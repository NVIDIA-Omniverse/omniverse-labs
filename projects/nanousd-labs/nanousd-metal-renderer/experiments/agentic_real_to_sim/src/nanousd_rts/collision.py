"""Owned adapter for splat-transform voxel and collision-mesh generation."""

from __future__ import annotations

import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .core import Bounds
from .core import RealToSimError, Workspace, sha256_file
from .gaussian import load_gaussians, write_gaussians


GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942
# splat-transform defines PLY space as a 180-degree rotation around Z, then
# converts to engine/glTF identity space before voxelization. NanoUSD's direct
# Gaussian path consumes the raw PLY rows, so collision GLBs must receive the
# inverse rotation exactly once. The rotation is self-inverse.
SPLAT_TRANSFORM_GLB_TO_NANOUSD = np.diag((-1.0, -1.0, 1.0))


def _bounds_corners(bounds: Bounds) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (bounds.minimum[0], bounds.maximum[0])
            for y in (bounds.minimum[1], bounds.maximum[1])
            for z in (bounds.minimum[2], bounds.maximum[2])
        ],
        dtype=np.float64,
    )


def _registration_for_transform(
    source: Bounds,
    derived: Bounds,
    matrix: np.ndarray,
) -> dict[str, Any]:
    source_center = np.asarray(source.center)
    source_size = np.asarray(source.size)
    scale = max(source.diagonal, 1e-9)
    corners = _bounds_corners(derived)
    transformed = corners @ matrix.T
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    center_error = float(np.linalg.norm((minimum + maximum) * 0.5 - source_center) / scale)
    size_error = float(np.linalg.norm((maximum - minimum) - source_size) / scale)
    score = center_error + size_error
    return {
        "normalized_residual": score,
        "center_error": center_error,
        "size_error": size_error,
        "matrix": np.asarray(matrix).tolist(),
        "determinant": float(np.linalg.det(matrix)),
        "registered_bounds": {
            "min": [float(item) for item in minimum],
            "max": [float(item) for item in maximum],
        },
    }


def _read_glb(path: Path) -> tuple[dict[str, Any], bytearray]:
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise RealToSimError("collision GLB is truncated")
    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total_length != len(raw):
        raise RealToSimError("collision output is not a valid GLB 2.0 file")
    offset = 12
    document = None
    binary = None
    while offset + 8 <= len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        payload = raw[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == GLB_JSON:
            document = json.loads(payload.rstrip(b" \0").decode("utf-8"))
        elif chunk_type == GLB_BIN:
            binary = bytearray(payload)
    if document is None or binary is None:
        raise RealToSimError("collision GLB must contain JSON and BIN chunks")
    return document, binary


def _write_glb(path: Path, document: dict[str, Any], binary: bytearray) -> None:
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary += b"\0" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    raw = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    raw.extend(struct.pack("<II", len(json_bytes), GLB_JSON))
    raw.extend(json_bytes)
    raw.extend(struct.pack("<II", len(binary), GLB_BIN))
    raw.extend(binary)
    Path(path).write_bytes(raw)


def _accessor_layout(document: dict[str, Any], accessor_index: int) -> tuple[dict[str, Any], int, int]:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 0))
    return accessor, offset, stride


def register_collision_glb(
    raw_path: Path,
    output_path: Path,
    *,
    expected_bounds: Bounds,
    max_normalized_residual: float = 0.2,
) -> dict[str, Any]:
    document, binary = _read_glb(raw_path)
    position_accessors = []
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            accessor_index = primitive.get("attributes", {}).get("POSITION")
            if accessor_index is not None:
                position_accessors.append(int(accessor_index))
    if not position_accessors:
        raise RealToSimError("collision GLB contains no POSITION accessor")
    derived_min = np.full(3, np.inf)
    derived_max = np.full(3, -np.inf)
    for accessor_index in position_accessors:
        accessor = document["accessors"][accessor_index]
        derived_min = np.minimum(derived_min, accessor["min"])
        derived_max = np.maximum(derived_max, accessor["max"])
    derived_bounds = Bounds(tuple(derived_min), tuple(derived_max))
    matrix = SPLAT_TRANSFORM_GLB_TO_NANOUSD.copy()
    registration = _registration_for_transform(expected_bounds, derived_bounds, matrix)
    registration["adapter_contract"] = (
        "splat-transform GLB (-x,-y,z) -> NanoUSD raw PLY (x,y,z)"
    )
    if registration["normalized_residual"] > max_normalized_residual:
        raise RealToSimError(
            "cannot register collision GLB to Gaussian coordinates: "
            f"normalized residual {registration['normalized_residual']:.4f} exceeds "
            f"{max_normalized_residual:.4f}"
        )
    registered_min = np.full(3, np.inf)
    registered_max = np.full(3, -np.inf)
    for accessor_index in position_accessors:
        accessor, offset, stride = _accessor_layout(document, accessor_index)
        if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
            raise RealToSimError("collision POSITION accessor must be float32 VEC3")
        stride = stride or 12
        local_min = np.full(3, np.inf)
        local_max = np.full(3, -np.inf)
        for index in range(int(accessor["count"])):
            position = np.asarray(struct.unpack_from("<3f", binary, offset + index * stride))
            transformed = matrix @ position
            struct.pack_into("<3f", binary, offset + index * stride, *transformed)
            local_min = np.minimum(local_min, transformed)
            local_max = np.maximum(local_max, transformed)
        accessor["min"] = [float(item) for item in local_min]
        accessor["max"] = [float(item) for item in local_max]
        registered_min = np.minimum(registered_min, local_min)
        registered_max = np.maximum(registered_max, local_max)
    if np.linalg.det(matrix) < 0:
        component_formats = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}
        for mesh in document.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                if "indices" not in primitive:
                    continue
                accessor, offset, stride = _accessor_layout(document, int(primitive["indices"]))
                if accessor.get("type") != "SCALAR" or accessor.get("componentType") not in component_formats:
                    raise RealToSimError("collision indices use an unsupported accessor format")
                code, width = component_formats[accessor["componentType"]]
                stride = stride or width
                count = int(accessor["count"])
                if count % 3:
                    raise RealToSimError("collision index count is not divisible by three")
                for index in range(0, count, 3):
                    second_offset = offset + (index + 1) * stride
                    third_offset = offset + (index + 2) * stride
                    second = struct.unpack_from("<" + code, binary, second_offset)[0]
                    third = struct.unpack_from("<" + code, binary, third_offset)[0]
                    struct.pack_into("<" + code, binary, second_offset, third)
                    struct.pack_into("<" + code, binary, third_offset, second)
    _write_glb(output_path, document, binary)
    registration.update(
        {
            "raw_bounds": derived_bounds.to_json(),
            "actual_registered_bounds": {
                "min": [float(item) for item in registered_min],
                "max": [float(item) for item in registered_max],
            },
            "max_normalized_residual": max_normalized_residual,
            "passed": True,
            "winding_reversed": bool(np.linalg.det(matrix) < 0),
        }
    )
    return registration


def voxelize(
    workspace: Workspace,
    *,
    node_id: str | None = None,
    voxel_size: float = 0.05,
    opacity_threshold: float = 0.1,
    mesh_shape: str = "faces",
    external_fill: float | None = None,
    floor_fill: float | None = None,
    carve: tuple[float, float] | None = None,
    seed: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if voxel_size <= 0 or not 0.0 <= opacity_threshold <= 1.0:
        raise RealToSimError("voxel_size must be positive and opacity_threshold must be in [0, 1]")
    if mesh_shape not in {"faces", "smooth"}:
        raise RealToSimError("collision mesh shape must be faces or smooth")
    workspace.verify_source()
    label = node_id or "scene"
    output_dir = workspace.root / "exports" / "voxel" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    source_scene = load_gaussians(workspace.source_path)
    source = workspace.source_path
    expected_bounds = source_scene.bounds
    selected_count = workspace.state["source"]["report"]["particle_count"]
    if node_id:
        node = workspace.node(node_id)
        selection = workspace.load_selection(node)
        selected_count = int(selection.size)
        selected_positions = source_scene.positions[selection]
        expected_bounds = Bounds(
            tuple(selected_positions.min(axis=0)),
            tuple(selected_positions.max(axis=0)),
        )
        source = write_gaussians(
            source_scene,
            output_dir / f"{label}.selected.ply",
            selection,
        )
    voxel_json = output_dir / f"{label}.voxel.json"
    command = [
        "npx",
        "--yes",
        "@playcanvas/splat-transform@2.7.1",
        "-w",
        str(source),
        "--voxel-params",
        f"{voxel_size},{opacity_threshold}",
    ]
    if external_fill is not None:
        command.extend(["--voxel-external-fill", str(external_fill)])
    if floor_fill is not None:
        command.extend(["--voxel-floor-fill", str(floor_fill)])
    if carve is not None:
        command.extend(["--voxel-carve", f"{carve[0]},{carve[1]}"])
    if seed is not None:
        command.extend(["--seed-pos", ",".join(str(item) for item in seed)])
    command.extend([str(voxel_json), "--collision-mesh", mesh_shape, "--mem"])
    started = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    voxel_bin = output_dir / f"{label}.voxel.bin"
    collision_glb = output_dir / f"{label}.collision.glb"
    missing = [str(path) for path in (voxel_json, voxel_bin, collision_glb) if not path.is_file()]
    if completed.returncode != 0 or missing:
        raise RealToSimError(
            "voxel/collision generation failed"
            + (f"; missing outputs: {missing}" if missing else "")
            + "\n"
            + (completed.stderr or completed.stdout or "no converter output")
        )
    registered_glb = output_dir / f"{label}.registered.collision.glb"
    registration = register_collision_glb(
        collision_glb,
        registered_glb,
        expected_bounds=expected_bounds,
    )
    voxel_metadata = json.loads(voxel_json.read_text(encoding="utf-8"))
    artifacts = {
        path.name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (voxel_json, voxel_bin, collision_glb, registered_glb)
    }
    if node_id:
        artifacts[source.name] = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    report = {
        "schema_version": 1,
        "scope": {"node": node_id, "selected_gaussians": selected_count},
        "settings": {
            "voxel_size": voxel_size,
            "opacity_threshold": opacity_threshold,
            "mesh_shape": mesh_shape,
            "external_fill": external_fill,
            "floor_fill": floor_fill,
            "carve": list(carve) if carve else None,
            "seed": list(seed) if seed else None,
        },
        "converter": "@playcanvas/splat-transform@2.7.1",
        "command": command,
        "elapsed_seconds": elapsed,
        "voxel_metadata": voxel_metadata,
        "registration": registration,
        "artifacts": artifacts,
        "stdout_tail": completed.stdout[-4000:],
        "passed": True,
        "provenance_note": "Collision geometry is derived, not measured; retain Gaussian source as visual truth.",
    }
    report_path = output_dir / "collision-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    workspace.trace(
        "voxelize",
        {"node": node_id, **report["settings"]},
        {"report": str(report_path), "artifacts": artifacts},
    )
    return report
