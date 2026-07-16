"""Gaussian ingest, strict 3DGS decoding, Metal AOV rendering, and fixtures."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .core import AXES, Bounds, RealToSimError, Workspace, sha256_file


MAX_GAUSSIANS = 2_000_000
MAX_HEADER_BYTES = 64 * 1024
REQUIRED = {
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


@dataclass(frozen=True, slots=True)
class PlyHeader:
    format: str
    count: int
    properties: tuple[str, ...]
    header_bytes: int

    @property
    def sh_degree(self) -> int:
        rest = sum(name.startswith("f_rest_") for name in self.properties)
        try:
            return {0: 0, 9: 1, 24: 2, 45: 3}[rest]
        except KeyError as exc:
            raise RealToSimError(f"unsupported f_rest property count: {rest}") from exc


@dataclass(frozen=True, slots=True)
class GaussianScene:
    source_path: Path
    source_sha256: str
    positions: np.ndarray
    scales: np.ndarray
    orientations: np.ndarray
    opacities: np.ndarray
    sh_coefficients: np.ndarray
    sh_degree: int

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def bounds(self) -> Bounds:
        return Bounds(tuple(self.positions.min(axis=0)), tuple(self.positions.max(axis=0)))

    def report(self) -> dict[str, Any]:
        quaternion_error = np.abs(np.linalg.norm(self.orientations, axis=1) - 1.0)
        return {
            "particle_count": self.count,
            "sh_degree": self.sh_degree,
            "bounds": {
                **self.bounds.to_json(),
                "diagonal": self.bounds.diagonal,
            },
            "scale": {
                "min": float(self.scales.min()),
                "max": float(self.scales.max()),
            },
            "opacity": {
                "min": float(self.opacities.min()),
                "max": float(self.opacities.max()),
            },
            "quaternion_norm_error_max": float(quaternion_error.max()),
            "finite": bool(
                np.isfinite(self.positions).all()
                and np.isfinite(self.scales).all()
                and np.isfinite(self.orientations).all()
                and np.isfinite(self.opacities).all()
                and np.isfinite(self.sh_coefficients).all()
            ),
        }


@dataclass(frozen=True, slots=True)
class Camera:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float = 60.0
    near_clip: float = 0.01
    far_clip: float = 10_000.0

    @classmethod
    def from_json(cls, value: dict[str, Any], *, up_axis: str) -> "Camera":
        up = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}[up_axis]
        return cls(
            name=value.get("name", "camera"),
            eye=tuple(float(item) for item in value["eye"]),
            target=tuple(float(item) for item in value["target"]),
            up=tuple(float(item) for item in value.get("up", up)),
            fov_degrees=float(value.get("fov_degrees", value.get("fov", 60.0))),
            near_clip=float(value.get("near_clip", 0.01)),
            far_clip=float(value.get("far_clip", 10_000.0)),
        )


def inspect_ply(path: Path) -> PlyHeader:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() != ".ply":
        raise RealToSimError(f"expected a standard 3DGS .ply file: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(MAX_HEADER_BYTES + 1)
    candidates = []
    for marker in (b"end_header\n", b"end_header\r\n"):
        index = prefix.find(marker)
        if index >= 0:
            candidates.append((index, marker))
    if not candidates:
        raise RealToSimError("PLY end_header was not found within 64 KiB")
    marker_offset, marker = min(candidates, key=lambda item: item[0])
    try:
        lines = prefix[:marker_offset].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise RealToSimError("PLY header must be ASCII") from exc
    if not lines or lines[0].strip() != "ply":
        raise RealToSimError("input is missing PLY magic")
    ply_format = None
    count = None
    properties: list[str] = []
    in_vertices = False
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            if len(parts) != 3 or parts[2] != "1.0":
                raise RealToSimError("only PLY version 1.0 is supported")
            ply_format = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertices = True
        elif parts[0] == "element":
            in_vertices = False
        elif parts[0] == "property" and in_vertices:
            if len(parts) != 3 or parts[1] not in {"float", "float32"}:
                raise RealToSimError("Gaussian vertex properties must be scalar float32")
            properties.append(parts[2])
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise RealToSimError(f"unsupported PLY format: {ply_format}")
    if count is None or not 1 <= count <= MAX_GAUSSIANS:
        raise RealToSimError(f"Gaussian count must be within [1, {MAX_GAUSSIANS}]")
    missing = sorted(REQUIRED.difference(properties))
    if missing:
        raise RealToSimError(f"PLY is missing required 3DGS properties: {missing}")
    if len(properties) != len(set(properties)):
        raise RealToSimError("PLY vertex property names must be unique")
    header = PlyHeader(ply_format, count, tuple(properties), marker_offset + len(marker))
    _ = header.sh_degree
    return header


def _load_matrix(path: Path, header: PlyHeader) -> np.ndarray:
    if header.format == "ascii":
        with Path(path).open("rb") as stream:
            stream.seek(header.header_bytes)
            matrix = np.loadtxt(stream, dtype=np.float32, max_rows=header.count, ndmin=2)
    else:
        order = "<" if header.format == "binary_little_endian" else ">"
        dtype = np.dtype([(name, f"{order}f4") for name in header.properties], align=False)
        expected = header.header_bytes + header.count * dtype.itemsize
        if Path(path).stat().st_size < expected:
            raise RealToSimError("binary PLY payload is truncated")
        with Path(path).open("rb") as stream:
            stream.seek(header.header_bytes)
            records = np.fromfile(stream, dtype=dtype, count=header.count)
        matrix = np.column_stack([records[name] for name in header.properties])
    if matrix.shape != (header.count, len(header.properties)):
        raise RealToSimError(
            f"PLY payload shape mismatch: expected {(header.count, len(header.properties))}, got {matrix.shape}"
        )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise RealToSimError("PLY payload contains non-finite values")
    return matrix


def _sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def load_gaussians(path: Path) -> GaussianScene:
    path = Path(path).resolve()
    header = inspect_ply(path)
    matrix = _load_matrix(path, header)
    column = {name: matrix[:, index] for index, name in enumerate(header.properties)}
    positions = np.column_stack([column["x"], column["y"], column["z"]]).astype(np.float32)
    with np.errstate(over="ignore", under="ignore"):
        scales = np.exp(
            np.column_stack([column["scale_0"], column["scale_1"], column["scale_2"]])
        ).astype(np.float32)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RealToSimError("Gaussian scale logits did not decode to finite positive values")
    orientations = np.column_stack(
        [column["rot_0"], column["rot_1"], column["rot_2"], column["rot_3"]]
    ).astype(np.float32)
    norms = np.linalg.norm(orientations, axis=1)
    if np.any(norms < 1e-12) or not np.isfinite(norms).all():
        raise RealToSimError("Gaussian orientations contain a zero quaternion")
    orientations /= norms[:, None]
    opacities = _sigmoid(column["opacity"].astype(np.float32))
    degree = header.sh_degree
    coefficient_count = (degree + 1) ** 2
    sh = np.empty((header.count, coefficient_count, 3), dtype=np.float32)
    sh[:, 0, :] = np.column_stack([column["f_dc_0"], column["f_dc_1"], column["f_dc_2"]])
    rest = coefficient_count - 1
    if rest:
        packed = np.column_stack([column[f"f_rest_{index}"] for index in range(rest * 3)])
        sh[:, 1:, :] = packed.reshape(header.count, 3, rest).transpose(0, 2, 1)
    return GaussianScene(
        source_path=path,
        source_sha256=sha256_file(path),
        positions=positions,
        scales=scales,
        orientations=orientations,
        opacities=opacities,
        sh_coefficients=sh,
        sh_degree=degree,
    )


def _sog_inputs(source: Path, lod: int | None) -> tuple[list[Path], dict[str, Any]]:
    metadata_path = source / "lod-meta.json"
    if not metadata_path.is_file():
        raise RealToSimError(f"SOG/LOD directory is missing lod-meta.json: {source}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lod_levels = int(metadata.get("lodLevels", len(metadata.get("counts", []))))
    selected_lod = lod_levels - 1 if lod is None else lod
    if not 0 <= selected_lod < lod_levels:
        raise RealToSimError(f"LOD must be in [0, {lod_levels - 1}]")
    filenames = [
        source / item
        for item in metadata.get("filenames", [])
        if Path(item).parts and Path(item).parts[0].startswith(f"{selected_lod}_")
    ]
    if not filenames or any(not item.is_file() for item in filenames):
        raise RealToSimError(f"LOD {selected_lod} does not resolve to complete meta.json chunks")
    return filenames, {
        "input_kind": "playcanvas-sog-lod",
        "lod": selected_lod,
        "lod_count": int(metadata.get("counts", [])[selected_lod]),
        "generator": metadata.get("asset", {}).get("generator"),
        "chunks": [str(item.resolve()) for item in filenames],
    }


def canonicalize_input(source: Path, *, lod: int | None = None) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() == ".ply":
        return source, {"input_kind": "standard-3dgs-ply", "original_path": str(source)}, None
    if source.is_dir():
        inputs, provenance = _sog_inputs(source, lod)
        temporary = tempfile.TemporaryDirectory(prefix="nanousd-rts-sog-")
        output = Path(temporary.name) / "source.ply"
        command = [
            "npx",
            "--yes",
            "@playcanvas/splat-transform@2.7.1",
            "-w",
            *(str(item) for item in inputs),
            str(output),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0 or not output.is_file():
            temporary.cleanup()
            raise RealToSimError(
                "splat-transform conversion failed:\n"
                + (completed.stderr or completed.stdout or "no converter output")
            )
        provenance.update(
            {
                "original_path": str(source),
                "converter": "@playcanvas/splat-transform@2.7.1",
                "converter_command": command,
                "converter_stdout_tail": completed.stdout[-2000:],
            }
        )
        return output, provenance, temporary
    raise RealToSimError(f"unsupported Gaussian input: {source}")


def ingest(
    source: Path,
    workspace: Path,
    *,
    lod: int | None = None,
    up_axis: str = "Y",
    meters_per_unit: float = 1.0,
    replace: bool = False,
) -> Workspace:
    if up_axis not in {"X", "Y", "Z"}:
        raise RealToSimError("up_axis must be X, Y, or Z")
    canonical, provenance, temporary = canonicalize_input(source, lod=lod)
    try:
        scene = load_gaussians(canonical)
        result = Workspace.create(
            workspace,
            canonical,
            scene.report(),
            source_provenance=provenance,
            up_axis=up_axis,
            meters_per_unit=meters_per_unit,
            replace=replace,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return result


def default_camera(scene: GaussianScene, *, up_axis: str, name: str = "auto") -> Camera:
    bounds = scene.bounds
    center = np.asarray(bounds.center)
    size = np.asarray(bounds.size)
    up_index = {"X": 0, "Y": 1, "Z": 2}[up_axis]
    horizontal = [index for index in range(3) if index != up_index]
    forward_index = horizontal[1]
    eye = center.copy()
    eye[up_index] += max(size[up_index] * 0.25, bounds.diagonal * 0.08)
    eye[forward_index] -= max(bounds.diagonal * 1.15, 1.0)
    up = np.zeros(3)
    up[up_index] = 1.0
    return Camera(
        name=name,
        eye=tuple(float(item) for item in eye),
        target=tuple(float(item) for item in center),
        up=tuple(float(item) for item in up),
        fov_degrees=60.0,
        near_clip=max(0.001, bounds.diagonal / 100_000.0),
        far_clip=max(100.0, bounds.diagonal * 20.0),
    )


def orbit_cameras(scene: GaussianScene, *, up_axis: str, count: int = 6) -> list[Camera]:
    if count < 1:
        raise RealToSimError("orbit count must be positive")
    center = np.asarray(scene.bounds.center, dtype=np.float64)
    radius = max(1.0, scene.bounds.diagonal * 1.15)
    up_index = {"X": 0, "Y": 1, "Z": 2}[up_axis]
    horizontal = [index for index in range(3) if index != up_index]
    up = np.zeros(3)
    up[up_index] = 1.0
    cameras = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        eye = center.copy()
        eye[horizontal[0]] += math.sin(angle) * radius
        eye[horizontal[1]] -= math.cos(angle) * radius
        eye[up_index] += radius * 0.18
        cameras.append(
            Camera(
                name=f"orbit_{index:03d}",
                eye=tuple(float(item) for item in eye),
                target=tuple(float(item) for item in center),
                up=tuple(float(item) for item in up),
                fov_degrees=60.0,
                near_clip=max(0.001, radius / 100_000.0),
                far_clip=radius * 20.0,
            )
        )
    return cameras


def _renderer_root() -> Path:
    override = os.environ.get("NANOUSD_METAL_RENDERER_ROOT")
    root = Path(override).resolve() if override else Path(__file__).resolve().parents[4]
    if not (root / "python" / "nusd_renderer" / "_bindings.py").is_file():
        raise RealToSimError(
            f"NanoUSD Metal renderer root is invalid: {root}; set NANOUSD_METAL_RENDERER_ROOT"
        )
    return root


def _load_renderer_binding() -> tuple[type[Any], dict[str, str]]:
    root = _renderer_root()
    library = root / "build" / "libnusd_renderer.dylib"
    nanousd_root = root.parent / "nanousd"
    dependency_candidates = [
        nanousd_root / "build" / "Release" / "libnanousd.dylib",
        nanousd_root / "build" / "libnanousd.dylib",
    ]
    dependency = next((item for item in dependency_candidates if item.is_file()), None)
    if not library.is_file() or dependency is None:
        raise RealToSimError(
            "NanoUSD renderer is not built; run cmake --build build in nanousd-metal-renderer"
        )
    ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
    os.environ["NUSD_RENDERER_LIB"] = str(library)
    binding_path = root / "python" / "nusd_renderer" / "_bindings.py"
    spec = importlib.util.spec_from_file_location("_nanousd_rts_bindings", binding_path)
    if spec is None or spec.loader is None:
        raise RealToSimError(f"cannot load renderer Python binding: {binding_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module.NuRenderer, "gs_fetch_id"):
        raise RealToSimError("renderer build is missing the stable Gaussian ID AOV")
    return module.NuRenderer, {
        "renderer_root": str(root),
        "library": str(library),
        "library_sha256": sha256_file(library),
        "nanousd_library": str(dependency),
    }


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    output = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        high = max(float(high), float(low) + 1e-6)
        output[valid] = np.clip((1.0 - (depth[valid] - low) / (high - low)) * 255, 0, 255)
    return output


def _normal_preview(normal: np.ndarray, depth: np.ndarray) -> np.ndarray:
    output = np.clip((normal * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    output[depth <= 0] = 0
    return output


def _id_preview(ids: np.ndarray) -> np.ndarray:
    values = ids.astype(np.uint64)
    output = np.zeros((*ids.shape, 3), dtype=np.uint8)
    output[..., 0] = ((values * 37 + 17) & 255).astype(np.uint8)
    output[..., 1] = ((values * 73 + 41) & 255).astype(np.uint8)
    output[..., 2] = ((values * 109 + 83) & 255).astype(np.uint8)
    output[ids == 0] = 0
    return output


def render(
    workspace: Workspace,
    *,
    name: str,
    camera: Camera | None = None,
    width: int = 960,
    height: int = 540,
    k: int = 16,
    max_passes: int = 200,
    min_transmittance: float = 0.03,
    iso_opacity_threshold: float = 0.5,
) -> dict[str, Any]:
    workspace.verify_source()
    scene = load_gaussians(workspace.source_path)
    selected_camera = camera or default_camera(scene, up_axis=workspace.up_axis)
    renderer_type, dependency = _load_renderer_binding()
    renderer = renderer_type(
        width=width,
        height=height,
        enable_rt=True,
        enable_materials=False,
        visible=False,
    )
    output_dir = workspace.root / "evidence" / "render" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        upload_started = time.perf_counter()
        renderer.gs_set_particles(
            scene.positions,
            scene.scales,
            scene.orientations,
            scene.opacities,
            scene.sh_coefficients,
            scene.sh_degree,
        )
        renderer.gs_set_k(k)
        renderer.gs_set_max_passes(max_passes)
        renderer.gs_set_min_transmittance(min_transmittance)
        renderer.gs_set_iso_opacity_threshold(iso_opacity_threshold)
        upload_seconds = time.perf_counter() - upload_started
        renderer.set_camera_explicit(
            selected_camera.eye,
            selected_camera.target,
            selected_camera.up,
            selected_camera.fov_degrees,
            selected_camera.near_clip,
            selected_camera.far_clip,
        )
        render_started = time.perf_counter()
        renderer.gs_render()
        render_seconds = time.perf_counter() - render_started
        read_started = time.perf_counter()
        rgb = np.ascontiguousarray(renderer.fetch_pixels()[..., :3])
        depth = np.ascontiguousarray(renderer.gs_fetch_depth())
        normal = np.ascontiguousarray(renderer.gs_fetch_normal())
        ids = np.ascontiguousarray(renderer.gs_fetch_id())
        readback_seconds = time.perf_counter() - read_started
        backend = renderer.get_backend_info()
    finally:
        renderer.close()
    valid_depth = depth > 0
    valid_ids = ids > 0
    if not np.array_equal(valid_depth, valid_ids):
        raise RealToSimError("depth and stable-ID AOV validity masks differ")
    if not np.any(valid_ids) or int(rgb.sum(dtype=np.uint64)) == 0:
        raise RealToSimError("Gaussian render is blank")
    if np.any(ids > scene.count):
        raise RealToSimError("stable-ID AOV references an unknown source particle")
    Image.fromarray(rgb, mode="RGB").save(output_dir / "rgb.png")
    np.save(output_dir / "depth.npy", depth, allow_pickle=False)
    Image.fromarray(_depth_preview(depth), mode="L").save(output_dir / "depth.png")
    np.save(output_dir / "normal.npy", normal, allow_pickle=False)
    Image.fromarray(_normal_preview(normal, depth), mode="RGB").save(output_dir / "normal.png")
    np.save(output_dir / "id.npy", ids, allow_pickle=False)
    Image.fromarray(_id_preview(ids), mode="RGB").save(output_dir / "id.png")
    metadata = {
        "schema_version": 1,
        "source_sha256": scene.source_sha256,
        "source_particle_count": scene.count,
        "camera": asdict(selected_camera),
        "settings": {
            "width": width,
            "height": height,
            "k": k,
            "max_passes": max_passes,
            "min_transmittance": min_transmittance,
            "iso_opacity_threshold": iso_opacity_threshold,
        },
        "timings": {
            "upload_seconds": upload_seconds,
            "render_seconds": render_seconds,
            "readback_seconds": readback_seconds,
        },
        "aov": {
            "visible_pixels": int(np.count_nonzero(valid_ids)),
            "unique_visible_particle_ids": int(np.unique(ids[valid_ids]).size),
            "rgb_sum": int(rgb.sum(dtype=np.uint64)),
            "depth_min": float(depth[valid_depth].min()),
            "depth_max": float(depth[valid_depth].max()),
            "id_semantics": "zero=background; nonzero=one-based immutable PLY row",
        },
        "backend": backend,
        "dependency": dependency,
    }
    (output_dir / "render.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    workspace.trace(
        "render",
        {"name": name, "camera": asdict(selected_camera), "settings": metadata["settings"]},
        {"evidence_dir": str(output_dir), "aov": metadata["aov"]},
    )
    return metadata


def select_bounds(scene: GaussianScene, bounds: Bounds) -> np.ndarray:
    minimum = np.asarray(bounds.minimum, dtype=np.float32)
    maximum = np.asarray(bounds.maximum, dtype=np.float32)
    mask = np.all((scene.positions >= minimum) & (scene.positions <= maximum), axis=1)
    return np.nonzero(mask)[0].astype(np.uint32)


def select_render_mask(id_aov: Path, mask_path: Path) -> np.ndarray:
    ids = np.load(Path(id_aov), allow_pickle=False)
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path, allow_pickle=False)
    else:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if ids.shape != mask.shape:
        raise RealToSimError(f"ID AOV and mask shapes differ: {ids.shape} != {mask.shape}")
    observed = np.unique(ids[np.asarray(mask, dtype=bool)])
    observed = observed[observed > 0]
    return (observed - 1).astype(np.uint32)


def _write_binary_ply(path: Path, records: np.ndarray) -> None:
    names = list(records.dtype.names or ())
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(records)}",
        *(f"property float {name}" for name in names),
        "end_header",
        "",
    ]
    with Path(path).open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        records.tofile(stream)


def write_gaussians(scene: GaussianScene, path: Path, source_indices: np.ndarray | None = None) -> Path:
    indices = (
        np.arange(scene.count, dtype=np.uint32)
        if source_indices is None
        else np.unique(np.asarray(source_indices, dtype=np.uint32))
    )
    if indices.size == 0 or int(indices.max()) >= scene.count:
        raise RealToSimError("Gaussian export selection is empty or out of range")
    rest_count = ((scene.sh_degree + 1) ** 2 - 1) * 3
    names = [
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *(f"f_rest_{index}" for index in range(rest_count)),
        "opacity",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    records = np.zeros(indices.size, dtype=np.dtype([(name, "<f4") for name in names]))
    positions = scene.positions[indices]
    scales = scene.scales[indices]
    sh = scene.sh_coefficients[indices]
    opacity = np.clip(scene.opacities[indices], 1e-6, 1.0 - 1e-6)
    records["x"], records["y"], records["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    logs = np.log(scales)
    records["scale_0"], records["scale_1"], records["scale_2"] = logs[:, 0], logs[:, 1], logs[:, 2]
    records["f_dc_0"], records["f_dc_1"], records["f_dc_2"] = sh[:, 0, 0], sh[:, 0, 1], sh[:, 0, 2]
    if rest_count:
        channel_major = sh[:, 1:, :].transpose(0, 2, 1).reshape(indices.size, rest_count)
        for index in range(rest_count):
            records[f"f_rest_{index}"] = channel_major[:, index]
    records["opacity"] = np.log(opacity / (1.0 - opacity))
    orientation = scene.orientations[indices]
    records["rot_0"], records["rot_1"], records["rot_2"], records["rot_3"] = (
        orientation[:, 0],
        orientation[:, 1],
        orientation[:, 2],
        orientation[:, 3],
    )
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(path, records)
    return path


def write_box_gaussians(
    path: Path,
    bounds: Bounds,
    *,
    color: tuple[float, float, float] = (0.55, 0.28, 0.12),
    count: int = 600,
    seed: int = 0,
) -> Path:
    if count < 24:
        raise RealToSimError("box completion requires at least 24 Gaussians")
    rng = np.random.default_rng(seed)
    dtype = np.dtype(
        [(name, "<f4") for name in (
            "x", "y", "z",
            "scale_0", "scale_1", "scale_2",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "rot_0", "rot_1", "rot_2", "rot_3",
        )]
    )
    records = np.zeros(count, dtype=dtype)
    points = rng.uniform(bounds.minimum, bounds.maximum, size=(count, 3)).astype(np.float32)
    surface_axis = rng.integers(0, 3, size=count)
    surface_side = rng.integers(0, 2, size=count)
    for index in range(count):
        points[index, surface_axis[index]] = (
            bounds.minimum[surface_axis[index]]
            if surface_side[index] == 0
            else bounds.maximum[surface_axis[index]]
        )
    records["x"], records["y"], records["z"] = points[:, 0], points[:, 1], points[:, 2]
    sigma = max(min(bounds.size) / 35.0, bounds.diagonal / 350.0, 0.002)
    for name in ("scale_0", "scale_1", "scale_2"):
        records[name] = math.log(sigma)
    c0 = 0.28209479177387814
    records["f_dc_0"] = (color[0] - 0.5) / c0
    records["f_dc_1"] = (color[1] - 0.5) / c0
    records["f_dc_2"] = (color[2] - 0.5) / c0
    records["opacity"] = 3.0
    records["rot_0"] = 1.0
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(path, records)
    return path


def write_surface_patch_gaussians(
    path: Path,
    patches: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int = 0,
) -> Path:
    """Write smooth axis-aligned Gaussian surface patches into one generated PLY."""

    normalized: list[dict[str, Any]] = []
    for patch in patches:
        bounds = patch.get("bounds")
        if not isinstance(bounds, Bounds):
            raise RealToSimError("surface patch bounds must be a Bounds instance")
        axis_value = patch.get("axis")
        axis = AXES[axis_value] if isinstance(axis_value, str) else int(axis_value)
        side = int(patch.get("side", 1))
        color = tuple(float(value) for value in patch.get("color", (0.65, 0.65, 0.65)))
        weight = float(patch.get("weight", 1.0))
        opacity = float(patch.get("opacity", 0.97))
        if axis not in (0, 1, 2) or side not in (-1, 1):
            raise RealToSimError("surface patch axis/side must be X/Y/Z and -1/+1")
        if len(color) != 3 or not all(0.0 <= value <= 1.0 for value in color):
            raise RealToSimError("surface patch color must contain three values in [0, 1]")
        if weight <= 0.0 or not 0.0 < opacity < 1.0:
            raise RealToSimError("surface patch weight and opacity must be positive")
        tangent = [index for index in range(3) if index != axis]
        area = float(bounds.size[tangent[0]] * bounds.size[tangent[1]])
        normalized.append(
            {
                "bounds": bounds,
                "axis": axis,
                "side": side,
                "color": color,
                "weight": weight,
                "opacity": opacity,
                "area": area,
                "tangent": tangent,
            }
        )
    if not normalized:
        raise RealToSimError("surface-patch Gaussian export requires at least one patch")
    if count < len(normalized) * 8:
        raise RealToSimError(
            f"surface-patch Gaussian export requires at least {len(normalized) * 8} Gaussians"
        )

    scores = np.asarray(
        [item["area"] * item["weight"] for item in normalized],
        dtype=np.float64,
    )
    raw_counts = scores / scores.sum() * count
    patch_counts = np.floor(raw_counts).astype(np.int64)
    patch_counts = np.maximum(patch_counts, 8)
    while int(patch_counts.sum()) > count:
        candidates = np.flatnonzero(patch_counts > 8)
        if not candidates.size:
            raise RealToSimError("surface-patch allocation could not satisfy the requested count")
        index = int(candidates[np.argmax(patch_counts[candidates] - raw_counts[candidates])])
        patch_counts[index] -= 1
    remainder = count - int(patch_counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - np.floor(raw_counts)))
        for index in order[:remainder]:
            patch_counts[index] += 1

    dtype = np.dtype(
        [(name, "<f4") for name in (
            "x", "y", "z",
            "scale_0", "scale_1", "scale_2",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "rot_0", "rot_1", "rot_2", "rot_3",
        )]
    )
    records = np.zeros(count, dtype=dtype)
    rng = np.random.default_rng(seed)
    c0 = 0.28209479177387814
    cursor = 0
    for patch, patch_count in zip(normalized, patch_counts, strict=True):
        stop = cursor + int(patch_count)
        bounds = patch["bounds"]
        axis = patch["axis"]
        tangent = patch["tangent"]
        sequence = np.arange(int(patch_count), dtype=np.float64)
        phase = rng.random(2)
        # A low-discrepancy R2 sequence avoids the visible holes and clumps of
        # independently random samples while retaining deterministic per-asset
        # variation.
        coordinates = (
            np.mod(phase[0] + sequence * 0.7548776662466927, 1.0),
            np.mod(phase[1] + sequence * 0.5698402909980532, 1.0),
        )
        points = np.empty((patch_count, 3), dtype=np.float64)
        points[:, axis] = (
            bounds.minimum[axis] if patch["side"] < 0 else bounds.maximum[axis]
        )
        for component, coordinate in zip(tangent, coordinates, strict=True):
            points[:, component] = (
                bounds.minimum[component]
                + coordinate * (bounds.maximum[component] - bounds.minimum[component])
            )
        records["x"][cursor:stop] = points[:, 0]
        records["y"][cursor:stop] = points[:, 1]
        records["z"][cursor:stop] = points[:, 2]

        spacing = math.sqrt(max(patch["area"], 1e-9) / max(int(patch_count), 1))
        tangent_sigma = float(np.clip(spacing * 0.56, 0.0028, 0.024))
        normal_sigma = max(tangent_sigma * 0.12, 0.0009)
        sigmas = [tangent_sigma, tangent_sigma, tangent_sigma]
        sigmas[axis] = normal_sigma
        for component, name in enumerate(("scale_0", "scale_1", "scale_2")):
            records[name][cursor:stop] = math.log(sigmas[component])

        color = patch["color"]
        records["f_dc_0"][cursor:stop] = (color[0] - 0.5) / c0
        records["f_dc_1"][cursor:stop] = (color[1] - 0.5) / c0
        records["f_dc_2"][cursor:stop] = (color[2] - 0.5) / c0
        opacity = patch["opacity"]
        records["opacity"][cursor:stop] = math.log(opacity / (1.0 - opacity))
        records["rot_0"][cursor:stop] = 1.0
        cursor = stop

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(path, records)
    return path


def _quaternion_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise RealToSimError("Gaussian face frame must be a finite 3x3 matrix")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(
                max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(
                max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)
            ) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise RealToSimError("could not derive a valid Gaussian face quaternion")
    return quaternion / norm


def _weighted_counts(
    scores: np.ndarray,
    *,
    count: int,
    minimum: int,
) -> np.ndarray:
    if count < len(scores) * minimum:
        raise RealToSimError(
            f"mesh-bound Gaussian export requires at least {len(scores) * minimum} Gaussians"
        )
    raw = scores / scores.sum() * count
    allocated = np.maximum(np.floor(raw).astype(np.int64), minimum)
    while int(allocated.sum()) > count:
        candidates = np.flatnonzero(allocated > minimum)
        if not candidates.size:
            raise RealToSimError("mesh-bound Gaussian allocation exceeded its budget")
        index = int(candidates[np.argmax(allocated[candidates] - raw[candidates])])
        allocated[index] -= 1
    order = np.argsort(-(raw - np.floor(raw)))
    cursor = 0
    while int(allocated.sum()) < count:
        allocated[int(order[cursor % len(order)])] += 1
        cursor += 1
    return allocated


def _sample_texture(
    texture: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    image = np.asarray(texture)
    if image.ndim != 3 or image.shape[2] < 3:
        raise RealToSimError("base-color texture must have shape HxWx3 or HxWx4")
    if image.dtype == np.uint8:
        colors = image[..., :3].astype(np.float64) / 255.0
    else:
        colors = image[..., :3].astype(np.float64)
        if not np.isfinite(colors).all():
            raise RealToSimError("base-color texture contains non-finite values")
        colors = np.clip(colors, 0.0, 1.0)
    height, width = colors.shape[:2]
    coordinates = np.clip(np.asarray(uv, dtype=np.float64), 0.0, 1.0)
    x = np.rint(coordinates[:, 0] * (width - 1)).astype(np.int64)
    # OBJ UVs use a bottom-left origin; image arrays use a top-left origin.
    y = np.rint((1.0 - coordinates[:, 1]) * (height - 1)).astype(np.int64)
    return colors[y, x]


def write_mesh_bound_gaussians(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    face_colors: np.ndarray,
    count: int,
    association_path: Path | None = None,
    face_uvs: np.ndarray | None = None,
    base_color_texture: np.ndarray | None = None,
    face_weights: np.ndarray | None = None,
    face_opacities: np.ndarray | None = None,
    face_groups: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample face-bound Gaussians and retain DRAWER-style mesh associations."""

    vertex_values = np.asarray(vertices, dtype=np.float64)
    face_values = np.asarray(faces, dtype=np.int64)
    color_values = np.asarray(face_colors, dtype=np.float64)
    if (
        vertex_values.ndim != 2
        or vertex_values.shape[1] != 3
        or not np.isfinite(vertex_values).all()
    ):
        raise RealToSimError("mesh vertices must have finite shape Nx3")
    if face_values.ndim != 2 or face_values.shape[1] != 3 or not len(face_values):
        raise RealToSimError("mesh faces must have non-empty shape Mx3")
    if int(face_values.min()) < 0 or int(face_values.max()) >= len(vertex_values):
        raise RealToSimError("mesh face indices are out of range")
    if color_values.shape != (len(face_values), 3) or not np.isfinite(color_values).all():
        raise RealToSimError("mesh face colors must have finite shape Mx3")
    color_values = np.clip(color_values, 0.0, 1.0)

    triangles = vertex_values[face_values]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(first_edges, second_edges)
    double_areas = np.linalg.norm(cross, axis=1)
    if np.any(double_areas <= 1e-12):
        raise RealToSimError("mesh contains a degenerate triangle")
    areas = double_areas * 0.5
    normals = cross / double_areas[:, None]
    tangents = first_edges / np.linalg.norm(first_edges, axis=1)[:, None]
    bitangents = np.cross(normals, tangents)
    bitangents /= np.linalg.norm(bitangents, axis=1)[:, None]
    frames = np.stack((tangents, bitangents, normals), axis=2)
    quaternions = np.stack(
        [
            # NanoUSD Metal expands a wxyz quaternion into three matrix rows
            # and treats those rows as ellipsoid axes. Transpose the conventional
            # column-basis face frame so the renderer receives tangent,
            # bitangent, and normal in that order.
            _quaternion_wxyz_from_matrix(frame.T)
            for frame in frames
        ],
        axis=0,
    )

    weights = (
        np.ones(len(face_values), dtype=np.float64)
        if face_weights is None
        else np.asarray(face_weights, dtype=np.float64)
    )
    opacities = (
        np.full(len(face_values), 0.97, dtype=np.float64)
        if face_opacities is None
        else np.asarray(face_opacities, dtype=np.float64)
    )
    if (
        weights.shape != (len(face_values),)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise RealToSimError("mesh face weights must be finite and positive")
    if (
        opacities.shape != (len(face_values),)
        or not np.isfinite(opacities).all()
        or np.any((opacities <= 0.0) | (opacities >= 1.0))
    ):
        raise RealToSimError("mesh face opacities must lie strictly within (0, 1)")
    uv_values = None
    if face_uvs is not None:
        uv_values = np.asarray(face_uvs, dtype=np.float64)
        if uv_values.shape != (len(face_values), 3, 2) or not np.isfinite(uv_values).all():
            raise RealToSimError("mesh face UVs must have finite shape Mx3x2")
    if base_color_texture is not None and uv_values is None:
        raise RealToSimError("texture sampling requires mesh face UVs")
    group_values = (
        np.arange(len(face_values), dtype=np.uint32)
        if face_groups is None
        else np.asarray(face_groups, dtype=np.uint32)
    )
    if group_values.shape != (len(face_values),):
        raise RealToSimError("mesh face groups must have shape M")

    allocations = _weighted_counts(areas * weights, count=count, minimum=4)
    dtype = np.dtype(
        [
            (name, "<f4")
            for name in (
                "x",
                "y",
                "z",
                "scale_0",
                "scale_1",
                "scale_2",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            )
        ]
    )
    records = np.zeros(count, dtype=dtype)
    association_faces = np.empty(count, dtype=np.uint32)
    association_barycentric = np.empty((count, 3), dtype=np.float32)
    association_uv = np.full((count, 2), np.nan, dtype=np.float32)
    rng = np.random.default_rng(seed)
    c0 = 0.28209479177387814
    cursor = 0
    for face_index, face_count in enumerate(allocations):
        stop = cursor + int(face_count)
        sequence = np.arange(int(face_count), dtype=np.float64)
        phase = rng.random(2)
        first = np.mod(phase[0] + sequence * 0.7548776662466927, 1.0)
        second = np.mod(phase[1] + sequence * 0.5698402909980532, 1.0)
        root = np.sqrt(first)
        barycentric = np.column_stack(
            (1.0 - root, root * (1.0 - second), root * second)
        )
        points = barycentric @ triangles[face_index]
        records["x"][cursor:stop] = points[:, 0]
        records["y"][cursor:stop] = points[:, 1]
        records["z"][cursor:stop] = points[:, 2]

        spacing = math.sqrt(max(areas[face_index], 1e-12) / max(int(face_count), 1))
        tangent_sigma = float(np.clip(spacing * 0.56, 0.0028, 0.024))
        normal_sigma = max(tangent_sigma * 0.10, 0.0008)
        records["scale_0"][cursor:stop] = math.log(tangent_sigma)
        records["scale_1"][cursor:stop] = math.log(tangent_sigma)
        records["scale_2"][cursor:stop] = math.log(normal_sigma)

        colors = np.repeat(color_values[[face_index]], int(face_count), axis=0)
        if uv_values is not None:
            sampled_uv = barycentric @ uv_values[face_index]
            association_uv[cursor:stop] = sampled_uv
            if base_color_texture is not None:
                colors = _sample_texture(base_color_texture, sampled_uv)
        records["f_dc_0"][cursor:stop] = (colors[:, 0] - 0.5) / c0
        records["f_dc_1"][cursor:stop] = (colors[:, 1] - 0.5) / c0
        records["f_dc_2"][cursor:stop] = (colors[:, 2] - 0.5) / c0
        opacity = float(opacities[face_index])
        records["opacity"][cursor:stop] = math.log(opacity / (1.0 - opacity))
        quaternion = quaternions[face_index]
        records["rot_0"][cursor:stop] = quaternion[0]
        records["rot_1"][cursor:stop] = quaternion[1]
        records["rot_2"][cursor:stop] = quaternion[2]
        records["rot_3"][cursor:stop] = quaternion[3]
        association_faces[cursor:stop] = face_index
        association_barycentric[cursor:stop] = barycentric
        cursor = stop

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(output, records)
    sidecar = (
        Path(association_path).resolve()
        if association_path is not None
        else output.with_suffix(".mesh-bindings.npz")
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar,
        schema_version=np.asarray([1], dtype=np.uint32),
        face_indices=association_faces,
        barycentric=association_barycentric,
        uv=association_uv,
        mesh_vertices=vertex_values.astype(np.float32),
        mesh_faces=face_values.astype(np.uint32),
        face_normals=normals.astype(np.float32),
        face_frames=frames.astype(np.float32),
        face_groups=group_values,
    )
    return {
        "ply": output,
        "associations": sidecar,
        "gaussian_count": int(count),
        "face_count": int(len(face_values)),
        "vertex_count": int(len(vertex_values)),
        "face_area": float(areas.sum()),
    }


def make_drawer_fixture(path: Path, *, seed: int = 7) -> dict[str, Bounds]:
    """Write a deterministic colored Gaussian cabinet, drawer, door, and floor."""
    rng = np.random.default_rng(seed)
    components = {
        "floor": (Bounds((-3.0, -0.15, -2.0), (3.0, 0.0, 2.0)), (0.25, 0.25, 0.28), 900),
        "cabinet": (Bounds((-1.2, 0.0, -0.6), (1.2, 2.0, 0.6)), (0.45, 0.22, 0.08), 1200),
        "drawer": (Bounds((0.1, 0.55, -0.75), (1.0, 1.15, -0.45)), (0.72, 0.22, 0.08), 650),
        "door": (Bounds((-1.25, 0.05, -0.68), (-0.05, 1.95, -0.48)), (0.12, 0.35, 0.78), 650),
        "obstacle": (Bounds((1.65, 0.0, -0.4), (2.25, 0.8, 0.4)), (0.1, 0.45, 0.75), 350),
    }
    dtype = np.dtype(
        [(name, "<f4") for name in (
            "x", "y", "z",
            "scale_0", "scale_1", "scale_2",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "rot_0", "rot_1", "rot_2", "rot_3",
        )]
    )
    records = np.zeros(sum(item[2] for item in components.values()), dtype=dtype)
    cursor = 0
    c0 = 0.28209479177387814
    for bounds, color, count in components.values():
        points = rng.uniform(bounds.minimum, bounds.maximum, size=(count, 3)).astype(np.float32)
        # Bias samples toward box surfaces so each component reads as a solid part.
        surface_axis = rng.integers(0, 3, size=count)
        surface_side = rng.integers(0, 2, size=count)
        for index in range(count):
            points[index, surface_axis[index]] = (
                bounds.minimum[surface_axis[index]]
                if surface_side[index] == 0
                else bounds.maximum[surface_axis[index]]
            )
        records["x"][cursor:cursor + count] = points[:, 0]
        records["y"][cursor:cursor + count] = points[:, 1]
        records["z"][cursor:cursor + count] = points[:, 2]
        for name in ("scale_0", "scale_1", "scale_2"):
            records[name][cursor:cursor + count] = math.log(0.035)
        records["f_dc_0"][cursor:cursor + count] = (color[0] - 0.5) / c0
        records["f_dc_1"][cursor:cursor + count] = (color[1] - 0.5) / c0
        records["f_dc_2"][cursor:cursor + count] = (color[2] - 0.5) / c0
        records["opacity"][cursor:cursor + count] = 3.5
        records["rot_0"][cursor:cursor + count] = 1.0
        cursor += count
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_binary_ply(path, records)
    return {name: item[0] for name, item in components.items()}
