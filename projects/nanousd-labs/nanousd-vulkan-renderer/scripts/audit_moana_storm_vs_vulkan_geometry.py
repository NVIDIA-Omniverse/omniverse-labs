#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit Moana geometry visible to Storm against Vulkan scene_load output.

Storm visibility is approximated with OpenUSD traversal using
Usd.TraverseInstanceProxies(), inherited visibility, and PointInstancer
invisibleIds. The Vulkan side uses the local libnusd_renderer scene_load entry
point through ctypes and summarizes loaded meshes plus compact instance batches.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = Path("$HOME/moana-island-scene-usd/island/usd/island.usda")
DEFAULT_PXR_PATH = Path("$HOME/OpenUSD_install/lib/python")
DEFAULT_RENDERER_LIB = REPO_ROOT / "build" / "libnusd_renderer.so"

VULKAN_ENV_DEFAULTS = {
    "NUSD_ENABLE_MATERIALS": "0",
    "NUSD_ENABLE_PTEX_MATERIALS": "0",
    "NUSD_CLAY_VIZ": "0",
    "NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL": "1",
    "NUSD_RENDER_PI_BATCHES": "1",
    "NUSD_NATIVE_ARC_CHASE_AFTER_DIRECT": "1",
    "NUSD_NATIVE_CURVES": "all",
    "NUSD_CURVE_SUBSEGS": "1",
}


@dataclass
class StormCounts:
    mesh: int = 0
    curves: int = 0
    point_instancers: int = 0
    point_instancer_instances: int = 0
    other_boundable: dict[str, int] = field(default_factory=dict)


@dataclass
class VulkanCounts:
    meshes: int = 0
    proto_only_meshes: int = 0
    curves: int = 0
    pi_batches: int = 0
    pi_transforms: int = 0
    point_instancer_batches: int = 0
    native_instance_batches: int = 0
    point_instancer_transform_slices: int = 0
    native_instance_transform_slices: int = 0
    point_instancer_transform_refs: int = 0
    native_instance_transform_refs: int = 0


def asset_key(path: str | None) -> str:
    if not path:
        return "<unknown>"
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "island":
        return parts[1]
    if parts:
        return parts[0]
    return "<root>"


def _add_default_pxr_path(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def _visible_instance_count(pi: Any) -> int:
    ids = pi.GetIdsAttr().Get()
    proto_indices = pi.GetProtoIndicesAttr().Get()
    positions = pi.GetPositionsAttr().Get()
    if ids is not None:
        count = len(ids)
    elif proto_indices is not None:
        count = len(proto_indices)
    elif positions is not None:
        count = len(positions)
    else:
        count = 0

    invisible = pi.GetInvisibleIdsAttr().Get()
    if invisible is None or count <= 0:
        return count
    if ids is not None:
        visible_ids = set(int(x) for x in ids)
        hidden = sum(1 for x in invisible if int(x) in visible_ids)
    else:
        hidden = sum(1 for x in invisible if 0 <= int(x) < count)
    return max(0, count - hidden)


def collect_storm_counts(usd_path: Path, pxr_path: Path) -> dict[str, Any]:
    _add_default_pxr_path(pxr_path)
    from pxr import Usd, UsdGeom  # type: ignore

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    per_asset: dict[str, StormCounts] = defaultdict(StormCounts)
    total = StormCounts()

    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        path = str(prim.GetPath())
        key = asset_key(path)
        type_name = prim.GetTypeName() or "<none>"

        if prim.IsA(UsdGeom.Mesh):
            per_asset[key].mesh += 1
            total.mesh += 1
        if prim.IsA(UsdGeom.BasisCurves):
            per_asset[key].curves += 1
            total.curves += 1
        if prim.IsA(UsdGeom.PointInstancer):
            ninst = _visible_instance_count(UsdGeom.PointInstancer(prim))
            per_asset[key].point_instancers += 1
            per_asset[key].point_instancer_instances += ninst
            total.point_instancers += 1
            total.point_instancer_instances += ninst
        if prim.IsA(UsdGeom.Boundable) and type_name not in (
            "Mesh",
            "BasisCurves",
            "PointInstancer",
        ):
            per_asset[key].other_boundable[type_name] = (
                per_asset[key].other_boundable.get(type_name, 0) + 1
            )
            total.other_boundable[type_name] = total.other_boundable.get(type_name, 0) + 1

    return {
        "total": dataclass_to_dict(total),
        "assets": {k: dataclass_to_dict(v) for k, v in sorted(per_asset.items())},
    }


class SceneMesh(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.POINTER(ctypes.c_float)),
        ("normals", ctypes.POINTER(ctypes.c_float)),
        ("colors", ctypes.POINTER(ctypes.c_float)),
        ("texcoords", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_uint32)),
        ("ptex_tri_colors", ctypes.POINTER(ctypes.c_uint32)),
        ("ptex_tri_color_count", ctypes.c_int),
        ("nvertices", ctypes.c_int),
        ("nindices", ctypes.c_int),
        ("world_xform", ctypes.c_double * 16),
        ("bounds_min", ctypes.c_float * 3),
        ("bounds_max", ctypes.c_float * 3),
        ("local_bounds_min", ctypes.c_float * 3),
        ("local_bounds_max", ctypes.c_float * 3),
        ("display_color", ctypes.c_float * 3),
        ("has_display_color", ctypes.c_int),
        ("material_index", ctypes.c_int),
        ("path", ctypes.c_char_p),
        ("vertex_offset", ctypes.c_uint32),
        ("index_offset", ctypes.c_uint32),
        ("prototype_idx", ctypes.c_int),
        ("is_proto_only", ctypes.c_int),
        ("lazy_prim_idx", ctypes.c_int),
        ("meshlet_offset", ctypes.c_uint32),
        ("meshlet_count", ctypes.c_uint32),
    ]


class SceneInstanceBatch(ctypes.Structure):
    _fields_ = [
        ("prototype_mesh_idx", ctypes.c_int),
        ("transform_offset", ctypes.c_uint32),
        ("transform_count", ctypes.c_uint32),
        ("source_prim_idx", ctypes.c_int),
        ("material_or_binding_id", ctypes.c_int),
        ("source_kind", ctypes.c_int),
    ]


class SceneInstanceTransform(ctypes.Structure):
    _fields_ = [("m", ctypes.c_float * 12)]


class Scene(ctypes.Structure):
    _fields_ = [
        ("meshes", ctypes.POINTER(SceneMesh)),
        ("nmeshes", ctypes.c_int),
        ("curves", ctypes.c_void_p),
        ("ncurves", ctypes.c_int),
        ("pi_batches", ctypes.POINTER(SceneInstanceBatch)),
        ("npi_batches", ctypes.c_int),
        ("pi_transforms", ctypes.POINTER(SceneInstanceTransform)),
        ("npi_transforms", ctypes.c_uint64),
    ]


def _decode_path(raw: bytes | None) -> str | None:
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace")


def configure_vulkan_env(metal_parity_native_arc: bool) -> dict[str, str]:
    for key, value in VULKAN_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if metal_parity_native_arc:
        os.environ["NUSD_METAL_PARITY_ARC_SEED"] = "1"
    return {
        key: os.environ.get(key, "")
        for key in sorted(
            set(VULKAN_ENV_DEFAULTS)
            | {"NUSD_METAL_PARITY_ARC_SEED", "NUSD_NATIVE_ARC_CHASE_AFTER_DIRECT"}
        )
    }


def collect_vulkan_counts(
    usd_path: Path, renderer_lib: Path, metal_parity_native_arc: bool
) -> dict[str, Any]:
    if not renderer_lib.exists():
        raise RuntimeError(f"Renderer library not found: {renderer_lib}")
    env = configure_vulkan_env(metal_parity_native_arc)
    lib = ctypes.CDLL(str(renderer_lib))
    lib.scene_load.argtypes = [ctypes.c_char_p]
    lib.scene_load.restype = ctypes.POINTER(Scene)
    lib.scene_free.argtypes = [ctypes.POINTER(Scene)]
    lib.scene_free.restype = None
    if hasattr(lib, "scene_set_load_materials"):
        lib.scene_set_load_materials.argtypes = [ctypes.c_int]
        lib.scene_set_load_materials.restype = None
        lib.scene_set_load_materials(0)

    t0 = time.perf_counter()
    scene_ptr = lib.scene_load(str(usd_path).encode("utf-8"))
    load_seconds = time.perf_counter() - t0
    if not scene_ptr:
        raise RuntimeError(f"scene_load failed: {usd_path}")

    try:
        scene = scene_ptr.contents
        per_asset: dict[str, VulkanCounts] = defaultdict(VulkanCounts)
        total = VulkanCounts(
            meshes=int(scene.nmeshes),
            curves=int(scene.ncurves),
            pi_batches=int(scene.npi_batches),
            pi_transforms=int(scene.npi_transforms),
        )

        mesh_paths: list[str | None] = []
        for i in range(scene.nmeshes):
            mesh = scene.meshes[i]
            path = _decode_path(mesh.path)
            mesh_paths.append(path)
            key = asset_key(path)
            per_asset[key].meshes += 1
            if mesh.is_proto_only:
                per_asset[key].proto_only_meshes += 1
                total.proto_only_meshes += 1

        point_slices: dict[str, set[tuple[int, int]]] = defaultdict(set)
        native_slices: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for i in range(scene.npi_batches):
            batch = scene.pi_batches[i]
            if 0 <= batch.prototype_mesh_idx < len(mesh_paths):
                key = asset_key(mesh_paths[batch.prototype_mesh_idx])
            else:
                key = "<bad-prototype>"
            if batch.source_kind == 0:
                per_asset[key].point_instancer_batches += 1
                per_asset[key].point_instancer_transform_refs += int(batch.transform_count)
                total.point_instancer_batches += 1
                total.point_instancer_transform_refs += int(batch.transform_count)
                point_slices[key].add((int(batch.transform_offset), int(batch.transform_count)))
            else:
                per_asset[key].native_instance_batches += 1
                per_asset[key].native_instance_transform_refs += int(batch.transform_count)
                total.native_instance_batches += 1
                total.native_instance_transform_refs += int(batch.transform_count)
                native_slices[key].add((int(batch.transform_offset), int(batch.transform_count)))

        for key, slices in point_slices.items():
            n = sum(count for _, count in slices)
            per_asset[key].point_instancer_transform_slices = n
            total.point_instancer_transform_slices += n
        for key, slices in native_slices.items():
            n = sum(count for _, count in slices)
            per_asset[key].native_instance_transform_slices = n
            total.native_instance_transform_slices += n

        return {
            "load_seconds": load_seconds,
            "environment": env,
            "total": dataclass_to_dict(total),
            "assets": {k: dataclass_to_dict(v) for k, v in sorted(per_asset.items())},
        }
    finally:
        lib.scene_free(scene_ptr)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in obj.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(obj, name)
        if isinstance(value, defaultdict):
            value = dict(value)
        out[name] = value
    return out


def build_comparison(storm: dict[str, Any], vulkan: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assets = set(storm["assets"])
    if vulkan:
        assets |= set(vulkan["assets"])
    for asset in sorted(assets):
        s = storm["assets"].get(asset, {})
        v = (vulkan or {}).get("assets", {}).get(asset, {})
        rows.append(
            {
                "asset": asset,
                "storm_mesh": int(s.get("mesh", 0)),
                "vulkan_mesh": int(v.get("meshes", 0)),
                "mesh_delta": int(s.get("mesh", 0)) - int(v.get("meshes", 0)),
                "storm_curves": int(s.get("curves", 0)),
                "storm_pi_instances": int(s.get("point_instancer_instances", 0)),
                "vulkan_pi_unique_transforms": int(v.get("point_instancer_transform_slices", 0)),
                "pi_delta": int(s.get("point_instancer_instances", 0))
                - int(v.get("point_instancer_transform_slices", 0)),
                "vulkan_native_unique_transforms": int(v.get("native_instance_transform_slices", 0)),
                "vulkan_proto_only_meshes": int(v.get("proto_only_meshes", 0)),
            }
        )
    return rows


def render_markdown(report: dict[str, Any], asset_filter: set[str] | None) -> str:
    rows = report["comparison"]
    if asset_filter:
        rows = [r for r in rows if r["asset"] in asset_filter]
    rows = sorted(
        rows,
        key=lambda r: (
            abs(int(r["pi_delta"])),
            abs(int(r["mesh_delta"])),
            int(r["storm_pi_instances"]),
        ),
        reverse=True,
    )

    lines = [
        "# Moana Storm vs Vulkan Geometry Audit",
        "",
        f"- USD: `{report['usd']}`",
        f"- Vulkan load: {'yes' if report.get('vulkan') else 'no'}",
    ]
    if report.get("vulkan"):
        lines.append(f"- Vulkan load seconds: {report['vulkan']['load_seconds']:.2f}")
    lines.extend(
        [
            "",
            "## Totals",
            "",
            "```json",
            json.dumps({"storm": report["storm"]["total"], "vulkan": (report.get("vulkan") or {}).get("total")}, indent=2),
            "```",
            "",
            "## Largest Asset Deltas",
            "",
            "| asset | Storm mesh | Vulkan mesh | mesh delta | Storm PI inst | Vulkan PI unique | PI delta | Vulkan native unique | proto-only meshes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if (
            row["storm_mesh"] == 0
            and row["storm_pi_instances"] == 0
            and row["vulkan_mesh"] == 0
            and row["vulkan_pi_unique_transforms"] == 0
        ):
            continue
        lines.append(
            "| {asset} | {storm_mesh} | {vulkan_mesh} | {mesh_delta} | "
            "{storm_pi_instances} | {vulkan_pi_unique_transforms} | {pi_delta} | "
            "{vulkan_native_unique_transforms} | {vulkan_proto_only_meshes} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--pxr-path", type=Path, default=DEFAULT_PXR_PATH)
    parser.add_argument("--renderer-lib", type=Path, default=DEFAULT_RENDERER_LIB)
    parser.add_argument("--run-vulkan", action="store_true", help="Run libnusd_renderer scene_load.")
    parser.add_argument(
        "--metal-parity-native-arc",
        action="store_true",
        help="Restore the old Metal-parity shortcut that skips child arcs once direct meshes are found.",
    )
    parser.add_argument("--asset", action="append", help="Limit markdown rows to an asset name.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full JSON report instead of markdown.")
    args = parser.parse_args()

    storm = collect_storm_counts(args.usd, args.pxr_path)
    vulkan = None
    if args.run_vulkan:
        vulkan = collect_vulkan_counts(args.usd, args.renderer_lib, args.metal_parity_native_arc)

    report = {
        "usd": str(args.usd),
        "storm": storm,
        "vulkan": vulkan,
        "comparison": build_comparison(storm, vulkan),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report, set(args.asset) if args.asset else None)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
