#!/usr/bin/env python3
"""Read-only JSON inspection of a USD stage using Blender's bundled pxr."""

import argparse
import json
import math
from pathlib import Path
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils


def _finite_matrix(matrix):
    return all(math.isfinite(float(matrix[row][column])) for row in range(4) for column in range(4))


def inspect_stage(stage_path, require_resolved=False):
    path = Path(stage_path)
    if not path.is_absolute():
        raise ValueError("stage path must be absolute")
    if not path.is_file():
        raise FileNotFoundError(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("USD stage failed to open")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(path)))
    prim_paths = set()
    duplicate_paths = []
    nonfinite_xforms = []
    mesh_issues = []
    type_counts = {}
    active = 0
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim_path in prim_paths:
            duplicate_paths.append(prim_path)
        prim_paths.add(prim_path)
        type_name = prim.GetTypeName() or "untyped"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        active += int(prim.IsActive())
        if prim.IsA(UsdGeom.Xformable):
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if not _finite_matrix(matrix):
                nonfinite_xforms.append(prim_path)
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get() or []
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            indices = mesh.GetFaceVertexIndicesAttr().Get() or []
            issues = []
            if sum(int(value) for value in counts) != len(indices):
                issues.append("face_count_index_length_mismatch")
            if indices and (min(indices) < 0 or max(indices) >= len(points)):
                issues.append("face_vertex_index_out_of_range")
            if any(not all(math.isfinite(float(axis)) for axis in point) for point in points):
                issues.append("nonfinite_points")
            if issues:
                mesh_issues.append({"prim": prim_path, "issues": issues})

    default_prim = stage.GetDefaultPrim()
    checks = {"opens": True,
              "dependencies_resolved": not unresolved if require_resolved else True,
              "prim_paths_unique": not duplicate_paths,
              "transforms_finite": not nonfinite_xforms,
              "mesh_topology_valid": not mesh_issues}
    return {"schema": "usd_stage_inspection.v1", "ok": all(checks.values()),
            "stage": str(path), "checks": checks,
            "metadata": {"default_prim": str(default_prim.GetPath()) if default_prim else None,
                         "up_axis": UsdGeom.GetStageUpAxis(stage),
                         "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                         "start_time_code": stage.GetStartTimeCode(),
                         "end_time_code": stage.GetEndTimeCode(),
                         "time_codes_per_second": stage.GetTimeCodesPerSecond()},
            "composition": {"layers": [layer.identifier for layer in layers],
                            "assets": [asset.path for asset in assets],
                            "unresolved": [asset.path for asset in unresolved]},
            "prims": {"count": len(prim_paths), "active": active,
                      "type_counts": type_counts, "duplicate_paths": duplicate_paths},
            "nonfinite_xforms": nonfinite_xforms, "mesh_issues": mesh_issues}


def _args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--require-resolved", action="store_true")
    return parser.parse_args(raw)


if __name__ == "__main__":
    args = _args()
    try:
        result = inspect_stage(args.stage, args.require_resolved)
    except Exception as error:
        result = {"schema": "usd_stage_inspection.v1", "ok": False,
                  "error_type": type(error).__name__, "error": str(error)}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)
