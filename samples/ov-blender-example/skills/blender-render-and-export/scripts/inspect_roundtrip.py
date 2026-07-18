#!/usr/bin/env python3
"""Import one GLB/USD in a disposable Blender process and print a JSON audit."""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args(raw)


def main():
    args = _args()
    source = args.input
    if not source.is_absolute() or source.suffix.lower() not in {".glb", ".usd", ".usda", ".usdc"}:
        raise ValueError("--input must be an absolute GLB or USD path")
    if not source.is_file():
        raise FileNotFoundError(source)
    # Destructive by design: this helper is only for a fresh disposable process.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source.suffix.lower() == ".glb":
        if not bpy.ops.import_scene.gltf.poll():
            raise RuntimeError("GLTF import operator is unavailable")
        result = bpy.ops.import_scene.gltf(filepath=str(source))
    else:
        if not bpy.ops.wm.usd_import.poll():
            raise RuntimeError("USD import operator is unavailable")
        result = bpy.ops.wm.usd_import(filepath=str(source))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects, invalid = [], []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        bounds = None
        if obj.type not in {"EMPTY", "CAMERA", "LIGHT", "SPEAKER"}:
            evaluated = obj.evaluated_get(depsgraph)
            points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
            if points and all(math.isfinite(float(v)) for point in points for v in point):
                bounds = {
                    "min": [min(float(point[i]) for point in points) for i in range(3)],
                    "max": [max(float(point[i]) for point in points) for i in range(3)],
                }
            else:
                invalid.append(obj.name)
        objects.append({
            "name": obj.name,
            "type": obj.type,
            "bounds": bounds,
            "materials": [slot.material.name for slot in obj.material_slots if slot.material],
            "action": obj.animation_data.action.name if obj.animation_data and obj.animation_data.action else None,
        })
    report = {
        "schema": "blender_roundtrip_inspection.v1",
        "ok": bool(objects) and not invalid and "FINISHED" in result,
        "source": str(source),
        "operator_result": sorted(result),
        "scene_unit": {
            "system": bpy.context.scene.unit_settings.system,
            "scale_length": bpy.context.scene.unit_settings.scale_length,
        },
        "objects": objects,
        "invalid_bounds": invalid,
        "actions": sorted(action.name for action in bpy.data.actions),
    }
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
