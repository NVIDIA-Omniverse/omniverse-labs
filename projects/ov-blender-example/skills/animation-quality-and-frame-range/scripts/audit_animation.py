#!/usr/bin/env python3
"""Read-only Blender animation audit for named objects and RNA properties."""

from __future__ import annotations

import argparse
import json
import math
import sys

import bpy


SCHEMA = "blender.animation-audit/v1"


def _script_args():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _parse_request():
    if "ANIMATION_AUDIT_REQUEST" in globals():
        request = dict(globals()["ANIMATION_AUDIT_REQUEST"])
        return request, True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument(
        "--properties", nargs="+", default=["location", "rotation_euler", "scale"]
    )
    parser.add_argument("--frames", nargs="+", type=float)
    parser.add_argument("--require-keyframes", action="store_true")
    parser.add_argument("--require-motion", action="store_true")
    parser.add_argument("--fail-on-frame-handlers", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    args = parser.parse_args(_script_args())
    return vars(args), False


def _finite_values(value):
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        raise TypeError(f"property is not a numeric scalar or vector: {type(value).__name__}")
    return values


def _resolve_property(obj, evaluated_obj, path):
    if path.startswith("shape_keys."):
        keys = obj.data.shape_keys if getattr(obj, "data", None) else None
        if keys is None:
            raise ValueError("object has no shape keys")
        return keys.path_resolve(path[len("shape_keys.") :])
    if path.startswith("data."):
        if getattr(evaluated_obj, "data", None) is None:
            raise ValueError("object has no data datablock")
        return evaluated_obj.data.path_resolve(path[len("data.") :])
    return evaluated_obj.path_resolve(path)


def _curves_for_animdata(animdata):
    if animdata is None or animdata.action is None:
        return []
    action = animdata.action
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    curves = []
    slot = getattr(animdata, "action_slot", None)
    for layer in action.layers:
        for strip in layer.strips:
            bag = None
            if slot is not None and hasattr(strip, "channelbag"):
                bag = strip.channelbag(slot, ensure=False)
            if bag is not None:
                curves.extend(bag.fcurves)
    return curves


def _animation_sources(obj):
    sources = []
    candidates = [("object", obj)]
    data = getattr(obj, "data", None)
    if data is not None:
        candidates.append(("data", data))
        if getattr(data, "shape_keys", None) is not None:
            candidates.append(("shape_keys", data.shape_keys))
    for owner, datablock in candidates:
        animdata = getattr(datablock, "animation_data", None)
        curves = _curves_for_animdata(animdata)
        curve_records = []
        for curve in curves:
            curve_records.append(
                {
                    "data_path": curve.data_path,
                    "array_index": curve.array_index,
                    "keyframes": sorted(float(point.co.x) for point in curve.keyframe_points),
                }
            )
        keyframes = [frame for curve in curve_records for frame in curve["keyframes"]]
        sources.append(
            {
                "owner": owner,
                "action": animdata.action.name if animdata and animdata.action else None,
                "slot": getattr(getattr(animdata, "action_slot", None), "identifier", None),
                "curve_paths": sorted({curve.data_path for curve in curves}),
                "curves": curve_records,
                "keyframes": sorted(set(keyframes)),
                "drivers": len(animdata.drivers) if animdata else 0,
                "nla_tracks": len(animdata.nla_tracks) if animdata else 0,
            }
        )
    return sources


def _default_frames(scene):
    start, end = float(scene.frame_start), float(scene.frame_end)
    return [start] if start == end else [start, (start + end) / 2.0, end]


def _set_frame(scene, frame):
    whole = math.floor(frame)
    scene.frame_set(whole, subframe=frame - whole)
    bpy.context.view_layer.update()


def audit(request):
    scene = bpy.context.scene
    names = list(dict.fromkeys(request.get("objects") or []))
    properties = list(dict.fromkeys(request.get("properties") or ["location"]))
    frames = sorted(set(float(frame) for frame in (request.get("frames") or _default_frames(scene))))
    epsilon = float(request.get("epsilon", 1.0e-6))
    require_keyframes = bool(request.get("require_keyframes", False))
    require_motion = bool(request.get("require_motion", False))
    fail_on_handlers = bool(request.get("fail_on_frame_handlers", False))

    checks = {
        "request_valid": bool(names and properties and frames and epsilon >= 0.0),
        "objects_exist": True,
        "properties_resolve": True,
        "samples_finite": True,
        "keyframes_present": True,
        "motion_present": True,
        "frame_handlers_allowed": True,
    }
    missing = [name for name in names if bpy.data.objects.get(name) is None]
    checks["objects_exist"] = not missing
    handler_names = [
        f"{kind}:{getattr(handler, '__name__', type(handler).__name__)}"
        for kind, handlers in (
            ("pre", bpy.app.handlers.frame_change_pre),
            ("post", bpy.app.handlers.frame_change_post),
        )
        for handler in handlers
    ]
    if fail_on_handlers:
        checks["frame_handlers_allowed"] = not handler_names

    records = []
    original_frame = scene.frame_current_final
    sampled_any = False
    try:
        for name in ([] if fail_on_handlers and handler_names else names):
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            sources = _animation_sources(obj)
            keyed_curves = {
                (source["owner"], curve["data_path"]): curve["keyframes"]
                for source in sources
                for curve in source["curves"]
            }
            samples = []
            errors = []
            for frame in frames:
                sampled_any = True
                _set_frame(scene, frame)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                evaluated = obj.evaluated_get(depsgraph)
                values = {}
                for path in properties:
                    try:
                        numeric = _finite_values(_resolve_property(obj, evaluated, path))
                        if not all(math.isfinite(value) for value in numeric):
                            checks["samples_finite"] = False
                            errors.append({"frame": frame, "property": path, "error": "non-finite"})
                            values[path] = None
                        else:
                            values[path] = numeric
                    except Exception as exc:
                        checks["properties_resolve"] = False
                        errors.append({"frame": frame, "property": path, "error": str(exc)})
                world_matrix = [float(value) for row in evaluated.matrix_world for value in row]
                if not all(math.isfinite(value) for value in world_matrix):
                    checks["samples_finite"] = False
                    errors.append({"frame": frame, "property": "matrix_world", "error": "non-finite"})
                    world_matrix = None
                samples.append(
                    {"frame": frame, "values": values, "world_matrix": world_matrix}
                )

            deltas = {}
            for path in properties:
                series = [sample["values"].get(path) for sample in samples]
                valid = [value for value in series if value is not None]
                deltas[path] = max(
                    (max(abs(a - b) for a, b in zip(left, right)) for left, right in zip(valid, valid[1:])),
                    default=0.0,
                )
            matrices = [sample["world_matrix"] for sample in samples if sample["world_matrix"] is not None]
            world_matrix_delta = max(
                (
                    max(abs(a - b) for a, b in zip(left, right))
                    for left, right in zip(matrices, matrices[1:])
                ),
                default=0.0,
            )
            has_motion = world_matrix_delta > epsilon or any(
                delta > epsilon for delta in deltas.values()
            )
            requested_keyed_paths = {}
            for path in properties:
                if path.startswith("shape_keys."):
                    key = ("shape_keys", path[len("shape_keys.") :])
                elif path.startswith("data."):
                    key = ("data", path[len("data.") :])
                else:
                    key = ("object", path)
                requested_keyed_paths[path] = any(
                    frames[0] <= frame <= frames[-1] for frame in keyed_curves.get(key, [])
                )
            has_keys = all(requested_keyed_paths.values())
            if require_motion and not has_motion:
                checks["motion_present"] = False
            if require_keyframes and not has_keys:
                checks["keyframes_present"] = False
            records.append(
                {
                    "object": name,
                    "type": obj.type,
                    "sources": sources,
                    "requested_keyed_paths": requested_keyed_paths,
                    "sampled_deltas": deltas,
                    "world_matrix_delta": world_matrix_delta,
                    "has_sampled_motion": has_motion,
                    "samples": samples,
                    "errors": errors,
                }
            )
    finally:
        if sampled_any:
            _set_frame(scene, original_frame)

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": SCHEMA,
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "request": {
            "objects": names,
            "properties": properties,
            "frames": frames,
            "require_keyframes": require_keyframes,
            "require_motion": require_motion,
            "epsilon": epsilon,
        },
        "scene": {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "fps": scene.render.fps,
            "fps_base": scene.render.fps_base,
        },
        "missing_objects": missing,
        "frame_handlers": handler_names,
        "objects": records,
    }


def main():
    request, mcp_mode = _parse_request()
    report = audit(request)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not mcp_mode and report["status"] != "pass":
        raise SystemExit(2)


if "ANIMATION_AUDIT_REQUEST" in globals():
    print(json.dumps(audit(ANIMATION_AUDIT_REQUEST), indent=2, sort_keys=True))
elif __name__ == "__main__":
    main()
