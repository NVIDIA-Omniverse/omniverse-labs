#!/usr/bin/env python3
"""Fit evaluated Blender object bounds into a render camera.

Blender CLI:
  blender --background scene.blend --python camera_framing.py -- \
    --camera Camera --objects Subject --margin 0.08

Blender MCP: prepend ``MCP_CONFIG = {...}`` to this complete file. The script
detects that global, runs once, and prints one structured JSON result.
"""

import argparse
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector


SCHEMA = "blender.camera-frame/v1"
NON_GEOMETRY_TYPES = {"EMPTY", "CAMERA", "LIGHT", "SPEAKER"}


def _vec(value):
    return [float(component) for component in value]


def _matrix(value):
    return [[float(component) for component in row] for row in value]


def _finite_vector(value):
    return all(math.isfinite(float(component)) for component in value)


def _object_key(obj):
    original = getattr(obj, "original", None) or obj
    return (original.name_full, original.library.filepath if original.library else None)


def _bbox_points(obj, matrix_world):
    if getattr(obj, "type", None) in NON_GEOMETRY_TYPES:
        return []
    bound_box = getattr(obj, "bound_box", None)
    if not bound_box or len(bound_box) != 8:
        return []
    points = []
    for corner in bound_box:
        point = matrix_world @ Vector(corner)
        if not _finite_vector(point):
            raise ValueError("non-finite evaluated bound for %s" % obj.name_full)
        points.append(point)
    return points


def _expand_objects(objects, include_descendants):
    expanded = []
    seen = set()

    def add(obj):
        key = _object_key(obj)
        if key in seen:
            return
        seen.add(key)
        expanded.append(obj)
        if include_descendants:
            for child in obj.children:
                add(child)

    for obj in objects:
        add(obj)
    return expanded


def evaluated_world_bounds(object_names, depsgraph, include_descendants=True,
                           include_instances=True):
    requested = list(dict.fromkeys(str(name) for name in object_names))
    resolved = []
    missing = []
    for name in requested:
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
        else:
            resolved.append(obj)
    if missing:
        raise ValueError("missing subject object(s): %s" % ", ".join(missing))

    expanded = _expand_objects(resolved, include_descendants)
    included_keys = {_object_key(obj) for obj in expanded}
    points = []
    direct_records = []
    instance_records = []

    for obj in expanded:
        evaluated = obj.evaluated_get(depsgraph)
        bound_points = _bbox_points(evaluated, evaluated.matrix_world)
        if bound_points:
            points.extend(bound_points)
            direct_records.append({
                "name": obj.name_full,
                "type": obj.type,
                "point_count": len(bound_points),
            })

    if include_instances:
        for instance in depsgraph.object_instances:
            if not instance.is_instance:
                continue
            source = instance.object
            parent = instance.parent
            source_key = _object_key(source) if source else None
            parent_key = _object_key(parent) if parent else None
            if source_key not in included_keys and parent_key not in included_keys:
                continue
            bound_points = _bbox_points(source, instance.matrix_world)
            if not bound_points:
                continue
            points.extend(bound_points)
            instance_records.append({
                "source": source.name_full,
                "parent": parent.name_full if parent else None,
                "point_count": len(bound_points),
            })

    if not points:
        raise ValueError("subject has no usable evaluated geometric bounds")
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return {
        "points": points,
        "requested": requested,
        "resolved": [obj.name_full for obj in resolved],
        "expanded": [obj.name_full for obj in expanded],
        "direct_records": direct_records,
        "instance_records": instance_records,
        "minimum": minimum,
        "maximum": maximum,
        "center": (minimum + maximum) * 0.5,
    }


def _safe_edges(camera_data, scene, margin, normalized_ortho=False):
    previous_scale = camera_data.ortho_scale
    try:
        if normalized_ortho:
            camera_data.ortho_scale = 1.0
        frame = camera_data.view_frame(scene=scene)
    finally:
        camera_data.ortho_scale = previous_scale
    if camera_data.type == "PERSP":
        xs = [point.x / -point.z for point in frame]
        ys = [point.y / -point.z for point in frame]
    else:
        xs = [point.x for point in frame]
        ys = [point.y for point in frame]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    width = right - left
    height = top - bottom
    safe = (
        left + margin * width,
        right - margin * width,
        bottom + margin * height,
        top - margin * height,
    )
    if not (safe[0] < 0.0 < safe[1] and safe[2] < 0.0 < safe[3]):
        raise ValueError("camera shift and margin place the fixed look axis outside the safe frame")
    return safe


def _orientation(view_from, roll_degrees):
    back = Vector(view_from)
    if not _finite_vector(back) or back.length <= 1.0e-12:
        raise ValueError("view_from must be a finite non-zero vector")
    back.normalize()
    rotation = (-back).to_track_quat("-Z", "Y")
    if roll_degrees:
        rotation = rotation @ Quaternion((0.0, 0.0, 1.0), math.radians(roll_degrees))
    return back, rotation.normalized()


def _fit_values(points, target, rotation, safe, camera_type):
    inverse = rotation.inverted()
    local = [inverse @ (point - target) for point in points]
    left, right, bottom, top = safe
    if camera_type == "PERSP":
        candidates = []
        for point in local:
            candidates.extend((
                point.z + point.x / right,
                point.z + point.x / left,
                point.z + point.y / top,
                point.z + point.y / bottom,
                point.z + 1.0e-6,
            ))
        return max(max(candidates), 1.0e-6), None, local
    if camera_type == "ORTHO":
        candidates = [1.0e-6]
        for point in local:
            candidates.extend((
                point.x / right,
                point.x / left,
                point.y / top,
                point.y / bottom,
            ))
        scale = max(candidates)
        depth_span = max(point.z for point in local) - min(point.z for point in local)
        distance = max(point.z for point in local) + max(depth_span * 0.1, scale * 0.01, 1.0e-3)
        return max(distance, 1.0e-6), max(scale, 1.0e-6), local
    raise ValueError("unsupported camera type: %s" % camera_type)


def _render_report(scene):
    render = scene.render
    scale = render.resolution_percentage / 100.0
    effective_x = max(1, round(render.resolution_x * scale))
    effective_y = max(1, round(render.resolution_y * scale))
    display_aspect = ((effective_x * render.pixel_aspect_x) /
                      (effective_y * render.pixel_aspect_y))
    return {
        "resolution": [int(render.resolution_x), int(render.resolution_y)],
        "resolution_percentage": int(render.resolution_percentage),
        "effective_resolution": [int(effective_x), int(effective_y)],
        "pixel_aspect": [float(render.pixel_aspect_x), float(render.pixel_aspect_y)],
        "display_aspect": float(display_aspect),
    }


def _camera_report(camera):
    data = camera.data
    return {
        "name": camera.name_full,
        "type": data.type,
        "matrix_world": _matrix(camera.matrix_world),
        "lens": float(data.lens),
        "sensor_fit": data.sensor_fit,
        "shift": [float(data.shift_x), float(data.shift_y)],
        "ortho_scale": float(data.ortho_scale),
        "clip_start": float(data.clip_start),
        "clip_end": float(data.clip_end),
    }


def _verify(scene, camera, points, margin, epsilon=2.0e-5):
    inverse = camera.matrix_world.inverted_safe()
    projected = []
    behind = 0
    outside = 0
    depths = []
    for point in points:
        local = inverse @ point
        depth = -float(local.z)
        depths.append(depth)
        uv = world_to_camera_view(scene, camera, point)
        values = (float(uv.x), float(uv.y), float(uv.z))
        if not all(math.isfinite(value) for value in values):
            outside += 1
            continue
        projected.append(values)
        if depth <= 0.0:
            behind += 1
        if (values[0] < margin - epsilon or values[0] > 1.0 - margin + epsilon or
                values[1] < margin - epsilon or values[1] > 1.0 - margin + epsilon):
            outside += 1
    return {
        "fits": bool(projected) and behind == 0 and outside == 0,
        "safe_uv": [float(margin), float(margin), float(1.0 - margin), float(1.0 - margin)],
        "uv_min": [min(value[0] for value in projected), min(value[1] for value in projected)] if projected else None,
        "uv_max": [max(value[0] for value in projected), max(value[1] for value in projected)] if projected else None,
        "depth_min": min(depths) if depths else None,
        "depth_max": max(depths) if depths else None,
        "outside_count": int(outside),
        "behind_count": int(behind),
        "point_count": len(points),
    }


def frame_camera(camera_name, object_names, *, scene_name=None, frame=None,
                 margin=0.08, include_descendants=True, include_instances=True,
                 target_empty_name=None, target_empty_mode="use", view_from=None,
                 roll_degrees=0.0, clip_policy="fit", clip_padding=0.05,
                 clip_start_min=0.001, clip_start=None, clip_end=None,
                 set_scene_camera=True, constraint_policy="error"):
    """Fit named evaluated objects and return a JSON-serializable report."""
    if not 0.0 <= float(margin) < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")
    if not 0.0 <= float(clip_padding) < 1.0:
        raise ValueError("clip_padding must satisfy 0 <= padding < 1")
    scene = bpy.data.scenes.get(scene_name) if scene_name else bpy.context.scene
    if scene is None:
        raise ValueError("missing scene: %s" % scene_name)
    camera = bpy.data.objects.get(camera_name)
    if camera is None or camera.type != "CAMERA":
        raise ValueError("missing Camera object: %s" % camera_name)
    if camera.data.type not in {"PERSP", "ORTHO"}:
        raise ValueError("only PERSP and ORTHO cameras are supported")
    if constraint_policy == "error" and camera.constraints:
        raise ValueError("camera has constraints; manage them explicitly before fitting")
    if constraint_policy not in {"error", "allow"}:
        raise ValueError("constraint_policy must be 'error' or 'allow'")
    if clip_policy not in {"fit", "preserve", "explicit"}:
        raise ValueError("clip_policy must be fit, preserve, or explicit")
    if target_empty_mode not in {"use", "create_at_bounds", "move_to_bounds"}:
        raise ValueError("invalid target_empty_mode")

    if frame is not None:
        scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bounds = evaluated_world_bounds(
        object_names, depsgraph,
        include_descendants=include_descendants,
        include_instances=include_instances,
    )

    target_empty = bpy.data.objects.get(target_empty_name) if target_empty_name else None
    if not target_empty_name and target_empty_mode != "use":
        raise ValueError("target_empty_mode requires target_empty_name")
    if target_empty_name and target_empty_mode in {"use", "move_to_bounds"} and target_empty is None:
        raise ValueError("missing requested target Empty: %s" % target_empty_name)
    if target_empty_name and target_empty_mode == "create_at_bounds" and target_empty is not None:
        raise ValueError("target Empty already exists: %s" % target_empty_name)
    if target_empty is not None and target_empty.type != "EMPTY":
        raise ValueError("target object is not an Empty: %s" % target_empty_name)
    if target_empty is not None and target_empty_mode == "use":
        target = target_empty.evaluated_get(depsgraph).matrix_world.translation.copy()
        target_source = "empty"
    else:
        target = bounds["center"].copy()
        target_source = "evaluated_bounds_center"

    if view_from is None:
        view_from = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))
    back, rotation = _orientation(view_from, float(roll_degrees))
    safe = _safe_edges(camera.data, scene, float(margin), camera.data.type == "ORTHO")
    distance, ortho_scale, _local = _fit_values(
        bounds["points"], target, rotation, safe, camera.data.type)

    before = _camera_report(camera)
    old_matrix = camera.matrix_world.copy()
    old_ortho = camera.data.ortho_scale
    old_clips = (camera.data.clip_start, camera.data.clip_end)
    old_scene_camera = scene.camera
    created_empty = None
    moved_empty_location = None
    try:
        if target_empty_name and target_empty is None and target_empty_mode == "create_at_bounds":
            target_empty = bpy.data.objects.new(target_empty_name, None)
            scene.collection.objects.link(target_empty)
            target_empty.matrix_world.translation = target
            created_empty = target_empty
        elif target_empty is not None and target_empty_mode == "move_to_bounds":
            moved_empty_location = target_empty.matrix_world.copy()
            target_empty.matrix_world.translation = target

        if ortho_scale is not None:
            camera.data.ortho_scale = ortho_scale
        pose = Matrix.Translation(target + back * distance) @ rotation.to_matrix().to_4x4()
        camera.matrix_world = pose
        if set_scene_camera:
            scene.camera = camera
        bpy.context.view_layer.update()

        preliminary = _verify(scene, camera, bounds["points"], float(margin))
        minimum_depth = preliminary["depth_min"]
        maximum_depth = preliminary["depth_max"]
        if minimum_depth is None or minimum_depth <= 0.0:
            raise RuntimeError("fitted camera left subject behind the camera")
        fitted_start = max(float(clip_start_min), minimum_depth * (1.0 - float(clip_padding)))
        fitted_end = max(fitted_start * 1.01, maximum_depth * (1.0 + float(clip_padding)))
        warnings = []
        if clip_policy == "fit":
            camera.data.clip_start = fitted_start
            camera.data.clip_end = fitted_end
        elif clip_policy == "explicit":
            if clip_start is None or clip_end is None:
                raise ValueError("explicit clip policy requires clip_start and clip_end")
            if float(clip_start) > minimum_depth or float(clip_end) < maximum_depth:
                raise ValueError("explicit clips exclude evaluated subject bounds")
            camera.data.clip_start = float(clip_start)
            camera.data.clip_end = float(clip_end)
        else:
            if camera.data.clip_start > minimum_depth or camera.data.clip_end < maximum_depth:
                warnings.append("preserved clips exclude evaluated subject bounds")
        if camera.data.clip_end / max(camera.data.clip_start, 1.0e-12) > 1.0e7:
            warnings.append("large clip_end/clip_start ratio may reduce depth precision")
        bpy.context.view_layer.update()
        verification = _verify(scene, camera, bounds["points"], float(margin))
        if not verification["fits"]:
            raise RuntimeError("independent camera projection verification failed")

        dimensions = bounds["maximum"] - bounds["minimum"]
        return {
            "schema": SCHEMA,
            "ok": True,
            "operation": "frame_render_camera",
            "blender_version": bpy.app.version_string,
            "scene": scene.name_full,
            "frame": int(scene.frame_current),
            "render": _render_report(scene),
            "selection": {
                "requested": bounds["requested"],
                "resolved": bounds["resolved"],
                "expanded": bounds["expanded"],
                "direct_bounds": bounds["direct_records"],
                "instance_bounds": bounds["instance_records"],
                "point_count": len(bounds["points"]),
            },
            "bounds": {
                "minimum": _vec(bounds["minimum"]),
                "maximum": _vec(bounds["maximum"]),
                "center": _vec(bounds["center"]),
                "dimensions": _vec(dimensions),
            },
            "target": {
                "source": target_source,
                "name": target_empty.name_full if target_empty else None,
                "location": _vec(target),
            },
            "fit": {
                "margin_per_edge": float(margin),
                "view_from": _vec(back),
                "roll_degrees": float(roll_degrees),
                "distance": float(distance),
                "safe_frame_edges": [float(value) for value in safe],
                "clip_policy": clip_policy,
            },
            "camera_before": before,
            "camera_after": _camera_report(camera),
            "verification": verification,
            "warnings": warnings,
        }
    except Exception:
        camera.matrix_world = old_matrix
        camera.data.ortho_scale = old_ortho
        camera.data.clip_start, camera.data.clip_end = old_clips
        scene.camera = old_scene_camera
        if created_empty is not None:
            bpy.data.objects.remove(created_empty, do_unlink=True)
        if moved_empty_location is not None and target_empty is not None:
            target_empty.matrix_world = moved_empty_location
        bpy.context.view_layer.update()
        raise


def _failure(error):
    return {
        "schema": SCHEMA,
        "ok": False,
        "operation": "frame_render_camera",
        "blender_version": bpy.app.version_string,
        "error": str(error),
        "error_type": type(error).__name__,
    }


def run_config(config):
    try:
        return frame_camera(**config)
    except Exception as error:
        return _failure(error)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--include-descendants", action="store_true")
    parser.add_argument("--include-instances", action="store_true")
    parser.add_argument("--target-empty")
    parser.add_argument("--target-empty-mode", default="use",
                        choices=("use", "create_at_bounds", "move_to_bounds"))
    parser.add_argument("--view-from", nargs=3, type=float)
    parser.add_argument("--roll-degrees", type=float, default=0.0)
    parser.add_argument("--clip-policy", default="fit", choices=("fit", "preserve", "explicit"))
    parser.add_argument("--clip-padding", type=float, default=0.05)
    parser.add_argument("--clip-start-min", type=float, default=0.001)
    parser.add_argument("--clip-start", type=float)
    parser.add_argument("--clip-end", type=float)
    parser.add_argument("--set-scene-camera", action="store_true")
    parser.add_argument("--save-as")
    return parser.parse_args(argv)


def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = _parse_args(argv)
    config = {
        "camera_name": args.camera,
        "object_names": args.objects,
        "scene_name": args.scene,
        "frame": args.frame,
        "margin": args.margin,
        "include_descendants": args.include_descendants,
        "include_instances": args.include_instances,
        "target_empty_name": args.target_empty,
        "target_empty_mode": args.target_empty_mode,
        "view_from": args.view_from,
        "roll_degrees": args.roll_degrees,
        "clip_policy": args.clip_policy,
        "clip_padding": args.clip_padding,
        "clip_start_min": args.clip_start_min,
        "clip_start": args.clip_start,
        "clip_end": args.clip_end,
        "set_scene_camera": args.set_scene_camera,
    }
    result = run_config(config)
    if result["ok"] and args.save_as:
        if not os.path.isabs(args.save_as):
            result = _failure(ValueError("save_as must be an absolute path"))
            print(json.dumps(result, sort_keys=True))
            return 2
        bpy.ops.wm.save_as_mainfile(filepath=args.save_as)
        result["saved_as"] = args.save_as
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 2


if "MCP_CONFIG" in globals():
    print(json.dumps(run_config(MCP_CONFIG), sort_keys=True))
elif __name__ == "__main__":
    raise SystemExit(_cli())
