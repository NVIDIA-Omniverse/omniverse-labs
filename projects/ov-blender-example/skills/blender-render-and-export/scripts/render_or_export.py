#!/usr/bin/env python3
"""Render one Blender still or export named objects to GLB/USD with JSON proof."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy


def _owned_output(path_value, operation):
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("output_path must be absolute")
    expected = {"render_still": {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"},
                "export_glb": {".glb"}, "export_usd": {".usd", ".usda", ".usdc"}}
    if path.suffix.lower() not in expected[operation]:
        raise ValueError("output extension does not match operation")
    if path.exists():
        raise FileExistsError("refuse to overwrite existing output: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _digest(path):
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("output file is empty")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return size, value.hexdigest()


def _select(names):
    resolved = []
    missing = []
    for name in dict.fromkeys(names or []):
        obj = bpy.data.objects.get(name)
        if obj is None:
            missing.append(name)
        else:
            resolved.append(obj)
    if missing or not resolved:
        raise ValueError("missing or empty export object set: %s" % ", ".join(missing))
    bpy.ops.object.select_all(action="DESELECT")
    for obj in resolved:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = resolved[0]
    return resolved


def run(operation, output_path, camera_name=None, frame=None, resolution=None,
        object_names=None, apply_modifiers=True, export_animation=False):
    if operation not in {"render_still", "export_glb", "export_usd"}:
        raise ValueError("unsupported operation")
    output = _owned_output(output_path, operation)
    scene = bpy.context.scene
    if operation != "render_still" and bpy.context.mode != "OBJECT":
        raise RuntimeError("export operations require Blender Object mode")
    prior_selected = list(bpy.context.selected_objects)
    prior_active = bpy.context.view_layer.objects.active
    prior_render_state = (scene.camera, scene.frame_current,
                          scene.render.resolution_x, scene.render.resolution_y,
                          scene.render.resolution_percentage,
                          scene.render.image_settings.file_format,
                          scene.render.filepath)
    try:
        if operation == "render_still":
            camera = bpy.data.objects.get(camera_name) if camera_name else scene.camera
            if camera is None or camera.type != "CAMERA":
                raise ValueError("render_still requires a valid camera")
            scene.camera = camera
            if frame is not None:
                scene.frame_set(int(frame))
            if resolution is not None:
                if len(resolution) != 3 or min(int(v) for v in resolution) <= 0:
                    raise ValueError("resolution must be [x, y, percentage] with positive values")
                scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = map(int, resolution)
            formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".exr": "OPEN_EXR", ".tif": "TIFF", ".tiff": "TIFF"}
            scene.render.image_settings.file_format = formats[output.suffix.lower()]
            scene.render.filepath = str(output)
            bpy.context.view_layer.update()
            result = bpy.ops.render.render(write_still=True)
            detail = {"camera": camera.name, "frame": scene.frame_current,
                      "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
                      "engine": scene.render.engine, "operator_result": sorted(result)}
        else:
            resolved = _select(object_names)
            bpy.context.view_layer.update()
            if operation == "export_glb":
                if not bpy.ops.export_scene.gltf.poll():
                    raise RuntimeError("GLTF export operator is unavailable")
                result = bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB",
                                                   use_selection=True, export_apply=bool(apply_modifiers),
                                                   export_animations=bool(export_animation))
            else:
                if not bpy.ops.wm.usd_export.poll():
                    raise RuntimeError("USD export operator is unavailable")
                result = bpy.ops.wm.usd_export(filepath=str(output), selected_objects_only=True,
                                               export_animation=bool(export_animation), evaluation_mode="RENDER")
            detail = {"objects": [obj.name for obj in resolved], "operator_result": sorted(result),
                      "export_animation": bool(export_animation)}
            if operation == "export_glb":
                detail["apply_modifiers"] = bool(apply_modifiers)
            else:
                detail["evaluation_mode"] = "RENDER"
        if not output.is_file():
            raise RuntimeError("operator finished without producing the requested file")
        size, sha256 = _digest(output)
        return {"schema": "blender_render_export.v1", "ok": True, "operation": operation,
                "blender_version": bpy.app.version_string, "output_path": str(output),
                "bytes": size, "sha256": sha256, "detail": detail}
    finally:
        if operation == "render_still":
            (scene.camera, prior_frame, scene.render.resolution_x,
             scene.render.resolution_y, scene.render.resolution_percentage,
             scene.render.image_settings.file_format,
             scene.render.filepath) = prior_render_state
            scene.frame_set(prior_frame)
            bpy.context.view_layer.update()
        else:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in prior_selected:
                if obj.name in bpy.context.view_layer.objects:
                    obj.select_set(True)
            if prior_active and prior_active.name in bpy.context.view_layer.objects:
                bpy.context.view_layer.objects.active = prior_active


def _args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("render_still", "export_glb", "export_usd"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--resolution", nargs=3, type=int)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--no-apply-modifiers", action="store_true")
    parser.add_argument("--export-animation", action="store_true")
    return parser.parse_args(raw)


def _failure(error, operation=None):
    return {"schema": "blender_render_export.v1", "ok": False, "operation": operation,
            "error_type": type(error).__name__, "error": str(error)}


if "RENDER_EXPORT_REQUEST" in globals():
    try:
        print(json.dumps(run(**RENDER_EXPORT_REQUEST), sort_keys=True))
    except Exception as error:
        print(json.dumps(_failure(error, RENDER_EXPORT_REQUEST.get("operation")), sort_keys=True))
elif __name__ == "__main__":
    args = _args()
    try:
        result = run(args.operation, args.output, args.camera, args.frame, args.resolution,
                     args.objects, not args.no_apply_modifiers, args.export_animation)
    except Exception as error:
        result = _failure(error, args.operation)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)
