#!/usr/bin/env python3
"""Configure the public OVRTX Blender scene-property surface transactionally."""

import argparse
import json
import sys
import bpy


FIELDS = ("min_samples", "max_samples", "color_presentation_mode", "sync_viewport_camera")


def configure(request):
    scene = bpy.context.scene
    settings = getattr(scene, "ovrtx_example", None)
    engine = bpy.types.RenderEngine.bl_rna_get_subclass_py("OVRTX_EXAMPLE")
    if settings is None or engine is None:
        raise RuntimeError("OVRTX_EXAMPLE engine or Scene.ovrtx_example is not registered")
    available = {item.identifier for item in settings.bl_rna.properties}
    requested = {key: request[key] for key in FIELDS if key in request}
    missing = sorted(set(requested) - available)
    if missing:
        raise RuntimeError(f"installed add-on lacks requested settings: {missing}")
    before = {key: getattr(settings, key) for key in requested}
    prior_engine = scene.render.engine
    try:
        for key, value in requested.items():
            prop = settings.bl_rna.properties[key]
            if prop.type == "ENUM":
                allowed = {item.identifier for item in prop.enum_items}
                if value not in allowed:
                    raise ValueError(f"unsupported {key}: {value}; allowed={sorted(allowed)}")
            setattr(settings, key, value)
        if "min_samples" in available and "max_samples" in available:
            if int(settings.min_samples) < 1 or int(settings.max_samples) < int(settings.min_samples):
                raise ValueError("sample bounds require 1 <= min_samples <= max_samples")
        if request.get("activate_engine", False):
            scene.render.engine = "OVRTX_EXAMPLE"
        bpy.context.view_layer.update()
    except Exception:
        for key, value in before.items():
            setattr(settings, key, value)
        scene.render.engine = prior_engine
        bpy.context.view_layer.update()
        raise
    after = {key: getattr(settings, key) for key in requested}
    return {"schema": "ovrtx_scene_settings.v1", "ok": True,
            "engine_before": prior_engine, "engine_after": scene.render.engine,
            "before": before, "after": after}


def args_request():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--color-presentation-mode")
    parser.add_argument("--sync-viewport-camera", choices=("true", "false"))
    parser.add_argument("--activate-engine", action="store_true")
    args = parser.parse_args(raw)
    result = {key: value for key, value in vars(args).items() if value is not None}
    if "sync_viewport_camera" in result:
        result["sync_viewport_camera"] = result["sync_viewport_camera"] == "true"
    return result


request = globals().get("OVRTX_SETTINGS_REQUEST")
try:
    print(json.dumps(configure(dict(request) if isinstance(request, dict) else args_request()), sort_keys=True))
except Exception as exc:
    print(json.dumps({"schema": "ovrtx_scene_settings.v1", "ok": False,
                      "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
    if __name__ == "__main__":
        raise SystemExit(2)
