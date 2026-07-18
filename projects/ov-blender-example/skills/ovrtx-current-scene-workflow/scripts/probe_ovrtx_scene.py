#!/usr/bin/env python3
"""Read-only JSON probe of the installed OVRTX Blender scene surface."""

import json

import bpy


def probe():
    scene = bpy.context.scene
    engine_class = bpy.types.RenderEngine.bl_rna_get_subclass_py("OVRTX_EXAMPLE")
    settings = getattr(scene, "ovrtx_example", None)
    required = {"min_samples", "max_samples", "color_presentation_mode"}
    optional = {"sync_viewport_camera"}
    available = {item.identifier for item in settings.bl_rna.properties} if settings else set()
    missing = sorted(required - available)
    values = {}
    enums = {}
    if settings:
        for name in sorted((required | optional) & available):
            values[name] = getattr(settings, name)
            prop = settings.bl_rna.properties[name]
            if prop.type == "ENUM":
                enums[name] = [item.identifier for item in prop.enum_items]
    checks = {"engine_registered": engine_class is not None,
              "settings_registered": settings is not None,
              "required_settings_present": not missing,
              "active_camera": scene.camera is not None}
    return {"schema": "ovrtx_blender_scene_probe.v1", "ok": all(checks.values()),
            "blender_version": bpy.app.version_string, "checks": checks,
            "active_engine": scene.render.engine,
            "active_camera": scene.camera.name if scene.camera else None,
            "settings_type": settings.bl_rna.identifier if settings else None,
            "values": values, "enums": enums, "missing_settings": missing,
            "boundary": "Installed Blender add-on surface only; worker/runtime health is not inferred."}


print(json.dumps(probe(), sort_keys=True))
