---
name: ovrtx-render-settings
description: Probe and configure the registered OVRTX Blender scene settings for sample bounds, color-presentation mode, viewport-camera sync, and engine selection through exact Blender RNA. Use when changing supported OVRTX scene controls or diagnosing a setting that may not exist in the installed add-on version.
---

# OVRTX render settings

Run `ovrtx-addon-install-and-preflight`, then use
`ovrtx-current-scene-workflow/scripts/probe_ovrtx_scene.py` to inspect the
installed surface. The tested public add-on registers engine `OVRTX_EXAMPLE`
and `Scene.ovrtx_example`; probe rather than assuming versioned fields.

Use `scripts/configure_scene_settings.py` in a bounded Blender process or append
its source to a `blender-python-execution` MCP call with
`OVRTX_SETTINGS_REQUEST`. The helper validates enum/range/order constraints and
restores all touched fields if any mutation fails.

```python
OVRTX_SETTINGS_REQUEST = {
    "min_samples": 1,
    "max_samples": 16,
    "color_presentation_mode": "scene_linear_hdr",
    "sync_viewport_camera": True,
    "activate_engine": True,
}
```

Require `ok: true` and inspect the returned before/after values. A successful
RNA assignment proves Blender add-on state only; verify native rendering with
`ovrtx-current-scene-workflow` and require the render result to identify engine
`OVRTX_EXAMPLE`.
