# OVRTX Blender scene API

Probe the running add-on before mutation; property availability is versioned.
The tested public Blender 5.1 add-on registers engine ID `OVRTX_EXAMPLE` and a
`Scene.ovrtx_example` property group containing `min_samples`, `max_samples`,
`color_presentation_mode`, `sync_viewport_camera`, and renderer-specific fields.

Use one bounded configuration transaction:

```python
import bpy, json

scene = bpy.context.scene
engine = bpy.types.RenderEngine.bl_rna_get_subclass_py("OVRTX_EXAMPLE")
settings = getattr(scene, "ovrtx_example", None)
if engine is None or settings is None:
    raise RuntimeError("OVRTX Example engine or settings are not registered")

required = {"min_samples", "max_samples", "color_presentation_mode"}
available = {item.identifier for item in settings.bl_rna.properties}
missing = sorted(required - available)
if missing:
    raise RuntimeError(f"installed add-on lacks settings: {missing}")

mode = "scene_linear_hdr"
allowed = {item.identifier for item in
           settings.bl_rna.properties["color_presentation_mode"].enum_items}
if mode not in allowed:
    raise RuntimeError(f"unsupported color mode: {mode}; available={sorted(allowed)}")
settings.min_samples = 1
settings.max_samples = 1
settings.color_presentation_mode = mode
scene.render.engine = "OVRTX_EXAMPLE"
bpy.context.view_layer.update()
print(json.dumps({"ok": True, "engine": scene.render.engine,
                  "min_samples": settings.min_samples,
                  "max_samples": settings.max_samples,
                  "color_presentation_mode": settings.color_presentation_mode},
                 sort_keys=True))
```

Use a one-sample render first. A Blender render operator completing is not by
itself proof of native ownership: inspect the add-on's documented session status
and requested render product, then verify the resulting dimensions and pixels.
Do not import worker/client internals or reconstruct the transport in Blender
Python.
