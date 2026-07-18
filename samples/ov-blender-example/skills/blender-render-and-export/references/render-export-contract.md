# Blender 5.x render and export contract

## Save a derivative `.blend`

Saving changes the current main-file identity, so refuse relative paths and
source overwrite, and restore the original file by reopening it only when the
caller explicitly wants to continue there. Prefer ending the bounded Blender
process after this write.

```python
from pathlib import Path
import bpy

output = Path("/absolute/caller-owned/scene-derivative.blend")
if not output.is_absolute() or output.suffix.lower() != ".blend":
    raise ValueError("derivative must be an absolute .blend path")
source = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
if source and output.resolve() == source:
    raise RuntimeError("refuse to overwrite the source blend")
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
if not output.is_file() or output.stat().st_size == 0:
    raise RuntimeError("save finished without a nonempty derivative")
```

## Still render

Set the active camera, frame, resolution, percentage, file format, and absolute
output path explicitly. Call `bpy.ops.render.render(write_still=True)` only
after `view_layer.update()` and a camera-containment audit. Verify the written
file rather than treating `{'FINISHED'}` as sufficient.

Keep engine ownership explicit. `BLENDER_EEVEE`, `BLENDER_WORKBENCH`, and
`CYCLES` are Blender results. Use the OVRTX workflow only when the registered
engine and installed add-on are intentionally in scope.

## GLB

Blender 5.1 exposes `bpy.ops.export_scene.gltf` with `filepath`,
`export_format='GLB'`, `use_selection`, and `export_apply`. Select exact named
objects and use `use_selection=True`. Decide whether modifiers are applied;
`export_apply=True` changes the exported derivative, not the source mesh.

## USD

Blender 5.1 exposes `bpy.ops.wm.usd_export` with `filepath`,
`selected_objects_only`, `export_animation`, and `evaluation_mode`. This is the
generic Blender USD exporter. It is not a substitute for an add-on-specific USD
stage or native renderer workflow.

Probe operator RNA in the running version before adding optional arguments:

```python
properties = {item.identifier for item in bpy.ops.wm.usd_export.get_rna_type().properties}
if "selected_objects_only" not in properties:
    raise RuntimeError("this Blender USD exporter lacks selected_objects_only")
```

For animation export, declare the frame range and sample behavior explicitly.
For portable USD dependency localization or flattening, route to the dedicated
USD handoff skill rather than assuming export produced a self-contained file.
