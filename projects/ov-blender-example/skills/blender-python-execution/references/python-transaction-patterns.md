# Blender Python transaction patterns

These snippets are templates for `execute_blender_code(code)`. Replace only the
explicit request values and stable names.

## Read, validate, mutate, report

```python
import bpy, json, math

name = "GEO-requested-target"
obj = bpy.data.objects.get(name)
if obj is None:
    raise RuntimeError(f"missing target: {name}")
if obj.library is not None:
    raise RuntimeError(f"target is linked and read-only: {name}")

requested = (1.0, 2.0, 3.0)
if not all(math.isfinite(v) for v in requested):
    raise ValueError("location must be finite")

obj.location = requested
bpy.context.view_layer.update()
actual = [float(v) for v in obj.location]
print(json.dumps({
    "ok": actual == list(requested),
    "operation": "set_location",
    "object": obj.name,
    "location": actual,
}, sort_keys=True))
```

## Get or create an owned collection

```python
import bpy, json

name = "COL-requested-output"
collection = bpy.data.collections.get(name)
created = collection is None
if collection is None:
    collection = bpy.data.collections.new(name)
if collection.name not in {c.name for c in bpy.context.scene.collection.children}:
    bpy.context.scene.collection.children.link(collection)

bpy.context.view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "ensure_collection",
    "collection": collection.name,
    "created": created,
}, sort_keys=True))
```

The stable name makes this safe to rerun. Do not unlink similarly named user
collections or assume ownership of pre-existing contents.

## Idempotent modifier update

```python
import bpy, json

object_name = "GEO-requested-target"
modifier_name = "REQ-Bevel"
obj = bpy.data.objects.get(object_name)
if obj is None or obj.type != "MESH":
    raise RuntimeError(f"mesh target not found: {object_name}")

modifier = obj.modifiers.get(modifier_name)
created = modifier is None
if modifier is None:
    modifier = obj.modifiers.new(name=modifier_name, type="BEVEL")
modifier.width = 0.02
modifier.segments = 3

bpy.context.view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "ensure_bevel_modifier",
    "object": obj.name,
    "modifier": modifier.name,
    "created": created,
    "width": float(modifier.width),
    "segments": int(modifier.segments),
}, sort_keys=True))
```

## Operator context

Prefer data APIs. If an operator is the supported path, scope selection and
active object, verify pollability, and restore previous state:

```python
import bpy, json

name = "GEO-requested-target"
obj = bpy.data.objects.get(name)
if obj is None:
    raise RuntimeError(f"missing target: {name}")

view_layer = bpy.context.view_layer
previous_active = view_layer.objects.active
previous_selected = [item for item in bpy.context.selected_objects]
previous_mode = bpy.context.mode

try:
    if previous_mode != "OBJECT" and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    view_layer.objects.active = obj
    if not bpy.ops.object.shade_smooth.poll():
        raise RuntimeError("shade_smooth is unavailable in the prepared context")
    bpy.ops.object.shade_smooth()
finally:
    if bpy.context.mode != "OBJECT" and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for item in previous_selected:
        if item.name in view_layer.objects:
            item.select_set(True)
    if previous_active is not None and previous_active.name in view_layer.objects:
        view_layer.objects.active = previous_active

view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "shade_smooth",
    "object": obj.name,
}, sort_keys=True))
```

Mode restoration may itself require a valid operator context. Do not force a
mode change when Blender reports that the mode is unavailable.

## Evaluated geometry read

```python
import bpy, json

name = "GEO-requested-target"
obj = bpy.data.objects.get(name)
if obj is None:
    raise RuntimeError(f"missing target: {name}")

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
mesh = evaluated.to_mesh() if evaluated.type == "MESH" else None
try:
    result = {
        "vertices": len(mesh.vertices) if mesh else None,
        "edges": len(mesh.edges) if mesh else None,
        "polygons": len(mesh.polygons) if mesh else None,
    }
finally:
    if mesh is not None:
        evaluated.to_mesh_clear()

print(json.dumps({
    "ok": True,
    "operation": "inspect_evaluated_geometry",
    "object": obj.name,
    "geometry": result,
}, sort_keys=True))
```

## Request a viewport redraw and frame the target

Use this as a separate transaction after a visible mutation and before the
screenshot call. It is interactive-only; background Blender has no screen or
3D View and should render to a file instead.

```python
import bpy, json

target_names = ["GEO-requested-target"]
view_layer = bpy.context.view_layer
targets = [bpy.data.objects.get(name) for name in target_names]
missing = [name for name, obj in zip(target_names, targets) if obj is None]
targets = [obj for obj in targets if obj is not None]
not_in_view_layer = [obj.name for obj in targets if obj.name not in view_layer.objects]
if missing or not_in_view_layer:
    raise RuntimeError(f"missing={missing} not_in_view_layer={not_in_view_layer}")

# Search every Blender window and choose the largest usable 3D View. The active
# context screen is often a Properties or Image Editor after MCP operations.
candidates = []
for window in bpy.context.window_manager.windows:
    screen = window.screen
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        region = next((item for item in area.regions if item.type == 'WINDOW'), None)
        space = area.spaces.active
        if region and space and space.type == 'VIEW_3D':
            candidates.append((area.width * area.height, window, screen, area, region, space))

previous_active = view_layer.objects.active
previous_selected = list(bpy.context.selected_objects)
framed = False
area_size = None
try:
    if bpy.context.mode != 'OBJECT' and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in targets:
        obj.select_set(True)
    view_layer.objects.active = targets[0]
    view_layer.update()

    if candidates:
        area_size, window, screen, area, region, space = max(candidates, key=lambda item: item[0])
        with bpy.context.temp_override(
            window=window,
            screen=screen,
            area=area,
            region=region,
            space_data=space,
            region_data=space.region_3d,
        ):
            if not bpy.ops.view3d.view_selected.poll():
                raise RuntimeError('view_selected failed its context poll')
            result = bpy.ops.view3d.view_selected(use_all_regions=False)
            framed = 'FINISHED' in result
        area.tag_redraw()
finally:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in previous_selected:
        if obj.name in view_layer.objects:
            obj.select_set(True)
    if previous_active and previous_active.name in view_layer.objects:
        view_layer.objects.active = previous_active

print(json.dumps({
    "ok": framed,
    "operation": "frame_viewport_objects",
    "objects": [obj.name for obj in targets],
    "candidate_view3d_areas": len(candidates),
    "chosen_area_pixels": area_size,
    "framed": framed,
    "background_or_no_view3d": not candidates,
}, sort_keys=True))
```

Return control after this call, then invoke the provider screenshot tool. A
redraw request still cannot prove renderer freshness; use an actual render or a
renderer-specific readiness/product check when the evidence claim requires it.
This operation changes only `RegionView3D`; it does not position the active
render camera. Use `blender-camera-framing` for render-camera containment.

## Save and render boundaries

Saving, opening files, exporting, and rendering are not implicit transaction
steps. Perform them only when explicitly requested, use a caller-selected
absolute output path, verify the active camera and engine first, and inspect the
resulting file or pixels afterward. Never overwrite the current source as a
side effect of a modeling transaction.
