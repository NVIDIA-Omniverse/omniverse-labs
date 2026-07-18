# Blender 5.x animation API

Use named datablocks and explicit RNA paths. Each mutation below is intended to
run as one bounded `blender-python-execution` transaction. Inspect the result in
a separate call.

## Object-property keyframes

```python
import bpy, json

obj = bpy.data.objects.get("GEO-subject")
if obj is None:
    raise RuntimeError("missing object: GEO-subject")

keys = {
    1:  (0.0, 0.0, 0.0),
    24: (2.0, 0.0, 1.0),
    48: (4.0, 0.0, 0.0),
}
for frame, value in keys.items():
    obj.location = value
    obj.keyframe_insert(data_path="location", frame=frame, group="Transform")

bpy.context.scene.frame_start = min(keys)
bpy.context.scene.frame_end = max(keys)
bpy.context.scene.frame_set(min(keys))
bpy.context.view_layer.update()
print(json.dumps({"ok": True, "object": obj.name, "frames": sorted(keys)}))
```

`keyframe_insert` creates or reuses the object's Action and, in Blender 5.x,
its layered Action slot. Do not assume `action.fcurves` exists: Blender 5 uses
`action.layers[*].strips[*].channelbags[*].fcurves`. Inspection code should
support both that layout and legacy `action.fcurves` for earlier files/builds.

## Set interpolation explicitly

Run this after inserting keys. Resolve the channel bag belonging to the
object's current Action slot; do not edit every slot in a shared Action.

```python
action = obj.animation_data.action
slot = obj.animation_data.action_slot
curves = []
if hasattr(action, "layers"):
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot, ensure=False) if slot else None
            if bag:
                curves.extend(bag.fcurves)
else:
    curves.extend(action.fcurves)

for curve in curves:
    if curve.data_path == "location":
        for point in curve.keyframe_points:
            point.interpolation = "BEZIER"  # or LINEAR / CONSTANT
```

Use `LINEAR` only for genuinely constant-rate properties and `CONSTANT` for
holds. Bezier handles can overshoot; sample the evaluated property between keys
when overshoot would violate a bound.

## Shape keys

```python
mesh_obj = bpy.data.objects["GEO-face"]
if mesh_obj.type != "MESH":
    raise RuntimeError("GEO-face is not a mesh")
if mesh_obj.data.shape_keys is None:
    mesh_obj.shape_key_add(name="Basis")
smile = mesh_obj.data.shape_keys.key_blocks.get("Smile")
if smile is None:
    smile = mesh_obj.shape_key_add(name="Smile")

for frame, value in ((1, 0.0), (12, 1.0), (24, 0.0)):
    smile.value = value
    smile.keyframe_insert(data_path="value", frame=frame)
```

Audit a shape value with the property path
`shape_keys.key_blocks["Smile"].value`. Shape-key animation lives on the Key
datablock, not on the Object Action.

## Drivers and constraints

Prefer keyframes or constraints over scripted expressions. For a simple driver,
use variables and a restricted expression:

```python
driven = bpy.data.objects["GEO-driven"]
source = bpy.data.objects["CTRL-source"]
curve = driven.driver_add("location", 2)
driver = curve.driver
driver.type = "SCRIPTED"
driver.expression = "source_x * 0.5"
var = driver.variables.new()
var.name = "source_x"
var.type = "TRANSFORMS"
target = var.targets[0]
target.id = source
target.transform_type = "LOC_X"
target.transform_space = "WORLD_SPACE"
```

Do not enable arbitrary Python or frame-change handlers to make an imported
scene animate. Record drivers and constraints in the handoff because not every
exporter or runtime evaluates them.

## Sampling and restoration

Use `scene.frame_set(frame)` followed by `view_layer.update()`, then read the
evaluated object from `evaluated_depsgraph_get()`. Always restore the original
frame in `finally`. Sample motion extrema in addition to first/middle/last when
Bezier curves, constraints, drivers, parents, or modifiers can overshoot.

Run `scripts/audit_animation.py` after authoring. Its pass means that requested
properties resolve, remain finite, and satisfy the requested sampled-motion and
keyframe gates. It samples each evaluated world matrix as well, so parent,
constraint, driver, and NLA motion remains observable even when the requested
local property is static. It does not certify renderer continuity, export compatibility,
camera composition, collision behavior, or visual quality.
