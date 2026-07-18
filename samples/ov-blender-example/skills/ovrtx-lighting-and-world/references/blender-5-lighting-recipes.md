# Blender 5.x lighting recipes

Use each mutation as a bounded `blender-python-execution` transaction. Inspect
first, mutate named data, call `bpy.context.view_layer.update()`, and print a
small JSON postcondition. These recipes author Blender state; they do not call
or emulate the OVRTX runtime.

## Idempotent named light

```python
import bpy
import json
from mathutils import Matrix, Vector

def ensure_light(name, light_type, collection=None):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != "LIGHT":
        raise TypeError(f"{name!r} exists and is not a light")
    if obj is None:
        data = bpy.data.lights.new(name=f"{name}-data", type=light_type)
        obj = bpy.data.objects.new(name, data)
        (collection or bpy.context.scene.collection).objects.link(obj)
    elif obj.data.type != light_type:
        obj.data.type = light_type
    return obj

key = ensure_light("Key", "AREA")
key.data.energy = 800.0
key.data.color = (1.0, 0.82, 0.68)
key.data.shape = "DISK"
key.data.size = 2.0
key.location = (4.0, -4.0, 5.0)
bpy.context.view_layer.update()
print(json.dumps({"name": key.name, "type": key.data.type,
                  "energy": key.data.energy, "size": key.data.size}))
```

Changing `Light.type` is a topology/conversion-family change for many render
bridges. Prefer a scene refresh after it. Common type-specific properties are:

- POINT: `energy`, `color`, `shadow_soft_size`;
- SPOT: POINT fields plus `spot_size` in radians and `spot_blend` in `[0, 1]`;
- SUN: `energy`, `color`, and `angle` in radians;
- AREA: `energy`, `color`, `shape`, `size`, and `size_y` for RECTANGLE/ELLIPSE.

## Aim a light at a target

Blender lights emit along local `-Z`. Preserve roll by using local `Y` as the
secondary axis:

```python
from mathutils import Vector

def aim_negative_z(obj, target):
    direction = Vector(target) - obj.matrix_world.translation
    if direction.length_squared <= 1.0e-12:
        raise ValueError("light and target positions coincide")
    location, _rotation, scale = obj.matrix_world.decompose()
    obj.matrix_world = Matrix.LocRotScale(
        location, direction.to_track_quat("-Z", "Y"), scale
    )

aim_negative_z(key, (0.0, 0.0, 1.0))
bpy.context.view_layer.update()
```

Use a `TRACK_TO` constraint only when the target relationship must remain live.
For a baked setup, write the rotation directly so export does not depend on
constraint evaluation support.

## Constant World Background

```python
import bpy

scene = bpy.context.scene
world = bpy.data.worlds.get("WORLD-requested") or bpy.data.worlds.new("WORLD-requested")
scene.world = world
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()
output = nodes.new("ShaderNodeOutputWorld")
output.name = "World Output"
background = nodes.new("ShaderNodeBackground")
background.name = "World Background"
background.inputs["Color"].default_value = (0.03, 0.04, 0.06, 1.0)
background.inputs["Strength"].default_value = 0.35
links.new(background.outputs["Background"], output.inputs["Surface"])
bpy.context.view_layer.update()
```

In Blender 5.x a newly created World has a usable `node_tree`; do not depend on
the deprecated `use_nodes` flag. Keep one active World Output. Multiple outputs
or shader mixtures may be valid in Blender but require explicit add-on support.

## Environment Texture World

```python
from pathlib import Path
import bpy

hdri_path = Path("/absolute/path/studio.exr")
if not hdri_path.is_file():
    raise FileNotFoundError(hdri_path)

world = bpy.data.worlds.get("WORLD-requested") or bpy.data.worlds.new("WORLD-requested")
bpy.context.scene.world = world
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()
output = nodes.new("ShaderNodeOutputWorld")
background = nodes.new("ShaderNodeBackground")
environment = nodes.new("ShaderNodeTexEnvironment")
environment.image = bpy.data.images.load(str(hdri_path.resolve()), check_existing=True)
# Keep Blender's detected file color space unless the image supplier or the
# project's OCIO policy specifies a different interpretation.
background.inputs["Strength"].default_value = 1.0
links.new(environment.outputs["Color"], background.inputs["Color"])
links.new(background.outputs["Background"], output.inputs["Surface"])
bpy.context.view_layer.update()
```

Add Texture Coordinate and Mapping nodes only when rotation is required. Record
the chosen rotation rather than relying on node placement. Environment maps are
scene-linear lighting data in this workflow; confirm the intended color-space
policy for the supplied file and installed add-on.

## Safe replacement and rollback

Before replacing a complex World graph, duplicate the World and assign the copy
to the scene. Before replacing a light data-block shared by several objects,
copy it with `obj.data = obj.data.copy()`. On failure, restore the prior
`scene.world`, transforms, and data-block references; remove only new orphaned
data created by the transaction.
