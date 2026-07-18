# Blender 5.x mesh recipes

## Named box through the data API

This creates a box without operator context and refuses incompatible reuse.

```python
import bpy, json

name = "GEO-box"
dimensions = (2.0, 1.0, 0.5)
obj = bpy.data.objects.get(name)
if obj is not None:
    if obj.type != "MESH" or obj.library is not None:
        raise RuntimeError(f"refuse incompatible target: {name}")
else:
    x, y, z = (value * 0.5 for value in dimensions)
    vertices = [(-x,-y,-z), (-x,-y,z), (-x,y,-z), (-x,y,z),
                (x,-y,-z), (x,-y,z), (x,y,-z), (x,y,z)]
    faces = [(0,4,6,2), (1,3,7,5), (0,1,5,4),
             (2,6,7,3), (0,2,3,1), (4,5,7,6)]
    mesh = bpy.data.meshes.new(name + "-mesh")
    mesh.from_pydata(vertices, [], faces)
    if mesh.validate(verbose=False):
        raise RuntimeError("new box topology required validation repairs")
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
obj.location = (0.0, 0.0, 0.25)
bpy.context.view_layer.update()
print(json.dumps({"ok": True, "object": obj.name,
                  "vertices": len(obj.data.vertices),
                  "polygons": len(obj.data.polygons)}, sort_keys=True))
```

If an existing owned mesh must change dimensions, update its vertices in a
separate explicit transaction or create a replacement derivative. Do not make
get-or-create code silently rewrite arbitrary existing topology.

## Preserve world transform while parenting

Create a caller-owned transform root without an operator when one is needed:

```python
root_name = "ROOT-assembly"
parent = bpy.data.objects.get(root_name)
if parent is not None and (parent.type != "EMPTY" or parent.library is not None):
    raise RuntimeError(f"refuse incompatible parent: {root_name}")
if parent is None:
    parent = bpy.data.objects.new(root_name, None)
    parent.empty_display_type = "PLAIN_AXES"
    bpy.context.scene.collection.objects.link(parent)
```

```python
child = bpy.data.objects["GEO-child"]
parent = bpy.data.objects["ROOT-assembly"]
world = child.matrix_world.copy()
child.parent = parent
child.matrix_world = world
bpy.context.view_layer.update()
```

Setting only `matrix_parent_inverse` is easy to misuse when the child already
has transforms. Preserve and restore `matrix_world`, then verify it numerically.

## Idempotent modifier setup

```python
modifier = obj.modifiers.get("MOD-bevel")
if modifier is not None and modifier.type != "BEVEL":
    raise RuntimeError("modifier name collision: MOD-bevel")
modifier = modifier or obj.modifiers.new("MOD-bevel", "BEVEL")
modifier.width = 0.04
modifier.segments = 3
modifier.limit_method = "ANGLE"
bpy.context.view_layer.update()
```

Use `obj.modifiers.move(current_index, requested_index)` when order matters.
Audit evaluated geometry after every order or parameter change. Applying a
modifier changes topology and may invalidate shape keys, UV assumptions, or
downstream object identity; do it only in a disposable derivative.

## Context-prepared edit operator

For an operation such as seam-aware editing that requires Edit Mode, preserve
the prior mode, selection, and active object; select the named target and faces;
check `operator.poll()`; execute once; restore context in `finally`. Never rely
on whatever object happened to be selected before the MCP call.
