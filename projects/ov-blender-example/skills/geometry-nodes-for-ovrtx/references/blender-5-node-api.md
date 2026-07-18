# Blender 5 Geometry Nodes API patterns

Use these patterns through `blender-python-execution`. They target Blender 5.x;
Blender 4.x also uses `NodeTree.interface`, but socket and node availability
must still be inspected in the running build.

## Idempotent geometry pass-through group

This transaction creates only caller-owned names, uses the current interface
API, wires geometry explicitly, and proves evaluated output. Do not clear or
replace an unrelated node group with the same name.

```python
import bpy, json

object_name = "GEO-subject"
modifier_name = "GN-requested"
group_name = "GN-requested"
obj = bpy.data.objects.get(object_name)
if obj is None or obj.type != "MESH" or obj.library is not None:
    raise RuntimeError(f"editable mesh not found: {object_name}")

group = bpy.data.node_groups.get(group_name)
if group is not None and group.bl_idname != "GeometryNodeTree":
    raise RuntimeError(f"refuse incompatible node group: {group_name}")
if group is None:
    group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

nodes, links = group.nodes, group.links
input_node = nodes.get("REQ-GroupInput") or nodes.new("NodeGroupInput")
input_node.name = "REQ-GroupInput"
output_node = nodes.get("REQ-GroupOutput") or nodes.new("NodeGroupOutput")
output_node.name = "REQ-GroupOutput"
links.new(input_node.outputs["Geometry"], output_node.inputs["Geometry"])

modifier = obj.modifiers.get(modifier_name)
if modifier is not None and modifier.type != "NODES":
    raise RuntimeError(f"refuse incompatible modifier: {modifier_name}")
modifier = modifier or obj.modifiers.new(modifier_name, "NODES")
modifier.node_group = group
bpy.context.view_layer.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
try:
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        raise RuntimeError("Geometry Nodes evaluated to empty mesh")
    result = {
        "ok": True,
        "operation": "ensure_geometry_node_group",
        "object": obj.name,
        "modifier": modifier.name,
        "node_group": group.name,
        "nodes": len(group.nodes),
        "links": len(group.links),
        "evaluated_vertices": len(mesh.vertices),
        "evaluated_polygons": len(mesh.polygons),
    }
finally:
    evaluated.to_mesh_clear()
print(json.dumps(result, sort_keys=True))
```

`group.inputs.new(...)` and `group.outputs.new(...)` are legacy patterns. Use
`group.interface.new_socket(...)` and inspect `group.interface.items_tree`.
Socket display names are not stable identifiers for modifier values. Resolve
the input interface item, then use its `identifier` as the modifier key:

```python
socket = next(item for item in group.interface.items_tree
              if item.item_type == "SOCKET" and item.in_out == "INPUT"
              and item.name == "Scale")
modifier[socket.identifier] = 1.25
```

## Nontrivial Transform Geometry recipe

Insert a caller-owned Transform Geometry node between Group Input and Group
Output to widen evaluated geometry. This is an absolute, idempotent assignment;
replace the scale with a request-derived value and verify evaluated bounds.

```python
transform = nodes.get("REQ-TransformGeometry") or nodes.new("GeometryNodeTransform")
transform.name = "REQ-TransformGeometry"
transform.inputs["Translation"].default_value = (0.0, 0.0, 0.0)
transform.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)
transform.inputs["Scale"].default_value = (7.0, 1.0, 1.0)
links.new(input_node.outputs["Geometry"], transform.inputs["Geometry"])
links.new(transform.outputs["Geometry"], output_node.inputs["Geometry"])
bpy.context.view_layer.update()
```

In Blender 5.1 the node id is `GeometryNodeTransform`; UI label guesses such as
`GeometryNodeTransformGeometry` fail. The named sockets above are stable in the
tested build, but inspect `[(s.name, s.bl_idname) for s in transform.inputs]`
when supporting another Blender version. A `(7, 1, 1)` scale must change the
evaluated X extent by seven relative to the same unmodified local mesh; test the
evaluated mesh, not the source `obj.data` bounds.

Call `view_layer.update()` after changing an interface or modifier property.
Read the evaluated object or a temporary evaluated mesh; reading the original
mesh only proves the pre-modifier topology. Always pair `to_mesh()` with
`to_mesh_clear()` in `finally`.

## Export and renderer boundary

Record whether instances and procedural attributes are left live, realized, or
baked. For USD or OVRTX, test a disposable derivative and compare evaluated
bounds, vertex/polygon counts, material slots, and required named attributes
before and after conversion. A Blender viewport match does not prove the target
runtime supports a node, simulation zone, anonymous attribute, or instance
representation.
