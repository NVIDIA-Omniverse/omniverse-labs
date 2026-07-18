# Blender 5.1 UV and material recipes

Use these patterns through `blender-python-execution`. Keep creation, unwrap,
material wiring, audit, and rendering as separate transactions. Replace only the
declared names, paths, and request values. Do not clear an existing material node
tree or UV layer unless the user explicitly owns it.

## Contents

- [Create deterministic planar UVs without operator context](#create-deterministic-planar-uvs-without-operator-context)
- [Run an operator unwrap with explicit context](#run-an-operator-unwrap-with-explicit-context)
- [Create a reachable Principled image material](#create-a-reachable-principled-image-material)
- [Wire a normal image correctly](#wire-a-normal-image-correctly)
- [Configure alpha deliberately](#configure-alpha-deliberately)
- [Assign material slots by polygon](#assign-material-slots-by-polygon)
- [Prepare a bake target](#prepare-a-bake-target)

## Create deterministic planar UVs without operator context

Use direct loop data when a known planar projection is correct. This example maps
local X/Y bounds into the unit tile and refuses a zero-width axis.

```python
import bpy, json

object_name = "GEO-target"
layer_name = "UV-primary"
obj = bpy.data.objects.get(object_name)
if obj is None or obj.type != "MESH" or obj.library is not None:
    raise RuntimeError(f"editable mesh not found: {object_name}")
if bpy.context.mode != "OBJECT":
    raise RuntimeError("unwrap recipe requires Object mode at transaction start")
mesh = obj.data
layer = mesh.uv_layers.get(layer_name) or mesh.uv_layers.new(name=layer_name)

xs = [vertex.co.x for vertex in mesh.vertices]
ys = [vertex.co.y for vertex in mesh.vertices]
minimum_x, maximum_x = min(xs), max(xs)
minimum_y, maximum_y = min(ys), max(ys)
extent_x, extent_y = maximum_x - minimum_x, maximum_y - minimum_y
if extent_x <= 1.0e-12 or extent_y <= 1.0e-12:
    raise RuntimeError("planar X/Y projection has a zero-width axis")

for polygon in mesh.polygons:
    for loop_index in polygon.loop_indices:
        coordinate = mesh.vertices[mesh.loops[loop_index].vertex_index].co
        layer.data[loop_index].uv = (
            (coordinate.x - minimum_x) / extent_x,
            (coordinate.y - minimum_y) / extent_y,
        )
mesh.uv_layers.active = layer
layer.active_render = True
mesh.update()
print(json.dumps({
    "ok": True,
    "operation": "planar_uv_xy",
    "object": obj.name,
    "uv_layer": layer.name,
    "loops": len(layer.data),
}, sort_keys=True))
```

For X/Z or Y/Z projection, change both selected coordinate components together.
Do not use this recipe for curved surfaces merely because it is context-free.

## Run an operator unwrap with explicit context

Use operators for seam-aware unwraps. Preserve selection and mode, make the
target active, select the intended faces, and verify the operator can run.

```python
import bpy, json

object_name = "GEO-target"
layer_name = "UV-primary"
obj = bpy.data.objects.get(object_name)
if obj is None or obj.type != "MESH" or obj.library is not None:
    raise RuntimeError(f"editable mesh not found: {object_name}")

view_layer = bpy.context.view_layer
previous_active = view_layer.objects.active
previous_selected = list(bpy.context.selected_objects)
try:
    if bpy.context.mode != "OBJECT" and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    view_layer.objects.active = obj
    layer = obj.data.uv_layers.get(layer_name) or obj.data.uv_layers.new(name=layer_name)
    obj.data.uv_layers.active = layer
    layer.active_render = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if not bpy.ops.uv.unwrap.poll():
        raise RuntimeError("uv.unwrap is unavailable in the prepared context")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
finally:
    if bpy.context.mode != "OBJECT" and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for item in previous_selected:
        if item.name in view_layer.objects:
            item.select_set(True)
    if previous_active and previous_active.name in view_layer.objects:
        view_layer.objects.active = previous_active

view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "unwrap",
    "object": obj.name,
    "uv_layer": layer.name,
    "loops": len(layer.data),
}, sort_keys=True))
```

Mark seams in a prior bounded transaction. Inspect a numbered checker after the
unwrap; operator success does not prove orientation, density, or island quality.

## Create a reachable Principled image material

Load only a caller-supplied path. Reuse owned, stably named nodes and wire them to
the active material output. Use `sRGB` only for color imagery.

```python
import bpy, json

object_name = "GEO-target"
material_name = "MAT-requested"
image_path = "/absolute/caller-selected/base-color.png"
obj = bpy.data.objects.get(object_name)
if obj is None or obj.type != "MESH":
    raise RuntimeError(f"mesh not found: {object_name}")

material = bpy.data.materials.get(material_name)
if material is not None and not material.get("_texture_uv_workflow_owned", False):
    raise RuntimeError(f"refuse to rewire unowned material: {material_name}")
if material is None:
    material = bpy.data.materials.new(material_name)
    material["_texture_uv_workflow_owned"] = True
if material.node_tree is None:  # compatibility for older node-disabled files
    material.use_nodes = True
nodes, links = material.node_tree.nodes, material.node_tree.links
output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
output = output or nodes.new("ShaderNodeOutputMaterial")
output.name = "REQ-MaterialOutput"
principled = nodes.get("REQ-Principled") or nodes.new("ShaderNodeBsdfPrincipled")
principled.name = "REQ-Principled"
texture = nodes.get("REQ-BaseColor") or nodes.new("ShaderNodeTexImage")
texture.name = "REQ-BaseColor"
image = bpy.data.images.load(image_path, check_existing=True)
image.colorspace_settings.name = "sRGB"
texture.image = image

links.new(texture.outputs["Color"], principled.inputs["Base Color"])
links.new(principled.outputs["BSDF"], output.inputs["Surface"])
if material.name not in {slot.material.name for slot in obj.material_slots if slot.material}:
    obj.data.materials.append(material)
bpy.context.view_layer.update()
print(json.dumps({
    "ok": True,
    "operation": "ensure_base_color_material",
    "object": obj.name,
    "material": material.name,
    "image": image.name,
    "colorspace": image.colorspace_settings.name,
    "surface_linked": output.inputs["Surface"].is_linked,
}, sort_keys=True))
```

If the existing material is not request-owned, create a new material rather than
rewiring it. Repeated `links.new` calls replace a single-input link but can leave
unowned nodes; inspect the result rather than clearing the tree.

## Wire a normal image correctly

Set normal, roughness, metallic, AO, height, and mask images to `Non-Color`.
Connect a tangent-space normal texture through a Normal Map node.

```python
normal_texture = nodes.get("REQ-NormalTexture") or nodes.new("ShaderNodeTexImage")
normal_texture.name = "REQ-NormalTexture"
normal_texture.image = bpy.data.images.load(normal_image_path, check_existing=True)
normal_texture.image.colorspace_settings.name = "Non-Color"
normal_map = nodes.get("REQ-NormalMap") or nodes.new("ShaderNodeNormalMap")
normal_map.name = "REQ-NormalMap"
normal_map.inputs["Strength"].default_value = 1.0
links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
```

Do not connect an RGB normal texture directly to Principled Normal. Treat a
DirectX-style green channel inversion as an explicit derived-image operation,
not a hidden material tweak.

## Configure alpha deliberately

Wire alpha first, then select a supported surface method by RNA inspection. In
Blender 5.1, `surface_render_method` exposes `DITHERED` and `BLENDED`; the legacy
`blend_method` may additionally expose `CLIP`.

```python
links.new(texture.outputs["Alpha"], principled.inputs["Alpha"])
requested = "CLIP"  # OPAQUE, CLIP, HASHED/DITHERED, or BLEND/BLENDED
if requested in {item.identifier for item in material.bl_rna.properties["blend_method"].enum_items}:
    material.blend_method = requested
elif requested in {item.identifier for item in material.bl_rna.properties["surface_render_method"].enum_items}:
    material.surface_render_method = requested
else:
    raise RuntimeError(f"unsupported alpha method in this Blender build: {requested}")
material.alpha_threshold = 0.5
```

Render alpha over a contrasting background. A node link and an enum assignment
do not prove that the target renderer preserved transparency.

## Assign material slots by polygon

Append each material once and set polygon indices from a deterministic face rule
or caller-supplied face map. Do not depend on the active slot.

```python
materials = [bpy.data.materials["MAT-body"], bpy.data.materials["MAT-trim"]]
for material in materials:
    if material.name not in {item.name for item in obj.data.materials if item}:
        obj.data.materials.append(material)
slot_by_name = {item.name: index for index, item in enumerate(obj.data.materials) if item}
for polygon in obj.data.polygons:
    requested_name = "MAT-trim" if polygon.normal.z > 0.9 else "MAT-body"
    polygon.material_index = slot_by_name[requested_name]
obj.data.update()
```

Report the exact face-selection rule. For imported topology, prefer a supplied
face map, material ID, or semantic region over normal-direction guesses.

## Prepare a bake target

Create or load the target image, add one selected image node to every baked
material, and make the mesh active before calling the bake operator. Baking and
saving are explicit writes and require a caller-selected output path.

```python
target = bpy.data.images.get("IMG-bake-target") or bpy.data.images.new(
    "IMG-bake-target", width=2048, height=2048, alpha=False, float_buffer=False
)
target.colorspace_settings.name = "Non-Color"  # change to sRGB for color bakes
target_node = nodes.get("REQ-BakeTarget") or nodes.new("ShaderNodeTexImage")
target_node.name = "REQ-BakeTarget"
target_node.image = target
for node in nodes:
    node.select = False
target_node.select = True
nodes.active = target_node
scene.render.engine = "CYCLES"
# Prepare object mode, selection, active object, UV layer, cage, and bake margin;
# then call bpy.ops.object.bake(type="NORMAL", margin=16) in its own transaction.
# Save with target.save_render(caller_selected_absolute_path, scene=scene).
```

Before baking, run `scripts/audit_material_uv.py`. Require a valid UV layer,
assigned materials, loaded sources, and a reachable material output. After
baking, rerun the audit and inspect the saved image for blank pixels and seams.
