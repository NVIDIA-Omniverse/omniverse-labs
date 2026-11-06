# SPDX-License-Identifier: Apache-2.0
"""Stock Blender fixture: authored visibility, restoration and generation lifecycle."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
import bpy
from pxr import Usd, UsdGeom
from ovrtx_blender_example import scene_generation as sg
from ovrtx_blender_example.object_visibility import scene_visibility, export_visibility

scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add()
parent = bpy.context.object
parent.name = "HiddenParent"
bpy.ops.mesh.primitive_cube_add(location=(3,0,0))
child = bpy.context.object
child.name = "IndependentChild"
child.parent = parent
bpy.ops.object.light_add(type="AREA", location=(0,0,3))
light = bpy.context.object
light.name = "RetainedLight"
parent.hide_render = True
light.hide_viewport = True
parent.visible_shadow = False
curve_data = bpy.data.curves.new("CurveData", "CURVE")
curve_data.dimensions = "3D"
curve_data.bevel_depth = .1
spline = curve_data.splines.new("POLY")
spline.points.add(1)
spline.points[0].co = (0,0,0,1)
spline.points[1].co = (1,0,0,1)
curve = bpy.data.objects.new("CurveOwner", curve_data)
scene.collection.objects.link(curve)
curve.hide_render = True
prototype = bpy.data.collections.new("Prototype")
proto_mesh = bpy.data.meshes.new("PrototypeMesh")
proto_mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
proto_obj = bpy.data.objects.new("PrototypeOwner", proto_mesh)
prototype.objects.link(proto_obj)
instance = bpy.data.objects.new("Instance", None)
instance.instance_type = "COLLECTION"
instance.instance_collection = prototype
scene.collection.objects.link(instance)
instance.hide_render = True
independent = bpy.data.objects.new("InstanceIndependentChild", parent.data.copy())
independent.parent = instance
scene.collection.objects.link(independent)
with tempfile.TemporaryDirectory() as temp:
    owner = sg.SceneGenerationOwner(Path(temp) / "generations")
    try:
        try:
            generation = owner.replace(scene)
        except sg.SceneGenerationError as exc:
            print(exc.diagnostics)
            raise
        parent_id, child_id, light_id = (sg.blender_id(obj, "OBJECT") for obj in (parent, child, light))
        def visibility(generation, identity):
            stage = Usd.Stage.Open(generation.usd_path)
            prim = stage.GetPrimAtPath(generation.blender_prim_paths[identity].schema_path)
            assert prim and prim.GetTypeName() != "Xform"
            return UsdGeom.Imageable(prim).ComputeVisibility()
        assert visibility(generation, parent_id) == "invisible"
        assert visibility(generation, child_id) == "inherited"
        assert visibility(generation, light_id) == "invisible"
        assert parent.hide_render and light.hide_viewport
        curve_id = sg.blender_id(curve, "OBJECT")
        curve_stage = Usd.Stage.Open(generation.usd_path)
        curve_children = curve_stage.GetPrimAtPath(generation.blender_prim_paths[curve_id].object_path).GetChildren()
        assert curve_children and all(UsdGeom.Imageable(prim).ComputeVisibility() == "invisible" for prim in curve_children)
        instance_id = sg.blender_id(instance, "OBJECT")
        instance_mapping = generation.blender_prim_paths[instance_id]
        instance_stage = Usd.Stage.Open(generation.usd_path)
        independent_id = sg.blender_id(independent, "OBJECT")
        assert visibility(generation, independent_id) == "inherited"
        branches = [prim for prim in instance_stage.GetPrimAtPath(instance_mapping.object_path).GetChildren() if str(prim.GetPath()) != generation.blender_prim_paths[independent_id].object_path]
        assert branches and all(UsdGeom.Imageable(prim).ComputeVisibility() == "invisible" for prim in branches)
        stage = Usd.Stage.Open(generation.usd_path)
        assert stage.GetPrimAtPath(generation.blender_prim_paths[parent_id].schema_path).GetAttribute("primvars:doNotCastShadows").Get() is True
        light.hide_viewport = False
        candidate = owner.reconcile(scene, {light_id})
        assert candidate.diagnostics["mode"] == "visibility_only"
        assert candidate.base_usd_path == generation.base_usd_path
        assert visibility(candidate, light_id) == "inherited"
        owner.reject(candidate)
        assert visibility(owner.current_generation, light_id) == "invisible"
        candidate = owner.reconcile(scene, {light_id})
        owner.accept(candidate)
        assert visibility(owner.current_generation, light_id) == "inherited"
        # Context scene A must not determine another scene's per-layer eye state.
        other = bpy.data.scenes.new("Other")
        other.collection.objects.link(light)
        layer_a, layer_b = scene.view_layers[0], other.view_layers[0]
        layer_a.update(); layer_b.update()
        light.hide_set(True, view_layer=layer_a)
        light.hide_set(False, view_layer=layer_b)
        assert not dict((obj.name,value) for obj,value in scene_visibility(scene))[light.name]
        assert dict((obj.name,value) for obj,value in scene_visibility(other))[light.name]
        light.hide_set(True, view_layer=layer_b)
        try:
            with export_visibility(bpy, other):
                assert not light.hide_get(view_layer=layer_b)
                assert light.hide_get(view_layer=layer_a)
                raise RuntimeError("test restoration")
        except RuntimeError:
            pass
        assert light.hide_get(view_layer=layer_a) and light.hide_get(view_layer=layer_b)
        print("VISIBILITY_PARITY_PASS: retained lights, independent children, shadow, rejection, owning layers and restoration")
    finally:
        owner.close()
