"""Blender RNA, stock-export and composed-stage SimReady conversion acceptance."""
import json
from pathlib import Path
import sys
import re
import bpy
from pxr import Sdf, Usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example import simready_authoring as conversion
from ovrtx_blender_example import simready_physics_conversion as legacy

out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
props.register()
checks = []

def check(label, value):
    assert value, label
    checks.append(label)

def fixture():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in tuple(bpy.data.collections):
        bpy.data.collections.remove(collection)
    scene = bpy.context.scene
    export = bpy.data.collections.new("Export")
    scene.collection.children.link(export)
    children = {}
    for name in ("Geometry", "ReferencePrims", "Colliders"):
        children[name] = bpy.data.collections.new(name)
        export.children.link(children[name])
    bpy.ops.mesh.primitive_cube_add()
    body = bpy.context.object
    body.name = "SimBody"
    for collection in tuple(body.users_collection):
        collection.objects.unlink(body)
    children["Geometry"].objects.link(body)
    reference = bpy.data.objects.new("Reference", None)
    children["ReferencePrims"].objects.link(reference)
    body.constraints.new("CHILD_OF").target = reference
    for key, value in zip(legacy.OBJECT_MASS_KEYS, (2.375, (0.1, -0.2, 0.3), (0.2, 0.4, 0.6), (1., 0., 0., 0.))):
        body[key] = value
    material = bpy.data.materials.new("SimMaterial")
    for key, value in zip(legacy.MATERIAL_PHYSICS_KEYS, (650., .35, .4, .5)):
        material[key] = value
    body.data.materials.append(material)
    bpy.context.view_layer.update()
    return body, material, children

def compiled(label):
    path = out / (label + ".usdc")
    generation._stock_export(bpy.context.scene, path)
    mappings = generation._validated_blender_prim_paths(bpy.context.scene, path)
    records = generation._compile_add_on_opinions(bpy.context.scene, mappings, path)
    stage = Usd.Stage.Open(str(path))
    text = conversion.composed_text(records, stage.GetRootLayer())
    text = re.sub(r"custom int ovrtx:(object|material)_session_uid = \d+", r"custom int ovrtx:\1_session_uid = 0", text)
    (out / (label + ".usda")).write_text(text)
    return text

body, material, children = fixture()
before = compiled("before")
body_keys = set(body.ov.physics.rigid_body.keys())
material_keys = set(material.ov.physics.keys())
def fail_after_write(*args):
    conversion.verify_equivalent(*args)
    raise RuntimeError("injected validation failure")
try:
    conversion.convert_in_scene(bpy.context.scene, body, verifier=fail_after_write)
except RuntimeError as exc:
    check("injected post-write failure", str(exc) == "injected validation failure")
check("rollback marker collection and explicit fields", body.ov.physics.simready_version == 0 and not body.ov.physics.colliders and set(body.ov.physics.rigid_body.keys()) == body_keys and set(material.ov.physics.keys()) == material_keys)
check("rollback composed equivalence", compiled("rollback") == before)

extra = body.constraints.new("COPY_LOCATION")
try:
    conversion.convert_in_scene(bpy.context.scene, body)
except Exception:
    check("reject arbitrary constraints without removal", len(body.constraints) == 2 and body.ov.physics.simready_version == 0)
else:
    raise AssertionError("unsupported constraint accepted")
body.constraints.remove(extra)
other = bpy.data.objects.new("SeparateCollider", body.data.copy())
children["Colliders"].objects.link(other)
try:
    conversion.convert_in_scene(bpy.context.scene, body)
except Exception:
    check("separate colliders rejected", not body.ov.physics.colliders)
else:
    raise AssertionError("separate collider accepted")
bpy.data.objects.remove(other, do_unlink=True)

bpy.context.preferences.edit.use_global_undo = True
bpy.ops.ed.undo_push(message="Before SimReady conversion")
check("explicit operator", bpy.ops.ov.physics_convert_simready() == {"FINISHED"})
bpy.ops.ed.undo_push(message="After SimReady conversion")
check("equivalent composed migration", conversion.equivalent_text(before, compiled("converted")))
check("typed body collider material", body.ov.physics.rigid_body.mass_kg == 2.375 and body.ov.physics.colliders[0].shape == "SIMREADY_CONVEX_DECOMPOSITION" and material.ov.physics.density == 650.)
check("idempotence", not conversion.convert_in_scene(bpy.context.scene, body))
check("native constraint preserved", len(body.constraints) == 1 and body.constraints[0].type == "CHILD_OF")
body.ov.physics.rigid_body.mass_kg = 4.0
check("converted body remains editable", "float physics:mass = 4" in compiled("edited_body"))
body.ov.physics.rigid_body.mass_kg = 2.375
body.ov.physics.colliders[0].hide_from_render = True
check("converted visibility remains editable", 'token visibility = "invisible"' in compiled("edited_visibility"))
body.ov.physics.colliders[0].property_unset("hide_from_render")
bpy.ops.ed.undo()
body = bpy.data.objects["SimBody"]
check("undo full conversion", body.ov.physics.simready_version == 0 and not body.ov.physics.colliders)
bpy.ops.ed.redo()
body = bpy.data.objects["SimBody"]
check("redo full conversion", body.ov.physics.simready_version == 1 and len(body.ov.physics.colliders) == 1)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "simready.blend"))
bpy.ops.wm.open_mainfile(filepath=str(out / "simready.blend"))
body = bpy.data.objects["SimBody"]
check("reload typed values", body.ov.physics.simready_version == 1 and body.ov.physics.rigid_body.center_of_mass[0] > .09)
check("reload composed equivalence", conversion.equivalent_text(before, compiled("reloaded")))
body[legacy.OBJECT_MASS_KEYS[0]] = 1000.
check("retained metadata inactive", conversion.equivalent_text(before, compiled("old_metadata_ignored")))
body.ov.physics.colliders.clear()
text = compiled("empty_collection")
check("empty collection never revives SimReady collision", "convexDecomposition" not in text and 'prepend apiSchemas = ["CollisionAPI"' not in text)
check("no native solver", bpy.context.scene.rigidbody_world is None and all(o.rigid_body is None and o.rigid_body_constraint is None for o in bpy.context.scene.objects))
body, material, children = fixture()
bpy.context.scene.unit_settings.scale_length = .01
before_cm = compiled("before_cm")
check("centimeter conversion", conversion.convert_in_scene(bpy.context.scene, body))
check("centimeter inverse units and legacy local COM scale", abs(body.ov.physics.rigid_body.center_of_mass[0] - .1) < 1e-6 and abs(material.ov.physics.density - .00065) < 1e-9)
check("centimeter composed equivalence", conversion.equivalent_text(before_cm, compiled("after_cm")))
cm_text = compiled("cm_persist_before")
bpy.ops.wm.save_as_mainfile(filepath=str(out / "simready_cm.blend"))
bpy.ops.wm.open_mainfile(filepath=str(out / "simready_cm.blend"))
check("centimeter save reload", compiled("cm_persist_after") == cm_text)
body = bpy.data.objects["SimBody"]
material = body.ov.physics.colliders[0].physics_material
saved_path = material.ov.physics.preserved_prim_path
material.ov.physics.preserved_prim_path = "/World/SimBody"
try:
    compiled("material_collision")
except Exception as exc:
    check("preserved material cannot overwrite source", "overlaps source" in str(exc))
else:
    raise AssertionError("source material collision accepted")
material.ov.physics.preserved_prim_path = saved_path
check("shape UI excludes imported cooking", "SIMREADY_CONVEX_DECOMPOSITION" not in bpy.ops.ov.physics_collider_shape.get_rna_type().properties["shape"].enum_items.keys())
bpy.context.view_layer.objects.active = body
check("shape edit operator", bpy.ops.ov.physics_collider_shape(collider_id=body.ov.physics.colliders[0].persistent_id, shape="CONVEX_HULL") == {"FINISHED"})
check("shape switch replaces imported cooking", 'physics:approximation = "convexHull"' in compiled("shape_switched") and "convexDecomposition" not in compiled("shape_switched"))

# Measure the actual RNA round-trip error, separately from the comparator.
old_layer, new_layer = Sdf.Layer.CreateAnonymous(".usda"), Sdf.Layer.CreateAnonymous(".usda")
old_layer.ImportFromString(before_cm)
new_layer.ImportFromString(cm_text)
old_stage, new_stage = Usd.Stage.Open(old_layer), Usd.Stage.Open(new_layer)
old_values = old_stage.GetPrimAtPath("/World/SimBody").GetAttribute("physics:centerOfMass").Get()
new_values = new_stage.GetPrimAtPath("/World/SimBody").GetAttribute("physics:centerOfMass").Get()
errors = {"max_absolute": max(abs(a-b) for a,b in zip(old_values,new_values)), "max_relative": max(abs((a-b)/a) for a,b in zip(old_values,new_values)), "relative_tolerance": 3e-7, "absolute_tolerance": 1e-10}
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks, "centimeter_float32_roundtrip": errors}, indent=2))
print("SIMREADY_AUTHORING_PASS", len(checks))
