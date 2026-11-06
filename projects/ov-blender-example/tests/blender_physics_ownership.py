"""Run with Blender --background --factory-startup --python this_file -- output_dir."""
import json
from pathlib import Path
import sys
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_endpoints as ep

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
checks = []

def check(name, condition):
    assert condition, name
    checks.append(name)

def rejected(name, call):
    try:
        call()
    except ep.EndpointError:
        checks.append(name)
    else:
        raise AssertionError(name)

def object_(name, data=None):
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj

def body(obj, bone=None):
    target = ep.initialize_owner(obj, bone)
    target.ov.physics.rigid_body.schema_opinion = "APPLY"
    return target

a = object_("Body")
body(a)
j = object_("Joint")
joint = ep.initialize_joint(j)
joint.endpoint_a.object = a
joint.endpoint_b.kind = "WORLD"
joint_copy = j.copy()
scene.collection.objects.link(joint_copy)
rejected("duplicate joint identity", lambda: ep.resolve_joint(joint, scene))
ep.repair_object_identity(joint_copy, scene)
check("duplicate joint repair", joint_copy.ov.physics.joint.persistent_id != joint.persistent_id)
check("semantic axes", tuple(x.name for x in joint.axes) == ep.AXIS_KEYS)
joint.axes["rotY"].drive.stiffness = 7
resolved = ep.resolve_joint(joint, scene)
old_id = resolved[0].body_identity
a.name = "Renamed"
check("object rename", ep.resolve_joint(joint, scene)[0].body_identity == old_id)
duplicate = a.copy()
scene.collection.objects.link(duplicate)
rejected("ambiguous object identity", lambda: ep.resolve_joint(joint, scene))
ep.repair_object_identity(duplicate, scene)
check("explicit object repair preserves pointer", ep.resolve_joint(joint, scene)[0].presentation_target == a)
check("distinct duplicate identity", duplicate.ov.physics.persistent_id != a.ov.physics.persistent_id)
joint.endpoint_b.kind = "OBJECT"
joint.endpoint_b.object = a
rejected("self joint", lambda: ep.resolve_joint(joint, scene))
joint.endpoint_b.object = object_("NoBody")
rejected("no manufactured bodies", lambda: ep.resolve_joint(joint, scene))
scene.collection.objects.unlink(joint.endpoint_b.object)
rejected("off scene", lambda: ep.resolve_joint(joint, scene))
joint.endpoint_b.object = None
rejected("missing object", lambda: ep.resolve_joint(joint, scene))
joint.endpoint_b.kind = "WORLD"
joint.endpoint_a.kind = "WORLD"
rejected("world world", lambda: ep.resolve_joint(joint, scene))
joint.endpoint_a.kind = "OBJECT"
rejected("namespace reservation", lambda: ep.physics_namespace(scene, ["/World/OVPhysics/StockMesh"]))

rig = object_("Rig", bpy.data.armatures.new("RigData"))
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
b = rig.data.edit_bones.new("Bone")
b.head, b.tail = (0, 0, 0), (0, 1, 0)
bpy.ops.object.mode_set(mode="OBJECT")
bone = rig.data.bones[0]
body(rig, bone)
ep.bind_bone_endpoint(joint.endpoint_a, rig, bone)
bone_id = bone.ov.persistent_id
before = ep.resolve_joint(joint, scene)[0]
bone.name = "RenamedBone"
check("bone rename", ep.resolve_joint(joint, scene)[0].body_identity == before.body_identity)
shared = object_("SharedRig", rig.data)
bpy.context.view_layer.update()
body(shared, bone)
shared.pose.bones[bone.name].ov.physics.rigid_body.mass_kg = 42
check("shared data per instance settings", rig.pose.bones[bone.name].ov.physics.rigid_body.mass_kg == 0)
joint.endpoint_b.kind = "BONE"
ep.bind_bone_endpoint(joint.endpoint_b, shared, bone)
ra, rb = ep.resolve_joint(joint, scene)
check("shared data distinct bodies", ra.body_identity != rb.body_identity and ra.body_path != rb.body_path)

rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
b = rig.data.edit_bones.new("CopyBone")
b.head, b.tail = (1, 0, 0), (1, 1, 0)
bpy.ops.object.mode_set(mode="OBJECT")
copy_bone = rig.data.bones["CopyBone"]
copy_bone.ov.persistent_id = bone_id
rejected("ambiguous bone", lambda: ep.resolve_joint(joint, scene))
ep.repair_bone_identity(rig, copy_bone)
check("repair does not guess endpoints", not joint.endpoint_a.bone_id and not joint.endpoint_b.bone_id)
ep.bind_bone_endpoint(joint.endpoint_a, rig, rig.data.bones["RenamedBone"])
ep.bind_bone_endpoint(joint.endpoint_b, shared, rig.data.bones["RenamedBone"])
check("rebinding after repair", len(ep.resolve_joint(joint, scene)) == 2)

library = out / "linked.blend"
bpy.data.libraries.write(str(library), {rig})
with bpy.data.libraries.load(str(library), link=True) as (source, dest):
    dest.objects = ["Rig"]
linked = dest.objects[0]
scene.collection.objects.link(linked)
rejected("linked read only", lambda: ep.initialize_owner(linked))
override = linked.override_create(remap_local_usages=False)
scene.collection.objects.link(override)
rejected("override explicit policy", lambda: ep.initialize_owner(override))
scene.collection.objects.unlink(linked)
scene.collection.objects.unlink(override)

# Explicit operator repair participates in Blender's actual undo stack.
undo_copy = bpy.data.objects["Renamed"].copy()
undo_copy.name = "UndoDuplicate"
scene.collection.objects.link(undo_copy)
bpy.context.view_layer.objects.active = undo_copy
bpy.context.preferences.edit.use_global_undo = True
bpy.ops.ed.undo_push(message="Before Identity Repair")
check("repair operator", bpy.ops.ov.physics_repair_identity() == {"FINISHED"})
repaired_id = undo_copy.ov.physics.persistent_id
bpy.ops.ed.undo_push(message="After Identity Repair")
bpy.ops.ed.undo()
check("undo identity repair", bpy.data.objects["UndoDuplicate"].ov.physics.persistent_id == bpy.data.objects["Renamed"].ov.physics.persistent_id)
bpy.ops.ed.redo()
check("redo identity repair", bpy.data.objects["UndoDuplicate"].ov.physics.persistent_id == repaired_id)
scene = bpy.context.scene

path = out / "ownership.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(path))
bpy.ops.wm.open_mainfile(filepath=str(path))
scene = bpy.context.scene
joint = bpy.data.objects["Joint"].ov.physics.joint
check("save reload body identities", ep.resolve_joint(joint, scene)[0].body_identity == before.body_identity)
check("save reload semantic drive", joint.axes["rotY"].drive.stiffness == 7)
check("no rigid body world", scene.rigidbody_world is None)
check("no native constraints", all(not o.constraints and o.rigid_body is None for o in scene.objects))
props.unregister()
check("unregister owner groups", not hasattr(bpy.types.PoseBone, "ov") and not hasattr(bpy.types.Bone, "ov"))
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("OWNERSHIP PASS", len(checks))
