"""Evaluated typed input identity, scale, units and output-feedback guards."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import uuid
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_kinematic_inputs import capture_kinematic_inputs
from ovrtx_blender_example.scene_generation import SceneGenerationOwner

props.register()
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
out = Path(sys.argv[sys.argv.index("--")+1])
out.mkdir(parents=True, exist_ok=True)
checks = []

def rejected(label, expected):
    bpy.context.view_layer.update()
    try:
        capture_kinematic_inputs(scene, generation)
    except ValueError as error:
        assert expected in str(error), str(error)
        checks.append(label)
    else:
        raise AssertionError(label)

bpy.ops.object.armature_add()
rig = bpy.context.object
bone = initialize_owner(rig, rig.data.bones[0])
dynamic = bpy.data.objects.new("Output", None)
scene.collection.objects.link(dynamic)
initialize_owner(dynamic)
for target, role in ((bone, "KINEMATIC"), (dynamic, "DYNAMIC")):
    target.ov.physics.rigid_body.schema_opinion, target.ov.physics.rigid_body.role = "APPLY", role
    c = target.ov.physics.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion, c.shape = "APPLY", "ANALYTIC_SPHERE"
bpy.context.view_layer.update()
generation = SceneGenerationOwner(out / "generation").replace(scene)
first = capture_kinematic_inputs(scene, generation)
bone.name = "Renamed Input"
rig.name = "Renamed Rig"
assert capture_kinematic_inputs(scene, generation).poses == first.poses
checks.append("stable Object and bone rename")
bone.location.x = 2
bpy.context.view_layer.update()
assert capture_kinematic_inputs(scene, generation).poses != first.poses
assert first.poses[0].translate == (0., 0., 0.)
checks.append("copied snapshot unaffected by evaluation")
bone.scale = (2, 2, 2)
rejected("animated scale rejected", "scale")
bone.scale = (1, 1, 1)
bone.ov.physics.rigid_body.offset.translation[0] = .2
rejected("offset edit requires restart", "offset")
bone.ov.physics.rigid_body.offset.translation[0] = 0
scene.unit_settings.scale_length = .01
rejected("unit edit requires restart", "units")
scene.unit_settings.scale_length = 1
bone.ov.physics.rigid_body.role = "DYNAMIC"
rejected("role cannot become dynamic input", "role")
bone.ov.physics.rigid_body.role = "KINEMATIC"
rig.parent = dynamic
rejected("dynamic parent feedback rejected", "dynamic output parent")
rig.parent = None
constraint = bone.constraints.new("COPY_LOCATION")
constraint.target = dynamic
rejected("dynamic constraint feedback rejected", "constraints")
bone.constraints.remove(constraint)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
sibling = rig.data.edit_bones.new("Sibling")
sibling.head, sibling.tail = (1, 0, 0), (1, 1, 0)
bpy.ops.object.mode_set(mode="OBJECT")
bone = rig.pose.bones["Renamed Input"]
constraint = rig.pose.bones["Sibling"].constraints.new("IK")
constraint.pole_target = dynamic
rejected("sibling IK pole dependency rejected", "indirect output feedback")
rig.pose.bones["Sibling"].constraints.remove(constraint)
curve = bone.driver_add("location", 0)
rejected("input driver unavailable", "drivers")
bone.driver_remove("location", 0)
duplicate = bpy.data.objects.new("Duplicate", None)
scene.collection.objects.link(duplicate)
duplicate.ov.physics.persistent_id = rig.ov.physics.persistent_id
rejected("duplicate Object identity rejected", "ambiguous")
bpy.data.objects.remove(duplicate, do_unlink=True)
old = bone.bone.ov.persistent_id
bone.bone.ov.persistent_id = uuid.uuid4().hex
rejected("missing stable bone rejected", "identity")
bone.bone.ov.persistent_id = old
assert capture_kinematic_inputs(scene, generation).poses
assert scene.rigidbody_world is None and not bone.constraints
checks.append("recovery and no native solver or persistent constraint edits")
(out / "results.json").write_text(json.dumps(checks, indent=2))
print("KINEMATIC_INPUTS_PASS", len(checks))
