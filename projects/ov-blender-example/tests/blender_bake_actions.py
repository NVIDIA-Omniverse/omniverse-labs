"""Installed Action slots, private copies, partial bone outputs and rollback."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import bpy
from mathutils import Quaternion
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props, physics_bake_ui as ui
from ovrtx_blender_example.physics_bake_actions import build_actions, ActionBuilder, activate as begin_activation, deactivate
from ovrtx_blender_example.physics_bake_capture import _conversion_context, _SampleWriter, BakeCaptureError
from ovrtx_blender_example.physics_bake_conversion import convert_capture, _target, _world, _body_error
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget
props.register()
ui.register()
def activate(scene, batch):
    import time
    transaction = begin_activation(scene, batch)
    while not transaction.step():
        time.sleep(.001)  # No live kinematic producer in this headless fixture.
scene = bpy.context.scene
root = Path(tempfile.mkdtemp(prefix="ov-bake-actions-"))
bpy.ops.object.armature_add()
rig = bpy.context.object
rig.name = "BakeRig"
bpy.ops.object.mode_set(mode="EDIT")
other = rig.data.edit_bones.new("Unaffected")
other.head, other.tail = (2, 0, 0), (2, 1, 0)
bpy.ops.object.mode_set(mode="OBJECT")
bone = rig.pose.bones["Bone"]
other = rig.pose.bones["Unaffected"]
initialize_owner(rig, bone.bone)
for frame in (1, 3):
    rig.location.x = frame * .1
    rig.keyframe_insert(data_path="location", frame=frame)
    other.location.z = frame * .2
    other.keyframe_insert(data_path="location", frame=frame)
original = rig.animation_data.action
slot = rig.animation_data.action_slot
shared = bpy.data.objects.new("SharedActionUser", None)
scene.collection.objects.link(shared)
shared.animation_data_create().action = original
shared.animation_data.action_slot = slot
target = PhysicsPresentationTarget("/Bone", bone.bone.ov.persistent_id, rig.session_uid,
    rig.ov.physics.persistent_id, bone.bone.ov.persistent_id, bone.name,
    "DYNAMIC", True, 1, 1, RigidFrame((.2, .1, .1), tuple(Quaternion((1, 0, 0), .2))), RigidFrame())
directory = root / "capture"
directory.mkdir()
writer = _SampleWriter(directory, ("/Bone",))
expected = []
for frame in (1, 2, 3):
    scene.frame_set(frame)
    expected.append((tuple(rig.location), tuple(other.location), tuple(shared.location)))
    writer.append(frame, frame - 1, [SimpleNamespace(prim_path="/Bone", translate=(0, 0, frame * .3),
        orient=(0, 0, 0, 1))], _conversion_context(scene, (target,)), 1)
capture = writer.finish(dict(targets=[asdict(target)], output_paths=["/Bone"], timing={}))
converted = convert_capture(scene, capture, root / "converted")
before_actions = set(bpy.data.actions)
old_delta = rig.delta_location.copy()
rig.delta_location.x += 3
try:
    build_actions(scene, converted)
    raise AssertionError("post-conversion delta edit accepted")
except BakeCaptureError:
    assert set(bpy.data.actions) == before_actions
rig.delta_location = old_delta
builder = ActionBuilder(scene, converted)
builder.step()
rig.delta_location.x += 3
try:
    while not builder.step():
        pass
    raise AssertionError("mid-build delta edit accepted")
except BakeCaptureError:
    assert set(bpy.data.actions) == before_actions
rig.delta_location = old_delta
batch = build_actions(scene, converted)
assert len(scene.ov_physics_bake.records) == 1
record = scene.ov_physics_bake.records[0]
assert record.previous_action == original and record.action != original
assert rig.animation_data.action == original and shared.animation_data.action == original
activate(scene, batch)
for frame, (location, unaffected, shared_location) in zip((1, 2, 3), expected):
    scene.frame_set(frame)
    assert tuple(rig.location) == location and tuple(other.location) == unaffected
    assert tuple(shared.location) == shared_location
    world = rig.matrix_world @ bone.matrix
    desired = _world(target, dict(position=(0, 0, frame * .3), rotation_xyzw=(0, 0, 0, 1)))
    pos, rot = _body_error(world, desired, target)
    assert pos <= 1e-4 and rot <= 1e-4
bpy.ops.wm.save_as_mainfile(filepath=str(root / "shared-active.blend"))
deactivate(scene)
assert rig.animation_data.action == original and rig.animation_data.action_slot == slot
assert shared.animation_data.action == original
# Async activation must revalidate after the drain future is complete, before assignment.
from ovrtx_blender_example.physics_bake_actions import Activation
activation = Activation(scene, batch)
rig.delta_location.x += 3
import time
try:
    while not activation.step():
        time.sleep(.001)
    raise AssertionError("mid-drain delta edit accepted")
except BakeCaptureError:
    assert rig.animation_data.action == original and not scene.ov_physics_bake.active
rig.delta_location = old_delta
before = set(bpy.data.actions)
builder = ActionBuilder(scene, converted)
builder.step()
builder.cancel()
assert set(bpy.data.actions) == before and rig.animation_data.action == original
sample_path = converted.directory / "samples.jsonl"
sample_path.rename(converted.directory / "samples-closed.jsonl").rename(sample_path)
second = build_actions(scene, converted)
assert second != batch and len(scene.ov_physics_bake.records) == 2
assert scene.ov_physics_bake.records[0].action != scene.ov_physics_bake.records[1].action
# Compatible NLA affects only a custom property and is preserved verbatim.
nla = bpy.data.actions.new("Compatible")
nla_slot = nla.slots.new(id_type="OBJECT", name=rig.name)
from bpy_extras.anim_utils import action_ensure_channelbag_for_slot
bag = action_ensure_channelbag_for_slot(nla, nla_slot)
rig["unrelated"] = 0.
curve = bag.fcurves.new(data_path='["unrelated"]', index=0)
curve.keyframe_points.insert(1, 2)
track = rig.animation_data.nla_tracks.new()
strip = track.strips.new("Compatible", 1, nla)
strip.action_slot = nla_slot
third = build_actions(scene, converted)
activate(scene, third)
scene.frame_set(2)
assert rig["unrelated"] == 2 and not track.mute and not strip.mute
deactivate(scene)
curve = bag.fcurves.new(data_path=bone.path_from_id() + ".location", index=0)
curve.keyframe_points.insert(1, 2)
before = set(bpy.data.actions)
try:
    build_actions(scene, converted)
    raise AssertionError("conflicting NLA accepted")
except BakeCaptureError:
    assert set(bpy.data.actions) == before and not track.mute and not strip.mute
area = bpy.context.screen.areas[0]
previous_type = area.type
area.type = "NLA_EDITOR"
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
strip.select = True
with bpy.context.temp_override(area=area, region=next(r for r in area.regions if r.type == "WINDOW")):
    bpy.ops.nla.meta_add()
assert track.strips[0].type == "META"
try:
    build_actions(scene, converted)
    raise AssertionError("nested conflicting NLA accepted")
except BakeCaptureError:
    assert set(bpy.data.actions) == before and not track.mute
area.type = previous_type
# Explicit clean mapping leaves the conflicting source NLA untouched.
from ovrtx_blender_example.physics_bake_destinations import remap_capture, rollback_mapping_identities
clean = rig.copy()
clean.data = rig.data.copy()
clean.name = "CleanOutput"
clean.animation_data_clear()
clean.scale = (2, 2, 2)
scene.collection.objects.link(clean)
mapping = SimpleNamespace(source=rig, source_bone=bone.name, destination=clean, destination_bone=bone.name)
# This synthetic capture used a minimal timing manifest; supply the fixed step.
import hashlib, json
from ovrtx_blender_example.physics_bake_capture import BakeCaptureArtifact
manifest = capture.manifest()
manifest.update(body_paths=["/Bone"], timing={"timestep_ns": 1})
raw = json.dumps(manifest).encode()
(capture.directory / "manifest.json").write_bytes(raw)
capture = BakeCaptureArtifact(capture.directory, hashlib.sha256(raw).hexdigest())
clean.pose.bones[bone.name].bone.ov.persistent_id = ""
existing_mapping = root / "existing-mapping"
existing_mapping.mkdir()
try:
    remap_capture(scene, capture, [mapping], existing_mapping)
    raise AssertionError("existing mapping directory accepted")
except FileExistsError:
    assert clean.pose.bones[bone.name].bone.ov.persistent_id == ""
cancelled_mapping = remap_capture(scene, capture, [mapping], root / "cancelled-mapping")
assert clean.pose.bones[bone.name].bone.ov.persistent_id
rollback_mapping_identities(scene, cancelled_mapping)
assert clean.pose.bones[bone.name].bone.ov.persistent_id == ""
mapped = remap_capture(scene, capture, [mapping], root / "mapped")
mapped_conversion = convert_capture(scene, mapped, root / "mapped-converted")
mapped_batch = build_actions(scene, mapped_conversion)
activate(scene, mapped_batch)
for frame in (1, 2, 3):
    scene.frame_set(frame)
    mapped_target = _target(mapped.manifest()["targets"][0])
    desired = _world(mapped_target, dict(position=(0, 0, frame * .3), rotation_xyzw=(0, 0, 0, 1)))
    pos, rot = _body_error(clean.matrix_world @ clean.pose.bones[bone.name].matrix, desired, mapped_target)
    assert pos <= 1e-4 and rot <= 1e-4
    assert abs((clean.matrix_world @ clean.pose.bones[bone.name].matrix).to_scale().x - 2) < 1e-4
assert rig.animation_data.action == original and not track.mute and not strip.mute
clean.pose.bones[bone.name].name = "RenamedWhileActive"
deactivate(scene)
assert not scene.ov_physics_bake.active and clean.animation_data.action is None
print("PASS Actions private copies, slots, shared users, partial bones, repeat, rollback and NLA", root, flush=True)
