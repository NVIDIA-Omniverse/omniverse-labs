"""Real Blender local-channel reconstruction, with no conversion RNA mutations."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Matrix, Quaternion, Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_bake_capture import BakeCaptureArtifact, BakeCaptureError, _conversion_context
from ovrtx_blender_example.physics_bake_conversion import convert_capture, _world, _body_error, _channels
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget

props.register()
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
root = Path(tempfile.mkdtemp(prefix="ov-bake-conversion-"))
checks = 0

def obj(name, parent=None):
    value = bpy.data.objects.new(name, None)
    scene.collection.objects.link(value)
    value.parent = parent
    initialize_owner(value)
    return value

def target(value, channel, scale):
    world = value.matrix_world if channel is value else value.matrix_world @ channel.matrix
    offset = RigidFrame((.12, -.09, .07), tuple(Quaternion((1, 0, 0), .31)))
    q = world.to_quaternion()
    initial = RigidFrame(tuple(x * scale for x in world.translation), tuple(q)).compose(offset)
    return PhysicsPresentationTarget("/" + value.name + ("_" + channel.name if channel is not value else ""),
        value.name + channel.name, value.session_uid, value.ov.physics.persistent_id,
        channel.bone.ov.persistent_id if channel is not value else "", channel.name if channel is not value else "",
        "DYNAMIC", True, scale, world.to_scale().x, offset, initial)

def fixture(scale, selected_parent, mode):
    global checks
    parent = obj(f"parent{checks}")
    intermediate = obj(f"intermediate{checks}", parent)
    intermediate.location = (.2, .3, -.4)
    intermediate.rotation_euler = (.1, .2, .3)
    intermediate.keyframe_insert(data_path="rotation_euler", frame=1)
    intermediate.rotation_euler.z += .4
    intermediate.keyframe_insert(data_path="rotation_euler", frame=3)
    child = obj(f"child{checks}", intermediate if selected_parent else parent)
    child.rotation_mode = mode
    child.matrix_parent_inverse = Matrix.Translation((.3, -.2, .4)) @ Quaternion((0, 1, 0), .2).to_matrix().to_4x4()
    child.delta_location = (.2, -.1, .4)
    child.delta_scale = (1.2, 1.2, 1.2)
    child.delta_rotation_euler = (.2, .1, -.3)
    child.delta_rotation_quaternion = Quaternion((1, 1, 0), .25)
    parent.location = (2, 3, 1)
    parent.keyframe_insert(data_path="location", frame=1)
    parent.location = (-2, 1, 4)
    parent.keyframe_insert(data_path="location", frame=3)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    targets = [target(parent, parent, scale), target(child, child, scale)]
    selected = targets if selected_parent else targets[1:]
    samples = []
    for frame in (1, 2, 3):
        scene.frame_set(frame)
        context = _conversion_context(scene, targets)
        poses = {}
        for i, t in enumerate(targets):
            q = Quaternion((0, 0, 1), 2.9 + .2 * frame + i * .1)
            poses[t.body_path] = {"position": (i + .2 * frame, .3, 2), "rotation_xyzw": (q.x, q.y, q.z, q.w)}
        samples.append(dict(frame=frame, poses=poses, context=context))
    check_capture(targets, selected, samples, [(parent, parent), (child, child)])

def check_capture(targets, selected, samples, owners):
    global checks, last_capture
    directory = root / str(checks)
    directory.mkdir()
    data = b"".join((json.dumps(s) + "\n").encode() for s in samples)
    (directory / "samples.jsonl").write_bytes(data)
    manifest = dict(targets=[asdict(t) for t in targets], output_paths=[t.body_path for t in selected],
                    sample_count=len(samples), samples_sha256=hashlib.sha256(data).hexdigest(), timing={})
    mdata = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(mdata)
    before = [(o, c, c.matrix_basis.copy(), o.animation_data.action if o.animation_data else None) for o, c in owners]
    last_capture = BakeCaptureArtifact(directory, hashlib.sha256(mdata).hexdigest())
    converted = convert_capture(scene, last_capture, root / (str(checks) + "-converted"))
    assert all(c.matrix_basis == m and (o.animation_data.action if o.animation_data else None) == a for o, c, m, a in before)
    rows = list(converted.samples())
    for sample, row in zip(samples, rows):
        scene.frame_set(sample["frame"])
        for t, (o, c) in zip(targets, owners):
            if t not in selected:
                continue
            for name, value in row["channels"][t.body_path].items():
                setattr(c, name, value)
        bpy.context.view_layer.update()
        for t, (o, c) in zip(targets, owners):
            if t in selected:
                world = o.matrix_world if c is o else o.matrix_world @ c.matrix
                pos, rot = _body_error(world, _world(t, sample["poses"][t.body_path]), t)
                assert pos <= 1e-4 and rot <= 1e-4, (t.body_path, pos, rot)
                checks += 1
    assert converted.manifest()["interpolation"] == "LINEAR"
    for t, (_, channel) in zip(targets, owners):
        if t in selected and channel.rotation_mode == "QUATERNION":
            quats = [Quaternion(row["channels"][t.body_path]["rotation_quaternion"]) for row in rows]
            assert all(a.dot(b) >= 0 for a, b in zip(quats, quats[1:]))
    if selected[-1] and owners[-1][1].rotation_mode == "XYZ":
        angles = [row["channels"][selected[-1].body_path]["rotation_euler"] for row in rows]
        assert all(max(abs(a - b) for a, b in zip(x, y)) < 1 for x, y in zip(angles, angles[1:]))
    for _, channel, basis, _ in before:
        channel.matrix_basis = basis
    bpy.context.view_layer.update()

for scale in (1, .01):
    for selection in (False, True):
        for mode in ("QUATERNION", "XYZ", "ZYX", "AXIS_ANGLE"):
            fixture(scale, selection, mode)

# Bone roots and parented children, animated armature Objects, full/no scale
# inheritance, and two independent armatures in the same immutable artifact.
for scale in (1, .01):
    for inherit in ("FULL", "NONE"):
        targets, owners = [], []
        for index in range(2):
            data = bpy.data.armatures.new(f"rig{checks}-{index}")
            rig = bpy.data.objects.new(data.name, data)
            scene.collection.objects.link(rig)
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            a = data.edit_bones.new("a")
            a.head, a.tail = (0, 0, .3), (.2, 1, .6)
            b = data.edit_bones.new("b")
            middle = data.edit_bones.new("middle")
            middle.head, middle.tail, middle.parent = (.2, 1, .6), (.3, 1.2, .8), a
            b.head, b.tail, b.parent = (.3, 1.2, .8), (.4, 2, 1), middle
            bpy.ops.object.mode_set(mode="OBJECT")
            initialize_owner(rig)
            rig.rotation_euler = (.2, -.1, .3)
            rig.location = (index * 4, 2, 1)
            rig.keyframe_insert(data_path="location", frame=1)
            rig.location.z += .6
            rig.keyframe_insert(data_path="location", frame=3)
            for c in rig.pose.bones:
                initialize_owner(rig, c.bone)
                c.bone.inherit_scale = inherit
                c.rotation_mode = "XYZ" if c.name == "a" else "QUATERNION"
                if c.name != "middle":
                    owners.append((rig, c))
            scene.frame_set(1)
            bpy.context.view_layer.update()
            targets.extend(target(rig, c, scale) for c in rig.pose.bones if c.name != "middle")
            if index == 1:
                targets.append(target(rig, rig, scale))
                owners.append((rig, rig))
            attachment = obj(f"boneAttachment{checks}-{index}", rig)
            attachment.parent_type, attachment.parent_bone = "BONE", "b"
            attachment.location = (.4, .2, .1)
            attachment.matrix_parent_inverse = Matrix.Translation((.3, -.1, .2))
            bpy.context.view_layer.update()
            targets.append(target(attachment, attachment, scale))
            owners.append((attachment, attachment))
        samples = []
        for frame in (1, 2, 3):
            scene.frame_set(frame)
            poses = {}
            for i, t in enumerate(targets):
                q = Quaternion((0, 1, 0), frame * .12 + i * .04)
                poses[t.body_path] = dict(position=(i * .5, .2, 2 + frame * .02), rotation_xyzw=(q.x, q.y, q.z, q.w))
            samples.append(dict(frame=frame, poses=poses, context=_conversion_context(scene, targets)))
        check_capture(targets, targets, samples, owners)
        check_capture(targets, [t for t in targets if t.bone_name != "a"], samples, owners)

# Rejected outputs never replace preexisting files or publish partial manifests.
existing = root / "existing"
existing.mkdir()
(existing / "sentinel").write_text("unchanged")
try:
    convert_capture(scene, last_capture, existing)
    raise AssertionError("existing output overwritten")
except FileExistsError:
    assert (existing / "sentinel").read_text() == "unchanged"
checks += 1
obj_ref = owners[-1][0]
original_identity = obj_ref.ov.physics.persistent_id
obj_ref.ov.physics.persistent_id = "changed-identity"
try:
    convert_capture(scene, last_capture, root / "changed-identity")
    raise AssertionError("changed output identity accepted")
except ValueError:
    assert not (root / "changed-identity").exists()
finally:
    obj_ref.ov.physics.persistent_id = original_identity
checks += 1
sample_file = last_capture.directory / "samples.jsonl"
original = sample_file.read_bytes()
for bad in (original.replace(b'"frame": 1', b'"frame": 0', 1), original[:original.rfind(b"\n", 0, -1) + 1]):
    sample_file.write_bytes(bad)
    destination = root / f"invalid-{checks}"
    try:
        convert_capture(scene, last_capture, destination)
        raise AssertionError("corrupt capture accepted")
    except BakeCaptureError:
        assert not destination.exists()
    checks += 1
sample_file.write_bytes(original)
obj_ref = owners[-1][0]
original_parent = obj_ref.parent
obj_ref.parent = None
try:
    convert_capture(scene, last_capture, root / "changed-parent")
    raise AssertionError("changed hierarchy accepted")
except BakeCaptureError:
    assert not (root / "changed-parent").exists()
finally:
    obj_ref.parent = original_parent
checks += 1
shear = Matrix.Identity(4)
shear[0][1] = .4
try:
    _channels(shear, {}, True, None)
    raise AssertionError("shear accepted")
except BakeCaptureError:
    checks += 1
print(json.dumps(dict(checks=checks, artifacts=str(root))), flush=True)
