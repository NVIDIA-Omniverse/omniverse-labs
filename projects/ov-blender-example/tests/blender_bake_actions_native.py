"""Actual mixed native capture -> Actions -> durable ordinary Blender skinning."""
from pathlib import Path
import json
import runpy
import socket
import sys
import bpy
from mathutils import Vector

state = runpy.run_path(str(Path(__file__).with_name("blender_bake_conversion_native.py")))
from ovrtx_blender_example import physics_bake_ui as ui
from ovrtx_blender_example.physics_bake_actions import build_actions, activate as begin_activation, deactivate
from ovrtx_blender_example.physics_bake_conversion import _world, _body_error
from ovrtx_blender_example.physics_bake_conversion import convert_capture
from ovrtx_blender_example.physics_bake_capture import BakeCaptureArtifact
from ovrtx_blender_example.physics_pose_presentation import resolve
from ovrtx_blender_example import scene_generation_sessions as sessions
ui.register()
def activate(scene, batch):
    import time
    transaction = begin_activation(scene, batch)
    while not transaction.step():
        time.sleep(.001)  # Capture has closed its producer before this transaction.
scene, capture, converted, targets, out = [state[k] for k in ("scene", "capture", "converted", "targets", "out")]
batch = build_actions(scene, converted)
assert len(scene.ov_physics_bake.records) == len({targets[p].object_session_uid for p in converted.manifest()["output_paths"]})
activate(scene, batch)
assert sessions.presentation_stopped(scene)
# A normal Armature modifier and vertex group prove stock Blender deformation.
target = next(t for t in targets.values() if t.bone_identity)
rig, bone = resolve(scene, target)
vertex = bone.bone.head_local + Vector((.12, .15, .08))
mesh = bpy.data.meshes.new("BakedSkin")
mesh.from_pydata([vertex, vertex + Vector((.1, 0, 0)), vertex + Vector((0, .1, 0))], [], [(0, 1, 2)])
skin = bpy.data.objects.new("BakedSkin", mesh)
scene.collection.objects.link(skin)
skin.parent = rig
group = skin.vertex_groups.new(name=bone.name)
group.add([0, 1, 2], 1., "REPLACE")
modifier = skin.modifiers.new("Ordinary Blender Armature", "ARMATURE")
modifier.object = rig
oracle = {"frames": [], "owners": [], "skin": skin.name}
for path in converted.manifest()["output_paths"]:
    t = targets[path]
    obj, channel = resolve(scene, t)
    oracle["owners"].append(dict(path=path, object=obj.name, bone=t.bone_name, units=t.meters_per_blender_unit,
                                  offset=list(t.body_offset.position)))
checks = 0
for sample in capture.samples():
    scene.frame_set(sample["frame"])
    assert sessions.presentation_stopped(scene)
    expected = {}
    for path in converted.manifest()["output_paths"]:
        t = targets[path]
        obj, channel = resolve(scene, t)
        world = obj.matrix_world if channel is obj else obj.matrix_world @ channel.matrix
        desired = _world(t, sample["poses"][path])
        pos, rot = _body_error(world, desired, t)
        assert pos <= 1e-4 and rot <= 1e-4, (path, pos, rot)
        expected[path] = [list(row) for row in desired]
        checks += 1
    expected_vertex = _world(target, sample["poses"][target.body_path]) @ bone.bone.matrix_local.inverted() @ vertex
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    actual = evaluated.matrix_world @ evaluated.data.vertices[0].co
    assert (actual - expected_vertex).length * target.meters_per_blender_unit <= 1e-4
    oracle["frames"].append(dict(frame=sample["frame"], worlds=expected, vertex=list(expected_vertex)))
    checks += 1
host, port = state["address"].rsplit(":", 1)
with socket.socket() as sock:
    sock.settimeout(.5)
    assert sock.connect_ex((host, int(port))) != 0, "capture worker still running"
bpy.ops.wm.save_as_mainfile(filepath=str(out / "baked.blend"))
(out / "offline-oracle.json").write_text(json.dumps(oracle, indent=2))
assert all(r.action.use_fake_user for r in scene.ov_physics_bake.records)
deactivate(scene)
assert not scene.ov_physics_bake.active
assert all(r.owner.animation_data.action == r.previous_action for r in scene.ov_physics_bake.records)
import hashlib
import shutil
for kind in ("objects", "bones", "selected-bone"):
    paths = [p for p, t in targets.items() if bool(t.bone_identity) == (kind != "objects")]
    if kind == "selected-bone":
        paths = paths[:1]
    directory = out / kind
    directory.mkdir()
    shutil.copyfile(capture.directory / "samples.jsonl", directory / "samples.jsonl")
    manifest = dict(capture.manifest(), output_paths=paths)
    data = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(data)
    selected_capture = BakeCaptureArtifact(directory, hashlib.sha256(data).hexdigest())
    selected_conversion = convert_capture(scene, selected_capture, out / (kind + "-converted"))
    selected_batch = build_actions(scene, selected_conversion)
    activate(scene, selected_batch)
    for sample in capture.samples():
        scene.frame_set(sample["frame"])
        for path in paths:
            t = targets[path]
            obj, channel = resolve(scene, t)
            world = obj.matrix_world if channel is obj else obj.matrix_world @ channel.matrix
            pos, rot = _body_error(world, _world(t, sample["poses"][path]), t)
            assert pos <= 1e-4 and rot <= 1e-4, (kind, path, pos, rot)
            checks += 1
    deactivate(scene)
activate(scene, batch)
removed = next(r.owner for r in scene.ov_physics_bake.records if r.batch == batch and r.owner.type == "EMPTY")
bpy.data.objects.remove(removed, do_unlink=True)
deactivate(scene)
assert not scene.ov_physics_bake.active
assert all(r.owner is None or r.owner.animation_data.action == r.previous_action
           for r in scene.ov_physics_bake.records if r.batch == batch)
print("PASS native Actions/skin", checks, "save", out / "baked.blend", flush=True)
(out / "actions-results.json").write_text(json.dumps({"checks": checks, "saved_blend": str(out / "baked.blend")}, indent=2) + "\n")
