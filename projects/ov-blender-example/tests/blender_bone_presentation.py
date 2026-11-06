"""Main-thread real Blender/native body output, hierarchy, ownership and cleanup."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import bpy
from mathutils import Matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock, prepare_runtime_pose_mirror, apply_pending_runtime_pose_mirror
from ovrtx_blender_example.physics_presentation_targets import presentation_targets
from ovrtx_blender_example.physics_pose_presentation import resolve
from ovrtx_blender_example.physics_frames import PhysicsUnits, evaluated_owner_frame, RigidFrame
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController

p = argparse.ArgumentParser()
p.add_argument("--blend", required=True, type=Path)
p.add_argument("--runtime", default=Path("."), type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
p.add_argument("--inherit-none", action="store_true")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--")+1:])
configure(a)
props.register()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
if a.inherit_none:
    for obj in scene.objects:
        if obj.type == "ARMATURE":
            for bone in obj.data.bones:
                if bone.parent:
                    bone.inherit_scale = "NONE"
    bpy.context.view_layer.update()
out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
checks = []
close_observations = []
def check(name, value):
    assert value, name
    checks.append(name)

def wait_for_native_close(label):
    # RNA ownership and channels must already be restored on the caller.
    # Native owner disposal is explicitly deferred by undo/disable/load.
    started = time.monotonic()
    while True:
        status = sessions.close_status()
        if status['status'] != 'closing' or time.monotonic() - started >= 30:
            break
        time.sleep(.01)
    row = dict(label=label, seconds=time.monotonic() - started, status=status,
               retained_runtime_uids=list(sessions._runtimes))
    close_observations.append(row)
    (out / 'close-observations.json').write_text(json.dumps(close_observations, indent=2))
    check(label + ' native cleanup reaches closed', status['status'] == 'closed')
    check(label + ' releases all retained runtimes', not sessions._runtimes)
owner = SceneGenerationOwner(out / "generations")
generation = owner.replace(scene)
sessions._owners[int(scene.session_uid)] = owner
targets = presentation_targets(generation)
lock = PhysicsPlaybackLock()
initial = {t.body_path: resolve(scene, t)[1].matrix_basis.copy() for t in targets}
initial_locks = {t.body_path: tuple(resolve(scene, t)[1].lock_location) for t in targets}
initial_named = {(resolve(scene,t)[0].name, resolve(scene,t)[1].name if t.kind == "BONE" else ""): matrix.copy()
                 for t in targets for matrix in (initial[t.body_path],)}
rig = bpy.data.objects["RigA"]
control = rig.pose.bones["Child"]
mesh = bpy.data.meshes.new("Skin")
mesh.from_pydata([(0, 100, 0), (10, 100, 0), (0, 110, 0)], [], [(0, 1, 2)])
skin = bpy.data.objects.new("Skin", mesh); scene.collection.objects.link(skin)
skin.matrix_world = rig.matrix_world.copy()
group = skin.vertex_groups.new(name=control.name); group.add([0, 1, 2], 1, "REPLACE")
modifier = skin.modifiers.new("Deform", "ARMATURE"); modifier.object = rig
attachment = bpy.data.objects.new("Attachment", None); scene.collection.objects.link(attachment)
attachment.parent = rig; attachment.parent_type = "BONE"; attachment.parent_bone = control.name
bpy.context.view_layer.update()
def skin_position():
    evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return evaluated.matrix_world @ evaluated.data.vertices[0].co
before_skin = skin_position().copy(); before_attachment = attachment.matrix_world.copy()
runtime = a.runtime.resolve(); server = runtime / "bin/ovphysx_grpc_server.exe"
if a.public_sdk:
    server = Path(__file__).resolve()
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0)); address = "127.0.0.1:" + str(sock.getsockname()[1])
def factory(config):
    return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
        ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", a.device]),
        device=a.device, physics_fps=240, update_fps=240, max_steps=240,
        worker_log_path=str(out / "worker.log"), ovphysx_native_client_path=str(runtime / "client"),
        ovphysx_native_client_module="ovphysx_bridge_client"))
adapter = OvphysxGenerationAdapter(factory)
try:
    check("actual CPU activation", adapter.activate(generation))
    for step in range(120):
        result = adapter.controller.tick()
        if step == 29:
            expected_replay = result.pose_set
    number, epoch, descriptors = sessions.current_presentation_context(scene)
    from ovrtx_blender_example.physics_edit_snapshot import capture as capture_edits
    pointer_flags = [c.is_property_set("mesh_source") for t in targets for c in resolve(scene,t)[1].ov.physics.colliders]
    first_capture = capture_edits(scene)
    check("repeated capture preserves sparse opinions", first_capture == capture_edits(scene)
          and pointer_flags == [c.is_property_set("mesh_source") for t in targets for c in resolve(scene,t)[1].ov.physics.colliders])
    sessions.observe_physics_edits(scene)
    pending, _ = prepare_runtime_pose_mirror(result.pose_set, bpy, bpy.context, presentation_targets=descriptors,
        scene_generation_number=number, presentation_epoch=epoch, owning_generation=result.generation)
    applied = apply_pending_runtime_pose_mirror(bpy, pending, lock)
    check("complete same cross mixed bone presentation", applied["mirrored_count"] == 7)
    previous_snapshot = sessions._physics_snapshots[int(scene.session_uid)]
    changes = sessions.observe_physics_edits(scene)
    if changes.restart or changes.body_input:
        after_snapshot = sessions._physics_snapshots[int(scene.session_uid)]
        before_values = {row[:4]: row[4:] for row in previous_snapshot.leaves}
        print("FEEDBACK", changes, [(row[:4], before_values.get(row[:4]), row[4:]) for row in after_snapshot.leaves if before_values.get(row[:4]) != row[4:]], flush=True)
    check("presentation does not queue physics authoring restart", not changes.restart and not changes.body_input)
    mass = control.ov.physics.rigid_body.mass_kg
    control.ov.physics.rigid_body.mass_kg = mass + 1
    check("durable authoring edit still detected", sessions.observe_physics_edits(scene).restart)
    control.ov.physics.rigid_body.mass_kg = mass
    sessions.observe_physics_edits(scene)
    for target in targets:
        obj, channel = resolve(scene, target)
        actual = evaluated_owner_frame(obj, bpy.context.evaluated_depsgraph_get(), bone_name=channel.name if target.kind == "BONE" else None,
                                       units=PhysicsUnits.from_scene(scene)).compose(target.body_offset)
        expected = next(x for x in result.pose_set if x.prim_path == target.body_path)
        check("position " + target.body_identity, math.dist(actual.position, expected.translate) < 2e-5)
        q = expected.orient
        dot = abs(sum(x*y for x,y in zip(actual.rotation_wxyz, (q[3], q[0], q[1], q[2]))))
        check("orientation " + target.body_identity, dot > 1-2e-6)
        check("owned location locked", all(channel.lock_location))
        check("scale channel left unchanged", not any(channel.lock_scale))
    check("Blender skinned deformation", (skin_position() - before_skin).length > .01)
    check("Blender rigid bone attachment", (attachment.matrix_world.translation - before_attachment.translation).length > .01)
    from ovrtx_blender_example.physics_authoring_snapshot import capture
    overlay = capture(scene, bpy.context.view_layer, bpy.context.evaluated_depsgraph_get(), generation=number)
    check("overlay follows evaluated bone output", len(overlay.colliders) == 7)
    for target in targets:
        collider = next(c for c in overlay.colliders if c.identity.startswith(target.body_identity + ":"))
        native = next(p for p in result.pose_set if p.prim_path == target.body_path)
        check("overlay world position " + target.body_identity,
              math.dist(tuple(v * scene.unit_settings.scale_length for v in collider.frame.position), native.translate) < 2e-5)
    lock.clear(reason="stop")
    bpy.context.view_layer.update()
    check("initial basis restored", all(resolve(scene,t)[1].matrix_basis == initial[t.body_path] for t in targets))
    check("original locks restored", all(tuple(resolve(scene,t)[1].lock_location) == initial_locks[t.body_path] for t in targets))
    check("restored skin", (skin_position() - before_skin).length < .001)
    constraint = control.constraints.new("COPY_LOCATION")
    constraint.target = attachment
    failed = apply_pending_runtime_pose_mirror(bpy, pending, lock)
    check("constraint conflict rejected", not failed["enabled"] and "constraints conflict" in failed["reason"])
    check("constraint preserved", control.constraints.get(constraint.name) is not None)
    control.constraints.remove(constraint)
    # Armature Object animation is not an output-bone channel conflict.
    rig.keyframe_insert(data_path="location", frame=1)
    check("unaffected armature animation admitted", apply_pending_runtime_pose_mirror(bpy, pending, lock)["mirrored_count"] == 7)
    lock.clear(reason="animation_test")
    control.keyframe_insert(data_path="location", frame=1)
    failed = apply_pending_runtime_pose_mirror(bpy, pending, lock)
    check("owned bone action conflict rejected", not failed["enabled"] and "animation or drivers" in failed["reason"])
    check("source Action preserved", rig.animation_data.action is not None)
    rig.animation_data_clear()
    driver = control.driver_add("location", 0)
    driver.driver.expression = "0"
    failed = apply_pending_runtime_pose_mirror(bpy, pending, lock)
    check("owned driver rejected", not failed["enabled"] and "animation or drivers" in failed["reason"])
    control.driver_remove("location", 0)
    check("mirror resumes after explicit conflict removal", apply_pending_runtime_pose_mirror(bpy, pending, lock)["mirrored_count"] == 7)
    from ovrtx_blender_example.physics_pose_presentation import enforce
    owned_location = control.location.copy()
    control.location.x += 20
    enforce(lock)
    check("attempted owned edit discarded", (control.location - owned_location).length < .0001)
    sessions._runtimes[int(scene.session_uid)] = SimpleNamespace(playback_lock=lock)
    sessions.restore_physics_presentation(scene)
    check("stop handler restores authored bases", all(resolve(scene,t)[1].matrix_basis == initial[t.body_path] for t in targets))
    check("old queued mirror invalid after stop", apply_pending_runtime_pose_mirror(bpy, pending, lock)["status"] == "stale")
    sessions._runtimes.pop(int(scene.session_uid))
    stale = dict(pending, scene_generation_number=number + 1)
    check("stale generation rejected", apply_pending_runtime_pose_mirror(bpy, stale, lock)["status"] == "stale")
    from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
    class Projection:
        def update_transforms(self, values):
            return OvrtxValueUpdateResult(updated_count=len(values), pending_simulation_time_ns=0 if values else None)
    scheduler = RuntimeScheduler(controller_provider=lambda: adapter.controller, controller_reset=adapter.reset)
    scheduler._last_timeline_frame = 121
    predecessor = adapter.controller
    replay = scheduler.tick_viewport(RuntimeTickRequest(generation.usd_path, timeline_controls_enabled=True,
        timeline_frame=31, timeline_start=1, timeline_end=241), ovrtx_updates=Projection())
    check("backward scrub recreates native solver", replay.timeline_reset and adapter.controller is not predecessor)
    check("backward scrub replays thirty fixed steps", replay.step_count == 30)
    check("replayed poses match original native history", all(math.dist(p.translate, next(q.translate for q in replay.physics_pose_set if q.prim_path == p.prim_path)) < 2e-5 for p in expected_replay))
    number, epoch, descriptors = sessions.current_presentation_context(scene)
    pending, _ = prepare_runtime_pose_mirror(replay.physics_pose_set, bpy, bpy.context, presentation_targets=descriptors,
        scene_generation_number=number, presentation_epoch=epoch, owning_generation=replay.generation)
    check("mirror resumed after replay", apply_pending_runtime_pose_mirror(bpy, pending, lock)["mirrored_count"] == 7)
    old_rest = rig.data.bones[control.name].inherit_scale
    rig.data.bones[control.name].inherit_scale = "FULL" if old_rest == "NONE" else "NONE"
    failed = apply_pending_runtime_pose_mirror(bpy, pending, lock)
    check("rest configuration edit rejects stale display", not failed["enabled"] and "rest topology changed" in failed["reason"])
    rig.data.bones[control.name].inherit_scale = old_rest
    check("error restored initial bases", all(resolve(scene,t)[1].matrix_basis == initial[t.body_path] for t in targets))
    apply_pending_runtime_pose_mirror(bpy, pending, lock)
    sessions._runtimes[int(scene.session_uid)] = SimpleNamespace(playback_lock=lock)
    bpy.app.handlers.save_pre.append(sessions.restore_physics_presentation)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "authoritative.blend"))
    finally:
        bpy.app.handlers.save_pre.remove(sessions.restore_physics_presentation)
        sessions._runtimes.pop(int(scene.session_uid))
    check("save restores transient ownership", not lock.is_active())
finally:
    lock.clear(reason="test_cleanup")
    adapter.deactivate()
    sessions._owners.pop(int(scene.session_uid), None)
bpy.ops.wm.open_mainfile(filepath=str(out / "authoritative.blend"))
check("save reload preserved authored initial basis", all(
    (bpy.data.objects[obj].pose.bones[bone].matrix_basis if bone else bpy.data.objects[obj].matrix_basis) == matrix
    for (obj, bone), matrix in initial_named.items()))

def owned_display(folder):
    current = bpy.context.scene
    owner = SceneGenerationOwner(out / folder)
    generation = owner.replace(current)
    sessions._owners[int(current.session_uid)] = owner
    runtime = sessions.AuthoringGenerationRuntime(SimpleNamespace(deactivate=lambda: "not_found"), owner)
    sessions._runtimes[int(current.session_uid)] = runtime
    number, epoch, targets = sessions.current_presentation_context(current)
    work, _ = prepare_runtime_pose_mirror(replay.physics_pose_set, bpy, bpy.context, presentation_targets=targets,
        scene_generation_number=number, presentation_epoch=epoch, owning_generation=replay.generation)
    applied = apply_pending_runtime_pose_mirror(bpy, work, runtime.playback_lock)
    check(folder + " owns live Blender channels", applied["mirrored_count"] == 7)
    return runtime

bpy.context.preferences.edit.use_global_undo = True
bpy.ops.ed.undo_push(message="authored initial")
bpy.data.objects["Skin"].location.x += 1
bpy.ops.ed.undo_push(message="independent edit")
undo_runtime = owned_display("undo-generation")
bpy.app.handlers.undo_pre.append(sessions.presentation_undo_pre)
try:
    bpy.ops.ed.undo()
finally:
    bpy.app.handlers.undo_pre.remove(sessions.presentation_undo_pre)
check("real undo releases all output ownership", not undo_runtime.playback_lock.is_active())
check("real undo preserves initial bone channels", all(
    (bpy.data.objects[obj].pose.bones[bone].matrix_basis if bone else bpy.data.objects[obj].matrix_basis) == matrix
    for (obj, bone), matrix in initial_named.items()))
wait_for_native_close('real undo')
disable_runtime = owned_display("disable-generation")
sessions.unregister_handlers()
bpy.context.view_layer.update()
check("disable releases all output ownership", not disable_runtime.playback_lock.is_active())
check("disable restores initial channels", all(
    (bpy.data.objects[obj].pose.bones[bone].matrix_basis if bone else bpy.data.objects[obj].matrix_basis) == matrix
    for (obj, bone), matrix in initial_named.items()))
wait_for_native_close('disable')
load_runtime = owned_display("load-generation")
bpy.app.handlers.load_pre.append(sessions.load_pre)
try:
    bpy.ops.wm.open_mainfile(filepath=str(out / "authoritative.blend"))
finally:
    bpy.app.handlers.load_pre.remove(sessions.load_pre)
check("real load releases stale output ownership", not load_runtime.playback_lock.is_active())
wait_for_native_close('real load')
(out / "results.json").write_text(json.dumps({"checks": checks, 'close_observations': close_observations}, indent=2))
print("BONE_PRESENTATION_OK", len(checks), flush=True)
