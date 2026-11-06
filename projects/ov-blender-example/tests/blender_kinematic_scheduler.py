"""Actual scheduler single-step inputs, playback gating and historical replay."""
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
import uuid
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.runtime_scheduler import RuntimeTickRequest, RuntimeTickStatus
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), required=True)
p.add_argument("--extended", action="store_true")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--")+1:])
configure(a)
a.output.mkdir(parents=True, exist_ok=True)
runtime = a.runtime.resolve()
server = Path(__file__).resolve() if a.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
props.register()
scene = bpy.context.scene
scene.render.fps, scene.render.fps_base = 24, 1.001
metrics = {}
for kind in ("OBJECT", "BONE"):
    for units in (1., .01):
        if a.extended and (kind != "OBJECT" or units != 1):
            continue
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene.unit_settings.scale_length = units
        scene.ov.physics_scene.schema_opinion = "APPLY"
        scene.ov.physics_scene.gravity_magnitude = 0
        if kind == "BONE":
            bpy.ops.object.armature_add()
            obj = bpy.context.object
            bpy.ops.object.mode_set(mode="EDIT")
            obj.data.edit_bones[0].tail = (0, 1, 0)
            bpy.ops.object.mode_set(mode="OBJECT")
            target = initialize_owner(obj, obj.data.bones[0])
        else:
            obj = bpy.data.objects.new("Input", None)
            scene.collection.objects.link(obj)
            target = initialize_owner(obj)
        obj.location = (-1/units, -.2/units, 1/units)
        obj.rotation_euler.z = math.pi/2
        obj.scale = (2, 2, 2)
        target.ov.physics.rigid_body.offset.translation = (.1/units, 0, 0)
        dyn = bpy.data.objects.new("Output", None)
        scene.collection.objects.link(dyn)
        dyn.location = (0, 0, 1/units)
        initialize_owner(dyn)
        for owner, role in ((target, "KINEMATIC"), (dyn, "DYNAMIC")):
            physics = owner.ov.physics
            physics.collider_version = 1
            physics.rigid_body.schema_opinion, physics.rigid_body.role = "APPLY", role
            c = physics.colliders.add()
            c.name = c.persistent_id = uuid.uuid4().hex
            c.schema_opinion, c.shape, c.radius_m = "APPLY", "ANALYTIC_SPHERE", .2
        channel = target if kind == "BONE" else obj
        axis = 1 if kind == "BONE" else 0
        channel.keyframe_insert("location", index=axis, frame=1)
        channel.location[axis] += -.5/units if kind == "BONE" else 1/units
        channel.keyframe_insert("location", index=axis, frame=25)
        action = obj.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation = "LINEAR"
        scene.frame_set(7)
        before = (tuple(channel.location), tuple(dyn.location), scene.frame_current, obj.animation_data.action)
        name = kind.lower()+"_"+str(units)
        out = a.output / name
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{sock.getsockname()[1]}"
        def factory(config):
            return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
                ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", a.device]),
                device=a.device, worker_log_path=str(a.output / (name+".log")),
                ovphysx_native_client_path=str(runtime / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))
        scene.frame_set(1)
        owner = SceneGenerationOwner(out / "generations")
        generation = owner.replace(scene)
        scene.frame_set(7)
        live = sessions.AuthoringGenerationRuntime(SimpleNamespace(), owner)
        live.scene_uid = int(scene.session_uid)
        live.ovphysx = OvphysxGenerationAdapter(lambda config: factory(replace(config,
            physics_fps=120, update_fps=24, max_steps=1000)))
        live.scheduler._controller_provider = lambda: live.ovphysx.controller
        live.scheduler._controller_reset = live.ovphysx.reset
        sessions._runtimes[live.scene_uid] = live
        sessions._owners[live.scene_uid] = owner
        sessions._presentation_stopped.add(live.scene_uid)
        class Port:
            def update_transforms(self, values):
                return OvrtxValueUpdateResult(len(values), 1 if values else None, {"status": "ok"})
        port = Port()
        request = RuntimeTickRequest(generation.usd_path, timeline_controls_enabled=True,
            timeline_playing=True, timeline_frame=25, timeline_start=1, timeline_end=25)
        executor = ThreadPoolExecutor(max_workers=1)
        def tick(request):
            if not a.extended:
                live.scheduler.prefetch_kinematic(request)  # as the render thread does, lock-free
                return live.scheduler.tick_viewport(request, ovrtx_updates=port)
            # Background Blender has no interactive timer loop. Service the same
            # dispatcher boundary on the main thread; all scheduler/native work
            # stays on one worker and the production sampler performs real RNA evaluation.
            from ovrtx_blender_example import engine
            work = queue.Queue()
            old = engine._run_on_main_thread
            def dispatch(callback):
                assert threading.current_thread() is not threading.main_thread()
                done = threading.Event()
                outcome = {}
                work.put((callback, done, outcome))
                assert done.wait(30)
                if "error" in outcome:
                    raise outcome["error"]
                return outcome["value"]
            engine._run_on_main_thread = dispatch
            try:
                def prefetch_then_tick():
                    live.scheduler.prefetch_kinematic(request)
                    return live.scheduler.tick_viewport(request, ovrtx_updates=port)
                future = executor.submit(prefetch_then_tick)
                while not future.done():
                    try:
                        callback, done, outcome = work.get(timeout=.01)
                    except queue.Empty:
                        continue
                    try:
                        outcome["value"] = callback()
                    except BaseException as exc:
                        outcome["error"] = exc
                    finally:
                        done.set()
                return future.result()
            finally:
                engine._run_on_main_thread = old
        try:
            assert live.ovphysx.activate(generation), live.ovphysx.last_error
            if a.extended:
                from ovrtx_blender_example.interactive_edit_planner import InteractiveEdit, EditShape, DataAuthority
                from ovrtx_blender_example.physics_kinematic_inputs import kinematic_targets
                input_path = kinematic_targets(generation)[0].body_path
                edit = InteractiveEdit(shape=EditShape.VALUE, data_authority=DataAuthority.SIM,
                    blender_property_path="matrix_world", usd_prim_path=input_path,
                    value=tuple(tuple(row) for row in obj.matrix_world))
                admission = live.submit_edit_group((edit,))[0]
                assert admission.accepted, admission
                assert admission.plan.edit.data_authority == DataAuthority.VIEW
                assert not live.scheduler.has_pending_sim_updates
            result = tick(request)
            assert result.status != RuntimeTickStatus.FAILED, result
            controller = live.ovphysx.controller
            assert controller.step_count == 120
            assert len(controller._kinematic_input_history) == 120
            assert controller.pose_publication_complete_count == 0
            paths = controller.dynamic_body_paths if hasattr(controller, "dynamic_body_paths") else controller.config.body_prims
            poses = controller.physics_pose_set(controller.current_simulation_time_ns)
            x = poses[0].translate[0]
            assert .39 < x < .42, x
            assert (tuple(channel.location), tuple(dyn.location), scene.frame_current, obj.animation_data.action) == before
            assert live.scene_uid in sessions._presentation_stopped
            assert abs(controller._kinematic_input_history[120].frame - (1+120*controller.config.timestep_ns/1e9*24/1.001)) < 1e-6
            backward = tick(replace(request, timeline_frame=13, timeline_playing=False))
            assert backward.status != RuntimeTickStatus.FAILED, backward
            assert live.ovphysx.controller.step_count == 60
            replay = tick(request)
            assert replay.status != RuntimeTickStatus.FAILED, replay
            actual = live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns)
            assert max(math.dist(p.translate, q.translate) for p, q in zip(poses, actual)) < 1e-6
            assert (tuple(channel.location), tuple(dyn.location), scene.frame_current, obj.animation_data.action) == before
            if a.extended:
                history = dict(live.ovphysx.controller._kinematic_input_history)
                capture = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 3, 24, substeps=5),
                    out / "cancelled-capture", controller_factory=factory)
                capture.future.result()
                capture.cancel()
                capture.close()
                restored = live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns)
                assert max(math.dist(p.translate, q.translate) for p, q in zip(actual, restored)) < 1e-6
                assert live.ovphysx.controller._kinematic_input_history == history
                assert live.scheduler._controller is live.ovphysx.controller
                next_request = replace(request, timeline_frame=26, timeline_end=26)
                assert tick(next_request).status != RuntimeTickStatus.FAILED
                continued = live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns)
                assert tick(replace(next_request, simulation_reset_token=1)).status != RuntimeTickStatus.FAILED
                fresh = live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns)
                assert max(math.dist(p.translate, q.translate) for p, q in zip(continued, fresh)) < 1e-6
                assert not (out / "cancelled-capture").exists()
                nonstart_owner = SceneGenerationOwner(out / "non-start")
                nonstart = nonstart_owner.replace(scene)
                assert nonstart.diagnostics["physics_initial_frame"] == 7
                assert live.ovphysx.activate(nonstart)
                predecessor = live.ovphysx.controller
                rejected = tick(request)
                assert rejected.status == RuntimeTickStatus.FAILED and "rebuild at the timeline start" in rejected.skipped_reason
                assert live.ovphysx.controller is predecessor and predecessor.step_count == 0
                assert (tuple(channel.location), tuple(dyn.location), scene.frame_current, obj.animation_data.action) == before
        finally:
            executor.shutdown(wait=True)
            live.ovphysx.deactivate()
            sessions._runtimes.pop(live.scene_uid)
            sessions._owners.pop(live.scene_uid)
            sessions._presentation_stopped.discard(live.scene_uid)
        metrics[name] = x
        print(name, x, flush=True)
        (a.output / "results.json").write_text(json.dumps(metrics, indent=2))
assert max(metrics.values())-min(metrics.values()) < 1e-4
print("KINEMATIC_SCHEDULER_PASS", a.device, len(metrics))
