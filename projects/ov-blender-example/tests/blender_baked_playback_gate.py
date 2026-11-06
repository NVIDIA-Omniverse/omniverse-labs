"""Actual native producer drain and durable render-only baked playback gate."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.physics_pose_presentation import apply
from ovrtx_blender_example.physics_presentation_targets import presentation_targets
from ovrtx_blender_example.runtime_scheduler import RuntimeTickRequest, RuntimeTickStatus
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from ovrtx_blender_example.interactive_edit_planner import InteractiveEdit, EditShape, DataAuthority
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--blend", required=True, type=Path)
p.add_argument("--runtime", default=Path("."), type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
configure(a)
props.register()
class TestBakeSettings(bpy.types.PropertyGroup):
    active: bpy.props.BoolProperty(default=False)
bpy.utils.register_class(TestBakeSettings)
bpy.types.Scene.ov_physics_bake = bpy.props.PointerProperty(type=TestBakeSettings)
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
scene.frame_set(42, subframe=.25)
before = {o.name: tuple(tuple(row) for row in o.matrix_basis) for o in scene.objects}
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
import faulthandler
_trace = (out / 'trace.log').open('w')
faulthandler.dump_traceback_later(30, repeat=True, file=_trace)
runtime_path = a.runtime.resolve()
server = runtime_path / "bin/ovphysx_grpc_server.exe"
if a.public_sdk:
    server = Path(__file__).resolve()
os.environ["OVPHYSX_ROOT"] = str(runtime_path / "private/ovphysx-runtime")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{sock.getsockname()[1]}"

calls = []
def factory(config):
    calls.append(config.input_usd_path)
    return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
        ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", a.device]),
        device=a.device, physics_fps=240, update_fps=240, max_steps=1000, worker_log_path=str(out / "worker.log"),
        ovphysx_native_client_path=str(runtime_path / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))

def wait(predicate):
    deadline = time.monotonic() + 45
    while not predicate():
        assert time.monotonic() < deadline, "gate deadline"
        time.sleep(.005)

owner = SceneGenerationOwner(out / "generation")
generation = owner.replace(scene)
uid = int(scene.session_uid)
runtime = sessions.AuthoringGenerationRuntime(SimpleNamespace(), owner)
runtime.scene_uid = uid
runtime.ovphysx = OvphysxGenerationAdapter(factory)
runtime.scheduler._controller_provider = lambda: runtime.ovphysx.controller
sessions._owners[uid], sessions._runtimes[uid] = owner, runtime
assert runtime.ovphysx.activate(generation)
controller = runtime.ovphysx.controller
for _ in range(5):
    controller.tick()
runtime.scheduler._borrow_controller(controller)
poses = {p.prim_path: {"translate": p.translate, "orient": p.orient}
         for p in controller.physics_pose_set(controller.current_simulation_time_ns)}
apply(scene, presentation_targets(generation), poses,
      runtime.playback_lock, generation.number, bpy.context.view_layer)
assert runtime.playback_lock.is_active()
controller.start_async_playback(max_steps_limit=1000)
scene.ov_physics_bake.active = True
future = sessions.set_baked_playback(scene, True)
wait(future.done)
sessions.finish_baked_playback_pause(scene, future)
assert runtime.ovphysx.controller is None and not runtime.playback_lock.is_active()
assert sessions.presentation_stopped(scene)
assert runtime.scheduler.baked_playback_paused
assert all(max(abs(value-before[name][i][j]) for i,row in enumerate(obj.matrix_basis) for j,value in enumerate(row)) < 1e-5
           for name,obj in ((o.name,o) for o in scene.objects))

class Port:
    writes = 0
    def update_transforms(self, values):
        self.writes += len(values)
        return OvrtxValueUpdateResult(len(values), 1 if values else None, {"status": "ok"})
port = Port()
path = presentation_targets(generation)[0].body_path
edit = InteractiveEdit(shape=EditShape.VALUE, data_authority=DataAuthority.SIM,
    blender_property_path="matrix_world", usd_prim_path=path,
    value=((1,0,0,0),(0,1,0,0),(0,0,1,0),(2,0,0,1)))
assert runtime.submit_edit_group((edit,))[0].accepted
assert not runtime.scheduler.has_pending_sim_updates
for frame in (3, 8, 2):
    scene.frame_set(frame)
    sessions.presentation_frame_changed(scene)
    sessions.animation_playback_pre(scene)
    result = runtime.scheduler.tick_viewport(RuntimeTickRequest(generation.usd_path,
        timeline_controls_enabled=True, timeline_playing=True, timeline_frame=frame), ovrtx_updates=port)
    assert result.status != RuntimeTickStatus.FAILED
    assert runtime.ovphysx.controller is None and sessions.presentation_stopped(scene)
assert port.writes > 0 and len(calls) == 1

# New runtimes inherit the gate; automatic preparation never starts physics.
sessions._runtimes.pop(uid)
runtime = sessions._runtime_for_owner(uid, owner)
runtime.ovphysx = OvphysxGenerationAdapter(factory)
runtime.scheduler._controller_provider = lambda: runtime.ovphysx.controller
assert runtime.scheduler.baked_playback_paused
sessions._wake_preparation(uid)
wait(lambda: runtime.preparation_status == "ready")
assert len(calls) == 1 and runtime.ovphysx.controller is None
scene.ov_physics_bake.active = False
future = sessions.set_baked_playback(scene, False)
wait(future.done)
sessions.finish_baked_playback_pause(scene, future)
sessions.presentation_frame_changed(scene)
assert sessions.presentation_stopped(scene) and runtime.scheduler.baked_playback_paused
joint = next(obj for obj in scene.objects if obj.ov.physics.joint.persistent_id)
joint_basis = joint.matrix_basis.copy()
joint.location.x += .5
bpy.context.view_layer.update()
try:
    sessions.demand_physics_playback(scene)
    raise AssertionError("pending joint admitted explicit resume")
except Exception as exc:
    assert "pending joint anchor" in str(exc)
assert runtime.scheduler.baked_playback_paused and sessions.presentation_stopped(scene)
runtime.preparation_status = "unavailable"
sessions.presentation_frame_changed(scene)
sessions._wake_preparation(uid)
wait(lambda: runtime.preparation_status == "ready")
assert runtime.ovphysx.controller is None and len(calls) == 1
assert runtime.scheduler.baked_playback_paused and sessions.presentation_stopped(scene)
joint.matrix_basis = joint_basis
bpy.context.view_layer.update()
sessions.demand_physics_playback(scene)
wait(lambda: runtime.ovphysx.controller is not None and runtime.preparation_status == "ready")
assert len(calls) == 2 and not runtime.scheduler.baked_playback_paused

# Persisted active Actions gate a fresh scene/runtime after load.
scene.ov_physics_bake.active = True
future = sessions.set_baked_playback(scene, True)
wait(future.done)
sessions.finish_baked_playback_pause(scene, future)
saved = out / "baked-active.blend"
sessions.register_handlers()
bpy.ops.wm.save_as_mainfile(filepath=str(saved))
bpy.ops.wm.open_mainfile(filepath=str(saved))
scene = bpy.context.scene
assert scene.ov_physics_bake.active
generation = sessions.generation_for_scene(scene, work_root=out / "loaded")
uid = int(scene.session_uid)
runtime = sessions._runtime_for_owner(uid, sessions._owners[uid])
assert runtime.scheduler.baked_playback_paused and sessions.presentation_stopped(scene)
try:
    sessions.demand_physics_playback(scene)
    raise AssertionError("active baked Actions admitted physics")
except RuntimeError as exc:
    assert "Deactivate baked Actions" in str(exc)
sessions.close()
(out / "results.json").write_text('{"passed_scenarios": 8}\n')
print("PASS baked playback gate 8 scenarios", flush=True)

