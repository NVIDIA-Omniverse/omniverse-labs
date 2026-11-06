"""Real full-generation bake capture, with no viewport publication consumer."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.physics_bake_capture import BakeCaptureError, BakeCaptureCancelled

p = argparse.ArgumentParser()
p.add_argument("--blend", type=Path, required=True)
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
a.device = "cpu"
configure(a)
props.register()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
for name, role, position, shape in (("ContactGround", "STATIC", (20, 0, -.5), "BOX"),
                                     ("UnconnectedContact", "DYNAMIC", (20, 0, .15), "ANALYTIC_SPHERE")):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = position
    initialize_owner(obj)
    body = obj.ov.physics.rigid_body
    body.schema_opinion = "APPLY"
    body.role = role
    collider = obj.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = shape
    collider.radius_m = .08
    collider.dimensions = (2, 2, 1)
bpy.context.view_layer.update()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
runtime = a.runtime.resolve()
server = Path(__file__).resolve() if a.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{sock.getsockname()[1]}"

def factory(config):
    return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
        ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", "cpu"]),
        device="cpu", worker_log_path=str(out / "worker.log"),
        ovphysx_native_client_path=str(runtime / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))

before_frame = (scene.frame_current, scene.frame_subframe)
before = [(channel, channel.matrix_basis.copy()) for obj in scene.objects
          for channel in (obj, *getattr(obj.pose, "bones", ()))]
settings = BakeCaptureSettings(1, 3, 8, 24, 1.001, interval=2, substeps=2)
session = BakeCaptureSession(scene, settings, out / "all", controller_factory=factory)
try:
    while not session.poll():
        time.sleep(.001)
finally:
    session.close()
samples = list(session.artifact.samples())
assert [x["frame"] for x in samples] == [3, 5, 7]
assert [x["step"] for x in samples] == [4, 8, 12]
assert all(len(x["poses"]) == 8 for x in samples)
assert all(len(x["context"]) == 8 for x in samples)
assert (scene.frame_current, scene.frame_subframe) == before_frame
assert all(channel.matrix_basis == matrix for channel, matrix in before)
assert session.artifact.manifest()["timing"]["duration_ns"] == 14 * settings.timestep_ns
contact = next(t["body_path"] for t in session.artifact.manifest()["targets"]
               if t["object_session_uid"] == bpy.data.objects["UnconnectedContact"].session_uid)
assert .06 < samples[-1]["poses"][contact]["position"][2] < .1
selected = BakeCaptureSession(scene, settings, out / "selected", output_paths=(contact,), controller_factory=factory)
while not selected.poll():
    time.sleep(.001)
assert selected.artifact.manifest()["output_paths"] == [contact]
assert list(selected.artifact.samples()) == samples

# Existing live native state is recreated and replayed, never teleported.
owner = SceneGenerationOwner(out / "accepted")
generation = owner.replace(scene)
live = sessions.AuthoringGenerationRuntime(SimpleNamespace(), owner)
live.ovphysx = OvphysxGenerationAdapter(factory)
assert live.ovphysx.activate(generation)
live.scheduler._borrow_controller(live.ovphysx.controller)
live.scheduler._last_timeline_frame = 37
for _ in range(5):
    live.ovphysx.controller.tick()
prior_steps = live.ovphysx.controller.step_count
prior_poses = live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns)
sessions._owners[int(scene.session_uid)] = owner
sessions._runtimes[int(scene.session_uid)] = live
try:
    cancelled = BakeCaptureSession(scene, settings, out / "cancelled", controller_factory=factory)
    while cancelled.phase == "begin":
        cancelled.poll()
        time.sleep(.001)
    cancelled.cancel()
    try:
        cancelled.poll()
        raise AssertionError("cancel accepted")
    except BakeCaptureCancelled:
        pass
    assert not (out / "cancelled").exists()
    assert live.ovphysx.controller.step_count == prior_steps
    assert live.ovphysx.controller.physics_pose_set(live.ovphysx.controller.current_simulation_time_ns) == prior_poses
    assert not live.scheduler.capture_active
    assert live.scheduler._controller is live.ovphysx.controller
    assert live.scheduler._last_timeline_frame == 37
    assert live.ovphysx.controller.tick().step_count > prior_steps
finally:
    live.ovphysx.deactivate()
    sessions._runtimes.pop(int(scene.session_uid))
    sessions._owners.pop(int(scene.session_uid))

failed = BakeCaptureSession(scene, settings, out / "missing", controller_factory=factory)
while failed.phase == "begin":
    failed.poll()
    time.sleep(.001)
# Fault injection only tests the failure path; successful samples above are native.
failed.future.result()
failed.worker.adapter.controller.physics_pose_set = lambda _time: ()
try:
    while not failed.poll():
        time.sleep(.001)
    raise AssertionError("missing pose accepted")
except BakeCaptureError as exc:
    assert "omitted" in str(exc)
assert not (out / "missing").exists()
assert all(channel.matrix_basis == matrix for channel, matrix in before)

changed = BakeCaptureSession(scene, settings, out / "changed", controller_factory=factory)
bpy.data.objects["UnconnectedContact"].ov.physics.rigid_body.mass_kg = 3
try:
    changed.poll()
    raise AssertionError("concurrent edit accepted")
except BakeCaptureError as exc:
    assert "authoring changed" in str(exc)
assert bpy.data.objects["UnconnectedContact"].ov.physics.rigid_body.mass_kg == 3
assert not (out / "changed").exists()
assert sessions._bake_capture_owner is None

# Preserve an existing destination Action and its bound slot. Offline sampling
# does not apply the transient pose mirror's Action ownership rejection.
destination = bpy.data.objects["MixedObject"]
destination.keyframe_insert(data_path="location", frame=1)
destination.keyframe_insert(data_path="location", frame=8)
action = destination.animation_data.action
slot = destination.animation_data.action_slot
action_capture = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 1, 24), out / "action", controller_factory=factory)
while not action_capture.poll():
    time.sleep(.001)
assert destination.animation_data.action is action and destination.animation_data.action_slot == slot
assert len(list(action_capture.artifact.samples())) == 1

# Collider geometry edited in place must cancel even without collection changes.
bpy.ops.mesh.primitive_cube_add(size=.1, location=(40, 0, 1))
mesh_source = bpy.context.object
mesh_source.name = "ColliderMeshSource"
contact_object = bpy.data.objects["UnconnectedContact"]
contact_object.ov.physics.colliders[0].shape = "CONVEX_HULL"
contact_object.ov.physics.colliders[0].mesh_source = mesh_source
geometry_capture = BakeCaptureSession(scene, settings, out / "geometry", controller_factory=factory)
mesh_source.data.vertices[0].co.x += .0125
try:
    geometry_capture.poll()
    raise AssertionError("concurrent geometry edit accepted")
except BakeCaptureError as exc:
    assert "authoring changed" in str(exc)
assert not (out / "geometry").exists()
assert sessions._bake_capture_owner is None
(out / "results.json").write_text(json.dumps({"passed_scenarios": 8, "samples": len(samples)}, indent=2))
print("PASS capture 8 scenarios", flush=True)
