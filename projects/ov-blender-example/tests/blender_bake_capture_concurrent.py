"""Native capture rejects curve edits and preserves concurrent authoring on exit."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import socket
import subprocess
import sys
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.physics_bake_capture import _source_stamp, BakeCaptureError
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController

p = argparse.ArgumentParser()
p.add_argument("--blend", required=True, type=Path)
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", required=True, type=Path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
a.device = "cpu"
configure(a)
props.register()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
scene.frame_set(42, subframe=.25)
before = {o.name: tuple(tuple(row) for row in o.matrix_basis) for o in scene.objects}
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

obj = bpy.data.objects["MixedObject"]
obj.keyframe_insert(data_path="location", frame=1)
obj.location.y += 2
obj.keyframe_insert(data_path="location", frame=42)
scene.frame_set(1)
curve = obj.animation_data.action.layers[0].strips[0].channelbag(obj.animation_data.action_slot).fcurves[0]
capture = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 4, 24), out / "modifier", controller_factory=factory)
capture.future.result()
stamp = _source_stamp(scene)
noise = curve.modifiers.new("NOISE")
noise.use_restricted_range = True
noise.frame_start, noise.frame_end, noise.strength = 3, 4, 12
assert _source_stamp(scene) != stamp
try:
    capture.poll()
    raise AssertionError("concurrent FCurve modifier accepted")
except BakeCaptureError as exc:
    assert "authoring changed" in str(exc)
assert capture.artifact is None and not (out / "modifier").exists()
curve.modifiers.remove(noise)

def changed_by(edit):
    stamp = _source_stamp(scene)
    edit()
    assert _source_stamp(scene) != stamp

envelope = curve.modifiers.new("ENVELOPE")
point = envelope.control_points.add(3)
changed_by(lambda: setattr(point, "min", -.25))
changed_by(lambda: setattr(curve, "extrapolation", "LINEAR"))
changed_by(lambda: setattr(curve.keyframe_points[0], "easing", "EASE_IN"))
curve.modifiers.remove(envelope)
generator = curve.modifiers.new("GENERATOR")
changed_by(lambda: setattr(generator, "coefficients", (2.0, 1.0)))
curve.modifiers.remove(generator)
driver = obj.driver_add("scale", 0)
driver.driver.expression = "1"
changed_by(lambda: driver.modifiers.new("NOISE"))
changed_by(lambda: setattr(driver.driver, "use_self", True))
obj.driver_remove("scale", 0)
action, slot = obj.animation_data.action, obj.animation_data.action_slot
track = obj.animation_data.nla_tracks.new()
strip = track.strips.new("Bound", 1, action)
strip.action_slot = slot
strip.use_animated_influence = True
strip.keyframe_insert(data_path="influence", frame=1)
changed_by(lambda: strip.fcurves[0].modifiers.new("NOISE"))
obj.animation_data.nla_tracks.remove(track)

# x is authored during capture, y belongs to the temporary evaluated frame.
# Every exit must preserve x but restore y to the original frame's value.
for route in ("cancel", "direct", "save", "disable"):
    scene.frame_set(42)
    original_y = obj.location.y
    capture = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 4, 24), out / route, controller_factory=factory)
    capture.future.result()
    assigned_action = obj.animation_data.action
    assigned_slot = obj.animation_data.action_slot
    if route == "direct":
        assigned_action = assigned_action.copy()
        obj.animation_data.action = assigned_action
        assigned_slot = assigned_action.slots[0]
        obj.animation_data.action_slot = assigned_slot
    expected_x = obj.location.x + 100
    obj.location.x = expected_x
    if route == "cancel":
        capture.cancel()
        capture.close()
    elif route == "direct":
        capture.close()
    elif route == "save":
        sessions.register_handlers()
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "concurrent-save.blend"))
        sessions.unregister_handlers()
    else:
        sessions.close()
    assert capture.closed and capture.artifact is None
    assert obj.location.x == expected_x, route
    assert obj.location.y == original_y, route
    assert obj.animation_data.action is assigned_action and obj.animation_data.action_slot == assigned_slot
    assert not (out / route).exists(), route
    assert sessions._bake_capture_owner is None
(out / "results.json").write_text('{"passed_scenarios": 12}\n')
print("PASS concurrent capture repair 12 scenarios", flush=True)
