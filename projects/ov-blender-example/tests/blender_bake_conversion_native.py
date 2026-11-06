"""Convert actual native mixed/cross-armature capture and evaluate local outputs."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.physics_bake_conversion import convert_capture, _target, _world, _error
from ovrtx_blender_example.physics_pose_presentation import resolve
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController

p = argparse.ArgumentParser()
p.add_argument("--blend", type=Path, required=True)
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
configure(a)
props.register()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
runtime = a.runtime.resolve()
server = runtime / "bin/ovphysx_grpc_server.exe"
if a.public_sdk:
    server = Path(__file__).resolve()
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{sock.getsockname()[1]}"

def factory(config):
    return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
        ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", a.device]),
        device=a.device, worker_log_path=str(out / "worker.log"),
        ovphysx_native_client_path=str(runtime / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))

session = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 9, 24), out / "capture", controller_factory=factory)
try:
    while not session.poll():
        time.sleep(.001)
finally:
    session.close()
capture = session.artifact
before = [(channel, channel.matrix_basis.copy()) for obj in scene.objects
          for channel in (obj, *getattr(obj.pose, "bones", ()))]
before_channels = [(channel, {name: tuple(getattr(channel, name)) for name in
                   ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")})
                   for channel, _ in before]
actions = tuple(bpy.data.actions)
frame = scene.frame_current
converted = convert_capture(scene, capture, out / "converted")
assert tuple(bpy.data.actions) == actions and scene.frame_current == frame
assert all(channel.matrix_basis == basis for channel, basis in before)
targets = {_target(row).body_path: _target(row) for row in capture.manifest()["targets"]}
checks = 0
maximum_position = maximum_rotation = 0.0
for original, local in zip(capture.samples(), converted.samples(), strict=True):
    scene.frame_set(original["frame"])
    for path, values in local["channels"].items():
        obj, channel = resolve(scene, targets[path])
        for name, value in values.items():
            setattr(channel, name, value)
    bpy.context.view_layer.update()
    for path in local["channels"]:
        t = targets[path]
        obj, channel = resolve(scene, t)
        world = obj.matrix_world if channel is obj else obj.matrix_world @ channel.matrix
        pos, rot = _error(world, _world(t, original["poses"][path]), t.meters_per_blender_unit)
        assert pos <= 1e-4 and rot <= 1e-4, (path, pos, rot)
        maximum_position, maximum_rotation = max(maximum_position, pos), max(maximum_rotation, rot)
        checks += 1
result = dict(checks=checks, max_position_error_m=maximum_position, max_rotation_error_rad=maximum_rotation,
              capture_digest=capture.manifest_digest, conversion_digest=converted.manifest_digest)
(out / "results.json").write_text(json.dumps(result, indent=2))
for channel, values in before_channels:
    for name, value in values.items():
        setattr(channel, name, value)
bpy.context.view_layer.update()
print(json.dumps(result), flush=True)
