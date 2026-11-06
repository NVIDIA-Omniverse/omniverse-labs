"""Saving during native capture cancels before serializing temporary view state."""
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

capture = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 4, 24), out / "capture", controller_factory=factory)
capture.future.result()
assert capture.worker.adapter.controller.started
assert scene.frame_current == 1
sessions.register_handlers()
saved = out / "cancelled-before-save.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(saved))
assert capture.closed and capture.cancelled and capture.artifact is None
assert not (out / "capture").exists()
assert sessions._bake_capture_owner is None
assert scene.frame_current == 42 and scene.frame_subframe == .25
assert before == {o.name: tuple(tuple(row) for row in o.matrix_basis) for o in scene.objects}
bpy.ops.wm.open_mainfile(filepath=str(saved))
assert bpy.context.scene.frame_current == 42 and bpy.context.scene.frame_subframe == .25
assert before == {o.name: tuple(tuple(row) for row in o.matrix_basis) for o in bpy.context.scene.objects}
(out / "results.json").write_text('{"passed": 9}\n')
print("PASS save capture lifecycle 9", flush=True)
