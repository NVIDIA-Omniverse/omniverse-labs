"""Animated evaluated Object/bone targets through actual fixed-step capture."""
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
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), required=True)
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
metrics = {}
for kind in ("OBJECT", "BONE"):
    for units in (1., .01):
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
        session = BakeCaptureSession(scene, BakeCaptureSettings(1, 1, 25, 24, substeps=5), out, controller_factory=factory)
        try:
            while not session.poll():
                time.sleep(.001)
        finally:
            session.close()
        samples = list(session.artifact.samples())
        manifest = session.artifact.manifest()
        assert len(manifest["body_paths"]) == len(manifest["kinematic_input_paths"]) == 1
        assert not set(manifest["body_paths"]) & set(manifest["kinematic_input_paths"])
        path = manifest["body_paths"][0]
        x = samples[-1]["poses"][path]["position"][0]
        assert .39 < x < .42, x
        assert (tuple(channel.location), tuple(dyn.location), scene.frame_current, obj.animation_data.action) == before
        assert len(samples) == 25 and samples[-1]["step"] == 120
        metrics[name] = x
        print(name, x, flush=True)
        (a.output / "results.json").write_text(json.dumps(metrics, indent=2))
assert max(metrics.values())-min(metrics.values()) < 1e-4
print("KINEMATIC_CAPTURE_PASS", a.device, len(metrics))
