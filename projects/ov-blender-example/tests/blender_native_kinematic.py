"""Actual existing pose-write transport: kinematic targets and dynamic contacts."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
import bpy
from pxr import Usd, UsdPhysics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from ovrtx_blender_example.shared_stage_composition import BodyPose
from ovrtx_blender_example.physics_presentation_targets import presentation_targets
from ovrtx_blender_example.physics_capabilities import binary_digest

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", choices=("cpu", "gpu"), required=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_sdk_physics_fixture as sdk_fixture
sdk_fixture.add_arguments(p)
args = p.parse_args(sys.argv[sys.argv.index("--")+1:])
args.output.mkdir(parents=True, exist_ok=True)
sdk_fixture.configure(args)
runtime = args.runtime.resolve()
server = Path(__file__).resolve() if args.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
props.register()
scene = bpy.context.scene
metrics = {}

for kind in ("OBJECT", "BONE"):
    for moving in (False, True):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        scene.ov.physics_scene.schema_opinion = "APPLY"
        scene.ov.physics_scene.gravity_magnitude = 0
        if kind == "BONE":
            bpy.ops.object.armature_add()
            obj = bpy.context.object
            target = initialize_owner(obj, obj.data.bones[0])
        else:
            obj = bpy.data.objects.new("Kinematic", None)
            scene.collection.objects.link(obj)
            target = initialize_owner(obj)
        dyn = bpy.data.objects.new("Dynamic", None)
        scene.collection.objects.link(dyn)
        dyn.location = (0, 0, 1)
        obj.location = (-1, 0, 1)
        initialize_owner(dyn)
        for body, role in ((target, "KINEMATIC"), (dyn, "DYNAMIC")):
            physics = body.ov.physics
            physics.rigid_body.schema_opinion, physics.rigid_body.role = "APPLY", role
            physics.rigid_body.mass_kg = 1
            collider = physics.colliders.add()
            collider.name = collider.persistent_id = uuid.uuid4().hex
            collider.schema_opinion, collider.shape, collider.radius_m = "APPLY", "ANALYTIC_SPHERE", .2
        bpy.context.view_layer.update()
        name = kind.lower() + ("_moving" if moving else "_held")
        out = args.output / name
        out.mkdir(exist_ok=True)
        generation = SceneGenerationOwner(out / "generations").replace(scene)
        descriptors = presentation_targets(generation)
        kin_path = next(t.body_path for t in descriptors if t.role == "KINEMATIC")
        dyn_path = next(t.body_path for t in descriptors if t.role == "DYNAMIC")
        stage = Usd.Stage.Open(generation.usd_path)
        assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(kin_path)).GetKinematicEnabledAttr().Get()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{sock.getsockname()[1]}"
        def factory(config):
            return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
                ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
                device=args.device, physics_fps=120, update_fps=120, max_steps=122,
                worker_log_path=str(out / "worker.log"), ovphysx_native_client_path=str(runtime / "client"),
                ovphysx_native_client_module="ovphysx_bridge_client"))
        adapter = OvphysxGenerationAdapter(factory)
        samples, writes = [], []
        try:
            assert adapter.activate(generation), adapter.last_error
            controller = adapter.controller
            assert controller.config.body_prims == (dyn_path,)
            initial = controller.tick()
            assert initial.status == OvphysxStageStatus.OK
            for step in range(1, 121):
                x = -1 + (step/120 if moving else 0)
                diagnostic = controller._physics_client.write_body_poses(
                    (BodyPose(kin_path, (x, 0., 1.), (0., 0., math.sqrt(.5), math.sqrt(.5))),),
                    simulation_time_ns=max(2, controller.current_simulation_time_ns+1))
                writes.append(dict(diagnostic))
                result = controller.tick()
                assert result.status == OvphysxStageStatus.OK, (step, result.status, controller.last_error)
                assert len(result.pose_set) == 1 and result.pose_set[0].prim_path == dyn_path
                samples.append(tuple(result.pose_set[0].translate))
            client = controller._physics_client
            original = client.config
            try:
                client.config = replace(original, body_prims=(kin_path,))
                states, _ = client.read_body_states(controller.current_simulation_time_ns)
            finally:
                client.config = original
            assert len(states) == 1 and states[0]["prim_path"] == kin_path
            assert states[0]["translate"]["found"] and states[0]["orient"]["found"]
            position = tuple(states[0]["translate"][key] for key in ("x", "y", "z"))
            rotation = tuple(states[0]["orient"][key] for key in ("i", "j", "k", "r"))
            assert math.dist(position, (x, 0., 1.)) < 1e-5
            assert abs(abs(sum(a*b for a,b in zip(rotation, (0., 0., math.sqrt(.5), math.sqrt(.5)))))-1) < 1e-5
            assert controller.config.body_prims == (dyn_path,)
            metrics[name] = {"final_dynamic": samples[-1], "target_x": x, "dynamic_output": dyn_path, "kinematic_input": kin_path, "input_readback": states[0]}
            print(name, metrics[name], flush=True)
            (out / "samples.json").write_text(json.dumps(samples))
            (out / "writes.json").write_text(json.dumps(writes, default=str))
        finally:
            adapter.deactivate()
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))

for kind in ("object", "bone"):
    assert abs(metrics[kind+"_held"]["final_dynamic"][0]) < .001
    assert metrics[kind+"_moving"]["final_dynamic"][0] > .3
(args.output / "results.json").write_text(json.dumps({"device": args.device, "server_sha256": binary_digest(server), "metrics": metrics}, indent=2))
print("KINEMATIC_NATIVE_PASS", args.device)
