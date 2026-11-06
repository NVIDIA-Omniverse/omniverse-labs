"""Real fixed-worker behavior for bounded body, collider, material and scene fields."""
import argparse
from dataclasses import replace
import importlib
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

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from ovrtx_blender_example.physics_capabilities import binary_digest

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, default=Path("."))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
parser.add_argument("--case", default="all")
parser.add_argument("--cases", help="Comma-separated exact case shard; complete numeric assertions run after aggregation")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_sdk_physics_fixture as sdk_fixture
sdk_fixture.add_arguments(parser)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
sdk_fixture.configure(args)
runtime = args.runtime.resolve()
server = Path(__file__).resolve() if args.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
props.register()
scene = bpy.context.scene
checks, metrics = [], {}

def check(label, value):
    assert value, label
    checks.append(label)

def body(name, position, shape="ANALYTIC_SPHERE", role="DYNAMIC"):
    if shape == "CONVEX_HULL":
        bpy.ops.mesh.primitive_cube_add(size=.4/scene.unit_settings.scale_length)
        obj = bpy.context.object
        obj.name = name
    else:
        obj = bpy.data.objects.new(name, None)
        scene.collection.objects.link(obj)
    obj.location = position
    initialize_owner(obj)
    p = obj.ov.physics
    p.rigid_body.schema_opinion, p.rigid_body.role = "APPLY", role
    if role != "STATIC":
        p.rigid_body.mass_kg = 1
    c = p.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion, c.shape, c.radius_m = "APPLY", shape, .2
    c.dimensions = (1, 1, 1)
    c.height_m = .6
    return obj, c

def material(collider, *, friction=0., restitution=0., density=None):
    m = bpy.data.materials.new("Physics")
    m.ov.physics.schema_opinion = "APPLY"
    m.ov.physics.static_friction = m.ov.physics.dynamic_friction = friction
    m.ov.physics.restitution = restitution
    if density is not None:
        m.ov.physics.density = density
    collider.physics_material = m
    return m

def fixture(name):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.unit_settings.scale_length = .01 if name.endswith("_cm") else 1.
    u = scene.unit_settings.scale_length
    scene.ov.physics_scene.schema_opinion = "APPLY"
    scene.ov.physics_scene.gravity_direction = (0, 0, -1)
    scene.ov.physics_scene.gravity_magnitude = 0 if name.startswith(("free", "damped")) else 9.81
    a, collider = body("Body", (0, 0, 2/u), "CONVEX_HULL" if "hull" in name else "BOX" if name.startswith(("box", "friction", "static")) else "CAPSULE" if name.startswith("capsule") else "ANALYTIC_SPHERE")
    p = a.ov.physics.rigid_body
    if name.startswith(("free", "damped")):
        p.initial_linear_velocity = (2/u, 0, 0)
        p.initial_angular_velocity = (0, 0, 1)
        p.linear_damping = 2 if name.startswith("damped") else 0
        p.angular_damping = 2 if name.startswith("damped") else 0
    elif name.startswith("hinge"):
        a.location = (0, 1/u, 3/u)
        p.angular_damping = 0
        p.diagonal_inertia = (1/u**2, 2/u**2, 4/u**2)
        if "mass2" in name:
            p.mass_kg = 2
        if "inertia2" in name:
            p.diagonal_inertia = (2/u**2, 2/u**2, 4/u**2)
        if "principal" in name:
            p.principal_axes = (math.sqrt(.5), 0, 0, math.sqrt(.5))
        if "com" in name:
            p.center_of_mass = (0, .5/u, 0)
        if "density" in name:
            density = 2 / (4/3 * math.pi * .2**3)
            if "material" in name:
                p.mass_mode = "INHERIT"
                p.property_unset("mass_kg")
                material(collider, density=density*u**3)
            else:
                p.mass_mode, p.density = "DENSITY", density*u**3
        handle = bpy.data.objects.new("Hinge", None)
        scene.collection.objects.link(handle)
        handle.location = (0, 0, 3/u)
        j = initialize_joint(handle)
        j.type, j.axis, j.enabled = "REVOLUTE", "X", True
        j.endpoint_a.kind, j.endpoint_b.object = "WORLD", a
        bpy.context.view_layer.update()
        apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
    else:
        floor, fc = body("Ground", (0, 0, -.5/u), "BOX", "STATIC")
        fc.dimensions = (20/u, 20/u, 1/u)
        collider.dimensions = (1/u, 1/u, 1/u)
        friction = .8 if name == "friction_high" else 0
        restitution = .9 if name == "restitution_high" else 0
        moving_material = material(collider, friction=friction, restitution=restitution)
        floor_material = material(fc, friction=friction, restitution=restitution)
        if name.startswith("friction"):
            moving_material.ov.physics.static_friction = floor_material.ov.physics.static_friction = .4
            a.location = (0, 0, .51/u)
            p.initial_linear_velocity = (2/u, 0, 0)
        if name.startswith("static"):
            angle = math.radians(15)
            floor.rotation_euler[1] = a.rotation_euler[1] = angle
            floor.location = (-.5*math.sin(angle), 0, -.5*math.cos(angle))
            a.location = (.5*math.sin(angle), 0, .5*math.cos(angle))
            moving_material.ov.physics.static_friction = floor_material.ov.physics.static_friction = .8 if name == "static_high" else .1
            moving_material.ov.physics.dynamic_friction = floor_material.ov.physics.dynamic_friction = .1
        if name == "gravity_half":
            scene.ov.physics_scene.gravity_magnitude = 4.905
        if name == "gravity_x":
            scene.ov.physics_scene.gravity_direction = (1, 0, 0)
    bpy.context.view_layer.update()
    return a

from body_controls_assertions import CASE_NAMES, assert_metrics, assert_partial_metrics
names = CASE_NAMES
selected = set(args.cases.split(",")) if args.cases else None
if selected is not None and (not selected or not selected <= set(names)):
    raise ValueError("Unknown body-controls case selection")
for name in names:
    if selected is not None and name not in selected:
        continue
    if args.case not in {"all", name} and not (args.case == "static" and name.startswith("static")) and not (args.case == "hull" and "hull" in name):
        continue
    a = fixture(name)
    directory = out / name
    directory.mkdir(exist_ok=True)
    generation = SceneGenerationOwner(directory / "generations").replace(scene)
    stage = Usd.Stage.Open(generation.usd_path)
    path = next(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    count = 24 if name.startswith("hinge") else 240
    def factory(config):
        return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
            ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
            device=args.device, physics_fps=120, update_fps=120, max_steps=count,
            worker_log_path=str(directory / "worker.log"), ovphysx_native_client_path=str(runtime / "client"),
            ovphysx_native_client_module="ovphysx_bridge_client"))
    adapter = OvphysxGenerationAdapter(factory)
    samples = []
    try:
        check(name + " worker activation", adapter.activate(generation))
        if args.device == "gpu" and not metrics:
            device_stage = Usd.Stage.Open(adapter.controller.config.input_usd_path)
            check("actual GPU scene enabled", device_stage.GetPrimAtPath(scene.ov.usd.physics_scene_path).GetAttribute("physxScene:enableGPUDynamics").Get() is True)
            health = sdk_fixture.current_health() if args.public_sdk else importlib.import_module("ovphysx_bridge_client").check_health()
            rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
            check("worker GPU compute process", any(row.split(",")[0].strip() == str(health["worker_pid"]) for row in rows.splitlines()))
            (out / "gpu-process.json").write_text(json.dumps({"worker_pid": health["worker_pid"], "compute_processes": rows}))
        for i in range(count):
            update = adapter.controller.tick()
            assert update.status == OvphysxStageStatus.OK, adapter.controller.last_error
            pose = next(p for p in update.pose_set if p.prim_path == path)
            samples.append({"position": tuple(pose.translate), "rotation": tuple(pose.orient)})
        metric = {"position": samples[-1]["position"], "rotation": samples[-1]["rotation"],
            "angle_x": 2*math.atan2(samples[-1]["rotation"][0], samples[-1]["rotation"][3]),
            "angle_z": 2*math.atan2(samples[-1]["rotation"][2], samples[-1]["rotation"][3]),
            "rebound_height": max(s["position"][2] for s in samples[100:]) if len(samples)>100 else None,
            "at_24": samples[23]["position"]}
        metrics[name] = metric
        (directory / "samples.json").write_text(json.dumps(samples))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(name, json.dumps(metric), flush=True)
    finally:
        adapter.deactivate()

if set(metrics) == set(CASE_NAMES):
    checks.extend(assert_metrics(metrics))
else:
    checks.extend(assert_partial_metrics(metrics))

(out / "results.json").write_text(json.dumps({"device": args.device, "blender": bpy.app.version_string, **({"public_sdk": sdk_fixture.current_provenance()} if args.public_sdk else {"server_sha256": binary_digest(server), "client_sha256": binary_digest(importlib.import_module("ovphysx_bridge_client").__file__)}), "checks": checks, "metrics": metrics}, indent=2))
print("BODY_CONTROLS_PASS", args.device, len(checks), flush=True)
