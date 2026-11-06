# SPDX-License-Identifier: Apache-2.0
"""Real worker: pose writes under parents are expressed in the parent's space.

The worker composes written body poses with the (static) USD parent while
readback is world-space. Initial conditions (at start, reset and mid-run) and
kinematic inputs under a translated, rotated parent must land at the requested
world pose.
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_kinematic_inputs as kinematic
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.scene_generation import SceneGenerationOwner, blender_id
from ovrtx_blender_example.shared_stage_composition import BodyPose

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, default=Path("."))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
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
checks, metrics = [], {}


def check(label, value):
    assert value, label
    checks.append(label)


def fresh_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    props.register()
    scene = bpy.context.scene
    scene.ov.physics_scene.schema_opinion = "APPLY"
    return scene


def cube(scene, name, location, role="DYNAMIC"):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    initialize_owner(obj)
    physics = obj.ov.physics
    physics.rigid_body.schema_opinion = "APPLY"
    physics.rigid_body.role = role
    physics.rigid_body.mass_kg = 1
    collider = physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "CONVEX_HULL"
    return obj


def floor(scene):
    obj = cube(scene, "Floor", (0, 0, -.5))
    obj.scale = (20, 20, .5)
    obj.ov.physics.rigid_body.schema_opinion = "INHERIT"
    return obj


def rotated_parent(scene, location):
    holder = bpy.data.objects.new("Holder", None)
    scene.collection.objects.link(holder)
    holder.location, holder.rotation_euler = location, (0, 0, math.pi / 2)
    return holder


def adapter(label):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"

    def factory(config):
        return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
            ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
            device=args.device, physics_fps=120, update_fps=120, max_steps=240,
            worker_log_path=str(out / f"{label}-worker.log"),
            ovphysx_native_client_path=str(runtime / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))
    return OvphysxGenerationAdapter(factory)


def pose_of(update, path):
    return next(p for p in update.pose_set if p.prim_path == path)


# Initial conditions for a child of a translated, rotated parent.
for mode in ("start", "reset", "mid"):
    scene = fresh_scene()
    floor(scene)
    holder = rotated_parent(scene, (5, 1, .5))
    child = cube(scene, "Child", (6, 1, 3))
    child.parent = holder
    child.matrix_parent_inverse = holder.matrix_world.inverted()
    bpy.context.view_layer.update()
    generation = SceneGenerationOwner(out / f"ic-{mode}").replace(scene)
    path = generation.blender_prim_paths[blender_id(child, "OBJECT")].object_path
    requested = BodyPose(path, (8.0, 0.0, 6.0), (0.0, 0.0, 0.0, 1.0))
    runner = adapter(f"ic-{mode}")
    try:
        if mode == "start":
            check("start activation", runner.activate(generation, initial_conditions=(requested,)))
        elif mode == "reset":
            check("reset activation", runner.activate(generation, initial_conditions=(requested,)))
            check("reset", runner.reset())
        else:
            check("mid activation", runner.activate(generation))
            update = runner.controller.apply_initial_condition_values((requested,))
            check("mid write applied", update.status == OvphysxStageStatus.OK)
        first = pose_of(runner.controller.tick(), path)
        metrics[f"ic_{mode}_first"] = first.translate
        check(f"initial condition at {mode} lands at the requested world pose",
              math.dist(first.translate, requested.translate) < .01)
    finally:
        check(f"ic {mode} shutdown", runner.deactivate() in {"stopped", "not_found"})

# Kinematic input under a translated, rotated parent carries a resting body.
scene = fresh_scene()
floor(scene)
holder = rotated_parent(scene, (3, 2, .5))
driven = cube(scene, "Kinematic", (3, 2, 1), role="KINEMATIC")
driven.parent = holder
driven.matrix_parent_inverse = holder.matrix_world.inverted()
rider = cube(scene, "Rider", (3, 2, 3.1))
holder.keyframe_insert("location", index=0, frame=1)
holder.location.x = 3.5
holder.keyframe_insert("location", index=0, frame=121)
scene.frame_set(1)
generation = SceneGenerationOwner(out / "kinematic").replace(scene)
rider_path = generation.blender_prim_paths[blender_id(rider, "OBJECT")].object_path
runner = adapter("kinematic")
try:
    check("kinematic activation", runner.activate(generation))
    last = None
    for step in range(120):
        scene.frame_set(min(step + 2, 121))
        kinematic.apply_kinematic_inputs(runner.controller, kinematic.capture_kinematic_inputs(scene, generation), generation)
        update = runner.controller.tick()
        check_ok = update.status == OvphysxStageStatus.OK
        assert check_ok, runner.controller.last_error
        last = pose_of(update, rider_path)
    metrics["kinematic_rider_last"] = last.translate
    check("rider stays on the parented kinematic body", abs(last.translate[2] - 3.0) < .05)
    check("rider is carried with the parent", last.translate[0] > 3.3)
finally:
    check("kinematic shutdown", runner.deactivate() in {"stopped", "not_found"})

(out / "results.json").write_text(json.dumps({"device": args.device, "checks": checks, "metrics": metrics}, indent=2))
print("PARENTED_POSES_PASS", args.device, len(checks))
