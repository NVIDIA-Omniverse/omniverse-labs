"""Blender-authored joints executed by the fixed native OVPhysX worker."""
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
from types import SimpleNamespace
import bpy
from pxr import Usd, UsdPhysics

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.scene_generation import SceneGenerationOwner, BlenderId
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter, generation_requires_physics
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from ovrtx_blender_example.physics_capabilities import validate_joint_runtime, joint_capability, binary_digest
from ovrtx_blender_example.physics_presentation_targets import presentation_targets

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, default=Path("."))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
parser.add_argument("--case", default="all")
parser.add_argument("--authoring-only", action="store_true",
                    help="Check compiled endpoints without starting native physics; writes no admission result")
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
checks = []
results = []
def check(label, value):
    assert value, label
    checks.append(label)
def body(name, position, role="DYNAMIC", enabled=True, radius=.2):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = position
    initialize_owner(obj)
    p = obj.ov.physics
    p.rigid_body.schema_opinion = "APPLY"
    p.rigid_body.role = role
    p.rigid_body.enabled = enabled
    if role != "STATIC":
        p.rigid_body.mass_kg = 1
    c = p.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion = "APPLY"
    c.shape = "ANALYTIC_SPHERE"
    c.radius_m = radius
    return obj

def device_stage(generation):
    # Device opinions are now supplied by the production controller's private
    # stage, so the fixture never edits the compiled generation for GPU tests.
    return Usd.Stage.Open(generation.usd_path)

def fixture(kind, axis, drive):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.ov.physics_scene.schema_opinion = "APPLY"
    direction = [0., 0., -1.]
    position = [0., 0., 2.]
    if kind in {"REVOLUTE", "STATIC_ANCHOR"}:
        position[1 if axis == "X" else 0] = 1.
    elif kind in {"PRISMATIC", "D6"}:
        direction = [0., 0., 0.]
        direction["XYZ".index(axis)] = -1.
    scene.ov.physics_scene.gravity_direction = direction
    scene.ov.physics_scene.gravity_magnitude = 9.81
    # The static anchor overlaps its pendulum, so only pair filtering keeps
    # the joint radius; the attachment itself compiles as a WORLD frame.
    a = body("DrivenBody", position, radius=.6 if kind == "STATIC_ANCHOR" else .2)
    contact = body("UnconnectedBody", (5, 0, 2))
    body("StaticSupport", (0, 0, -100), role="STATIC", radius=100)
    body("KinematicInput", (8, 0, 2), role="KINEMATIC")
    body("DisabledBody", (10, 0, 2), enabled=False)
    handle = bpy.data.objects.new("Joint", None)
    scene.collection.objects.link(handle)
    handle.location = (0, 0, 2)
    joint = initialize_joint(handle)
    joint.enabled = True
    joint.type, joint.axis = ("REVOLUTE" if kind == "STATIC_ANCHOR" else kind), axis
    # Drive targets describe body B relative to A. Use world A so a positive
    # target is a positive body displacement against the negative gravity load.
    joint.endpoint_a.kind = "WORLD"
    if kind == "STATIC_ANCHOR":
        joint.endpoint_a.kind = "OBJECT"
        joint.endpoint_a.object = body("StaticAnchor", (0, 0, 2), role="STATIC", radius=.6)
    joint.endpoint_b.object = a
    if kind == "D6":
        for record in joint.axes:
            record.state = "LOCKED"
    if kind not in {"FIXED", "STATIC_ANCHOR"}:
        key = ("rot" if kind == "REVOLUTE" else "trans") + axis
        record = joint.axes[key]
        record.state = "LIMITED"
        limit = math.radians(20) if kind == "REVOLUTE" else .4
        record.lower, record.upper = -limit, limit
        if drive:
            record.drive.enabled = True
            record.drive.target_position = math.radians(15) if kind == "REVOLUTE" else .3
            record.drive.stiffness = 300 if kind == "REVOLUTE" else 100
            record.drive.damping = 20
            record.drive.max_effort = 100
    bpy.context.view_layer.update()
    apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
    return a, contact, joint, position

cases = [("FIXED", "X", False), ("REVOLUTE", "X", False), ("REVOLUTE", "Y", False),
         ("PRISMATIC", "X", False), ("PRISMATIC", "Z", True),
         ("D6", "Y", False), ("D6", "Z", True), ("STATIC_ANCHOR", "X", False)]
generic = out / "generic-joint-only.usda"
generic.write_text('#usda 1.0\ndef PhysicsJoint "GenericJoint" {}\n')
check("generic PhysicsJoint schema independently activates physics",
      generation_requires_physics(SimpleNamespace(materialize_usd=lambda: str(generic))))
for kind, axis, drive in cases:
    name = f"{kind.lower()}-{axis.lower()}-{'drive' if drive else 'limit'}"
    if args.case not in {"all", name}:
        continue
    case_dir = out / name
    case_dir.mkdir(exist_ok=True)
    a, contact, joint, initial = fixture(kind, axis, drive)
    owner = SceneGenerationOwner(case_dir / "generations")
    generation = owner.replace(scene)
    check(name + " activates physics", generation_requires_physics(generation))
    stage = device_stage(generation)
    source_hash = hashlib.sha256(Path(generation.usd_path).read_bytes()).hexdigest()
    source_layers = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(generation.usd_path).parent.rglob("*") if p.suffix in {".usda", ".usdc"}}
    # Object bodies can retain their stock-export path. Resolve the generation's
    # stable Object mapping rather than searching for an ID in the USD spelling.
    paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
             and p.GetAttribute("physics:rigidBodyEnabled").Get() is not False
             and not p.GetAttribute("physics:kinematicEnabled").Get()]
    targets = [target for target in presentation_targets(generation)
               if target.object_session_uid == a.session_uid and target.kind == "OBJECT"]
    check(name + " unique compiled Object mapping", len(targets) == 1)
    target = targets[0]
    expected = target.body_path
    check(name + " compiled Object identity", target.object_identity == a.ov.physics.persistent_id)
    check(name + " compiled dynamic roster", expected in paths and len(paths) == 2
          and target.enabled and target.role == "DYNAMIC")
    compiled_joint = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
    check(name + " composed driven endpoint",
          [str(path) for path in compiled_joint.GetRelationship("physics:body1").GetTargets()] == [expected])
    if args.authoring_only:
        results.append({"case": name, "body_path": expected, "dynamic_paths": paths,
                        "joint_path": str(compiled_joint.GetPath())})
        continue
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    def controller_factory(config):
        return OvphysxStageController(replace(config,
            server=str(server), ovphysx_address=address,
            ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
            device=args.device, physics_fps=120, update_fps=120, max_steps=360,
            worker_log_path=str(case_dir / "worker.log"),
            ovphysx_native_client_path=str(runtime / "client"),
            ovphysx_native_client_module="ovphysx_bridge_client"))
    adapter = OvphysxGenerationAdapter(controller_factory)
    samples = []
    try:
        check(name + " real worker activation", adapter.activate(generation))
        check(name + " dynamic role roster", set(adapter.dynamic_body_paths) == set(paths) and len(paths) == 2)
        controller = adapter.controller
        if args.device == "gpu":
            check(name + " private GPU device stage", controller.config.input_usd_path != generation.usd_path)
            device_prim = Usd.Stage.Open(controller.config.input_usd_path)
            check(name + " actual GPU scene opinions", device_prim.GetPrimAtPath(scene.ov.usd.physics_scene_path).GetAttribute("physxScene:enableGPUDynamics").Get() is True)
        else:
            check(name + " CPU has no GPU override", controller.config.input_usd_path == generation.usd_path)
        joint_kind = "REVOLUTE" if kind == "STATIC_ANCHOR" else kind
        check(name + " admission distinguishes restart from live", joint_capability(joint_kind, args.device).restart_simulation and not joint_capability(joint_kind, args.device).live)
        samples.append(controller.physics_pose_set(0))
        for step in range(360):
            result = controller.tick()
            check_label = name + " complete finite poses"
            assert result.status == OvphysxStageStatus.OK, (result, controller.last_error)
            assert {p.prim_path for p in result.pose_set} == set(paths)
            assert all(math.isfinite(v) for p in result.pose_set for v in (*p.translate, *p.orient))
            assert all(abs(sum(v*v for v in p.orient) - 1) < 1e-4 for p in result.pose_set)
            assert result.simulation_time_ns == (step + 1) * controller.config.timestep_ns
            samples.append(result.pose_set)
        check(check_label, True)
        capabilities = sdk_fixture.current_capabilities() if args.public_sdk else importlib.import_module("ovphysx_bridge_client").capabilities()
        if args.device == "gpu":
            health = sdk_fixture.current_health() if args.public_sdk else importlib.import_module("ovphysx_bridge_client").check_health()
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
            evidence = [row for row in processes.splitlines() if row.split(",")[0].strip() == str(health["worker_pid"])]
            check(name + " GPU worker compute process", bool(evidence))
            (case_dir / "gpu-process.json").write_text(json.dumps({"worker_pid": health["worker_pid"], "rows": evidence}, indent=2))
        if kind == "FIXED":
            rejected_path = case_dir / "unsupported.usda"
            stage.Flatten().Export(str(rejected_path))
            rejected_stage = Usd.Stage.Open(str(rejected_path))
            rejected_joint = next(p for p in rejected_stage.Traverse() if p.IsA(UsdPhysics.Joint))
            rejected_joint.SetTypeName("PhysicsSphericalJoint")
            rejected_stage.GetRootLayer().Save()
            previous_controller = adapter.controller
            rejected_generation = SimpleNamespace(number=999, materialize_usd=lambda: str(rejected_path))
            check("unvalidated joint fails admission", not adapter.activate(rejected_generation))
            check("admission rejection preserves active controller", adapter.controller is previous_controller and adapter.active_generation == generation.number)
            if args.public_sdk:
                for refusal in sdk_fixture.reject_changed_identities(previous_controller.config):
                    check(refusal['label'] + ' rejected', True)
            else:
                native = importlib.import_module("ovphysx_bridge_client")
                for label, config, native_supplier in (
                    ("missing native RPC", previous_controller.config, lambda: SimpleNamespace(capabilities=lambda: {"rpcs": []})),
                    ("unknown runtime identity", replace(previous_controller.config, server=sys.executable,
                        ovphysx_worker_command=subprocess.list2cmdline([sys.executable])), lambda: native),
                ):
                    try:
                        validate_joint_runtime(generation.usd_path, config, native_supplier)
                    except ValueError:
                        check(label + " rejected", True)
                    else:
                        check(label + " rejected", False)
            for switched in ("PRISMATIC", "FIXED"):
                joint.type, joint.axis = switched, "Y"
                scene.ov.physics_scene.gravity_direction = (0, -1, 0)
                record = joint.axes["transY"]
                record.state, record.lower, record.upper = "LIMITED", -.4, .4
                candidate = owner.reconcile(scene, {BlenderId("PHYSICS", scene.session_uid)})
                switched_stage = device_stage(candidate)
                check("type switch " + switched + " actual activation", adapter.activate(candidate))
                for step in range(180):
                    update = adapter.controller.tick()
                    assert update.status == OvphysxStageStatus.OK, adapter.controller.last_error
                    assert {p.prim_path for p in update.pose_set} == set(paths)
                pose = next(p for p in update.pose_set if p.prim_path == expected)
                check("type switch " + switched + " real response",
                      abs(pose.translate[1] + .4) < .03 if switched == "PRISMATIC" else math.dist(pose.translate, initial) < .01)
                owner.accept(candidate)
    finally:
        status = adapter.deactivate()
        check(name + " worker shutdown", status in {"stopped", "not_found"})
        (case_dir / "poses.json").write_text(json.dumps([[{"path": p.prim_path, "translate": p.translate, "orient": p.orient} for p in sample] for sample in samples], indent=2))
    tracked = [next(p for p in sample if p.prim_path == expected) for sample in samples]
    displacement = [p.translate["XYZ".index(axis)] - initial["XYZ".index(axis)] for p in tracked]
    angle = [2 * math.atan2(p.orient["XYZ".index(axis)], p.orient[3]) for p in tracked]
    metric = {"case": name, "initial": tracked[0].translate, "final": tracked[-1].translate,
              "final_angle_deg": math.degrees(angle[-1]), "max_angle_deg": max(abs(math.degrees(v)) for v in angle),
              "last_second_max_angle_deg": max(abs(math.degrees(v)) for v in angle[-120:]),
              "timestep_ns": controller.config.timestep_ns, "steps": 360,
              "minimum_displacement": min(displacement), "maximum_displacement": max(displacement)}
    print(json.dumps(metric), flush=True)
    results.append(metric)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    if kind == "FIXED":
        check(name + " gravity attachment", max(math.dist(p.translate, initial) for p in tracked) < .01)
    elif kind == "STATIC_ANCHOR":
        check(name + " swings under gravity", min(p.translate[2] for p in tracked) < 1.1)
        check(name + " static anchor holds joint radius",
              all(abs(math.dist(p.translate, (0, 0, 2)) - 1) < .01 for p in tracked))
    elif kind == "REVOLUTE":
        check(name + " moves under gravity", max(abs(v) for v in angle) > math.radians(10))
        check(name + " angular limit", max(abs(v) for v in angle) < math.radians(23))
    else:
        check(name + " configured axis", max(abs(p.translate[i] - initial[i]) for p in tracked for i in range(3) if i != "XYZ".index(axis)) < .01)
        check(name + " linear limits", min(displacement) > -.43 and max(displacement) < .43)
        check(name + " loaded drive" if drive else name + " gravity translation",
              displacement[-1] > .12 if drive else min(displacement) < -.3)
    check(name + " input unchanged", hashlib.sha256(Path(generation.usd_path).read_bytes()).hexdigest() == source_hash)
    check(name + " all compiled source layers unchanged", all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in source_layers.items()))
    check(name + " no authoring pose writeback", math.dist(a.location, initial) < 1e-6)

if args.authoring_only:
    (out / "authoring-results.json").write_text(json.dumps({"checks": checks, "cases": results}, indent=2))
    print(f"JOINT_AUTHORING_ONLY_PASS {len(results)} cases {len(checks)} checks", flush=True)
    raise SystemExit(0)

result = {"status": "pass", "device": args.device, "blender": bpy.app.version_string,
          **({"public_sdk": sdk_fixture.current_provenance()} if args.public_sdk else {
              "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
              "client_sha256": binary_digest(importlib.import_module("ovphysx_bridge_client").__file__)}),
          "capabilities": capabilities, "metrics": results, "checks": checks}
(out / "results.json").write_text(json.dumps(result, indent=2))
print(f"NATIVE_JOINTS_OK {args.device} {len(checks)} checks", flush=True)
