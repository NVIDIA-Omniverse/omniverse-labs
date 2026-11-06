"""Real Blender flat bone compilation and native ordinary-joint mechanisms."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
import bpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint, bind_bone_endpoint
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.physics_frames import PhysicsUnits, RigidFrame, evaluated_owner_frame
from ovrtx_blender_example.physics_presentation_targets import presentation_targets
from ovrtx_blender_example.scene_generation import SceneGenerationOwner, BlenderId
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, default=Path("."))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
parser.add_argument("--aligned", action="store_true", help="Exercise the GPU-admitted identity local joint rotations")
parser.add_argument("--world-only", action="store_true", help="Native GPU-admitted isolated WORLD fixed bone")
parser.add_argument("--authoring-only", action="store_true",
                    help="Validate composed bodies/endpoints without native physics; writes no admission result")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_sdk_physics_fixture as sdk_fixture
sdk_fixture.add_arguments(parser)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
sdk_fixture.configure(args)
out, runtime = args.output.resolve(), args.runtime.resolve()
out.mkdir(parents=True, exist_ok=True)
server = Path(__file__).resolve() if args.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
props.register()
scene = bpy.context.scene
checks, metrics, equivalent = [], [], []
def check(label, condition):
    assert condition, label
    checks.append(label)

for unit_scale in (1., .01):
    label = f"{args.device}-{unit_scale}"
    case = out / label
    case.mkdir(exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Remove orphan IDs so independently authored unit cases keep identical names.
    for datablock in tuple(bpy.data.objects):
        if datablock.users == 0:
            bpy.data.objects.remove(datablock)
    scene.unit_settings.scale_length = unit_scale
    scene.ov.physics_scene.schema_opinion = "APPLY"
    scene.ov.physics_scene.gravity_direction = (0, 0, -1)
    # Existing radius_m and gravity_magnitude authoring are explicitly SI;
    # frame offsets/rest geometry are Blender units and scale below.
    scene.ov.physics_scene.gravity_magnitude = 9.81
    def configure(obj, bone=None, mass=1):
        target = initialize_owner(obj, bone)
        p = target.ov.physics
        p.rigid_body.schema_opinion = "APPLY"
        p.rigid_body.mass_kg = mass
        p.rigid_body.offset.translation = (0, .1 / unit_scale, .05 / unit_scale)
        p.rigid_body.offset.rotation = (1, 0, 0, 0) if args.aligned else (.98078528, 0, 0, .19509032)
        c = p.colliders.add()
        c.name = c.persistent_id = uuid.uuid4().hex
        c.schema_opinion = "APPLY"
        c.shape = "ANALYTIC_SPHERE"
        c.radius_m = .08
        return target
    def rig(name, position, rotation, scale=1, shared=None):
        obj = bpy.data.objects.new(name, shared or bpy.data.armatures.new(name + "Data"))
        scene.collection.objects.link(obj)
        obj.location = tuple(v / unit_scale for v in position)
        obj.rotation_euler = (0, 0, 0) if args.aligned else rotation
        obj.scale = (scale,) * 3
        if shared is None:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode="EDIT")
            root = obj.data.edit_bones.new("Root")
            root.head, root.tail = (0, 0, 0), (0, 1 / unit_scale, 0)
            child = obj.data.edit_bones.new("Child")
            child.head, child.tail = root.tail, (0, 2 / unit_scale, 0)
            child.parent = root
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.update()
        for bone in obj.data.bones:
            configure(obj, bone, mass=2 if shared else 1)
        return obj
    r = rig("RigA", (0, 0, 4), (.1, .2, .3), 2)
    t = rig("RigB", (3, 0, 5), (-.2, .1, -.3), 1.5)
    shared = rig("SharedInstance", (10, 0, 7), (.2, -.1, .4), 1, r.data)
    obj = bpy.data.objects.new("MixedObject", None)
    scene.collection.objects.link(obj)
    obj.location = (3 / unit_scale, 1 / unit_scale, 5 / unit_scale)
    configure(obj)
    bpy.context.view_layer.update()
    units = PhysicsUnits.from_scene(scene)
    def endpoint(field, owner, name=None):
        if owner is None:
            field.kind = "WORLD"
        elif name:
            bind_bone_endpoint(field, owner, owner.data.bones[name])
        else:
            field.kind, field.object = "OBJECT", owner
    def joint(name, a, aname, b, bname, kind="FIXED"):
        handle = bpy.data.objects.new(name, None)
        scene.collection.objects.link(handle)
        handle.location = r.matrix_world.translation
        j = initialize_joint(handle)
        j.enabled, j.type, j.axis = True, kind, "X"
        endpoint(j.endpoint_a, a, aname)
        endpoint(j.endpoint_b, b, bname)
        if kind == "REVOLUTE":
            j.axes["rotX"].state = "LIMITED"
            j.axes["rotX"].lower, j.axes["rotX"].upper = -.6, .6
        bpy.context.view_layer.update()
        apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
        return handle
    handles = [joint("WorldBone", None, None, r, "Root"),
               joint("SameArmature", r, "Root", r, "Child", "REVOLUTE"),
               joint("CrossArmature", r, "Child", t, "Root"),
               joint("MixedObjectBone", t, "Root", obj, None)]
    if args.world_only:
        for handle in handles[1:]:
            handle.ov.physics.joint.enabled = False
    owner = SceneGenerationOwner(case / "generations")
    generation = owner.replace(scene)
    targets = presentation_targets(generation)
    check(label + " complete immutable presentation mapping", len(targets) == 7 and all(isinstance(x.body_offset, RigidFrame) for x in targets))
    check(label + " shared definition scoped by object", r.data is shared.data and len({x.body_identity for x in targets}) == 7)
    stage = Usd.Stage.Open(generation.usd_path)
    cache = UsdGeom.XformCache()
    initial = {}
    for target in targets:
        prim = stage.GetPrimAtPath(target.body_path)
        check(label + " compiled rigid body " + target.body_identity, prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if target.bone_identity:
            check(label + " flat bone body " + target.body_identity,
                  prim.GetParent().GetPath() == Sdf.Path(scene.ov.physics_namespace + "/Bodies"))
        else:
            mapping = generation.blender_prim_paths[BlenderId("OBJECT", target.object_session_uid)]
            check(label + " stock Object body mapping", target.body_path == mapping.object_path)
        # The runtime composes parents; body frames are parent-relative, never reset.
        check(label + " parent-relative body frame", not UsdGeom.Xformable(prim).GetResetXformStack())
        initial[target.body_path] = tuple(cache.GetLocalToWorldTransform(prim).ExtractTranslation())
        check(label + " copied initial meter frame", math.dist(initial[target.body_path], target.initial_body_frame.position) < 2e-5)
        source_obj = next(o for o in scene.objects if o.session_uid == target.object_session_uid)
        raw = evaluated_owner_frame(source_obj, bpy.context.evaluated_depsgraph_get(), bone_name=target.bone_name or None, units=units)
        reconstructed = raw.compose(target.body_offset)
        check(label + " body offset includes scale exactly once", math.dist(reconstructed.position, target.initial_body_frame.position) < 2e-5)
        check(label + " no RNA in target", not hasattr(target.initial_body_frame, "id_data"))
        check(label + " per-instance mass opinion", UsdPhysics.MassAPI(prim).GetMassAttr().Get() == (2 if source_obj is shared else 1))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    check(label + " ordinary joint records", len(joints) == (1 if args.world_only else 4))
    check(label + " all relationships resolve actual flat bodies", all(str(path) in initial for p in joints for key in ("physics:body0", "physics:body1") for path in p.GetRelationship(key).GetTargets()))
    # Preserve stock-backed Object definitions and ancestor transforms while
    # proving ordinary physics bodies do not depend on exported USD skeletons.
    # Overlay-only composition loses stock ancestor transforms.
    physics_stage = Usd.Stage.Open(stage.Flatten())
    skeleton_paths = [p.GetPath() for p in physics_stage.Traverse() if "Skel" in p.GetTypeName()]
    check(label + " physics bodies independent of skeleton subtrees",
          all(not Sdf.Path(path).HasPrefix(skeleton) for path in initial for skeleton in skeleton_paths))
    for path in sorted(skeleton_paths, key=lambda path: len(str(path)), reverse=True):
        physics_stage.RemovePrim(path)
    physics_path = case / "physics-without-skeleton.usda"
    physics_stage.Flatten().Export(str(physics_path))
    check(label + " no skeleton or articulation dependency", not any("Skel" in p.GetTypeName() or "PhysicsArticulationRootAPI" in p.GetAppliedSchemas() for p in physics_stage.Traverse()))
    check(label + " isolated complete composed body roster", all(
        physics_stage.GetPrimAtPath(path).IsDefined() and
        physics_stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI) for path in initial))
    isolated_cache = UsdGeom.XformCache()
    check(label + " isolated composed world frames", all(math.dist(
        tuple(isolated_cache.GetLocalToWorldTransform(physics_stage.GetPrimAtPath(path)).ExtractTranslation()),
        position) < 2e-5 for path, position in initial.items()))
    if args.authoring_only:
        equivalent.append(sorted((x.bone_name, next(o.name for o in scene.objects
            if o.session_uid == x.object_session_uid), x.initial_body_frame.position) for x in targets))
        continue
    physics_generation = generation
    source_hash = hashlib.sha256(physics_path.read_bytes()).hexdigest()
    source_layers = {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                     for layer in stage.GetUsedLayers() if layer.realPath}
    before = {o.name: tuple(tuple(row) for row in o.matrix_world) for o in (r, t, shared, obj)}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    def factory(config):
        return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
            ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
            device=args.device, physics_fps=240, update_fps=240, max_steps=240,
            worker_log_path=str(case / "worker.log"), ovphysx_native_client_path=str(runtime / "client"),
            ovphysx_native_client_module="ovphysx_bridge_client"))
    adapter, samples = OvphysxGenerationAdapter(factory), []
    if args.device == "gpu" and (not args.world_only or not args.aligned):
        baseline_layer = Sdf.Layer.CreateNew(str(case / "accepted-free-bodies.usda"))
        baseline_layer.subLayerPaths = [generation.usd_path]
        baseline_stage = Usd.Stage.Open(baseline_layer)
        for prim in joints:
            baseline_stage.OverridePrim(prim.GetPath()).CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        baseline_layer.Save()
        try:
            check(label + " real GPU predecessor activation", adapter.activate(replace(generation, usd_path=baseline_layer.realPath)))
            predecessor = adapter.controller
            check(label + " GPU mechanism fails closed", not adapter.activate(generation))
            check(label + " CPU alternative diagnostic", "select CPU explicitly" in adapter.last_error)
            check(label + " rejected GPU keeps accepted controller", adapter.controller is predecessor)
            check(label + " accepted predecessor remains usable", predecessor.tick().status == OvphysxStageStatus.OK)
        finally:
            check(label + " predecessor shutdown", adapter.deactivate() in {"stopped", "not_found"})
        continue
    try:
        check(label + " real full-generation native activation", adapter.activate(physics_generation))
        for step in range(240):
            result = adapter.controller.tick()
            assert result.status == OvphysxStageStatus.OK, adapter.controller.last_error
            assert {p.prim_path for p in result.pose_set} == set(initial)
            assert all(math.isfinite(v) for p in result.pose_set for v in (*p.translate, *p.orient))
            samples.append({p.prim_path: p.translate for p in result.pose_set})
            if step == 0:
                check(label + " native initial meter poses", all(math.dist(samples[-1][path], position) < .003 for path, position in initial.items()))
    finally:
        check(label + " shutdown", adapter.deactivate() in {"stopped", "not_found"})
        (case / "poses.json").write_text(json.dumps(samples, indent=2))
    root_target = next(x for x in targets if x.object_session_uid == r.session_uid and x.bone_name == "Root")
    moving = [x for x in targets if (x.object_session_uid == r.session_uid and x.bone_name == "Child") or x.object_session_uid == obj.session_uid]
    root_error = max(math.dist(sample[root_target.body_path], initial[root_target.body_path]) for sample in samples)
    motion = [math.dist(samples[-1][x.body_path], initial[x.body_path]) for x in moving]
    print(label, "root_error", root_error, "motion", motion, flush=True)
    check(label + " fixed transformed root under gravity", root_error < .01)
    check(label + " same and mixed mechanism real motion", min(motion) > .02)
    shared_target = next(x for x in targets if x.object_session_uid == shared.session_uid and x.bone_name == "Root")
    check(label + " independent shared-rig instance falls", initial[shared_target.body_path][2] - samples[-1][shared_target.body_path][2] > 3)
    check(label + " worker does not mutate authoring RNA", before == {o.name: tuple(tuple(row) for row in o.matrix_world) for o in (r, t, shared, obj)})
    check(label + " source USD untouched", hashlib.sha256(physics_path.read_bytes()).hexdigest() == source_hash)
    check(label + " full generation layers untouched", all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in source_layers.items()))
    isolated_samples = []
    try:
        check(label + " native activation without UsdSkel", adapter.activate(replace(generation, usd_path=str(physics_path))))
        for step in range(24):
            result = adapter.controller.tick()
            assert result.status == OvphysxStageStatus.OK, adapter.controller.last_error
            isolated_samples.append({p.prim_path: p.translate for p in result.pose_set})
        check(label + " isolated and full physics trajectories agree", all(
            math.dist(isolated_samples[i][path], samples[i][path]) < .001
            for i in range(24) for path in initial))
    finally:
        check(label + " isolated shutdown", adapter.deactivate() in {"stopped", "not_found"})
    equivalent.append(sorted((x.bone_name, bpy.data.objects.get(next(o.name for o in scene.objects if o.session_uid == x.object_session_uid)).name, x.initial_body_frame.position) for x in targets))
    metrics.append({"case": label, "root_error_m": root_error, "moving_displacement_m": motion})
    # Full compiler must reject unsupported transforms without replacing accepted.
    saved_scale = r.scale.copy()
    for value in ((1, 2, 1), (-1, 1, 1), (0, 1, 1)):
        r.scale = value
        bpy.context.view_layer.update()
        try:
            owner.replace(scene)
        except ValueError:
            check(label + " rejects scale " + str(value), owner.current_generation is generation)
        else:
            check(label + " rejects scale " + str(value), False)
    r.scale = saved_scale
    r.data.bones["Child"].inherit_scale = "AVERAGE"
    bpy.context.view_layer.update()
    try:
        owner.replace(scene)
    except ValueError:
        check(label + " rejects unvalidated inherit scale", owner.current_generation is generation)
    else:
        check(label + " rejects unvalidated inherit scale", False)
    r.data.bones["Child"].inherit_scale = "FULL"
    bpy.context.view_layer.update()
    child_id = r.data.bones["Child"].ov.persistent_id
    r.data.bones["Child"].ov.persistent_id = r.data.bones["Root"].ov.persistent_id
    try:
        owner.replace(scene)
    except ValueError:
        check(label + " duplicate bone definitions reject without replacement", owner.current_generation is generation)
    else:
        check(label + " duplicate bone definitions reject without replacement", False)
    finally:
        r.data.bones["Child"].ov.persistent_id = child_id
    # Names are hints only; descriptors carry immutable definition identities.
    old_bone_id = r.data.bones["Root"].ov.persistent_id
    r.data.bones["Root"].name = "RenamedRoot"
    check(label + " generation targets retain stable bone identity", root_target.bone_identity == old_bone_id)
    renamed = owner.replace(scene)
    renamed_targets = presentation_targets(renamed)
    check(label + " renamed bone preserves compiled paths", {x.body_path for x in targets} == {x.body_path for x in renamed_targets})
    check(label + " renamed label resolves same stable identity", any(x.bone_name == "RenamedRoot" and x.bone_identity == old_bone_id for x in renamed_targets))
    bpy.ops.wm.save_as_mainfile(filepath=str(case / "mechanism.blend"))
(out / "equivalent.json").write_text(json.dumps(equivalent, indent=2))
if equivalent:
    check("meter-centimeter equivalent compiled initial frames", all(a[:2] == b[:2] and math.dist(a[2], b[2]) < 2e-5 for a, b in zip(*equivalent)))
(out / ("authoring-results.json" if args.authoring_only else "results.json")).write_text(json.dumps({"device": args.device, "blender": bpy.app.version_string, "metrics": metrics, "checks": checks}, indent=2))
print(f"{'BONE_AUTHORING_ONLY_PASS' if args.authoring_only else 'BONE_MECHANISMS_OK'} {args.device} {len(checks)} checks", flush=True)
