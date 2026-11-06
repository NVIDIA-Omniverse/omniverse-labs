"""Native meter/cm regression for stock and generated owners beneath /World."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid
import bpy
from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController, OvphysxStageStatus

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
out, runtime = args.output.resolve(), args.runtime.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
authoring_properties.register()
scene = bpy.context.scene
results = {}
for scale in (1., .01):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.unit_settings.scale_length = scale
    scene.ov.physics_scene.schema_opinion = "APPLY"
    scene.ov.physics_scene.gravity_direction = (0, 0, -1)
    scene.ov.physics_scene.gravity_magnitude = 9.81
    bodies = []
    for i, kind in enumerate(("hull", "mapped", "generated", "fixed")):
        if kind in {"hull", "mapped"}:
            bpy.ops.mesh.primitive_cube_add(size=.4/scale)
            body = bpy.context.object
        else:
            body = bpy.data.objects.new(kind, None)
            scene.collection.objects.link(body)
        body.name = kind
        body.location = (i*2/scale, 0, 2/scale)
        initialize_owner(body)
        p = body.ov.physics
        p.rigid_body.schema_opinion, p.rigid_body.mass_kg = "APPLY", 1
        c = p.colliders.add()
        c.name = c.persistent_id = uuid.uuid4().hex
        c.schema_opinion, c.shape, c.radius_m = "APPLY", "CONVEX_HULL" if kind == "hull" else "ANALYTIC_SPHERE", .2
        bodies.append(body)
        if kind == "fixed":
            handle = bpy.data.objects.new("Joint", None)
            scene.collection.objects.link(handle)
            handle.location = body.location
            joint = initialize_joint(handle)
            joint.enabled, joint.type = True, "FIXED"
            joint.endpoint_a.kind, joint.endpoint_b.object = "WORLD", body
            bpy.context.view_layer.update()
            apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
    folder = out / str(scale)
    generation = SceneGenerationOwner(folder / "generations").replace(scene)
    stage = Usd.Stage.Open(generation.usd_path)
    prims = sorted((p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)), key=lambda p: UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[0])
    paths = [str(p.GetPath()) for p in prims]
    composed = [tuple(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()) for p in prims]
    assert all(abs(p[2]-2)<1e-5 for p in composed), composed
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    server = runtime / "bin/ovphysx_grpc_server.exe"
    def factory(config):
        return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
            ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", args.device]),
            device=args.device, physics_fps=120, update_fps=120, max_steps=24,
            worker_log_path=str(folder / "worker.log"), ovphysx_native_client_path=str(runtime / "client"),
            ovphysx_native_client_module="ovphysx_bridge_client"))
    adapter = OvphysxGenerationAdapter(factory)
    try:
        assert adapter.activate(generation), "worker activation"
        for step in range(24):
            update = adapter.controller.tick()
            assert update.status == OvphysxStageStatus.OK, adapter.controller.last_error
        final = [next(tuple(p.translate) for p in update.pose_set if p.prim_path == path) for path in paths]
        assert all(abs(p[2]-1.795625)<.002 for p in final[:3]), final
        assert abs(final[3][2]-2)<.001, final
        results[str(scale)] = {"composed": composed, "native": final}
        print(scale, final, flush=True)
    finally:
        adapter.deactivate()
assert all(abs(a-b)<1e-4 for arow,brow in zip(results["1.0"]["native"], results["0.01"]["native"]) for a,b in zip(arow,brow)), results
(out / "results.json").write_text(json.dumps(results, indent=2))
print("NATIVE_UNIT_SCALE_PASS", flush=True)
