"""Read-only fixed native runtime investigation, deliberately bypassing admission.

This hand-authored USD distinguishes native behavior from Blender compilation.
At angle30 with a world-fixed anchor, CPU is stationary and GPU moves body B
0.43m by tick4 despite zero gravity. Angle0, --free and --fixedpair controls
can pass: the product's restriction is conservative, not a claim that every
GPU body pair fails. The production adapter rejects unvalidated GPU inputs.
Run in Blender with --runtime, --output, --device and optional control flags.
"""
import argparse, os, sys, socket, subprocess, json, math
from pathlib import Path
from dataclasses import replace
from pxr import Usd, UsdGeom, UsdPhysics, Gf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig
p = argparse.ArgumentParser()
p.add_argument('--device')
p.add_argument('--angle', type=float, default=30)
p.add_argument('--free', action='store_true')
p.add_argument('--fixedpair', action='store_true')
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = a.output.resolve()
out.mkdir(exist_ok=True)
runtime = a.runtime.resolve()
server = runtime / 'bin/ovphysx_grpc_server.exe'
os.environ['OVPHYSX_ROOT'] = str(runtime / 'private/ovphysx-runtime')
stage = Usd.Stage.CreateNew(str(out / 'input.usda'))
UsdGeom.SetStageMetersPerUnit(stage, 1)
UsdGeom.SetStageUpAxis(stage, 'Z')
UsdPhysics.Scene.Define(stage, '/World/PhysicsScene').CreateGravityMagnitudeAttr(0)
rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), a.angle)
q = rot.GetQuat()
inv = rot.GetInverse().GetQuat()
positions = {'/World/A': Gf.Vec3d(0, 0, 2), '/World/B': Gf.Vec3d(-math.sin(math.radians(a.angle)), math.cos(math.radians(a.angle)), 2)}
for path, pos in positions.items():
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTransformOp().Set(Gf.Matrix4d().SetRotate(rot) * Gf.Matrix4d().SetTranslate(pos))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(1)
    sphere = UsdGeom.Sphere.Define(stage, path + '/Sphere')
    sphere.CreateRadiusAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
j = UsdPhysics.FixedJoint.Define(stage, '/World/Fixed')
j.CreateBody1Rel().SetTargets(['/World/A'])
j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 2))
j.CreateLocalRot0Attr(Gf.Quatf(q))
j.CreateLocalRot1Attr(Gf.Quatf(1))
j = UsdPhysics.RevoluteJoint.Define(stage, '/World/Hinge')
j.CreateBody0Rel().SetTargets(['/World/A'])
j.CreateBody1Rel().SetTargets(['/World/B'])
j.CreateAxisAttr('X')
j.CreateLocalPos0Attr(Gf.Vec3f(0))
j.CreateLocalPos1Attr(Gf.Vec3f(0, -1, 0))
j.CreateLocalRot0Attr(Gf.Quatf(inv))
j.CreateLocalRot1Attr(Gf.Quatf(inv))
j.CreateLowerLimitAttr(-30)
j.CreateUpperLimitAttr(30)
if a.free:
    stage.GetPrimAtPath('/World/Fixed').GetAttribute('physics:jointEnabled').Set(False)
if a.fixedpair:
    stage.RemovePrim('/World/Fixed')
    stage.RemovePrim('/World/Hinge')
    positions['/World/B'] = Gf.Vec3d(0, 1, 2)
    stage.GetPrimAtPath('/World/B').GetAttribute('xformOp:transform').Set(Gf.Matrix4d().SetTranslate(positions['/World/B']))
    j = UsdPhysics.FixedJoint.Define(stage, '/World/Pair')
    j.CreateBody0Rel().SetTargets(['/World/A'])
    j.CreateBody1Rel().SetTargets(['/World/B'])
    j.CreateLocalPos1Attr(Gf.Vec3f(0, -1, 0))
    j.CreateLocalRot0Attr(Gf.Quatf(inv))
    j.CreateLocalRot1Attr(Gf.Quatf(1))
stage.GetRootLayer().Save()
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    address = '127.0.0.1:' + str(sock.getsockname()[1])
config = InteractiveSharedStageConfig.from_env(str(out / 'input.usda'))
config = replace(config, server=str(server), ovphysx_address=address, ovphysx_worker_command=subprocess.list2cmdline([str(server), '--listen', address, '--device', a.device]), device=a.device, physics_fps=240, update_fps=240, max_steps=4, body_prims=tuple(positions), worker_log_path=str(out / 'worker.log'), ovphysx_native_client_path=str(runtime / 'client'), ovphysx_native_client_module='ovphysx_bridge_client')
c = OvphysxStageController(config)
results = []
try:
    for i in range(4):
        update = c.tick()
        results.append({p.prim_path: {'position': p.translate, 'rotation': p.orient, 'error': math.dist(p.translate, positions[p.prim_path])} for p in update.pose_set})
finally:
    c.shutdown()
(out / 'results.json').write_text(json.dumps(results, indent=2))
print(results, flush=True)
