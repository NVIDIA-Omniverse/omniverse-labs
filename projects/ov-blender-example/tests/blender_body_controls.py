"""Composed USD, sparse reset, invalidation and RNA persistence of body controls."""
import json
import math
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import uuid
import bpy
from pxr import Usd, UsdGeom, UsdPhysics
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_edit_snapshot import capture, classify
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.physics_body_controls import FIELD_MATRIX, UNAVAILABLE

props.register()
out = Path(sys.argv[sys.argv.index("--")+1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.scale_length = .01
obj = bpy.data.objects.new("Body", None)
scene.collection.objects.link(obj)
initialize_owner(obj)
p = obj.ov.physics
p.rigid_body.schema_opinion = "APPLY"
body = p.rigid_body
checks = []
def check(name, value):
    assert value, name
    checks.append(name)

for field, value in (("mass_mode", "DENSITY"), ("density", .002), ("center_of_mass", (10, 20, 30)),
    ("diagonal_inertia", (10000, 20000, 30000)), ("principal_axes", (.70710678, 0, 0, .70710678)),
    ("initial_linear_velocity", (100, 200, 300)), ("initial_angular_velocity", (0, 0, math.pi)),
    ("linear_damping", 1.2), ("angular_damping", .7)):
    before = capture(scene)
    setattr(body, field, value)
    change = classify(before, capture(scene))
    check(field + " explicit restart invalidation", change.restart and not change.body_input)
colliders = []
for shape in ("BOX", "CAPSULE"):
    c = p.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion, c.shape = "APPLY", shape
    c.dimensions, c.radius_m, c.height_m = (100, 200, 300), .2, .6
    colliders.append(c)
material = bpy.data.materials.new("Physics")
material.ov.physics.schema_opinion = "APPLY"
material.ov.physics.density = .003
colliders[0].physics_material = material
scene.ov.physics_scene.schema_opinion = "APPLY"
scene.ov.physics_scene.gravity_magnitude = 4.5
scene.ov.physics_scene.gravity_direction = (0, -1, 0)
bpy.context.view_layer.update()
owner = SceneGenerationOwner(out / "generations")
generation = owner.replace(scene)
stage = Usd.Stage.Open(generation.usd_path)
rigid = next(prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI))
def attribute(name):
    return rigid.GetAttribute(name).Get()
check("body density unit conversion and mass precedence", abs(attribute("physics:density")-2000)<.001 and attribute("physics:mass")==0)
check("COM unit conversion", all(abs(a-b)<1e-6 for a,b in zip(attribute("physics:centerOfMass"),(.1,.2,.3))))
check("inertia unit conversion", all(abs(a-b)<1e-5 for a,b in zip(attribute("physics:diagonalInertia"),(1,2,3))))
check("linear velocity units", all(abs(a-b)<1e-5 for a,b in zip(attribute("physics:velocity"),(1,2,3))))
check("angular velocity degrees", abs(attribute("physics:angularVelocity")[2]-180)<1e-4)
check("damping authored", abs(attribute("physxRigidBody:linearDamping")-1.2)<1e-6 and abs(attribute("physxRigidBody:angularDamping")-.7)<1e-6)
cube = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Cube))
extent = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(cube).ComputeAlignedRange().GetSize()
check("box dimensional world size", all(abs(a-b)<1e-5 for a,b in zip(extent,(1,2,3))))
capsule = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Capsule))
check("capsule dimensions axis visibility", abs(capsule.GetAttribute("height").Get()-.6)<1e-6 and capsule.GetAttribute("axis").Get()=="Z" and capsule.GetAttribute("visibility").Get()=="invisible")
check("material density unit conversion", any(p.HasAPI(UsdPhysics.MaterialAPI) and abs(p.GetAttribute("physics:density").Get()-3000)<.002 for p in stage.Traverse()))
before = capture(scene)
colliders[0].dimensions = (120, 200, 300)
check("box edit structural invalidation", classify(before,capture(scene)).structural)
body.mass_mode, body.mass_kg = "MASS", 3.5
body.property_unset("initial_linear_velocity")
body.property_unset("linear_damping")
generation = owner.replace(scene)
stage = Usd.Stage.Open(generation.usd_path)
rigid = next(prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI))
check("mode switch clears density precedence", attribute("physics:mass")==3.5 and attribute("physics:density")==0)
check("unset removes velocity and damping opinions", not rigid.GetAttribute("physics:velocity").HasAuthoredValueOpinion() and not rigid.GetAttribute("physxRigidBody:linearDamping").HasAuthoredValueOpinion())
from ovrtx_blender_example.physics_capabilities import validate_joint_runtime
fake_config = SimpleNamespace(device="cpu", server=sys.executable, ovphysx_worker_command=subprocess.list2cmdline([sys.executable]))
fake_client = SimpleNamespace(__file__=sys.executable, capabilities=lambda: {"rpcs": ("CreateSimulation", "ReadWorldState", "WriteWorldState", "DeleteSimulation")})
try:
    validate_joint_runtime(generation.usd_path, fake_config, lambda: fake_client)
except ValueError as exc:
    check("unknown expanded runtime fails closed before RPC", "identity" in str(exc))
else:
    raise AssertionError("unknown body-control runtime accepted")
body.diagonal_inertia = (1, 0, 2)
try:
    owner.replace(scene)
except ValueError:
    check("invalid partial inertia rejected", True)
else:
    raise AssertionError("invalid inertia accepted")
body.diagonal_inertia = (10000, 20000, 30000)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "controls.blend"))
bpy.ops.wm.open_mainfile(filepath=str(out / "controls.blend"))
check("save reload controls", bpy.data.objects["Body"].ov.physics.rigid_body.mass_mode=="MASS" and len(bpy.data.objects["Body"].ov.physics.colliders)==2)
check("explicit unavailable controls", all(word in UNAVAILABLE for word in ("CCD", "sleep", "solver", "telemetry")))
check("no native Blender physics", bpy.context.scene.rigidbody_world is None)
(out / "results.json").write_text(json.dumps({"checks":checks,"field_matrix":FIELD_MATRIX,"unavailable":UNAVAILABLE}, indent=2))
print("BODY_CONTROLS_COMPOSITION_PASS",len(checks))
