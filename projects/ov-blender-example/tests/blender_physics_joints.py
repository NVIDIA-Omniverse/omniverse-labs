"""Concrete ordinary-joint composed USD acceptance; no simulation claim."""
import json
import math
from pathlib import Path
import sys
import uuid
import bpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.scale_length = 0.01
checks = []
def check(label, value):
    assert value, label
    checks.append(label)

def close(a, b):
    return abs(a-b) < 2e-5

def body(name, location, mesh=False):
    if mesh:
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.object
        obj.name = name
    else:
        obj = bpy.data.objects.new(name, None)
        scene.collection.objects.link(obj)
    obj.location = location
    initialize_owner(obj)
    obj.ov.physics.rigid_body.schema_opinion = "APPLY"
    c = obj.ov.physics.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion = "APPLY"
    c.shape = "ANALYTIC_SPHERE"
    return obj

a, b = body("A", (0, 0, 200), mesh=True), body("B", (400, 0, 200))
a.rotation_euler = (.2, .1, 0)
obj = bpy.data.objects.new("JointHandle", None)
scene.collection.objects.link(obj)
obj.location = (200, 100, 200)
obj.rotation_euler = (.1, .2, .3)
joint = initialize_joint(obj)
joint.enabled = True
joint.endpoint_a.object, joint.endpoint_b.object = a, b
joint.break_force, joint.break_torque = 25, 100
joint.collision_enabled = True
bpy.context.view_layer.update()
apply_anchor(obj, scene, bpy.context.evaluated_depsgraph_get())
path = scene.ov.physics_namespace + "/Joints/j_" + joint.persistent_id
owner = generation.SceneGenerationOwner(out / "generations")

def compile(label, sparse=False):
    if sparse:
        result = owner.reconcile(scene, {generation.blender_id(obj, "OBJECT")})
        check(label + " creates sparse candidate", result is not None)
        owner.accept(result)
    else:
        result = owner.replace(scene)
        if result is None:
            result = owner.current_generation
    stage = Usd.Stage.Open(result.usd_path)
    stage.Flatten().Export(str(out / (label + ".usda")))
    return stage, stage.GetPrimAtPath(path)

stage, prim = compile("fixed")
check("fixed schema", prim.IsA(UsdPhysics.FixedJoint))
check("two body references", len(prim.GetRelationship("physics:body0").GetTargets()) == len(prim.GetRelationship("physics:body1").GetTargets()) == 1)
mapping = owner.current_generation.blender_prim_paths[generation.blender_id(a, "OBJECT")]
check("relationship targets mapped body not collider", str(prim.GetRelationship("physics:body0").GetTargets()[0]) == mapping.object_path != mapping.schema_path)
check("break force units", close(prim.GetAttribute("physics:breakForce").Get(), .25))
check("break torque units", close(prim.GetAttribute("physics:breakTorque").Get(), .01))
check("connected collision flag", prim.GetAttribute("physics:collisionEnabled").Get())
for side in (0, 1):
    endpoint = stage.GetPrimAtPath(prim.GetRelationship(f"physics:body{side}").GetTargets()[0])
    local = Gf.Vec3d(prim.GetAttribute(f"physics:localPos{side}").Get())
    world = UsdGeom.XformCache().GetLocalToWorldTransform(endpoint).Transform(local)
    check("endpoint frame common anchor", (world-Gf.Vec3d(2, 1, 2)).GetLength() < 2e-5)
    rotation = UsdGeom.XformCache().GetLocalToWorldTransform(endpoint).ExtractRotationQuat() * Gf.Quatd(prim.GetAttribute(f"physics:localRot{side}").Get())
    expected = obj.matrix_world.to_quaternion()
    check("joint quaternion frame convention", abs(sum(x*y for x, y in zip((rotation.GetReal(), *rotation.GetImaginary()), expected))) > 1-2e-5)

joint.type, joint.axis = "REVOLUTE", "Y"
axis = joint.axes["rotY"]
axis.state, axis.lower, axis.upper = "LIMITED", -math.pi/2, math.pi/4
axis.drive.enabled = True
axis.drive.target_position, axis.drive.target_velocity = math.pi/3, math.pi/6
axis.drive.stiffness, axis.drive.damping, axis.drive.max_effort = 1200, 600, 300
stage, prim = compile("revolute-y", sparse=True)
check("nondefault revolute axis", prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetAttribute("physics:axis").Get() == "Y")
check("revolute degree limits", close(prim.GetAttribute("physics:lowerLimit").Get(), -90) and close(prim.GetAttribute("physics:upperLimit").Get(), 45))
drive = UsdPhysics.DriveAPI(prim, "angular")
check("angular instance and target", prim.HasAPI(UsdPhysics.DriveAPI, "angular") and close(drive.GetTargetPositionAttr().Get(), 60))
check("angular velocity degrees", close(drive.GetTargetVelocityAttr().Get(), 30))
check("angular dimensional gain", close(drive.GetStiffnessAttr().Get(), 1200*.0001*math.pi/180))
check("angular effort dimensional gain", close(drive.GetMaxForceAttr().Get(), .03))
joint.axis = "Z"
stage, prim = compile("revolute-axis-switch", sparse=True)
check("axis switch drops incompatible drive", not prim.HasAPI(UsdPhysics.DriveAPI, "angular") and not prim.GetAttribute("physics:lowerLimit").HasAuthoredValueOpinion())
check("axis switch preserves semantic source", joint.axes["rotY"].drive.enabled and close(joint.axes["rotY"].drive.target_position, math.pi/3))
joint.axis = "Y"
stage, prim = compile("revolute-axis-restored", sparse=True)
check("axis restore uses original semantics", prim.HasAPI(UsdPhysics.DriveAPI, "angular") and close(UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Get(), 60))
axis.state = "LOCKED"
stage, prim = compile("revolute-locked", sparse=True)
check("revolute lock encoding", prim.GetAttribute("physics:lowerLimit").Get() > prim.GetAttribute("physics:upperLimit").Get())
axis.state = "FREE"
stage, prim = compile("revolute-free", sparse=True)
check("revolute free clears limits", not prim.GetAttribute("physics:lowerLimit").HasAuthoredValueOpinion() and math.isinf(prim.GetAttribute("physics:upperLimit").Get()))

joint.type, joint.axis = "PRISMATIC", "Z"
axis = joint.axes["transZ"]
axis.state, axis.lower, axis.upper = "LIMITED", -100, 200
axis.drive.enabled = True
axis.drive.target_position, axis.drive.target_velocity = 50, 25
axis.drive.stiffness, axis.drive.damping, axis.drive.max_effort = 12, 6, 300
stage, prim = compile("prismatic-z", sparse=True)
check("nondefault prismatic axis", prim.IsA(UsdPhysics.PrismaticJoint) and prim.GetAttribute("physics:axis").Get() == "Z")
check("prismatic meter limits", close(prim.GetAttribute("physics:lowerLimit").Get(), -1) and close(prim.GetAttribute("physics:upperLimit").Get(), 2))
drive = UsdPhysics.DriveAPI(prim, "linear")
check("linear instance", prim.HasAPI(UsdPhysics.DriveAPI, "linear") and not prim.HasAPI(UsdPhysics.DriveAPI, "angular"))
check("linear targets", close(drive.GetTargetPositionAttr().Get(), .5) and close(drive.GetTargetVelocityAttr().Get(), .25))
check("linear gains", close(drive.GetStiffnessAttr().Get(), 12) and close(drive.GetMaxForceAttr().Get(), 3))
axis.state = "LOCKED"
stage, prim = compile("prismatic-locked", sparse=True)
check("prismatic lock encoding", prim.GetAttribute("physics:lowerLimit").Get() > prim.GetAttribute("physics:upperLimit").Get())
axis.state = "FREE"
stage, prim = compile("prismatic-free", sparse=True)
check("prismatic free clears limits", not prim.GetAttribute("physics:upperLimit").HasAuthoredValueOpinion())

joint.type = "D6"
joint.axes["transX"].state = "LOCKED"
joint.axes["rotZ"].state = "LOCKED"
joint.axes["transZ"].state = "LIMITED"
joint.axes["rotY"].state = "LIMITED"
joint.axes["rotY"].lower, joint.axes["rotY"].upper = -math.pi/6, math.pi/6
stage, prim = compile("d6", sparse=True)
check("generic D6 schema", prim.GetTypeName() == "PhysicsJoint")
check("D6 semantic drives", prim.HasAPI(UsdPhysics.DriveAPI, "rotY") and prim.HasAPI(UsdPhysics.DriveAPI, "transZ") and not prim.HasAPI(UsdPhysics.DriveAPI, "linear"))
check("D6 free has no LimitAPI", not prim.HasAPI(UsdPhysics.LimitAPI, "transY"))
check("D6 locked semantic axis", UsdPhysics.LimitAPI(prim, "transX").GetLowAttr().Get() > UsdPhysics.LimitAPI(prim, "transX").GetHighAttr().Get())
check("D6 angular degree limits", close(UsdPhysics.LimitAPI(prim, "rotY").GetHighAttr().Get(), 30))
joint.axes["rotY"].state = "FREE"
joint.axes["rotY"].drive.enabled = False
stage, prim = compile("d6-free", sparse=True)
check("D6 free removes stale limit and drive", not prim.HasAPI(UsdPhysics.LimitAPI, "rotY") and not prim.HasAPI(UsdPhysics.DriveAPI, "rotY"))
joint.type = "FIXED"
stage, prim = compile("fixed-again", sparse=True)
check("fixed removes all stale axis properties", not any(p.GetName().startswith(("limit:", "drive:")) for p in prim.GetProperties()))
check("inactive settings retained", joint.axes["transZ"].drive.enabled and joint.axes["transZ"].lower == -100)

joint.endpoint_b.kind = "WORLD"
apply_anchor(obj, scene, bpy.context.evaluated_depsgraph_get())
stage, prim = compile("world-b")
check("explicit world relationship empty", prim.GetRelationship("physics:body1").GetTargets() == [])
old_frame = tuple(prim.GetAttribute("physics:localPos0").Get())
obj.location.x += 10
bpy.context.view_layer.update()
stage, prim = compile("pending-committed")
check("pending handle retains committed frames", tuple(prim.GetAttribute("physics:localPos0").Get()) == old_frame)
a.location.x += 10
bpy.context.view_layer.update()
stage, prim = compile("moved-body")
check("moving body never recaptures frames", tuple(prim.GetAttribute("physics:localPos0").Get()) == old_frame)
joint.endpoint_a.kind, joint.endpoint_b.kind = "WORLD", "OBJECT"
joint.endpoint_b.object = b
apply_anchor(obj, scene, bpy.context.evaluated_depsgraph_get())
stage, prim = compile("world-a")
check("world on first endpoint", prim.GetRelationship("physics:body0").GetTargets() == [] and len(prim.GetRelationship("physics:body1").GetTargets()) == 1)

def reject(label):
    current = owner.current_generation
    try:
        owner.replace(scene)
    except ValueError:
        check(label + " preserves accepted generation", owner.current_generation is current)
    else:
        raise AssertionError(label)
for unsupported in ("SPHERICAL", "DISTANCE"):
    joint.type = unsupported
    reject(unsupported + " diagnosed")
joint.type = "PRISMATIC"
joint.axes["transZ"].drive.mode = "ACCELERATION"
reject("acceleration drive unavailable")
joint.axes["transZ"].drive.mode = "FORCE"
b.ov.physics.rigid_body.role = "STATIC"
reject("static-to-world joint without a simulated body diagnosed")
b.ov.physics.rigid_body.role = "DYNAMIC"
# A static endpoint compiles as a WORLD attachment at its (rotated, scaled)
# body frame, filtered against its partner unless joint collision is enabled.
joint.endpoint_a.kind, joint.endpoint_a.object = "OBJECT", a
a.ov.physics.rigid_body.role = "STATIC"
bpy.context.view_layer.update()
apply_anchor(obj, scene, bpy.context.evaluated_depsgraph_get())
stage, prim = compile("static-a")
a_path = owner.current_generation.blender_prim_paths[generation.blender_id(a, "OBJECT")].object_path
b_path, = prim.GetRelationship("physics:body1").GetTargets()
check("static endpoint compiles as world attachment", prim.GetRelationship("physics:body0").GetTargets() == [])
b_matrix = UsdGeom.Xformable(stage.GetPrimAtPath(b_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
anchor_world = b_matrix.Transform(Gf.Vec3d(prim.GetAttribute("physics:localPos1").Get()))
check("static world position matches partner frame",
      (anchor_world - Gf.Vec3d(prim.GetAttribute("physics:localPos0").Get())).GetLength() < 1e-4)
rotation_world = Gf.Quatd(b_matrix.RemoveScaleShear().ExtractRotationQuat()) * Gf.Quatd(prim.GetAttribute("physics:localRot1").Get())
check("static world rotation matches partner frame",
      abs(abs(Gf.Dot(rotation_world.GetNormalized(), Gf.Quatd(prim.GetAttribute("physics:localRot0").Get()).GetNormalized())) - 1) < 1e-5)
check("static body is not a rigid body", not stage.GetPrimAtPath(a_path).HasAPI(UsdPhysics.RigidBodyAPI))
check("collision-enabled static joint is not filtered", not stage.GetPrimAtPath(b_path).HasAPI(UsdPhysics.FilteredPairsAPI))
joint.collision_enabled = False
stage, prim = compile("static-a-filtered")
filtered = stage.GetPrimAtPath(b_path)
check("static partner contacts filtered",
      filtered.HasAPI(UsdPhysics.FilteredPairsAPI)
      and UsdPhysics.FilteredPairsAPI(filtered).GetFilteredPairsRel().GetTargets() == [Sdf.Path(a_path)])
check("filtered body keeps rigid body", filtered.HasAPI(UsdPhysics.RigidBodyAPI))
joint.collision_enabled = True
a.ov.physics.rigid_body.role = "DYNAMIC"
joint.endpoint_a.kind = "WORLD"
apply_anchor(obj, scene, bpy.context.evaluated_depsgraph_get())
scene.ov.physics_scene.schema_opinion = "APPLY"
scene.ov.usd.physics_scene_path = path
reject("generated family namespace collision")
scene.ov.physics_scene.schema_opinion = "INHERIT"
joint.axes["rotY"].drive.keyframe_insert(data_path="target_position", frame=1)
reject("inactive animated semantic axis diagnosed")
obj.animation_data_clear()
joint.axes["transZ"].drive.keyframe_insert(data_path="target_position", frame=1)
reject("active animated drive unavailable")
obj.animation_data_clear()
joint.keyframe_insert(data_path="show_indicator", frame=1)
stage, prim = compile("display-animation")
check("display animation does not become motor control", prim.IsA(UsdPhysics.PrismaticJoint))
obj.animation_data_clear()
joint.enabled = False
stage, prim = compile("joint-disabled", sparse=True)
check("disabled joint removes stale prim", not prim or not prim.IsActive())
joint.enabled = True
stage, prim = compile("joint-restored", sparse=True)
deleted_identity = generation.blender_id(obj, "OBJECT")
bpy.data.objects.remove(obj, do_unlink=True)
deleted = owner.reconcile(scene, {deleted_identity})
owner.accept(deleted)
deleted_stage = Usd.Stage.Open(deleted.usd_path)
check("deleted joint removes sparse opinion", not deleted_stage.GetPrimAtPath(path))
# Mesh bodies with convex-hull collider records honor Static/disabled roles, and a
# static mesh endpoint anchors a mesh chain like an authored WORLD joint.
def hull_mesh(name, location):
    bpy.ops.mesh.primitive_cube_add(location=location)
    mesh = bpy.context.object
    mesh.name = name
    initialize_owner(mesh)
    mesh.ov.physics.rigid_body.schema_opinion = "APPLY"
    mesh.ov.physics.rigid_body.mass_kg = 1
    c = mesh.ov.physics.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion = "APPLY"
    c.shape = "CONVEX_HULL"
    return mesh

anchor, hanging = hull_mesh("MeshAnchor", (0, 1000, 500)), hull_mesh("MeshHanging", (0, 1250, 500))
anchor.ov.physics.rigid_body.role = "STATIC"
mesh_handle = bpy.data.objects.new("MeshJoint", None)
scene.collection.objects.link(mesh_handle)
mesh_handle.location = (0, 1125, 500)
mesh_joint = initialize_joint(mesh_handle)
mesh_joint.type, mesh_joint.axis = "REVOLUTE", "X"
mesh_joint.endpoint_a.object, mesh_joint.endpoint_b.object = anchor, hanging
bpy.context.view_layer.update()
apply_anchor(mesh_handle, scene, bpy.context.evaluated_depsgraph_get())
hanging.ov.physics.rigid_body.enabled = False
stage = Usd.Stage.Open(owner.replace(scene).usd_path)
mapped = owner.current_generation.blender_prim_paths
anchor_prim = stage.GetPrimAtPath(mapped[generation.blender_id(anchor, "OBJECT")].object_path)
hanging_prim = stage.GetPrimAtPath(mapped[generation.blender_id(hanging, "OBJECT")].object_path)
anchor_hull = stage.GetPrimAtPath(str(anchor_prim.GetPath()) + "/Colliders/c_" + anchor.ov.physics.colliders[0].persistent_id)
check("static mesh is collision only", not anchor_prim.HasAPI(UsdPhysics.RigidBodyAPI) and not anchor_prim.HasAPI(UsdPhysics.MassAPI))
check("static mesh keeps generated hull collider",
      anchor_hull.IsA(UsdGeom.Mesh) and anchor_hull.HasAPI(UsdPhysics.CollisionAPI)
      and anchor_hull.GetAttribute("physics:approximation").Get() == "convexHull")
check("disabled mesh keeps disabled rigid body",
      hanging_prim.HasAPI(UsdPhysics.RigidBodyAPI) and hanging_prim.GetAttribute("physics:rigidBodyEnabled").Get() is False)
hanging.ov.physics.rigid_body.enabled = True
mesh_joint.enabled = True
stage = Usd.Stage.Open(owner.replace(scene).usd_path)
mesh_joint_prim = stage.GetPrimAtPath(scene.ov.physics_namespace + "/Joints/j_" + mesh_joint.persistent_id)
check("static mesh endpoint compiles as world attachment",
      mesh_joint_prim.GetRelationship("physics:body0").GetTargets() == []
      and mesh_joint_prim.GetRelationship("physics:body1").GetTargets() == [hanging_prim.GetPath()])
check("static mesh partner contacts filtered",
      UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(hanging_prim.GetPath())).GetFilteredPairsRel().GetTargets() == [anchor_prim.GetPath()])
check("enabled mesh rigid body", stage.GetPrimAtPath(hanging_prim.GetPath()).GetAttribute("physics:rigidBodyEnabled").Get() is not False)
# The composed body keeps the matrix override and collision filtering together.
hanging[generation.MATRIX_OVERRIDE_PROP] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 12.5, 7, 1]
stage = Usd.Stage.Open(owner.replace(scene).usd_path)
root_records = [r for r in owner.current_generation.opinion_records if r.usd_prim_path == str(hanging_prim.GetPath())]
override_z = UsdGeom.Xformable(stage.GetPrimAtPath(hanging_prim.GetPath())).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
# Current generation compaction can merge same-prim records. Assert their
# observable opinions, not the number of intermediate records.
check("static filter keeps matrix override", bool(root_records) and abs(override_z - 7) < 1e-4)
check("static filter survives matrix override",
      UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(hanging_prim.GetPath())).GetFilteredPairsRel().GetTargets() == [anchor_prim.GetPath()])
del hanging[generation.MATRIX_OVERRIDE_PROP]
check("no native rigid body world", scene.rigidbody_world is None)
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("JOINT COMPILATION PASS", len(checks))
