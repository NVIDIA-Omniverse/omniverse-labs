"""Actual private USD compilation for independent owners and compound colliders."""
import hashlib
import json
from pathlib import Path
import sys
import uuid
import bpy
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_owner_compilation import gather_physics_owners

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
checks = []
def check(label, result):
    assert result, label
    checks.append(label)

def body(obj, bone=None):
    target = initialize_owner(obj, bone)
    target.ov.physics.rigid_body.schema_opinion = "APPLY"
    return target

def collider(owner, shape, material=None):
    record = owner.ov.physics.colliders.add()
    record.name = record.persistent_id = uuid.uuid4().hex
    record.shape = shape
    record.schema_opinion = "APPLY"
    record.physics_material = material
    return record

bpy.ops.mesh.primitive_cube_add()
mesh = bpy.context.object
mesh.name = "MeshBody"
mesh.location = (3, 4, 5)
mesh.scale = (2, 2, 2)
mesh.rotation_euler = (0.2, 0.3, 0.1)
body(mesh)
mesh.ov.physics.rigid_body.offset.translation = (0.3, 0.1, 0.2)
mesh.ov.physics.rigid_body.offset.rotation = (0.9238795, 0, 0, 0.3826834)
material = bpy.data.materials.new("OnlyPhysicsMaterial")
material.ov.physics.schema_opinion = "APPLY"
material.ov.physics.static_friction = 0.73
hull = collider(mesh, "CONVEX_HULL", material)
sphere = collider(mesh, "ANALYTIC_SPHERE", material)
sphere.offset.translation = (1, 0, 0)
sphere.radius_m = 0.25
empty = bpy.data.objects.new("NonRenderBody", None)
scene.collection.objects.link(empty)
empty.location = (0, 0, 8)
body(empty)
collider(empty, "ANALYTIC_SPHERE", material)
arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("RigData"))
scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bone = arm.data.edit_bones.new("BodyBone")
bone.head, bone.tail = (0, 0, 0), (0, 1, 0)
bpy.ops.object.mode_set(mode="OBJECT")
target = body(arm, arm.data.bones[0])
collider(target, "ANALYTIC_SPHERE", material)
bpy.context.view_layer.update()

def compile_scene(label):
    stock_path = out / (label + "-stock.usdc")
    generation._stock_export(scene, stock_path)
    mappings = generation._validated_blender_prim_paths(scene, stock_path)
    before = hashlib.sha256(stock_path.read_bytes()).hexdigest()
    records = generation._compile_add_on_opinions(scene, mappings, stock_path)
    check(label + " source unchanged", hashlib.sha256(stock_path.read_bytes()).hexdigest() == before)
    layers = []
    for record in records:
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(record.layer_text)
        layers.append(layer)
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [layer.identifier for layer in layers] + [str(stock_path)]
    stage = Usd.Stage.Open(root)
    stage.Flatten().Export(str(out / (label + "-composed.usda")))
    return stage, mappings, layers, root, Usd.Stage.Open(str(stock_path))

stage, mappings, layers, root, stock = compile_scene("meters")
owners = gather_physics_owners(scene, mappings, stock)
check("independent object empty bone owners", len(owners) == 3)
paths = {o.target.name: o.body_path for o in owners}
check("ordinary Object identity retained", paths[mesh.name] == mappings[generation.blender_id(mesh, "OBJECT")].object_path)
check("flat bone namespace", paths[target.name].startswith("/World/OVPhysics/Bodies/"))
check("three body APIs", sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 3)
check("four colliders", sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()) == 4)
check("physics only material dependency", any(p.HasAPI(UsdPhysics.MaterialAPI) for p in stage.Traverse()))
mesh_path = mappings[generation.blender_id(mesh, "OBJECT")].schema_path
visual = stage.GetPrimAtPath(mesh_path)
check("visual remains visible", UsdGeom.Imageable(visual).ComputeVisibility() != "invisible")
cache, source_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
for point in UsdGeom.Mesh(visual).GetPointsAttr().Get():
    a = cache.GetLocalToWorldTransform(visual).Transform(Gf.Vec3d(point))
    b = source_cache.GetLocalToWorldTransform(stock.GetPrimAtPath(mesh_path)).Transform(Gf.Vec3d(point))
    check("body offset preserves visual", (a-b).GetLength() < 2e-5)
check("no helper custom properties", all(not a.GetName().startswith("userProperties:ov:") for p in stage.Traverse() for a in p.GetAttributes()))
meter_colliders = {str(p.GetPath()): (UsdGeom.XformCache().GetLocalToWorldTransform(p),
    p.GetAttribute("radius").Get() if p.IsA(UsdGeom.Sphere) else tuple(tuple(v) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()))
    for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}

# Equivalent centimeter setup: all Blender lengths increase 100x; meters remain
# explicit meters for radius_m. Uniform object scale remains unchanged.
meter_positions = {o.body_identity: tuple(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(o.body_path)).ExtractTranslation()) for o in owners}
scene.unit_settings.scale_length = 0.01
for obj in (mesh, empty, arm):
    obj.location *= 100
for vertex in mesh.data.vertices:
    vertex.co *= 100
mesh.ov.physics.rigid_body.offset.translation = tuple(v * 100 for v in mesh.ov.physics.rigid_body.offset.translation)
sphere.offset.translation = tuple(v * 100 for v in sphere.offset.translation)
bpy.context.view_layer.update()
cm, cm_mappings, cm_layers, cm_root, cm_stock = compile_scene("centimeters")
check("stock normalized exactly once without scaling future physics siblings", all(sum(op.GetOpName() == "xformOp:scale:ovSceneUnits" for op in UsdGeom.Xformable(child).GetOrderedXformOps()) == 1 for child in cm_stock.GetPrimAtPath("/World").GetChildren() if child.IsA(UsdGeom.Xformable)) and not UsdGeom.Xformable(cm_stock.GetPrimAtPath("/World")).GetOrderedXformOps())
for path, (meter_transform, geometry) in meter_colliders.items():
    prim = cm.GetPrimAtPath(path)
    cm_transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    check("meter centimeter collider pose", max(abs(cm_transform[i][j]-meter_transform[i][j]) for i in range(4) for j in range(4)) < 2e-5)
    current = prim.GetAttribute("radius").Get() if prim.IsA(UsdGeom.Sphere) else tuple(tuple(v) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    check("meter centimeter collider geometry", current == geometry)
for owner in gather_physics_owners(scene, cm_mappings, cm_stock):
    position = UsdGeom.XformCache().GetLocalToWorldTransform(cm.GetPrimAtPath(owner.body_path)).ExtractTranslation()
    check("meter centimeter body equivalence", (position-Gf.Vec3d(meter_positions[owner.body_identity])).GetLength() < 2e-5)
for point in UsdGeom.Mesh(cm.GetPrimAtPath(mesh_path)).GetPointsAttr().Get():
    a = UsdGeom.XformCache().GetLocalToWorldTransform(cm.GetPrimAtPath(mesh_path)).Transform(Gf.Vec3d(point))
    b = UsdGeom.XformCache().GetLocalToWorldTransform(cm_stock.GetPrimAtPath(mesh_path)).Transform(Gf.Vec3d(point))
    check("centimeter visual preserved", (a-b).GetLength() < 2e-5)

old_collider_paths = [str(p.GetPath()) for p in cm.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
generation_owner = generation.SceneGenerationOwner(out / "generations")
first = generation_owner.replace(scene)
first_stage = Usd.Stage.Open(first.usd_path)
check("private generation full colliders", sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in first_stage.Traverse()) == 4)
for owner in (mesh, empty, target):
    owner.ov.physics.colliders.clear()
    owner.ov.physics.rigid_body.schema_opinion = "REMOVE"
removed, _, keep_layers, keep_root, _ = compile_scene("removed")
check("no stale colliders", not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in removed.Traverse()))
check("no stale materials", not any(p.HasAPI(UsdPhysics.MaterialAPI) for p in removed.Traverse()))
check("body schema removal", not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in removed.Traverse()))
second = generation_owner.replace(scene)
second_stage = Usd.Stage.Open(second.usd_path)
check("replacement no stale colliders", not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in second_stage.Traverse()))
check("replacement no stale material schemas", not any(p.HasAPI(UsdPhysics.MaterialAPI) for p in second_stage.Traverse()))
check("replacement no generated stale children", not any(second_stage.GetPrimAtPath(p) and second_stage.GetPrimAtPath(p).IsActive() for p in old_collider_paths))
stage_root = scene.ov.physics_namespace
scene.ov.physics_namespace = "/World/MeshBody"
try:
    generation._compile_add_on_opinions(scene, mappings, out / "meters-stock.usdc")
except ValueError:
    checks.append("namespace collision rejected")
else:
    raise AssertionError("namespace collision accepted")
scene.ov.physics_namespace = stage_root
scene.unit_settings.scale_length = 1.0
bpy.ops.mesh.primitive_cube_add()
hull_body = bpy.context.object
hull_body.name = "HullBody"
body(hull_body)
hull_record = collider(hull_body, "CONVEX_HULL", material)
hull_stage, hull_mappings, hull_layers, hull_root, hull_stock = compile_scene("hull")
hull_mapping = hull_mappings[generation.blender_id(hull_body, "OBJECT")]
hull_path = hull_mapping.object_path + "/Colliders/c_" + hull_record.persistent_id
hull_prim = hull_stage.GetPrimAtPath(hull_path)
check("hull compiles as generated collider mesh", hull_prim.IsA(UsdGeom.Mesh) and hull_prim.HasAPI(UsdPhysics.CollisionAPI)
      and hull_prim.GetAttribute("physics:approximation").Get() == "convexHull")
check("generated hull invisible", UsdGeom.Imageable(hull_prim).ComputeVisibility() == "invisible")
check("generated hull copies source topology",
      len(UsdGeom.Mesh(hull_prim).GetPointsAttr().Get()) == len(UsdGeom.Mesh(hull_stock.GetPrimAtPath(hull_mapping.schema_path)).GetPointsAttr().Get()))
check("generated hull binds physics material", len(hull_prim.GetRelationship("material:binding:physics").GetTargets()) == 1)
check("mapped visual mesh is not a collider", not hull_stage.GetPrimAtPath(hull_mapping.schema_path).HasAPI(UsdPhysics.CollisionAPI))
check("hull keeps visible mesh", UsdGeom.Imageable(hull_stage.GetPrimAtPath(hull_mapping.schema_path)).ComputeVisibility() != "invisible")
check("hull body rigid", hull_stage.GetPrimAtPath(hull_mapping.object_path).HasAPI(UsdPhysics.RigidBodyAPI))
collider(hull_body, "ANALYTIC_SPHERE", material)
compound, _, compound_layers, compound_root, _ = compile_scene("hull-compound")
check("hull plus sphere compound", sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in compound.Traverse()
      if str(p.GetPath()).startswith(hull_mapping.object_path + "/")) == 2)
check("compound keeps visible mesh", UsdGeom.Imageable(compound.GetPrimAtPath(hull_mapping.schema_path)).ComputeVisibility() != "invisible")
hull_record.shape = "ANALYTIC_SPHERE"
switched, _, switched_layers, switched_root, _ = compile_scene("hull-switched")
check("shape switch replaces generated hull", switched.GetPrimAtPath(hull_path).IsA(UsdGeom.Sphere)
      and not any(p.IsA(UsdGeom.Mesh) for p in switched.Traverse() if str(p.GetPath()).startswith(hull_mapping.object_path + "/Colliders/")))
check("shape switch keeps two sphere colliders", sum(p.IsA(UsdGeom.Sphere) and p.HasAPI(UsdPhysics.CollisionAPI) for p in switched.Traverse()) == 2)
scripted_empty = bpy.data.objects.new("ScriptedEmpty", None)
scene.collection.objects.link(scripted_empty)
body(scripted_empty)
scripted_record = collider(scripted_empty, "ANALYTIC_SPHERE")
scripted_record.radius_m = 0.4
scripted, scripted_mappings, scripted_layers, scripted_root, scripted_stock = compile_scene("scripted-empty")
scripted_owner = next(o for o in gather_physics_owners(scene, scripted_mappings, scripted_stock) if o.object == scripted_empty)
check("scripted non-render body compiled", scripted.GetPrimAtPath(scripted_owner.body_path).HasAPI(UsdPhysics.RigidBodyAPI))
scripted_sphere = scripted.GetPrimAtPath(scripted_owner.body_path + "/Colliders/c_" + scripted_record.persistent_id)
check("scripted non-render sphere collider", scripted_sphere.IsA(UsdGeom.Sphere) and scripted_sphere.HasAPI(UsdPhysics.CollisionAPI)
      and abs(scripted_sphere.GetAttribute("radius").Get() - 0.4) < 1e-6)
from ovrtx_blender_example.physics_colliders import ColliderError
mesh.ov.physics.rigid_body.schema_opinion = "APPLY"
conflicting_record = collider(mesh, "ANALYTIC_SPHERE")
conflicting_record.name = conflicting_record.persistent_id = "a" * 32
container = bpy.data.objects.new("Colliders", None)
scene.collection.objects.link(container)
container.parent = mesh
bpy.ops.mesh.primitive_cube_add()
source_object = bpy.context.object
source_object.name = "c_" + "a" * 32
source_object.parent = container
bpy.context.view_layer.update()
accepted_generation = generation_owner.current_generation
accepted_bytes = Path(accepted_generation.usd_path).read_bytes()
def reject_source_collision(label):
    stock_path = out / (label + "-stock.usdc")
    generation._stock_export(scene, stock_path)
    collision_mappings = generation._validated_blender_prim_paths(scene, stock_path)
    before_bytes = stock_path.read_bytes()
    try:
        generation._compile_add_on_opinions(scene, collision_mappings, stock_path)
    except ColliderError as error:
        check(label + " actionable diagnostic", "overlaps stock/source USD" in str(error))
    else:
        raise AssertionError("generated collider overwrote source")
    check(label + " source unchanged on rejection", before_bytes == stock_path.read_bytes())
    try:
        generation_owner.replace(scene)
    except ColliderError:
        pass
    else:
        raise AssertionError("invalid replacement accepted")
    check(label + " accepted generation unchanged", generation_owner.current_generation is accepted_generation and Path(accepted_generation.usd_path).read_bytes() == accepted_bytes)
reject_source_collision("exact-collider-collision")
source_object.name = "UnrelatedDescendant"
reject_source_collision("collider-container-descendant")
bpy.data.objects.remove(source_object, do_unlink=True)
reject_source_collision("collider-container-only")
bpy.data.objects.remove(container, do_unlink=True)
mesh.ov.physics.colliders.clear()
bpy.ops.mesh.primitive_cube_add()
static_sphere = bpy.context.object
static_sphere.name = "StaticSphere"
initialize_owner(static_sphere)
static_record = collider(static_sphere, "ANALYTIC_SPHERE")
static_record.name = static_record.persistent_id = "b" * 32
bpy.ops.mesh.primitive_cube_add()
source_object = bpy.context.object
source_object.name = "Colliders"
source_object.parent = static_sphere
bpy.context.view_layer.update()
reject_source_collision("collision-only-sphere-container")
bpy.data.objects.remove(source_object, do_unlink=True)
bpy.data.objects.remove(static_sphere, do_unlink=True)
from ovrtx_blender_example.physics_owner_compilation import reserve_generated_subtree
inactive_stage = Usd.Stage.CreateInMemory()
inactive_stage.DefinePrim("/World/Body/Colliders", "Xform").SetActive(False)
try:
    reserve_generated_subtree(inactive_stage, "/World/Body/Colliders")
except ColliderError:
    checks.append("inactive source subtree reserved")
else:
    raise AssertionError("inactive source collision accepted")
check("no rigid body world", scene.rigidbody_world is None)
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("COMPILATION PASS", len(checks))
