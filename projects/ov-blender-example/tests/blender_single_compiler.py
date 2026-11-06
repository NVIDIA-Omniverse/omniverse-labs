# SPDX-License-Identifier: Apache-2.0
"""The owner compiler is the only compiler: scale, identity and visibility cases."""
import json
from pathlib import Path
import sys
import uuid

import bpy
from mathutils import Vector
from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_pose_presentation
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock, suppress_interactive_edit_bridge
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_presentation_targets import presentation_targets

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
checks = []


def check(label, value):
    assert value, label
    checks.append(label)


def mesh_body(name, location, scale, *, role="DYNAMIC", body=True, identity=True):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name, obj.scale = name, scale
    if identity:
        initialize_owner(obj)
    physics = obj.ov.physics
    if body:
        physics.rigid_body.schema_opinion = "APPLY"
        physics.rigid_body.role = role
        physics.rigid_body.mass_kg = 1
    collider = physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "CONVEX_HULL"
    return obj


box = mesh_body("StretchedBox", (0, 0, 3), (2, 1, .5))
floor = mesh_body("Floor", (0, 0, -.5), (10, 10, 1), body=False)
unnamed = mesh_body("NoIdentity", (4, 0, 3), (1, 1, 1), identity=False)
hidden = mesh_body("HiddenRender", (-4, 0, 3), (1, 1, 1), role="STATIC")
hidden.ov.physics.colliders[0].hide_from_render = True
bpy.context.view_layer.update()

check("owner without identity initially", not unnamed.ov.physics.persistent_id)
sessions.repair_duplicate_identities(scene)
check("repair pass initializes missing identity", bool(unnamed.ov.physics.persistent_id))

owner = generation.SceneGenerationOwner(out / "generations")
result = owner.replace(scene)
stage = Usd.Stage.Open(result.usd_path)
paths = {o.name: result.blender_prim_paths[generation.blender_id(o, "OBJECT")] for o in scene.objects}


def hull(name):
    root = stage.GetPrimAtPath(paths[name].object_path + "/Colliders")
    return next(p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh))


box_prim = stage.GetPrimAtPath(paths["StretchedBox"].object_path)
check("nonuniform dynamic body compiles", box_prim.HasAPI(UsdPhysics.RigidBodyAPI))
cache = UsdGeom.XformCache()
check("dynamic body frame is rigid",
      all(abs(v - 1) < 1e-6 for v in (Vector(r[:3]).length for r in list(cache.GetLocalToWorldTransform(box_prim))[:3])))
extent = UsdGeom.Mesh(hull("StretchedBox")).GetPointsAttr().Get()
span = [max(p[i] for p in extent) - min(p[i] for p in extent) for i in range(3)]
check("hull bakes per-axis scale", all(abs(a - b) < 1e-5 for a, b in zip(span, (4, 2, 1))))
render = stage.GetPrimAtPath(paths["StretchedBox"].schema_path)
world_render = cache.GetLocalToWorldTransform(render)
check("render mesh keeps its stretched world transform",
      all(abs(Vector(world_render.GetRow3(i)).length - s) < 1e-5 for i, s in enumerate((2, 1, .5))))
floor_span = UsdGeom.Mesh(hull("Floor")).GetPointsAttr().Get()
check("collision-only floor bakes nonuniform scale",
      abs(max(p[0] for p in floor_span) - min(p[0] for p in floor_span) - 20) < 1e-4)
check("collision-only floor has no rigid body", not stage.GetPrimAtPath(paths["Floor"].object_path).HasAPI(UsdPhysics.RigidBodyAPI))
check("hide render geometry hides the mapped mesh",
      UsdGeom.Imageable(stage.GetPrimAtPath(paths["HiddenRender"].schema_path)).ComputeVisibility() == "invisible")
check("identity-less owner compiles", stage.GetPrimAtPath(paths["NoIdentity"].object_path).HasAPI(UsdPhysics.RigidBodyAPI))

target = next(t for t in presentation_targets(result) if t.body_path == paths["StretchedBox"].object_path)
check("presentation target carries axis scale", all(abs(a - b) < 1e-6 for a, b in zip(target.axis_scale, (2, 1, .5))))
lock = PhysicsPlaybackLock()
sessions._runtimes[int(scene.session_uid)] = type("Runtime", (), {"playback_lock": lock})()
try:
    with suppress_interactive_edit_bridge():
        physics_pose_presentation.apply(scene, (target,), {target.body_path: {"translate": (1, 2, 1), "orient": (0, 0, 0, 1)}},
                                        lock, result.number, bpy.context.view_layer)
    bpy.context.view_layer.update()
    check("presented pose keeps nonuniform Blender scale",
          all(abs(a - b) < 1e-5 for a, b in zip(box.matrix_world.to_scale(), (2, 1, .5)))
          and (box.matrix_world.translation - Vector((1, 2, 1))).length < 1e-5)
finally:
    lock.clear(reason="test_complete")
    sessions._runtimes.pop(int(scene.session_uid), None)
owner.close()


def compile_error(label, build):
    """Compile a scene variant and require an explicit ColliderError."""
    objects = build()
    bpy.context.view_layer.update()
    try:
        generation.SceneGenerationOwner(out / ("gen-" + label)).replace(scene)
    except Exception as exc:
        check(label, "ColliderError" in type(exc).__name__ or "shear" in str(exc) or "parent" in str(exc))
    else:
        raise AssertionError(label + " compiled")
    finally:
        for obj in objects:
            bpy.data.objects.remove(obj, do_unlink=True)


# Parented bodies: the runtime composes parents, so the body frame is parent-relative.
parent = bpy.data.objects.new("Parent", None)
scene.collection.objects.link(parent)
parent.location, parent.scale = (24, 0, 0), (2, 2, 2)
child = mesh_body("ParentedChild", (0, 0, 0), (1, 1, 1))
child.parent = parent
child.location = (0, 0, 3)
bpy.context.view_layer.update()
result = generation.SceneGenerationOwner(out / "gen-parented").replace(scene)
stage = Usd.Stage.Open(result.usd_path)
child_prim = stage.GetPrimAtPath(result.blender_prim_paths[generation.blender_id(child, "OBJECT")].object_path)
world = UsdGeom.XformCache().GetLocalToWorldTransform(child_prim)
check("parented body composes to its Blender world position",
      (Vector(world.ExtractTranslation()) - child.matrix_world.translation).length < 1e-5)
check("parented body frame is rigid", all(abs(Vector(world.GetRow3(i)).length - 1) < 1e-6 for i in range(3)))
check("parented body keeps no reset marker", not UsdGeom.Xformable(child_prim).GetResetXformStack())
bpy.data.objects.remove(child, do_unlink=True)


def nonuniform_parent():
    stretched = bpy.data.objects.new("StretchedParent", None)
    scene.collection.objects.link(stretched)
    stretched.scale = (2, 1, 1)
    moving = mesh_body("ChildOfStretched", (0, 5, 3), (1, 1, 1))
    moving.parent = stretched
    return [moving, stretched]


def rotated_offset():
    rotated = mesh_body("RotatedOffset", (0, -5, 3), (2, 1, .5))
    rotated.ov.physics.rigid_body.offset.rotation = (0.9238795, 0, 0, 0.3826834)
    return [rotated]


compile_error("moving body under nonuniform parent is rejected", nonuniform_parent)
compile_error("rotated offset on nonuniform owner is rejected", rotated_offset)
bpy.data.objects.remove(parent, do_unlink=True)

# A body parented to a scaled body authors its own rigid frame; the parent's
# compensation must not override it.
outer = mesh_body("OuterBody", (0, 12, 3), (2, 2, 2))
inner = mesh_body("InnerBody", (0, 0, 0), (1, 1, 1))
inner.parent = outer
inner.location, inner.scale = (3, 0, 0), (.75, .75, .75)
inner.ov.physics.rigid_body.offset.translation = (.2 / 1.5, 0, 0)
bpy.context.view_layer.update()
result = generation.SceneGenerationOwner(out / "gen-nested").replace(scene)
stage = Usd.Stage.Open(result.usd_path)
inner_path = result.blender_prim_paths[generation.blender_id(inner, "OBJECT")].object_path
inner_prim = stage.GetPrimAtPath(inner_path)
world = UsdGeom.XformCache().GetLocalToWorldTransform(inner_prim)
expected = inner.matrix_world @ Vector((.2 / 1.5, 0, 0))
check("nested body keeps its own frame and offset", (Vector(world.ExtractTranslation()) - expected).length < 1e-5)
check("nested body frame is rigid", all(abs(Vector(world.GetRow3(i)).length - 1) < 1e-6 for i in range(3)))
check("nested body has only its own transform op",
      [str(op.GetOpName()) for op in UsdGeom.Xformable(inner_prim).GetOrderedXformOps()] == ["xformOp:transform:ovPhysics"])
points = UsdGeom.Mesh(next(p for p in Usd.PrimRange(stage.GetPrimAtPath(inner_path + "/Colliders")) if p.IsA(UsdGeom.Mesh))).GetPointsAttr().Get()
check("nested hull is scaled once", abs(max(p[0] for p in points) - min(p[0] for p in points) - 3.0) < 1e-4)
for obj in (inner, outer):
    bpy.data.objects.remove(obj, do_unlink=True)


def kinematic_under_scaled_parent():
    holder = bpy.data.objects.new("ScaledHolder", None)
    scene.collection.objects.link(holder)
    holder.scale = (2, 2, 2)
    kinematic = mesh_body("KinematicChild", (0, -9, 1), (1, 1, 1), role="KINEMATIC")
    kinematic.parent = holder
    return [kinematic, holder]


compile_error("kinematic body under a scaled parent is rejected", kinematic_under_scaled_parent)

# Hide Render Geometry applies to the owner's mesh even on an inherited record.
inherit = mesh_body("InheritHidden", (6, 6, 3), (1, 1, 1), role="STATIC")
record = inherit.ov.physics.colliders[0]
record.schema_opinion, record.hide_from_render = "INHERIT", True
bpy.context.view_layer.update()
result = generation.SceneGenerationOwner(out / "gen-inherit").replace(scene)
stage = Usd.Stage.Open(result.usd_path)
mapping = result.blender_prim_paths[generation.blender_id(inherit, "OBJECT")]
check("inherited record hides render geometry",
      UsdGeom.Imageable(stage.GetPrimAtPath(mapping.schema_path)).ComputeVisibility() == "invisible")
check("inherited record writes no collider override", not stage.GetPrimAtPath(mapping.object_path + "/Colliders"))
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("SINGLE_COMPILER_PASS", len(checks))
