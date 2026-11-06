# SPDX-License-Identifier: Apache-2.0
"""Complete, independent owner/collider compilation into private USD opinions."""
from dataclasses import dataclass
import math

from .physics_colliders import collider_records, ColliderError
from .physics_endpoints import _id, physics_namespace, require_writable
from .physics_frames import PhysicsUnits, RigidFrame, evaluated_owner_transform
from .usd_opinion_records import AddOnUsdOpinionRecord


@dataclass(frozen=True)
class PhysicsOwner:
    object: object
    target: object
    bone_name: str | None
    body_identity: str
    body_path: str
    render_mapping: object = None


def authoring_owners(scene):
    """All physics property owners, including non-renderable objects and bones."""
    for obj in scene.objects:
        yield obj, obj, None
        if obj.type == "ARMATURE" and obj.pose is not None:
            for bone in obj.pose.bones:
                yield obj, bone, bone.name


def collider_material(record):
    # Reading a null RNA ID pointer can itself materialize an explicit value.
    if hasattr(record, "is_property_set") and not record.is_property_set("physics_material"):
        return None
    return getattr(record, "physics_material", None)


def reserve_generated_subtree(stock_stage, root):
    """A generated container owns its entire subtree, including inactive source."""
    if any(str(prim.GetPath()) == root or str(prim.GetPath()).startswith(root + "/")
           for prim in stock_stage.TraverseAll()):
        raise ColliderError(f"Generated physics subtree {root} overlaps stock/source USD")


def _mapping(mappings, obj):
    from .scene_generation import blender_id
    return mappings.get(blender_id(obj, "OBJECT"))


def gather_physics_owners(scene, mappings, stock_stage):
    result, used = [], set()
    root = None
    for obj, target, bone_name in authoring_owners(scene):
        records = collider_records(target)
        physics = target.ov.physics
        body = physics.rigid_body
        if body.schema_opinion == "INHERIT" and not any(c.schema_opinion != "INHERIT" or c.hide_from_render for c in records):
            continue
        require_writable(obj)
        if bone_name:
            require_writable(obj.data)
        if root is None:
            root = physics_namespace(scene, (str(p.GetPath()) for p in stock_stage.TraverseAll()))
        identity = _id(physics.persistent_id if bone_name is None else obj.ov.physics.persistent_id)
        if sum(candidate.ov.physics.persistent_id == identity for candidate in scene.objects) != 1:
            raise ColliderError("Duplicate Object identity; explicitly repair duplicates")
        if bone_name:
            bone_id = _id(obj.data.bones[bone_name].ov.persistent_id)
            if sum(bone.ov.persistent_id == bone_id for bone in obj.data.bones) != 1:
                raise ColliderError("Duplicate bone definition identity; explicitly repair duplicates")
            identity += "_" + bone_id
        if identity in used:
            raise ColliderError("Duplicate physics owner identity; explicitly repair duplicates")
        used.add(identity)
        mapping = _mapping(mappings, obj) if bone_name is None else None
        path = mapping.object_path if mapping is not None else root + "/Bodies/b_" + identity
        result.append(PhysicsOwner(obj, target, bone_name, identity, path, mapping))
    return tuple(result)


def _owner_frame(owner, depsgraph, units):
    settings = owner.target.ov.physics.rigid_body
    offset = RigidFrame(tuple(settings.offset.translation), tuple(settings.offset.rotation))
    return evaluated_owner_transform(owner.object, depsgraph, bone_name=owner.bone_name, offset=offset, units=units)


def _set_parent_relative_frame(prim, frame, stock, body_path, body_frames):
    """Author a rigid world frame relative to the body's composed stock parent.

    The runtime composes parent transforms and does not honour
    !resetXformStack!, so the local op is world * parent_world^-1.
    """
    from pxr import Gf, Sdf, UsdGeom
    world = Gf.Matrix4d(*[v for row in frame.gf_matrix() for v in row])
    parent_path = str(Sdf.Path(body_path).GetParentPath())
    if parent_path in body_frames:
        # A body under another body: that body's composed frame is its rigid frame.
        parent_world = body_frames[parent_path]
    else:
        # Non-body parents keep their stock world (bodies compensate their children).
        parent = stock.GetPrimAtPath(parent_path)
        parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent) if parent else Gf.Matrix4d(1)
    operation = UsdGeom.Xformable(prim).AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ovPhysics")
    operation.Set(world * parent_world.GetInverse())
    UsdGeom.Xformable(prim).SetXformOpOrder([operation], False)
    return parent_world


def _set_frame(prim, frame, *, reset=False, scale=1.0):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    matrix = Gf.Matrix4d(*[v for row in frame.gf_matrix() for v in row])
    if scale != 1:
        matrix = Gf.Matrix4d().SetScale(scale) * matrix
    operation = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ovPhysics")
    operation.Set(matrix)
    xform.SetXformOpOrder([operation], reset)


def compile_physics_owners(scene, mappings, stock_path, material_paths):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from .scene_generation import _session_uid
    stock = Usd.Stage.Open(str(stock_path))
    owners = gather_physics_owners(scene, mappings, stock)
    units = PhysicsUnits.from_scene(scene)
    try:
        import bpy
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except ModuleNotFoundError:
        depsgraph = None
    body_frames = {
        owner.body_path: Gf.Matrix4d(*[v for row in _owner_frame(owner, depsgraph, units).frame.gf_matrix() for v in row])
        for owner in owners
    }
    result = []
    for owner in owners:
        obj, target = owner.object, owner.target
        settings = target.ov.physics.rigid_body
        decomposition = [r for r in collider_records(target) if r.shape == "SIMREADY_CONVEX_DECOMPOSITION"]
        if decomposition and (len(collider_records(target)) != 1 or owner.render_mapping is None
                              or decomposition[0].legacy_target_path != owner.render_mapping.schema_path):
            raise ColliderError("Converted SimReady decomposition requires its preserved single mesh mapping; compound/offset/role edits are not validated")
        offset = RigidFrame(tuple(settings.offset.translation), tuple(settings.offset.rotation))
        # Body frames are rigid; nonuniform owner scale is baked into collider
        # geometry and restored per axis when poses are presented to Blender.
        transform = evaluated_owner_transform(obj, depsgraph, bone_name=owner.bone_name, offset=offset, units=units)
        scale = transform.uniform_scale * units.meters_per_blender_unit
        axis_scale = tuple(v * units.meters_per_blender_unit for v in transform.axis_scale)
        nonuniform = max(axis_scale) - min(axis_scale) > 1e-6 * max(axis_scale)
        simulated = settings.schema_opinion == "APPLY" and settings.role != "STATIC"
        if simulated and owner.bone_name is None and obj.parent is not None:
            from .physics_frames import FrameError, matrix_to_transform
            try:
                matrix_to_transform(obj.parent.matrix_world)
            except FrameError as exc:
                # A rotating child of a nonuniformly scaled parent would shear.
                raise ColliderError(f"{obj.name}: a moving body needs a uniformly scaled parent chain ({exc})") from exc
        if nonuniform and (any(abs(v) > 1e-6 for v in offset.rotation_wxyz[1:])
                           or any(any(abs(v) > 1e-6 for v in tuple(r.offset.rotation)[1:]) for r in collider_records(target))):
            raise ColliderError(f"{obj.name}: rotated body or collider offsets on a nonuniformly scaled Object would shear")
        layer = Sdf.Layer.CreateAnonymous(".usda")
        stage = Usd.Stage.Open(layer)
        body = stage.OverridePrim(owner.body_path) if owner.render_mapping else UsdGeom.Xform.Define(stage, owner.body_path).GetPrim()
        # Generated and mapped body frames are rigid and meter-valued in world space.
        parent_world = _set_parent_relative_frame(body, transform.frame, stock, owner.body_path, body_frames)
        if simulated and settings.role == "KINEMATIC" and any(
                abs(Gf.Vec3d(*parent_world.GetRow3(i)).GetLength() - 1) > 1e-6 for i in range(3)):
            # Kinematic poses are written in parent space and cannot carry scale.
            raise ColliderError(f"{obj.name}: a kinematic body needs an unscaled parent chain")
        if settings.schema_opinion == "APPLY" and settings.role != "STATIC":
            rigid = UsdPhysics.RigidBodyAPI.Apply(body)
            rigid.CreateRigidBodyEnabledAttr().Set(settings.enabled)
            rigid.CreateKinematicEnabledAttr().Set(settings.role == "KINEMATIC")
            from .physics_body_controls import author_body_controls
            author_body_controls(body, settings, units)
        elif settings.schema_opinion == "REMOVE" or settings.role == "STATIC":
            body.RemoveAPI(UsdPhysics.RigidBodyAPI)
            body.RemoveAPI(UsdPhysics.MassAPI)
        if owner.render_mapping:
            # Preserve every stock child world transform when changing its
            # parent's pivot and removing inherited scene/object scale.
            stock_body = stock.GetPrimAtPath(owner.body_path)
            # Offset translation scales with owner scale as in evaluated frame.
            scaled_offset = RigidFrame(tuple(v * s for v, s in zip(offset.position, axis_scale)), offset.rotation_wxyz)
            compensation = Gf.Matrix4d().SetScale(Gf.Vec3d(*axis_scale)) * Gf.Matrix4d(*[v for row in scaled_offset.inverse().gf_matrix() for v in row])
            for child in stock_body.GetChildren():
                # A child that is itself a body authors its own parent-relative frame.
                if child.IsA(UsdGeom.Xformable) and str(child.GetPath()) not in body_frames:
                    prim = stage.OverridePrim(child.GetPath())
                    child_xform = UsdGeom.Xformable(child)
                    matrix = child_xform.GetLocalTransformation() * compensation
                    xform = UsdGeom.Xformable(prim)
                    op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ovPhysicsVisual")
                    op.Set(matrix)
                    xform.SetXformOpOrder([op], child_xform.GetResetXformStack())
        records = collider_records(target)
        if owner.render_mapping is not None and any(record.hide_from_render for record in records):
            # "Hide Render Geometry" hides the owner's own render mesh.
            UsdGeom.Imageable(stage.OverridePrim(owner.render_mapping.schema_path)).CreateVisibilityAttr().Set("invisible")
        collider_ids = set()
        for record in records:
            if record.schema_opinion == "INHERIT":
                continue
            identity = _id(record.persistent_id)
            if identity in collider_ids:
                raise ColliderError("Duplicate collider identity")
            collider_ids.add(identity)
            # A preserved target is the SimReady-converted single render mesh.
            legacy_path = getattr(record, "legacy_target_path", "")
            if legacy_path:
                if owner.render_mapping is None or legacy_path != owner.render_mapping.schema_path:
                    raise ColliderError("Preserved collider target is not owned by its mapped Object")
                if record.shape not in {"CONVEX_HULL", "SIMREADY_CONVEX_DECOMPOSITION"} or record.schema_opinion == "REMOVE":
                    # A shape switch or removal suppresses collision on the mesh itself.
                    old = stage.OverridePrim(legacy_path)
                    old.RemoveAPI(UsdPhysics.CollisionAPI)
                    old.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                    old.CreateRelationship("material:binding:physics").SetTargets([])
                    legacy_path = ""
            if record.schema_opinion == "REMOVE":
                continue
            path = legacy_path or owner.body_path + "/Colliders/c_" + identity
            if not legacy_path:
                reserve_generated_subtree(stock, owner.body_path + "/Colliders")
            shape_scale = 1.0
            if record.shape == "BOX":
                from .physics_body_controls import positive
                dimensions = tuple(positive(v * s, "Box dimension") for v, s in zip(record.dimensions, axis_scale))
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr().Set(1.)
                cube.CreateExtentAttr().Set([Gf.Vec3f(-.5), Gf.Vec3f(.5)])
                prim, shape_scale = cube.GetPrim(), Gf.Vec3d(*dimensions)
            elif record.shape == "CAPSULE":
                from .physics_body_controls import positive
                radius = positive(record.radius_m, "Capsule radius")
                height = positive(record.height_m, "Capsule cylinder height", zero=True)
                capsule = UsdGeom.Capsule.Define(stage, path)
                capsule.CreateRadiusAttr().Set(radius)
                capsule.CreateHeightAttr().Set(height)
                capsule.CreateAxisAttr().Set("Z")
                extent = Gf.Vec3f(radius, radius, height / 2 + radius)
                capsule.CreateExtentAttr().Set([-extent, extent])
                prim = capsule.GetPrim()
            elif record.shape == "ANALYTIC_SPHERE":
                radius = float(record.radius_m)
                if radius <= 0 or not math.isfinite(radius):
                    raise ColliderError("Sphere radius must be finite and positive meters")
                sphere = UsdGeom.Sphere.Define(stage, path)
                sphere.CreateRadiusAttr().Set(radius)
                sphere.CreateExtentAttr().Set([Gf.Vec3f(-radius), Gf.Vec3f(radius)])
                prim = sphere.GetPrim()
            elif record.shape in {"CONVEX_HULL", "SIMREADY_CONVEX_DECOMPOSITION"}:
                if legacy_path:
                    # Preserved single-mesh target (SimReady conversion): collide on the mesh itself.
                    prim = stage.OverridePrim(path)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                        "convexDecomposition" if record.shape == "SIMREADY_CONVEX_DECOMPOSITION" else "convexHull")
                    collision = UsdPhysics.CollisionAPI.Apply(prim)
                    collision.CreateCollisionEnabledAttr().Set(record.enabled)
                    material = collider_material(record)
                    material_path = material_paths.get(_session_uid(material, "MATERIAL"), "") if material else ""
                    if material_path:
                        UsdShade.MaterialBindingAPI.Apply(prim)
                        prim.CreateRelationship("material:binding:physics").SetTargets([Sdf.Path(material_path)])
                    continue
                source_obj = (record.mesh_source if hasattr(record, "mesh_source") and record.is_property_set("mesh_source") else None) or obj
                source_mapping = _mapping(mappings, source_obj)
                if source_obj.type != "MESH" or source_mapping is None:
                    raise ColliderError("Hull collider needs a mapped evaluated Mesh source")
                source = stock.GetPrimAtPath(source_mapping.schema_path)
                if not source.IsA(UsdGeom.Mesh):
                    raise ColliderError("Hull source mapping is not a USD Mesh")
                prim = UsdGeom.Mesh.Define(stage, path).GetPrim()
                source_scale = (axis_scale if source_obj == obj else tuple(
                    v * units.meters_per_blender_unit for v in evaluated_owner_transform(
                        source_obj, depsgraph, units=units, allow_nonuniform=True).axis_scale))
                for attr in source.GetAttributes():
                    name = attr.GetName()
                    if name in {"points", "extent", "faceVertexCounts", "faceVertexIndices", "subdivisionScheme", "orientation", "holeIndices"}:
                        value = attr.Get()
                        if value is not None:
                            if name in {"points", "extent"}:
                                value = [Gf.Vec3f(*(float(c) * k for c, k in zip(p, source_scale))) for p in value]
                            prim.CreateAttribute(name, attr.GetTypeName()).Set(value)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            else:
                raise ColliderError("Unsupported collider shape")
            local = (RigidFrame(tuple(v * s for v, s in zip(record.offset.translation, axis_scale)), tuple(record.offset.rotation))
                     if hasattr(record, "offset") else RigidFrame())
            _set_frame(prim, local, scale=shape_scale)
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr().Set(getattr(record, "enabled", True))
            # Generated collision geometry is never a second rendered mesh.
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
            material = collider_material(record)
            if material:
                material_path = material_paths.get(_session_uid(material, "MATERIAL"), "")
                if not material_path:
                    raise ColliderError("Collider material has no compiled physics material; apply a material schema explicitly")
                UsdShade.MaterialBindingAPI.Apply(prim)
                prim.CreateRelationship("material:binding:physics").SetTargets([Sdf.Path(material_path)])
        offset_meters = RigidFrame(tuple(v * s for v, s in zip(offset.position, axis_scale)), offset.rotation_wxyz)
        result.append(AddOnUsdOpinionRecord(owner.body_path, layer, {
            "physics_owner": owner.body_identity,
            "presentation_object_uid": int(obj.session_uid),
            "presentation_object_id": str(obj.ov.physics.persistent_id),
            "presentation_bone_id": str(obj.data.bones[owner.bone_name].ov.persistent_id) if owner.bone_name else "",
            "presentation_bone_name": owner.bone_name or "",
            "body_role": settings.role,
            "body_enabled": bool(settings.enabled and settings.schema_opinion == "APPLY"),
            "meters_per_blender_unit": units.meters_per_blender_unit,
            "owner_uniform_scale": transform.uniform_scale,
            "owner_axis_scale": tuple(transform.axis_scale),
            "body_offset_meters": (offset_meters.position, offset_meters.rotation_wxyz),
            "initial_body_frame": (transform.frame.position, transform.frame.rotation_wxyz),
        }))
    return tuple(result)
