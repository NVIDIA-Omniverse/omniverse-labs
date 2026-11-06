# SPDX-License-Identifier: Apache-2.0
"""Explicit, undoable conversion of the existing SimReady uni-body contract.

The typed OV fields use the shared Blender-unit dimensional contract and wxyz
principal axes, not a retained SimReady metadata source.
The original CHILD_OF constraint is ordinary user data and is never changed.
"""
from pathlib import Path
import tempfile
import uuid
import math

from .physics_colliders import ColliderError, collider_records
from .physics_endpoints import require_writable


def _snapshot(group, fields):
    result = {}
    for field in fields:
        explicit = group.is_property_set(field)
        value = getattr(group, field)
        if hasattr(value, "to_tuple"):
            value = value.to_tuple()
        result[field] = (explicit, value)
        if not explicit and group.is_property_set(field):
            group.property_unset(field)
    return result


def _restore(group, snapshot):
    for field, (explicit, value) in snapshot.items():
        if explicit:
            setattr(group, field, value)
        else:
            group.property_unset(field)


def composed_text(records, source):
    from pxr import Sdf, Usd
    layers = []
    for record in records:
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(record.layer_text)
        layers.append(layer)
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [layer.identifier for layer in layers] + [source.identifier]
    return Usd.Stage.Open(root).Flatten().ExportToString()


def _physics_facts(text):
    """Physically meaningful composed facts, independent of how they are authored.

    The owner compiler authors rigid body frames with compensated children and
    explicit default flags; conversion must preserve behaviour, not USD text.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    cache = UsdGeom.XformCache()
    facts = {}
    for prim in stage.Traverse():
        entry = {}
        imageable = UsdGeom.Imageable(prim)
        if prim.IsA(UsdGeom.Xformable):
            world = cache.GetLocalToWorldTransform(prim)
            if not prim.IsA(UsdGeom.Gprim):
                # Body frames are rigid by design; geometry keeps full world transforms.
                world = world.RemoveScaleShear()
            entry["world"] = tuple(v for row in world for v in row)
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            entry["visibility"] = str(imageable.ComputeVisibility())
        def authored(attr):
            return attr.Get() if attr and attr.HasAuthoredValue() else None

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim)
            entry["rigid"] = (body.GetRigidBodyEnabledAttr().Get() is not False,
                              bool(body.GetKinematicEnabledAttr().Get()),
                              tuple(authored(body.GetVelocityAttr()) or ()),
                              tuple(authored(body.GetAngularVelocityAttr()) or ()),
                              authored(prim.GetAttribute("physxRigidBody:linearDamping")),
                              authored(prim.GetAttribute("physxRigidBody:angularDamping")))
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI(prim)
            entry["physics_material"] = tuple(authored(a) for a in (
                material.GetStaticFrictionAttr(), material.GetDynamicFrictionAttr(),
                material.GetRestitutionAttr(), material.GetDensityAttr()))
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            world = cache.GetLocalToWorldTransform(prim)
            com = authored(mass.GetCenterOfMassAttr())
            axes = authored(mass.GetPrincipalAxesAttr())
            if axes is not None:
                # Compare principal axes in world orientation (body frames may be re-authored).
                rotation = world.RemoveScaleShear().ExtractRotationQuat()
                axes = rotation * axes
                axes = (axes.GetReal(), *axes.GetImaginary()) if axes.GetReal() >= 0 else (-axes.GetReal(), *(-axes.GetImaginary()))
            entry["mass"] = (authored(mass.GetMassAttr()), authored(mass.GetDensityAttr()),
                             tuple(world.Transform(com)) if com is not None else None,
                             tuple(authored(mass.GetDiagonalInertiaAttr()) or ()), tuple(axes) if axes else None)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approximation = (UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                             if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None)
            binding = prim.GetRelationship("material:binding:physics")
            points = ()
            if prim.IsA(UsdGeom.Mesh):
                world = cache.GetLocalToWorldTransform(prim)
                points = tuple(sorted(tuple(round(v, 5) for v in world.Transform(p)) for p in (UsdGeom.Mesh(prim).GetPointsAttr().Get() or ())))
            entry["collision"] = (UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,
                                  approximation, tuple(str(t) for t in binding.GetTargets()) if binding else (), points)
        if entry:
            facts[str(prim.GetPath())] = entry
    return facts


def _close(a, b):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
    return a == b


def equivalent_text(before, after):
    """Composed physics, material and visibility are equivalent within float tolerance."""
    if before == after:
        return True
    old, new = _physics_facts(before), _physics_facts(after)
    if set(old) != set(new):
        return False
    return all(set(old[p]) == set(new[p]) and all(_close(old[p][k], new[p][k]) for k in old[p]) for p in old)


def verify_equivalent(before, after, source):
    if not equivalent_text(composed_text(before, source), composed_text(after, source)):
        raise ColliderError("SimReady conversion changed composed physics/material/visibility; original data restored")


def convert_in_scene(scene, owner, *, verifier=verify_equivalent):
    """Convert the supported single body, comparing the complete private stage.

    Export/mapping/validation precede mutation. Failed post-write validation
    restores explicit-property state as well as values on both datablocks.
    """
    from pxr import Usd
    from . import scene_generation as generation, simready_physics_conversion as legacy
    from .physics_frames import PhysicsUnits
    from . import physics_authoring
    physics_authoring.require_stopped(scene)
    units = PhysicsUnits.from_scene(scene)
    require_writable(owner)
    physics = owner.ov.physics
    initial_keys = (set(physics.keys()), set(physics.rigid_body.keys()))
    if physics.simready_version == 1:
        collider_records(owner)
        return False
    if physics.simready_version != 0:
        raise ColliderError("Unsupported SimReady conversion version")
    unibodies = legacy.read_scene_unibodies(scene)
    if len(unibodies) != 1 or unibodies[0].body != owner:
        raise ColliderError("Select the single supported SimReady Geometry body")
    uni = unibodies[0]
    require_writable(uni.material)
    collider_records(owner)
    # Explicit OV opinions and pre-existing collection edits have no unique
    # merge with legacy SimReady ownership. Do not guess their precedence.
    if physics.colliders or physics.rigid_body.schema_opinion != "INHERIT" or uni.material.ov.physics.schema_opinion != "INHERIT":
        raise ColliderError("SimReady conversion requires unambiguous legacy-only physics authoring")
    constraints = tuple(owner.constraints)
    if len(constraints) != 1 or constraints[0].type != "CHILD_OF" or constraints[0].target != uni.reference:
        raise ColliderError("SimReady conversion requires exactly one reference constraint; joints are never inferred")
    with tempfile.TemporaryDirectory(prefix="ov-simready-conversion-") as folder:
        stock_path = Path(folder) / "stock.usdc"
        generation._stock_export(scene, stock_path)
        mappings = generation._validated_blender_prim_paths(scene, stock_path)
        mapping = mappings.get(generation.blender_id(owner, "OBJECT"))
        if mapping is None:
            raise ColliderError("SimReady body has no unique exported Object mapping")
        before = generation._compile_add_on_opinions(scene, mappings, stock_path)
        material_records = [record for record in before if record.diagnostics.get("blender_property_path") == "simready.physics_material"]
        if len(material_records) != 1:
            raise ColliderError("SimReady physics material has no unique generated mapping")
        body = physics.rigid_body
        body_scale = generation._stock_body_scale(stock_path, mapping.object_path)
        material = uni.material.ov.physics
        body_state = _snapshot(body, ("persistent_id", "schema_opinion", "mass_kg", "center_of_mass", "diagonal_inertia", "principal_axes"))
        owner_state = _snapshot(physics, ("persistent_id", "simready_version"))
        material_state = _snapshot(material, ("schema_opinion", "density", "static_friction", "dynamic_friction", "restitution", "preserved_prim_path"))
        try:
            physics.persistent_id = physics.persistent_id or uuid.uuid4().hex
            body.persistent_id = body.persistent_id or uuid.uuid4().hex
            body.schema_opinion = "APPLY"
            body.mass_kg = uni.mass
            body.center_of_mass = tuple(units.length_from_meters(v * body_scale) for v in uni.center_of_mass)
            body.diagonal_inertia = tuple(v / units.inertia(1.0) for v in uni.diagonal_inertia)
            body.principal_axes = uni.principal_axes
            collider = physics.colliders.add()
            collider.name = collider.persistent_id = uuid.uuid4().hex
            collider.schema_opinion = "APPLY"
            collider.shape = "SIMREADY_CONVEX_DECOMPOSITION"
            collider.legacy_target_path = mapping.schema_path
            collider.physics_material = uni.material
            material.schema_opinion = "APPLY"
            material.density = uni.density / units.density(1.0)
            material.static_friction = uni.static_friction
            material.dynamic_friction = uni.dynamic_friction
            material.restitution = uni.restitution
            material.preserved_prim_path = material_records[0].usd_prim_path
            physics.simready_version = 1
            after = generation._compile_add_on_opinions(scene, mappings, stock_path)
            source_stage = Usd.Stage.Open(str(stock_path))
            verifier(before, after, source_stage.GetRootLayer())
        except Exception:
            physics.colliders.clear()
            _restore(body, body_state)
            _restore(material, material_state)
            _restore(physics, owner_state)
            # Null nested RNA pointers may materialize during compile reads.
            for group, keys in ((body, initial_keys[1]), (physics, initial_keys[0])):
                for field in set(group.keys()) - keys:
                    group.property_unset(field)
            raise
    physics_authoring.mark_dirty(scene, owner)
    return True


try:
    import bpy
except ModuleNotFoundError:
    bpy = None

if bpy is not None:
    class OVPHYSICS_OT_convert_simready(bpy.types.Operator):
        bl_idname = "ov.physics_convert_simready"
        bl_label = "Convert SimReady Body to OV Physics"
        bl_description = "Explicitly convert the supported single-body layout, preserving its reference constraint and composed physics"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context):
            return context.object is not None and context.object.type == "MESH"

        def execute(self, context):
            try:
                convert_in_scene(context.scene, context.object)
            except Exception as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            return {"FINISHED"}


def register():
    if bpy is not None and not getattr(OVPHYSICS_OT_convert_simready, "is_registered", False):
        bpy.utils.register_class(OVPHYSICS_OT_convert_simready)


def unregister():
    if bpy is not None and getattr(OVPHYSICS_OT_convert_simready, "is_registered", False):
        bpy.utils.unregister_class(OVPHYSICS_OT_convert_simready)
