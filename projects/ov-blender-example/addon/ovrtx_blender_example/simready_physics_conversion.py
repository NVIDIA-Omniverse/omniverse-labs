# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimReady 2026.4.0 uni-body physics opinions for scene generations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from . import authoring_properties, usd_paths
from .scene_generation import BlenderId, BlenderPrimPath, _session_uid
from .usd_opinion_records import AddOnUsdOpinionRecord, SceneGenerationError


OBJECT_MASS_KEYS = (
    "pxr:usd:physics_mass",
    "pxr:usd:physics_centerofmass",
    "pxr:usd:physics_inertia",
    "pxr:physics:principalAxes",
)
MATERIAL_PHYSICS_KEYS = (
    "pxr:usd:physics_density",
    "pxr:usd:physics_dynamicFriction",
    "pxr:usd:physics_staticFriction",
    "pxr:usd:physics_restitution",
)
IGNORED_MATERIAL_KEYS = (
    "pxr:usd:physics_type",
    "pxr:usd:physics_fillRatio",
    "pxr:usd:physics_thickness",
)


@dataclass(frozen=True)
class SimReadyUniBody:
    body: Any
    reference: Any
    material: Any
    mass: float
    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]
    principal_axes: tuple[float, float, float, float]
    density: float
    dynamic_friction: float
    static_friction: float
    restitution: float


def read_scene_unibodies(scene: Any) -> tuple[SimReadyUniBody, ...]:
    """Return supported SimReady uni-body authoring from Blender data."""

    export = _child_collection(_root_collection(scene), "Export")
    if export is None:
        return ()
    geometry = _child_collection(export, "Geometry")
    references = _child_collection(export, "ReferencePrims")
    colliders = _child_collection(export, "Colliders")
    if geometry is None and references is None and colliders is None:
        return ()
    if geometry is None or references is None or colliders is None:
        raise _error(None, "simready_collections_incomplete")
    collider_meshes = _mesh_objects(colliders)
    if collider_meshes:
        raise _error(collider_meshes[0], "separate_simready_colliders_unsupported")
    bodies = _mesh_objects(geometry)
    if len(bodies) != 1:
        raise _error(None, "simready_unibody_requires_one_geometry_mesh")
    reference_objects = _objects(references)
    if len(reference_objects) != 1:
        raise _error(bodies[0], "simready_unibody_requires_one_reference_prim")
    body = bodies[0]
    reference = reference_objects[0]
    constraints = [
        constraint
        for constraint in getattr(body, "constraints", ())
        if str(getattr(constraint, "type", "")) == "CHILD_OF"
        and getattr(constraint, "target", None) is reference
    ]
    if len(constraints) != 1:
        raise _error(body, "simready_unibody_requires_one_child_of_reference")
    unsupported = [
        constraint
        for constraint in getattr(body, "constraints", ())
        if str(getattr(constraint, "type", "")) != "CHILD_OF"
    ]
    if unsupported:
        raise _error(body, "simready_joint_constraints_unsupported")
    materials = _simready_materials(body)
    if len(materials) != 1:
        raise _error(body, "simready_unibody_requires_one_physics_material")
    material = materials[0]
    mass = _required_float(body, "pxr:usd:physics_mass")
    if mass <= 0.0:
        raise _error(body, "simready_mass_must_be_positive")
    return (
        SimReadyUniBody(
            body=body,
            reference=reference,
            material=material,
            mass=mass,
            center_of_mass=_required_float_tuple(
                body, "pxr:usd:physics_centerofmass", 3
            ),
            diagonal_inertia=_required_float_tuple(
                body, "pxr:usd:physics_inertia", 3
            ),
            principal_axes=_required_float_tuple(
                body, "pxr:physics:principalAxes", 4
            ),
            density=_required_float(material, "pxr:usd:physics_density"),
            dynamic_friction=_required_float(
                material, "pxr:usd:physics_dynamicFriction"
            ),
            static_friction=_required_float(
                material, "pxr:usd:physics_staticFriction"
            ),
            restitution=_required_float(material, "pxr:usd:physics_restitution"),
        ),
    )


def explicit_materials(
    unibodies: tuple[SimReadyUniBody, ...],
) -> tuple[Any, ...]:
    return tuple(
        sorted(
            {
                unibody.material
                for unibody in unibodies
                if _schema_opinion(unibody.material) != authoring_properties.INHERIT
            },
            key=lambda material: str(getattr(material, "name_full", "")),
        )
    )


def convert_scene_unibodies(
    unibodies: tuple[SimReadyUniBody, ...],
    mappings: Mapping[BlenderId, BlenderPrimPath],
    occupied_paths: set[str],
    explicit_material_paths: Mapping[int, str],
) -> tuple[AddOnUsdOpinionRecord, ...]:
    records: list[AddOnUsdOpinionRecord] = []
    simready_material_paths: dict[int, str] = {}
    for unibody in unibodies:
        if (
            _schema_opinion(unibody.body, "rigid_body", "collision")
            != authoring_properties.INHERIT
        ):
            continue
        body_uid = _session_uid(unibody.body, "OBJECT")
        mapping = mappings.get(BlenderId("OBJECT", body_uid))
        if mapping is None:
            raise _error(unibody.body, "simready_body_has_no_mapped_prim")
        material_uid = _session_uid(unibody.material, "MATERIAL")
        material_path = explicit_material_paths.get(material_uid)
        if material_path is None:
            material_path = simready_material_paths.get(material_uid)
        if material_path is None:
            material_path = usd_paths.reserve_unique_child_path(
                "/World/PhysicsMaterials",
                "physics_mat_"
                + str(getattr(unibody.material, "name_full", "Material")),
                occupied_paths,
            )
            simready_material_paths[material_uid] = material_path
            records.append(_convert_material(unibody, material_path))
        records.append(_convert_body(unibody, mapping, material_path))
    return tuple(records)


def fingerprint_for_object(obj: Any) -> tuple[Any, ...]:
    """Return SimReady authoring that affects generated physics opinions."""

    if str(getattr(obj, "type", "")) != "MESH":
        return ()
    materials = tuple(
        _material_fingerprint(material) for material in _slot_materials(obj)
    )
    constraints = tuple(
        sorted(
            (
                str(getattr(constraint, "type", "")),
                _session_uid(getattr(constraint, "target", None), "OBJECT")
                if getattr(constraint, "target", None) is not None
                else 0,
            )
            for constraint in getattr(obj, "constraints", ())
        )
    )
    object_values = tuple(
        (key, _property_value(obj, key))
        for key in OBJECT_MASS_KEYS
        if _has_property(obj, key)
    )
    collections = tuple(
        sorted(
            str(getattr(collection, "name", ""))
            for collection in getattr(obj, "users_collection", ())
        )
    )
    return (object_values, materials, constraints, collections)


def _convert_body(
    unibody: SimReadyUniBody,
    mapping: BlenderPrimPath,
    material_path: str,
) -> AddOnUsdOpinionRecord:
    Gf, Sdf, Usd, UsdPhysics, UsdShade = _pxr_modules()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    _override_ancestors(stage, Sdf.Path(mapping.object_path).GetParentPath())
    root_prim = stage.OverridePrim(mapping.object_path)
    mesh_prim = stage.OverridePrim(mapping.schema_path)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr().Set(float(unibody.mass))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*unibody.center_of_mass))
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*unibody.diagonal_inertia))
    mass.CreatePrincipalAxesAttr().Set(
        Gf.Quatf(
            unibody.principal_axes[0],
            Gf.Vec3f(*unibody.principal_axes[1:]),
        )
    )
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set(
        "convexDecomposition"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    mesh_prim.CreateRelationship("material:binding:physics").SetTargets(
        (Sdf.Path(material_path),)
    )
    return AddOnUsdOpinionRecord(
        mapping.object_path,
        layer,
        _diagnostics(unibody, mapping.object_path, "simready.unibody"),
    )


def _convert_material(
    unibody: SimReadyUniBody,
    material_path: str,
) -> AddOnUsdOpinionRecord:
    _Gf, Sdf, Usd, UsdPhysics, UsdShade = _pxr_modules()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    _override_ancestors(stage, Sdf.Path(material_path).GetParentPath())
    prim = UsdShade.Material.Define(stage, material_path).GetPrim()
    material = UsdPhysics.MaterialAPI.Apply(prim)
    material.CreateDensityAttr().Set(float(unibody.density))
    material.CreateDynamicFrictionAttr().Set(float(unibody.dynamic_friction))
    material.CreateStaticFrictionAttr().Set(float(unibody.static_friction))
    material.CreateRestitutionAttr().Set(float(unibody.restitution))
    return AddOnUsdOpinionRecord(
        material_path,
        layer,
        _diagnostics(unibody, material_path, "simready.physics_material"),
    )


def _pxr_modules() -> tuple[Any, Any, Any, Any, Any]:
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade  # type: ignore

    return Gf, Sdf, Usd, UsdPhysics, UsdShade


def _root_collection(scene: Any) -> Any:
    return getattr(scene, "collection", None)


def _child_collection(collection: Any, name: str) -> Any:
    if collection is None:
        return None
    children = getattr(collection, "children", ())
    get = getattr(children, "get", None)
    if get is not None:
        found = get(name)
        if found is not None:
            return found
    for child in children:
        if str(getattr(child, "name", "")) == name:
            return child
    return None


def _objects(collection: Any) -> tuple[Any, ...]:
    return tuple(getattr(collection, "objects", ())) if collection is not None else ()


def _mesh_objects(collection: Any) -> tuple[Any, ...]:
    return tuple(
        obj for obj in _objects(collection) if str(getattr(obj, "type", "")) == "MESH"
    )


def _simready_materials(obj: Any) -> tuple[Any, ...]:
    materials = []
    seen = set()
    for material in _slot_materials(obj):
        if material is None or id(material) in seen:
            continue
        seen.add(id(material))
        if any(_has_property(material, key) for key in MATERIAL_PHYSICS_KEYS):
            materials.append(material)
    return tuple(materials)


def _slot_materials(obj: Any) -> tuple[Any, ...]:
    slots = getattr(obj, "material_slots", None)
    if slots is not None:
        return tuple(getattr(slot, "material", None) for slot in slots)
    return tuple(getattr(getattr(obj, "data", None), "materials", ()))


def _schema_opinion(owner: Any, *groups: str) -> str:
    physics = getattr(getattr(owner, "ov", None), "physics", None)
    if not groups:
        return str(
            getattr(physics, "schema_opinion", authoring_properties.INHERIT)
        )
    for group in groups:
        opinion = str(
            getattr(
                getattr(physics, group, None),
                "schema_opinion",
                authoring_properties.INHERIT,
            )
        )
        if opinion != authoring_properties.INHERIT:
            return opinion
    return authoring_properties.INHERIT


def _required_float(owner: Any, key: str) -> float:
    if not _has_property(owner, key):
        raise _error(owner, f"missing_{key}")
    value = float(_property_value(owner, key))
    if not math.isfinite(value):
        raise _error(owner, f"invalid_{key}")
    return value


def _required_float_tuple(owner: Any, key: str, size: int) -> tuple[float, ...]:
    if not _has_property(owner, key):
        raise _error(owner, f"missing_{key}")
    try:
        values = tuple(float(value) for value in _property_value(owner, key))
    except TypeError as exc:
        raise _error(owner, f"invalid_{key}") from exc
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise _error(owner, f"invalid_{key}")
    return values


def _has_property(owner: Any, key: str) -> bool:
    try:
        return key in owner
    except TypeError:
        return getattr(owner, key, None) is not None


def _property_value(owner: Any, key: str) -> Any:
    get = getattr(owner, "get", None)
    return get(key) if get is not None else getattr(owner, key)


def _material_fingerprint(material: Any) -> tuple[Any, ...]:
    if material is None:
        return ()
    return (
        _session_uid(material, "MATERIAL"),
        tuple(
            (key, _property_value(material, key))
            for key in (*MATERIAL_PHYSICS_KEYS, *IGNORED_MATERIAL_KEYS)
            if _has_property(material, key)
        ),
        _schema_opinion(material),
    )


def _override_ancestors(stage: Any, parent_path: Any) -> None:
    paths = []
    current = parent_path
    while current and not current.IsAbsoluteRootPath():
        paths.append(current)
        current = current.GetParentPath()
    for path in reversed(paths):
        stage.OverridePrim(path)


def _diagnostics(
    unibody: SimReadyUniBody,
    root: str,
    property_path: str,
) -> dict[str, Any]:
    return {
        "blender_owner": str(
            getattr(unibody.body, "name", getattr(unibody.body, "name_full", ""))
        ),
        "blender_property_path": property_path,
        "intended_usd_prim_path": root,
        "source": "SimReady 2026.4.0 uni-body",
        "body": str(getattr(unibody.body, "name_full", "")),
        "reference_prim": str(getattr(unibody.reference, "name_full", "")),
        "physics_material": str(getattr(unibody.material, "name_full", "")),
    }


def _error(owner: Any, reason: str) -> SceneGenerationError:
    name = (
        ""
        if owner is None
        else str(getattr(owner, "name_full", getattr(owner, "name", "")))
    )
    return SceneGenerationError(
        "SimReady uni-body physics cannot be converted",
        (
            {
                "blender_owner": name,
                "blender_property_path": "simready.unibody",
                "reason": reason,
                "source": "SimReady 2026.4.0 uni-body",
            },
        ),
    )


__all__ = [
    "MATERIAL_PHYSICS_KEYS",
    "OBJECT_MASS_KEYS",
    "SimReadyUniBody",
    "convert_scene_unibodies",
    "explicit_materials",
    "fingerprint_for_object",
    "read_scene_unibodies",
]
