# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse Blender physics opinions for private scene generations.

The scene-generation track produces :class:`AddOnUsdOpinionRecord` opinions.
"""

from __future__ import annotations

from typing import Any

from . import authoring_properties
from .usd_opinion_records import AddOnUsdOpinionRecord, SceneGenerationError


def _pxr_modules() -> tuple[Any, Any, Any, Any, Any]:
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade  # type: ignore

    return Gf, Sdf, Usd, UsdPhysics, UsdShade


def object_has_physics_opinion(obj: Any) -> bool:
    return any(
        value != authoring_properties.INHERIT
        for value in (
            obj.ov.physics.rigid_body.schema_opinion,
            obj.ov.physics.collision.schema_opinion,
        )
    )


def convert_generation_physics_scene(scene: Any) -> AddOnUsdOpinionRecord | None:
    settings = scene.ov.physics_scene
    if settings.schema_opinion == authoring_properties.INHERIT:
        return None
    root = str(scene.ov.usd.physics_scene_path or "")
    if settings.schema_opinion != authoring_properties.APPLY or not root:
        raise _conversion_error(
            scene,
            root,
            "ov.physics_scene.schema_opinion",
            "physics_scene_requires_apply_and_absolute_path",
        )
    Gf, Sdf, Usd, UsdPhysics, _UsdShade = _pxr_modules()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    _override_ancestors(stage, Sdf.Path(root).GetParentPath())
    physics_scene = UsdPhysics.Scene.Define(stage, root)
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*settings.gravity_direction))
    physics_scene.CreateGravityMagnitudeAttr().Set(float(settings.gravity_magnitude))
    return AddOnUsdOpinionRecord(
        root,
        layer,
        _diagnostics(scene, root, "ov.physics_scene"),
    )


def convert_mapped_object_physics(
    obj: Any,
    mapping: Any,
    physics_material_path: str = "",
) -> AddOnUsdOpinionRecord:
    """Compile sparse physics opinions for one stock-exported Blender object."""

    Gf, Sdf, Usd, UsdPhysics, UsdShade = _pxr_modules()
    from pxr import UsdGeom  # type: ignore

    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    _override_ancestors(stage, Sdf.Path(mapping.object_path).GetParentPath())
    root_prim = stage.OverridePrim(mapping.object_path)
    mesh_prim = stage.OverridePrim(mapping.schema_path)
    rigid_opinion = obj.ov.physics.rigid_body.schema_opinion
    rigid_settings = obj.ov.physics.rigid_body
    collision_settings = obj.ov.physics.collision
    collision_opinion = collision_settings.schema_opinion
    _apply_api_opinion(root_prim, UsdPhysics.RigidBodyAPI, rigid_opinion)
    if rigid_opinion == authoring_properties.APPLY and _is_set(rigid_settings, "mass_kg"):
        mass = float(rigid_settings.mass_kg)
        if mass <= 0.0:
            raise _conversion_error(
                obj,
                mapping.object_path,
                "ov.physics.rigid_body.mass_kg",
                "rigid_body_mass_must_be_positive",
            )
        UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr().Set(mass)

    collider_prim = mesh_prim
    if (
        collision_opinion == authoring_properties.APPLY
        and collision_settings.shape == "ANALYTIC_SPHERE"
    ):
        radius = float(collision_settings.radius_m)
        if radius <= 0.0:
            raise _conversion_error(
                obj,
                mapping.object_path,
                "ov.physics.collision.radius_m",
                "analytic_sphere_radius_must_be_positive",
            )
        collider = UsdGeom.Sphere.Define(stage, mapping.object_path + "/Collider")
        collider.CreateRadiusAttr().Set(radius)
        collider.CreateExtentAttr().Set(
            [Gf.Vec3f(-radius), Gf.Vec3f(radius)]
        )
        collider_prim = collider.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)
        UsdGeom.Imageable(collider_prim).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible
        )
    else:
        _apply_api_opinion(mesh_prim, UsdPhysics.CollisionAPI, collision_opinion)
        if collision_opinion == authoring_properties.APPLY:
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set(
                "convexHull"
            )
        elif collision_opinion == authoring_properties.REMOVE:
            mesh_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if collision_settings.hide_from_render:
            UsdGeom.Imageable(mesh_prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            )
    if physics_material_path and collision_opinion == authoring_properties.APPLY:
        UsdShade.MaterialBindingAPI.Apply(collider_prim)
        collider_prim.CreateRelationship("material:binding:physics").SetTargets(
            (Sdf.Path(physics_material_path),)
        )
    return AddOnUsdOpinionRecord(
        mapping.object_path,
        layer,
        _diagnostics(obj, mapping.object_path, "ov.physics"),
    )


def convert_generation_physics_material(
    material: Any,
    usd_prim_path: str,
    *,
    mapped: bool,
) -> AddOnUsdOpinionRecord:
    """Compile one mapped override or complete add-on-only physics material."""

    settings = material.ov.physics
    _Gf, Sdf, Usd, UsdPhysics, UsdShade = _pxr_modules()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    _override_ancestors(stage, Sdf.Path(usd_prim_path).GetParentPath())
    prim = (
        stage.OverridePrim(usd_prim_path)
        if mapped
        else UsdShade.Material.Define(stage, usd_prim_path).GetPrim()
    )
    if settings.schema_opinion == authoring_properties.APPLY:
        schema = UsdPhysics.MaterialAPI.Apply(prim)
    elif settings.schema_opinion == authoring_properties.REMOVE and mapped:
        prim.RemoveAPI(UsdPhysics.MaterialAPI)
        schema = UsdPhysics.MaterialAPI(prim)
    else:
        raise _conversion_error(
            material,
            usd_prim_path,
            "ov.physics",
            "add_on_physics_material_requires_apply",
        )
    for property_name, creator in (
        ("static_friction", schema.CreateStaticFrictionAttr),
        ("dynamic_friction", schema.CreateDynamicFrictionAttr),
        ("restitution", schema.CreateRestitutionAttr),
    ):
        if not mapped or _is_set(settings, property_name):
            creator().Set(float(getattr(settings, property_name)))
    return AddOnUsdOpinionRecord(
        usd_prim_path,
        layer,
        _diagnostics(material, usd_prim_path, "ov.physics"),
    )


def _apply_api_opinion(prim: Any, schema: Any, opinion: str) -> None:
    if opinion == authoring_properties.APPLY:
        schema.Apply(prim)
    elif opinion == authoring_properties.REMOVE:
        prim.RemoveAPI(schema)


def _override_ancestors(stage: Any, parent_path: Any) -> None:
    paths = []
    current = parent_path
    while current and not current.IsAbsoluteRootPath():
        paths.append(current)
        current = current.GetParentPath()
    for path in reversed(paths):
        stage.OverridePrim(path)


def _is_set(group: Any, name: str) -> bool:
    return bool(group.is_property_set(name))


def _diagnostics(owner: Any, root: str, property_path: str) -> dict[str, str]:
    return {
        "blender_owner": str(getattr(owner, "name", "")),
        "blender_property_path": property_path,
        "intended_usd_prim_path": root,
    }


def _conversion_error(
    owner: Any,
    root: str,
    property_path: str,
    reason: str,
) -> SceneGenerationError:
    return SceneGenerationError(
        "Blender physics opinion cannot be converted",
        (
            {
                **_diagnostics(owner, root, property_path),
                "reason": reason,
            },
        ),
    )



__all__ = [
    "convert_generation_physics_material",
    "convert_generation_physics_scene",
    "convert_mapped_object_physics",
    "object_has_physics_opinion",
]
