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
    from .physics_body_controls import positive
    import math
    positive(settings.gravity_magnitude, "Gravity magnitude", zero=True)
    if not all(math.isfinite(v) for v in settings.gravity_direction) or not any(settings.gravity_direction):
        raise ValueError("Gravity direction must be finite and nonzero")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*settings.gravity_direction))
    physics_scene.CreateGravityMagnitudeAttr().Set(float(settings.gravity_magnitude))
    return AddOnUsdOpinionRecord(
        root,
        layer,
        _diagnostics(scene, root, "ov.physics_scene"),
    )


def convert_generation_physics_material(
    material: Any,
    usd_prim_path: str,
    *,
    mapped: bool,
    units=None,
) -> AddOnUsdOpinionRecord:
    """Compile one mapped override or complete add-on-only physics material."""

    settings = material.ov.physics
    from .physics_frames import PhysicsUnits
    units = units or PhysicsUnits()
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
            from .physics_body_controls import positive
            value = positive(getattr(settings, property_name), property_name, zero=True)
            if property_name == "restitution" and value > 1:
                raise ValueError("Restitution must be between zero and one")
            creator().Set(value)
    if _is_set(settings, "density"):
        from .physics_body_controls import positive
        schema.CreateDensityAttr().Set(units.density(positive(settings.density, "Material density", zero=True)))
    return AddOnUsdOpinionRecord(
        usd_prim_path,
        layer,
        _diagnostics(material, usd_prim_path, "ov.physics"),
    )


def author_mass_distribution(prim, settings, units, *, body_scale=1.0):
    """Shared typed kg/BU/wxyz source; USD output is kg/m/wxyz."""
    Gf, _Sdf, _Usd, UsdPhysics, _Shade = _pxr_modules()
    for field, creator in (("center_of_mass", "CreateCenterOfMassAttr"), ("diagonal_inertia", "CreateDiagonalInertiaAttr"), ("principal_axes", "CreatePrincipalAxesAttr")):
        if _is_set(settings, field):
            value = tuple(getattr(settings, field))
            import math
            from .physics_frames import normalize_quaternion
            from .physics_colliders import ColliderError
            if not all(math.isfinite(v) for v in value):
                raise ColliderError("Mass distribution must be finite")
            if field == "diagonal_inertia" and (any(v < 0 for v in value) or any(value) and not all(v > 0 for v in value)):
                raise ColliderError("Diagonal inertia must be all positive or all zero for automatic inertia")
            if field == "principal_axes":
                value = normalize_quaternion(value)
            if field == "center_of_mass":
                value = tuple(units.length(v) / body_scale for v in value)
            elif field == "diagonal_inertia":
                value = tuple(units.inertia(v) for v in value)
            value = Gf.Quatf(value[0], Gf.Vec3f(*value[1:])) if field == "principal_axes" else Gf.Vec3f(*value)
            getattr(UsdPhysics.MassAPI.Apply(prim), creator)().Set(value)


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
]
