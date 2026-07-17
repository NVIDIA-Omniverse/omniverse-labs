# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable Blender authoring source under one ``ov`` property namespace."""

from __future__ import annotations

from typing import Any

try:
    import bpy  # type: ignore
    from bpy.props import (  # type: ignore
        BoolProperty,
        EnumProperty,
        FloatProperty,
        FloatVectorProperty,
        PointerProperty,
        StringProperty,
    )
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]
    BoolProperty = EnumProperty = FloatProperty = FloatVectorProperty = None  # type: ignore[assignment]
    PointerProperty = StringProperty = None  # type: ignore[assignment]


INHERIT = "INHERIT"
APPLY = "APPLY"
REMOVE = "REMOVE"
SCHEMA_OPINION_ITEMS = (
    (INHERIT, "Use Source", "Author no stronger applied-schema opinion"),
    (APPLY, "Apply", "Apply the USD schema in this authoring session"),
    (REMOVE, "Remove", "Remove a weaker applied USD schema in this authoring session"),
)


def _schema_opinion() -> Any:
    return EnumProperty(
        name="Schema Opinion",
        items=SCHEMA_OPINION_ITEMS,
        default=INHERIT,
    )


if bpy is not None:

    class OvUsdIdentityProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        prim_path: StringProperty(  # type: ignore[valid-type]
            name="USD Prim Path",
            description="Stable current-file USD prim path for this Blender authoring source",
            default="",
        )


    class OvRigidBodyProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        schema_opinion: _schema_opinion()  # type: ignore[valid-type]
        mass_kg: FloatProperty(  # type: ignore[valid-type]
            name="Mass",
            default=0.0,
            min=0.0,
            unit="MASS",
        )


    class OvCollisionProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        schema_opinion: _schema_opinion()  # type: ignore[valid-type]
        shape: EnumProperty(  # type: ignore[valid-type]
            name="Collision Shape",
            items=(
                ("CONVEX_HULL", "Convex Hull", "Use the mapped stock-exported mesh"),
                ("ANALYTIC_SPHERE", "Sphere", "Author an add-on analytic sphere collider"),
            ),
            default="CONVEX_HULL",
        )
        radius_m: FloatProperty(  # type: ignore[valid-type]
            name="Radius",
            default=0.5,
            min=0.0,
            unit="LENGTH",
        )
        hide_from_render: BoolProperty(  # type: ignore[valid-type]
            name="Hide Collider From Render",
            default=False,
        )
        physics_material: PointerProperty(  # type: ignore[valid-type]
            name="Physics Material",
            type=bpy.types.Material,
        )


    class OvObjectPhysicsProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        rigid_body: PointerProperty(type=OvRigidBodyProperties)  # type: ignore[valid-type]
        collision: PointerProperty(type=OvCollisionProperties)  # type: ignore[valid-type]


    class OvObjectAuthoringProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        usd: PointerProperty(type=OvUsdIdentityProperties)  # type: ignore[valid-type]
        physics: PointerProperty(type=OvObjectPhysicsProperties)  # type: ignore[valid-type]


    class OvPhysicsMaterialProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        schema_opinion: _schema_opinion()  # type: ignore[valid-type]
        static_friction: FloatProperty(  # type: ignore[valid-type]
            name="Static Friction",
            default=0.5,
            min=0.0,
        )
        dynamic_friction: FloatProperty(  # type: ignore[valid-type]
            name="Dynamic Friction",
            default=0.5,
            min=0.0,
        )
        restitution: FloatProperty(  # type: ignore[valid-type]
            name="Restitution",
            default=0.0,
            min=0.0,
            max=1.0,
        )


    class OvMaterialAuthoringProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        usd: PointerProperty(type=OvUsdIdentityProperties)  # type: ignore[valid-type]
        physics: PointerProperty(type=OvPhysicsMaterialProperties)  # type: ignore[valid-type]


    class OvPhysicsSceneProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        schema_opinion: _schema_opinion()  # type: ignore[valid-type]
        gravity_direction: FloatVectorProperty(  # type: ignore[valid-type]
            name="Gravity Direction",
            size=3,
            subtype="DIRECTION",
            default=(0.0, 0.0, -1.0),
        )
        gravity_magnitude: FloatProperty(  # type: ignore[valid-type]
            name="Gravity Magnitude",
            default=9.81,
            min=0.0,
        )


    class OvSceneUsdProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        physics_scene_path: StringProperty(  # type: ignore[valid-type]
            name="Physics Scene Prim Path",
            default="/World/PhysicsScene",
        )


    class OvSceneAuthoringProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        usd: PointerProperty(type=OvSceneUsdProperties)  # type: ignore[valid-type]
        physics_scene: PointerProperty(type=OvPhysicsSceneProperties)  # type: ignore[valid-type]


    _CLASSES = (
        OvUsdIdentityProperties,
        OvRigidBodyProperties,
        OvCollisionProperties,
        OvObjectPhysicsProperties,
        OvObjectAuthoringProperties,
        OvPhysicsMaterialProperties,
        OvMaterialAuthoringProperties,
        OvPhysicsSceneProperties,
        OvSceneUsdProperties,
        OvSceneAuthoringProperties,
    )
else:
    OvUsdIdentityProperties = None  # type: ignore[assignment]
    OvRigidBodyProperties = None  # type: ignore[assignment]
    OvCollisionProperties = None  # type: ignore[assignment]
    OvObjectPhysicsProperties = None  # type: ignore[assignment]
    OvObjectAuthoringProperties = None  # type: ignore[assignment]
    OvPhysicsMaterialProperties = None  # type: ignore[assignment]
    OvMaterialAuthoringProperties = None  # type: ignore[assignment]
    OvPhysicsSceneProperties = None  # type: ignore[assignment]
    OvSceneUsdProperties = None  # type: ignore[assignment]
    OvSceneAuthoringProperties = None  # type: ignore[assignment]
    _CLASSES: tuple[type[Any], ...] = ()


def register() -> None:
    if bpy is None:
        raise RuntimeError("authoring properties require Blender")
    for cls in _CLASSES:
        if not getattr(cls, "is_registered", False):
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Object, "ov"):
        bpy.types.Object.ov = PointerProperty(type=OvObjectAuthoringProperties)
    if not hasattr(bpy.types.Material, "ov"):
        bpy.types.Material.ov = PointerProperty(type=OvMaterialAuthoringProperties)
    if not hasattr(bpy.types.Scene, "ov"):
        bpy.types.Scene.ov = PointerProperty(type=OvSceneAuthoringProperties)


def unregister() -> None:
    if bpy is None:
        raise RuntimeError("authoring properties require Blender")
    for owner in (bpy.types.Scene, bpy.types.Material, bpy.types.Object):
        if hasattr(owner, "ov"):
            del owner.ov
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError) as exc:
            if "missing bl_rna" not in str(exc) and "not registered" not in str(exc):
                raise


__all__ = [
    "APPLY",
    "INHERIT",
    "REMOVE",
    "register",
    "unregister",
]
