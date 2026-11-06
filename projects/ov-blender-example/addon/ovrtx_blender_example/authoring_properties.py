# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable Blender authoring source under one ``ov`` property namespace."""

from __future__ import annotations

from typing import Any

try:
    import bpy  # type: ignore
    from bpy.props import (  # type: ignore
        BoolProperty,
        CollectionProperty,
        IntProperty,
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


def _rendering_property_updated(group: Any, _context: Any) -> None:
    """Queue exact owner identities; no export or runtime work in RNA callbacks."""
    from .scene_generation import blender_id
    from .scene_generation_sessions import mark_scene_dirty

    owner = getattr(group, 'id_data', None)
    if owner is None or bpy is None:
        return
    identity = blender_id(owner, 'OBJECT')
    for scene in bpy.data.scenes:
        if scene.objects.get(owner.name) is owner:
            mark_scene_dirty(scene, {identity})


if bpy is not None:

    class OvPhysicsIdentityProperties(bpy.types.PropertyGroup):
        persistent_id: StringProperty(name="Persistent Identity", default="")

    class OvPhysicsFrameProperties(bpy.types.PropertyGroup):
        translation: FloatVectorProperty(size=3, default=(0, 0, 0))
        rotation: FloatVectorProperty(size=4, subtype="QUATERNION", default=(1, 0, 0, 0))

    class OvPhysicsEndpointProperties(bpy.types.PropertyGroup):
        kind: EnumProperty(items=(("OBJECT", "Object", ""), ("BONE", "Bone", ""), ("WORLD", "World", "")), default="OBJECT")
        object: PointerProperty(type=bpy.types.Object)
        bone_id: StringProperty(default="")
        bone_name: StringProperty(default="")

    class OvPhysicsDriveProperties(bpy.types.PropertyGroup):
        enabled: BoolProperty(default=False)
        mode: EnumProperty(items=(("FORCE", "Force", ""), ("ACCELERATION", "Acceleration", "")), default="FORCE")
        target_position: FloatProperty(default=0)
        target_velocity: FloatProperty(default=0)
        stiffness: FloatProperty(default=0, min=0)
        damping: FloatProperty(default=0, min=0)
        max_effort: FloatProperty(default=1e10, min=0)

    class OvPhysicsAxisProperties(bpy.types.PropertyGroup):
        # Collection item names are semantic keys, used in animation paths.
        state: EnumProperty(items=(("FREE", "Free", ""), ("LOCKED", "Locked", ""), ("LIMITED", "Limited", "")), default="FREE")
        lower: FloatProperty(default=0)
        upper: FloatProperty(default=0)
        drive: PointerProperty(type=OvPhysicsDriveProperties)

    class OvPhysicsJointProperties(bpy.types.PropertyGroup):
        persistent_id: StringProperty(default="")
        enabled: BoolProperty(default=False)
        show_indicator: BoolProperty(default=True)
        type: EnumProperty(items=tuple((k, k.title(), "") for k in ("FIXED", "REVOLUTE", "PRISMATIC", "D6", "SPHERICAL", "DISTANCE")), default="FIXED")
        axis: EnumProperty(items=tuple((k, k, "") for k in "XYZ"), default="X")
        endpoint_a: PointerProperty(type=OvPhysicsEndpointProperties)
        endpoint_b: PointerProperty(type=OvPhysicsEndpointProperties)
        frame_a: PointerProperty(type=OvPhysicsFrameProperties)
        frame_b: PointerProperty(type=OvPhysicsFrameProperties)
        committed_handle: PointerProperty(type=OvPhysicsFrameProperties)
        frames_initialized: BoolProperty(default=False)
        axes: CollectionProperty(type=OvPhysicsAxisProperties)
        break_force: FloatProperty(default=1e10, min=0)
        break_torque: FloatProperty(default=1e10, min=0)
        collision_enabled: BoolProperty(default=False)
        mechanism_id: StringProperty(default="")

    class OvUsdIdentityProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        prim_path: StringProperty(  # type: ignore[valid-type]
            name="USD Prim Path",
            description="Stable current-file USD prim path for this Blender authoring source",
            default="",
        )


    class OvRigidBodyProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        mass_mode: EnumProperty(items=(("MASS", "Explicit Mass", ""), ("DENSITY", "Body Density", ""), ("INHERIT", "Use Source / Material", "")), default="MASS")
        density: FloatProperty(default=1000.0, min=0.0)
        initial_linear_velocity: FloatVectorProperty(size=3, default=(0, 0, 0))
        initial_angular_velocity: FloatVectorProperty(size=3, default=(0, 0, 0))
        linear_damping: FloatProperty(default=0.0, min=0.0)
        angular_damping: FloatProperty(default=0.05, min=0.0)
        center_of_mass: FloatVectorProperty(size=3, default=(0, 0, 0))
        diagonal_inertia: FloatVectorProperty(size=3, default=(0, 0, 0))
        principal_axes: FloatVectorProperty(size=4, subtype="QUATERNION", default=(1, 0, 0, 0))
        persistent_id: StringProperty(default="")
        role: EnumProperty(items=(("DYNAMIC", "Dynamic", ""), ("KINEMATIC", "Kinematic", ""), ("STATIC", "Static", "")), default="DYNAMIC")
        enabled: BoolProperty(default=True)
        offset: PointerProperty(type=OvPhysicsFrameProperties)
        schema_opinion: _schema_opinion()  # type: ignore[valid-type]
        mass_kg: FloatProperty(  # type: ignore[valid-type]
            name="Mass",
            default=0.0,
            min=0.0,
            unit="MASS",
        )


    class OvColliderProperties(bpy.types.PropertyGroup):
        persistent_id: StringProperty(default="")
        schema_opinion: _schema_opinion()
        enabled: BoolProperty(default=True)
        shape: EnumProperty(items=tuple((k, k.title(), "") for k in ("CONVEX_HULL", "ANALYTIC_SPHERE", "BOX", "CAPSULE", "SIMREADY_CONVEX_DECOMPOSITION")), default="CONVEX_HULL")
        offset: PointerProperty(type=OvPhysicsFrameProperties)
        dimensions: FloatVectorProperty(size=3, default=(1, 1, 1), min=0)
        radius_m: FloatProperty(default=0.5, min=0)
        height_m: FloatProperty(default=1, min=0)
        physics_material: PointerProperty(type=bpy.types.Material)
        mesh_source: PointerProperty(type=bpy.types.Object)
        hide_from_render: BoolProperty(default=False)
        legacy_target_path: StringProperty(default="")

    class OvObjectPhysicsProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        simready_version: IntProperty(default=0, min=0)
        persistent_id: StringProperty(default="")
        colliders: CollectionProperty(type=OvColliderProperties)
        joint: PointerProperty(type=OvPhysicsJointProperties)
        rigid_body: PointerProperty(type=OvRigidBodyProperties)  # type: ignore[valid-type]


    class OvObjectRenderingProperties(bpy.types.PropertyGroup):
        matte: EnumProperty(name='Matte', description='Overrides are unavailable in supported rendering; use source USD opinions', items=(('INHERIT', 'Use Source', ''), ('ON', 'On', ''), ('OFF', 'Off', '')), default='INHERIT', update=_rendering_property_updated)
        holdout: EnumProperty(name='Holdout', description='Overrides are unavailable in supported rendering; use source USD opinions', items=(('INHERIT', 'Use Source', ''), ('ON', 'On', ''), ('OFF', 'Off', '')), default='INHERIT', update=_rendering_property_updated)

    class OvObjectAuthoringProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        rendering: PointerProperty(type=OvObjectRenderingProperties)
        usd: PointerProperty(type=OvUsdIdentityProperties)  # type: ignore[valid-type]
        physics: PointerProperty(type=OvObjectPhysicsProperties)  # type: ignore[valid-type]


    class OvPhysicsMaterialProperties(bpy.types.PropertyGroup):  # type: ignore[misc]
        density: FloatProperty(default=0.0, min=0.0)
        preserved_prim_path: StringProperty(default="")
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
        indicators_enabled: BoolProperty(name="Physics Indicators", default=True)
        indicators_selected_only: BoolProperty(name="Selected Only", default=False)
        indicators_colliders: BoolProperty(name="Collider Wires", default=True)
        indicators_size: FloatProperty(name="Indicator Size", default=0.25, min=0.001, subtype="DISTANCE")
        # A playback setting, not physics authoring: no authoring update callback.
        sync_playback_to_ovrtx_frames: BoolProperty(  # type: ignore[valid-type]
            name="Sync Playback to OVRTX Frames",
            description=(
                "While live physics presents in an OVRTX viewport, playback waits for the OVRTX image "
                "of the current frame, so keyframed objects stay in step with it. Not applied under "
                "Sync to Audio. Playback continues after 2 s without a new image"
            ),
            default=True,
            update=None,
        )
        physics_namespace: StringProperty(default="/World/OVPhysics")
        usd: PointerProperty(type=OvSceneUsdProperties)  # type: ignore[valid-type]
        physics_scene: PointerProperty(type=OvPhysicsSceneProperties)  # type: ignore[valid-type]


    _CLASSES = (
        OvPhysicsIdentityProperties,
        OvPhysicsFrameProperties,
        OvPhysicsEndpointProperties,
        OvPhysicsDriveProperties,
        OvPhysicsAxisProperties,
        OvPhysicsJointProperties,
        OvUsdIdentityProperties,
        OvRigidBodyProperties,
        OvColliderProperties,
        OvObjectPhysicsProperties,
        OvObjectRenderingProperties,
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
    OvObjectPhysicsProperties = None  # type: ignore[assignment]
    OvObjectAuthoringProperties = None  # type: ignore[assignment]
    OvPhysicsMaterialProperties = None  # type: ignore[assignment]
    OvMaterialAuthoringProperties = None  # type: ignore[assignment]
    OvPhysicsSceneProperties = None  # type: ignore[assignment]
    OvSceneUsdProperties = None  # type: ignore[assignment]
    OvSceneAuthoringProperties = None  # type: ignore[assignment]
    _CLASSES: tuple[type[Any], ...] = ()


def _physics_property_updated(group: Any, _context: Any) -> None:
    from .scene_generation_sessions import note_physics_authoring_changed
    note_physics_authoring_changed(getattr(group, 'id_data', None))


def _prepare_authoring_updates(cls: type[Any]) -> None:
    # Postponed annotations are trusted declarations from this source module.
    # Install one notification policy for every authoring scalar/vector/pointer;
    # collection add/remove operations have explicit owner notifications instead.
    for name, declaration in tuple(getattr(cls, '__annotations__', {}).items()):
        value = eval(declaration, globals()) if isinstance(declaration, str) else declaration
        keywords = getattr(value, 'keywords', None)
        if keywords is None:
            continue
        if getattr(getattr(value, 'function', None), '__name__', '') != 'CollectionProperty':
            keywords.setdefault('update', _physics_property_updated)
        cls.__annotations__[name] = value


def register() -> None:
    if bpy is None:
        raise RuntimeError("authoring properties require Blender")
    for cls in _CLASSES:
        if not getattr(cls, "is_registered", False):
            _prepare_authoring_updates(cls)
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Object, "ov"):
        bpy.types.Object.ov = PointerProperty(type=OvObjectAuthoringProperties)
    if not hasattr(bpy.types.PoseBone, "ov"):
        bpy.types.PoseBone.ov = PointerProperty(type=OvObjectAuthoringProperties)
    if not hasattr(bpy.types.Bone, "ov"):
        bpy.types.Bone.ov = PointerProperty(type=OvPhysicsIdentityProperties)
    if not hasattr(bpy.types.Material, "ov"):
        bpy.types.Material.ov = PointerProperty(type=OvMaterialAuthoringProperties)
    if not hasattr(bpy.types.Scene, "ov"):
        bpy.types.Scene.ov = PointerProperty(type=OvSceneAuthoringProperties)
    from . import physics_endpoints
    physics_endpoints.register()
    from . import physics_colliders
    physics_colliders.register()
    from . import simready_authoring
    simready_authoring.register()
    from . import physics_ui
    physics_ui.register()
    from . import physics_overlay
    physics_overlay.register()


def unregister() -> None:
    if bpy is None:
        raise RuntimeError("authoring properties require Blender")
    from . import physics_overlay
    physics_overlay.unregister()
    from . import physics_ui
    physics_ui.unregister()
    from . import physics_endpoints
    physics_endpoints.unregister()
    from . import physics_colliders
    physics_colliders.unregister()
    from . import simready_authoring
    simready_authoring.unregister()
    for owner in (bpy.types.Scene, bpy.types.Material, bpy.types.PoseBone, bpy.types.Bone, bpy.types.Object):
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
