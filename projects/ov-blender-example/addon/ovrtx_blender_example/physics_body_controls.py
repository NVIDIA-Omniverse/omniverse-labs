# SPDX-License-Identifier: Apache-2.0
"""Bounded body controls in the shared BU/kg/s dimensional contract.

CCD, sleep thresholds, solver iterations, contacts, filtering, and telemetry are
not admitted controls. PhysicsUnits converts to meter/kg/degree USD exactly once.
"""
import math
from .physics_colliders import ColliderError


def _set(settings, field):
    return hasattr(settings, field) and settings.is_property_set(field)


def positive(value, label, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ColliderError(label + " must be finite and " + ("nonnegative" if zero else "positive"))
    return value


def author_body_controls(prim, settings, units, *, body_scale=1.0):
    from pxr import Gf, Sdf, UsdPhysics
    from .physics_topology_conversion import author_mass_distribution
    mode = getattr(settings, "mass_mode", "MASS")
    if mode == "MASS" and _set(settings, "mass_kg"):
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(positive(settings.mass_kg, "Body mass"))
        if _set(settings, "mass_mode"):
            UsdPhysics.MassAPI(prim).CreateDensityAttr().Set(0.)
    elif mode == "DENSITY":
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(0.)  # Override weaker explicit mass precedence.
        mass.CreateDensityAttr().Set(units.density(positive(settings.density, "Body density")))
    elif mode == "INHERIT":
        pass
    elif mode != "MASS":
        raise ColliderError("Unsupported body mass mode")
    author_mass_distribution(prim, settings, units, body_scale=body_scale)
    rigid = UsdPhysics.RigidBodyAPI(prim)
    if _set(settings, "initial_linear_velocity"):
        rigid.CreateVelocityAttr().Set(Gf.Vec3f(*units.vector(settings.initial_linear_velocity)))
    if _set(settings, "initial_angular_velocity"):
        rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(*units.angular_velocity(settings.initial_angular_velocity)))
    for field, attribute in (("linear_damping", "physxRigidBody:linearDamping"), ("angular_damping", "physxRigidBody:angularDamping")):
        if _set(settings, field):
            # Author the codeless API identifier without registering another USD
            # runtime/plugin into Blender. The fixed worker owns its schemas.
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
            prim.CreateAttribute(attribute, Sdf.ValueTypeNames.Float).Set(positive(getattr(settings, field), field, zero=True))


FIELD_MATRIX = (
    ("mass_kg / mass_mode", "PhysicsMassAPI.mass", "kg", "explicit restart"),
    ("density", "PhysicsMassAPI.density", "kg/BU³ -> kg/m³", "explicit restart"),
    ("center_of_mass", "PhysicsMassAPI.centerOfMass", "BU -> m", "explicit restart"),
    ("diagonal_inertia", "PhysicsMassAPI.diagonalInertia", "kg BU² -> kg m²", "explicit restart"),
    ("principal_axes", "PhysicsMassAPI.principalAxes", "normalized wxyz", "explicit restart"),
    ("initial_linear_velocity", "PhysicsRigidBodyAPI.velocity", "BU/s -> m/s", "initial condition on restart"),
    ("initial_angular_velocity", "PhysicsRigidBodyAPI.angularVelocity", "rad/s -> deg/s", "initial condition on restart"),
    ("linear_damping / angular_damping", "PhysxRigidBodyAPI.*Damping", "1/s", "explicit restart"),
    ("BOX dimensions", "Cube.size + local scale", "BU -> m", "recompile"),
    ("CAPSULE radius_m / height_m", "Capsule.radius / height, Z axis", "m", "recompile"),
    ("material friction / restitution / density", "PhysicsMaterialAPI", "unitless / kg BU^-3", "explicit restart"),
    ("gravity_direction / gravity_magnitude", "PhysicsScene.gravity*", "direction / m/s²", "explicit restart"),
)
UNAVAILABLE = "CCD, sleep tuning, solver options, contact/rest offsets, collision filters and telemetry require separate runtime validation"
