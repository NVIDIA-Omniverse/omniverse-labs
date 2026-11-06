# SPDX-License-Identifier: Apache-2.0
"""Concrete ordinary-joint USD authoring; runtime support is verified separately.

Limit/drive semantics follow the installed usdPhysics schema and
https://openusd.org/release/api/class_usd_physics_limit_a_p_i.html .
Every generation authors a fresh complete joint record, so inactive schemas,
attributes and drive instances cannot leak through a previous joint type.
"""
import math

from .physics_authoring import active_axis_keys, inactive_animation, owner_actions, action_curves, read_frame
from .physics_endpoints import resolve_joint, validate_axes, _id, require_writable, physics_namespace
from .physics_frames import PhysicsUnits, RigidFrame, matrix_to_transform
from .physics_owner_compilation import reserve_generated_subtree
from .usd_opinion_records import AddOnUsdOpinionRecord


class JointError(ValueError):
    pass


def limits_for_axis(axis, units, *, angular):
    if axis.state == "FREE":
        return None
    if axis.state == "LOCKED":
        return (1.0, -1.0)  # USD low>high denotes a locked degree of freedom.
    if axis.state != "LIMITED":
        raise JointError("Invalid joint limit state")
    lower, upper = float(axis.lower), float(axis.upper)
    if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
        raise JointError("Limited joint axes need finite ordered lower/upper bounds")
    convert = units.angle if angular else units.length
    return convert(lower), convert(upper)


def _constant_parameters(obj):
    errors = inactive_animation(obj)
    if errors:
        raise JointError(errors[0])
    animation = getattr(obj, "animation_data", None)
    curves = [curve for action in owner_actions(obj) for _, curve in action_curves(action)]
    curves += list(getattr(animation, "drivers", ()))
    if any(curve.data_path.startswith("ov.physics.joint.") and curve.data_path != "ov.physics.joint.show_indicator" for curve in curves):
        raise JointError("Animated joint parameters require unavailable live joint writes; use a constant setup and explicit restart")


def _drive(prim, instance, settings, units, *, angular):
    from pxr import UsdPhysics
    if not settings.enabled:
        return
    if settings.mode != "FORCE":
        raise JointError("Acceleration drives require separate runtime and dimensional validation")
    for key in ("target_position", "target_velocity", "stiffness", "damping", "max_effort"):
        value = float(getattr(settings, key))
        if not math.isfinite(value) or (key in {"stiffness", "damping", "max_effort"} and value < 0):
            raise JointError(f"Invalid drive {key}")
    drive = UsdPhysics.DriveAPI.Apply(prim, instance)
    drive.CreateTypeAttr().Set("force")
    convert = units.angle if angular else units.length
    drive.CreateTargetPositionAttr().Set(convert(settings.target_position))
    drive.CreateTargetVelocityAttr().Set(convert(settings.target_velocity))
    drive.CreateStiffnessAttr().Set(units.drive_gain(settings.stiffness, angular=angular))
    drive.CreateDampingAttr().Set(units.drive_gain(settings.damping, angular=angular))
    drive.CreateMaxForceAttr().Set(units.drive_effort(settings.max_effort, angular=angular))


def _usd_local_frame(frame, endpoint, candidate):
    """Stored frames use rigid meters; USD positions use the body's local space.

    Compiled bodies have rigid (unit-scale) world frames; dividing by the
    composed scale is a no-op safeguard. Keep the durable frame and its rigid
    orientation unchanged.
    WORLD frames are already absolute stage meters.
    """
    if endpoint.body_path is None:
        return frame
    from pxr import UsdGeom
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(candidate.GetPrimAtPath(endpoint.body_path))
    transform = matrix_to_transform(tuple(tuple(matrix[column][row] for column in range(4)) for row in range(4)))
    return RigidFrame(tuple(value / transform.uniform_scale for value in frame.position), frame.rotation_wxyz)


def _static_world_frame(frame, endpoint, candidate):
    """A static body cannot move, so its joint frame is a WORLD frame.

    Compose the stored body-relative meters frame with the static body's
    composed rigid world frame; scale is excluded as in _usd_local_frame.
    """
    from pxr import UsdGeom
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(candidate.GetPrimAtPath(endpoint.body_path))
    body = matrix_to_transform(tuple(tuple(matrix[column][row] for column in range(4)) for row in range(4))).frame
    return body.compose(frame)


def _filtered_body_record(record, static_paths):
    """Replace a body's owner record, adding contact filters to static partners."""
    from pxr import Sdf, Usd, UsdPhysics
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(record.layer_text)
    stage = Usd.Stage.Open(layer)
    pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.OverridePrim(record.usd_prim_path)).CreateFilteredPairsRel()
    for path in sorted(static_paths):
        pairs.AddTarget(Sdf.Path(path))
    return AddOnUsdOpinionRecord(record.usd_prim_path, layer, record.diagnostics)


def compile_scene_joints(scene, mappings, stock_path, owner_records):
    """Validate all joint endpoints against the complete compiled body scene.

    Returns (joint_records, replacements), where replacements maps id() of an
    owner record to the record that supersedes it.
    """
    from pxr import Gf, Sdf, Usd, UsdPhysics
    from .scene_generation import blender_id
    joints = [obj for obj in scene.objects if obj.ov.physics.joint.enabled]
    if not joints:
        return (), {}
    stock = Usd.Stage.Open(str(stock_path))
    root = physics_namespace(scene, (str(p.GetPath()) for p in stock.TraverseAll()))
    layers = []
    for record in owner_records:
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(record.layer_text)
        layers.append(layer)
    complete = Sdf.Layer.CreateAnonymous(".usda")
    complete.subLayerPaths = [layer.identifier for layer in layers] + [str(stock_path)]
    candidate = Usd.Stage.Open(complete)
    reserve_generated_subtree(candidate, root + "/Joints")
    object_paths = {obj: mappings[blender_id(obj, "OBJECT")].object_path for obj in scene.objects
                    if blender_id(obj, "OBJECT") in mappings}
    units = PhysicsUnits.from_scene(scene)
    try:
        import bpy
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except ModuleNotFoundError:
        depsgraph = None
    schemas = {"FIXED": UsdPhysics.FixedJoint, "REVOLUTE": UsdPhysics.RevoluteJoint,
               "PRISMATIC": UsdPhysics.PrismaticJoint, "D6": UsdPhysics.Joint}
    validated = []
    for obj in joints:
        require_writable(obj)
        joint = obj.ov.physics.joint
        if obj.type != "EMPTY":
            raise JointError("A joint owner must be an ordinary Empty")
        if joint.type not in schemas:
            raise JointError(f"{joint.type} joint authoring is unavailable until separately validated")
        if not joint.frames_initialized:
            raise JointError("Initialize/apply the joint anchor before compiling")
        validate_axes(joint)
        _constant_parameters(obj)
        endpoints = resolve_joint(joint, scene, depsgraph, object_body_paths=object_paths)
        if all(endpoint.kind == "WORLD" or endpoint.role == "STATIC" for endpoint in endpoints):
            raise JointError("A joint needs at least one dynamic or kinematic body")
        for endpoint in endpoints:
            if endpoint.kind == "WORLD":
                continue
            prim = candidate.GetPrimAtPath(endpoint.body_path)
            if endpoint.role == "STATIC":
                # Compiled as a WORLD attachment at the static body's frame.
                if not prim:
                    raise JointError("Static joint endpoint does not resolve to a compiled body")
                continue
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise JointError("Joint endpoint does not resolve to a compiled rigid body")
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False:
                raise JointError("Joint endpoint rigid body is disabled")
        validated.append((obj, endpoints))
    result = []
    static_partners = {}
    for obj, endpoints in validated:
        joint = obj.ov.physics.joint
        path = root + "/Joints/j_" + _id(joint.persistent_id)
        layer = Sdf.Layer.CreateAnonymous(".usda")
        stage = Usd.Stage.Open(layer)
        schema = schemas[joint.type].Define(stage, path)
        prim = schema.GetPrim()
        static = [endpoint for endpoint in endpoints if endpoint.role == "STATIC"]
        for i, (endpoint, frame_record) in enumerate(zip(endpoints, (joint.frame_a, joint.frame_b))):
            frame = units.frame(read_frame(frame_record))
            if endpoint.role == "STATIC":
                frame, body_path = _static_world_frame(frame, endpoint, candidate), None
            else:
                frame, body_path = _usd_local_frame(frame, endpoint, candidate), endpoint.body_path
            getattr(schema, f"CreateBody{i}Rel")().SetTargets([Sdf.Path(body_path)] if body_path else [])
            getattr(schema, f"CreateLocalPos{i}Attr")().Set(Gf.Vec3f(*frame.position))
            w, x, y, z = frame.rotation_wxyz
            getattr(schema, f"CreateLocalRot{i}Attr")().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        schema.CreateJointEnabledAttr().Set(True)
        schema.CreateCollisionEnabledAttr().Set(joint.collision_enabled)
        if static and not joint.collision_enabled:
            # The world attachment drops the pair's joint filtering; restore it.
            moving = next(endpoint for endpoint in endpoints if endpoint.body_path and endpoint.role != "STATIC")
            static_partners.setdefault(moving.body_path, set()).add(static[0].body_path)
        schema.CreateBreakForceAttr().Set(units.force(joint.break_force))
        schema.CreateBreakTorqueAttr().Set(units.torque(joint.break_torque))
        if joint.type in {"REVOLUTE", "PRISMATIC"}:
            schema.CreateAxisAttr().Set(joint.axis)
        for key in active_axis_keys(joint):
            axis = joint.axes[key]
            angular = key.startswith("rot")
            bounds = limits_for_axis(axis, units, angular=angular)
            if joint.type == "D6":
                if bounds is not None:
                    limit = UsdPhysics.LimitAPI.Apply(prim, key)
                    limit.CreateLowAttr().Set(bounds[0])
                    limit.CreateHighAttr().Set(bounds[1])
                instance = key
            else:
                if bounds is not None:
                    schema.CreateLowerLimitAttr().Set(bounds[0])
                    schema.CreateUpperLimitAttr().Set(bounds[1])
                instance = "angular" if angular else "linear"
            _drive(prim, instance, axis.drive, units, angular=angular)
        result.append(AddOnUsdOpinionRecord(path, layer, {"joint_owner": obj.name, "joint_id": joint.persistent_id,
                                                        "joint_session_uid": str(obj.session_uid)}))
    # A mapped body can have several records at its root (e.g. a matrix
    # override); replace only the one that authors its rigid body.
    replaced = {}
    for body_path, static_paths in static_partners.items():
        body = next((record for record in owner_records if record.usd_prim_path == body_path
                     and _authors_rigid_body(record)), None)
        if body is None:
            raise JointError("Static-attached joint body has no owner record to filter contacts")
        replaced[id(body)] = _filtered_body_record(body, static_paths)
    return tuple(result), replaced


def _authors_rigid_body(record):
    from pxr import Sdf
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(record.layer_text)
    spec = layer.GetPrimAtPath(record.usd_prim_path)
    schemas = spec.GetInfo("apiSchemas") if spec is not None and spec.HasInfo("apiSchemas") else None
    return schemas is not None and "PhysicsRigidBodyAPI" in schemas.ApplyOperations([])
