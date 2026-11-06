# SPDX-License-Identifier: Apache-2.0
"""Durable ownership and strict endpoint resolution; resolution never mutates RNA.

Identity creation/repair is explicit and undoable. Bone definitions belong to
armature data, body instances to the armature Object and its PoseBones.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import uuid
from typing import Any

AXIS_KEYS = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")


class EndpointError(ValueError):
    pass


def require_writable(owner):
    data = getattr(owner, "id_data", owner)
    # Overrides are intentionally unavailable until independently certified.
    if getattr(data, "library", None) or getattr(data, "override_library", None):
        raise EndpointError("Physics authoring requires local data; linked data and library overrides are not yet supported")
    if not getattr(data, "is_editable", True):
        raise EndpointError("Physics owner is read-only")


def _id(value):
    if not re.fullmatch(r"[a-f0-9]{32}", value or ""):
        raise EndpointError("Missing or invalid persistent identity; initialize or explicitly repair ownership")
    return value


def initialize_owner(obj, bone=None):
    """Explicit creation action: assign persistent physics identities."""
    require_writable(obj)
    if bone is not None:
        require_writable(bone)
    if not obj.ov.physics.persistent_id:
        obj.ov.physics.persistent_id = uuid.uuid4().hex
    target = obj
    if bone is not None:
        if not bone.ov.persistent_id:
            bone.ov.persistent_id = uuid.uuid4().hex
        target = obj.pose.bones[bone.name]
    body = target.ov.physics.rigid_body
    if not body.persistent_id:
        body.persistent_id = uuid.uuid4().hex
    return target


def initialize_joint(obj):
    require_writable(obj)
    if obj.type != "EMPTY":
        raise EndpointError("A joint must be an Empty")
    if not obj.ov.physics.persistent_id:
        obj.ov.physics.persistent_id = uuid.uuid4().hex
    joint = obj.ov.physics.joint
    if not joint.persistent_id:
        joint.persistent_id = uuid.uuid4().hex
    if not joint.axes:
        for key in AXIS_KEYS:
            joint.axes.add().name = key
        from .scene_generation_sessions import note_physics_authoring_changed
        note_physics_authoring_changed(obj)
    validate_axes(joint)
    return joint


def repair_bone_identity(obj, bone):
    """Explicit selection disambiguates the definition; endpoints stay unresolved.

    Call bind_bone_endpoint for each affected joint after selecting its intended
    bone. Repair never infers that a joint should follow the original or copy.
    """
    require_writable(obj)
    require_writable(bone)
    old = _id(bone.ov.persistent_id)
    matches = [b for b in obj.data.bones if b.ov.persistent_id == old]
    if len(matches) < 2:
        return False
    # Invalidate every reference to the ambiguous identity before repair,
    # including references through other Objects sharing this armature data.
    if bpy is not None:
        for owner in bpy.data.objects:
            for endpoint in (owner.ov.physics.joint.endpoint_a, owner.ov.physics.joint.endpoint_b):
                if endpoint.kind == "BONE" and endpoint.object is not None and endpoint.object.data == obj.data and endpoint.bone_id == old:
                    require_writable(owner)
    replacement = uuid.uuid5(uuid.UUID(old), bone.name).hex
    if any(b.ov.persistent_id == replacement for b in obj.data.bones):
        raise EndpointError("Repair identity collision; rename selected bone and retry")
    if bpy is not None:
        for owner in bpy.data.objects:
            for endpoint in (owner.ov.physics.joint.endpoint_a, owner.ov.physics.joint.endpoint_b):
                if endpoint.kind == "BONE" and endpoint.object is not None and endpoint.object.data == obj.data and endpoint.bone_id == old:
                    endpoint.bone_id = ""
    bone.ov.persistent_id = replacement
    return True


def bind_bone_endpoint(endpoint, obj, bone):
    require_writable(endpoint)
    if obj.type != "ARMATURE" or bone.id_data != obj.data:
        raise EndpointError("Selected bone does not belong to endpoint armature")
    identity = _id(bone.ov.persistent_id)
    if sum(b.ov.persistent_id == identity for b in obj.data.bones) != 1:
        raise EndpointError("Repair duplicate bone identities before binding")
    endpoint.kind, endpoint.object = "BONE", obj
    endpoint.bone_id, endpoint.bone_name = identity, bone.name


def validate_axes(joint):
    if tuple(axis.name for axis in joint.axes) != AXIS_KEYS:
        raise EndpointError("Joint axes must retain unique semantic names and canonical ordering")


def physics_namespace(scene, occupied_paths=()):
    """Check reservation against source/stock paths before generating children."""
    root = scene.ov.physics_namespace
    if not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", root):
        raise EndpointError("Invalid physics namespace")
    if any(p == root or p.startswith(root + "/") for p in occupied_paths):
        raise EndpointError("Physics namespace overlaps source USD; choose an unused namespace")
    return root


@dataclass(frozen=True)
class ResolvedEndpoint:
    kind: str
    body_identity: str | None
    body_path: str | None
    authoring_world_frame: Any
    presentation_target: Any
    role: str | None


def resolve_endpoint(endpoint, scene, depsgraph=None, *, object_body_paths=None, occupied_paths=()):
    from .physics_frames import RigidFrame, PhysicsUnits, evaluated_owner_frame
    if endpoint.kind == "WORLD":
        return ResolvedEndpoint("WORLD", None, None, RigidFrame(), None, None)
    if endpoint.kind not in {"OBJECT", "BONE"}:
        raise EndpointError("Unknown endpoint kind")
    obj = endpoint.object
    if obj is None:
        raise EndpointError("Endpoint Object is missing; WORLD must be explicit")
    if not any(candidate == obj for candidate in scene.objects):
        raise EndpointError("Endpoint Object is outside the active scene")
    require_writable(obj)
    owner_id = _id(obj.ov.physics.persistent_id)
    if sum(o.ov.physics.persistent_id == owner_id for o in scene.objects) != 1:
        raise EndpointError("Duplicate Object identity requires explicit repair")
    target, bone_name = obj, None
    identity = owner_id
    if endpoint.kind == "BONE":
        if obj.type != "ARMATURE":
            raise EndpointError("Bone endpoint requires an armature Object")
        require_writable(obj.data)
        matches = [b for b in obj.data.bones if b.ov.persistent_id == _id(endpoint.bone_id)]
        if len(matches) != 1:
            raise EndpointError("Bone identity is missing or ambiguous; explicitly select the intended bone")
        bone_name = matches[0].name
        target = obj.pose.bones[bone_name]
        identity += "_" + endpoint.bone_id
    body = target.ov.physics.rigid_body
    if body.schema_opinion != "APPLY" or not body.enabled:
        raise EndpointError("Endpoint has no enabled authored OV body")
    _id(body.persistent_id)
    root = physics_namespace(scene, occupied_paths)
    path = (object_body_paths or {}).get(obj) if endpoint.kind == "OBJECT" else None
    if path is None:
        path = root + "/Bodies/b_" + identity
    offset = RigidFrame(tuple(body.offset.translation), tuple(body.offset.rotation))
    frame = evaluated_owner_frame(obj, depsgraph, bone_name=bone_name, offset=offset,
                                  units=PhysicsUnits(scene.unit_settings.scale_length))
    return ResolvedEndpoint(endpoint.kind, identity, path, frame, target, body.role)


def resolve_joint(joint, scene, depsgraph=None, **kwargs):
    joint_id = _id(joint.persistent_id)
    if sum(o.ov.physics.joint.persistent_id == joint_id for o in scene.objects) != 1:
        raise EndpointError("Duplicate joint identity requires explicit repair")
    a = resolve_endpoint(joint.endpoint_a, scene, depsgraph, **kwargs)
    b = resolve_endpoint(joint.endpoint_b, scene, depsgraph, **kwargs)
    if a.body_identity == b.body_identity:
        raise EndpointError("A joint cannot connect a body to itself or WORLD to WORLD")
    return a, b


def repair_object_identity(obj, scene):
    """Caller selects the duplicate to reidentify; endpoint Object pointers survive.

    UUID5 makes a repeated repair after undo deterministic. No bone endpoint is
    rewritten: ambiguous copied bone IDs need an explicit bone selection first.
    """
    require_writable(obj)
    old = _id(obj.ov.physics.persistent_id)
    peers = [o for o in scene.objects if o.ov.physics.persistent_id == old]
    if len(peers) < 2:
        return False
    replacement = uuid.uuid5(uuid.UUID(old), obj.name_full).hex
    used = {o.ov.physics.persistent_id for o in scene.objects}
    counter = 0
    while replacement in used:
        counter += 1
        replacement = uuid.uuid5(uuid.UUID(old), obj.name_full + str(counter)).hex
    obj.ov.physics.persistent_id = replacement
    obj.ov.physics.rigid_body.persistent_id = uuid.uuid5(uuid.UUID(replacement), "body").hex
    if obj.ov.physics.joint.persistent_id:
        obj.ov.physics.joint.persistent_id = uuid.uuid5(uuid.UUID(replacement), "joint").hex
    return True


try:
    import bpy
except ModuleNotFoundError:
    bpy = None

if bpy is not None:
    class OVPHYSICS_OT_repair_identity(bpy.types.Operator):
        bl_idname = "ov.physics_repair_identity"
        bl_label = "Reidentify Selected Duplicate"
        bl_description = "Give the selected duplicate Object or bone a new persistent physics identity"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            try:
                if context.active_pose_bone is not None:
                    repair_bone_identity(context.object, context.active_pose_bone.bone)
                else:
                    repair_object_identity(context.object, context.scene)
            except (EndpointError, AttributeError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            return {"FINISHED"}


def register():
    if bpy is not None and not getattr(OVPHYSICS_OT_repair_identity, "is_registered", False):
        bpy.utils.register_class(OVPHYSICS_OT_repair_identity)


def unregister():
    if bpy is not None and getattr(OVPHYSICS_OT_repair_identity, "is_registered", False):
        bpy.utils.unregister_class(OVPHYSICS_OT_repair_identity)
