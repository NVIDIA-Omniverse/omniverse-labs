# SPDX-License-Identifier: Apache-2.0
"""Stopped authoring transactions. Pending anchors derive from durable frames.

All stored translations are Blender units; resolver output is meters. No runtime
poses, dirty flags, or simulation singleton are persisted here.
"""
import math
import re
import sys

from .physics_endpoints import AXIS_KEYS, repair_object_identity, require_writable, resolve_joint
from .physics_frames import RigidFrame, PhysicsUnits, matrix_to_transform, local_frame


class AuthoringError(ValueError):
    pass


def read_frame(record):
    return RigidFrame(record.translation, record.rotation)


def write_frame(record, frame):
    record.translation = frame.position
    record.rotation = frame.rotation_wxyz


def handle_frame(obj):
    transform = matrix_to_transform(obj.matrix_world)
    if not math.isclose(transform.uniform_scale, 1, abs_tol=1e-5):
        raise AuthoringError("Joint handle scale must be one")
    if any(v <= 0 for v in obj.scale):
        raise AuthoringError("Negative joint handle scale is unsupported")
    return transform.frame


def anchor_pending(obj):
    joint = obj.ov.physics.joint
    if not joint.frames_initialized:
        return True
    try:
        current, saved = handle_frame(obj), read_frame(joint.committed_handle)
    except ValueError:
        return True
    return (max(abs(a-b) for a, b in zip(current.position, saved.position)) > 1e-6
            or abs(sum(a*b for a, b in zip(current.rotation_wxyz, saved.rotation_wxyz))) < 1-1e-10)


def simulation_owns_scene(scene):
    # Read the established viewport owners without creating an engine or worker.
    module = sys.modules.get(__package__ + ".engine")
    if module is None:
        return False
    for engine in module._live_viewport_engines():
        lock = getattr(engine, "_physics_playback_lock", None)
        if lock is not None and lock.is_active():
            return True
    return False


def require_stopped(scene, *, starting_playback=False):
    if simulation_owns_scene(scene):
        raise AuthoringError("Stop/reset OVPhysX and release pose ownership before editing")
    bpy = sys.modules.get("bpy")
    if not starting_playback and bpy is not None and any(s.is_animation_playing for s in getattr(getattr(bpy, "data", None), "screens", ())):
        raise AuthoringError("Stop timeline playback before editing physics")


def mark_dirty(scene, obj):
    from .scene_generation_sessions import note_physics_authoring_changed
    note_physics_authoring_changed(scene)
    from .scene_generation import blender_id
    from .scene_generation_sessions import mark_scene_dirty
    owner = getattr(obj, "id_data", obj)
    mark_scene_dirty(scene, {blender_id(owner, "OBJECT")})


def apply_anchor(obj, scene, depsgraph=None, *, repair_duplicate_objects=False,
                 starting_playback=False, mark_scene=True, report_repair=True):
    require_stopped(scene, starting_playback=starting_playback)
    require_writable(obj)
    if obj.type != "EMPTY":
        raise AuthoringError("Joint handle must be an Empty")
    handle = handle_frame(obj)
    joint = obj.ov.physics.joint
    repaired = []
    identities = []
    session_owners = None
    if repair_duplicate_objects:
        from . import scene_generation_sessions as sessions
        uid = int(getattr(scene, "session_uid", 0) or 0)
        session_owners = dict(sessions._identity_owners.get(uid, {}))
        # Global duplicate repair participates in the same anchor transaction.
        # Snapshot peers too: a rejected anchor must not silently reidentify them.
        for owner in scene.objects:
            physics = owner.ov.physics
            identities.append((owner, physics.persistent_id, physics.rigid_body.persistent_id,
                               physics.joint.persistent_id))
    try:
        if repair_duplicate_objects:
            repaired.extend(sessions.repair_duplicate_identities(scene, report=False))
            # The selected joint and its endpoint pointers identify exactly
            # which copies the artist is editing. Never change unrelated peers.
            candidates = [obj] + [endpoint.object for endpoint in (joint.endpoint_a, joint.endpoint_b)
                                  if endpoint.kind != "WORLD"]
            for owner in candidates:
                if owner is None:
                    continue
                physics = owner.ov.physics
                old = (owner, physics.persistent_id, physics.rigid_body.persistent_id,
                       physics.joint.persistent_id)
                if not any(owner == previous for previous, *_ in identities):
                    identities.append(old)
                if repair_object_identity(owner, scene):
                    repaired.append(owner.name)
        a, b = resolve_joint(joint, scene, depsgraph)
        units = PhysicsUnits.from_scene(scene)
        anchor = units.frame(handle)
        # Resolve/validate everything before writing frames or retaining a repair.
        frames = [units.frame_from_meters(local_frame(e.authoring_world_frame, anchor)) for e in (a, b)]
    except Exception:
        for owner, owner_id, body_id, joint_id in identities:
            owner.ov.physics.persistent_id = owner_id
            owner.ov.physics.rigid_body.persistent_id = body_id
            owner.ov.physics.joint.persistent_id = joint_id
        if session_owners is not None:
            sessions._identity_owners[uid] = session_owners
        raise
    write_frame(joint.frame_a, frames[0])
    write_frame(joint.frame_b, frames[1])
    write_frame(joint.committed_handle, handle)
    joint.frames_initialized = True
    if mark_scene:
        mark_dirty(scene, obj)
    if repaired and report_repair:
        from . import user_messages
        user_messages.report_info("Gave duplicated physics Objects their own identity: " + ", ".join(repaired),
                                  context=f"physics-identity:{uid}", dedup=False)
    return tuple(repaired)


def apply_pending_anchors_for_playback(scene, depsgraph=None):
    """Commit staged joint handles at the explicit timeline-play boundary."""
    require_scene_ready(scene, allow_pending_anchors=True)
    pending = []
    for obj in tuple(getattr(scene, "objects", ())):
        joint = getattr(getattr(getattr(obj, "ov", None), "physics", None), "joint", None)
        if joint is not None and joint.enabled and anchor_pending(obj):
            pending.append(obj)
    if not pending:
        return ()
    # Each anchor can repair duplicates anywhere in the scene. The outer batch
    # must restore those peers and identity history if a later anchor rejects.
    from . import scene_generation_sessions as sessions
    uid = int(getattr(scene, "session_uid", 0) or 0)
    had_history = uid in sessions._identity_owners
    session_owners = dict(sessions._identity_owners.get(uid, {}))
    identities = [(owner, owner.ov.physics.persistent_id,
                   owner.ov.physics.rigid_body.persistent_id,
                   owner.ov.physics.joint.persistent_id) for owner in scene.objects]
    frames = [(obj.ov.physics.joint, read_frame(obj.ov.physics.joint.frame_a),
               read_frame(obj.ov.physics.joint.frame_b),
               read_frame(obj.ov.physics.joint.committed_handle),
               obj.ov.physics.joint.frames_initialized) for obj in pending]
    repaired = []
    try:
        for obj in pending:
            repaired.extend(apply_anchor(obj, scene, depsgraph, repair_duplicate_objects=True,
                         starting_playback=True, mark_scene=False, report_repair=False))
    except Exception:
        for owner, owner_id, body_id, joint_id in identities:
            owner.ov.physics.persistent_id = owner_id
            owner.ov.physics.rigid_body.persistent_id = body_id
            owner.ov.physics.joint.persistent_id = joint_id
        if had_history:
            sessions._identity_owners[uid] = session_owners
        else:
            sessions._identity_owners.pop(uid, None)
        for joint, frame_a, frame_b, committed, initialized in frames:
            write_frame(joint.frame_a, frame_a)
            write_frame(joint.frame_b, frame_b)
            write_frame(joint.committed_handle, committed)
            joint.frames_initialized = initialized
        raise
    for obj in pending:
        mark_dirty(scene, obj)
    if repaired:
        from . import user_messages
        user_messages.report_info("Gave duplicated physics Objects their own identity: " + ", ".join(dict.fromkeys(repaired)),
                                  context=f"physics-identity:{uid}", dedup=False)
    return tuple(obj.name for obj in pending)


def cancel_anchor(obj, scene):
    require_stopped(scene)
    require_writable(obj)
    if not obj.ov.physics.joint.frames_initialized:
        raise AuthoringError("Joint has no committed anchor")
    from mathutils import Matrix
    obj.matrix_world = Matrix(read_frame(obj.ov.physics.joint.committed_handle).column_matrix())


def edit_frame(obj, scene, side, frame):
    require_stopped(scene)
    require_writable(obj)
    if side not in {"A", "B"}:
        raise AuthoringError("Choose frame A or B")
    if anchor_pending(obj):
        raise AuthoringError("Apply or cancel the pending common anchor before editing one frame")
    write_frame(getattr(obj.ov.physics.joint, "frame_" + side.lower()), frame)
    mark_dirty(scene, obj)


def action_curves(action):
    """Yield actual (collection, curve) pairs for legacy and layered Actions."""
    if action is None:
        return
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        for curve in legacy:
            yield legacy, curve
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                for curve in bag.fcurves:
                    yield bag.fcurves, curve


def owner_actions(owner):
    animation = getattr(owner, "animation_data", None)
    if animation is None:
        return ()
    actions = []
    if animation.action is not None:
        actions.append(animation.action)
    for track in animation.nla_tracks:
        for strip in track.strips:
            if strip.action is not None and strip.action not in actions:
                actions.append(strip.action)
    return tuple(actions)


def animation_owners():
    """Visit all Blender IDs, including embedded shader/compositor node trees.

    Drivers are not restricted to Objects: materials, Keys, scenes and their
    embedded node trees can reference an Object's collider properties too.
    """
    bpy = sys.modules.get("bpy")
    data = getattr(bpy, "data", None)
    properties = getattr(getattr(data, "bl_rna", None), "properties", ())
    names = [p.identifier for p in properties if p.type == "COLLECTION"]
    if not names:
        names = ("objects", "materials", "node_groups", "shape_keys", "scenes", "worlds", "lights", "meshes")
    seen = set()
    def visit(owner):
        identity = owner.as_pointer() if hasattr(owner, "as_pointer") else id(owner)
        if identity in seen:
            return
        seen.add(identity)
        yield owner
        for attribute in ("node_tree", "shape_keys"):
            child = getattr(owner, attribute, None)
            if child is not None:
                yield from visit(child)
    for name in names:
        for owner in getattr(data, name, ()):
            yield from visit(owner)


def active_axis_keys(joint):
    if joint.type == "D6":
        return AXIS_KEYS
    if joint.type == "REVOLUTE":
        return ("rot" + joint.axis,)
    if joint.type == "PRISMATIC":
        return ("trans" + joint.axis,)
    return ()


def inactive_animation(obj):
    joint = obj.ov.physics.joint
    active = active_axis_keys(joint)
    curves = [curve for action in owner_actions(obj) for _, curve in action_curves(action)]
    animation = getattr(obj, "animation_data", None)
    curves += list(getattr(animation, "drivers", ()))
    errors = []
    for curve in curves:
        match = re.search(r'ov\.physics\.joint\.axes\[(?:"([^"\\]+)"|(\d+))\]', curve.data_path)
        if match:
            name = match[1]
            if name is None:
                index = int(match[2])
                name = AXIS_KEYS[index] if index < len(AXIS_KEYS) else "invalid axis"
            if name not in active:
                errors.append(f"Inactive animated joint setting: {curve.data_path}; remap or remove its animation")
    return tuple(errors)


def set_joint_type(obj, scene, joint_type, axis):
    require_stopped(scene)
    require_writable(obj)
    if joint_type not in {"FIXED", "REVOLUTE", "PRISMATIC", "D6", "SPHERICAL", "DISTANCE"} or axis not in "XYZ" or len(axis) != 1:
        raise AuthoringError("Invalid joint type or axis")
    obj.ov.physics.joint.type, obj.ov.physics.joint.axis = joint_type, axis
    mark_dirty(scene, obj)
    return inactive_animation(obj)


def require_scene_ready(scene, *, allow_pending_anchors=False):
    """Shared mandatory admission guard for current-scene simulation and baking."""
    for obj in getattr(scene, "objects", ()):
        joint = getattr(getattr(getattr(obj, "ov", None), "physics", None), "joint", None)
        if joint is None or not joint.enabled:
            continue
        if not allow_pending_anchors and anchor_pending(obj):
            raise AuthoringError(f"{obj.name}: Apply or cancel the pending joint anchor before simulation or baking")
        errors = inactive_animation(obj)
        if errors:
            raise AuthoringError(f"{obj.name}: {errors[0]}")


def repair_collection_animation(owner, records, removed_id=None):
    """Convert numeric paths to stable names and delete only removed channels.

    Validate shared Action safety before any edit. Removal cannot silently move
    another collider's animation; reordered collections also retain their IDs.
    """
    data_owner = getattr(owner, "id_data", owner)
    owner_path = "" if owner == data_owner else owner.path_from_id()
    prefix = (owner_path + "." if owner_path else "") + "ov.physics.colliders"
    expression = re.compile(re.escape(prefix) + r'\[(?:"([^"\\]+)"|(\d+))\]')
    names = [record.name for record in records]
    actions = owner_actions(data_owner)
    edits = []
    animation = getattr(data_owner, "animation_data", None)
    sources = [(action, collection, curve) for action in actions for collection, curve in action_curves(action)]
    sources += [(None, animation.drivers, curve) for curve in getattr(animation, "drivers", ())]
    for action, collection, curve in sources:
        match = expression.search(curve.data_path)
        if not match:
            continue
        if action is not None and action.users > 1:
            raise AuthoringError("Make shared collider animation single-user before collection edits")
        index = int(match[2]) if match[2] else None
        if index is not None and index >= len(names):
            raise AuthoringError("Collider animation index is out of range; repair the animation first")
        name = match[1] or names[index]
        edits.append((collection, curve, None if name == removed_id else
                      curve.data_path[:match.start()] + prefix + '["' + name + '"]' + curve.data_path[match.end():]))
    # Driver variable references can live on a different owner. Normalize
    # surviving references; a removed input needs explicit user repair first.
    external = []
    for other in animation_owners():
        for curve in getattr(getattr(other, "animation_data", None), "drivers", ()):
            for variable in curve.driver.variables:
                for target in variable.targets:
                    if target.id != data_owner:
                        continue
                    match = expression.search(target.data_path)
                    if match is None:
                        continue
                    require_writable(other)
                    index = int(match[2]) if match[2] else None
                    if index is not None and index >= len(names):
                        raise AuthoringError("Collider driver index is out of range; repair it first")
                    name = match[1] or names[index]
                    if name == removed_id:
                        raise AuthoringError("Collider is a driver input; remap or remove that driver before deleting it")
                    external.append((target, target.data_path[:match.start()] + prefix + '["' + name + '"]' + target.data_path[match.end():]))
    for collection, curve, path in edits:
        if path is None:
            collection.remove(curve)
        else:
            curve.data_path = path
    for target, path in external:
        target.data_path = path
