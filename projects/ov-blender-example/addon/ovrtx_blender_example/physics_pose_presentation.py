# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main-thread transient output; ownership is stored on the existing playback lock."""
import threading
from .physics_frames import RigidFrame
from .physics_presentation_targets import target_axis_scale


def main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Physics presentation requires Blender's main thread")


class ResolutionIndex:
    """One-transaction cache: rebuilding invalidates delete/rename/undo/topology.

    Scan the scene once and each touched armature once, independent of body
    count. Never retain RNA references between writable transactions.
    """
    def __init__(self, scene):
        self.objects = {}
        self.bones = {}
        self.object_count = 0
        for obj in scene.objects:
            self.object_count += 1
            key = (getattr(obj, 'session_uid', None),
                   getattr(getattr(getattr(obj, 'ov', None), 'physics', None), 'persistent_id', None))
            self.objects.setdefault(key, []).append(obj)

    def resolve(self, target):
        matches = self.objects.get((target.object_session_uid, target.object_identity), ())
        if len(matches) != 1:
            raise ValueError("Physics presentation Object identity is stale or missing")
        obj = matches[0]
        if not target.bone_identity:
            return obj, obj
        key = target.object_session_uid
        if key not in self.bones:
            index = {}
            for bone in obj.data.bones:
                index.setdefault(bone.ov.persistent_id, []).append(bone)
            self.bones[key] = index
        bones = self.bones[key].get(target.bone_identity, ())
        if len(bones) != 1:
            raise ValueError("Physics presentation bone identity is stale or ambiguous")
        return obj, obj.pose.bones[bones[0].name]


def resolve(scene, target, index=None):
    return (index if index is not None else ResolutionIndex(scene)).resolve(target)


def _curves(animation):
    if animation is None:
        return
    yield from animation.drivers
    actions = [(animation.action, getattr(animation, "action_slot", None))] if animation.action else []
    for track in animation.nla_tracks:
        if not track.mute:
            actions.extend((strip.action, getattr(strip, "action_slot", None))
                           for strip in track.strips if strip.action and not strip.mute)
    for action, slot in actions:
        if not hasattr(action, "slots") or getattr(action, "is_action_legacy", False):
            yield from getattr(action, "fcurves", ())
            continue
        # A modern Action may animate many unrelated IDs. Only this binding's
        # slot evaluates on the output owner; an unbound slot has no channels.
        if slot is None:
            continue
        for layer in getattr(action, "layers", ()):
            for strip in layer.strips:
                getter = getattr(strip, "channelbag", None)
                bag = getter(slot) if callable(getter) else None
                if bag is not None:
                    yield from bag.fcurves


def validate_output(obj, channel):
    if obj.library or obj.override_library or (obj.data and (obj.data.library or obj.data.override_library)):
        raise ValueError("Physics presentation requires writable local output")
    if any(not c.mute for c in channel.constraints):
        raise ValueError(f"{channel.name}: output constraints conflict; use a clean display rig")
    prefix = channel.path_from_id() + "." if channel is not obj else ""
    names = {prefix + name for name in ("location", "rotation_euler", "rotation_quaternion",
                                      "rotation_axis_angle", "scale", "delta_location",
                                      "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")}
    if any(curve.data_path in names and not curve.mute for curve in _curves(obj.animation_data)):
        raise ValueError(f"{channel.name}: output animation or drivers conflict; use a clean display rig")


def _rest(channel, obj):
    if channel is obj:
        return None
    return (tuple(tuple(row) for row in channel.bone.matrix_local), channel.bone.inherit_scale,
            channel.parent.bone.ov.persistent_id if channel.parent else "")


def release(lock):
    main_thread()
    records = getattr(lock, "_presentation_records", {})
    # Basis channels are independent persistent values; restore all, then evaluate.
    for record in records.values():
        try:
            channel = record["channel"]
            channel.matrix_basis = record["initial_basis"].copy()
            # Matrix decomposition is not an exact inverse of RNA channels:
            # it can alter Euler signed zero and inactive rotation values.
            # Preserve the authored components, including their representation.
            for name, value in record["initial_channels"].items():
                setattr(channel, name, value)
            for name, value in record["locks"].items():
                setattr(channel, name, value)
        except (ReferenceError, RuntimeError):
            pass
    records.clear()


def apply(scene, targets, poses, lock, generation, view_layer, *, index=None, evaluate=True):
    """Validate a complete batch before mutation; parents use matching sample poses."""
    main_thread()
    from mathutils import Matrix, Quaternion, Vector
    rows = []
    index = index if index is not None else ResolutionIndex(scene)
    for target in targets:
        if target.body_path not in poses or not target.enabled or target.role != "DYNAMIC":
            continue
        obj, channel = resolve(scene, target, index)
        validate_output(obj, channel)
        rows.append((target, obj, channel))
    records = lock._presentation_records
    if records and any(r["generation"] != generation for r in records.values()):
        release(lock)
        view_layer.update()
    desired = {}
    for target, obj, channel in rows:
        record = records.get(target.body_path)
        if record and record["rest"] != _rest(channel, obj):
            raise ValueError("Bone rest topology changed during presentation; restart required")
        pose = poses[target.body_path]
        q = pose["orient"]
        world = RigidFrame(tuple(pose["translate"]), (q[3], q[0], q[1], q[2])).compose(target.body_offset.inverse())
        world_matrix = Matrix.LocRotScale(Vector(tuple(v / target.meters_per_blender_unit for v in world.position)),
                                        Quaternion(world.rotation_wxyz), Vector(target_axis_scale(target)))
        desired[target.body_path] = world_matrix
    # Parent Object outputs precede child Objects and armature bone outputs.
    def depth(row):
        target, obj, channel = row
        return (0 if channel is obj else 1, len(channel.parent_recursive) if channel is not obj else len(_parents(obj)))
    rows.sort(key=depth)
    bone_paths = {(int(obj.session_uid), channel.name): target.body_path
                  for target, obj, channel in rows if channel is not obj}
    previous_depth = None
    try:
        for target, obj, channel in rows:
            current_depth = depth((target, obj, channel))
            # Evaluate by hierarchy layer so indirect/unbound parents are
            # current without evaluating every independent sibling.
            if previous_depth is not None and current_depth != previous_depth:
                view_layer.update()
            previous_depth = current_depth
            if channel is not obj:
                # Object ancestors (including unbound armatures) were evaluated
                # at the layer boundary. Convert the immutable world goal now,
                # never against the armature's previous-frame world matrix.
                desired[target.body_path] = obj.matrix_world.inverted() @ desired[target.body_path]
            if target.body_path not in records:
                locks = {name: tuple(getattr(channel, name)) for name in ("lock_location", "lock_rotation")}
                if channel is not obj:
                    locks["lock_rotation_w"] = channel.lock_rotation_w
                    locks["lock_rotations_4d"] = channel.lock_rotations_4d
                records[target.body_path] = {"channel": channel, "object": obj, "generation": generation,
                    "initial_basis": channel.matrix_basis.copy(),
                    "initial_channels": {name: tuple(getattr(channel, name)) for name in
                        ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")},
                    "rest": _rest(channel, obj), "locks": locks}
            if channel is obj:
                channel.matrix_world = desired[target.body_path]
            else:
                kwargs = {}
                if channel.parent:
                    parent_path = bone_paths.get((int(obj.session_uid), channel.parent.name))
                    kwargs = {"parent_matrix": desired[parent_path] if parent_path else channel.parent.matrix.copy(),
                              "parent_matrix_local": channel.parent.bone.matrix_local}
                channel.matrix_basis = channel.bone.convert_local_to_pose(desired[target.body_path], channel.bone.matrix_local,
                                                                          invert=True, **kwargs)
            channel.lock_location = (True,) * 3
            channel.lock_rotation = (True,) * 3
            if channel is not obj:
                channel.lock_rotation_w = True
                channel.lock_rotations_4d = True
            records[target.body_path]["display_basis"] = channel.matrix_basis.copy()
        if evaluate:
            view_layer.update()
    except Exception:
        release(lock)
        view_layer.update()
        raise
    return tuple(target.body_path for target, _, _ in rows)


def _parents(obj):
    result = []
    while obj.parent:
        obj = obj.parent
        result.append(obj)
    return result


def enforce(lock):
    """Discard edits to owned channels while preserving unrelated rig controls."""
    main_thread()
    for record in tuple(lock._presentation_records.values()):
        channel, obj = record["channel"], record["object"]
        try:
            validate_output(obj, channel)
            if _rest(channel, obj) != record["rest"]:
                raise ValueError("Bone rest changed; restart required")
            if channel.matrix_basis != record["display_basis"]:
                channel.matrix_basis = record["display_basis"].copy()
        except Exception:
            lock.clear(reason="physics_output_conflict")
            raise
