# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only world-pose conversion into validated, disk-backed local channels.

No Action, frame, or RNA value is changed here.  Only selected outputs replace
parent motion; every other parent retains its captured evaluated motion.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .physics_bake_capture import BakeCaptureArtifact, BakeCaptureError, _main_thread
from .physics_presentation_targets import target_axis_scale
from .physics_frames import RigidFrame
from .physics_presentation_targets import PhysicsPresentationTarget

POSITION_TOLERANCE_M = 1e-4
ROTATION_TOLERANCE_RAD = 1e-4


@dataclass(frozen=True)
class BakeConversionArtifact(BakeCaptureArtifact):
    """Same verified manifest()/samples() streaming contract as capture."""


def _target(data):
    data = dict(data)
    for name in ("body_offset", "initial_body_frame"):
        data[name] = RigidFrame(**data[name])
    return PhysicsPresentationTarget(**data)


def _inverse(matrix):
    try:
        return matrix.inverted()
    except ValueError as exc:
        raise BakeCaptureError("singular transform cannot be baked") from exc


def _world(target, pose):
    from mathutils import Matrix, Quaternion, Vector
    q = pose["rotation_xyzw"]
    owner = RigidFrame(tuple(pose["position"]), (q[3], *q[:3])).compose(target.body_offset.inverse())
    return Matrix.LocRotScale(Vector(tuple(x / target.meters_per_blender_unit for x in owner.position)),
        Quaternion(owner.rotation_wxyz), Vector(target_axis_scale(target)))


def _delta(context, bone):
    from mathutils import Quaternion, Euler, Vector
    if bone:
        return Vector((0, 0, 0)), Quaternion(), Vector((1, 1, 1))
    mode = context["rotation_mode"]
    rotation = (Quaternion(context["delta_rotation_quaternion"]) if mode == "QUATERNION" else
                Quaternion() if mode == "AXIS_ANGLE" else
                Euler(context["delta_rotation_euler"], mode).to_quaternion())
    return Vector(context["delta_location"]), rotation, Vector(context["delta_scale"])


def _channels(basis, context, bone, previous):
    from mathutils import Matrix, Quaternion, Vector
    location, rotation, scale = basis.decompose()
    if any(v <= 0 or not math.isfinite(v) for v in scale):
        raise BakeCaptureError("bake requires positive nonsingular local scale")
    rebuilt = Matrix.LocRotScale(location, rotation, scale)
    if max(abs(basis[i][j] - rebuilt[i][j]) for i in range(4) for j in range(4)) > 1e-5:
        raise BakeCaptureError("local shear cannot be represented by animation channels")
    dloc, drot, dscale = _delta(context, bone)
    if any(abs(x) < 1e-12 for x in dscale):
        raise BakeCaptureError("zero delta scale cannot be baked")
    location -= dloc
    rotation = drot.inverted() @ rotation
    rotation.normalize()
    if previous is not None and rotation.dot(Quaternion(previous["quaternion"])) < 0:
        rotation.negate()
    mode = context["rotation_mode"]
    result = {"location": tuple(location), "scale": tuple(scale[i] / dscale[i] for i in range(3))}
    if mode == "QUATERNION":
        result["rotation_quaternion"] = tuple(rotation)
    elif mode == "AXIS_ANGLE":
        axis, angle = rotation.to_axis_angle()
        result["rotation_axis_angle"] = (angle, *axis)
    else:
        from mathutils import Euler
        compatible = Euler(previous["euler"], mode) if previous else None
        euler = rotation.to_euler(mode, compatible) if compatible else rotation.to_euler(mode)
        result["rotation_euler"] = tuple(euler)
    state = {"quaternion": tuple(rotation), "euler": result.get("rotation_euler")}
    if mode == "AXIS_ANGLE":
        angle, *axis = result["rotation_axis_angle"]
        serialized_rotation = Quaternion(axis, angle)
    elif mode == "QUATERNION":
        serialized_rotation = Quaternion(result["rotation_quaternion"])
    else:
        serialized_rotation = Euler(result["rotation_euler"], mode).to_quaternion()
    reconstructed = Matrix.LocRotScale(Vector(result["location"]) + dloc, drot @ serialized_rotation,
        Vector(tuple(result["scale"][i] * dscale[i] for i in range(3))))
    return result, state, reconstructed


def _bone_pose(channel, context, basis, parent_pose, *, invert):
    from mathutils import Matrix
    kwargs = {}
    if context["parent_rest"] is not None:
        kwargs = dict(parent_matrix=parent_pose, parent_matrix_local=Matrix(context["parent_rest"]))
    return channel.bone.convert_local_to_pose(basis, Matrix(context["rest"]), invert=invert, **kwargs)


def _error(world, desired, meters):
    position = (world.translation - desired.translation).length * meters
    # Double precision dot avoids float32 acos falsely reporting >1e-4 near identity.
    a, b = world.to_quaternion(), desired.to_quaternion()
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    norm = math.sqrt(sum(float(x) ** 2 for x in a) * sum(float(x) ** 2 for x in b))
    rotation = 2 * math.acos(min(1.0, dot / norm))
    return position, rotation


def _body_error(world, desired, target):
    from mathutils import Vector
    actual_body, desired_body = world.copy(), desired.copy()
    offset = Vector(target.body_offset.position) / target.meters_per_blender_unit
    actual_body.translation += world.to_quaternion() @ offset
    desired_body.translation += desired.to_quaternion() @ offset
    return _error(actual_body, desired_body, target.meters_per_blender_unit)


def convert_capture(scene, capture, directory):
    """Convert an intact capture; return only after all samples are hash checked.

    A failed conversion removes only its newly created private directory.
    """
    _main_thread()
    import shutil
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        return _convert_capture(scene, capture, directory)
    except BaseException:
        shutil.rmtree(directory)
        raise


def _convert_capture(scene, capture, directory):
    from mathutils import Matrix
    from .physics_pose_presentation import resolve
    from .physics_bake_configuration import channel_configuration, DELTAS
    source = capture.manifest()
    targets = {_target(row).body_path: _target(row) for row in source["targets"]}
    outputs = tuple(source["output_paths"])
    if not outputs or len(set(outputs)) != len(outputs) or not set(outputs) <= targets.keys():
        raise BakeCaptureError("invalid bake output selection")
    resolved = {path: resolve(scene, targets[path]) for path in outputs}
    for obj, channel in resolved.values():
        if any(not c.mute for c in channel.constraints):
            raise BakeCaptureError("constrained output requires a clean bake destination")
        if obj.library or obj.override_library:
            raise BakeCaptureError("bake output must be a local editable Object")
    objects = {t.object_session_uid: p for p, t in targets.items() if p in outputs and not t.bone_identity}
    bones = {(t.object_session_uid, t.bone_name): p for p, t in targets.items() if p in outputs and t.bone_identity}
    digest, count, previous = hashlib.sha256(), 0, {}
    max_position = max_rotation = 0.0
    with (directory / "samples.jsonl").open("wb") as sink:
        for sample in capture.samples():
            desired = {p: _world(targets[p], sample["poses"][p]) for p in outputs}
            hierarchy = _Hierarchy(scene, sample["context"], desired, objects, bones)
            converted = {}
            for path in outputs:
                target = targets[path]
                obj, channel = resolved[path]
                context = sample["context"][path]
                if channel.rotation_mode != context["rotation_mode"]:
                    raise BakeCaptureError("destination rotation mode changed after capture")
                if (int(obj.parent.session_uid) if obj.parent else 0) != context["parent_object_uid"]:
                    raise BakeCaptureError("Object parent changed after capture")
                if target.bone_identity:
                    if (channel.bone.inherit_scale not in {"FULL", "NONE"} or
                        channel.bone.inherit_scale != context["inherit_scale"] or
                        channel.bone.use_local_location != context["use_local_location"] or
                        channel.bone.use_inherit_rotation != context["use_inherit_rotation"] or
                        (channel.parent.name if channel.parent else "") != context["parent_bone"] or
                        any(abs(channel.bone.matrix_local[i][j] - context["rest"][i][j]) > 1e-6
                            for i in range(4) for j in range(4))):
                        raise BakeCaptureError("bone rest or inheritance changed or is unsupported")
                    armature = (hierarchy.object(target.object_session_uid) if hierarchy.available else
                                desired[objects[target.object_session_uid]] if target.object_session_uid in objects else Matrix(context["object_world"]))
                    parent_path = bones.get((target.object_session_uid, context["parent_bone"]))
                    parent = (hierarchy.bone(target.object_session_uid, context["parent_bone"]) if hierarchy.available and context["parent_bone"] else
                              _inverse(armature) @ desired[parent_path] if parent_path else
                              Matrix(context["parent_pose"]) if context["parent_pose"] else None)
                    basis = _bone_pose(channel, context, _inverse(armature) @ desired[path], parent, invert=True)
                else:
                    if "delta_configuration" in context:
                        if channel_configuration(obj, obj, DELTAS) != context["delta_configuration"]:
                            raise BakeCaptureError("destination delta configuration changed after capture")
                    elif any(tuple(getattr(obj, name)) != tuple(context[name]) for name in DELTAS):
                        raise BakeCaptureError("legacy capture delta configuration differs; recapture required")
                    if (obj.parent_type != context["parent_type"] or obj.parent_bone != context["parent_bone"] or
                        any(abs(obj.matrix_parent_inverse[i][j] - context["parent_inverse"][i][j]) > 1e-6
                            for i in range(4) for j in range(4))):
                        raise BakeCaptureError("Object parent configuration changed after capture")
                    if context["parent_type"] not in {"OBJECT", "BONE"}:
                        raise BakeCaptureError("unsupported Object parenting arrangement")
                    # Captured evaluated effective parent includes inverse, bone-tail,
                    # and relative-parent conventions without guessing their order.
                    effective = Matrix(context["object_world"]) @ _inverse(Matrix(context["basis"]))
                    parent_uid = context["parent_object_uid"]
                    parent_path = objects.get(parent_uid)
                    bone_path = bones.get((parent_uid, context["parent_bone"])) if context["parent_type"] == "BONE" else None
                    if hierarchy.available and parent_uid:
                        old = Matrix(context["parent_world"])
                        new = hierarchy.object(parent_uid)
                        if context["parent_type"] == "BONE":
                            old = old @ Matrix(context["object_parent_bone_pose"])
                            new = new @ hierarchy.bone(parent_uid, context["parent_bone"])
                        effective = new @ _inverse(old) @ effective
                    elif bone_path:
                        old = Matrix(context["parent_world"]) @ Matrix(context["object_parent_bone_pose"])
                        effective = desired[bone_path] @ _inverse(old) @ effective
                    elif parent_path:
                        effective = desired[parent_path] @ _inverse(Matrix(context["parent_world"])) @ effective
                    basis = _inverse(effective) @ desired[path]
                values, previous[path], reconstructed = _channels(basis, context, bool(target.bone_identity), previous.get(path))
                world = (armature @ _bone_pose(channel, context, reconstructed, parent, invert=False)
                         if target.bone_identity else effective @ reconstructed)
                pos, rot = _body_error(world, desired[path], target)
                if pos > POSITION_TOLERANCE_M or rot > ROTATION_TOLERANCE_RAD:
                    raise BakeCaptureError(f"{path}: reconstruction exceeds tolerance ({pos:g} m, {rot:g} rad)")
                max_position, max_rotation = max(max_position, pos), max(max_rotation, rot)
                converted[path] = values
            line = (json.dumps({"frame": sample["frame"], "channels": converted}, sort_keys=True, allow_nan=False) + "\n").encode()
            sink.write(line)
            digest.update(line)
            count += 1
    if not count:
        raise BakeCaptureError("empty bake capture")
    manifest = {"format_version": 1, "kind": "local_channels", "capture_digest": capture.manifest_digest,
        "capture_directory": str(capture.directory), "targets": source["targets"], "output_paths": outputs,
        "rotation_modes": {p: c.rotation_mode for p, (_, c) in resolved.items()}, "interpolation": "LINEAR",
        "timing": source["timing"], "sample_count": count, "samples_sha256": digest.hexdigest(),
        "max_position_error_m": max_position, "max_rotation_error_rad": max_rotation,
        "position_tolerance_m": POSITION_TOLERANCE_M, "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD}
    data = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode()
    (directory / "manifest.json").write_bytes(data)
    return BakeConversionArtifact(directory, hashlib.sha256(data).hexdigest())


def validate_captured_configuration(scene, capture):
    """Revalidate captured immutable inputs without changing frame or channels."""
    from .physics_pose_presentation import resolve
    source = capture.manifest()
    outputs = set(source["output_paths"])
    targets = [_target(row) for row in source["targets"] if row["body_path"] in outputs]
    objects = {t.object_session_uid: t.body_path for t in targets if not t.bone_identity}
    bones = {(t.object_session_uid, t.bone_name): t.body_path for t in targets if t.bone_identity}
    for sample in capture.samples():
        _Hierarchy(scene, sample["context"], {}, objects, bones)
        for target in targets:
            _, channel = resolve(scene, target)
            if channel.rotation_mode != sample["context"][target.body_path]["rotation_mode"]:
                raise BakeCaptureError("destination rotation mode changed after capture")


class _Hierarchy:
    """Evaluate unchanged intermediate channels under selected ancestor poses."""
    def __init__(self, scene, contexts, desired, objects, bones):
        from .physics_bake_configuration import channel_configuration, TRANSFORMS, DELTAS
        self.scene, self.desired, self.selected_objects, self.selected_bones = scene, desired, objects, bones
        self.objects, self.bones, self.object_cache, self.bone_cache = {}, {}, {}, {}
        for context in contexts.values():
            if not context.get("hierarchy_objects"):
                raise BakeCaptureError("capture lacks ancestor configuration; recapture required")
            self.objects.update((row["uid"], row) for row in context.get("hierarchy_objects", ()))
            self.bones.update(((row["object_uid"], row["name"]), row) for row in context.get("hierarchy_bones", ()))
        self.available = bool(self.objects)
        live = {int(o.session_uid): o for o in scene.objects}
        for uid, row in self.objects.items():
            obj = live.get(uid)
            if (obj is None or (int(obj.parent.session_uid) if obj.parent else 0) != row["parent_uid"] or
                obj.parent_type != row["parent_type"] or obj.parent_bone != row["parent_bone"]):
                raise BakeCaptureError("captured Object hierarchy changed")
            if "parent_inverse" not in row:
                raise BakeCaptureError("capture lacks ancestor configuration; recapture required")
            if (obj.ov.physics.persistent_id != row["identity"] or obj.rotation_mode != row["rotation_mode"] or
                any(abs(obj.matrix_parent_inverse[i][j] - row["parent_inverse"][i][j]) > 1e-6
                    for i in range(4) for j in range(4)) or
                channel_configuration(obj, obj, DELTAS) != row["delta_configuration"]):
                raise BakeCaptureError("captured Object configuration changed")
            if uid not in self.selected_objects:
                if (any(not c.mute for c in obj.constraints) or
                    channel_configuration(obj, obj, TRANSFORMS + DELTAS) != row["configuration"]):
                    raise BakeCaptureError("unchanged ancestor channels or constraints differ from capture")
        for (uid, name), row in self.bones.items():
            channel = live[uid].pose.bones.get(name)
            if (channel is None or (channel.parent.name if channel.parent else "") != row["parent"] or
                any(getattr(channel.bone, flag) != row[flag] for flag in
                    ("inherit_scale", "use_local_location", "use_inherit_rotation")) or
                any(abs(channel.bone.matrix_local[i][j] - row["rest"][i][j]) > 1e-6
                    for i in range(4) for j in range(4))):
                raise BakeCaptureError("captured bone hierarchy or rest changed")
            if (channel.bone.ov.persistent_id != row["identity"] or
                channel.bone.use_relative_parent != row["use_relative_parent"]):
                raise BakeCaptureError("captured bone identity or relative parenting changed")
            if (uid, name) not in self.selected_bones:
                if (any(not c.mute for c in channel.constraints) or
                    channel_configuration(live[uid], channel, TRANSFORMS) != row["configuration"]):
                    raise BakeCaptureError("unchanged bone channels or constraints differ from capture")

    def object(self, uid):
        from mathutils import Matrix
        if uid in self.selected_objects:
            return self.desired[self.selected_objects[uid]]
        if uid not in self.object_cache:
            row = self.objects[uid]
            world = Matrix(row["world"])
            if row["parent_uid"]:
                parent = row["parent_uid"]
                old, new = Matrix(self.objects[parent]["world"]), self.object(parent)
                if row["parent_type"] == "BONE":
                    old = old @ Matrix(self.bones[parent, row["parent_bone"]]["pose"])
                    new = new @ self.bone(parent, row["parent_bone"])
                elif row["parent_type"] != "OBJECT":
                    raise BakeCaptureError("unsupported intermediate parent type")
                changed = max(abs(old[i][j] - new[i][j]) for i in range(4) for j in range(4)) > 1e-6
                if changed and row["constrained"]:
                    raise BakeCaptureError("constrained intermediate parent requires a clean bake destination")
                world = new @ _inverse(old) @ world
            self.object_cache[uid] = world
        return self.object_cache[uid]

    def bone(self, uid, name):
        from mathutils import Matrix
        key = (uid, name)
        if key in self.selected_bones:
            return _inverse(self.object(uid)) @ self.desired[self.selected_bones[key]]
        if key not in self.bone_cache:
            row = self.bones[key]
            pose = Matrix(row["pose"])
            if row["parent"]:
                parent_row = self.bones[uid, row["parent"]]
                parent = self.bone(uid, row["parent"])
                changed = max(abs(parent[i][j] - parent_row["pose"][i][j]) for i in range(4) for j in range(4)) > 1e-6
                if changed:
                    if row["constrained"]:
                        raise BakeCaptureError("constrained intermediate bone requires a clean bake destination")
                    obj = next(o for o in self.scene.objects if int(o.session_uid) == uid)
                    channel = obj.pose.bones[name]
                    if any(getattr(channel.bone, flag) != row[flag] for flag in
                           ("inherit_scale", "use_local_location", "use_inherit_rotation")):
                        raise BakeCaptureError("intermediate bone inheritance changed")
                    pose = channel.bone.convert_local_to_pose(Matrix(row["basis"]), Matrix(row["rest"]),
                        parent_matrix=parent, parent_matrix_local=Matrix(parent_row["rest"]))
            self.bone_cache[key] = pose
        return self.bone_cache[key]
