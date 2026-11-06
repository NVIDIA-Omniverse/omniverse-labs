# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit clean destination contexts for captured body poses, without retargeting."""
from dataclasses import asdict, replace
import shutil
from types import SimpleNamespace
import uuid
from pathlib import Path

from .physics_bake_capture import BakeCaptureError, _SampleWriter, _conversion_context, _main_thread
from .physics_bake_conversion import _target


def remap_capture(scene, capture, mappings, directory):
    previous = [(row.destination, row.destination.pose.bones.get(row.destination_bone),
                 row.destination.pose.bones[row.destination_bone].bone.ov.persistent_id)
                for row in mappings if row.destination and row.destination.pose and
                row.destination_bone in row.destination.pose.bones]
    try:
        return _remap_capture(scene, capture, mappings, directory)
    except BaseException:
        for obj, bone, identity in previous:
            bone.bone.ov.persistent_id = identity
        raise


def rollback_mapping_identities(scene, capture):
    for row in capture.manifest().get("initialized_bones", ()):
        obj = next((o for o in scene.objects if o.session_uid == row["object_uid"]), None)
        bone = obj.pose.bones.get(row["bone"]) if obj and obj.pose else None
        if bone and bone.bone.ov.persistent_id == row["identity"]:
            bone.bone.ov.persistent_id = ""


def _remap_capture(scene, capture, mappings, directory):
    _main_thread()
    import bpy
    from .physics_pose_presentation import resolve
    from .physics_bake_actions import _validate_owner
    from .interactive_operator_state import suppress_interactive_edit_bridge
    original = capture.manifest()
    targets = {_target(row).body_path: _target(row) for row in original["targets"]}
    destinations = []
    initialized = []
    seen = set()
    for path in original["output_paths"]:
        target = targets[path]
        source, channel = resolve(scene, target)
        rows = [row for row in mappings if row.source == source and row.source_bone == target.bone_name]
        if len(rows) != 1 or not rows[0].destination:
            raise BakeCaptureError(f"{source.name}/{target.bone_name}: exactly one clean output mapping required")
        row = rows[0]
        obj = row.destination
        dest = obj.pose.bones.get(row.destination_bone) if row.destination_bone and obj.pose else obj
        if row.destination_bone and (dest is obj or dest is None):
            raise BakeCaptureError("mapped destination bone is missing")
        if obj == source:
            raise BakeCaptureError("clean output must use a different Object/armature")
        identity = obj.session_uid, row.destination_bone
        if identity in seen:
            raise BakeCaptureError("two bodies cannot write the same clean output")
        seen.add(identity)
        names = {"location", "scale", "rotation_quaternion", "rotation_euler", "rotation_axis_angle"}
        _validate_owner(obj, [(path, dest, names)])
        if dest is not obj and not dest.bone.ov.persistent_id:
            dest.bone.ov.persistent_id = uuid.uuid4().hex
            initialized.append(dict(object_uid=obj.session_uid, bone=dest.name, identity=dest.bone.ov.persistent_id))
        from .physics_frames import evaluated_owner_transform, PhysicsUnits
        transform = evaluated_owner_transform(obj, bpy.context.evaluated_depsgraph_get(),
            bone_name=dest.name if dest is not obj else "", units=PhysicsUnits(target.meters_per_blender_unit))
        destinations.append(replace(target, object_session_uid=obj.session_uid,
            object_identity=obj.ov.physics.persistent_id,
            bone_identity=dest.bone.ov.persistent_id if dest is not obj else "",
            bone_name=dest.name if dest is not obj else "", owner_uniform_scale=transform.uniform_scale,
            owner_axis_scale=transform.axis_scale))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    writer = _SampleWriter(directory, original["body_paths"])
    saved_frame = scene.frame_current, scene.frame_subframe
    saved = [(channel, {name: tuple(getattr(channel, name)) for name in
                       ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")}) for obj in scene.objects
             for channel in (obj, *getattr(obj.pose, "bones", ()))]
    try:
        with suppress_interactive_edit_bridge():
            for sample in capture.samples():
                scene.frame_set(sample["frame"])
                for layer in scene.view_layers:
                    layer.update()
                poses = [SimpleNamespace(prim_path=path, translate=pose["position"], orient=pose["rotation_xyzw"])
                         for path, pose in sample["poses"].items()]
                writer.append(sample["frame"], sample["step"], poses, _conversion_context(scene, destinations),
                              original["timing"]["timestep_ns"])
        manifest = dict(original, targets=[asdict(t) for t in destinations],
                        remapped_from_digest=capture.manifest_digest,
                        initialized_bones=initialized,
                        source_targets=original["targets"])
        return writer.finish(manifest)
    except BaseException:
        writer.abort()
        if directory.exists():
            shutil.rmtree(directory)
        raise
    finally:
        with suppress_interactive_edit_bridge():
            scene.frame_set(saved_frame[0], subframe=saved_frame[1])
            for channel, values in saved:
                for name, value in values.items():
                    setattr(channel, name, value)
            for layer in scene.view_layers:
                layer.update()
