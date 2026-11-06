# SPDX-License-Identifier: Apache-2.0
"""Copied evaluated kinematic targets, separate from dynamic output ownership."""
from dataclasses import dataclass
import math
import threading

from .physics_frames import PhysicsUnits, RigidFrame, evaluated_owner_transform
from .physics_presentation_targets import presentation_targets, target_axis_scale
from .shared_stage_composition import BodyPose


@dataclass(frozen=True)
class KinematicInputs:
    generation_digest: str
    frame: float
    poses: tuple = ()

    def __post_init__(self):
        if not math.isfinite(self.frame):
            raise ValueError("Kinematic input frame must be finite")
        copied = []
        for pose in self.poses:
            if not isinstance(pose, BodyPose):
                raise ValueError("Kinematic inputs require typed body poses")
            validated = BodyPose(pose.prim_path, tuple(pose.translate), tuple(pose.orient))
            norm = math.hypot(*validated.orient)
            copied.append(BodyPose(validated.prim_path, validated.translate, tuple(v/norm for v in validated.orient)))
        object.__setattr__(self, "poses", tuple(copied))


def kinematic_targets(generation):
    return tuple(t for t in presentation_targets(generation) if t.enabled and t.role == "KINEMATIC")


def _validate_authored_dependencies(obj, channel, dynamic):
    """Bounded authored hierarchy only; uncertain constraint/driver graphs reject.

    This deliberately does not try to traverse arbitrary Blender dependency
    graphs (IK poles, multi-target constraints and scripted drivers included).
    Pure keyed Object/bone hierarchies are supported without output feedback.
    """
    owner, bone = obj, channel if channel is not obj else None
    while owner is not None:
        if (int(owner.session_uid), "") in dynamic:
            raise ValueError("Kinematic input depends on a dynamic output parent")
        if getattr(getattr(owner, "animation_data", None), "drivers", ()):
            raise ValueError("Kinematic input drivers are unavailable: output feedback cannot be excluded")
        if any(not c.mute for c in owner.constraints):
            raise ValueError("Kinematic input constraints are unavailable: dynamic output feedback cannot be excluded")
        # A sibling IK constraint can alter an input's ancestor without living
        # on its direct chain. Reject constrained rigs as a whole, including
        # nested/pole targets, instead of guessing an incomplete dependency set.
        if any(not c.mute for p in getattr(owner.pose, "bones", ()) for c in p.constraints):
            raise ValueError("Kinematic rig constraints are unavailable: indirect output feedback cannot be excluded")
        while bone is not None:
            if (int(owner.session_uid), bone.bone.ov.persistent_id) in dynamic:
                raise ValueError("Kinematic input depends on a dynamic output bone")
            if any(not c.mute for c in bone.constraints):
                raise ValueError("Kinematic input constraints are unavailable: dynamic output feedback cannot be excluded")
            bone = bone.parent
        bone = (owner.parent.pose.bones.get(owner.parent_bone)
                if owner.parent is not None and owner.parent_type == "BONE" and owner.parent.pose else None)
        owner = owner.parent


def capture_kinematic_inputs(scene, generation, depsgraph=None):
    """Main-thread evaluated RNA becomes immutable meter/XYZW input values."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("Kinematic evaluation requires the main thread")
    if depsgraph is None:
        import bpy
        depsgraph = bpy.context.evaluated_depsgraph_get()
    units = PhysicsUnits.from_scene(scene)
    objects = {int(obj.session_uid): obj for obj in scene.objects}
    dynamic = {(t.object_session_uid, t.bone_identity) for t in presentation_targets(generation)
               if t.enabled and t.role == "DYNAMIC"}
    poses = []
    for target in kinematic_targets(generation):
        obj = objects.get(target.object_session_uid)
        if obj is None or obj.ov.physics.persistent_id != target.object_identity:
            raise ValueError("Kinematic owner identity changed; restart simulation")
        if sum(other.ov.physics.persistent_id == target.object_identity for other in scene.objects) != 1:
            raise ValueError("Kinematic Object identity is ambiguous; repair duplicates")
        if units.meters_per_blender_unit != target.meters_per_blender_unit:
            raise ValueError("Kinematic units changed; restart simulation")
        bone_name = None
        channel = obj
        if target.bone_identity:
            matches = [p for p in obj.pose.bones if p.bone.ov.persistent_id == target.bone_identity]
            if len(matches) != 1:
                raise ValueError("Kinematic bone identity is missing or ambiguous")
            channel = matches[0]
            bone_name = channel.name
        body = channel.ov.physics.rigid_body
        if body.schema_opinion != "APPLY" or body.role != "KINEMATIC" or not body.enabled:
            raise ValueError("Kinematic input role changed; restart simulation")
        _validate_authored_dependencies(obj, channel, dynamic)
        transform = evaluated_owner_transform(obj, depsgraph, bone_name=bone_name, units=units)
        if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(transform.axis_scale, target_axis_scale(target))):
            raise ValueError("Animated kinematic scale is unavailable; restart simulation")
        from .physics_authoring import read_frame
        authored_offset = read_frame(body.offset)
        offset = RigidFrame(tuple(units.length(v*s) for v, s in zip(authored_offset.position, transform.axis_scale)), authored_offset.rotation_wxyz)
        if math.dist(offset.position, target.body_offset.position) > 1e-7 or abs(sum(a*b for a,b in zip(offset.rotation_wxyz, target.body_offset.rotation_wxyz))) < 1-1e-10:
            raise ValueError("Kinematic body offset changed; restart simulation")
        frame = transform.frame.compose(target.body_offset)
        poses.append(BodyPose(target.body_path, *frame.bridge_pose()))
    return KinematicInputs(generation.digest, float(scene.frame_current)+float(scene.frame_subframe), tuple(poses))


def apply_kinematic_inputs(controller, inputs, generation):
    """Session-owner thread, immediately before one native step; never reset."""
    if inputs.generation_digest != generation.digest:
        raise ValueError("Stale kinematic input generation")
    expected = {t.body_path for t in kinematic_targets(generation)}
    paths = [p.prim_path for p in inputs.poses]
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise ValueError("Kinematic input roster is incomplete or contains non-kinematic bodies")
    if set(paths) & set(controller.config.body_prims):
        raise ValueError("Dynamic output cannot be used as kinematic input")
    if not paths:
        return {}
    if not controller.started or controller.failed or controller.closed:
        raise ValueError("Kinematic input requires an active simulation")
    # The generation's immutable composition is the authority, not a caller's
    # role label. Cache only on the owning controller and generation lifetime.
    key = (generation.digest, tuple(sorted(expected)))
    if getattr(controller, "_validated_kinematic_roster", None) != key:
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.Open(controller.config.input_usd_path)
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    or prim.GetAttribute("physics:kinematicEnabled").Get() is not True
                    or prim.GetAttribute("physics:rigidBodyEnabled").Get() is False):
                raise ValueError("Input target is not an enabled native kinematic body: " + path)
        controller._validated_kinematic_roster = key
    from .ovphysx_stage import _physics_write_error
    try:
        from .ovphysx_stage import parent_local_poses
        diagnostic = controller._physics_client.write_body_poses(parent_local_poses(controller, inputs.poses),
            simulation_time_ns=max(2, int(controller.current_simulation_time_ns)+1), reset=False)
        if error := _physics_write_error(diagnostic, len(paths)):
            raise ValueError("Kinematic input rejected: " + error)
    except Exception as exc:
        # A rejected/partial write may already have changed solver input. Never
        # retry against that state or publish it as an acknowledged target.
        controller.failed = controller.completed = True
        controller.last_error = str(exc)
        raise
    return diagnostic
