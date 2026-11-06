# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame-independent configuration required by captured local conversion."""
import hashlib
import json

TRANSFORMS = ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")
DELTAS = ("delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")


def _rna(value):
    result = []
    for prop in value.bl_rna.properties:
        name = prop.identifier
        if name == "rna_type":
            continue
        if ((name == "influence" and getattr(value, "use_animated_influence", False)) or
            (name == "strip_time" and getattr(value, "use_animated_time", False))):
            # Definitions live in strip.fcurves; these RNA values are evaluated.
            continue
        child = getattr(value, name)
        if prop.type == "COLLECTION":
            result.append((name, tuple(_rna(item) for item in child)))
        elif not prop.is_readonly:
            if prop.type == "POINTER":
                child = (getattr(getattr(child, "bl_rna", None), "identifier", ""), getattr(child, "name_full", getattr(child, "name", "")))
            elif getattr(prop, "is_array", False):
                child = tuple(child)
            result.append((name, child))
    return result


def channel_configuration(obj, channel, names, *, allow_drivers=False):
    """Compare static components, but definitions (not evaluated values) of animation.

    Delta drivers and constrained unchanged ancestors are rejected separately:
    arbitrary external dependencies cannot be certified from a pose capture.
    """
    from .physics_pose_presentation import _curves
    from .physics_bake_capture import BakeCaptureError
    prefix = "" if obj is channel else channel.path_from_id() + "."
    paths = {prefix + name: name for name in names}
    animated = set()
    definitions = []
    animation = obj.animation_data
    for curve in _curves(animation):
        if curve.data_path not in paths or curve.mute:
            continue
        if curve.driver:
            if not allow_drivers:
                raise BakeCaptureError("bake conversion requires channels without external drivers")
            definitions.append(("driver", _rna(curve.driver)))
        animated.add((paths[curve.data_path], curve.array_index))
        definitions.append(_rna(curve))
    binding = None
    if definitions:
        def relevant(strip):
            action = strip.action
            if not action:
                return False
            if getattr(action, "is_action_legacy", False) and hasattr(action, "fcurves"):
                return any(c.data_path in paths for c in action.fcurves)
            slot = getattr(strip, "action_slot", None)
            return any(c.data_path in paths for layer in action.layers for item in layer.strips
                       if slot and (bag := item.channelbag(slot)) for c in bag.fcurves)
        tracks = [(track, [strip for strip in track.strips if relevant(strip)]) for track in animation.nla_tracks]
        binding = (getattr(animation.action, "name_full", ""),
                   getattr(getattr(animation, "action_slot", None), "identifier", ""),
                   animation.action_blend_type, animation.action_extrapolation,
                   animation.action_influence, animation.use_nla,
                   tuple((track.name, track.mute, track.is_solo,
                          tuple(_rna(strip) for strip in strips)) for track, strips in tracks if strips))
    # Matrix/channel round trips can change the sign of an exact zero without
    # changing the authored transform. Preserve every nonzero value exactly.
    static = [(name, tuple(None if (name, index) in animated else
                          (0.0 if value == 0.0 else float(value))
                          for index, value in enumerate(getattr(channel, name)))) for name in names]
    return hashlib.sha256(json.dumps((static, definitions, binding), sort_keys=True).encode()).hexdigest()
