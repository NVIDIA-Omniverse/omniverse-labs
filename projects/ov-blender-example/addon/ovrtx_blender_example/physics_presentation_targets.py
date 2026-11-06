# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable generation-owned body-to-Object/bone mappings, with no RNA handles."""
from dataclasses import dataclass
from .physics_frames import RigidFrame


@dataclass(frozen=True)
class PhysicsPresentationTarget:
    body_path: str
    body_identity: str
    object_session_uid: int
    object_identity: str
    bone_identity: str
    bone_name: str
    role: str
    enabled: bool
    meters_per_blender_unit: float
    owner_uniform_scale: float
    body_offset: RigidFrame
    initial_body_frame: RigidFrame
    owner_axis_scale: tuple = None

    @property
    def axis_scale(self):
        """Per-axis owner scale; nonuniform scale is baked into colliders."""
        return tuple(self.owner_axis_scale) if self.owner_axis_scale else (self.owner_uniform_scale,) * 3

    @property
    def kind(self):
        return "BONE" if self.bone_identity else "OBJECT"


def target_axis_scale(target):
    """Per-axis owner scale of any presentation target (uniform when unspecified)."""
    axis = getattr(target, "owner_axis_scale", None)
    return tuple(axis) if axis else (target.owner_uniform_scale,) * 3


def presentation_targets(generation):
    """Use stable bone identity, not the diagnostic name, when resolving RNA.

    Frames/offsets are rigid meters. Offset translations already include owner
    uniform scale and scene units; consumers must not multiply either twice.
    The owning generation supplies lifetime/version identity to its consumers.
    """
    result = []
    for record in generation.opinion_records:
        data = record.diagnostics
        if "presentation_object_uid" not in data:
            continue
        result.append(PhysicsPresentationTarget(
            record.usd_prim_path, data["physics_owner"], int(data["presentation_object_uid"]),
            data["presentation_object_id"], data["presentation_bone_id"], data["presentation_bone_name"],
            data["body_role"], data["body_enabled"], data["meters_per_blender_unit"], data["owner_uniform_scale"],
            RigidFrame(*data["body_offset_meters"]), RigidFrame(*data["initial_body_frame"]),
            tuple(data["owner_axis_scale"]) if "owner_axis_scale" in data else None))
    return tuple(result)
