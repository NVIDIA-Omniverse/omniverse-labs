# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics pose set validation for shared-stage composition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .shared_stage_composition import BodyPose, pose_from_ovphysx_state
from .shared_stage_errors import SharedStageCompositionError


UNKNOWN = "???"


def complete_physics_pose_set(
    states: Sequence[Mapping[str, Any]],
    expected_paths: Sequence[str],
) -> tuple[BodyPose, ...]:
    expected = tuple(str(path) for path in expected_paths)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise SharedStageCompositionError("Physics body prim set contains duplicate paths")
    by_path: dict[str, Mapping[str, Any]] = {}
    for state in states:
        prim_path = str(state.get("prim_path", ""))
        if prim_path not in expected_set:
            raise SharedStageCompositionError(f"Unexpected OVPhysX body pose path: {prim_path or UNKNOWN}")
        if prim_path in by_path:
            raise SharedStageCompositionError(f"Duplicate OVPhysX body pose path: {prim_path}")
        by_path[prim_path] = state
    missing = [path for path in expected if path not in by_path]
    if missing:
        raise SharedStageCompositionError(f"OVPhysX read omitted {len(missing)} body pose(s): {missing[:3]}")
    try:
        return tuple(pose_from_ovphysx_state(by_path[path]) for path in expected)
    except ValueError as exc:
        raise SharedStageCompositionError(str(exc)) from exc


def apply_initial_condition_values(
    poses: Sequence[BodyPose],
    values: Sequence[BodyPose],
) -> tuple[BodyPose, ...]:
    by_path = {pose.prim_path: pose for pose in values}
    result = [by_path.get(pose.prim_path, pose) for pose in poses]
    existing_paths = {pose.prim_path for pose in poses}
    result.extend(pose for pose in values if pose.prim_path not in existing_paths)
    return tuple(result)


__all__ = [
    "apply_initial_condition_values",
    "complete_physics_pose_set",
]
