# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.physics_pose_set import (  # noqa: E402
    apply_initial_condition_values,
    complete_physics_pose_set,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402
from ovrtx_blender_example.shared_stage_errors import SharedStageCompositionError  # noqa: E402


def _state(path: str, *, y: float = 5.0, orient: bool = True) -> dict[str, object]:
    state: dict[str, object] = {
        "prim_path": path,
        "translate": {"found": True, "x": 1.0, "y": y, "z": 3.0},
    }
    if orient:
        state["orient"] = {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0}
    return state


def _pose(path: str, *, y: float) -> BodyPose:
    return BodyPose(
        prim_path=path,
        translate=(1.0, y, 3.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )


def test_complete_physics_pose_set_returns_expected_path_order() -> None:
    poses = complete_physics_pose_set(
        [
            _state("/World/BodyB", y=2.0),
            _state("/World/BodyA", y=1.0),
        ],
        ["/World/BodyA", "/World/BodyB"],
    )

    assert poses == (
        _pose("/World/BodyA", y=1.0),
        _pose("/World/BodyB", y=2.0),
    )


def test_complete_physics_pose_set_rejects_duplicate_expected_paths() -> None:
    with pytest.raises(SharedStageCompositionError, match="duplicate paths"):
        complete_physics_pose_set([_state("/World/BodyA")], ["/World/BodyA", "/World/BodyA"])


def test_complete_physics_pose_set_rejects_unexpected_state_path() -> None:
    with pytest.raises(SharedStageCompositionError, match="Unexpected OVPhysX body pose path"):
        complete_physics_pose_set([_state("/World/Other")], ["/World/BodyA"])


def test_complete_physics_pose_set_rejects_duplicate_state_path() -> None:
    with pytest.raises(SharedStageCompositionError, match="Duplicate OVPhysX body pose path"):
        complete_physics_pose_set(
            [_state("/World/BodyA"), _state("/World/BodyA", y=2.0)],
            ["/World/BodyA"],
        )


def test_complete_physics_pose_set_rejects_missing_expected_path() -> None:
    with pytest.raises(SharedStageCompositionError, match="omitted 1 body pose"):
        complete_physics_pose_set([_state("/World/BodyA")], ["/World/BodyA", "/World/BodyB"])


def test_complete_physics_pose_set_rejects_malformed_pose_attributes() -> None:
    with pytest.raises(SharedStageCompositionError, match="missing orient"):
        complete_physics_pose_set([_state("/World/BodyA", orient=False)], ["/World/BodyA"])


def test_apply_initial_condition_values_replaces_existing_and_appends_new_paths() -> None:
    base_a = _pose("/World/BodyA", y=1.0)
    base_b = _pose("/World/BodyB", y=2.0)
    override_b = _pose("/World/BodyB", y=3.0)
    override_c = _pose("/World/BodyC", y=4.0)

    assert apply_initial_condition_values([base_a, base_b], [override_b, override_c]) == (
        base_a,
        override_b,
        override_c,
    )
