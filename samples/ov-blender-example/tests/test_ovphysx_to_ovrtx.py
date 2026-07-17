# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_to_ovrtx import translate_values
from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue
from ovrtx_blender_example.shared_stage_composition import BodyPose


def _pose(path: str = "/World/Body") -> BodyPose:
    return BodyPose(path, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))


def test_translate_values_emits_complete_scaled_row_vector_matrix() -> None:
    assert translate_values([_pose()], 2.0) == [
        OvrtxTransformValue(
            "/World/Body",
            [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
            ],
        )
    ]


def test_translate_values_preserves_input_order() -> None:
    poses = [_pose("/World/B"), _pose("/World/A")]
    values = translate_values(poses, 1.0)
    assert [value.prim_path for value in values] == ["/World/B", "/World/A"]


def test_translate_values_accepts_empty_input() -> None:
    assert translate_values([], 1.0) == []


@pytest.mark.parametrize(
    "poses,scale,match",
    [
        ([_pose(), _pose()], 1.0, "duplicate"),
        ([BodyPose("/World/Body", (0.0, 0.0, 0.0), (1e154, 0.0, 0.0, 1.0))], 1.0, "matrix"),
        ([BodyPose("/World/Body", (0.0, 0.0, 0.0), (1e155, 0.0, 0.0, 1.0))], 1.0, "matrix"),
        ([_pose()], math.nan, "body_scale"),
    ],
)
def test_translate_values_rejects_invalid_batches(poses, scale, match) -> None:
    with pytest.raises(ValueError, match=match):
        translate_values(poses, scale)
