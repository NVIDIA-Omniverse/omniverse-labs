# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate authoritative OVPhysX poses into OVRTX transform values."""

from __future__ import annotations

import math
from typing import Sequence

from .shared_stage_composition import BodyPose
from .ovrtx_value_updates import OvrtxTransformValue


def translate_values(
    poses: Sequence[BodyPose], body_scale: float
) -> list[OvrtxTransformValue]:
    scale = float(body_scale)
    if not math.isfinite(scale):
        raise ValueError("body_scale must be finite")
    paths: set[str] = set()
    for pose in poses:
        if pose.prim_path in paths:
            raise ValueError(f"duplicate body pose path: {pose.prim_path}")
        paths.add(pose.prim_path)
    values: list[OvrtxTransformValue] = []
    for pose in poses:
        matrix = _matrix4d_rows(pose.translate, pose.orient, scale)
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError(f"body pose matrix must be finite: {pose.prim_path}")
        values.append(OvrtxTransformValue(pose.prim_path, matrix))
    return values


def _matrix4d_rows(
    translate: tuple[float, float, float],
    orient: tuple[float, float, float, float],
    scale: float,
) -> list[list[float]]:
    tx, ty, tz = translate
    x, y, z, w = orient
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    flat = [
        scale * (1.0 - 2.0 * (yy + zz)), scale * 2.0 * (xy + wz), scale * 2.0 * (xz - wy), 0.0,
        scale * 2.0 * (xy - wz), scale * (1.0 - 2.0 * (xx + zz)), scale * 2.0 * (yz + wx), 0.0,
        scale * 2.0 * (xz + wy), scale * 2.0 * (yz - wx), scale * (1.0 - 2.0 * (xx + yy)), 0.0,
        tx, ty, tz, 1.0,
    ]
    return [flat[index : index + 4] for index in range(0, 16, 4)]
