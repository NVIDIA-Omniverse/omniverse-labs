# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.physics_frames import (
    FrameError, PhysicsUnits, RigidFrame, local_frame, matrix_to_transform,
    owner_frame_from_body, wxyz_to_xyzw, xyzw_to_wxyz,
)


def assert_frame(a, b):
    assert a.position == pytest.approx(b.position)
    assert abs(sum(x*y for x, y in zip(a.rotation_wxyz, b.rotation_wxyz))) == pytest.approx(1)


def test_frames_and_offsets_use_column_vector_composition():
    body = RigidFrame((3, 2, 1), (1, 0, 0, 1))
    anchor = RigidFrame((3, 4, 1), (1, 0, 0, 0))
    local = local_frame(body, anchor)
    assert local.position == pytest.approx((2, 0, 0))
    assert_frame(body.compose(local), anchor)
    assert_frame(owner_frame_from_body(body.compose(local), local), body)
    assert body.gf_matrix()[3] == (3, 2, 1, 1)
    assert_frame(matrix_to_transform(body.column_matrix()).frame, body)


@pytest.mark.parametrize("q", [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 2, 3, 4)])
def test_quaternion_branches_and_bridge_order(q):
    frame = RigidFrame((1, 2, 3), q)
    assert_frame(matrix_to_transform(frame.column_matrix()).frame, frame)
    assert xyzw_to_wxyz(wxyz_to_xyzw(q)) == pytest.approx(frame.rotation_wxyz)
    assert frame.bridge_pose()[1] == pytest.approx(wxyz_to_xyzw(q))


def test_meter_and_centimeter_equivalence_for_every_dimension():
    m, cm = PhysicsUnits(), PhysicsUnits(.01)
    assert cm.length(100) == m.length(1)
    assert cm.vector((100, 200, 300)) == m.vector((1, 2, 3))
    assert cm.density(.001) == pytest.approx(m.density(1000))
    assert cm.inertia(20000) == m.inertia(2)
    assert cm.force(1000) == m.force(10)
    assert cm.torque(100000) == m.torque(10)
    assert cm.drive_gain(10) == m.drive_gain(10)
    assert cm.drive_gain(100000, angular=True) == pytest.approx(m.drive_gain(10, angular=True))
    # Same torque for equivalent errors expressed in radians or USD degrees.
    assert m.drive_gain(10, angular=True) * m.angle(.5) == pytest.approx(5)
    assert cm.drive_effort(1000) == m.drive_effort(10)
    assert cm.drive_effort(100000, angular=True) == m.drive_effort(10, angular=True)
    assert cm.angular_velocity((math.pi, 0, -math.pi/2)) == pytest.approx((180, 0, -90))
    frame = RigidFrame((100, 200, 300), (1, 2, 3, 4))
    assert_frame(cm.frame_from_meters(cm.frame(frame)), frame)
    with pytest.raises(FrameError, match="runtime validation"):
        cm.drive_gain(10, mode="acceleration")


@pytest.mark.parametrize("scale, message", [((0, 1, 1), "singular"), ((-1, 1, 1), "negative"), ((1, 2, 1), "nonuniform")])
def test_bad_scale_rejected(scale, message):
    matrix = [[0.0]*4 for _ in range(4)]
    for i, v in enumerate((*scale, 1)):
        matrix[i][i] = v
    with pytest.raises(FrameError, match=message):
        matrix_to_transform(matrix)


def test_shear_and_invalid_rotations_rejected():
    matrix = [list(row) for row in RigidFrame().column_matrix()]
    matrix[0][1] = .3
    with pytest.raises(FrameError, match="sheared"):
        matrix_to_transform(matrix)
    for q in [(0, 0, 0, 0), (math.nan, 0, 0, 1)]:
        with pytest.raises(FrameError):
            RigidFrame(rotation_wxyz=q)
    for scale in [0, -1, math.nan, math.inf]:
        with pytest.raises(FrameError):
            PhysicsUnits(scale)
