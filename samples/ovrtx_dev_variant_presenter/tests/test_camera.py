# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from dev_variant_presenter.render.camera import CameraController


def test_xform_is_row_vector_4x4_float64():
    c = CameraController(target=(0, 0, 0), distance=10.0, up_axis="Y")
    m = c.to_xform()
    assert m.shape == (4, 4) and m.dtype == np.float64
    eye = m[3, :3]  # translation in the last ROW (row-vector convention)
    assert np.isclose(np.linalg.norm(eye - np.array([0, 0, 0])), 10.0, atol=1e-6)


def test_orbit_changes_eye_but_keeps_distance():
    c = CameraController(target=(0, 0, 0), distance=10.0, up_axis="Y")
    before = c.to_xform()[3, :3].copy()
    c.orbit(d_az=0.5, d_el=0.2)
    after = c.to_xform()[3, :3]
    assert not np.allclose(before, after)
    assert np.isclose(np.linalg.norm(after), 10.0, atol=1e-6)


def test_snap_to_recovers_authored_eye():
    authored = CameraController(target=(0, 0, 0), distance=20.0, up_axis="Y")
    M = authored.to_xform()
    c = CameraController(target=(5, 5, 5), distance=1.0, up_axis="Y")
    c.snap_to(M, focus_distance=20.0)
    assert np.allclose(c.to_xform()[3, :3], M[3, :3], atol=1e-4)
    # free navigation continues from there
    c.orbit(0.3, 0.0)
    assert np.isclose(np.linalg.norm(c.to_xform()[3, :3] - c.target), 20.0, atol=1e-4)
