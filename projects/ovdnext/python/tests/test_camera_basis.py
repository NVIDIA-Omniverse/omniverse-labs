# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The original camera's exact float boundary, through real normalization."""

import numpy as np
import pytest

from ovd_ui.camera import Camera
from reference_data import load


@pytest.fixture(scope="module")
def original_basis():
    return load("camera_basis")


@pytest.mark.parametrize("handed",[False,True])
@pytest.mark.parametrize("case",range(58))
def test_float_threshold_basis_and_mirror_follow_unchanged_qt(original_basis,case,handed):
    rows=[r for r in original_basis if r["case"]==case and r["handed"]==handed]
    camera=Camera(left_handed=handed,near=.01)
    incoming=np.asarray(rows[0]["input"],dtype="f4")
    camera.set_world_up(camera.mirror(incoming))
    camera.reset()
    for row in rows:
        if row["stage"]=="orbit":camera.turn(17,-11,orbit=True)
        elif row["stage"]=="move":camera.move((.3,.4,-.2),world_up=True)
        for key,value in (("world_up",camera.world_up),("forward",camera.basis[2]),
            ("up",camera.basis[1]),("right",camera.basis[0]),("position",camera.position),
            ("distance",camera.distance),("mirror",camera.mirror((2,3,5)))):
            np.testing.assert_allclose(value,row[key],rtol=3e-5,atol=3e-5,
                err_msg=f"{case}/{handed}/{row['stage']}/{key}")
        np.testing.assert_allclose(camera.view_matrix().ravel(),row["view"],rtol=3e-5,atol=3e-5)
