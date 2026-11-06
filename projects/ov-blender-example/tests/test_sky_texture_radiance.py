# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from ovrtx_blender_example import sky_texture_radiance as sky  # noqa: E402


def _node(direction=(0.0, 0.0, 1.0), turbidity=2.9, albedo=0.3, sky_type="HOSEK_WILKIE"):
    return SimpleNamespace(sky_type=sky_type, sun_direction=direction, turbidity=turbidity, ground_albedo=albedo)


def test_zenith_sun_matches_measured_horizon_band_mean():
    color, note = sky.sky_texture_radiance(_node())
    assert color == pytest.approx((0.1044, 0.1595, 0.2070), rel=0.02)
    assert note == "sky_texture_horizon_band_mean"


def test_low_sun_is_brighter_and_less_blue_near_the_horizon():
    # Measured with Cycles for the Classroom sun direction: (0.1200, 0.1373, 0.1255).
    color, _ = sky.sky_texture_radiance(_node(direction=(-0.9461538, 0.0615385, 0.3178143)))
    assert color == pytest.approx((0.1200, 0.1373, 0.1255), rel=0.03)
    zenith, _ = sky.sky_texture_radiance(_node())
    assert color[2] / color[0] < zenith[2] / zenith[0]


def test_azimuth_does_not_change_the_mean():
    a, _ = sky.sky_texture_radiance(_node(direction=(0.6, 0.0, 0.8)))
    b, _ = sky.sky_texture_radiance(_node(direction=(0.0, -0.6, 0.8)))
    assert a == pytest.approx(b)


def test_parameters_are_clamped_and_table_nodes_are_exact():
    exact = sky.hosek_wilkie_mean_radiance(30.0, 4.0, 0.6)
    assert exact == pytest.approx(sky._TABLE[(sky.ELEVATIONS.index(30) * len(sky.TURBIDITIES) + sky.TURBIDITIES.index(4.0)) * len(sky.ALBEDOS) + sky.ALBEDOS.index(0.6)])
    assert sky.hosek_wilkie_mean_radiance(-20.0, 99.0, 5.0) == pytest.approx(sky.hosek_wilkie_mean_radiance(0.0, 10.0, 1.0))


def test_other_sky_types_fall_back_and_report():
    color, note = sky.sky_texture_radiance(_node(sky_type="NISHITA"))
    assert color == pytest.approx(sky.hosek_wilkie_mean_radiance())
    assert note.startswith("sky_texture_non_hosek_wilkie_default_used:")
