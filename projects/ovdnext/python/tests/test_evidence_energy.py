# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent mass-frame/COM energy oracles used by evidence validation."""
import math
import pytest


def test_mass_frame_com_oracle_is_85_point_5_joules():
    from ovd_ui.evidence_energy import mechanical_energy
    value = mechanical_energy(2, (1, 2, 3), (0, 0, 0, 1, 0, 0, 1),
                              (0, 0, 0, 1, 1, 2, 3), (2, 0, 0), (0, 0, 1), (0, 0, -10), (0, 0, 0))
    assert value.components == (4, 1.5, 80) and value.total == 85.5
    assert value.roundoff > 0


def test_rotated_com_and_mass_frame_inertia_use_world_angular_velocity():
    from ovd_ui.evidence_energy import mechanical_energy
    half = math.sqrt(.5)
    value = mechanical_energy(2, (1, 2, 3), (0, 0, 0, 1, 1, 0, 0),
                              (0, half, 0, half, 0, 0, 4), (0, 0, 0), (0, 0, 1), (0, 0, -10), (0, 0, 0))
    assert value.total == pytest.approx(60.5)
    assert value.components[:2] == pytest.approx((0, .5))


def test_invalid_physical_setup_is_inconclusive():
    from ovd_ui.evidence_energy import mechanical_energy
    for mass, inertia, pose in [(0, (1, 1, 1), (0, 0, 0, 1, 0, 0, 0)),
                                 (1, (1, -1, 1), (0, 0, 0, 1, 0, 0, 0)),
                                 (1, (1, 1, 1), (0, 0, 0, 0, 0, 0, 0))]:
        with pytest.raises(ValueError):
            mechanical_energy(mass, inertia, pose, pose, (0, 0, 0), (0, 0, 0), (0, 0, -10), (0, 0, 0))
