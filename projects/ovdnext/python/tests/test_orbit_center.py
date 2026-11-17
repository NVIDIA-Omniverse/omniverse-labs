# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original hemisphere source oracle, owned camera math and settings behavior."""

import numpy as np
import pytest

from ovd_ui.camera import Camera
from ovd_ui.orbit_center import orbit_sphere, orbit_transform
from ovd_ui.render_settings import RenderingSettings
from test_application_render_settings import application, dialog_controls
from reference_data import load


@pytest.fixture(scope="module")
def original_topology():
    return load("orbit_topology")


def test_wire_topology_is_the_original_two_unshifted_hemispheres(original_topology):
    points, indices = orbit_sphere()
    expected_points = np.concatenate([part["vertices"] for part in original_topology])
    expected_indices = np.concatenate([part["indices"] + np.array(i * 85)
                                      for i, part in enumerate(original_topology)])
    np.testing.assert_allclose(points, expected_points, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(indices, expected_indices)
    assert points.shape == (170, 3) and indices.shape == (768,)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1., atol=1e-7)
    assert not points.flags.writeable and not indices.flags.writeable
    assert orbit_sphere()[0] is points  # Only one immutable CPU topology.


@pytest.mark.parametrize("scale", [.01, 1., 100.])
@pytest.mark.parametrize("handed", [False, True])
@pytest.mark.parametrize("up", [(0., 1., 0.), (0., 0., 1.), (1., 2., 3.)])
def test_marker_tracks_the_actual_pivot_and_scales_radius_once(scale, handed, up):
    camera = Camera(tolerance_length=3., left_handed=handed)
    camera.look_at((11., 7., -9.), (2., -1., 4.), up)
    for move in (lambda: None, lambda: camera.turn(17., -11., orbit=True),
                 lambda: camera.pan(23., -12., 600),
                 lambda: camera.center_at((3., 8., -2.))):
        move()
        before = (camera.position.copy(), camera.world_up.copy(), camera.distance, camera.near)
        transform = orbit_transform(camera, 2., 3.4, scale)
        world_scale = np.diag([scale, scale, scale, 1.])
        pivot = world_scale @ transform @ np.array([0., 0., 0., 1.])
        np.testing.assert_allclose(pivot[:3], camera.mirror(camera.target), rtol=2e-7, atol=2e-6)
        camera_pivot = camera.view_matrix() @ pivot
        np.testing.assert_allclose(camera_pivot[:2], 0., atol=3e-6)
        np.testing.assert_allclose(np.linalg.norm((world_scale @ transform)[:3, :3], axis=0),
                                   .25 * 2. * 3.4 * scale, rtol=1e-7)
        np.testing.assert_array_equal(camera.position, before[0])
        np.testing.assert_array_equal(camera.world_up, before[1])
        assert (camera.distance, camera.near) == before[2:]


def test_no_calibration_uses_current_owned_tolerance_without_pose_reset():
    camera = Camera(tolerance_length=2.)
    assert orbit_transform(camera, 0., 1., 1.)[0, 0] == .5
    camera.tolerance_length = 8.
    assert orbit_transform(camera, 0., 1., 1.)[0, 0] == 2.
    assert orbit_transform(camera, 3., 1., 1.)[0, 0] == .75


@pytest.mark.parametrize("value", [1, 0, "yes", None, 0.5])
def test_orbit_preference_is_a_boolean(value):
    with pytest.raises(ValueError):
        RenderingSettings(draw_orbit_center=value)


def test_orbit_setting_round_trip_apply_failure_and_defaults():
    assert not RenderingSettings.from_dict({}).draw_orbit_center
    assert RenderingSettings.from_dict(RenderingSettings(draw_orbit_center=True).to_dict()).draw_orbit_center
    app, applied, saved = application()
    controls = dialog_controls(app.render_settings_view)
    controls["draw_orbit_center"].model.set_value(True)
    controls["geometry_scale"].model.set_value(2.)
    app.render_settings_view._apply()
    assert applied[-1].draw_orbit_center and applied[-1].geometry_scale == 2.
    assert saved[-1][1]["draw_orbit_center"] is True
    app.render_settings_view._apply(True)
    assert not applied[-1].draw_orbit_center
    app, applied, saved = application(fail_save=True)
    controls = dialog_controls(app.render_settings_view)
    controls["draw_orbit_center"].model.set_value(True)
    app.render_settings_view._apply()
    assert not applied and not saved
    assert not app.render_settings.draw_orbit_center and app.render_settings_view._status_error
