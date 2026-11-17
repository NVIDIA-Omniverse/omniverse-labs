# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-click highlights; double-click centers without moving the camera eye."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_viewport_menu import app, menu_app, Menu, camera_state
from ovd_ui.camera import Camera, CameraController
from ovd_ui.orbit_center import orbit_transform


@pytest.fixture
def focus_app(menu_app):
    app = menu_app
    app.hit = int(app.renderer.snapshot.object_ids[0])
    app.picks = []

    def pick(x, y):
        app.picks.append((x, y))
        return app.hit  # GPU readback only; selection/bounds/camera stay real.

    app.renderer.pick = pick
    app._build_viewport_input()
    yield app
    app.viewport_input.destroy()


def click(app, *, modifiers=0):
    widget = app.viewport_input
    widget.call_mouse_pressed_fn(300, 400, 0, modifiers)
    widget.call_mouse_released_fn(300, 400, 0, modifiers)


def move(app, x, y):
    app._workspace_mouse_moved(x, y)
    callback, args = app.queued.pop()
    callback(*args)


@pytest.mark.parametrize("source", ["viewport", "objects", "find"])
@pytest.mark.parametrize("first", ["orbit", "zoom"])
def test_selection_anchors_navigation_without_an_initial_jump(focus_app, source, first):
    app = focus_app
    camera = app.renderer.camera
    bounds = app.renderer.snapshot.bounds[app.renderer.snapshot.object_ids == app.hit]
    center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) * .5
    camera.position = center + [7., 3., 30.]  # Deliberately off-axis.
    before = camera_state(app)
    if source == "viewport": click(app)
    elif source == "find": assert app._find_object(app.hit)
    else: app._tree_selection([app.object_model.canonical_item(app.hit)])
    assert app.interaction.selected == (app.hit,)
    assert camera_state(app) == before

    widget = app.viewport_input
    for gesture in (["orbit", "zoom"] if first == "orbit" else ["zoom", "orbit"]):
        eye, radius = camera.position.copy(), np.linalg.norm(camera.position - center)
        local = camera.view_matrix() @ np.r_[center, 1.]
        if gesture == "orbit":
            widget.call_mouse_pressed_fn(300, 400, 0, 0)
            assert camera_state(app) == before
            move(app, 304, 400)
            widget.call_mouse_released_fn(304, 400, 0, 0)
            assert np.linalg.norm(camera.position - center) == pytest.approx(radius)
            assert 0 < np.linalg.norm(camera.position - eye) < radius * .03
        else:
            angles = camera.yaw, camera.pitch
            widget.call_mouse_wheel_fn(0, 1, 0)
            np.testing.assert_allclose(camera.position, center + (eye - center) / 1.1)
            assert (camera.yaw, camera.pitch) == angles
            closer = camera.position.copy()
            widget.call_mouse_wheel_fn(0, -1, 0)
            assert np.linalg.norm(camera.position - center) > np.linalg.norm(closer - center)
        moved = camera.view_matrix() @ np.r_[center, 1.]
        # The rendered matrix is float32: allow less than .01 viewport pixels.
        np.testing.assert_allclose(moved[:2] / moved[2], local[:2] / local[2], atol=3e-6)
        before = camera_state(app)
    # Navigation does not pick again or change the selection.
    assert app.picks == ([(300, 400)] if source == "viewport" else [])
    assert app.interaction.selected == (app.hit,)


@pytest.mark.parametrize("modifiers", [0, 1 << 30])
def test_double_click_centers_without_moving_or_zooming_then_orbits(focus_app, modifiers):
    app = focus_app
    camera = app.renderer.camera
    bounds = app.renderer.snapshot.bounds[app.renderer.snapshot.object_ids == app.hit]
    center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) * .5
    camera.position = center + [7., 3., 30.]
    eye = camera.position.copy()
    radius = np.linalg.norm(eye - center)
    before = camera_state(app)
    click(app, modifiers=modifiers)
    assert camera_state(app) == before  # The first click only selects/highlights.
    assert app.renderer.selected == {app.hit}
    widget = app.viewport_input
    widget.call_mouse_pressed_fn(300, 400, 0, modifiers)
    widget.call_mouse_double_clicked_fn(300, 400, 0, modifiers)
    assert camera_state(app) == before
    widget.call_mouse_released_fn(300, 400, 0, modifiers)
    np.testing.assert_allclose(camera.position, eye)
    np.testing.assert_allclose(camera.target, center)
    assert camera.distance == pytest.approx(radius)
    local = camera.view_matrix() @ np.r_[center, 1.]
    np.testing.assert_allclose(local[:2], 0., atol=3e-5)
    assert app.interaction.selected == (app.hit,)
    widget.call_mouse_pressed_fn(300, 400, 0, modifiers)
    move(app, 320, 410)
    widget.call_mouse_released_fn(320, 410, 0, modifiers)
    assert np.linalg.norm(camera.position - center) == pytest.approx(radius)
    widget.call_mouse_wheel_fn(0, 1, 0)
    assert np.linalg.norm(camera.position - center) == pytest.approx(radius / 1.1)
    local = camera.view_matrix() @ np.r_[center, 1.]
    np.testing.assert_allclose(local[:2], 0., atol=3e-5)


def test_explicit_framing_remains_available(focus_app):
    app = focus_app
    click(app)
    camera = app.renderer.camera
    before = camera.position.copy()
    app.frame_selected()
    assert not np.allclose(camera.position, before)
    np.testing.assert_allclose(camera.target, app.renderer.selection_pivot())
    camera.turn(15., 0.)
    eye = camera.position.copy()
    app.center_selected()
    np.testing.assert_allclose(camera.position, eye)
    np.testing.assert_allclose(camera.target, app.renderer.selection_pivot())


@pytest.mark.parametrize("case", ["empty", "missing", "hidden", "scene", "capture"])
def test_unavailable_selection_keeps_normal_camera_navigation(focus_app, case):
    app = focus_app
    click(app)
    if case == "empty": app.selection.publish([], source="objects")
    elif case == "missing": app.renderer.selected = {-999}
    elif case == "hidden": app.renderer.hidden = {app.hit}
    elif case == "scene": app.renderer.scene_id = -999
    elif case == "capture": app.renderer.snapshot = None
    expected = deepcopy(app.navigation)
    expected.wheel(1)
    app.viewport_input.call_mouse_wheel_fn(0, 1, 0)
    np.testing.assert_allclose(app.renderer.camera.position, expected.camera.position)
    assert app.renderer.camera.distance == expected.camera.distance


@pytest.mark.parametrize("amount", [-1, 1])
@pytest.mark.parametrize("radius", [0., .001, 1e8])
def test_selected_zoom_cannot_jump_at_existing_distance_limits(amount, radius):
    camera = Camera(position=np.array([radius, 0., 0.]))
    control = CameraController(camera)
    control.scene_length = 1. if radius == 0 else 0.
    control.wheel(amount, pivot=np.zeros(3))
    expected = radius * (1.1 if amount < 0 else 1 / 1.1)
    if radius < .003 and amount > 0 or radius > 300000 and amount < 0:
        expected = radius
    np.testing.assert_allclose(camera.position, [expected, 0., 0.])
    assert (camera.yaw, camera.pitch) == (0., 0.)


@pytest.mark.parametrize("distance,amount", [(300000., -1), (.003, 1)])
def test_deselection_after_selected_zoom_keeps_legacy_distance_limits(distance, amount):
    camera = Camera(position=np.array([0., 0., 1.]), distance=distance)
    control = CameraController(camera)
    control.wheel(amount, pivot=np.zeros(3))
    assert .003 <= camera.distance <= 300000.
    eye = camera.position.copy()
    control.wheel(amount)  # Selection cleared; never jump back from an invalid distance.
    np.testing.assert_allclose(camera.position, eye, atol=1e-9, rtol=0)


@pytest.mark.parametrize("identity", ["object_ids", "shape_ids", "geometry_ids"])
@pytest.mark.parametrize("scale,handed,up", [(1., False, [0, 1, 0]),
                                           (.1, True, [0, 1, 0]), (2., True, [0, 0, 1])])
def test_selected_bounds_and_marker_share_scaled_mirrored_pivot(focus_app, identity, scale, handed, up):
    renderer = focus_app.renderer
    snapshot = renderer.snapshot
    renderer.selected = {int(getattr(snapshot, identity)[0])}
    renderer.settings = SimpleNamespace(geometry_scale=scale)
    renderer._prepare = lambda: None  # GPU upload sink; all owned rows visible.
    renderer._visible_rows = np.arange(len(snapshot.object_ids))
    rows = getattr(snapshot, identity) == next(iter(renderer.selected))
    bounds = snapshot.bounds[rows]
    center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) * .5
    expected = center.astype("f8") * scale
    if handed: expected[0 if up[2] else 2] *= -1
    renderer.camera.left_handed, renderer.camera.world_up = handed, np.array(up)
    pivot = renderer.selection_pivot()
    np.testing.assert_allclose(pivot, expected)
    marker = orbit_transform(renderer.camera, 1., 1., scale, pivot=pivot)
    np.testing.assert_allclose(marker[:3, 3], center)
    renderer.camera.position = expected + [7., 3., 30.]
    focus_app.viewport_input.call_mouse_wheel_fn(0, 1, 0)
    np.testing.assert_allclose(renderer.camera.position, expected + np.array([7., 3., 30.]) / 1.1)


@pytest.mark.parametrize("case", ["empty", "drag", "popup", "unfocused", "scene", "ctrl", "shift", "alt", "super", "right"])
def test_double_click_cannot_focus_a_cancelled_or_modified_pick(focus_app, case):
    app = focus_app
    widget = app.viewport_input
    button, modifiers = (1 if case == "right" else 0), {"shift":1,"ctrl":2,"alt":4,"super":8}.get(case,0)
    widget.call_mouse_pressed_fn(300, 400, button, modifiers)
    widget.call_mouse_double_clicked_fn(300, 400, button, modifiers)
    if case == "empty": app.hit = None
    elif case == "drag": move(app, 320, 420)
    elif case == "popup": Menu.current = object()
    elif case == "unfocused": app.window.focused = False
    elif case == "scene": app.scene_id += 1
    before = camera_state(app)
    widget.call_mouse_released_fn(300, 400, button, modifiers)
    assert camera_state(app) == before
