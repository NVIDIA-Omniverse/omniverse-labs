# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track action semantics with real owned frames and GPU-free widget sinks."""
from capture_semantics import test_scene_roles, role_key, dynamic_geometry_owner
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.camera import Camera, CameraController, tracking_center
from ovd_ui.widgets.camera_view import CameraView
from test_application_presentation import (app as base_app, live, open_capture,
    HANDSHAKE, PAYLOAD, FRAME2_END, wait_until)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app(base_app):
    base_app.navigation = CameraController(Camera())
    base_app.window = SimpleNamespace(focused=True)
    base_app.viewport_keyboard_focus = True
    base_app.object_keyboard_focus = False
    return base_app


def open_scene(app, tmp_path, name="test_scene.ovd", frame=2):
    assert app._display(*open_capture(tmp_path, name))
    assert app.seek(frame, snap=False)
    if name == 'test_scene.ovd':
        app.fixture_roles = test_scene_roles(app.session)
    else:
        box = dynamic_geometry_owner(app.session, 2, frame)
        sphere = dynamic_geometry_owner(app.session, 1, frame)
        geometry = app.presentation.geometry_snapshot()
        shapes = set(map(int, geometry.shape_ids[geometry.object_ids == box]))
        assert len(shapes) == 1
        app.fixture_roles = SimpleNamespace(box=box, sphere=sphere, shape=shapes.pop())


def select(app, identity):
    app.selection.publish([f"/objects/{identity}"] if identity is not None else [], source="test")
    assert app._refresh()


def pose(app):
    camera = app.navigation.camera
    return (camera.position.copy(), camera.target.copy(), camera.world_up.copy(),
            camera.fov, camera.near, camera.distance)


def assert_pose(actual, expected):
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(got, wanted, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("left_handed", [False, True])
def test_alt_t_twice_stops_without_moving_camera_or_playhead(app, tmp_path, left_handed):
    open_scene(app, tmp_path)
    ids = app.fixture_roles
    select(app, ids.actor)
    app.navigation.camera.left_handed = left_handed
    before = pose(app)
    view = app.presentation
    app._key_pressed(84, 4, True)
    assert app.navigation.tracking_id == ids.actor and app.navigation.tracking_offset is None
    assert not app.navigation.captured and app.presentation is view
    assert_pose(pose(app), before)
    assert app._refresh()
    assert app.navigation.tracking_offset is not None
    following = pose(app)
    generation = app.presentation.generation
    app._key_pressed(84, 4, True)
    assert app.navigation.tracking_id is None and app.navigation.tracking_offset is None
    assert app.status_label.text == "Camera tracking stopped."
    assert app.presentation.generation == generation and app.presentation_dirty
    assert_pose(pose(app), following)
    assert app._refresh()
    assert app.frame == app.presentation.frame == app.playback.frame == 2
    assert_pose(pose(app), following)
    assert app.selection.get_snapshot().paths() == [f"/objects/{ids.actor}"]


def test_new_selection_still_stops_first_then_next_action_tracks_its_owned_pose(app, tmp_path):
    open_scene(app, tmp_path, "multi_geom.ovd")
    ids = app.fixture_roles
    select(app, ids.box)
    app.track_selected()
    assert app._refresh()
    old_offset = app.navigation.tracking_offset.copy()
    select(app, ids.sphere)
    before = pose(app)
    app.track_selected()
    assert app.navigation.tracking_id is None and app.navigation.tracking_offset is None
    assert_pose(pose(app), before)
    assert app._refresh()
    app.track_selected()
    assert app.navigation.tracking_id == ids.sphere and app.navigation.tracking_offset is None
    assert_pose(pose(app), before)
    assert app._refresh()
    view = app.presentation
    center = tracking_center(view, ids.sphere, app.adapter)
    assert center is not None
    np.testing.assert_allclose(app.navigation.tracking_offset,
        app.navigation.camera.mirror(before[0]) - center)
    assert not np.array_equal(app.navigation.tracking_offset, old_offset)
    assert app.selection.get_snapshot().paths() == [f"/objects/{ids.sphere}"]
    assert ids.sphere in view.inspected_ids
    app.track_selected()
    assert app.navigation.tracking_id is None


@pytest.mark.parametrize("selection", [None, "shape"])
def test_stop_with_empty_or_ineligible_selection_removes_tracking_only_read(app, tmp_path, selection):
    open_scene(app, tmp_path, "multi_geom.ovd")
    ids = app.fixture_roles
    selection = ids.shape if selection == 'shape' else None
    select(app, ids.box)
    app.track_selected()
    assert app._refresh()
    if selection is not None:
        assert app.adapter.class_name(selection) != "PxRigidDynamic"
    select(app, selection)
    assert ids.box in app.presentation.inspected_ids
    before = pose(app)
    app.track_selected()
    assert app.navigation.tracking_id is None and app.navigation.tracking_offset is None
    assert app._refresh()
    assert ids.box not in app.presentation.inspected_ids
    assert app.presentation.inspected_ids == ([selection] if selection is not None else [])
    assert_pose(pose(app), before)


def test_sleeping_death_and_rewind_do_not_reenable_a_stopped_target(app, tmp_path):
    open_scene(app, tmp_path, frame=1)
    ids = app.fixture_roles
    select(app, ids.actor)
    app.track_selected()
    before = pose(app)
    assert app._refresh()
    assert app.navigation.tracking_id == ids.actor and app.navigation.tracking_offset is None
    assert tracking_center(app.presentation, ids.actor, app.adapter) is None
    assert_pose(pose(app), before)
    assert app.seek(2, snap=False)
    assert app.navigation.tracking_offset is not None
    owned = app.presentation
    center = tracking_center(owned, ids.actor)
    assert app.seek(3, snap=False)
    assert app.navigation.tracking_id == ids.actor and not app.adapter.instances[ids.actor].alive
    assert app.selection.get_snapshot().paths() == []
    before = pose(app)
    app.track_selected()
    assert app.navigation.tracking_id is None
    assert app._refresh() and ids.actor not in app.presentation.inspected_ids
    assert app.seek(2, snap=False)
    assert app.navigation.tracking_id is None
    assert_pose(pose(app), before)
    np.testing.assert_array_equal(tracking_center(owned, ids.actor), center)


def test_inactive_dead_find_target_is_not_enabled_and_new_capture_drops_tracking(app, tmp_path):
    open_scene(app, tmp_path, frame=3)
    ids = app.fixture_roles
    assert app._find_object(ids.actor)
    app.track_selected()
    assert app.navigation.tracking_id is None
    assert "Select a dynamic rigid body" in app.status_label.text
    assert app.seek(2, snap=False)
    select(app, ids.actor)
    app.track_selected()
    assert app._refresh()
    old_capture = app.presentation.capture_key
    assert app._display(*open_capture(tmp_path / "other", "test_scene.ovd"))
    assert app.presentation.capture_key != old_capture
    assert app.navigation.tracking_id is None and app.navigation.tracking_offset is None


def editor(app):
    view = CameraView(app.navigation, app._camera_changed,
        frame_all=lambda: None, frame_selected=lambda: None, center_selected=lambda: None,
        track_selected=app.track_selected, reset=app.reset_camera)
    view.window = SimpleNamespace(visible=True, destroy=lambda: None)
    for field in ("speed", "near", "fov"):
        setattr(view, field, SimpleNamespace(model=ui.SimpleFloatModel()))
    view.near_slider = SimpleNamespace(model=ui.SimpleIntModel(1, min=1, max=100))
    view.handed = SimpleNamespace(model=ui.SimpleBoolModel())
    for field in ("source", "mode", "up"):
        model = ui.SimpleIntModel()
        setattr(view, field, SimpleNamespace(model=SimpleNamespace(get_item_value_model=lambda value=model: value)))
    view.coordinates = [SimpleNamespace(model=ui.SimpleFloatModel()) for _ in range(3)]
    view.status = SimpleNamespace(text="")
    view.track_button = SimpleNamespace(text="")
    view._sync()
    return view


def test_camera_button_uses_same_toggle_and_updates_its_tracking_label(app, tmp_path):
    open_scene(app, tmp_path)
    ids = app.fixture_roles
    select(app, ids.actor)
    view = editor(app)
    callback = dict(view.actions)["Track selection"]
    view._action(callback)
    assert app.navigation.tracking_id == ids.actor and f"Tracking {ids.actor}" in view.status.text
    assert view.track_button.text == "Stop tracking"
    assert app._refresh()
    before = pose(app)
    view._action(callback)
    assert app.navigation.tracking_id is None and "Tracking " not in view.status.text
    assert view.track_button.text == "Track selection"
    assert app.status_label.text == "Camera tracking stopped."
    assert_pose(pose(app), before)


@pytest.mark.parametrize("cancel", ["keyboard", "drag", "reset"])
def test_external_cancellation_refreshes_tracking_state_without_overwriting_drafts(app, tmp_path, cancel):
    open_scene(app, tmp_path)
    ids = app.fixture_roles
    select(app, ids.actor)
    view = editor(app)
    app.camera_view = view
    app.track_selected()
    assert view.track_button.text == "Stop tracking"
    view.fov.model.set_value(97.)
    view.speed.model.set_value(123.)
    view.coordinates[0].model.set_value(987.)
    if cancel == "keyboard":app.navigation.key(87, 0, True)
    elif cancel == "drag":app.navigation.drag(20, 5, 800, 600, 0)
    else:
        assert app._display(*open_capture(tmp_path / "other", "test_scene.ovd"))
    assert app.navigation.tracking_id is None
    # The production draw loop calls this after rendering changed camera state.
    view.refresh_tracking()
    assert view.track_button.text == "Track selection" and "Tracking " not in view.status.text
    assert view.fov.model.as_float == 97 and view.speed.model.as_float == 123
    assert view.coordinates[0].model.as_float == 987


def test_draw_time_tracking_refresh_preserves_invalid_draft_and_error(app, tmp_path):
    open_scene(app, tmp_path)
    ids = app.fixture_roles
    select(app, ids.actor)
    view = editor(app)
    app.camera_view = view
    app.track_selected()
    view.near.model.set_value(0.)
    view._apply()
    error = view.status.text
    assert "positive finite" in error and app.navigation.tracking_id == ids.actor
    app.navigation.manual_change()
    view.refresh_tracking()
    assert view.track_button.text == "Track selection"
    assert view.near.model.as_float == 0. and view.status.text == error


def test_closed_window_keeps_valid_state_but_destroyed_window_never_touches_children(app):
    view = editor(app)
    view.window.visible = False  # Closing the dialog hides, rather than destroys, it.
    app.navigation.tracking_id = 22
    view.refresh_tracking()
    assert view.track_button.text == "Stop tracking"
    view.destroy()
    assert view.window is None
    class DestroyedWidget:
        @property
        def text(self):
            raise AssertionError("Destroyed widget was accessed")
        @text.setter
        def text(self, _):
            raise AssertionError("Destroyed widget was written")
    view.track_button = view.status = DestroyedWidget()
    app.navigation.manual_change()
    view.refresh_tracking()


def test_partial_live_start_then_stop_keeps_prior_owned_pose_and_never_resumes(app, live):
    source, connection = live
    app.workflow.live_session = source
    connection.sendall(HANDSHAKE + PAYLOAD[:FRAME2_END])
    wait_until(lambda: source.info().completed_serial == 2)
    assert app._display(None, source)
    ids = test_scene_roles(source)
    assert app.seek(2, snap=False)
    select(app, ids.actor)
    before, owned = pose(app), app.presentation
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda: source.info().recording_size == len(PAYLOAD)
               and source.info().unpublished_changes)
    app.track_selected()
    assert app.navigation.tracking_id == ids.actor and app.navigation.tracking_offset is None
    assert not app._refresh()
    assert app.presentation is owned and app.navigation.tracking_offset is None
    assert_pose(pose(app), before)
    app.track_selected()
    assert app.navigation.tracking_id is None
    assert not app._refresh()
    assert app.presentation is owned
    source.stop_live()
    assert app._refresh()
    assert app.navigation.tracking_id is None and app.navigation.tracking_offset is None
    assert app.frame == app.presentation.frame == 2
    assert_pose(pose(app), before)


def test_alt_t_respects_existing_viewport_and_text_focus_ownership(app, tmp_path):
    open_scene(app, tmp_path)
    ids = app.fixture_roles
    select(app, ids.actor)
    for focused, viewport, modifiers in ((False, True, 4), (True, False, 4),
                                       (True, False, 4 | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD)):
        app.window.focused, app.viewport_keyboard_focus = focused, viewport
        app._key_pressed(84, modifiers, True)
        assert app.navigation.tracking_id is None
