# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport routing over real owned captures; only widget/GPU creation is a sink."""
from capture_semantics import test_scene_roles, role_key, dynamic_geometry_owner
from types import MethodType, SimpleNamespace as NS
import struct

import numpy as np
import pytest

from ovd_ui.camera import Camera, CameraController
from ovd_ui.renderer import ViewportRenderer
from test_application_presentation import app, live, open_capture, ROOT, HANDSHAKE, wait_until
from test_application_object_actions import (ALIAS, GROUP, choose, settle,
    selection_and_publication, PresentationForbiddenSession)


class Menu:
    current = None
    building = None

    def __init__(self, title, **options):
        self.title, self.options, self.items = title, options, []
        self.destroyed = 0

    @staticmethod
    def get_current(): return Menu.current
    def __enter__(self): Menu.building = self; return self
    def __exit__(self, *_): Menu.building = None
    def set_shown_changed_fn(self, callback): self.changed = callback
    def show_at(self, x, y): self.position = (x, y); Menu.current = self
    def dismiss(self):
        Menu.current = None
        self.changed(False)
    def destroy(self):
        self.destroyed += 1
        self.dismiss()


@pytest.fixture
def menu_app(app, tmp_path, monkeypatch):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    app.window = NS(focused=True)
    app.viewport = NS(screen_position_x=0, screen_position_y=0)
    app.renderer.camera, app.renderer.settings, app.renderer.size = Camera(), None, (800, 600)
    app.renderer.frame_all = MethodType(ViewportRenderer.frame_all, app.renderer)
    app.navigation = CameraController(app.renderer.camera)
    app.renderer.pick = lambda *_: pytest.fail("A right gesture must not repick")
    app.mouse_anchor = None
    Menu.current = None
    monkeypatch.setattr("ovd_ui.application.ui.Menu", Menu)
    monkeypatch.setattr("ovd_ui.application.ui.MenuItem",
        lambda label, **kw: Menu.building.items.append(NS(label=label, **kw)))
    monkeypatch.setattr("ovd_ui.application.ui.Separator", lambda: None)
    app.queued = []
    monkeypatch.setattr("ovd_ui.application.asyncio.get_event_loop",
        lambda: NS(call_soon=lambda fn, *args: app.queued.append((fn, args))))
    yield app
    Menu.current = None


def open_menu(app):
    app._mouse_pressed(300, 400, 1, 0)
    app._mouse_released(300, 400, 1)
    assert len(app.queued) == 1
    assert app.viewport_menu is None  # Native mouse dispatch must finish first.
    fn, args = app.queued.pop()
    fn(*args)
    return app.viewport_menu


def activate(menu, label):
    item = next(item for item in menu.items if item.label == label)
    assert getattr(item, "enabled", True)
    result = item.triggered_fn()
    menu.dismiss()  # OVUI triggers first, then closes its popup.
    return result


def camera_state(app):
    camera = app.renderer.camera
    return (camera.position.tobytes(), camera.yaw, camera.pitch, camera.distance)


@pytest.mark.parametrize("keys", [("object:{actor}",), (GROUP,), (ALIAS,), (GROUP, ALIAS)])
def test_six_actions_preserve_native_selection_camera_and_publication(menu_app, keys):
    app = menu_app
    ids = test_scene_roles(app.session)
    keys = tuple(role_key(key, ids) for key in keys)
    choose(app, *keys)
    before = selection_and_publication(app), camera_state(app)
    app.session = PresentationForbiddenSession(app.session)
    menu = open_menu(app)
    assert menu.options == {"tearable": False} and menu.position == (300., 400.)
    assert [item.label for item in menu.items] == ["Hide selected", "Show selected",
        "Hide all", "Show all", "Center at selected", "Zoom to selected"]
    assert all(item.width == 180 for item in menu.items)
    assert activate(menu, "Hide selected")
    assert (selection_and_publication(app), camera_state(app)) == before
    assert app._row_tags(app.object_model.row_identity(keys[0]))[0]
    if keys == (role_key(ALIAS, ids),): assert not app.interaction.hidden
    else: assert {ids.actor, ids.shape, ids.geometry} <= app.interaction.hidden
    assert activate(open_menu(app), "Show selected")
    assert not app.interaction.hidden
    assert activate(open_menu(app), "Hide all")
    assert app.interaction.hidden == app.interaction.known
    assert activate(open_menu(app), "Show all")
    assert not app.interaction.hidden
    assert (selection_and_publication(app), camera_state(app)) == before


def test_visibility_uses_frozen_scope_without_retargeting_later_selection(menu_app):
    app = menu_app
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids))
    menu = open_menu(app)
    choose(app, f"object:{ids.material}")
    before = selection_and_publication(app), camera_state(app)
    assert activate(menu, "Hide selected")
    assert {ids.actor, ids.shape, ids.geometry} <= app.interaction.hidden and ids.material not in app.interaction.hidden
    assert (selection_and_publication(app), camera_state(app)) == before


@pytest.mark.parametrize("label,center_only", [("Center at selected", True), ("Zoom to selected", False)])
def test_camera_action_uses_owned_selected_bounds_without_seeking(menu_app, label, center_only):
    app = menu_app
    ids = test_scene_roles(app.session)
    choose(app, "object:{actor}")
    before = selection_and_publication(app)
    bounds = app.renderer.snapshot.bounds[app.renderer.snapshot.object_ids == ids.actor]
    expected = Camera()
    expected.frame_bounds((bounds[:, 0].min(0), bounds[:, 1].max(0)), center_only=center_only)
    app.session = PresentationForbiddenSession(app.session)
    assert activate(open_menu(app), label)
    assert np.allclose(app.renderer.camera.position, expected.position)
    assert app.renderer.camera.distance == expected.distance
    assert selection_and_publication(app) == before


@pytest.mark.parametrize("label", ["Center at selected", "Zoom to selected"])
def test_camera_rejects_changed_selection(menu_app, label):
    app = menu_app
    ids = test_scene_roles(app.session)
    choose(app, "object:{actor}")
    menu = open_menu(app)
    choose(app, f"object:{ids.material}")
    before = selection_and_publication(app), camera_state(app)
    assert not activate(menu, label)
    assert "Selection or object contents changed" in app.error_message
    assert (selection_and_publication(app), camera_state(app)) == before


@pytest.mark.parametrize("label", ["Hide selected", "Center at selected", "Zoom to selected"])
def test_dead_current_frame_scope_rejected(menu_app, label):
    app = menu_app
    ids = test_scene_roles(app.session)
    assert app.seek(1, snap=False)
    choose(app, "object:{actor}")
    menu = open_menu(app)
    assert app.seek(3, snap=False)
    settle(app)
    before = selection_and_publication(app), camera_state(app)
    assert not activate(menu, label)
    assert not app.interaction.hidden
    assert (selection_and_publication(app), camera_state(app)) == before


@pytest.mark.parametrize("dx,dy,opens", [(0, 0, True), (1.9, 0, True), (1.4, 1.4, True),
    (2, 0, False), (-2, 0, False), (1.5, 1.5, False)])
def test_right_release_has_strict_radial_two_logical_pixel_tolerance(menu_app, dx, dy, opens):
    app = menu_app
    ids = test_scene_roles(app.session)
    app._mouse_pressed(300, 400, 1, 0)
    app._mouse_released(300 + dx, 400 + dy, 1)
    assert bool(app.queued) is opens and app.mouse_anchor is None


def test_right_drag_cannot_become_click_by_returning_to_origin(menu_app):
    app = menu_app
    ids = test_scene_roles(app.session)
    camera = camera_state(app)
    app._mouse_pressed(300, 400, 1, 0)
    app._mouse_moved(303, 400)
    assert camera_state(app) != camera
    app._mouse_moved(300, 400)
    app._mouse_released(300, 400, 1)
    assert not app.queued and app.mouse_dragged


@pytest.mark.parametrize("button", [0, 1, 2])
@pytest.mark.parametrize("blocked", ["popup", "window"])
def test_popup_or_foreground_window_cancels_press_move_release(menu_app, button, blocked):
    app = menu_app
    ids = test_scene_roles(app.session)
    before = camera_state(app), selection_and_publication(app)
    app._mouse_pressed(300, 400, button, 0)
    if blocked == "popup": Menu.current = object()
    else: app.window.focused = False
    app._mouse_moved(360, 420)
    app._mouse_released(360, 420, button)
    app._mouse_pressed(300, 400, button, 0)
    assert app.mouse_anchor is None and not app.queued
    assert (camera_state(app), selection_and_publication(app)) == before


@pytest.mark.parametrize("change", ["capture", "new", "shutdown", "dismiss", "scene", "segment"])
def test_lifecycle_invalidates_saved_callback_without_mutating_useful_display(menu_app, tmp_path, change):
    app = menu_app
    ids = test_scene_roles(app.session)
    choose(app, "object:{actor}")
    menu = open_menu(app)
    callback = next(item.triggered_fn for item in menu.items if item.label == "Hide all")
    if change == "capture":
        assert app._display(*open_capture(tmp_path / "replacement", "test_scene.ovd"))
        assert menu.destroyed == 1
    elif change == "new":
        app.cancel_load = lambda **_: None
        app.load_generation = 0
        app.workflow.new = lambda: app._display(None, None)
        app.new_capture()
        assert app.session is None and menu.destroyed == 1
    elif change == "shutdown":
        app.running = True
        app.request_exit()
        assert not app.running and menu.destroyed == 1
    elif change == "dismiss": menu.dismiss()
    elif change == "scene": app.scene_id += 1
    else:
        view = app.presentation
        app.presentation = NS(segment_epoch=view.segment_epoch + 1, generation=view.generation)
    before = set(app.interaction.hidden), camera_state(app), app.frame
    assert not callback()
    assert (set(app.interaction.hidden), camera_state(app), app.frame) == before


def test_queued_popup_after_new_capture_is_discarded_and_old_press_cannot_pick(menu_app, tmp_path):
    app = menu_app
    ids = test_scene_roles(app.session)
    app._mouse_pressed(300, 400, 1, 0)
    app._mouse_released(300, 400, 1)
    queued = app.queued.pop()
    app._mouse_pressed(300, 400, 0, 0)
    assert app._display(*open_capture(tmp_path / "replacement", "test_scene.ovd"))
    queued[0](*queued[1])
    app._mouse_released(300, 400, 0)
    assert app.viewport_menu is None and app.mouse_anchor is None


def test_unmatched_release_and_empty_capture_do_not_open_menu(menu_app):
    app = menu_app
    ids = test_scene_roles(app.session)
    app._mouse_pressed(300, 400, 0, 0)
    app._mouse_released(300, 400, 1)
    assert not app.queued
    assert app._display(None, None)
    assert open_menu(app) is None


@pytest.mark.parametrize("owner,secondary", [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)])
def test_camera_drag_retains_initiating_button(menu_app, owner, secondary):
    app = menu_app
    app._mouse_pressed(300, 400, owner, 0)
    app._mouse_moved(320, 410)
    anchor, origin = app.mouse_anchor, app.mouse_origin
    app._mouse_pressed(320, 410, secondary, 1)
    app._mouse_released(320, 410, secondary)
    assert app.mouse_anchor == anchor and app.mouse_origin == origin
    assert app.mouse_modifiers == 0 and app.mouse_dragged
    before = camera_state(app)
    app._mouse_moved(330, 420)
    assert camera_state(app) != before
    app._mouse_released(330, 420, owner)
    before = camera_state(app)
    app._mouse_moved(340, 430)
    assert app.mouse_anchor is None and camera_state(app) == before and not app.queued


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_fresh_owner_press_recovers_a_missed_release(menu_app, owner):
    app = menu_app
    app._mouse_pressed(300, 400, owner, 0)
    app._mouse_double_clicked(300, 400, owner, 0)
    app._mouse_moved(320, 410)
    app._workspace_mouse_moved(360, 440)
    before = camera_state(app)
    # No release callback: the next physical press must start a fresh gesture.
    app._mouse_pressed(500, 500, owner, 1)
    callback, args = app.queued.pop()
    callback(*args)
    assert camera_state(app) == before
    assert app.mouse_modifiers == 1 and not app.mouse_double_clicked
    app._mouse_moved(501, 500)
    assert camera_state(app) == before and not app.mouse_dragged
    app._mouse_moved(520, 510)
    assert camera_state(app) != before
    app._mouse_released(520, 510, owner)
    assert app.mouse_anchor is None and not app.queued


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_camera_does_not_move_after_owner_released_with_another_button_held(menu_app, owner):
    app = menu_app
    app._mouse_pressed(300, 400, owner, 0)
    app._mouse_moved(320, 410)
    before = camera_state(app)
    # Native OVUI dispatches workspace movement before viewport release.
    app._workspace_mouse_moved(360, 440)
    app._mouse_released(360, 440, owner)
    for callback, args in app.queued:
        callback(*args)
    app.queued.clear()
    assert app.mouse_anchor is None and camera_state(app) == before and not app.queued


@pytest.mark.parametrize("owner", [0, 1, 2])
def test_escape_cancels_camera_until_a_fresh_viewport_press(menu_app, owner):
    app = menu_app
    app._mouse_pressed(300, 400, owner, 0)
    app._mouse_moved(320, 410)
    app._key_pressed(87, 0, True)
    assert app.navigation.moving
    app._key_pressed(256, 0, True)
    before = camera_state(app)
    app._mouse_moved(360, 440)
    app._mouse_released(360, 440, owner)
    assert app.mouse_anchor is None and not app.navigation.moving
    assert camera_state(app) == before and not app.queued
    app._mouse_pressed(300, 400, owner, 0)
    app._mouse_moved(320, 410)
    assert camera_state(app) != before


def test_queued_camera_movement_cannot_transfer_to_a_new_gesture(menu_app):
    app = menu_app
    app._mouse_pressed(300, 400, 1, 0)
    app._workspace_mouse_moved(360, 440)
    app._key_pressed(256, 0, True)
    app._mouse_pressed(500, 500, 2, 0)
    before = camera_state(app)
    callback, args = app.queued.pop()
    callback(*args)
    assert camera_state(app) == before and app.mouse_anchor == (500, 500, 2)


def test_left_drag_released_before_queued_movement_cannot_pick(menu_app):
    app = menu_app
    app._mouse_pressed(300, 400, 0, 0)
    app._workspace_mouse_moved(360, 440)
    app._mouse_released(360, 440, 0)
    callback, args = app.queued.pop()
    callback(*args)
    assert app.mouse_anchor is None


def test_real_tcp_resampling_publication_closes_old_segment_menu(menu_app, live):
    app = menu_app
    ids = test_scene_roles(app.session)
    source, connection = live
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    first_end = 42117
    stop = struct.pack("<BQQ", 12, 2929590687920, 1)
    second_end = payload.index(stop, first_end) + len(stop)
    connection.sendall(HANDSHAKE + payload[:first_end])
    wait_until(lambda: source.info().completed_serial == 120)
    app.workflow.live_session = source
    assert app._display(None, source)
    menu = open_menu(app)
    context = app._viewport_context()
    callback = next(item.triggered_fn for item in menu.items if item.label == "Hide all")
    connection.sendall(payload[first_end:second_end])
    wait_until(lambda: source.info().completed_serial == 121)
    update = source.refresh_live()
    app.workflow.follow_live = True
    event = NS(kind="frames", session=source, scenes=update.scenes, segment_changed=True,
               max_frame=update.max_frame, current_frame=update.current_frame)
    app.workflow.poll = lambda: [event]
    app._poll_workflow()
    assert app._viewport_context() != context
    assert menu.destroyed == 1 and app.viewport_menu is None
    before = camera_state(app), selection_and_publication(app)
    assert not callback() and not app.interaction.hidden
    assert (camera_state(app), selection_and_publication(app)) == before
