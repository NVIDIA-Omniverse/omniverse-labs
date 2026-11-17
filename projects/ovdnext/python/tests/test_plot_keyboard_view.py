# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyboard/mouse integration over owned native scalar series and display state."""
from types import SimpleNamespace as NS

import pytest
import omni.ui as ui

from ovd_ui.analysis import HistoryRequest, read_scalar_history
from ovd_ui.graph_keyboard import KeyboardCrosshair, accepted_rectangle
from ovd_ui.plotting import Curve, PlotModel
from ovd_ui.widgets.analysis_views import PlotView
from test_application_presentation import app, open_capture, Frame
from test_application_object_actions import choose, selection_and_publication


def probe(series):
    view = PlotView.__new__(PlotView)
    view.series = series
    view.model = PlotModel(series.frames, [Curve(series.name, series.values, series.unit)])
    view.keyboard = KeyboardCrosshair()
    view.keyboard_navigation_flags = None
    view.window = NS(focused=True, visible=True, flags=8)
    view.canvas = NS(screen_position_x=100., screen_position_y=70., computed_width=500., computed_height=300.)
    view.keyboard.configure(500, 300)
    view.destroyed = False
    view.editing_fields = set()
    view.export_dialog = view.context_menu = view.drag = None
    view.cursor = NS(text="")
    view.overlay = Frame()
    view._sync = lambda: view.overlay.rebuild()
    view.navigated = []
    view.on_frame = view.navigated.append
    return view


@pytest.fixture
def native_view(app, tmp_path):
    path, session = open_capture(tmp_path, "test_scene.ovd")
    assert app._display(path, session)
    actor = next(instance for instance in app.adapter.instances.values()
                 if instance.alive and app.adapter.class_name(instance.id) == "PxRigidDynamic")
    choose(app, f"object:{actor.id}")
    prop = next(p for p in session.properties(actor.class_id) if p.name == "mass")
    request = HistoryRequest.from_display(path, session, actor.id, prop.id, scene_id=app.scene_id)
    series = read_scalar_history(request)
    assert series.frames == (1, 2, 3) and series.values == (2., 2., None)
    return app, probe(series)


def press(view, key, modifiers=0):
    view._key(key, modifiers | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD, True)
    view._key(key, modifiers | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD, False)


def test_keyboard_rectangle_keeps_owned_series_and_native_playhead(native_view):
    app, view = native_view
    before = selection_and_publication(app)
    series = view.series
    original = view.model.view
    view._passive_hover(.2, .3)
    press(view, 257)
    assert view.keyboard.anchor == (100, 90)
    snapped = view.model.cursor_frame
    press(view, 262)
    assert view.keyboard.position == (101, 90) and view.model.cursor_frame == snapped
    press(view, 264)
    assert view.keyboard.position == (101, 91)
    assert view.window.flags == 8 | (1 << 16)
    press(view, 335)  # Keypad Enter also finishes, without a frame jump.
    assert not view.keyboard.active and view.model.history_index == 1
    rectangle = accepted_rectangle((100, 90), (101, 91))
    expected = PlotModel(series.frames, [Curve(series.name, series.values, series.unit)])
    expected.rectangle(rectangle[0]/500, rectangle[1]/300, rectangle[2]/500, rectangle[3]/300, origin=original)
    assert view.model.view == expected.view and view.model.history == expected.history
    assert view.series is series and view.model.curves[0].values is series.values
    assert not view.navigated and selection_and_publication(app) == before
    press(view, 79, 1)
    assert view.model.view == original and view.model.history_index == 0
    assert view.window.flags == 8
    press(view, 73, 1)
    assert view.model.view == expected.view and view.model.history_index == 1
    assert selection_and_publication(app) == before


@pytest.mark.parametrize("key", [257, 335, 525, 627])
def test_return_aliases_start_and_rejected_point_end_without_menu(native_view, key):
    _, view = native_view
    before = view.model.view, list(view.model.history)
    press(view, key)
    assert view.keyboard.active
    press(view, key)
    assert not view.keyboard.active and view.context_menu is None
    assert (view.model.view, view.model.history) == before


@pytest.mark.parametrize("modifiers", range(1, 16))
@pytest.mark.parametrize("key", [262, 263, 264, 265, 257, 335, 256])
def test_new_routes_accept_exact_unmodified_chords_only(native_view, key, modifiers):
    _, view = native_view
    before = view.model.view, list(view.model.history)
    press(view, key, modifiers)
    assert not view.keyboard.visible and not view.keyboard.active
    assert (view.model.view, view.model.history) == before


@pytest.mark.parametrize("block", ["focus", "editor", "export", "menu", "resize", "hidden", "destroy"])
def test_pending_rectangle_and_repeat_cancel_without_history_or_navigation(native_view, monkeypatch, block):
    app, view = native_view
    before = selection_and_publication(app), view.model.view, list(view.model.history)
    press(view, 257)
    view._key(262, 0, True)
    assert view.keyboard.active and view.keyboard.held == 262
    if block == "focus": view.window.focused = False
    elif block == "editor": view._field_edit("start", True)
    elif block == "export": view.export_dialog = NS(window=NS(visible=True))
    elif block == "menu": monkeypatch.setattr(ui.Menu, "get_current", staticmethod(lambda: object()))
    elif block == "resize": view.canvas.computed_width += .25
    elif block == "hidden": view.window.visible = False
    else:
        view.job = NS(cancel=lambda: None)
        view.hover_overlay = NS(destroy=lambda: None)
        view.chart_scene = None
        view.window.destroy = lambda: None
        view.destroy()
    if block != "destroy": view._poll_keyboard()
    assert not view.keyboard.active and view.keyboard.held is None and not view.keyboard.visible
    assert not view.navigated and view.window.flags == 8
    assert (selection_and_publication(app), view.model.view, view.model.history) == before


def test_physical_mouse_motion_and_mouse_press_take_over_without_stale_repeat(native_view):
    _, view = native_view
    view._passive_hover(.2, .3)
    press(view, 257)
    view._key(262, 0, True)
    assert view.keyboard.position == (101, 90)
    view._passive_hover(.2, .3)
    assert view.keyboard.position == (101, 90)
    view._passive_hover(.4, .5)
    assert view.keyboard.active and view.keyboard.position == (200, 150)
    assert view.keyboard.held is None and view.keyboard.anchor == (100, 90)
    view._press(350, 220, 1, 0)
    assert not view.keyboard.active and not view.keyboard.visible and view.window.flags == 8
    view._release(450, 270, 1, 0)
    assert view.model.history_index == 1 and not view.navigated


def test_mouse_and_keyboard_commit_the_same_integer_pixel_rectangle(native_view):
    _, view = native_view
    other = probe(view.series)
    first, last = (100, 90), (201, 181)
    view._press(200, 160, 1, 0)
    view._release(301, 251, 1, 0)
    other._passive_hover(.2, .3)
    press(other, 257)
    other._passive_hover(last[0]/500, last[1]/300)
    press(other, 257)
    assert view.model.history == other.model.history and view.model.history_index == 1
    assert view.model.view == other.model.view and not view.navigated and not other.navigated


def test_completed_analysis_poll_still_repeats_without_a_native_or_job_read(native_view, monkeypatch):
    _, view = native_view
    clock = [10.]
    monkeypatch.setattr("ovd_ui.widgets.analysis_views.time.monotonic", lambda: clock[0])
    view.finished = True
    view.chart_signature = (500., 300., ui.ColorStore.find("pvd_accent"), ui.ColorStore.find("pvd_warning"))
    view._key(262, 0, True)
    initial = view.keyboard.position
    clock[0] += .401
    view.poll()  # No job is supplied: a finished graph must not read its worker.
    assert view.keyboard.position == (initial[0] + 5, initial[1])
    view._key(262, 0, False)
    clock[0] += 100
    view.poll()
    assert view.keyboard.position == (initial[0] + 5, initial[1])
