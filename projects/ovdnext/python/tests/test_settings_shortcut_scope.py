# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Settings dispatch with real native views/models and window/focus sinks.

The explicit focus transitions reproduce dispatcher inputs, not physical
desktop focus delivery. GPU/native window validation is separate.
"""
from copy import deepcopy
from types import SimpleNamespace

import omni.ui as ui
import pytest

from test_application_presentation import app, displayed, open_capture
from test_clip_shortcuts import keyboard
from test_general_settings import make_view


@pytest.fixture
def settings_app(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd"))
    assert app.seek(2, snap=False)
    calls = keyboard(app, viewport=False)
    view, preferences, applied = make_view(tmp_path)
    view.window.visible = False
    view.window.focus = lambda: calls.append("settings")
    for name, _ in view.SECTIONS:
        view.sections[name] = SimpleNamespace(visible=False)
        view.tabs[name] = SimpleNamespace(name="tab")
    app.render_settings_view = view
    before = (displayed(app), deepcopy(app.playback.__dict__), preferences.path.read_bytes())
    return app, view, preferences, applied, calls, before


def assert_unchanged(probe):
    app, view, preferences, applied, calls, before = probe
    assert displayed(app) == before[0]
    assert app.playback.__dict__ == before[1]
    assert preferences.path.read_bytes() == before[2]
    assert not applied and not app.presentation_dirty


@pytest.mark.parametrize("capture", (0, ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD))
def test_settings_window_shortcut_opens_real_controller_without_publication(settings_app, capture):
    app, view, _, _, calls, _ = settings_app
    # OVUI uses the same broad capture bit for main-window button and editor
    # focus. Both must reach the real dialog controller without claiming camera.
    app._key_pressed(83, 3 | capture, True)
    assert calls == ["settings"] and view.window.visible
    assert view.section == "rendering" and view.sections["rendering"].visible
    assert not app.viewport_keyboard_focus
    app._key_pressed(83, 3 | capture, False)
    assert calls == ["settings"]
    assert_unchanged(settings_app)


def test_settings_window_shortcut_keeps_an_existing_dialog_draft(settings_app):
    app, view, _, _, calls, _ = settings_app
    view.window.visible = True
    view._select_section("properties")
    draft = view.property_controls["float_precision"].model
    draft.set_value("25")
    app._key_pressed(83, 3 | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD, True)
    assert calls == ["settings"] and view.section == "rendering"
    assert draft.as_string == "25"
    assert_unchanged(settings_app)


def test_after_child_close_requires_returned_main_window_focus(settings_app):
    app, view, _, _, calls, _ = settings_app
    app.view_window = SimpleNamespace(visible=True)
    app.window.focused = False
    app._window_focused(False)
    chord = 3 | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD
    app._key_pressed(83, chord, True)
    assert not calls and not view.window.visible
    # Hiding a child alone cannot authorize another window's keyboard input.
    app.view_window.visible = False
    app._key_pressed(83, chord, True)
    assert not calls and not view.window.visible
    app.window.focused = True
    app._window_focused(True)
    app._key_pressed(83, chord, True)
    assert calls == ["settings"] and view.window.visible
    assert not app.viewport_keyboard_focus
    assert_unchanged(settings_app)


def test_other_s_commands_keep_the_editor_capture_guard(settings_app):
    app, view, _, _, calls, _ = settings_app
    for chord in range(16):
        if chord == 3:
            continue
        app._key_pressed(83, chord | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD, True)
    assert not calls and not view.window.visible
    assert_unchanged(settings_app)
