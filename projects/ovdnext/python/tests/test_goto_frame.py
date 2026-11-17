# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame-jump semantics, keyboard cancellation and real native publication."""
from copy import deepcopy
from types import SimpleNamespace

import omni.ui as ui
import pytest

from ovd_ui.playback import Playback, FrameMode
from ovd_ui.widgets.goto_frame import GoToFrameDialog, frame_target
from test_application_presentation import app, open_capture


def test_frame_jump_uses_native_numbers_and_current_clip_bounds():
    playback = Playback()
    playback.reset(101)
    playback.seek(25)
    playback.trim_start()
    playback.seek(70)
    playback.trim_end()
    before = deepcopy(playback.__dict__)
    assert frame_target("25", playback) == (25, "Frame 25")
    assert frame_target(" +0069 ", playback) == (69, "Frame 69")
    for value in ("1", "24", "70", "101", "-1", "1.5", "1_0", "0x20", "１２", ""):
        with pytest.raises(ValueError):
            frame_target(value, playback)
    assert playback.__dict__ == before


def test_frame_jump_snaps_pre_post_and_handles_last_partial_step():
    playback = Playback()
    playback.reset(8, ovd=True)
    assert frame_target("3", playback) == (4, "Frame 4 · 2 Post")
    assert frame_target("7", playback) == (6, "Frame 6 · 3 Post")
    playback.set_mode(FrameMode.PRE)
    assert frame_target("4", playback) == (5, "Frame 5 · 3 Pre")
    playback.set_mode(FrameMode.FULL)
    assert frame_target("4", playback) == (4, "Frame 4 · 2 Post")
    playback.reset(2, ovd=True)
    assert frame_target("1", playback) == (1, "Frame 1 · 1 Pre")
    playback.reset(1, ovd=True)
    assert frame_target("0", playback) == (0, "Frame 0 · 0 Post")


def test_frame_jump_does_not_parse_unbounded_integer_or_missing_capture():
    playback = Playback()
    with pytest.raises(ValueError, match="Open a capture"):
        frame_target("1", playback)
    playback.reset(3)
    assert frame_target("0" * 5000 + "1", playback)[0] == 1
    with pytest.raises(ValueError, match="Choose a frame"):
        frame_target("1" * 5000, playback)


def dialog_for(playback, callback):
    dialog = GoToFrameDialog(callback)
    dialog.window = SimpleNamespace(visible=True)
    dialog.model = ui.SimpleStringModel(str(playback.frame))
    dialog.playback = deepcopy(playback)
    dialog.context = ("owned context",)
    dialog.preview = SimpleNamespace(text="")
    dialog.go = SimpleNamespace(enabled=True)
    return dialog


def test_modal_return_validation_escape_and_failure_keep_owned_snapshot():
    playback = Playback()
    playback.reset(11, ovd=True)
    selected = []
    dialog = dialog_for(playback, lambda frame, context: selected.append((frame, context)))
    dialog.model.set_value("99")
    dialog.key(257, 0, True)
    assert dialog.window.visible and not selected and not dialog.go.enabled
    dialog.model.set_value("3")
    dialog.key(257, 0, False)
    assert not selected
    dialog.key(256, 0, True)
    assert not dialog.window.visible and not selected and playback.frame == 2
    dialog.window.visible = True
    dialog.key(257, 0, True)
    assert selected == [(4, ("owned context",))] and not dialog.window.visible
    assert playback.frame == 2
    dialog.window.visible = True
    def reject(*_):
        raise ValueError("The capture changed")
    dialog.on_apply = reject
    assert not dialog.apply() and dialog.window.visible
    assert dialog.preview.text == "The capture changed"


def test_jump_publishes_native_frame_and_pauses_without_changing_clip(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd"))
    app.playback.seek(20)
    app.playback.trim_start()
    app.playback.toggle()
    context = app._frame_jump_context()
    target, _ = frame_target("23", app.playback)
    assert target == 24
    app._goto_frame(target, context)
    assert app.presentation.frame == app.playback.frame == app.frame == 24
    assert not app.playback.playing
    assert app.playback.clip.start == 20
    assert next(scene.current_frame for scene in app.scenes if scene.id == app.scene_id) == 24


def test_jump_context_rejects_changed_scene_clip_mode_and_capture(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd"))
    context = app._frame_jump_context()
    before = app.frame
    app.playback.set_mode(FrameMode.PRE)
    with pytest.raises(ValueError, match="changed"):
        app._goto_frame(10, context)
    assert app.frame == before
    app.playback.set_mode(FrameMode.POST)
    app.playback.clip.stop -= 1
    with pytest.raises(ValueError, match="changed"):
        app._goto_frame(10, context)
    app.playback.clip.stop += 1
    app.scene_id += 1
    with pytest.raises(ValueError, match="changed"):
        app._goto_frame(10, context)
    app.scene_id -= 1
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    with pytest.raises(ValueError, match="changed"):
        app._goto_frame(10, context)


def test_ctrl_g_opens_from_text_focus_without_camera_navigation(app):
    opened = []
    app.window = SimpleNamespace(focused=True)
    app.show_goto_frame = lambda: opened.append(True)
    app._key_pressed(71, 2 | ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD, True)
    assert opened == [True]


def test_missing_capture_reports_status_without_opening_dialog(app):
    opened = []
    app.goto_frame_dialog = SimpleNamespace(show=lambda *_: opened.append(True))
    app.show_goto_frame()
    assert not opened and app.status_label.text == "Open a capture with published frames first."
