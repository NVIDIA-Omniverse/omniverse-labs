# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact clip chords through real Application, Playback and native publication.

UI widgets and GPU are sinks, matching the existing presentation fixture.
No desktop application or OpenGL context is started by these checks.
"""
from copy import deepcopy
from types import SimpleNamespace

import omni.ui as ui
import pytest

from ovd_ui.camera import Camera, CameraController
from ovd_ui.playback import Playback
from test_application_presentation import app, open_capture


def keyboard(application, *, viewport=True):
    application.window=SimpleNamespace(focused=True)
    application.viewport_keyboard_focus=viewport
    application.object_keyboard_focus=False
    application.navigation=CameraController(Camera())
    calls=[]
    application.show_open=lambda:calls.append("open")
    application.new_capture=lambda:calls.append("new")
    application.show_save=lambda:calls.append("save")
    application.render_settings_view=SimpleNamespace(show=lambda:calls.append("settings"))
    return calls


@pytest.mark.parametrize("key,expected_frame,expected_ranges",[
    (89,1,[(1,31),(31,772)]),
    (83,30,[(30,772)]),
    (69,29,[(1,30)]),
])
@pytest.mark.parametrize("panel_open",[False,True])
def test_clip_shortcuts_publish_real_native_frame_before_or_after_panel_opens(
        app,tmp_path,key,expected_frame,expected_ranges,panel_open):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(30,snap=False)
    assert app.playback.clip.stop==772
    calls=keyboard(app)
    if panel_open:
        before=app.clips_view.refreshes
    else:
        del app.clips_view
    app._key_pressed(key,6,True)
    assert calls==[]
    assert [(c.start,c.stop) for c in app.playback.clips]==expected_ranges
    assert app.frame==app.playback.frame==app.presentation.frame==expected_frame
    assert app.presentation.request_applied and not app.presentation_dirty
    if panel_open:assert app.clips_view.refreshes==before+1
    else:assert not hasattr(app,"clips_view")
    app._request_presentation()
    assert app._refresh()
    assert [(c.start,c.stop) for c in app.playback.clips]==expected_ranges


@pytest.mark.parametrize("key",[89,83,69])
@pytest.mark.parametrize("blocked",["release","unfocused","editing","shift","super"])
def test_clip_chord_rejects_release_other_windows_text_and_extra_modifiers(app,tmp_path,key,blocked):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(30,snap=False)
    calls=keyboard(app)
    before=deepcopy(app.playback.__dict__)
    generation=app.presentation.generation
    modifiers=6
    pressed=True
    if blocked=="release":pressed=False
    elif blocked=="unfocused":app.window.focused=False
    elif blocked=="editing":
        app.viewport_keyboard_focus=False
        modifiers|=ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD
    elif blocked=="shift":modifiers|=1
    elif blocked=="super":modifiers|=8
    app._key_pressed(key,modifiers,pressed)
    assert calls==[]
    assert app.playback.__dict__==before
    assert app.presentation.generation==generation


def test_clip_chords_have_workspace_scope_without_claimed_text_input(app,tmp_path):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(30,snap=False)
    keyboard(app,viewport=False)
    app._key_pressed(83,6,True)
    assert app.playback.clip.start==app.presentation.frame==30


@pytest.mark.parametrize("key,name",[(79,"open"),(78,"new"),(83,"save")])
def test_file_shortcuts_require_exact_control_chord(app,key,name):
    calls=keyboard(app)
    for modifiers in (0,1,4,8,3,10,7,14,15):
        app._key_pressed(key,modifiers,True)
    # The S key with Ctrl+Alt belongs only to Trim; it is harmless without a
    # published clip. O/N with Alt must not become file commands either.
    app._key_pressed(key,6,True)
    assert calls==(["settings"] if key==83 else [])
    app._key_pressed(key,2,True)
    assert calls==(["settings"] if key==83 else [])+[name]
    app._key_pressed(key,2,False)
    assert calls==(["settings"] if key==83 else [])+[name]


@pytest.mark.parametrize("blocked",("release","unfocused","alt","super"))
@pytest.mark.parametrize("capture",(0,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD))
def test_settings_shortcut_respects_window_and_exact_modifiers(app,blocked,capture):
    calls=keyboard(app,viewport=False)
    modifiers,pressed=3|capture,True
    if blocked=="release":pressed=False
    elif blocked=="unfocused":app.window.focused=False
    elif blocked=="alt":modifiers|=4
    else:modifiers|=8
    app._key_pressed(83,modifiers,pressed)
    assert calls==[]


def test_settings_shortcut_has_workspace_scope_without_a_capture(app):
    calls=keyboard(app,viewport=False)
    app._key_pressed(83,3,True)
    assert calls==["settings"]


@pytest.mark.parametrize("missing",["session","scenes","clip"])
@pytest.mark.parametrize("key",[89,83,69])
def test_clip_shortcuts_require_a_displayed_clip_without_a_panel(app,key,missing):
    keyboard(app)
    app.session=object()
    app.scenes=[object()]
    app.playback=Playback()
    app.playback.reset(20)
    app.playback.seek(10)
    if missing=="session":app.session=None
    elif missing=="scenes":app.scenes=[]
    else:app.playback.clips=[]
    del app.clips_view
    app.seek=lambda *_args,**_kwargs:pytest.fail("An unavailable clip must not seek")
    before=deepcopy(app.playback.__dict__)
    app._key_pressed(key,6,True)
    assert app.playback.__dict__==before
