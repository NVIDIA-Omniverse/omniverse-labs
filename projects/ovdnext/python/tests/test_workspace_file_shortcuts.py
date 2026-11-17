# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main-window file shortcuts must survive embedded analysis/editor focus."""
from types import SimpleNamespace
import omni.ui as ui
import pytest
from ovd_ui.application import Application


@pytest.mark.parametrize('key,expected',[(79,'open'),(78,'new'),(83,'save')])
@pytest.mark.parametrize('capture_keyboard',[0,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD])
def test_file_actions_work_with_embedded_page_navigation_focus(key,expected,capture_keyboard):
    app=Application.__new__(Application)
    calls=[]
    app.window=SimpleNamespace(focused=True)
    app.navigation=None
    app.viewport_keyboard_focus=False
    app.workspace=SimpleNamespace(key=lambda *_:False,navigation_focused=True)
    app.show_open=lambda:calls.append('open')
    app.new_capture=lambda:calls.append('new')
    app.show_save=lambda:calls.append('save')
    app._key_pressed(key,2|capture_keyboard,True)
    assert calls==[expected]


@pytest.mark.parametrize('modifiers,pressed,focused',[
    (2,False,True),(2,True,False),(0,True,True),(1,True,True),
    (3,True,True),(6,True,True),(10,True,True)])
def test_file_shortcuts_reject_release_other_windows_and_extra_modifiers(modifiers,pressed,focused):
    app=Application.__new__(Application)
    calls=[]
    app.window=SimpleNamespace(focused=focused)
    app.navigation=None
    app.viewport_keyboard_focus=False
    app.show_open=lambda:calls.append('open')
    app.new_capture=lambda:calls.append('new')
    app.show_save=lambda:calls.append('save')
    app._key_pressed(79,modifiers|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,pressed)
    assert not calls


@pytest.mark.parametrize('key',[79,78,83])
def test_embedded_text_editor_keeps_file_shortcut_capture_guard(key):
    app=Application.__new__(Application)
    app.window=SimpleNamespace(focused=True)
    app.navigation=None
    app.viewport_keyboard_focus=False
    app.workspace=SimpleNamespace(key=lambda *_:False,navigation_focused=False)
    calls=[]
    app.show_open=lambda:calls.append('open')
    app.new_capture=lambda:calls.append('new')
    app.show_save=lambda:calls.append('save')
    app._key_pressed(key,2|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert not calls
