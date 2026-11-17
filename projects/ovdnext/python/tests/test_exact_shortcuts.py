# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared command chords, collisions and owned-state safety without a GPU."""
from capture_semantics import test_scene_roles, unique_instance_id
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.application import Application
from ovd_ui.camera import Camera, CameraController, normalize_key
from ovd_ui.playback import Playback
from ovd_ui.plotting import Curve, PlotModel
from ovd_ui.widgets.analysis_views import PlotView
from test_application_presentation import app, open_capture
from test_find_objects import adapter, dialog_for as find_dialog
from test_goto_frame import dialog_for as frame_dialog


# Complete commands sharing each tested physical key, including F's three scopes.
COMMANDS={70:{0:'frame_selected',2:'find',4:'profile'},71:{2:'goto'},
          82:{3:'reset_visual_settings',4:'reset_camera'},84:{4:'track_selected'},67:{4:'center_selected'},
          90:{2:'frame_all',4:'frame_selected'},32:{0:'play'},
          72:{0:'hide',1:'show_all'},262:{2:'step:1'},263:{2:'step:-1'},
          268:{0:'first'},269:{0:'last'}}


def command_probe(*,viewport=True):
    view=Application.__new__(Application)
    view.window=NS(focused=True)
    view.viewport_keyboard_focus=viewport
    view.object_keyboard_focus=False
    view.navigation=CameraController(Camera())
    view.playback=Playback();view.playback.reset(20)
    calls=[]
    for name in ('reset_visual_settings','reset_camera','track_selected','center_selected','frame_selected','frame_all'):
        setattr(view,name,lambda command=name:calls.append(command))
    view.show_find_profile=lambda:calls.append('profile')
    view.show_goto_frame=lambda:calls.append('goto')
    view.object_browser=NS(show_find=lambda:calls.append('find'),search_key=lambda *_:False)
    view._object_action=lambda action:calls.append(action)
    view.toggle_play=lambda:calls.append('play')
    view._sync_transport=lambda:calls.append('pause')
    view._transport=lambda action,*args:calls.append(action.__name__+(':'+str(args[0]) if args else ''))
    return view,calls


@pytest.mark.parametrize('key',COMMANDS)
@pytest.mark.parametrize('chord',range(16))
def test_all_command_modifier_combinations_have_only_the_declared_destination(key,chord):
    view,calls=command_probe()
    view._key_pressed(key,chord,True)
    expected=COMMANDS[key].get(chord)
    assert calls==([expected] if expected else [])
    view._key_pressed(key,chord,False)
    assert calls==([expected] if expected else []),'Key release activated a command'


@pytest.mark.parametrize('key,chord',[(82,4),(84,4),(67,4),(90,4),(90,2),
    (32,0),(70,0),(72,0),(72,1),(262,2),(263,2),(268,0),(269,0)])
@pytest.mark.parametrize('blocked',('background','another_surface','editing'))
def test_viewport_commands_keep_existing_window_and_text_ownership(key,chord,blocked):
    view,calls=command_probe(viewport=blocked=='background')
    if blocked=='background':view.window.focused=False
    if blocked=='editing':chord|=ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD
    view._key_pressed(key,chord,True)
    assert not calls


@pytest.mark.parametrize('key,name',[(70,'find'),(71,'goto')])
def test_find_commands_accept_exact_control_in_text_but_reject_super(key,name):
    view,calls=command_probe(viewport=False)
    view._key_pressed(key,10|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert not calls
    view._key_pressed(key,2|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert calls==[name]
    view.window.focused=False
    view._key_pressed(key,2,True)
    assert calls==[name]


@pytest.mark.parametrize('chord',range(16))
def test_object_tree_hide_show_conveniences_are_exact_without_viewport_ownership(chord):
    view,calls=command_probe(viewport=False)
    view.object_keyboard_focus=True
    view._key_pressed(72,chord,True)
    assert calls==({0:['hide'],1:['show_all']}.get(chord,[]))


def test_rejected_commands_leave_real_owned_frame_selection_visibility_and_camera_unchanged(app,tmp_path):
    app.navigation=CameraController(Camera())
    app.window=NS(focused=True)
    app.viewport_keyboard_focus=True;app.object_keyboard_focus=False
    assert app._display(*open_capture(tmp_path,'test_scene.ovd'))
    assert app.seek(1,snap=False)
    app.selection.publish(['/objects/1'],source='test')
    assert app._refresh()
    camera=app.navigation.camera
    camera.turn(12,7)
    before=(camera.position.copy(),camera.target.copy(),camera.view_matrix().copy(),camera.fov)
    presentation=app.presentation;playback=deepcopy(app.playback.__dict__)
    interaction=deepcopy(app.interaction.__dict__)
    selected=app.selection.get_snapshot().paths()
    for key,chord in ((82,6),(84,5),(67,12),(90,3),(90,6),(70,6),
                      (32,2),(262,3),(263,10),(268,4),(269,8),(72,8),(72,9)):
        app._key_pressed(key,chord,True)
    assert app.presentation is presentation and not app.presentation_dirty
    assert app.playback.__dict__==playback and app.interaction.__dict__==interaction
    assert app.selection.get_snapshot().paths()==selected
    for actual,wanted in zip((camera.position,camera.target,camera.view_matrix(),camera.fov),before):
        np.testing.assert_array_equal(actual,wanted)


@pytest.mark.parametrize('key,expected',[(262,3),(263,1),(268,1),(269,3)])
def test_exact_transport_chords_still_publish_native_frame(app,tmp_path,key,expected):
    app.navigation=CameraController(Camera())
    app.window=NS(focused=True);app.viewport_keyboard_focus=True;app.object_keyboard_focus=False
    assert app._display(*open_capture(tmp_path,'test_scene.ovd'))
    # This OVD fixture's native clip spans frames 1 through 3.
    from ovd_ui.playback import FrameMode
    app.playback.set_mode(FrameMode.FULL)
    assert app.seek(2,snap=False)
    app._key_pressed(key,2 if key in (262,263) else 0,True)
    assert app.frame==app.playback.frame==app.presentation.frame==expected
    assert app.presentation.request_applied and not app.presentation_dirty


@pytest.mark.parametrize('key',(257,335,525,627))
def test_find_enter_variants_search_then_activate_only_one_owned_result(adapter,key):
    dialog,selected=find_dialog(adapter)
    dialog.expression.set_value('sphere')
    dialog.key(key,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert dialog.focus=='results' and len(dialog.model.items)==1 and not selected
    dialog.key(key,0,False)
    assert not selected
    dialog.key(key,0,True)
    assert selected==[adapter.fixture_roles.actor] and not dialog.window.visible
    assert dialog.browser.model.filter=='unrelated sidebar filter'


@pytest.mark.parametrize('key',(257,335,525,627))
def test_find_enter_keeps_invalid_and_stale_results_unselected(adapter,key):
    dialog,selected=find_dialog(adapter)
    dialog.mode=1;dialog.expression.set_value('not a number')
    dialog.key(key,0,True);dialog.key(key,0,True)
    assert not selected and dialog.window.visible and not dialog.model.items
    dialog.mode=0;dialog.focus='expression';dialog.expression.set_value('sphere')
    dialog.key(key,0,True)
    dialog.capture_key='replaced capture'
    dialog.key(key,0,True)
    assert not selected and dialog.window.visible and 'capture changed' in dialog.status.text


@pytest.mark.parametrize('key',(257,335,525,627))
def test_frame_enter_variants_keep_validation_and_stale_owner_guard(key):
    playback=Playback();playback.reset(11,ovd=True)
    selected=[]
    dialog=frame_dialog(playback,lambda *args:selected.append(args))
    dialog.model.set_value('99');dialog.key(key,0,True)
    assert not selected and dialog.window.visible and not dialog.go.enabled
    dialog.model.set_value('3');dialog.key(key,0,False)
    assert not selected
    dialog.key(key,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert selected==[(4,('owned context',))] and not dialog.window.visible
    dialog.window.visible=True
    def reject(*_):raise ValueError('The capture changed')
    dialog.on_apply=reject;dialog.key(key,0,True)
    assert dialog.window.visible and dialog.preview.text=='The capture changed'
    dialog.window.destroy=lambda:None
    dialog.destroy();dialog.key(key,0,True)
    assert selected==[(4,('owned context',))]


@pytest.mark.parametrize('key',(257,335))
@pytest.mark.parametrize('chord',range(1,16))
def test_frame_acceptance_rejects_every_extra_modifier(key,chord):
    playback=Playback();playback.reset(20)
    selected=[];dialog=frame_dialog(playback,lambda *args:selected.append(args))
    dialog.key(key,chord|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert not selected and dialog.window.visible


def test_keypad_normalization_agrees_with_public_window_translation():
    assert normalize_key(627)==normalize_key(335)==335


def plot_probe():
    view=PlotView.__new__(PlotView)
    from ovd_ui.graph_keyboard import KeyboardCrosshair
    view.keyboard=KeyboardCrosshair();view.destroyed=False
    view.keyboard_navigation_flags=None
    view.window=NS(focused=True,visible=True,flags=0);view.editing_fields=set();view.export_dialog=None
    view.model=PlotModel(range(10),[Curve('v',tuple(float(i) for i in range(10)))])
    view.model.rectangle(.1,.1,.9,.9);view.model.rectangle(.2,.2,.8,.8)
    view._sync=lambda:None
    view.drag=('mark',);view.overlay=NS(rebuild=lambda:None)
    return view


@pytest.mark.parametrize('key,accepted',[(268,{0}),(79,{1}),(73,{1}),
    (61,{0,1}),(334,{0}),(45,{0}),(333,{0}),(256,{0})])
@pytest.mark.parametrize('chord',range(16))
def test_plot_complete_key_patterns_preserve_history_and_aliases(key,accepted,chord):
    view=plot_probe()
    if key==73:view.model.undo()
    before=(view.model.view,list(view.model.history),view.model.history_index,view.drag)
    view._key(key,chord|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    after=(view.model.view,list(view.model.history),view.model.history_index,view.drag)
    assert (after!=before)==(chord in accepted)


@pytest.mark.parametrize('blocked',('release','background','field','export'))
def test_plot_keyboard_guard_keeps_modal_and_editor_ownership(blocked):
    view=plot_probe();before=(view.model.view,list(view.model.history),view.model.history_index)
    if blocked=='background':view.window.focused=False
    if blocked=='field':view.editing_fields.add('range')
    if blocked=='export':view.export_dialog=NS(window=NS(visible=True))
    view._key(268,0,blocked!='release')
    assert (view.model.view,view.model.history,view.model.history_index)==before
