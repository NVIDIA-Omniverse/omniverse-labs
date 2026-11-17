# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchy shortcuts over real native matches, with lightweight widget sinks."""
from pathlib import Path
from types import SimpleNamespace as NS

import omni.ui as ui
import pytest
from _pvd_native import Session

from capture_semantics import test_scene_roles
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.application import Application
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets.object_browser import ObjectBrowser


ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def browser(tmp_path):
    source=Session(str(ROOT/'test/data/test_scene.ovd'),str(tmp_path))
    model=ObjectTreeModel()
    model.populate(PvdDataAdapter(source.presentation(frame=1)))
    view=ObjectBrowser.__new__(ObjectBrowser)
    view.model=model
    view.source=source
    view.ids=test_scene_roles(source)
    view._destroyed=False
    view._search_editing=False
    view._search_refocus=False
    view.owner=NS(focused=True)
    view.search_window_focused=lambda:view.owner.focused
    view.search_model=ui.SimpleStringModel('')
    view.search_editor_model=ui.SimpleStringModel('')
    view.search_editor_model.add_value_changed_fn(view._search_edited)
    view.search_model.add_value_changed_fn(view._search_changed)
    view.match_case=ui.SimpleBoolModel(False)
    view.whole_word=ui.SimpleBoolModel(False)
    view.fields_index=0
    view._last_search_signature=view._last_projection=None
    view._matches=[]
    view.search_hint=view.clear_search_host=NS(visible=False)
    view.summary=NS(text='')
    view.previous_button=NS(enabled=False)
    view.next_button=NS(enabled=False)
    view.focus_requests=[]
    view.search_field=NS(focus_keyboard=lambda:view.focus_requests.append(True))
    view.tree=NS(selection=[],set_expanded=lambda *_:None)
    return view


def selected(browser):
    return [item.instance_id for item in browser.tree.selection]


def enter(browser,key=257,mods=0):
    assert browser.search_key(key,mods,True)
    # Native AbstractField ends editing after accepting Enter, then applies
    # focus_keyboard on the following draw. The real GUI verifies this order.
    browser._end_search_edit(browser.search_editor_model)
    browser._begin_search_edit(browser.search_editor_model)
    assert not browser.search_key(key,mods,False)


@pytest.mark.parametrize('key',[257,335,525])
def test_enter_and_shift_enter_wrap_native_matches_without_changing_expression_or_frame(browser,key):
    browser.search_editor_model.set_value('sphere')
    assert selected(browser)==[browser.ids.actor]
    browser._begin_search_edit(browser.search_editor_model)
    frame=browser.source.scenes()[0].current_frame
    enter(browser,key)
    assert selected(browser)==[browser.ids.geometry]
    enter(browser,key)
    assert selected(browser)==[browser.ids.actor]
    enter(browser,key,1)
    assert selected(browser)==[browser.ids.geometry]
    enter(browser,key,1)
    assert selected(browser)==[browser.ids.actor]
    assert browser.model.filter==browser.search_model.as_string==browser.search_editor_model.as_string=='sphere'
    assert browser.source.scenes()[0].current_frame==frame==1
    assert len(browser.focus_requests)==4


@pytest.mark.parametrize('mods',[2,3,4,5,6,7,8,9,10,11,12,13,14,15])
@pytest.mark.parametrize('key',[257,335])
def test_extra_modifiers_do_not_navigate_or_restore_focus(browser,key,mods):
    browser.search_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    assert not browser.search_key(key,mods|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert selected(browser)==[browser.ids.actor]
    assert not browser._search_refocus and not browser.focus_requests


@pytest.mark.parametrize('blocked',['not_editing','other_window','destroyed','release','other_key'])
def test_search_ownership_and_lifetime_guards(browser,blocked):
    browser.search_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    if blocked=='not_editing':browser._end_search_edit(browser.search_editor_model)
    if blocked=='other_window':browser.owner.focused=False
    if blocked=='destroyed':browser._destroyed=True
    assert not browser.search_key(65 if blocked=='other_key' else 257,0,blocked!='release')
    assert selected(browser)==[browser.ids.actor]
    assert not browser._search_refocus and not browser.focus_requests


def test_keyboard_capture_flag_is_not_a_modifier_and_pointer_location_is_not_required(browser):
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    assert browser.search_key(335,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD|1,True)
    assert selected(browser)==[browser.ids.geometry]


def test_shared_filter_updates_sync_text_without_granting_sidebar_ownership(browser):
    # The existing Filter options view uses search_model; its edit state is
    # intentionally not used to grant keyboard ownership to the sidebar.
    browser.search_model.set_value('sphere')
    assert browser.search_editor_model.as_string=='sphere'
    assert not browser._search_editing
    assert not browser.search_key(257,0,True)
    browser._begin_search_edit(browser.search_editor_model)
    enter(browser)
    assert selected(browser)==[browser.ids.geometry]
    browser.search_model.set_value('PxRigidDynamic')
    assert browser.search_editor_model.as_string=='PxRigidDynamic'
    assert selected(browser)==[browser.ids.actor]
    enter(browser)
    assert selected(browser)==[browser.ids.actor]


def test_zero_matches_and_clear_do_not_navigate_using_old_results(browser):
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    enter(browser)
    assert selected(browser)==[browser.ids.geometry]
    browser.search_editor_model.set_value('no recorded matches')
    enter(browser)
    # This lightweight sink does not emulate TreeView's filter-driven selection
    # removal. The shortcut itself must leave the sink unchanged on zero matches.
    assert selected(browser)==[browser.ids.geometry] and not browser._matches
    browser.search_model.set_value('')
    enter(browser,mods=1)
    assert selected(browser)==[browser.ids.geometry] and not browser._matches


def test_refocus_does_not_steal_another_window_or_revive_a_destroyed_browser(browser):
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    assert browser.search_key(257,0,True)
    browser.owner.focused=False
    browser._end_search_edit(browser.search_editor_model)
    assert not browser.focus_requests and not browser._search_refocus
    browser.owner.focused=True
    browser._begin_search_edit(browser.search_editor_model)
    assert browser.search_key(257,0,True)
    browser._destroyed=True
    browser._end_search_edit(browser.search_editor_model)
    assert not browser.focus_requests


def test_changed_native_projection_uses_current_matches_instead_of_stale_selection(browser):
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    enter(browser)
    assert selected(browser)==[browser.ids.geometry]
    # Native frame 3 has ended the sphere actor's lifetime and changed names.
    browser.model.populate(PvdDataAdapter(browser.source.presentation(frame=3)))
    browser.refresh()
    current=browser.model.find_matches()
    assert all(item.kind=='object' for item in current)
    enter(browser)
    assert not current or browser.tree.selection[-1] in current
    assert browser.source.scenes()[0].current_frame==3


def test_application_delegates_before_text_guard_and_does_not_require_hover(browser):
    application=Application.__new__(Application)
    application.window=browser.owner
    application.object_browser=browser
    application.viewport_keyboard_focus=application.object_keyboard_focus=False
    application.navigation=None
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    application._key_pressed(335,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert selected(browser)==[browser.ids.geometry]
    assert browser._search_refocus


def test_application_does_not_route_background_window_enter(browser):
    application=Application.__new__(Application)
    application.window=browser.owner
    application.object_browser=browser
    application.viewport_keyboard_focus=application.object_keyboard_focus=False
    application.navigation=None
    browser.search_editor_model.set_value('sphere')
    browser._begin_search_edit(browser.search_editor_model)
    browser.owner.focused=False
    application._key_pressed(257,0,True)
    assert selected(browser)==[browser.ids.actor] and not browser._search_refocus
