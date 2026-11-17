# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native publication/lifecycle evidence; widget sinks are not GUI validation."""
from capture_semantics import test_scene_roles, role_key, dynamic_geometry_owner
from collections import Counter
from copy import copy
from types import SimpleNamespace

import pytest

from ovd_ui.frame_summary import TypeCount,native_object_counts,sorted_counts
from ovd_ui.widgets import properties
from ovd_ui.widgets.properties import DisplayedValue,PropertyInspector,_CopyEntry
from test_application_presentation import app,open_capture
from test_application_object_actions import choose,GROUP,ALIAS,PresentationForbiddenSession


def test_counts_use_alive_native_creation_ids_only_and_sort_numerically():
    values={i:SimpleNamespace(alive=True) for i in range(1,15)}
    values.update({0:SimpleNamespace(alive=True),-6:SimpleNamespace(alive=True),99:SimpleNamespace(alive=False)})
    adapter=SimpleNamespace(instances=values,class_name=lambda i:"A" if i<3 else "B" if i<13 else "C")
    rows=native_object_counts(adapter)
    assert rows==(TypeCount("A",2),TypeCount("B",10),TypeCount("C",2))
    assert [row.type_name for row in sorted_counts(rows,"count")]==["A","C","B"]
    assert [row.type_name for row in sorted_counts(rows,"count",True)]==["B","A","C"]
    assert [row.type_name for row in sorted_counts(rows,"type",True)]==["C","B","A"]
    assert native_object_counts(None)==()
    with pytest.raises(ValueError):sorted_counts(rows,"unsupported")


@pytest.mark.parametrize("keys",[("object:{actor}",),(GROUP,),(GROUP,ALIAS,"object:{actor}")])
def test_summary_clears_every_selection_store_without_native_or_camera_work(app,tmp_path,keys):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app,*keys)
    before=(app.presentation,app.frame,app.playback.frame,app.pending_presentation,
            app.renderer.snapshot.transforms.tobytes())
    app.session=PresentationForbiddenSession(app.session)
    app.navigation=SimpleNamespace(camera=object())
    camera=app.navigation.camera
    assert app.show_frame_summary()
    assert app.frame_summary
    assert not app.interaction.sticky and not app.interaction.selected
    assert not app.row_actions.selected and not app.tree.selection
    assert app.selection.get_snapshot().paths()==[] and app.selected_record is None
    assert not app.presentation_dirty
    assert before==(app.presentation,app.frame,app.playback.frame,app.pending_presentation,
                    app.renderer.snapshot.transforms.tobytes())
    assert app.navigation.camera is camera


def test_counts_survive_filter_grouping_visibility_and_owned_frame_advance(app,tmp_path):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    assert app.seek(1,snap=False)
    adapter=copy(app.adapter)
    before=native_object_counts(adapter)
    assert dict((row.type_name,row.count) for row in before)==Counter(
        adapter.class_name(i) for i,instance in adapter.instances.items() if i>0 and instance.alive)
    app.object_model.set_filter("no possible match")
    app.interaction.set_visible([ids.actor],False)
    app.object_model.grouping="class"
    app.object_model.populate(adapter)
    assert native_object_counts(adapter)==before
    assert app.show_frame_summary()
    assert app.seek(3,snap=False)
    assert app.frame_summary
    assert native_object_counts(app.adapter)!=before
    assert native_object_counts(adapter)==before  # old publication stays owned
    assert app.seek(1,snap=False)
    assert native_object_counts(app.adapter)==before


@pytest.mark.parametrize("source",["objects","property","viewport","find","interaction"])
def test_next_object_selection_returns_to_properties(app,tmp_path,source):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    assert app.show_frame_summary()
    app.selection.publish([f"/objects/{ids.actor}"],source=source)
    assert not app.frame_summary


def test_container_selection_also_returns_from_summary(app,tmp_path):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    assert app.show_frame_summary()
    choose(app,GROUP)
    assert not app.frame_summary


def test_new_capture_resets_summary_and_rejects_old_context(app,tmp_path):
    assert not app.show_frame_summary()
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    key=app.presentation.capture_key
    assert app.show_frame_summary(key)
    assert app._display(*open_capture(tmp_path/"second","test_scene.ovd"))
    assert not app.frame_summary
    assert not app.show_frame_summary(key)
    assert app._display(None,None)
    assert not app.frame_summary and not app.show_frame_summary()


@pytest.mark.parametrize("modifiers,dragged,button,expected",[(0,False,0,True),(1,False,0,False),
    (2,False,0,False),(4,False,0,False),(16,False,0,True),(0,True,0,False),(0,False,1,False)])
def test_only_plain_empty_viewport_pick_opens_summary(app,tmp_path,monkeypatch,modifiers,dragged,button,expected):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids = test_scene_roles(app.session)
    app.renderer.pick=lambda *_:None
    app.viewport=SimpleNamespace(screen_position_x=0,screen_position_y=0)
    app.window=SimpleNamespace(focused=True)
    app._show_viewport_menu=lambda *_:None
    monkeypatch.setattr("ovd_ui.application.asyncio.get_event_loop",lambda:SimpleNamespace(call_soon=lambda *_:None))
    app._mouse_pressed(10,10,button,modifiers)
    app.mouse_dragged,app.mouse_modifiers=dragged,modifiers
    app._mouse_released(10,10,button)
    assert app.frame_summary is expected


def test_context_action_is_capture_guarded_and_has_no_copy_on_whitespace(monkeypatch):
    widgets=[]
    class Widget:
        def __init__(self,text="",**kwargs):self.text,self.kwargs=text,kwargs;widgets.append(self)
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def show_at(self,*_):pass
        def destroy(self):pass
    monkeypatch.setattr(properties,"ui",SimpleNamespace(Menu=Widget,MenuItem=Widget,Separator=Widget))
    requested=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_summary=lambda key:requested.append(key) or True)
    inspector.capture_key=13
    inspector.summary_available=True
    inspector.background_pressed(1,2,1,0)
    assert [widget.text for widget in widgets]==["Property value","Frame summary"]
    action=widgets[-1].kwargs["triggered_fn"]
    assert action() and requested==[13]
    # Compound/flags children own only copy text, not plot/capture metadata.
    inspector.show_context_menu(_CopyEntry(DisplayedValue("1, 2, 3")),1,2)
    action=widgets[-1].kwargs["triggered_fn"]
    assert action() and requested==[13,13]
    inspector.capture_key=14
    assert not action() and requested==[13,13]
    inspector.capture_key=13
    inspector.destroy()
    assert not action()
