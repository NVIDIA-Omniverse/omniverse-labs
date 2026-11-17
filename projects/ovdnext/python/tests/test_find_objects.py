# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original Find semantics over real immutable native presentations."""
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path
from types import SimpleNamespace

import omni.ui as ui
import pytest
from _pvd_native import Session

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.application import Application
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets.object_browser import (
    FindObjectsDialog, FoundObject, _FindModel, find_objects, find_sort_key, signed_decimal,
)
from test_application_presentation import app, inspector_ui, open_capture

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def adapter(tmp_path):
    source=Session(str(ROOT/"test/data/test_scene.ovd"),str(tmp_path))
    result=PvdDataAdapter(source.presentation(frame=1))
    result.fixture_roles=test_scene_roles(source)
    return result


@pytest.mark.parametrize("text,expected",[(" +0001 ",1),("-2147483648",-2147483648),
    ("2147483647",2147483647),("2147483648",None),("-2147483649",None),
    ("1.0",None),("0x10",None),("1_0",None),("١",None),("１２",None),
    ("",None),("1 0",None),("0"*5000+"1",1),("1"*5000,None)])
def test_id_parser_matches_qt_signed_decimal_domain(text,expected):
    assert signed_decimal(text)==expected


def test_name_is_recorded_case_sensitive_literal_with_ovd_lowercase_support(adapter):
    ids=adapter.fixture_roles
    assert find_objects(adapter,"")==[FoundObject(ids.actor,"sphere","PxRigidDynamic",True)]
    assert find_objects(adapter,"sphere")==find_objects(adapter,"pher")
    for expression in ("Sphere"," sphere","sphere ","PxRigidDynamic","ph er"):
        assert not find_objects(adapter,expression)
    assert find_objects(adapter,"Rigid",2)==[FoundObject(ids.actor,"PxRigidDynamic","PxRigidDynamic",True)]
    assert not find_objects(adapter,"rigid",2)


def test_id_lookup_is_exact_and_query_records_stay_in_sidebar(tmp_path):
    source=Session(str(ROOT/"test/data/cloth_physx34.pxd2"),str(tmp_path))
    adapter=PvdDataAdapter(source.presentation(frame=1))
    assert [row.identity for row in find_objects(adapter,"1",1)]==[1]
    assert [row.identity for row in find_objects(adapter," +011 ",1)]==[11]
    assert adapter.query_records()
    assert not find_objects(adapter,str(adapter.query_records()[0].query_id),1)


def test_type_and_id_keep_enumerated_dead_objects(tmp_path):
    source=Session(str(ROOT/"test/data/test_scene.ovd"),str(tmp_path))
    adapter=PvdDataAdapter(source.presentation(frame=3))
    ids=test_scene_roles(source)
    dead=FoundObject(ids.actor,"PxRigidDynamic","PxRigidDynamic",False)
    assert find_objects(adapter,str(ids.actor),1)==[dead]
    assert find_objects(adapter,"PxRigidDynamic",2)==[dead]
    assert {row.identity for row in find_objects(adapter,"",2)}==set(adapter.instances)
    # The owned view cannot invent names for unavailable objects.
    assert find_objects(adapter,"sphere")==[]


def test_numeric_sort_and_qt_default_descending_id_order():
    rows=[FoundObject(i,label,"Class",True) for i,label in
          ((2,"10"),(11,"2"),(1,"a"),(3,"-3"),(4,"A"),(5,"9223372036854775808"))]
    assert [r.label for r in sorted(rows,key=lambda row:find_sort_key(row,1))]==[
        "-3","2","10","9223372036854775808","A","a"]
    model=_FindModel()
    model.set_rows(rows)
    assert [item.result.identity for item in model.items]==[11,5,4,3,2,1]
    model.sort(0)
    assert [item.result.identity for item in model.items]==[1,2,3,4,5,11]


def dialog_for(adapter):
    """Real value/result models and callbacks; no desktop needed for semantics."""
    model=ObjectTreeModel()
    model.populate(adapter)
    model.set_filter("unrelated sidebar filter")
    selected=[]
    dialog=FindObjectsDialog.__new__(FindObjectsDialog)
    dialog.browser=SimpleNamespace(model=model,activate_found=lambda identity:selected.append(identity))
    dialog.model=_FindModel()
    dialog.expression=ui.SimpleStringModel("")
    dialog.mode=0
    dialog.focus="expression"
    dialog.capture_key=None
    dialog._selecting=False
    dialog.window=SimpleNamespace(visible=True)
    dialog.field=SimpleNamespace(focus_keyboard=lambda *_:None)
    dialog.status=SimpleNamespace(text="")
    dialog.select_button=SimpleNamespace(enabled=False)
    dialog.tree=SimpleNamespace(selection=[])
    dialog.tree.clear_selection=lambda:setattr(dialog.tree,"selection",[])
    return dialog,selected


def test_return_search_select_escape_and_mode_change_do_not_mutate_sidebar(adapter):
    ids=adapter.fixture_roles
    dialog,selected=dialog_for(adapter)
    dialog.expression.set_value("sphere")
    dialog.key(257,0,True)
    assert len(dialog.model.items)==1 and dialog.select_button.enabled
    assert dialog.focus=="results" and selected==[]
    assert dialog.browser.model.filter=="unrelated sidebar filter"
    dialog.key(257,0,True)
    assert selected==[ids.actor] and not dialog.window.visible
    dialog.window.visible=True
    dialog.key(256,0,True)
    assert selected==[ids.actor] and not dialog.window.visible
    dialog._mode_changed(SimpleNamespace(get_item_value_model=lambda:ui.SimpleIntModel(2)),None)
    assert not dialog.model.items and not dialog.select_button.enabled
    assert dialog.expression.as_string=="sphere"


def test_result_single_selection_capture_change_and_unknown_activation_guard(adapter):
    ids=adapter.fixture_roles
    dialog,selected=dialog_for(adapter)
    dialog.mode=2
    dialog.search()
    assert not dialog.select_button.enabled
    assert not dialog.activate()

    items=dialog.model.items
    dialog.tree.selection=items[:2]
    dialog._selection_changed(dialog.tree.selection)
    assert dialog.tree.selection==[items[1]]
    dialog.capture_key=-123
    assert not dialog.activate() and dialog.window.visible and not selected
    assert "capture changed" in dialog.status.text
    dialog.browser.model._adapter=None
    assert not dialog.activate()


def test_results_own_arrow_navigation_and_tab_returns_to_expression(adapter):
    ids=adapter.fixture_roles
    dialog,_=dialog_for(adapter)
    dialog.mode=2
    dialog.search()
    assert dialog.window.flags & (1<<16)
    dialog.key(264,0,True)
    assert dialog.tree.selection==[dialog.model.items[0]]
    dialog.key(264,0,True)
    assert dialog.tree.selection==[dialog.model.items[1]]
    dialog.key(265,0,True)
    assert dialog.tree.selection==[dialog.model.items[0]]
    dialog.key(258,0,True)
    assert dialog.focus=="expression" and not dialog.window.flags & (1<<16)


def test_explicit_dead_find_inspection_survives_publication_without_selecting_geometry(app,tmp_path,inspector_ui):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids=test_scene_roles(app.session)
    assert app.seek(3,snap=False)
    framed=[]
    app.frame_selected=lambda:framed.append(True)
    assert app._find_object(ids.actor)
    assert app.interaction.sticky==(ids.actor,) and not app.renderer.selected and not framed
    assert app._refresh()
    assert app.selection.get_snapshot().paths()==[]
    assert app._inspection_ids()==[ids.actor]
    assert app.presentation.inspected_ids==[] and app.presentation.unavailable_ids==[ids.actor]
    assert not app.presentation_dirty
    app._build_properties()
    assert f"PxRigidDynamic · #{ids.actor}" in inspector_ui[1]
    assert "Not alive at this frame" in inspector_ui[1]
    assert app._refresh() and not app.presentation_dirty
    assert app._inspection_ids()==[ids.actor]
    # Unknown IDs cannot create an inspection request loop.
    assert app._find_object(999999) is False
    app.selection.publish([f"/objects/{ids.metadata}"],source="objects")
    assert app.find_inspection is None and app._inspection_ids()==[ids.metadata]
    assert app._find_object(ids.actor)
    assert app._display(*open_capture(tmp_path,"cloth_physx34.pxd2"))
    assert app.find_inspection is None and app._inspection_ids()==[]


def test_live_find_does_not_frame_and_history_change_clears_explicit_dead_inspection(app,tmp_path):
    assert app._display(*open_capture(tmp_path,"test_scene.ovd"))
    ids=test_scene_roles(app.session)
    framed=[]
    app.frame_selected=lambda:framed.append(True)
    assert app._find_object(ids.actor)
    assert framed==[] and app.find_inspection is None
    assert app.seek(3,snap=False)
    assert app._find_object(ids.actor)
    app._interaction_action(app.interaction.select,[ids.metadata])
    assert app._inspection_ids()==[ids.metadata] and app.find_inspection is None


def test_ctrl_f_owns_shortcut_even_when_text_input_wants_keyboard(app):
    shown=[]
    app.object_browser.show_find=lambda:shown.append(True)
    app.window=SimpleNamespace(focused=True)
    app.viewport_keyboard_focus=True
    app.navigation=None
    app._key_pressed(70,2|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert shown==[True] and not app.viewport_keyboard_focus
    app._key_pressed(70,2,False)
    app.window.focused=False
    app._key_pressed(70,2,True)
    assert shown==[True]
