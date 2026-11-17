# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent original Find over real native-owned profiler results."""
from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace as NS

import omni.ui as ui
import pytest
import _pvd_native as native

from ovd_ui.profiling import EventKey, ProfileModel
from ovd_ui.profile_find import (FoundProfileEvent, activate_profile_event, find_profile_events,
                                profile_find_sort_key, unsigned_decimal)
from ovd_ui.widgets.profile_find import FindProfileDialog, ProfileFindModel
from ovd_ui.widgets.profile_view import ProfileView, ProfileTree
from test_application_presentation import app

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def owned(tmp_path):
    source=native.Session(str(ROOT/'test/data/cloth_physx34.pxd2'),str(tmp_path))
    scene=source.scenes()[0].id
    assert source.seek(scene,2)
    return source,native.profile_snapshot(source,scene)


@pytest.mark.parametrize('text,expected',[
    ('0',0),(' +00017 ',17),('18446744073709551615',2**64-1),
    ('18446744073709551616',None),('-0',None),('-1',None),('++1',None),
    ('1.0',None),('1e2',None),('0x10',None),('1_0',None),('١',None),('１２',None),
    ('',None),('1 0',None),('0'*5000+'1',1),('1'*5000,None),
    ('\u00a0+42\u00a0',42),('\u200342\u2028',42),('\u008542\u0085',42),
    ('\x1c42\x1c',None),('\x1d42\x1d',None),('\x1e42\x1e',None),('\x1f42\x1f',None),
])
def test_unsigned_64_decimal_conversion(text,expected):
    assert unsigned_decimal(text)==expected


def test_actual_qt_conversion_oracle_corpus():
    corpus=json.loads((Path(__file__).parent/'fixtures/qt_profile_find_unsigned.json').read_text(encoding='utf-8'))
    assert corpus['qt']=='5.15.3'
    for row in corpus['rows']:
        assert unsigned_decimal(row['text'])==(int(row['unsigned_value']) if row['unsigned_ok'] else None)


@pytest.mark.parametrize('filename,frame',[
    ('cloth_physx34.pxd2',2),('ue4_shooter.pxd2',2),('apex_destruction.pxd2',3)])
def test_all_criteria_exactly_match_owned_native_summaries_after_seek_and_close(tmp_path,filename,frame):
    source=native.Session(str(ROOT/'test/data'/filename),str(tmp_path))
    scene=source.scenes()[0].id
    assert source.seek(scene,frame)
    data=native.profile_snapshot(source,scene)
    assert data.supported and data.summaries
    row=next(row for row in data.summaries if row.name and any(c.isupper() for c in row.name))
    all_keys={EventKey.of(r) for r in data.summaries}
    assert {r.key for r in find_profile_events(data,'')}==all_keys
    assert {r.key for r in find_profile_events(data,row.name)}=={
        EventKey.of(r) for r in data.summaries if row.name in r.name}
    assert EventKey.of(row) not in {r.key for r in find_profile_events(data,row.name.swapcase())}
    for criterion,field in enumerate(('context_id','thread_id','event_id'),1):
        value=getattr(row,field)
        expected={EventKey.of(r) for r in data.summaries if getattr(r,field)==value}
        assert {r.key for r in find_profile_events(data,f' +000{value} ',criterion)}==expected
    # The source moves and dies; search only retains the independently owned result.
    source.seek(scene,1)
    del source
    assert {r.key for r in find_profile_events(data,'')}==all_keys
    assert data.current_frame==frame


def test_widths_are_not_truncated_and_zone_collisions_keep_distinct_identities():
    rows=[NS(name='Step',zone_id=z,event_id=17,context_id=2**64-1,thread_id=2**32-1) for z in (1,2)]
    data=NS(supported=True,summaries=rows)
    assert len(find_profile_events(data,str(2**64-1),1))==2
    assert find_profile_events(data,str(2**32+17),3)==()
    found=find_profile_events(data,'Step')
    assert found[0].key!=found[1].key
    assert found[0].cells==('Step',str(2**64-1),str(2**32-1),'17')
    cross=FoundProfileEvent(EventKey(1,17,1,native.CROSS_THREAD_ID),'cross')
    assert cross.cells[2]=='Cross thread'
    assert not find_profile_events(None,'')
    assert not find_profile_events(NS(supported=False),'')
    with pytest.raises(ValueError):find_profile_events(data,'',4)


def test_find_sort_uses_original_signed64_numeric_then_text_columns():
    rows=[FoundProfileEvent(EventKey(i,1,1,1),name) for i,name in enumerate(
        ('10','2','A','a','-3','9223372036854775808'))]
    assert [r.name for r in sorted(rows,key=profile_find_sort_key)]==[
        '-3','2','10','9223372036854775808','A','a']
    model=ProfileFindModel();model.set_rows(rows)
    assert model.items[0].result.name=='a'
    model.sort(0)
    assert model.items[0].result.name=='-3'
    contexts=[FoundProfileEvent(EventKey(1,1,c,1),'same') for c in (2,11,2**64-1)]
    assert [r.key.context_id for r in sorted(contexts,key=lambda r:profile_find_sort_key(r,1))]==[2,11,2**64-1]


def test_activation_is_one_identity_both_modes_hidden_preserved_and_stale_result_rejected(owned):
    source,data=owned
    model=ProfileModel();model.load(data)
    row=next(row for row in data.summaries if row.total_ms>0)
    key=EventKey.of(row)
    model.hide([key]);hidden=set(model.hidden)
    assert activate_profile_event(model,data,key)
    assert model.selected=={key} and model.hidden==hidden
    assert model.view_range[0]>=0 and model.view_range[1]<=data.duration_ms
    assert source.scenes()[0].current_frame==2
    before=model.selected,model.view_range,set(model.hidden)
    assert not activate_profile_event(model,NS(supported=True),key)
    assert not activate_profile_event(model,data,EventKey(-1,1,1,1))
    assert (model.selected,model.view_range,model.hidden)==before
    model.load(native.profile_snapshot(source,source.scenes()[0].id))
    assert not activate_profile_event(model,data,key)  # same frame, different owned publication


def view_for(data):
    view=ProfileView.__new__(ProfileView)
    view.model=ProfileModel();view.model.load(data)
    view.finished=view.find_ready=True
    view.find_dialog=None;view.found_key=None
    view.capture_key=1;view.source_label='cloth_physx34.pxd2'
    view.window=NS(visible=True,focused=True,focus=lambda:None)
    view.mode=0;view.query='unrelated inline filter';view.search_by='name'
    view.search_editing=False;view.page_size=1;view.lane_page=0
    view.scroll=NS(scroll_y=0,computed_width=800,computed_height=300)
    view.bands=[]
    view.tree_model=ProfileTree();view.tree_guard=False;view.restore_tree=0
    view.restore_expansion=None;view.pending_sort=None
    view.sort_column=0;view.sort_descending=False
    view.table=NS(update=lambda:None,set_sort=lambda *_:None,displayed=(146,52,74,76),
                  headers=('Event / context','Count','Total ms','% frame'))
    view.tree=NS(selection=[],set_expanded=lambda *_:None,is_expanded=lambda *_:False)
    view.selection_label=NS(text='')
    view.search_field=NS(model=ui.SimpleStringModel(view.query))
    view.search_field.model.add_value_changed_fn(view._search)
    view.find_button=NS(enabled=True)
    view.progress=NS(text='');view.dirty=False
    return view


def dialog_for(view):
    dialog=FindProfileDialog.__new__(FindProfileDialog)
    dialog.view=view;dialog.model=ProfileFindModel();dialog.expression=ui.SimpleStringModel('')
    dialog.mode=0;dialog.focus='expression';dialog.snapshot=None;dialog._selecting=False
    dialog.window=NS(visible=True,focused=True)
    dialog.field=NS(focus_keyboard=lambda *_:None)
    dialog.status=NS(text='');dialog.select_button=NS(enabled=False)
    dialog.tree=NS(selection=[])
    dialog.tree.clear_selection=lambda:setattr(dialog.tree,'selection',[])
    return dialog


def test_dialog_return_one_result_activation_closes_clears_filter_and_preserves_native_frame(owned):
    source,data=owned;view=view_for(data);dialog=dialog_for(view)
    row=next(row for row in data.summaries if sum(r.name==row.name for r in data.summaries)==1)
    dialog.expression.set_value(row.name)
    dialog.key(257,0,True)
    assert view.query=='unrelated inline filter' and not view.model.selected
    assert len(dialog.model.items)==1 and dialog.select_button.enabled
    assert dialog.focus=='results'
    dialog.key(257,0,True)
    assert view.model.selected=={EventKey.of(row)} and not dialog.window.visible
    assert view.query==view.search_field.model.as_string==''
    assert source.scenes()[0].current_frame==2
    assert EventKey.of(row) in view.tree_model.leaves


def test_result_arrow_tab_sort_and_cancel_do_not_change_existing_selection_or_filter(owned):
    _,data=owned;view=view_for(data);dialog=dialog_for(view)
    first=EventKey.of(data.summaries[0]);view.model.select([first])
    before=(view.query,view.model.selected,view.model.view_range)
    dialog.key(257,0,True)
    assert len(dialog.model.items)>1 and not dialog.select_button.enabled
    assert dialog.window.flags&(1<<16)
    dialog.key(264,0,True);dialog.key(264,0,True);dialog.key(265,0,True)
    assert dialog.tree.selection==[dialog.model.items[0]]
    chosen=dialog.tree.selection[0]
    dialog.sort(3)
    assert dialog.tree.selection==[chosen]
    dialog._selection_changed(dialog.model.items[:2])
    assert dialog.tree.selection==[dialog.model.items[1]]
    dialog.key(258,0,True)
    assert dialog.focus=='expression' and not dialog.window.flags&(1<<16)
    dialog.key(256,0,True)
    assert not dialog.window.visible and (view.query,view.model.selected,view.model.view_range)==before


def test_hidden_result_selects_statistics_without_unhiding_and_lane_mode_keeps_same_key(owned):
    _,data=owned;view=view_for(data)
    key=EventKey.of(data.summaries[0]);view.model.hide([key])
    assert view.activate_found(data,key)
    assert key in view.model.hidden and key in view.tree_model.leaves
    assert 'Hidden event' in view.selection_label.text
    view._mode(NS(get_item_value_model=lambda:ui.SimpleIntModel(1)),None)
    assert view.model.selected=={key} and key in view.model.hidden
    view.visibility(True)
    view._mode(NS(get_item_value_model=lambda:ui.SimpleIntModel(0)),None)
    assert view.model.selected=={key} and key not in view.model.hidden


def test_named_context_cells_keep_unsigned_queries_and_snapshot_identity(owned):
    source,data=owned
    assert data.context_names
    for row in find_profile_events(data,''):
        assert row.cells[1]==data.context_names.get(row.key.context_id,str(row.key.context_id))
    records=[NS(name='Step',zone_id=1,event_id=17,context_id=c,thread_id=7) for c in (2**63,2**63+1)]
    data=NS(supported=True,summaries=records,context_names={c:'Same caption' for c in (2**63,2**63+1)})
    found=find_profile_events(data,'Step')
    assert found[0].cells==found[1].cells and found[0].key!=found[1].key
    assert [r.key.context_id for r in find_profile_events(data,str(2**63+1),1)]==[2**63+1]


@pytest.mark.parametrize('mode',[0,1])
def test_hide_filter_select_and_show_keep_statistics_but_remove_intervals(owned,mode):
    _,data=owned;view=view_for(data);view.mode=mode
    key=EventKey.of(next(row for row in data.summaries if row.total_ms>0))
    assert view.activate_found(data,key)
    view.visibility(False)
    assert view.tree_model.leaves[key].parent is view.tree_model.hidden_root
    assert view.model.selected=={key}
    assert not any(EventKey.of(scope)==key for scope in view.model.scopes())
    assert any(EventKey.of(row)==key for row in view.rows())
    view.search_by='event_id'
    view.search_field.model.set_value(str(key.event_id))
    hidden=view.tree_model.hidden_root
    assert hidden and key in view.tree_model.leaves
    view.restore_tree=0
    view._select_tree([hidden])
    assert view.model.selected=={key}
    view.visibility(True)
    assert view.tree_model.hidden_root is None and key not in view.model.hidden
    assert view.model.selected=={key} and not view.tree_model.leaves[key].hidden
    assert any(EventKey.of(scope)==key for scope in view.model.scopes())


def test_refresh_invalidates_old_results_and_disables_find_until_success(owned):
    _,data=owned;view=view_for(data);dialog=dialog_for(view);view.find_dialog=dialog
    dialog.search();item=dialog.model.items[0]
    cancelled=[];view.job=NS(cancel=lambda:cancelled.append(True))
    view.cancel=NS(enabled=False,set_clicked_fn=lambda fn:None)
    pending=Future();view.set_job(NS(future=pending,cancel=lambda:None))
    assert cancelled==[True] and not dialog.window.visible and not dialog.model.items
    assert not view.can_find and not view.find_button.enabled
    assert not dialog.activate(item) and not view.model.selected
    assert not view.activate_found(data,item.result.key)
    # A failed refresh keeps the old display readable but does not re-enable stale Find.
    pending.set_exception(RuntimeError('cancelled'))
    view.canvas=NS(height=0,width=0);view.chart=NS(rebuild=lambda:None);view.chart_width=0
    view.poll()
    assert not view.can_find and not view.find_button.enabled
    assert view.model.snapshot is data and 'cancelled' in view.progress.text


def test_criterion_change_invalid_number_closed_parent_and_foreign_result_guard(owned):
    _,data=owned;view=view_for(data);dialog=dialog_for(view)
    dialog.search();item=dialog.model.items[0]
    dialog._mode_changed(NS(get_item_value_model=lambda:ui.SimpleIntModel(2)),None)
    assert not dialog.model.items and dialog.snapshot is None and not dialog.activate(item)
    dialog.expression.set_value('-1');dialog.search()
    assert not dialog.model.items and 'decimal ID' in dialog.status.text
    dialog.mode=0;dialog.expression.set_value('');dialog.search()
    item=dialog.model.items[0];view.window.visible=False
    assert not dialog.activate(item) and not view.model.selected


def test_profile_alt_f_exact_chord_focused_window_and_inline_edit_scope(owned):
    _,data=owned;view=view_for(data);calls=[]
    view.show_find=lambda:calls.append(True)
    view.search_editing=True
    for mods in (0,2,5,6,12):view._key(70,mods,True)
    view._key(70,4,False)
    view.window.focused=False;view._key(70,4,True)
    assert calls==[]
    view.window.focused=True;view._key(70,4|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert calls==[True]


def test_workspace_routes_only_loaded_visible_current_capture_and_exact_alt_f(app,owned):
    source,data=owned;first=view_for(data);second=view_for(data)
    app.session=source
    first.capture_key=second.capture_key=source.capture_key
    first.window.focused=False;second.window.focused=True
    calls=[];first.show_find=lambda:calls.append('first') or True
    second.show_find=lambda:calls.append('second') or True
    app.analysis_views=[first,object(),second]
    app.window=NS(focused=True);app.navigation=None;app.viewport_keyboard_focus=False
    app._key_pressed(70,4|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert calls==['second']
    second.window.visible=False
    assert app.show_find_profile() and calls[-1]=='first'
    first.capture_key=-1
    assert not app.show_find_profile() and len(calls)==2
    assert 'Open Profiler' in app.status_label.text
    first.capture_key=source.capture_key;first.find_ready=False
    assert not app.show_find_profile()
    first.find_ready=True;app.session=None
    assert not app.show_find_profile()


def test_workspace_menu_snapshot_and_capture_guards_do_not_retarget_new_results(app,owned):
    source,data=owned;view=view_for(data)
    view.capture_key=source.capture_key;app.session=source;app.analysis_views=[view]
    app.window=NS(focused=True);app.navigation=None
    calls=[];view.show_find=lambda:calls.append(True)
    app.show_find_profile(view,data)
    assert calls==[True]
    view.model.load(native.profile_snapshot(source,source.scenes()[0].id))
    assert not app.show_find_profile(view,data)
    assert 'result changed' in app.status_label.text
    view.model.load(data);app.session=None
    assert not app.show_find_profile(view,data) and calls==[True]


def test_old_worker_completion_cannot_reenable_find_for_replacement_request(owned):
    _,data=owned;view=view_for(data);old=Future();new=Future()
    view.job=NS(future=old,cancel=lambda:None)
    view.cancel=NS(enabled=False,set_clicked_fn=lambda fn:None)
    view.canvas=NS(height=0,width=0);view.chart=NS(rebuild=lambda:None);view.chart_width=0
    view.set_job(NS(future=new,cancel=lambda:None))
    old.set_result(data);view.poll()
    assert not view.can_find and not view.find_button.enabled
    new.set_result(data);view.poll()
    assert view.can_find and view.find_button.enabled
