# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array policies exercised against native owned properties, not a UI-sized mock."""
import gc
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import pytest
import _pvd_native
from _pvd_native import Session, SnapshotDataUnavailable

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.array_settings import ArrayDisplaySettings, MAX_ARRAY_WINDOW_BYTES
from ovd_ui.widgets.properties import PropertyInspector
from test_property_context import display, prepare, ObservedView
from native_fixtures import executable

ROOT=Path(__file__).resolve().parents[2]


def inspector(prefs=None,requests=None):
    return PropertyInspector(lambda *_:None,lambda *_:None,prefs or {},
        on_request=lambda:requests.append(True) if requests is not None else None)


def expand(display):
    section=next(widget for widget in reversed(display.widgets) if widget.kind=='CollapsableFrame')
    section.collapsed(False)
    frame=next(widget for widget in reversed(display.widgets) if widget.kind=='Frame')
    frame.rebuild()
    return section,frame


@pytest.mark.parametrize('count_limit,expanded',[(0,False),(1051,False),(1052,True)])
def test_actual_cloth_array_parent_obeys_inclusive_expansion_limit(tmp_path,display,count_limit,expanded):
    source,adapter=prepare(tmp_path,'cloth_physx34.pxd2',18,1)
    prop=adapter.property(18,'Restvalues')
    assert adapter.session.property_size(18,prop.id)==1052*4
    observed=ObservedView(adapter.session);adapter.session=observed
    view=inspector({'max_array_to_load':count_limit})
    view._property(adapter,18,prop)
    assert bool([w for w in display.widgets if w.kind=='Frame'])==expanded
    assert not observed.reads and view.read_requests(adapter,[18])==[]
    if expanded:
        expand(display)
        assert observed.reads==[((18,prop.id),{'offset':0,'limit':40},40)]
        assert view.read_requests(adapter,[18])==[(18,prop.id,0,40)]
    else:
        assert display.find('1,052 elements')
        assert view.array_positions=={} and view.array_expanded==set()


def test_actual_native_class_array_uses_exact_stride_for_limit_and_preserves_raw_preview(tmp_path,display):
    source,adapter=prepare(tmp_path,'cloth_physx34.pxd2',11,1)
    prop=adapter.property(11,'Samples')
    assert prop.type=='PxHeightFieldSample'
    assert inspector()._array_stride(adapter,prop)==4
    assert adapter.session.property_size(11,prop.id)==4096
    view=inspector({'array_group_size':9999,'max_array_to_load':1024})
    observed=ObservedView(adapter.session);adapter.session=observed
    view._property(adapter,11,prop)
    assert observed.reads==[((11,prop.id),{'limit':256},256)]
    assert any('[32 of 4,096 bytes]' in w.text for w in display.widgets)
    assert view.read_requests(adapter,[11])==[]
    view.set_array_settings(ArrayDisplaySettings(9999,1023))
    view._property(adapter,11,prop)
    assert display.find('1,024 elements')
    assert len(observed.reads)==1


def test_ovd_reference_count_comes_from_owned_references_not_zero_raw_size(tmp_path,display):
    source,adapter=prepare(tmp_path,'multi_geom.ovd','PxPhysics',2)
    identity=adapter.fixture_id
    prop=adapter.property(identity,'shapes')
    assert adapter.session.property_size(identity,prop.id)==0
    refs=adapter.session.references(identity,prop.id);assert len(refs)==11
    view=inspector({'max_array_to_load':10})
    view._property(adapter,identity,prop)
    assert display.find('11 elements') and not view.array_positions
    assert view.set_array_settings(ArrayDisplaySettings(10,11))
    view._property(adapter,identity,prop);expand(display)
    assert len([w for w in display.widgets if w.kind=='Label' and w.text.startswith('[')])==10
    assert view.read_requests(adapter,[identity])==[]


def large_capture(tmp_path):
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    path=tmp_path/'large.ovd'
    subprocess.run([str(binary),str(path)],check=True,capture_output=True)
    payload=path.read_bytes()
    # Replace one complete SET_ATTRIBUTE payload in the actual writer fixture,
    # checking the wire header before changing its length. No parser shortcut.
    old=struct.pack('<Iff',8,4,5)
    assert payload.count(old)==1
    offset=payload.index(old)
    command,context,instance,arity,attribute=struct.unpack('<BQQBI',payload[offset-22:offset])
    assert (command,context,instance,arity)==(6,1,110,1)
    values=struct.pack('<20000f',*range(20000))
    path.write_bytes(payload[:offset]+struct.pack('<I',len(values))+values+payload[offset+len(old):])
    source=Session(str(path),str(tmp_path))
    initial=source.presentation(frame=3)
    cls=next(c.id for c in initial.classes() if c.name=='AnalysisSample')
    identity=next(i.id for i in initial.instances() if i.class_id==cls)
    return source,identity


def test_large_group_requests_one_owned_window_and_retains_it_after_source_close(tmp_path,display):
    source,identity=large_capture(tmp_path)
    adapter=PvdDataAdapter(source.presentation(frame=3,inspect_ids=[identity]))
    prop=adapter.property(identity,'variable')
    assert adapter.session.property_size(identity,prop.id)==80000
    with pytest.raises(SnapshotDataUnavailable):adapter.session.read(identity,prop.id,offset=72000,limit=256)
    requested=[]
    view=inspector({'array_group_size':9999,'max_array_to_load':99999},requested)
    view.build(adapter,[identity])
    view.array_positions[(identity,prop.id)]=18000
    view.array_expanded.add((identity,prop.id))
    requests=view.read_requests(adapter,[identity])
    assert requests==[(identity,prop.id,72000,256)]
    owned=source.presentation(frame=3,inspect_ids=[identity],read_requests=requests)
    adapter=PvdDataAdapter(owned)
    view.build(adapter,[identity])
    display.widgets.clear()
    observed=ObservedView(owned);adapter.session=observed
    view._property(adapter,identity,prop)
    section,frame=expand(display)
    assert observed.reads==[((identity,prop.id),{'offset':72000,'limit':256},256)]
    assert display.find('18000.000000') and display.find('18063.000000')
    # A paused/replaced reader cannot redirect the retained builder's owned bytes.
    source.presentation(frame=2,inspect_ids=[identity])
    del source;gc.collect()
    frame.rebuild()
    assert observed.reads[-1][2]==256
    assert view.read_requests(adapter,[identity])==requests


def test_new_capture_and_policy_invalidate_old_control_callbacks(tmp_path,display):
    source,adapter=prepare(tmp_path,'cloth_physx34.pxd2',18,1)
    requested=[]
    view=inspector({'max_array_to_load':99999},requested)
    view.build(adapter,[18]);old_sections=[w for w in display.widgets if w.kind=='CollapsableFrame']
    other,other_adapter=prepare(tmp_path/'other')
    view.build(other_adapter,[1]);before=(dict(view.array_positions),set(view.array_expanded),len(requested))
    for section in old_sections:section.collapsed(False)
    assert (view.array_positions,view.array_expanded,len(requested))==before
    assert view.read_requests(adapter,[18])==[]
    view.set_array_settings(ArrayDisplaySettings(20,0))
    for section in old_sections:section.collapsed(False)
    assert not view.array_positions and not view.array_expanded


def test_malformed_preferences_recover_at_construction_and_oversized_element_never_reads(display):
    view=inspector({'array_group_size':False,'max_array_to_load':-1})
    assert view.array_settings==ArrayDisplaySettings()
    def no_read(*_):raise AssertionError('oversized element read')
    view._array(None,1,SimpleNamespace(id=2),1,no_read,stride=MAX_ARRAY_WINDOW_BYTES+1)
    assert not [w for w in display.widgets if w.kind=='Frame']
    assert display.find('One recorded element exceeds the owned window budget.')


def test_suppressed_parent_does_not_hide_explicit_native_query_record(tmp_path,display):
    from ovd_ui.models import ObjectTreeModel
    source,adapter=prepare(tmp_path,'cloth_physx34.pxd2',6,1)
    view=inspector({'max_array_to_load':0})
    prop=adapter.property(6,'SceneQueries.Raycasts')
    assert view._array_stride(adapter,prop)==64
    view._property(adapter,6,prop)
    assert display.find('5 elements') and not view.array_positions
    model=ObjectTreeModel();model.populate(adapter)
    item=model.canonical_item(-6)
    owned=source.presentation(frame=1,record_requests=[(6,prop.id,0)])
    adapter=PvdDataAdapter(owned)
    del source;gc.collect()
    view.build(adapter,[],item)
    assert display.find('origin') and display.find('distance')
    assert view.read_requests(adapter,[])==[]


def test_unknown_layout_retains_native_raw_total_without_guessed_elements(tmp_path,display):
    source,adapter=prepare(tmp_path,'cloth_physx34.pxd2',6,1)
    prop=adapter.property(6,'SceneQueries.Raycasts')
    adapter.classes={}
    observed=ObservedView(adapter.session);adapter.session=observed
    view=inspector({'max_array_to_load':0})
    view._property(adapter,6,prop)
    assert observed.reads==[((6,prop.id),{'limit':256},256)]
    assert any('[32 of 320 bytes]' in w.text for w in display.widgets)
    assert not view.array_positions  # Unknown element count is not invented.


def test_large_group_does_not_raise_native_record_or_window_request_budgets(tmp_path):
    from ovd_ui.models import ObjectTreeModel
    source=Session(str(ROOT/'test/data/cloth_physx34.pxd2'),str(tmp_path))
    owned=source.presentation(frame=1)
    model=ObjectTreeModel(page_size=9999);model.populate(PvdDataAdapter(owned))
    record=model.canonical_item(-6)
    assert record.element_index==0
    for kwargs in ({'record_requests':[(6,401,0)]*1025},{'read_requests':[(6,401,0,64)]*1025}):
        with pytest.raises(ValueError,match='1024 windows/records'):source.presentation(frame=1,**kwargs)
    explicit=source.presentation(frame=1,record_requests=[(6,401,0)])
    assert explicit.array_record(6,401,0).fields
    assert owned.frame==1  # Model expansion did not publish or consume records.


def test_copy_budget_denial_and_missing_window_do_not_schedule_repeating_reads(tmp_path,display):
    source,identity=large_capture(tmp_path)
    owned=source.presentation(frame=3,inspect_ids=[identity])
    adapter=PvdDataAdapter(owned);prop=adapter.property(identity,'variable')
    requested=[]
    view=inspector({'array_group_size':9999,'max_array_to_load':99999},requested)
    view.build(adapter,[identity]);view.array_positions[(identity,prop.id)]=18000
    display.widgets.clear();view._property(adapter,identity,prop)
    section,frame=expand(display)
    for _ in range(5):frame.rebuild()
    assert len(requested)==1  # Missing owned bytes request once, not once per draw.
    reads=view.read_requests(adapter,[identity]);assert reads==[(identity,prop.id,72000,256)]
    view.reject_read_requests(reads,'Array window paused after native copy budget rejection.')
    for _ in range(5):frame.rebuild()
    assert len(requested)==1 and view.read_requests(adapter,[identity])==[]
    assert display.find('Array window paused after native copy budget rejection.')
    section.collapsed(True);section.collapsed(False)
    assert len(requested)==2 and view.read_requests(adapter,[identity])==reads
    # Aggregate budget uses native current copied bytes, with8MiB reserved.
    class NearBudget:
        copied_bytes=128*1024*1024-8*1024*1024-255
        def __getattr__(self,name):return getattr(owned,name)
    adapter.session=NearBudget()
    assert view.read_requests(adapter,[identity])==[]
    frame.rebuild()
    assert any('owned-data budget' in w.text for w in display.widgets)
    assert len(requested)==2


def test_native_unavailable_window_has_explicit_status_without_retry_loop(tmp_path,display):
    source,identity=large_capture(tmp_path)
    first=source.presentation(frame=3,inspect_ids=[identity])
    adapter=PvdDataAdapter(first);prop=adapter.property(identity,'variable')
    request=(identity,prop.id,72000,256)
    later=source.presentation(frame=2,inspect_ids=[identity],read_requests=[request],allow_unavailable_requests=True)
    assert later.unavailable_reads==[request]
    adapter=PvdDataAdapter(later)
    requested=[]
    view=inspector({'array_group_size':9999,'max_array_to_load':99999},requested)
    # Exercise a deferred array builder whose range became unavailable. The
    # native rejection marker is authoritative even before a parent rebuild.
    view.array_positions[(identity,prop.id)]=18000
    view.array_expanded.add((identity,prop.id))
    view._array(adapter,identity,prop,20000,
        lambda start,count:later.read(identity,prop.id,offset=start*4,limit=count*4),stride=4)
    frame=next(w for w in reversed(display.widgets) if w.kind=='Frame')
    for _ in range(5):frame.rebuild()
    assert not requested
    assert any('unavailable at this frame' in w.text for w in display.widgets)
    # Applying the current parent clamps the range; an empty earlier frame
    # removes it completely. Neither stale marker follows a new window.
    view.build(adapter,[identity])
    assert view.array_positions[(identity,prop.id)]==0
    adapter=PvdDataAdapter(source.presentation(frame=1,inspect_ids=[identity]))
    view.build(adapter,[identity])
    assert (identity,prop.id) not in view.array_positions
    assert view.read_requests(adapter,[identity])==[]
