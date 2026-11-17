# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded profiler grouping preserves exact native keys and measurement totals."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import _pvd_native as native
from ovd_ui.profiling import EventKey, thread_name
from ovd_ui.widgets.profile_view import ProfileTree

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('filename,frame',[
    ('cloth_physx34.pxd2',2),('apex_destruction.pxd2',3),('ue4_shooter.pxd2',30)])
@pytest.mark.parametrize('mode',[0,1])
def test_original_hierarchies_cover_each_native_event_once(tmp_path,filename,frame,mode):
    session=native.Session(str(ROOT/'test/data'/filename),str(tmp_path))
    scene=session.scenes()[-1].id
    assert session.seek(scene,frame)
    snapshot=native.profile_snapshot(session,scene)
    rows=snapshot.summaries
    assert len(rows)>5
    model=ProfileTree()
    model.populate(rows,{zone.id:zone.name for zone in snapshot.zones},mode)
    assert set(model.leaves)=={EventKey.of(row) for row in rows}
    expected_roots={f'Context {row.context_id}' if mode else thread_name(row.thread_id) for row in rows}
    assert {root.values[0].as_string for root in model.roots}==expected_roots
    visited=[]
    for root in model.roots:
        assert root.parent is None and not root.keys
        group_keys=[]
        for item in root.children:
            assert item.parent is root and not item.children
            keys=set(item.keys)
            group_keys.extend(item.keys);visited.extend(item.keys)
            records=[row for row in rows if EventKey.of(row) in keys]
            assert item.count==sum(row.count for row in records)
            assert item.total_ms==pytest.approx(sum(row.total_ms for row in records))
            assert item.percent==pytest.approx(sum(row.percent for row in records))
            assert item.values[3].as_string==f'{item.percent:.2f}'
            assert item.values[1].as_string==str(item.count)
            assert all(model.leaves[key] is item for key in keys)
            if mode:
                assert len({key.context_id for key in keys})==len({key.thread_id for key in keys})==1
                assert item.values[0].as_string==thread_name(records[0].thread_id)
                assert root.values[0].as_string==f'Context {records[0].context_id}'
            else:
                assert len(keys)==1 and item.key in keys
                row=records[0]
                assert item.values[0].as_string==f'{row.name} [Context {row.context_id}]'
                assert root.values[0].as_string==thread_name(row.thread_id)
                assert f'Zone {row.zone_id} · Event {row.event_id}' in item.tooltip
        records=[row for row in rows if EventKey.of(row) in group_keys]
        assert root.count==sum(row.count for row in records)
        assert root.total_ms==pytest.approx(sum(row.total_ms for row in records))
        assert root.percent==pytest.approx(sum(row.percent for row in records))
    assert len(visited)==len(set(visited))==len(rows)
    identities=dict(model.leaves)
    for column in range(4):
        for descending in (False,True):
            model.sort(column,descending)
            for siblings in (model.roots,*(root.children for root in model.roots)):
                values=[item.values[0].as_string if column==0 else
                        (item.count,item.total_ms,item.percent)[column-1] for item in siblings]
                assert values==sorted(values,reverse=descending)
            assert model.leaves==identities


@pytest.mark.parametrize('mode',[0,1])
def test_sort_uses_unrounded_numbers_and_percentages_are_not_clamped(mode):
    records=[NS(zone_id=1,event_id=i,context_id=1,thread_id=i,name=name,
                count=count,total_ms=duration,percent=percent)
             for i,(name,count,duration,percent) in enumerate([
                 ('Zeta',2,9.99994,60.001),('Alpha',10,9.99991,60.004),('Beta',100,2.,10.)])]
    tree=ProfileTree();tree.populate(records,{1:'Test'},mode)
    items=[tree.leaves[EventKey.of(row)] for row in records]
    assert items[0].values[2].as_string==items[1].values[2].as_string
    assert items[0].values[3].as_string==items[1].values[3].as_string
    siblings=tree.roots[0].children if mode else tree.roots
    tree.sort(1);assert [item.count for item in siblings]==[2,10,100]
    tree.sort(2);assert [item.count for item in siblings]==[100,10,2]
    tree.sort(3,True);assert [item.count for item in siblings]==[10,2,100]
    if mode:assert tree.roots[0].percent==pytest.approx(130.005)


@pytest.mark.parametrize('mode',[0,1])
def test_duplicate_names_and_large_unsigned_contexts_never_alias(mode):
    records=[NS(zone_id=zone,event_id=event,context_id=context,thread_id=thread,
                name='Same name',count=1,total_ms=1.,percent=1.)
             for zone in (1,2) for event in (3,4)
             for context in (2**63,2**63+1) for thread in (7,native.CROSS_THREAD_ID)]
    tree=ProfileTree();tree.populate(records,{1:'First zone',2:'Second zone'},mode)
    assert len(tree.leaves)==16
    assert sum(item.count for item in tree.roots)==16
    for row in records:
        item=tree.leaves[EventKey.of(row)]
        expected={EventKey.of(other) for other in records if
                  (other.context_id,other.thread_id)==(row.context_id,row.thread_id)} if mode else {EventKey.of(row)}
        assert set(item.keys)==expected
    # A later filtered/empty publication must not retain another event's mapping.
    tree.populate(records[:1],{1:'First zone'},mode)
    assert set(tree.leaves)=={EventKey.of(records[0])}
    tree.populate([],{},mode)
    assert tree.roots==[] and tree.leaves=={}


@pytest.mark.parametrize('mode',[0,1])
def test_hidden_statistics_are_recoverable_without_aliasing_native_groups(mode):
    records=[NS(zone_id=z,event_id=e,context_id=c,thread_id=t,name=f'Event {e}',
                count=e,total_ms=float(e),percent=float(e))
             for z in (1,2) for e in (1,2) for c in (2**63,2**63+1)
             for t in (native.CROSS_THREAD_ID,2**32-1)]
    hidden={EventKey.of(r) for r in records if r.event_id==1}
    names={2**63:'Same scene caption',2**63+1:'Same scene caption'}
    tree=ProfileTree();tree.populate(records,{},mode,hidden=hidden,context_names=names)
    assert tree.hidden_root is tree.roots[0]
    assert {item.key for item in tree.hidden_root.children}==hidden
    assert tree.hidden_root.count==8 and tree.hidden_root.total_ms==8
    assert all(item.hidden and item.parent is tree.hidden_root for item in tree.hidden_root.children)
    visible=[item for root in tree.roots if root is not tree.hidden_root for item in root.children]
    assert {key for item in visible for key in item.keys}=={EventKey.of(r) for r in records}-hidden
    assert sum(root.count for root in tree.roots)==24
    for row in records:
        item=tree.leaves[EventKey.of(row)]
        assert str(row.context_id) in item.tooltip
        assert 'Same scene caption' in item.tooltip
    # No synthetic thread-ID sentinel: even UINT32_MAX remains a real thread.
    for column in range(4):
        for descending in (True,False):
            tree.sort(column,descending)
            assert tree.roots[0] is tree.hidden_root
            assert len(tree.leaves)==len(records)
    # Hidden keys survive frame gaps without an empty recovery group.
    tree.populate([r for r in records if r.event_id==2],{},mode,hidden=hidden,context_names=names)
    assert tree.hidden_root is None
    tree.populate(records,{},mode,hidden=hidden,context_names=names)
    assert tree.hidden_root and len(tree.hidden_root.children)==8
    tree.populate(records,{},mode,hidden=(),context_names=names)
    assert tree.hidden_root is None and all(not item.hidden for item in tree.leaves.values())
