# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection gates use owned native overlay IDs, without an engine/GPU read."""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _pvd_native import Session
from capture_semantics import dynamic_geometry_owner, multi_geom_roles

from ovd_ui.application import Application
from ovd_ui.overlays import OverlayRenderer, SELECTION_OVERLAYS, overlay_visibility
from ovd_ui.render_settings import DisplayMode
from ovd_ui.interaction import InteractionState
from ovd_ui.settings import Preferences
from ovd_ui.renderer import ViewportRenderer

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def overlay_bundle(tmp_path):
    session=Session(str(ROOT/'test/data/multi_geom.ovd'),str(tmp_path))
    actor=dynamic_geometry_owner(session,2,2)
    scene=session.scenes()[0]
    presentation=session.presentation(scene.id,2,overlay_mask=511)
    snapshot=presentation.overlay_snapshot()
    articulation_root=multi_geom_roles(presentation).root
    assert presentation.frame==2 and set(snapshot.scene_ids)=={scene.instance_id}
    assert Counter(snapshot.categories.tolist())=={1:132,2:33,4:8,8:8,16:18,256:24}
    del session, presentation
    return snapshot,actor,articulation_root


@pytest.fixture
def owned_overlay(overlay_bundle):
    return overlay_bundle[0]


@pytest.fixture
def selection_ids(owned_overlay,overlay_bundle):
    snapshot=owned_overlay
    moving=set(snapshot.object_ids[snapshot.categories==4])
    actor=overlay_bundle[1]
    axes=(snapshot.categories==2)&(snapshot.object_ids==actor)
    joint_actor=overlay_bundle[2]
    # The root link has exactly one incident joint (20 body + 6 joint rows),
    # unlike an interior link which legitimately contributes two joints.
    joints=np.flatnonzero((snapshot.categories==16)&(snapshot.related_ids==joint_actor).any(axis=1))
    assert len(joints)==6 and len(set(snapshot.object_ids[joints]))==1
    joint=joints[0]
    return SimpleNamespace(scene=int(snapshot.scene_ids[0]),actor=actor,
        shape=int(snapshot.related_ids[axes,0][0]),
        static=int(next(i for i in snapshot.object_ids[snapshot.categories==1] if i not in moving)),
        joint=int(snapshot.object_ids[joint]),joint_actor=joint_actor)


@pytest.mark.parametrize('category,total,selected_count', [(1,132,12),(2,33,3),(4,8,1),(8,8,1),(256,24,3)])
def test_real_native_actor_gates_and_unrelated_categories(owned_overlay, selection_ids, category, total, selected_count):
    snapshot=owned_overlay
    ids=selection_ids
    original=snapshot.positions.copy()
    for mode,count in ((DisplayMode.NONE,0),(DisplayMode.ALL,total),
                       (DisplayMode.SELECTED,selected_count),(DisplayMode.UNSELECTED,total-selected_count)):
        mask=overlay_visibility(snapshot,ids.scene,selected={ids.actor},modes={category:mode})
        assert np.count_nonzero(mask & (snapshot.categories==category))==count
        assert mask[snapshot.categories!=category].all()  # Unconfigured families retain ALL.
        if mode==DisplayMode.SELECTED:
            assert set(snapshot.object_ids[mask & (snapshot.categories==category)])=={ids.actor}
    np.testing.assert_array_equal(snapshot.positions,original)
    assert not snapshot.positions.flags.writeable


def test_empty_selection_and_shape_selection_do_not_promote_owning_actor(owned_overlay, selection_ids):
    ids=selection_ids
    modes={bit:DisplayMode.SELECTED for bit,_ in SELECTION_OVERLAYS}
    assert ids.shape!=ids.actor
    for selected in (set(),{ids.shape},{-6},{999999}):
        # Selecting an attached shape does not select its actor, COM, or joints.
        mask=overlay_visibility(owned_overlay,ids.scene,selected=selected,modes=modes)
        assert not mask.any()
    modes={bit:DisplayMode.UNSELECTED for bit,_ in SELECTION_OVERLAYS}
    assert overlay_visibility(owned_overlay,ids.scene,selected=set(),modes=modes).all()


def test_hidden_and_scene_filters_still_win_over_selected_mode(owned_overlay, selection_ids):
    ids=selection_ids
    modes={bit:DisplayMode.SELECTED for bit,_ in SELECTION_OVERLAYS}
    assert not overlay_visibility(owned_overlay,ids.scene+999,selected={ids.actor},modes=modes).any()
    mask=overlay_visibility(owned_overlay,ids.scene,hidden={ids.actor},selected={ids.actor},modes=modes)
    assert not mask.any()
    # Existing shape-related visibility still suppresses its axes and velocity.
    mask=overlay_visibility(owned_overlay,ids.scene,hidden={ids.shape},selected={ids.actor},modes=modes)
    assert np.count_nonzero(mask & (owned_overlay.categories==1))==12
    assert np.count_nonzero(mask & (owned_overlay.categories==256))==3
    assert not mask[np.isin(owned_overlay.categories,[2,4,8])].any()


@pytest.fixture
def owned_contacts(tmp_path):
    session=Session(str(ROOT/'test/data/2026_04_28_15_24_52_000000a751246bf6.ovd'),str(tmp_path))
    view=session.presentation(session.scenes()[0].id,1,overlay_mask=1|32)
    snapshot=view.overlay_snapshot()
    pairs=Counter(map(tuple,snapshot.related_ids[snapshot.categories==32].tolist()))
    assert sorted(pairs.values())==[18,18]  # Two pairs, six contacts of three lines each.
    return snapshot


def test_contacts_filter_either_actor_not_scene_and_keep_visibility_gates(owned_contacts):
    snapshot=owned_contacts
    contact=snapshot.categories==32
    first,second=np.unique(snapshot.related_ids[contact],axis=0)
    common=set(first)&set(second)
    assert len(common)==1
    single={int(next(iter(set(first)-common)))}
    scene=int(snapshot.scene_ids[contact][0])
    for selected,yes,no in ((single,18,18),(common,36,0),(set(),0,36),({scene},0,36)):
        for mode,count in ((DisplayMode.NONE,0),(DisplayMode.ALL,36),
                           (DisplayMode.SELECTED,yes),(DisplayMode.UNSELECTED,no)):
            mask=overlay_visibility(snapshot,scene,selected=selected,modes={32:mode})
            assert np.count_nonzero(mask & contact)==count
            assert mask[~contact].all()
    assert not overlay_visibility(snapshot,scene+999,selected=common,modes={32:2}).any()
    hidden=overlay_visibility(snapshot,scene,hidden=set(first),selected=single,modes={32:2})
    assert not hidden[contact].any()


def test_contact_draw_and_pick_filter_the_same_owned_rows(owned_contacts):
    renderer,emitted=drawing_sink(owned_contacts)
    renderer.set_mode(1,DisplayMode.NONE)
    renderer.set_mode(32,DisplayMode.SELECTED)
    first,second=np.unique(owned_contacts.related_ids[owned_contacts.categories==32],axis=0)
    selected=set(first)-set(second)
    scene=int(owned_contacts.scene_ids[0])
    original=owned_contacts.positions.copy()
    for picking in (False,True):
        renderer.draw(np.eye(4),np.array([100.,100.,100.]),20,scene,selected=selected,picking=picking)
        assert renderer.primitive_count==18
    for visible,pick in zip(emitted[:3],emitted[3:]):
        np.testing.assert_array_equal(visible[1],pick[1])
        np.testing.assert_array_equal(visible[2],pick[2])
    assert renderer.snapshot is owned_contacts
    np.testing.assert_array_equal(owned_contacts.positions,original)


@pytest.mark.parametrize('selected',[{7},{7,0,-1}])
def test_contacts_without_pair_metadata_remain_visible_but_unresolved_actors_are_not_selected(selected):
    snapshot=SimpleNamespace(kinds=np.full(4,2),categories=np.full(4,32),
        object_ids=np.full(4,5),scene_ids=np.full(4,5),
        related_ids=np.array([[0,0],[-1,-1],[7,-1],[7,8]]))
    assert overlay_visibility(snapshot,5,selected=selected,modes={32:2}).tolist()==[True,False,True,True]
    assert overlay_visibility(snapshot,5,selected=selected,modes={32:3}).tolist()==[True,True,False,False]


def test_joint_frames_follow_joint_or_either_recorded_actor_and_com_follows_its_actor(tmp_path):
    session=Session(str(ROOT/'test/data/articulation.ovd'),str(tmp_path))
    view=session.presentation(session.scenes()[0].id,1,overlay_mask=16|256)
    snapshot=view.overlay_snapshot()
    joints=snapshot.categories==16
    com=snapshot.categories==256
    scene=int(snapshot.scene_ids[0])
    joint=int(snapshot.object_ids[joints][0])
    parent,child=map(int,snapshot.related_ids[joints][0])
    for selected in ({joint},{parent},{child},set(),{scene}):
        expected=joints & ((snapshot.object_ids==joint) if selected=={joint} else
            np.array([bool(set(pair)&selected) for pair in snapshot.related_ids]))
        yes=overlay_visibility(snapshot,scene,selected=selected,modes={16:2,256:2})
        no=overlay_visibility(snapshot,scene,selected=selected,modes={16:3,256:3})
        np.testing.assert_array_equal(yes[joints],expected[joints])
        assert np.all(yes^no)
        assert np.count_nonzero(yes & com)==(3 if selected in ({parent},{child}) else 0)
    assert not overlay_visibility(snapshot,scene,modes={16:0,256:0}).any()


class Uniform:
    def write(self,data):self.data=data


def drawing_sink(snapshot):
    renderer=OverlayRenderer.__new__(OverlayRenderer)
    renderer.snapshot=snapshot
    renderer.context=SimpleNamespace()
    renderer.program={'view_projection':Uniform(),'picking':Uniform()}
    renderer.modes={}
    renderer.palette=None
    emitted=[]
    def upload(mode,positions,colors,ids):
        emitted.append((mode,positions.copy(),ids.copy()))
    renderer._upload=upload
    return renderer,emitted


def test_visible_and_pick_batches_use_identical_native_gate_without_republication(owned_overlay, selection_ids):
    ids=selection_ids
    renderer,emitted=drawing_sink(owned_overlay)
    for bit,_ in SELECTION_OVERLAYS:renderer.set_mode(bit,DisplayMode.SELECTED)
    for selected,count in (({ids.actor},20),({ids.static},15),({ids.joint_actor},26),
                           ({ids.joint},6),(set(),0)):
        renderer.draw(np.eye(4),np.array([100.,100.,100.]),20,ids.scene,selected=selected)
        visible=emitted[:];emitted.clear()
        assert renderer.primitive_count==count
        renderer.draw(np.eye(4),np.array([100.,100.,100.]),20,ids.scene,selected=selected,picking=True)
        assert renderer.primitive_count==count
        for first,second in zip(visible,emitted):
            assert first[0]==second[0]
            np.testing.assert_array_equal(first[1],second[1])
            np.testing.assert_array_equal(first[2],second[2])
        emitted.clear()
    assert renderer.snapshot is owned_overlay


def application(tmp_path):
    app=Application.__new__(Application)
    app.overlay_mask=16|32
    app.overlay_modes={}
    app.gizmo_scale=1.
    app.interaction=InteractionState()
    app.workflow=SimpleNamespace(settings=Preferences(tmp_path/'preferences.json'))
    app.renderer=SimpleNamespace(overlays=OverlayRenderer.__new__(OverlayRenderer))
    app.renderer.overlays.modes={}
    app.dirty=False
    requested=[]
    app._request_presentation=lambda:requested.append(app.overlay_mask)
    return app,requested


def test_only_category_enable_changes_request_owned_overlay_publication(tmp_path):
    app,requested=application(tmp_path)
    app._set_overlay_mode(1,DisplayMode.SELECTED)
    assert app.overlay_mask==16|32|1 and requested==[16|32|1]
    requested.clear()
    for mode in (DisplayMode.ALL,DisplayMode.UNSELECTED,DisplayMode.SELECTED):
        app.dirty=False
        app._set_overlay_mode(1,mode)
        assert app.dirty and requested==[]
        assert app.renderer.overlays.modes[1]==mode==app._overlay_mode(1)
    app.dirty=False
    app._set_overlay_mode(1,DisplayMode.SELECTED)
    assert not app.dirty and requested==[]
    app._set_overlay_mode(1,DisplayMode.NONE)
    assert app.overlay_mask==16|32 and requested==[16|32]
    assert app.renderer.overlays.modes[1]==DisplayMode.NONE


def test_existing_other_toggles_and_boolean_overlay_entrypoint_still_work(tmp_path):
    app,requested=application(tmp_path)
    app._set_overlay_mode(2,DisplayMode.UNSELECTED)
    app._set_overlay(128,True)
    app._set_overlay(32,False)
    assert app._overlay_mode(2)==DisplayMode.UNSELECTED
    assert app.overlay_mask==2|16|128
    app._set_overlay(1,True)
    assert app._overlay_mode(1)==DisplayMode.ALL
    app._set_overlay(1,False)
    assert app._overlay_mode(1)==DisplayMode.NONE
    assert app.overlay_mask==2|16|128
    requested.clear()
    app._set_overlay(128,True)
    assert requested==[]


@pytest.mark.parametrize('category,mode',[(64,1),(0,1),(1,-1),(1,4),(1,True)])
def test_invalid_modes_do_not_partially_change_state(tmp_path,category,mode):
    app,requested=application(tmp_path)
    with pytest.raises(ValueError):app._set_overlay_mode(category,mode)
    assert app.overlay_modes=={} and app.overlay_mask==16|32 and requested==[]


def test_viewport_pick_forwards_active_native_selection_to_overlay_draw():
    # Exercise production pick orchestration without owning a GPU context.
    renderer=ViewportRenderer.__new__(ViewportRenderer)
    renderer.size=(100,100)
    renderer.pick_buffer=SimpleNamespace(clear=lambda *args:None,read=lambda **kwargs:bytes(4))
    renderer.context=SimpleNamespace(scope=lambda **kwargs:nullcontext())
    renderer._draw=lambda **kwargs:None
    renderer._draw_queries=lambda **kwargs:None
    renderer._view_projection=lambda:np.eye(4)
    renderer.eye=lambda:np.zeros(3)
    renderer.camera=SimpleNamespace(distance=10)
    renderer.scene_id=30
    renderer.hidden={25}
    renderer.selected={33}
    calls=[]
    renderer.overlays=SimpleNamespace(draw=lambda *args,**kwargs:calls.append((args,kwargs)))
    assert renderer.pick(20,20) is None
    assert calls[0][1]=={'selected':{33},'picking':True}
    assert calls[0][0][-1]=={25}
