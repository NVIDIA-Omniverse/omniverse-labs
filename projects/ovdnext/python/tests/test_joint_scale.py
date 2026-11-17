# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native joint scale over real OVD and native-writer indexed PXD2 captures."""
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest
import _pvd_native
from _pvd_native import Session
from test_native_presentation import live, HANDSHAKE, wait_until
from native_fixtures import executable

ROOT=Path(__file__).resolve().parents[2]
ARRAYS=('positions','colors','kinds','categories','object_ids','related_ids','scene_ids','color_roles')

@pytest.fixture(scope='session')
def joint_captures(tmp_path_factory):
    directory=tmp_path_factory.mktemp('native_joints')
    binary=executable('pvd_geometry_fixture','PVD_JOINT_FIXTURE')
    paths={}
    for name,arguments in [('sdk.ovd',[]),('sdk.pxd2',['--legacy-joints']),('sdk-live.ovd',['--joint-live-tail'])]:
        path=directory/name;subprocess.run([str(binary),str(path),*arguments],check=True);paths[name]=path
    return paths

def capture(path,tmp_path,frame=1):
    source=Session(str(path),str(tmp_path));scene=source.scenes()[0]
    assert source.seek(scene.id,frame)
    return source,scene

def same_metadata(a,b):
    for name in ARRAYS[1:]:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

def test_real_ovd_joint_floor_parent_ratio_and_general_gizmo_independence(tmp_path):
    source,scene=capture(ROOT/'test/data/articulation.ovd',tmp_path)
    omitted=source.overlay_snapshot(16,gizmo_scale=2)
    assert omitted.joint_scale is None
    np.testing.assert_allclose(np.linalg.norm(omitted.positions[:,1]-omitted.positions[:,0],axis=1),[3]*3+[2]*3+[3]*3+[2]*3,rtol=1e-5)
    for scale in (0,.5,1,2):
        o=source.overlay_snapshot(16,gizmo_scale=7,joint_scale=scale,joint_base_length=.25)
        assert o.joint_scale==scale and o.joint_base_length==.25
        same_metadata(omitted,o)
        np.testing.assert_array_equal(omitted.positions[:,0],o.positions[:,0])
        lengths=np.linalg.norm(o.positions[:,1]-o.positions[:,0],axis=1)
        np.testing.assert_allclose(sorted(lengths),[.25*max(scale,.5)]*6+[.375*max(scale,.5)]*6,rtol=1e-5,atol=1e-6)
        other=source.overlay_snapshot(16,gizmo_scale=.2,joint_scale=scale,joint_base_length=.25)
        np.testing.assert_array_equal(other.positions,o.positions)

@pytest.mark.parametrize('name',['sdk.ovd','sdk.pxd2'])
def test_captured_sdk_events_use_raw_scale_even_in_ovd_container(name,joint_captures,tmp_path):
    source,scene=capture(joint_captures[name],tmp_path)
    assert source.info().ovd==(name.endswith('.ovd'))
    baseline=source.overlay_snapshot(16|128,gizmo_scale=1)
    assert np.count_nonzero(baseline.kinds==4)==6
    assert np.count_nonzero(baseline.categories==16)==242
    for scale in (0,.5,1,2):
        o=source.overlay_snapshot(16|128,gizmo_scale=9,joint_scale=scale,joint_base_length=123)
        same_metadata(baseline,o)
        # All emitted non-joint geometry, including recorded colors/IDs, is exact.
        nonjoint=o.categories!=16
        np.testing.assert_array_equal(o.positions[nonjoint],baseline.positions[nonjoint])
        arrows=o.kinds==4
        np.testing.assert_array_equal(o.positions[arrows,0],baseline.positions[arrows,0])
        np.testing.assert_allclose(o.positions[arrows,1]-o.positions[arrows,0],(baseline.positions[arrows,1]-baseline.positions[arrows,0])*scale,atol=1e-6)
        # First20limit segments have a physical anchor at x=2; their circle
        # radius scales but the captured limit distance never changes.
        limits=np.flatnonzero((o.categories==16)&(o.kinds==2))
        linear=limits[:20];anchor=np.array([2,0,0])
        np.testing.assert_allclose(o.positions[linear,:2]-anchor,(baseline.positions[linear,:2]-anchor)*scale,atol=1e-6)
        # Angular/single-cone/double-cone frames in the native fixtures are identity.
        np.testing.assert_allclose(o.positions[limits[20:],:2],baseline.positions[limits[20:],:2]*scale,atol=1e-6)
        assert o.joint_scale==scale and o.joint_base_length==123


def test_indexed_event_history_and_owned_request_survive_new_frames_and_close(joint_captures,tmp_path):
    source,scene=capture(joint_captures['sdk.pxd2'],tmp_path)
    first=source.presentation(scene.id,1,overlay_mask=16,joint_scale=.5,joint_base_length=10)
    positions=first.overlay_snapshot().positions.copy()
    second=source.presentation(scene.id,2,overlay_mask=16,joint_scale=2,joint_base_length=20)
    assert second.frame==2 and second.joint_scale==2 and second.joint_base_length==20
    assert (second.overlay_snapshot().object_ids==2).all()
    np.testing.assert_allclose(second.overlay_snapshot().positions[:3,0],[[2,4,5]]*3)
    third=source.presentation(scene.id,3,overlay_mask=16,joint_scale=0)
    assert not len(third.overlay_snapshot().kinds)  # No debug update in this frame.
    del source,second,third
    assert first.joint_scale==.5 and first.joint_base_length==10
    assert first.overlay_snapshot().joint_scale==.5
    np.testing.assert_array_equal(first.overlay_snapshot().positions,positions)
    assert not first.overlay_snapshot().positions.flags.writeable

@pytest.mark.parametrize('kwargs',[{'joint_scale':-.1},{'joint_scale':2.01},{'joint_scale':float('nan')},
    {'joint_scale':float('inf')},{'joint_base_length':0},{'joint_base_length':float('nan')},
    {'joint_scale':2,'joint_base_length':3e38}])
def test_invalid_joint_request_rejected_before_native_frame_change(kwargs,tmp_path):
    source,scene=capture(ROOT/'test/data/articulation.ovd',tmp_path)
    with pytest.raises(ValueError):source.presentation(scene.id,2,overlay_mask=16,**kwargs)
    assert source.scenes()[0].current_frame==1
    with pytest.raises(ValueError):source.overlay_snapshot(16,**kwargs)
    assert source.scenes()[0].current_frame==1


def test_new_arguments_are_keyword_only_and_old_positional_calls_remain(tmp_path):
    source,scene=capture(ROOT/'test/data/articulation.ovd',tmp_path)
    old=source.presentation(scene.id,1,[],[],[],16,2,10,False)
    assert old.joint_scale is None
    with pytest.raises(TypeError):source.presentation(scene.id,1,[],[],[],16,2,10,False,1)
    with pytest.raises(TypeError):source.overlay_snapshot(16,2,10,1)
    current=source.overlay_snapshot(16,2,10,joint_scale=None)
    np.testing.assert_array_equal(old.overlay_snapshot().positions,current.positions)


def test_partial_live_retains_geometry_and_its_joint_request_identity(live,joint_captures):
    source,connection=live
    payload=joint_captures['sdk-live.ovd'].read_bytes()
    end2=payload.index(struct.pack('<BQQ',12,1,2))+17
    stop3=payload.index(struct.pack('<BQQ',12,1,3))
    connection.sendall(HANDSHAKE+payload[:end2])
    wait_until(lambda:source.info().completed_serial==2)
    first=source.presentation(frame=2,overlay_mask=16,joint_scale=.5,joint_base_length=10)
    assert first.ready and first.request_applied and len(first.overlay_snapshot().kinds)
    old=first.overlay_snapshot().positions.copy()
    connection.sendall(payload[end2:stop3])
    wait_until(lambda:source.info().unpublished_changes)
    later=source.presentation(first.scene_id,1,overlay_mask=16,joint_scale=2,joint_base_length=20)
    assert later.ready and not later.request_applied
    assert later.joint_scale==.5 and later.joint_base_length==10
    np.testing.assert_array_equal(later.overlay_snapshot().positions,old)
    cached=source.overlay_snapshot(16,joint_scale=2,joint_base_length=20)
    assert cached.joint_scale==.5 and cached.joint_base_length==10
    with pytest.raises(ValueError):source.presentation(first.scene_id,1,joint_scale=3)
    with pytest.raises(ValueError):source.overlay_snapshot(16,joint_base_length=0)
    connection.sendall(payload[stop3:])
    wait_until(lambda:source.info().completed_serial==3)
    applied=source.presentation(first.scene_id,1,overlay_mask=16,joint_scale=2,joint_base_length=20)
    assert applied.request_applied and applied.joint_scale==2 and applied.joint_base_length==20
    np.testing.assert_array_equal(first.overlay_snapshot().positions,old)


@pytest.mark.parametrize('name',['articulation.ovd','cloth_physx34.pxd2','sdk.ovd','sdk.pxd2'])
def test_independent_joint_request_preserves_every_nonjoint_overlay_byte(name,joint_captures,tmp_path):
    source,scene=capture(joint_captures.get(name,ROOT/'test/data'/name),tmp_path)
    before=source.overlay_snapshot(511,gizmo_scale=2,tolerance_speed=7)
    after=source.overlay_snapshot(511,gizmo_scale=2,tolerance_speed=7,joint_scale=0,joint_base_length=.25)
    same_metadata(before,after)
    nonjoint=before.categories!=16
    np.testing.assert_array_equal(before.positions[nonjoint],after.positions[nonjoint])
    assert before.diagnostics==after.diagnostics
    if before.query_geometry is not None:
        for field in ('transforms','dimensions','bounds','object_ids','shape_ids','geometry_ids','scene_ids','kinds','override_colors'):
            np.testing.assert_array_equal(getattr(before.query_geometry,field),getattr(after.query_geometry,field))


def test_owned_joint_request_exists_even_when_overlays_are_disabled(tmp_path):
    source,scene=capture(ROOT/'test/data/articulation.ovd',tmp_path)
    view=source.presentation(scene.id,1,joint_scale=.25,joint_base_length=3)
    assert view.joint_scale==.25 and view.joint_base_length==3 and view.overlay_snapshot() is None
    newer=source.presentation(scene.id,1)
    assert newer.joint_scale is None and view.joint_scale==.25
