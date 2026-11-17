# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-derived velocity scaling belongs to the first owned publication."""
import gc
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest
import _pvd_native
from _pvd_native import Session
from test_native_presentation import live, wait_until, HANDSHAKE
from test_application_presentation import app
from native_fixtures import executable

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def writer():
    return executable("pvd_tolerance_fixture", "PVD_TOLERANCE_FIXTURE")

@pytest.fixture
def capture(writer,tmp_path):
    def make(mode):
        path=tmp_path/(mode+".ovd")
        subprocess.run([str(writer),str(path),mode],check=True)
        return path,Session(str(path),str(tmp_path))
    return make

def view_for(source,index=0,frame=1,**kw):
    return source.presentation(sorted(source.scenes(),key=lambda s:s.stream_id)[index].id,frame,overlay_mask=4,gizmo_scale=2,tolerance_speed=None,**kw)

def assert_velocity(view,speed):
    overlay=view.overlay_snapshot()
    selected=next(s.instance_id for s in view.scenes() if s.id==view.scene_id)
    rows=overlay.positions[overlay.scene_ids==selected]
    assert len(rows)==1
    np.testing.assert_allclose(rows[:,1]-rows[:,0],[[24/speed,12/speed,0]],rtol=1e-6)

@pytest.mark.parametrize("mode,speeds,reason",[("scene",[2,3],"scene"),("owner",[5,7],"physics-owner"),("single",[5,5],"single-physics")])
def test_first_view_scene_switch_and_rewind_use_exact_captured_speed(capture,mode,speeds,reason):
    _,source=capture(mode)
    first=view_for(source)
    assert first.tolerance_speed==speeds[0]
    assert first.tolerance_speed_resolution==reason
    assert first.tolerance_speed_source_id>0 and first.tolerance_speed_property_id>=0
    assert_velocity(first,speeds[0])
    second=view_for(source,1)
    assert second.tolerance_speed==speeds[1]
    assert_velocity(second,speeds[1])
    third=view_for(source,frame=2)
    assert third.tolerance_speed==speeds[0]*2
    assert_velocity(third,speeds[0]*2)
    restored=view_for(source)
    assert restored.tolerance_speed_source_id==first.tolerance_speed_source_id
    np.testing.assert_array_equal(restored.overlay_snapshot().positions,first.overlay_snapshot().positions)
    del source,second,third,restored
    gc.collect();assert_velocity(first,speeds[0])

@pytest.mark.parametrize("mode,reason",[("ambiguous","ambiguous-physics"),("conflicting","ambiguous-physics"),("missing","missing"),
    *[(mode,"invalid") for mode in ("wrong-type","variable","zero","negative","nan","infinity","bad-length")]])
def test_missing_ambiguous_and_malformed_scales_use_fresh_fallback(capture,mode,reason):
    _,source=capture(mode);view=view_for(source)
    assert view.tolerance_speed==1
    assert view.tolerance_speed_source_id==0 and view.tolerance_speed_property_id==-1
    assert view.tolerance_speed_resolution==reason
    assert_velocity(view,1)

def test_explicit_and_default_api_values_keep_existing_overlay_contract(capture):
    _,source=capture("scene");scene=source.scenes()[0]
    for kwargs,speed in [({},10),({"tolerance_speed":4},4)]:
        view=source.presentation(scene.id,1,overlay_mask=4,gizmo_scale=2,**kwargs)
        assert view.tolerance_speed==speed and view.tolerance_speed_resolution=="explicit"
        assert view.tolerance_speed_source_id==0
        assert_velocity(view,speed)
        direct=source.overlay_snapshot(4,gizmo_scale=2,tolerance_speed=speed)
        np.testing.assert_array_equal(view.overlay_snapshot().positions,direct.positions)
    for speed in (0,-1,float("nan"),float("inf")):
        with pytest.raises(ValueError):source.presentation(scene.id,2,tolerance_speed=speed)
    assert source.scenes()[0].current_frame==1

def test_dead_physics_and_capture_replacement_never_reuse_prior_scale(capture):
    _,source=capture("dead");first=view_for(source);assert first.tolerance_speed==5
    later=view_for(source,frame=2);assert later.tolerance_speed==1
    _,other=capture("missing");fresh=view_for(other);assert fresh.tolerance_speed==1
    assert fresh.capture_key!=first.capture_key
    assert_velocity(first,5)

@pytest.mark.parametrize("filename,expected",[("test_scene.ovd",10),("cloth_physx34.pxd2",10),("ue4_shooter.pxd2",1000)])
def test_real_native_schema_speed_is_owned_without_inspection(filename,expected,tmp_path):
    source=Session(str(ROOT/"test/data"/filename),str(tmp_path))
    view=source.presentation(frame=1,tolerance_speed=None,overlay_mask=4)
    assert view.tolerance_speed==expected
    assert view.tolerance_speed_source_id>0 and view.inspected_ids==[]
    raw=view.read(view.tolerance_speed_source_id,view.tolerance_speed_property_id)
    assert struct.unpack("<f",raw[-4:])[0]==expected
    direct=source.overlay_snapshot(4,tolerance_speed=expected)
    np.testing.assert_array_equal(view.overlay_snapshot().positions,direct.positions)

def test_partial_live_speed_mutation_retains_whole_prior_view(capture,live):
    path,_=capture("scene");payload=path.read_bytes();source,connection=live
    stop=struct.pack("<BQQ",12,1,1);boundary=payload.index(stop)+len(stop)
    next_stop=payload.index(struct.pack("<BQQ",12,1,2))
    connection.sendall(HANDSHAKE+payload[:boundary]);wait_until(lambda:source.info().completed_serial==1)
    source.refresh_live()
    first=view_for(source)
    assert first.request_applied and first.tolerance_speed==2
    connection.sendall(payload[boundary:next_stop]);wait_until(lambda:source.info().unpublished_changes)
    deferred=source.presentation(frame=2,tolerance_speed=None,overlay_mask=4,gizmo_scale=2)
    assert not deferred.request_applied
    assert (deferred.frame,deferred.generation,deferred.tolerance_speed,deferred.tolerance_speed_source_id)==(first.frame,first.generation,2,first.tolerance_speed_source_id)
    np.testing.assert_array_equal(deferred.overlay_snapshot().positions,first.overlay_snapshot().positions)
    connection.sendall(payload[next_stop:]);wait_until(lambda:source.info().completed_serial==3)
    applied=source.presentation(first.scene_id,2,tolerance_speed=None,overlay_mask=4,gizmo_scale=2)
    assert applied.request_applied and applied.tolerance_speed==4
    assert_velocity(applied,4);assert_velocity(first,2)

def test_application_first_overlay_and_scene_switch_adopt_same_owned_speed(app,capture):
    path,source=capture("scene");app.overlay_mask=4;app.gizmo_scale=2
    assert app._display(path,source)
    assert app.tolerance_speed==app.presentation.tolerance_speed
    assert_velocity(app.presentation,app.tolerance_speed)
    other=next(s for s in app.scenes if s.id!=app.scene_id)
    assert app._refresh(scene_id=other.id,frame=1)
    assert app.tolerance_speed==(3 if other.stream_id==20 else 2)
    assert app.renderer.overlays is app.presentation.overlay_snapshot()
    assert_velocity(app.presentation,app.tolerance_speed)
    path,missing=capture("missing")
    assert app._display(path,missing)
    assert app.tolerance_speed==1 and app.presentation.tolerance_speed_source_id==0
    assert_velocity(app.presentation,1)

@pytest.mark.parametrize("mode",["short","empty"])
def test_truncated_fixed_schema_is_rejected_by_original_native_loader(capture,mode):
    with pytest.raises(RuntimeError,match="payload does not match"):
        capture(mode)


def test_new_live_segment_reused_handles_get_owned_current_source_identity(live):
    source,connection=live
    payload=(ROOT/"test/data/resample_two_segment.ovd").read_bytes()
    first_end,second_end=42117,98924
    connection.sendall(HANDSHAKE+payload[:first_end])
    wait_until(lambda:source.info().completed_serial==120)
    first=source.presentation(frame=1,tolerance_speed=None,overlay_mask=4)
    assert first.tolerance_speed==10 and first.tolerance_speed_source_id>0
    assert first.tolerance_length==1 and first.tolerance_length_source_id==first.tolerance_speed_source_id
    connection.sendall(payload[first_end:first_end+1024])
    wait_until(lambda:source.info().unpublished_changes)
    old=source.presentation(frame=1,tolerance_speed=None,overlay_mask=4)
    assert not old.request_applied and old.generation==first.generation
    assert old.tolerance_length_source_id==first.tolerance_length_source_id
    connection.sendall(payload[first_end+1024:second_end])
    wait_until(lambda:source.info().completed_serial==241)
    indexed=source.presentation(frame=1,tolerance_speed=None)
    scene=next(s for s in indexed.scenes() if s.id!=first.scene_id)
    new=source.presentation(scene.id,1,tolerance_speed=None,overlay_mask=4)
    assert new.request_applied and new.segment_epoch>first.segment_epoch
    assert new.tolerance_speed_source_id!=first.tolerance_speed_source_id
    assert new.tolerance_speed_source_id==scene.instance_id
    assert new.tolerance_speed==10
    assert new.tolerance_length==1 and new.tolerance_length_source_id==scene.instance_id
    assert new.tolerance_length_source_id!=first.tolerance_length_source_id
    assert struct.unpack("<2f",new.read(new.tolerance_speed_source_id,new.tolerance_speed_property_id))==(1,10)
    assert first.tolerance_speed==10 and first.capture_key==new.capture_key
    assert first.tolerance_speed_source_id==first.scene_id
