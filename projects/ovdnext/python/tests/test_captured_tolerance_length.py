# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current captured units affect later camera actions, never the adopted pose."""
import gc
from pathlib import Path
import struct

import numpy as np
import pytest
from _pvd_native import Session
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.camera import Camera,CameraController
from test_captured_tolerance import writer,capture,view_for
from test_native_presentation import live,wait_until,HANDSHAKE
from test_application_presentation import app

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("mode,lengths,reason",[("scene",[10,30],"scene"),("owner",[50,70],"physics-owner"),("single",[50,50],"single-physics")])
def test_length_follows_selected_owned_scene_and_frame(capture,mode,lengths,reason):
    _,source=capture(mode)
    first=view_for(source);second=view_for(source,1);later=view_for(source,frame=2)
    with pytest.raises(AttributeError):first.tolerance_length=42
    assert first.tolerance_length==lengths[0] and second.tolerance_length==lengths[1]
    assert later.tolerance_length==lengths[0]*2
    assert first.tolerance_length_resolution==reason
    assert first.tolerance_length_source_id==first.tolerance_speed_source_id
    assert first.tolerance_length_property_id==first.tolerance_speed_property_id
    assert struct.unpack("<2f",first.read(first.tolerance_length_source_id,first.tolerance_length_property_id))==(lengths[0],first.tolerance_speed)
    restored=view_for(source)
    assert (restored.tolerance_length,restored.tolerance_length_source_id)==(first.tolerance_length,first.tolerance_length_source_id)
    del source,second,later,restored;gc.collect()
    assert first.tolerance_length==lengths[0]

@pytest.mark.parametrize("mode",["missing","ambiguous","conflicting","wrong-type","variable","bad-length","length-negative","length-nan","length-infinity"])
def test_invalid_or_ambiguous_length_never_reuses_a_previous_value(capture,mode):
    _,valid=capture("scene");assert view_for(valid).tolerance_length==10
    _,source=capture(mode);view=view_for(source)
    assert view.tolerance_length==1 and view.tolerance_length_source_id==0
    assert view.tolerance_length_property_id==-1
    assert view.tolerance_length_resolution in ("missing","ambiguous-physics","invalid")

@pytest.mark.parametrize("mode",["zero","negative","nan","infinity"])
def test_valid_length_is_independent_of_invalid_speed(capture,mode):
    _,source=capture(mode);view=view_for(source)
    assert view.tolerance_length==10 and view.tolerance_length_resolution=="scene"
    assert view.tolerance_speed==1 and view.tolerance_speed_resolution=="invalid"

@pytest.mark.parametrize("filename,length",[("test_scene.ovd",1),("cloth_physx34.pxd2",1),("ue4_shooter.pxd2",100)])
def test_length_is_resolved_even_with_explicit_default_speed(tmp_path,filename,length):
    source=Session(str(ROOT/"test/data"/filename),str(tmp_path))
    for kwargs,speed in [({},10),({"tolerance_speed":3},3)]:
        view=source.presentation(frame=1,**kwargs)
        assert view.tolerance_length==length and view.tolerance_length_source_id>0
        assert view.tolerance_speed==speed and view.tolerance_speed_resolution=="explicit"
        assert struct.unpack("<f",view.read(view.tolerance_length_source_id,view.tolerance_length_property_id)[:4])[0]==length


def camera_state(camera,controller):
    return (camera.position.copy(),camera.world_up.copy(),camera.distance,camera.yaw,camera.pitch,
        camera.near,camera.fov,controller.speed_scale,controller.scene_length)

def same_state(before,after):
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)


def test_frame_adoption_preserves_camera_but_later_zoom_and_reset_use_current_units(capture):
    _,source=capture("scene");first=view_for(source)
    camera=Camera(tolerance_length=1000,distance=100_000_000,near=.123,fov=73)
    camera.set_world_up([0,0,1]);camera.set_rotation(47,-13)
    controller=CameraController(camera);controller.speed_scale=77;controller.scene_length=8
    before=camera_state(camera,controller)
    controller.apply_frame(first,PvdDataAdapter(first))
    assert camera.tolerance_length==10
    same_state(before,camera_state(camera,controller))
    # Qt only applies its new tolerance-dependent bound when setOrbitDistance
    # is next called (zoom); loading a setting does not clamp the old distance.
    target=camera.target.copy();camera.zoom(1)
    assert camera.distance==300000*10**2
    np.testing.assert_allclose(camera.target,target)
    second=view_for(source,1);before=camera_state(camera,controller)
    controller.apply_frame(second,PvdDataAdapter(second))
    assert camera.tolerance_length==30
    same_state(before,camera_state(camera,controller))
    camera.reset()
    assert camera.distance==30*30
    np.testing.assert_array_equal(camera.world_up,[0,0,1])
    assert camera.near==.123 and controller.speed_scale==77


def test_application_scene_switch_updates_units_without_reframing_manual_camera(app,capture):
    path,source=capture("scene");camera=Camera();controller=CameraController(camera)
    app.navigation=controller
    assert app._display(path,source)
    camera.set_world_up([1,0,0]);camera.set_rotation(37,-12);camera.near=.321
    controller.speed_scale=17
    before=camera_state(camera,controller)
    other=next(s for s in app.scenes if s.id!=app.scene_id)
    assert app._refresh(scene_id=other.id,frame=1)
    assert camera.tolerance_length==(30 if other.stream_id==20 else 10)
    same_state(before,camera_state(camera,controller))
    app.reset_camera()
    assert camera.distance==30*camera.tolerance_length
    assert camera.near==.321
    np.testing.assert_array_equal(camera.world_up,[1,0,0])
    path,missing=capture("missing")
    assert app._display(path,missing)
    assert camera.tolerance_length==1 and app.presentation.tolerance_length_source_id==0


def test_live_partial_length_and_dead_physics_keep_owned_values(capture,live):
    path,_=capture("owner");source,connection=live;payload=path.read_bytes()
    marker=struct.pack("<BQQ",12,1,1);first_end=payload.index(marker)+len(marker)
    second_stop=payload.index(struct.pack("<BQQ",12,1,2))
    connection.sendall(HANDSHAKE+payload[:first_end]);wait_until(lambda:source.info().completed_serial==1)
    source.refresh_live();first=view_for(source)
    assert first.tolerance_length==50
    connection.sendall(payload[first_end:second_stop]);wait_until(lambda:source.info().unpublished_changes)
    old=source.presentation(first.scene_id,2,tolerance_speed=None)
    assert not old.request_applied and old.generation==first.generation
    assert (old.tolerance_length,old.tolerance_length_source_id)==(50,first.tolerance_length_source_id)
    connection.sendall(payload[second_stop:]);wait_until(lambda:source.info().completed_serial==3)
    later=view_for(source,frame=2)
    assert later.tolerance_length==100 and first.tolerance_length==50
    _,dead=capture("dead");alive=view_for(dead);gone=view_for(dead,frame=2)
    assert alive.tolerance_length==50 and gone.tolerance_length==1
    assert gone.tolerance_length_source_id==0
