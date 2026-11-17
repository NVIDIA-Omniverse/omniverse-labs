# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured up uses declared native vectors and the original Qt camera rules."""
from pathlib import Path
import struct

import numpy as np
import pytest
from _pvd_native import Session
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.camera import Camera,CameraController,captured_gravity
from test_camera import original_camera
from test_captured_tolerance import writer,capture,view_for
from test_captured_tolerance_length import camera_state,same_state
from test_native_presentation import live,wait_until,HANDSHAKE
from test_application_presentation import app

ROOT=Path(__file__).resolve().parents[2]

def scene_instance(view):
    identity=next(s.instance_id for s in view.scenes() if s.id==view.scene_id)
    return next(i for i in view.instances() if i.id==identity)


@pytest.mark.parametrize("mode",["oblique","z","tiny","small","zero"])
@pytest.mark.parametrize("handed",[False,True])
def test_native_captured_gravity_pose_matches_original_qt(capture,original_camera,mode,handed):
    _,source=capture("gravity-"+mode);view=view_for(source)
    camera=Camera(left_handed=handed,near=.7)
    control=CameraController(camera)
    control.apply_frame(view,PvdDataAdapter(view),reset=True)
    ref=original_camera["gravity_"+mode+("_left" if handed else "_right")]
    for key,value in (("world_up",camera.world_up),("position",camera.position),
            ("up",camera.basis[1]),("forward",camera.basis[2]),("distance",camera.distance)):
        np.testing.assert_allclose(value,ref[key],rtol=3e-5,atol=3e-5,err_msg=key)
    np.testing.assert_allclose(camera.view_matrix().ravel(),ref["view"],rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(camera.projection(300,900).ravel(),ref["projection"],rtol=1e-6,atol=1e-6)
    if mode in ("zero","tiny"):
        assert captured_gravity(view,scene_instance(view)) is None
    elif mode=="oblique":
        np.testing.assert_allclose(camera.world_up,[0,.6,-.8 if handed else .8])


@pytest.mark.parametrize("mode",["wrong-type","variable","pair","four","nan","infinity","missing"])
def test_malformed_gravity_retains_capture_local_default_up(capture,mode):
    _,source=capture("gravity-"+mode);view=view_for(source)
    camera=Camera(left_handed=True);camera.set_world_up([1,0,0])
    control=CameraController(camera)
    assert captured_gravity(view,scene_instance(view)) is None
    control.apply_frame(view,PvdDataAdapter(view),reset=True)
    np.testing.assert_array_equal(camera.world_up,[0,1,0])
    assert np.isfinite(camera.view_matrix()).all()


@pytest.mark.parametrize("mode",["short","empty"])
def test_fixed_truncated_gravity_is_rejected_by_native_loader(capture,mode):
    with pytest.raises(RuntimeError,match="payload does not match"):
        capture("gravity-"+mode)


def test_frame_and_scene_switch_preserve_manual_camera_until_capture_reset(capture):
    _,source=capture("gravity-oblique")
    first=view_for(source);second=view_for(source,frame=2);other=view_for(source,1)
    camera=Camera(left_handed=True);control=CameraController(camera)
    control.apply_frame(first,PvdDataAdapter(first),reset=True)
    camera.set_world_up([1,0,0]);camera.set_rotation(29,-11);camera.near=.123
    control.speed_scale=42;before=camera_state(camera,control)
    for view in (second,other):
        control.apply_frame(view,PvdDataAdapter(view))
        same_state(before,camera_state(camera,control))
    # Explicit new capture initialization uses its selected scene, even when
    # another scene with actors appears first in the native list.
    control.apply_frame(other,PvdDataAdapter(other),reset=True)
    np.testing.assert_allclose(camera.world_up,[.8,0,-.6])
    np.testing.assert_array_equal(captured_gravity(first,scene_instance(first)),[0,-3,-4])
    np.testing.assert_array_equal(captured_gravity(second,scene_instance(second)),[-3,-4,0])


@pytest.mark.parametrize("filename,expected",[("test_scene.ovd",[0,1,0]),
    ("cloth_physx34.pxd2",[0,1,0]),("ue4_shooter.pxd2",[0,0,1])])
def test_real_ovd_and_legacy_scalar_vector_schema(filename,expected,tmp_path):
    source=Session(str(ROOT/"test/data"/filename),str(tmp_path))
    view=source.presentation(frame=1);camera=Camera();control=CameraController(camera)
    assert captured_gravity(view,scene_instance(view)) is not None
    control.apply_frame(view,PvdDataAdapter(view),reset=True)
    np.testing.assert_array_equal(camera.world_up,expected)


def test_dead_scene_without_owned_gravity_uses_default_instead_of_reading_metadata(tmp_path):
    source=Session(str(ROOT/"test/data/test_scene.ovd"),str(tmp_path))
    first=source.presentation(frame=1);dead=source.presentation(frame=3)
    instance=scene_instance(dead)
    assert not instance.alive and captured_gravity(dead,instance) is None
    camera=Camera();camera.set_world_up([1,0,0]);control=CameraController(camera)
    control.apply_frame(dead,PvdDataAdapter(dead),reset=True)
    np.testing.assert_array_equal(camera.world_up,[0,1,0])
    assert captured_gravity(first,scene_instance(first)) is not None


def test_application_live_partial_gravity_cannot_replace_manual_up(app,capture,live):
    path,_=capture("gravity-oblique");payload=path.read_bytes();source,connection=live
    marker=struct.pack("<BQQ",12,1,1);boundary=payload.index(marker)+len(marker)
    next_stop=payload.index(struct.pack("<BQQ",12,1,2))
    connection.sendall(HANDSHAKE+payload[:boundary]);wait_until(lambda:source.info().completed_serial==1)
    source.refresh_live();camera=Camera(left_handed=True);app.navigation=CameraController(camera)
    assert app._display(path,source)
    selected=next(s for s in app.scenes if s.stream_id==10)
    assert app._refresh(scene_id=selected.id,frame=1)
    camera.set_world_up([1,0,0]);camera.set_rotation(41,-19);camera.near=.234
    before=camera_state(camera,app.navigation);owned=app.presentation
    connection.sendall(payload[boundary:next_stop]);wait_until(lambda:source.info().unpublished_changes)
    assert not app._refresh(frame=2)
    assert app.presentation is owned
    same_state(before,camera_state(camera,app.navigation))
    connection.sendall(payload[next_stop:]);wait_until(lambda:source.info().completed_serial==3)
    assert app._refresh(frame=2)
    same_state(before,camera_state(camera,app.navigation))
    np.testing.assert_array_equal(captured_gravity(owned,scene_instance(owned)),[0,-3,-4])
    np.testing.assert_array_equal(captured_gravity(app.presentation,scene_instance(app.presentation)),[-3,-4,0])
