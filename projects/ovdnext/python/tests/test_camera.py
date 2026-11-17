# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera behavior checked against original Qt camera code and native captures."""
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path

import numpy as np
import pytest

from _pvd_native import Session
from ovd_ui.camera import Camera, CameraController, MovementMode, captured_cameras, tracking_center
from reference_data import load

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def original_camera():
    return {row["name"]: row for row in load("camera")}


def test_camera_matches_original_pose_projection_and_navigation(original_camera):
    camera = Camera(near=.0123, tolerance_length=2.)
    control = CameraController(camera)
    control.speed_scale = 2.5
    size = (800, 600)
    def check(name):
        reference = original_camera[name]
        for key, value in (("position", camera.position), ("forward", camera.basis[2]),
                           ("up", camera.basis[1]), ("distance", camera.distance), ("fov", camera.fov)):
            np.testing.assert_allclose(value, reference[key], rtol=3e-5, atol=3e-5, err_msg=name+" "+key)
        np.testing.assert_allclose(camera.view_matrix().ravel(), reference["view"], atol=3e-5, rtol=3e-5, err_msg=name)
        np.testing.assert_allclose(camera.projection(*size).ravel(), reference["projection"], atol=1e-6, rtol=1e-6, err_msg=name)
    camera.look_at([3, 4, 10], [1, 2, 0]);check("initial")
    camera.turn(-20, 15, orbit=True);check("orbit")
    camera.turn(-10, -8);check("local")
    control.key(87, 0, True);control.key(68, 1, True);control.update(.125);check("flight")
    control.key(87, 0, False);control.key(68, 0, False)
    control.mode = MovementMode.THIRD_PERSON
    control.key(87, 0, True);control.key(69, 0, True);control.update(.1);check("world_up")
    control.key(87, 0, False);control.key(69, 0, False)
    control.key(262, 1, True);control.key(265, 1, True);control.update(.05);check("keyboard_look")
    control.stop();control.update(1);check("focus_lost")
    control.scene_length = 5
    camera.set_distance(.1);control.wheel(-1);check("small_dolly")
    camera.set_world_up([0, 0, 1]);check("z_up")
    size = (300, 900);check("portrait")
    camera.left_handed = True;check("left_handed")
    camera.left_handed=False;camera.set_world_up([0,1,0])
    camera.set_rotation(0.,0.);camera.center_at([0,0,0])
    camera.tolerance_length=100.;camera.set_distance(3000000.);check("large_length")
    camera.tolerance_length=1.;check("length_setting_no_motion")
    camera.zoom(1.);check("length_zoom_guard")
    camera.tolerance_length=7.;camera.reset();check("length_reset")


def test_navigation_focus_aliases_fov_and_shift_wheel_are_independent():
    camera = Camera()
    control = CameraController(camera)
    assert not control.key(87, 0, True, active=False)
    assert not control.key(87, 0, True, editing=True)
    assert not control.key(87, 2, True)
    assert not control.moving
    control.key(265, 0, True);control.key(73, 0, True)
    control.key(73, 0, False)
    assert control.moving and control.update(.1)
    assert camera.pitch == pytest.approx(9)
    control.stop();assert not control.update(.1)
    control.key(61, 0, True);assert camera.fov == 59
    control.key(45, 0, True);assert camera.fov == 60
    camera.set_fov(-100);assert camera.fov == 1
    camera.set_fov(1000);assert camera.fov == 179
    position = camera.position.copy()
    control.wheel(1, 1)
    assert control.speed_scale == pytest.approx(10**(3/250))
    np.testing.assert_array_equal(camera.position, position)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError): camera.set_fov(bad)
        with pytest.raises(ValueError): camera.set_distance(bad)


def test_native_captured_camera_and_tracking_are_owned(tmp_path):
    source = Session(str(ROOT / "test/data/apex_destruction.pxd2"), str(tmp_path))
    view = source.presentation(frame=1)
    cameras = captured_cameras(view)
    assert len(cameras) == 1 and cameras[0][0] == 13
    np.testing.assert_array_equal(cameras[0][1], [0, 2, 16])
    np.testing.assert_array_equal(cameras[0][2], [0, 2, 15])
    np.testing.assert_array_equal(cameras[0][3], [0, 1, 0])
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    source.presentation(frame=1)
    actor=unique_instance_id(source,'PxRigidDynamic')
    first = source.presentation(frame=1, inspect_ids=[actor])
    assert captured_cameras(first) == []
    assert tracking_center(first, actor) is None  # Recorded sleeping bodies do not drive tracking.
    later = source.presentation(frame=2, inspect_ids=[actor], overlay_mask=256)
    center = tracking_center(later, actor)
    native_com = later.overlay_snapshot().positions[later.overlay_snapshot().object_ids == actor][0, :2].mean(0)
    np.testing.assert_allclose(center, native_com, rtol=1e-6)
    dead = source.presentation(frame=3, inspect_ids=[actor])
    assert tracking_center(dead, actor) is None
    np.testing.assert_array_equal(tracking_center(later, actor), center)


def test_scene_extent_speed_and_coordinate_target_preserve_original_semantics(tmp_path):
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    view = source.presentation(frame=1)
    camera = Camera()
    camera.reset()
    control = CameraController(camera)
    control.calibrate(view.geometry_snapshot(), view.scenes()[0].instance_id)
    assert control.scene_length == pytest.approx(27.14016342163086)
    assert control.effective_speed == pytest.approx(271.4016342163086)
    assert camera.near == pytest.approx(.02714016342163086)
    before = camera.basis[2].copy()
    camera.center_at([1, 2, 3])
    np.testing.assert_allclose(camera.target, [1, 2, 3])
    np.testing.assert_allclose(camera.basis[2], before)
    eye = camera.position.copy()
    camera.frame_bounds([[9, 19, 29], [11, 21, 31]], center_only=True)
    np.testing.assert_array_equal(camera.position, eye)
    np.testing.assert_allclose(camera.target, [10, 20, 30])


def test_application_navigation_is_scoped_and_releases_after_focus_changes():
    from types import SimpleNamespace
    import omni.ui as ui
    from ovd_ui.application import Application
    app=Application.__new__(Application)
    app.navigation=CameraController(Camera())
    app.window=SimpleNamespace(focused=True)
    app.viewport_hovered=True
    app.viewport_keyboard_focus=True
    app.dirty=False
    app._key_pressed(87,0,True)
    assert app.navigation.moving
    app._window_focused(False)
    assert not app.navigation.moving
    app._key_pressed(87,0,True)
    assert not app.navigation.moving
    app.viewport_hovered=True
    app._key_pressed(87,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert not app.navigation.moving
    app.viewport_keyboard_focus=True
    app._key_pressed(87,0,True)
    app._window_focused(False)
    assert not app.navigation.moving and app.mouse_anchor is None
    app._key_pressed(87,0,False)
    app.viewport_hovered=True
    app.viewport_keyboard_focus=True
    calls=[]
    app.playback=SimpleNamespace(step=lambda *args:calls.append(args))
    app._transport=lambda callback,*args:callback(*args)
    app._key_pressed(262,0,True)
    assert app.navigation.moving and not calls
    app._key_pressed(262,0,False)
    app._key_pressed(262,2,True)
    assert not app.navigation.moving and calls==[(1,)]


def test_orbit_and_local_drag_preserve_distinct_native_pivots():
    camera=Camera()
    camera.look_at([3,4,10],[1,2,0])
    control=CameraController(camera)
    pivot=camera.target.copy()
    control.drag(100,-50,800,600,0)
    np.testing.assert_allclose(camera.target,pivot)
    eye=camera.position.copy()
    control.drag(100,-50,800,600,1)
    np.testing.assert_array_equal(camera.position,eye)
    assert not np.allclose(camera.target,pivot)


def test_application_owned_keyboard_suppresses_only_viewport_navigation():
    from types import SimpleNamespace
    from ovd_ui.application import Application
    app=Application.__new__(Application)
    app.navigation=CameraController(Camera())
    app.window=SimpleNamespace(flags=1<<13,focused=True)
    app.viewport=SimpleNamespace(screen_position_x=100,screen_position_y=100,
                                 computed_width=400,computed_height=300)
    app.tree=SimpleNamespace(screen_position_x=0,screen_position_y=100,
                             computed_width=90,computed_height=300)
    app._workspace_pressed(200,200,0,0)
    assert app.viewport_keyboard_focus and app.window.flags==(1<<13|1<<16)
    app._key_pressed(514,1<<30,True)  # Inspector named arrow and ImGui navigation bit.
    assert app.navigation.update(.1)
    assert app.navigation.camera.yaw==pytest.approx(351)
    app._key_pressed(514,1<<30,False)
    app._workspace_pressed(50,150,0,0)
    assert not app.viewport_keyboard_focus and app.window.flags==1<<13
    assert app.object_keyboard_focus
    actions=[]
    app._object_action=actions.append
    app._key_pressed(72,1<<30,True)
    app._key_pressed(72,(1<<30)|1,True)
    assert actions==["hide","show_all"]
    app._workspace_pressed(50,50,0,0)  # Search field is outside the tree.
    app._key_pressed(72,1<<30,True)
    assert not app.object_keyboard_focus and actions==["hide","show_all"]
    app._key_pressed(568,1<<30,True)  # Text input W belongs to its field.
    assert not app.navigation.moving
    app._workspace_pressed(200,200,0,0)
    app._key_pressed(258,0,True)
    assert not app.viewport_keyboard_focus and app.window.flags==1<<13
