# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accepted visual reset over real owned frames, preferences and OVUI models."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.camera import Camera, CameraController, MovementMode, tracking_center
from ovd_ui.general_settings import GeneralSettings
from ovd_ui.interaction import TransparencyMode
from ovd_ui.render_settings import DisplayMode, RenderingSettings
from ovd_ui.settings import Preferences
from ovd_ui.visual_reset import VisualReset, NEAR, GIZMO_SCALE, SPEED_SCALE, RENDER_FIELDS
from ovd_ui.widgets.camera_view import CameraView
from test_application_presentation import (app as base_app, open_capture, live,
    HANDSHAKE, PAYLOAD, FRAME2_END, wait_until)
from test_application_general_settings import ObservedSession
from test_general_settings import make_view
from test_exact_shortcuts import command_probe


class OverlayModes:
    """Only GPU filtering is a sink; native extraction remains authoritative."""
    def __init__(self, snapshot=None):
        self.snapshot, self.modes = snapshot, {}
    def set_mode(self, bit, mode):self.modes[bit] = mode


@pytest.fixture
def app(base_app, tmp_path):
    app = base_app
    initial = GeneralSettings(rendering=RenderingSettings(face_culling=1,wireframe=DisplayMode.UNSELECTED,
        sleeping_objects=DisplayMode.NONE,trigger_shapes=DisplayMode.SELECTED,particle_size=2.3,
        geometry_scale=2.5,plane_scale=3.,anti_aliasing=True,background_mode="custom",background_color="#123456"))
    view, preferences, _ = make_view(tmp_path, initial)
    app.workflow.settings = preferences
    app.render_settings_view, app.render_settings = view, initial.rendering
    view.on_apply = lambda value: app._render_settings_changed(value.rendering)
    app.navigation = CameraController(Camera())
    app.renderer.set_settings = lambda settings: setattr(app.renderer,"settings",settings)
    app.renderer.set_overlays = lambda snapshot: setattr(app.renderer,"overlays",OverlayModes(snapshot))
    app.renderer.overlays = OverlayModes()
    app.overlay_modes = {bit:DisplayMode.SELECTED for bit in (1,2,4,8,32)}
    app.overlay_mask, app.gizmo_scale = 511, 2.
    yield app
    app.selection_subscription = None


def opened(app, tmp_path, name="multi_geom.ovd"):
    path, native = open_capture(tmp_path, name)
    observed = ObservedSession(native)
    assert app._display(path, observed) and app.seek(2,snap=False)
    identity = int(app.presentation.geometry_snapshot().object_ids[0])
    app.selection.publish([f"/objects/{identity}"],source="test")
    assert app._refresh()
    observed.calls.clear()
    return observed


def state(app):
    return (app.session.capture_key,app.presentation.scene_id,app.scene_id,app.frame,
            app.playback.frame,tuple(app.selection.get_snapshot().paths()),
            tuple(app.interaction.hidden),tuple(app.interaction.transparent))


def camera_controls(app):
    view = CameraView(app.navigation,lambda:pytest.fail("Reset model syncing called a camera action"),
        frame_all=lambda:None,frame_selected=lambda:None,center_selected=lambda:None,
        track_selected=lambda:None,reset=lambda:None)
    for key in ("mode","source","up"):
        model = ui.SimpleIntModel()
        setattr(view,key,NS(model=NS(get_item_value_model=lambda m=model:m)))
    for key in ("near","speed","fov"):
        setattr(view,key,NS(model=ui.SimpleFloatModel()))
    view.near_slider=NS(model=ui.SimpleIntModel())
    view.near_slider.model.add_value_changed_fn(view._near_slider_changed)
    view.near.model.add_value_changed_fn(view._near_field_changed)
    view.handed=NS(model=ui.SimpleBoolModel())
    view.coordinates=[NS(model=ui.SimpleFloatModel()) for _ in range(3)]
    view.window=NS(visible=True)
    view.status=NS(text="")
    view._sync()
    app.camera_view=view
    return view


def test_reset_is_one_durable_transaction_and_one_owned_same_frame_publication(app,tmp_path):
    source=opened(app,tmp_path)
    app.interaction.set_visible([25],False)
    app.interaction.set_transparent([22],True)
    app.interaction.set_transparency(TransparencyMode.TAGGED,.73)
    camera=app.navigation.camera
    camera.set_world_up((0,0,1));camera.left_handed=True;camera.set_rotation(17,23)
    camera.fov=47
    old_target=camera.target.copy()
    before=state(app);old_view=app.presentation
    original=app.render_settings
    prefs=deepcopy(app.workflow.settings.values)
    writes=[];save=app.workflow.settings.set_many
    def observe(values):
        # Persistence precedes every live mutation.
        assert app.render_settings==original and app.presentation is old_view
        assert camera.left_handed and app.gizmo_scale==2.
        writes.append(deepcopy(values));save(values)
    app.workflow.settings.set_many=observe
    assert app.reset_visual_settings()
    assert writes==[{"rendering":app.render_settings.to_preferences(),"joint_scale":1.,
                     "viewport_display":app._viewport_settings().to_dict()}]
    assert {k:v for k,v in app.workflow.settings.values.items() if k not in ("rendering","joint_scale","viewport_display")}=={
        k:v for k,v in prefs.items() if k not in ("rendering","joint_scale","viewport_display")}
    assert GeneralSettings.from_preferences(Preferences(app.workflow.settings.path)).rendering==app.render_settings
    assert app.render_settings==replace(original,wireframe=DisplayMode.NONE,
        sleeping_objects=DisplayMode.ALL,trigger_shapes=DisplayMode.ALL,particle_size=5.)
    assert camera is app.navigation.camera and camera.fov==47 and camera.near==NEAR
    assert not camera.left_handed
    np.testing.assert_allclose(camera.target,old_target,atol=1e-12)
    assert app.navigation.mode==MovementMode.FIRST_PERSON and app.navigation.speed_scale==SPEED_SCALE
    assert app.overlay_mask==16|8|128|256 and app.gizmo_scale==GIZMO_SCALE
    assert app.overlay_modes[8]==DisplayMode.SELECTED
    assert all(app.overlay_modes[bit]==DisplayMode.NONE for bit in (1,2,4,32))
    assert app.interaction.transparency_mode==TransparencyMode.NONE and app.interaction.transparency_amount==.5
    assert state(app)==before and app.presentation is old_view and source.calls==[]
    assert app.presentation_dirty and app._refresh()
    assert len(source.calls)==1 and state(app)==before
    args,kwargs=source.calls[0]
    assert args==(app.scene_id,2) and kwargs["overlay_mask"]==app.overlay_mask
    assert kwargs["gizmo_scale"]==GIZMO_SCALE
    assert app.presentation.overlay_snapshot().mask==app.overlay_mask
    assert old_view.overlay_snapshot().mask==511
    source.calls.clear()
    app.workflow.settings.set_many=save
    view=app.presentation
    pose=deepcopy(app.navigation.state())
    assert app.reset_visual_settings()
    assert app.presentation is view and not app.presentation_dirty and source.calls==[]
    assert app.navigation.state()==pose


def test_partial_live_reset_retains_complete_owned_view_until_safe_same_frame_publication(app,live):
    native,connection=live
    source=ObservedSession(native)
    app.workflow.live_session=source
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END])
    wait_until(lambda:native.info().completed_serial==2)
    assert app._display(None,source) and app.seek(2,snap=False)
    app.selection.publish(["/objects/1"],source="test")
    assert app._refresh()
    before=state(app);old_view=app.presentation
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda:native.info().recording_size==len(PAYLOAD)
               and native.info().unpublished_changes)
    source.calls.clear()
    assert app.reset_visual_settings()
    assert not app._refresh()
    assert len(source.calls)==1 and app.presentation is old_view and state(app)==before
    assert app.presentation_dirty and app.pending_presentation is None  # No seek was introduced.
    native.stop_live()
    assert app._refresh()
    assert len(source.calls)==2 and state(app)==before
    assert app.presentation.ready and app.presentation.request_applied
    assert app.presentation.overlay_snapshot().mask==app.overlay_mask
    assert old_view.overlay_snapshot().mask==511


def test_failed_replace_preserves_owned_state_all_drafts_and_preferences(app,tmp_path,monkeypatch):
    source=opened(app,tmp_path)
    view=app.render_settings_view
    camera_view=camera_controls(app)
    camera_view.fov.model.set_value(79)
    view.controls["background_color"].model.set_value("unfinished")
    view._status_error=True;view.status.text="Existing validation error"
    before=(state(app),deepcopy(app.navigation.state()),app.render_settings,app.overlay_mask,app.gizmo_scale)
    file_bytes=app.workflow.settings.path.read_bytes();values=deepcopy(app.workflow.settings.values)
    import ovd_ui.settings as preferences_module
    monkeypatch.setattr(preferences_module.os,"replace",lambda *_:(_ for _ in ()).throw(OSError("read-only target")))
    assert not app.reset_visual_settings()
    assert (state(app),app.navigation.state(),app.render_settings,app.overlay_mask,app.gizmo_scale)==before
    assert app.workflow.settings.values==values and app.workflow.settings.path.read_bytes()==file_bytes
    assert view.controls["background_color"].model.as_string=="unfinished"
    assert view._status_error and view.status.text=="Existing validation error"
    assert camera_view.fov.model.as_float==79 and source.calls==[] and not app.presentation_dirty


def test_built_views_keep_unrelated_drafts_and_apply_cannot_restore_old_reset_fields(app,tmp_path):
    opened(app,tmp_path)
    camera_view=camera_controls(app)
    camera_view.fov.model.set_value(79)
    camera_view.source.model.get_item_value_model().set_value(1)
    camera_view.coordinates[0].model.set_value(123)
    camera_view._status_error=True;camera_view.status.text="Unfinished camera draft"
    view=app.render_settings_view
    view.controls["geometry_scale"].model.set_value(4.5)
    view.controls["background_color"].model.set_value("unfinished")
    view.property_controls["float_precision"].model.set_value("25")
    view._status_error=True;view.status.text="Unfinished general draft"
    # Real reactive models must not schedule a publication while synchronizing.
    # Embedded Display controls must synchronize even with no floating window.
    app.view_window=None;app.syncing_view_options=False
    app.view_option_controls={key:ui.SimpleFloatModel(9) for key in
        ("transparency","amount","gizmo_scale",1,2,4,8,16,32,64,128,256)}
    for model in app.view_option_controls.values():
        model.add_value_changed_fn(lambda _m:app._view_option_changed(lambda:pytest.fail("recursive view action")))
    assert app.reset_visual_settings()
    assert camera_view.fov.model.as_float==79 and camera_view.source.model.get_item_value_model().as_int==1
    assert camera_view.coordinates[0].model.as_float==123
    assert camera_view._status_error and camera_view.status.text=="Unfinished camera draft"
    assert camera_view.near.model.as_float==NEAR and camera_view.near_slider.model.as_int==57
    assert app.view_option_controls["gizmo_scale"].as_float==GIZMO_SCALE
    assert all(app.view_option_controls[bit].as_float==9 for bit in (8,16,128,256))
    assert view.controls["geometry_scale"].model.as_float==4.5
    assert view.controls["background_color"].model.as_string=="unfinished"
    assert view.property_controls["float_precision"].model.as_string=="25"
    assert view._status_error and view.status.text=="Unfinished general draft"
    for key in RENDER_FIELDS:
        model=view.controls[key].model
        if hasattr(model,"get_item_value_model"):model=model.get_item_value_model()
        actual=model.as_float if key=="particle_size" else model.as_int
        assert actual==getattr(app.render_settings,key)
    view.controls["background_color"].model.set_value("#123456")
    assert view._apply()
    assert app.render_settings.wireframe==DisplayMode.NONE and app.render_settings.particle_size==5
    assert app.render_settings.geometry_scale==4.5


def test_unbuilt_camera_adopts_effective_near_and_original_slider_position_on_first_sync(app,tmp_path):
    opened(app,tmp_path)
    assert app.reset_visual_settings()
    view=camera_controls(app)
    assert view.near.model.as_float==NEAR and view.near_slider.model.as_int==57


def test_tracking_keeps_existing_offset_and_resumes_it_on_next_owned_frame(app,tmp_path):
    opened(app,tmp_path,"test_scene.ovd")
    controller=app.navigation;camera=controller.camera
    identity=int(app.presentation.geometry_snapshot().object_ids[0])
    camera.set_world_up((0,0,1));camera.set_rotation(17,23)
    controller.tracking_id=identity;controller.tracking_offset=None
    controller.apply_frame(app.presentation,app.adapter)
    old_offset=controller.tracking_offset.copy()
    controller.captured=True  # This fixture has no recorded cameras to apply.
    assert app.reset_visual_settings()
    assert controller.tracking_id==identity and controller.captured
    np.testing.assert_array_equal(controller.tracking_offset,old_offset)
    immediate=camera.position.copy()
    assert app.seek(3,snap=False)  # Actor death does not consume the intent.
    assert controller.tracking_id==identity
    assert app.seek(2,snap=False)
    center=tracking_center(app.presentation,identity,app.adapter)
    np.testing.assert_allclose(camera.position,center+old_offset,atol=1e-12)
    assert not np.allclose(immediate,camera.position)
    assert controller.captured and controller.tracking_id==identity


def test_reset_shortcut_respects_other_window_ownership():
    view,calls=command_probe(viewport=False)
    view.window.focused=False
    view._key_pressed(82,3|ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True)
    assert calls==[]


@pytest.mark.parametrize("flags",(0,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD))
def test_reset_window_shortcut_also_accepts_toolbar_and_main_editor_focus(flags):
    view,calls=command_probe(viewport=False)
    view._key_pressed(82,3|flags,True)
    assert calls==["reset_visual_settings"]
