# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable choices over real native publications and public OVUI value models."""
from copy import deepcopy
import os
import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.application import Application
from ovd_ui.interaction import TransparencyMode
from ovd_ui.overlays import overlay_visibility
from ovd_ui.render_settings import DisplayMode
from ovd_ui.settings import Preferences
from ovd_ui.viewport_settings import ViewportDisplaySettings
from ovd_ui.visual_reset import GIZMO_SCALE
from test_application_presentation import app as base_app, open_capture, live, HANDSHAKE, PAYLOAD, FRAME2_END, wait_until
from test_application_general_settings import ObservedSession
from test_visual_reset import app, opened, state, camera_controls


PREFERENCE_ENVIRONMENT = "LOCALAPPDATA" if os.name == "nt" else "XDG_DATA_HOME"


def defaults(app):
    assert app._set_viewport_settings(**ViewportDisplaySettings().to_dict())
    for bit in (8,128,256):app._set_overlay(bit,False)
    assert app._refresh()


def test_real_constructor_restores_nine_choices_without_capture_ids(tmp_path,monkeypatch):
    monkeypatch.setenv(PREFERENCE_ENVIRONMENT,str(tmp_path))
    first=Application()
    try:
        assert first._viewport_settings()==ViewportDisplaySettings()
        assert first._set_overlay_mode(1,DisplayMode.SELECTED)
        assert first._set_overlay_mode(2,DisplayMode.UNSELECTED)
        assert first._set_overlay_mode(4,DisplayMode.ALL)
        assert first._set_overlay(16,True) and first._set_overlay(32,True) and first._set_overlay(64,True)
        assert first._set_transparency(TransparencyMode.TAGGED,.75)
        assert first._set_gizmo_scale(3.25)
        expected=first._viewport_settings()
        # Session-only extensions/IDs never enter the new block.
        first._set_overlay(128,True)
        first.interaction.hidden.add(33)
        path=first.workflow.settings.path
    finally:first.close()
    restored=Application()
    try:
        assert restored._viewport_settings()==expected
        assert not restored.overlay_mask & (8|128|256)
        assert not restored.interaction.hidden and not restored.interaction.sticky
        assert restored.joint_calibration is None and restored.presentation is None
        assert ViewportDisplaySettings.from_preferences(Preferences(path))==expected
        assert set(Preferences(path).get('viewport_display'))==set(expected.to_dict())
    finally:restored.close()


def test_malformed_preferences_leave_real_constructor_at_existing_defaults(tmp_path,monkeypatch):
    monkeypatch.setenv(PREFERENCE_ENVIRONMENT,str(tmp_path))
    preferences=Preferences();preferences.set('viewport_display',{'bounds':True,'gizmo_scale':float('inf')})
    before=preferences.path.read_bytes()
    application=Application()
    try:
        assert application._viewport_settings()==ViewportDisplaySettings()
        assert preferences.path.read_bytes()==before
    finally:application.close()


def test_enable_changes_coalesce_but_modes_and_transparency_use_owned_rows(app,tmp_path):
    from capture_semantics import dynamic_geometry_owner
    source=opened(app,tmp_path)
    identity=dynamic_geometry_owner(source,2,app.frame)
    app.selection.publish([f'/objects/{identity}'],source='test')
    assert app._refresh()
    defaults(app);source.calls.clear()
    before=app.presentation;base=app.joint_calibration
    assert app._set_overlay(1,True)
    assert app._set_overlay_mode(2,DisplayMode.SELECTED)
    assert app._set_overlay(16,True)
    assert app._set_gizmo_scale(2)
    assert source.calls==[] and app.presentation_dirty
    assert app._refresh() and len(source.calls)==1
    owned=app.presentation;source.calls.clear()
    assert owned is not before and app.joint_calibration is base
    assert app._set_overlay_mode(1,DisplayMode.UNSELECTED)
    assert app._set_transparency(TransparencyMode.SELECTED,.75)
    assert not app.presentation_dirty and source.calls==[] and app.presentation is owned
    geometry=owned.geometry_snapshot()
    selected=np.asarray(geometry.object_ids)==identity
    np.testing.assert_array_equal(app.renderer.instance_alpha[selected],np.full(selected.sum(),.25,dtype='f4'))
    assert not np.any(app.renderer.instance_alpha[~selected]<1)
    overlay=owned.overlay_snapshot()
    mask=overlay_visibility(overlay,app.renderer.scene_id,selected={identity},modes=app.overlay_modes)
    assert not np.any(mask & (overlay.categories==1) & (overlay.object_ids==identity))
    assert np.count_nonzero(mask & (overlay.categories==2))==3
    prefs=Preferences(app.workflow.settings.path)
    assert prefs.get('viewport_display')['transparency_amount']==.75  # UI amount, not opacity .25.
    before_bytes=app.workflow.settings.path.read_bytes();app.dirty=False
    assert app._set_transparency(TransparencyMode.SELECTED,.75)
    assert not app.dirty and app.workflow.settings.path.read_bytes()==before_bytes


def test_existing_generic_transparency_entrypoint_cannot_bypass_persistence(app,tmp_path):
    source=opened(app,tmp_path);old=app.presentation
    assert app._interaction_action(app.interaction.set_transparency,TransparencyMode.ALL,.2)
    assert Preferences(app.workflow.settings.path).get('viewport_display')['transparency_amount']==.2
    np.testing.assert_allclose(app.renderer.instance_alpha,.8)
    assert app.renderer.all_transparent and app.presentation is old and source.calls==[]


def test_scale_with_no_enabled_category_is_durable_without_optional_native_work(app,tmp_path):
    source=opened(app,tmp_path);defaults(app);source.calls.clear()
    before=app.presentation;base=app.joint_calibration
    assert app._set_gizmo_scale(7.)
    assert app.gizmo_scale==7. and app.dirty and not app.presentation_dirty
    assert source.calls==[] and app.presentation is before and app.joint_calibration is base
    assert Preferences(app.workflow.settings.path).get('viewport_display')['gizmo_scale']==7.
    assert app._set_overlay(64,True) and app._refresh()
    assert len(source.calls)==1 and source.calls[0][1]['gizmo_scale']==7.


def test_family_scale_refreshes_only_enabled_families_and_preserves_other_sizes(app,tmp_path):
    source=opened(app,tmp_path);defaults(app);source.calls.clear()
    assert app._set_viewport_settings(linear_velocity_scale=7.,com_scale=.5)
    assert not app.presentation_dirty and source.calls==[]
    assert app._set_overlay_mode(2,1) and app._set_overlay_mode(256,1) and app._refresh()
    assert len(source.calls)==1
    assert source.calls[0][1]['family_scales']=={2:1.,4:7.,8:1.,32:1.,256:.5}
    old=app.presentation.overlay_snapshot();source.calls.clear()
    assert app._set_viewport_settings(axes_scale=2.) and app.presentation_dirty
    assert app._refresh() and len(source.calls)==1
    current=app.presentation.overlay_snapshot()
    axes=current.categories==2
    np.testing.assert_array_equal(current.positions[axes,0],old.positions[axes,0])
    np.testing.assert_allclose(current.positions[axes,1]-current.positions[axes,0],
        2*(old.positions[axes,1]-old.positions[axes,0]),atol=1e-6)
    np.testing.assert_array_equal(current.positions[~axes],old.positions[~axes])
    source.calls.clear()
    for bit in (16,256):
        assert app._set_overlay_mode(bit,1) and app._refresh()
        source.calls.clear();owned=app.presentation
        for mode in (2,3,1):assert app._set_overlay_mode(bit,mode)
        assert app.presentation is owned and not app.presentation_dirty and source.calls==[]
    saved=Preferences(app.workflow.settings.path)
    assert ViewportDisplaySettings.from_preferences(saved)==app._viewport_settings()


def test_master_refreshes_enabled_gizmos_without_changing_family_factors_or_geometry(app,tmp_path):
    source=opened(app,tmp_path);defaults(app);source.calls.clear()
    assert app._set_viewport_settings(master_scale=2.)
    assert source.calls==[] and not app.presentation_dirty
    assert app._set_overlay_mode(2,1) and app._set_overlay_mode(16,1) and app._refresh()
    old=app.presentation;factors=app.overlay_scales.copy();base=app.joint_calibration
    source.calls.clear()
    assert app._set_viewport_settings(master_scale=4.) and app.presentation_dirty
    assert app._refresh() and len(source.calls)==1
    assert source.calls[0][1]['master_scale']==4.
    assert source.calls[0][1]['family_scales']==factors==app.overlay_scales
    current=app.presentation.overlay_snapshot();before=old.overlay_snapshot()
    assert current.master_scale==4. and old.overlay_snapshot().master_scale==2.
    np.testing.assert_array_equal(current.positions[:,0],before.positions[:,0])
    np.testing.assert_allclose(current.positions[:,1]-current.positions[:,0],
        2*(before.positions[:,1]-before.positions[:,0]),atol=2e-6)
    np.testing.assert_array_equal(app.presentation.geometry_snapshot().transforms,old.geometry_snapshot().transforms)
    assert app.joint_calibration is base
    assert app.reset_visual_settings() and app._viewport_settings().master_scale==1.


def test_master_rejects_raw_joint_overflow_before_saving_even_without_a_capture(tmp_path,monkeypatch):
    monkeypatch.setenv(PREFERENCE_ENVIRONMENT,str(tmp_path))
    app=Application()
    app.status_label=ui.Label('')
    try:
        assert app._set_joint_scale(2.)
        before=app.workflow.settings.path.read_bytes()
        assert not app._set_viewport_settings(master_scale=2e38)
        assert app.master_scale==1. and app.workflow.settings.path.read_bytes()==before
    finally:
        app.status_label.destroy()
        app.close()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),10**400,True])
def test_invalid_native_gizmo_request_never_replaces_preferences_or_owned_view(app,tmp_path,value):
    source=opened(app,tmp_path);before=app.presentation
    saved=app.workflow.settings.path.read_bytes();old=app.gizmo_scale
    app.view_option_controls={'gizmo_scale':ui.SimpleFloatModel(19),
                              'amount':ui.SimpleFloatModel(17)}
    assert not app._set_gizmo_scale(value)
    assert app.gizmo_scale==old and app.view_option_controls['gizmo_scale'].as_float==old
    # An unrelated public model has not participated in this attempted action.
    assert app.view_option_controls['amount'].as_float==17
    assert app.presentation is before and source.calls==[] and not app.presentation_dirty
    assert app.workflow.settings.path.read_bytes()==saved


def test_failed_durable_changes_restore_only_attempted_models_and_keep_drafts(app,tmp_path,monkeypatch):
    source=opened(app,tmp_path);before_view=app.presentation;before_state=state(app);base=app.joint_calibration
    preferences=app.workflow.settings;bytes_before=preferences.path.read_bytes()
    accepted=app._viewport_settings();before_values=deepcopy(preferences.values)
    camera=camera_controls(app);camera.fov.model.set_value(79)
    general=app.render_settings_view;general.controls['background_color'].model.set_value('unfinished')
    general._status_error=True;general.status.text='Keep this draft error'
    app.syncing_view_options=False
    controls={1:ui.SimpleIntModel(accepted.bounds),32:ui.SimpleIntModel(accepted.contacts),
        'amount':ui.SimpleFloatModel(accepted.transparency_amount*100),
        'gizmo_scale':ui.SimpleFloatModel(accepted.gizmo_scale)}
    app.view_option_controls=controls
    controls[1].add_value_changed_fn(lambda m:app._view_option_changed(app._set_overlay_mode,1,m.as_int))
    controls[32].add_value_changed_fn(lambda m:app._view_option_changed(app._set_overlay_mode,32,m.as_int))
    controls['amount'].add_value_changed_fn(lambda m:app._view_option_changed(app._set_viewport_settings,transparency_amount=m.as_float/100))
    controls['gizmo_scale'].add_value_changed_fn(lambda m:app._view_option_changed(app._set_gizmo_scale,m.as_float))
    import ovd_ui.settings as module
    monkeypatch.setattr(module.os,'replace',lambda *_:(_ for _ in ()).throw(OSError('disk denied')))
    for key,value in [(1,1),(32,0),('amount',75),('gizmo_scale',19)]:
        controls[key].set_value(value)
        assert app._viewport_settings()==accepted and app.presentation is before_view
        assert app.joint_calibration is base and state(app)==before_state and source.calls==[]
        assert not app.presentation_dirty and not app.syncing_view_options
        assert preferences.values==before_values and preferences.path.read_bytes()==bytes_before
        assert controls[1].as_int==accepted.bounds and controls[32].as_int==accepted.contacts
        assert controls['amount'].as_float==accepted.transparency_amount*100
        assert controls['gizmo_scale'].as_float==accepted.gizmo_scale
        assert camera.fov.model.as_float==79 and general.controls['background_color'].model.as_string=='unfinished'
        assert general._status_error and general.status.text=='Keep this draft error'


def test_reset_saves_all_three_blocks_once_and_keeps_original_exemptions(app,tmp_path):
    source=opened(app,tmp_path)
    assert app._set_transparency(TransparencyMode.TAGGED,.75)
    assert app._set_joint_scale(.25) and app._refresh()
    source.calls.clear();base=app.joint_calibration
    before=app._viewport_settings();assert before.joints
    extension_mode=app._overlay_mode(8)
    app.workflow.settings.set('unrelated_future_setting',{'value':[1,2]})
    writes=[];save=app.workflow.settings.set_many
    def observe(values):
        assert app._viewport_settings()==before and app.joint_scale==.25
        writes.append(deepcopy(values));save(values)
    app.workflow.settings.set_many=observe
    assert app.reset_visual_settings()
    assert len(writes)==1 and set(writes[0])=={'rendering','joint_scale','viewport_display'}
    assert app._viewport_settings()==before.visual_reset(GIZMO_SCALE)
    assert app.overlay_mask==16|8|128|256 and app._overlay_mode(8)==extension_mode
    assert app.joint_scale==1 and app.joint_calibration is base
    assert app._refresh() and len(source.calls)==1
    restored=Preferences(app.workflow.settings.path)
    assert ViewportDisplaySettings.from_preferences(restored)==app._viewport_settings()
    assert restored.get('unrelated_future_setting')=={'value':[1,2]}


def test_new_capture_reuses_native_ids_without_restoring_tags_or_recalibrating_from_them(app,tmp_path):
    source=opened(app,tmp_path)
    assert app._set_overlay_mode(1,DisplayMode.SELECTED)
    assert app._set_transparency(TransparencyMode.TAGGED,.8)
    accepted=app._viewport_settings();old=app.joint_calibration
    app.interaction.set_visible([33],False);app.interaction.set_transparent([22],True)
    path,native=open_capture(tmp_path,'multi_geom.ovd');replacement=ObservedSession(native)
    assert app._display(path,replacement)
    assert app._viewport_settings()==accepted
    assert app.presentation.capture_key!=old.capture_key and app.joint_calibration.matches(app.presentation)
    assert app.joint_calibration is not old and not app.interaction.hidden and not app.interaction.transparent
    assert app.interaction.selected==() and 33 in app.adapter.instances
    np.testing.assert_array_equal(app.renderer.instance_alpha,np.ones(len(app.presentation.geometry_snapshot().kinds),dtype='f4'))


def test_partial_live_change_retains_owned_view_until_complete(app,live):
    native,connection=live;source=ObservedSession(native)
    app.workflow.live_session=source
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END]);wait_until(lambda:native.info().completed_serial==2)
    assert app._display(None,source) and app.seek(2,snap=False)
    before=app.presentation;base=app.joint_calibration
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda:native.info().recording_size==len(PAYLOAD)
               and native.info().unpublished_changes)
    source.calls.clear();assert app._set_overlay(32,False)
    assert not app._refresh() and len(source.calls)==1 and app.presentation is before
    assert app.joint_calibration is base and app.presentation_dirty
    assert not Preferences(app.workflow.settings.path).get('viewport_display')['contacts']
    native.stop_live();assert app._refresh() and len(source.calls)==2
    assert app.joint_calibration is base and app.presentation is not before
