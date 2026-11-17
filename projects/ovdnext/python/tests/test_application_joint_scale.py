# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual owned publications and atomic preferences; only GPU/widgets are sinks."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
from pathlib import Path
import subprocess
import struct

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.joint_scale import resolve_joint_base, validate_joint_scale
from ovd_ui.render_settings import RenderingSettings
from ovd_ui.settings import Preferences
from test_application_general_settings import ObservedSession
from test_application_presentation import (app as base_app, open_capture, live,
    HANDSHAKE, PAYLOAD, FRAME2_END, wait_until)
from test_visual_reset import app, opened, state, camera_controls
from test_joint_calibration import calibration_writer, calibration_capture


def test_new_capture_first_applied_joint_base_uses_its_own_geometry(app, tmp_path, calibration_capture):
    app.render_settings=RenderingSettings()
    app.joint_scale=1.
    assert app._display(*open_capture(tmp_path,"ue4_shooter.pxd2"))
    prior=app.joint_calibration
    app.navigation.scene_length=9999.
    # IDs are reused by this separate capture; neither tags nor base carry over.
    app.interaction.hidden.update(range(1,200))
    path,native=calibration_capture("box");source=ObservedSession(native)
    assert app._display(path,source)
    assert len(source.calls)==4
    assert app.joint_calibration.capture_key!=prior.capture_key
    assert app.joint_calibration.base_length==16  # actual initial Post frame2, not frame1
    assert app.presentation.joint_scale==1 and app.presentation.joint_base_length==16
    assert not app.interaction.hidden
    initial=app.joint_calibration
    source.calls.clear()
    assert app.seek(1,snap=False)
    assert len(source.calls)==1 and app.joint_calibration is initial
    assert app.presentation.joint_base_length==16
    fresh=resolve_joint_base(app.presentation,app.adapter,app.render_settings)
    assert fresh.base_length==4  # geometry changed but no ordinary-frame recalibration
    source.calls.clear();assert app._refresh() and len(source.calls)==1


@pytest.mark.parametrize("mode",["empty","plane","zero","overflow"])
def test_replacement_fallback_never_uses_previous_camera_or_capture_base(app, tmp_path, calibration_capture, mode):
    app.render_settings=RenderingSettings()
    opened(app,tmp_path)
    old=app.joint_calibration
    app.navigation.scene_length=12345
    if mode=="overflow":
        # Existing Camera.calibrate still performs its original unchecked size
        # subtraction; this task changes only the independent joint base.
        with pytest.warns(RuntimeWarning,match="overflow"):
            assert app._display(*calibration_capture(mode))
    else:
        assert app._display(*calibration_capture(mode))
    assert app.joint_calibration.capture_key!=old.capture_key
    assert app.presentation.joint_base_length==app.presentation.tolerance_length==9
    assert app.joint_calibration.source=="captured-tolerance"
    assert app._display(None,None) and app.joint_calibration is None


def test_raw_changes_persist_coalesce_and_do_not_publish_disabled_category(app,tmp_path):
    app.render_settings=RenderingSettings()
    source=opened(app,tmp_path)
    first=app.presentation;calibration=app.joint_calibration
    app.overlay_mask=0
    assert app._set_joint_scale(.3)
    assert not app.presentation_dirty and source.calls==[]
    assert app.presentation is first and first.joint_scale==1
    stored=Preferences(app.workflow.settings.path)
    assert stored.get("joint_scale")==validate_joint_scale(.3)
    app.overlay_mask=16
    assert app._set_joint_scale(.4) and app._set_joint_scale(.5)
    assert app.presentation_dirty and source.calls==[]
    assert app._refresh() and len(source.calls)==1
    assert app.presentation.joint_scale==.5 and app.joint_calibration is calibration
    source.calls.clear();assert app._set_joint_scale(.5)
    assert not app.presentation_dirty and source.calls==[]


def test_raw_and_reset_write_failure_preserve_values_controls_drafts_and_view(app,tmp_path,monkeypatch):
    source=opened(app,tmp_path)
    assert app._set_joint_scale(.5) and app._refresh()
    source.calls.clear()
    old_view=app.presentation;old_state=state(app);old_values=deepcopy(app.workflow.settings.values)
    file_bytes=app.workflow.settings.path.read_bytes()
    camera=camera_controls(app);camera.fov.model.set_value(79)
    general=app.render_settings_view;general.controls["background_color"].model.set_value("unfinished")
    general._status_error=True;general.status.text="Keep this error"
    app.view_window=NS(visible=True);app.syncing_view_options=False
    model=ui.SimpleFloatModel(.5);app.view_option_controls={"joint_scale":model}
    model.add_value_changed_fn(lambda m:app._view_option_changed(app._set_joint_scale,m.as_float))
    import ovd_ui.settings as settings_module
    monkeypatch.setattr(settings_module.os,"replace",lambda *_:(_ for _ in ()).throw(OSError("unwritable")))
    model.set_value(.9)
    assert app.joint_scale==model.as_float==.5 and not app.reset_visual_settings()
    assert app.presentation is old_view and state(app)==old_state and source.calls==[]
    assert app.workflow.settings.values==old_values and app.workflow.settings.path.read_bytes()==file_bytes
    assert general.controls["background_color"].model.as_string=="unfinished" and general._status_error
    assert camera.fov.model.as_float==79


def test_reset_commits_raw_with_rendering_once_and_preserves_calibration(app,tmp_path):
    source=opened(app,tmp_path)
    assert app._set_joint_scale(0) and app._refresh()
    source.calls.clear();calibration=app.joint_calibration
    writes=[];save=app.workflow.settings.set_many
    def observe(values):
        assert app.joint_scale==0
        writes.append(deepcopy(values));save(values)
    app.workflow.settings.set_many=observe
    assert app.reset_visual_settings()
    assert len(writes)==1 and set(writes[0])=={"rendering","joint_scale","viewport_display"} and writes[0]["joint_scale"]==1
    assert app.joint_scale==1 and app.joint_calibration is calibration
    assert app._refresh() and len(source.calls)==1
    assert app.presentation.joint_scale==1 and app.presentation.joint_base_length==calibration.base_length


def test_partial_live_raw_request_keeps_old_owned_base_and_exact_view(app,live):
    native,connection=live;source=ObservedSession(native)
    app.workflow.live_session=source
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END]);wait_until(lambda:native.info().completed_serial==2)
    assert app._display(None,source) and app.seek(2,snap=False)
    old=app.presentation;calibration=app.joint_calibration
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda:native.info().recording_size==len(PAYLOAD)
               and native.info().unpublished_changes)
    source.calls.clear()
    assert app._set_joint_scale(.5) and not app._refresh()
    assert len(source.calls)==1 and app.presentation is old and app.joint_calibration is calibration
    assert old.joint_scale==1 and app.presentation_dirty
    native.stop_live();assert app._refresh()
    assert len(source.calls)==2 and app.presentation.joint_scale==.5
    assert app.joint_calibration is calibration and app.presentation.joint_base_length==old.joint_base_length


def test_partial_new_capture_does_not_replace_previous_base(app,tmp_path,live):
    opened(app,tmp_path);old=app.presentation;calibration=app.joint_calibration
    native,connection=live
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END]);wait_until(lambda:native.info().completed_serial==2)
    native.refresh_live()  # The workflow indexes each completed prefix before promotion.
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda:native.info().recording_size==len(PAYLOAD)
               and native.info().unpublished_changes)
    assert not app._display(None,native)
    assert app.presentation is old and app.joint_calibration is calibration
    native.stop_live();assert app._display(None,native)
    assert app.joint_calibration.capture_key!=calibration.capture_key
    assert app.joint_calibration.matches(app.presentation)


def test_explicit_scene_adoption_recalibrates_once_then_seek_retains_base(app,tmp_path):
    app.render_settings=RenderingSettings()
    path,native=open_capture(tmp_path,"resample_two_segment.ovd")
    source=ObservedSession(native);assert app._display(path,source)
    before=app.joint_calibration
    other=next(scene for scene in app.scenes if scene.id!=app.scene_id)
    source.calls.clear()
    assert app._refresh(scene_id=other.id,frame=1)
    assert len(source.calls)==2
    assert app.joint_calibration.scene_id==other.id and app.joint_calibration is not before
    expected=resolve_joint_base(app.presentation,app.adapter,app.render_settings,
        selected=app.interaction.selected,hidden=app.interaction.hidden)
    assert app.joint_calibration==expected
    current=app.joint_calibration;source.calls.clear()
    assert app.seek(2,snap=False)
    assert len(source.calls)==1 and app.joint_calibration is current


def test_new_live_segment_calibrates_without_previous_namespace_tags(app,live):
    native,connection=live;source=ObservedSession(native)
    root=Path(__file__).resolve().parents[2]
    payload=(root/"test/data/resample_two_segment.ovd").read_bytes()
    first_end=42117
    stop=struct.pack("<BQQ",12,2929590687920,1)
    second_first_end=payload.index(stop,first_end)+len(stop)
    connection.sendall(HANDSHAKE+payload[:first_end])
    wait_until(lambda:native.info().completed_serial==120)
    app.workflow.live_session=source;app.render_settings=RenderingSettings()
    assert app._display(None,source)
    old=app.joint_calibration
    # This intentionally includes IDs a later recording may reuse. They belong
    # to the old interaction namespace and cannot filter the new calibration.
    app.interaction.hidden.update(range(1,10000))
    connection.sendall(payload[first_end:second_first_end])
    wait_until(lambda:native.info().completed_serial==121)
    update=native.refresh_live();app.workflow.follow_live=True
    app.workflow.poll=lambda:[NS(kind="frames",session=source,scenes=update.scenes,
        segment_changed=True,max_frame=update.max_frame,current_frame=update.current_frame)]
    source.calls.clear();app._poll_workflow()
    actual=app.joint_calibration
    assert len(source.calls)==2 and actual.capture_key==old.capture_key
    assert actual.segment_epoch!=old.segment_epoch and actual.matches(app.presentation)
    assert actual.vote_count>0
    assert actual==resolve_joint_base(app.presentation,app.adapter,app.render_settings)


def test_calibrated_scene_call_uses_the_same_bounded_optional_read_retry(app,tmp_path):
    app.render_settings=RenderingSettings()
    path,native=open_capture(tmp_path,"resample_two_segment.ovd")
    source=ObservedSession(native);assert app._display(path,source)
    other=next(scene for scene in app.scenes if scene.id!=app.scene_id)
    actor=int(app.presentation.geometry_snapshot().object_ids[0])
    prop=app.adapter.property(actor,"GlobalPose")
    request=(actor,prop.id,0,4)
    app.property_inspector.read_requests=lambda *_:[request]
    source.calls.clear()
    # Only allocation failure is injected; all successful views/data are real.
    source.failures.extend((None,ValueError("Presentation metadata/inspector data exceeds 128 MiB")))
    assert app._refresh(scene_id=other.id,frame=1)
    assert len(source.calls)==3
    assert [call[1]["read_requests"] for call in source.calls]==[[request],[request],[]]
    assert app.presentation.scene_id==other.id and app.joint_calibration.matches(app.presentation)
    assert not app.presentation_dirty
    # A core failure is not swallowed, and leaves the previous display/retry.
    old=app.presentation;old_base=app.joint_calibration
    source.failures.extend((None,RuntimeError("core publication failed")))
    previous=next(scene for scene in app.scenes if scene.id!=app.scene_id)
    with pytest.raises(RuntimeError,match="core publication failed"):
        app._refresh(scene_id=previous.id,frame=1)
    assert app.presentation is old and app.joint_calibration is old_base and app.presentation_dirty


def test_partial_packet_between_calibration_and_first_overlay_cannot_adopt(app,tmp_path,live):
    opened(app,tmp_path);old=app.presentation;calibration=app.joint_calibration
    native,connection=live
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END]);wait_until(lambda:native.info().completed_serial==2)
    class ArrivingSource(ObservedSession):
        def presentation(self,*args,**kwargs):
            result=super().presentation(*args,**kwargs)
            if len(self.calls)==3:
                connection.sendall(PAYLOAD[FRAME2_END:])
                wait_until(lambda:native.info().recording_size==len(PAYLOAD)
                           and native.info().unpublished_changes)
            return result
    source=ArrivingSource(native)
    assert not app._display(None,source) and len(source.calls)==4
    assert app.presentation is old and app.joint_calibration is calibration
    native.stop_live();assert app._display(None,source)
    assert app.joint_calibration.matches(app.presentation)
    assert app.joint_calibration.capture_key==native.capture_key!=calibration.capture_key


def test_sdk_joint_control_uses_raw_scale_independently_of_scene_and_gizmo(app,tmp_path):
    from _pvd_native import Session
    from native_fixtures import executable
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    path=tmp_path/"sdk.ovd";subprocess.run([str(binary),str(path)],check=True)
    app.render_settings=RenderingSettings();app.overlay_mask=16;app.overlay_modes={}
    assert app._display(path,Session(str(path),str(tmp_path))) and app.seek(1,snap=False)
    first=app.presentation.overlay_snapshot();mask=first.kinds==4
    assert np.count_nonzero(mask)==6
    vectors=first.positions[mask,1]-first.positions[mask,0]
    app._set_gizmo_scale(19.)
    assert app._set_joint_scale(.25) and app._refresh()
    after=app.presentation.overlay_snapshot()
    np.testing.assert_array_equal(after.positions[mask,0],first.positions[mask,0])
    np.testing.assert_allclose(after.positions[mask,1]-after.positions[mask,0],vectors*.25,atol=1e-6)
