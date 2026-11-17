# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native presence, bounded uncertainty and asynchronous capture lifecycle."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import _pvd_native as native
from ovd_ui.analysis_availability import (ACTIONS, AnalysisAvailability, AvailabilityContext,
                                          read_availability)
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def fixture_executable():
    return executable("pvd_availability_fixture", "PVD_AVAILABILITY_FIXTURE")


@pytest.fixture
def generated(tmp_path, fixture_executable):
    def create(mode="dealloc"):
        path = tmp_path / (mode + ".pxd2")
        subprocess.run([str(fixture_executable), str(path), mode], check=True, capture_output=True)
        return path
    return create


@pytest.mark.parametrize("name", ["cloth_physx34.pxd2", "ue4_shooter.pxd2", "apex_destruction.pxd2"])
def test_recorded_metadata_enables_empty_profile_and_memory_without_moving_display(tmp_path, name):
    session = native.Session(str(ROOT/"test/data"/name), str(tmp_path))
    scene = session.scenes()[0].id
    geometry = session.geometry_snapshot().transforms.copy()
    data = native.analysis_availability(session, scene)
    assert data.profile.available and data.profile_zones > 0
    assert data.memory.available and data.memory_entries > 0
    assert data.memory_bytes == data.memory_events == 0  # metadata suffices; no parser/aggregation
    assert native.profile_snapshot(session, scene).scopes == []
    assert native.memory_snapshot(session, scene).rows == []
    assert data.frame_timing.available == session.frame_timing().supported
    assert not data.errors.available and data.errors.known
    assert session.scenes()[0].current_frame == 0
    np.testing.assert_array_equal(session.geometry_snapshot().transforms, geometry)


def test_zero_metadata_deallocation_empty_log_and_budget_uncertainty(generated, tmp_path):
    session = native.Session(str(generated()), str(tmp_path))
    scene = session.scenes()[0].id
    session.seek(scene, 2)
    data = native.analysis_availability(session, scene)
    assert data.memory_entries == 0 and data.memory_bytes == 4 and data.memory_events == 1
    assert data.memory.known and data.memory.available
    assert native.memory_snapshot(session, scene).raw_event_count == 1
    assert data.errors.available and data.error_log >= 0
    errors = native.error_snapshot(session, scene)
    assert errors.has_log and errors.total == 0  # log presence, not record count
    assert not data.profile.available and data.profile.known
    for options in ({"max_bytes": 1}, {"max_events": 1}):
        bounded = native.analysis_availability(session, scene, **options)
        assert not bounded.memory.available and not bounded.memory.known
        assert "budget" in bounded.memory.reason
        assert bounded.memory_bytes == 0  # bounded before parser entry
    bounded = native.analysis_availability(session, scene, max_instances=1)
    assert not bounded.errors.known and not bounded.errors.available
    assert bounded.scanned_instances == 1 and "budget" in bounded.errors.reason
    session.seek(scene, 3)
    empty = native.analysis_availability(session, scene)
    assert empty.memory.known and not empty.memory.available
    assert empty.errors.available
    assert data.memory.available and data.memory_events == 1  # owned earlier result


def test_true_empty_memory_stream_is_absent(generated, tmp_path):
    session = native.Session(str(generated("empty")), str(tmp_path))
    scene = session.scenes()[0].id
    session.seek(scene, 2)
    data = native.analysis_availability(session, scene)
    assert data.memory.known and not data.memory.available and data.memory_entries == 0
    assert not native.memory_snapshot(session, scene).supported


def test_native_validation_and_pre_cancel_are_explicit(tmp_path):
    session = native.Session(str(ROOT/"test/data/cloth_physx34.pxd2"), str(tmp_path))
    for options in ({"max_instances": 0}, {"max_events": 65537}, {"max_bytes": 8*1024*1024+1}):
        with pytest.raises(ValueError, match="budget"):
            native.analysis_availability(session, **options)
    with pytest.raises(IndexError, match="scene"):
        native.analysis_availability(session, 123)
    control = native.LoadControl(); control.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        native.analysis_availability(session, control=control)
    assert session.scenes()[0].current_frame == 0


def test_real_ue4_error_log_presence_changes_with_frame(tmp_path):
    session = native.Session(str(ROOT/"test/data/ue4_shooter.pxd2"), str(tmp_path))
    scene = session.scenes()[0].id
    assert not native.analysis_availability(session, scene).errors.available
    session.seek(scene, 921)
    data = native.analysis_availability(session, scene)
    assert data.errors.available and data.error_log == native.error_snapshot(session, scene).log_instance_id
    session.seek(scene, 0)
    assert not native.analysis_availability(session, scene).errors.available


class ControlledExecutor:
    def __init__(self): self.calls = []; self.closed = False
    def submit(self, function, *args, **kwargs):
        future = Future(); future.set_running_or_notify_cancel()
        self.calls.append((future, lambda: function(*args, **kwargs)))
        return future
    def shutdown(self, **kwargs): self.closed = True


def context(**changes):
    return replace(AvailabilityContext(1, 0, ROOT/"test/data/cloth_physx34.pxd2", 6, 2, False, False, 10), **changes)


def facts(**changes):
    values = {name: SimpleNamespace(known=True, available=True, reason="") for name in ACTIONS if name != "messages"}
    values["memory_entries"] = 0
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def controller():
    executor = ControlledExecutor()
    value = AnalysisAvailability(executor=executor, reader=lambda *_: facts())
    try: yield value, executor
    finally: value.close()


def test_pending_and_same_frame_cache_never_rescan_on_menu_open(controller):
    value, executor = controller
    value.sync(context()); value.request()
    assert value.state("profile").checking and not value.state("profile").enabled
    for _ in range(10): value.request(); value.poll()
    assert len(executor.calls) == 1
    executor.calls[0][0].set_result(facts())
    assert value.poll() and value.state("profile").enabled
    for _ in range(10): value.request()
    assert len(executor.calls) == 1
    value.sync(context(clip_frames=1))
    assert not value.state("frame_timing").enabled and "two frames" in value.state("frame_timing").reason
    assert value.state("memory").enabled and len(executor.calls) == 1


@pytest.mark.parametrize("change", [{"capture_key": 2}, {"segment_epoch": 1}, {"scene_id": 123}, {"frame": 3}])
def test_stale_running_completion_cannot_enable_new_context(controller, change):
    value, executor = controller
    value.sync(context()); value.request()
    old_control = value.pending[1]
    value.sync(context(**change))
    assert old_control.cancelled
    executor.calls[0][0].set_result(facts())
    assert not value.poll() and not value.state("profile").enabled
    value.request()
    assert len(executor.calls) == 2
    executor.calls[1][0].set_result(facts(memory=SimpleNamespace(known=True, available=False, reason="No measurements")))
    assert value.poll() and value.state("profile").enabled and not value.state("memory").enabled


def test_new_close_and_failed_checks_do_not_reuse_success(controller):
    value, executor = controller
    value.sync(context()); value.request(); first_control=value.pending[1]
    value.sync(None)
    executor.calls[0][0].set_result(facts())
    assert first_control.cancelled and not value.poll()
    assert all(not value.state(name).enabled for name in ACTIONS)
    value.sync(context(capture_key=2)); value.request()
    executor.calls[1][0].set_exception(OSError("capture disappeared"))
    assert value.poll() and "capture disappeared" in value.state("errors").reason
    value.request(); assert len(executor.calls) == 2
    value.sync(context(frame=3)); value.request(); control=value.pending[1]
    value.close()
    assert control.cancelled and executor.closed and not value.poll()


def test_unknown_is_cached_with_reason_and_frame_revisit_uses_bounded_cache(controller):
    value, executor = controller
    value.sync(context()); value.request()
    executor.calls[0][0].set_result(facts(memory=SimpleNamespace(known=False,available=False,reason="Memory scan budget exceeded")))
    value.poll(); value.request()
    assert "budget" in value.state("memory").reason and not value.state("memory").checking
    value.sync(context(frame=3))
    assert value.state("profile").enabled and len(executor.calls) == 1  # whole-file profile metadata reused
    assert not value.state("memory").enabled and not value.state("errors").enabled
    value.sync(context())
    assert value.state("profile").enabled and len(executor.calls) == 1


def test_empty_and_active_ovd_messages_are_available_without_any_scan(controller, tmp_path):
    value, executor = controller
    source = native.Session(str(ROOT/"test/data/test_scene.ovd"), str(tmp_path))
    assert source.message_count() == 0
    for active in (False, True):
        value.sync(context(ovd=True, active=active, path=None, clip_frames=1)); value.request()
        assert value.state("messages").enabled
        assert all(not value.state(name).enabled for name in ACTIONS if name != "messages")
    assert executor.calls == []
    value.sync(context(active=True)); value.request()
    assert "Detach" in value.state("profile").reason and executor.calls == []


def test_private_read_leaves_native_display_on_its_original_frame(tmp_path):
    path = ROOT/"test/data/ue4_shooter.pxd2"
    display = native.Session(str(path), str(tmp_path))
    scene = display.scenes()[0].id
    data = read_availability(context(path=path, scene_id=scene, frame=921), native.LoadControl())
    assert data.errors.available
    assert display.scenes()[0].current_frame == 0
    with pytest.raises(ValueError, match="existing saved capture"):
        read_availability(context(path=tmp_path/"missing.pxd2"), native.LoadControl())


def test_queued_availability_retains_extracted_payload_after_display_close(tmp_path):
    import gc
    from threading import Event
    import zipfile
    from ovd_ui.ingestion import load_capture
    archive = tmp_path/"capture.pvd"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(ROOT/"test/data/cloth_physx34.pxd2", "nested/data.pxd2")
    loaded = load_capture(archive)
    payload = loaded.data_path
    value = AnalysisAvailability()
    gate = Event()
    value.executor.submit(gate.wait)
    try:
        value.sync(context(path=payload, scene_id=loaded.session.scenes()[0].id))
        value.request()
        future = value.pending[2]
        loaded.close(); gc.collect()
        assert payload.exists(), "Queued availability lost the retained extracted payload"
        gate.set()
        assert future.result(timeout=10).profile.available
        assert value.poll() and value.state("profile").enabled
    finally:
        gate.set(); value.close()
    gc.collect()
    assert not payload.exists()


def test_close_cancels_running_worker_and_reaps_its_private_session(tmp_path):
    from concurrent.futures import CancelledError
    from threading import Event
    started, finished = Event(), Event()
    def reader(request, control):
        # A real indexed native load, then a controlled safe cancellation point.
        session = native.Session(str(request.path), str(tmp_path), control)
        started.set()
        try:
            while not control.cancelled:
                finished.wait(.005)
            raise CancelledError()
        finally:
            del session
            finished.set()
    value = AnalysisAvailability(reader=reader)
    value.sync(context()); value.request()
    future = value.pending[2]
    assert started.wait(5)
    value.close()
    assert finished.is_set() and future.done() and value.closed
    with pytest.raises(CancelledError): future.result()
    assert not value.poll() and all(not value.state(name).enabled for name in ACTIONS)


def test_owned_whole_file_facts_keep_toolbar_actions_usable_without_frame_scans(controller):
    value, executor = controller
    value.sync(context()); value.request()
    executor.calls[0][0].set_result(facts(memory_entries=12))
    value.poll(); value.sync(context(frame=3))
    assert all(value.state(name).enabled for name in ("profile","memory","frame_timing"))
    assert not value.state("errors").enabled and len(executor.calls)==1
    value.sync(context(frame=3,segment_epoch=1))
    assert all(not value.state(name).enabled for name in ("profile","memory","frame_timing","errors"))


def test_submission_failures_obey_the_same_bounded_cache():
    class FailingExecutor(ControlledExecutor):
        def submit(self,*args,**kwargs): raise RuntimeError("executor rejected submission")
    value=AnalysisAvailability(executor=FailingExecutor(),cache_size=2)
    try:
        for frame in (1,2,3,4):
            value.sync(context(frame=frame));value.request()
            assert "rejected submission" in value.state("profile").reason
        assert len(value.cache)==2 and [key[-1] for key in value.cache]==[3,4]
    finally:value.close()


@pytest.mark.parametrize('action',['profile','memory','errors','frame_timing'])
def test_unchecked_data_is_requestable_but_never_advertised_as_present(controller,action):
    value,executor=controller
    value.sync(context())
    state=value.state(action)
    assert state.requestable and not state.enabled and not state.checking
    assert not executor.calls  # Reading button state never starts native work.
    value.request()
    state=value.state(action)
    assert state.requestable and state.checking and not state.enabled
    absent=SimpleNamespace(known=True,available=False,reason='No recorded data')
    executor.calls[0][0].set_result(facts(**{action:absent}))
    assert value.poll()
    state=value.state(action)
    assert not state.requestable and not state.enabled and not state.checking
    value.sync(context(frame=921))
    state=value.state(action)
    if action in ('memory','errors'):
        assert state.requestable and not state.enabled  # Presence is frame-specific.
    else:
        assert not state.requestable  # Whole-file metadata is still authoritative.
    assert len(executor.calls)==1


@pytest.mark.parametrize('changes',[{'ovd':True},{'active':True},{'path':None},{'scene_id':0}])
def test_unsupported_or_unsaved_views_cannot_request_a_probe(controller,changes):
    value,executor=controller
    value.sync(context(**changes))
    for action in ('profile','memory','errors'):
        state=value.state(action)
        assert not state.requestable and not state.enabled
    assert not executor.calls
