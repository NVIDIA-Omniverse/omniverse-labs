# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completed/error Futures must not keep archive payloads after native reads."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from threading import Event
import time
import weakref
import zipfile

import pytest
import _pvd_native as native

from ovd_ui.ingestion import load_capture, retain_capture, submit_capture
from ovd_ui.profiling import DiagnosticsTasks, ProfileHistoryRequest

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path=tmp_path_factory.mktemp("lease-source")/"cloth.pvd"
    with zipfile.ZipFile(path,"x",zipfile.ZIP_STORED) as output:
        output.write(ROOT/"test/data/cloth_physx34.pxd2","cloth.pxd2")
    return path


@pytest.fixture
def opened(archive,tmp_path):
    loaded=load_capture(archive,temporary_directory=tmp_path)
    yield loaded
    loaded.close()


def test_running_read_retains_payload_after_display_close_until_success(opened):
    path=opened.data_path
    entered=Event();finish=Event()
    def read():
        with retain_capture(path):
            entered.set()
            assert finish.wait(5)
            return path.stat().st_size
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=submit_capture(executor,path,read)
        try:
            assert entered.wait(5)
            opened.close()
            assert path.exists() and not future.cancel()
        finally:finish.set()
        assert future.result(timeout=5)>0
    assert not path.exists()  # Keep the successful Future alive deliberately.


def test_failure_future_preserves_original_error_traceback_without_retaining_payload(opened):
    path=opened.data_path
    resource=weakref.ref(opened._resources)
    original=ValueError("native-reader-shaped failure")
    def fail():
        with retain_capture(path):
            assert path.exists()
            raise original
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=submit_capture(executor,path,fail)
        with pytest.raises(ValueError) as caught:future.result(timeout=5)
        assert caught.value is original
    opened.close()
    assert future.exception() is original and original.__traceback__ is not None
    assert not path.exists() and resource() is None


def test_queued_cancel_releases_payload_without_starting_native_work(opened):
    path=opened.data_path
    occupied=Event();finish=Event();ran=[]
    def busy():occupied.set();assert finish.wait(5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker=executor.submit(busy)
        try:
            assert occupied.wait(5)
            future=submit_capture(executor,path,lambda:ran.append(True))
            opened.close()
            assert path.exists()  # Queued submission owns the extracted file.
            assert future.cancel()
            assert not path.exists()  # Work item is still in the occupied queue.
            assert ran==[]
        finally:finish.set()
        blocker.result(timeout=5)
    with pytest.raises(CancelledError):future.result()


def test_submit_rejection_preserves_exception_but_drops_submission_lease(opened):
    path=opened.data_path
    class Rejected:
        def submit(self,function):raise RuntimeError("executor rejected submission")
    with pytest.raises(RuntimeError) as caught:
        submit_capture(Rejected(),path,lambda:None)
    opened.close()
    assert str(caught.value)=="executor rejected submission"
    assert caught.value.__traceback__ is not None and not path.exists()


def test_direct_context_error_keeps_payload_until_unwind_and_preserves_error(opened):
    path=opened.data_path
    original=RuntimeError("direct context failure")
    with pytest.raises(RuntimeError) as caught:
        with retain_capture(path):
            opened.close()
            assert path.exists()
            raise original
    assert caught.value is original and not path.exists()


def test_real_cancelled_profile_history_releases_payload_while_service_keeps_future(opened):
    path=opened.data_path
    scene=opened.session.scenes()[0].id
    assert opened.session.seek(scene,1)
    snapshot=native.profile_snapshot(opened.session,scene)
    row=next(row for row in snapshot.summaries if row.count>0)
    request=ProfileHistoryRequest(path,scene,row.zone_id,row.event_id,row.context_id,None)
    service=DiagnosticsTasks()
    try:
        job=service.submit_history(request)
        deadline=time.monotonic()+5
        while not job.future.done() and job.progress[0]==0 and time.monotonic()<deadline:
            time.sleep(.001)  # Observe real progress; never delay the worker.
        assert job.future.running() and 0<job.progress[0]<job.progress[1]
        opened.close()
        assert path.exists()
        service.close()
        assert service.jobs[0] is job and job.cancel_event.is_set()
        with pytest.raises(CancelledError,match="Analysis cancelled"):
            job.future.result()
        assert job.future.exception().__traceback__ is not None
        assert not path.exists()  # No gc.collect or dropping error/job history.
    finally:service.close()
