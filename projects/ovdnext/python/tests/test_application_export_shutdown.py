# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real executor outcomes remain observable when the window pump has stopped."""
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from ovd_ui.lifecycle import run_standalone
from test_application_lifecycle import application, leaves


def observe_result(monkeypatch, future):
    calls=[]
    original=future.result
    def result(*args,**kwargs):
        calls.append((future.done(),future.cancelled()))
        return original(*args,**kwargs)
    monkeypatch.setattr(future,"result",result)
    return calls


def worker_failure(error):
    raise error


def test_completed_export_success_is_read_once_and_cleared(application,monkeypatch):
    future=application.export_executor.submit(lambda:(Path("saved.pxd2"),123))
    application.export_executor.shutdown(wait=True)
    application.pending_export=future
    calls=observe_result(monkeypatch,future)
    application.close()
    application.close()
    assert calls==[(True,False)]
    assert application.pending_export is None and application.closed


@pytest.mark.parametrize("error",[FileExistsError("destination exists"),CancelledError("worker cancelled")])
def test_completed_worker_failure_is_reported_with_original_traceback(application,monkeypatch,error):
    future=application.export_executor.submit(worker_failure,error)
    application.export_executor.shutdown(wait=True)
    assert future.done() and not future.cancelled()
    original_traceback=error.__traceback__
    application.pending_export=future
    calls=observe_result(monkeypatch,future)
    with pytest.raises(ExceptionGroup) as caught:application.close()
    assert leaves(caught.value)==[error]
    tracebacks=[]
    traceback=error.__traceback__
    while traceback is not None:
        tracebacks.append(traceback)
        traceback=traceback.tb_next
    assert original_traceback in tracebacks
    assert any(value.tb_frame.f_code.co_name=="worker_failure" for value in tracebacks)
    assert calls==[(True,False)] and application.pending_export is None
    assert application.workflow.closed and not application.closed
    assert not Path(application.temporary.name).exists()
    with pytest.raises(ExceptionGroup) as repeated:application.close()
    assert repeated.value is caught.value and calls==[(True,False)]


@pytest.mark.parametrize("fail",[False,True])
def test_running_export_is_joined_before_its_result_is_read(application,monkeypatch,fail):
    entered=Event()
    release=Event()
    error=OSError("native export failed")
    executor=application.export_executor
    def work():
        entered.set()
        assert release.wait(5),"test worker was not released"
        if fail:worker_failure(error)
        return "completed"
    future=executor.submit(work)
    assert entered.wait(5) and future.running()
    application.pending_export=future
    calls=observe_result(monkeypatch,future)
    joins=[]
    def shutdown(**kwargs):
        assert future.running() and calls==[]
        release.set()
        executor.shutdown(**kwargs)
        joins.append(future.done())
    application.export_executor=SimpleNamespace(shutdown=shutdown)
    if fail:
        with pytest.raises(ExceptionGroup) as caught:application.close()
        assert leaves(caught.value)==[error]
    else:application.close()
    assert joins==[True] and calls==[(True,False)]
    assert application.pending_export is None and application.closed is not fail


def test_queued_cancel_is_normal_and_does_not_read_result(application,monkeypatch):
    entered=Event()
    release=Event()
    executor=application.export_executor
    def occupied():
        entered.set()
        assert release.wait(5),"test executor was not released"
    blocker=executor.submit(occupied)
    assert entered.wait(5) and blocker.running()
    executed=[]
    future=executor.submit(lambda:executed.append(True))
    application.pending_export=future
    calls=observe_result(monkeypatch,future)
    def shutdown(**kwargs):
        executor.shutdown(wait=False,cancel_futures=True)
        release.set()
        executor.shutdown(**kwargs)
    application.export_executor=SimpleNamespace(shutdown=shutdown)
    application.close()
    assert future.cancelled() and executed==[] and calls==[]
    assert application.pending_export is None and application.closed


def test_export_and_renderer_failures_both_survive_backend_cleanup(application,monkeypatch):
    export_error=FileExistsError("destination raced with export")
    renderer_error=RuntimeError("renderer close failed")
    future=application.export_executor.submit(worker_failure,export_error)
    application.export_executor.shutdown(wait=True)
    application.pending_export=future
    calls=observe_result(monkeypatch,future)
    events=[]
    def render_close():
        assert application.pending_export is None
        events.append("renderer")
        raise renderer_error
    application.renderer=SimpleNamespace(close=render_close)
    async def complete():pass
    def shutdown():
        assert not Path(application.temporary.name).exists()
        assert application.workflow.closed and application.session is None
        events.append("backend")
    with pytest.raises(ExceptionGroup) as caught:
        run_standalone(complete,complete,setup=lambda:None,close=application.close,shutdown=shutdown)
    assert leaves(caught.value)==[export_error,renderer_error]
    assert events==["renderer","backend"] and calls==[(True,False)]
    assert application.pending_export is None and not application.closed
    with pytest.raises(ExceptionGroup) as repeated:application.close()
    assert repeated.value is application._close_error
    assert events==["renderer","backend"] and calls==[(True,False)]
