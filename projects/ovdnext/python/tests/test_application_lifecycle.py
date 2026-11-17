# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real idle native capture/services, plus injected close failures; no GPU."""
from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from ovd_ui.application import Application
from ovd_ui.lifecycle import run_standalone

ROOT=Path(__file__).resolve().parents[2]


def leaves(error):
    if isinstance(error,BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


@pytest.fixture
def application(tmp_path,monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA" if os.name == "nt" else "XDG_DATA_HOME",
                       str(tmp_path/"preferences"))
    value=Application(ROOT/"test/data/test_scene.ovd")
    yield value
    if not value._close_started:value.close()


def test_unstarted_application_closes_real_constructor_services_once(application):
    temporary=Path(application.temporary.name)
    assert temporary.is_dir()
    application.close()
    application.close()
    assert application.closed and not application.running
    assert application.capture_opener.closed and application.analysis_availability.closed
    assert application.capture_selection_view._closed
    assert application.evidence_controller._closed
    assert application.workflow.closed
    assert not temporary.exists()
    for executor in (application.export_executor,application.analysis_tasks.executor,
                     application.diagnostics_tasks.executor,application.error_tasks.executor,
                     application.evidence_controller._executor):
        with pytest.raises(RuntimeError,match="shutdown"):
            executor.submit(lambda:None)


def test_idle_native_session_and_renderer_close_before_backend_shutdown(application):
    events=[]
    assert application.workflow.open_file(ROOT/"test/data/test_scene.ovd")
    application.session=application.workflow.display_session
    application.adapter=object()
    assert application.session.info().instance_count==8
    temporary=Path(application.temporary.name)
    def render_close():
        assert application.workflow.closed
        assert temporary.exists()
        events.append("renderer close")
    application.renderer=SimpleNamespace(close=render_close)
    async def complete():pass
    def backend_shutdown():
        assert application.closed and application.session is application.adapter is None
        assert not temporary.exists()
        events.append("backend shutdown")
    run_standalone(complete,complete,setup=lambda:None,close=application.close,shutdown=backend_shutdown)
    assert events==["renderer close","backend shutdown"]


def test_independent_cleanup_failures_do_not_skip_later_resources_or_hide_original(application):
    events=[]
    failure=ValueError("application failed")
    grip_error=RuntimeError("grip failed")
    renderer_error=OSError("renderer failed")
    def fail_grip():events.append("bad grip");raise grip_error
    def fail_renderer():events.append("bad renderer");raise renderer_error
    application.browser_grip=SimpleNamespace(destroy=fail_grip)
    application.inspector_grip=SimpleNamespace(destroy=lambda:events.append("other grip"))
    application.analysis_views=[SimpleNamespace(destroy=fail_grip),
                                SimpleNamespace(destroy=lambda:events.append("other analysis"))]
    application.renderer=SimpleNamespace(close=fail_renderer)
    temporary=Path(application.temporary.name)
    async def bad():raise failure
    async def pump():
        import asyncio
        await asyncio.Future()
    with pytest.raises(ExceptionGroup) as caught:
        run_standalone(bad,pump,setup=lambda:None,close=application.close,
                       shutdown=lambda:events.append("backend shutdown"))
    assert failure in leaves(caught.value)
    assert grip_error in leaves(caught.value) and renderer_error in leaves(caught.value)
    assert events==["bad grip","other grip","bad grip","other analysis","bad renderer","backend shutdown"]
    assert application.workflow.closed and application.capture_opener.closed
    assert not temporary.exists() and not application.closed
    assert application.analysis_views==[]
    before=list(events)
    with pytest.raises(ExceptionGroup) as repeated:application.close()
    assert repeated.value is application._close_error
    assert events==before


def test_partial_setup_failure_uses_actual_application_owner(application,monkeypatch):
    from ovd_ui import application as module
    failure=RuntimeError("style setup failed")
    events=[]
    def setup():events.append("partial setup");raise failure
    def shutdown():
        assert application.closed
        events.append("backend shutdown")
    monkeypatch.setattr(module.ui,"shutdown",shutdown)
    with pytest.raises(RuntimeError) as caught:
        application.run_with_backend(setup=setup)
    assert caught.value is failure
    assert events==["partial setup","backend shutdown"]


def test_application_constructor_failure_is_preserved_and_backend_still_closed(application,monkeypatch):
    from ovd_ui import application as module
    failure=RuntimeError("renderer initialization failed")
    def renderer():raise failure
    async def pump():
        import asyncio
        await asyncio.Future()
    monkeypatch.setattr(module,"ViewportRenderer",renderer)
    monkeypatch.setattr(module.ui,"run_async",pump)
    events=[]
    monkeypatch.setattr(module.ui,"shutdown",lambda:events.append("shutdown"))
    with pytest.raises(RuntimeError) as caught:
        application.run_with_backend(setup=lambda:None)
    assert caught.value is failure and application.run_error is failure
    assert application.closed and events==["shutdown"]
