# SPDX-License-Identifier: Apache-2.0
"""Callback latency, coalesced ownership, and fail-closed recovery boundaries."""
import threading
import time
from types import SimpleNamespace

import pytest

from ovrtx_blender_example import engine, scene_generation_sessions as sessions
from ovrtx_blender_example import direct_sdk_client


def test_restart_is_nonblocking_coalesced_and_discards_only_after_drain(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    events = []
    monkeypatch.setattr(engine, '_WORKER_RESTART', None)
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    monkeypatch.setattr(engine, '_visit_active_viewport_engines', lambda fn: ([], []))
    monkeypatch.setattr(direct_sdk_client, 'enabled', lambda: True)
    monkeypatch.setattr(sessions, 'request_preparation_pause', lambda: events.append('admission'))
    def drain():
        entered.set()
        assert release.wait(2)
        events.append('drained')
        return True
    monkeypatch.setattr(sessions, 'pause_preparation', drain)
    monkeypatch.setattr(sessions, 'discard_unacknowledged_renderer_intent', lambda: events.append('discard') or 3)
    monkeypatch.setattr(sessions, 'deactivate_all_ovrtx', lambda: events.append('deactivate') or True)
    monkeypatch.setattr(sessions, 'resume_preparation', lambda: events.append('resume'))
    try:
        start = time.monotonic()
        assert engine.restart_ovrtx_workers()['status'] == 'restart_pending'
        assert time.monotonic() - start < .1
        assert entered.wait(2)
        task = engine._WORKER_RESTART
        for _ in range(100):
            assert engine.restart_ovrtx_workers()['status'] == 'restart_pending'
        assert engine._WORKER_RESTART is task
        assert events == ['admission']
        release.set()
        assert task['completion'].wait(2)
        assert events == ['admission', 'drained', 'discard', 'deactivate', 'resume']
        assert task['result']['status'] == 'restarted'
        assert task['result']['discarded_unacknowledged_count'] == 3
    finally:
        release.set()
        engine._WORKER_RESTART['completion'].wait(2)


@pytest.mark.parametrize('stage', ['setup', 'thread'])
def test_restart_start_failure_settles_and_releases_admission(monkeypatch, stage):
    monkeypatch.setattr(engine, '_WORKER_RESTART', None)
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    events = []
    monkeypatch.setattr(sessions, 'request_preparation_pause', lambda: events.append('pause'))
    monkeypatch.setattr(sessions, 'resume_preparation', lambda: events.append('resume'))
    monkeypatch.setattr(direct_sdk_client, 'enabled', lambda: True)
    def fail(*args, **kwargs):
        raise RuntimeError('injected start failure')
    monkeypatch.setattr(engine, '_visit_active_viewport_engines', fail if stage == 'setup' else lambda fn: ([], []))
    if stage == 'thread':
        monkeypatch.setattr(threading.Thread, 'start', fail)
    result = engine.restart_ovrtx_workers()
    assert result['status'] == 'teardown_unconfirmed'
    assert result['restart_pending'] is False
    assert engine._WORKER_RESTART['completion'].is_set()
    assert events == ['pause', 'resume']


def test_cleanup_thread_start_failure_does_not_leave_session_guard_latched(monkeypatch):
    monkeypatch.setattr(engine, '_SESSION_ENDS_IN_PROGRESS', 0)
    monkeypatch.setattr(engine, '_DIRECT_VIEWPORT_REUSE_BLOCKED', False)
    def fail(*args, **kwargs):
        raise RuntimeError('injected thread resource exhaustion')
    monkeypatch.setattr(threading.Thread, 'start', fail)
    runtime = {'render_thread': object()}
    assert engine._teardown_engine_runtime(runtime) is False
    assert runtime['cleanup_completion'].is_set()
    assert runtime['stop_confirmed'] is False
    assert not engine._viewport_session_end_in_progress()
    assert engine._DIRECT_VIEWPORT_REUSE_BLOCKED is True


@pytest.mark.parametrize('fail', [False, True])
def test_deferred_close_blocks_reenable_until_exact_terminal_cleanup(monkeypatch, fail):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(sessions, '_CLOSE_TASK', None)
    monkeypatch.setattr(sessions, '_runtimes', {})
    monkeypatch.setattr(sessions, '_owners', {})
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    events = []
    def stop(**kwargs):
        entered.set()
        assert release.wait(2)
        events.append(threading.get_ident())
        if fail:
            raise RuntimeError('native cleanup failed')
        return set()
    monkeypatch.setattr(sessions, '_stop_preparation_worker', stop)
    try:
        started = time.monotonic()
        sessions.close(defer_native=True)
        assert time.monotonic() - started < .1
        assert entered.wait(2)
        task = sessions._CLOSE_TASK
        with pytest.raises(Exception, match='cleanup is still in progress'):
            sessions._require_session_open()
        sessions.close(defer_native=True)
        assert sessions._CLOSE_TASK is task
        release.set()
        assert task['completion'].wait(2)
        assert events == [task['thread'].ident]
        assert events[0] != threading.get_ident()
        if fail:
            assert sessions.close_status()['status'] == 'failed'
            with pytest.raises(RuntimeError, match='cleanup failed'):
                sessions._require_session_open()
        else:
            assert sessions.close_status()['status'] == 'closed'
            sessions._require_session_open()
    finally:
        release.set()
        sessions._CLOSE_TASK['completion'].wait(2)


@pytest.mark.parametrize('handler', ['load_pre', 'presentation_undo_pre'])
def test_file_and_undo_callbacks_defer_blocked_native_close(monkeypatch, handler):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(sessions, '_CLOSE_TASK', None)
    monkeypatch.setattr(sessions, '_load_pre_completed', False)
    monkeypatch.setattr(sessions, '_runtimes', {})
    monkeypatch.setattr(sessions, '_owners', {})
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    monkeypatch.setattr(engine, 'stop_viewport_render_threads_for_file_load', lambda: False)
    def stop(**kwargs):
        entered.set()
        assert release.wait(2)
        return set()
    monkeypatch.setattr(sessions, '_stop_preparation_worker', stop)
    try:
        started = time.monotonic()
        getattr(sessions, handler)()
        assert time.monotonic() - started < .1
        assert entered.wait(2)
        assert sessions.close_status()['status'] == 'closing'
        release.set()
        assert sessions._CLOSE_TASK['completion'].wait(2)
        assert sessions.close_status()['status'] == 'closed'
    finally:
        release.set()
        sessions._CLOSE_TASK['completion'].wait(2)


@pytest.mark.parametrize('fail_finalizer', [False, True])
def test_later_unregister_finalizers_join_pending_file_close_once(monkeypatch, fail_finalizer):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(sessions, '_CLOSE_TASK', None)
    monkeypatch.setattr(sessions, '_runtimes', {})
    monkeypatch.setattr(sessions, '_owners', {})
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    events = []
    def stop(**kwargs):
        entered.set()
        assert release.wait(2)
        return set()
    def finalizer():
        events.append(('services', threading.get_ident()))
        if fail_finalizer:
            raise RuntimeError('service close failed')
    def second():
        events.append(('second', threading.get_ident()))
    monkeypatch.setattr(sessions, '_stop_preparation_worker', stop)
    try:
        sessions.close(defer_native=True)
        assert entered.wait(2)
        task = sessions._CLOSE_TASK
        for _ in range(100):
            sessions.close(defer_native=True, after_native=finalizer)
        sessions.close(defer_native=True, after_native=second)
        assert task['finalizers'] == [finalizer, second]
        release.set()
        assert task['completion'].wait(2)
        assert [event[0] for event in events] == ['services', 'second']
        assert all(event[1] != threading.get_ident() for event in events)
        assert task['result']['status'] == ('failed' if fail_finalizer else 'closed')
    finally:
        release.set()
        sessions._CLOSE_TASK['completion'].wait(2)


def test_close_waits_for_existing_restart_and_suppresses_new_preparation(monkeypatch):
    finished = threading.Event()
    restart = {'completion': finished, 'result': {'status': 'restart_pending'}}
    monkeypatch.setattr(engine, '_WORKER_RESTART', restart)
    monkeypatch.setattr(engine, '_ENGINE_RUNTIMES', {})
    monkeypatch.setattr(sessions, '_CLOSE_TASK', None)
    monkeypatch.setattr(sessions, '_runtimes', {})
    monkeypatch.setattr(sessions, '_owners', {})
    monkeypatch.setattr(sessions, '_preparation_pending', {})
    calls = []
    monkeypatch.setattr(sessions, '_stop_preparation_worker', lambda **kw: calls.append('close') or set())
    try:
        sessions.close(defer_native=True)
        task = sessions._CLOSE_TASK
        sessions._wake_preparation(17)
        assert sessions._preparation_pending == {}
        assert not task['completion'].is_set() and not calls
        assert engine.restart_ovrtx_workers()['status'] == 'restart_pending'
        assert engine._WORKER_RESTART is restart
        finished.set()
        assert task['completion'].wait(2)
        assert calls == ['close'] and task['result']['status'] == 'closed'
    finally:
        finished.set()
        sessions._CLOSE_TASK['completion'].wait(2)
