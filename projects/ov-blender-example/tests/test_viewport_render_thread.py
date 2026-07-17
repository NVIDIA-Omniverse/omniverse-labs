# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.viewport_render_thread import (
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    RenderThreadError,
    RenderThreadRejectedError,
    RenderThreadResult,
    RenderThreadTimeoutError,
    ViewportRenderThread,
)


WAIT_S = 5.0


def _stopped(thread: ViewportRenderThread) -> None:
    outcome = thread.stop(timeout=WAIT_S)
    assert outcome["joined"] is True


def test_commands_run_in_order_on_one_named_daemon_thread() -> None:
    thread = ViewportRenderThread("OVRTX:abc123")
    thread.start()
    seen: list[tuple[int, int]] = []
    for index in range(8):
        thread.submit(lambda index=index: seen.append((index, threading.get_ident())))
    ident = thread.call(threading.get_ident, label="ident").result(WAIT_S)

    assert [index for index, _ in seen] == list(range(8))
    assert {command_ident for _, command_ident in seen} == {ident}
    assert ident != threading.get_ident()
    diagnostics = thread.diagnostics()
    assert diagnostics["name"] == "ovrtx-render-OVRTX:abc123"
    assert diagnostics["thread_ident"] == ident
    assert diagnostics["daemon"] is True
    assert diagnostics["status"] == STATUS_RUNNING
    assert thread.is_alive() is True
    _stopped(thread)


def test_call_returns_value_and_delivers_exception_without_failing_thread() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()

    assert thread.call(lambda: 21 * 2).result(WAIT_S) == 42

    failing = thread.call(lambda: (_ for _ in ()).throw(ValueError("rpc failed")))
    with pytest.raises(ValueError, match="rpc failed"):
        failing.result(WAIT_S)
    assert isinstance(failing.exception(WAIT_S), ValueError)

    # The waiting caller handled the exception; the thread keeps serving.
    assert thread.call(lambda: "still-alive").result(WAIT_S) == "still-alive"
    assert thread.status() == STATUS_RUNNING
    diagnostics = thread.diagnostics()
    assert diagnostics["commands_failed"] == 1
    assert diagnostics["commands_completed"] == 2
    _stopped(thread)


def test_submit_exception_transitions_to_failed_and_drains_pending() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    gate = threading.Event()
    thread.submit(gate.wait, label="gate")

    def _boom() -> None:
        raise RuntimeError("unhandled loop failure")

    thread.submit(_boom, label="boom")
    pending = thread.call(lambda: "never-runs", label="pending")
    gate.set()

    with pytest.raises(RenderThreadRejectedError, match="unhandled loop failure"):
        pending.result(WAIT_S)
    deadline = time.monotonic() + WAIT_S
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert thread.is_alive() is False
    assert thread.status() == STATUS_FAILED
    assert "RuntimeError: unhandled loop failure" in thread.failure()
    with pytest.raises(RenderThreadRejectedError, match="unhandled loop failure"):
        thread.submit(lambda: None)

    # Join stays possible after failure.
    outcome = thread.stop(timeout=WAIT_S)
    assert outcome["joined"] is True
    assert outcome["status"] == STATUS_FAILED
    assert outcome["leaked_thread"] is False


def test_stop_join_is_bounded_under_hung_rpc_and_publishes_leaked_thread() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    hang = threading.Event()
    thread.submit(hang.wait, label="hung-rpc")

    started = time.monotonic()
    outcome = thread.stop(timeout=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert outcome["joined"] is False
    assert outcome["leaked_thread"] is True
    assert outcome["status"] == STATUS_FAILED
    assert "leaked_thread" in outcome["failure"]
    diagnostics = thread.diagnostics()
    assert diagnostics["leaked_thread"] is True
    assert diagnostics["alive"] is True
    with pytest.raises(RenderThreadRejectedError):
        thread.call(lambda: None)
    hang.set()  # release the abandoned daemon thread


def test_stop_waits_for_inflight_work_and_runs_queued_teardown_once() -> None:
    thread = ViewportRenderThread("sid", join_timeout_seconds=6.0)
    thread.start()
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []

    def _operation() -> None:
        entered.set()
        release.wait()
        events.append("operation")

    thread.submit(_operation)
    assert entered.wait(WAIT_S)
    thread.submit(lambda: events.append("teardown"), label="session-teardown")
    # The operation outlives the former production join deadline (5s) but
    # still completes within this session's configured deadline.
    threading.Timer(5.05, release.set).start()

    started = time.monotonic()
    outcome = thread.stop()

    assert outcome["joined"] is True
    assert outcome["leaked_thread"] is False
    assert events == ["operation", "teardown"]
    assert time.monotonic() - started >= 5.0


def test_commands_after_stop_are_rejected_with_typed_error() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    _stopped(thread)

    with pytest.raises(RenderThreadRejectedError):
        thread.submit(lambda: None)
    with pytest.raises(RenderThreadRejectedError):
        thread.call(lambda: None)
    assert thread.diagnostics()["commands_rejected"] == 2
    assert thread.status() == STATUS_STOPPED


def test_stop_before_start_rejects_queued_futures_and_is_idempotent() -> None:
    thread = ViewportRenderThread("sid")
    queued = thread.call(lambda: "queued-before-start")
    assert thread.status() == STATUS_CREATED

    first = thread.stop(timeout=0.1)
    second = thread.stop(timeout=0.1)

    assert first["joined"] is True and second["joined"] is True
    assert thread.status() == STATUS_STOPPED
    with pytest.raises(RenderThreadRejectedError):
        queued.result(WAIT_S)
    with pytest.raises(RenderThreadError, match="already started or stopped"):
        thread.start()


def test_double_start_raises() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    with pytest.raises(RenderThreadError, match="already started or stopped"):
        thread.start()
    _stopped(thread)


def test_result_wait_timeout_raises_typed_timeout() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    hang = threading.Event()
    slow = thread.call(hang.wait, label="slow")

    with pytest.raises(RenderThreadTimeoutError):
        slow.result(timeout=0.05)
    with pytest.raises(TimeoutError):
        slow.exception(timeout=0.05)
    assert slow.done() is False

    hang.set()
    assert slow.result(WAIT_S) is True
    _stopped(thread)


def test_stop_requested_from_inside_a_command_does_not_deadlock() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()

    inner = thread.call(lambda: thread.stop(timeout=0.1)).result(WAIT_S)

    assert inner["joined"] is False
    assert inner["leaked_thread"] is False
    outcome = thread.stop(timeout=WAIT_S)
    assert outcome["joined"] is True
    assert outcome["status"] == STATUS_STOPPED


def test_diagnostics_expose_identity_and_timing_metadata() -> None:
    thread = ViewportRenderThread("sid", join_timeout_seconds=2.5)
    thread.start()
    thread.call(lambda: time.sleep(0.01), label="tick").result(WAIT_S)
    _stopped(thread)

    diagnostics = thread.diagnostics()
    assert diagnostics["session_id"] == "sid"
    assert diagnostics["status"] == STATUS_STOPPED
    assert diagnostics["alive"] is False
    assert diagnostics["thread_ident"] not in (0, threading.get_ident())
    assert diagnostics["join_timeout_seconds"] == 2.5
    assert diagnostics["started_time_ns"] > 0
    assert diagnostics["started_monotonic_ns"] > 0
    assert diagnostics["ended_time_ns"] >= diagnostics["started_time_ns"]
    assert diagnostics["commands_submitted"] == 1
    assert diagnostics["commands_completed"] == 1
    assert diagnostics["pending_commands"] == 0
    assert diagnostics["last_command_label"] == "tick"
    assert diagnostics["last_command_ms"] >= 0.0


def test_failed_thread_start_rejects_queued_futures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = ViewportRenderThread("sid")
    queued = thread.call(lambda: "queued-before-start")

    def _fail_start(self: threading.Thread) -> None:
        raise RuntimeError("no more threads")

    monkeypatch.setattr(threading.Thread, "start", _fail_start)
    with pytest.raises(RuntimeError, match="no more threads"):
        thread.start()

    assert thread.status() == STATUS_FAILED
    assert "no more threads" in thread.failure()
    with pytest.raises(RenderThreadRejectedError, match="failed to start"):
        queued.result(WAIT_S)
    with pytest.raises(RenderThreadRejectedError):
        thread.submit(lambda: None)
    assert thread.diagnostics()["pending_commands"] == 0


def test_pending_commands_diagnostic_excludes_stop_sentinels() -> None:
    thread = ViewportRenderThread("sid")
    thread.start()
    entered = threading.Event()
    hang = threading.Event()

    def _hung_rpc() -> None:
        entered.set()
        hang.wait()

    thread.submit(_hung_rpc, label="hung-rpc")
    queued = thread.call(lambda: "queued-behind-hang", label="queued")
    assert entered.wait(WAIT_S) is True  # the hung command is in flight

    # Two stops each enqueue a sentinel; only the one real queued command
    # (not the in-flight hung one, not the sentinels) is pending.
    first = thread.stop(timeout=0.05)
    second = thread.stop(timeout=0.05)

    assert first["leaked_thread"] is True and second["leaked_thread"] is True
    assert thread.diagnostics()["pending_commands"] == 1

    # Abandoned-daemon semantics: on release the thread drains the queue and
    # exits on the sentinel.
    hang.set()
    assert queued.result(WAIT_S) == "queued-behind-hang"
    deadline = time.monotonic() + WAIT_S
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert thread.is_alive() is False
    assert thread.diagnostics()["pending_commands"] == 0


def test_non_callable_command_is_a_type_error() -> None:
    thread = ViewportRenderThread("sid")
    with pytest.raises(TypeError, match="callable"):
        thread.submit("not-callable")  # type: ignore[arg-type]
    thread.stop(timeout=0.1)


def test_result_wrapper_resolves_once_and_reports_done() -> None:
    result = RenderThreadResult()
    assert result.done() is False
    assert result.wait(0.01) is False
    result._resolve("value")
    assert result.done() is True
    assert result.result(0.0) == "value"
    assert result.exception(0.0) is None
