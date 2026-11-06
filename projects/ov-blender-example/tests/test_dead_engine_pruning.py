# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dead render-engine wrappers must be pruned, not crash consumers.

Blender frees an engine's StructRNA at destruction (viewport closed,
workspace switched, F12 render finished) while the Python wrapper lingers
until GC — and ``bpy_struct`` validates the RNA pointer before ANY
attribute lookup, so even ``getattr(engine, "_render_loop", None)`` raises
``ReferenceError`` on a dead wrapper (panel-draw error storm plus silently
skipped thread teardown, 2026-07-07). Teardown-critical handles therefore
live in the ``_ENGINE_RUNTIMES`` sidecar keyed by ``id(engine)``."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example.viewport_render_thread import ViewportRenderThread


class _DeadEngine:
    """Wrapper whose RNA has been freed: EVERY attribute access raises."""

    def __getattribute__(self, name: str):
        raise ReferenceError(
            "StructRNA of type OvrtxExampleRenderEngine has been removed"
        )


class _StoppableRuntime:
    def __init__(self) -> None:
        self.stop_requests = 0
        self.thread_stops = 0

    def request_stop(self) -> None:
        self.stop_requests += 1

    def stop(self) -> dict:
        self.thread_stops += 1
        return {"leaked_thread": False}


class _LiveEngine:
    def as_pointer(self) -> int:
        return 1

    def _viewport_session_status(self) -> dict:
        return {"status": "live"}


def _tracked(*engines):
    engine._ACTIVE_VIEWPORT_ENGINES.clear()
    for item in engines:
        engine._ACTIVE_VIEWPORT_ENGINES.add(item)


def test_tracking_preserves_unconfirmed_sidecar_on_object_id_reuse() -> None:
    replacement = _LiveEngine()
    unconfirmed = {"stop_confirmed": False, "render_thread": object()}
    engine._ENGINE_RUNTIMES[id(replacement)] = unconfirmed
    try:
        engine._track_viewport_engine(replacement)

        assert engine._ENGINE_RUNTIMES[id(replacement)] is not unconfirmed
        assert unconfirmed in engine._ENGINE_RUNTIMES.values()
        assert any(key < 0 for key in engine._ENGINE_RUNTIMES)
    finally:
        engine._ACTIVE_VIEWPORT_ENGINES.clear()
        engine._ENGINE_RUNTIMES.clear()


def test_statuses_skip_prune_and_stop_dead_engines() -> None:
    dead = _DeadEngine()
    live = _LiveEngine()
    runtime = _StoppableRuntime()
    entered, release = threading.Event(), threading.Event()
    original_stop = runtime.stop

    def blocked_stop():
        entered.set()
        assert release.wait(2)
        return original_stop()

    runtime.stop = blocked_stop
    sidecar = {
        "signal_id": "OVRTX_EXAMPLE:dead",
        "render_loop": runtime,
        "render_thread": runtime,
    }
    engine._ENGINE_RUNTIMES[id(dead)] = sidecar
    _tracked(dead, live)
    try:
        result = engine.viewport_session_statuses()

        assert result["active_session_count"] == 1
        assert result["sessions"] == [{"status": "live"}]
        # Pruning returns while native cleanup is still blocked. The plain
        # Python sidecar retains ownership until the cleanup actually finishes.
        remaining = list(engine._ACTIVE_VIEWPORT_ENGINES)
        assert live in remaining and len(remaining) == 1
        assert entered.wait(2)
        assert runtime.stop_requests == 1
        assert runtime.thread_stops == 0
        assert engine._ENGINE_RUNTIMES[id(dead)] is sidecar
        assert sidecar["stop_pending"] is True
        release.set()
        assert sidecar["cleanup_completion"].wait(2)
        assert engine._finish_pending_engine_teardowns() is True
        assert runtime.thread_stops == 1
        assert id(dead) not in engine._ENGINE_RUNTIMES
        # A second pass is clean: nothing dead remains.
        assert engine.viewport_session_statuses()["active_session_count"] == 1
        assert runtime.stop_requests == runtime.thread_stops == 1
    finally:
        release.set()
        completion = sidecar.get("cleanup_completion")
        if completion is not None:
            assert completion.wait(2)
        engine._ACTIVE_VIEWPORT_ENGINES.clear()
        engine._ENGINE_RUNTIMES.clear()


def test_final_render_host_ignores_dead_engines() -> None:
    dead = _DeadEngine()
    _tracked(dead)
    try:
        assert engine._viewport_final_render_host(
            SimpleNamespace(session_uid=7)
        ) == (None, True)
        assert list(engine._ACTIVE_VIEWPORT_ENGINES) == []
    finally:
        engine._ACTIVE_VIEWPORT_ENGINES.clear()
        engine._ENGINE_RUNTIMES.clear()


def test_final_render_host_reports_unconfirmed_dead_engine_stop() -> None:
    dead = _DeadEngine()

    class Thread:
        def stop(self) -> dict[str, object]:
            return {"joined": False, "leaked_thread": True}

    engine._ENGINE_RUNTIMES[id(dead)] = {
        "authored": False,
        "render_thread": Thread(),
    }
    _tracked(dead)
    try:
        assert engine._viewport_final_render_host(
            SimpleNamespace(session_uid=7)
        ) == (None, False)
        assert engine._viewport_final_render_host(
            SimpleNamespace(session_uid=7)
        ) == (None, False)
    finally:
        engine._ACTIVE_VIEWPORT_ENGINES.clear()
        engine._ENGINE_RUNTIMES.clear()


def test_confirmed_dead_thread_runs_sidecar_teardown_on_caller() -> None:
    events: list[str] = []
    state = {"ran": False}

    def teardown() -> None:
        state["ran"] = True
        events.append("teardown")

    class Thread:
        def submit(self, _fn: object, *, label: str = "") -> None:
            assert label == "session-teardown"
            raise RuntimeError("thread already stopped")

        def stop(self) -> dict[str, object]:
            events.append("join")
            return {"joined": True, "leaked_thread": False}

    runtime = {
        "render_thread": Thread(),
        "render_loop": None,
        "teardown": teardown,
        "teardown_state": state,
    }

    engine._teardown_engine_runtime(runtime)
    assert runtime["cleanup_completion"].wait(2)
    assert engine._teardown_engine_runtime(runtime) is True
    assert events == ["join", "teardown"]


def test_teardown_engine_runtime_tolerates_missing_and_partial_state() -> None:
    engine._teardown_engine_runtime(None)
    engine._teardown_engine_runtime({})
    runtime = _StoppableRuntime()
    state = {"signal_id": "x", "render_loop": runtime, "render_thread": None}
    engine._teardown_engine_runtime(state)
    assert state["cleanup_completion"].wait(2)
    assert runtime.stop_requests == 1
    assert state["render_loop"] is None


def test_owner_thread_finalizer_defers_join_without_poisoning_reuse(monkeypatch):
    monkeypatch.setattr(engine, "_DIRECT_VIEWPORT_REUSE_BLOCKED", False)
    thread = ViewportRenderThread("owner-finalizer", join_timeout_seconds=2)
    teardown_entered = threading.Event()
    finish_teardown = threading.Event()
    state = {"ran": False, "errors": []}
    calls = []

    def teardown():
        calls.append(threading.get_ident())
        teardown_entered.set()
        assert finish_teardown.wait(2)
        state["ran"] = True

    runtime = {"render_thread": thread, "teardown": teardown, "teardown_state": state}
    engine._ENGINE_RUNTIMES[-123] = runtime

    class Finalizer:
        def __del__(self):
            engine._teardown_engine_runtime(runtime)

    def release_last_reference():
        wrapper = Finalizer()
        del wrapper

    thread.start()
    try:
        thread.call(release_last_reference).result(2)
        assert teardown_entered.wait(2)
        assert runtime.get("stop_confirmed") is not False
        assert runtime.get("stop_pending") is True
        assert engine._DIRECT_VIEWPORT_REUSE_BLOCKED is False
        assert runtime["stop_outcome"]["join_deferred"] is True
        finish_teardown.set()
        assert runtime["cleanup_completion"].wait(2)
        assert engine._finish_pending_engine_teardowns() is True
        assert runtime["stop_confirmed"] is True
        assert runtime["stop_pending"] is False
        assert -123 not in engine._ENGINE_RUNTIMES
        assert state["ran"] is True
        assert len(calls) == 1
        assert thread.diagnostics()["commands_completed"] == 2
        assert thread.diagnostics()["failure"] == ""
        assert engine._DIRECT_VIEWPORT_REUSE_BLOCKED is False
    finally:
        finish_teardown.set()
        thread.stop(2)
        engine._ENGINE_RUNTIMES.pop(-123, None)


def test_live_engine_self_stop_retains_sidecar_until_external_join(monkeypatch):
    from test_engine_nonblocking_view_callbacks import _load_engine_with_fake_bpy

    module = _load_engine_with_fake_bpy(monkeypatch)
    wrapper = module.OvrtxExampleRenderEngine()
    thread = ViewportRenderThread("live-owner-stop", join_timeout_seconds=2)
    teardown_entered = threading.Event()
    release = threading.Event()
    state = {"ran": False, "errors": []}

    def teardown():
        teardown_entered.set()
        assert release.wait(2)
        state["ran"] = True

    wrapper._render_thread = thread
    wrapper._render_loop = SimpleNamespace(request_stop=lambda: None)
    runtime = {
        "render_thread": thread, "render_loop": wrapper._render_loop,
        "teardown": teardown, "teardown_state": state,
    }
    module._ENGINE_RUNTIMES[id(wrapper)] = runtime
    thread.start()
    try:
        outcome = thread.call(lambda: wrapper._stop_render_loop(teardown=teardown)).result(2)
        assert outcome["join_deferred"] is True
        assert teardown_entered.wait(2)
        assert runtime["stop_pending"] is True
        assert runtime["stop_confirmed"] is None
        assert module._DIRECT_VIEWPORT_REUSE_BLOCKED is False
        # The wrapper may disappear while this command still owns cleanup.
        module._untrack_viewport_engine(wrapper, runtime)
        release.set()
        assert runtime["cleanup_completion"].wait(2)
        # Repeated live teardown has cleared wrapper handles but must still
        # confirm the retained owner thread, not report a false no-thread join.
        assert wrapper._stop_render_loop()["joined"] is True
        assert module._finish_pending_engine_teardowns() is True
        assert runtime["stop_confirmed"] is True
        assert state["ran"] is True
        assert thread.diagnostics()["commands_completed"] == 2
        assert thread.diagnostics()["commands_failed"] == 0
    finally:
        release.set()
        thread.stop(2)
        module._ENGINE_RUNTIMES.clear()


def test_pending_join_retains_actual_owner_thread_cleanup_errors(monkeypatch):
    monkeypatch.setattr(engine, "_DIRECT_VIEWPORT_REUSE_BLOCKED", False)
    thread = ViewportRenderThread("errored-owner-cleanup", join_timeout_seconds=2)
    state = {"ran": False, "errors": []}

    def teardown():
        state["ran"] = True
        state["errors"].append("controller_shutdown: rejected")

    runtime = {"render_thread": thread, "teardown": teardown, "teardown_state": state}
    engine._ENGINE_RUNTIMES[-124] = runtime
    thread.start()
    try:
        thread.call(lambda: engine._teardown_engine_runtime(runtime)).result(2)
        assert runtime["cleanup_completion"].wait(2)
        assert engine._finish_pending_engine_teardowns() is False
        assert runtime["stop_confirmed"] is False
        assert runtime["teardown_state"] is state
        assert runtime["stop_outcome"]["teardown_errors"] == state["errors"]
        assert engine._ENGINE_RUNTIMES[-124] is runtime
        assert engine._DIRECT_VIEWPORT_REUSE_BLOCKED is True
        assert thread.diagnostics()["commands_completed"] == 2
        assert thread.diagnostics()["failure"] == ""
    finally:
        thread.stop(2)
        engine._ENGINE_RUNTIMES.pop(-124, None)


def test_pending_teardown_survives_wrapper_id_reuse(monkeypatch):
    replacement = _LiveEngine()
    pending = {"stop_confirmed": None, "stop_pending": True, "render_thread": object()}
    engine._ENGINE_RUNTIMES[id(replacement)] = pending
    try:
        engine._track_viewport_engine(replacement)
        assert engine._ENGINE_RUNTIMES[id(replacement)] is not pending
        assert pending in engine._ENGINE_RUNTIMES.values()
    finally:
        engine._ACTIVE_VIEWPORT_ENGINES.clear()
        engine._ENGINE_RUNTIMES.clear()


def test_real_join_deadline_remains_terminal_after_thread_later_finishes(monkeypatch):
    monkeypatch.setattr(engine, "_DIRECT_VIEWPORT_REUSE_BLOCKED", False)
    entered = threading.Event()
    release = threading.Event()
    thread = ViewportRenderThread("real-deadline", join_timeout_seconds=0.01)
    runtime = {"render_thread": thread}
    thread.start()
    try:
        thread.submit(lambda: (entered.set(), release.wait(2)))
        assert entered.wait(2)
        assert engine._teardown_engine_runtime(runtime) is False
        assert runtime["cleanup_completion"].wait(2)
        assert runtime["stop_confirmed"] is False
        assert engine._DIRECT_VIEWPORT_REUSE_BLOCKED is True
        release.set()
        assert thread.stop(2)["joined"] is True
        assert engine._teardown_engine_runtime(runtime) is False
        assert runtime["stop_confirmed"] is False
    finally:
        release.set()
        thread.stop(2)


def test_cleanup_fallback_never_blocks_callback_and_repeated_requests_coalesce():
    import time
    entered = threading.Event()
    release = threading.Event()
    calls = []
    state = {"ran": False, "errors": []}
    def teardown():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        state["ran"] = True
    runtime = {"teardown": teardown, "teardown_state": state}
    try:
        started = time.monotonic()
        assert engine._teardown_engine_runtime(runtime) is False
        assert time.monotonic() - started < 0.1
        assert entered.wait(2)
        owner = runtime['cleanup_thread']
        for _ in range(100):
            assert engine._teardown_engine_runtime(runtime) is False
        assert runtime['cleanup_thread'] is owner
        assert runtime['stop_confirmed'] is None
        assert engine._viewport_session_end_in_progress()
        release.set()
        assert runtime['cleanup_completion'].wait(2)
        assert runtime['stop_confirmed'] is True
        assert runtime['stop_pending'] is False
        assert state['ran'] is True
        assert len(calls) == 1 and calls[0] != threading.get_ident()
    finally:
        release.set()
        runtime['cleanup_completion'].wait(2)
