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
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402


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
    engine._ENGINE_RUNTIMES[id(dead)] = {
        "signal_id": "OVRTX_EXAMPLE:dead",
        "render_loop": runtime,
        "render_thread": runtime,
    }
    _tracked(dead, live)
    try:
        result = engine.viewport_session_statuses()

        assert result["active_session_count"] == 1
        assert result["sessions"] == [{"status": "live"}]
        # Pruned via set rebuild (dead-wrapper hash/eq is unreliable),
        # sidecar runtime stopped with the bounded thread stop, and the
        # sidecar entry consumed.
        remaining = list(engine._ACTIVE_VIEWPORT_ENGINES)
        assert live in remaining and len(remaining) == 1
        assert runtime.stop_requests == 1
        assert runtime.thread_stops == 1
        assert id(dead) not in engine._ENGINE_RUNTIMES
        # A second pass is clean: nothing dead remains.
        assert engine.viewport_session_statuses()["active_session_count"] == 1
    finally:
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

    assert engine._teardown_engine_runtime(runtime) is True
    assert events == ["join", "teardown"]


def test_teardown_engine_runtime_tolerates_missing_and_partial_state() -> None:
    engine._teardown_engine_runtime(None)
    engine._teardown_engine_runtime({})
    runtime = _StoppableRuntime()
    state = {"signal_id": "x", "render_loop": runtime, "render_thread": None}
    engine._teardown_engine_runtime(state)
    assert runtime.stop_requests == 1
    assert state["render_loop"] is None
