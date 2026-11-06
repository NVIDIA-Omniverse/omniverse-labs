# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A final render after a file load waits for the previous scene's close.

Scene replacement in one process (render, ``read_homefile``, build, render)
closes the first scene's retained final-render runtime on a cleanup thread.
The second render must wait for that close instead of refusing, and a worker
stop slower than five seconds must still confirm cleanup.
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RuntimeServicesPreparingError  # noqa: E402

SCENE = SimpleNamespace(render=SimpleNamespace(engine=engine.ENGINE_ID))


@pytest.fixture
def close_state(monkeypatch):
    # Other suites may leave admission paused; a renderer restart always
    # resumes it before a scene close proceeds, so start from that state.
    monkeypatch.setattr(sessions, "_preparation_paused", False)
    sessions._stop_preparation_worker()
    monkeypatch.setattr(sessions, "_CLOSE_TASK", None)
    monkeypatch.setattr(sessions, "_runtimes", {})
    monkeypatch.setattr(sessions, "_owners", {})
    monkeypatch.setattr(sessions, "_unclosed_owners", [])
    monkeypatch.setattr(sessions, "_deferred_cleanup_paths", [])
    monkeypatch.setattr(sessions, "_physics_snapshots", {})
    monkeypatch.setattr(sessions, "_preparation_pending", {})
    monkeypatch.setattr(sessions, "_preparation_revisions", {})
    monkeypatch.setattr(engine, "_ENGINE_RUNTIMES", {})
    monkeypatch.setattr(engine, "_WORKER_RESTART", None)
    monkeypatch.setattr(engine, "_FINAL_RENDER_ACTIVE", False)
    yield sessions
    sessions._stop_preparation_worker()


class SlowRuntime:
    """A retained final-render runtime whose native worker stop is slow."""

    reuse_blocked = False
    playback_lock = SimpleNamespace(clear=lambda **_kwargs: None)

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.lifecycle_status = "open"

    def close(self) -> bool:
        time.sleep(self.seconds)
        self.lifecycle_status = "closed"
        return True


def closing_task(status: str = "closing", thread: threading.Thread | None = None) -> dict:
    return {"completion": threading.Event(), "result": {"status": status},
            "finalizers": [], "completed_finalizers": [], "thread": thread}


def finish_later(task: dict, status: str, delay: float) -> threading.Thread:
    def finish() -> None:
        time.sleep(delay)
        task["result"] = {"status": status}
        task["completion"].set()

    thread = threading.Thread(target=finish, name="ovrtx-scene-cleanup", daemon=True)
    task["thread"] = thread
    thread.start()
    return thread


def test_worker_stop_slower_than_five_seconds_confirms_cleanup(close_state) -> None:
    owner_closed = []
    runtime = SlowRuntime(5.5)
    close_state._runtimes[7] = runtime
    close_state._owners[7] = SimpleNamespace(close=lambda: owner_closed.append(True))

    close_state.close(defer_native=True)
    assert close_state._CLOSE_TASK["completion"].wait(30)

    assert close_state.close_status() == {"status": "closed"}
    assert runtime.lifecycle_status == "closed" and not runtime.reuse_blocked
    assert close_state._runtimes == {} and close_state._owners == {}
    assert owner_closed == [True]
    close_state._require_session_open()


def test_join_deadline_covers_each_reusable_runtime_stop(close_state, monkeypatch) -> None:
    monkeypatch.setattr(sessions, "_PREPARATION_JOIN_SECONDS", 1.0)
    monkeypatch.setattr(sessions, "_RUNTIME_CLOSE_JOIN_SECONDS", 10.0)
    blocked = SlowRuntime(0)
    blocked.reuse_blocked = True
    close_state._runtimes.update({1: SlowRuntime(0), 2: SlowRuntime(0), 3: blocked})

    assert sessions._preparation_join_timeout() == 21.0


def test_join_deadline_still_fails_closed_when_the_stop_never_finishes(close_state, monkeypatch) -> None:
    monkeypatch.setattr(sessions, "_PREPARATION_JOIN_SECONDS", 0.05)
    monkeypatch.setattr(sessions, "_RUNTIME_CLOSE_JOIN_SECONDS", 0.1)
    release = threading.Event()

    class StuckRuntime(SlowRuntime):
        def close(self) -> bool:
            release.wait(10)
            return super().close()

    stuck = StuckRuntime(0)
    close_state._runtimes[4] = stuck
    try:
        assert close_state._stop_preparation_worker(close_runtimes=True) == set()
        assert stuck.reuse_blocked
    finally:
        release.set()


def test_final_render_init_waits_for_deferred_close_then_prepares(close_state, monkeypatch) -> None:
    task = closing_task()
    monkeypatch.setattr(sessions, "_CLOSE_TASK", task)
    monkeypatch.setattr(engine, "_FINAL_RENDER_REQUEST", None)
    monkeypatch.setattr(engine, "_FINAL_RENDER_INIT_ERROR", None)

    prepared = SimpleNamespace(current_scene_generation=False)

    def prepare(_build):
        # The real preparation path refuses while the previous close is open.
        sessions._require_session_open()
        return prepared

    monkeypatch.setattr(engine, "_prepare_final_render_request", prepare)
    cleanup = finish_later(task, "closed", 0.2)

    engine._final_render_init_handler(SCENE)
    cleanup.join(2)

    assert engine._consume_final_render_request(SCENE) == (prepared, None)


def test_gui_job_thread_init_waits_for_deferred_close(close_state, monkeypatch) -> None:
    task = closing_task()
    monkeypatch.setattr(sessions, "_CLOSE_TASK", task)
    monkeypatch.setattr(engine, "_prepare_final_render_request",
                        lambda *_args: pytest.fail("job thread accessed Blender"))
    finish_later(task, "closed", 0.2)
    observed = {}

    def job() -> None:
        engine._final_render_init_handler(SCENE)
        observed["status"] = sessions.close_status()

    thread = threading.Thread(target=job)
    thread.start()
    thread.join(5)

    assert not thread.is_alive()
    assert observed["status"] == {"status": "closed"}


def test_final_render_init_still_refuses_a_failed_close(close_state, monkeypatch) -> None:
    task = closing_task()
    monkeypatch.setattr(sessions, "_CLOSE_TASK", task)
    monkeypatch.setattr(engine, "_prepare_final_render_request",
                        lambda _build: (sessions._require_session_open(), "prepared")[1])
    cleanup = finish_later(task, "failed", 0.05)

    engine._final_render_init_handler(SCENE)
    cleanup.join(2)

    request, error = engine._consume_final_render_request(SCENE)
    assert request is None
    assert isinstance(error, RuntimeError) and "cleanup failed" in str(error)


def test_final_render_init_wait_is_bounded(close_state, monkeypatch) -> None:
    monkeypatch.setattr(sessions, "_CLOSE_TASK", closing_task(thread=threading.Thread()))
    monkeypatch.setattr(engine, "FINAL_RENDER_CLOSE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(engine, "_prepare_final_render_request",
                        lambda _build: (sessions._require_session_open(), "prepared")[1])

    started = time.monotonic()
    engine._final_render_init_handler(SCENE)

    assert time.monotonic() - started < 2
    request, error = engine._consume_final_render_request(SCENE)
    assert request is None and isinstance(error, RuntimeServicesPreparingError)


def test_cleanup_thread_never_waits_for_its_own_close(close_state, monkeypatch) -> None:
    monkeypatch.setattr(sessions, "_CLOSE_TASK", closing_task(thread=threading.current_thread()))

    started = time.monotonic()
    assert sessions.wait_for_close(5) == {"status": "closing"}
    assert time.monotonic() - started < 1
