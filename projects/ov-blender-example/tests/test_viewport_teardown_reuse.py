# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport teardown keeps the runtime reusable (n-11-02).

F12, shading changes and file loads end viewport sessions from places that
cannot write Blender data (the render job thread, panel drawing), and can
free the engine while its render thread is starting a session. None of that
is an unconfirmed teardown: the thread joined, so the runtime stays usable.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example  # noqa: E402,F401
from ovrtx_blender_example import interactive_operator_state  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions  # noqa: E402
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402

WAIT_S = 5.0


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        pass

    def report(self, levels: set[str], message: str) -> None:
        pass


class _Timers:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, fn, first_interval: float = 0.0) -> None:
        self.registered.append(fn)

    def is_registered(self, fn) -> bool:
        return fn in self.registered

    def run_all(self) -> None:
        pending, self.registered = self.registered, []
        for fn in pending:
            fn()


@pytest.fixture
def timers(monkeypatch: pytest.MonkeyPatch) -> _Timers:
    timers = _Timers()
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=timers),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    return timers


def _load_engine(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_teardown_reuse_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _request(module):
    return module.RenderRequest(
        input_usd_path="/tmp/scene.usda",
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=2,
        height=2,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        camera_matrix=((1.0, 0, 0, 0), (0, 1.0, 0, 0), (0, 0, 1.0, 0), (0, 0, 0, 1.0)),
        worker_command="worker",
        native_client_module="client",
    )


class _Lock:
    """Playback lock whose release writes Blender data (main thread only)."""

    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear(self, *, reason: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("physics authoring snapshots require Blender's main thread")
        self.cleared.append(reason)


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            out["error"] = exc

    worker = threading.Thread(target=run, name="render-job")
    worker.start()
    worker.join(WAIT_S)
    if "error" in out:
        raise out["error"]
    return out.get("value")


def test_detach_on_render_job_thread_defers_lock_release(timers: _Timers, monkeypatch) -> None:
    lock = _Lock()
    runtime = SimpleNamespace(viewport_ids={"vp"}, playback_lock=lock,
                              detach=lambda vid: runtime.viewport_ids.discard(vid))
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {991: runtime})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)

    assert _in_thread(lambda: scene_generation_sessions.detach_viewport("vp")) is True

    assert lock.cleared == [] and len(timers.registered) == 1
    timers.run_all()
    assert lock.cleared == ["last_viewport_detached"]


def test_deferred_lock_release_skips_a_reattached_runtime(timers: _Timers, monkeypatch) -> None:
    lock = _Lock()
    runtime = SimpleNamespace(viewport_ids={"vp"}, playback_lock=lock,
                              detach=lambda vid: runtime.viewport_ids.discard(vid))
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {992: runtime})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)

    _in_thread(lambda: scene_generation_sessions.detach_viewport("vp"))
    runtime.viewport_ids.add("vp2")
    timers.run_all()

    assert lock.cleared == []


def test_restricted_context_lock_release_retries_from_timer(timers: _Timers) -> None:
    attempts: list[str] = []

    class DrawContextLock:
        def clear(self, *, reason: str) -> None:
            attempts.append(reason)
            if len(attempts) == 1:
                raise AttributeError("Writing to ID classes in this context is not allowed")

    interactive_operator_state.clear_playback_lock_when_writable(DrawContextLock(), reason="r")

    assert attempts == ["r"] and len(timers.registered) == 1
    timers.run_all()
    assert attempts == ["r", "r"]


def test_dead_engine_teardown_on_render_job_thread_stays_confirmed(
    timers: _Timers, monkeypatch
) -> None:
    """F12's render() prunes a freed viewport engine off the main thread."""

    module = _load_engine(monkeypatch)
    lock = _Lock()
    generation_runtime = SimpleNamespace(
        viewport_ids={"OVRTX_EXAMPLE:dead"}, playback_lock=lock, reuse_blocked=False,
        detach=lambda vid: generation_runtime.viewport_ids.discard(vid),
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {993: generation_runtime})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)
    blocked: list[object] = []
    monkeypatch.setattr(scene_generation_sessions, "fail_closed_runtime_reuse", blocked.append)
    entered, release = threading.Event(), threading.Event()

    class Thread:
        def submit(self, fn, *, label: str = "") -> None:
            fn()

        def stop(self) -> dict[str, object]:
            # A thread that died on the freed engine still joined.
            entered.set()
            assert release.wait(WAIT_S)
            return {"status": "failed", "joined": True, "leaked_thread": False}

    sidecar = {
        "signal_id": "OVRTX_EXAMPLE:dead",
        "authored": True,
        "generation_runtime": generation_runtime,
        "render_thread": Thread(),
        "render_loop": None,
        "teardown": lambda: None,
        "teardown_state": {"ran": True, "errors": []},
    }

    try:
        assert _in_thread(lambda: module._teardown_engine_runtime(sidecar)) is False
        assert entered.wait(WAIT_S)
        assert sidecar["stop_pending"] is True and sidecar["stop_confirmed"] is None
        assert blocked == [] and lock.cleared == []
        release.set()
        assert sidecar["cleanup_completion"].wait(WAIT_S)
        assert module._teardown_engine_runtime(sidecar) is True
        assert sidecar["stop_confirmed"] is True
        assert blocked == [] and generation_runtime.viewport_ids == set()
        timers.run_all()
        assert lock.cleared == ["last_viewport_detached"]
    finally:
        release.set()
        assert sidecar["cleanup_completion"].wait(WAIT_S)


def test_session_end_blocks_reentrant_view_update_start(timers: _Timers, monkeypatch) -> None:
    """A session end evaluates the view layer; Blender re-enters view_update."""

    module = _load_engine(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    entered, release = threading.Event(), threading.Event()

    def shutdown():
        entered.set()
        assert release.wait(WAIT_S)

    engine._runtime_scheduler = SimpleNamespace(shutdown=shutdown)
    starts: list[object] = []
    engine._begin_async_viewport_session = lambda *args, **kwargs: starts.append(args)
    monkeypatch.setattr(
        module.scene_generation_sessions, "is_authoring", lambda _scene: False
    )
    depsgraph = SimpleNamespace(scene=SimpleNamespace())

    def outputs(**_kwargs) -> None:
        # Pose mirror / playback-lock release -> view_layer.update() -> view_update.
        engine.view_update(SimpleNamespace(), depsgraph)

    engine._write_viewport_session_outputs = outputs
    try:
        assert engine._end_viewport_session(
            module.ViewportSessionEndReason.ENGINE_DESTROYED
        ) is False
        assert entered.wait(WAIT_S)
        sidecar = module._ENGINE_RUNTIMES[id(engine)]
        assert sidecar["stop_pending"] is True
        assert module._viewport_session_end_in_progress() is True
        assert starts == []
        assert module._restore_viewport_redraw in timers.registered
        release.set()
        assert sidecar["cleanup_completion"].wait(WAIT_S)
        assert module._teardown_engine_runtime(sidecar) is True
        assert module._viewport_session_end_in_progress() is False
        assert starts == []
    finally:
        release.set()
        sidecar = module._ENGINE_RUNTIMES.get(id(engine))
        if sidecar is not None:
            assert sidecar["cleanup_completion"].wait(WAIT_S)
        module._ENGINE_RUNTIMES.pop(id(engine), None)


def test_freed_engine_during_session_start_is_a_release_not_a_thread_failure(
    timers: _Timers, monkeypatch
) -> None:
    module = _load_engine(monkeypatch)
    monkeypatch.setattr(controller_module, "_runtime_client_from_request", lambda _r: None)
    controller = controller_module.OvrtxSessionController()
    scheduler = RuntimeScheduler(config_factory=lambda _path: SimpleNamespace(enabled=False))
    runtime = SimpleNamespace(ovrtx=SimpleNamespace(controller=controller), scheduler=scheduler,
                              playback_lock=SimpleNamespace())
    monkeypatch.setattr(module.scene_generation_sessions, "runtime_for_viewport",
                        lambda *args, **kwargs: runtime)
    blocked: list[object] = []
    monkeypatch.setattr(module, "_fail_closed_runtime_reuse", blocked.append)
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_teardown_state = lambda: (lambda: None, {"ran": False, "errors": []})

    def ensure(_request):
        # Activation finished, but Blender freed the engine meanwhile.
        raise ReferenceError("StructRNA of type OvrtxExampleRenderEngine has been removed")

    engine._thread_ensure_session = ensure
    request = _request(module)
    engine._ensure_render_loop(request, scene=SimpleNamespace(), authored=True)
    thread = engine._render_thread
    sidecar = module._ENGINE_RUNTIMES[id(engine)]
    try:
        engine._camera_mailbox.write(module.viewport_handoff.snapshot_from_render_request(request))
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        assert frame.status == module.viewport_handoff.FRAME_STATUS_FAILED
        assert "Viewport engine was released" in frame.detail
        time.sleep(0.05)
        assert thread.status() == module.viewport_render_thread.STATUS_RUNNING
        assert thread.failure() == ""

        module._teardown_engine_runtime(sidecar)
        assert sidecar["cleanup_completion"].wait(2)
        assert module._teardown_engine_runtime(sidecar) is True
        assert sidecar["stop_confirmed"] is True
        assert thread.status() == module.viewport_render_thread.STATUS_STOPPED
        assert blocked == []
    finally:
        if thread.is_alive():
            thread.stop(timeout=WAIT_S)
        engine._render_loop = engine._render_thread = engine._ovrtx_session_controller = None
        module._ENGINE_RUNTIMES.pop(id(engine), None)


def test_render_init_generation_is_serialized_with_main_thread_observation(monkeypatch) -> None:
    """render_init runs on the job thread; handlers must not interleave with it."""

    inside = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def generation(scene, *, work_root, explicit_physics_restart):
        order.append("generation-start")
        inside.set()
        assert release.wait(WAIT_S)
        order.append("generation-end")

    def observe(scene, *, explicit_physics_restart):
        order.append("observe")

    monkeypatch.setattr(scene_generation_sessions, "_generation_for_scene", generation)
    monkeypatch.setattr(scene_generation_sessions, "_observe_physics_edits", observe)
    job = threading.Thread(
        target=lambda: scene_generation_sessions.generation_for_scene(SimpleNamespace()),
        name="render-job",
    )
    job.start()
    assert inside.wait(WAIT_S)
    releaser = threading.Timer(0.1, release.set)
    releaser.start()
    scene_generation_sessions.observe_physics_edits(SimpleNamespace())
    job.join(WAIT_S)

    assert order == ["generation-start", "generation-end", "observe"]


def test_main_thread_observation_gives_up_on_a_stuck_final_render(monkeypatch) -> None:
    monkeypatch.setattr(scene_generation_sessions, "_SESSION_STATE_MAIN_THREAD_WAIT_S", 0.05)
    held = threading.Event()
    done = threading.Event()

    def hold():
        with scene_generation_sessions._SessionState():
            held.set()
            done.wait(WAIT_S)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(WAIT_S)
        with pytest.raises(RuntimeError, match="busy"):
            scene_generation_sessions.observe_physics_edits(SimpleNamespace())
    finally:
        done.set()
        holder.join(WAIT_S)
