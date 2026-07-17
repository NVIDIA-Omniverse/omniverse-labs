# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-blocking ``view_update``/``view_draw`` callbacks (task02-04).

Headless coverage for the async main-thread contract: ``view_update``
translates + reconciles and hands off without waiting; ``view_draw``
writes the newest snapshot to the mailbox, presents the newest published
frame (or the loading status before the first publication), and performs
no service RPCs and no sleeps. Failure publications flow through the
existing ``_report_viewport_error`` dedupe. One end-to-end test drives the
real ``ViewportRenderThread`` + ``LatestViewRenderLoop`` against a fake
srtx client and asserts every RPC ran off the main thread.
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
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402


WAIT_S = 5.0


def _wait_for(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []
        self.redraw_requested = False
        self.reports: list[tuple[set[str], str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        self.redraw_requested = True

    def report(self, levels: set[str], message: str) -> None:
        self.reports.append((set(levels), message))


def _load_engine_with_fake_bpy(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_nonblocking_callbacks_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=SimpleNamespace()),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _viewport_context() -> SimpleNamespace:
    return SimpleNamespace(
        region_data=SimpleNamespace(view_perspective="PERSP"),
        scene=SimpleNamespace(),
        space_data=None,
    )


def _matrix(tx: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _request(module, **overrides):
    fields = {
        "input_usd_path": "/tmp/scene.usda",
        "sensor_paths": ("/Render/Product",),
        "selected_sensor_paths": ("/Render/Product",),
        "width": 2,
        "height": 2,
        "min_samples": 1,
        "max_samples": 4,
        "camera_prim_path": "/World/Camera",
        "camera_matrix": _matrix(1.0),
        "worker_command": "worker",
        "native_client_module": "client",
    }
    fields.update(overrides)
    return module.RenderRequest(**fields)


def _result(completed_samples: int = 4) -> RenderResult:
    return RenderResult(
        width=2,
        height=2,
        rgba8=b"\x00" * 16,
        completed_samples=completed_samples,
        session_completed_samples=completed_samples,
        simulation_time_ns=0,
    )


class _PoisonController:
    """Fails the test if the draw callback performs any session RPC."""

    def ensure(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not call controller.ensure")

    def render(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not call controller.render")

    def apply_runtime_updates(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not apply runtime updates")

    def deactivate(self, *_args, **_kwargs):
        # Benign: engine teardown (__del__ at GC) legitimately closes the
        # runtime; only the draw-path RPCs above are poisoned.
        return "stopped"

    def shutdown(self, *_args, **_kwargs):
        return None

    def diagnostics(self):
        return {
            "simulation_id": "sim",
            "session_reuse": {},
            "lifecycle_events": (),
            "startup": {"render_worker": {"status": "running"}},
            "render_timings": {},
            "value_update_timings": {},
            "active": True,
        }


def _engine_with_stubbed_session(module, monkeypatch: pytest.MonkeyPatch, request):
    """Engine whose async session seam is stubbed out (no thread)."""

    engine = module.OvrtxExampleRenderEngine()
    begin_calls: list[tuple[object, object, object]] = []

    class _Adapter:
        def view_update(self, context: object, depsgraph: object) -> object:
            return request

        def view_draw(self, context: object, depsgraph: object) -> object:
            return request

    monkeypatch.setattr(module, "_render_callback_adapter", lambda engine_id="": _Adapter())

    def begin_session(seen_request, scene=None, depsgraph=None) -> None:
        begin_calls.append((seen_request, scene, depsgraph))
        engine._viewport_request = seen_request

    engine._begin_async_viewport_session = begin_session
    engine._ovrtx_session_controller = _PoisonController()
    engine._write_viewport_artifact = lambda *_args, **_kwargs: None
    return engine, begin_calls


def test_view_update_translates_hands_off_and_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    context = _viewport_context()

    engine.view_update(context, depsgraph)

    assert [call[0] for call in begin_calls] == [request]
    assert begin_calls[0][1] is depsgraph.scene
    assert begin_calls[0][2] is depsgraph
    written = engine._camera_mailbox.peek()
    assert written is not None
    assert written.key == module.viewport_handoff.snapshot_from_render_request(request).key
    assert engine.redraw_requested
    # No publication exists yet: view_update reports the loading status.
    assert engine.stats[-1][1] == "Starting OVRTX"


def test_view_draw_shows_loading_status_before_first_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._viewport_lifecycle_phase = module.session_lifecycle.PHASE_LOADING
    draw_calls: list[object] = []
    engine._draw_viewport_texture = lambda *args: draw_calls.append(args)

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    # Nothing was drawn (existing background kept), the loading status is
    # shown, and the newest snapshot still reached the mailbox. The draw
    # path no longer polls for the first frame: the first publication
    # schedules the redraw that presents it (task02-05).
    assert draw_calls == []
    assert engine.stats[-1][1] == "Loading scene in OVRTX"
    assert engine._camera_mailbox.peek() is not None
    assert engine.redraw_requested is False


def test_view_draw_presents_newest_publication_without_rpcs_or_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", "1")
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    engine._end_viewport_session = lambda _reason: False

    def _no_sleep(_seconds: float) -> None:
        raise AssertionError("view_draw must not sleep")

    monkeypatch.setattr(module.time, "sleep", _no_sleep)
    draw_calls: list[object] = []
    engine._upload_viewport_texture = lambda result: ("texture", result)[0]
    engine._draw_viewport_texture = (
        lambda _context, _texture, result, _scene=None: draw_calls.append(result)
    )

    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    stale = _result(completed_samples=1)
    newest = _result(completed_samples=4)
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=stale, snapshot_key=snapshot_key, completed_samples=1
        )
    )
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=newest, snapshot_key=snapshot_key, completed_samples=4
        )
    )

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    # Only the newest publication is presented; refinement is complete for
    # the written view so no interim redraw poll is requested.
    assert draw_calls == [newest]
    assert engine.stats[-1][1] == "Viewport samples 4/4"
    assert engine._presented_publication_index == 2
    assert engine.redraw_requested is False
    # The profile record attributes only main-thread phases to the callback.
    record = engine._viewport_artifact_recorder._profile["recent_draws"][-1]
    assert record["timings_ms"]["ensure_session_ms"] == 0.0


def test_view_draw_reuses_texture_identity_for_unchanged_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    upload_indices: list[int] = []

    def _upload(result) -> str:
        upload_indices.append(engine._snapshot_index)
        # Mirror the real helper's identity bookkeeping.
        engine._texture_snapshot_index = engine._snapshot_index
        return "texture"

    engine._upload_viewport_texture = _upload
    engine._draw_viewport_texture = lambda *_args: None
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(), snapshot_key=snapshot_key, completed_samples=4
        )
    )

    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_draw(_viewport_context(), depsgraph)
    engine.view_draw(_viewport_context(), depsgraph)

    # The publication index is the texture identity: the same publication
    # keeps the same identity, so the cached-texture reuse path applies.
    assert upload_indices == [1, 1]
    assert engine._texture_snapshot_index == 1


def test_view_draw_reports_published_failure_with_dedupe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    draw_calls: list[object] = []
    engine._draw_viewport_texture = lambda *args: draw_calls.append(args)
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            status=module.viewport_handoff.FRAME_STATUS_FAILED,
            detail="RenderClientError: render failed",
        )
    )

    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_draw(_viewport_context(), depsgraph)
    engine.view_draw(_viewport_context(), depsgraph)

    assert draw_calls == []
    # One report despite two draws (existing dedupe), status text on both.
    assert engine.reports == [({"ERROR"}, "RenderClientError: render failed")]
    failed_stats = [text for _engine, text in engine.stats if text.startswith("Viewport failed:")]
    assert len(failed_stats) == 2


def test_view_draw_recovers_after_failure_when_frame_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            status=module.viewport_handoff.FRAME_STATUS_FAILED,
            detail="RenderClientError: render failed",
        )
    )
    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_draw(_viewport_context(), depsgraph)
    assert engine._viewport_reported_error == "RenderClientError: render failed"

    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(), snapshot_key=snapshot_key, completed_samples=4
        )
    )
    engine.view_draw(_viewport_context(), depsgraph)

    assert engine._viewport_reported_error == ""
    assert engine.stats[-1][1] == "Viewport samples 4/4"


def test_view_draw_keeps_presenting_last_frame_during_resync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resyncing publication presents the cached frame, never a freeze."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    engine._upload_viewport_texture = lambda _result: "texture"
    draw_calls: list[object] = []
    engine._draw_viewport_texture = (
        lambda _context, _texture, result, _scene=None: draw_calls.append(result)
    )
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    result = _result()
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=result, snapshot_key=snapshot_key, completed_samples=4
        )
    )
    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_draw(_viewport_context(), depsgraph)
    assert draw_calls == [result]

    # The render thread starts a background replacement (task02-06).
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            status=module.viewport_handoff.FRAME_STATUS_RESYNCING,
            detail="Re-syncing scene",
        )
    )
    engine.view_draw(_viewport_context(), depsgraph)

    # The last presented frame stays up, the resync status is shown, and
    # nothing is reported as an error. Lifecycle transitions surface as
    # Info-panel reports (stdout log routing), once per transition — a
    # repeated resync draw must not report again.
    assert draw_calls == [result, result]
    assert engine.stats[-1][1] == "Re-syncing scene"
    assert all("ERROR" not in levels for levels, _message in engine.reports)
    info_reports = [message for levels, message in engine.reports if "INFO" in levels]
    assert info_reports.count("Re-syncing scene") == 1
    engine.view_draw(_viewport_context(), depsgraph)
    info_reports = [message for levels, message in engine.reports if "INFO" in levels]
    assert info_reports.count("Re-syncing scene") == 1


def test_session_end_resets_publications_so_no_stale_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine, _begin_calls = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(), snapshot_key=snapshot_key, completed_samples=4
        )
    )
    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_draw(_viewport_context(), depsgraph)
    assert engine._presented_publication_index == 1

    # _end_viewport_session would shut the poison controller down; the
    # publication reset lives in _close_ovrtx_runtime.
    engine._ovrtx_session_controller = None
    engine._close_ovrtx_runtime()

    assert engine._frame_slot.peek_latest() is None
    assert engine._presented_publication_index == 0
    assert engine._presented_frame is None


class _ThreadRecordingClient:
    """Fake srtx client recording the thread ident of every RPC."""

    def __init__(self) -> None:
        self.rpc_thread_idents: list[tuple[str, int]] = []
        self.render_calls = 0
        self.startup_diagnostics = {"render_worker": {"status": "running"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def _record(self, name: str) -> None:
        self.rpc_thread_idents.append((name, threading.get_ident()))

    def start_session(self, _spec: object, simulation_id: str | None = None) -> str:
        self._record("start_session")
        return simulation_id or "sim"

    def render_result(self, _simulation_id: str, **kwargs: object) -> RenderResult:
        self._record("render_result")
        self.render_calls += 1
        return RenderResult(
            width=2,
            height=2,
            rgba8=b"\x00" * 16,
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, _simulation_id: str, values) -> OvrtxValueUpdateResult:
        self._record("update_transforms")
        return OvrtxValueUpdateResult(len(tuple(values)), pending_simulation_time_ns=1)

    def delete_simulation(self, _simulation_id: str) -> str:
        self._record("delete_simulation")
        return "stopped"

    def shutdown(self) -> None:
        self._record("shutdown")


def test_direct_route_ensures_and_renders_on_the_render_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: handoff → thread-side ensure → loop render → presentation."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    client = _ThreadRecordingClient()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    engine = module.OvrtxExampleRenderEngine()
    # Physics-disabled scheduler (same pattern as the loop tests).
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        assert engine._render_thread is not None
        assert engine._render_loop is not None
        assert module.VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS == 600.0
        assert engine._render_thread.diagnostics()["join_timeout_seconds"] == 600.0
        assert module._ENGINE_RUNTIMES[id(engine)]["authored"] is False
        # The handoff never blocks on activation: the loading phase is set
        # immediately on the main thread.
        assert engine._viewport_lifecycle_phase == module.session_lifecycle.PHASE_LOADING

        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        assert frame.status == module.viewport_handoff.FRAME_STATUS_FRAME

        # Refinement continues on the thread to max_samples without any
        # further main-thread involvement.
        deadline = time.monotonic() + WAIT_S
        newest = frame
        while time.monotonic() < deadline and newest.completed_samples < request.max_samples:
            candidate = engine._frame_slot.wait_for_newer(
                newest.publication_index, timeout=WAIT_S
            )
            if candidate is not None:
                newest = candidate
        assert newest.completed_samples == request.max_samples

        main_ident = threading.get_ident()
        assert client.rpc_thread_idents, "expected srtx RPCs"
        foreign = {ident for _name, ident in client.rpc_thread_idents}
        assert main_ident not in foreign
        assert {name for name, _ident in client.rpc_thread_idents} >= {
            "start_session",
            "render_result",
        }
    finally:
        stopped_thread = engine._render_thread
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
        assert engine._render_thread is None
        assert engine._render_loop is None
        if stopped_thread is not None:
            assert stopped_thread.status() in (
                module.viewport_render_thread.STATUS_STOPPED,
                module.viewport_render_thread.STATUS_FAILED,
            )
            assert stopped_thread.is_alive() is False
    names = [name for name, _ident in client.rpc_thread_idents]
    assert names.count("delete_simulation") == 1
    assert names.count("shutdown") == 1
    cleanup_idents = {
        ident
        for name, ident in client.rpc_thread_idents
        if name in {"delete_simulation", "shutdown"}
    }
    assert cleanup_idents == foreign


def test_teardown_deadline_fails_closed_without_caller_thread_rpcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(module, "_DIRECT_VIEWPORT_REUSE_BLOCKED", False)
    shutdowns: list[str] = []
    submitted: list[object] = []
    warnings: list[str] = []

    class _Thread:
        name = "ovrtx-render-timeout"

        def submit(self, fn: object, *, label: str = "") -> None:
            assert label == "session-teardown"
            submitted.append(fn)

        def stop(self) -> dict[str, object]:
            return {
                "status": "failed",
                "joined": False,
                "leaked_thread": True,
                "join_timeout_seconds": 0.01,
                "failure": "deadline exceeded",
            }

        def diagnostics(self) -> dict[str, object]:
            return {"name": self.name, "alive": True}

    engine = module.OvrtxExampleRenderEngine()
    engine._render_loop = SimpleNamespace(
        request_stop=lambda: None,
        diagnostics=lambda: {},
        iteration_records=lambda: [],
    )
    engine._render_thread = _Thread()
    engine._runtime_scheduler = SimpleNamespace(
        shutdown=lambda: shutdowns.append("scheduler")
    )
    engine._ovrtx_session_controller = SimpleNamespace(
        shutdown=lambda: shutdowns.append("controller")
    )
    engine._write_viewport_session_outputs = lambda **_kwargs: None
    monkeypatch.setattr(
        module.user_messages,
        "report_warning",
        lambda message, **_kwargs: warnings.append(message),
    )

    assert engine._end_viewport_session(
        module.ViewportSessionEndReason.RECONNECT_REQUESTED
    ) is False

    assert submitted and shutdowns == []
    assert engine._render_thread is None
    assert engine._runtime_scheduler is None
    assert engine._ovrtx_session_controller is None
    assert module._DIRECT_VIEWPORT_REUSE_BLOCKED is True
    assert engine._viewport_cleanup_diagnostics["status"] == "teardown_deadline_exceeded"
    assert warnings and "runtime reuse disabled" in warnings[0]
    with pytest.raises(RenderClientError, match="cannot restart"):
        engine._begin_async_viewport_session(_request(module))


def test_authored_viewport_teardown_does_not_close_shared_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    events: list[str] = []
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = SimpleNamespace(
        shutdown=lambda: events.append("scheduler")
    )
    engine._viewport_generation_runtime = SimpleNamespace(
        ovrtx=SimpleNamespace(
            deactivate=lambda: (_ for _ in ()).throw(
                AssertionError("shared OVRTX runtime was closed")
            )
        ),
        ovphysx=SimpleNamespace(
            deactivate=lambda: (_ for _ in ()).throw(
                AssertionError("shared OVPhysX runtime was closed")
            )
        ),
    )

    teardown, state = engine._runtime_teardown_state()
    teardown()

    assert state == {"ran": True, "errors": []}
    assert events == []


def test_fail_close_requests_stop_for_every_loop_sharing_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    runtime = SimpleNamespace(reuse_blocked=False)
    stopped: list[str] = []

    class _DeadScene:
        def __getattribute__(self, _name: str) -> object:
            raise ReferenceError("StructRNA has been removed")

    monkeypatch.setattr(
        module,
        "_ENGINE_RUNTIMES",
        {
            0: {
                "authored": True,
                "generation_runtime": None,
                "scene": _DeadScene(),
                "render_loop": SimpleNamespace(
                    request_stop=lambda: stopped.append("stale")
                ),
            },
            1: {
                "authored": True,
                "generation_runtime": runtime,
                "render_loop": SimpleNamespace(
                    request_stop=lambda: stopped.append("first")
                ),
            },
            2: {
                "authored": True,
                "generation_runtime": runtime,
                "render_loop": SimpleNamespace(
                    request_stop=lambda: stopped.append("second")
                ),
            },
        },
    )

    module._fail_closed_runtime_reuse(runtime)

    assert runtime.reuse_blocked is True
    assert stopped == ["first", "second"]


def test_sidecar_resolves_runtime_created_during_authored_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    scene = SimpleNamespace(session_uid=12)
    runtime = object()
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "active_runtime_for_scene",
        lambda value: runtime if value is scene else None,
    )

    assert module._sidecar_generation_runtime(
        {"authored": True, "scene": scene, "generation_runtime": None}
    ) is runtime


def test_direct_route_identity_ignores_unrelated_scene_runtime_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(module, "_DIRECT_VIEWPORT_REUSE_BLOCKED", False)
    scene = SimpleNamespace(session_uid=12)
    unrelated_runtime = SimpleNamespace(reuse_blocked=False)
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "owns_request",
        lambda _scene, _request: False,
    )
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "active_runtime_for_scene",
        lambda value: unrelated_runtime if value is scene else None,
    )
    engine = module.OvrtxExampleRenderEngine()
    engine._prepare_direct_session = lambda _request: None
    routes_before_start: list[bool] = []
    engine._ensure_render_loop = (
        lambda _request, *, scene, authored: routes_before_start.append(authored)
    )

    engine._begin_async_viewport_session(_request(module), scene=scene)
    module._ENGINE_RUNTIMES[id(engine)]["authored"] = routes_before_start[0]
    authored_stops: list[str] = []
    module._ENGINE_RUNTIMES[999] = {
        "authored": True,
        "scene": scene,
        "generation_runtime": unrelated_runtime,
        "render_loop": SimpleNamespace(
            request_stop=lambda: authored_stops.append("unrelated")
        ),
    }
    engine._record_unconfirmed_teardown(
        SimpleNamespace(name="direct-timeout"),
        {"joined": False, "leaked_thread": True, "join_timeout_seconds": 0.01},
    )

    assert routes_before_start == [False]
    assert module._DIRECT_VIEWPORT_REUSE_BLOCKED is True
    assert unrelated_runtime.reuse_blocked is False
    assert authored_stops == []


def test_running_thread_keeps_its_original_route_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    scene = SimpleNamespace(session_uid=12)
    engine = module.OvrtxExampleRenderEngine()
    module._ENGINE_RUNTIMES[id(engine)]["authored"] = True
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING
    )
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "owns_request",
        lambda _scene, _request: False,
    )

    with pytest.raises(RenderClientError, match="route cannot change"):
        engine._begin_async_viewport_session(_request(module), scene=scene)

    assert module._ENGINE_RUNTIMES[id(engine)]["authored"] is True


def test_route_change_restarts_before_publishing_opposite_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    scene = SimpleNamespace(session_uid=12)
    engine = module.OvrtxExampleRenderEngine()
    engine._viewport_scene = scene
    module._ENGINE_RUNTIMES[id(engine)]["authored"] = True
    events: list[str] = []
    engine._end_viewport_session = lambda _reason: events.append("end") or True
    engine._begin_async_viewport_session = (
        lambda _request, scene=None: events.append("begin")
    )
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "owns_request",
        lambda _scene, _request: False,
    )

    engine._note_translated_request(_request(module))

    assert events == ["end", "begin"]


def test_destructor_retains_sidecar_through_normal_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    sidecar = module._ENGINE_RUNTIMES[id(engine)]
    observed: list[bool] = []
    engine._end_viewport_session = lambda _reason: observed.append(
        module._ENGINE_RUNTIMES.get(id(engine)) is sidecar
    )

    engine.__del__()

    assert observed == [True]
    assert id(engine) not in module._ENGINE_RUNTIMES


def test_pre_activation_teardown_does_not_close_promoted_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    events: list[str] = []
    scene = SimpleNamespace(session_uid=12)
    controller = SimpleNamespace(
        shutdown=lambda: events.append("controller"),
        adopt_owning_thread=lambda: None,
    )
    runtime = SimpleNamespace(ovrtx=SimpleNamespace(controller=controller))
    engine = module.OvrtxExampleRenderEngine()
    engine._viewport_scene = scene
    engine._ovrtx_session_controller = controller
    engine._runtime_scheduler = SimpleNamespace(
        shutdown=lambda: events.append("scheduler")
    )
    monkeypatch.setattr(
        module.scene_generation_sessions,
        "active_runtime_for_scene",
        lambda value: runtime if value is scene else None,
    )

    teardown, state = engine._runtime_teardown_state()
    teardown()

    assert state == {"ran": True, "errors": []}
    assert events == ["scheduler"]
    engine._ovrtx_session_controller = None
    engine._runtime_scheduler = None


def test_thread_side_ensure_failure_publishes_failure_and_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup ensure failure follows the task02-06 on-thread retry policy."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    start_attempts: list[object] = []

    def _fail_client(_request: object) -> object:
        start_attempts.append(_request)
        raise RenderClientError("worker launch failed")

    monkeypatch.setattr(controller_module, "_runtime_client_from_request", _fail_client)
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        # The startup ensure is mailbox-triggered (task02-06): nothing runs
        # until the first snapshot is written.
        assert start_attempts == []
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        assert frame.status == module.viewport_handoff.FRAME_STATUS_FAILED
        assert "worker launch failed" in frame.detail
        assert len(start_attempts) == 1
        assert engine._viewport_start_failure_count == 1
        # The thread stays up: the callbacks never restart it for retries —
        # the loop retries on the next snapshot per should_auto_retry.
        assert engine._render_loop_needs_start() is False
        # A failed first activation never stamped a session start (so a
        # retry attempt presents as loading, not resyncing) and cleared
        # the busy phase.
        assert engine._viewport_session_started_ns == 0
        assert engine._viewport_lifecycle_phase == ""

        last_index = frame.publication_index
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        retry_frame = engine._frame_slot.wait_for_newer(last_index, timeout=WAIT_S)
        assert retry_frame is not None
        assert retry_frame.status == module.viewport_handoff.FRAME_STATUS_FAILED
        assert len(start_attempts) == 2
        assert engine._viewport_start_failure_count == 2
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)


def test_thread_side_ensure_retries_stop_after_auto_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshots beyond should_auto_retry never re-attempt the ensure."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    start_attempts: list[object] = []

    def _fail_client(_request: object) -> object:
        start_attempts.append(_request)
        raise RenderClientError("worker launch failed")

    monkeypatch.setattr(controller_module, "_runtime_client_from_request", _fail_client)
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    max_retries = module.session_lifecycle.MAX_AUTO_RETRIES
    try:
        engine._begin_async_viewport_session(request)
        last_index = 0
        for _attempt in range(max_retries):
            engine._camera_mailbox.write(
                module.viewport_handoff.snapshot_from_render_request(request)
            )
            frame = engine._frame_slot.wait_for_newer(last_index, timeout=WAIT_S)
            assert frame is not None
            assert frame.status == module.viewport_handoff.FRAME_STATUS_FAILED
            last_index = frame.publication_index
        assert len(start_attempts) == max_retries
        assert engine._viewport_start_failure_count == max_retries

        # Retries exhausted: further snapshots publish nothing new and do
        # not attempt another ensure; the loop parks in the failed state.
        loop = engine._render_loop
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        assert _wait_for(
            lambda: loop.diagnostics().get("snapshots_taken", 0) > max_retries
        )
        assert loop.diagnostics().get("retry_blocked") is True
        assert len(start_attempts) == max_retries
        assert engine._frame_slot.latest_index() == last_index
        # The gave-up state presents through the existing lifecycle status.
        status = engine._session_lifecycle_diagnostics(running=False)
        assert status["status"] == module.session_lifecycle.STATUS_FAILED
        assert status["auto_retry_allowed"] is False
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
