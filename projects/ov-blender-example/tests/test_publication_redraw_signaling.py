# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame publication and redraw signaling (task02-05).

Publication of a new frame (or resync/failure state) latches exactly one
pending redraw for a recurring ``bpy.app.timers`` poller registered on the
main thread at session start. The render thread never touches Blender's
lock-free application-timer registry. Duplicate publications before a poll
are absorbed by an atomic ``redraw_pending`` flag per engine; the poller
holds a weakref to the engine (dead ref → no-op) and falls back to the
VIEW_3D area scan when ``tag_redraw`` raises.
``redraw_requested_monotonic_ns`` is stamped at publication (thread side)
so the publish→draw span includes poll latency. The ``view_draw``-tail
``tag_redraw`` refinement polling is gone.
"""

from __future__ import annotations

import gc
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
from ovrtx_blender_example import viewport_handoff  # noqa: E402
from ovrtx_blender_example import viewport_render_thread  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402


WAIT_S = 5.0


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []
        self.redraw_requested = False
        self.redraw_count = 0
        self.reports: list[tuple[set[str], str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        self.redraw_requested = True
        self.redraw_count += 1

    def report(self, levels: set[str], message: str) -> None:
        self.reports.append((set(levels), message))


class _FakeTimers:
    """``bpy.app.timers`` stand-in: records registrations, fires on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pending: list[object] = []
        self.register_calls = 0

    def register(self, fn, first_interval: float = 0.0) -> None:
        with self._lock:
            self.register_calls += 1
            self.pending.append(fn)

    def run_pending(self) -> int:
        """Fire pending timer callbacks as Blender's main loop would."""

        with self._lock:
            callbacks = list(self.pending)
            self.pending.clear()
        recurring: list[object] = []
        for fn in callbacks:
            interval = fn()
            if interval is not None:
                assert interval >= 0.0
                recurring.append(fn)
        with self._lock:
            self.pending.extend(recurring)
        return len(callbacks)


class _FakeWindowManager:
    """Records session-scoped event timers that wake Blender's event loop."""

    def __init__(self) -> None:
        self.windows = ()
        self.add_calls: list[tuple[float, object]] = []
        self.remove_calls: list[object] = []
        self.active: list[object] = []

    def event_timer_add(self, interval: float, *, window: object) -> object:
        timer = object()
        self.add_calls.append((float(interval), window))
        self.active.append(timer)
        return timer

    def event_timer_remove(self, timer: object) -> None:
        self.remove_calls.append(timer)
        self.active.remove(timer)


def _load_engine_with_fake_bpy(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_publication_redraw_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    window = object()
    window_manager = _FakeWindowManager()
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=_FakeTimers()),
        context=SimpleNamespace(window=window, window_manager=window_manager),
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


class _DummyEngine:
    """Weakref-able tag_redraw target for signaler unit tests."""

    def __init__(self, *, tag_raises: bool = False) -> None:
        self.tag_calls = 0
        self.area_scan_calls = 0
        self._tag_raises = tag_raises

    def tag_redraw(self) -> None:
        if self._tag_raises:
            raise ReferenceError("engine freed")
        self.tag_calls += 1

    def _tag_viewport_redraws(self) -> None:
        self.area_scan_calls += 1


class _PoisonController:
    """Fails the test if the draw callback performs any session RPC."""

    def ensure(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not call controller.ensure")

    def render(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not call controller.render")

    def apply_runtime_updates(self, *_args, **_kwargs):
        raise AssertionError("view_draw must not apply runtime updates")

    def deactivate(self, *_args, **_kwargs):
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

    class _Adapter:
        def view_update(self, context: object, depsgraph: object) -> object:
            return request

        def view_draw(self, context: object, depsgraph: object) -> object:
            return request

    monkeypatch.setattr(module, "_render_callback_adapter", lambda engine_id="": _Adapter())

    def begin_session(seen_request, scene=None, depsgraph=None) -> None:
        engine._viewport_request = seen_request

    engine._begin_async_viewport_session = begin_session
    engine._ovrtx_session_controller = _PoisonController()
    engine._render_thread = SimpleNamespace(
        status=lambda: viewport_render_thread.STATUS_RUNNING
    )
    engine._write_viewport_artifact = lambda *_args, **_kwargs: None
    return engine


# ---------------------------------------------------------------------------
# Signaler unit tests (fake timers)
# ---------------------------------------------------------------------------


def test_publications_coalesce_into_one_pending_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N publications → one latch consumed by one main-thread poll."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()
    window_manager = module.bpy.context.window_manager

    assert window_manager.add_calls == [
        (module.PUBLICATION_REDRAW_WAKE_INTERVAL_SECONDS, module.bpy.context.window)
    ]
    assert len(window_manager.active) == 1
    assert signaler.diagnostics()["event_timer_active"] is True

    for _ in range(5):
        signaler.signal()

    # Session start registered one recurring main-thread poller; none of the
    # render-thread publications touches the timer registry.
    assert timers.register_calls == 1
    assert signaler.diagnostics()["redraw_pending"] is True
    assert signaler.diagnostics()["signals"] == 5

    assert timers.run_pending() == 1
    assert target.tag_calls == 1
    assert target.area_scan_calls == 1
    assert signaler.diagnostics()["redraw_pending"] is False
    assert signaler.diagnostics()["redraw_dispatches"] == 1

    # The next publication reuses the same poller registration.
    signaler.signal()
    assert timers.register_calls == 1
    assert timers.run_pending() == 1
    assert target.tag_calls == 2
    assert target.area_scan_calls == 2
    signaler.stop()
    assert window_manager.remove_calls
    assert window_manager.active == []
    assert signaler.diagnostics()["event_timer_removals"] == 1
    assert timers.run_pending() == 1
    assert signaler.diagnostics()["poller_active"] is False


def test_redraw_mark_stamped_at_publication_and_consumed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()

    before = time.perf_counter_ns()
    signaler.signal()
    after = time.perf_counter_ns()
    first_mark = signaler._requested_monotonic_ns
    assert before <= first_mark <= after

    # Absorbed publications do not restamp: the span starts at the first
    # (unserviced) redraw request, which is what timer dispatch latency
    # measurement needs.
    signaler.signal()
    assert signaler._requested_monotonic_ns == first_mark

    assert signaler.consume_request_mark() == first_mark
    assert signaler.consume_request_mark() is None
    signaler.stop()


def test_signal_after_stop_does_not_leak_into_restarted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()
    signaler.stop()

    signaler.signal()
    assert signaler.diagnostics()["redraw_pending"] is False
    assert signaler.consume_request_mark() is None

    signaler.start()
    assert timers.run_pending() == 2
    assert target.tag_calls == 0


def test_render_thread_signal_never_registers_a_blender_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()

    publisher = threading.Thread(
        target=lambda: [signaler.signal() for _ in range(8)],
        name="publication-test-thread",
    )
    publisher.start()
    publisher.join(WAIT_S)

    assert not publisher.is_alive()
    assert timers.register_calls == 1
    assert signaler.diagnostics()["signals"] == 8
    assert timers.run_pending() == 1
    assert target.tag_calls == 1
    assert target.area_scan_calls == 1


def test_stopped_poller_generation_cannot_consume_restarted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()
    signaler.stop()
    signaler.start()
    signaler.signal()

    # Both callbacks are still queued in the fake registry. The retired one
    # exits; only the current generation consumes and dispatches the frame.
    assert timers.register_calls == 2
    assert timers.run_pending() == 2
    assert target.tag_calls == 1
    assert len(timers.pending) == 1


def test_timer_callback_with_dead_engine_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()
    signaler.signal()

    del target
    gc.collect()

    # Dead weakref: the callback is a no-op (no exception, no fallback).
    assert timers.run_pending() == 1
    diagnostics = signaler.diagnostics()
    assert diagnostics["timer_fires"] == 1
    assert diagnostics["fallback_redraws"] == 0
    assert diagnostics["redraw_pending"] is False


def test_tag_redraw_failure_falls_back_to_area_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine(tag_raises=True)
    signaler = module._PublicationRedrawSignaler(target)
    signaler.start()
    signaler.signal()

    assert timers.run_pending() == 1
    # Setting the engine flag failed, but the required VIEW_3D area scan still
    # tags the actual viewport redraw.
    assert target.tag_calls == 0
    assert target.area_scan_calls == 1
    assert signaler.diagnostics()["fallback_redraws"] == 1


def test_registration_failure_releases_latch_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    target = _DummyEngine()
    signaler = module._PublicationRedrawSignaler(target)

    def _broken_register(_fn, first_interval: float = 0.0) -> None:
        raise RuntimeError("timers unavailable")

    original_register = timers.register
    timers.register = _broken_register
    signaler.start()
    diagnostics = signaler.diagnostics()
    assert diagnostics["registration_failures"] == 1
    # The latch is released so the next publication retries registration
    # instead of wedging redraws forever.
    assert diagnostics["redraw_pending"] is False

    timers.register = original_register
    signaler.start()
    signaler.signal()
    assert timers.pending
    assert timers.run_pending() == 1
    assert target.tag_calls == 1
    assert target.area_scan_calls == 1


# ---------------------------------------------------------------------------
# Slot wrapper
# ---------------------------------------------------------------------------


def test_slot_wrapper_signals_after_publish_and_delegates() -> None:
    slot = viewport_handoff.LatestFrameSlot()
    observed_at_signal: list[int] = []

    wrapper = viewport_render_thread.RedrawSignalingFrameSlot(
        slot,
        lambda: observed_at_signal.append(slot.latest_index()),
    )

    published = wrapper.publish(
        viewport_handoff.FrameState(
            render_result=_result(),
            snapshot_key=("key",),
            completed_samples=4,
        )
    )

    # The signal fires after the publication is stored, so a redraw
    # request can never observe the slot without the frame it announces.
    assert observed_at_signal == [1]
    assert published.publication_index == 1
    # Reads delegate to the wrapped slot.
    assert wrapper.peek_latest() is slot.peek_latest()
    assert wrapper.latest_index() == 1

    wrapper.publish(
        viewport_handoff.FrameState(
            status=viewport_handoff.FRAME_STATUS_FAILED,
            detail="RenderClientError: boom",
        )
    )
    # Non-frame state changes (resync/failure) signal too.
    assert observed_at_signal == [1, 2]


# ---------------------------------------------------------------------------
# view_draw integration (stubbed session, fake timers)
# ---------------------------------------------------------------------------


def test_view_draw_records_publication_stamped_redraw_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", "1")
    request = _request(module)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None

    publisher = viewport_render_thread.RedrawSignalingFrameSlot(
        engine._frame_slot, engine._redraw_signaler.signal
    )
    engine._redraw_signaler.start()
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    before = time.perf_counter_ns()
    publisher.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(), snapshot_key=snapshot_key, completed_samples=4
        )
    )
    after = time.perf_counter_ns()
    module.bpy.app.timers.run_pending()
    assert engine.redraw_requested

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    record = engine._viewport_artifact_recorder._profile["recent_draws"][-1]
    redraw_mark = record["span_boundaries"]["redraw_requested_monotonic_ns"]
    started_mark = record["span_boundaries"]["render_callback_started_monotonic_ns"]
    # The span starts at publication (thread side): the publish→draw span
    # includes timer dispatch latency instead of hiding it.
    assert before <= redraw_mark <= after
    assert redraw_mark <= started_mark
    # Consumed once: a draw without a new publication has no request mark.
    assert engine._redraw_signaler.consume_request_mark() is None
    engine._redraw_signaler.stop()


def test_view_draw_does_not_poll_while_refinement_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(completed_samples=1),
            snapshot_key=snapshot_key,
            completed_samples=1,
        )
    )

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    # Refinement is incomplete (1/4) but the draw tail no longer tags a
    # redraw: the next published step schedules the next redraw instead of
    # a busy tag_redraw loop.
    assert engine.stats[-1][1] == "Viewport samples 1/4"
    assert engine.redraw_requested is False
    assert module.bpy.app.timers.pending == []


def test_view_draw_reports_unbounded_sampling_as_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module, max_samples=0)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    artifact_running: list[bool] = []
    engine._write_viewport_artifact = (
        lambda *, running: artifact_running.append(running)
    )
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(completed_samples=8),
            snapshot_key=snapshot_key,
            completed_samples=8,
        )
    )

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    assert engine.stats[-1][1] == "Viewport samples 8/continuous"
    assert artifact_running == [True]


def test_script_stamped_mark_overrides_publication_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", "1")
    request = _request(module)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    snapshot_key = module.viewport_handoff.snapshot_from_render_request(request).key
    publisher = viewport_render_thread.RedrawSignalingFrameSlot(
        engine._frame_slot, engine._redraw_signaler.signal
    )
    publisher.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(), snapshot_key=snapshot_key, completed_samples=4
        )
    )
    # Measurement scripts (scripts/run_blender_navigation.py) stamp the
    # attribute directly for injected input events; it wins over the
    # publication stamp and both are consumed.
    engine._redraw_requested_monotonic_ns = 123456

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    record = engine._viewport_artifact_recorder._profile["recent_draws"][-1]
    assert record["span_boundaries"]["redraw_requested_monotonic_ns"] == 123456
    assert engine._redraw_requested_monotonic_ns is None
    assert engine._redraw_signaler.consume_request_mark() is None


# ---------------------------------------------------------------------------
# End to end: real thread + loop publications drive the timer
# ---------------------------------------------------------------------------


class _ThreadRecordingClient:
    """Fake srtx client recording the thread ident of every RPC."""

    def __init__(self) -> None:
        self.render_calls = 0
        self.startup_diagnostics = {"render_worker": {"status": "running"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, _spec: object, simulation_id: str | None = None) -> str:
        return simulation_id or "sim"

    def render_result(self, _simulation_id: str, **kwargs: object) -> RenderResult:
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
        return OvrtxValueUpdateResult(len(tuple(values)), pending_simulation_time_ns=1)

    def delete_simulation(self, _simulation_id: str) -> str:
        return "stopped"

    def shutdown(self) -> None:
        return None


def test_loop_publications_register_one_coalesced_redraw_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real thread + loop: publications coalesce into one pending timer."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    client = _ThreadRecordingClient()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        deadline = time.monotonic() + WAIT_S
        newest = None
        index = 0
        while time.monotonic() < deadline:
            candidate = engine._frame_slot.wait_for_newer(index, timeout=WAIT_S)
            if candidate is None:
                continue
            newest = candidate
            index = candidate.publication_index
            if (
                candidate.status == module.viewport_handoff.FRAME_STATUS_FRAME
                and candidate.completed_samples >= request.max_samples
            ):
                break
        assert newest is not None
        assert newest.completed_samples == request.max_samples
        # Refinement published multiple steps (min-first then doubling), but
        # session start registered exactly one main-thread poller. Render-
        # thread publications only update its pending bit.
        assert engine._frame_slot.latest_index() >= 2
        assert engine._redraw_signaler.diagnostics()["timer_registrations"] == 1
        assert timers.run_pending() >= 1
        assert engine.redraw_requested is True
        # After the redraw the latch is open for the next publication.
        assert engine._redraw_signaler.diagnostics()["redraw_pending"] is False
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)


def test_thread_side_ensure_failure_schedules_redraw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers

    def _fail_client(_request: object) -> object:
        raise RenderClientError("worker launch failed")

    monkeypatch.setattr(controller_module, "_runtime_client_from_request", _fail_client)
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        # The startup ensure is mailbox-triggered (task02-06): the failure
        # publishes once the first snapshot reaches the loop.
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        assert frame.status == module.viewport_handoff.FRAME_STATUS_FAILED
        # State-change publications (failure) signal presentation too, so
        # the failed state becomes visible without user input.
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline and timers.register_calls < 1:
            time.sleep(0.01)
        assert timers.register_calls == 1
        timers.run_pending()
        assert engine.redraw_requested is True
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
