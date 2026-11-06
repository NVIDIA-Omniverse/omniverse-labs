# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest-view render loop tests (task02-03, ADR 0013).

Fake-client loop coverage: one-sample acquisition on change, refinement to
``max_samples`` after stability, snapshot supersession between acquisitions,
live camera value updates on the existing session, idle parking
(no busy polling), edit- and pose-publication wakes during physics playback,
and failure publication that never kills the thread.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import camera_value_conversion
from ovrtx_blender_example import ovrtx_session_controller as controller_module
from ovrtx_blender_example import viewport_handoff
from ovrtx_blender_example.interactive_edit_planner import (
    DataAuthority,
    EditIntent,
    EditShape,
    EditStatus,
    edit_location,
)
from ovrtx_blender_example.ovrtx_runtime_client import (
    DEFAULT_RENDER_SIMULATION_STEP_NS,
    RenderClientError,
    RenderResult,
    RuntimeServicesPreparingError,
)
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from ovrtx_blender_example.render_requests import CameraProjectionState, RenderRequest
from ovrtx_blender_example.runtime_scheduler import (
    EditSubmissionResult,
    RuntimeScheduler,
    RuntimeTickResult,
    RuntimeTickStatus,
)
from ovrtx_blender_example.shared_stage_errors import SharedStageCompositionError
from ovrtx_blender_example.view_update_stream import ViewUpdateStream
from ovrtx_blender_example.viewport_handoff import (
    FRAME_STATUS_FAILED,
    FRAME_STATUS_FRAME,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (
    LatestViewRenderLoop,
    RenderThreadError,
    RenderThreadRejectedError,
    SessionLifecycleHooks,
    ViewportRenderThread,
)


WAIT_S = 5.0


def test_native_completion_wait_skips_resolved_frames_and_bounds_wait():
    calls = []
    pending = SimpleNamespace(
        tickets={"done": SimpleNamespace(wait=lambda ms: pytest.fail("resolved ticket waited")),
                 "pending": SimpleNamespace(wait=calls.append)},
        resolved_frames={"done": object()},
    )
    LatestViewRenderLoop._wait_pending_native(pending, 2000.0)
    assert calls == [1000.0]
    LatestViewRenderLoop._wait_pending_native(SimpleNamespace(tickets={"old": object()}), 8.0)


def _warm_public_pipeline(tmp_path, monkeypatch, *, timeline_controls_enabled=False):
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_POLL_BUDGET_MS", "0")
    client = _Client()
    client.worker_owned = True
    client.camera_successor_deferral = True
    client.async_supported = True
    client.poll_resolve_after = 100
    harness = _Harness(monkeypatch, tmp_path, client)
    harness.loop._adopt_snapshot(_snapshot(tx=2.0, max_samples=4,
        timeline_controls_enabled=timeline_controls_enabled))
    harness.loop._iterate()  # Session cold start remains synchronous.
    harness.loop._iterate()  # Retain a genuinely pending exact-view ticket.
    assert len(harness.loop._pending_samples) == 1
    return harness


@pytest.mark.parametrize("timeline_controls_enabled", [False, True])
def test_public_camera_successor_coalesces_and_keeps_predecessor_attribution(tmp_path, monkeypatch, timeline_controls_enabled):
    harness = _warm_public_pipeline(tmp_path, monkeypatch,
        timeline_controls_enabled=timeline_controls_enabled)
    loop, client = harness.loop, harness.client
    original = loop._snapshot
    pending = loop._pending_samples[0]
    try:
        loop._adopt_snapshot(_snapshot(tx=3.0, max_samples=4,
            timeline_controls_enabled=timeline_controls_enabled))
        latest = _snapshot(tx=4.0, max_samples=4,
            timeline_controls_enabled=timeline_controls_enabled)
        loop._adopt_snapshot(latest)
        assert loop._snapshot is original
        assert loop._deferred_camera_snapshot is latest
        assert not any(t.cancelled for t in pending.tickets.values())
        pending.remaining = 0
        loop._iterate()
        assert harness.slot.frames()[-1].snapshot_key == original.key
        assert loop._deferred_camera_due()
        loop._adopt_snapshot(loop._deferred_camera_snapshot)
        loop._iterate()  # Warm successor begins asynchronously, even at sample 0.
        assert loop._snapshot is latest
        assert client.render_additional_samples == [1]
        assert client.begin_render_calls == 2
        loop._pending_samples[0].remaining = 0
        loop._iterate()
        assert harness.slot.frames()[-1].snapshot_key == latest.key
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


@pytest.mark.parametrize("override", [
    {"width": 2}, {"render_var": "HdrColor"}, {"max_samples": 3},
    {"camera_projection": CameraProjectionState(source="blender_view", focal_length=35.0,
        horizontal_aperture=36.0, vertical_aperture=24.0)},
    {"timeline_playing": True}, {"camera_prim_path": "/World/OtherCamera"},
    {"timeline_frame": 2}, {"timeline_start": 2}, {"timeline_end": 20},
    {"simulation_reset_token": 1},
])
@pytest.mark.parametrize("timeline_controls_enabled", [False, True])
def test_public_camera_deferral_rejects_non_pose_changes(tmp_path, monkeypatch, override, timeline_controls_enabled):
    harness = _warm_public_pipeline(tmp_path, monkeypatch,
        timeline_controls_enabled=timeline_controls_enabled)
    loop = harness.loop
    pending = loop._pending_samples[0]
    try:
        changes = {"max_samples": 4, "timeline_controls_enabled": timeline_controls_enabled, **override}
        loop._adopt_snapshot(_snapshot(tx=3.0, **changes))
        assert loop._deferred_camera_snapshot is None
        assert all(t.cancelled for t in pending.tickets.values())
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_public_camera_deferral_bound_and_edit_revision_interrupt(tmp_path, monkeypatch):
    harness = _warm_public_pipeline(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        loop._adopt_snapshot(_snapshot(tx=3.0, max_samples=4))
        assert not loop._deferred_camera_due()
        loop._camera_deferred_at_ns -= 250_000_000
        assert loop._deferred_camera_due()
        loop._camera_deferred_at_ns = time.perf_counter_ns()
        harness.scheduler.note_applied_content()
        assert loop._deferred_camera_due()
        loop._adopt_snapshot(loop._deferred_camera_snapshot)
        assert not loop._pending_samples
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_public_session_revision_change_requires_fresh_cold_render(tmp_path, monkeypatch):
    harness = _warm_public_pipeline(tmp_path, monkeypatch)
    loop, client = harness.loop, harness.client
    pending = loop._pending_samples[0]
    try:
        harness.controller._session_revision += 1
        loop._adopt_snapshot(_snapshot(tx=3.0, max_samples=4))
        assert loop._deferred_camera_snapshot is None
        assert all(t.cancelled for t in pending.tickets.values())
        loop._iterate()
        assert client.render_additional_samples == [1, 1]
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


@pytest.mark.parametrize("timeline_controls_enabled", [False, True])
def test_public_refinement_prearm_respects_capacity_and_sample_limit(tmp_path, monkeypatch, timeline_controls_enabled):
    harness = _warm_public_pipeline(tmp_path, monkeypatch,
        timeline_controls_enabled=timeline_controls_enabled)
    loop = harness.loop
    try:
        loop._pending_samples[0].remaining = 0
        loop._iterate()
        assert loop._readback_prearm_count == 1
        assert len(loop._pending_samples) <= loop._pipeline_depth
        loop._prearm_refinement_before_copy()
        assert loop._readback_prearm_count == 1  # At capacity/sample limit.
        for pending in loop._pending_samples:
            pending.remaining = 0
        loop._iterate()
        loop._iterate()
        assert loop._completed_samples == 4
        assert harness.client.begin_render_calls == 3
        assert not loop._pending_samples
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_public_refinement_prearm_rejects_running_timeline(tmp_path, monkeypatch):
    harness = _warm_public_pipeline(tmp_path, monkeypatch, timeline_controls_enabled=True)
    loop = harness.loop
    try:
        # A paused authored viewport can prearm; starting playback cannot
        # silently add another native sample for the old timeline state.
        loop._snapshot = replace(loop._snapshot, timeline_playing=True)
        begin_calls = harness.client.begin_render_calls
        loop._prearm_refinement_before_copy()
        assert harness.client.begin_render_calls == begin_calls
        assert loop._readback_prearm_count == 0
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _matrix(tx: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _request(tmp_path: Path) -> RenderRequest:
    return RenderRequest(
        input_usd_path=str(tmp_path / "scene.usda"),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=1,
        height=1,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        camera_matrix=_matrix(1.0),
        worker_command="worker",
        native_client_module="client",
    )


def _transform_intent(matrix, prim_path: str = "/World/Cube") -> EditIntent:
    """View-authoritative object transform intent (omni:xform value update)."""

    return EditIntent(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=prim_path,
            usd_attribute="omni:xform",
            blender_property_path="matrix_world",
        ),
        value=matrix,
    )


def _snapshot(tx: float = 2.0, **overrides) -> ViewSnapshot:
    fields = {
        "camera_matrix": _matrix(tx),
        "camera_prim_path": "/World/Camera",
        "min_samples": 1,
        "max_samples": 4,
        "selected_sensor_paths": ("/Render/Product",),
        "width": 1,
        "height": 1,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


def _projection(focal_length: float) -> CameraProjectionState:
    return CameraProjectionState(
        source="active_camera_view",
        focal_length=focal_length,
        horizontal_aperture=36.0,
        vertical_aperture=24.0,
        clipping_range=(0.1, 100.0),
    )


def _honored_camera_probe() -> camera_value_conversion.CameraValueProbe:
    probe = camera_value_conversion.CameraValueProbe()
    for probe_class in camera_value_conversion.CAMERA_VALUE_PROBE_CLASSES:
        probe.record_result(probe_class, honored=True)
    return probe


class _FakeReadTicket:
    """Mirrors ``ReadWorldStateTicket``'s ``cancel()`` for teardown tests."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _PendingSample:
    """Fake ``PendingRenderSample`` returned by ``_Client.begin_render_sample``."""

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining
        self.tickets = {"/Render/Product/LdrColor": _FakeReadTicket()}
        self.native_started_ns = time.perf_counter_ns()


class _Client:
    """Fake srtx client compatible with OvrtxSessionController."""

    def __init__(self, simulation_id: str = "sim") -> None:
        self.simulation_id = simulation_id
        self.fail_render = False
        self.starts = 0
        self.deletes = 0
        self.closed = 0
        self.render_calls = 0
        self.render_additional_samples: list[int] = []
        self.render_camera_translations: list[float | None] = []
        self.render_focal_lengths: list[float | None] = []
        self.render_thread_idents: list[int] = []
        self.camera_translation: float | None = None
        self.focal_length: float | None = None
        self.transform_update_batches: list[tuple] = []
        self.transform_update_attempts = 0
        self.transform_failures_remaining = 0
        self.render_hook = None
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}
        # Async read pipelining (performance-improvements Phase 03).
        self.async_supported = False
        self.begin_render_calls = 0
        self.poll_render_calls = 0
        self.poll_resolve_after = 0
        self._outstanding = 0
        self.max_outstanding = 0
        self.pending_samples: list[_PendingSample] = []

    def supports_async_render_sample(self) -> bool:
        return self.async_supported

    def resize_output(self, simulation_id, width, height):
        self.output_shape = (width, height)

    def begin_render_sample(self, simulation_id: str, **kwargs: object) -> "_PendingSample":
        self.render_thread_idents.append(threading.get_ident())
        self.begin_render_calls += 1
        self._outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self._outstanding)
        pending = _PendingSample(self.poll_resolve_after)
        self.pending_samples.append(pending)
        return pending

    def poll_render_sample(self, pending: "_PendingSample", *, before_copy=None) -> RenderResult | None:
        self.render_thread_idents.append(threading.get_ident())
        self.poll_render_calls += 1
        if pending.remaining > 0:
            pending.remaining -= 1
            return None
        if before_copy is not None:
            before_copy()
        self._outstanding -= 1
        self.render_calls += 1
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=1,
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.starts += 1
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        call_index = self.render_calls
        self.render_calls += 1
        self.render_additional_samples.append(int(kwargs["additional_samples"]))
        self.render_camera_translations.append(self.camera_translation)
        self.render_focal_lengths.append(self.focal_length)
        self.render_thread_idents.append(threading.get_ident())
        hook = self.render_hook
        if hook is not None:
            hook(call_index)
        if self.fail_render:
            raise RenderClientError("render failed")
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        self.transform_update_attempts += 1
        if self.transform_failures_remaining:
            self.transform_failures_remaining -= 1
            raise RenderClientError("camera update rejected_for_test")
        batch = tuple(values)
        self.transform_update_batches.append(batch)
        for value in batch:
            if value.prim_path == "/World/Camera":
                self.camera_translation = float(value.matrix[0][3])
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def update_attribute_values(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        batch = tuple(values)
        for value in batch:
            if value.prim_path == "/World/Camera" and value.attribute == "focalLength":
                self.focal_length = float(value.value)
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        self.deletes += 1
        return "stopped"

    def shutdown(self) -> None:
        self.closed += 1


class _RecordingMailbox(CameraRequestMailbox):
    """Records each take's timeout (at entry) and whether it took a snapshot.

    Recording at entry is what lets tests observe a *parked* loop: the
    ``timeout=None`` record exists while the take is still blocked.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[list] = []

    def take(self, timeout: float | None = None) -> ViewSnapshot | None:
        entry = [timeout, False]
        self.records.append(entry)
        snapshot = super().take(timeout)
        entry[1] = snapshot is not None
        return snapshot

    def park_count(self) -> int:
        return sum(1 for timeout, _taken in list(self.records) if timeout is None)


class _RecordingSlot(LatestFrameSlot):
    """Keeps every publication (the slot itself only holds the newest)."""

    def __init__(self) -> None:
        super().__init__()
        self._record_lock = threading.Lock()
        self.published: list[FrameState] = []

    def publish(self, frame_state: FrameState) -> FrameState:
        stamped = super().publish(frame_state)
        with self._record_lock:
            self.published.append(stamped)
        return stamped

    def frames(self) -> list[FrameState]:
        with self._record_lock:
            return list(self.published)


class _Harness:
    """Real controller + real scheduler (physics disabled) + fake client."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        client: _Client | None = None,
    ) -> None:
        self.client = client or _Client()
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = _request(tmp_path)
        self.controller.ensure(self.base_request)
        self.scheduler = RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = _RecordingMailbox()
        self.slot = _RecordingSlot()
        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
        )


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="latest-view-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


# --- Async read pipelining (performance-improvements Phase 03) -------------


def test_pipelined_acquisition_overlaps_write_with_pending_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth 2: the first sample stays synchronous (cold-start cook), then
    a later sample's write goes out while a prior sample's read is pending."""

    # n-03-08: the bounded inner poll loop (default budget) resolves a
    # 2-poll ticket inside the same outer tick it was begun in, which
    # avoids ever needing a second write in flight -- exactly the overhead
    # n-03-07 found and this fix removes. This test asserts the pre-fix
    # single-poll-per-tick contract still holds with budget=0 (its escape
    # hatch); genuine overlap-for-fast-tickets is covered instead by
    # test_pipeline_poll_budget_avoids_unnecessary_overlap_for_fast_ticket.
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_POLL_BUDGET_MS", "0")
    client = _Client()
    client.async_supported = True
    client.poll_resolve_after = 2
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=4))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)

    frames = [f for f in harness.slot.frames() if f.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in frames[:4]] == [1, 2, 3, 4]
    assert client.render_additional_samples == [1]
    assert client.begin_render_calls == 3
    assert client.max_outstanding == 2
    assert client.render_calls == 4
    assert [
        frame.timing_marks["render_call_started_monotonic_ns"]
        for frame in frames[1:4]
    ] == [pending.native_started_ns for pending in client.pending_samples]


@pytest.mark.parametrize('public_successor', [False, True])
def test_pending_sample_keeps_submission_snapshot_after_same_view_rewrite(tmp_path, monkeypatch, public_successor):
    """A resolving old ticket must not borrow a successor's snapshot timestamp."""
    harness = _warm_public_pipeline(tmp_path, monkeypatch)
    loop, client = harness.loop, harness.client
    client.camera_successor_deferral = public_successor
    try:
        original = loop._snapshot
        oldest = loop._pending_samples[0]
        newer = replace(original, written_monotonic_ns=oldest.native_started_ns + 1,
                        timing_marks={'redraw_requested_monotonic_ns': oldest.native_started_ns + 1})
        loop._adopt_snapshot(newer)
        assert loop._snapshot is newer and newer.key == original.key
        oldest.remaining = 0
        loop._iterate()  # Resolves oldest; before_copy also submits a successor.
        frame = [frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FRAME][-1]
        assert len(client.pending_samples) > 1
        assert frame.timing_marks['render_call_started_monotonic_ns'] == oldest.native_started_ns
        assert frame.timing_marks['snapshot_written_monotonic_ns'] == original.written_monotonic_ns
        assert frame.timing_marks['snapshot_written_monotonic_ns'] <= frame.timing_marks['render_call_started_monotonic_ns']
        assert frame.snapshot_key == original.key
        assert 'redraw_requested_monotonic_ns' not in frame.timing_marks
        # The newly submitted ticket owns the newer context, not the resolved one.
        successor = loop._pending_samples[0]
        successor.remaining = 0
        loop._iterate()
        frame = [frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FRAME][-1]
        assert frame.timing_marks['render_call_started_monotonic_ns'] == successor.native_started_ns
        assert frame.timing_marks['snapshot_written_monotonic_ns'] == newer.written_monotonic_ns
        assert frame.timing_marks['redraw_requested_monotonic_ns'] == newer.written_monotonic_ns
    finally:
        loop._cancel_pending_samples()
        assert not loop._pending_sample_snapshots
        loop._pipeline_loop.close()


def test_pipeline_poll_budget_avoids_unnecessary_overlap_for_fast_ticket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n-03-08: with the default poll budget, a ticket that resolves in a
    couple of polls is picked up within the outer tick that began it,
    instead of forcing a second write into flight before the first
    resolves. Frame delivery is unaffected; only the overlap that n-03-07
    showed wasn't earning its keep (worker resolves in single-digit ms,
    well under the default budget) goes away for the fast-ticket case."""

    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    client = _Client()
    client.async_supported = True
    client.poll_resolve_after = 1
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=4))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)

    frames = [f for f in harness.slot.frames() if f.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in frames[:4]] == [1, 2, 3, 4]
    assert client.begin_render_calls == 3
    assert client.max_outstanding == 1
    assert client.render_calls == 4


def test_pipeline_falls_back_to_synchronous_render_when_async_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Depth 2 requested, but the native client advertises no async
    ReadWorldState support: the pre-existing synchronous path is used."""

    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    client = _Client()
    client.async_supported = False
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=3))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)

    frames = [f for f in harness.slot.frames() if f.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in frames[:3]] == [1, 2, 3]
    assert client.begin_render_calls == 0
    assert client.render_additional_samples == [1, 1, 1]


def test_nonphysics_playback_uses_bounded_async_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    client = _Client()
    client.async_supported = True
    harness = _Harness(monkeypatch, tmp_path, client=client)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=3, timeline_playing=True))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
    assert client.begin_render_calls >= 2
    assert client.render_additional_samples == [1]
    assert client.max_outstanding == 1


def test_pipeline_cancels_pending_tickets_on_session_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session replacement mid-pipeline cancels every outstanding ticket
    instead of leaking it across the boundary (README: cancel() is
    non-blocking and idempotent)."""

    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    client = _Client()
    client.async_supported = True
    client.poll_resolve_after = 1_000_000  # never resolves on its own
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: client.begin_render_calls >= 2)
        pending_before = list(client.pending_samples)
        assert pending_before
        assert not any(
            ticket.cancelled
            for pending in pending_before
            for ticket in pending.tickets.values()
        )

        # A new snapshot with a different camera pose forces a refinement
        # reset (task02-03), which must cancel the outstanding tickets.
        harness.mailbox.write(_snapshot(tx=9.0, max_samples=0))
        assert _wait_until(
            lambda: all(
                ticket.cancelled
                for pending in pending_before
                for ticket in pending.tickets.values()
            )
        )


def test_pipeline_cooldown_falls_back_to_sync_after_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n-03-09: continuous camera motion resets ``_completed_samples`` on
    nearly every tick, which re-enables pipelining the very next tick --
    eagerly starting a new WriteWorldState+ticket that the *next* camera
    change then cancels before it ever resolves. Measured live: 100% of
    pipelined attempts wasted during active navigation, doubling the
    worker's real write/render volume per published frame versus depth=1.
    After an actual cancellation, no *new* pipelined ticket should start
    until a short cooldown elapses -- the loop falls back to the
    synchronous path (matching depth=1) instead of repeating the waste."""

    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_CANCEL_COOLDOWN_MS", "400")
    client = _Client()
    client.async_supported = True
    client.poll_resolve_after = 1_000_000  # never resolves on its own
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: client.begin_render_calls >= 1)
        begin_calls_before_cancel = client.begin_render_calls

        # A new camera pose forces a refinement reset, cancelling the
        # outstanding ticket -- the same trigger as the sibling
        # session-replacement test above. The render thread may already be
        # mid-tick when the new snapshot lands, so at most one more begin
        # can race the reset and get cancelled immediately after -- that is
        # expected. What the cooldown must prevent is any *further* begin
        # while the camera keeps "moving" (further cancels arriving).
        harness.mailbox.write(_snapshot(tx=9.0, max_samples=0))
        renders_at_cancel = client.render_calls
        assert _wait_until(lambda: client.render_calls >= renders_at_cancel + 2)
        begin_calls_after_reset = client.begin_render_calls
        assert begin_calls_after_reset <= begin_calls_before_cancel + 1

        # Several more synchronous samples complete with no *additional*
        # pipelined ticket begun: the cooldown suppressed re-entry.
        renders_at_settle = client.render_calls
        assert _wait_until(lambda: client.render_calls >= renders_at_settle + 20)
        assert client.begin_render_calls == begin_calls_after_reset

        # Once the cooldown elapses, pipelining resumes.
        assert _wait_until(
            lambda: client.begin_render_calls > begin_calls_after_reset,
            timeout=2.0,
        )


def test_pipeline_survives_applied_revision_bump_before_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live scene activation calls ``note_applied_content`` before any FRAME.

    Depth>1 must keep the first in-flight tickets until they resolve instead
    of treating unpublished applied_revision as a per-tick cancel.
    """

    monkeypatch.setenv("OVRTX_RENDER_PIPELINE_DEPTH", "2")
    client = _Client()
    client.async_supported = True
    client.poll_resolve_after = 12
    harness = _Harness(monkeypatch, tmp_path, client=client)
    harness.scheduler.note_applied_content()

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=4))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4, timeout=2.0)

    frames = [frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in frames[:4]] == [1, 2, 3, 4]
    cancelled = [
        ticket.cancelled
        for pending in client.pending_samples
        for ticket in pending.tickets.values()
    ]
    assert cancelled.count(True) < client.begin_render_calls
    assert client.begin_render_calls >= 3


# --- Fresh snapshot: one sample per iteration, refine to max ---------------


def test_fresh_snapshot_acquires_one_sample_per_iteration_to_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot = _snapshot(tx=2.0)

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        # Refinement complete -> the loop parks again (a second timeout=None
        # take after the initial empty park).
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)

    frames = harness.slot.frames()[:4]
    assert [frame.completed_samples for frame in frames] == [1, 2, 3, 4]
    assert all(frame.status == FRAME_STATUS_FRAME for frame in frames)
    assert all(frame.snapshot_key == snapshot.key for frame in frames)
    indices = [frame.publication_index for frame in frames]
    assert indices == sorted(indices) and len(set(indices)) == 4
    # Latency-sensitive viewport refinement advances one sample per native call.
    assert harness.client.starts == 1
    assert harness.loop.diagnostics()["last_reset_reason"] == "camera_changed"


def test_no_sleep_between_publication_and_next_take_while_refining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)

    records = harness.mailbox.records
    snapshot_take = next(
        index for index, (_timeout, taken) in enumerate(records) if taken
    )
    park_after = next(
        index
        for index, (timeout, _taken) in enumerate(records)
        if index > snapshot_take and timeout is None
    )
    refining = [timeout for timeout, _taken in records[snapshot_take + 1 : park_after]]
    # Every mailbox consult between the first render and full refinement is
    # a non-blocking poll: no sleeps while input may be active.
    assert refining
    assert all(timeout == 0.0 for timeout in refining)


# --- Supersession -----------------------------------------------------------


def test_newer_snapshot_supersedes_refinement_and_final_view_refines_fully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot_a = _snapshot(tx=2.0)
    snapshot_b = _snapshot(tx=9.0)

    def write_b_during_first_step(call_index: int) -> None:
        if call_index == 0:
            harness.mailbox.write(snapshot_b)

    harness.client.render_hook = write_b_during_first_step

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 5)

    frames = harness.slot.frames()[:5]
    keys = [frame.snapshot_key for frame in frames]
    # The completed in-flight result remains useful visual feedback, then the
    # newest pose gets the next render call.
    assert keys == [snapshot_a.key] + [snapshot_b.key] * 4
    # The final view refines completely after input stops (ADR 0013).
    assert [frame.completed_samples for frame in frames] == [1, 1, 2, 3, 4]
    assert harness.loop.diagnostics()["snapshots_superseded"] == 1


def test_camera_motion_publishes_completed_visual_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot_a = _snapshot(tx=2.0)
    snapshot_b = _snapshot(tx=9.0)
    second_render_started = threading.Event()
    release_second_render = threading.Event()

    def keep_camera_moving(call_index: int) -> None:
        if call_index == 0:
            harness.mailbox.write(snapshot_b)
        elif call_index == 1:
            second_render_started.set()
            assert release_second_render.wait(WAIT_S)

    harness.client.render_hook = keep_camera_moving

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert second_render_started.wait(WAIT_S)
        frames_during_motion = harness.slot.frames()
        release_second_render.set()

    assert [frame.snapshot_key for frame in frames_during_motion] == [snapshot_a.key]


def test_newer_snapshot_waits_for_at_most_one_stale_unbounded_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot_a = _snapshot(tx=2.0, max_samples=0)
    snapshot_b = _snapshot(tx=9.0, max_samples=0)
    queued_call = 4
    published_before_queue = 0

    def write_b_during_refinement(call_index: int) -> None:
        nonlocal published_before_queue
        if call_index == queued_call:
            published_before_queue = len(harness.slot.frames())
            harness.mailbox.write(snapshot_b)

    harness.client.render_hook = write_b_during_refinement

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert _wait_until(
            lambda: any(
                frame.snapshot_key == snapshot_b.key
                for frame in harness.slot.frames()
            )
        )

    assert harness.client.render_additional_samples[queued_call] == 1
    assert harness.client.render_additional_samples[queued_call + 1] == 1
    assert harness.client.render_camera_translations[
        queued_call : queued_call + 2
    ] == [2.0, 9.0]
    frames_after_queue = harness.slot.frames()[published_before_queue:]
    assert frames_after_queue[0].snapshot_key == snapshot_a.key
    assert frames_after_queue[1].snapshot_key == snapshot_b.key


def test_stable_view_refines_one_sample_at_a_time_to_128(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=128))
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)

    frames = harness.slot.frames()
    assert frames[-1].completed_samples == 128
    assert harness.client.render_additional_samples == [1] * 128


# --- Camera pose as live value update --------------------------------------


def test_camera_pose_applies_as_value_update_without_session_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot_a = _snapshot(tx=2.0)
    snapshot_b = _snapshot(tx=9.0)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 8)

    # Navigation never replaces the session: one start, no deletes.
    assert harness.client.starts == 1
    assert harness.client.deletes == 0
    batches = harness.client.transform_update_batches
    assert len(batches) == 2
    for batch in batches:
        assert [value.prim_path for value in batch] == ["/World/Camera"]
    assert batches[1][0].matrix == [list(row) for row in snapshot_b.camera_matrix]
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["camera_update_count"] == 2
    assert diagnostics["camera_controls_mode"] == "blender_view"
    # Second view change also restarted refinement at min samples.
    assert [frame.completed_samples for frame in harness.slot.frames()[4:8]] == [1, 2, 3, 4]


def test_snapshot_without_camera_target_uses_usd_camera_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot = _snapshot(tx=2.0, camera_prim_path="")

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)

    assert harness.client.transform_update_batches == []
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["camera_update_count"] == 0
    assert diagnostics["camera_controls_mode"] == "usd_camera"
    assert diagnostics["last_reset_reason"] == "snapshot_changed"


# --- Idle parking and wake sources ------------------------------------------


def test_idle_loop_parks_without_polling_and_wakes_on_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        iterations = harness.loop.diagnostics()["iterations"]
        renders = harness.client.render_calls
        time.sleep(0.15)
        # Parked: no busy polling, no renders, no scheduler ticks.
        assert harness.loop.diagnostics()["iterations"] == iterations
        assert harness.client.render_calls == renders
        harness.mailbox.write(_snapshot(tx=9.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 8)

    assert [frame.completed_samples for frame in harness.slot.frames()[4:8]] == [1, 2, 3, 4]


def test_edit_submission_wakes_parked_loop_and_resets_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot = _snapshot(tx=2.0)

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        submission = harness.scheduler.submit_edit(
            _transform_intent(_matrix(5.0))
        )
        assert submission.accepted
        assert _wait_until(lambda: len(harness.slot.frames()) >= 8)

    refreshed = harness.slot.frames()[4:8]
    # The applied value update reset refinement to one sample, then refined
    # the same view identity back to max.
    assert [frame.completed_samples for frame in refreshed] == [1, 2, 3, 4]
    assert all(frame.snapshot_key == snapshot.key for frame in refreshed)
    applied_paths = [
        value.prim_path
        for batch in harness.client.transform_update_batches
        for value in batch
    ]
    assert "/World/Cube" in applied_paths
    # task04-06: the pure value-update reset records the distinct
    # value_edit reason (camera/composition/snapshot changes keep theirs),
    # per applied batch in the iteration records and in diagnostics.
    assert harness.loop.diagnostics()["last_reset_reason"] == "value_edit"
    edit_batch_records = [
        record
        for record in harness.loop.iteration_records()
        if record.get("reset_reason") == "value_edit"
    ]
    assert edit_batch_records


def test_shared_edit_revision_reaches_both_parked_presentations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = _RecordingMailbox()
    second_slot = _RecordingSlot()
    first_loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=second_slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    harness.scheduler.set_edit_wake_hook(
        lambda: (harness.mailbox.wake(), second_mailbox.wake())
    )

    with _running(first_loop) as first_thread, _running(second_loop) as second_thread:
        snapshot = _snapshot(tx=2.0)
        harness.mailbox.write(snapshot)
        second_mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        assert _wait_until(lambda: len(second_slot.frames()) >= 1)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        assert _wait_until(lambda: second_mailbox.park_count() >= 2)
        assert [frame.completed_samples for frame in harness.slot.frames()[-4:]] == [
            1,
            2,
            3,
            4,
        ]
        assert [frame.completed_samples for frame in second_slot.frames()[-4:]] == [
            1,
            2,
            3,
            4,
        ]
        first_count = len(harness.slot.frames())
        second_count = len(second_slot.frames())

        submission = harness.scheduler.submit_edit(
            _transform_intent(_matrix(5.0))
        )

        assert submission.accepted
        assert _wait_until(lambda: len(harness.slot.frames()) >= first_count + 4)
        assert _wait_until(lambda: len(second_slot.frames()) >= second_count + 4)
        assert [
            frame.completed_samples
            for frame in harness.slot.frames()[first_count : first_count + 4]
        ] == [1, 2, 3, 4]
        assert [
            frame.completed_samples
            for frame in second_slot.frames()[second_count : second_count + 4]
        ] == [1, 2, 3, 4]
        assert harness.scheduler.has_pending_view_updates is False
        revision = harness.scheduler.presentation_revision
        assert first_loop.diagnostics()["presented_scheduler_revision"] == revision
        assert second_loop.diagnostics()["presented_scheduler_revision"] == revision
        assert harness.slot.frames()[-1].presentation_revision == revision
        assert harness.slot.frames()[-1].applied_revision > 0
        assert second_slot.frames()[-1].presentation_revision == revision

        # A generation activation/replay has no queued presentation revision,
        # but every parked pane must still discard its pre-activation frame.
        first_count = len(harness.slot.frames())
        second_count = len(second_slot.frames())
        harness.scheduler.note_applied_content()
        assert _wait_until(lambda: len(harness.slot.frames()) >= first_count + 4)
        assert _wait_until(lambda: len(second_slot.frames()) >= second_count + 4)
        assert [
            frame.completed_samples
            for frame in harness.slot.frames()[first_count : first_count + 4]
        ] == [1, 2, 3, 4]
        assert [
            frame.completed_samples
            for frame in second_slot.frames()[second_count : second_count + 4]
        ] == [1, 2, 3, 4]


def test_shared_edit_reaches_two_continuously_sampling_presentations_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = _RecordingMailbox()
    second_slot = _RecordingSlot()
    first_loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=second_slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    harness.scheduler.set_edit_wake_hook(
        lambda: (harness.mailbox.wake(), second_mailbox.wake())
    )
    first_snapshot = _snapshot(tx=2.0, max_samples=0)
    second_snapshot = _snapshot(tx=9.0, max_samples=0)

    with _running(first_loop) as first_thread, _running(second_loop) as second_thread:
        harness.mailbox.write(first_snapshot)
        second_mailbox.write(second_snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: len(second_slot.frames()) >= 4)
        assert first_loop.diagnostics()["completed_samples"] >= 4
        assert second_loop.diagnostics()["completed_samples"] >= 4
        first_count = len(harness.slot.frames())
        second_count = len(second_slot.frames())

        submission = harness.scheduler.submit_edit(_transform_intent(_matrix(5.0)))

        assert submission.accepted
        assert _wait_until(lambda: len(harness.slot.frames()) > first_count)
        assert _wait_until(lambda: len(second_slot.frames()) > second_count)
        revision = harness.scheduler.presentation_revision
        assert _wait_until(
            lambda: first_loop.diagnostics()["presented_scheduler_revision"]
            == revision
        )
        assert _wait_until(
            lambda: second_loop.diagnostics()["presented_scheduler_revision"]
            == revision
        )

    assert harness.slot.frames()[-1].snapshot_key == first_snapshot.key
    assert second_slot.frames()[-1].snapshot_key == second_snapshot.key
    assert first_loop.diagnostics()["completed_samples"] > 0
    assert second_loop.diagnostics()["completed_samples"] > 0
    rendered_cameras = list(
        zip(
            harness.client.render_thread_idents,
            harness.client.render_camera_translations,
        )
    )
    assert {
        camera for thread_ident, camera in rendered_cameras
        if thread_ident == first_thread.ident
    } == {2.0}
    assert {
        camera for thread_ident, camera in rendered_cameras
        if thread_ident == second_thread.ident
    } == {9.0}
    assert sum(
        value.prim_path == "/World/Cube"
        for batch in harness.client.transform_update_batches
        for value in batch
    ) == 1


def test_shared_continuous_presentations_rebind_projection_per_pane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.base_request = replace(
        harness.base_request,
        camera_projection=_projection(35.0),
        camera_value_route_classes=("clip", "ortho", "projection"),
    )
    harness.controller.ensure(harness.base_request)
    harness.controller._allow_serialized_threads()
    second_mailbox = _RecordingMailbox()
    second_slot = _RecordingSlot()
    first_loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
        camera_value_probe=_honored_camera_probe(),
    )
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=second_slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
        camera_value_probe=_honored_camera_probe(),
    )

    with _running(first_loop) as first_thread, _running(second_loop) as second_thread:
        harness.mailbox.write(
            _snapshot(tx=2.0, max_samples=0, camera_projection=_projection(35.0))
        )
        second_mailbox.write(
            _snapshot(tx=9.0, max_samples=0, camera_projection=_projection(85.0))
        )
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: len(second_slot.frames()) >= 4)
        assert _wait_until(
            lambda: all(
                any(
                    ident == thread.ident and focal_length == expected
                    for ident, focal_length in zip(
                        harness.client.render_thread_idents,
                        harness.client.render_focal_lengths,
                    )
                )
                for thread, expected in (
                    (first_thread, 35.0),
                    (second_thread, 85.0),
                )
            )
        )
        scheduler_revision = harness.scheduler.presentation_revision
        second_completed = second_slot.frames()[-1].completed_samples
        second_count = len(second_slot.frames())
        harness.mailbox.write(
            _snapshot(tx=2.0, max_samples=0, camera_projection=_projection(45.0))
        )
        assert _wait_until(
            lambda: any(
                ident == first_thread.ident and focal_length == 45.0
                for ident, focal_length in zip(
                    harness.client.render_thread_idents,
                    harness.client.render_focal_lengths,
                )
            )
        ), first_loop.diagnostics()["last_failure_detail"]
        assert _wait_until(lambda: len(second_slot.frames()) > second_count)
        assert second_slot.frames()[second_count].completed_samples > second_completed
        assert harness.scheduler.presentation_revision == scheduler_revision

    rendered = zip(
        harness.client.render_thread_idents,
        harness.client.render_focal_lengths,
    )
    by_thread: dict[int, set[float | None]] = {}
    for thread_ident, focal_length in rendered:
        by_thread.setdefault(thread_ident, set()).add(focal_length)
    assert by_thread[first_thread.ident] - {None} == {35.0, 45.0}
    assert by_thread[second_thread.ident] - {None} == {85.0}


def test_shared_continuous_presentations_use_one_session_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = _RecordingMailbox()
    second_slot = _RecordingSlot()
    first_loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=SessionLifecycleHooks(
            ensure_session=lambda request: (
                harness.controller.ensure(request),
                harness.scheduler,
            )[1],
            replacement_reason=harness.controller.would_replace,
            retry_allowed=lambda: True,
        ),
        owns_scheduler=False,
    )
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=second_slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )

    with _running(first_loop):
        harness.mailbox.write(
            _snapshot(tx=2.0, width=640, height=480, max_samples=0)
        )
        with _running(second_loop):
            second_mailbox.write(
                _snapshot(tx=9.0, width=320, height=720, max_samples=0)
            )
            assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
            assert _wait_until(lambda: len(second_slot.frames()) >= 4)
            shared_starts = harness.client.starts
            harness.mailbox.write(
                _snapshot(tx=3.0, width=800, height=600, max_samples=0)
            )
        assert _wait_until(lambda: harness.controller._spec.width == 800)
        assert harness.client.starts == shared_starts
        restored_spec = harness.controller._spec
        assert restored_spec.width == 800
        assert restored_spec.height == 600

    assert harness.client.deletes == harness.client.starts - 1


def test_shared_projection_takeover_probes_accept_but_ignore_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.base_request = replace(
        harness.base_request,
        camera_projection=_projection(35.0),
        camera_value_route_classes=("clip", "ortho", "projection"),
    )
    harness.controller.ensure(harness.base_request)
    harness.controller._allow_serialized_threads()
    second_mailbox = _RecordingMailbox()
    second_slot = _RecordingSlot()
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=second_slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    first_loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )

    with _running(first_loop), _running(second_loop):
        harness.mailbox.write(
            _snapshot(tx=2.0, max_samples=0, camera_projection=_projection(35.0))
        )
        second_mailbox.write(
            _snapshot(tx=9.0, max_samples=0, camera_projection=_projection(85.0))
        )
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for slot in (harness.slot, second_slot)
                for frame in slot.frames()
            )
        )
        failures = [
            frame
            for slot in (harness.slot, second_slot)
            for frame in slot.frames()
            if frame.status == FRAME_STATUS_FAILED
        ]
        assert "shared viewport projections require live camera value updates" in (
            failures[-1].detail
        )


def test_rapid_edits_apply_latest_value_per_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # task04-06: rapid successive edits to one target coalesce latest-wins
    # before render-thread application (no per-intermediate-value RPCs);
    # distinct targets in the batch all apply.
    harness = _Harness(monkeypatch, tmp_path)
    stale = _matrix(3.0)
    other = _matrix(4.0)
    newest = _matrix(5.0)

    with _running(harness.loop):
        # Queued before the first view: all three edits land in the first
        # iteration's single applied batch.
        assert harness.scheduler.submit_edit(_transform_intent(stale)).accepted
        assert harness.scheduler.submit_edit(
            _transform_intent(other, prim_path="/World/Cube_B")
        ).accepted
        assert harness.scheduler.submit_edit(_transform_intent(newest)).accepted
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)

    # The shared scheduler applies one value per distinct edit target first,
    # newest value winning per target; the presentation camera binds after
    # that accepted content and before acquisition.
    edit_batch, camera_batch = harness.client.transform_update_batches
    assert [value.prim_path for value in edit_batch] == [
        "/World/Cube",
        "/World/Cube_B",
    ]
    assert [value.prim_path for value in camera_batch] == ["/World/Camera"]
    assert edit_batch[0].matrix == [list(row) for row in newest]
    assert edit_batch[1].matrix == [list(row) for row in other]
    records = harness.loop.iteration_records()
    # The camera change owns the reset reason for the applied batch.
    assert records[0]["reset_reason"] == "camera_changed"
    # Refinement iterations apply no batch.
    assert "reset_reason" not in records[1]


def test_session_activation_adopts_authoring_scheduler_without_owning_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    activated: list[object] = []
    wakes: list[int] = []
    shutdowns: list[int] = []
    original_shutdown = harness.scheduler.shutdown
    harness.scheduler.shutdown = lambda: shutdowns.append(1) or original_shutdown()  # type: ignore[method-assign]
    dispatcher = lambda: wakes.append(1)
    harness.scheduler.set_edit_wake_hook(dispatcher)
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=None,
        owns_scheduler=False,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=SessionLifecycleHooks(
            ensure_session=lambda request: (
                activated.append(request) or harness.scheduler
            ),
            replacement_reason=lambda _request: "",
            retry_allowed=lambda: True,
        ),
    )

    with _running(loop):
        assert loop._scheduler is None
        harness.mailbox.write(_snapshot())
        assert _wait_until(lambda: loop._scheduler is harness.scheduler)
        assert harness.scheduler._wake_hook is dispatcher

    assert len(activated) == 1
    assert harness.scheduler._wake_hook is dispatcher
    assert shutdowns == []


def test_session_activation_cannot_replace_an_existing_runtime_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    other = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=SessionLifecycleHooks(
            ensure_session=lambda _request: other,
            replacement_reason=lambda _request: "",
            retry_allowed=lambda: False,
        ),
    )

    with _running(loop):
        harness.mailbox.write(_snapshot())
        assert _wait_until(lambda: bool(harness.slot.frames()))

    assert harness.slot.frames()[0].status == FRAME_STATUS_FAILED
    assert "replace the runtime scheduler" in harness.slot.frames()[0].detail
    assert loop._scheduler is harness.scheduler


def test_retryable_activation_with_pending_sim_retries_without_new_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    attempts = 0

    def ensure(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RenderClientError("physics runtime busy")
        return harness.scheduler

    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=SessionLifecycleHooks(
            ensure_session=ensure,
            replacement_reason=lambda _request: "",
            retry_allowed=lambda: True,
        ),
        failure_retry_backoff_seconds=0.01,
    )
    assert harness.scheduler.submit_edit(_sim_intent(tx=5.0)).accepted

    with _running(loop):
        harness.mailbox.write(_snapshot())
        assert _wait_until(lambda: attempts >= 2)

    assert attempts == 2


def test_scheduler_due_work_parks_until_publication_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeScheduler:
        has_pending_view_updates = False

        def __init__(self) -> None:
            self.wake_hook = None
            self.edits: list[EditIntent] = []

        def set_edit_wake_hook(self, hook) -> None:
            self.wake_hook = hook

        def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
            self.edits.append(intent)
            if self.wake_hook is not None:
                self.wake_hook()
            return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued")

    class _FakeController:
        def __init__(self) -> None:
            self.adopted = 0
            self.apply_calls = 0
            self.applied = threading.Event()
            self.render_calls = 0
            self.should_request_redraw = True

        def adopt_owning_thread(self) -> None:
            self.adopted += 1

        def apply_runtime_updates(self, operation) -> RuntimeTickResult:
            self.apply_calls += 1
            self.applied.set()
            return RuntimeTickResult(
                status=RuntimeTickStatus.NOOP,
                enabled=True,
                should_request_redraw=self.should_request_redraw,
                generation=3,
            )

        def render(self, request, *, additional_samples: int) -> RenderResult:
            self.render_calls += 1
            return RenderResult(
                width=1,
                height=1,
                rgba8=b"\x00\x00\x00\xff",
                completed_samples=additional_samples,
                session_completed_samples=self.render_calls,
                simulation_time_ns=0,
            )

    controller = _FakeController()
    scheduler = _FakeScheduler()
    mailbox = _RecordingMailbox()
    slot = _RecordingSlot()
    loop = LatestViewRenderLoop(
        mailbox=mailbox,
        frame_slot=slot,
        controller=controller,
        scheduler=scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            _request(Path("unused")), snapshot
        ),
    )

    with _running(loop):
        mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(slot.frames()) >= 1)
        # Physics playback parks after completing available work. A pose
        # publication wakes exactly one new scheduler/render iteration.
        assert _wait_until(lambda: mailbox.park_count() >= 2)
        parked_before = mailbox.park_count()
        applies = controller.apply_calls
        time.sleep(0.05)
        assert controller.apply_calls == applies
        assert scheduler.wake_hook is not None
        controller.applied.clear()
        scheduler.wake_hook()
        assert controller.applied.wait(timeout=0.03)
        assert controller.apply_calls == applies + 1
        assert _wait_until(lambda: mailbox.park_count() > parked_before)
        controller.should_request_redraw = False

    assert controller.adopted == 1
    assert slot.frames()[0].generation == 3


# --- Failure publication -----------------------------------------------------


def test_render_failure_publishes_failed_state_and_loop_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.fail_render = True

    with _running(harness.loop) as thread:
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        failure = harness.slot.frames()[0]
        assert failure.status == FRAME_STATUS_FAILED
        assert "RenderClientError" in failure.detail
        assert "render failed" in failure.detail
        assert thread.is_alive()
        # A failed view parks (no busy retry of the same failure).
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        renders = harness.client.render_calls
        time.sleep(0.1)
        assert harness.client.render_calls == renders
        # Fresh input retries: same view identity is enough.
        harness.client.fail_render = False
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 5)

    frames = harness.slot.frames()
    assert [frame.completed_samples for frame in frames[1:5]] == [1, 2, 3, 4]
    assert all(frame.status == FRAME_STATUS_FRAME for frame in frames[1:5])
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["failures"] == 1
    assert diagnostics["failed_state"] is False


def test_persistent_failure_with_snapshot_noise_has_bounded_rpcs_and_publications(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.fail_render = True
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    for _ in range(1000):
        harness.loop._adopt_snapshot(_snapshot(tx=2.0, max_samples=1))
        harness.loop._iterate()
        clock[0] += 0.01
    assert harness.client.render_calls == 3
    failures = [f for f in harness.slot.frames() if f.status == FRAME_STATUS_FAILED]
    assert len(failures) == 3
    assert all("render failed" in f.detail for f in failures)
    assert not harness.loop._retry_allowed()
    harness.client.fail_render = False
    harness.loop._adopt_snapshot(_snapshot(tx=3.0, max_samples=1))
    harness.loop._iterate()
    assert harness.slot.frames()[-1].status == FRAME_STATUS_FRAME
    harness.controller.shutdown()


def test_camera_validation_runs_before_scheduler_and_transform_writes(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    def invalid(*args):
        raise ValueError("injected complete camera validation failure")
    harness.client.validate_attribute_values = invalid
    harness.loop._adopt_snapshot(_snapshot(tx=2.0, max_samples=1))
    harness.loop._iterate()
    assert harness.client.transform_update_batches == []
    assert harness.client.render_calls == 0
    assert "injected complete camera validation failure" in harness.slot.frames()[-1].detail
    harness.controller.shutdown()


@pytest.mark.parametrize("fault", ["prefix", "ambiguous"])
def test_one_explicit_recovery_merges_newer_camera_and_queued_edits(tmp_path, monkeypatch, fault):
    # Exercise the production SDK client/controller/scheduler/producer path. Values
    # adopted during quarantine have not yet reached the client's desired ledger.
    from public_sdk_recovery_fixture import fault_client
    real, native, identity = fault_client(tmp_path, monkeypatch)
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.update_transforms = lambda sid, values: real.update_transforms(identity, values)
    harness.client.update_attribute_values = lambda sid, values: real.update_attribute_values(identity, values)
    harness.client.validate_attribute_values = lambda sid, values: real.validate_attribute_values(identity, values)
    harness.client.recover_value_updates = lambda sid, **kwargs: real.recover_value_updates(identity, **kwargs)
    harness.client.begin_recovery_update = lambda sid: real.begin_recovery_update(identity)
    harness.client.end_recovery_update = lambda sid, **kwargs: real.end_recovery_update(identity, **kwargs)
    harness.loop._camera_value_probe = _honored_camera_probe()
    native.remote_fault = fault
    harness.loop._adopt_snapshot(_snapshot(tx=2., max_samples=1, camera_projection=_projection(40.)))
    harness.loop._iterate()
    assert harness.loop._failure_permanent, harness.slot.frames()[-1].detail
    harness.loop._adopt_snapshot(_snapshot(tx=3., max_samples=1,
        camera_projection=replace(_projection(55.), clipping_range=(0.5, 800.))))
    harness.scheduler.submit_edit(_transform_intent(_matrix(7.)))
    harness.loop._iterate()
    assert native.write_attempts == 1
    native.remote_fault = ""
    assert harness.controller.recover_value_updates()
    assert native.accepted == {}  # Latest producers must contribute before replay.
    harness.loop._iterate()
    assert harness.slot.frames()[-1].status == FRAME_STATUS_FRAME
    assert native.accepted[("/World/Camera", "omni:xform")][0][3] == 3.
    assert native.accepted[("/World/Cube", "omni:xform")][0][3] == 7.
    assert native.accepted[("/World/Camera", "clippingRange")] == [0.5, 800.]
    assert native.accepted[("/World/Camera", "focalLength")] == 55.
    assert sum(op == "recover" for op, _ in native.calls) == 1
    assert not harness.scheduler.has_pending_view_updates
    assert not real.states[identity].recovery_pending
    assert not harness.loop._failure_permanent
    assert not native.shutdown_called
    harness.controller.shutdown()
    real.shutdown()


def test_composition_failure_publishes_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _ExplodingScheduler:
        has_pending_view_updates = False

        def set_edit_wake_hook(self, hook) -> None:
            pass

        def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
            return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued")

        def tick_viewport(self, request, *, ovrtx_updates, project_complete_pose):
            raise SharedStageCompositionError("stage exploded")

    harness = _Harness(monkeypatch, tmp_path)
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=_ExplodingScheduler(),
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
    )

    with _running(loop) as thread:
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        assert thread.is_alive()

    failure = harness.slot.frames()[0]
    assert failure.status == FRAME_STATUS_FAILED
    assert "Shared-stage composition failed" in failure.detail
    assert "stage exploded" in failure.detail


def test_failed_tick_status_publishes_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingTickScheduler:
        has_pending_view_updates = False

        def set_edit_wake_hook(self, hook) -> None:
            pass

        def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
            return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued")

        def tick_viewport(self, request, *, ovrtx_updates, project_complete_pose):
            return RuntimeTickResult(
                status=RuntimeTickStatus.FAILED,
                enabled=True,
                skipped_reason="ovphysx_reset_failed",
            )

    harness = _Harness(monkeypatch, tmp_path)
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=_FailingTickScheduler(),
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
    )

    with _running(loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)

    failure = harness.slot.frames()[0]
    assert failure.status == FRAME_STATUS_FAILED
    assert "ovphysx_reset_failed" in failure.detail


def test_rejected_camera_update_retries_on_fresh_same_key_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.transform_failures_remaining = 1
    snapshot = _snapshot(tx=2.0)
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snap: viewport_handoff.request_from_snapshot(
            harness.base_request, snap
        ),
    )

    with _running(loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        failure = harness.slot.frames()[0]
        assert failure.status == FRAME_STATUS_FAILED
        assert "rejected_for_test" in failure.detail
        # Fresh input with the SAME view identity must retry the camera pose
        # because the rejected update never reached the worker.
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 5)

    assert harness.client.transform_update_attempts == 2
    batches = harness.client.transform_update_batches
    assert [value.prim_path for batch in batches for value in batch] == [
        "/World/Camera"
    ]
    frames = harness.slot.frames()[1:5]
    assert [frame.completed_samples for frame in frames] == [1, 2, 3, 4]
    assert loop.diagnostics()["camera_update_count"] == 1


# --- Stability of the view identity ------------------------------------------


def test_same_key_snapshot_does_not_reset_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        publications = len(harness.slot.frames())
        # Same view identity, different timeline cursor: taken and ticked,
        # but no refinement reset and no camera resubmission.
        harness.mailbox.write(_snapshot(tx=2.0, timeline_frame=7))
        assert _wait_until(
            lambda: harness.loop.diagnostics()["snapshots_taken"] >= 2
        )
        assert _wait_until(lambda: harness.mailbox.park_count() >= 3)
        assert len(harness.slot.frames()) == publications

    diagnostics = harness.loop.diagnostics()
    assert diagnostics["completed_samples"] == 4
    assert diagnostics["camera_update_count"] == 1
    assert diagnostics["snapshots_superseded"] == 0


def test_sample_limit_changes_preserve_progress_and_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=SessionLifecycleHooks(
            ensure_session=lambda request: (
                harness.controller.ensure(request) and harness.scheduler
            ),
            replacement_reason=harness.controller.would_replace,
            retry_allowed=lambda: True,
        ),
    )

    with _running(loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=4))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        session_counts = (harness.client.starts, harness.client.deletes)

        harness.mailbox.write(_snapshot(tx=2.0, max_samples=6))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 6)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 3)

        publications = len(harness.slot.frames())
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=3))
        assert _wait_until(
            lambda: loop.diagnostics()["snapshots_taken"] >= 3
        )
        assert _wait_until(lambda: harness.mailbox.park_count() >= 4)
        assert len(harness.slot.frames()) == publications

        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= publications + 4)

    assert [frame.completed_samples for frame in harness.slot.frames()[:10]] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    diagnostics = loop.diagnostics()
    assert diagnostics["completed_samples"] >= 10
    assert diagnostics["max_samples"] == 0
    assert diagnostics["camera_update_count"] == 1
    assert diagnostics["session_replacements"] == 0
    assert diagnostics["resync_publications"] == 0
    assert (harness.client.starts, harness.client.deletes) == session_counts


# --- Thread integration -------------------------------------------------------


def test_loop_runs_as_render_thread_submit_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    thread = ViewportRenderThread("latest-view-loop")
    thread.start()
    try:
        thread.submit(harness.loop.run, label="latest_view_loop")
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
        # The loop adopted the controller as its first act on the thread.
        assert harness.controller.diagnostics()["rpc_thread"]["adopted"] is True
        harness.loop.request_stop()
        # run() returned: the thread serves subsequent commands.
        assert thread.call(lambda: "alive").result(WAIT_S) == "alive"
    finally:
        outcome = thread.stop()
    assert outcome["joined"] is True
    assert outcome["leaked_thread"] is False


def test_run_rejects_concurrent_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        assert _wait_until(lambda: harness.loop.diagnostics()["running"])
        with pytest.raises(RenderThreadError, match="already running"):
            harness.loop.run()


def test_setup_failure_releases_running_latch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Controller adoption failing is thread-fatal per the 02-01 contract,
    # but the loop instance must not latch "running" (diagnostics would
    # lie and a retry would be misreported as a concurrent run).
    harness = _Harness(monkeypatch, tmp_path)

    def _explode() -> None:
        raise RuntimeError("adoption failed")

    monkeypatch.setattr(harness.controller, "adopt_owning_thread", _explode)

    with pytest.raises(RuntimeError, match="adoption failed"):
        harness.loop.run()
    assert harness.loop.diagnostics()["running"] is False
    # A retry surfaces the real failure again, not "already running".
    with pytest.raises(RuntimeError, match="adoption failed"):
        harness.loop.run()


def test_unexpected_loop_error_uninstalls_wake_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unexpected exception fails the thread (02-01 contract), but the
    # edit wake hook must be uninstalled on the way out: a dead loop must
    # not keep receiving wakes from main-thread edit submissions.
    harness = _Harness(monkeypatch, tmp_path)

    def _boom(snapshot: ViewSnapshot):
        raise ValueError("request translation exploded")

    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=_boom,
    )
    errors: list[BaseException] = []

    def _target() -> None:
        try:
            loop.run()
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            errors.append(exc)

    thread = threading.Thread(target=_target, name="latest-view-fatal", daemon=True)
    thread.start()
    harness.mailbox.write(_snapshot(tx=2.0))
    thread.join(WAIT_S)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert loop.diagnostics()["running"] is False
    wakes_before = harness.mailbox.diagnostics()["wakes"]
    submission = harness.scheduler.submit_edit(_transform_intent(_matrix(5.0)))
    assert submission.accepted
    assert harness.mailbox.diagnostics()["wakes"] == wakes_before


# --- Tick-result handoff seam (task02-07) ------------------------------------


def _sim_intent(tx: float = 2.0) -> EditIntent:
    """Sim-authoritative initial-condition pose intent."""

    return EditIntent(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.SIM,
        **edit_location(
            usd_prim_path="/World/Cube",
            usd_attribute="omni:xform",
            blender_property_path="matrix_world",
        ),
        value={"translate": (float(tx), 0.0, 0.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    )


def test_tick_result_sink_receives_successful_tick_on_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    sink_calls: list[tuple] = []
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        tick_result_sink=lambda result, request: sink_calls.append(
            (result, request, threading.get_ident())
        ),
    )
    snapshot = _snapshot(tx=2.0, timeline_frame=7)

    with _running(loop) as thread:
        harness.mailbox.write(snapshot)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
        loop_ident = thread.ident

    assert sink_calls
    result, request, ident = sink_calls[0]
    # The sink runs on the loop's thread, never the writer's.
    assert ident == loop_ident
    assert ident != threading.get_ident()
    assert result.status is not RuntimeTickStatus.FAILED
    # The request is the snapshot-derived one: timeline facts ride the
    # snapshot (no bpy.context reads on the thread).
    assert request.timeline_frame == 7
    assert request.camera_matrix == snapshot.camera_matrix
    assert loop.diagnostics()["last_timeline_reset"] is False


def test_tick_result_waits_for_synchronous_image_before_mirroring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    sink_calls = []
    entered, release = threading.Event(), threading.Event()

    def block_first_render(call_index):
        if call_index == 0:
            entered.set()
            assert release.wait(WAIT_S)

    harness.client.render_hook = block_first_render
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox, frame_slot=harness.slot,
        controller=harness.controller, scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        tick_result_sink=lambda result, request: sink_calls.append((result, request)),
    )
    with _running(loop):
        try:
            harness.mailbox.write(_snapshot(tx=2.0, timeline_frame=7))
            assert entered.wait(WAIT_S)
            assert sink_calls == []
            assert harness.slot.frames() == []
        finally:
            release.set()
        assert _wait_until(lambda: bool(harness.slot.frames()))
        assert sink_calls


def test_tick_result_sink_not_called_for_failed_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingTickScheduler:
        has_pending_view_updates = False

        def set_edit_wake_hook(self, hook) -> None:
            pass

        def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
            return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued")

        def tick_viewport(self, request, *, ovrtx_updates, project_complete_pose):
            return RuntimeTickResult(
                status=RuntimeTickStatus.FAILED,
                enabled=True,
                skipped_reason="ovphysx_reset_failed",
            )

    harness = _Harness(monkeypatch, tmp_path)
    sink_calls: list[tuple] = []
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=_FailingTickScheduler(),
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        tick_result_sink=lambda result, request: sink_calls.append((result, request)),
    )

    with _running(loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)

    assert harness.slot.frames()[0].status == FRAME_STATUS_FAILED
    assert sink_calls == []


def test_tick_result_hands_off_after_render_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Solver state still hands off when rendering fails, allowing restoration
    # and playback-lock transitions without a successful image publication.
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.fail_render = True
    sink_calls: list[tuple] = []
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        tick_result_sink=lambda result, request: sink_calls.append((result, request)),
    )

    with _running(loop) as thread:
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        assert thread.is_alive()

    assert harness.slot.frames()[0].status == FRAME_STATUS_FAILED
    assert len(sink_calls) == 1


def test_loop_shuts_down_scheduler_on_loop_thread_before_join_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _ShutdownRecordingScheduler:
        """Delegates to the real scheduler; records shutdown thread idents."""

        def __init__(self, inner: RuntimeScheduler) -> None:
            self._inner = inner
            self.shutdown_idents: list[int] = []
            self.tick_idents: list[int] = []

        @property
        def has_pending_view_updates(self) -> bool:
            return self._inner.has_pending_view_updates

        @property
        def has_pending_sim_updates(self) -> bool:
            return self._inner.has_pending_sim_updates

        def set_edit_wake_hook(self, hook) -> None:
            self._inner.set_edit_wake_hook(hook)

        def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
            return self._inner.submit_edit(intent)

        def tick_viewport(self, request, *, ovrtx_updates, project_complete_pose):
            self.tick_idents.append(threading.get_ident())
            return self._inner.tick_viewport(
                request,
                ovrtx_updates=ovrtx_updates,
                project_complete_pose=project_complete_pose,
            )

        def shutdown(self) -> None:
            self.shutdown_idents.append(threading.get_ident())
            self._inner.shutdown()

    harness = _Harness(monkeypatch, tmp_path)
    scheduler = _ShutdownRecordingScheduler(harness.scheduler)
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
    )
    thread = ViewportRenderThread("scheduler-shutdown-ordering")
    thread.start()
    try:
        thread.submit(loop.run, label="latest_view_loop")
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
        assert scheduler.shutdown_idents == []
        loop.request_stop()
    finally:
        outcome = thread.stop()
    # Shutdown ordering (task02-07): loop exit -> scheduler.shutdown() on
    # the render thread, completed before the join returned.
    assert outcome["joined"] is True
    assert len(scheduler.shutdown_idents) == 1
    assert scheduler.shutdown_idents == scheduler.tick_idents[:1]
    assert scheduler.shutdown_idents[0] != threading.get_ident()


def test_sim_edit_wakes_parked_loop_once_without_busy_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        iterations = harness.loop.diagnostics()["iterations"]
        parks = harness.mailbox.park_count()
        submission = harness.scheduler.submit_edit(_sim_intent(tx=5.0))
        assert submission.accepted
        # The sim-edit wake runs one tick (the tick is what applies the
        # pending initial-condition value when physics is enabled)...
        assert _wait_until(
            lambda: harness.loop.diagnostics()["iterations"] > iterations
        )
        # ...and then re-parks. With physics disabled the tick cannot
        # drain the pending sim edit, and the loop must not busy-poll on
        # it (drain-once semantics: the next edit wake retries).
        assert _wait_until(lambda: harness.mailbox.park_count() > parks)
        settled = harness.loop.diagnostics()["iterations"]
        time.sleep(0.15)
        assert harness.loop.diagnostics()["iterations"] == settled
        assert harness.scheduler.has_pending_sim_updates is True


# --- Wake-hook seam -----------------------------------------------------------


def test_view_update_stream_wake_hook_fires_on_queue() -> None:
    stream = ViewUpdateStream()
    wakes: list[int] = []
    stream.set_wake_hook(lambda: wakes.append(1))

    stream.queue(_transform_intent(_matrix(5.0)))
    assert wakes == [1]

    stream.set_wake_hook(None)
    stream.queue(_transform_intent(_matrix(6.0)))
    assert wakes == [1]


def test_scheduler_exposes_pending_view_updates_and_wake_hook() -> None:
    scheduler = RuntimeScheduler(
        config_factory=lambda path: SimpleNamespace(enabled=False)
    )
    wakes: list[int] = []
    scheduler.set_edit_wake_hook(lambda: wakes.append(1))
    assert scheduler.has_pending_view_updates is False
    assert scheduler.presentation_revision == 0

    submission = scheduler.submit_edit(_transform_intent(_matrix(5.0)))

    assert submission.accepted
    assert wakes == [1]
    assert scheduler.presentation_revision == 1
    assert scheduler.has_pending_view_updates is True


def test_scheduler_exposes_pending_sim_updates_and_wake_hook() -> None:
    # Sim edits are cross-thread once the scheduler ticks on the render
    # thread (task02-07): submission wakes the loop and the pending state
    # is visible for the due-work check.
    scheduler = RuntimeScheduler(
        config_factory=lambda path: SimpleNamespace(enabled=False)
    )
    wakes: list[int] = []
    scheduler.set_edit_wake_hook(lambda: wakes.append(1))
    assert scheduler.has_pending_sim_updates is False
    assert scheduler.presentation_revision == 0

    submission = scheduler.submit_edit(_sim_intent(tx=5.0))

    assert submission.accepted
    assert wakes == [1]
    assert scheduler.presentation_revision == 1
    assert scheduler.has_pending_sim_updates is True
    assert scheduler.has_pending_view_updates is False


# --- Regression: loop follows a mid-session engine controller swap -----------
# The authored ensure (activate_for_viewport) can replace the engine's
# controller after the loop exists — on an output-shape replacement or a
# Cycles->OVRTX re-create. A captured reference would render the torn-down one
# forever; the provider makes the loop follow the live controller.


def test_active_controller_resolves_current_and_raises_when_absent() -> None:
    holder = SimpleNamespace(controller="controller-a")
    loop = LatestViewRenderLoop(
        mailbox=CameraRequestMailbox(),
        frame_slot=LatestFrameSlot(),
        controller_provider=lambda: holder.controller,
        scheduler=RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        ),
        request_for_snapshot=lambda snapshot: snapshot,
    )

    assert loop._active_controller() == "controller-a"
    holder.controller = "controller-b"
    assert loop._active_controller() == "controller-b"
    # A cleared controller (teardown) surfaces as the retryable render error
    # the loop's failure policy already handles, not an AttributeError crash.
    holder.controller = None
    with pytest.raises(RenderClientError):
        loop._active_controller()


@pytest.mark.parametrize("provider_state", ["dead_rna", "cleared"])
def test_stop_and_detach_survive_released_engine_provider(tmp_path, monkeypatch, provider_state):
    harness = _Harness(monkeypatch, tmp_path)
    released = threading.Event()
    def provider():
        if released.is_set():
            if provider_state == "dead_rna":
                raise ReferenceError("StructRNA of type OvrtxExampleRenderEngine has been removed")
            return None
        return harness.controller
    harness.loop._controller_provider = provider
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(max_samples=1))
        assert _wait_until(lambda: len(harness.slot.frames()) == 1)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        assert id(harness.loop) in harness.controller._presentations
        released.set()
        # Rendering must not fall back to stale ownership after RNA removal.
        with pytest.raises(RenderClientError):
            harness.loop._active_controller()
        # The teardown path can still wake a queued transaction and detach.
        harness.loop.request_stop()
        assert _wait_until(lambda: not harness.loop._running)
    assert id(harness.loop) not in harness.controller._presentations
    assert harness.client.render_calls == 1
    harness.controller.shutdown()


@pytest.mark.parametrize("stop_requested", [True, False])
@pytest.mark.parametrize("provider_state", ["dead_rna", "cleared", "render_error"])
def test_stop_during_idle_controller_lookup_runs_queued_teardown(
    tmp_path, monkeypatch, provider_state, stop_requested
):
    harness = _Harness(monkeypatch, tmp_path)
    entered = threading.Event()
    resume = threading.Event()
    released = threading.Event()
    work_due = harness.loop._work_due_without_new_snapshot

    def provider():
        if released.is_set():
            if provider_state == "dead_rna":
                raise ReferenceError("StructRNA has been removed")
            if provider_state == "render_error":
                raise RenderClientError("unrelated transport failure")
            return None
        return harness.controller

    def gated_work_due():
        entered.set()
        assert resume.wait(WAIT_S)
        return work_due()

    harness.loop._controller_provider = provider
    thread = ViewportRenderThread("stop-during-controller-lookup")
    thread.start()
    try:
        thread.submit(harness.loop.run, label="latest-view-loop")
        harness.mailbox.write(_snapshot(max_samples=1))
        assert _wait_until(lambda: len(harness.slot.frames()) == 1)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        monkeypatch.setattr(harness.loop, "_work_due_without_new_snapshot", gated_work_due)
        harness.mailbox.wake()
        assert entered.wait(WAIT_S)
        # The loop has passed its stop checks, just as fixed03 did. Queue the
        # real controller cleanup behind run(), then invalidate the provider.
        if stop_requested:
            harness.loop.request_stop()
        teardown = thread.call(harness.controller.shutdown, label="teardown")
        released.set()
        resume.set()
        expected_failure = not stop_requested or provider_state == "render_error"
        if expected_failure:
            with pytest.raises(RenderThreadError, match="render thread failed"):
                teardown.result(WAIT_S)
        else:
            teardown.result(WAIT_S)
    finally:
        released.clear()
        harness.loop.request_stop()
        resume.set()
        outcome = thread.stop(WAIT_S)
    assert outcome["joined"] is True
    assert outcome["leaked_thread"] is False
    diagnostics = thread.diagnostics()
    if expected_failure:
        assert diagnostics["status"] == "failed"
        assert diagnostics["commands_failed"] == 1
        assert diagnostics["commands_rejected"] == 1
        if provider_state == "render_error":
            assert "unrelated transport failure" in diagnostics["failure"]
        harness.controller.shutdown()
        return
    assert diagnostics["status"] == "stopped"
    assert diagnostics["commands_completed"] == 2
    assert diagnostics["commands_failed"] == 0
    assert diagnostics["commands_rejected"] == 0
    assert diagnostics["failure"] == ""
    assert harness.loop.diagnostics()["failures"] == 0
    assert id(harness.loop) not in harness.controller._presentations
    assert harness.client.render_calls == 1


@pytest.mark.parametrize("stop_requested", [True, False])
@pytest.mark.parametrize("provider_state", ["dead_rna", "cleared"])
def test_controller_loss_after_job_dequeue_settles_every_future(
    tmp_path, monkeypatch, provider_state, stop_requested
):
    harness = _Harness(monkeypatch, tmp_path)
    entered = threading.Event()
    resume = threading.Event()
    released = threading.Event()
    main_thread = threading.get_ident()

    def provider():
        if (harness.loop._job_active and not resume.is_set()
                and threading.get_ident() != main_thread):
            entered.set()
            assert resume.wait(WAIT_S)
        if released.is_set():
            if provider_state == "dead_rna":
                raise ReferenceError("StructRNA has been removed")
            return None
        return harness.controller

    harness.loop._controller_provider = provider
    first = harness.loop.call(lambda: pytest.fail("dequeued job ran after ownership loss"))
    second = harness.loop.call(lambda: pytest.fail("queued job ran after ownership loss"))
    thread = ViewportRenderThread("stop-after-job-dequeue")
    thread.start()
    try:
        thread.submit(harness.loop.run, label="latest-view-loop")
        assert entered.wait(WAIT_S)
        teardown = thread.call(harness.controller.shutdown, label="teardown")
        if stop_requested:
            harness.loop.request_stop()
        released.set()
        resume.set()
        if stop_requested:
            teardown.result(WAIT_S)
        else:
            with pytest.raises(RenderThreadRejectedError):
                teardown.result(WAIT_S)
        assert first.done(), "dequeued job future was lost during controller lookup"
        assert second.done()
        with pytest.raises(RenderThreadRejectedError if stop_requested else RenderClientError):
            first.result(0)
        with pytest.raises(RenderThreadRejectedError):
            second.result(0)
    finally:
        released.clear()
        resume.set()
        harness.loop.request_stop()
        outcome = thread.stop(WAIT_S)
    assert outcome["joined"] is True
    assert outcome["leaked_thread"] is False
    assert not harness.loop._job_active
    assert not harness.loop._pending_jobs
    assert harness.controller._exclusive_pending == 0
    diagnostics = thread.diagnostics()
    assert diagnostics["commands_failed"] == (0 if stop_requested else 1)
    assert diagnostics["commands_completed"] == (2 if stop_requested else 0)
    if not stop_requested:
        harness.controller.shutdown()


@pytest.mark.parametrize("failure_phase", ["admission", "after_result"])
def test_job_transport_failure_settles_only_unresolved_result(
    tmp_path, monkeypatch, failure_phase
):
    harness = _Harness(monkeypatch, tmp_path)
    failure = RenderClientError("exclusive transport failed")
    calls = []

    @contextmanager
    def transaction():
        if failure_phase == "admission":
            raise failure
        yield
        raise failure

    monkeypatch.setattr(harness.controller, "_exclusive_transport", transaction)
    future = harness.loop.call(lambda: calls.append("ran") or 42)
    with pytest.raises(RenderClientError) as error:
        harness.loop._run_pending_jobs()
    assert error.value is failure
    assert future.done()
    if failure_phase == "admission":
        assert future.exception(0) is failure
        assert calls == []
    else:
        assert future.result(0) == 42
        assert calls == ["ran"]
    assert not harness.loop._job_active
    assert not harness.loop._pending_jobs
    harness.controller.shutdown()


def test_loop_follows_engine_controller_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_a = _Client(simulation_id="sim-a")
    client_b = _Client(simulation_id="sim-b")
    active = {"client": client_a}
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: active["client"],
    )
    base_request = _request(tmp_path)
    controller_a = controller_module.OvrtxSessionController()
    controller_a.ensure(base_request)
    active["client"] = client_b
    controller_b = controller_module.OvrtxSessionController()
    controller_b.ensure(base_request)

    holder = SimpleNamespace(controller=controller_a)
    mailbox = _RecordingMailbox()
    slot = _RecordingSlot()
    loop = LatestViewRenderLoop(
        mailbox=mailbox,
        frame_slot=slot,
        controller_provider=lambda: holder.controller,
        scheduler=RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        ),
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            base_request, snapshot
        ),
    )

    with _running(loop):
        mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(lambda: len(slot.frames()) >= 4)
        assert _wait_until(lambda: mailbox.park_count() >= 2)
        a_calls_at_swap = client_a.render_calls
        assert a_calls_at_swap >= 4
        # The engine swaps its controller out from under the running loop
        # (activate_for_viewport reusing a persistent generation runtime).
        holder.controller = controller_b
        mailbox.write(_snapshot(tx=9.0))
        assert _wait_until(lambda: client_b.render_calls >= 1)

    # The loop followed the swap: B rendered the new view; A saw nothing more.
    assert client_b.render_calls >= 1
    assert client_a.render_calls == a_calls_at_swap


def test_ensure_defers_while_runtime_services_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A worker restart respawns the runtime services off-thread, and the session
    # start raises RuntimeServicesPreparingError until they serve. The loop must
    # treat that as a transient loading state -- retry, paced, without a failure
    # publication or spending the failure budget -- not a hard failure.
    harness = _Harness(monkeypatch, tmp_path)
    calls = {"n": 0}

    def _ensure(_request) -> None:
        calls["n"] += 1
        raise RuntimeServicesPreparingError("Runtime services are still preparing")

    lifecycle = SessionLifecycleHooks(
        ensure_session=_ensure,
        replacement_reason=lambda _request: "",
        retry_allowed=lambda: True,
    )
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        lifecycle=lifecycle,
    )

    with _running(loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        # It keeps retrying (paced) rather than giving up after one attempt.
        assert _wait_until(lambda: calls["n"] >= 2)

    # No failure publication, no failure counted, and the ensure stays pending.
    assert all(frame.status != FRAME_STATUS_FAILED for frame in harness.slot.frames())
    diagnostics = loop.diagnostics()
    assert diagnostics["failures"] == 0
    assert diagnostics["ensure_failures"] == 0
    assert diagnostics["ensure_pending"] is True


# --- Worker death without new input (build 29 review) ----------------------


def _worker_lost_error(message: str = "renderer render: connection reset") -> RenderClientError:
    """A lost worker as the loop sees it: the client wraps the session error."""
    from ovrtx_blender_example.direct_runtime_session import RuntimeWorkerLost

    try:
        raise RuntimeWorkerLost(message)
    except RuntimeWorkerLost as cause:
        try:
            raise RenderClientError(f"Public SDK render: {cause}") from cause
        except RenderClientError as error:
            return error


def _failing_renders(client: _Client, failures: list[BaseException]) -> None:
    def hook(_call_index: int) -> None:
        if failures:
            raise failures.pop(0)

    client.render_hook = hook


def test_static_view_retries_once_after_worker_death(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    harness.loop._failure_retry_backoff_seconds = 0.01
    _failing_renders(harness.client, [_worker_lost_error()])

    with _running(harness.loop):
        # One input only: the view never changes again.
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
        assert _wait_until(lambda: any(
            frame.status == FRAME_STATUS_FRAME and frame.completed_samples >= 2
            for frame in harness.slot.frames()))

    frames = harness.slot.frames()
    assert frames[0].status == FRAME_STATUS_FAILED
    assert "connection reset" in frames[0].detail
    assert harness.mailbox.diagnostics()["writes"] == 1
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["worker_failure_retries"] == 1
    assert diagnostics["failed_state"] is False


def test_static_view_worker_retry_is_one_shot_and_paced(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    harness.loop._failure_retry_backoff_seconds = 0.01
    _failing_renders(harness.client, [_worker_lost_error("first"), _worker_lost_error("second"),
                                      _worker_lost_error("third")])

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
        assert _wait_until(lambda: harness.client.render_calls == 2)
        # The retry failed too: the view parks until new input.
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        time.sleep(0.3)
        assert harness.client.render_calls == 2
        failures = [frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FAILED]
        assert len(failures) == 2
        # Fresh input still retries under the existing attempt limit.
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
        assert _wait_until(lambda: harness.client.render_calls == 3)
        assert harness.loop.diagnostics()["worker_failure_retries"] == 1


def test_static_view_does_not_auto_retry_rejections_or_recovery_required(tmp_path, monkeypatch):
    from ovrtx_blender_example.ovrtx_runtime_client import ValueUpdateRecoveryRequired

    lost = _worker_lost_error()
    recovery_required = ValueUpdateRecoveryRequired(str(lost))
    recovery_required.__cause__ = lost
    for failure in (RenderClientError("render failed"), recovery_required):
        harness = _Harness(monkeypatch, tmp_path)
        harness.loop._failure_retry_backoff_seconds = 0.01
        _failing_renders(harness.client, [failure])
        with _running(harness.loop):
            harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
            assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
            time.sleep(0.2)
            assert harness.client.render_calls == 1
        assert harness.loop.diagnostics()["worker_failure_retries"] == 0
    # ValueUpdateRecoveryRequired stays permanent: fresh input does not retry it.
    assert harness.loop._failure_permanent is True
    assert not harness.loop._retry_allowed()


class _WatchedClient(_Client):
    """Fake client whose worker can exit while the view is idle."""

    def __init__(self) -> None:
        super().__init__()
        self.exited = False
        self.watch_calls = 0

    def worker_exited(self) -> bool:
        self.watch_calls += 1
        return self.exited

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        # The session respawns the worker on its next request.
        self.exited = False
        return super().render_result(simulation_id, **kwargs)


def test_idle_worker_exit_rerenders_the_static_view_without_input(tmp_path, monkeypatch):
    from ovrtx_blender_example import viewport_render_thread

    monkeypatch.setattr(viewport_render_thread, "WORKER_EXIT_WATCH_SECONDS", 0.02)
    client = _WatchedClient()
    harness = _Harness(monkeypatch, tmp_path, client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
        assert _wait_until(lambda: harness.client.render_calls == 2)
        # Converged and idle: the loop watches the worker instead of parking forever.
        assert _wait_until(lambda: client.watch_calls >= 3)
        assert harness.client.render_calls == 2
        client.exited = True
        assert _wait_until(lambda: harness.client.render_calls == 4)

    frames = [frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in frames] == [1, 2, 1, 2]
    assert harness.mailbox.diagnostics()["writes"] == 1
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["idle_worker_exits"] == 1
    assert diagnostics["last_reset_reason"] == "worker_exited"


def test_idle_loop_without_worker_watch_still_parks(tmp_path, monkeypatch):
    harness = _Harness(monkeypatch, tmp_path)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=1))
        assert _wait_until(lambda: harness.mailbox.park_count() >= 1)
        assert all(timeout in (None, 0.0) for timeout, _taken in harness.mailbox.records)
