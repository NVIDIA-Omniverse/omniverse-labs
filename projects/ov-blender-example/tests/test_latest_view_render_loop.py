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
    ASYNC_RENDER_READ_ENV,
    LatestViewRenderLoop,
    RenderThreadError,
    SessionLifecycleHooks,
    ViewportRenderThread,
    _async_render_read_enabled,
)


WAIT_S = 5.0


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


class _Client:
    """Fake srtx client compatible with OvrtxSessionController."""

    def __init__(self, simulation_id: str = "sim") -> None:
        self.simulation_id = simulation_id
        self.fail_render = False
        self.starts = 0
        self.deletes = 0
        self.delete_failures_remaining = 0
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
        if self.delete_failures_remaining:
            self.delete_failures_remaining -= 1
            return "failed"
        return "stopped"

    def shutdown(self) -> None:
        self.closed += 1


class _AsyncClient(_Client):
    """Nonblocking exact-time read fake with observable ticket ordering."""

    def __init__(self, simulation_id: str = "sim") -> None:
        super().__init__(simulation_id)
        self.async_events: list[tuple[str, int, float | None]] = []
        self.pending_tickets: dict[int, float | None] = {}
        self.peak_pending_tickets = 0
        self.active_read_tickets: set[int] = set()
        self.peak_active_read_tickets = 0
        self.release_read: threading.Event | None = None
        self.read_started = threading.Event()
        self.blocked_tickets: set[int] = set()
        self.pending_polls = 1
        self.poll_counts: dict[int, int] = {}
        self.submit_failures_remaining = 0
        self.begin_failures_remaining = 0
        self.terminal_poll_failures_remaining = 0
        self.cancel_failures_remaining = 0

    def supports_async_render_read(self) -> bool:
        return True

    def submit_render_sample(self, simulation_id: str, **kwargs: object) -> int:
        if self.submit_failures_remaining:
            self.submit_failures_remaining -= 1
            raise RenderClientError("successor submit rejected_for_test")
        ticket = self.render_calls + 1
        self.render_calls += 1
        self.render_thread_idents.append(threading.get_ident())
        camera_translation = self.camera_translation
        self.render_camera_translations.append(camera_translation)
        self.pending_tickets[ticket] = camera_translation
        self.peak_pending_tickets = max(
            self.peak_pending_tickets, len(self.pending_tickets)
        )
        self.async_events.append(("submit", ticket, camera_translation))
        return ticket

    def begin_render_sample_read(self, ticket: int) -> int:
        camera_translation = self.pending_tickets[ticket]
        if self.begin_failures_remaining:
            self.begin_failures_remaining -= 1
            self.async_events.append(("begin_failed", ticket, camera_translation))
            raise RenderClientError("read begin rejected_for_test")
        if self.active_read_tickets:
            raise AssertionError("more than one native read began concurrently")
        self.active_read_tickets.add(ticket)
        self.peak_active_read_tickets = max(
            self.peak_active_read_tickets,
            len(self.active_read_tickets),
        )
        self.async_events.append(("begin", ticket, camera_translation))
        return ticket

    def poll_render_sample_read(self, ticket: int) -> RenderResult | None:
        self.read_started.set()
        self.poll_counts[ticket] = self.poll_counts.get(ticket, 0) + 1
        if self.terminal_poll_failures_remaining:
            self.terminal_poll_failures_remaining -= 1
            camera_translation = self.pending_tickets.pop(ticket)
            self.active_read_tickets.remove(ticket)
            self.async_events.append(("poll_failed", ticket, camera_translation))
            raise RenderClientError("read poll terminalized_for_test")
        if ticket in self.blocked_tickets:
            return None
        if self.release_read is not None and not self.release_read.is_set():
            return None
        if self.poll_counts[ticket] <= self.pending_polls:
            return None
        camera_translation = self.pending_tickets.pop(ticket)
        self.active_read_tickets.remove(ticket)
        self.async_events.append(("complete", ticket, camera_translation))
        self.last_render_timings = {"native_render_ms": 1.0, "ticket": ticket}
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=ticket,
            session_completed_samples=ticket,
            simulation_time_ns=ticket * 10,
        )

    def cancel_render_sample_read(self, ticket: int) -> None:
        if self.cancel_failures_remaining:
            self.cancel_failures_remaining -= 1
            camera_translation = self.pending_tickets[ticket]
            self.async_events.append(("cancel_failed", ticket, camera_translation))
            raise RenderClientError("read cancel rejected_for_test")
        camera_translation = self.pending_tickets.pop(ticket)
        self.active_read_tickets.remove(ticket)
        self.async_events.append(("cancel", ticket, camera_translation))

    def discard_render_sample(self, ticket: int) -> None:
        camera_translation = self.pending_tickets.pop(ticket)
        self.async_events.append(("discard", ticket, camera_translation))

    def delete_simulation(self, simulation_id: str) -> str:
        self.async_events.append(("delete", 0, self.camera_translation))
        status = super().delete_simulation(simulation_id)
        self.pending_tickets.clear()
        self.active_read_tickets.clear()
        return status


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


def _prime_camera_pipeline_session(
    harness: _Harness,
    snapshot: ViewSnapshot,
) -> None:
    """Match production's routed startup ensure before pipeline assertions."""

    request = viewport_handoff.request_from_snapshot(
        harness.base_request,
        snapshot,
    )
    request = harness.loop._with_camera_value_route(request)
    request = harness.loop._with_rtpt_digest_route(request)
    harness.controller.ensure(request)
    harness.loop._observed_session_revision = harness.controller._session_revision


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


def test_async_read_keeps_one_ticket_and_does_not_queue_same_camera_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=4))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)

    assert client.async_events == [
        ("submit", 1, 2.0),
        ("begin", 1, 2.0),
        ("complete", 1, 2.0),
        ("submit", 2, 2.0),
        ("begin", 2, 2.0),
        ("complete", 2, 2.0),
        ("submit", 3, 2.0),
        ("begin", 3, 2.0),
        ("complete", 3, 2.0),
        ("submit", 4, 2.0),
        ("begin", 4, 2.0),
        ("complete", 4, 2.0),
    ]
    assert client.peak_pending_tickets == 1
    assert [
        frame.completed_samples for frame in harness.slot.frames()[:4]
    ] == [1, 2, 3, 4]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["peak_active_reads"] == 1
    assert client.peak_active_read_tickets == 1
    assert pipeline["async_read_begins"] == 4
    assert pipeline["async_read_completions"] == 4
    assert pipeline["synchronous_acquisitions"] == 0
    assert pipeline["last_mode"] == "async"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_prim_path", "/World/OtherCamera"),
        ("camera_projection", _projection(35.0)),
        ("min_samples", 2),
        ("max_samples", 8),
        ("selected_sensor_paths", ("/Render/Other",)),
        ("render_var", "HdrColor"),
        ("width", 2),
        ("height", 2),
        ("timeline_controls_enabled", True),
        ("timeline_playing", True),
        ("timeline_frame", 2),
        ("timeline_start", 0),
        ("timeline_end", 2),
        ("simulation_reset_token", 1),
    ],
)
def test_camera_successor_rejects_every_non_pose_snapshot_change(
    field: str,
    value: object,
) -> None:
    current = _snapshot(tx=2.0)
    candidate = replace(_snapshot(tx=5.0), **{field: value})

    assert not LatestViewRenderLoop._pure_camera_successor(current, candidate)


def test_newer_mailbox_snapshot_wins_over_preserved_deferred_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    snapshot_b = _snapshot(tx=5.0)
    snapshot_c = _snapshot(tx=9.0)
    harness.loop._deferred_snapshot = snapshot_b
    harness.mailbox.write(snapshot_c)

    assert harness.loop._take_next_snapshot() is snapshot_c
    assert harness.loop._deferred_snapshot is None


def test_camera_successor_failure_is_counted_once_across_nested_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    error = RenderClientError("successor failed_for_test")

    harness.loop._note_camera_successor_failure(error)
    harness.loop._note_camera_successor_failure(error)

    assert harness.loop.diagnostics()["render_pipeline"][
        "camera_successor_failures"
    ] == 1


def test_async_read_pipelines_two_fifo_successors_and_one_deferred_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    # Once the predecessor gate is released, every rendered result is already
    # terminal at its first poll. This pins the producer/consumer boundary:
    # B/C remain FIFO, B's read begins, then D replenishes its freed slot.
    client.pending_polls = 0
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    original_publish = harness.slot.publish

    def _record_ordered_publication(frame: FrameState) -> FrameState:
        if frame.status == FRAME_STATUS_FRAME and frame.snapshot_key == snapshot_a.key:
            client.async_events.append(("publish_a", 0, None))
        return original_publish(frame)

    harness.slot.publish = _record_ordered_publication

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: harness.loop.diagnostics()["render_pipeline"][
                "camera_successor_submissions"
            ]
            >= 1
        )
        # B is rendered while A owns the only native read, but B's read has
        # not begun and A is still nonterminal.
        assert ("submit", 2, 5.0) in client.async_events
        assert not any(
            event[0] == "begin" and event[1] == 2
            for event in client.async_events
        )
        assert not any(
            event[0] == "complete" and event[1] == 1
            for event in client.async_events
        )
        harness.mailbox.write(snapshot_c)
        assert _wait_until(
            lambda: ("submit", 3, 9.0) in client.async_events
        )
        pipeline = harness.loop.diagnostics()["render_pipeline"]
        assert pipeline["queued_camera_successor_count"] == 2
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        # A owns the read; B/C own the two prepared slots; D is data-only.
        assert [event[0] for event in client.async_events].count("submit") == 3
        client.release_read.set()
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)

    assert [frame.snapshot_key for frame in harness.slot.frames()[:4]] == [
        snapshot_a.key,
        snapshot_b.key,
        snapshot_c.key,
        snapshot_d.key,
    ]
    assert [
        event for event in client.async_events if event[0] == "submit"
    ][:4] == [
        ("submit", 1, 2.0),
        ("submit", 2, 5.0),
        ("submit", 3, 9.0),
        ("submit", 4, 12.0),
    ]
    events = client.async_events
    assert events.index(("submit", 2, 5.0)) < events.index(
        ("complete", 1, 2.0)
    )
    assert events.index(("submit", 3, 9.0)) < events.index(
        ("complete", 1, 2.0)
    )
    assert events.index(("complete", 1, 2.0)) < events.index(
        ("begin", 2, 5.0)
    )
    assert events.index(("begin", 2, 5.0)) < events.index(
        ("submit", 4, 12.0)
    )
    assert events.index(("submit", 4, 12.0)) < events.index(
        ("publish_a", 0, None)
    )
    assert events.index(("submit", 4, 12.0)) < events.index(
        ("complete", 2, 5.0)
    )
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["peak_active_reads"] == 1
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["peak_queued_camera_successors"] == 2
    assert pipeline["peak_deferred_snapshots"] == 1
    assert client.peak_pending_tickets == 3


def test_two_successor_capacity_retains_only_latest_deferred_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    snapshot_e = _snapshot(tx=15.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        harness.mailbox.write(snapshot_e)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_e
        )
        assert [event[0] for event in client.async_events].count("submit") == 3
        client.release_read.set()
        assert _wait_until(lambda: len(harness.slot.frames()) >= 4)

    assert [frame.snapshot_key for frame in harness.slot.frames()[:4]] == [
        snapshot_a.key,
        snapshot_b.key,
        snapshot_c.key,
        snapshot_e.key,
    ]
    submitted_cameras = [
        event[2] for event in client.async_events if event[0] == "submit"
    ]
    assert submitted_cameras[:4] == [2.0, 5.0, 9.0, 15.0]
    assert 12.0 not in submitted_cameras
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["peak_queued_camera_successors"] == 2
    assert pipeline["peak_deferred_snapshots"] == 1
    assert pipeline["deferred_snapshot_replacements"] >= 1


def test_prefetched_head_keeps_two_tail_slots_and_latest_post_prefetch_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshots = [
        _snapshot(tx=translation, max_samples=1)
        for translation in (2.0, 5.0, 9.0, 12.0, 15.0, 18.0)
    ]
    snapshot_a, snapshot_b, snapshot_c, snapshot_d, snapshot_e, snapshot_f = (
        snapshots
    )
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)

        harness.mailbox.write(snapshot_e)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_e)
        harness.mailbox.write(snapshot_f)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_f)
        pipeline = harness.loop.diagnostics()["render_pipeline"]
        assert pipeline["prefetched_camera_successor"] is True
        assert pipeline["queued_camera_successor_count"] == 2
        assert len(client.pending_tickets) == 3

        client.blocked_tickets.remove(2)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 5)

    assert [frame.snapshot_key for frame in harness.slot.frames()[:5]] == [
        snapshot_a.key,
        snapshot_b.key,
        snapshot_c.key,
        snapshot_d.key,
        snapshot_f.key,
    ]
    submitted_cameras = [
        event[2] for event in client.async_events if event[0] == "submit"
    ]
    assert submitted_cameras[:5] == [2.0, 5.0, 9.0, 12.0, 18.0]
    assert 15.0 not in submitted_cameras
    assert client.peak_pending_tickets == 3
    assert client.peak_active_read_tickets == 1
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["peak_queued_camera_successors"] == 2
    assert pipeline["deferred_snapshot_replacements"] >= 1


def test_dynamic_second_pane_waits_for_prefetched_read_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(controller_module.RPC_THREAD_GUARD_ENV, "1")
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    snapshot_other = _snapshot(tx=21.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
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

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)
        assert _wait_until(
            lambda: harness.controller._prefetched_read_reservation
            == id(harness.loop)
        )
        deletes_before_attach = client.deletes

        with _running(second_loop):
            assert _wait_until(harness.controller._has_shared_presentations)
            assert harness.controller.diagnostics()["rpc_thread"] == {
                "owning_thread_ident": 0,
                "adopted": False,
                "guard_active": False,
            }
            second_mailbox.write(snapshot_other)
            assert _wait_until(
                lambda: len(harness.controller._transport_waiters) >= 1
            )
            time.sleep(0.02)
            assert second_slot.frames() == []
            assert client.render_additional_samples == []
            assert not any(
                float(value.matrix[0][3]) == 21.0
                for batch in client.transform_update_batches
                for value in batch
                if value.prim_path == "/World/Camera"
            )

            client.blocked_tickets.remove(2)
            assert _wait_until(lambda: len(second_slot.frames()) >= 1)
            assert _wait_until(
                lambda: harness.controller._prefetched_read_reservation
                is controller_module._PRESENTATION_UNSET
            )

    assert client.deletes == deletes_before_attach
    assert not client.active_read_tickets
    assert not any(
        frame.status == FRAME_STATUS_FAILED
        for frame in harness.slot.frames() + second_slot.frames()
    )
    assert harness.controller.would_replace(harness.base_request) != (
        "render_operation_failed"
    )


def test_prefetch_begin_failure_publishes_predecessor_then_drains_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: ("submit", 2, 5.0) in client.async_events
        )
        harness.mailbox.write(snapshot_c)
        assert _wait_until(
            lambda: ("submit", 3, 9.0) in client.async_events
        )
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        client.begin_failures_remaining = 1
        client.release_read.set()
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        assert _wait_until(
            lambda: harness.loop._snapshot_key == snapshot_d.key
        )

    events = client.async_events
    assert events.index(("complete", 1, 2.0)) < events.index(
        ("begin_failed", 2, 5.0)
    )
    assert ("begin", 2, 5.0) not in events
    assert ("discard", 2, 5.0) in events
    assert ("discard", 3, 9.0) in events
    assert not any(event[0] == "submit" and event[2] == 12.0 for event in events)
    assert not any(event[0] == "discard" and event[2] == 12.0 for event in events)
    assert not client.pending_tickets
    assert [frame.snapshot_key for frame in harness.slot.frames()[:2]] == [
        snapshot_a.key,
        snapshot_b.key,
    ]
    assert [frame.status for frame in harness.slot.frames()[:2]] == [
        FRAME_STATUS_FRAME,
        FRAME_STATUS_FAILED,
    ]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["queued_camera_successor"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["camera_successor_discards"] == 1
    assert pipeline["camera_successor_failures"] == 1
    assert pipeline["retirement_recovery_unavailable"] is True


def test_revision_change_discards_entire_successor_fifo_and_keeps_newest_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: ("submit", 2, 5.0) in client.async_events
        )
        harness.mailbox.write(snapshot_c)
        assert _wait_until(
            lambda: ("submit", 3, 9.0) in client.async_events
        )
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        harness.scheduler.note_applied_content()
        client.release_read.set()
        assert _wait_until(
            lambda: [event[0] for event in client.async_events].count("discard")
            >= 2
        )
        assert _wait_until(lambda: len(harness.slot.frames()) >= 2)

    assert ("discard", 2, 5.0) in client.async_events
    assert ("discard", 3, 9.0) in client.async_events
    assert ("begin", 2, 5.0) not in client.async_events
    assert ("begin", 3, 9.0) not in client.async_events
    assert [frame.snapshot_key for frame in harness.slot.frames()[:2]] == [
        snapshot_a.key,
        snapshot_d.key,
    ]
    assert [frame.applied_revision for frame in harness.slot.frames()[:2]] == [
        0,
        1,
    ]
    assert [
        event[2]
        for event in client.async_events
        if event[0] == "submit"
    ][:4] == [2.0, 5.0, 9.0, 12.0]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["camera_successor_discards"] == 2


def test_terminal_poll_failure_discards_fifo_and_preserves_newest_deferred_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        client.terminal_poll_failures_remaining = 1
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        assert _wait_until(
            lambda: harness.loop._snapshot_key == snapshot_d.key
        )

    assert ("poll_failed", 1, 2.0) in client.async_events
    assert ("discard", 2, 5.0) in client.async_events
    assert ("discard", 3, 9.0) in client.async_events
    assert not any(
        event[0] == "submit" and event[2] == 12.0
        for event in client.async_events
    )
    assert not client.pending_tickets
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["camera_successor_discards"] == 2
    assert pipeline["retirement_recovery_unavailable"] is True


def test_successor_submit_failure_is_fatal_and_never_retried_as_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        client.submit_failures_remaining = 1
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        assert _wait_until(
            lambda: harness.loop._snapshot_key == snapshot_b.key
        )

    assert client.camera_translation == 5.0
    assert [event for event in client.async_events if event[0] == "submit"] == [
        ("submit", 1, 2.0)
    ]
    assert ("cancel", 1, 2.0) in client.async_events
    assert ("delete", 0, 5.0) in client.async_events
    assert client.async_events.index(("cancel", 1, 2.0)) < client.async_events.index(
        ("delete", 0, 5.0)
    )
    # The first delete is the test's routed startup replacement; the second
    # retires the ambiguous camera-successor session.
    assert client.deletes == 2
    assert client.closed == 1
    assert harness.controller.would_replace(harness.base_request) == "no_active_session"
    failed_frames = [
        frame
        for frame in harness.slot.frames()
        if frame.status == FRAME_STATUS_FAILED
    ]
    assert len(failed_frames) == 1
    assert failed_frames[0].snapshot_key == snapshot_a.key
    assert "OvrtxSessionRetirementRequiredError" in failed_frames[0].detail
    assert "No active OVRTX session" not in failed_frames[0].detail
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["camera_successor_failures"] == 1
    assert pipeline["camera_successor_submissions"] == 0
    assert pipeline["retirement_recovery_unavailable"] is True


def test_ambiguous_successor_failure_recreates_session_and_renders_same_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    harness.loop = LatestViewRenderLoop(
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
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    starts_before_failure = client.starts

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        client.submit_failures_remaining = 1
        client.delete_failures_remaining = 1
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        client.release_read.set()
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FRAME
                and frame.snapshot_key == snapshot_b.key
                for frame in harness.slot.frames()
            )
        ), harness.loop.diagnostics()["last_failure_detail"]

    assert client.starts == starts_before_failure + 1
    assert any(
        event[0] == "submit" and event[2] == 5.0
        for event in client.async_events
    )
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_terminal_poll_failure_retries_retirement_without_new_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    harness.loop = LatestViewRenderLoop(
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
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    starts_before_failure = client.starts
    client.terminal_poll_failures_remaining = 1
    client.delete_failures_remaining = 1

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        assert harness.loop.diagnostics()["render_pipeline"]["active_read"] is False
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FRAME
                and frame.snapshot_key == snapshot_a.key
                for frame in harness.slot.frames()
            )
        ), harness.loop.diagnostics()["last_failure_detail"]

    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["async_read_cancel_failures"] >= 2
    assert pipeline["active_read"] is False
    assert client.starts == starts_before_failure + 1
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_retirement_error_retires_origin_not_current_controller_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path, client=_AsyncClient("origin"))
    origin = harness.controller
    replacement_client = _AsyncClient("replacement")
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: replacement_client,
    )
    replacement = controller_module.OvrtxSessionController()
    replacement.ensure(harness.base_request)
    assert replacement._session_revision == origin._session_revision
    loop = LatestViewRenderLoop(
        mailbox=harness.mailbox,
        frame_slot=harness.slot,
        controller_provider=lambda: replacement,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
    )
    error = origin._render_retirement_error(
        "render read poll",
        RenderClientError("failed_for_test"),
    )

    loop._publish_failure(error, {})

    assert harness.client.deletes == 1
    assert replacement_client.deletes == 0
    assert origin.would_replace(harness.base_request) == "no_active_session"
    assert replacement.would_replace(harness.base_request) == ""
    assert harness.slot.frames()[-1].status == FRAME_STATUS_FAILED


def test_stop_cancels_active_read_and_discards_entire_prepared_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(
            lambda: ("submit", 2, 5.0) in client.async_events
        )
        harness.mailbox.write(snapshot_c)
        assert _wait_until(
            lambda: ("submit", 3, 9.0) in client.async_events
        )
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )

    assert ("cancel", 1, 2.0) in client.async_events
    assert ("discard", 2, 5.0) in client.async_events
    assert ("discard", 3, 9.0) in client.async_events
    assert ("begin", 2, 5.0) not in client.async_events
    assert not any(
        event[0] == "submit" and event[2] == 12.0
        for event in client.async_events
    )
    assert not client.pending_tickets
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["camera_successor_discards"] == 2
    assert pipeline["queued_camera_successor"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["peak_queued_camera_successors"] == 2


def test_stop_cancels_prefetched_head_and_discards_each_unstarted_tail_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        pipeline = harness.loop.diagnostics()["render_pipeline"]
        assert pipeline["prefetched_camera_successor"] is True
        assert pipeline["queued_camera_successor_count"] == 2

    assert client.async_events.count(("cancel", 2, 5.0)) == 1
    assert client.async_events.count(("discard", 3, 9.0)) == 1
    assert client.async_events.count(("discard", 4, 12.0)) == 1
    assert ("begin", 3, 9.0) not in client.async_events
    assert not client.pending_tickets
    assert [frame.snapshot_key for frame in harness.slot.frames()] == [
        snapshot_a.key
    ]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["prefetched_camera_successor"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["async_read_cancels"] == 1
    assert pipeline["camera_successor_discards"] == 2
    assert pipeline["peak_active_reads"] == 1


def test_live_prefetched_cancel_failure_retires_and_releases_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    deletes_before_failure = client.deletes

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)
        client.cancel_failures_remaining = 10

    assert client.async_events.count(("cancel_failed", 2, 5.0)) >= 3
    assert ("cancel", 2, 5.0) not in client.async_events
    assert client.deletes == deletes_before_failure + 1
    assert not client.pending_tickets
    assert not client.active_read_tickets
    assert harness.controller._prefetched_read_reservation is (
        controller_module._PRESENTATION_UNSET
    )
    assert harness.controller.would_replace(harness.base_request) == (
        "no_active_session"
    )
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["prefetched_camera_successor"] is False
    assert pipeline["async_read_cancel_failures"] >= 3


def test_prefetched_poll_failure_discards_each_unstarted_tail_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        client.terminal_poll_failures_remaining = 1
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )

    assert client.async_events.count(("poll_failed", 2, 5.0)) == 1
    assert client.async_events.count(("discard", 3, 9.0)) == 1
    assert client.async_events.count(("discard", 4, 12.0)) == 1
    assert ("begin", 3, 9.0) not in client.async_events
    assert not client.pending_tickets
    assert [frame.snapshot_key for frame in harness.slot.frames()[:2]] == [
        snapshot_a.key,
        snapshot_b.key,
    ]
    assert [frame.status for frame in harness.slot.frames()[:2]] == [
        FRAME_STATUS_FRAME,
        FRAME_STATUS_FAILED,
    ]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["prefetched_camera_successor"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["camera_successor_discards"] == 2
    assert pipeline["camera_successor_failures"] == 1
    assert pipeline["peak_active_reads"] == 1


def test_prefetch_replenish_failure_cancels_head_and_preserves_latest_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.submit_failures_remaining = 1
        client.release_read.set()
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FAILED
                for frame in harness.slot.frames()
            )
        )
        assert _wait_until(lambda: harness.loop._snapshot_key == snapshot_d.key)

    events = client.async_events
    assert events.index(("complete", 1, 2.0)) < events.index(("begin", 2, 5.0))
    assert events.index(("begin", 2, 5.0)) < events.index(("cancel", 2, 5.0))
    assert events.count(("cancel", 2, 5.0)) == 1
    assert events.count(("discard", 3, 9.0)) == 1
    assert not any(event[0] == "submit" and event[2] == 12.0 for event in events)
    assert not client.pending_tickets
    assert [frame.snapshot_key for frame in harness.slot.frames()[:2]] == [
        snapshot_a.key,
        snapshot_b.key,
    ]
    assert [frame.status for frame in harness.slot.frames()[:2]] == [
        FRAME_STATUS_FRAME,
        FRAME_STATUS_FAILED,
    ]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["prefetched_camera_successor"] is False
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["async_read_cancels"] == 1
    assert pipeline["camera_successor_discards"] == 1
    assert pipeline["camera_successor_failures"] == 1


def test_exclusive_job_cancels_prefetched_head_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    deletes_before_job = client.deletes

    def _job() -> tuple[int, ...]:
        pending = tuple(sorted(client.pending_tickets))
        client.async_events.append(("prefetched_job", 0, None))
        return pending

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)

        future = harness.loop.call(_job, label="prefetched-cleanup")
        assert future.result(WAIT_S) == ()
        assert _wait_until(
            lambda: any(
                frame.snapshot_key == snapshot_d.key
                and frame.status == FRAME_STATUS_FRAME
                for frame in harness.slot.frames()
            )
        )

    events = client.async_events
    assert events.index(("cancel", 2, 5.0)) < events.index(("discard", 3, 9.0))
    assert events.index(("discard", 3, 9.0)) < events.index(
        ("prefetched_job", 0, None)
    )
    assert events.index(("discard", 4, 12.0)) < events.index(
        ("prefetched_job", 0, None)
    )
    assert events.index(("prefetched_job", 0, None)) < events.index(
        ("submit", 5, 12.0)
    )
    assert client.deletes == deletes_before_job
    assert harness.controller._prefetched_read_reservation is (
        controller_module._PRESENTATION_UNSET
    )
    assert not client.pending_tickets


def test_controller_wide_exclusive_cancels_blocked_prefetched_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)
    deletes_before_exclusive = client.deletes
    exclusive_entered = threading.Event()
    release_exclusive = threading.Event()

    def _foreign_exclusive() -> None:
        harness.controller._request_exclusive_transport()
        try:
            with harness.controller._exclusive_transport():
                client.async_events.append(("foreign_exclusive", 0, None))
                exclusive_entered.set()
                release_exclusive.wait(WAIT_S)
        finally:
            harness.controller._release_exclusive_transport()

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)

        exclusive = threading.Thread(
            target=_foreign_exclusive,
            name="foreign-controller-exclusive",
            daemon=True,
        )
        exclusive.start()
        try:
            assert exclusive_entered.wait(WAIT_S)
            assert 2 in client.blocked_tickets
            assert not client.active_read_tickets
            assert harness.controller._prefetched_read_reservation is (
                controller_module._PRESENTATION_UNSET
            )
        finally:
            release_exclusive.set()
            exclusive.join(WAIT_S)
        assert not exclusive.is_alive()
        assert _wait_until(
            lambda: not harness.controller._has_pending_exclusive_transport()
        )

    events = client.async_events
    assert events.index(("cancel", 2, 5.0)) < events.index(
        ("foreign_exclusive", 0, None)
    )
    assert client.deletes == deletes_before_exclusive
    assert harness.controller.would_replace(harness.base_request) != (
        "render_operation_failed"
    )


def test_revision_change_while_prefetched_discards_tail_and_renders_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    client.blocked_tickets.add(2)
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(lambda: harness.loop._deferred_snapshot is snapshot_d)
        client.release_read.set()
        assert _wait_until(lambda: ("begin", 2, 5.0) in client.async_events)
        assert _wait_until(lambda: ("submit", 4, 12.0) in client.async_events)

        harness.scheduler.note_applied_content()
        client.blocked_tickets.remove(2)
        assert _wait_until(lambda: len(harness.slot.frames()) >= 3)

    assert [frame.snapshot_key for frame in harness.slot.frames()[:3]] == [
        snapshot_a.key,
        snapshot_b.key,
        snapshot_d.key,
    ]
    assert [frame.applied_revision for frame in harness.slot.frames()[:3]] == [
        0,
        0,
        1,
    ]
    events = client.async_events
    assert events.count(("discard", 3, 9.0)) == 1
    assert events.count(("discard", 4, 12.0)) == 1
    assert ("begin", 3, 9.0) not in events
    assert ("begin", 4, 12.0) not in events
    assert ("submit", 5, 12.0) in events
    assert harness.controller._prefetched_read_reservation is (
        controller_module._PRESENTATION_UNSET
    )
    assert harness.controller.would_replace(harness.base_request) != (
        "render_operation_failed"
    )


def test_exclusive_job_discards_prepared_fifo_before_running_and_resumes_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    client.pending_polls = 0
    harness = _Harness(monkeypatch, tmp_path, client=client)
    snapshot_a = _snapshot(tx=2.0, max_samples=1)
    snapshot_b = _snapshot(tx=5.0, max_samples=1)
    snapshot_c = _snapshot(tx=9.0, max_samples=1)
    snapshot_d = _snapshot(tx=12.0, max_samples=1)
    _prime_camera_pipeline_session(harness, snapshot_a)

    def job() -> tuple[int, ...]:
        pending = tuple(sorted(client.pending_tickets))
        client.async_events.append(("job", 0, None))
        client.release_read.set()
        return pending

    with _running(harness.loop):
        harness.mailbox.write(snapshot_a)
        assert client.read_started.wait(WAIT_S)
        harness.mailbox.write(snapshot_b)
        assert _wait_until(lambda: ("submit", 2, 5.0) in client.async_events)
        harness.mailbox.write(snapshot_c)
        assert _wait_until(lambda: ("submit", 3, 9.0) in client.async_events)
        harness.mailbox.write(snapshot_d)
        assert _wait_until(
            lambda: harness.loop._deferred_snapshot is snapshot_d
        )
        future = harness.loop.call(job, label="pipeline-cleanup")
        assert future.result(WAIT_S) == ()
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)

    events = client.async_events
    assert events.index(("cancel", 1, 2.0)) < events.index(("job", 0, None))
    assert events.index(("discard", 2, 5.0)) < events.index(("job", 0, None))
    assert events.index(("discard", 3, 9.0)) < events.index(("job", 0, None))
    assert events.index(("job", 0, None)) < events.index(("submit", 4, 12.0))
    assert [frame.snapshot_key for frame in harness.slot.frames()[:1]] == [
        snapshot_d.key
    ]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["queued_camera_successor_count"] == 0
    assert pipeline["camera_successor_discards"] == 2
    assert harness.loop.diagnostics()["exclusive_jobs"] == 1


def test_async_environment_switch_uses_synchronous_render_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ASYNC_RENDER_READ_ENV, "0")
    client = _AsyncClient()
    harness = _Harness(monkeypatch, tmp_path, client=client)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=2))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 2)

    assert client.async_events == []
    assert client.render_additional_samples == [1, 1]
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["async_enabled"] is False
    assert pipeline["synchronous_acquisitions"] == 2
    assert pipeline["last_mode"] == "synchronous"


def test_async_read_falls_back_while_controller_is_shared_by_two_panes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    harness = _Harness(monkeypatch, tmp_path, client=client)
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

    with _running(harness.loop), _running(second_loop):
        assert _wait_until(harness.controller._has_shared_presentations)
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=1))
        second_mailbox.write(_snapshot(tx=9.0, max_samples=1))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        assert _wait_until(lambda: len(second_slot.frames()) >= 1)

    assert client.async_events == []
    assert client.render_additional_samples == [1, 1]
    assert harness.loop.diagnostics()["render_pipeline"]["last_mode"] == (
        "synchronous"
    )
    assert second_loop.diagnostics()["render_pipeline"]["last_mode"] == (
        "synchronous"
    )


def test_async_render_read_environment_parser() -> None:
    assert _async_render_read_enabled({}) is True
    assert _async_render_read_enabled({ASYNC_RENDER_READ_ENV: "invalid"}) is True
    assert _async_render_read_enabled({ASYNC_RENDER_READ_ENV: "false"}) is False
    assert _async_render_read_enabled({ASYNC_RENDER_READ_ENV: "yes"}) is True


def test_stop_cancels_active_async_read_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _AsyncClient()
    client.release_read = threading.Event()
    harness = _Harness(monkeypatch, tmp_path, client=client)
    thread = threading.Thread(
        target=harness.loop.run,
        name="latest-view-async-stop-test",
        daemon=True,
    )
    thread.start()
    harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
    assert client.read_started.wait(WAIT_S)
    harness.loop.request_stop()
    thread.join(WAIT_S)

    assert not thread.is_alive()
    assert client.async_events == [
        ("submit", 1, 2.0),
        ("begin", 1, 2.0),
        ("cancel", 1, 2.0),
    ]
    assert harness.slot.frames() == []
    pipeline = harness.loop.diagnostics()["render_pipeline"]
    assert pipeline["active_read"] is False
    assert pipeline["async_read_cancels"] == 1


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
        assert _wait_until(lambda: harness.client.starts == shared_starts + 1)
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


def test_render_failure_retires_once_then_waits_for_fresh_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.client.fail_render = True
    harness.loop = LatestViewRenderLoop(
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

    with _running(harness.loop) as thread:
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(
            lambda: len(
                [
                    frame
                    for frame in harness.slot.frames()
                    if frame.status == FRAME_STATUS_FAILED
                ]
            )
            >= 2
        )
        failure = harness.slot.frames()[0]
        assert failure.status == FRAME_STATUS_FAILED
        assert "OvrtxSessionRetirementRequiredError" in failure.detail
        assert "render failed" in failure.detail
        assert thread.is_alive()
        # One automatic replacement is allowed. A persistent failure then
        # parks instead of restarting the session in a busy loop.
        assert _wait_until(lambda: harness.mailbox.park_count() >= 2)
        renders = harness.client.render_calls
        time.sleep(0.1)
        assert harness.client.render_calls == renders
        assert harness.loop.diagnostics()["render_pipeline"][
            "retirement_recovery_wait_for_input"
        ] is True
        # Failure-driven redraw feedback produces an identical snapshot; it
        # must not reset the bounded recovery budget.
        snapshots_before_duplicate = harness.loop.diagnostics()["snapshots_taken"]
        harness.mailbox.write(_snapshot(tx=2.0, max_samples=0))
        assert _wait_until(
            lambda: harness.loop.diagnostics()["snapshots_taken"]
            > snapshots_before_duplicate
        )
        time.sleep(0.1)
        assert harness.client.render_calls == renders
        # A semantically changed view is genuine fresh input and may retry.
        harness.client.fail_render = False
        harness.mailbox.write(_snapshot(tx=3.0, max_samples=0))
        assert _wait_until(
            lambda: len(
                [
                    frame
                    for frame in harness.slot.frames()
                    if frame.status == FRAME_STATUS_FRAME
                ]
            )
            >= 4
        )

    frames = harness.slot.frames()
    rendered = [frame for frame in frames if frame.status == FRAME_STATUS_FRAME]
    assert [frame.completed_samples for frame in rendered[:4]] == [1, 2, 3, 4]
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["failures"] == 2
    assert diagnostics["failed_state"] is False
    assert diagnostics["render_pipeline"][
        "retirement_recovery_wait_for_input"
    ] is False


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


def test_tick_result_hands_off_before_render_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The tick result is handed off before controller.render runs, so physics
    # poses mirror even when the render step fails.
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
