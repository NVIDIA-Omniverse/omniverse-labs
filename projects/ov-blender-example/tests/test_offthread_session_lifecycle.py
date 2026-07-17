# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Off-thread session lifecycle and resyncing presentation (task02-06).

The latest-view render loop owns session ensure/replace on the render
thread, triggered by mailbox/generation state: startup ensure on the first
adopted snapshot, replacement triggers evaluated per snapshot (authored
generation change, ``reuse_decision`` blockers), resize replacement
debounced by two consecutive same-size snapshots, resyncing presentation
(the last published frame stays presented until the new session's first
frame publishes), and ensure-failure retries gated by the existing
``session_lifecycle.should_auto_retry`` policy with paced (never busy)
retries while pending view updates keep requesting work.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session_controller as controller_module
from ovrtx_blender_example import viewport_handoff
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from ovrtx_blender_example.render_requests import RenderRequest
from ovrtx_blender_example.interactive_edit_planner import EditStatus
from ovrtx_blender_example.runtime_scheduler import (
    EditSubmissionResult,
    RuntimeScheduler,
    RuntimeTickResult,
    RuntimeTickStatus,
)
from ovrtx_blender_example.viewport_handoff import (
    FRAME_STATUS_FAILED,
    FRAME_STATUS_FRAME,
    FRAME_STATUS_RESYNCING,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (
    LatestViewRenderLoop,
    SessionLifecycleHooks,
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


class _Client:
    """Fake srtx client compatible with OvrtxSessionController."""

    def __init__(self, simulation_id: str = "sim") -> None:
        self.simulation_id = simulation_id
        self.fail_render = False
        self.starts = 0
        self.deletes = 0
        self.closed = 0
        self.render_calls = 0
        self.transform_update_batches: list[tuple] = []
        self.start_hook = None
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.starts += 1
        hook = self.start_hook
        if hook is not None:
            hook(self.starts)
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
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
        batch = tuple(values)
        self.transform_update_batches.append(batch)
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        self.deletes += 1
        return "stopped"

    def shutdown(self) -> None:
        self.closed += 1


class _RecordingMailbox(CameraRequestMailbox):
    """Records each take's timeout (at entry) and whether it took a snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[list] = []

    def take(self, timeout: float | None = None) -> ViewSnapshot | None:
        entry = [timeout, False]
        self.records.append(entry)
        snapshot = super().take(timeout)
        entry[1] = snapshot is not None
        return snapshot


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

    def statuses(self) -> list[str]:
        return [frame.status for frame in self.frames()]


class _Harness:
    """Real controller + real scheduler (physics disabled) + fake client.

    Unlike the task02-03 harness, the controller is NOT pre-ensured: the
    loop's lifecycle hooks own the startup ensure (task02-06).
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        client: _Client | None = None,
        ensure_session=None,
        replacement_reason=None,
        retry_allowed=None,
        scheduler=None,
        failure_retry_backoff_seconds: float | None = None,
    ) -> None:
        self.client = client or _Client()
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = _request(tmp_path)
        self.scheduler = scheduler or RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = _RecordingMailbox()
        self.slot = _RecordingSlot()
        self.ensure_calls: list[RenderRequest] = []

        def _default_ensure(request: RenderRequest) -> None:
            self.controller.ensure(request)

        # Reassignable after construction (tests that force a replacement
        # despite a reusable spec swap in a deactivate-then-ensure).
        self.ensure_impl = ensure_session or _default_ensure

        def _counting_ensure(request: RenderRequest) -> None:
            self.ensure_calls.append(request)
            self.ensure_impl(request)

        self.lifecycle = SessionLifecycleHooks(
            ensure_session=_counting_ensure,
            replacement_reason=replacement_reason
            or (lambda request: self.controller.would_replace(request)),
            retry_allowed=retry_allowed or (lambda: True),
        )
        kwargs = {}
        if failure_retry_backoff_seconds is not None:
            kwargs["failure_retry_backoff_seconds"] = failure_retry_backoff_seconds
        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
            lifecycle=self.lifecycle,
            **kwargs,
        )

    def wait_refined(self, timeout: float = WAIT_S) -> None:
        assert _wait_until(
            lambda: any(
                frame.status == FRAME_STATUS_FRAME
                and frame.completed_samples >= self.base_request.max_samples
                for frame in self.slot.frames()
            ),
            timeout,
        ), f"never refined; publications: {self.slot.statuses()}"


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="lifecycle-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


def test_startup_ensure_is_mailbox_triggered_and_precedes_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    with _running(harness.loop):
        # Parked with no snapshot: no session work happens.
        time.sleep(0.05)
        assert harness.client.starts == 0
        assert harness.ensure_calls == []

        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()

    # Exactly one startup ensure, before any render call, on the loop.
    assert harness.client.starts == 1
    assert len(harness.ensure_calls) == 1
    assert harness.loop.diagnostics()["session_ensures"] == 1
    assert harness.loop.diagnostics()["session_replacements"] == 0
    # Startup publishes no resyncing state (nothing to keep presenting);
    # the first publication is the session's first frame.
    assert harness.slot.statuses()[0] == FRAME_STATUS_FRAME


def test_slow_replacement_keeps_last_frame_available_and_never_blocks_writers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replace_after: list[str] = []
    harness = _Harness(
        monkeypatch,
        tmp_path,
        replacement_reason=lambda request: replace_after[0] if replace_after else "",
    )
    ensure_gate = threading.Event()
    ensure_entered = threading.Event()

    def _start_hook(start_index: int) -> None:
        if start_index >= 2:
            ensure_entered.set()
            assert ensure_gate.wait(WAIT_S)

    harness.client.start_hook = _start_hook

    def _replacing_ensure(request: RenderRequest) -> None:
        # The faked replacement reason is not one controller.ensure would
        # derive itself, so mirror break-before-make explicitly.
        if harness.client.starts:
            harness.controller.deactivate()
        harness.controller.ensure(request)

    harness.ensure_impl = _replacing_ensure
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()
        first_frame = harness.slot.peek_latest()
        assert first_frame.status == FRAME_STATUS_FRAME

        # Trigger a replacement whose ensure blocks on the worker.
        replace_after.append("scene_composition_changed")
        harness.mailbox.write(_snapshot(3.0))
        assert ensure_entered.wait(WAIT_S)

        # While the thread is inside the slow ensure: the resyncing state
        # is published (presentation keeps the last frame), and mailbox
        # writes return immediately — the main thread never blocks.
        latest = harness.slot.peek_latest()
        assert latest.status == FRAME_STATUS_RESYNCING
        assert latest.detail == "Re-syncing scene"
        started = time.perf_counter()
        harness.mailbox.write(_snapshot(4.0))
        write_seconds = time.perf_counter() - started
        assert write_seconds < 0.5
        # The last completed frame is still what a presenter would draw.
        previous_frames = [
            frame for frame in harness.slot.frames() if frame.status == FRAME_STATUS_FRAME
        ]
        assert previous_frames[-1].publication_index == first_frame.publication_index

        # Release the ensure: the replacement session's first frame is the
        # next FRAME publication (presentation gating), starting from
        # min_samples again.
        replace_after.clear()
        ensure_gate.set()
        assert _wait_until(
            lambda: harness.slot.latest_index() > latest.publication_index
            and harness.slot.peek_latest().status == FRAME_STATUS_FRAME
        )
        statuses = harness.slot.statuses()
        resync_at = statuses.index(FRAME_STATUS_RESYNCING)
        first_new_frame = next(
            frame
            for frame in harness.slot.frames()[resync_at + 1 :]
            if frame.status == FRAME_STATUS_FRAME
        )
        assert first_new_frame.completed_samples == harness.base_request.min_samples
    assert harness.client.starts == 2
    assert harness.loop.diagnostics()["session_replacements"] == 1
    assert harness.loop.diagnostics()["last_resync_reason"] == "scene_composition_changed"


def test_replacement_reapplies_camera_pose_to_the_new_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replace_once: list[str] = []
    harness = _Harness(
        monkeypatch,
        tmp_path,
        replacement_reason=lambda request: replace_once.pop() if replace_once else "",
    )

    def _replacing_ensure(request: RenderRequest) -> None:
        if harness.client.starts:
            harness.controller.deactivate()
        harness.controller.ensure(request)

    harness.ensure_impl = _replacing_ensure
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()
        batches_before = len(harness.client.transform_update_batches)
        assert batches_before >= 1  # the live pose reached session one

        replace_once.append("generation_changed")
        # Same view identity: the replacement, not a view change, is what
        # must re-apply the pose to the new session.
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(lambda: harness.client.starts == 2)
        assert _wait_until(
            lambda: len(harness.client.transform_update_batches) > batches_before
        )
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_resync_state_recovers_even_when_the_ensure_reuses_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A replacement whose ensure reuses the session must not wedge RESYNCING.

    A generation change can materialize to an identical composition digest,
    making ``controller.ensure`` reuse the session with no refinement
    reset: no rendered step would follow the RESYNCING publication. The
    loop re-publishes the session's still-valid current result so
    presentation recovers.
    """

    replace_once: list[str] = []
    harness = _Harness(
        monkeypatch,
        tmp_path,
        replacement_reason=lambda request: replace_once.pop() if replace_once else "",
    )
    # Default ensure: controller.ensure derives reuse (unchanged spec).
    # Camera-less snapshots keep the post-ensure iteration render-free
    # (no live pose edit → no refinement reset), exercising the recovery
    # re-publication rather than an ordinary rendered step.
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, camera_prim_path="", camera_matrix=None))
        harness.wait_refined()

        replace_once.append("generation_changed")
        harness.mailbox.write(_snapshot(2.0, camera_prim_path="", camera_matrix=None))
        assert _wait_until(
            lambda: harness.slot.statuses().count(FRAME_STATUS_RESYNCING) == 1
        )
        # The session was reused (no second start) yet a FRAME publication
        # follows the resync state, so the status never wedges.
        assert _wait_until(
            lambda: harness.slot.statuses()
            and harness.slot.statuses()[-1] == FRAME_STATUS_FRAME
        )
        assert harness.client.starts == 1
        recovery_frame = harness.slot.peek_latest()
        assert recovery_frame.render_result is not None
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_resize_replacement_waits_for_two_consecutive_same_size_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)  # real would_replace probe
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()
        assert harness.client.starts == 1

        # First snapshot at the new size: debounced, no replacement yet
        # (the session keeps rendering; the draw path scales the old frame).
        harness.mailbox.write(_snapshot(2.0, width=4, height=4))
        assert _wait_until(
            lambda: harness.loop.diagnostics()["resize_debounce_pending"] is True
        )
        assert harness.client.starts == 1
        assert FRAME_STATUS_RESYNCING not in harness.slot.statuses()

        # Second consecutive snapshot at the same new size: replacement.
        harness.mailbox.write(_snapshot(2.0, width=4, height=4))
        assert _wait_until(lambda: harness.client.starts == 2)
        assert FRAME_STATUS_RESYNCING in harness.slot.statuses()
        assert harness.loop.diagnostics()["last_resync_reason"] == "output_shape_changed"
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_oscillating_resize_never_replaces_until_the_size_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()

        # A drag-resize storm: sizes never repeat consecutively.
        for index, size in enumerate(((2, 2), (3, 3), (2, 2), (3, 3)), start=2):
            harness.mailbox.write(
                _snapshot(2.0 + index, width=size[0], height=size[1])
            )
            assert _wait_until(
                lambda expected=index: harness.loop.diagnostics()["snapshots_taken"]
                >= expected
            )
        assert harness.client.starts == 1
        assert harness.loop.diagnostics()["session_replacements"] == 0

        # The drag ends: two agreeing snapshots replace once.
        harness.mailbox.write(_snapshot(9.0, width=3, height=3))
        harness.mailbox.write(_snapshot(9.5, width=3, height=3))
        assert _wait_until(lambda: harness.client.starts == 2)
    assert harness.loop.diagnostics()["session_replacements"] == 1


def test_ensure_failure_retries_follow_the_auto_retry_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failures: list[int] = []

    def _failing_ensure(_request: RenderRequest) -> None:
        failures.append(1)
        raise RenderClientError("worker launch failed")

    harness = _Harness(
        monkeypatch,
        tmp_path,
        ensure_session=_failing_ensure,
        retry_allowed=lambda: len(failures) < 2,
    )
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(lambda: len(harness.slot.frames()) >= 1)
        assert harness.slot.peek_latest().status == FRAME_STATUS_FAILED
        assert "worker launch failed" in harness.slot.peek_latest().detail
        assert len(failures) == 1

        # Fresh snapshot retries while the policy allows it.
        harness.mailbox.write(_snapshot(3.0))
        assert _wait_until(lambda: len(failures) == 2)

        # Policy exhausted: further snapshots attempt nothing.
        harness.mailbox.write(_snapshot(4.0))
        assert _wait_until(
            lambda: harness.loop.diagnostics()["snapshots_taken"] >= 3
        )
        assert len(failures) == 2
        assert harness.loop.diagnostics()["retry_blocked"] is True
        assert harness.loop.diagnostics()["ensure_failures"] == 2
    assert harness.client.starts == 0


class _PendingEditsScheduler:
    """Scheduler stub whose pending view updates are never drained."""

    has_pending_view_updates = True

    def __init__(self) -> None:
        self.wake_hook = None

    def set_edit_wake_hook(self, hook) -> None:
        self.wake_hook = hook

    def submit_edit(self, intent):
        return EditSubmissionResult(status=EditStatus.QUEUED, reason="")

    def tick_viewport(self, request, *, ovrtx_updates=None, project_complete_pose=False):
        return RuntimeTickResult(status=RuntimeTickStatus.NOOP, enabled=False)

    def shutdown(self) -> None:
        pass


def test_persistent_failure_with_pending_edits_paces_retries_no_busy_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """task02-03 follow-up: pending updates must not busy-retry a failure."""

    backoff = 0.02
    harness = _Harness(
        monkeypatch,
        tmp_path,
        scheduler=_PendingEditsScheduler(),
        failure_retry_backoff_seconds=backoff,
    )
    harness.client.fail_render = True
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        # Multiple paced retries happen (the failure persists, pending
        # updates keep requesting work) ...
        assert _wait_until(
            lambda: harness.loop.diagnostics()["failures"] >= 3
        )
    # ... and every mailbox consult after the first failure waited the
    # backoff interval — no timeout=0 busy polling in the failed state.
    records = list(harness.mailbox.records)
    first_failure_take = next(
        index for index, (timeout, taken) in enumerate(records) if taken
    )
    post_failure_timeouts = [
        timeout for timeout, _taken in records[first_failure_take + 1 :]
    ]
    assert post_failure_timeouts, "expected retry waits after the failure"
    assert all(
        timeout is None or timeout >= backoff for timeout in post_failure_timeouts
    )
    assert harness.loop.diagnostics()["retry_waits"] >= 2
