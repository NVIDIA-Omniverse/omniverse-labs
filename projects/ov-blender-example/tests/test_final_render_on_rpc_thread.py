# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final render job on the session RPC thread (task05-01/task05-03).

F12 with an active viewport session for the same scene submits its
final-render work — a dedicated session spec on the already-running
worker, chunked raw sample acquisitions, result readback,
simulation teardown — to the viewport session's render thread through the
render loop's exclusive-job seam (``LatestViewRenderLoop.call``). The
viewport loop yields between iterations while the job is queued/running.
The worker loads one simulation at a time, so the job first suspends the
viewport simulation (controller kept open) and the loop re-ensures the
session before resuming after the job; the Blender render job
thread blocks on the future, polling ``test_break()`` so long renders stay
cancellable; cancellation deletes the final-render simulation. Scenes
without a matching viewport session take the standalone path (task05-03):
the same job runs on a short-lived ``ViewportRenderThread`` constructed
for the render's duration — worker launch, chunked cancellable batches,
simulation delete, worker shutdown at render end, bounded thread join —
so the single-RPC-thread invariant holds even when the viewport never ran.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example import viewport_handoff  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import (  # noqa: E402
    RenderClientError,
    RenderResult,
    SIMULATION_ID_PREFIX,
)
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402
from ovrtx_blender_example.viewport_handoff import (  # noqa: E402
    FRAME_STATUS_FRAME,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (  # noqa: E402
    LatestViewRenderLoop,
    RenderThreadRejectedError,
    SessionLifecycleHooks,
)


WAIT_S = 5.0

FINAL_SIMULATION_ID = f"{SIMULATION_ID_PREFIX}final-{os.getpid()}"


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _final_render_threads() -> list[threading.Thread]:
    """Live short-lived standalone final-render RPC threads (task05-03)."""

    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("ovrtx-render-final-")
    ]


def _matrix(tx: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _viewport_request(tmp_path: Path, **overrides) -> RenderRequest:
    fields = dict(
        input_usd_path=str(tmp_path / "scene.usda"),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=2,
        height=2,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        camera_matrix=_matrix(1.0),
        worker_command="worker",
        native_client_module="client",
    )
    fields.update(overrides)
    return RenderRequest(**fields)


def _final_request(tmp_path: Path, **overrides) -> RenderRequest:
    fields = dict(
        input_usd_path=str(tmp_path / "scene.usda"),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=4,
        height=4,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        camera_matrix=None,
        worker_command="worker",
        native_client_module="client",
    )
    fields.update(overrides)
    return RenderRequest(**fields)


def _snapshot(tx: float = 2.0, **overrides) -> ViewSnapshot:
    fields = {
        "camera_matrix": _matrix(tx),
        "camera_prim_path": "/World/Camera",
        "min_samples": 1,
        "max_samples": 4,
        "selected_sensor_paths": ("/Render/Product",),
        "width": 2,
        "height": 2,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


class _Client:
    """Fake srtx client shared by the viewport and final-render sessions."""

    def __init__(self) -> None:
        self.rpc_thread_idents: list[tuple[str, int]] = []
        self.started: list[tuple[object, str | None]] = []
        self.deleted: list[str] = []
        self.render_calls: list[tuple[str, int]] = []
        self.render_camera_translations: list[float | None] = []
        self.camera_translation: float | None = None
        self.render_hook = None
        self.start_hook = None
        self.markers: list[str] | None = None
        self.startup_diagnostics = {"render_worker": {"status": "running"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def _record(self, name: str) -> None:
        self.rpc_thread_idents.append((name, threading.get_ident()))

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self._record("start_session")
        self.started.append((spec, simulation_id))
        if self.markers is not None:
            self.markers.append("start_session")
        hook = self.start_hook
        if hook is not None:
            hook(spec, simulation_id)
        return simulation_id or "sim-viewport"

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self._record("render_result")
        additional = int(kwargs["additional_samples"])
        self.render_calls.append((str(simulation_id), additional))
        self.render_camera_translations.append(self.camera_translation)
        if self.markers is not None:
            self.markers.append("render_result")
        hook = self.render_hook
        if hook is not None:
            hook(str(simulation_id), additional)
        width = 2 if simulation_id == "sim-viewport" else 4
        return RenderResult(
            width=width,
            height=width,
            rgba8=b"\x00" * (width * width * 4),
            completed_samples=additional,
            session_completed_samples=len(self.render_calls),
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        self._record("update_transforms")
        batch = tuple(values)
        for value in batch:
            if value.prim_path == "/World/Camera":
                self.camera_translation = float(value.matrix[0][3])
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        self._record("delete_simulation")
        self.deleted.append(str(simulation_id))
        return "stopped"

    def shutdown(self) -> None:
        self._record("shutdown")


class _Harness:
    """Real controller + real loop lifecycle + fake client (02-06 pattern)."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        replacement_reason=None,
    ) -> None:
        self.client = _Client()
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = _viewport_request(tmp_path)
        self.scheduler = RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = CameraRequestMailbox()
        self.slot = LatestFrameSlot()
        self.markers: list[str] = []

        def _ensure(request: RenderRequest) -> None:
            self.markers.append("ensure")
            self.controller.ensure(request)

        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
            lifecycle=SessionLifecycleHooks(
                ensure_session=_ensure,
                replacement_reason=replacement_reason
                or (lambda request: self.controller.would_replace(request)),
                retry_allowed=lambda: True,
            ),
        )

    def wait_refined(self, after_index: int = 0, timeout: float = WAIT_S) -> FrameState:
        newest: list[FrameState] = []

        def _refined() -> bool:
            frame = self.slot.peek_latest()
            if (
                frame is not None
                and frame.publication_index > after_index
                and frame.status == FRAME_STATUS_FRAME
                and frame.completed_samples >= self.base_request.max_samples
            ):
                newest.append(frame)
                return True
            return False

        assert _wait_until(_refined, timeout), "viewport never refined"
        return newest[-1]


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="final-render-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Loop-level exclusive-job seam
# ---------------------------------------------------------------------------


def test_exclusive_job_runs_on_the_loop_thread_and_viewport_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    with _running(harness.loop) as thread:
        harness.mailbox.write(_snapshot(2.0))
        refined = harness.wait_refined()

        future = harness.loop.call(threading.get_ident, label="ident-probe")
        assert future.result(WAIT_S) == thread.ident
        assert harness.loop.diagnostics()["exclusive_jobs"] == 1
        assert harness.loop.diagnostics()["last_exclusive_job_label"] == "ident-probe"

        # The viewport resumes: a fresh view still renders and publishes.
        harness.mailbox.write(_snapshot(3.0))
        harness.wait_refined(after_index=refined.publication_index)


def test_exclusive_job_wakes_a_fully_refined_parked_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()
        # Parked on take(timeout=None): the job submission alone (no
        # mailbox write) must wake the loop and run the job promptly.
        future = harness.loop.call(lambda: "ran", label="parked-wake")
        assert future.result(WAIT_S) == "ran"


def test_viewport_yields_while_a_job_runs_and_resumes_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    gate = threading.Event()
    entered = threading.Event()

    def _blocking_job() -> str:
        entered.set()
        assert gate.wait(WAIT_S)
        return "done"

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, max_samples=0))
        harness.wait_refined()

        future = harness.loop.call(_blocking_job, label="final-render")
        assert entered.wait(WAIT_S)
        paused_index = harness.slot.latest_index()
        renders_before = len(harness.client.render_camera_translations)
        # While the job occupies the thread the viewport yields: new
        # snapshots coalesce in the mailbox and nothing publishes.
        harness.mailbox.write(_snapshot(3.0))
        harness.mailbox.write(_snapshot(4.0))
        time.sleep(0.05)
        assert harness.slot.latest_index() == paused_index
        assert harness.loop.diagnostics()["exclusive_job_active"] is True

        gate.set()
        assert future.result(WAIT_S) == "done"
        # The loop resumes and renders the newest coalesced view.
        harness.wait_refined(after_index=paused_index)
        assert harness.client.render_camera_translations[renders_before] == 4.0


def test_exclusive_job_pauses_every_pane_sharing_the_controller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = CameraRequestMailbox()
    second_slot = LatestFrameSlot()
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
    gate = threading.Event()
    entered = threading.Event()

    def _blocking_final_job() -> None:
        entered.set()
        assert gate.wait(WAIT_S)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, max_samples=0))
        harness.wait_refined()
        with _running(second_loop):
            second_mailbox.write(_snapshot(9.0, max_samples=0))
            assert _wait_until(lambda: second_slot.latest_index() >= 4)

            future = harness.loop.call(_blocking_final_job, label="final-render")
            assert entered.wait(WAIT_S)
            first_index = harness.slot.latest_index()
            second_index = second_slot.latest_index()
            time.sleep(0.05)
            assert harness.slot.latest_index() == first_index
            assert second_slot.latest_index() == second_index

            gate.set()
            assert future.result(WAIT_S) is None
            assert _wait_until(lambda: harness.slot.latest_index() > first_index)
            assert _wait_until(lambda: second_slot.latest_index() > second_index)


def test_suspending_final_job_restores_shared_panes_before_resuming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = CameraRequestMailbox()
    second_slot = LatestFrameSlot()
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
        harness.mailbox.write(_snapshot(2.0, max_samples=0))
        harness.wait_refined()
        with _running(second_loop):
            second_mailbox.write(_snapshot(9.0, max_samples=0))
            assert _wait_until(lambda: second_slot.latest_index() >= 4)
            first_index = harness.slot.latest_index()
            second_index = second_slot.latest_index()
            future = harness.loop.call(harness.controller.suspend, label="final-render")
            assert future.result(WAIT_S) == "stopped"
            assert _wait_until(lambda: len(harness.client.started) == 2)
            assert _wait_until(lambda: harness.slot.latest_index() > first_index)
            assert _wait_until(lambda: second_slot.latest_index() > second_index)
            assert harness.loop.diagnostics()["failed_state"] is False
            assert second_loop.diagnostics()["failed_state"] is False


def test_failed_shared_restore_wakes_every_pane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = CameraRequestMailbox()
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=LatestFrameSlot(),
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, max_samples=0))
        harness.wait_refined()
        with _running(second_loop):
            second_mailbox.write(_snapshot(9.0, max_samples=0))
            assert _wait_until(lambda: second_loop.diagnostics()["iterations"] >= 2)

            def _fail_restore(spec: object, simulation_id: str | None) -> None:
                raise RenderClientError("restore failed")

            harness.client.start_hook = _fail_restore
            future = harness.loop.call(harness.controller.suspend, label="final-render")
            assert future.result(WAIT_S) == "stopped"
            assert _wait_until(
                lambda: harness.loop.diagnostics()["failed_state"]
                and second_loop.diagnostics()["failed_state"]
            )


def test_shared_pane_stop_is_not_blocked_by_another_panes_final_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    harness.controller._allow_serialized_threads()
    second_mailbox = CameraRequestMailbox()
    second_loop = LatestViewRenderLoop(
        mailbox=second_mailbox,
        frame_slot=LatestFrameSlot(),
        controller=harness.controller,
        scheduler=harness.scheduler,
        request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
            harness.base_request, snapshot
        ),
        owns_scheduler=False,
    )
    gate = threading.Event()
    entered = threading.Event()

    def _blocking_final_job() -> None:
        entered.set()
        assert gate.wait(WAIT_S)

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, max_samples=0))
        harness.wait_refined()
        with _running(second_loop) as second_thread:
            second_mailbox.write(_snapshot(9.0, max_samples=0))
            future = harness.loop.call(_blocking_final_job, label="final-render")
            assert entered.wait(WAIT_S)
            second_loop.request_stop()
            second_thread.join(0.5)
            assert not second_thread.is_alive()
            gate.set()
            assert future.result(WAIT_S) is None


def test_job_exception_delivers_to_the_future_and_loop_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    def _failing_job() -> None:
        raise RenderClientError("final render exploded")

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        refined = harness.wait_refined()

        future = harness.loop.call(_failing_job, label="final-render")
        with pytest.raises(RenderClientError, match="final render exploded"):
            future.result(WAIT_S)
        assert harness.loop.diagnostics()["exclusive_job_failures"] == 1

        # The failure belongs to the waiting caller: the loop keeps
        # serving the viewport.
        harness.mailbox.write(_snapshot(3.0))
        harness.wait_refined(after_index=refined.publication_index)


def test_jobs_never_interleave_with_ensure_or_render_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Job markers are contiguous: no RPC or ensure lands inside a job."""

    replace_once: list[str] = []
    harness = _Harness(
        monkeypatch,
        tmp_path,
        replacement_reason=lambda request: replace_once.pop() if replace_once else "",
    )
    markers = harness.markers
    harness.client.markers = markers

    def _job() -> None:
        markers.append("job-start")
        markers.append("job-end")

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()

        # A pending replacement and a queued job around the same snapshot.
        replace_once.append("generation_changed")
        future = harness.loop.call(_job, label="final-render")
        harness.mailbox.write(_snapshot(2.0))
        future.result(WAIT_S)
        assert _wait_until(lambda: markers.count("ensure") >= 2)

    starts = [index for index, mark in enumerate(markers) if mark == "job-start"]
    assert starts, "job never ran"
    for index in starts:
        assert markers[index + 1] == "job-end"


def test_loop_exit_rejects_queued_jobs_and_later_submissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    gate = threading.Event()
    entered = threading.Event()

    def _blocking_job() -> str:
        status = harness.controller.suspend()
        entered.set()
        assert gate.wait(WAIT_S)
        return status

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined()
        first = harness.loop.call(_blocking_job, label="first")
        assert entered.wait(WAIT_S)
        queued = harness.loop.call(lambda: "second", label="second")
        harness.loop.request_stop()
        gate.set()
        # The in-flight job completes and resolves; the queued one is
        # rejected when the loop exits (typed, never silently dropped).
        assert first.result(WAIT_S) == "stopped"
        with pytest.raises(RenderThreadRejectedError):
            queued.result(WAIT_S)
        assert len(harness.client.started) == 1
    with pytest.raises(RenderThreadRejectedError):
        harness.loop.call(lambda: "late", label="late")


# ---------------------------------------------------------------------------
# Final-render job body
# ---------------------------------------------------------------------------


def _load_engine_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_final_render_rpc_thread_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"

    class _FakeRenderEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stats: list[tuple[str, str]] = []
            self.reports: list[tuple[set[str], str]] = []
            self.test_break_result = False
            self.test_break_calls = 0

        def update_stats(self, engine: str, message: str) -> None:
            self.stats.append((engine, message))

        def tag_redraw(self) -> None:
            pass

        def report(self, levels: set[str], message: str) -> None:
            self.reports.append((set(levels), message))

        def test_break(self) -> bool:
            self.test_break_calls += 1
            return self.test_break_result

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


def test_final_render_job_creates_its_own_session_and_deletes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ovrtx_blender_example import engine as engine_module

    client = _Client()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: client,
    )
    request = _final_request(tmp_path)

    outcome = engine_module._run_final_render_job(request, threading.Event())

    assert outcome["status"] == engine_module.FINAL_RENDER_STATUS_COMPLETED
    assert outcome["result"].completed_samples == request.max_samples
    assert outcome["composition"] is not None
    # One dedicated session in the final lane (never the viewport's), the
    # pose-override isolation structural: scene camera stays composed.
    assert [sim for _spec, sim in client.started] == [FINAL_SIMULATION_ID]
    spec = client.started[0][0]
    assert spec.camera_pose_source == "composed_scene"
    assert (spec.width, spec.height) == (request.width, request.height)
    # Chunked batches to the fixed endpoint: 1 → 2 → 4 samples.
    assert [
        additional for sim, additional in client.render_calls if sim == FINAL_SIMULATION_ID
    ] == [1, 1, 2]
    # The final-render simulation is always deleted before returning.
    assert client.deleted == [FINAL_SIMULATION_ID]


@pytest.mark.parametrize("cancelled", (False, True))
def test_final_render_job_owns_progress_when_borrowing_a_nonzero_session_cursor(
    cancelled: bool,
) -> None:
    from ovrtx_blender_example import engine as engine_module

    class BorrowedController:
        render_calls: list[int] = []
        shutdown_calls = 0

        def ensure(self, _request: RenderRequest) -> object:
            raise AssertionError("prepared controller must not be ensured again")

        def render(
            self,
            _request: RenderRequest,
            *,
            additional_samples: int,
        ) -> RenderResult:
            self.render_calls.append(additional_samples)
            return RenderResult(
                width=1,
                height=1,
                rgba8=b"\x00\x00\x00\xff",
                completed_samples=additional_samples,
                session_completed_samples=50 + len(self.render_calls),
                simulation_time_ns=0,
            )

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    controller = BorrowedController()
    composition = object()
    outcome = engine_module._run_final_render_job(
        RenderRequest(max_samples=4),
        threading.Event(),
        controller=controller,
        composition=composition,
        cancel_requested=lambda: cancelled,
    )

    assert outcome["status"] == (
        engine_module.FINAL_RENDER_STATUS_CANCELLED
        if cancelled
        else engine_module.FINAL_RENDER_STATUS_COMPLETED
    )
    assert outcome["composition"] is composition
    assert controller.render_calls == ([] if cancelled else [1, 1, 2])
    if not cancelled:
        assert outcome["result"].completed_samples == 4
        assert outcome["result"].session_completed_samples == 53
    assert controller.shutdown_calls == 0


@pytest.mark.parametrize(
    "controller, composition",
    ((object(), None), (None, object())),
)
def test_final_render_job_rejects_partial_borrowed_state(
    controller: object | None,
    composition: object | None,
) -> None:
    from ovrtx_blender_example import engine as engine_module

    with pytest.raises(ValueError, match="must be supplied together"):
        engine_module._run_final_render_job(
            RenderRequest(),
            threading.Event(),
            controller=controller,
            composition=composition,
        )


def test_final_render_job_cancel_deletes_the_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ovrtx_blender_example import engine as engine_module

    client = _Client()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: client,
    )
    cancel_event = threading.Event()
    # The cancel lands while the first batch renders; the job observes it
    # at the next batch boundary.
    client.render_hook = lambda _sim, _additional: cancel_event.set()
    request = _final_request(tmp_path)

    outcome = engine_module._run_final_render_job(request, cancel_event)

    assert outcome["status"] == engine_module.FINAL_RENDER_STATUS_CANCELLED
    assert outcome["result"] is None
    assert len(client.render_calls) == 1
    assert client.deleted == [FINAL_SIMULATION_ID]


def test_final_render_job_suspends_the_host_session_before_creating_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker loads one simulation at a time: the hosting viewport
    simulation must be suspended before the final lane is created."""

    from ovrtx_blender_example import engine as engine_module

    client = _Client()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: client,
    )
    request = _final_request(tmp_path)
    order: list[str] = []
    client.markers = order

    def _suspend() -> str:
        order.append("suspend")
        return "stopped"

    outcome = engine_module._run_final_render_job(
        request, threading.Event(), suspend_host_session=_suspend
    )

    assert outcome["status"] == engine_module.FINAL_RENDER_STATUS_COMPLETED
    assert order[:2] == ["suspend", "start_session"]
    assert client.deleted == [FINAL_SIMULATION_ID]


def test_final_render_job_fails_when_the_host_suspend_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ovrtx_blender_example import engine as engine_module

    client = _Client()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: client,
    )
    request = _final_request(tmp_path)

    with pytest.raises(RenderClientError, match="suspend the viewport"):
        engine_module._run_final_render_job(
            request, threading.Event(), suspend_host_session=lambda: "failed"
        )

    # The final lane was never created against the still-loaded viewport
    # simulation.
    assert client.started == []


# ---------------------------------------------------------------------------
# Engine-level F12 flow (fake bpy)
# ---------------------------------------------------------------------------


SCENE_UID = 77


class _FinalRenderAdapter:
    def __init__(self, request: RenderRequest) -> None:
        self._request = request

    def final_render(self, depsgraph: object) -> RenderRequest:
        return self._request


@contextmanager
def _viewport_session(module, monkeypatch: pytest.MonkeyPatch, client, request):
    """Running viewport engine (real thread/loop, direct route, fake client)."""

    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    try:
        engine._begin_async_viewport_session(request)
        engine._viewport_scene = SimpleNamespace(session_uid=SCENE_UID)
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        assert frame.status == module.viewport_handoff.FRAME_STATUS_FRAME
        yield engine
    finally:
        engine._viewport_scene = None
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)


def _f12_engine(module, monkeypatch: pytest.MonkeyPatch, request: RenderRequest):
    monkeypatch.setattr(
        module,
        "_render_callback_adapter",
        lambda engine_id="": _FinalRenderAdapter(request),
    )
    monkeypatch.setattr(
        module, "_final_render_color_presentation_from_scene", lambda scene: {}
    )
    engine = module.OvrtxExampleRenderEngine()
    written: list[RenderResult] = []
    engine._write_blender_result = written.append
    return engine, written


@pytest.mark.parametrize(
    "job_status, restore_status, expected_writes, expected_report",
    (
        ("completed", "stopped", 1, None),
        ("cancelled", "stopped", 0, None),
        ("error", "stopped", 0, "render failed"),
        ("completed", "failed", 0, "Could not restore prepared viewport state"),
        ("error", "failed", 0, "after render failure: render failed"),
    ),
)
def test_current_scene_final_render_restores_once_before_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    job_status: str,
    restore_status: str,
    expected_writes: int,
    expected_report: str | None,
) -> None:
    module = _load_engine_module(monkeypatch)
    request = _final_request(tmp_path)
    scene = SimpleNamespace(session_uid=SCENE_UID)
    monkeypatch.setattr(
        module,
        "_render_callback_adapter",
        lambda engine_id="": _FinalRenderAdapter(request),
    )
    presentation = {"mode": "scene_linear_hdr"}
    monkeypatch.setattr(
        module,
        "_final_render_color_presentation_from_scene",
        lambda _scene: presentation,
    )
    monkeypatch.setattr(module.scene_generation_sessions, "owns_request", lambda *_: True)
    monkeypatch.setattr(module, "_end_active_viewport_sessions", lambda *_: True)

    lifecycle = {"activated": 0, "restored": 0, "shutdown": 0, "jobs": 0}
    composition = object()
    borrowed_controller = object()

    class Lease:
        controller = borrowed_controller
        last_ensure_result = SimpleNamespace(composition=composition)

        def deactivate(self) -> str:
            lifecycle["restored"] += 1
            return restore_status

    def activate(*_args: object, **_kwargs: object) -> Lease:
        lifecycle["activated"] += 1
        return Lease()

    class Controller:
        def shutdown(self) -> None:
            lifecycle["shutdown"] += 1

    def run_job(render_request: RenderRequest, *_args: object, **kwargs: object):
        lifecycle["jobs"] += 1
        assert render_request.color_presentation == presentation
        assert kwargs["controller"] is borrowed_controller
        assert kwargs["composition"] is composition
        if job_status == "error":
            raise module.RenderClientError("render failed")
        return {
            "status": job_status,
            "result": SimpleNamespace(completed_samples=request.max_samples),
            "composition": composition,
        }

    monkeypatch.setattr(module.scene_generation_sessions, "activate_for_final_render", activate)
    monkeypatch.setattr(module, "OvrtxSessionController", Controller)
    monkeypatch.setattr(module, "_run_final_render_job", run_job)
    engine = module.OvrtxExampleRenderEngine()
    writes: list[object] = []
    artifacts: list[object] = []
    engine._write_blender_result = writes.append
    engine._write_result_artifact = lambda *args, **kwargs: artifacts.append((args, kwargs))

    engine.render(SimpleNamespace(scene=scene))

    assert lifecycle == {"activated": 1, "restored": 1, "shutdown": 1, "jobs": 1}
    assert len(writes) == len(artifacts) == expected_writes
    if expected_report is None:
        assert engine.reports == []
    else:
        assert expected_report in engine.reports[-1][1]


def test_f12_rides_the_viewport_rpc_thread_and_the_viewport_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    viewport_request = _viewport_request(tmp_path, max_samples=0)
    final_request = _final_request(tmp_path)
    with _viewport_session(module, monkeypatch, client, viewport_request) as viewport:
        render_thread_ident = viewport._render_thread._thread_ident
        starts_before = len(client.started)
        f12, written = _f12_engine(module, monkeypatch, final_request)
        depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

        latest = viewport._frame_slot.latest_index()
        f12.render(depsgraph)

        # The final frame reached Blender's render result at the endpoint.
        assert len(written) == 1
        assert written[0].completed_samples == final_request.max_samples
        assert f12.stats[-1] == ("ovrtx", "Done")
        assert f12.reports == []

        # No second worker/thread: the F12 engine started no render thread
        # and every srtx RPC — the viewport's and the final render's —
        # ran on the one session RPC thread.
        assert f12._render_thread is None
        foreign = {ident for _name, ident in client.rpc_thread_idents}
        assert foreign == {render_thread_ident}
        assert threading.get_ident() not in foreign

        # The F12 session is its own spec on the shared worker: final
        # lane simulation ID, final resolution, composed scene camera
        # (the viewport's live pose override cannot leak in).
        final_starts = client.started[starts_before:]
        assert [sim for _spec, sim in final_starts] == [FINAL_SIMULATION_ID]
        final_spec = final_starts[0][0]
        assert final_spec.camera_pose_source == "composed_scene"
        assert (final_spec.width, final_spec.height) == (4, 4)
        viewport_spec = client.started[0][0]
        assert viewport_spec.camera_pose_source == "runtime_update"

        # The worker loads one simulation at a time: the job suspended the
        # viewport simulation before creating the final lane, then deleted
        # the final simulation before returning.
        assert client.deleted == ["sim-viewport", FINAL_SIMULATION_ID]

        # The continuously eligible viewport recovers without a new input:
        # the loop sees the suspended session, re-ensures it, and publishes.

        def _fresh_frame() -> bool:
            frame = viewport._frame_slot.peek_latest()
            return (
                frame is not None
                and frame.publication_index > latest
                and frame.status == module.viewport_handoff.FRAME_STATUS_FRAME
            )

        assert _wait_until(_fresh_frame)
        # The re-ensured session is a fresh viewport-lane simulation.
        assert [sim for _spec, sim in client.started].count(None) == 2
    # Session end tears the recreated viewport simulation down as usual.
    assert client.deleted.count("sim-viewport") == 2


def test_f12_for_another_scene_takes_the_standalone_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    viewport_request = _viewport_request(tmp_path)
    final_request = _final_request(tmp_path)
    with _viewport_session(module, monkeypatch, client, viewport_request) as viewport:
        starts_before = len(client.started)
        f12, written = _f12_engine(module, monkeypatch, final_request)
        depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID + 1))

        f12.render(depsgraph)

        assert len(written) == 1
        # The standalone path (task05-03) owns its RPC context: the same
        # job machinery on a short-lived RPC thread — final simulation
        # lane, chunked batches — never Blender's render job thread and
        # never the other scene's viewport thread.
        standalone_starts = client.started[starts_before:]
        assert [sim for _spec, sim in standalone_starts] == [FINAL_SIMULATION_ID]
        standalone_start_ident = [
            ident
            for name, ident in client.rpc_thread_idents
            if name == "start_session"
        ][-1]
        assert standalone_start_ident != threading.get_ident()
        # The viewport thread is stopped before this short-lived thread starts;
        # OS thread identifiers may therefore be reused.
        # Chunked batches to the fixed endpoint, same as the
        # viewport-hosted route (translated min_samples preserved).
        assert [
            additional
            for sim, additional in client.render_calls
            if sim == FINAL_SIMULATION_ID
        ] == [1, 1, 2]
        # Render-end teardown ran before render() returned: simulation
        # deleted and the short-lived thread joined.
        assert FINAL_SIMULATION_ID in client.deleted
        assert not _final_render_threads()
        # The process GPU lease is exclusive per worker launch: the other
        # scene's viewport session ends (session replaced, simulation
        # deleted) before the standalone worker launch so the lease is
        # free for the F12 lane.
        assert "sim-viewport" in client.deleted
        assert client.deleted.index("sim-viewport") < client.deleted.index(
            FINAL_SIMULATION_ID
        )


def test_f12_cancel_via_test_break_deletes_the_final_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    viewport_request = _viewport_request(tmp_path)
    final_request = _final_request(tmp_path)
    with _viewport_session(module, monkeypatch, client, viewport_request):
        f12, written = _f12_engine(module, monkeypatch, final_request)
        f12.test_break_result = True

        # Capture the job's cancel event so the fake render can hold the
        # first batch until the render job thread has actually flagged the
        # cancel — the job then deterministically observes it at the next
        # batch boundary.
        captured: dict[str, threading.Event] = {}
        original_job = module._run_final_render_job

        def _capturing_job(
            request: object,
            cancel_event: threading.Event,
            progress: dict | None = None,
            suspend_host_session=None,
        ) -> dict:
            captured["cancel"] = cancel_event
            return original_job(
                request,
                cancel_event,
                progress,
                suspend_host_session=suspend_host_session,
            )

        monkeypatch.setattr(module, "_run_final_render_job", _capturing_job)

        def _final_render_blocks_until_cancel(sim: str, _additional: int) -> None:
            if sim == FINAL_SIMULATION_ID:
                assert _wait_until(lambda: captured.get("cancel") is not None)
                assert captured["cancel"].wait(WAIT_S)

        client.render_hook = _final_render_blocks_until_cancel
        depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

        f12.render(depsgraph)

        # Cancelled: nothing written, no error, the final-render
        # simulation deleted, at most the in-flight batch rendered.
        assert written == []
        assert f12.reports == []
        assert _wait_until(lambda: FINAL_SIMULATION_ID in client.deleted)
        final_batches = [
            additional
            for sim, additional in client.render_calls
            if sim == FINAL_SIMULATION_ID
        ]
        assert len(final_batches) == 1


def test_f12_job_failure_reports_error_and_the_viewport_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    viewport_request = _viewport_request(tmp_path)
    final_request = _final_request(tmp_path)

    def _fail_final_session(_spec: object, simulation_id: str | None) -> None:
        if simulation_id == FINAL_SIMULATION_ID:
            raise RenderClientError("final session refused")

    client.start_hook = _fail_final_session
    with _viewport_session(module, monkeypatch, client, viewport_request) as viewport:
        f12, written = _f12_engine(module, monkeypatch, final_request)
        depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

        f12.render(depsgraph)

        assert written == []
        assert len(f12.reports) == 1
        levels, message = f12.reports[0]
        assert "ERROR" in levels
        assert "final session refused" in message
        # The failed call rejected only the F12 future: the loop and the
        # viewport session keep serving.
        latest = viewport._frame_slot.latest_index()
        viewport._camera_mailbox.write(_snapshot(9.0))
        assert viewport._frame_slot.wait_for_newer(latest, timeout=WAIT_S) is not None


# ---------------------------------------------------------------------------
# Standalone final render path — no viewport session (task05-03)
# ---------------------------------------------------------------------------


def _standalone_f12(
    module,
    monkeypatch: pytest.MonkeyPatch,
    client: _Client,
    final_request: RenderRequest,
):
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    return _f12_engine(module, monkeypatch, final_request)


def _capture_cancel_event(module, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Expose the standalone job's cancel event to fake-client hooks."""

    captured: dict[str, threading.Event] = {}
    original_job = module._run_final_render_job

    def _capturing_job(
        request: object,
        cancel_event: threading.Event,
        progress: dict | None = None,
        suspend_host_session=None,
    ) -> dict:
        captured["cancel"] = cancel_event
        return original_job(
            request,
            cancel_event,
            progress,
            suspend_host_session=suspend_host_session,
        )

    monkeypatch.setattr(module, "_run_final_render_job", _capturing_job)
    return captured


def test_standalone_f12_renders_on_a_short_lived_rpc_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No viewport session at all: F12 launches, renders, and shuts down."""

    module = _load_engine_module(monkeypatch)
    client = _Client()
    final_request = _final_request(tmp_path)
    f12, written = _standalone_f12(module, monkeypatch, client, final_request)
    depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

    f12.render(depsgraph)

    # The frame reached Blender's render result at the fixed endpoint,
    # with no errors reported.
    assert len(written) == 1
    assert written[0].completed_samples == final_request.max_samples
    assert f12.stats[-1] == ("ovrtx", "Done")
    assert f12.reports == []

    # One code path with the viewport-hosted route: the F12 session rides
    # the final simulation lane with the composed scene camera and renders
    # chunked batches (translated min_samples doubling to the endpoint) —
    # no collapsed single fixed-endpoint batch.
    assert [sim for _spec, sim in client.started] == [FINAL_SIMULATION_ID]
    spec = client.started[0][0]
    assert spec.camera_pose_source == "composed_scene"
    assert (spec.width, spec.height) == (final_request.width, final_request.height)
    assert [additional for _sim, additional in client.render_calls] == [1, 1, 2]

    # Every srtx RPC — session start, batches, teardown — ran on one
    # dedicated short-lived RPC thread, never Blender's render job thread.
    idents = {ident for _name, ident in client.rpc_thread_idents}
    assert len(idents) == 1
    assert threading.get_ident() not in idents

    # Render-end teardown (recorded decision: worker shutdown at render
    # end) completed before render() returned: simulation deleted, then
    # the client/worker shut down, and the bounded join retired the thread.
    assert client.deleted == [FINAL_SIMULATION_ID]
    assert [name for name, _ident in client.rpc_thread_idents][-2:] == [
        "delete_simulation",
        "shutdown",
    ]
    assert not _final_render_threads()


def test_standalone_f12_cancel_via_test_break_still_tears_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    final_request = _final_request(tmp_path)
    f12, written = _standalone_f12(module, monkeypatch, client, final_request)
    f12.test_break_result = True
    captured = _capture_cancel_event(module, monkeypatch)

    def _final_render_blocks_until_cancel(_sim: str, _additional: int) -> None:
        # Hold the first batch until the render job thread has actually
        # flagged the cancel; the job observes it at the batch boundary.
        assert _wait_until(lambda: captured.get("cancel") is not None)
        assert captured["cancel"].wait(WAIT_S)

    client.render_hook = _final_render_blocks_until_cancel
    depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

    f12.render(depsgraph)

    # Cancelled: nothing written, no error (Blender's cancel semantics).
    assert written == []
    assert f12.reports == []
    # The bounded stop/join drained the job's render-end teardown before
    # render() returned: at most the in-flight batch rendered, the
    # simulation was deleted, the worker shut down, the thread retired.
    assert len(client.render_calls) == 1
    assert client.deleted == [FINAL_SIMULATION_ID]
    assert [name for name, _ident in client.rpc_thread_idents][-1] == "shutdown"
    assert not _final_render_threads()


def test_standalone_f12_failure_reports_error_and_retires_the_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_engine_module(monkeypatch)
    client = _Client()
    final_request = _final_request(tmp_path)

    def _fail_final_session(_spec: object, simulation_id: str | None) -> None:
        if simulation_id == FINAL_SIMULATION_ID:
            raise RenderClientError("standalone session refused")

    client.start_hook = _fail_final_session
    f12, written = _standalone_f12(module, monkeypatch, client, final_request)
    depsgraph = SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID))

    f12.render(depsgraph)

    assert written == []
    assert len(f12.reports) == 1
    levels, message = f12.reports[0]
    assert "ERROR" in levels
    assert "standalone session refused" in message
    # No session was created, so nothing to delete; the short-lived
    # thread still joined cleanly.
    assert client.deleted == []
    assert not _final_render_threads()


def test_viewport_session_launches_cleanly_after_a_standalone_f12(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Worker lifetime decision: standalone F12 shuts the worker down at
    render end, so a viewport session that follows launches fresh (a fresh
    worker has nothing to sweep) instead of attaching to a survivor."""

    module = _load_engine_module(monkeypatch)
    client = _Client()
    final_request = _final_request(tmp_path)
    f12, written = _standalone_f12(module, monkeypatch, client, final_request)

    f12.render(SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID)))

    assert len(written) == 1
    assert [name for name, _ident in client.rpc_thread_idents][-1] == "shutdown"
    assert not _final_render_threads()

    # A subsequent viewport session starts, serves frames, and tears down
    # normally (the _viewport_session context asserts the first frame).
    viewport_request = _viewport_request(tmp_path)
    with _viewport_session(module, monkeypatch, client, viewport_request) as viewport:
        assert viewport._frame_slot.peek_latest() is not None
    assert "sim-viewport" in client.deleted


# ---------------------------------------------------------------------------
# Scene binding (host selection)
# ---------------------------------------------------------------------------


class _FakeEngineEntry:
    """Weakref-able stand-in for a tracked viewport engine."""

    def __init__(
        self,
        *,
        scene_uid: int | None = SCENE_UID,
        thread_status: str = "running",
        with_loop: bool = True,
        with_thread: bool = True,
    ) -> None:
        self._viewport_scene = (
            SimpleNamespace(session_uid=scene_uid) if scene_uid is not None else None
        )
        self._render_thread = (
            SimpleNamespace(status=lambda: thread_status) if with_thread else None
        )
        self._render_loop = object() if with_loop else None


def _fake_viewport_engine(**kwargs) -> _FakeEngineEntry:
    return _FakeEngineEntry(**kwargs)


def test_viewport_final_render_host_scene_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovrtx_blender_example import engine as engine_module

    match = _fake_viewport_engine()
    others = [
        _fake_viewport_engine(scene_uid=SCENE_UID + 1),
        _fake_viewport_engine(scene_uid=None),
        _fake_viewport_engine(thread_status="stopped"),
        _fake_viewport_engine(with_thread=False),
        _fake_viewport_engine(with_loop=False),
    ]
    for candidate in (*others, match):
        engine_module._ACTIVE_VIEWPORT_ENGINES.add(candidate)
    try:
        scene = SimpleNamespace(session_uid=SCENE_UID)
        assert engine_module._viewport_final_render_host(scene) == (match, True)
        # No session serves the other scene; and a scene without a
        # session_uid never matches (standalone path both times).
        assert (
            engine_module._viewport_final_render_host(
                SimpleNamespace(session_uid=SCENE_UID + 5)
            )
            == (None, True)
        )
        assert engine_module._viewport_final_render_host(SimpleNamespace()) == (
            None,
            True,
        )
        assert engine_module._viewport_final_render_host(None) == (None, True)
    finally:
        for candidate in (*others, match):
            engine_module._ACTIVE_VIEWPORT_ENGINES.discard(candidate)
