# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import render_requests
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult
from ovrtx_blender_example.render_requests import RenderRequest
from ovrtx_blender_example.viewport_handoff import (
    FRAME_STATUS_FAILED,
    FRAME_STATUS_FRAME,
    FRAME_STATUS_RESYNCING,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
    request_from_snapshot,
    snapshot_from_render_request,
)


WAIT_S = 5.0

RAW_MATRIX = (
    (1.00000004, 0.0, 0.0, 0.123456789),
    (0.0, 1.0, 0.0, -0.0),
    (0.0, 0.0, 1.0, 4.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _snapshot(**overrides) -> ViewSnapshot:
    fields = {
        "camera_matrix": RAW_MATRIX,
        "camera_prim_path": "/World/OVRTXCamera",
        "min_samples": 1,
        "max_samples": 64,
        "selected_sensor_paths": ("/World/Sensors/Camera",),
        "width": 960,
        "height": 540,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


def _render_result(sample: int = 1) -> RenderResult:
    return RenderResult(
        width=4,
        height=4,
        rgba8=bytes(4 * 4 * 4),
        completed_samples=sample,
        session_completed_samples=sample,
        simulation_time_ns=0,
    )


def _frame(sample: int = 1, **overrides) -> FrameState:
    fields = {
        "status": FRAME_STATUS_FRAME,
        "render_result": _render_result(sample),
        "snapshot_key": _snapshot().key,
        "completed_samples": sample,
        "generation": 1,
    }
    fields.update(overrides)
    return FrameState(**fields)


# --- ViewSnapshot -----------------------------------------------------------


def test_snapshot_stabilizes_camera_matrix_on_construction() -> None:
    snapshot = _snapshot()
    assert snapshot.camera_matrix == render_requests.stable_camera_matrix(RAW_MATRIX)
    # Negative zero is normalized so identical poses share one identity.
    assert snapshot.camera_matrix[1][3] == 0.0
    assert str(snapshot.camera_matrix[1][3]) == "0.0"


def test_snapshot_sample_limit_is_not_rendered_view_identity() -> None:
    request = RenderRequest(
        selected_sensor_paths=("/World/Sensors/Camera",),
        min_samples=2,
        max_samples=32,
        camera_matrix=RAW_MATRIX,
        camera_prim_path="/World/OVRTXCamera",
        color_presentation={"render_var": "HdrColor"},
    )
    snapshot = snapshot_from_render_request(request)
    raised = dataclasses.replace(snapshot, max_samples=64)
    unbounded = dataclasses.replace(snapshot, max_samples=0)

    assert raised.key == snapshot.key == unbounded.key
    assert (snapshot.max_samples, raised.max_samples, unbounded.max_samples) == (
        32,
        64,
        0,
    )


def test_snapshot_from_render_request_maps_fields() -> None:
    request = RenderRequest(
        selected_sensor_paths=("/World/Sensors/A", "/World/Sensors/B"),
        width=800,
        height=450,
        min_samples=4,
        max_samples=256,
        camera_prim_path="/World/Cam",
        camera_matrix=RAW_MATRIX,
        timeline_controls_enabled=True,
        timeline_playing=True,
        timeline_frame=7,
        timeline_start=2,
        timeline_end=90,
        simulation_reset_token=3,
        color_presentation={"render_var": "HdrColor"},
    )
    snapshot = snapshot_from_render_request(
        request, timing_marks={"redraw_requested_monotonic_ns": 123}
    )
    assert snapshot.camera_prim_path == "/World/Cam"
    assert snapshot.selected_sensor_paths == ("/World/Sensors/A", "/World/Sensors/B")
    assert (snapshot.min_samples, snapshot.max_samples) == (4, 256)
    assert (snapshot.width, snapshot.height) == (800, 450)
    assert snapshot.render_var == "HdrColor"
    assert snapshot.timeline_controls_enabled is True
    assert snapshot.timeline_playing is True
    assert (snapshot.timeline_frame, snapshot.timeline_start, snapshot.timeline_end) == (7, 2, 90)
    assert snapshot.simulation_reset_token == 3
    assert snapshot.timing_marks["redraw_requested_monotonic_ns"] == 123


def test_snapshot_render_var_defaults_to_ldr_color() -> None:
    snapshot = snapshot_from_render_request(RenderRequest())
    assert snapshot.render_var == "LdrColor"


def test_snapshot_written_mark_is_stamped_and_preserved() -> None:
    before = time.perf_counter_ns()
    stamped = _snapshot()
    after = time.perf_counter_ns()
    assert before <= stamped.written_monotonic_ns <= after
    explicit = _snapshot(written_monotonic_ns=42)
    assert explicit.written_monotonic_ns == 42


def test_snapshot_is_frozen() -> None:
    snapshot = _snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.width = 1


def test_snapshot_normalizes_render_var_on_construction() -> None:
    # color_presentation.get("render_var") is untyped mapping data; the
    # snapshot owns the str coercion so field and key never disagree.
    snapshot = _snapshot(render_var=123)
    assert snapshot.render_var == "123"
    assert snapshot.key[1] == "123"


def test_snapshot_timing_marks_do_not_alias_caller_mapping() -> None:
    # The engine's live timings dict keeps mutating; a frozen snapshot that
    # already crossed threads must not see those mutations.
    live_timings = {"redraw_requested_monotonic_ns": 1}
    snapshot = _snapshot(timing_marks=live_timings)
    live_timings["redraw_requested_monotonic_ns"] = 2
    live_timings["late_mark"] = 3
    assert snapshot.timing_marks == {"redraw_requested_monotonic_ns": 1}


# --- CameraRequestMailbox ---------------------------------------------------


def test_mailbox_take_returns_and_clears() -> None:
    mailbox = CameraRequestMailbox()
    written = _snapshot()
    mailbox.write(written)
    assert mailbox.take(timeout=0) is written
    assert mailbox.take(timeout=0) is None
    diagnostics = mailbox.diagnostics()
    assert diagnostics["writes"] == 1
    assert diagnostics["takes"] == 1
    assert diagnostics["occupied"] is False


def test_mailbox_newest_write_overwrites_older_snapshot() -> None:
    mailbox = CameraRequestMailbox()
    for frame in range(5):
        mailbox.write(_snapshot(timeline_frame=frame))
    newest = mailbox.take(timeout=0)
    assert newest is not None
    assert newest.timeline_frame == 4
    assert mailbox.take(timeout=0) is None
    diagnostics = mailbox.diagnostics()
    assert diagnostics["writes"] == 5
    assert diagnostics["overwrites"] == 4
    assert diagnostics["takes"] == 1


def test_mailbox_take_parks_until_write_wakes_it() -> None:
    mailbox = CameraRequestMailbox()
    taken: list[ViewSnapshot | None] = []
    started = threading.Event()

    def _take() -> None:
        started.set()
        taken.append(mailbox.take(timeout=WAIT_S))

    taker = threading.Thread(target=_take, daemon=True)
    taker.start()
    assert started.wait(WAIT_S)
    written = _snapshot()
    mailbox.write(written)
    taker.join(WAIT_S)
    assert not taker.is_alive()
    assert taken == [written]


def test_mailbox_take_times_out_empty() -> None:
    mailbox = CameraRequestMailbox()
    started = time.perf_counter()
    assert mailbox.take(timeout=0.05) is None
    assert time.perf_counter() - started < WAIT_S


def test_mailbox_wake_interrupts_parked_take_without_snapshot() -> None:
    mailbox = CameraRequestMailbox()
    taken: list = ["sentinel"]
    started = threading.Event()

    def _take() -> None:
        started.set()
        taken[0] = mailbox.take(timeout=WAIT_S)

    taker = threading.Thread(target=_take, daemon=True)
    taker.start()
    assert started.wait(WAIT_S)
    mailbox.wake()
    taker.join(WAIT_S)
    assert not taker.is_alive()
    assert taken[0] is None
    diagnostics = mailbox.diagnostics()
    assert diagnostics["wakes"] == 1
    assert diagnostics["takes"] == 0


def test_mailbox_wake_latches_for_the_next_take() -> None:
    # No lost wakeups: a wake with no waiting taker makes the next take
    # return immediately, and the latch is one-shot.
    mailbox = CameraRequestMailbox()
    mailbox.wake()
    started = time.perf_counter()
    assert mailbox.take(timeout=WAIT_S) is None
    assert time.perf_counter() - started < 1.0
    written = _snapshot()
    mailbox.write(written)
    assert mailbox.take(timeout=0) is written


def test_request_from_snapshot_overlays_view_fields_on_base_request() -> None:
    base_request = RenderRequest(
        input_usd_path="/tmp/scene.usda",
        current_scene_generation=True,
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        worker_command="worker",
        native_client_module="client",
        color_presentation={"render_var": "LdrColor", "display_gamma": 2.2},
    )
    snapshot = _snapshot(
        min_samples=2,
        max_samples=32,
        render_var="HdrColor",
        selected_sensor_paths=("/World/Sensors/Camera",),
        timeline_controls_enabled=True,
        timeline_playing=True,
        timeline_frame=7,
        timeline_start=2,
        timeline_end=90,
        simulation_reset_token=3,
    )

    request = request_from_snapshot(base_request, snapshot)

    # Snapshot view fields win.
    assert (request.width, request.height) == (960, 540)
    assert (request.min_samples, request.max_samples) == (2, 32)
    assert request.camera_prim_path == "/World/OVRTXCamera"
    assert request.camera_matrix == snapshot.camera_matrix
    assert request.selected_sensor_paths == ("/World/Sensors/Camera",)
    assert request.timeline_controls_enabled is True
    assert request.timeline_playing is True
    assert (request.timeline_frame, request.timeline_start, request.timeline_end) == (7, 2, 90)
    assert request.simulation_reset_token == 3
    assert request.color_presentation["render_var"] == "HdrColor"
    # Base session identity survives untouched.
    assert request.input_usd_path == "/tmp/scene.usda"
    assert request.current_scene_generation
    assert request.worker_command == "worker"
    assert request.sensor_paths == ("/Render/Product",)
    assert request.color_presentation["display_gamma"] == 2.2
    assert base_request.color_presentation["render_var"] == "LdrColor"
    # Snapshot key identity round-trips through the rebuilt request.
    assert snapshot_from_render_request(request).key == snapshot.key


def test_mailbox_peek_does_not_clear() -> None:
    mailbox = CameraRequestMailbox()
    assert mailbox.peek() is None
    written = _snapshot()
    mailbox.write(written)
    assert mailbox.peek() is written
    assert mailbox.peek() is written
    assert mailbox.take(timeout=0) is written


def test_mailbox_rejects_non_snapshot_writes() -> None:
    mailbox = CameraRequestMailbox()
    with pytest.raises(TypeError):
        mailbox.write(object())  # type: ignore[arg-type]


def test_mailbox_concurrent_stress_taker_only_sees_newer_snapshots() -> None:
    mailbox = CameraRequestMailbox()
    total_writes = 400
    observed: list[int] = []
    writer_done = threading.Event()

    def _writer() -> None:
        for frame in range(total_writes):
            mailbox.write(_snapshot(timeline_frame=frame))
        writer_done.set()

    def _taker() -> None:
        while True:
            snapshot = mailbox.take(timeout=0.05)
            if snapshot is not None:
                observed.append(snapshot.timeline_frame)
                if snapshot.timeline_frame == total_writes - 1:
                    return
            elif writer_done.is_set() and mailbox.peek() is None:
                return

    writer = threading.Thread(target=_writer, daemon=True)
    taker = threading.Thread(target=_taker, daemon=True)
    taker.start()
    writer.start()
    writer.join(WAIT_S)
    taker.join(WAIT_S)
    assert not writer.is_alive() and not taker.is_alive()
    # Latest-wins: every observation is strictly newer than the previous
    # (dropped intermediates are fine; going backwards never is), and the
    # newest snapshot is the one the render thread ends on.
    assert observed == sorted(set(observed))
    assert observed[-1] == total_writes - 1
    diagnostics = mailbox.diagnostics()
    assert diagnostics["writes"] == total_writes
    assert diagnostics["takes"] == len(observed)
    assert diagnostics["overwrites"] == total_writes - len(observed)


# --- FrameState -------------------------------------------------------------


def test_frame_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        _frame(status="mystery")


def test_frame_status_requires_result_and_snapshot_key() -> None:
    with pytest.raises(ValueError):
        _frame(render_result=None)
    with pytest.raises(ValueError):
        _frame(snapshot_key=None)


def test_non_frame_statuses_carry_no_render_result() -> None:
    resync = FrameState(
        status=FRAME_STATUS_RESYNCING, detail="session_replacement"
    )
    failed = FrameState(status=FRAME_STATUS_FAILED, detail="worker_exited")
    assert resync.render_result is None
    assert failed.detail == "worker_exited"


def test_frame_state_is_frozen() -> None:
    frame = _frame()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.completed_samples = 99


def test_frame_state_timing_marks_do_not_alias_caller_mapping() -> None:
    live_timings = {"rgba_available_monotonic_ns": 1}
    frame = _frame(timing_marks=live_timings)
    live_timings["rgba_available_monotonic_ns"] = 2
    assert frame.timing_marks == {"rgba_available_monotonic_ns": 1}


# --- LatestFrameSlot --------------------------------------------------------


def test_slot_publish_stamps_monotonic_index_and_time() -> None:
    slot = LatestFrameSlot()
    assert slot.peek_latest() is None
    assert slot.latest_index() == 0
    before = time.perf_counter_ns()
    first = slot.publish(_frame(1))
    second = slot.publish(_frame(2))
    after = time.perf_counter_ns()
    assert (first.publication_index, second.publication_index) == (1, 2)
    assert before <= first.published_monotonic_ns <= second.published_monotonic_ns <= after
    assert slot.latest_index() == 2


def test_slot_publish_overrides_caller_supplied_stamps() -> None:
    slot = LatestFrameSlot()
    stamped = slot.publish(
        _frame(1, publication_index=777, published_monotonic_ns=777)
    )
    assert stamped.publication_index == 1
    assert stamped.published_monotonic_ns != 777


def test_slot_peek_latest_is_nonblocking_and_never_clears() -> None:
    slot = LatestFrameSlot()
    stamped = slot.publish(_frame(3))
    assert slot.peek_latest() is stamped
    assert slot.peek_latest() is stamped
    newer = slot.publish(_frame(4))
    assert slot.peek_latest() is newer


def test_slot_passes_render_result_bytes_by_reference() -> None:
    slot = LatestFrameSlot()
    result = _render_result()
    stamped = slot.publish(_frame(1, render_result=result))
    assert stamped.render_result is result
    assert stamped.render_result.rgba8 is result.rgba8


def test_slot_rejects_non_frame_state_publications() -> None:
    slot = LatestFrameSlot()
    with pytest.raises(TypeError):
        slot.publish(object())  # type: ignore[arg-type]


def test_slot_wait_for_newer_wakes_on_publication_and_times_out() -> None:
    slot = LatestFrameSlot()
    assert slot.wait_for_newer(0, timeout=0.05) is None
    received: list[FrameState | None] = []
    started = threading.Event()

    def _wait() -> None:
        started.set()
        received.append(slot.wait_for_newer(0, timeout=WAIT_S))

    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    assert started.wait(WAIT_S)
    stamped = slot.publish(_frame(1))
    waiter.join(WAIT_S)
    assert not waiter.is_alive()
    assert received == [stamped]


def test_slot_resync_and_failure_flow_through_same_slot() -> None:
    slot = LatestFrameSlot()
    frame = slot.publish(_frame(8))
    resync = slot.publish(
        FrameState(status=FRAME_STATUS_RESYNCING, detail="generation_activation")
    )
    assert resync.publication_index == frame.publication_index + 1
    latest = slot.peek_latest()
    assert latest is not None
    assert latest.status == FRAME_STATUS_RESYNCING
    assert slot.diagnostics()["status"] == FRAME_STATUS_RESYNCING


def test_slot_concurrent_stress_reader_never_observes_stale_publication() -> None:
    slot = LatestFrameSlot()
    total_publications = 400
    publisher_done = threading.Event()
    reader_indices: list[int] = []
    reader_samples: list[int] = []

    def _publisher() -> None:
        for sample in range(1, total_publications + 1):
            slot.publish(_frame(sample, snapshot_key=("stress",)))
        publisher_done.set()

    def _reader() -> None:
        while True:
            latest = slot.peek_latest()
            if latest is not None:
                reader_indices.append(latest.publication_index)
                reader_samples.append(latest.completed_samples)
            if publisher_done.is_set() and (
                latest is not None
                and latest.publication_index == total_publications
            ):
                return

    reader = threading.Thread(target=_reader, daemon=True)
    publisher = threading.Thread(target=_publisher, daemon=True)
    reader.start()
    publisher.start()
    publisher.join(WAIT_S)
    reader.join(WAIT_S)
    assert not publisher.is_alive() and not reader.is_alive()
    # Monotonic publication index: reads never go backwards, and the index
    # order matches the payload order (a stale frame can never pair with a
    # newer index).
    assert reader_indices == sorted(reader_indices)
    assert reader_samples == sorted(reader_samples)
    assert reader_indices[-1] == total_publications
    assert slot.latest_index() == total_publications
