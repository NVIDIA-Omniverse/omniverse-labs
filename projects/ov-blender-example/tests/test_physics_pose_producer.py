# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
import threading
import time
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.physics_pose_producer import PhysicsPoseProducer  # noqa: E402
from ovrtx_blender_example import physics_pose_producer  # noqa: E402


BODY_PRIM = "/World/PhysicsIsland/DynamicBodies/Cube_00"


class _FakePhysicsClient:
    def __init__(self, *, incomplete: bool = False) -> None:
        self.incomplete = incomplete
        self.calls: list[tuple[int, int, int]] = []

    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.calls.append((int(start_step_count), int(steps), int(timestep_ns)))
        step_count = int(start_step_count) + int(steps)
        simulation_time_ns = step_count * int(timestep_ns)
        state = {
            "prim_path": BODY_PRIM,
            "translate": {"found": True, "x": 0.0, "y": 4.0, "z": 0.0},
        }
        if not self.incomplete:
            state["orient"] = {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0}
        return (
            [state],
            {
                "step_count": step_count,
                "simulation_time_ns": simulation_time_ns,
                "step_ms": 1.0,
                "read_ms": 0.25,
                "total_ms": 1.25,
            },
        )


class _BlockingFirstPhysicsClient(_FakePhysicsClient):
    def __init__(self) -> None:
        super().__init__()
        self.first_call_started = threading.Event()
        self.release_first_call = threading.Event()

    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if not self.calls:
            self.first_call_started.set()
            assert self.release_first_call.wait(timeout=2.0)
        return super().advance_and_read_body_states(start_step_count, steps, timestep_ns)


class _BlockingFailingFirstPhysicsClient(_BlockingFirstPhysicsClient):
    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if not self.calls:
            self.first_call_started.set()
            assert self.release_first_call.wait(timeout=2.0)
            self.calls.append((int(start_step_count), int(steps), int(timestep_ns)))
            raise RuntimeError("advance failed")
        return super().advance_and_read_body_states(start_step_count, steps, timestep_ns)


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def _producer(
    physics: _FakePhysicsClient,
    *,
    steps_per_update: int = 1,
    timestep_ns: int = 16,
    sync_lock=None,
) -> PhysicsPoseProducer:
    return PhysicsPoseProducer(
        physics_client=physics,
        body_prims=(BODY_PRIM,),
        steps_per_update=steps_per_update,
        timestep_ns=timestep_ns,
        update_interval_ns=1,
        config_fingerprint="cfg",
        simulation_id="sim",
        sync_lock=sync_lock or threading.Lock(),
        trace=lambda *_args, **_kwargs: None,
        on_error=lambda exc: (_ for _ in ()).throw(exc),
    )


def test_pose_producer_publishes_latest_complete_pose_set() -> None:
    physics = _FakePhysicsClient()
    producer = _producer(physics)
    try:
        producer.start(generation=0, max_steps_limit=2)
        _wait_for(lambda: producer.complete_count >= 2)
        handoff = producer.take_latest(generation=0, current_step_count=0)
        publication = handoff.publication
    finally:
        producer.stop(wait=True, invalidate_generation=1)

    assert publication is not None
    assert publication.sequence == 2
    assert publication.step_count == 2
    assert publication.simulation_time_ns == 32
    assert publication.poses[0].prim_path == BODY_PRIM
    assert producer.overwrite_drop_count >= 1


def test_complete_pose_publication_fires_wake_hook() -> None:
    wakes: list[int] = []
    producer = _producer(_FakePhysicsClient())
    producer.set_publication_wake_hook(lambda: wakes.append(1))
    try:
        producer.start(generation=0, max_steps_limit=1)
        _wait_for(lambda: producer.complete_count == 1)
    finally:
        producer.stop(wait=True, invalidate_generation=1)

    assert wakes == [1]


def test_60hz_physics_30hz_publication_advances_two_fixed_steps() -> None:
    physics = _FakePhysicsClient()
    producer = _producer(
        physics,
        steps_per_update=2,
        timestep_ns=16_666_666,
    )
    try:
        producer.start(generation=0, max_steps_limit=2)
        _wait_for(lambda: producer.complete_count == 1)
        publication = producer.take_latest(
            generation=0,
            current_step_count=0,
        ).publication
    finally:
        producer.stop(wait=True, invalidate_generation=1)

    assert physics.calls == [(0, 2, 16_666_666)]
    assert publication is not None
    assert publication.step_count == 2
    assert publication.simulation_time_ns == 33_333_332


def test_composition_lock_acquisition_is_nonblocking() -> None:
    class _RecordingLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.acquire_calls: list[tuple[tuple, dict]] = []

        def acquire(self, *args, **kwargs) -> bool:
            self.acquire_calls.append((args, kwargs))
            return self.lock.acquire(*args, **kwargs)

        def release(self) -> None:
            self.lock.release()

    sync_lock = _RecordingLock()
    producer = _producer(_FakePhysicsClient(), sync_lock=sync_lock)
    try:
        producer.start(generation=0, max_steps_limit=1)
        _wait_for(lambda: producer.complete_count == 1)
    finally:
        producer.stop(wait=True, invalidate_generation=1)

    assert sync_lock.acquire_calls == [((), {"blocking": False})]


def test_contended_composition_lock_skips_without_advancing() -> None:
    sync_lock = threading.Lock()
    sync_lock.acquire()
    physics = _FakePhysicsClient()
    producer = _producer(physics, sync_lock=sync_lock)
    try:
        producer.start(generation=0, max_steps_limit=1)
        _wait_for(
            lambda: producer.diagnostics()["composition_lock_skip_count"] >= 1
        )
    finally:
        producer.stop(wait=True, invalidate_generation=1)
        sync_lock.release()

    assert physics.calls == []


def test_pose_producer_rejects_incomplete_pose_sets() -> None:
    producer = _producer(_FakePhysicsClient(incomplete=True))
    try:
        producer.start(generation=0, max_steps_limit=1)
        _wait_for(lambda: producer.pose_read_incomplete_count >= 1)
        handoff = producer.take_latest(generation=0, current_step_count=0)
    finally:
        producer.stop(wait=True, invalidate_generation=1)

    assert handoff.publication is None
    assert producer.complete_count == 0


def test_pose_producer_restarts_after_nonblocking_stop() -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics)
    restart_thread = threading.Thread(
        target=lambda: producer.start(
            generation=1,
            max_steps_limit=1000,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)
        producer.stop(wait=False, invalidate_generation=1)

        restart_thread.start()
        time.sleep(0.01)
        assert restart_thread.is_alive()
        physics.release_first_call.set()
        restart_thread.join(timeout=2.0)
        _wait_for(lambda: len(physics.calls) >= 2)
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=2)

    assert not restart_thread.is_alive()
    assert physics.calls[1][0] == 1


def test_pose_producer_defers_replacement_when_join_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics)
    monkeypatch.setattr(physics_pose_producer, "_PRODUCER_JOIN_TIMEOUT_SECONDS", 0.01)
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)

        started = time.monotonic()
        producer.start(generation=1, max_steps_limit=1000)

        assert time.monotonic() - started < 1.0
        assert len(physics.calls) == 0
        assert producer._thread is not None
        assert producer._thread.is_alive()
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=2)


def test_pose_producer_restart_is_cancelled_by_error_during_join() -> None:
    physics = _BlockingFailingFirstPhysicsClient()
    producer = _producer(physics)
    producer._on_error = lambda _exc: producer.stop(wait=False, invalidate_generation=2)
    restart_thread = threading.Thread(
        target=lambda: producer.start(
            generation=1,
            max_steps_limit=1000,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)
        producer.stop(wait=False, invalidate_generation=1)

        restart_thread.start()
        time.sleep(0.01)
        assert restart_thread.is_alive()
        physics.release_first_call.set()
        restart_thread.join(timeout=2.0)
        time.sleep(0.01)
        diagnostics = producer.diagnostics()
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=3)

    assert not restart_thread.is_alive()
    assert len(physics.calls) == 1
    assert diagnostics["active"] is False


def test_generation_changing_start_replaces_live_producer() -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics)
    restart_thread = threading.Thread(
        target=lambda: producer.start(
            generation=1,
            max_steps_limit=1000,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)

        restart_thread.start()
        time.sleep(0.01)
        assert restart_thread.is_alive()
        physics.release_first_call.set()
        restart_thread.join(timeout=2.0)
        _wait_for(lambda: len(physics.calls) >= 2)
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=2)

    assert not restart_thread.is_alive()
    assert physics.calls[0][0] == 0
    assert physics.calls[1][0] == 1


def test_stop_cannot_join_thread_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _FakePhysicsClient()
    producer = _producer(physics)
    producer_thread_starting = threading.Event()
    release_thread_start = threading.Event()
    original_start = threading.Thread.start

    def blocking_start(thread: threading.Thread) -> None:
        if thread.name == "ovrtx-physics-pose-publication":
            producer_thread_starting.set()
            assert release_thread_start.wait(timeout=2.0)
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", blocking_start)
    start_caller = threading.Thread(target=lambda: producer.start(generation=0, max_steps_limit=1000))
    stop_caller = threading.Thread(target=lambda: producer.stop(wait=True, invalidate_generation=1))
    try:
        start_caller.start()
        assert producer_thread_starting.wait(timeout=2.0)
        stop_caller.start()
        time.sleep(0.01)
        assert stop_caller.is_alive()

        release_thread_start.set()
        start_caller.join(timeout=2.0)
        stop_caller.join(timeout=2.0)
    finally:
        release_thread_start.set()
        producer.stop(wait=True, invalidate_generation=2)

    assert not start_caller.is_alive()
    assert not stop_caller.is_alive()
    assert producer.diagnostics()["active"] is False


def test_latest_generation_start_wins_concurrent_restart_join() -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics)
    generation_one = threading.Thread(
        target=lambda: producer.start(
            generation=1,
            max_steps_limit=1000,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    generation_two = threading.Thread(
        target=lambda: producer.start(
            generation=2,
            max_steps_limit=1000,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)
        generation_one.start()
        time.sleep(0.01)
        generation_two.start()
        time.sleep(0.01)

        physics.release_first_call.set()
        generation_one.join(timeout=2.0)
        generation_two.join(timeout=2.0)
        _wait_for(lambda: producer.complete_count >= 1)
        handoff = producer.take_latest(generation=2, current_step_count=0)
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=3)

    assert not generation_one.is_alive()
    assert not generation_two.is_alive()
    assert handoff.publication is not None
    assert handoff.publication.generation == 2


def test_thread_start_failure_rolls_back_producer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer(_FakePhysicsClient())
    original_start = threading.Thread.start

    def failing_start(thread: threading.Thread) -> None:
        if thread.name == "ovrtx-physics-pose-publication":
            raise RuntimeError("thread resource unavailable")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)

    with pytest.raises(RuntimeError, match="thread resource unavailable"):
        producer.start(generation=0, max_steps_limit=1)

    diagnostics = producer.diagnostics()
    producer.stop(wait=True, invalidate_generation=1)
    assert diagnostics["active"] is False
    assert diagnostics["in_flight"] is False


def test_same_generation_start_updates_active_max_step_limit() -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics)
    limit_thread = threading.Thread(
        target=lambda: producer.start(
            generation=0,
            max_steps_limit=1,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)

        limit_thread.start()
        time.sleep(0.01)
        assert limit_thread.is_alive()
        physics.release_first_call.set()
        limit_thread.join(timeout=2.0)
        assert producer.diagnostics()["max_steps"] == 1
        _wait_for(lambda: producer.state().producer_completed)
        producer.stop(wait=True, invalidate_generation=1)
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=2)

    assert len(physics.calls) == 1
    assert producer.state().producer_step_count == 1


def test_live_limit_change_waits_for_in_flight_batch_boundary() -> None:
    physics = _BlockingFirstPhysicsClient()
    producer = _producer(physics, steps_per_update=8)
    limit_thread = threading.Thread(
        target=lambda: producer.start(
            generation=0,
            max_steps_limit=3,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert physics.first_call_started.wait(timeout=2.0)
        limit_thread.start()
        time.sleep(0.01)
        assert limit_thread.is_alive()

        physics.release_first_call.set()
        limit_thread.join(timeout=2.0)
        handoff = producer.take_latest(generation=0, current_step_count=0)
        producer.stop(wait=True, invalidate_generation=1)
    finally:
        physics.release_first_call.set()
        producer.stop(wait=True, invalidate_generation=2)

    assert not limit_thread.is_alive()
    assert physics.calls == [(0, 8, 16)]
    assert handoff.publication is None


def test_live_limit_change_rejects_publication_already_at_store_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _FakePhysicsClient()
    producer = _producer(physics, steps_per_update=8)
    store_entered = threading.Event()
    release_store = threading.Event()
    original_store = producer._store

    def blocking_store(publication) -> None:
        store_entered.set()
        assert release_store.wait(timeout=2.0)
        original_store(publication)

    monkeypatch.setattr(producer, "_store", blocking_store)
    limit_thread = threading.Thread(
        target=lambda: producer.start(
            generation=0,
            max_steps_limit=3,
            step_count=0,
            simulation_time_ns=0,
        )
    )
    try:
        producer.start(generation=0, max_steps_limit=1000)
        assert store_entered.wait(timeout=2.0)
        limit_thread.start()
        time.sleep(0.01)
        assert limit_thread.is_alive()

        release_store.set()
        limit_thread.join(timeout=2.0)
        handoff = producer.take_latest(generation=0, current_step_count=0)
    finally:
        release_store.set()
        producer.stop(wait=True, invalidate_generation=1)

    assert not limit_thread.is_alive()
    assert handoff.publication is None
