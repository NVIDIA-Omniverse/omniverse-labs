# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest-pose publication state for interactive OVPhysX playback."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .native_client_support import coerce_mapping_float as _mapping_float
from .native_client_support import coerce_mapping_int as _mapping_int
from .physics_pose_set import complete_physics_pose_set
from .shared_stage_composition import BodyPose
from .shared_stage_errors import SharedStageCompositionError


# Bound the join on the (daemon) producer thread. A producer blocked in a native
# call to an unreachable worker must not hang the caller -- notably Blender's
# shutdown, which stops the producer synchronously. On timeout the daemon thread
# is abandoned and reaped at interpreter exit, the same policy the viewport
# render loop uses for its worker thread.
_PRODUCER_JOIN_TIMEOUT_SECONDS = 10.0


class PhysicsPoseClient(Protocol):
    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


@dataclass(frozen=True)
class PhysicsPosePublication:
    generation: int
    config_fingerprint: str
    simulation_id: str
    simulation_time_ns: int
    sequence: int
    step_count: int
    produced_monotonic_ns: int
    produced_time_ns: int
    body_count: int
    source_authority: str
    poses: tuple[BodyPose, ...]
    step_ms: float
    read_ms: float
    total_ms: float


@dataclass(frozen=True)
class PhysicsPoseHandoff:
    publication: PhysicsPosePublication | None
    handoff_wait_ms: float
    pose_age_ms_at_apply: float
    latest_pose_lag_steps: int


@dataclass(frozen=True)
class PhysicsPoseProducerState:
    producer_completed: bool
    producer_step_count: int
    producer_simulation_time_ns: int
    in_flight: bool


class LatestPhysicsPosePublication:
    """Stores the latest complete pose publication for the render thread."""

    def __init__(self) -> None:
        self.latest: PhysicsPosePublication | None = None
        self.publication_sequence = 0
        self.applied_publication_sequence = 0
        self.complete_count = 0
        self.overwrite_drop_count = 0
        self.stale_generation_drop_count = 0
        self.first_publication_monotonic_ns = 0
        self.last_publication_monotonic_ns = 0

    @property
    def has_latest(self) -> bool:
        return self.latest is not None

    def clear_latest(self) -> None:
        self.latest = None

    def next_sequence(self) -> int:
        self.publication_sequence += 1
        return self.publication_sequence

    def take(self, current_generation: int) -> PhysicsPosePublication | None:
        publication = self.latest
        self.latest = None
        if publication is not None and publication.generation != current_generation:
            self.stale_generation_drop_count += 1
            return None
        return publication

    def store(self, publication: PhysicsPosePublication, *, current_generation: int) -> bool:
        if publication.generation != current_generation:
            self.stale_generation_drop_count += 1
            return False
        if self.latest is not None:
            self.overwrite_drop_count += 1
        self.latest = publication
        self.complete_count += 1
        if self.first_publication_monotonic_ns == 0:
            self.first_publication_monotonic_ns = publication.produced_monotonic_ns
        self.last_publication_monotonic_ns = publication.produced_monotonic_ns
        return True

    def mark_applied(self, publication: PhysicsPosePublication) -> None:
        self.applied_publication_sequence = max(
            self.applied_publication_sequence,
            publication.sequence,
        )

    def diagnostics(
        self,
        *,
        active: bool,
        in_flight: bool,
        producer_completed: bool,
        producer_step_count: int,
        producer_simulation_time_ns: int,
        max_steps: int,
    ) -> dict[str, float | int | bool]:
        duration_s = (
            (self.last_publication_monotonic_ns - self.first_publication_monotonic_ns)
            / 1_000_000_000.0
            if self.first_publication_monotonic_ns
            and self.last_publication_monotonic_ns > self.first_publication_monotonic_ns
            else 0.0
        )
        return {
            "active": active,
            "in_flight": in_flight,
            "producer_completed": producer_completed,
            "producer_step_count": producer_step_count,
            "producer_simulation_time_ns": producer_simulation_time_ns,
            "max_steps": max_steps,
            "latest_pose_publication_sequence": self.publication_sequence,
            "applied_pose_publication_sequence": self.applied_publication_sequence,
            "has_unapplied_publication": self.latest is not None,
            "unapplied_publication_sequence": self.latest.sequence if self.latest is not None else 0,
            "pose_publication_hz": self.complete_count / duration_s if duration_s > 0 else 0.0,
        }


class PhysicsPoseProducer:
    """Runs OVPhysX ahead of the render thread and publishes complete pose sets."""

    def __init__(
        self,
        *,
        physics_client: PhysicsPoseClient,
        body_prims: Sequence[str],
        steps_per_update: int,
        timestep_ns: int,
        update_interval_ns: int,
        config_fingerprint: str,
        simulation_id: str,
        sync_lock: threading.Lock,
        trace: Callable[[str], None] | Callable[..., None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.physics_client = physics_client
        self.body_prims = tuple(body_prims)
        self.steps_per_update = max(1, int(steps_per_update))
        self.timestep_ns = int(timestep_ns)
        self.update_interval_ns = int(update_interval_ns)
        self.config_fingerprint = config_fingerprint
        self.simulation_id = simulation_id
        self._sync_lock = sync_lock
        self._trace = trace
        self._on_error = on_error
        self._on_publication: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._in_flight = False
        self._playback_active = False
        self._producer_step_count = 0
        self._producer_simulation_time_ns = 0
        self._max_steps_limit = 1
        self._producer_completed = False
        self._current_generation = 0
        self._intent_epoch = 0
        self._pose_publications = LatestPhysicsPosePublication()
        self.pose_read_incomplete_count = 0
        self.composition_lock_skip_count = 0
        self.last_step_diagnostics: Mapping[str, Any] | None = None
        self.last_read_diagnostics: Mapping[str, Any] | None = None

    def set_publication_wake_hook(self, hook: Callable[[], None] | None) -> None:
        with self._lock:
            self._on_publication = hook

    @property
    def complete_count(self) -> int:
        return self._pose_publications.complete_count

    @property
    def overwrite_drop_count(self) -> int:
        return self._pose_publications.overwrite_drop_count

    @property
    def stale_generation_drop_count(self) -> int:
        return self._pose_publications.stale_generation_drop_count

    def start(
        self,
        *,
        generation: int,
        max_steps_limit: int,
        step_count: int | None = None,
        simulation_time_ns: int | None = None,
    ) -> None:
        self._start_locked(
            generation=generation,
            max_steps_limit=max_steps_limit,
            step_count=step_count,
            simulation_time_ns=simulation_time_ns,
        )

    def _start_locked(
        self,
        *,
        generation: int,
        max_steps_limit: int,
        step_count: int | None = None,
        simulation_time_ns: int | None = None,
        _expected_intent_epoch: int | None = None,
    ) -> None:
        stopping_thread: threading.Thread | None = None
        requested_generation = int(generation)
        requested_max_steps = max(1, int(max_steps_limit))
        with self._lock:
            if _expected_intent_epoch is None:
                self._intent_epoch += 1
                _expected_intent_epoch = self._intent_epoch
            elif self._intent_epoch != _expected_intent_epoch:
                return
            if self._thread is not None and self._thread.is_alive():
                generation_changed = self._current_generation != requested_generation
                limit_changed = self._max_steps_limit != requested_max_steps
                if (
                    (self._stop_event is not None and self._stop_event.is_set())
                    or generation_changed
                    or limit_changed
                ):
                    stopping_thread = self._thread
                    self._pose_publications.clear_latest()
                    if self._stop_event is not None:
                        self._playback_active = False
                        self._stop_event.set()
                else:
                    self._max_steps_limit = max(1, int(max_steps_limit))
                    if self._producer_step_count >= self._max_steps_limit:
                        self._producer_completed = True
                        self._playback_active = False
                        if self._stop_event is not None:
                            self._stop_event.set()
                    else:
                        self._playback_active = True
                    return
            if stopping_thread is None:
                self._current_generation = requested_generation
                self._max_steps_limit = requested_max_steps
                if step_count is not None:
                    self._producer_step_count = max(self._producer_step_count, int(step_count))
                if simulation_time_ns is not None:
                    self._producer_simulation_time_ns = max(
                        self._producer_simulation_time_ns,
                        int(simulation_time_ns),
                    )
                if self._producer_step_count >= self._max_steps_limit:
                    self._producer_completed = True
                    return
                stop_event = threading.Event()
                self._stop_event = stop_event
                self._playback_active = True
                self._producer_completed = False
                thread = threading.Thread(
                    target=self._publication_loop,
                    args=(requested_generation, stop_event),
                    name="ovrtx-physics-pose-publication",
                    daemon=True,
                )
                self._thread = thread
                try:
                    thread.start()
                except Exception:
                    if self._thread is thread:
                        self._thread = None
                    if self._stop_event is stop_event:
                        self._stop_event = None
                    self._playback_active = False
                    self._in_flight = False
                    raise
                return
        if stopping_thread is not None:
            if stopping_thread is threading.current_thread():
                return
            stopping_thread.join(_PRODUCER_JOIN_TIMEOUT_SECONDS)
            if stopping_thread.is_alive():
                return
            with self._lock:
                if self._intent_epoch != _expected_intent_epoch:
                    return
                joined_step_count = self._producer_step_count
                joined_simulation_time_ns = self._producer_simulation_time_ns
            self._start_locked(
                generation=generation,
                max_steps_limit=max_steps_limit,
                step_count=max(joined_step_count, int(step_count or 0)),
                simulation_time_ns=max(joined_simulation_time_ns, int(simulation_time_ns or 0)),
                _expected_intent_epoch=_expected_intent_epoch,
            )
            return

    def needs_invalidation(self, invalidate: bool) -> bool:
        if not invalidate:
            return False
        with self._lock:
            return self._playback_active or self._pose_publications.has_latest

    def stop(
        self,
        *,
        wait: bool,
        invalidate_generation: int | None = None,
    ) -> bool:
        thread: threading.Thread | None
        with self._lock:
            self._intent_epoch += 1
            if (
                not self._playback_active
                and self._thread is None
                and not self._pose_publications.has_latest
            ):
                return True
            if invalidate_generation is not None:
                self._current_generation = int(invalidate_generation)
                self._pose_publications.clear_latest()
            self._playback_active = False
            if self._stop_event is not None:
                self._stop_event.set()
            thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(_PRODUCER_JOIN_TIMEOUT_SECONDS)
        return thread is None or not thread.is_alive()

    def sync_position(self, *, step_count: int, simulation_time_ns: int) -> None:
        with self._lock:
            self._producer_step_count = int(step_count)
            self._producer_simulation_time_ns = int(simulation_time_ns)

    def adopt_position(
        self,
        *,
        current_step_count: int,
        current_simulation_time_ns: int,
    ) -> tuple[int, int] | None:
        del current_simulation_time_ns
        with self._lock:
            if self._producer_step_count <= int(current_step_count):
                return None
            return self._producer_step_count, self._producer_simulation_time_ns

    def take_latest(self, *, generation: int, current_step_count: int) -> PhysicsPoseHandoff:
        started_ns = time.perf_counter_ns()
        with self._lock:
            self._current_generation = int(generation)
            publication = self._pose_publications.take(int(generation))
            producer_step_count = self._producer_step_count
        handoff_wait_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if publication is None:
            return PhysicsPoseHandoff(
                publication=None,
                handoff_wait_ms=handoff_wait_ms,
                pose_age_ms_at_apply=0.0,
                latest_pose_lag_steps=max(0, producer_step_count - int(current_step_count)),
            )
        return PhysicsPoseHandoff(
            publication=publication,
            handoff_wait_ms=handoff_wait_ms,
            pose_age_ms_at_apply=(time.monotonic_ns() - publication.produced_monotonic_ns) / 1_000_000.0,
            latest_pose_lag_steps=max(0, producer_step_count - publication.step_count),
        )

    def mark_applied(self, publication: PhysicsPosePublication) -> int:
        with self._lock:
            self._pose_publications.mark_applied(publication)
            return max(0, self._producer_step_count - publication.step_count)

    def state(self) -> PhysicsPoseProducerState:
        with self._lock:
            return PhysicsPoseProducerState(
                producer_completed=self._producer_completed,
                producer_step_count=self._producer_step_count,
                producer_simulation_time_ns=self._producer_simulation_time_ns,
                in_flight=self._in_flight,
            )

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            diagnostics = dict(
                self._pose_publications.diagnostics(
                    active=self._playback_active,
                    in_flight=self._in_flight,
                    producer_completed=self._producer_completed,
                    producer_step_count=self._producer_step_count,
                    producer_simulation_time_ns=self._producer_simulation_time_ns,
                    max_steps=self._max_steps_limit,
                )
            )
            diagnostics["composition_lock_skip_count"] = self.composition_lock_skip_count
            return diagnostics

    def _publication_loop(self, generation: int, stop_event: threading.Event) -> None:
        next_tick_ns = time.monotonic_ns()
        try:
            while not stop_event.is_set():
                now_ns = time.monotonic_ns()
                if now_ns < next_tick_ns:
                    time.sleep((next_tick_ns - now_ns) / 1_000_000_000.0)
                    if stop_event.is_set():
                        break
                with self._lock:
                    max_steps_limit = self._max_steps_limit
                    if generation != self._current_generation:
                        break
                    if self._producer_step_count >= max_steps_limit:
                        self._producer_completed = True
                        break
                self._produce_publication(generation, max_steps_limit, stop_event)
                next_tick_ns = max(next_tick_ns + self.update_interval_ns, time.monotonic_ns())
        except Exception as exc:
            self._on_error(exc)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                if self._stop_event is stop_event:
                    self._stop_event = None
                self._in_flight = False
                self._playback_active = False

    def _produce_publication(
        self,
        generation: int,
        max_steps_limit: int,
        stop_event: threading.Event,
    ) -> None:
        if not self._sync_lock.acquire(blocking=False):
            with self._lock:
                self.composition_lock_skip_count += 1
            self._trace(
                "async_pose_publication.composition_busy",
                generation=generation,
            )
            return
        if stop_event.is_set():
            self._sync_lock.release()
            return
        started_ns = time.perf_counter_ns()
        try:
            with self._lock:
                if generation != self._current_generation or self._producer_step_count >= max_steps_limit:
                    return
                self._in_flight = True
                start_step_count = self._producer_step_count
                steps_to_run = min(self.steps_per_update, max_steps_limit - start_step_count)

            self._trace(
                "async_advance_and_read_body_states.begin",
                generation=generation,
                start_step_count=start_step_count,
                steps=steps_to_run,
            )
            states, read_diagnostics = self.physics_client.advance_and_read_body_states(
                start_step_count,
                steps_to_run,
                self.timestep_ns,
            )
            target_step_count = _mapping_int(read_diagnostics, "step_count", start_step_count + steps_to_run)
            target_simulation_time_ns = _mapping_int(
                read_diagnostics,
                "simulation_time_ns",
                target_step_count * self.timestep_ns,
            )
            step_ms = _mapping_float(read_diagnostics, "step_ms", 0.0)
            read_ms = _mapping_float(read_diagnostics, "read_ms", 0.0)
            with self._lock:
                self._producer_step_count = target_step_count
                self._producer_simulation_time_ns = target_simulation_time_ns
                if target_step_count >= max_steps_limit:
                    self._producer_completed = True
            if stop_event.is_set():
                return
            self._trace(
                "async_advance_and_read_body_states.end",
                generation=generation,
                step_count=target_step_count,
                simulation_time_ns=target_simulation_time_ns,
                state_count=len(states),
            )
            try:
                poses = complete_physics_pose_set(states, self.body_prims)
            except SharedStageCompositionError as exc:
                self.pose_read_incomplete_count += 1
                self._trace(
                    "async_pose_publication.incomplete",
                    generation=generation,
                    simulation_time_ns=target_simulation_time_ns,
                    error=str(exc),
                )
                return

            total_ms = _mapping_float(read_diagnostics, "total_ms", (time.perf_counter_ns() - started_ns) / 1_000_000.0)
            with self._lock:
                sequence = self._pose_publications.next_sequence()
            publication = PhysicsPosePublication(
                generation=generation,
                config_fingerprint=self.config_fingerprint,
                simulation_id=self.simulation_id,
                simulation_time_ns=target_simulation_time_ns,
                sequence=sequence,
                step_count=target_step_count,
                produced_monotonic_ns=time.monotonic_ns(),
                produced_time_ns=time.time_ns(),
                body_count=len(poses),
                source_authority="OVPhysX",
                poses=poses,
                step_ms=step_ms,
                read_ms=read_ms,
                total_ms=total_ms,
            )
            self.last_step_diagnostics = read_diagnostics
            self.last_read_diagnostics = read_diagnostics
            self._store(publication)
        except Exception as exc:
            stop_event.set()
            self._on_error(exc)
        finally:
            with self._lock:
                self._in_flight = False
            self._sync_lock.release()

    def _store(self, publication: PhysicsPosePublication) -> None:
        wake_hook: Callable[[], None] | None
        with self._lock:
            if self._stop_event is not None and self._stop_event.is_set():
                return
            stored = self._pose_publications.store(publication, current_generation=self._current_generation)
            if not stored:
                return
            self._producer_step_count = publication.step_count
            self._producer_simulation_time_ns = publication.simulation_time_ns
            if publication.step_count >= self._max_steps_limit:
                self._producer_completed = True
            wake_hook = self._on_publication
        self._trace(
            "async_pose_publication.complete",
            generation=publication.generation,
            sequence=publication.sequence,
            step_count=publication.step_count,
            simulation_time_ns=publication.simulation_time_ns,
            body_count=publication.body_count,
            step_ms=publication.step_ms,
            read_ms=publication.read_ms,
            total_ms=publication.total_ms,
        )
        if wake_hook is not None:
            wake_hook()


__all__ = [
    "LatestPhysicsPosePublication",
    "PhysicsPoseHandoff",
    "PhysicsPoseProducer",
    "PhysicsPoseProducerState",
    "PhysicsPosePublication",
]
