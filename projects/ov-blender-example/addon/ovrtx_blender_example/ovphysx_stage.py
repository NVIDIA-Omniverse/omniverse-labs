# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVPhysX advancement and authoritative runtime-stage publication."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .ovphysx_runtime_client import (
    UNKNOWN,
    OvphysxRuntimeClient as _OvphysxRuntimeClient,
    coerce_mapping_int as _mapping_int,
)
from .physics_pose_set import apply_initial_condition_values, complete_physics_pose_set
from .physics_pose_producer import PhysicsPoseProducer, PhysicsPosePublication
from .shared_stage_config import InteractiveSharedStageConfig as _InteractiveSharedStageConfig
from .shared_stage_composition import BodyPose as _BodyPose, BodyVelocity as _BodyVelocity, RuntimeStageHost


COMPOSITION_SYNC_LOCK = threading.Lock()


def _physics_client_from_config(config: _InteractiveSharedStageConfig, simulation_id: str) -> _OvphysxRuntimeClient:
    return _OvphysxRuntimeClient(config, simulation_id)


class OvphysxStageStatus(str, Enum):
    OK = "ok"
    COMPLETED = "completed"
    BUSY = "busy"
    FAILED = "failed"


@dataclass(frozen=True)
class OvphysxStageResult:
    status: OvphysxStageStatus
    reason: str
    pose_set: tuple[_BodyPose, ...]
    dirty_paths: tuple[str, ...]
    step_count: int
    simulation_time_ns: int
    generation: int


class OvphysxStageController:
    def __init__(
        self,
        config: _InteractiveSharedStageConfig,
        physics_client: _OvphysxRuntimeClient | None = None,
        simulation_id: str | None = None,
    ) -> None:
        self.config = config
        self.physics_simulation_id = simulation_id or f"ovphysx-blender-{os.getpid()}"
        self.composition_mode = "async_latest_pose"
        self.composition_config_fingerprint = _composition_config_fingerprint(config)
        self._stage_host = RuntimeStageHost(scene_id=config.input_usd_path)
        self._physics_client = physics_client or _physics_client_from_config(config, self.physics_simulation_id)
        self.started = False
        self.completed = False
        self.failed = False
        self.closed = False
        self.step_count = 0
        self.sync_count = 0
        self.stage_update_count = 0
        self.completed_sync_count = 0
        self.busy_sync_count = 0
        self._sync_in_progress = False
        self.current_simulation_time_ns = 0
        self.changed_body_paths: set[str] = set()
        self.last_update: dict[str, Any] | None = None
        self.create_diagnostics: Mapping[str, Any] | None = None
        self.last_read_diagnostics: Mapping[str, Any] | None = None
        self.last_step_diagnostics: Mapping[str, Any] | None = None
        self.last_initial_condition_write: Mapping[str, Any] | None = None
        self.last_body_velocity_write: Mapping[str, Any] | None = None
        self.last_error = ""
        self.max_steps_limit = config.max_steps
        self.composition_generation = 0
        self.playback_intent_generation = 0
        self.same_pose_reuse_count = 0
        self.last_pose_handoff_wait_ms = 0.0
        self.last_pose_age_ms_at_publish = 0.0
        self.last_latest_pose_lag_steps = 0
        self._runtime_transition_lock = threading.RLock()
        self._async_failure_lock = threading.RLock()
        self._pose_producer = PhysicsPoseProducer(
            physics_client=self._physics_client,
            body_prims=self.config.body_prims,
            steps_per_update=self.config.steps_per_update,
            timestep_ns=self.config.timestep_ns,
            update_interval_ns=self.config.update_interval_ns,
            config_fingerprint=self.composition_config_fingerprint,
            simulation_id=self.physics_simulation_id,
            sync_lock=COMPOSITION_SYNC_LOCK,
            trace=self._trace,
            on_error=self._record_async_error,
        )

    @property
    def pose_publication_complete_count(self) -> int:
        return self._pose_producer.complete_count

    @property
    def pose_publication_overwrite_drop_count(self) -> int:
        return self._pose_producer.overwrite_drop_count

    @property
    def stale_generation_drop_count(self) -> int:
        return self._pose_producer.stale_generation_drop_count

    @property
    def pose_read_incomplete_count(self) -> int:
        return self._pose_producer.pose_read_incomplete_count

    def set_pose_publication_wake_hook(self, hook: Callable[[], None] | None) -> None:
        self._pose_producer.set_publication_wake_hook(hook)

    def physics_pose_set(self, simulation_time_ns: int) -> tuple[_BodyPose, ...]:
        """Return the complete current pose set, or empty before it is available."""

        if not self.started:
            return ()
        mutation = self._stage_host.last_mutation
        if mutation is None or mutation.simulation_time_ns != int(simulation_time_ns):
            return ()
        try:
            return self._stage_host.body_poses_for(self.config.body_prims)
        except KeyError:
            return ()

    def tick(
        self,
        *,
        max_steps: int | None = None,
        initial_condition_values: Sequence[_BodyPose] = (),
    ) -> OvphysxStageResult:
        with self._runtime_transition_lock:
            if self.closed:
                update = self._record_update(
                    "closed", False, [], "composition_shutdown", extra={"closed": True, "failed": True}
                )
            else:
                self._stop_async_playback(wait=True, invalidate=True)
                self.adopt_async_producer_position()
                update = self._tick_update(
                    max_steps=max_steps,
                    initial_condition_values=initial_condition_values,
                )
            return self._stage_result(update)

    def _tick_update(
        self,
        *,
        max_steps: int | None = None,
        initial_condition_values: Sequence[_BodyPose] = (),
    ) -> dict[str, Any]:
        max_steps_limit = max(1, int(max_steps if max_steps is not None else self.config.max_steps))
        self.max_steps_limit = max_steps_limit
        if self._sync_in_progress:
            return self._record_update("busy", False, [], "in_progress")
        if not COMPOSITION_SYNC_LOCK.acquire(blocking=False):
            return self._record_update("busy", False, [], "in_progress")
        self._sync_in_progress = True
        try:
            if self.failed:
                return self._record_update("failed", False, [], "composition_failed")
            if not self.started:
                return self._start(initial_condition_values=initial_condition_values)
            if self.completed and self.step_count < max_steps_limit:
                self.completed = False
            if self.completed:
                return self._record_update("completed", False, [], "max_steps")
            if self.step_count >= max_steps_limit:
                self.completed = True
                return self._record_update("completed", False, [], "max_steps")
            steps_to_run = min(self.config.steps_per_update, max_steps_limit - self.step_count)
            self._trace(
                "physics_advance_and_read.begin",
                start_step_count=self.step_count,
                steps=steps_to_run,
            )
            try:
                states, read_diagnostics = self._physics_client.advance_and_read_body_states(
                    self.step_count,
                    steps_to_run,
                    self.config.timestep_ns,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.failed = True
                self.completed = True
                self.last_error = message
                return self._record_update(
                    "step",
                    False,
                    [],
                    "physics_step_error",
                    extra={"failed": True, "physics_error": message},
                )
            self.step_count = _mapping_int(read_diagnostics, "step_count", self.step_count + steps_to_run)
            self.current_simulation_time_ns = _mapping_int(
                read_diagnostics,
                "simulation_time_ns",
                self.step_count * self.config.timestep_ns,
            )
            self.last_step_diagnostics = read_diagnostics
            self.last_read_diagnostics = read_diagnostics
            self._trace(
                "physics_advance_and_read.end",
                step_count=self.step_count,
                simulation_time_ns=self.current_simulation_time_ns,
                state_count=len(states),
            )
            should_complete = self.step_count >= max_steps_limit
            if should_complete:
                self.completed = True
            return self._publish_poses(states, "step")
        finally:
            self._sync_in_progress = False
            COMPOSITION_SYNC_LOCK.release()

    def publish_latest_pose(
        self,
        *,
        max_steps: int | None = None,
    ) -> OvphysxStageResult:
        with self._runtime_transition_lock:
            if self.closed:
                update = self._record_update(
                    "closed", False, [], "composition_shutdown", extra={"closed": True, "failed": True}
                )
            else:
                update = self._publish_latest_pose_update(max_steps=max_steps)
            return self._stage_result(update)

    def _publish_latest_pose_update(
        self,
        *,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        max_steps_limit = max(1, int(max_steps if max_steps is not None else self.config.max_steps))
        self.max_steps_limit = max_steps_limit
        if not self.started:
            update = self._tick_update(
                max_steps=max_steps_limit,
            )
            self.sync_async_producer_position()
            if not self.failed and not self.completed:
                self.start_async_playback(max_steps_limit=max_steps_limit)
            if self.failed:
                if update.get("failed"):
                    return update
                update = {
                    **update,
                    "reason": "failed",
                    "failed": True,
                    "skipped_reason": "composition_failed",
                }
                self.last_update = update
            return update
        if self.failed:
            self.stop_async_playback(wait=True, invalidate=True)
            self.adopt_async_producer_position()
            return self._record_update("failed", False, [], "composition_failed")
        if self.completed and self.step_count < max_steps_limit:
            self.completed = False
        if self.completed:
            self.stop_async_playback(wait=True, invalidate=False)
            self.adopt_async_producer_position()
            return self._record_update("completed", False, [], "max_steps")

        publication, handoff_wait_ms = self._take_latest_pose_publication()
        if publication is not None and publication.step_count < self.step_count:
            publication = None
        if publication is not None and publication.step_count > max_steps_limit:
            self.completed = True
            self.stop_async_playback(wait=True, invalidate=False)
            self.adopt_async_producer_position()
            return self._record_update(
                "completed",
                False,
                [],
                "max_steps",
                extra={"dropped_publication_step_count": publication.step_count},
            )
        failed_update: dict[str, Any] | None = None
        completed_update: dict[str, Any] | None = None
        with self._async_failure_lock:
            if self.failed:
                self.stop_async_playback(wait=False, invalidate=True)
                failed_update = self._record_update("failed", False, [], "composition_failed")
            elif publication is not None:
                self.step_count = publication.step_count
                self.current_simulation_time_ns = publication.simulation_time_ns
                self._sync_pose_producer_diagnostics()
                update = self._publish_poses(
                    (),
                    "async_latest_pose",
                    publication=publication,
                    pose_handoff_wait_ms=handoff_wait_ms,
                )
                if not self.failed:
                    self.last_latest_pose_lag_steps = self._pose_producer.mark_applied(publication)
                if self.failed:
                    self.stop_async_playback(wait=False, invalidate=True)
                    update["composition_generation"] = self.composition_generation
                    update["playback_intent_generation"] = self.playback_intent_generation
                    failed_update = update
                elif publication.step_count >= max_steps_limit:
                    self.completed = True
                    self.stop_async_playback(wait=False, invalidate=False)
                    completed_update = update
                if failed_update is None and completed_update is None:
                    return update

        if failed_update is not None:
            self.stop_async_playback(wait=True, invalidate=False)
            self.adopt_async_producer_position()
            failed_update["step_count"] = self.step_count
            failed_update["simulation_time_ns"] = self.current_simulation_time_ns
            failed_update["composition_generation"] = self.composition_generation
            failed_update["playback_intent_generation"] = self.playback_intent_generation
            self.last_update = failed_update
            return failed_update

        if completed_update is not None:
            self.stop_async_playback(wait=True, invalidate=False)
            self.adopt_async_producer_position()
            completed_update["step_count"] = self.step_count
            completed_update["simulation_time_ns"] = self.current_simulation_time_ns
            self.last_update = completed_update
            return completed_update

        producer_state = self._pose_producer.state()
        if not (
            producer_state.producer_completed
            and producer_state.producer_step_count >= max_steps_limit
        ):
            self.start_async_playback(max_steps_limit=max_steps_limit)
            producer_state = self._pose_producer.state()
        failed_update = None
        completed_update = None
        with self._async_failure_lock:
            if self.failed:
                self.stop_async_playback(wait=False, invalidate=True)
                failed_update = self._record_update("failed", False, [], "composition_failed")
            elif self.completed:
                self.stop_async_playback(wait=False, invalidate=False)
                completed_update = self._record_update("completed", False, [], "max_steps")
            elif (
                producer_state.producer_completed
                and producer_state.producer_step_count >= max_steps_limit
            ):
                self.completed = True
                self.stop_async_playback(wait=False, invalidate=False)
                completed_update = self._record_update("completed", False, [], "max_steps")
            else:
                self.same_pose_reuse_count += 1
                return self._record_update(
                    "async_reuse",
                    False,
                    [],
                    "no_new_pose_publication",
                    extra={
                        "pose_handoff_wait_ms": handoff_wait_ms,
                        "latest_pose_lag_steps": max(0, producer_state.producer_step_count - self.step_count),
                    },
                )

        self.stop_async_playback(wait=True, invalidate=False)
        self.adopt_async_producer_position()
        terminal_update = failed_update or completed_update
        terminal_update["step_count"] = self.step_count
        terminal_update["simulation_time_ns"] = self.current_simulation_time_ns
        terminal_update["composition_generation"] = self.composition_generation
        terminal_update["playback_intent_generation"] = self.playback_intent_generation
        self.last_update = terminal_update
        return terminal_update

    def start_async_playback(self, *, max_steps_limit: int | None = None) -> None:
        with self._runtime_transition_lock:
            try:
                self._start_async_playback(max_steps_limit=max_steps_limit)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.failed = True
                self.completed = True
                self.last_error = message
                self._stop_async_playback(wait=False, invalidate=True)
                self._trace("async_playback_start.error", error=message)

    def _start_async_playback(self, *, max_steps_limit: int | None = None) -> None:
        requested_max_steps = max(
            1,
            int(self.config.max_steps if max_steps_limit is None else max_steps_limit),
        )
        self.max_steps_limit = requested_max_steps
        if not self.started or self.closed or self.failed:
            return
        producer_state = self._pose_producer.state()
        if self.step_count >= requested_max_steps or (
            producer_state.producer_completed
            and producer_state.producer_step_count >= requested_max_steps
        ):
            self._stop_async_playback(wait=True, invalidate=False)
            self.adopt_async_producer_position()
            self.completed = self.step_count >= requested_max_steps
            return
        self.completed = False
        self._pose_producer.start(
            generation=self.composition_generation,
            max_steps_limit=requested_max_steps,
            step_count=self.step_count,
            simulation_time_ns=self.current_simulation_time_ns,
        )
        producer_state = self._pose_producer.state()
        if producer_state.producer_completed:
            self.adopt_async_producer_position()
            self.completed = self.step_count >= requested_max_steps

    def stop_async_playback(self, *, wait: bool, invalidate: bool = True) -> bool:
        with self._runtime_transition_lock:
            return self._stop_async_playback(wait=wait, invalidate=invalidate)

    def _stop_async_playback(self, *, wait: bool, invalidate: bool = True) -> bool:
        invalidate_generation = None
        if self._pose_producer.needs_invalidation(invalidate):
            self.playback_intent_generation += 1
            self.composition_generation += 1
            invalidate_generation = self.composition_generation
        return self._pose_producer.stop(
            wait=wait, invalidate_generation=invalidate_generation
        )

    def sync_async_producer_position(self) -> None:
        self._pose_producer.sync_position(
            step_count=self.step_count,
            simulation_time_ns=self.current_simulation_time_ns,
        )

    def adopt_async_producer_position(self) -> None:
        position = self._pose_producer.adopt_position(
            current_step_count=self.step_count,
            current_simulation_time_ns=self.current_simulation_time_ns,
        )
        if position is None:
            return
        self.step_count, self.current_simulation_time_ns = position

    def _failed_initial_condition_edit_update(self) -> dict[str, Any]:
        update = {
            "reason": "initial_condition_value_edit",
            "sim_value_write_applied": False,
            "physics_generation_reset": False,
            "stage_updated": False,
            "dirty_paths": [],
            "failed": True,
            "skipped_reason": "composition_failed",
            "result": (
                {"status": "error", "error": self.last_error}
                if self.last_error
                else {}
            ),
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "mutation_authority": "Blender initial-condition edit",
        }
        self.last_update = update
        return update

    def apply_initial_condition_values(
        self,
        poses: Sequence[_BodyPose],
        *,
        reset: bool = False,
    ) -> OvphysxStageResult:
        with self._runtime_transition_lock:
            if self.closed:
                update = {
                    "reason": "initial_condition_value_edit",
                    "sim_value_write_applied": False,
                    "physics_generation_reset": False,
                    "stage_updated": False,
                    "failed": True,
                    "closed": True,
                    "skipped_reason": "composition_shutdown",
                    "composition_generation": self.composition_generation,
                    "playback_intent_generation": self.playback_intent_generation,
                    "mutation_authority": "Blender initial-condition edit",
                }
                self.last_update = update
            else:
                update = self._apply_initial_condition_values(poses, reset=reset)
            return self._stage_result(update)

    def _apply_initial_condition_values(
        self,
        poses: Sequence[_BodyPose],
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        if self.failed:
            self.stop_async_playback(wait=True, invalidate=True)
            self.adopt_async_producer_position()
            return self._failed_initial_condition_edit_update()
        if not poses:
            return {
                "reason": "initial_condition_value_edit",
                "sim_value_write_applied": False,
                "skipped_reason": "no_initial_condition_value_edits",
            }
        if error := _initial_condition_values_error(poses):
            self.stop_async_playback(wait=True, invalidate=True)
            self.failed = True
            self.completed = True
            self.last_error = error
            self.last_initial_condition_write = {"status": "error", "error": error}
            update = {
                "reason": "initial_condition_value_edit",
                "sim_value_write_applied": False,
                "value_requested_count": len(poses),
                "sim_value_paths": [pose.prim_path for pose in poses],
                "physics_reset": bool(reset),
                "physics_generation_reset": False,
                "stage_updated": False,
                "dirty_paths": [],
                "failed": True,
                "skipped_reason": "initial_condition_value_validation_error",
                "result": dict(self.last_initial_condition_write),
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
                "mutation_authority": "Blender initial-condition edit",
            }
            self.last_update = update
            return update
        previous_playback_intent_generation = self.playback_intent_generation
        self.stop_async_playback(wait=True, invalidate=True)
        if self.failed:
            return self._failed_initial_condition_edit_update()
        simulation_time_ns = max(1, int(self.current_simulation_time_ns) + 1)
        self._trace(
            "write_body_poses.begin",
            simulation_time_ns=simulation_time_ns,
            body_count=len(poses),
            reset=reset,
        )
        try:
            write_diagnostics = self._physics_client.write_body_poses(
                poses,
                simulation_time_ns=simulation_time_ns,
                reset=reset,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.completed = True
            self.last_error = message
            self.last_initial_condition_write = {"status": "error", "error": message}
            update = {
                "reason": "initial_condition_value_edit",
                "sim_value_write_applied": False,
                "value_requested_count": len(poses),
                "sim_value_paths": [pose.prim_path for pose in poses],
                "physics_reset": bool(reset),
                "physics_generation_reset": False,
                "stage_updated": False,
                "dirty_paths": [],
                "failed": True,
                "skipped_reason": "initial_condition_value_write_error",
                "result": dict(self.last_initial_condition_write),
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
                "mutation_authority": "Blender initial-condition edit",
            }
            self.last_update = update
            return update
        if error := _physics_write_error(write_diagnostics, len(poses)):
            self.failed = True
            self.completed = True
            self.last_error = error
            self.last_initial_condition_write = {**dict(write_diagnostics), "status": "error", "error": error}
            update = {
                "reason": "initial_condition_value_edit",
                "sim_value_write_applied": False,
                "sim_value_write": dict(write_diagnostics),
                "value_requested_count": len(poses),
                "sim_value_paths": [pose.prim_path for pose in poses],
                "physics_reset": bool(reset),
                "physics_generation_reset": False,
                "stage_updated": False,
                "dirty_paths": [],
                "failed": True,
                "skipped_reason": "initial_condition_value_write_error",
                "result": dict(self.last_initial_condition_write),
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
                "mutation_authority": "Blender initial-condition edit",
            }
            self.last_update = update
            return update
        self.last_initial_condition_write = dict(write_diagnostics)
        self._trace(
            "write_body_poses.end",
            simulation_time_ns=simulation_time_ns,
            body_count=len(poses),
            reset=reset,
        )
        self.current_simulation_time_ns = simulation_time_ns
        mutation = self._stage_host.publish_ovphysx_poses(poses, simulation_time_ns)
        dirty_paths = list(mutation.dirty_paths)
        self.changed_body_paths.update(dirty_paths)
        self.completed = False
        self.step_count = 0
        if self.playback_intent_generation == previous_playback_intent_generation:
            self.playback_intent_generation += 1
        self.sync_async_producer_position()
        self.composition_generation += 1
        update = {
            "reason": "initial_condition_value_edit",
            "sim_value_write_applied": True,
            "sim_value_write": dict(write_diagnostics),
            "value_requested_count": len(poses),
            "sim_value_paths": [pose.prim_path for pose in poses],
            "physics_reset": bool(reset),
            "physics_generation_reset": True,
            "step_count": self.step_count,
            "simulation_time_ns": simulation_time_ns,
            "stage_updated": bool(dirty_paths),
            "dirty_paths": dirty_paths,
            "failed": False,
            "skipped_reason": "",
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "mutation_authority": "Blender initial-condition edit",
        }
        self.last_update = update
        if dirty_paths:
            self.stage_update_count += 1
        return update

    def apply_body_velocity_edits(
        self,
        velocities: Sequence[_BodyVelocity],
        *,
        reset: bool = False,
    ) -> OvphysxStageResult:
        with self._runtime_transition_lock:
            update = self._apply_body_velocity_edits(velocities, reset=reset)
            return self._stage_result(update)

    def _apply_body_velocity_edits(
        self,
        velocities: Sequence[_BodyVelocity],
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        paths = [velocity.prim_path for velocity in velocities]
        base = {
            "reason": "body_velocity_edit",
            "velocity_requested_count": len(velocities),
            "sim_value_paths": paths,
            "physics_reset": bool(reset),
            "physics_generation_reset": False,
            "stage_updated": False,
            "dirty_paths": [],
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "mutation_authority": "Blender body-velocity edit",
        }
        validation_error = ""
        if self.closed:
            validation_error = "composition_shutdown"
        elif self.failed:
            validation_error = "composition_failed"
        elif not self.started:
            validation_error = "physics_simulation_not_started"
        elif self.step_count >= self.max_steps_limit:
            validation_error = "no_remaining_physics_step_budget"
        elif not velocities:
            validation_error = "no_body_velocity_edits"
        elif len(set(paths)) != len(paths):
            validation_error = "body_velocity_paths_not_unique"
        else:
            unknown = sorted(set(paths).difference(self.config.body_prims))
            if unknown:
                validation_error = f"unknown_dynamic_body_paths: {', '.join(unknown)}"
        if validation_error:
            update = {
                **base,
                "sim_value_write_applied": False,
                "failed": validation_error != "no_body_velocity_edits",
                "skipped_reason": validation_error,
            }
            self.last_update = update
            return update

        previous_playback_generation = self.playback_intent_generation
        self.stop_async_playback(wait=True, invalidate=True)
        self.adopt_async_producer_position()
        if self.step_count >= self.max_steps_limit:
            update = {
                **base,
                "sim_value_write_applied": False,
                "failed": True,
                "skipped_reason": "no_remaining_physics_step_budget",
                "step_count": self.step_count,
                "simulation_time_ns": self.current_simulation_time_ns,
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
            }
            self.last_update = update
            return update
        simulation_time_ns = max(1, int(self.current_simulation_time_ns) + 1)
        self._trace(
            "write_body_velocities.begin",
            simulation_time_ns=simulation_time_ns,
            body_count=len(velocities),
            reset=reset,
        )
        try:
            write_diagnostics = self._physics_client.write_body_velocities(
                velocities,
                simulation_time_ns=simulation_time_ns,
                reset=reset,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.completed = True
            self.last_error = message
            self.last_body_velocity_write = {"status": "error", "error": message}
            update = {
                **base,
                "sim_value_write_applied": False,
                "failed": True,
                "skipped_reason": "body_velocity_write_error",
                "result": dict(self.last_body_velocity_write),
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
            }
            self.last_update = update
            return update
        if error := _physics_write_error(write_diagnostics, len(velocities)):
            self.failed = True
            self.completed = True
            self.last_error = error
            self.last_body_velocity_write = {**dict(write_diagnostics), "status": "error", "error": error}
            update = {
                **base,
                "sim_value_write_applied": False,
                "failed": True,
                "skipped_reason": "body_velocity_write_error",
                "result": dict(self.last_body_velocity_write),
                "composition_generation": self.composition_generation,
                "playback_intent_generation": self.playback_intent_generation,
            }
            self.last_update = update
            return update

        self.last_body_velocity_write = dict(write_diagnostics)
        self.current_simulation_time_ns = simulation_time_ns
        self.completed = False
        if self.playback_intent_generation == previous_playback_generation:
            self.playback_intent_generation += 1
        self.composition_generation += 1
        self.sync_async_producer_position()
        self._trace(
            "write_body_velocities.end",
            simulation_time_ns=simulation_time_ns,
            body_count=len(velocities),
            reset=reset,
        )
        update = {
            **base,
            "sim_value_write_applied": True,
            "sim_value_write": dict(write_diagnostics),
            "simulation_time_ns": simulation_time_ns,
            "step_count": self.step_count,
            "failed": False,
            "skipped_reason": "",
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "velocities": [
                {
                    "prim_path": velocity.prim_path,
                    "linear": list(velocity.linear),
                    "angular": list(velocity.angular),
                }
                for velocity in velocities
            ],
        }
        self.last_update = update
        return update

    def _take_latest_pose_publication(self) -> tuple[PhysicsPosePublication | None, float]:
        handoff = self._pose_producer.take_latest(
            generation=self.composition_generation,
            current_step_count=self.step_count,
        )
        self.last_pose_handoff_wait_ms = handoff.handoff_wait_ms
        self.last_pose_age_ms_at_publish = handoff.pose_age_ms_at_apply
        self.last_latest_pose_lag_steps = handoff.latest_pose_lag_steps
        return handoff.publication, handoff.handoff_wait_ms

    def _record_async_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._async_failure_lock:
            self.failed = True
            self.completed = True
            self.last_error = message
            self._stop_async_playback(wait=False, invalidate=True)
            self._trace("async_pose_publication.error", error=message)

    def deactivate(self) -> str:
        with self._runtime_transition_lock:
            return self._shutdown()

    def shutdown(self) -> str:
        return self.deactivate()

    def _shutdown(self) -> str:
        if self.closed:
            return "not_found"
        if not self.stop_async_playback(wait=True, invalidate=True):
            return "failed"
        self.adopt_async_producer_position()
        status = str(self._physics_client.shutdown() or "stopped")
        if status == "failed":
            return status
        self.closed = True
        self.completed = True
        return status

    def diagnostics(self) -> dict[str, Any]:
        async_snapshot = self._async_diagnostics_snapshot()
        return {
            "schema_version": 1,
            "artifact_id": "interactive-shared-stage-composition",
            "status": "failed" if self.failed else "complete" if self.completed else "running" if self.started else "not-started",
            "closed": self.closed,
            "composition_mode": self.composition_mode,
            "composition_config_fingerprint": self.composition_config_fingerprint,
            "physics_client_transport": _physics_client_transport(self._physics_client),
            "topology": {
                "stage_host": "demo-local Python RuntimeStageHost",
                "physics_pose_producer": "demo-local playback/runtime publication source",
                "physics_worker": _physics_worker_description(self._physics_client),
                "physics_client": _physics_client_description(self._physics_client),
            },
            "input_usd_path": self.config.input_usd_path,
            "body_root": self.config.body_root,
            "body_prims": list(self.config.body_prims),
            "physics_simulation_id": self.physics_simulation_id,
            "sync_count": self.sync_count,
            "step_count": self.step_count,
            "max_steps": self.max_steps_limit,
            "configured_max_steps": self.config.max_steps,
            "steps_per_update": self.config.steps_per_update,
            "stage_update_count": self.stage_update_count,
            "completed_sync_count": self.completed_sync_count,
            "busy_sync_count": self.busy_sync_count,
            "current_simulation_time_ns": self.current_simulation_time_ns,
            "stage_host": self._stage_host.diagnostics(),
            "pose_handoff_wait_ms": self.last_pose_handoff_wait_ms,
            "pose_age_ms_at_publish": self.last_pose_age_ms_at_publish,
            "latest_pose_lag_steps": self.last_latest_pose_lag_steps,
            "latest_pose_publication_sequence": async_snapshot["latest_pose_publication_sequence"],
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "pose_publication_complete_count": self.pose_publication_complete_count,
            "pose_read_incomplete_count": self.pose_read_incomplete_count,
            "pose_publication_overwrite_drop_count": self.pose_publication_overwrite_drop_count,
            "stale_generation_drop_count": self.stale_generation_drop_count,
            "same_pose_reuse_count": self.same_pose_reuse_count,
            "pose_publication_hz": async_snapshot["pose_publication_hz"],
            "physics_step_hz": self.config.physics_fps,
            "async_publication": async_snapshot,
            "changed_body_paths": sorted(self.changed_body_paths),
            "last_update": self.last_update,
            "last_initial_condition_write": self.last_initial_condition_write,
            "last_body_velocity_write": self.last_body_velocity_write,
            "last_physics_step_diagnostics": _physics_diagnostics_summary(self.last_step_diagnostics),
            "last_physics_read_diagnostics": _physics_diagnostics_summary(self.last_read_diagnostics),
            "last_error": self.last_error,
            "mutation_authority": "OVPhysX",
            "zero_copy_unverified": True,
            "direct_native_stage_sharing_unverified": True,
            "final_ovstage_ipc_topology_unverified": True,
            "worker_log": self.config.worker_log_path,
        }

    def _async_diagnostics_snapshot(self) -> dict[str, Any]:
        return self._pose_producer.diagnostics()

    def _sync_pose_producer_diagnostics(self) -> None:
        if self._pose_producer.last_step_diagnostics is not None:
            self.last_step_diagnostics = self._pose_producer.last_step_diagnostics
        if self._pose_producer.last_read_diagnostics is not None:
            self.last_read_diagnostics = self._pose_producer.last_read_diagnostics

    def _start(
        self,
        *,
        initial_condition_values: Sequence[_BodyPose] = (),
    ) -> dict[str, Any]:
        starting_values = tuple(initial_condition_values)
        if error := _initial_condition_values_error(starting_values):
            self.failed = True
            self.completed = True
            self.last_error = error
            self.last_initial_condition_write = {"status": "error", "error": error}
            return self._record_update(
                "initial",
                False,
                [],
                "initial_condition_value_validation_error",
                extra={
                    "failed": True,
                    "sim_value_write_applied": False,
                    "physics_generation_reset": False,
                    "value_requested_count": len(starting_values),
                    "sim_value_paths": [pose.prim_path for pose in starting_values],
                    "initial_condition_value_error": error,
                },
            )
        startup_phase = "start"
        try:
            self._trace("physics_start.begin")
            self._physics_client.start()
            self._trace("physics_start.end")
            startup_phase = "create_simulation"
            self._trace("create_simulation.begin", simulation_id=self.physics_simulation_id)
            self.create_diagnostics = self._physics_client.create_simulation()
            self._trace("create_simulation.end", simulation_id=self.physics_simulation_id)
            startup_phase = "read_body_states"
            if self.config.body_prims:
                self._trace("read_body_states.begin", simulation_time_ns=0)
                states, read_diagnostics = self._physics_client.read_body_states(0)
                self._trace("read_body_states.end", simulation_time_ns=0, state_count=len(states))
            else:
                states = []
                read_diagnostics = {
                    "status": "skipped",
                    "reason": "empty_dynamic_body_set",
                    "body_count": 0,
                    "simulation_time_ns": 0,
                }
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.completed = True
            self.last_error = message
            return self._record_update(
                "initial",
                False,
                [],
                "physics_startup_error",
                extra={
                    "failed": True,
                    "physics_startup_phase": startup_phase,
                    "physics_error": message,
                },
            )
        self.last_read_diagnostics = read_diagnostics
        value_write_diagnostics: Mapping[str, Any] | None = None
        if starting_values:
            simulation_time_ns = max(1, int(self.current_simulation_time_ns) + 1)
            self._trace(
                "initial_condition_values.write.begin",
                simulation_time_ns=simulation_time_ns,
                body_count=len(starting_values),
            )
            try:
                value_write_diagnostics = self._physics_client.write_body_poses(
                    starting_values,
                    simulation_time_ns=simulation_time_ns,
                    reset=False,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.failed = True
                self.completed = True
                self.last_error = message
                self.last_initial_condition_write = {"status": "error", "error": message}
                return self._record_update(
                    "initial",
                    False,
                    [],
                    "initial_condition_value_write_error",
                    extra={
                        "failed": True,
                        "sim_value_write_applied": False,
                        "physics_generation_reset": False,
                        "value_requested_count": len(starting_values),
                        "sim_value_paths": [pose.prim_path for pose in starting_values],
                        "initial_condition_value_error": message,
                    },
                )
            if error := _physics_write_error(value_write_diagnostics, len(starting_values)):
                self.failed = True
                self.completed = True
                self.last_error = error
                self.last_initial_condition_write = {
                    **dict(value_write_diagnostics),
                    "status": "error",
                    "error": error,
                }
                return self._record_update(
                    "initial",
                    False,
                    [],
                    "initial_condition_value_write_error",
                    extra={
                        "failed": True,
                        "sim_value_write_applied": False,
                        "physics_generation_reset": False,
                        "value_requested_count": len(starting_values),
                        "sim_value_paths": [pose.prim_path for pose in starting_values],
                        "sim_value_write": dict(value_write_diagnostics),
                        "initial_condition_value_error": error,
                    },
                )
            self.last_initial_condition_write = dict(value_write_diagnostics)
            self._trace(
                "initial_condition_values.write.end",
                simulation_time_ns=simulation_time_ns,
                body_count=len(starting_values),
            )
            self.current_simulation_time_ns = simulation_time_ns
        self.started = True
        return self._publish_poses(
            states,
            "initial",
            initial_condition_values=starting_values,
            extra={
                "reason": "initial_condition_values",
                "value_requested_count": len(starting_values),
                "sim_value_paths": [pose.prim_path for pose in starting_values],
                "sim_value_write": dict(value_write_diagnostics or {}),
                "sim_value_write_applied": True,
                "physics_generation_reset": True,
            }
            if starting_values
            else None,
        )

    def _publish_poses(
        self,
        states: Sequence[Mapping[str, Any]],
        reason: str,
        *,
        publication: PhysicsPosePublication | None = None,
        pose_handoff_wait_ms: float = 0.0,
        initial_condition_values: Sequence[_BodyPose] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.failed:
            return self._record_update("failed", False, [], "composition_failed")
        self.last_pose_handoff_wait_ms = pose_handoff_wait_ms
        self._trace("publish_poses.begin", reason=reason, state_count=len(states))
        try:
            poses = (
                list(publication.poses)
                if publication is not None
                else list(complete_physics_pose_set(states, self.config.body_prims))
            )
            if initial_condition_values:
                poses = list(apply_initial_condition_values(poses, initial_condition_values))
            mutation = self._stage_host.publish_ovphysx_poses(
                poses, self.current_simulation_time_ns
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed = True
            self.completed = True
            if reason == "initial":
                self.started = False
            self.last_error = message
            self._trace("publish_poses.error", reason=reason, error=message)
            return self._record_update(
                reason,
                False,
                [],
                "pose_publication_error",
                extra={"failed": True, "pose_publication_error": message},
            )
        dirty_paths = list(mutation.dirty_paths)
        self.changed_body_paths.update(dirty_paths)
        self._trace(
            "publish_poses.end",
            reason=reason,
            dirty_path_count=len(dirty_paths),
            stage_revision=self._stage_host.revision,
        )
        return self._record_update(
            reason,
            bool(dirty_paths),
            dirty_paths,
            "",
            extra={
                "pose_handoff_wait_ms": pose_handoff_wait_ms,
                "latest_pose_lag_steps": self.last_latest_pose_lag_steps,
                "latest_pose_publication_sequence": publication.sequence if publication is not None else 0,
                **dict(extra or {}),
            },
        )

    def _record_update(
        self,
        reason: str,
        stage_updated: bool,
        dirty_paths: Sequence[str],
        skipped_reason: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        update = {
            "reason": reason,
            "composition_mode": self.composition_mode,
            "composition_config_fingerprint": self.composition_config_fingerprint,
            "stage_updated": stage_updated,
            "failed": self.failed,
            "sim_value_write_applied": False,
            "physics_generation_reset": False,
            "skipped_reason": skipped_reason,
            "dirty_paths": list(dirty_paths),
            "step_count": self.step_count,
            "simulation_time_ns": self.current_simulation_time_ns,
            "stage_revision": self._stage_host.revision,
            "composition_generation": self.composition_generation,
            "playback_intent_generation": self.playback_intent_generation,
            "mutation_authority": "OVPhysX",
        }
        if extra:
            update.update(dict(extra))
        self.sync_count += 1
        if stage_updated:
            self.stage_update_count += 1
        if skipped_reason == "max_steps":
            self.completed_sync_count += 1
        if skipped_reason == "in_progress":
            self.busy_sync_count += 1
        self.last_update = update
        return update

    def _stage_result(self, update: Mapping[str, Any]) -> OvphysxStageResult:
        raw_reason = str(update.get("reason", ""))
        reason = str(update.get("skipped_reason", "")) or raw_reason
        if self.failed or bool(update.get("failed", False)):
            status = OvphysxStageStatus.FAILED
        elif raw_reason == "busy":
            status = OvphysxStageStatus.BUSY
        elif raw_reason == "completed":
            status = OvphysxStageStatus.COMPLETED
        else:
            status = OvphysxStageStatus.OK
        simulation_time_ns = _mapping_int(
            update, "simulation_time_ns", self.current_simulation_time_ns
        )
        return OvphysxStageResult(
            status=status,
            reason=reason,
            pose_set=self.physics_pose_set(simulation_time_ns),
            dirty_paths=tuple(str(path) for path in update.get("dirty_paths", ())),
            step_count=_mapping_int(update, "step_count", self.step_count),
            simulation_time_ns=simulation_time_ns,
            generation=_mapping_int(
                update, "composition_generation", self.composition_generation
            ),
        )

    def _trace(self, event: str, **fields: Any) -> None:
        if not self.config.trace_log_path:
            return
        payload = {"event": event, "time_ns": time.time_ns(), **fields}
        try:
            path = Path(self.config.trace_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass


def _physics_write_error(result: Mapping[str, Any], requested_count: int) -> str:
    status = str(result.get("status", "")).lower()
    if result.get("ok") is False or bool(result.get("failed", False)) or status in {"error", "failed", "unavailable"}:
        return str(result.get("error") or result.get("skipped_reason") or status or "physics write failed")
    if result.get("error"):
        return str(result["error"])
    if result.get("skipped_reason"):
        return str(result["skipped_reason"])
    grpc_status = str(result.get("grpc_status", "OK")).upper()
    if grpc_status != "OK":
        return f"physics write gRPC status is {grpc_status}"
    body_count = result.get("body_count")
    if type(body_count) is not int:
        return "physics write returned an invalid body_count"
    if body_count != requested_count:
        return f"physics write accepted {body_count} of {requested_count} poses"
    return ""


def _initial_condition_values_error(poses: Sequence[_BodyPose]) -> str:
    paths = [pose.prim_path for pose in poses]
    if len(set(paths)) != len(paths):
        return "ValueError: initial-condition body pose paths must be unique"
    return ""


def _composition_config_fingerprint(config: _InteractiveSharedStageConfig) -> str:
    payload = repr(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _physics_client_transport(physics_client: Any) -> str:
    return str(getattr(physics_client, "transport", "custom"))


def _physics_client_description(physics_client: Any) -> str:
    return str(getattr(physics_client, "topology_description", "custom Python physics client"))


def _physics_worker_description(physics_client: Any) -> str:
    return str(getattr(physics_client, "worker_description", "custom Python physics worker"))


def _physics_diagnostics_summary(diagnostics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    summary_keys = (
        "name",
        "transport",
        "method",
        "exit_code",
        "simulation_time_ns",
        "step_count",
        "body_count",
        "step_ms",
        "read_ms",
        "total_ms",
        "python_call_ms",
        "step_timings_ms",
    )
    return {key: diagnostics[key] for key in summary_keys if key in diagnostics}


def _body_states_from_read_response(
    response: Mapping[str, Any],
    requested_paths: Sequence[str],
    simulation_time_ns: int,
) -> list[dict[str, Any]]:
    columns = _columns_by_attribute(response)
    paths = _packed_strings(columns.get("usd-path"))
    if not paths:
        paths = list(requested_paths)
    row_count = max(
        len(paths),
        _packed_len(columns.get("xformOp:translate"), ("packedFloat3", "packed_float3")),
        _packed_len(columns.get("xformOp:orient"), ("packedQuatf", "packedQuatF", "packed_quatf")),
    )
    states: list[dict[str, Any]] = []
    for index in range(row_count):
        prim_path = paths[index] if index < len(paths) else requested_paths[index] if index < len(requested_paths) else UNKNOWN
        states.append(
            {
                "prim_path": prim_path,
                "simulation_time_ns": int(simulation_time_ns),
                "translate": _float3_at(columns.get("xformOp:translate"), index),
                "orient": _quatf_at(columns.get("xformOp:orient"), index),
                "linear_velocity": _float3_at(columns.get("physics:velocity"), index),
                "angular_velocity": _float3_at(columns.get("physics:angularVelocity"), index),
            }
        )
    return states


def _columns_by_attribute(response: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    columns: dict[str, Mapping[str, Any]] = {}
    for column in response.get("resultSet", {}).get("columns", []):
        if not isinstance(column, Mapping):
            continue
        attribute = column.get("attribute", {})
        if not isinstance(attribute, Mapping):
            continue
        name = attribute.get("attributeName")
        if isinstance(name, str):
            columns[name] = column
    return columns


def _packed_strings(column: Mapping[str, Any] | None) -> list[str]:
    values = _packed_values(column, ("packedString", "packed_string"))
    return [str(value) for value in values]


def _float3_at(column: Mapping[str, Any] | None, index: int) -> dict[str, Any]:
    values = _packed_values(column, ("packedFloat3", "packed_float3"))
    if index >= len(values) or not isinstance(values[index], Mapping):
        return {"found": False}
    value = values[index]
    return {
        "found": True,
        "x": float(value.get("x", 0.0)),
        "y": float(value.get("y", 0.0)),
        "z": float(value.get("z", 0.0)),
    }


def _quatf_at(column: Mapping[str, Any] | None, index: int) -> dict[str, Any]:
    values = _packed_values(column, ("packedQuatf", "packedQuatF", "packed_quatf"))
    if index >= len(values) or not isinstance(values[index], Mapping):
        return {"found": False}
    value = values[index]
    return {
        "found": True,
        "i": float(value.get("i", 0.0)),
        "j": float(value.get("j", 0.0)),
        "k": float(value.get("k", 0.0)),
        "r": float(value.get("r", 1.0)),
    }


def _packed_len(column: Mapping[str, Any] | None, packed_names: Sequence[str]) -> int:
    return len(_packed_values(column, packed_names))


def _packed_values(column: Mapping[str, Any] | None, packed_names: Sequence[str]) -> list[Any]:
    if not isinstance(column, Mapping):
        return []
    values_column = column.get("values", {})
    if not isinstance(values_column, Mapping):
        return []
    for packed_name in packed_names:
        packed = values_column.get(packed_name)
        if not isinstance(packed, Mapping):
            continue
        values = packed.get("values", [])
        return list(values) if isinstance(values, list) else []
    return []


__all__ = [
    "OvphysxStageController",
    "_body_states_from_read_response",
]
