# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import fields
from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    COMPOSITION_SYNC_LOCK,
    OvphysxStageController,
    OvphysxStageResult,
    OvphysxStageStatus,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose, BodyVelocity  # noqa: E402
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig  # noqa: E402


BODY = "/World/PhysicsIsland/DynamicBodies/Cube_00"


def _config(*, max_steps: int = 3, update_fps: float = 1000.0) -> InteractiveSharedStageConfig:
    return InteractiveSharedStageConfig(
        enabled=True,
        input_usd_path="/tmp/stair_drop_ovrtx_ovphysx.usda",
        server="/tmp/ovphysx-bridge-server/bin/ovphysx-bridge-server",
        ovphysx_address="127.0.0.1:50094",
        ovphysx_worker_command="worker",
        device="cpu",
        body_root="/World/PhysicsIsland/DynamicBodies",
        body_prims=(BODY,),
        physics_fps=60.0,
        update_fps=update_fps,
        max_steps=max_steps,
        body_scale=1.0,
        worker_log_path="/tmp/ovphysx-worker.log",
        ovphysx_native_client_module="ovphysx_bridge_client",
    )


class _Physics:
    def __init__(self) -> None:
        self.started = False
        self.steps: list[int] = []
        self.writes: list[tuple[tuple[BodyPose, ...], int, bool]] = []
        self.velocity_writes: list[tuple[tuple[BodyVelocity, ...], int, bool]] = []
        self.shutdown_count = 0

    def start(self) -> None:
        self.started = True

    def create_simulation(self) -> dict[str, object]:
        return {"status": "created"}

    def _states(self, simulation_time_ns: int):
        y = 5.0 - len(self.steps) * 0.5
        return ([{
            "prim_path": BODY,
            "translate": {"found": True, "x": 0.0, "y": y, "z": 0.0},
            "orient": {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0},
        }], {"simulation_time_ns": simulation_time_ns})

    def read_body_states(self, simulation_time_ns: int):
        return self._states(simulation_time_ns)

    def advance_and_read_body_states(self, start_step_count: int, steps: int, timestep_ns: int):
        step_count = start_step_count + steps
        self.steps.extend(range(start_step_count + 1, step_count + 1))
        states, _ = self._states(step_count * timestep_ns)
        return states, {
            "step_count": step_count,
            "simulation_time_ns": step_count * timestep_ns,
        }

    def write_body_poses(
        self,
        poses,
        *,
        simulation_time_ns: int,
        reset: bool,
    ) -> dict[str, object]:
        pose_set = tuple(poses)
        self.writes.append((pose_set, simulation_time_ns, reset))
        return {"body_count": len(pose_set), "simulation_time_ns": simulation_time_ns}

    def write_body_velocities(
        self,
        velocities,
        *,
        simulation_time_ns: int,
        reset: bool,
    ) -> dict[str, object]:
        velocity_set = tuple(velocities)
        self.velocity_writes.append((velocity_set, simulation_time_ns, reset))
        return {"body_count": len(velocity_set), "simulation_time_ns": simulation_time_ns}

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _StartupFailure(_Physics):
    def read_body_states(self, simulation_time_ns: int):
        raise RuntimeError("startup read failed")


class _StepFailure(_Physics):
    def advance_and_read_body_states(self, start_step_count: int, steps: int, timestep_ns: int):
        raise RuntimeError("physics step failed")


class _WriteFailure(_Physics):
    def write_body_poses(self, poses, *, simulation_time_ns: int, reset: bool):
        raise RuntimeError("physics write failed")


class _MalformedStartupPose(_Physics):
    def read_body_states(self, simulation_time_ns: int):
        states, diagnostics = super().read_body_states(simulation_time_ns)
        del states[0]["orient"]
        return states, diagnostics


class _MalformedStepPose(_Physics):
    def advance_and_read_body_states(self, start_step_count: int, steps: int, timestep_ns: int):
        states, diagnostics = super().advance_and_read_body_states(
            start_step_count, steps, timestep_ns
        )
        del states[0]["orient"]
        return states, diagnostics


class _BlockingPoseWrite(_Physics):
    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def write_body_poses(self, poses, *, simulation_time_ns: int, reset: bool):
        self.write_started.set()
        assert self.release_write.wait(timeout=2.0)
        return super().write_body_poses(
            poses, simulation_time_ns=simulation_time_ns, reset=reset
        )


class _IncompleteSecondBlockingThird(_Physics):
    def __init__(self) -> None:
        super().__init__()
        self.third_advance_started = threading.Event()
        self.release_third_advance = threading.Event()

    def advance_and_read_body_states(self, start_step_count: int, steps: int, timestep_ns: int):
        step_count = start_step_count + steps
        self.steps.extend(range(start_step_count + 1, step_count + 1))
        if start_step_count == 2:
            self.third_advance_started.set()
            assert self.release_third_advance.wait(timeout=2.0)
        states, _ = self._states(step_count * timestep_ns)
        if step_count == 2:
            del states[0]["orient"]
        return states, {
            "step_count": step_count,
            "simulation_time_ns": step_count * timestep_ns,
        }


def _controller(physics: _Physics, *, max_steps: int = 3) -> OvphysxStageController:
    return OvphysxStageController(_config(max_steps=max_steps), physics_client=physics, simulation_id="sim")


def _wait_for(predicate) -> None:
    deadline = time.monotonic() + 2.0
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out")
        time.sleep(0.001)


def test_result_contract_has_exact_closed_field_set() -> None:
    assert [field.name for field in fields(OvphysxStageResult)] == [
        "status", "reason", "pose_set", "dirty_paths", "step_count",
        "simulation_time_ns", "generation",
    ]
    assert set(OvphysxStageStatus) == {
        OvphysxStageStatus.OK, OvphysxStageStatus.COMPLETED,
        OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED,
    }


def test_tick_starts_physics_and_publishes_authoritative_pose() -> None:
    physics = _Physics()
    controller = _controller(physics)
    try:
        result = controller.tick()

        assert result.status == OvphysxStageStatus.OK
        assert result.reason == "initial"
        assert result.dirty_paths == (BODY,)
        assert result.pose_set[0].prim_path == BODY
        assert controller.diagnostics()["stage_host"]["body_paths"] == [BODY]
        assert physics.started is True
    finally:
        controller.shutdown()


def test_tick_advances_then_reports_completion() -> None:
    controller = _controller(_Physics(), max_steps=1)
    try:
        controller.tick()
        stepped = controller.tick(max_steps=1)
        completed = controller.tick(max_steps=1)

        assert stepped.status == OvphysxStageStatus.OK
        assert stepped.step_count == 1
        assert completed.status == OvphysxStageStatus.COMPLETED
        assert completed.reason == "max_steps"
    finally:
        controller.shutdown()


def test_tick_reports_busy_without_mutating_stage() -> None:
    controller = _controller(_Physics())
    assert COMPOSITION_SYNC_LOCK.acquire(blocking=False)
    try:
        result = controller.tick()
    finally:
        COMPOSITION_SYNC_LOCK.release()
        controller.shutdown()

    assert result.status == OvphysxStageStatus.BUSY
    assert result.reason == "in_progress"
    assert result.pose_set == ()


@pytest.mark.parametrize("physics", [_StartupFailure(), _StepFailure()])
def test_physics_failures_are_terminal(physics: _Physics) -> None:
    controller = _controller(physics)
    try:
        first = controller.tick()
        failed = first if first.status == OvphysxStageStatus.FAILED else controller.tick()
        retried = controller.tick()

        assert failed.status == OvphysxStageStatus.FAILED
        assert retried.status == OvphysxStageStatus.FAILED
        assert controller.failed is True
    finally:
        controller.shutdown()


def test_malformed_startup_pose_returns_terminal_typed_failure() -> None:
    controller = _controller(_MalformedStartupPose())
    try:
        result = controller.tick()

        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "pose_publication_error"
        assert controller.started is False
        assert controller.failed is True
        assert "orient" in controller.last_error
    finally:
        controller.shutdown()


def test_malformed_step_pose_returns_terminal_typed_failure() -> None:
    controller = _controller(_MalformedStepPose())
    try:
        initial = controller.tick()
        result = controller.tick()

        assert initial.status == OvphysxStageStatus.OK
        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "pose_publication_error"
        assert controller.started is True
        assert controller.failed is True
        assert "orient" in controller.last_error
    finally:
        controller.shutdown()


def test_startup_initial_conditions_are_applied_before_publication() -> None:
    physics = _Physics()
    controller = _controller(physics)
    pose = BodyPose(BODY, (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0))
    try:
        result = controller.tick(initial_condition_values=(pose,))

        assert result.status == OvphysxStageStatus.OK
        assert result.pose_set == (pose,)
        assert physics.writes[0][0] == (pose,)
        assert result.dirty_paths == (BODY,)
    finally:
        controller.shutdown()


def test_startup_rejects_duplicate_initial_condition_paths_before_external_calls() -> None:
    physics = _Physics()
    controller = _controller(physics)
    poses = (
        BodyPose(BODY, (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
        BodyPose(BODY, (3.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
    )
    try:
        result = controller.tick(initial_condition_values=poses)

        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "initial_condition_value_validation_error"
        assert physics.started is False
        assert physics.writes == []
    finally:
        controller.shutdown()


def test_initial_condition_application_resets_generation_and_publishes_stage() -> None:
    physics = _Physics()
    controller = _controller(physics)
    pose = BodyPose(BODY, (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0))
    try:
        initial = controller.tick()
        result = controller.apply_initial_condition_values((pose,))

        assert result.status == OvphysxStageStatus.OK
        assert result.generation > initial.generation
        assert result.step_count == 0
        assert result.pose_set == (pose,)
        assert result.dirty_paths == (BODY,)
    finally:
        controller.shutdown()


def test_live_initial_conditions_reject_duplicate_paths_before_physics_write() -> None:
    physics = _Physics()
    controller = _controller(physics)
    poses = (
        BodyPose(BODY, (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
        BodyPose(BODY, (3.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
    )
    try:
        initial = controller.tick()
        result = controller.apply_initial_condition_values(poses)

        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "initial_condition_value_validation_error"
        assert physics.writes == []
        assert result.pose_set == initial.pose_set
    finally:
        controller.shutdown()


def test_initial_condition_write_failure_is_terminal() -> None:
    controller = _controller(_WriteFailure())
    pose = BodyPose(BODY, (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0))
    try:
        controller.tick()
        failed = controller.apply_initial_condition_values((pose,))

        assert failed.status == OvphysxStageStatus.FAILED
        assert failed.reason == "initial_condition_value_write_error"
        assert controller.failed is True
    finally:
        controller.shutdown()


def test_body_velocity_edit_advances_generation_without_immediate_stage_pose() -> None:
    physics = _Physics()
    controller = _controller(physics)
    velocity = BodyVelocity(BODY, (6.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    try:
        initial = controller.tick()
        result = controller.apply_body_velocity_edits((velocity,))

        assert result.status == OvphysxStageStatus.OK
        assert result.generation > initial.generation
        assert result.simulation_time_ns > initial.simulation_time_ns
        assert result.pose_set == ()
        assert result.dirty_paths == ()
        assert physics.velocity_writes == [((velocity,), result.simulation_time_ns, False)]
        assert controller.diagnostics()["stage_host"]["revision"] == 1
    finally:
        controller.shutdown()


def test_body_velocity_unknown_path_fails_before_playback_invalidation() -> None:
    physics = _Physics()
    controller = _controller(physics)
    try:
        controller.tick()
        generation = controller.composition_generation
        playback_generation = controller.playback_intent_generation
        result = controller.apply_body_velocity_edits((
            BodyVelocity("/World/Unknown", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ))

        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason.startswith("unknown_dynamic_body_paths")
        assert controller.composition_generation == generation
        assert controller.playback_intent_generation == playback_generation
        assert controller.failed is False
        assert physics.velocity_writes == []
    finally:
        controller.shutdown()


def test_body_velocity_rejects_exhausted_step_budget_before_write() -> None:
    physics = _Physics()
    controller = _controller(physics, max_steps=1)
    try:
        controller.tick()
        completed = controller.tick(max_steps=1)
        generation = controller.composition_generation
        result = controller.apply_body_velocity_edits((
            BodyVelocity(BODY, (1.0, 0.0, 0.0)),
        ))

        assert completed.step_count == 1
        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "no_remaining_physics_step_budget"
        assert controller.composition_generation == generation
        assert physics.velocity_writes == []
    finally:
        controller.shutdown()


def test_body_velocity_rechecks_budget_after_adopting_async_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _Physics()
    controller = _controller(physics, max_steps=2)
    try:
        controller.tick()

        def adopt_async_limit() -> None:
            controller.step_count = controller.max_steps_limit
            controller.current_simulation_time_ns = (
                controller.step_count * controller.config.timestep_ns
            )

        monkeypatch.setattr(controller, "adopt_async_producer_position", adopt_async_limit)
        result = controller.apply_body_velocity_edits((
            BodyVelocity(BODY, (1.0, 0.0, 0.0)),
        ))

        assert result.status == OvphysxStageStatus.FAILED
        assert result.reason == "no_remaining_physics_step_budget"
        assert result.step_count == controller.max_steps_limit
        assert result.simulation_time_ns == controller.current_simulation_time_ns
        assert physics.velocity_writes == []
    finally:
        controller.shutdown()


def test_diagnostics_include_last_body_velocity_write() -> None:
    physics = _Physics()
    controller = _controller(physics)
    try:
        controller.tick()
        controller.apply_body_velocity_edits((BodyVelocity(BODY, (1.0, 0.0, 0.0)),))

        assert controller.diagnostics()["last_body_velocity_write"]["body_count"] == 1
    finally:
        controller.shutdown()


def test_async_latest_pose_publication_updates_authoritative_stage() -> None:
    physics = _Physics()
    controller = _controller(physics, max_steps=2)
    try:
        initial = controller.publish_latest_pose(max_steps=2)
        _wait_for(lambda: controller.pose_publication_complete_count >= 1)
        latest = controller.publish_latest_pose(max_steps=2)

        assert initial.status == OvphysxStageStatus.OK
        assert latest.status in {OvphysxStageStatus.OK, OvphysxStageStatus.COMPLETED}
        assert latest.step_count >= 1
        assert latest.pose_set
        assert controller.diagnostics()["stage_host"]["body_paths"] == [BODY]
    finally:
        controller.shutdown()


def test_direct_edit_serializes_async_restart_until_write_completes() -> None:
    physics = _BlockingPoseWrite()
    controller = _controller(physics, max_steps=100_000)
    controller.tick()
    pose = BodyPose(BODY, (1.0, 5.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    results: dict[str, OvphysxStageResult] = {}
    direct_thread = threading.Thread(
        target=lambda: results.setdefault(
            "direct", controller.apply_initial_condition_values((pose,))
        )
    )
    async_thread = threading.Thread(
        target=lambda: results.setdefault(
            "async", controller.publish_latest_pose(max_steps=100_000)
        )
    )
    try:
        direct_thread.start()
        assert physics.write_started.wait(timeout=2.0)
        async_thread.start()
        time.sleep(0.01)
        assert async_thread.is_alive()
        assert physics.steps == []

        physics.release_write.set()
        direct_thread.join(timeout=2.0)
        async_thread.join(timeout=2.0)

        assert not direct_thread.is_alive()
        assert not async_thread.is_alive()
        assert results["direct"].status == OvphysxStageStatus.OK
        assert results["async"].status in {
            OvphysxStageStatus.OK,
            OvphysxStageStatus.COMPLETED,
        }
    finally:
        physics.release_write.set()
        controller.shutdown()


def test_shutdown_waits_for_direct_runtime_transition() -> None:
    physics = _BlockingPoseWrite()
    controller = _controller(physics)
    controller.tick()
    pose = BodyPose(BODY, (1.0, 5.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    results: dict[str, OvphysxStageResult] = {}
    direct_thread = threading.Thread(
        target=lambda: results.setdefault(
            "direct", controller.apply_initial_condition_values((pose,))
        )
    )
    shutdown_thread = threading.Thread(target=controller.shutdown)
    try:
        direct_thread.start()
        assert physics.write_started.wait(timeout=2.0)
        shutdown_thread.start()
        time.sleep(0.01)
        assert shutdown_thread.is_alive()
        assert physics.shutdown_count == 0

        physics.release_write.set()
        direct_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=2.0)

        assert not direct_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert results["direct"].status == OvphysxStageStatus.OK
        assert physics.shutdown_count == 1
        assert controller.closed is True
    finally:
        physics.release_write.set()


def test_async_failure_after_handoff_prevents_stage_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(_Physics(), max_steps=2)
    try:
        controller.publish_latest_pose(max_steps=2)
        _wait_for(lambda: controller.pose_publication_complete_count >= 1)
        initial_revision = controller.diagnostics()["stage_host"]["revision"]
        original_take = controller._take_latest_pose_publication

        def take_then_fail():
            handoff = original_take()
            controller._record_async_error(RuntimeError("async handoff failed"))
            return handoff

        monkeypatch.setattr(controller, "_take_latest_pose_publication", take_then_fail)
        failed = controller.publish_latest_pose(max_steps=2)

        assert failed.status == OvphysxStageStatus.FAILED
        assert failed.reason == "composition_failed"
        assert controller.diagnostics()["stage_host"]["revision"] == initial_revision
        assert controller.diagnostics()["async_publication"]["active"] is False
    finally:
        controller.shutdown()


def test_public_stop_cannot_invalidate_during_async_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(_Physics(), max_steps=2)
    controller.publish_latest_pose(max_steps=2)
    _wait_for(lambda: controller.pose_publication_complete_count >= 1)
    handoff_taken = threading.Event()
    release_handoff = threading.Event()
    original_take = controller._take_latest_pose_publication

    def blocking_take():
        handoff = original_take()
        handoff_taken.set()
        assert release_handoff.wait(timeout=2.0)
        return handoff

    monkeypatch.setattr(controller, "_take_latest_pose_publication", blocking_take)
    results: dict[str, OvphysxStageResult] = {}
    apply_thread = threading.Thread(
        target=lambda: results.setdefault(
            "apply", controller.publish_latest_pose(max_steps=2)
        )
    )
    stop_thread = threading.Thread(
        target=lambda: controller.stop_async_playback(wait=True, invalidate=True)
    )
    try:
        initial_generation = controller.composition_generation
        apply_thread.start()
        assert handoff_taken.wait(timeout=2.0)
        stop_thread.start()
        time.sleep(0.01)
        assert stop_thread.is_alive()
        assert controller.composition_generation == initial_generation

        release_handoff.set()
        apply_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

        assert not apply_thread.is_alive()
        assert not stop_thread.is_alive()
        assert results["apply"].status in {
            OvphysxStageStatus.OK,
            OvphysxStageStatus.COMPLETED,
        }
        assert controller.composition_generation >= initial_generation
        assert controller.diagnostics()["async_publication"]["active"] is False
    finally:
        release_handoff.set()
        controller.shutdown()


def test_older_retained_publication_does_not_rewind_adopted_cursor() -> None:
    physics = _IncompleteSecondBlockingThird()
    controller = _controller(physics, max_steps=3)
    try:
        controller.publish_latest_pose(max_steps=2)
        _wait_for(lambda: controller.pose_read_incomplete_count >= 1)
        controller.start_async_playback(max_steps_limit=2)
        assert controller.step_count == 2
        controller.start_async_playback(max_steps_limit=3)
        assert physics.third_advance_started.wait(timeout=2.0)

        stale_handoff = controller.publish_latest_pose(max_steps=3)
        assert controller.step_count == 2
        physics.release_third_advance.set()
        _wait_for(lambda: controller.pose_publication_complete_count >= 2)
        fresh_handoff = controller.publish_latest_pose(max_steps=3)

        assert stale_handoff.reason == "no_new_pose_publication"
        assert fresh_handoff.step_count == 3
    finally:
        physics.release_third_advance.set()
        controller.shutdown()


def test_async_to_sync_transition_adopts_producer_cursor() -> None:
    physics = _Physics()
    controller = _controller(physics, max_steps=2)
    try:
        controller.publish_latest_pose(max_steps=2)
        _wait_for(lambda: controller.pose_publication_complete_count >= 1)
        controller.stop_async_playback(wait=True, invalidate=True)
        controller.adopt_async_producer_position()
        result = controller.tick(max_steps=2)

        assert physics.steps == [1, 2]
        assert result.step_count == 2
        assert result.simulation_time_ns == 2 * controller.config.timestep_ns
    finally:
        controller.shutdown()


def test_shutdown_is_idempotent_and_closed_operations_fail() -> None:
    physics = _Physics()
    controller = _controller(physics)
    controller.tick()

    controller.shutdown()
    controller.shutdown()
    closed = controller.tick()

    assert physics.shutdown_count == 1
    assert closed.status == OvphysxStageStatus.FAILED
    assert closed.reason == "composition_shutdown"
