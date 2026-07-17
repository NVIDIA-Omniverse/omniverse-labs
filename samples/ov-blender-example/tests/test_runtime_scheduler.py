# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    OvphysxStageController,
    OvphysxStageResult,
    OvphysxStageStatus,
)
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    EditMechanism,
    EditPersistence,
    EditStatus,
    DataAuthority,
    EditShape,
    EditIntent,
    edit_location,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    RuntimeScheduler,
    RuntimeTickRequest,
    RuntimeTickStatus,
    _timeline_max_steps,
    _timeline_should_reset,
    _timeline_should_step,
)
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxSessionUpdatePort,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose, BodyVelocity  # noqa: E402
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig  # noqa: E402


def test_shared_scheduler_serializes_viewport_ticks() -> None:
    scheduler = RuntimeScheduler()
    state_lock = threading.Lock()
    active = 0
    maximum = 0

    def tick(_request: object, **_kwargs: object) -> object:
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return object()

    scheduler._tick_viewport = tick  # type: ignore[method-assign]
    threads = [
        threading.Thread(
            target=scheduler.tick_viewport,
            args=(RuntimeTickRequest("/tmp/scene.usda"),),
            kwargs={"ovrtx_updates": object()},
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum == 1


def _config(
    *,
    enabled: bool = True,
    physics_fps: float = 60.0,
    update_fps: float = 60.0,
    max_steps: int = 2,
) -> InteractiveSharedStageConfig:
    return InteractiveSharedStageConfig(
        enabled=enabled,
        input_usd_path="/tmp/stair_drop_ovrtx_ovphysx.usda",
        server="/tmp/ovphysx-bridge-server/bin/ovphysx-bridge-server",
        ovphysx_address="127.0.0.1:50094",
        ovphysx_worker_command="worker",
        device="cpu",
        body_root="/World/PhysicsIsland/DynamicBodies",
        body_prims=("/World/PhysicsIsland/DynamicBodies/Cube_00",),
        physics_fps=physics_fps,
        update_fps=update_fps,
        max_steps=max_steps,
        body_scale=1.0,
        worker_log_path="/tmp/ovphysx-worker.log",
    )


class _FakePhysicsClient:
    def __init__(self) -> None:
        self.started = False
        self.created = False
        self.steps: list[int] = []
        self.pose_writes: list[dict[str, object]] = []
        self.velocity_writes: list[dict[str, object]] = []
        self.write_order: list[str] = []
        self.fail_velocity_writes = False
        self.shutdown_called = False

    def start(self) -> None:
        self.started = True

    def create_simulation(self) -> dict[str, object]:
        self.created = True
        return {"status": "created"}

    def read_body_states(self, simulation_time_ns: int) -> tuple[list[dict[str, object]], dict[str, object]]:
        y = 5.0 if simulation_time_ns == 0 else 4.5
        return (
            [
                {
                    "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
                    "translate": {"found": True, "x": 0.0, "y": y, "z": 0.0},
                    "orient": {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0},
                }
            ],
            {"simulation_time_ns": simulation_time_ns},
        )

    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        step_count = int(start_step_count) + int(steps)
        simulation_time_ns = step_count * int(timestep_ns)
        for step in range(int(start_step_count) + 1, step_count + 1):
            self.steps.append(step * int(timestep_ns))
        states, _ = self.read_body_states(simulation_time_ns)
        return (
            states,
            {
                "name": "advance_and_read_body_states",
                "transport": "native",
                "step_count": step_count,
                "simulation_time_ns": simulation_time_ns,
                "step_ms": 1.0,
                "read_ms": 0.25,
                "total_ms": 1.25,
                "body_count": len(states),
            },
        )

    def write_body_poses(
        self,
        poses: tuple[BodyPose, ...] | list[BodyPose],
        *,
        simulation_time_ns: int,
        reset: bool = False,
    ) -> dict[str, object]:
        self.pose_writes.append(
            {
                "simulation_time_ns": int(simulation_time_ns),
                "reset": bool(reset),
                "poses": tuple(poses),
            }
        )
        self.write_order.append("pose")
        return {
            "name": "write_body_poses",
            "simulation_time_ns": int(simulation_time_ns),
            "body_count": len(poses),
            "reset": bool(reset),
        }

    def write_body_velocities(
        self,
        velocities: tuple[BodyVelocity, ...] | list[BodyVelocity],
        *,
        simulation_time_ns: int,
        reset: bool = False,
    ) -> dict[str, object]:
        self.velocity_writes.append({
            "simulation_time_ns": int(simulation_time_ns),
            "reset": bool(reset),
            "velocities": tuple(velocities),
        })
        self.write_order.append("velocity")
        return {
            "name": "write_body_velocities",
            "simulation_time_ns": int(simulation_time_ns),
            "body_count": 0 if self.fail_velocity_writes else len(velocities),
            "reset": bool(reset),
        }

    def shutdown(self) -> None:
        self.shutdown_called = True


class _StablePhysicsClient(_FakePhysicsClient):
    def read_body_states(self, simulation_time_ns: int) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [
                {
                    "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
                    "translate": {"found": True, "x": 0.0, "y": 5.0, "z": 0.0},
                    "orient": {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0},
                }
            ],
            {"simulation_time_ns": simulation_time_ns},
        )


class _ThrowingStartPhysicsClient(_FakePhysicsClient):
    def start(self) -> None:
        raise RuntimeError("physics startup rejected")


class _MalformedInitialPosePhysicsClient(_FakePhysicsClient):
    def read_body_states(
        self,
        simulation_time_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        states, diagnostics = super().read_body_states(simulation_time_ns)
        del states[0]["orient"]
        return states, diagnostics


class _IncompleteAfterInitialPhysicsClient(_FakePhysicsClient):
    def read_body_states(
        self,
        simulation_time_ns: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if simulation_time_ns == 0:
            return super().read_body_states(simulation_time_ns)
        return (
            [
                {
                    "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
                    "translate": {"found": True, "x": 0.0, "y": 4.5, "z": 0.0},
                }
            ],
            {"simulation_time_ns": simulation_time_ns},
        )


class _BusyThenSteppedController(OvphysxStageController):
    def __init__(self, config: InteractiveSharedStageConfig) -> None:
        super().__init__(config, physics_client=_FakePhysicsClient(), simulation_id="sim")
        self.sync_calls = 0
        self._pose = BodyPose(
            "/World/PhysicsIsland/DynamicBodies/Cube_00",
            (0.0, 5.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

    def tick(self, **_kwargs: object) -> OvphysxStageResult:
        self.sync_calls += 1
        if self.sync_calls == 1:
            return OvphysxStageResult(
                OvphysxStageStatus.BUSY, "in_progress", (), (), 0, 0,
                self.composition_generation,
            )
        self.started = True
        return OvphysxStageResult(
            OvphysxStageStatus.OK,
            "initial" if self.sync_calls == 2 else "step",
            (self._pose,),
            (self._pose.prim_path,),
            0 if self.sync_calls == 2 else 1,
            0 if self.sync_calls == 2 else self.config.timestep_ns,
            self.composition_generation,
        )

    def physics_pose_set(self, _simulation_time_ns: int) -> tuple[BodyPose, ...]:
        return (self._pose,) if self.started else ()


class _FakeRenderClient:
    def __init__(self) -> None:
        self.transform_updates: list[list[OvrtxTransformValue]] = []
        self.material_updates: list[list[OvrtxAttributeValue]] = []
        self.attribute_updates: list[list[OvrtxAttributeValue]] = []

    def update_transforms(
        self, _session: object, values: list[OvrtxTransformValue]
    ) -> OvrtxValueUpdateResult:
        self.transform_updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)

    def update_attribute_values(
        self, _session: object, values: list[OvrtxAttributeValue]
    ) -> OvrtxValueUpdateResult:
        self.attribute_updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


class _FailingRenderClient(_FakeRenderClient):
    def update_transforms(self, _session: object, values: list[OvrtxTransformValue]) -> OvrtxValueUpdateResult:
        self.transform_updates.append(list(values))
        raise RuntimeError("native transform value write rejected")


class _ErrorResultRenderClient(_FakeRenderClient):
    def update_transforms(self, _session: object, values: list[OvrtxTransformValue]) -> OvrtxValueUpdateResult:
        self.transform_updates.append(list(values))
        return OvrtxValueUpdateResult(
            len(values),
            7 if values else None,
            {"status": "error", "error": "native transform value write rejected"},
        )


class _NoopRenderPort:
    def update_transforms(self, values: list[OvrtxTransformValue]) -> OvrtxValueUpdateResult:
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def _scheduler(
    config: InteractiveSharedStageConfig,
    physics: _FakePhysicsClient,
) -> RuntimeScheduler:
    return RuntimeScheduler(
        config_factory=lambda _input_usd_path: config,
        controller_factory=lambda controller_config: OvphysxStageController(
            controller_config,
            physics_client=physics,
            simulation_id="sim",
        ),
    )


def _edit_target(*, usd_layer_id: str = "/layers/scene.usda") -> dict[str, object]:
    return edit_location(
        usd_layer_id=usd_layer_id,
        usd_prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        usd_attribute="xformOp:transform",
        blender_property_path="location",
        provenance={"source": "test"},
    )


def _camera_edit_target() -> dict[str, object]:
    return edit_location(
        usd_prim_path="/World/Camera",
        usd_attribute="omni:xform",
        blender_property_path="viewport_camera_matrix",
        provenance={"source": "test"},
    )


def _material_edit_target() -> dict[str, object]:
    return edit_location(
        usd_prim_path="/World/Looks/Paint/Shader",
        usd_attribute="inputs:diffuseColor",
        blender_property_path="diffuse_color",
        provenance={"source": "test", "match_source": "hierarchy_path"},
    )


def _velocity_edit_target() -> dict[str, object]:
    return edit_location(
        usd_prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        usd_attribute="physics:velocity",
        provenance={"source": "test"},
    )


def _velocity_intent(value: tuple[float, float, float] = (5.0, 0.0, 0.0)):
    return InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.SIM,
            **_velocity_edit_target(),
            value=value,
        )
    ).to_intent()


def _request(
    *,
    now_ns: int = 0,
    timeline_controls_enabled: bool = True,
    timeline_playing: bool = False,
    timeline_frame: int = 1,
    timeline_start: int = 1,
    timeline_end: int = 2,
    simulation_reset_token: int = 0,
) -> RuntimeTickRequest:
    return RuntimeTickRequest(
        input_usd_path="/tmp/stair_drop_ovrtx_ovphysx.usda",
        now_ns=now_ns,
        timeline_controls_enabled=timeline_controls_enabled,
        timeline_playing=timeline_playing,
        timeline_frame=timeline_frame,
        timeline_start=timeline_start,
        timeline_end=timeline_end,
        simulation_reset_token=simulation_reset_token,
    )


def _tick(
    scheduler: RuntimeScheduler,
    render: _FakeRenderClient,
    session: object,
    **request_changes: object,
):
    return scheduler.tick_viewport(
        _request(**request_changes),
        ovrtx_updates=OvrtxSessionUpdatePort(render, session),
    )


def test_timeline_frame_range_maps_to_physics_step_limit() -> None:
    request = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_start=1,
        timeline_end=24,
    )

    assert _timeline_max_steps(request, steps_per_update=5, configured_max_steps=240) == 120


def test_scheduler_borrows_scene_owned_physics_and_routes_adapter_values() -> None:
    physics = _FakePhysicsClient()
    controller = OvphysxStageController(
        _config(),
        physics_client=physics,
        simulation_id="borrowed",
    )
    retained_transforms: list[tuple[OvrtxTransformValue, ...]] = []
    retained_poses: list[tuple[BodyPose, ...]] = []
    reset_calls: list[str] = []
    scheduler = RuntimeScheduler(
        config_factory=lambda _path: pytest.fail("borrowed physics must not rebuild config"),
        controller_provider=lambda: controller,
        controller_reset=lambda: (reset_calls.append("reset") or True),
        ovrtx_transform_sink=retained_transforms.append,
        ovphysx_initial_condition_sink=retained_poses.append,
    )
    render = _FakeRenderClient()
    port = OvrtxSessionUpdatePort(render, object())

    initial = scheduler.tick_viewport(
        _request(),
        ovrtx_updates=port,
        project_complete_pose=True,
    )
    assert initial.enabled is True
    assert physics.created is True
    assert physics.steps == []
    reset = scheduler.tick_viewport(
        _request(simulation_reset_token=1),
        ovrtx_updates=port,
    )
    assert reset.timeline_reset is True
    assert reset_calls == ["reset"]

    view_plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_camera_edit_target(),
            value=((1.0, 0.0, 0.0, 0.0),) * 4,
        )
    )
    sim_plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.SIM,
            **_edit_target(),
            value={
                "translate": (1.0, 2.0, 3.0),
                "orient": (0.0, 0.0, 0.0, 1.0),
            },
        )
    )
    assert scheduler.submit_edit(view_plan.to_intent()).accepted
    assert scheduler.submit_edit(sim_plan.to_intent()).accepted
    applied = scheduler.tick_viewport(_request(), ovrtx_updates=port)

    assert applied.status != RuntimeTickStatus.FAILED
    assert retained_transforms == [
        (
            OvrtxTransformValue(
                "/World/Camera",
                [[1.0, 0.0, 0.0, 0.0]] * 4,
            ),
        )
    ]
    assert retained_poses == [
        (
            BodyPose(
                "/World/PhysicsIsland/DynamicBodies/Cube_00",
                (1.0, 2.0, 3.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
    ]
    assert scheduler.diagnostics()["sim_updates"]["value_count"] == 0

    scheduler.shutdown()
    assert physics.shutdown_called is False
    controller.shutdown()
    assert physics.shutdown_called is True


def test_scheduler_routes_view_values_without_starting_physics() -> None:
    retained: list[tuple[OvrtxTransformValue, ...]] = []
    scheduler = RuntimeScheduler(
        controller_provider=lambda: None,
        ovrtx_transform_sink=retained.append,
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_camera_edit_target(),
            value=((1.0, 0.0, 0.0, 0.0),) * 4,
        )
    )

    assert scheduler.submit_edit(plan.to_intent()).accepted
    result = scheduler.tick_viewport(
        _request(),
        ovrtx_updates=OvrtxSessionUpdatePort(_FakeRenderClient(), object()),
    )

    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    assert retained == [
        (
            OvrtxTransformValue(
                "/World/Camera",
                [[1.0, 0.0, 0.0, 0.0]] * 4,
            ),
        )
    ]


def test_borrowed_physics_reset_failure_is_not_retried() -> None:
    physics = _FakePhysicsClient()
    controller = OvphysxStageController(
        _config(),
        physics_client=physics,
        simulation_id="borrowed",
    )
    reset_calls: list[str] = []
    scheduler = RuntimeScheduler(
        controller_provider=lambda: controller,
        controller_reset=lambda: (reset_calls.append("reset") or False),
    )
    port = OvrtxSessionUpdatePort(_FakeRenderClient(), object())
    scheduler.tick_viewport(_request(), ovrtx_updates=port, project_complete_pose=True)

    failed = scheduler.tick_viewport(
        _request(simulation_reset_token=1),
        ovrtx_updates=port,
    )
    scheduler.tick_viewport(
        _request(simulation_reset_token=1),
        ovrtx_updates=port,
    )

    assert failed.status == RuntimeTickStatus.FAILED
    assert failed.skipped_reason == "ovphysx_reset_failed"
    assert reset_calls == ["reset"]
    controller.shutdown()


def test_runtime_tick_request_rejects_removed_path_field() -> None:
    removed_field = "fixture" + "_path"

    with pytest.raises(TypeError):
        RuntimeTickRequest(
            **{
                removed_field: "/fixture.usda",
            }
        )


def test_timeline_controls_keep_configured_limit_when_disabled() -> None:
    request = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=False,
        timeline_start=1,
        timeline_end=24,
    )

    assert _timeline_max_steps(request, steps_per_update=5, configured_max_steps=240) == 240


def test_timeline_reset_on_token_change_or_backward_frame() -> None:
    request = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_frame=10,
        simulation_reset_token=2,
    )
    backward = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_frame=4,
        simulation_reset_token=1,
    )

    assert _timeline_should_reset(request, last_frame=10, last_reset_token=1) is True
    assert _timeline_should_reset(backward, last_frame=10, last_reset_token=1) is True
    assert _timeline_should_reset(backward, last_frame=4, last_reset_token=1) is False


def test_timeline_step_when_playing_or_frame_advanced() -> None:
    paused = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_playing=False,
        timeline_frame=10,
    )
    playing = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_playing=True,
        timeline_frame=10,
    )
    advanced = RuntimeTickRequest(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_playing=False,
        timeline_frame=11,
    )

    assert _timeline_should_step(paused, last_frame=10, controller_started=True) is False
    assert _timeline_should_step(playing, last_frame=10, controller_started=True) is True
    assert _timeline_should_step(advanced, last_frame=10, controller_started=True) is True
    assert _timeline_should_step(paused, last_frame=10, controller_started=False) is True


def test_scheduler_paused_timeline_steps_only_on_first_or_advanced_frame() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        initial = _tick(scheduler, render, session, timeline_frame=1)
        same_frame = _tick(scheduler, render, session, timeline_frame=1)
        advanced = _tick(scheduler, render, session, timeline_frame=2)
    finally:
        scheduler.shutdown()

    assert initial.status == RuntimeTickStatus.INITIAL
    assert same_frame.status == RuntimeTickStatus.NOOP
    assert advanced.status == RuntimeTickStatus.STEPPED
    assert physics.steps == [_config().timestep_ns]
    assert initial.should_reset_refinement is True
    assert same_frame.should_reset_refinement is False
    assert advanced.should_reset_refinement is True
    assert len(render.transform_updates) == 2


def test_scheduler_playback_applies_latest_background_pose() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    config = _config(update_fps=1000.0, max_steps=2)
    scheduler = _scheduler(config, physics)
    try:
        initial = _tick(scheduler, render, session, timeline_playing=True, timeline_frame=1, timeline_end=2)
        _wait_for(lambda: scheduler.diagnostics()["pose_publication_complete_count"] >= 1)
        applied = _tick(
                scheduler,
                render,
                session,
                now_ns=config.update_interval_ns,
                timeline_playing=True,
                timeline_frame=1,
                timeline_end=2,
        )
        diagnostics = scheduler.diagnostics()
    finally:
        scheduler.shutdown()

    assert initial.status == RuntimeTickStatus.INITIAL
    assert applied.status == RuntimeTickStatus.PLAYBACK_ADVANCED
    assert applied.should_reset_refinement is True
    assert diagnostics["latest_pose_publication_sequence"] >= 1
    assert diagnostics["async_publication"]["applied_pose_publication_sequence"] >= 1
    assert len(render.transform_updates) == 2


def test_scheduler_omits_pose_set_when_async_pose_time_is_stale() -> None:
    physics = _IncompleteAfterInitialPhysicsClient()
    render = _FakeRenderClient()
    config = _config(update_fps=1000.0, max_steps=1)
    scheduler = _scheduler(config, physics)
    try:
        initial = _tick(
            scheduler,
            render,
            object(),
            timeline_playing=True,
            timeline_frame=1,
            timeline_end=1,
        )
        _wait_for(lambda: scheduler.diagnostics()["pose_read_incomplete_count"] >= 1)
        completed = _tick(
            scheduler,
            render,
            object(),
            now_ns=config.update_interval_ns,
            timeline_playing=True,
            timeline_frame=1,
            timeline_end=1,
        )
    finally:
        scheduler.shutdown()

    assert initial.status == RuntimeTickStatus.INITIAL
    assert initial.simulation_time_ns == 0
    assert len(initial.physics_pose_set) == 1
    assert completed.status == RuntimeTickStatus.COMPLETED
    assert completed.simulation_time_ns == config.timestep_ns
    assert completed.physics_pose_set == ()


def test_scheduler_paused_reuse_invalidates_async_generation_without_stepping() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    config = _config(update_fps=1000.0, max_steps=3)
    scheduler = _scheduler(config, physics)
    try:
        _tick(scheduler, render, session, timeline_playing=True, timeline_frame=1, timeline_end=3)
        paused = _tick(
                scheduler,
                render,
                session,
                now_ns=config.update_interval_ns,
                timeline_playing=False,
                timeline_frame=1,
                timeline_end=3,
        )
        diagnostics = scheduler.diagnostics()
    finally:
        scheduler.shutdown()

    assert paused.status == RuntimeTickStatus.NOOP
    assert paused.should_reset_refinement is False
    assert diagnostics["composition_generation"] == 1
    assert diagnostics["playback_intent_generation"] == 1
    assert diagnostics["stale_generation_drop_count"] == 0


def test_scheduler_stable_pose_does_not_reset_refinement() -> None:
    physics = _StablePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        initial = _tick(scheduler, render, session, timeline_frame=1)
        unchanged = _tick(scheduler, render, session, timeline_frame=2)
        diagnostics = scheduler.diagnostics()
    finally:
        scheduler.shutdown()

    assert initial.should_reset_refinement is True
    assert unchanged.status == RuntimeTickStatus.STEPPED
    assert unchanged.stage_changed is False
    assert unchanged.values_written is False
    assert unchanged.should_reset_refinement is False
    assert diagnostics["pose_projection_application_count"] == 1
    assert len(render.transform_updates) == 1


def test_scheduler_reports_busy_from_injected_controller() -> None:
    controller = _BusyThenSteppedController(_config(max_steps=4))
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: controller.config,
        controller_factory=lambda _config: controller,
    )
    render = _FakeRenderClient()
    try:
        busy = _tick(scheduler, render, object(), timeline_frame=1)
        retried = _tick(scheduler, render, object(), timeline_frame=1)
    finally:
        scheduler.shutdown()

    assert busy.status == RuntimeTickStatus.BUSY
    assert busy.skipped_reason == "in_progress"
    assert busy.should_reset_refinement is False
    assert busy.physics_pose_set == ()
    assert retried.status == RuntimeTickStatus.INITIAL
    assert [pose.prim_path for pose in retried.physics_pose_set] == [
        "/World/PhysicsIsland/DynamicBodies/Cube_00"
    ]
    assert controller.sync_calls == 2


def test_scheduler_startup_failure_returns_empty_physics_pose_set() -> None:
    physics = _ThrowingStartPhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        result = _tick(scheduler, _FakeRenderClient(), object(), timeline_frame=1)
    finally:
        scheduler.shutdown()

    assert result.status == RuntimeTickStatus.FAILED
    assert result.skipped_reason == "physics_startup_error"
    assert result.physics_pose_set == ()


def test_scheduler_maps_malformed_startup_pose_to_failed_status() -> None:
    scheduler = _scheduler(_config(max_steps=4), _MalformedInitialPosePhysicsClient())
    try:
        result = _tick(scheduler, _FakeRenderClient(), object(), timeline_frame=1)
    finally:
        scheduler.shutdown()

    assert result.status == RuntimeTickStatus.FAILED
    assert result.skipped_reason == "pose_publication_error"
    assert result.physics_pose_set == ()


def test_scheduler_maps_view_value_failure_to_failed_status() -> None:
    physics = _FakePhysicsClient()
    render = _FailingRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        failed = _tick(scheduler, render, session, timeline_frame=1)
        after_failure = _tick(scheduler, render, session, timeline_frame=2)
        diagnostics = scheduler.diagnostics()
    finally:
        scheduler.shutdown()

    assert failed.status == RuntimeTickStatus.FAILED
    assert failed.skipped_reason == "pose_projection_application_error"
    assert failed.should_reset_refinement is True
    assert after_failure.status == RuntimeTickStatus.FAILED
    assert after_failure.update["reason"] == "step"
    assert diagnostics["status"] == "running"
    assert "native transform value write rejected" in diagnostics[
        "last_pose_projection_application"
    ]["result"]["error"]


def test_scheduler_maps_returned_value_error_to_failed_status() -> None:
    physics = _FakePhysicsClient()
    render = _ErrorResultRenderClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        failed = _tick(scheduler, render, object(), timeline_frame=1)
        diagnostics = scheduler.diagnostics()
    finally:
        scheduler.shutdown()

    assert failed.status == RuntimeTickStatus.FAILED
    assert failed.values_written is False
    assert failed.skipped_reason == "pose_projection_application_error"
    assert diagnostics["status"] == "running"
    assert diagnostics["pose_projection_application_failure_count"] == 1


def test_scheduler_reports_completion_after_max_steps() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    config = _config(max_steps=1)
    scheduler = _scheduler(config, physics)
    try:
        _tick(scheduler, render, session, now_ns=0, timeline_controls_enabled=False)
        stepped = _tick(scheduler, render, session, now_ns=config.update_interval_ns, timeline_controls_enabled=False)
        completed = _tick(scheduler, render, session, now_ns=2 * config.update_interval_ns, timeline_controls_enabled=False)
    finally:
        scheduler.shutdown()

    assert stepped.status == RuntimeTickStatus.STEPPED
    assert completed.status == RuntimeTickStatus.COMPLETED
    assert completed.skipped_reason == "max_steps"
    assert completed.should_request_redraw is False
    assert physics.steps == [_config().timestep_ns]


def test_pose_publication_fires_scheduler_wake_hook() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    scheduler = _scheduler(_config(max_steps=1), physics)
    wakes: list[int] = []
    scheduler.set_edit_wake_hook(lambda: wakes.append(1))
    try:
        _tick(
            scheduler,
            render,
            object(),
            timeline_playing=True,
        )
        _wait_for(lambda: bool(wakes))
    finally:
        scheduler.set_edit_wake_hook(None)
        scheduler.shutdown()

    assert wakes == [1]


def test_scheduler_replaces_physics_on_timeline_reset_with_observable_lifecycle() -> None:
    physics_clients: list[_FakePhysicsClient] = []

    def controller_factory(config: InteractiveSharedStageConfig) -> OvphysxStageController:
        physics = _FakePhysicsClient()
        physics_clients.append(physics)
        return OvphysxStageController(
            config,
            physics_client=physics,
            simulation_id=f"sim-{len(physics_clients)}",
        )

    render = _FakeRenderClient()
    session = object()
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: _config(max_steps=4),
        controller_factory=controller_factory,
    )
    try:
        initial = _tick(scheduler, render, session, timeline_frame=3, simulation_reset_token=1)
        reset = _tick(scheduler, render, session, timeline_frame=2, simulation_reset_token=1)

        assert initial.generation == 0
        assert reset.timeline_reset is True
        assert reset.status == RuntimeTickStatus.INITIAL
        assert reset.generation == initial.generation
        assert len(physics_clients) == 2
        assert physics_clients[0].shutdown_called is True
        assert physics_clients[1].created is True
        assert physics_clients[1].started is True
        assert scheduler.diagnostics()["ovphysx_simulation_reuse"] == {
            "reuse": False,
            "reason": "explicit_reset",
        }
    finally:
        scheduler.shutdown()


def test_scheduler_accepts_resolved_view_update() -> None:
    scheduler = RuntimeScheduler()
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=((1.0, 0.0, 0.0, 0.0),),
        )
    )

    result = scheduler.submit_edit(plan.to_intent())

    assert result.accepted is True
    assert result.reason == "queued"
    assert result.physics_generation_reset is False
    assert result.diagnostics["data_authority"] == "view"
    assert result.diagnostics["physics_generation_reset"] is False
    assert result.diagnostics["target"]["usd_layer_id"] == "/layers/scene.usda"
    assert result.diagnostics["target"]["usd_prim_path"] == "/World/PhysicsIsland/DynamicBodies/Cube_00"
    assert result.diagnostics["target"]["provenance"] == {"source": "test"}
    assert "plan_impact" not in result.diagnostics
    assert result.diagnostics["queued"] is True
    diagnostics = scheduler.diagnostics()
    assert diagnostics["enabled"] is False
    assert diagnostics["last_edit_update"]["queued"] is True


def test_scheduler_accepts_resolved_sim_value_update() -> None:
    scheduler = RuntimeScheduler()
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.SIM,
            **_edit_target(),
            value={"translate": (1.0, 2.0, 3.0), "orient": (0.0, 0.0, 0.0, 1.0)},
        )
    )

    result = scheduler.submit_edit(plan.to_intent())

    assert result.accepted is True
    assert result.reason == "queued"
    assert result.physics_generation_reset is False
    assert result.diagnostics["shape"] == "value"
    assert result.diagnostics["data_authority"] == "sim"
    assert result.diagnostics["physics_generation_reset"] is False


def test_pending_view_and_sim_values_are_bounded_latest_wins() -> None:
    scheduler = RuntimeScheduler()
    planner = InteractiveEditPlanner()

    for translation in (1.0, 2.0):
        view = planner.plan(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.VIEW,
                **_edit_target(),
                value=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (translation, 0.0, 0.0, 1.0),
                ),
            )
        )
        sim = planner.plan(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.SIM,
                **_edit_target(),
                value={
                    "translate": (translation, 0.0, 0.0),
                    "orient": (0.0, 0.0, 0.0, 1.0),
                },
            )
        )
        assert scheduler.submit_edit(view.to_intent()).accepted is True
        assert scheduler.submit_edit(sim.to_intent()).accepted is True

    render = _FakeRenderClient()

    class Controller:
        applied: list[BodyPose] = []

        def apply_initial_condition_values(
            self,
            poses: tuple[BodyPose, ...],
            *,
            reset: bool,
        ) -> OvphysxStageResult:
            assert reset is False
            self.applied.extend(poses)
            return OvphysxStageResult(
                OvphysxStageStatus.OK,
                "updated",
                poses,
                tuple(pose.prim_path for pose in poses),
                0,
                0,
                1,
            )

    controller = Controller()
    view_result = scheduler.apply_pending_view_values(
        OvrtxSessionUpdatePort(render, object())
    )
    sim_result = scheduler.apply_pending_sim_values(controller)

    assert view_result.values_written is True
    assert sim_result.values_written is True
    assert len(render.transform_updates) == 1
    assert render.transform_updates[0][0].matrix[3][0] == 2.0
    assert [pose.translate for pose in controller.applied] == [(2.0, 0.0, 0.0)]


def test_scheduler_submits_a_group_atomically_with_one_revision() -> None:
    scheduler = RuntimeScheduler()
    planner = InteractiveEditPlanner()
    supported = planner.plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0, 1.0),
            ),
        )
    ).to_intent()
    unsupported = EditIntent(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Unsupported",
            usd_attribute="unknown:value",
            blender_property_path="unknown",
        ),
        value=1,
    )

    revision = scheduler.presentation_revision
    rejected = scheduler.submit_edits((supported, unsupported))

    assert all(result.accepted is False for result in rejected)
    assert scheduler.has_pending_view_updates is False
    assert scheduler.presentation_revision == revision

    second = replace(
        supported,
        usd_prim_path="/World/Second",
        usd_property_path="/World/Second.xformOp:transform",
    )
    accepted = scheduler.submit_edits((supported, second))
    render = _FakeRenderClient()
    applied = scheduler.apply_pending_view_values(
        OvrtxSessionUpdatePort(render, object())
    )

    assert all(result.accepted is True for result in accepted)
    assert scheduler.presentation_revision == revision + 1
    assert applied.values_written is True
    assert [value.prim_path for value in render.transform_updates[0]] == [
        "/World/PhysicsIsland/DynamicBodies/Cube_00",
        "/World/Second",
    ]


def test_retryable_sim_application_remains_pending() -> None:
    scheduler = RuntimeScheduler()
    intent = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.SIM,
            **_edit_target(),
            value={
                "translate": (1.0, 2.0, 3.0),
                "orient": (0.0, 0.0, 0.0, 1.0),
            },
        )
    ).to_intent()

    class Controller:
        calls = 0

        def apply_initial_condition_values(
            self,
            poses: tuple[BodyPose, ...],
            *,
            reset: bool,
        ) -> OvphysxStageResult:
            assert reset is False
            self.calls += 1
            status = (
                OvphysxStageStatus.BUSY
                if self.calls == 1
                else OvphysxStageStatus.OK
            )
            return OvphysxStageResult(
                status,
                "in_progress" if status == OvphysxStageStatus.BUSY else "updated",
                poses,
                tuple(pose.prim_path for pose in poses),
                0,
                0,
                1,
            )

    controller = Controller()
    assert scheduler.submit_edit(intent).accepted is True

    busy = scheduler.apply_pending_sim_values(controller)
    applied = scheduler.apply_pending_sim_values(controller)

    assert busy.status == RuntimeTickStatus.BUSY
    assert busy.values_written is False
    assert applied.status == RuntimeTickStatus.NOOP
    assert applied.values_written is True
    assert scheduler.has_pending_sim_updates is False


def test_scheduler_applies_pending_view_value_update() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: _config(enabled=False))
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=matrix,
        )
    )

    scheduler.submit_edit(plan.to_intent())
    result = _tick(scheduler, render, object(), timeline_controls_enabled=False)

    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    assert result.should_reset_refinement is True
    assert render.transform_updates == [[
        OvrtxTransformValue(
            "/World/PhysicsIsland/DynamicBodies/Cube_00",
            [list(row) for row in matrix],
        )
    ]]
    update_result = result.update["update_result"]
    assert update_result["values_written"] is True
    assert update_result["value_paths"] == ["/World/PhysicsIsland/DynamicBodies/Cube_00"]
    assert scheduler.diagnostics()["last_edit_update"]["values_written"] is True


def test_scheduler_applies_pending_camera_value_before_initial_playback_step() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=4), physics)
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_camera_edit_target(),
            value=matrix,
        )
    )

    try:
        submission_result = scheduler.submit_edit(plan.to_intent())
        result = _tick(scheduler, render, session, timeline_controls_enabled=False)

        assert submission_result.status == EditStatus.QUEUED
        assert submission_result.accepted is True
        assert result.values_written is True
        assert result.should_reset_refinement is True
        assert render.transform_updates[0] == [
            OvrtxTransformValue("/World/Camera", [list(row) for row in matrix])
        ]
        update_result = result.update["update_result"]
        assert update_result["values_written"] is True
        view_update = next(
            update for update in update_result["updates"]
            if update.get("data_authority") == "view"
        )
        assert view_update["physics_generation_reset"] is False
        assert view_update["value_paths"] == ["/World/Camera"]
    finally:
        scheduler.shutdown()


def test_scheduler_applies_pending_material_value_update_through_ovrtx_updates() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: _config(enabled=False))
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **_material_edit_target(),
            value=(0.1, 0.2, 0.3, 1.0),
        )
    )

    submission_result = scheduler.submit_edit(plan.to_intent())
    result = _tick(scheduler, render, object(), timeline_controls_enabled=False)

    assert submission_result.status == EditStatus.QUEUED
    assert submission_result.accepted is True
    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    assert result.should_reset_refinement is True
    assert render.material_updates == []
    assert render.attribute_updates == [[
        OvrtxAttributeValue(
            "/World/Looks/Paint/Shader",
            "inputs:diffuseColor",
            [0.1, 0.2, 0.3],
            "Color3f",
        )
    ]]
    update_result = result.update["update_result"]
    assert update_result["values_written"] is True
    assert update_result["data_authority"] == "view"
    assert update_result["value_requested_count"] == 1
    assert update_result["accepted_by_worker"] is True
    assert scheduler.diagnostics()["last_edit_update"]["values_written"] is True


def test_scheduler_applies_pending_uv_value_without_physics_reset() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: _config(enabled=False))
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Quad",
                usd_attribute="primvars:st",
                blender_property_path="uv_layers.active",
                provenance={
                    "source": "test",
                    "uv_loop_order_validation": {
                        "status": "resolved",
                        "validation_kind": "uv_loop_order",
                        "mesh_prim_path": "/World/Quad",
                        "target_attribute": "primvars:st",
                        "value_type": "Float2Array",
                        "uv_layer_name": "UVMap",
                        "interpolation": "faceVarying",
                        "indexed": False,
                        "primvar_shape_status": "resolved",
                        "element_count": 4,
                        "topology_fingerprint": "test-topology",
                        "blender_uv_digest": "test-blender-digest",
                        "source_uv_digest": "test-usd-digest",
                        "tolerance": 1.0e-6,
                    },
                },
            ),
            value=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),
            metadata={"usd_value_type": "Float2Array"},
        )
    )

    submission_result = scheduler.submit_edit(plan.to_intent())
    result = _tick(scheduler, render, object(), timeline_controls_enabled=False)

    assert submission_result.status == EditStatus.QUEUED
    assert submission_result.accepted is True
    assert submission_result.physics_generation_reset is False
    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    assert result.should_reset_refinement is True
    assert render.attribute_updates == [[
        OvrtxAttributeValue(
            "/World/Quad",
            "primvars:st",
            [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
            "Float2Array",
        )
    ]]
    update_result = result.update["update_result"]
    assert update_result["data_authority"] == "view"
    assert update_result["physics_generation_reset"] is False


def test_scheduler_applies_pending_light_value_update_through_ovrtx_updates() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: _config(enabled=False))
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Key/KeyLight",
                usd_attribute="inputs:intensity",
                blender_property_path="energy",
                provenance={"source": "test"},
            ),
            value=900.0,
            metadata={"usd_value_type": "Float"},
        )
    )

    submission_result = scheduler.submit_edit(plan.to_intent())
    result = _tick(scheduler, render, object(), timeline_controls_enabled=False)

    assert submission_result.status == EditStatus.QUEUED
    assert submission_result.accepted is True
    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    assert result.should_reset_refinement is True
    assert render.attribute_updates == [[
        OvrtxAttributeValue("/World/Key/KeyLight", "inputs:intensity", 900.0, "Float")
    ]]
    update_result = result.update["update_result"]
    assert update_result["values_written"] is True
    assert update_result["data_authority"] == "view"
    assert update_result["value_requested_count"] == 1
    assert update_result["accepted_by_worker"] is True


def test_scheduler_reports_runtime_transform_apply_failure() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: _config(enabled=False))
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=(1.0, 2.0),
        )
    )

    scheduler.submit_edit(plan.to_intent())
    result = _tick(scheduler, render, object(), timeline_controls_enabled=False)

    assert result.status == RuntimeTickStatus.FAILED
    assert result.skipped_reason == "view_value_update_error"
    assert result.values_written is False
    assert render.transform_updates == []
    update_result = result.update["update_result"]
    assert update_result["values_written"] is False
    assert update_result["failed"] is True
    assert "ValueError" in update_result["result"]["error"]


def test_scheduler_applies_pending_sim_value_update_before_playback_step() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=4), physics)
    planner = InteractiveEditPlanner()
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 7.0, 4.0, 1.0),
    )
    plan = planner.plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.SIM,
            **_edit_target(),
            value=matrix,
        )
    )

    try:
        _tick(scheduler, render, session, timeline_frame=1, simulation_reset_token=1)
        submission_result = scheduler.submit_edit(plan.to_intent())
        result = _tick(
                scheduler,
                render,
                session,
                timeline_controls_enabled=True,
                timeline_playing=True,
                timeline_frame=1,
                timeline_start=1,
                simulation_reset_token=1,
        )
        assert submission_result.status == EditStatus.QUEUED
        assert submission_result.accepted is True
        assert result.status in {RuntimeTickStatus.PLAYBACK_ADVANCED, RuntimeTickStatus.REUSED_LATEST}
        assert physics.pose_writes
        assert physics.pose_writes[0]["simulation_time_ns"] == 1
        assert physics.pose_writes[0]["reset"] is False
        written_pose = physics.pose_writes[0]["poses"][0]
        assert written_pose.translate == (2.0, 7.0, 4.0)
        assert written_pose.orient == (0.0, 0.0, 0.0, 1.0)
        _wait_for(lambda: bool(physics.steps))
        assert physics.steps
        assert physics.pose_writes[0]["simulation_time_ns"] < physics.steps[0]
        update_result = result.update["update_result"]
        assert update_result["values_written"] is True
        sim_update = next(
            update for update in update_result["updates"]
            if update.get("data_authority") == "sim"
        )
        assert sim_update["sim_value_paths"] == ["/World/PhysicsIsland/DynamicBodies/Cube_00"]
        assert sim_update["physics_reset"] is False
        assert submission_result.physics_generation_reset is False
    finally:
        scheduler.shutdown()


def test_scheduler_replays_runtime_sim_value_after_timeline_wrap() -> None:
    physics = _FakePhysicsClient()
    render = _FakeRenderClient()
    session = object()
    scheduler = _scheduler(_config(max_steps=8), physics)
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 7.0, 4.0, 1.0),
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.SIM,
            **_edit_target(),
            value=matrix,
        )
    )

    try:
        _tick(scheduler, render, session, timeline_frame=0, timeline_start=0, timeline_end=4, simulation_reset_token=1)
        scheduler.submit_edit(plan.to_intent())
        edit_result = _tick(scheduler, render, session, timeline_frame=0, timeline_start=0, timeline_end=4, simulation_reset_token=1)
        _tick(
                scheduler,
                render,
                session,
                timeline_playing=True,
                timeline_frame=4,
                timeline_start=0,
                timeline_end=4,
                simulation_reset_token=1,
        )
        render_update_count_before_reset = len(render.transform_updates)
        loop_result = _tick(
                scheduler,
                render,
                session,
                timeline_playing=False,
                timeline_frame=0,
                timeline_start=0,
                timeline_end=4,
                simulation_reset_token=1,
        )
        assert edit_result.update["update_result"]["values_written"] is True
        assert loop_result.timeline_reset is True
        reset_render_updates = render.transform_updates[render_update_count_before_reset:]
        assert len(reset_render_updates) == 1
        assert [
            value.prim_path
            for update in reset_render_updates
            for value in update
            if value.prim_path == "/World/PhysicsIsland/DynamicBodies/Cube_00"
        ] == ["/World/PhysicsIsland/DynamicBodies/Cube_00"]
        assert len(physics.pose_writes) >= 2
        replayed_pose = physics.pose_writes[-1]["poses"][0]
        assert replayed_pose.translate == (2.0, 7.0, 4.0)
        replay_update = next(
            update for update in loop_result.update["update_result"]["updates"]
            if update.get("reason") == "initial_condition_values"
        )
        assert replay_update["value_requested_count"] == 1
    finally:
        scheduler.shutdown()


def test_scheduler_reuses_physics_for_non_spec_config_change() -> None:
    physics = _FakePhysicsClient()
    config = _config(update_fps=30.0)
    current = {"config": config}
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: current["config"],
        controller_factory=lambda controller_config: OvphysxStageController(
            controller_config,
            physics_client=physics,
            simulation_id="sim",
        ),
    )
    render = _FakeRenderClient()
    try:
        _tick(scheduler, render, object(), timeline_frame=1)
        current["config"] = _config(update_fps=120.0, max_steps=99)
        _tick(scheduler, render, object(), timeline_frame=2)

        assert physics.shutdown_called is False
        assert scheduler.diagnostics()["ovphysx_simulation_reuse"] == {
            "reuse": True,
            "reason": "same_simulation",
        }
    finally:
        scheduler.shutdown()


def test_complete_pose_projection_precedes_pending_view_edit_once() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    first_render = _FakeRenderClient()
    replacement_render = _FakeRenderClient()
    view_matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (9.0, 8.0, 7.0, 1.0),
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=view_matrix,
        )
    )
    try:
        _tick(scheduler, first_render, object(), timeline_frame=1)
        scheduler.submit_edit(plan.to_intent())
        request = _request(timeline_frame=1)
        result = scheduler.tick_viewport(
            request,
            ovrtx_updates=OvrtxSessionUpdatePort(replacement_render, object()),
            project_complete_pose=True,
        )

        assert result.status == RuntimeTickStatus.NOOP
        assert result.should_reset_refinement is True
        assert len(replacement_render.transform_updates) == 2
        assert replacement_render.transform_updates[0][0].matrix[3][1] == 5.0
        assert replacement_render.transform_updates[1][0].matrix == [list(row) for row in view_matrix]
        diagnostics = scheduler.diagnostics()
        assert diagnostics["last_pose_projection_application"]["reason"] == "complete_pose_projection"
        assert result.complete_pose_projected is True
        assert diagnostics["last_edit_update"]["values_written"] is True
    finally:
        scheduler.shutdown()


def test_complete_pose_projection_starts_replacement_physics_before_view_edit() -> None:
    physics_clients: list[_FakePhysicsClient] = []

    def controller_factory(config: InteractiveSharedStageConfig) -> OvphysxStageController:
        physics = _FakePhysicsClient()
        physics_clients.append(physics)
        return OvphysxStageController(
            config,
            physics_client=physics,
            simulation_id=f"sim-{len(physics_clients)}",
        )

    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: _config(max_steps=4),
        controller_factory=controller_factory,
    )
    first_render = _FakeRenderClient()
    replacement_render = _FakeRenderClient()
    view_matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (9.0, 8.0, 7.0, 1.0),
    )
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value=view_matrix,
        )
    )
    try:
        _tick(scheduler, first_render, object(), timeline_frame=1, simulation_reset_token=0)
        scheduler.submit_edit(plan.to_intent())

        result = scheduler.tick_viewport(
            _request(timeline_frame=1, simulation_reset_token=1),
            ovrtx_updates=OvrtxSessionUpdatePort(replacement_render, object()),
            project_complete_pose=True,
        )

        assert result.timeline_reset is True
        assert len(physics_clients) == 2
        assert physics_clients[1].started is True
        assert replacement_render.transform_updates[0][0].matrix[3][1] == 5.0
        assert replacement_render.transform_updates[1][0].matrix == [list(row) for row in view_matrix]
    finally:
        scheduler.shutdown()


def test_failed_replacement_projection_does_not_repeat_explicit_reset() -> None:
    physics_clients: list[_FakePhysicsClient] = []

    def controller_factory(config: InteractiveSharedStageConfig) -> OvphysxStageController:
        physics = _FakePhysicsClient()
        physics_clients.append(physics)
        return OvphysxStageController(
            config,
            physics_client=physics,
            simulation_id=f"sim-{len(physics_clients)}",
        )

    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: _config(max_steps=4),
        controller_factory=controller_factory,
    )
    try:
        _tick(scheduler, _FakeRenderClient(), object(), simulation_reset_token=0)
        failed = scheduler.tick_viewport(
            _request(simulation_reset_token=1),
            ovrtx_updates=OvrtxSessionUpdatePort(_FailingRenderClient(), object()),
            project_complete_pose=True,
        )

        retry = scheduler.tick_viewport(
            _request(simulation_reset_token=1),
            ovrtx_updates=OvrtxSessionUpdatePort(_FakeRenderClient(), object()),
            project_complete_pose=True,
        )

        assert failed.status == RuntimeTickStatus.FAILED
        assert failed.skipped_reason == "pose_projection_application_error"
        assert failed.complete_pose_projected is False
        assert len(physics_clients) == 2
        assert retry.timeline_reset is False
        assert scheduler.diagnostics()["last_pose_projection_application"]["reason"] == "complete_pose_projection"
    finally:
        scheduler.shutdown()


def test_complete_pose_projection_failure_preserves_physics_controller() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    try:
        _tick(scheduler, _FakeRenderClient(), object(), timeline_frame=1)
        plan = InteractiveEditPlanner().plan(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.VIEW,
                **_edit_target(),
                value=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (9.0, 8.0, 7.0, 1.0),
                ),
            )
        )
        scheduler.submit_edit(plan.to_intent())
        result = scheduler.tick_viewport(
            _request(timeline_frame=1),
            ovrtx_updates=OvrtxSessionUpdatePort(_FailingRenderClient(), object()),
            project_complete_pose=True,
        )

        assert result.status == RuntimeTickStatus.FAILED
        assert result.skipped_reason == "pose_projection_application_error"
        assert result.complete_pose_projected is False
        assert physics.shutdown_called is False
        assert scheduler._view_updates.has_pending is True
        assert scheduler.diagnostics()["last_pose_projection_application"]["reason"] == "complete_pose_projection"

        replacement_render = _FakeRenderClient()
        retry = scheduler.tick_viewport(
            _request(timeline_frame=1),
            ovrtx_updates=OvrtxSessionUpdatePort(replacement_render, object()),
            project_complete_pose=True,
        )
        assert retry.status == RuntimeTickStatus.NOOP
        assert scheduler._view_updates.has_pending is False
        assert len(replacement_render.transform_updates) == 2
    finally:
        scheduler.shutdown()


def test_scheduler_does_not_accept_persistence_plan_as_update() -> None:
    scheduler = RuntimeScheduler()
    export_plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.TOPOLOGY,

            data_authority=DataAuthority.VIEW,
            **_edit_target(),
            value="new collider",
        )
    )

    result = scheduler.submit_edit(export_plan)  # type: ignore[arg-type]

    assert export_plan.persistence == EditPersistence.WRITE
    assert result.accepted is False
    assert result.reason == "unsupported:unresolved_edit"


def test_scheduler_applies_fresh_velocity_while_paused_without_ovrtx_projection() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    render = _FakeRenderClient()
    try:
        assert scheduler.submit_edit(_velocity_intent()).accepted

        result = _tick(scheduler, render, object(), timeline_playing=False)

        assert result.status == RuntimeTickStatus.NOOP
        assert result.values_written is True
        assert result.simulation_time_ns == physics.velocity_writes[0]["simulation_time_ns"]
        assert result.update["physics_generation_reset"] is False
        assert result.update["transform_updated"] is False
        assert result.update["render_value_write_applied"] is False
        assert len(physics.velocity_writes) == 1
        assert render.transform_updates == []
    finally:
        scheduler.shutdown()


def test_scheduler_replays_retained_velocity_once_and_adopts_replay_result() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    render = _FakeRenderClient()
    try:
        scheduler.submit_edit(_velocity_intent())
        _tick(scheduler, render, object(), timeline_playing=False)

        replay = _tick(
            scheduler,
            render,
            object(),
            timeline_playing=False,
            simulation_reset_token=1,
        )
        unchanged = _tick(
            scheduler,
            render,
            object(),
            timeline_playing=False,
            simulation_reset_token=1,
        )

        assert replay.status == RuntimeTickStatus.NOOP
        assert replay.simulation_time_ns == physics.velocity_writes[1]["simulation_time_ns"]
        assert replay.update["update_result"]["reason"] == "body_velocity_values"
        assert len(physics.velocity_writes) == 2
        assert unchanged.status == RuntimeTickStatus.NOOP
        assert len(physics.velocity_writes) == 2
        assert render.transform_updates == []
    finally:
        scheduler.shutdown()


def test_scheduler_surfaces_retained_velocity_replay_failure() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    render = _FakeRenderClient()
    try:
        scheduler.submit_edit(_velocity_intent())
        _tick(scheduler, render, object(), timeline_playing=False)
        physics.fail_velocity_writes = True

        replay = _tick(
            scheduler,
            render,
            object(),
            timeline_playing=False,
            simulation_reset_token=1,
        )

        assert replay.status == RuntimeTickStatus.FAILED
        assert replay.skipped_reason == "body_velocity_values_error"
        assert replay.simulation_time_ns == 0
        assert replay.update["update_result"]["result"]["simulation_time_ns"] == 0
        assert len(physics.velocity_writes) == 2
        assert render.transform_updates == []
    finally:
        scheduler.shutdown()


def test_scheduler_applies_paused_velocity_before_pause_invalidation() -> None:
    physics = _FakePhysicsClient()
    scheduler = _scheduler(_config(max_steps=4), physics)
    render = _FakeRenderClient()
    try:
        scheduler.submit_edit(_velocity_intent((3.0, 0.0, 0.0)))

        result = _tick(scheduler, render, object(), timeline_playing=False)

        assert result.status == RuntimeTickStatus.NOOP
        assert physics.write_order == ["velocity"]
        assert scheduler._controller is not None
        assert scheduler._controller.playback_intent_generation >= 1
        assert result.generation == scheduler._controller.composition_generation
        assert render.transform_updates == []
    finally:
        scheduler.shutdown()
