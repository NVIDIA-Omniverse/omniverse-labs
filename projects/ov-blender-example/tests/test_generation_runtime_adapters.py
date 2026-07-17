# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.generation_runtime_adapters import (  # noqa: E402
    OvphysxGenerationAdapter,
    OvrtxGenerationAdapter,
)
from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    OvphysxStageResult,
    OvphysxStageStatus,
)
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402


class _Generation:
    def __init__(self, number: int, path: Path) -> None:
        self.number = number
        self.path = path

    def materialize_usd(self) -> str:
        return str(self.path)


class _Port:
    def __init__(self, calls: list[tuple[str, tuple[object, ...]]], fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def update_transforms(self, values: tuple[object, ...]) -> OvrtxValueUpdateResult:
        self.calls.append(("transforms", tuple(values)))
        if self.fail:
            raise RuntimeError("transform replay failed")
        return OvrtxValueUpdateResult(len(values), 1 if values else None)

    def update_attribute_values(self, values: tuple[object, ...]) -> OvrtxValueUpdateResult:
        self.calls.append(("attributes", tuple(values)))
        return OvrtxValueUpdateResult(len(values), 2 if values else None)


class _OvrtxController:
    def __init__(self) -> None:
        self.requests: list[RenderRequest] = []
        self.replays: list[tuple[str, tuple[object, ...]]] = []
        self.fail_replay = False
        self.fail_ensure = False
        self.stop_status = "stopped"

    def ensure(self, request: RenderRequest) -> None:
        self.requests.append(request)
        if self.fail_ensure:
            raise RuntimeError("ensure failed")
        return SimpleNamespace(composition=f"composition-{len(self.requests)}", session_started=True)

    def apply_runtime_updates(self, operation: object) -> object:
        return operation(_Port(self.replays, self.fail_replay), False)

    def deactivate(self) -> str:
        return self.stop_status


def test_ovrtx_adapter_replays_scene_owned_values(tmp_path: Path) -> None:
    controller = _OvrtxController()
    adapter = OvrtxGenerationAdapter(controller)  # type: ignore[arg-type]
    adapter.update_request(RenderRequest(worker_command="worker"))

    first_transform = OvrtxTransformValue("/World/Body", ((1.0,),))
    first_attribute = OvrtxAttributeValue("/World/Body", "inputs:value", 1.0, "Float")
    assert adapter.activate(  # type: ignore[arg-type]
        _Generation(1, tmp_path / "one.usdc"),
        transform_values=(first_transform,),
        attribute_values=(first_attribute,),
    )
    assert adapter.request is not None
    assert adapter.request.input_usd_path == str(tmp_path / "one.usdc")
    assert adapter.composition == "composition-1"
    assert controller.replays == [
        ("transforms", (first_transform,)),
        ("attributes", (first_attribute,)),
    ]

    second_transform = OvrtxTransformValue("/World/Body", ((2.0,),))
    controller.replays.clear()
    assert adapter.activate(  # type: ignore[arg-type]
        _Generation(2, tmp_path / "two.usdc"),
        transform_values=(second_transform,),
    )
    assert controller.replays == [("transforms", (second_transform,))]

    controller.replays.clear()
    assert adapter.activate(_Generation(3, tmp_path / "three.usdc"))  # type: ignore[arg-type]
    assert controller.replays == []


def test_ovrtx_adapter_reuses_active_generation_for_presentation_request_changes(
    tmp_path: Path,
) -> None:
    controller = _OvrtxController()
    adapter = OvrtxGenerationAdapter(controller)  # type: ignore[arg-type]
    adapter.update_request(RenderRequest(input_usd_path="export.usdc", width=10))
    assert adapter.activate(_Generation(1, tmp_path / "authored.usdc"))  # type: ignore[arg-type]

    adapter.update_request(RenderRequest(input_usd_path="new-export.usdc", width=20))
    assert adapter.ensure_request()

    assert controller.requests[-1].input_usd_path == str(tmp_path / "authored.usdc")
    assert controller.requests[-1].width == 20
    assert adapter.composition == "composition-2"


def test_ovrtx_adapter_does_not_retry_failed_presentation_replacement(
    tmp_path: Path,
) -> None:
    controller = _OvrtxController()
    adapter = OvrtxGenerationAdapter(controller)  # type: ignore[arg-type]
    adapter.update_request(RenderRequest(width=10))
    assert adapter.activate(_Generation(1, tmp_path / "authored.usdc"))  # type: ignore[arg-type]
    controller.fail_ensure = True
    adapter.update_request(RenderRequest(width=20))

    assert not adapter.ensure_request()
    assert not adapter.ensure_request()
    assert len(controller.requests) == 2


def test_ovrtx_adapter_replay_failure_never_marks_generation_ready(
    tmp_path: Path,
) -> None:
    controller = _OvrtxController()
    controller.fail_replay = True
    adapter = OvrtxGenerationAdapter(controller)  # type: ignore[arg-type]
    adapter.update_request(RenderRequest(worker_command="worker"))
    assert not adapter.activate(  # type: ignore[arg-type]
        _Generation(1, tmp_path / "scene.usdc"),
        transform_values=(OvrtxTransformValue("/World/Body", ((1.0,),)),),
    )
    assert adapter.active_generation is None
    assert "retained_value_replay_failed" in adapter.last_error


class _PhysicsController:
    def __init__(self, config: object, instances: list["_PhysicsController"]) -> None:
        self.config = config
        self.instances = instances
        self.instances.append(self)
        self.starting_values: tuple[BodyPose, ...] = ()
        self.stop_status = "stopped"

    def tick(
        self,
        *,
        max_steps: int,
        initial_condition_values: tuple[BodyPose, ...],
    ) -> OvphysxStageResult:
        assert max_steps > 0
        self.starting_values = initial_condition_values
        return OvphysxStageResult(
            status=OvphysxStageStatus.OK,
            reason="",
            pose_set=initial_condition_values,
            dirty_paths=tuple(value.prim_path for value in initial_condition_values),
            step_count=0,
            simulation_time_ns=0,
            generation=1,
        )

    def deactivate(self) -> str:
        return self.stop_status


def test_ovphysx_adapter_discovers_arbitrary_roots_prunes_values_and_accepts_empty_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dynamic_path = "/Arbitrary/Nested/Dynamic"
    scene = tmp_path / "physics.usda"
    scene.write_text(
        f'''#usda 1.0
def Xform "Arbitrary"
{{
    def Xform "Nested"
    {{
        def Cube "Dynamic" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
        )
        {{
        }}
    }}
}}
''',
        encoding="utf-8",
    )
    empty = tmp_path / "static.usda"
    empty.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    instances: list[_PhysicsController] = []
    adapter = OvphysxGenerationAdapter(
        lambda config: _PhysicsController(config, instances)
    )
    retained = BodyPose(
        prim_path=dynamic_path,
        translate=(0.0, 0.0, 2.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )
    removed = BodyPose(
        prim_path="/Removed",
        translate=(0.0, 0.0, 0.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )
    assert adapter.activate(  # type: ignore[arg-type]
        _Generation(1, scene),
        initial_conditions=(retained, removed),
    )
    assert adapter.dynamic_body_paths == (dynamic_path,)
    assert instances[-1].config.body_root == ""
    assert instances[-1].starting_values == (retained,)
    assert adapter.activate(_Generation(2, empty))  # type: ignore[arg-type]
    assert adapter.dynamic_body_paths == ()
    assert instances[-1].config.body_prims == ()
    assert instances[-1].starting_values == ()
    assert adapter.physics_generation == 2


def test_ovphysx_adapter_reset_replaces_only_physics_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene = tmp_path / "physics.usda"
    scene.write_text(
        '''#usda 1.0
def Xform "World"
{
    def Cube "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    ) {}
}
''',
        encoding="utf-8",
    )
    instances: list[_PhysicsController] = []
    adapter = OvphysxGenerationAdapter(
        lambda config: _PhysicsController(config, instances)
    )
    generation = _Generation(4, scene)

    assert adapter.activate(generation)  # type: ignore[arg-type]
    assert adapter.reset()

    assert len(instances) == 2
    assert adapter.active_generation == 4
    assert adapter.physics_generation == 2


def test_ovphysx_adapter_preserves_controller_failure_detail(tmp_path: Path) -> None:
    scene = tmp_path / "physics.usda"
    scene.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")

    class _FailingPhysicsController(_PhysicsController):
        def __init__(self, config: object) -> None:
            super().__init__(config, [])
            self.last_error = "native module ABI is incompatible"

        def tick(
            self,
            *,
            max_steps: int,
            initial_condition_values: tuple[BodyPose, ...],
        ) -> OvphysxStageResult:
            return OvphysxStageResult(
                status=OvphysxStageStatus.FAILED,
                reason="physics_startup_error",
                pose_set=(),
                dirty_paths=(),
                step_count=0,
                simulation_time_ns=0,
                generation=0,
            )

    adapter = OvphysxGenerationAdapter(_FailingPhysicsController)

    assert not adapter.activate(_Generation(1, scene))  # type: ignore[arg-type]
    assert adapter.last_error == (
        "physics_startup_error: native module ABI is incompatible"
    )
