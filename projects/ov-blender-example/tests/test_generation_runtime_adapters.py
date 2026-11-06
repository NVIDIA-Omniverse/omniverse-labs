# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


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


def test_healthy_reset_reuses_admitted_controller_and_active_authored_source(tmp_path):
    from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig
    calls = []
    config = InteractiveSharedStageConfig.from_env(str(tmp_path / 'old.usda'))
    controller = SimpleNamespace(started=True, failed=False, closed=False, config=config,
        _physics_client=SimpleNamespace(supports_verified_simulation_reload=True), last_error='')
    def reset_simulation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status=OvphysxStageStatus.OK)
    controller.reset_simulation = reset_simulation
    adapter = OvphysxGenerationAdapter(lambda config: pytest.fail('Healthy reset constructed candidate'))
    adapter._controller = controller
    adapter.active_generation = 2
    adapter._active_scene_generation = _Generation(2, tmp_path / 'accepted.usda')
    adapter._initial_conditions = (BodyPose('/World/Body', (1., 2., 3.), (0., 0., 0., 1.)),)
    assert adapter.reset()
    assert adapter.controller is controller
    assert adapter.physics_generation == 1 and adapter.active_generation == 2
    assert calls[0]['config'].input_usd_path == str(tmp_path / 'accepted.usda')
    assert calls[0]['initial_condition_values'] == adapter._initial_conditions


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


@pytest.mark.parametrize('cleanup', ['stopped', 'failed', 'exception'])
def test_failed_admission_closes_candidate_and_preserves_accepted_owner(tmp_path, cleanup):
    valid = tmp_path / 'valid.usda'
    rejected = tmp_path / 'rejected.usda'
    for path in (valid, rejected):
        path.write_text('#usda 1.0\ndef Xform "World" {}\n')
    instances = []
    class Controller(_PhysicsController):
        def __init__(self, config):
            super().__init__(config, instances)
            self.reject = Path(config.input_usd_path).name == rejected.name
            self.stop_status = cleanup if self.reject else 'stopped'
            self.close_calls = 0
        def validate_joint_admission(self):
            if self.reject:
                raise ValueError('measured joint family unavailable')
        def deactivate(self):
            self.close_calls += 1
            if self.stop_status == 'exception':
                raise RuntimeError('owned worker has not exited')
            return self.stop_status
    adapter = OvphysxGenerationAdapter(Controller)
    assert adapter.activate(_Generation(1, valid))
    accepted = adapter.controller
    assert not adapter.activate(_Generation(2, rejected))
    candidate = instances[-1]
    assert candidate.close_calls == 1 and accepted.close_calls == 0
    assert adapter.controller is accepted and adapter.active_generation == 1
    assert 'measured joint family unavailable' in adapter.last_error
    if cleanup != 'stopped':
        assert not adapter.activate(_Generation(3, valid))
        assert len(instances) == 2 and candidate.close_calls == 2
        assert 'cleanup failed' in adapter.last_error
        candidate.stop_status = 'stopped'
    assert adapter.deactivate() == 'stopped'
    assert adapter._rejected_controller is None and adapter.controller is None
    assert accepted.close_calls == 1


def test_public_physics_keeps_live_state_across_visual_generation_then_resets(tmp_path):
    pytest.importorskip('pxr.Usd', reason='Requires USD composition; also run under Blender Python')
    source = '#usda 1.0\ndef Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}\n'
    first = tmp_path / 'one.usda'; first.write_text(source)
    second = tmp_path / 'two.usda'; second.write_text(source + 'def Cube "VisualOnly" {}\n')
    instances = []
    class PublicController(_PhysicsController):
        def __init__(self, config):
            super().__init__(config, instances)
            self._physics_client = SimpleNamespace(supports_unchanged_generation_reuse=True)
            self.failed = self.closed = False
            self.step_count = 37
    adapter = OvphysxGenerationAdapter(PublicController)
    assert adapter.activate(_Generation(1, first))
    original = adapter.controller
    assert adapter.activate(_Generation(2, second))
    assert adapter.controller is original and original.step_count == 37
    assert adapter.active_generation == 2 and adapter.physics_generation == 1
    assert adapter.reused_generation_count == 1
    assert adapter.reset()
    assert adapter.controller is not original and adapter.physics_generation == 2


def test_failed_live_mass_write_cannot_implicitly_restore_initial_physics(tmp_path, monkeypatch):
    pytest.importorskip('pxr.Usd', reason='Requires USD composition; also run under Blender Python')
    from ovrtx_blender_example import physics_capabilities
    monkeypatch.setattr(physics_capabilities, 'validate_joint_runtime', lambda *args: None)
    monkeypatch.setattr(physics_capabilities, 'validate_fixed_runtime', lambda *args, **kwargs: None)
    source = '#usda 1.0\ndef Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {\n float physics:mass = 1\n}\n'
    first = tmp_path / 'one.usda'; first.write_text(source)
    second = tmp_path / 'two.usda'; second.write_text(source.replace('mass = 1', 'mass = 2'))
    instances = []
    class PublicController(_PhysicsController):
        def __init__(self, config):
            super().__init__(config, instances)
            self.failed = self.closed = False
            self._physics_client = SimpleNamespace(supports_unchanged_generation_reuse=True,
                write_body_masses=lambda values: None, _import_native_client=lambda: None, public_sdk_identity=lambda: {})
        def apply_body_mass_edits(self, values):
            self.failed = True
            raise RuntimeError('native partial write')
    adapter = OvphysxGenerationAdapter(PublicController)
    assert adapter.activate(_Generation(1, first))
    original = adapter.controller
    assert not adapter.activate(_Generation(2, second))
    assert 'partial write' in adapter.last_error
    assert not adapter.activate(_Generation(1, first))
    assert adapter.controller is original and len(instances) == 1
    assert adapter.reset() and len(instances) == 2


@pytest.mark.parametrize('preserved,unchanged,new_generation,expected_calls', [
    (True, True, False, 1), (False, True, False, 2),
    (True, False, False, 2), (True, True, True, 2),
])
def test_retained_values_are_lazy_only_when_accepted_state_is_preserved(
    tmp_path, preserved, unchanged, new_generation, expected_calls,
):
    class Controller(_OvrtxController):
        def ensure(self, request):
            return SimpleNamespace(composition='accepted', session_started=False,
                                   value_state_preserved=preserved)
    controller = Controller()
    adapter = OvrtxGenerationAdapter(controller)
    adapter.update_request(RenderRequest())
    calls = []
    def retained():
        calls.append('materialize retained values')
        return ((OvrtxTransformValue('/World/Body', ((1.0,),)),), (), ())
    generation = _Generation(1, tmp_path / 'one.usdc')
    assert adapter.activate(generation, retained_values=retained)
    if new_generation: generation = _Generation(2, tmp_path / 'two.usdc')
    assert adapter.activate(generation, retained_values=retained, skip_unchanged_retained=unchanged)
    assert len(calls) == expected_calls
    assert len(controller.replays) == expected_calls


def test_static_geometry_generation_reuses_running_controller(tmp_path):
    pytest.importorskip('pxr.Usd')
    source = '#usda 1.0\ndef Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]) {}\ndef Cube "Ground" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {\n double size = 2\n}\n'
    first=tmp_path/'one.usda';first.write_text(source)
    second=tmp_path/'two.usda';second.write_text(source.replace('size = 2','size = 3'))
    instances=[]
    class PublicController(_PhysicsController):
        def __init__(self, config):
            super().__init__(config, instances)
            self._physics_client=SimpleNamespace(supports_unchanged_generation_reuse=True, supports_static_geometry_updates=True)
            self.failed=self.closed=False;self.step_count=37;self.edits=[]
        def apply_static_geometry_edits(self, changes):
            self.edits.extend(changes)
    adapter=OvphysxGenerationAdapter(PublicController)
    assert adapter.activate(_Generation(1, first))
    original=adapter.controller
    assert adapter.activate(_Generation(2, second))
    assert adapter.controller is original and original.step_count==37 and len(instances)==1
    assert original.edits==[dict(prim_path='/Ground',attribute_name='size',value_type='Double',value=3.)]
    assert adapter.active_generation==2 and adapter.physics_generation==1
