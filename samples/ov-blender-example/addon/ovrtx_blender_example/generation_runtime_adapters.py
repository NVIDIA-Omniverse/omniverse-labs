# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete OVRTX and OVPhysX activation for scene generations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .scene_generation import SceneGeneration
from .ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from .ovrtx_session_controller import OvrtxSessionController
from .ovrtx_value_updates import OvrtxAttributeValue, OvrtxTransformValue
from .physics_body_prims import discover_dynamic_body_prims
from .render_requests import RenderRequest
from .runtime_scheduler import RuntimeTickResult, RuntimeTickStatus
from .shared_stage_composition import BodyPose
from .shared_stage_config import InteractiveSharedStageConfig


class OvrtxGenerationAdapter:
    """Replace one OVRTX session and replay scene-owned values."""

    def __init__(self, controller: OvrtxSessionController | None = None) -> None:
        self._controller = controller or OvrtxSessionController()
        self._request: RenderRequest | None = None
        self._active_usd_path = ""
        self._composition: Any | None = None
        self._last_ensure_result: Any | None = None
        self._request_activation_failed = False
        self.active_generation: int | None = None
        self.last_error = ""

    @property
    def controller(self) -> OvrtxSessionController:
        return self._controller

    @property
    def request(self) -> RenderRequest | None:
        return self._request

    @property
    def composition(self) -> Any | None:
        return self._composition

    @property
    def last_ensure_result(self) -> Any | None:
        return self._last_ensure_result

    def update_request(self, request: RenderRequest) -> None:
        if not isinstance(request, RenderRequest):
            raise TypeError("OVRTX generation adapter request must be a RenderRequest")
        self._request = (
            replace(request, input_usd_path=self._active_usd_path)
            if self._active_usd_path
            else request
        )

    def ensure_request(self) -> bool:
        if self._request is None or self.active_generation is None:
            self.last_error = "scene_generation_unavailable"
            return False
        if self._request_activation_failed:
            return False
        try:
            self._record_ensure(self._controller.ensure(self._request))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._request_activation_failed = True
            return False
        self.last_error = ""
        return True

    def activate(
        self,
        generation: SceneGeneration,
        *,
        transform_values: tuple[OvrtxTransformValue, ...] = (),
        attribute_values: tuple[OvrtxAttributeValue, ...] = (),
    ) -> bool:
        if self._request is None:
            self.last_error = "render_request_unavailable"
            return False
        self._request_activation_failed = False
        self._active_usd_path = generation.materialize_usd()
        request = replace(
            self._request,
            input_usd_path=self._active_usd_path,
        )
        try:
            self._record_ensure(self._controller.ensure(request))
            if transform_values or attribute_values:
                replay = self._controller.apply_runtime_updates(
                    lambda port, _project: _replay_ovrtx_values(
                        port,
                        transform_values,
                        attribute_values,
                    )
                )
                if replay.status == RuntimeTickStatus.FAILED:
                    self.last_error = replay.skipped_reason or "retained_value_replay_failed"
                    return False
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self._request = request
        self.active_generation = generation.number
        self.last_error = ""
        return True

    def deactivate(self) -> str:
        status = self._controller.deactivate()
        if status in {"stopped", "not_found"}:
            self.active_generation = None
            self._active_usd_path = ""
            self._composition = None
            self._last_ensure_result = None
            self._request_activation_failed = False
        return status

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_generation": self.active_generation,
            "last_error": self.last_error,
        }

    def _record_ensure(self, result: Any) -> None:
        self._last_ensure_result = result
        self._composition = getattr(result, "composition", None)

class OvphysxGenerationAdapter:
    """Replace one scene-owned OVPhysX simulation and its pose producer."""

    def __init__(self, controller_factory: Any = OvphysxStageController) -> None:
        self._controller_factory = controller_factory
        self._controller: OvphysxStageController | None = None
        self._initial_conditions: tuple[BodyPose, ...] = ()
        self.active_generation: int | None = None
        self.physics_generation = 0
        self.dynamic_body_paths: tuple[str, ...] = ()
        self._active_scene_generation: SceneGeneration | None = None
        self.last_error = ""

    @property
    def controller(self) -> OvphysxStageController | None:
        return self._controller

    def activate(
        self,
        generation: SceneGeneration,
        *,
        initial_conditions: tuple[BodyPose, ...] = (),
    ) -> bool:
        if self._controller is not None:
            status = self.deactivate()
            if status not in {"stopped", "not_found"}:
                self.last_error = "predecessor_deactivation_failed"
                return False
        usd_path = generation.materialize_usd()
        body_paths = discover_dynamic_body_prims(usd_path, root="")
        self.dynamic_body_paths = body_paths
        self._initial_conditions = tuple(
            value for value in initial_conditions if value.prim_path in body_paths
        )
        config = InteractiveSharedStageConfig.from_env(
            usd_path,
            authored_body_prims=body_paths,
        )
        controller = self._controller_factory(config)
        self._controller = controller
        result = controller.tick(
            max_steps=config.max_steps,
            initial_condition_values=self._initial_conditions,
        )
        if result.status in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}:
            reason = result.reason or result.status.value
            controller_error = str(getattr(controller, "last_error", "")).strip()
            self.last_error = (
                f"{reason}: {controller_error}"
                if controller_error and controller_error != reason
                else reason
            )
            return False
        self.active_generation = generation.number
        self._active_scene_generation = generation
        self.physics_generation += 1
        self.last_error = ""
        return True

    def reset(self) -> bool:
        generation = self._active_scene_generation
        if generation is None or self.active_generation is None:
            self.last_error = "active_scene_generation_unavailable"
            return False
        return self.activate(generation, initial_conditions=self._initial_conditions)

    def deactivate(self) -> str:
        if self._controller is None:
            return "not_found"
        status = self._controller.deactivate()
        if status in {"stopped", "not_found"}:
            self._controller = None
            self.active_generation = None
            self.dynamic_body_paths = ()
            self._active_scene_generation = None
        return status

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_generation": self.active_generation,
            "physics_generation": self.physics_generation,
            "dynamic_body_paths": list(self.dynamic_body_paths),
            "retained_initial_condition_paths": sorted(
                value.prim_path for value in self._initial_conditions
            ),
            "last_error": self.last_error,
        }


def generation_requires_physics(generation: SceneGeneration) -> bool:
    """Return whether a generation needs the OVPhysX runtime."""

    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(generation.materialize_usd())
    if not stage:
        raise RuntimeError("scene generation could not be inspected for physics")
    applied_names = {
        "PhysicsRigidBodyAPI",
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsArticulationRootAPI",
    }
    physics_types = {
        "PhysicsScene",
        "PhysicsFixedJoint",
        "PhysicsRevoluteJoint",
        "PhysicsPrismaticJoint",
        "PhysicsSphericalJoint",
        "PhysicsDistanceJoint",
    }
    for prim in stage.Traverse():
        if prim.GetTypeName() in physics_types:
            return True
        if applied_names.intersection(str(name) for name in prim.GetAppliedSchemas()):
            return True
        if prim.HasRelationship("material:binding:physics"):
            return True
    return False


def _replay_ovrtx_values(
    port: Any,
    transforms: tuple[OvrtxTransformValue, ...],
    attributes: tuple[OvrtxAttributeValue, ...],
) -> RuntimeTickResult:
    try:
        if transforms:
            transform_result = port.update_transforms(transforms)
            if transform_result.updated_count != len(transforms):
                raise RuntimeError("OVRTX transform replay was incomplete")
        if attributes:
            attribute_result = port.update_attribute_values(attributes)
            if attribute_result.updated_count != len(attributes):
                raise RuntimeError("OVRTX attribute replay was incomplete")
    except Exception as exc:
        return RuntimeTickResult(
            status=RuntimeTickStatus.FAILED,
            enabled=True,
            skipped_reason=f"retained_value_replay_failed:{type(exc).__name__}:{exc}",
        )
    return RuntimeTickResult(
        status=RuntimeTickStatus.NOOP,
        enabled=True,
        values_written=bool(transforms or attributes),
        should_reset_refinement=bool(transforms or attributes),
    )


__all__ = [
    "OvphysxGenerationAdapter",
    "OvrtxGenerationAdapter",
    "generation_requires_physics",
]
