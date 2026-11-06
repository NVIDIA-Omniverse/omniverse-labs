# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete OVRTX and OVPhysX activation for scene generations."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from .scene_generation import SceneGeneration
from .ovphysx_stage import OvphysxStageController, OvphysxStageStatus
from .ovrtx_session_controller import OvrtxSessionController
from .ovrtx_value_updates import OvrtxAttributeValue, OvrtxTransformValue
from .physics_body_prims import discover_dynamic_body_prims
from .ovphysx_to_ovrtx import read_body_shape_transforms
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
        self.activation_timings: dict[str, Any] = {}
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
        retained_values: Any | None = None,
        skip_unchanged_retained: bool = False,
    ) -> bool:
        if self._request is None:
            self.last_error = "render_request_unavailable"
            return False
        started = time.perf_counter()
        phase_started = started
        self.activation_timings = {}
        same_generation = (self.active_generation == generation.number
                           and self._active_usd_path == generation.materialize_usd())
        self._request_activation_failed = False
        self._active_usd_path = generation.materialize_usd()
        request = replace(
            self._request,
            input_usd_path=self._active_usd_path,
        )
        self.activation_timings['materialize_request_ms'] = (time.perf_counter()-phase_started)*1000
        try:
            phase_started = time.perf_counter()
            self._record_ensure(self._controller.ensure(request))
            self.activation_timings['ensure_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            preserved = bool(getattr(self._last_ensure_result, 'value_state_preserved', False))
            if retained_values is not None and not (same_generation and skip_unchanged_retained and preserved):
                transform_values, attribute_values, _initial = retained_values()
            self.activation_timings['retained_value_resolution_ms'] = (time.perf_counter()-phase_started)*1000
            self.activation_timings['retained_transform_count'] = len(transform_values)
            self.activation_timings['retained_attribute_count'] = len(attribute_values)
            phase_started = time.perf_counter()
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
        self.activation_timings['retained_value_replay_ms'] = (time.perf_counter()-phase_started)*1000
        self.activation_timings['total_ms'] = (time.perf_counter()-started)*1000
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
        self._rejected_controller: OvphysxStageController | None = None
        self._rejected_reason = ""
        self._initial_conditions: tuple[BodyPose, ...] = ()
        self.active_generation: int | None = None
        self.physics_generation = 0
        self.dynamic_body_paths: tuple[str, ...] = ()
        self._active_scene_generation: SceneGeneration | None = None
        self._physics_reuse_key: Any = None
        self.reused_generation_count = 0
        self.live_mass_generation_count = 0
        self._live_mass_failed = False
        self.last_error = ""

    @property
    def controller(self) -> OvphysxStageController | None:
        return self._controller

    def activate(
        self,
        generation: SceneGeneration,
        *,
        initial_conditions: tuple[BodyPose, ...] = (),
        force_reset: bool = False,
    ) -> bool:
        if not self._close_rejected_controller():
            return False
        if self._live_mass_failed and not force_reset:
            self.last_error = "A live mass update failed; explicit physics reset is required"
            return False
        usd_path = generation.materialize_usd()
        body_paths = discover_dynamic_body_prims(usd_path, root="")
        retained_initial_conditions = tuple(
            value for value in initial_conditions if value.prim_path in body_paths
        )
        config = InteractiveSharedStageConfig.from_env(
            usd_path,
            authored_body_prims=body_paths,
        )
        config = replace(config, body_shape_transforms=read_body_shape_transforms(usd_path, body_paths))
        current = self._controller
        client = getattr(current, "_physics_client", None)
        reuse_key = None
        if getattr(client, "supports_unchanged_generation_reuse", False):
            from .physics_generation_fingerprint import physics_generation_fingerprint
            fingerprint = physics_generation_fingerprint(config.input_usd_path)
            if fingerprint is not None:
                reuse_key = (fingerprint, replace(config, input_usd_path=""), retained_initial_conditions)
        if (not force_reset and reuse_key is not None and current is not None
                and getattr(client, "supports_unchanged_generation_reuse", False)
                and not getattr(current, "failed", True) and not getattr(current, "closed", True)
                and self._physics_reuse_key == reuse_key):
            # The old immutable source remains owned by this live controller.
            # It represents the same complete physics closure, so neither a
            # load nor a pose replay is necessary for visual-only edits.
            self.active_generation = generation.number
            self._active_scene_generation = generation
            self.reused_generation_count += 1
            self.last_error = ""
            return True
        if (not force_reset and reuse_key is not None and self._physics_reuse_key is not None
                and reuse_key[1:] == self._physics_reuse_key[1:] and current is not None
                and not getattr(current, "failed", True) and not getattr(current, "closed", True)
                and callable(getattr(client, "write_body_masses", None))
                and self._active_scene_generation is not None):
            from .physics_generation_fingerprint import physics_mass_delta
            changes = physics_mass_delta(self._active_scene_generation.materialize_usd(),
                                         config.input_usd_path, body_paths)
            if changes:
                # The candidate retains all ordinary admission requirements.
                from .physics_capabilities import validate_joint_runtime, validate_fixed_runtime
                try:
                    validate_joint_runtime(config.input_usd_path, config, client._import_native_client,
                                           client.public_sdk_identity)
                    validate_fixed_runtime(config, client._import_native_client,
                                           client.public_sdk_identity, family="live_mass")
                    current.apply_body_mass_edits(changes)
                except Exception as exc:
                    if getattr(current, "failed", False):
                        self._live_mass_failed = True
                    self.last_error = f"Live physics mass update failed: {exc}"
                    return False
                self.active_generation = generation.number
                self._active_scene_generation = generation
                self._physics_reuse_key = reuse_key
                self.live_mass_generation_count += 1
                self.last_error = ""
                return True
        if (not force_reset and reuse_key is not None and self._physics_reuse_key is not None
                and reuse_key[1:] == self._physics_reuse_key[1:] and current is not None
                and not getattr(current, "failed", True) and not getattr(current, "closed", True)
                and getattr(client, "supports_static_geometry_updates", False)
                and self._active_scene_generation is not None):
            from .physics_generation_fingerprint import physics_geometry_delta
            changes = physics_geometry_delta(self._active_scene_generation.materialize_usd(), config.input_usd_path, allow_dynamic=True)
            if changes:
                try:
                    current.apply_static_geometry_edits(changes)
                except Exception as exc:
                    if getattr(current, "failed", False):
                        self._live_mass_failed = True
                    self.last_error = f"Live collider geometry update failed: {exc}"
                    return False
                self.active_generation = generation.number
                self._active_scene_generation = generation
                self._physics_reuse_key = reuse_key
                self.reused_generation_count += 1
                self.last_error = ""
                return True
        controller = self._controller_factory(config)
        if reuse_key is None and getattr(getattr(controller, "_physics_client", None), "supports_unchanged_generation_reuse", False):
            from .physics_generation_fingerprint import physics_generation_fingerprint
            fingerprint = physics_generation_fingerprint(config.input_usd_path)
            if fingerprint is not None:
                reuse_key = (fingerprint, replace(config, input_usd_path=""), retained_initial_conditions)
        validate = getattr(controller, "validate_joint_admission", None)
        if callable(validate):
            try:
                validate()
            except Exception as exc:
                self._rejected_controller = controller
                self._rejected_reason = f"Joint admission failed: {exc}"
                self._close_rejected_controller()
                return False
        if self._controller is not None:
            status = self.deactivate()
            if status not in {"stopped", "not_found"}:
                self._rejected_controller = controller
                self._rejected_reason = "predecessor_deactivation_failed"
                self._close_rejected_controller()
                return False
        self._controller = controller
        self.dynamic_body_paths = body_paths
        self._initial_conditions = retained_initial_conditions
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
        self._physics_reuse_key = reuse_key
        self._live_mass_failed = False
        self.physics_generation += 1
        self.last_error = ""
        return True

    def reset(self) -> bool:
        generation = self._active_scene_generation
        if generation is None or self.active_generation is None:
            self.last_error = "active_scene_generation_unavailable"
            return False
        current = self._controller
        client = getattr(current, "_physics_client", None)
        if (current is not None and not self._live_mass_failed
                and getattr(current, "started", False)
                and not getattr(current, "failed", True)
                and not getattr(current, "closed", True)
                and getattr(client, "supports_verified_simulation_reload", False)):
            # The active immutable generation includes accepted authored edits;
            # loading it restores velocities, joints and solver state as well as
            # poses. The existing worker retains its production admission.
            config = replace(current.config, input_usd_path=generation.materialize_usd())
            try:
                result = current.reset_simulation(
                    config=config, initial_condition_values=self._initial_conditions)
            except Exception as exc:
                current.failed = True
                self.last_error = f"Physics reset failed: {exc}"
                return False
            if result.status in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}:
                current.failed = True
                self.last_error = current.last_error or result.reason or result.status.value
                return False
            self.physics_generation += 1
            self.last_error = ""
            return True
        return self.activate(generation, initial_conditions=self._initial_conditions, force_reset=True)

    def _close_rejected_controller(self) -> bool:
        if self._rejected_controller is None:
            return True
        self.last_error = self._rejected_reason
        try:
            status = self._rejected_controller.deactivate()
            if status not in {"stopped", "not_found"}:
                raise RuntimeError(str(status))
        except Exception as exc:
            self.last_error += f"; rejected candidate cleanup failed: {exc}"
            return False
        self._rejected_controller = None
        self._rejected_reason = ""
        return True

    def deactivate(self) -> str:
        had_rejected = self._rejected_controller is not None
        rejected_closed = self._close_rejected_controller()
        if self._controller is None:
            return ("stopped" if had_rejected else "not_found") if rejected_closed else "failed"
        status = self._controller.deactivate()
        if status in {"stopped", "not_found"}:
            self._controller = None
            self.active_generation = None
            self.dynamic_body_paths = ()
            self._active_scene_generation = None
            self._physics_reuse_key = None
            self._live_mass_failed = False
        return status if rejected_closed else "failed"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_generation": self.active_generation,
            "physics_generation": self.physics_generation,
            "reused_generation_count": self.reused_generation_count,
            "live_mass_generation_count": self.live_mass_generation_count,
            "dynamic_body_paths": list(self.dynamic_body_paths),
            "retained_initial_condition_paths": sorted(
                value.prim_path for value in self._initial_conditions
            ),
            "last_error": self.last_error,
        }


def generation_requires_physics(generation: SceneGeneration) -> bool:
    """Return whether a generation needs the OVPhysX runtime."""

    from pxr import Usd, UsdPhysics  # type: ignore

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
        if prim.IsA(UsdPhysics.Joint):
            return True
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
