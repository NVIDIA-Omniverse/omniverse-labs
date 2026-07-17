# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pending and accepted sim-authoritative initial condition values."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from .interactive_edit_planner import DataAuthority, EditIntent
from .native_client_support import coerce_mapping_int as _mapping_int
from .ovphysx_stage import OvphysxStageResult, OvphysxStageStatus
from .shared_stage_composition import BodyPose, BodyVelocity


_POSE_ATTRIBUTES = frozenset({"omni:xform", "xformOp:transform"})
_VELOCITY_ATTRIBUTES = frozenset({"physics:velocity", "physics:angularVelocity"})
_SUPPORTED_ATTRIBUTES = _POSE_ATTRIBUTES | _VELOCITY_ATTRIBUTES


class SimUpdateStream:
    """Own sim edits and accepted values across OVPhysX replacements."""

    def __init__(
        self,
        *,
        value_sink: Callable[[Sequence[BodyPose]], None] | None = None,
        retain_values: bool = True,
    ) -> None:
        self._pending: dict[tuple[str, str, str], EditIntent] = {}
        # latest desired value per semantic target is the queue.
        # Dict replacement preserves deterministic first-target order.
        self._pending_lock = threading.Lock()
        self._accepted: dict[str, BodyPose] = {}
        self._accepted_velocities: dict[str, BodyVelocity] = {}
        self._value_sink = value_sink
        self._retain_values = bool(retain_values)
        self._last_result: dict[str, Any] = {}
        self._last_controller_result: OvphysxStageResult | None = None
        self._wake_hook: Callable[[], None] | None = None

    @property
    def has_pending(self) -> bool:
        with self._pending_lock:
            return bool(self._pending)

    def set_wake_hook(self, hook: Callable[[], None] | None) -> None:
        """Install a hook fired after each queued edit (task02-07 wake source).

        The render loop parks on the camera mailbox when idle; a sim edit
        submitted from the main thread must wake it so the next tick can
        apply the pending initial-condition value. ``None`` uninstalls the
        hook (loop exit).
        """

        self._wake_hook = hook

    @property
    def has_values(self) -> bool:
        return bool(self._accepted or self._accepted_velocities)

    @property
    def last_result(self) -> dict[str, Any]:
        return dict(self._last_result)

    @property
    def last_controller_result(self) -> OvphysxStageResult | None:
        return self._last_controller_result

    @staticmethod
    def supports(intent: EditIntent) -> bool:
        target = intent
        return intent.data_authority == DataAuthority.SIM and (
            target.usd_attribute in _SUPPORTED_ATTRIBUTES
            or target.blender_property_path == "matrix_world"
        )

    def queue(self, intent: EditIntent, *, notify: bool = True) -> dict[str, Any]:
        with self._pending_lock:
            self._pending[_pending_target(intent)] = intent
        result = _base_result(intent, queued=True)
        self._last_result = result
        wake_hook = self._wake_hook
        if notify and wake_hook is not None:
            wake_hook()
        return result

    def unsupported_result(self, intent: EditIntent) -> dict[str, Any]:
        result = _base_result(intent, queued=False)
        result.update(failed=True, skipped_reason="unsupported_sim_value_update")
        self._last_result = result
        return result

    def values_for_controller_start(self, *, controller_started: bool) -> tuple[BodyPose, ...]:
        if controller_started:
            return ()
        return tuple(self._accepted[path] for path in sorted(self._accepted))

    def apply_pending(self, controller: Any) -> dict[str, Any]:
        with self._pending_lock:
            if not self._pending:
                return {}
            # Atomic swap: an edit queued from another thread between a
            # copy and a clear would otherwise be dropped silently
            # (task02-03 review follow-up, applied here where sim edits
            # become cross-thread).
            intents = list(self._pending.values())
            self._pending = {}
        self._last_controller_result = None
        kind_order: list[str] = []
        for intent in intents:
            kind = "velocity" if intent.usd_attribute in _VELOCITY_ATTRIBUTES else "pose"
            if kind not in kind_order:
                kind_order.append(kind)
        results: list[dict[str, Any]] = []
        retry_intents: list[EditIntent] = []
        for kind in kind_order:
            batch = [
                intent for intent in intents
                if (intent.usd_attribute in _VELOCITY_ATTRIBUTES) == (kind == "velocity")
            ]
            try:
                values: tuple[BodyPose | BodyVelocity, ...]
                if kind == "velocity":
                    values = _body_velocities_from_intents(batch, self._accepted_velocities)
                    update = controller.apply_body_velocity_edits(values, reset=False)
                    reason = "body_velocity_update"
                else:
                    values = tuple(_body_pose_from_intent(intent) for intent in batch)
                    update = controller.apply_initial_condition_values(values, reset=False)
                    reason = "initial_condition_value_update"
                self._last_controller_result = update
                failed = update.status in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}
                if update.status == OvphysxStageStatus.BUSY:
                    retry_intents.extend(batch)
            except Exception as exc:
                values = ()
                update = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                failed = True
                reason = "body_velocity_update" if kind == "velocity" else "initial_condition_value_update"
            if not failed:
                if self._retain_values:
                    for value in values:
                        if isinstance(value, BodyVelocity):
                            self._accepted_velocities[value.prim_path] = value
                        else:
                            self._accepted[value.prim_path] = value
                if self._value_sink is not None and values and isinstance(values[0], BodyPose):
                    self._value_sink(values)
            results.append(
                _result(batch[0], values, update, failed=failed, reason=reason, value_kind=kind)
            )
        if retry_intents:
            with self._pending_lock:
                # prepend retryable work, then let any value queued
                # during native application win for the same semantic target.
                retry = {_pending_target(intent): intent for intent in retry_intents}
                retry.update(self._pending)
                self._pending = retry
        result = _combine_results(results)
        self._last_result = result
        return result

    def record_controller_start(self, update: OvphysxStageResult, controller: Any | None = None) -> dict[str, Any]:
        self._last_controller_result = None
        poses = self.values_for_controller_start(controller_started=False)
        results: list[dict[str, Any]] = []
        failed = update.status in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}
        if poses:
            results.append(_result(None, poses, update, failed=failed, reason="initial_condition_values", value_kind="pose"))
        velocities = tuple(self._accepted_velocities[path] for path in sorted(self._accepted_velocities))
        if velocities and controller is not None and not failed:
            velocity_update = controller.apply_body_velocity_edits(velocities, reset=False)
            self._last_controller_result = velocity_update
            velocity_failed = velocity_update.status in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}
            results.append(_result(None, velocities, velocity_update, failed=velocity_failed, reason="body_velocity_values", value_kind="velocity"))
        if not results:
            return {}
        result = results[-1]
        if any(item["failed"] for item in results):
            self._last_result = result
        return result

    def diagnostics(self) -> dict[str, Any]:
        with self._pending_lock:
            pending_count = len(self._pending)
        return {
            "pending_count": pending_count,
            "value_count": len(self._accepted),
            "value_paths": sorted(self._accepted),
            "velocity_count": len(self._accepted_velocities),
            "velocity_paths": sorted(self._accepted_velocities),
            "process_lifetime": True,
        }


def _base_result(intent: EditIntent, *, queued: bool) -> dict[str, Any]:
    return {
        "queued": queued,
        "values_written": False,
        "shape": intent.shape.value,
        "data_authority": intent.data_authority.value,
        "physics_generation_reset": False,
        "failed": False,
        "skipped_reason": "",
    }


def _pending_target(intent: EditIntent) -> tuple[str, str, str]:
    kind = "velocity" if intent.usd_attribute in _VELOCITY_ATTRIBUTES else "pose"
    attribute = intent.usd_attribute if kind == "velocity" else ""
    return kind, intent.usd_prim_path, attribute


def _result(
    intent: EditIntent | None,
    poses: Sequence[BodyPose | BodyVelocity],
    update: OvphysxStageResult | Mapping[str, Any],
    *,
    failed: bool,
    reason: str,
    value_kind: str,
) -> dict[str, Any]:
    typed = isinstance(update, OvphysxStageResult)
    physics_generation_reset = value_kind == "pose" and not failed
    simulation_time_ns = update.simulation_time_ns if typed else _mapping_int(update, "simulation_time_ns", 0)
    update_diagnostics = (
        {
            "status": update.status.value,
            "reason": update.reason,
            "dirty_paths": list(update.dirty_paths),
            "step_count": update.step_count,
            "simulation_time_ns": update.simulation_time_ns,
            "generation": update.generation,
        }
        if typed
        else dict(update)
    )
    return {
        **(_base_result(intent, queued=False) if intent is not None else {
            "queued": False,
            "data_authority": "sim",
        }),
        "reason": reason,
        "values_written": not failed,
        "physics_generation_reset": physics_generation_reset,
        "physics_generation_invalidated": not failed,
        "transform_updated": value_kind == "pose" and not failed,
        "failed": failed,
        "retryable": typed and update.status == OvphysxStageStatus.BUSY,
        "skipped_reason": (
            f"{reason}_error" if failed else ""
        ),
        "value_requested_count": len(poses),
        "value_paths": [],
        "value_count": 0,
        "sim_value_paths": [pose.prim_path for pose in poses],
        "simulation_time_ns": simulation_time_ns,
        "physics_reset": False,
        "sim_value_write_applied": not failed,
        "render_value_write_applied": False,
        "result": update_diagnostics,
    }


def _combine_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(results) == 1:
        return dict(results[0])
    failed = any(result.get("failed", False) for result in results)
    return {
        "reason": "sim_value_updates",
        "queued": False,
        "data_authority": "sim",
        "values_written": any(result.get("values_written", False) for result in results),
        "physics_generation_reset": any(result.get("physics_generation_reset", False) for result in results),
        "physics_generation_invalidated": any(result.get("physics_generation_invalidated", False) for result in results),
        "transform_updated": any(result.get("transform_updated", False) for result in results),
        "failed": failed,
        "retryable": any(result.get("retryable", False) for result in results),
        "skipped_reason": ";".join(
            str(result["skipped_reason"]) for result in results if result.get("skipped_reason")
        ),
        "value_requested_count": sum(int(result.get("value_requested_count", 0)) for result in results),
        "sim_value_paths": [
            str(path) for result in results for path in result.get("sim_value_paths", ())
        ],
        "sim_value_write_applied": all(result.get("sim_value_write_applied", False) for result in results),
        "render_value_write_applied": False,
        "updates": [dict(result) for result in results],
    }


def _body_pose_from_intent(intent: EditIntent) -> BodyPose:
    prim_path = intent.usd_prim_path
    if not prim_path:
        raise ValueError("body pose update is missing usd_prim_path")
    if isinstance(intent.value, Mapping) and "translate" in intent.value and "orient" in intent.value:
        return BodyPose(
            prim_path=prim_path,
            translate=_float_tuple(intent.value["translate"], ("x", "y", "z")),
            orient=_float_tuple(intent.value["orient"], ("i", "j", "k", "r")),
        )
    matrix = _matrix4d_rows(intent.value)
    return BodyPose(
        prim_path=prim_path,
        translate=(matrix[3][0], matrix[3][1], matrix[3][2]),
        orient=_quatf_from_row_vector_matrix(matrix),
    )


def _body_velocities_from_intents(
    intents: Sequence[EditIntent],
    accepted: Mapping[str, BodyVelocity],
) -> tuple[BodyVelocity, ...]:
    components: dict[str, dict[str, tuple[float, float, float]]] = {}
    for intent in intents:
        path = intent.usd_prim_path
        if not path:
            raise ValueError("body velocity update is missing usd_prim_path")
        previous = accepted.get(path)
        entry = components.setdefault(
            path,
            {
                "linear": previous.linear if previous is not None else (0.0, 0.0, 0.0),
                "angular": previous.angular if previous is not None else (0.0, 0.0, 0.0),
            },
        )
        key = "linear" if intent.usd_attribute == "physics:velocity" else "angular"
        entry[key] = _float_tuple(intent.value, ("x", "y", "z"))
    return tuple(
        BodyVelocity(path, values["linear"], values["angular"])
        for path, values in sorted(components.items())
    )


def _float_tuple(value: Any, keys: tuple[str, ...]) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        try:
            return tuple(float(value[key]) for key in keys)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"value must have numeric {', '.join(keys)}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != len(keys):
        raise ValueError(f"value must be a {len(keys)}-value sequence")
    return tuple(float(item) for item in value)


def _matrix4d_rows(value: Any) -> list[list[float]]:
    if isinstance(value, Mapping):
        value = value.get("matrix", value.get("omni:xform"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("transform value must be a 4x4 or flat 16-value matrix")
    rows = list(value)
    if len(rows) == 16 and all(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes))
        for item in rows
    ):
        flat = [_float_matrix_value(item) for item in rows]
        return [flat[index : index + 4] for index in range(0, 16, 4)]
    if len(rows) != 4:
        raise ValueError("transform value must have four rows")
    matrix: list[list[float]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
            raise ValueError("transform matrix rows must have four values")
        matrix.append([_float_matrix_value(item) for item in row])
    return matrix


def _float_matrix_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transform matrix value is not numeric: {value!r}") from exc


def _quatf_from_row_vector_matrix(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    rotation: list[list[float]] = []
    for row_index in range(3):
        row = [float(matrix[row_index][column]) for column in range(3)]
        length = math.sqrt(sum(value * value for value in row))
        if length <= 1.0e-12:
            raise ValueError("body pose transform matrix has a zero-length rotation row")
        rotation.append([value / length for value in row])
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        r, i, j, k = 0.25 * scale, (m12 - m21) / scale, (m20 - m02) / scale, (m01 - m10) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        r, i, j, k = (m12 - m21) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        r, i, j, k = (m20 - m02) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        r, i, j, k = (m01 - m10) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale
    length = math.sqrt(i * i + j * j + k * k + r * r)
    return (i / length, j / length, k / length, r / length)


__all__ = ["SimUpdateStream"]
