# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pending view-authoritative edit application."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from .interactive_edit_planner import (
    DataAuthority,
    EditIntent,
    RENDER_SETTING_VALUE_SOURCE,
)
from . import uv_usd_prim
from . import usd_value_edit_support
from .ovrtx_value_updates import (
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxUpdatePort,
    OvrtxValueUpdateResult,
)


SUPPORTED_UV_VALUE_ATTRIBUTES = {
    uv_usd_prim.TARGET_USD_ATTRIBUTE: uv_usd_prim.VALUE_TYPE,
}
_VIEW_VALUE_KINDS = frozenset(
    {
        "camera",
        "camera_value",
        "render_setting",
        "transform",
        "material",
        "light",
        "world",
        "uv",
    }
)
def _pending_by_kind(
    pending: Sequence[EditIntent],
) -> tuple[dict[str, list[EditIntent]], list[EditIntent]]:
    view_grouped = {kind: [] for kind in _VIEW_VALUE_KINDS}
    unsupported: list[EditIntent] = []
    for intent in pending:
        kind = _view_value_kind(intent)
        if intent.data_authority == DataAuthority.VIEW and kind in view_grouped:
            view_grouped[kind].append(intent)
        else:
            unsupported.append(intent)
    return view_grouped, unsupported


def _combined_update_result(updates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    non_empty = [dict(update) for update in updates if update]
    if not non_empty:
        return {}
    if len(non_empty) == 1:
        return non_empty[0]
    failed = any(update.get("failed", False) for update in non_empty)
    return {
        "values_written": any(update.get("values_written", False) for update in non_empty),
        "value_paths": [
            str(path)
            for update in non_empty
            for path in update.get("value_paths", ())
        ],
        "value_count": sum(int(update.get("value_count", 0)) for update in non_empty),
        "value_requested_count": sum(
            int(update.get("value_requested_count", update.get("value_count", 0)))
            for update in non_empty
        ),
        "physics_generation_reset": any(update.get("physics_generation_reset", False) for update in non_empty),
        "failed": failed,
        "skipped_reason": ";".join(
            str(update.get("skipped_reason", "")) for update in non_empty if update.get("skipped_reason", "")
        ),
        "updates": non_empty,
    }


def retain_view_value(
    intent: EditIntent,
    *,
    transform_sink: Callable[[Sequence[OvrtxTransformValue]], None] | None = None,
    attribute_sink: Callable[[Sequence[OvrtxAttributeValue]], None] | None = None,
) -> None:
    """Retain one supported view value at a scene-owned sink."""

    kind = _view_value_kind(intent)
    try:
        if kind in {"camera", "transform"} and transform_sink is not None:
            transform_sink((_view_transform_value(intent),))
        elif attribute_sink is not None:
            value_fn = {
                "material": _material_value,
                "light": _light_value,
                "world": _world_value,
                "uv": _uv_value,
            }.get(kind)
            if value_fn is not None:
                attribute_sink((value_fn(intent),))
    except (TypeError, ValueError):
        pass


class ViewUpdateStream:
    """Queues and applies view-authoritative edits to the active OVRTX session."""

    def __init__(
        self,
        *,
        transform_sink: Callable[[Sequence[OvrtxTransformValue]], None] | None = None,
        attribute_sink: Callable[[Sequence[OvrtxAttributeValue]], None] | None = None,
    ) -> None:
        self._pending_updates: dict[tuple[Any, ...], EditIntent] = {}
        # one insertion-ordered map is the bounded cross-thread
        # state; replace by semantic target, keep first-target order.
        self._pending_lock = threading.Lock()
        self._last_result: dict[str, Any] = {}
        self._transform_sink = transform_sink
        self._attribute_sink = attribute_sink
        self._wake_hook: Callable[[], None] | None = None

    @property
    def has_pending(self) -> bool:
        with self._pending_lock:
            return bool(self._pending_updates)

    def pending_targets(
        self,
    ) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
        """Return targets whose current value is already queued."""

        with self._pending_lock:
            transforms = frozenset(
                intent.usd_prim_path
                for intent in self._pending_updates.values()
                if _view_value_kind(intent) in {"camera", "transform"}
            )
            attributes = frozenset(
                (intent.usd_prim_path, intent.usd_attribute)
                for intent in self._pending_updates.values()
                if _view_value_kind(intent) not in {"camera", "transform"}
            )
        return transforms, attributes

    def set_wake_hook(self, hook: Callable[[], None] | None) -> None:
        """Install a hook fired after each queued edit (task02-03 wake source).

        The render loop parks on the camera mailbox when idle; edit
        submission must wake it so pending value updates apply promptly.
        ``None`` uninstalls the hook (loop exit).
        """

        self._wake_hook = hook

    @property
    def last_result(self) -> dict[str, Any]:
        return dict(self._last_result)

    @staticmethod
    def supports(intent: EditIntent) -> bool:
        return intent.data_authority == DataAuthority.VIEW and _view_value_kind(intent) in _VIEW_VALUE_KINDS

    def queue(self, intent: EditIntent, *, notify: bool = True) -> dict[str, Any]:
        self._retain_queued_value(intent)
        with self._pending_lock:
            key = _pending_target(intent)
            self._pending_updates[key] = intent
        result = update_result(intent, physics_generation_reset=False, queued=True)
        self._last_result = result
        wake_hook = self._wake_hook
        if notify and wake_hook is not None:
            wake_hook()
        return result

    def _retain_queued_value(self, intent: EditIntent) -> None:
        retain_view_value(
            intent,
            transform_sink=self._transform_sink,
            attribute_sink=self._attribute_sink,
        )

    def unsupported_result(self, intent: EditIntent) -> dict[str, Any]:
        result = update_result(intent, physics_generation_reset=False, queued=False)
        result["failed"] = True
        result["skipped_reason"] = _unsupported_update_application_reason(intent)
        self._last_result = result
        return result

    def apply_pending(
        self,
        ovrtx_updates: OvrtxUpdatePort,
    ) -> dict[str, Any]:
        with self._pending_lock:
            if not self._pending_updates:
                return {}
            # Atomic swap: an edit queued from another thread between a
            # copy and a clear would otherwise be dropped silently.
            pending = self._pending_updates
            self._pending_updates = {}
        pending_intents = list(pending.values())
        view_grouped, unsupported = _pending_by_kind(pending_intents)
        camera_intents = view_grouped["camera"]
        camera_value_intents = view_grouped["camera_value"]
        render_setting_intents = view_grouped["render_setting"]
        transform_intents = view_grouped["transform"]
        material_intents = view_grouped["material"]
        light_intents = view_grouped["light"]
        world_intents = view_grouped["world"]
        uv_intents = view_grouped["uv"]
        updates: list[dict[str, Any]] = []

        transform_updates = [*camera_intents, *transform_intents]
        if transform_updates:
            updates.append(
                self._apply_transform_updates(
                    ovrtx_updates,
                    transform_updates,
                )
            )

        for intents, value_fn, result_fn, unsupported_error, unsupported_reason in (
            (
                camera_value_intents,
                _camera_value,
                camera_value_update_result,
                "unsupported camera value attribute",
                "unsupported_camera_value_attribute",
            ),
            (
                render_setting_intents,
                _render_setting_value,
                render_setting_value_update_result,
                "unsupported render setting value type",
                "unsupported_render_setting_value_type",
            ),
            (
                material_intents,
                _material_value,
                material_value_update_result,
                "unsupported material value attribute",
                "unsupported_material_value_attribute",
            ),
            (
                light_intents,
                _light_value,
                light_value_update_result,
                "unsupported light value attribute",
                "unsupported_light_value_attribute",
            ),
            (
                world_intents,
                _world_value,
                world_value_update_result,
                "unsupported world value attribute",
                "unsupported_world_value_attribute",
            ),
            (
                uv_intents,
                _uv_value,
                uv_value_update_result,
                "unsupported UV value attribute",
                "unsupported_uv_value_attribute",
            ),
        ):
            if intents:
                updates.append(
                    self._apply_attribute_updates(
                        ovrtx_updates,
                        intents,
                        value_fn=value_fn,
                        result_fn=result_fn,
                        unsupported_error=unsupported_error,
                        unsupported_reason=unsupported_reason,
                    )
                )

        if unsupported:
            result = update_result(
                unsupported[0],
                physics_generation_reset=False,
                queued=False,
            )
            result["failed"] = True
            result["skipped_reason"] = _unsupported_update_application_reason(unsupported[0])
            updates.append(result)

        if not updates:
            return {}
        if len(updates) == 1:
            self._last_result = updates[0]
            return updates[0]
        result = _combined_update_result(updates)
        self._last_result = result
        return result

    def _apply_transform_updates(
        self,
        ovrtx_updates: OvrtxUpdatePort,
        intents: Sequence[EditIntent],
    ) -> dict[str, Any]:
        values: list[OvrtxTransformValue] = []
        try:
            values = [_view_transform_value(intent) for intent in intents]
            started_ns = time.perf_counter_ns()
            outcome = ovrtx_updates.update_transforms(values)
            apply_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            failed = outcome.updated_count != len(values)
            result = _value_update_result_mapping(outcome)
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            apply_ms = 0.0
            failed = True

        update = view_value_update_result(
            intents,
            values=values,
            result=result,
            apply_ms=apply_ms,
            failed=failed,
        )
        self._last_result = update
        return update

    def _apply_attribute_updates(
        self,
        ovrtx_updates: OvrtxUpdatePort,
        intents: Sequence[EditIntent],
        *,
        value_fn: Callable[[EditIntent], OvrtxAttributeValue],
        result_fn: Callable[..., dict[str, Any]],
        unsupported_error: str,
        unsupported_reason: str,
    ) -> dict[str, Any]:
        values: list[OvrtxAttributeValue] = []
        try:
            values = [value_fn(intent) for intent in intents]
            started_ns = time.perf_counter_ns()
            updater = getattr(ovrtx_updates, "update_attribute_values", None)
            if not callable(updater):
                raise RuntimeError("OVRTX client does not support attribute value updates")
            outcome = updater(values)
            apply_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            result_mapping = _value_update_result_mapping(outcome)
            failed = outcome.updated_count != len(values)
        except Exception as exc:
            skipped_reason = (
                "value_update_unavailable"
                if "does not support attribute value updates" in str(exc)
                else ""
            )
            if not skipped_reason and unsupported_error in str(exc):
                skipped_reason = unsupported_reason
            result_mapping = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            if skipped_reason:
                result_mapping["skipped_reason"] = skipped_reason
            apply_ms = 0.0
            failed = True

        result = result_fn(
            intents,
            values=values,
            result=result_mapping,
            apply_ms=apply_ms,
            failed=failed,
        )
        self._last_result = result
        return result


def _pending_target(intent: EditIntent) -> tuple[Any, ...]:
    kind = _view_value_kind(intent)
    if kind in {"camera", "transform"}:
        return ("transform", intent.usd_prim_path)
    return ("attribute", intent.usd_prim_path, intent.usd_attribute)


def combine_update_results(updates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _combined_update_result(updates)


def _value_update_result_mapping(result: OvrtxValueUpdateResult) -> dict[str, Any]:
    return {
        **dict(result.diagnostics),
        "updated_count": result.updated_count,
        "pending_simulation_time_ns": result.pending_simulation_time_ns,
    }


def update_result(
    intent: EditIntent,
    *,
    physics_generation_reset: bool,
    queued: bool = False,
) -> dict[str, Any]:
    target = intent
    return {
        "queued": queued,
        "values_written": False,
        "shape": intent.shape.value,
        "data_authority": intent.data_authority.value,
        "physics_generation_reset": physics_generation_reset,
        "target": {
            "usd_prim_path": target.usd_prim_path,
            "usd_attribute": target.usd_attribute,
            "usd_property_path": target.usd_property_path,
            "usd_layer_id": target.usd_layer_id,
            "blender_property_path": target.blender_property_path,
            "provenance": dict(target.provenance),
        },
    }


def view_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxTransformValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    first = intents[0]
    skipped_reason = "view_value_update_error" if failed else ""
    return {
        **update_result(first, physics_generation_reset=False, queued=False),
        "values_written": bool(values) and not failed,
        "failed": failed,
        "skipped_reason": skipped_reason,
        "value_apply_ms": float(apply_ms),
        # Requested intents vs applied values: transform batches coalesce
        # latest-wins per prim path before application (task04-01).
        "value_requested_count": len(intents),
        "value_count": len(values),
        "value_paths": [value.prim_path for value in values],
        "value_attributes": ["omni:xform" for _value in values],
        "targets": [_update_target_details(intent) for intent in intents],
        "result": dict(result),
    }


def material_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    return _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="material_value",
        unsupported_reason="unsupported_material_value_attribute",
    )


def camera_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    result = _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="camera_value",
        unsupported_reason="unsupported_camera_value_attribute",
    )
    first = intents[0]
    result["camera_value_probe_class"] = str(
        first.provenance.get("probe_class", "")
    )
    return result


def render_setting_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    update = _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="render_setting_value",
        unsupported_reason="unsupported_render_setting_value_type",
    )
    # Value types applied this batch (task01-04): int32 authors as USD ``Int``,
    # the firefly filter as ``Bool`` — evidence the exact dtype was written.
    update["value_types"] = [value.value_type for value in values]
    # The authored (attribute, value) pairs of this batch, carried so a rejected
    # live write can be folded back into the composition digest and re-keyed
    # (the task01-04 fallback needs the values that never landed live).
    update["render_setting_values"] = [
        {
            "attribute": value.attribute,
            "value": value.value,
            "value_type": value.value_type,
        }
        for value in values
    ]
    return update


RENDER_SETTING_WRITE_REASON = "render_setting_value_update_error"


def _update_lanes(update: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The per-lane update dicts of a (possibly combined) update result."""

    lanes = update.get("updates")
    if isinstance(lanes, (list, tuple)):
        return [lane for lane in lanes if isinstance(lane, Mapping)]
    return [update]


def _is_render_setting_lane(lane: Mapping[str, Any]) -> bool:
    target = lane.get("target")
    if isinstance(target, Mapping):
        provenance = target.get("provenance")
        if (
            isinstance(provenance, Mapping)
            and str(provenance.get("source", "")) == RENDER_SETTING_VALUE_SOURCE
        ):
            return True
    return str(lane.get("skipped_reason", "")).startswith("render_setting_value")


def render_setting_write_rejection(update: Mapping[str, Any]) -> dict[str, Any] | None:
    """Describe a rejected live render-setting write, or ``None``.

    A worker that raises on the render-setting ``update_attribute_values`` write
    surfaces as a failed render-setting lane. The returned mapping carries the
    authored attributes/values so the render loop can fold them back into the
    composition digest and re-key the session (task01-04 fallback), plus the
    rejection reason and the render product path(s) for diagnostics.
    """

    if not isinstance(update, Mapping):
        return None
    failed = [
        lane
        for lane in _update_lanes(update)
        if lane.get("failed") and _is_render_setting_lane(lane)
    ]
    if not failed:
        return None
    attributes: list[str] = []
    values: list[dict[str, Any]] = []
    reasons: list[str] = []
    paths: list[str] = []
    for lane in failed:
        attributes.extend(str(attribute) for attribute in lane.get("value_attributes", ()) or ())
        for value in lane.get("render_setting_values", ()) or ():
            if isinstance(value, Mapping):
                values.append(dict(value))
        reason = str(lane.get("skipped_reason", "") or "")
        if reason:
            reasons.append(reason)
        paths.extend(str(path) for path in lane.get("value_paths", ()) or ())
    return {
        "attributes": attributes,
        "values": values,
        "skipped_reason": ";".join(dict.fromkeys(reasons)) or RENDER_SETTING_WRITE_REASON,
        "render_product_paths": list(dict.fromkeys(paths)),
    }


def has_non_render_setting_failure(update: Mapping[str, Any]) -> bool:
    """True when a non-render-setting lane failed (a genuine, fatal failure)."""

    if not isinstance(update, Mapping):
        return False
    return any(
        lane.get("failed") and not _is_render_setting_lane(lane)
        for lane in _update_lanes(update)
    )


def light_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    return _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="light_value",
        unsupported_reason="unsupported_light_value_attribute",
    )


def world_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    first_target = intents[0]
    result = _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="world_value",
        unsupported_reason="unsupported_world_value_attribute",
    )
    result["world_dome_owner_path"] = first_target.usd_prim_path
    result["world_dome_conversion"] = dict(first_target.provenance.get("world_dome_conversion", {}))
    return result


def uv_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
) -> dict[str, Any]:
    result = _attribute_value_update_result(
        intents,
        values=values,
        result=result,
        apply_ms=apply_ms,
        failed=failed,
        prefix="uv_value",
        unsupported_reason="unsupported_uv_value_attribute",
    )
    result["value_types"] = [value.value_type for value in values]
    result["value_element_counts"] = [
        len(value.value) if isinstance(value.value, Sequence) else 0
        for value in values
    ]
    return result


def _attribute_value_update_result(
    intents: Sequence[EditIntent],
    *,
    values: Sequence[OvrtxAttributeValue],
    result: Mapping[str, Any],
    apply_ms: float,
    failed: bool,
    prefix: str,
    unsupported_reason: str,
) -> dict[str, Any]:
    first = intents[0]
    requested_count = len(intents)
    status = str(result.get("status", ""))
    result_skipped_reason = str(result.get("skipped_reason", ""))
    supported_by_client = (
        status != "unavailable"
        and result_skipped_reason != "value_update_unavailable"
    )
    supported_attribute = result_skipped_reason != unsupported_reason
    accepted_by_worker = (
        bool(values)
        and not failed
        and status not in {"unavailable", "error"}
        and not bool(result.get("failed", False))
    )
    if not failed:
        skipped_reason = ""
    elif result_skipped_reason:
        skipped_reason = result_skipped_reason
    else:
        skipped_reason = f"{prefix}_update_error"
    return {
        **update_result(first, physics_generation_reset=False, queued=False),
        "values_written": accepted_by_worker,
        "failed": failed,
        "skipped_reason": skipped_reason,
        "supported_by_client": supported_by_client,
        "supported_attribute": supported_attribute,
        "accepted_by_worker": accepted_by_worker,
        "value_apply_ms": float(apply_ms),
        "value_requested_count": requested_count,
        "value_count": len(values),
        "value_paths": [value.prim_path for value in values],
        "value_attributes": [value.attribute for value in values],
        "targets": [_update_target_details(intent) for intent in intents],
        "result": dict(result),
    }


def _view_value_kind(intent: EditIntent) -> str | None:
    target = intent
    attribute = target.usd_attribute
    blender_property_path = target.blender_property_path
    if str(target.provenance.get("source", "")) == RENDER_SETTING_VALUE_SOURCE:
        return "render_setting"
    if _is_viewport_camera_value(intent):
        return "camera"
    if str(target.provenance.get("source", "")) == "viewport_camera_projection":
        return "camera_value"
    if attribute in {"omni:xform", "xformOp:transform"} or blender_property_path == "matrix_world":
        return "transform"
    if attribute == uv_usd_prim.TARGET_USD_ATTRIBUTE or blender_property_path == uv_usd_prim.DEFAULT_BLENDER_PROPERTY_PATH:
        return "uv"
    if blender_property_path == "world_dome" or "world_dome_conversion" in target.provenance:
        return "world"
    if "light_path" in target.provenance:
        return "light"
    if "material_path" in target.provenance:
        return "material"
    if blender_property_path in {
        "energy",
        "data.energy",
        "color",
        "data.color",
        "normalize",
        "exposure",
        "size",
        "size_y",
        "spot_size",
        "spot_blend",
        "data.type",
        "data.shape",
    }:
        return "light"
    if (
        attribute.startswith("inputs:")
        or blender_property_path.startswith("material.")
        or blender_property_path.startswith("principled:")
        or blender_property_path in {
            "diffuse_color",
            "roughness",
            "metallic",
            "alpha",
            "base_color",
        }
    ):
        return "material"
    return None


def _is_viewport_camera_value(intent: EditIntent) -> bool:
    target = intent
    return (
        target.blender_property_path == "viewport_camera_matrix"
        or str(target.provenance.get("source", "")) == "viewport_camera"
    )


def _unsupported_update_application_reason(intent: EditIntent) -> str:
    if intent.data_authority == DataAuthority.VIEW:
        return "unsupported_view_value_update"
    return "unsupported_data_authority"


def _view_transform_value(intent: EditIntent) -> OvrtxTransformValue:
    prim_path = intent.usd_prim_path
    if not prim_path:
        raise ValueError("view value update is missing usd_prim_path")
    return OvrtxTransformValue(prim_path, _matrix4d_rows(intent.value))


def _camera_value(intent: EditIntent) -> OvrtxAttributeValue:
    return _attribute_value(
        intent,
        domain="camera",
        supported_attributes=usd_value_edit_support.CAMERA_USD_VALUE_TYPES,
    )


def _render_setting_value(intent: EditIntent) -> OvrtxAttributeValue:
    """Typed RTPT render-setting attribute value for the render product write.

    The dispatcher stamps the USD value-update type role (``Int`` for the
    ``int32`` bounce counts — never ``Int64`` — ``Bool`` for the firefly
    filter) into provenance so the exact documented dtype is written, and
    the value is coerced to match. The target prim is the active
    ``RenderProduct`` path resolved from the running session's request.
    """

    prim_path = intent.usd_prim_path
    attribute = intent.usd_attribute
    if not prim_path:
        raise ValueError("render setting update is missing usd_prim_path")
    if not attribute:
        raise ValueError("render setting update is missing usd_attribute")
    value_type = str(intent.provenance.get("value_type", ""))
    if value_type == "Int":
        typed_value: Any = int(intent.value)
    elif value_type == "Bool":
        typed_value = bool(intent.value)
    else:
        raise ValueError(
            f"unsupported render setting value type: {value_type!r}"
        )
    return OvrtxAttributeValue(prim_path, attribute, typed_value, value_type)


def _material_value(intent: EditIntent) -> OvrtxAttributeValue:
    return _attribute_value(
        intent,
        domain="material",
        supported_attributes=usd_value_edit_support.MATERIAL_USD_VALUE_TYPES,
    )


def _light_value(intent: EditIntent) -> OvrtxAttributeValue:
    return _attribute_value(
        intent,
        domain="light",
        supported_attributes=usd_value_edit_support.LIGHT_USD_VALUE_TYPES,
        bool_supported=True,
    )


def _world_value(intent: EditIntent) -> OvrtxAttributeValue:
    return _attribute_value(
        intent,
        domain="world",
        supported_attributes=usd_value_edit_support.WORLD_USD_VALUE_TYPES,
    )


def _attribute_value(
    intent: EditIntent,
    *,
    domain: str,
    supported_attributes: Mapping[str, str],
    bool_supported: bool = False,
) -> OvrtxAttributeValue:
    prim_path = intent.usd_prim_path
    attribute = intent.usd_attribute
    if not prim_path:
        raise ValueError(f"{domain} value update is missing usd_prim_path")
    if not attribute:
        raise ValueError(f"{domain} value update is missing usd_attribute")
    value_type = _value_type(attribute, supported_attributes, domain=domain)
    return OvrtxAttributeValue(
        prim_path,
        attribute,
        _typed_value(
            intent.value,
            attribute,
            value_type,
            domain=domain,
            bool_supported=bool_supported,
        ),
        value_type,
    )


def _uv_value(intent: EditIntent) -> OvrtxAttributeValue:
    prim_path = intent.usd_prim_path
    attribute = intent.usd_attribute
    if not prim_path:
        raise ValueError("UV value update is missing usd_prim_path")
    if not attribute:
        raise ValueError("UV value update is missing usd_attribute")
    value = _uv_float2_array(intent.value, attribute)
    _validate_uv_loop_order_validation(intent, prim_path, attribute, element_count=len(value))
    return OvrtxAttributeValue(
        prim_path,
        attribute,
        value,
        _uv_value_type(attribute),
    )


def _validate_uv_loop_order_validation(
    intent: EditIntent,
    prim_path: str,
    attribute: str,
    *,
    element_count: int,
) -> None:
    validation = intent.provenance.get("uv_loop_order_validation")
    error_prefix = "unsupported UV value attribute"
    if not isinstance(validation, Mapping):
        raise ValueError(f"{error_prefix}: missing uv_loop_order_validation")
    if str(validation.get("status", "")) != uv_usd_prim.RESOLVED:
        raise ValueError(f"{error_prefix}: unresolved uv_loop_order_validation")
    if str(validation.get("validation_kind", "")) != uv_usd_prim.VALIDATION_KIND:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation kind mismatch")
    if str(validation.get("mesh_prim_path", "")) != prim_path:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation mesh mismatch")
    if str(validation.get("target_attribute", "")) != attribute:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation attribute mismatch")
    if str(validation.get("value_type", "")) != uv_usd_prim.VALUE_TYPE:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation value type mismatch")
    if str(validation.get("interpolation", "")) != "faceVarying":
        raise ValueError(f"{error_prefix}: uv_loop_order_validation interpolation mismatch")
    if bool(validation.get("indexed", True)):
        raise ValueError(f"{error_prefix}: uv_loop_order_validation indexed primvar")
    if str(validation.get("primvar_shape_status", "")) != uv_usd_prim.RESOLVED:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation shape unresolved")
    for key in ("uv_layer_name", "topology_fingerprint", "blender_uv_digest", "source_uv_digest"):
        if not str(validation.get(key, "")):
            raise ValueError(f"{error_prefix}: uv_loop_order_validation missing {key}")
    try:
        float(validation.get("tolerance", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation tolerance invalid") from exc
    try:
        validated_count = int(validation.get("element_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation element count invalid") from exc
    if validated_count != element_count:
        raise ValueError(f"{error_prefix}: uv_loop_order_validation element count mismatch")


def _update_target_details(intent: EditIntent) -> dict[str, Any]:
    target = intent
    return {
        "shape": intent.shape.value,
        "data_authority": intent.data_authority.value,
        "usd_prim_path": target.usd_prim_path,
        "usd_attribute": target.usd_attribute,
        "usd_property_path": target.usd_property_path,
        "usd_layer_id": target.usd_layer_id,
        "blender_property_path": target.blender_property_path,
        "provenance": dict(target.provenance),
    }


def _matrix4d_rows(value: Any) -> list[list[float]]:
    if isinstance(value, Mapping):
        value = value.get("matrix", value.get("omni:xform"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("transform value must be a 4x4 or flat 16-value matrix")

    rows = list(value)
    if len(rows) == 16 and all(not isinstance(item, Sequence) or isinstance(item, (str, bytes)) for item in rows):
        flat = [_float_matrix_value(item) for item in rows]
        return [flat[index : index + 4] for index in range(0, 16, 4)]

    if len(rows) != 4:
        raise ValueError("transform value must have four rows")
    matrix: list[list[float]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("transform matrix rows must be sequences")
        values = list(row)
        if len(values) != 4:
            raise ValueError("transform matrix rows must have four values")
        matrix.append([_float_matrix_value(item) for item in values])
    return matrix


def _value_type(attribute: str, supported_attributes: Mapping[str, str], *, domain: str) -> str:
    try:
        return supported_attributes[attribute]
    except KeyError as exc:
        raise ValueError(f"unsupported {domain} value attribute: {attribute}") from exc


def _typed_value(
    value: Any,
    attribute: str,
    value_type: str,
    *,
    domain: str,
    bool_supported: bool = False,
) -> Any:
    if value_type == "Bool" and bool_supported:
        return bool(value)
    if value_type == "Float":
        return _float_value(value, attribute, domain)
    if value_type == "Color3f":
        sequence = _float_sequence(value, attribute, domain)
        if len(sequence) not in {3, 4}:
            raise ValueError(f"{attribute} {domain} value must have three or four numeric values")
        return sequence[:3]
    if value_type == "Float2":
        sequence = _float_sequence(value, attribute, domain)
        if len(sequence) != 2:
            raise ValueError(f"{attribute} {domain} value must have two numeric values")
        return sequence
    raise ValueError(f"unsupported {domain} value attribute: {attribute}")


def _uv_float2_array(value: Any, attribute: str) -> list[tuple[float, float]]:
    value_type = _uv_value_type(attribute)
    if value_type != uv_usd_prim.VALUE_TYPE:
        raise ValueError(f"unsupported UV value attribute: {attribute}")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{attribute} UV value must be a sequence of float2 values")
    pairs: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError(f"{attribute} UV value entries must be 2-value sequences")
        try:
            pairs.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{attribute} UV value entries must be numeric") from exc
    return pairs


def _uv_value_type(attribute: str) -> str:
    try:
        return SUPPORTED_UV_VALUE_ATTRIBUTES[attribute]
    except KeyError as exc:
        raise ValueError(f"unsupported UV value attribute: {attribute}") from exc


def _float_sequence(value: Any, attribute: str, domain: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{attribute} {domain} value must be a numeric sequence")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{attribute} {domain} value must be numeric") from exc


def _float_value(value: Any, attribute: str, domain: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{attribute} {domain} value must be numeric") from exc


def _float_matrix_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transform matrix value is not numeric: {value!r}") from exc
