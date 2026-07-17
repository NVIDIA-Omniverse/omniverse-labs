# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contracts for Blender-authored value edit conversion policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


STATUS_SUPPORTED = "supported_value"
STATUS_TOPOLOGY = "topology"
STATUS_UNSUPPORTED = "unsupported_first_scope"
STATUS_NON_RENDER = "non_render"

# Diagnostic classification vocabulary (blender-live-render task04-07):
# exactly four values, normalized across the material/light/world/camera
# policies. Edit records, workflow events, and user-visible reports use
# these; the ``STATUS_*`` constants above remain the policies' internal
# classification statuses.
CLASSIFICATION_SUPPORTED = "supported"
CLASSIFICATION_UNSUPPORTED = "unsupported"
CLASSIFICATION_NON_RENDERING = "non_rendering"
CLASSIFICATION_TOPOLOGY = "topology"

DIAGNOSTIC_CLASSIFICATIONS = (
    CLASSIFICATION_SUPPORTED,
    CLASSIFICATION_UNSUPPORTED,
    CLASSIFICATION_NON_RENDERING,
    CLASSIFICATION_TOPOLOGY,
)

_STATUS_CLASSIFICATIONS = {
    STATUS_SUPPORTED: CLASSIFICATION_SUPPORTED,
    STATUS_TOPOLOGY: CLASSIFICATION_TOPOLOGY,
    STATUS_UNSUPPORTED: CLASSIFICATION_UNSUPPORTED,
    STATUS_NON_RENDER: CLASSIFICATION_NON_RENDERING,
    # Idempotent on already-normalized values (``topology`` is shared).
    CLASSIFICATION_SUPPORTED: CLASSIFICATION_SUPPORTED,
    CLASSIFICATION_UNSUPPORTED: CLASSIFICATION_UNSUPPORTED,
    CLASSIFICATION_NON_RENDERING: CLASSIFICATION_NON_RENDERING,
}

# Non-rendering classification reasons share the ``non_runtime`` prefix
# across every policy's non-render table (asserted by tests).
_NON_RENDERING_REASON_PREFIX = "non_runtime"


def normalized_classification(status: Any) -> str:
    """Map a policy classification status onto the diagnostic vocabulary.

    Every policy-internal status normalizes to one of the four
    ``DIAGNOSTIC_CLASSIFICATIONS``; unknown non-empty statuses fail closed
    to ``unsupported`` (a status the vocabulary cannot express is by
    definition not expressible as a live value update), and empty input
    stays empty (no classification available).
    """

    key = str(status or "").strip()
    if not key:
        return ""
    return _STATUS_CLASSIFICATIONS.get(key, CLASSIFICATION_UNSUPPORTED)


def classification_for_unsupported_reason(reason: Any) -> str:
    """Classify an unsupported-reason string as unsupported/non-rendering."""

    if str(reason or "").startswith(_NON_RENDERING_REASON_PREFIX):
        return CLASSIFICATION_NON_RENDERING
    return CLASSIFICATION_UNSUPPORTED


def display_field_name(blender_property_path: Any) -> str:
    """Human-facing field name for user-visible reports (names the field)."""

    field = str(blender_property_path or "").strip()
    if field.startswith("principled:"):
        return field.split(":", 1)[1]
    return field


def classification_report_message(classification: str, *, field: str) -> str:
    """One-line user-visible report per classification (task04-07 phrasing).

    ``unsupported`` names the field and says value updates cannot express
    it; ``topology`` reports the generation route ("applies on next scene
    update"); ``non_rendering`` says the field does not affect rendering.
    ``supported`` (and unknown) produce no user-visible report.
    """

    label = str(field or "").strip() or "This edit"
    if classification == CLASSIFICATION_TOPOLOGY:
        return f"OVRTX: '{label}' applies on next scene update."
    if classification == CLASSIFICATION_NON_RENDERING:
        return f"OVRTX: '{label}' does not affect rendering; no update sent."
    if classification == CLASSIFICATION_UNSUPPORTED:
        return f"OVRTX: '{label}' is not supported by OVRTX value updates."
    return ""

BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS = {
    field: "non_runtime_blender_datablock_field"
    for field in (
        "use_fake_user",
        "use_extra_user",
        "asset_data",
        "is_runtime_data",
        "tag",
    )
}


@dataclass(frozen=True)
class UsdAttributeValue:
    name: str
    value: Any
    value_type: str
    blender_property_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldClassification:
    status: str
    reason: str
    usd_attributes: tuple[str, ...] = ()


class ValueEditConversionPolicy(Protocol):
    """Module-shaped policy for one Blender value-edit family."""

    SUPPORTED_USD_ATTRIBUTES: Mapping[str, str]
    EDIT_VALUE_ATTRIBUTES_BY_FIELD: Mapping[str, tuple[str, ...]]
    EDIT_VALUE_CONCEPTS: frozenset[str]
    EDIT_TOPOLOGY_CONCEPTS: frozenset[str]
    EDIT_TOPOLOGY_KINDS: frozenset[str]
    classify_field: Callable[..., FieldClassification]
    usd_attribute_values: Callable[[Any], tuple[UsdAttributeValue, ...]]


@dataclass(frozen=True)
class ValueEditConversionPolicies:
    material: ValueEditConversionPolicy
    light: ValueEditConversionPolicy
    world: ValueEditConversionPolicy
    #: Camera projection/framing value policy (blender-live-render
    #: task04-05). Defaults to ``None`` so injected three-policy fakes keep
    #: their prior behavior; consumers read it with ``getattr`` guards.
    camera: ValueEditConversionPolicy | None = None


def default_value_edit_conversion_policies() -> ValueEditConversionPolicies:
    """Return the concrete value edit conversion policy modules."""

    from . import camera_value_conversion
    from . import light_value_conversion
    from . import material_value_conversion
    from . import world_dome_conversion

    return ValueEditConversionPolicies(
        material=material_value_conversion,
        light=light_value_conversion,
        world=world_dome_conversion,
        camera=camera_value_conversion,
    )


def classify_mapped_field(
    field: str,
    *,
    non_render: Mapping[str, str] | None = None,
    topology: Mapping[str, str] | None = None,
    unsupported: Mapping[str, str] | None = None,
) -> FieldClassification | None:
    for status, reasons in (
        (STATUS_NON_RENDER, non_render),
        (STATUS_TOPOLOGY, topology),
        (STATUS_UNSUPPORTED, unsupported),
    ):
        if reasons is not None and field in reasons:
            return FieldClassification(status, reasons[field])
    return None


def node_input(node: Any, name: str) -> Any:
    inputs = getattr(node, "inputs", {})
    getter = getattr(inputs, "get", None)
    if callable(getter):
        return getter(name)
    try:
        return inputs[name]
    except (KeyError, TypeError, IndexError):
        return None


def socket_is_linked(socket: Any) -> bool:
    return bool(getattr(socket, "is_linked", False)) if socket is not None else False


def float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
