# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caller-facing OVRTX value-update operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class OvrtxTransformValue:
    prim_path: str
    matrix: Any

    def __post_init__(self) -> None:
        if not isinstance(self.prim_path, str) or not self.prim_path.strip():
            raise ValueError("OVRTX transform value requires a prim path")


@dataclass(frozen=True)
class OvrtxAttributeValue:
    prim_path: str
    attribute: str
    value: Any
    value_type: str

    def __post_init__(self) -> None:
        for field_name in ("prim_path", "attribute", "value_type"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"OVRTX attribute value requires {field_name.replace('_', ' ')}")


@dataclass(frozen=True)
class OvrtxValueUpdateResult:
    updated_count: int
    pending_simulation_time_ns: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.updated_count) is not int or self.updated_count < 0:
            raise ValueError("OVRTX value update count must be a nonnegative integer")
        pending = self.pending_simulation_time_ns
        if pending is not None and (type(pending) is not int or pending < 0):
            raise ValueError("OVRTX value update pending time must be a nonnegative integer")
        if self.updated_count == 0 and pending is not None:
            raise ValueError("empty OVRTX value update cannot have a pending time")
        if self.updated_count > 0 and pending is None:
            raise ValueError("nonempty OVRTX value update requires a pending time")


class OvrtxUpdatePort(Protocol):
    def update_transforms(
        self,
        values: Sequence[OvrtxTransformValue],
    ) -> OvrtxValueUpdateResult: ...

    def update_attribute_values(
        self,
        values: Sequence[OvrtxAttributeValue],
    ) -> OvrtxValueUpdateResult: ...


@dataclass(frozen=True)
class OvrtxSessionUpdatePort:
    client: Any
    simulation_id: str

    def update_transforms(
        self,
        values: Sequence[OvrtxTransformValue],
    ) -> OvrtxValueUpdateResult:
        batch = _transform_values(values)
        if not batch:
            return OvrtxValueUpdateResult(0)
        return _result(
            self.client.update_transforms(self.simulation_id, batch),
            "update_transforms",
            len(batch),
        )

    def update_attribute_values(
        self,
        values: Sequence[OvrtxAttributeValue],
    ) -> OvrtxValueUpdateResult:
        batch = _attribute_values(values)
        if not batch:
            return OvrtxValueUpdateResult(0)
        updater = getattr(self.client, "update_attribute_values", None)
        if not callable(updater):
            raise RuntimeError("OVRTX client does not support attribute value updates")
        return _result(
            updater(self.simulation_id, batch),
            "update_attribute_values",
            len(batch),
        )


def _transform_values(
    values: Sequence[OvrtxTransformValue],
) -> tuple[OvrtxTransformValue, ...]:
    batch = tuple(values)
    for index, value in enumerate(batch):
        if not isinstance(value, OvrtxTransformValue):
            raise TypeError(f"transform values[{index}] must be an OvrtxTransformValue")
    _reject_duplicates((value.prim_path for value in batch), "transform prim path")
    return batch


def _attribute_values(
    values: Sequence[OvrtxAttributeValue],
) -> tuple[OvrtxAttributeValue, ...]:
    batch = tuple(values)
    for index, value in enumerate(batch):
        if not isinstance(value, OvrtxAttributeValue):
            raise TypeError(f"attribute values[{index}] must be an OvrtxAttributeValue")
    _reject_duplicates(
        ((value.prim_path, value.attribute) for value in batch),
        "attribute target",
    )
    return batch


def _reject_duplicates(values: Any, label: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate OVRTX {label}: {value}")
        seen.add(value)


def _result(value: Any, operation: str, expected_count: int) -> OvrtxValueUpdateResult:
    if not isinstance(value, OvrtxValueUpdateResult):
        raise TypeError(f"OVRTX client {operation} must return OvrtxValueUpdateResult")
    if value.updated_count != expected_count:
        raise RuntimeError(
            f"OVRTX client {operation} updated {value.updated_count} of {expected_count} values"
        )
    return value


__all__ = [
    "OvrtxAttributeValue",
    "OvrtxSessionUpdatePort",
    "OvrtxTransformValue",
    "OvrtxUpdatePort",
    "OvrtxValueUpdateResult",
]
