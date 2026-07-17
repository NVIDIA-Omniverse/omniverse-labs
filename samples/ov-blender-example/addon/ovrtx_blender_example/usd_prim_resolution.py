# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared result contract for USD prim resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


class UsdPrimResolutionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class UsdPrimResolution(Generic[T]):
    status: UsdPrimResolutionStatus
    value: T | None = None
    error_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, UsdPrimResolutionStatus):
            raise ValueError(f"unsupported USD prim resolution status: {self.status!r}")
        if self.status == UsdPrimResolutionStatus.OK:
            if self.value is None:
                raise ValueError("successful USD prim resolution requires a value")
            if self.error_reason:
                raise ValueError("successful USD prim resolution cannot have an error reason")
        elif self.status == UsdPrimResolutionStatus.ERROR:
            if self.value is not None:
                raise ValueError("failed USD prim resolution cannot have a value")
            if not isinstance(self.error_reason, str) or not self.error_reason.strip():
                raise ValueError("failed USD prim resolution requires an error reason")
        else:
            raise ValueError(f"unsupported USD prim resolution status: {self.status!r}")
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    def diagnostics_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.diagnostics.items()}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw(item) for item in value]
    return value


__all__ = ["UsdPrimResolution", "UsdPrimResolutionStatus"]
