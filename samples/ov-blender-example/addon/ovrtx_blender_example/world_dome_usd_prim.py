# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the configured World dome to an existing USD light prim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import usd_paths as usd_paths
from .usd_prim_resolution import UsdPrimResolution, UsdPrimResolutionStatus
from .world_dome_conversion import DEFAULT_DOME_OWNER_PATH, SUPPORTED_USD_ATTRIBUTES


ERROR_USD_STAGE_UNAVAILABLE = "usd_stage_unavailable"
ERROR_NO_DOME_PRIM = "no_dome_prim"
ERROR_WRONG_DOME_PRIM_TYPE = "wrong_dome_prim_type"
ERROR_MISSING_DOME_PRIM_ATTRIBUTE = "missing_dome_prim_attribute"


@dataclass(frozen=True)
class WorldDomeUsdPrim:
    prim_path: str
    usd_family: str

    def __post_init__(self) -> None:
        if not self.prim_path.strip():
            raise ValueError("World dome USD prim requires prim path")
        if not self.usd_family.strip():
            raise ValueError("World dome USD prim requires USD family")


def _world_dome_prim_index_from_prims(
    prims: Iterable[Any],
    *,
    dome_prim_path: str = DEFAULT_DOME_OWNER_PATH,
) -> dict[str, Any]:
    target_path = usd_paths.clean_usd_path(dome_prim_path) or DEFAULT_DOME_OWNER_PATH
    for prim in prims:
        if usd_paths.usd_prim_path_from_prim(prim) != target_path:
            continue
        family = usd_paths.usd_prim_type_name_from_prim(prim)
        return {
            "available": True,
            "stage_reason": "",
            "prim_path": target_path,
            "usd_family": family,
            "missing_attributes": _missing_attributes(prim),
        }
    return {
        "available": True,
        "stage_reason": "",
        "prim_path": target_path,
        "usd_family": "",
        "missing_attributes": (),
    }


def resolve_world_dome_usd_prim(
    index: Mapping[str, Any],
) -> UsdPrimResolution[WorldDomeUsdPrim]:
    diagnostics = {
        "prim_path": str(index.get("prim_path", DEFAULT_DOME_OWNER_PATH)),
        "usd_family": str(index.get("usd_family", "")),
        "missing_attributes": tuple(index.get("missing_attributes", ())),
    }
    if not index.get("available", False):
        return _error(
            ERROR_USD_STAGE_UNAVAILABLE,
            {**diagnostics, "stage_reason": str(index.get("stage_reason", ""))},
        )
    if not diagnostics["usd_family"]:
        return _error(ERROR_NO_DOME_PRIM, diagnostics)
    if diagnostics["usd_family"] != "DomeLight":
        return _error(ERROR_WRONG_DOME_PRIM_TYPE, diagnostics)
    if diagnostics["missing_attributes"]:
        return _error(ERROR_MISSING_DOME_PRIM_ATTRIBUTE, diagnostics)
    return UsdPrimResolution(
        UsdPrimResolutionStatus.OK,
        WorldDomeUsdPrim(diagnostics["prim_path"], diagnostics["usd_family"]),
        diagnostics=diagnostics,
    )


def _error(
    reason: str,
    diagnostics: Mapping[str, Any],
) -> UsdPrimResolution[WorldDomeUsdPrim]:
    return UsdPrimResolution(
        UsdPrimResolutionStatus.ERROR,
        error_reason=reason,
        diagnostics=diagnostics,
    )


def _missing_attributes(prim: Any) -> tuple[str, ...]:
    return tuple(
        attribute for attribute in SUPPORTED_USD_ATTRIBUTES if not _prim_has_attribute(prim, attribute)
    )


def _prim_has_attribute(prim: Any, attribute: str) -> bool:
    attributes = getattr(prim, "attributes", None)
    if attributes is not None:
        return attribute in set(attributes)
    has_attribute = getattr(prim, "HasAttribute", None)
    if callable(has_attribute):
        try:
            return bool(has_attribute(attribute))
        except Exception:
            return False
    get_attribute = getattr(prim, "GetAttribute", None)
    if callable(get_attribute):
        try:
            usd_attribute = get_attribute(attribute)
        except Exception:
            return False
        is_valid = getattr(usd_attribute, "IsValid", None)
        if callable(is_valid):
            try:
                return bool(is_valid())
            except Exception:
                return False
        return usd_attribute is not None
    return False


__all__ = [
    "WorldDomeUsdPrim",
    "resolve_world_dome_usd_prim",
]
