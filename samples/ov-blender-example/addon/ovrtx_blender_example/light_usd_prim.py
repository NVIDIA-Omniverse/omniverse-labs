# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Blender lights to existing USD light prims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import light_value_conversion as light_conversion
from . import usd_paths as usd_paths
from .usd_prim_resolution import UsdPrimResolution, UsdPrimResolutionStatus


ERROR_AMBIGUOUS = "ambiguous"
ERROR_USD_STAGE_UNAVAILABLE = "usd_stage_unavailable"
ERROR_MISSING_LIGHT_NAME = "missing_light_name"
ERROR_NO_PATH_MATCH = "no_path_match"
ERROR_AUTHORING_PATH_NOT_IN_SCENE = "authoring_prim_path_not_in_scene"
MATCH_SOURCE_USD_PATH = "sourceUsdPath"
MATCH_HIERARCHY_PATH = "hierarchy_path"
MATCH_SOURCE_SESSION_UID = "blender_session_uid"
MATCH_AUTHORING_PRIM_PATH = "authoring_prim_path"

USD_LIGHT_FAMILIES = frozenset({"SphereLight", "RectLight", "DiskLight", "DistantLight"})
SPOT_SHAPING_ATTRIBUTES = frozenset(
    {"inputs:shaping:cone:angle", "inputs:shaping:cone:softness"}
)


@dataclass(frozen=True)
class LightUsdPrim:
    prim_path: str
    usd_family: str
    authored_light_form: str

    def __post_init__(self) -> None:
        if not self.prim_path.strip():
            raise ValueError("light USD prim requires prim path")
        if not self.usd_family.strip():
            raise ValueError("light USD prim requires USD family")


def _light_prim_index_from_prims(prims: Iterable[Any]) -> dict[str, Any]:
    candidates = []
    for prim in prims:
        path = usd_paths.usd_prim_path_from_prim(prim)
        family = usd_paths.usd_prim_type_name_from_prim(prim)
        if not path or family not in USD_LIGHT_FAMILIES:
            continue
        candidates.append(
            {
                "prim_path": path,
                "parent_path": usd_paths.parent_usd_path(path),
                "usd_family": family,
                "authored_light_form": _authored_light_form_from_prim(prim, family),
            }
        )
    return {
        "available": True,
        "reason": "",
        "prim_paths": tuple(candidate["prim_path"] for candidate in candidates),
        "candidates": tuple(candidates),
    }


def resolve_light_usd_prim(
    light_object: Any,
    index: Mapping[str, Any],
    *,
    known_prim_path: str = "",
) -> UsdPrimResolution[LightUsdPrim]:
    name = str(getattr(light_object, "name_full", getattr(light_object, "name", "")))
    # Read the authoring identity through the shared helper: evaluated
    # depsgraph copies of some ID types drop add-on PointerProperty data in
    # Blender 5.1 (materials verified; task04-02), and the helper falls back
    # to the original datablock.
    authoring_path = usd_paths.authoring_prim_path(light_object)
    diagnostics = {
        "light_name": name,
        "match_source": "",
        "source_usd_path": usd_paths.source_usd_path_from_blender_id(light_object),
        "authoring_prim_path": authoring_path,
        "candidate_count": 0,
        "candidates": (),
    }
    if not index.get("available", False):
        return _error(
            ERROR_USD_STAGE_UNAVAILABLE,
            {**diagnostics, "stage_reason": str(index.get("reason", ""))},
        )

    if known_prim_path:
        return _resolve_path(
            diagnostics,
            index,
            known_prim_path,
            match_source=MATCH_SOURCE_SESSION_UID,
        )

    # Authored scene composition (blender-live-render task04-03): the
    # topology orchestrator assigns each converted light object's root prim
    # path to the ``ov.usd.prim_path`` authoring property and the light
    # converter authors the UsdLux prim at exactly that path. Authored
    # generations carry no ``sourceUsdPath`` metadata and light names can
    # sanitize differently from their USD leaf names ("Key Light" ->
    # ``Key_Light``), so the authoring identity — verified against the
    # scanned light index — is the primary resolution source; the
    # source-path and leaf-name matches remain the direct-USD fallbacks.
    if authoring_path:
        result = _resolve_path(
            diagnostics,
            index,
            authoring_path,
            match_source=MATCH_AUTHORING_PRIM_PATH,
        )
        if result.error_reason != ERROR_NO_PATH_MATCH:
            return result

    source_path = usd_paths.clean_usd_path(diagnostics["source_usd_path"])
    if source_path:
        return _authoring_fail_closed(
            _resolve_path(
                diagnostics,
                index,
                source_path,
                match_source=MATCH_SOURCE_USD_PATH,
            ),
            authoring_path,
        )

    hierarchy_path = usd_paths.hierarchy_usd_path(light_object)
    if hierarchy_path:
        result = _resolve_path(
            diagnostics,
            index,
            hierarchy_path,
            match_source=MATCH_HIERARCHY_PATH,
        )
        if result.error_reason != ERROR_NO_PATH_MATCH:
            return result

    normalized_name = usd_paths.normalized_blender_object_name(name)
    if not normalized_name:
        return _authoring_fail_closed(
            _error(ERROR_MISSING_LIGHT_NAME, diagnostics),
            authoring_path,
        )
    candidates = [
        candidate
        for candidate in index.get("candidates", ())
        if usd_paths.normalized_usd_leaf_name(str(candidate.get("parent_path", "")))
        == normalized_name
        or usd_paths.normalized_usd_leaf_name(str(candidate.get("prim_path", "")))
        == normalized_name
    ]
    return _authoring_fail_closed(
        _resolve_candidates(diagnostics, candidates, match_source=MATCH_HIERARCHY_PATH),
        authoring_path,
    )


def _authoring_fail_closed(
    result: UsdPrimResolution[LightUsdPrim],
    authoring_path: str,
) -> UsdPrimResolution[LightUsdPrim]:
    """Downgrade generic misses to the precise fail-closed reason.

    A light that claims an authored identity the scanned composition does
    not contain (stale reconcile, or a stage the converters did not emit)
    must fail with the authoring-specific reason instead of a generic name
    miss (04-01/04-02 precedent).
    """

    if (
        authoring_path
        and result.status is UsdPrimResolutionStatus.ERROR
        and result.error_reason in {ERROR_NO_PATH_MATCH, ERROR_MISSING_LIGHT_NAME}
    ):
        return UsdPrimResolution(
            UsdPrimResolutionStatus.ERROR,
            error_reason=ERROR_AUTHORING_PATH_NOT_IN_SCENE,
            diagnostics=result.diagnostics,
        )
    return result


def _resolve_path(
    diagnostics: Mapping[str, Any],
    index: Mapping[str, Any],
    source_path: str,
    *,
    match_source: str,
) -> UsdPrimResolution[LightUsdPrim]:
    candidates = [
        candidate
        for candidate in index.get("candidates", ())
        if candidate.get("prim_path") == source_path
        or candidate.get("parent_path") == source_path
    ]
    return _resolve_candidates(diagnostics, candidates, match_source=match_source)


def _resolve_candidates(
    diagnostics: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    *,
    match_source: str,
) -> UsdPrimResolution[LightUsdPrim]:
    candidate_list = tuple(dict(candidate) for candidate in candidates)
    evidence = {
        **diagnostics,
        "match_source": match_source,
        "candidate_count": len(candidate_list),
        "candidates": candidate_list,
    }
    if not candidate_list:
        return _error(ERROR_NO_PATH_MATCH, evidence)
    if len(candidate_list) > 1:
        return _error(ERROR_AMBIGUOUS, evidence)
    candidate = candidate_list[0]
    return UsdPrimResolution(
        UsdPrimResolutionStatus.OK,
        LightUsdPrim(
            prim_path=str(candidate["prim_path"]),
            usd_family=str(candidate["usd_family"]),
            authored_light_form=str(candidate.get("authored_light_form", "")),
        ),
        diagnostics=evidence,
    )


def _error(reason: str, diagnostics: Mapping[str, Any]) -> UsdPrimResolution[LightUsdPrim]:
    return UsdPrimResolution(
        UsdPrimResolutionStatus.ERROR,
        error_reason=reason,
        diagnostics=diagnostics,
    )


def _authored_light_form_from_prim(prim: Any, family: str) -> str:
    fixed_form = light_conversion.authored_light_form_from_usd_family(family)
    if fixed_form:
        return fixed_form
    if family != "SphereLight":
        return ""
    authored_names = _authored_attribute_names(prim)
    if authored_names is None:
        return ""
    if any(name in authored_names for name in SPOT_SHAPING_ATTRIBUTES):
        return light_conversion.AUTHORED_LIGHT_FORM_SPOT
    return light_conversion.AUTHORED_LIGHT_FORM_POINT


def _authored_attribute_names(prim: Any) -> frozenset[str] | None:
    direct = getattr(prim, "authored_attributes", None)
    if direct is None:
        direct = getattr(prim, "attributes", None)
    if isinstance(direct, Mapping):
        return frozenset(str(name) for name in direct)
    if direct is not None and not isinstance(direct, (str, bytes)):
        try:
            return frozenset(str(name) for name in direct)
        except TypeError:
            pass
    getter = getattr(prim, "GetAuthoredProperties", None)
    if callable(getter):
        try:
            names = [_property_name(prop) for prop in getter()]
            return frozenset(name for name in names if name)
        except Exception:
            return None
    attr_getter = getattr(prim, "GetAttribute", None)
    if callable(attr_getter):
        names = []
        for name in SPOT_SHAPING_ATTRIBUTES:
            try:
                attr = attr_getter(name)
            except Exception:
                continue
            if _attribute_has_authored_value(attr):
                names.append(name)
        return frozenset(names)
    return None


def _property_name(prop: Any) -> str:
    getter = getattr(prop, "GetName", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    return str(getattr(prop, "name", ""))


def _attribute_has_authored_value(attr: Any) -> bool:
    if attr is None:
        return False
    authored = getattr(attr, "HasAuthoredValueOpinion", None)
    if callable(authored):
        try:
            return bool(authored())
        except Exception:
            return False
    valid = getattr(attr, "IsValid", None)
    if callable(valid):
        try:
            return bool(valid())
        except Exception:
            return False
    return bool(attr)


__all__ = [
    "ERROR_AUTHORING_PATH_NOT_IN_SCENE",
    "LightUsdPrim",
    "MATCH_AUTHORING_PRIM_PATH",
    "USD_LIGHT_FAMILIES",
    "resolve_light_usd_prim",
]
