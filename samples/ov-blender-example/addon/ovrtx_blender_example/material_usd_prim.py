# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Blender materials to existing USD material value prims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import usd_paths as usd_paths
from .usd_prim_resolution import UsdPrimResolution, UsdPrimResolutionStatus


DEFAULT_BLENDER_MATERIAL_PROPERTY = "diffuse_color"
DEFAULT_USD_MATERIAL_ATTRIBUTE = "inputs:diffuseColor"
ERROR_AMBIGUOUS = "ambiguous"
ERROR_USD_STAGE_UNAVAILABLE = "usd_stage_unavailable"
ERROR_MISSING_MATERIAL_NAME = "missing_material_name"
ERROR_MISSING_PRIM_ATTRIBUTE = "missing_prim_attribute"
ERROR_NO_PATH_MATCH = "no_path_match"
ERROR_AUTHORING_PATH_NOT_IN_SCENE = "authoring_prim_path_not_in_scene"
MATCH_SOURCE_USD_PATH = "sourceUsdPath"
MATCH_HIERARCHY_PATH = "material_path"
MATCH_AUTHORING_PRIM_PATH = "authoring_prim_path"


@dataclass(frozen=True)
class MaterialUsdPrim:
    material_prim_path: str
    prim_path: str
    usd_attribute: str
    # Whether the resolved shader attribute has an authored connection
    # (texture-driven input) on the scanned stage. Connected inputs are not
    # value-editable; the edit builder uses this to classify texture
    # connects/disconnects as material-graph topology (task04-02).
    connected: bool = False

    def __post_init__(self) -> None:
        for name in ("material_prim_path", "prim_path", "usd_attribute"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"material USD prim requires {name.replace('_', ' ')}")


def _material_prim_index_from_prims(
    prims: Iterable[Any],
    *,
    usd_attribute: str = DEFAULT_USD_MATERIAL_ATTRIBUTE,
    material_type: Any | None = None,
) -> dict[str, Any]:
    prim_facts = [_prim_facts(prim, usd_attribute, material_type) for prim in prims]
    prim_facts = [facts for facts in prim_facts if facts["path"]]
    material_paths = tuple(facts["path"] for facts in prim_facts if facts["is_material"])
    value_facts = tuple(facts for facts in prim_facts if facts["has_attribute"])
    value_paths = tuple(facts["path"] for facts in value_facts)
    candidates = tuple(
        {
            "material_prim_path": material_path,
            "prim_path": facts["path"],
            "usd_attribute": usd_attribute,
            "connected": bool(facts["connected"]),
        }
        for material_path in material_paths
        for facts in value_facts
        if facts["path"] == material_path or facts["path"].startswith(material_path + "/")
    )
    return {
        "available": True,
        "reason": "",
        "usd_attribute": usd_attribute,
        "material_prim_paths": material_paths,
        "value_prim_paths": value_paths,
        "candidates": candidates,
    }


def resolve_material_usd_prim(
    material: Any,
    index: Mapping[str, Any],
    *,
    usd_attribute: str = DEFAULT_USD_MATERIAL_ATTRIBUTE,
    property_name: str = DEFAULT_BLENDER_MATERIAL_PROPERTY,
) -> UsdPrimResolution[MaterialUsdPrim]:
    material_name = str(getattr(material, "name_full", getattr(material, "name", "")))
    authoring_path = usd_paths.authoring_prim_path(material)
    diagnostics = {
        "material_name": material_name,
        "property_name": property_name,
        "usd_attribute": usd_attribute,
        "match_source": "",
        "source_usd_path": usd_paths.source_usd_path_from_blender_id(material),
        "authoring_prim_path": authoring_path,
        "candidate_count": 0,
        "candidates": (),
    }
    if not index.get("available", False):
        return _error(
            ERROR_USD_STAGE_UNAVAILABLE,
            {**diagnostics, "stage_reason": str(index.get("reason", ""))},
        )

    # Authored scene composition (blender-live-render task04-02): the
    # topology orchestrator assigns each visual material's root prim path to
    # the ``ov.usd.prim_path`` authoring property and the converter authors
    # the Material prim at exactly that path. Authored generations carry no
    # ``sourceUsdPath`` metadata and material names can sanitize differently
    # from their USD leaf names, so the authoring identity — verified
    # against the scanned index — is the primary resolution source; the
    # source-path and leaf-name matches remain the direct-USD fallbacks.
    if authoring_path and _known_material_path(index, authoring_path):
        return _resolve_path(
            diagnostics,
            index,
            authoring_path,
            match_source=MATCH_AUTHORING_PRIM_PATH,
        )

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

    normalized_name = usd_paths.normalized_blender_object_name(material_name)
    if not normalized_name:
        return _authoring_fail_closed(
            _error(ERROR_MISSING_MATERIAL_NAME, diagnostics),
            authoring_path,
        )
    material_paths = [
        path
        for path in index.get("material_prim_paths", ())
        if usd_paths.normalized_usd_leaf_name(path) == normalized_name
    ]
    return _authoring_fail_closed(
        _resolve_material_paths(
            diagnostics,
            index,
            material_paths,
            match_source=MATCH_HIERARCHY_PATH,
        ),
        authoring_path,
    )


def _known_material_path(index: Mapping[str, Any], path: str) -> bool:
    if path in set(index.get("material_prim_paths", ())):
        return True
    return any(
        candidate.get("material_prim_path") == path or candidate.get("prim_path") == path
        for candidate in index.get("candidates", ())
    )


def _authoring_fail_closed(
    result: UsdPrimResolution[MaterialUsdPrim],
    authoring_path: str,
) -> UsdPrimResolution[MaterialUsdPrim]:
    """Downgrade generic misses to the precise fail-closed reason.

    A material that claims an authored identity the scanned composition does
    not contain (stale reconcile, or a stage the converters did not emit)
    must fail with the authoring-specific reason instead of a generic name
    miss (04-01 precedent).
    """

    if (
        authoring_path
        and result.status is UsdPrimResolutionStatus.ERROR
        and result.error_reason in {ERROR_NO_PATH_MATCH, ERROR_MISSING_MATERIAL_NAME}
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
) -> UsdPrimResolution[MaterialUsdPrim]:
    candidates = [
        candidate
        for candidate in index.get("candidates", ())
        if candidate.get("material_prim_path") == source_path
        or candidate.get("prim_path") == source_path
    ]
    if candidates:
        return _resolve_candidates(diagnostics, candidates, match_source=match_source)
    if source_path in set(index.get("material_prim_paths", ())):
        return _error(
            ERROR_MISSING_PRIM_ATTRIBUTE,
            {**diagnostics, "match_source": match_source, "material_prim_path": source_path},
        )
    return _error(ERROR_NO_PATH_MATCH, {**diagnostics, "match_source": match_source})


def _resolve_material_paths(
    diagnostics: Mapping[str, Any],
    index: Mapping[str, Any],
    material_paths: Iterable[str],
    *,
    match_source: str,
) -> UsdPrimResolution[MaterialUsdPrim]:
    paths = tuple(material_paths)
    if not paths:
        return _error(ERROR_NO_PATH_MATCH, {**diagnostics, "match_source": match_source})
    candidates = [
        candidate
        for candidate in index.get("candidates", ())
        if candidate.get("material_prim_path") in paths
    ]
    if candidates:
        return _resolve_candidates(diagnostics, candidates, match_source=match_source)
    return _error(
        ERROR_MISSING_PRIM_ATTRIBUTE,
        {
            **diagnostics,
            "match_source": match_source,
            "candidate_count": len(paths),
            "candidates": tuple({"material_prim_path": path} for path in paths),
        },
    )


def _resolve_candidates(
    diagnostics: Mapping[str, Any],
    candidates: Iterable[Mapping[str, str]],
    *,
    match_source: str,
) -> UsdPrimResolution[MaterialUsdPrim]:
    candidate_list = tuple(dict(candidate) for candidate in candidates)
    evidence = {
        **diagnostics,
        "match_source": match_source,
        "candidate_count": len(candidate_list),
        "candidates": candidate_list,
    }
    if len(candidate_list) != 1:
        return _error(ERROR_AMBIGUOUS, evidence)
    candidate = candidate_list[0]
    return UsdPrimResolution(
        UsdPrimResolutionStatus.OK,
        MaterialUsdPrim(
            material_prim_path=str(candidate["material_prim_path"]),
            prim_path=str(candidate["prim_path"]),
            usd_attribute=str(candidate["usd_attribute"]),
            connected=bool(candidate.get("connected", False)),
        ),
        diagnostics=evidence,
    )


def _error(
    reason: str,
    diagnostics: Mapping[str, Any],
) -> UsdPrimResolution[MaterialUsdPrim]:
    return UsdPrimResolution(
        UsdPrimResolutionStatus.ERROR,
        error_reason=reason,
        diagnostics=diagnostics,
    )


def _prim_facts(prim: Any, usd_attribute: str, material_type: Any | None) -> dict[str, Any]:
    path = usd_paths.usd_prim_path_from_prim(prim)
    has_attribute = _prim_is_material_value_shader(prim) and _prim_has_attribute(prim, usd_attribute)
    return {
        "path": path,
        "is_material": _prim_is_material(prim, material_type),
        "has_attribute": has_attribute,
        "connected": has_attribute and _prim_attribute_connected(prim, usd_attribute),
    }


def _prim_is_material(prim: Any, material_type: Any | None) -> bool:
    if material_type is not None:
        is_a = getattr(prim, "IsA", None)
        if callable(is_a):
            try:
                return bool(is_a(material_type))
            except Exception:
                pass
    type_name = str(getattr(prim, "type_name", ""))
    if not type_name:
        getter = getattr(prim, "GetTypeName", None)
        if callable(getter):
            try:
                type_name = str(getter())
            except Exception:
                type_name = ""
    return type_name == "Material"


def _prim_has_attribute(prim: Any, name: str) -> bool:
    getter = getattr(prim, "GetAttribute", None)
    if callable(getter):
        try:
            attr = getter(name)
        except Exception:
            attr = None
        if attr is not None:
            is_valid = getattr(attr, "IsValid", None)
            return bool(is_valid()) if callable(is_valid) else True
    return name in getattr(prim, "attributes", ())


def _prim_attribute_connected(prim: Any, name: str) -> bool:
    """Whether the attribute has an authored connection (texture-driven)."""

    getter = getattr(prim, "GetAttribute", None)
    if callable(getter):
        try:
            attr = getter(name)
        except Exception:
            attr = None
        if attr is not None:
            has_connections = getattr(attr, "HasAuthoredConnections", None)
            if callable(has_connections):
                try:
                    return bool(has_connections())
                except Exception:
                    return False
    return name in getattr(prim, "connected_attributes", ())


def _prim_shader_type_name(prim: Any) -> str:
    type_name = str(getattr(prim, "type_name", ""))
    if not type_name:
        getter = getattr(prim, "GetTypeName", None)
        if callable(getter):
            try:
                type_name = str(getter())
            except Exception:
                type_name = ""
    return type_name


def _prim_is_material_value_shader(prim: Any) -> bool:
    """Whether the prim is a shader whose inputs accept live value edits.

    Covers both the authored UsdPreviewSurface shaders and the OpenPBR
    MaterialX surface shaders authored by the material scene layer.
    """

    return _prim_shader_type_name(prim) == "Shader" and _prim_info_id(prim) in {
        "UsdPreviewSurface",
        "ND_open_pbr_surface_surfaceshader",
    }


def _prim_is_preview_surface_shader(prim: Any) -> bool:
    return (
        _prim_shader_type_name(prim) == "Shader"
        and _prim_info_id(prim) == "UsdPreviewSurface"
    )


def _prim_info_id(prim: Any) -> str:
    getter = getattr(prim, "GetAttribute", None)
    if callable(getter):
        try:
            attr = getter("info:id")
        except Exception:
            attr = None
        if attr is not None:
            value_getter = getattr(attr, "Get", None)
            if callable(value_getter):
                try:
                    return str(value_getter())
                except Exception:
                    return ""
    return str(getattr(prim, "info_id", ""))


__all__ = [
    "DEFAULT_BLENDER_MATERIAL_PROPERTY",
    "DEFAULT_USD_MATERIAL_ATTRIBUTE",
    "MaterialUsdPrim",
    "resolve_material_usd_prim",
]
