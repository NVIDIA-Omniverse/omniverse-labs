# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Blender meshes to existing USD UV primvar values."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from . import usd_paths as usd_paths
from .usd_prim_resolution import UsdPrimResolution, UsdPrimResolutionStatus


TARGET_USD_ATTRIBUTE = "primvars:st"
VALUE_TYPE = "Float2Array"
DEFAULT_BLENDER_PROPERTY_PATH = "uv_layers.active"
VALIDATION_KIND = "uv_loop_order"

RESOLVED = "resolved"
ERROR_USD_STAGE_UNAVAILABLE = "usd_stage_unavailable"
ERROR_MISSING_UV_LAYER = "unsupported_uv_layer_missing"
ERROR_PRIM_UNRESOLVED = "unsupported_uv_prim_unresolved"
ERROR_PRIM_AMBIGUOUS = "unsupported_uv_prim_ambiguous"
ERROR_PRIM_INFERRED_ONLY = "unsupported_uv_prim_inferred_only"
ERROR_MISSING_PRIMVAR = "unsupported_uv_primvar_missing"
ERROR_INDEXED_PRIMVAR = "unsupported_uv_indexed_primvar"
ERROR_UNSUPPORTED_INTERPOLATION = "unsupported_uv_interpolation"
ERROR_MALFORMED_PRIMVAR = "unsupported_uv_primvar_malformed"
ERROR_COUNT_MISMATCH = "topology_uv_count_mismatch"
ERROR_LOOP_ORDER_UNPROVEN = "unsupported_uv_loop_order_unproven"

MATCH_SOURCE_USD_PATH = "sourceUsdPath"
MATCH_HIERARCHY_PATH = "hierarchy_path"
MATCH_NAME = "mesh_name"


@dataclass(frozen=True)
class UvUsdPrim:
    prim_path: str
    target_attribute: str
    value_type: str
    interpolation: str
    element_count: int
    source_uv_values: tuple[tuple[float, float], ...]
    source_uv_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_uv_values", _pair_tuple(self.source_uv_values))
        for name in ("prim_path", "target_attribute", "value_type", "interpolation", "source_uv_digest"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"UV USD prim requires {name.replace('_', ' ')}")
        if self.element_count <= 0 or self.element_count != len(self.source_uv_values):
            raise ValueError("UV USD prim element count must match its nonempty USD UV values")
        if self.source_uv_digest != uv_digest(self.source_uv_values):
            raise ValueError("UV USD prim digest must match its USD UV values")


def _uv_prim_index_from_prims(prims: Iterable[Any]) -> dict[str, Any]:
    mesh_facts = tuple(facts for prim in prims if (facts := _mesh_facts(prim))["prim_path"])
    return {
        "available": True,
        "reason": "",
        "target_attribute": TARGET_USD_ATTRIBUTE,
        "prim_paths": tuple(facts["prim_path"] for facts in mesh_facts),
        "candidates": tuple(_candidate_diagnostics(facts) for facts in mesh_facts),
        "mesh_facts": mesh_facts,
    }


def resolve_uv_usd_prim(
    mesh: Any,
    index: Mapping[str, Any],
) -> UsdPrimResolution[UvUsdPrim]:
    object_name = _string_value(getattr(mesh, "name_full", getattr(mesh, "name", "")))
    diagnostics = {
        "object_name": object_name,
        "match_source": "",
        "source_usd_path": usd_paths.source_usd_path_from_blender_id(mesh),
        "candidate_count": 0,
        "candidates": (),
    }
    if not index.get("available", False):
        return _error(
            ERROR_USD_STAGE_UNAVAILABLE,
            {**diagnostics, "stage_reason": str(index.get("reason", ""))},
        )

    source_path = usd_paths.clean_usd_path(diagnostics["source_usd_path"])
    if source_path:
        return _resolve_mesh_facts(
            diagnostics,
            _facts_matching_path(index, source_path),
            match_source=MATCH_SOURCE_USD_PATH,
        )

    hierarchy_path = usd_paths.hierarchy_usd_path(mesh)
    if hierarchy_path:
        matches = _facts_matching_path(index, hierarchy_path)
        if matches:
            return _inferred_error(diagnostics, matches, MATCH_HIERARCHY_PATH)

    normalized_name = usd_paths.normalized_blender_object_name(object_name)
    if not normalized_name:
        return _error(ERROR_PRIM_UNRESOLVED, diagnostics)
    matches = [
        facts
        for facts in index.get("mesh_facts", ())
        if usd_paths.normalized_usd_leaf_name(str(facts.get("prim_path", "")))
        == normalized_name
    ]
    if matches:
        return _inferred_error(diagnostics, matches, MATCH_NAME)
    return _error(ERROR_PRIM_UNRESOLVED, {**diagnostics, "match_source": MATCH_NAME})


def active_uv_snapshot(mesh: Any) -> dict[str, Any]:
    mesh_data = getattr(mesh, "data", None) or mesh
    active_layer = _active_uv_layer(mesh_data)
    if active_layer is None:
        return {
            "status": ERROR_MISSING_UV_LAYER,
            "uv_layer_name": "",
            "element_count": 0,
            "uv_values": (),
            "uv_digest": "",
            "topology_fingerprint": mesh_topology_fingerprint(mesh_data),
        }
    values = _uv_values(active_layer)
    return {
        "status": RESOLVED,
        "uv_layer_name": _string_value(getattr(active_layer, "name", "")),
        "element_count": len(values),
        "uv_values": values,
        "uv_digest": uv_digest(values),
        "topology_fingerprint": mesh_topology_fingerprint(mesh_data),
    }


def validate_loop_order(
    snapshot: Mapping[str, Any],
    prim: UvUsdPrim,
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    if snapshot.get("status") != RESOLVED:
        return _validation_result(snapshot, prim, str(snapshot.get("status", ERROR_MISSING_UV_LAYER)))
    uv_values = _pair_tuple(snapshot.get("uv_values", ()))
    if len(uv_values) != prim.element_count or len(prim.source_uv_values) != len(uv_values):
        return _validation_result(snapshot, prim, ERROR_COUNT_MISMATCH)
    if not _pairs_close(uv_values, prim.source_uv_values, tolerance=tolerance):
        return _validation_result(snapshot, prim, ERROR_LOOP_ORDER_UNPROVEN)
    return _validation_result(snapshot, prim, RESOLVED, tolerance=tolerance)


def cached_loop_order_validation_is_valid(
    validation: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any],
    prim: UvUsdPrim,
) -> bool:
    if not isinstance(validation, Mapping) or validation.get("status") != RESOLVED:
        return False
    return (
        str(validation.get("mesh_prim_path", "")) == prim.prim_path
        and str(validation.get("validation_kind", "")) == VALIDATION_KIND
        and str(validation.get("target_attribute", "")) == prim.target_attribute
        and str(validation.get("value_type", "")) == prim.value_type
        and str(validation.get("interpolation", "")) == prim.interpolation
        and str(validation.get("primvar_shape_status", "")) == RESOLVED
        and bool(validation.get("indexed", True)) is False
        and str(validation.get("uv_layer_name", "")) == str(snapshot.get("uv_layer_name", ""))
        and int(validation.get("element_count", -1)) == int(snapshot.get("element_count", -2))
        and str(validation.get("topology_fingerprint", ""))
        == str(snapshot.get("topology_fingerprint", ""))
        and str(validation.get("source_uv_digest", "")) == prim.source_uv_digest
    )


def validation_record(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": str(validation.get("status", "")),
        "validation_kind": str(validation.get("validation_kind", VALIDATION_KIND)),
        "mesh_prim_path": str(validation.get("mesh_prim_path", "")),
        "target_attribute": str(validation.get("target_attribute", TARGET_USD_ATTRIBUTE)),
        "value_type": str(validation.get("value_type", VALUE_TYPE)),
        "uv_layer_name": str(validation.get("uv_layer_name", "")),
        "interpolation": str(validation.get("interpolation", "")),
        "indexed": bool(validation.get("indexed", False)),
        "primvar_shape_status": str(validation.get("primvar_shape_status", "")),
        "element_count": int(validation.get("element_count", 0) or 0),
        "topology_fingerprint": str(validation.get("topology_fingerprint", "")),
        "blender_uv_digest": str(validation.get("blender_uv_digest", "")),
        "source_uv_digest": str(validation.get("source_uv_digest", "")),
        "tolerance": float(validation.get("tolerance", 0.0) or 0.0),
    }


def mesh_topology_fingerprint(mesh: Any) -> str:
    loops = tuple(_loop_vertex_index(loop) for loop in getattr(mesh, "loops", ()) or ())
    polygons = tuple(_polygon_record(poly) for poly in getattr(mesh, "polygons", ()) or ())
    vertices = len(tuple(getattr(mesh, "vertices", ()) or ()))
    edges = len(tuple(getattr(mesh, "edges", ()) or ()))
    payload = repr((vertices, edges, loops, polygons)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def uv_digest(values: Sequence[Sequence[float]]) -> str:
    payload = repr(
        tuple((round(float(u), 9), round(float(v), 9)) for u, v in values)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_mesh_facts(
    diagnostics: Mapping[str, Any],
    mesh_facts: Iterable[Mapping[str, Any]],
    *,
    match_source: str,
) -> UsdPrimResolution[UvUsdPrim]:
    facts_list = tuple(dict(facts) for facts in mesh_facts)
    evidence = {
        **diagnostics,
        "match_source": match_source,
        "candidate_count": len(facts_list),
        "candidates": tuple(_candidate_diagnostics(facts) for facts in facts_list),
    }
    if not facts_list:
        return _error(ERROR_PRIM_UNRESOLVED, evidence)
    if len(facts_list) > 1:
        return _error(ERROR_PRIM_AMBIGUOUS, evidence)
    facts = facts_list[0]
    if facts["error_reason"]:
        return _error(str(facts["error_reason"]), evidence)
    return UsdPrimResolution(
        UsdPrimResolutionStatus.OK,
        UvUsdPrim(
            prim_path=str(facts["prim_path"]),
            target_attribute=TARGET_USD_ATTRIBUTE,
            value_type=VALUE_TYPE,
            interpolation=str(facts["interpolation"]),
            element_count=int(facts["element_count"]),
            source_uv_values=tuple(facts["source_uv_values"]),
            source_uv_digest=str(facts["source_uv_digest"]),
        ),
        diagnostics=evidence,
    )


def _inferred_error(
    diagnostics: Mapping[str, Any],
    mesh_facts: Iterable[Mapping[str, Any]],
    match_source: str,
) -> UsdPrimResolution[UvUsdPrim]:
    facts_list = tuple(dict(facts) for facts in mesh_facts)
    return _error(
        ERROR_PRIM_INFERRED_ONLY,
        {
            **diagnostics,
            "match_source": match_source,
            "candidate_count": len(facts_list),
            "candidates": tuple(_candidate_diagnostics(facts) for facts in facts_list),
        },
    )


def _error(reason: str, diagnostics: Mapping[str, Any]) -> UsdPrimResolution[UvUsdPrim]:
    return UsdPrimResolution(
        UsdPrimResolutionStatus.ERROR,
        error_reason=reason,
        diagnostics=diagnostics,
    )


def _facts_matching_path(index: Mapping[str, Any], path: str) -> list[Mapping[str, Any]]:
    return [facts for facts in index.get("mesh_facts", ()) if facts.get("prim_path") == path]


def _mesh_facts(prim: Any) -> dict[str, Any]:
    path = usd_paths.usd_prim_path_from_prim(prim)
    if usd_paths.usd_prim_type_name_from_prim(prim) != "Mesh":
        return _empty_mesh_facts("")
    attr = _prim_attribute(prim, TARGET_USD_ATTRIBUTE)
    if attr is None:
        return {**_empty_mesh_facts(path), "error_reason": ERROR_MISSING_PRIMVAR}
    interpolation = _primvars_st_interpolation(prim)
    if _prim_attribute(prim, TARGET_USD_ATTRIBUTE + ":indices") is not None:
        return {
            **_empty_mesh_facts(path),
            "error_reason": ERROR_INDEXED_PRIMVAR,
            "interpolation": interpolation,
        }
    if interpolation != "faceVarying":
        return {
            **_empty_mesh_facts(path),
            "error_reason": ERROR_UNSUPPORTED_INTERPOLATION,
            "interpolation": interpolation,
        }
    values = _primvars_st_values(prim)
    if values is None:
        return {
            **_empty_mesh_facts(path),
            "error_reason": ERROR_MALFORMED_PRIMVAR,
            "interpolation": interpolation,
        }
    return {
        "prim_path": path,
        "error_reason": "",
        "interpolation": interpolation,
        "element_count": len(values),
        "source_uv_values": values,
        "source_uv_digest": uv_digest(values),
    }


def _empty_mesh_facts(path: str) -> dict[str, Any]:
    return {
        "prim_path": path,
        "error_reason": "",
        "interpolation": "",
        "element_count": 0,
        "source_uv_values": (),
        "source_uv_digest": "",
    }


def _candidate_diagnostics(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prim_path": str(facts.get("prim_path", "")),
        "error_reason": str(facts.get("error_reason", "")),
        "interpolation": str(facts.get("interpolation", "")),
        "element_count": int(facts.get("element_count", 0)),
        "source_uv_digest": str(facts.get("source_uv_digest", "")),
    }


def _primvars_st_values(prim: Any) -> tuple[tuple[float, float], ...] | None:
    attr = _prim_attribute(prim, TARGET_USD_ATTRIBUTE)
    if attr is None:
        return None
    getter = getattr(attr, "Get", None)
    raw = getter() if callable(getter) else attr
    try:
        values = _pair_tuple(raw)
    except ValueError:
        return None
    return values or None


def _primvars_st_interpolation(prim: Any) -> str:
    attr = _prim_attribute(prim, TARGET_USD_ATTRIBUTE)
    if attr is None:
        return ""
    getter = getattr(attr, "GetMetadata", None)
    if callable(getter):
        try:
            value = getter("interpolation")
            if value:
                return str(value)
        except Exception:
            pass
    return _string_value(getattr(attr, "interpolation", ""))


def _prim_attribute(prim: Any, name: str) -> Any | None:
    getter = getattr(prim, "GetAttribute", None)
    if callable(getter):
        try:
            attr = getter(name)
            is_valid = getattr(attr, "IsValid", None)
            if callable(is_valid) and not bool(is_valid()):
                return None
            return attr
        except Exception:
            return None
    attrs = getattr(prim, "attributes", None)
    return attrs.get(name) if isinstance(attrs, Mapping) else None


def _active_uv_layer(mesh: Any) -> Any | None:
    uv_layers = getattr(mesh, "uv_layers", None)
    if uv_layers is None:
        return None
    active = getattr(uv_layers, "active", None)
    if active is not None:
        return active
    try:
        return uv_layers[0]
    except (KeyError, IndexError, TypeError):
        return None


def _uv_values(layer: Any) -> tuple[tuple[float, float], ...]:
    return tuple(_uv_pair(item) for item in getattr(layer, "data", ()) or ())


def _uv_pair(item: Any) -> tuple[float, float]:
    raw = getattr(item, "uv", item)
    if isinstance(raw, (str, bytes)):
        raise ValueError("UV loop value must be a 2-value sequence")
    try:
        count = len(raw)
    except TypeError as exc:
        raise ValueError("UV loop value must be a 2-value sequence") from exc
    if count != 2:
        raise ValueError("UV loop value must be a 2-value sequence")
    try:
        return (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ValueError("UV loop value must be a numeric 2-value sequence") from exc


def _pair_tuple(values: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(values, (str, bytes)):
        return ()
    try:
        count = len(values)
    except TypeError:
        return ()
    return tuple(_uv_pair(values[index]) for index in range(count))


def _pairs_close(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(float(a[0]) - float(b[0])) <= tolerance
        and abs(float(a[1]) - float(b[1])) <= tolerance
        for a, b in zip(left, right)
    )


def _validation_result(
    snapshot: Mapping[str, Any],
    prim: UvUsdPrim,
    status: str,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    return {
        "status": status,
        "validation_kind": VALIDATION_KIND,
        "mesh_prim_path": prim.prim_path,
        "target_attribute": prim.target_attribute,
        "value_type": prim.value_type,
        "uv_layer_name": str(snapshot.get("uv_layer_name", "")),
        "interpolation": prim.interpolation,
        "indexed": False,
        "primvar_shape_status": RESOLVED,
        "element_count": int(snapshot.get("element_count", 0) or 0),
        "topology_fingerprint": str(snapshot.get("topology_fingerprint", "")),
        "blender_uv_digest": str(snapshot.get("uv_digest", "")),
        "source_uv_digest": prim.source_uv_digest,
        "tolerance": float(tolerance),
    }


def _loop_vertex_index(loop: Any) -> int:
    try:
        return int(getattr(loop, "vertex_index"))
    except (AttributeError, TypeError, ValueError):
        try:
            return int(loop)
        except (TypeError, ValueError):
            return -1


def _polygon_record(poly: Any) -> tuple[int, int, tuple[int, ...]]:
    loop_start = _int_attr(poly, "loop_start", -1)
    loop_total = _int_attr(poly, "loop_total", -1)
    try:
        vertices = tuple(int(value) for value in getattr(poly, "vertices", ()) or ())
    except (TypeError, ValueError):
        vertices = ()
    return (loop_start, loop_total, vertices)


def _int_attr(value: Any, name: str, default: int) -> int:
    try:
        return int(getattr(value, name))
    except (AttributeError, TypeError, ValueError):
        return default


def _string_value(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "DEFAULT_BLENDER_PROPERTY_PATH",
    "RESOLVED",
    "TARGET_USD_ATTRIBUTE",
    "UvUsdPrim",
    "VALIDATION_KIND",
    "VALUE_TYPE",
    "active_uv_snapshot",
    "cached_loop_order_validation_is_valid",
    "mesh_topology_fingerprint",
    "resolve_uv_usd_prim",
    "uv_digest",
    "validate_loop_order",
    "validation_record",
]
