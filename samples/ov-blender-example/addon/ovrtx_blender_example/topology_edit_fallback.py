# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topology edit fallback policy and compatibility diagnostics for ADR 0009."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import light_value_conversion, material_value_conversion, world_dome_conversion


MESH_TOPOLOGY_CHANGED = "topology_mesh_topology_changed"
MATERIAL_GRAPH_CHANGED = material_value_conversion.MATERIAL_GRAPH_CHANGED
MATERIAL_BINDING_CHANGED = material_value_conversion.MATERIAL_BINDING_CHANGED
LIGHT_TYPE_CHANGED = light_value_conversion.LIGHT_TYPE_CHANGED
LIGHT_FAMILY_CHANGED = light_value_conversion.LIGHT_FAMILY_CHANGED
LIGHT_FORM_CHANGED = light_value_conversion.LIGHT_FORM_CHANGED
PRIM_CREATE_DELETE = "topology_prim_create_delete"
COLLIDER_TOPOLOGY_CHANGED = "topology_collider_topology_changed"
UV_COUNT_MISMATCH = "topology_uv_count_mismatch"
ENVIRONMENT_TEXTURE_CHANGED = world_dome_conversion.ENVIRONMENT_TEXTURE_CHANGED
WORLD_NODE_GRAPH_CHANGED = world_dome_conversion.WORLD_NODE_GRAPH_CHANGED
WORLD_ASSIGNMENT_CHANGED = world_dome_conversion.WORLD_ASSIGNMENT_CHANGED
SCENE_TOPOLOGY_CHANGED = "topology_scene_topology_changed"

TOPOLOGY_REASON_BY_CHANGE = {
    "mesh_topology": MESH_TOPOLOGY_CHANGED,
    "material_graph": MATERIAL_GRAPH_CHANGED,
    "material_binding": MATERIAL_BINDING_CHANGED,
    "light_type": LIGHT_TYPE_CHANGED,
    "light_form": LIGHT_FORM_CHANGED,
    "light_family": LIGHT_FAMILY_CHANGED,
    "prim_create_delete": PRIM_CREATE_DELETE,
    "collider_topology": COLLIDER_TOPOLOGY_CHANGED,
    "uv_count_mismatch": UV_COUNT_MISMATCH,
    "environment_texture": ENVIRONMENT_TEXTURE_CHANGED,
    "world_node_graph": WORLD_NODE_GRAPH_CHANGED,
    "world_assignment": WORLD_ASSIGNMENT_CHANGED,
}

DEFAULT_TOPOLOGY_REASON_BY_DEFAULT = {
    "collider_topology": COLLIDER_TOPOLOGY_CHANGED,
    "collider_structure": COLLIDER_TOPOLOGY_CHANGED,
    "material_topology": MATERIAL_GRAPH_CHANGED,
    "material_structure": MATERIAL_GRAPH_CHANGED,
    "scene_topology": SCENE_TOPOLOGY_CHANGED,
    "scene_structure": SCENE_TOPOLOGY_CHANGED,
}

_REASON_ORDER = tuple(dict.fromkeys((
    MESH_TOPOLOGY_CHANGED,
    MATERIAL_GRAPH_CHANGED,
    MATERIAL_BINDING_CHANGED,
    LIGHT_TYPE_CHANGED,
    LIGHT_FORM_CHANGED,
    LIGHT_FAMILY_CHANGED,
    PRIM_CREATE_DELETE,
    COLLIDER_TOPOLOGY_CHANGED,
    UV_COUNT_MISMATCH,
    ENVIRONMENT_TEXTURE_CHANGED,
    WORLD_NODE_GRAPH_CHANGED,
    WORLD_ASSIGNMENT_CHANGED,
    SCENE_TOPOLOGY_CHANGED,
)))


def topology_reasons_for_edit(
    default_topology: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    metadata = metadata or {}
    reasons = _metadata_reasons(metadata)
    if reasons:
        return coalesce_topology_reasons(reasons)
    change_reasons = _metadata_change_reasons(metadata)
    if change_reasons:
        return coalesce_topology_reasons(change_reasons)
    reason = DEFAULT_TOPOLOGY_REASON_BY_DEFAULT.get(str(default_topology), "")
    return (reason,) if reason else ()


def coalesce_topology_reasons(reasons: Iterable[Any]) -> tuple[str, ...]:
    unique = {str(reason).strip() for reason in reasons if str(reason).strip()}
    ordered = [reason for reason in _REASON_ORDER if reason in unique]
    ordered.extend(sorted(unique.difference(ordered)))
    return tuple(ordered)


def topology_rekey_diagnostics(
    *,
    reasons: Iterable[Any],
    old_composition: Any | None = None,
    new_composition: Any | None = None,
    requested_write_path: str = "",
    session_rekey_status: str = "requested",
    write_requested: bool = False,
) -> dict[str, Any]:
    old_identity = _composition_identity(old_composition)
    new_identity = _composition_identity(new_composition)
    identity_changed = bool(
        old_identity
        and new_identity
        and old_identity.get("composition_digest") != new_identity.get("composition_digest")
    )
    rekey_requested = bool(new_identity) or session_rekey_status in {"requested", "rekeyed"}
    return {
        "topology_reasons": list(coalesce_topology_reasons(reasons)),
        "old_composition_identity": old_identity,
        "new_composition_identity": new_identity,
        "composition_identity_status": _composition_identity_status(old_identity, new_identity),
        "requested_write_path": str(requested_write_path),
        "composition_identity_changed": identity_changed,
        "write_requested": bool(write_requested),
        "session_rekey_status": str(session_rekey_status or ""),
        "session_rekey_requested": rekey_requested,
        "refinement_reset": bool(identity_changed or rekey_requested),
    }


def _metadata_reasons(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("topology_reasons", ())
    if not raw:
        raw = metadata.get("topology_reason", ())
    if not raw:
        raw = metadata.get("topology_reasons", ())
    if not raw:
        raw = metadata.get("topology_reason", ())
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(reason) for reason in raw)
    except TypeError:
        return ()


def _metadata_change_reasons(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("topology_change_kinds", ())
    if not raw:
        raw = metadata.get("topology_change_kind", ())
    if not raw:
        raw = metadata.get("topology_change_kinds", ())
    if not raw:
        raw = metadata.get("topology_change_kind", ())
    if isinstance(raw, str):
        raw_values = (raw,)
    else:
        try:
            raw_values = tuple(raw)
        except TypeError:
            raw_values = ()
    return tuple(
        reason
        for change_kind in raw_values
        if (reason := TOPOLOGY_REASON_BY_CHANGE.get(_change_key(str(change_kind))))
    )


def _composition_identity(composition: Any | None) -> dict[str, Any]:
    if composition is None:
        return {}
    layers = tuple(getattr(composition, "presentation_layers", ()) or ())
    return {
        "source_scene_path": str(getattr(composition, "source_scene_path", "") or ""),
        "composed_scene_path": str(getattr(composition, "composed_scene_path", "") or ""),
        "composition_digest": str(getattr(composition, "digest", "") or ""),
        "pass_through": bool(getattr(composition, "pass_through", False)),
        "presentation_layer_count": len(layers),
        "presentation_sources": [
            str(layer.get("source", ""))
            for layer in layers
            if isinstance(layer, Mapping)
        ],
    }


def _composition_identity_status(old_identity: Mapping[str, Any], new_identity: Mapping[str, Any]) -> str:
    if old_identity and new_identity:
        return "old_and_new_recorded"
    if old_identity:
        return "old_recorded"
    if new_identity:
        return "new_recorded"
    return "not_recorded"


def _change_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(".", "_")
