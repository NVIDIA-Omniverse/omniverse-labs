# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD stack-based write target resolution for exact edit targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


TARGET_KIND_PRIM = "prim"
TARGET_KIND_ATTRIBUTE = "attribute"
TARGET_KIND_RELATIONSHIP = "relationship"

REASON_WRITE_TARGET_MISMATCH = "write_target_mismatch"
REASON_MISSING_WRITE_TARGET = "missing_write_target"
REASON_SESSION_LAYER_ONLY = "session_layer_only"
REASON_PXR_UNAVAILABLE = "pxr_unavailable"
REASON_STAGE_OPEN_FAILED = "stage_open_failed"
REASON_MISSING_TARGET = "missing_target"
REASON_MISSING_PROPERTY = "missing_property"
REASON_TARGET_STACK_UNAVAILABLE = "target_stack_unavailable"
REASON_UNSUPPORTED_TARGET_KIND = "unsupported_target_kind"
REASON_UNSUPPORTED_TOPOLOGY_FIELDS = "unsupported_topology_fields"

_SUPPORTED_TARGET_KINDS = {
    TARGET_KIND_PRIM,
    TARGET_KIND_ATTRIBUTE,
    TARGET_KIND_RELATIONSHIP,
}

_UNSUPPORTED_PRIM_TOPOLOGY_FIELDS = {
    "references": ("referenceList", "references"),
    "payloads": ("payloadList", "payloads"),
    "variants": ("variantSets", "variantSetNameList", "variantSelections", "variantSetNames"),
    "inherits": ("inheritPathList", "inherits"),
    "specializes": ("specializesList", "specializes"),
}


class WriteTargetResolutionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class WriteTargetResolutionResult:
    status: WriteTargetResolutionStatus
    usd_layer_id: str | None = None
    error_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier = self.usd_layer_id
        if self.status is WriteTargetResolutionStatus.OK:
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("successful write-target resolution requires an identifier")
            if self.error_reason is not None:
                raise ValueError("successful write-target resolution cannot have an error reason")
        elif self.status is WriteTargetResolutionStatus.ERROR:
            if identifier is not None:
                raise ValueError("failed write-target resolution cannot have an identifier")
            if not isinstance(self.error_reason, str) or not self.error_reason.strip():
                raise ValueError("failed write-target resolution requires an error reason")
        else:
            raise ValueError(f"invalid write-target resolution status: {self.status}")


def resolve_write_target(
    input_usd_path: str,
    *,
    usd_prim_path: str,
    target_kind: str,
    usd_property_name: str = "",
    explicit_usd_layer_id: str = "",
    ignored_layer_identifiers: Iterable[str] = (),
    usd_module: Any | None = None,
) -> WriteTargetResolutionResult:
    """Resolve durable layer ownership from the exact USD target stack."""

    normalized_target_kind = str(target_kind or "").strip()
    ignored_identifiers = tuple(str(identifier) for identifier in ignored_layer_identifiers if str(identifier))
    explicit_identifier = str(explicit_usd_layer_id or "").strip()
    diagnostics = {
        "input_usd_path": str(input_usd_path or ""),
        "target_kind": normalized_target_kind,
        "usd_prim_path": str(usd_prim_path or ""),
        "usd_property_name": str(usd_property_name or ""),
        "usd_property_path": _usd_property_path(usd_prim_path, usd_property_name),
        "explicit_usd_layer_id": explicit_identifier,
        "ignored_layer_identifiers": list(ignored_identifiers),
        "stack_resolved_identifier": "",
        "winning_spec": {},
        "candidate_specs": [],
    }
    if normalized_target_kind not in _SUPPORTED_TARGET_KINDS:
        return _error(diagnostics, REASON_UNSUPPORTED_TARGET_KIND)

    usd, reason = _load_usd_module(usd_module)
    if usd is None:
        return _error(diagnostics, REASON_PXR_UNAVAILABLE, detail=reason)

    stage = _open_stage(usd, input_usd_path)
    if stage is None:
        return _error(diagnostics, REASON_STAGE_OPEN_FAILED)

    prim = _stage_prim_at_path(stage, usd_prim_path)
    if not _usd_object_is_valid(prim):
        return _error(diagnostics, REASON_MISSING_TARGET)

    stack_result = _target_stack(prim, normalized_target_kind, usd_property_name)
    error_reason = str(stack_result.get("error_reason", ""))
    if error_reason:
        return _error(
            diagnostics,
            error_reason,
            detail=str(stack_result.get("detail", "")),
        )

    scan = _scan_stack(
        stack_result.get("stack", ()),
        target_kind=normalized_target_kind,
        ignored_layer_identifiers=frozenset(ignored_identifiers),
    )
    candidates = list(scan["candidate_specs"])
    resolved_identifier = str(scan["usd_layer_id"])
    winning_spec = dict(scan["winning_spec"])
    diagnostics = {
        **diagnostics,
        "stack_resolved_identifier": resolved_identifier,
        "winning_spec": winning_spec,
        "candidate_specs": candidates,
    }
    error_reason = str(scan["error_reason"])
    if not resolved_identifier:
        return _error(diagnostics, error_reason)
    if explicit_identifier and explicit_identifier != resolved_identifier:
        return _error(diagnostics, REASON_WRITE_TARGET_MISMATCH)
    return _ok(diagnostics, resolved_identifier)


def _load_usd_module(usd_module: Any | None) -> tuple[Any | None, str]:
    if usd_module is not None:
        return usd_module, ""
    try:
        from pxr import Usd  # type: ignore
    except Exception as exc:
        return None, type(exc).__name__
    return Usd, ""


def _open_stage(usd: Any, input_usd_path: str) -> Any | None:
    try:
        opener = getattr(getattr(usd, "Stage", None), "Open", None)
        if not callable(opener):
            return None
        return opener(str(input_usd_path or ""))
    except Exception:
        return None


def _stage_prim_at_path(stage: Any, usd_prim_path: str) -> Any | None:
    getter = getattr(stage, "GetPrimAtPath", None)
    if not callable(getter):
        return None
    try:
        return getter(str(usd_prim_path or ""))
    except Exception:
        return None


def _target_stack(prim: Any, target_kind: str, usd_property_name: str) -> dict[str, Any]:
    if target_kind == TARGET_KIND_PRIM:
        getter = getattr(prim, "GetPrimStack", None)
        if not callable(getter):
            return {
                "error_reason": REASON_TARGET_STACK_UNAVAILABLE,
                "detail": "missing_GetPrimStack",
            }
        try:
            return {"stack": tuple(getter())}
        except Exception as exc:
            return {
                "error_reason": REASON_TARGET_STACK_UNAVAILABLE,
                "detail": type(exc).__name__,
            }

    property_name = str(usd_property_name or "").strip()
    if not property_name:
        return {
            "error_reason": REASON_MISSING_PROPERTY,
            "detail": "missing_usd_property_name",
        }
    getter_name = "GetAttribute" if target_kind == TARGET_KIND_ATTRIBUTE else "GetRelationship"
    property_getter = getattr(prim, getter_name, None)
    if not callable(property_getter):
        return {
            "error_reason": REASON_MISSING_PROPERTY,
            "detail": f"missing_{getter_name}",
        }
    try:
        prop = property_getter(property_name)
    except Exception:
        prop = None
    if not _usd_object_is_valid(prop):
        return {"error_reason": REASON_MISSING_PROPERTY, "detail": "property_not_found"}
    stack_getter = getattr(prop, "GetPropertyStack", None)
    if not callable(stack_getter):
        return {
            "error_reason": REASON_TARGET_STACK_UNAVAILABLE,
            "detail": "missing_GetPropertyStack",
        }
    try:
        return {"stack": tuple(stack_getter())}
    except Exception as exc:
        return {
            "error_reason": REASON_TARGET_STACK_UNAVAILABLE,
            "detail": type(exc).__name__,
        }


def _scan_stack(
    stack: Iterable[Any],
    *,
    target_kind: str,
    ignored_layer_identifiers: frozenset[str],
) -> dict[str, Any]:
    candidate_specs: list[dict[str, Any]] = []
    ignored_durable_count = 0
    anonymous_count = 0
    unsupported_fields: tuple[str, ...] = ()
    for index, spec in enumerate(stack):
        record = _spec_record(spec, index=index, ignored_layer_identifiers=ignored_layer_identifiers)
        candidate_specs.append(record)
        layer_status = str(record["layer_status"])
        if layer_status == "ignored_layer":
            ignored_durable_count += 1
            continue
        if layer_status == "anonymous_layer":
            anonymous_count += 1
            continue
        if layer_status != "durable":
            continue
        if target_kind == TARGET_KIND_PRIM:
            unsupported_fields = tuple(record["unsupported_topology_fields"])
            if unsupported_fields:
                return {
                    "error_reason": REASON_UNSUPPORTED_TOPOLOGY_FIELDS,
                    "usd_layer_id": "",
                    "winning_spec": {},
                    "candidate_specs": candidate_specs,
                }
            if not _prim_spec_is_definition(spec):
                record["candidate_status"] = "not_prim_definition"
                continue
        record["candidate_status"] = "selected"
        record["selected"] = True
        return {
            "error_reason": "",
            "usd_layer_id": str(record["layer_identifier"]),
            "winning_spec": record,
            "candidate_specs": candidate_specs,
        }
    if ignored_durable_count and ignored_durable_count + anonymous_count == len(candidate_specs):
        error_reason = REASON_SESSION_LAYER_ONLY
    else:
        error_reason = REASON_MISSING_WRITE_TARGET
    return {
        "error_reason": error_reason,
        "usd_layer_id": "",
        "winning_spec": {},
        "candidate_specs": candidate_specs,
    }


def _spec_record(
    spec: Any,
    *,
    index: int,
    ignored_layer_identifiers: frozenset[str],
) -> dict[str, Any]:
    layer = _spec_layer(spec)
    layer_identifier = _layer_identifier(layer)
    ignored = bool(layer_identifier and layer_identifier in ignored_layer_identifiers)
    anonymous = _layer_is_anonymous(layer, layer_identifier)
    if ignored:
        layer_status = "ignored_layer"
    elif anonymous:
        layer_status = "anonymous_layer"
    elif layer_identifier:
        layer_status = "durable"
    else:
        layer_status = "missing_layer_identifier"
    unsupported_fields = _unsupported_prim_topology_fields(spec)
    return {
        "index": index,
        "specifier": _spec_specifier(spec),
        "type_name": _spec_type_name(spec),
        "spec_path": _spec_path(spec),
        "property_name": _spec_name(spec),
        "layer_identifier": layer_identifier,
        "layer_real_path": _layer_real_path(layer),
        "layer_resolved_path": _layer_resolved_path(layer),
        "layer_display_name": _layer_display_name(layer),
        "layer_anonymous": anonymous,
        "layer_ignored": ignored,
        "layer_status": layer_status,
        "unsupported_topology_fields": list(unsupported_fields),
        "candidate_status": layer_status,
        "selected": False,
    }


def _ok(
    diagnostics: Mapping[str, Any],
    usd_layer_id: str,
) -> WriteTargetResolutionResult:
    return WriteTargetResolutionResult(
        status=WriteTargetResolutionStatus.OK,
        usd_layer_id=usd_layer_id,
        diagnostics=dict(diagnostics),
    )


def _error(
    diagnostics: Mapping[str, Any],
    error_reason: str,
    *,
    detail: str = "",
) -> WriteTargetResolutionResult:
    result_diagnostics = dict(diagnostics)
    if detail:
        result_diagnostics["failure_detail"] = detail
    return WriteTargetResolutionResult(
        status=WriteTargetResolutionStatus.ERROR,
        error_reason=error_reason,
        diagnostics=result_diagnostics,
    )


def _usd_object_is_valid(value: Any) -> bool:
    if value is None:
        return False
    is_valid = getattr(value, "IsValid", None)
    if callable(is_valid):
        try:
            return bool(is_valid())
        except Exception:
            return False
    return True


def _usd_property_path(usd_prim_path: Any, usd_property_name: Any) -> str:
    prim_path = str(usd_prim_path or "")
    property_name = str(usd_property_name or "")
    return f"{prim_path}.{property_name}" if prim_path and property_name else ""


def _spec_layer(spec: Any) -> Any | None:
    for name in ("layer", "Layer"):
        try:
            layer = getattr(spec, name)
        except Exception:
            layer = None
        if layer is not None:
            return layer
    getter = getattr(spec, "GetLayer", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def _layer_identifier(layer: Any | None) -> str:
    return _string_attr(layer, "identifier")


def _layer_real_path(layer: Any | None) -> str:
    return _string_attr(layer, "realPath")


def _layer_resolved_path(layer: Any | None) -> str:
    value = _string_attr(layer, "resolvedPath")
    if value:
        return value
    getter = getattr(layer, "GetResolvedPath", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return _string_attr(layer, "resolved_path")


def _layer_display_name(layer: Any | None) -> str:
    getter = getattr(layer, "GetDisplayName", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return _string_attr(layer, "displayName") or _string_attr(layer, "display_name")


def _layer_is_anonymous(layer: Any | None, identifier: str) -> bool:
    for name in ("anonymous", "isAnonymous", "is_anonymous"):
        try:
            value = getattr(layer, name)
        except Exception:
            value = None
        if callable(value):
            try:
                if bool(value()):
                    return True
            except Exception:
                continue
        elif value:
            return True
    return str(identifier).startswith("anon:")


def _spec_specifier(spec: Any) -> str:
    return str(_spec_value(spec, "specifier") or "")


def _spec_type_name(spec: Any) -> str:
    return str(_spec_value(spec, "typeName") or _spec_value(spec, "type_name") or "")


def _spec_path(spec: Any) -> str:
    value = _spec_value(spec, "path")
    if value:
        return str(value)
    getter = getattr(spec, "GetPath", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return ""


def _spec_name(spec: Any) -> str:
    value = _spec_value(spec, "name")
    if value:
        return str(value)
    getter = getattr(spec, "GetName", None)
    if callable(getter):
        try:
            return str(getter() or "")
        except Exception:
            return ""
    return ""


def _spec_value(spec: Any, name: str) -> Any:
    try:
        value = getattr(spec, name)
    except Exception:
        value = None
    if value is not None:
        return value
    getter = getattr(spec, "GetInfo", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            return None
    return None


def _prim_spec_is_definition(spec: Any) -> bool:
    specifier = _spec_specifier(spec).lower()
    if specifier in {"def", "specifierdef"} or specifier.endswith(".specifierdef"):
        return True
    return bool(_spec_type_name(spec))


def _unsupported_prim_topology_fields(spec: Any) -> tuple[str, ...]:
    fields: list[str] = []
    for field_name, aliases in _UNSUPPORTED_PRIM_TOPOLOGY_FIELDS.items():
        if any(_has_authored_value(_spec_value(spec, alias)) for alias in aliases):
            fields.append(field_name)
    return tuple(fields)


def _has_authored_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, Mapping):
        return bool(value)
    for name in (
        "explicitItems",
        "prependedItems",
        "appendedItems",
        "addedItems",
        "deletedItems",
        "orderedItems",
    ):
        try:
            items = getattr(value, name)
        except Exception:
            items = None
        if items:
            return True
    try:
        return len(value) > 0
    except Exception:
        return False


def _string_attr(value: Any, name: str) -> str:
    try:
        return str(getattr(value, name) or "")
    except Exception:
        return ""


__all__ = [
    "WriteTargetResolutionResult",
    "WriteTargetResolutionStatus",
    "REASON_MISSING_PROPERTY",
    "REASON_MISSING_WRITE_TARGET",
    "REASON_MISSING_TARGET",
    "REASON_WRITE_TARGET_MISMATCH",
    "REASON_PXR_UNAVAILABLE",
    "REASON_SESSION_LAYER_ONLY",
    "REASON_STAGE_OPEN_FAILED",
    "REASON_TARGET_STACK_UNAVAILABLE",
    "REASON_UNSUPPORTED_TARGET_KIND",
    "REASON_UNSUPPORTED_TOPOLOGY_FIELDS",
    "TARGET_KIND_ATTRIBUTE",
    "TARGET_KIND_PRIM",
    "TARGET_KIND_RELATIONSHIP",
    "resolve_write_target",
]
