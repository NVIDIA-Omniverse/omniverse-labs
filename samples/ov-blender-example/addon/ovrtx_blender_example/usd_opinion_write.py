# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete USD opinion writer adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import light_usd_prim
from . import usd_value_edit_support
from .edit_persistence import WriteRequest, WriteResult
from .interactive_edit_planner import EditShape, InteractiveEdit


_SUPPORTED_USD_VALUE_TYPES = {
    "Bool",
    "Color3f",
    "Color4f",
    "Double",
    "Float",
    "Float2",
    "Float3",
    "Float4",
    "Int",
    "Matrix4d",
    "String",
}

_USD_OPINION_ATTRIBUTE_TYPES = usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES
SUPPORTED_TOPOLOGY_KINDS = frozenset({"light_form"})


@dataclass(frozen=True)
class UsdOpinionWriter:
    filepath: str
    usd_layer_id: str
    usd_module: Any | None = None
    sdf_module: Any | None = None
    gf_module: Any | None = None

    def __call__(self, request: WriteRequest) -> WriteResult:
        output_path = Path(self.filepath)
        layer_path = str(output_path)
        resolved_usd_layer_id = (
            request.usd_layer_id or self.usd_layer_id
        )
        validation = _validate_usd_opinion_write_request(
            request,
            writer_target_identifier=self.usd_layer_id,
        )
        if validation:
            return WriteResult(
                requested=False,
                completed=False,
                reason="usd_opinion_write_unsupported_edits",
                path=layer_path,
                usd_layer_id=resolved_usd_layer_id,
                diagnostics=_usd_opinion_write_diagnostics(
                    request,
                    layer_path,
                    rejected_edit_count=len(request.edits),
                    unsupported_edits=validation,
                ),
            )

        try:
            usd, sdf, gf = self._usd_modules()
        except Exception as exc:
            return WriteResult(
                requested=False,
                completed=False,
                reason="pxr_unavailable",
                path=layer_path,
                usd_layer_id=resolved_usd_layer_id,
                diagnostics=_usd_opinion_write_diagnostics(
                    request,
                    layer_path,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stage = usd.Stage.CreateNew(layer_path)
            opinions = []
            for edit in request.edits:
                topology_type = _topology_type_for_edit(edit)
                if topology_type:
                    prim = _define_topology_prim(stage, edit.usd_prim_path, topology_type)
                    topology_attributes = _topology_attribute_values(edit)
                    for attribute_edit in topology_attributes:
                        value_type = _usd_value_type_for_edit(attribute_edit)
                        type_name, value = _usd_attribute_type_and_value(attribute_edit, value_type, sdf, gf)
                        attribute = prim.CreateAttribute(attribute_edit.usd_attribute, type_name, custom=True)
                        attribute.Set(value)
                    opinions.append(
                        {
                            **_target_details(edit),
                            "topology_type": topology_type,
                            "topology_attributes": [
                                _target_details(attribute_edit)
                                for attribute_edit in topology_attributes
                            ],
                        }
                    )
                else:
                    prim = stage.OverridePrim(edit.usd_prim_path)
                    value_type = _usd_value_type_for_edit(edit)
                    type_name, value = _usd_attribute_type_and_value(edit, value_type, sdf, gf)
                    attribute = prim.CreateAttribute(edit.usd_attribute, type_name, custom=True)
                    attribute.Set(value)
                    opinions.append(
                        {
                            **_target_details(edit),
                            "usd_value_type": value_type,
                        }
                    )
            stage.GetRootLayer().Save()
        except Exception as exc:
            return WriteResult(
                requested=True,
                completed=False,
                reason="usd_opinion_write_failed",
                path=layer_path,
                usd_layer_id=resolved_usd_layer_id,
                diagnostics=_usd_opinion_write_diagnostics(
                    request,
                    layer_path,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

        return WriteResult(
            requested=True,
            completed=True,
            reason="usd_opinion_write_completed",
            path=layer_path,
            usd_layer_id=resolved_usd_layer_id,
            diagnostics=_usd_opinion_write_diagnostics(
                request,
                layer_path,
                accepted_edit_count=len(request.edits),
                opinions=opinions,
                usd_opinion_layer_written=True,
            ),
        )

    def _usd_modules(self) -> tuple[Any, Any, Any]:
        if self.usd_module is not None and self.sdf_module is not None and self.gf_module is not None:
            return self.usd_module, self.sdf_module, self.gf_module
        from pxr import Gf, Sdf, Usd  # type: ignore

        return Usd, Sdf, Gf


def _usd_opinion_write_diagnostics(
    request: WriteRequest,
    layer_path: str,
    *,
    accepted_edit_count: int = 0,
    rejected_edit_count: int = 0,
    unsupported_edits: Sequence[Mapping[str, Any]] = (),
    opinions: Sequence[Mapping[str, Any]] = (),
    error: str = "",
    usd_opinion_layer_written: bool = False,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "mutated_source_stage": False,
        "usd_layer_path": layer_path,
        "whole_scene_export_requested": False,
        "edit_count": len(request.edits),
        "accepted_edit_count": int(accepted_edit_count),
        "rejected_edit_count": int(rejected_edit_count),
    }
    if unsupported_edits:
        diagnostics["unsupported_edits"] = [dict(item) for item in unsupported_edits]
    if error:
        diagnostics["error"] = error
    if usd_opinion_layer_written:
        diagnostics["usd_opinion_layer_written"] = True
    if opinions:
        diagnostics["opinions"] = [dict(item) for item in opinions]
    return diagnostics


def _target_details(edit: InteractiveEdit) -> dict[str, Any]:
    target = edit
    return {
        "shape": edit.shape.value,
        "data_authority": edit.data_authority.value,
        "usd_prim_path": target.usd_prim_path,
        "usd_attribute": target.usd_attribute,
        "usd_property_path": target.usd_property_path,
        "usd_layer_id": target.usd_layer_id,
        "blender_property_path": target.blender_property_path,
        "provenance": dict(target.provenance),
    }


def _validate_usd_opinion_write_request(
    request: WriteRequest,
    *,
    writer_target_identifier: str,
) -> list[dict[str, Any]]:
    unsupported: list[dict[str, Any]] = []
    if not request.edits:
        unsupported.append({"reason": "no_edits"})
    usd_layer_ids = sorted(
        {
            edit.usd_layer_id
            for edit in request.edits
            if edit.usd_layer_id
        }
    )
    if len(usd_layer_ids) > 1:
        unsupported.append(
            {
                "reason": "mixed_write_targets",
                "usd_layer_ids": usd_layer_ids,
            }
        )
    if not writer_target_identifier:
        unsupported.append({"reason": "missing_writer_target_identifier"})
    elif (
        request.usd_layer_id
        and request.usd_layer_id != writer_target_identifier
    ):
        unsupported.append(
            {
                "reason": "writer_target_mismatch",
                "request_usd_layer_id": request.usd_layer_id,
                "writer_target_identifier": writer_target_identifier,
            }
        )
    for index, edit in enumerate(request.edits):
        target = edit
        reasons: list[str] = []
        if edit.shape == EditShape.TOPOLOGY:
            if not _topology_type_for_edit(edit):
                reasons.append("topology_edit_requires_writer")
            reasons.extend(_validate_topology_attribute_values(edit))
        if not target.usd_prim_path:
            reasons.append("missing_usd_prim_path")
        if edit.shape != EditShape.TOPOLOGY and not target.usd_attribute:
            reasons.append("missing_usd_attribute")
        if not target.has_persistence_identity():
            reasons.append("missing_write_target_identity")
        if (
            request.usd_layer_id
            and target.usd_layer_id
            and request.usd_layer_id != target.usd_layer_id
        ):
            reasons.append("write_target_mismatch")
        if (
            writer_target_identifier
            and target.usd_layer_id
            and writer_target_identifier != target.usd_layer_id
        ):
            reasons.append("writer_target_mismatch")
        value_type = ""
        if edit.shape != EditShape.TOPOLOGY:
            value_type = _usd_value_type_for_edit(edit)
            if not value_type:
                reasons.append("unsupported_usd_value_type")
            elif value_type not in _SUPPORTED_USD_VALUE_TYPES:
                reasons.append("unknown_usd_value_type")
            else:
                value_error = _validate_usd_opinion_value_shape(edit.value, value_type, target.usd_attribute)
                if value_error:
                    reasons.append(value_error)
        if reasons:
            unsupported.append(
                {
                    "index": index,
                    "shape": edit.shape.value,
                    "data_authority": edit.data_authority.value,
                    "reasons": reasons,
                    "usd_value_type": value_type,
                    "target": _target_details(edit),
                }
            )
    return unsupported


def _topology_type_for_edit(edit: InteractiveEdit) -> str:
    if edit.shape != EditShape.TOPOLOGY:
        return ""
    kinds = frozenset(_topology_change_kinds(edit.metadata))
    if not kinds or not kinds <= SUPPORTED_TOPOLOGY_KINDS:
        return ""
    family = str(edit.metadata.get("current_usd_family", "") or "").strip()
    return family if family in light_usd_prim.USD_LIGHT_FAMILIES else ""


def _topology_attribute_values(edit: InteractiveEdit) -> tuple[InteractiveEdit, ...]:
    attributes = []
    for item in _mapping_sequence(edit.metadata.get("topology_attribute_values", ())):
        name = str(item.get("name", "") or "")
        attributes.append(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=edit.data_authority,
                usd_prim_path=edit.usd_prim_path,
                usd_property_path=f"{edit.usd_prim_path}.{name}" if edit.usd_prim_path and name else "",
                usd_layer_id=edit.usd_layer_id,
                blender_property_path=str(item.get("blender_property_path", "") or ""),
                provenance=dict(edit.provenance),
                value=item.get("value"),
                metadata=dict(item.get("metadata", {}) if isinstance(item.get("metadata", {}), Mapping) else {}),
            )
        )
    return tuple(attributes)


def _validate_topology_attribute_values(edit: InteractiveEdit) -> list[str]:
    reasons: list[str] = []
    attributes = _topology_attribute_values(edit)
    if not attributes:
        reasons.append("topology_edit_requires_attribute_values")
    for attribute_edit in attributes:
        value_type = _usd_value_type_for_edit(attribute_edit)
        if not value_type:
            reasons.append("unsupported_topology_attribute")
        elif value_type not in _SUPPORTED_USD_VALUE_TYPES:
            reasons.append("unknown_topology_attribute_type")
        else:
            value_error = _validate_usd_opinion_value_shape(
                attribute_edit.value,
                value_type,
                attribute_edit.usd_attribute,
            )
            if value_error:
                reasons.append(f"topology_attribute_{value_error}")
    return reasons


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(item for item in value if isinstance(item, Mapping))
    except TypeError:
        return ()


def _topology_change_kinds(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("topology_change_kinds", ())
    if not raw:
        raw = metadata.get("topology_change_kind", ())
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(value) for value in raw)
    except TypeError:
        return ()


def _define_topology_prim(stage: Any, path: str, type_name: str) -> Any:
    define_prim = getattr(stage, "DefinePrim", None)
    if callable(define_prim):
        return define_prim(path, type_name)
    prim = stage.OverridePrim(path)
    setattr(prim, "typeName", type_name)
    return prim


def _usd_value_type_for_edit(edit: InteractiveEdit) -> str:
    return _USD_OPINION_ATTRIBUTE_TYPES.get(edit.usd_attribute, "")


def _usd_attribute_type_and_value(edit: InteractiveEdit, value_type: str, sdf: Any, gf: Any) -> tuple[Any, Any]:
    try:
        type_name = getattr(sdf.ValueTypeNames, value_type)
    except AttributeError as exc:
        raise ValueError(f"unknown USD value type: {value_type!r}") from exc
    return type_name, _coerce_usd_value(edit.value, value_type, gf, edit.usd_attribute)


def _coerce_usd_value(value: Any, value_type: str, gf: Any, attribute: str) -> Any:
    if value_type == "Bool":
        return bool(value)
    if value_type == "Int":
        return int(value)
    if value_type == "Double":
        return float(value)
    if value_type == "Float":
        return float(value)
    if value_type == "String":
        return str(value)
    if value_type == "Float2":
        return gf.Vec2f(*_fixed_float_sequence(value, 2))
    if value_type == "Float3":
        return gf.Vec3f(*_fixed_float_sequence(value, 3))
    if value_type == "Float4":
        return gf.Vec4f(*_fixed_float_sequence(value, 4))
    if value_type == "Color3f":
        return gf.Vec3f(*_color3f_sequence(value, attribute))
    if value_type == "Color4f":
        return gf.Vec4f(*_fixed_float_sequence(value, 4))
    if value_type == "Matrix4d":
        return _matrix4d_value(value, gf)
    raise ValueError(f"unsupported USD value type: {value_type!r}")


def _validate_usd_opinion_value_shape(value: Any, value_type: str, attribute: str) -> str:
    try:
        if value_type == "Bool" and not isinstance(value, bool):
            return "bool_value_required"
        if value_type == "Int" and (not isinstance(value, int) or isinstance(value, bool)):
            return "int_value_required"
        if value_type == "Double" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return "numeric_value_required"
        if value_type == "Float" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return "numeric_value_required"
        if value_type == "String" and not isinstance(value, str):
            return "string_value_required"
        if value_type == "Float2":
            _fixed_float_sequence(value, 2)
        elif value_type == "Float3":
            _fixed_float_sequence(value, 3)
        elif value_type == "Float4":
            _fixed_float_sequence(value, 4)
        elif value_type == "Color3f":
            _color3f_sequence(value, attribute)
        elif value_type == "Color4f":
            _fixed_float_sequence(value, 4)
        elif value_type == "Matrix4d" and not _is_matrix4(value):
            return "matrix4d_value_required"
    except (TypeError, ValueError):
        return "invalid_value_shape"
    return ""


def _plain_sequence(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)):
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _float_sequence(value: list[Any]) -> list[float]:
    return [float(item) for item in value]


def _fixed_float_sequence(value: Any, length: int) -> list[float]:
    sequence = _plain_sequence(value)
    if len(sequence) != length:
        raise ValueError(f"expected {length} values")
    return _float_sequence(sequence)


def _color3f_sequence(value: Any, attribute: str) -> list[float]:
    sequence = _plain_sequence(value)
    if len(sequence) == 4 and attribute == "inputs:diffuseColor":
        sequence = sequence[:3]
    if len(sequence) != 3:
        raise ValueError("expected three color channels")
    return _float_sequence(sequence)


def _is_matrix4(value: Any) -> bool:
    rows = _plain_sequence(value)
    if len(rows) == 16:
        return True
    return len(rows) == 4 and all(len(_plain_sequence(row)) == 4 for row in rows)


def _matrix4d_value(value: Any, gf: Any) -> Any:
    rows = _plain_sequence(value)
    if len(rows) == 16:
        matrix_rows = [_float_sequence(rows[index : index + 4]) for index in range(0, 16, 4)]
    else:
        if len(rows) != 4:
            raise ValueError("matrix4d USD opinion value must have four rows")
        matrix_rows = []
        for row in rows:
            row_values = _plain_sequence(row)
            if len(row_values) != 4:
                raise ValueError("matrix4d USD opinion rows must have four values")
            matrix_rows.append(_float_sequence(row_values))

    matrix = gf.Matrix4d(1.0)
    for index, row in enumerate(matrix_rows):
        matrix.SetRow(index, gf.Vec4d(*row))
    return matrix
