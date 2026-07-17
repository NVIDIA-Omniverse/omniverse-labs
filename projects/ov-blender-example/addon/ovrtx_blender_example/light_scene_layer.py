# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-scene light presentation layer for OVRTX parity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from . import light_value_conversion
from . import ovrtx_scene_composition
from .render_requests import MaterialPresentationLayer
from .value_edit_conversion import UsdAttributeValue


_SOURCE = "light_value_policy"
_USD_LIGHT_TYPES = {"RectLight", "DiskLight", "SphereLight", "DistantLight"}


@dataclass(frozen=True)
class _LightRecord:
    blender_object_name: str
    usd_path: str
    usd_family: str
    attributes: tuple[UsdAttributeValue, ...]


def scene_layer_from_lights(
    light_objects: Iterable[Any],
    input_usd_path: str,
) -> MaterialPresentationLayer | None:
    """Author initial light values with the same policy used for live edits."""

    index = _stock_light_index(input_usd_path)
    if not index:
        return None
    records: list[_LightRecord] = []
    skipped: list[dict[str, str]] = []
    for light_object in light_objects:
        if str(getattr(light_object, "type", "")) != "LIGHT":
            continue
        name = _blender_object_name(light_object)
        light_data = getattr(light_object, "data", light_object)
        expected_family = light_value_conversion.exported_light_family(
            str(getattr(light_data, "type", "")),
            str(getattr(light_data, "shape", "")),
        )
        exported = index.get(name)
        if exported is None:
            skipped.append({"name": name, "reason": "missing_stock_light_prim"})
            continue
        if expected_family and exported["usd_family"] != expected_family:
            skipped.append(
                {
                    "name": name,
                    "reason": "stock_light_family_mismatch",
                    "expected": expected_family,
                    "actual": exported["usd_family"],
                }
            )
            continue
        attributes = light_value_conversion.usd_attribute_values(light_object)
        if attributes:
            records.append(
                _LightRecord(
                    name,
                    exported["path"],
                    exported["usd_family"],
                    attributes,
                )
            )
    return _layer_from_records(records, skipped)


def _stock_light_index(input_usd_path: str) -> dict[str, dict[str, str]]:
    if not input_usd_path:
        return {}
    try:
        from pxr import Usd  # type: ignore
    except ModuleNotFoundError:
        return {}
    stage = Usd.Stage.Open(str(input_usd_path))
    if stage is None:
        return {}
    index: dict[str, dict[str, str]] = {}
    for prim in stage.Traverse():
        usd_family = str(prim.GetTypeName())
        if usd_family not in _USD_LIGHT_TYPES:
            continue
        name = _prim_attribute_value(
            prim.GetParent(), "userProperties:blender:object_name"
        )
        if not name or name in index:
            continue
        index[name] = {"path": str(prim.GetPath()), "usd_family": usd_family}
    return index


def _layer_from_records(
    records: Iterable[_LightRecord],
    skipped: Iterable[Mapping[str, str]] = (),
) -> MaterialPresentationLayer | None:
    selected = tuple(records)
    if not selected:
        return None
    definitions = [
        (
            record.usd_path,
            record.usd_family,
            [_attribute_line(attribute) for attribute in record.attributes],
        )
        for record in selected
    ]
    layer_body = "\n".join(ovrtx_scene_composition._usda_typed_def_tree(definitions)).rstrip()
    authored_properties = tuple(
        (record.usd_path, attribute.name)
        for record in selected
        for attribute in record.attributes
    )
    digest_records = [
        {
            "blender_object_name": record.blender_object_name,
            "usd_path": record.usd_path,
            "usd_family": record.usd_family,
            "attributes": [
                {
                    "name": attribute.name,
                    "value": _json_value(attribute.value),
                    "value_type": attribute.value_type,
                }
                for attribute in record.attributes
            ],
        }
        for record in selected
    ]
    return MaterialPresentationLayer(
        target_path=min(record.usd_path for record in selected),
        layer_body=layer_body,
        authored_properties=authored_properties,
        digest_content={"source": _SOURCE, "records": digest_records},
        diagnostics={
            "source": _SOURCE,
            "status": "generated",
            "light_count": len(selected),
            "skipped_lights": [dict(item) for item in skipped],
        },
    )


def _attribute_line(attribute: UsdAttributeValue) -> str:
    value_type = str(attribute.value_type)
    if value_type == "Float":
        return f"float {attribute.name} = {_float(attribute.value)}"
    if value_type == "Bool":
        return f"bool {attribute.name} = {_bool(attribute.value)}"
    if value_type == "Color3f":
        color = tuple(attribute.value)
        return (
            f"color3f {attribute.name} = "
            f"({_float(color[0])}, {_float(color[1])}, {_float(color[2])})"
        )
    raise ValueError(f"unsupported light attribute type: {value_type}")


def _prim_attribute_value(prim: Any, name: str) -> str:
    if prim is None:
        return ""
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.IsValid():
        return ""
    value = attribute.Get()
    return str(value) if value is not None else ""


def _blender_object_name(value: Any) -> str:
    return str(getattr(value, "name_full", None) or getattr(value, "name", "") or "")


def _float(value: Any) -> str:
    return f"{float(value):.9g}"


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
