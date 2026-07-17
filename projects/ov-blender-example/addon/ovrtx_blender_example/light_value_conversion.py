# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender light value edit conversion policy."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import usd_value_edit_support
from .value_edit_conversion import (
    BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    FieldClassification,
    STATUS_NON_RENDER,
    STATUS_SUPPORTED,
    STATUS_TOPOLOGY,
    STATUS_UNSUPPORTED,
    UsdAttributeValue,
    classify_mapped_field,
)


MEASURED_LIGHT_SCALE = 120.0 * math.pi * math.pi
PARITY_INTENSITY_SCALE = 120.0 * math.pi
INTENSITY_POLICY_VERSION = "ovrtx_0_3_cycles_parity_2026_06_11"
MIN_SPHERE_RADIUS = 1.0e-3
MIN_EMITTER_AREA = 1.0e-6

LIGHT_TYPE_CHANGED = "light_type_changes_are_topology"
LIGHT_FAMILY_CHANGED = "light_family_changes_are_topology"
LIGHT_FORM_CHANGED = "light_form_changes_are_topology"
MISSING_PREVIOUS_AUTHORED_LIGHT_FORM = "missing_previous_authored_light_form"

AUTHORED_LIGHT_FORM_POINT = "POINT"
AUTHORED_LIGHT_FORM_SPOT = "SPOT"
AUTHORED_LIGHT_FORM_SUN = "SUN"
AUTHORED_LIGHT_FORM_AREA_RECT = "AREA_RECT"
AUTHORED_LIGHT_FORM_AREA_DISK = "AREA_DISK"

SUPPORTED_USD_ATTRIBUTES = usd_value_edit_support.LIGHT_USD_VALUE_TYPES
EDIT_VALUE_ATTRIBUTES_BY_FIELD = MappingProxyType(
    {
        "energy": ("inputs:intensity",),
        "color": ("inputs:color",),
        "use_temperature": ("inputs:color", "inputs:enableColorTemperature"),
        "temperature": ("inputs:color", "inputs:enableColorTemperature"),
        "shadow_soft_size": ("inputs:radius",),
        "spot_size": ("inputs:shaping:cone:angle",),
        "spot_blend": ("inputs:shaping:cone:softness",),
        "angle": ("inputs:angle",),
        "shape": ("inputs:width", "inputs:height", "inputs:radius"),
        "size": ("inputs:width", "inputs:height", "inputs:radius"),
        "size_y": ("inputs:width", "inputs:height", "inputs:radius"),
    }
)
EDIT_VALUE_CONCEPTS = frozenset(
    {
        "light.intensity",
        "light.color",
        "light.size_shape_spot",
    }
)
EXPORT_VALUE_CONCEPTS = EDIT_VALUE_CONCEPTS
EDIT_TOPOLOGY_CONCEPTS = frozenset()
EDIT_TOPOLOGY_KINDS = frozenset({"light_form"})

_UNSUPPORTED_FIELD_REASONS = {
    "specular_factor": "unsupported_light_contribution_factor",
    "diffuse_factor": "unsupported_light_contribution_factor",
    "transmission_factor": "unsupported_light_contribution_factor",
    "volume_factor": "unsupported_light_contribution_factor",
    "use_custom_distance": "unsupported_light_cutoff_distance",
    "cutoff_distance": "unsupported_light_cutoff_distance",
    "use_shadow": "unsupported_light_shadow_toggle",
    "exposure": "unsupported_light_exposure",
    "normalize": "unsupported_light_normalize_policy_fixed",
    "use_nodes": "unsupported_light_nodes",
    "node_tree": "unsupported_light_nodes",
    "shadow_buffer_clip_start": "unsupported_light_shadow_quality",
    "shadow_filter_radius": "unsupported_light_shadow_quality",
    "shadow_maximum_resolution": "unsupported_light_shadow_quality",
    "use_shadow_jitter": "unsupported_light_shadow_quality",
    "shadow_jitter_overblur": "unsupported_light_shadow_quality",
    "use_absolute_resolution": "unsupported_light_shadow_quality",
    "use_soft_falloff": "unsupported_light_falloff_mode",
    "use_square": "unsupported_spot_square_shape",
    "shadow_cascade_max_distance": "unsupported_sun_shadow_cascade",
    "shadow_cascade_count": "unsupported_sun_shadow_cascade",
    "shadow_cascade_exponent": "unsupported_sun_shadow_cascade",
    "shadow_cascade_fade": "unsupported_sun_shadow_cascade",
    "spread": "unsupported_area_spread",
}

_NON_RENDER_FIELD_REASONS = {
    "name": "non_runtime_light_identifier",
    "name_full": "non_runtime_light_identifier",
    **BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    "show_cone": "non_runtime_viewport_field",
}


def exported_light_family(light_type: str, shape: str | None = None) -> str:
    if light_type == "AREA":
        return "DiskLight" if shape in {"DISK", "ELLIPSE"} else "RectLight"
    return {
        "POINT": "SphereLight",
        "SPOT": "SphereLight",
        "SUN": "DistantLight",
    }.get(light_type, "")


def authored_light_form(light_type: str, shape: str | None = None) -> str:
    light_type = str(light_type or "").strip().upper()
    shape = str(shape or "").strip().upper()
    if light_type == "AREA":
        return AUTHORED_LIGHT_FORM_AREA_DISK if shape in {"DISK", "ELLIPSE"} else AUTHORED_LIGHT_FORM_AREA_RECT
    return {
        "POINT": AUTHORED_LIGHT_FORM_POINT,
        "SPOT": AUTHORED_LIGHT_FORM_SPOT,
        "SUN": AUTHORED_LIGHT_FORM_SUN,
    }.get(light_type, "")


def authored_light_form_from_usd_family(usd_family: str) -> str:
    return {
        "RectLight": AUTHORED_LIGHT_FORM_AREA_RECT,
        "DiskLight": AUTHORED_LIGHT_FORM_AREA_DISK,
        "DistantLight": AUTHORED_LIGHT_FORM_SUN,
    }.get(str(usd_family or ""), "")


def classify_field(
    light: Any,
    property_name: str,
    *,
    previous_usd_family: str = "",
) -> FieldClassification:
    light_type = _light_type(light)
    shape = _light_shape(light)
    field = str(property_name or "").strip()

    if field == "type":
        return FieldClassification(STATUS_TOPOLOGY, LIGHT_TYPE_CHANGED)
    if field == "shape":
        if light_type != "AREA":
            return FieldClassification(STATUS_UNSUPPORTED, "non_applicable_area_shape")
        current_family = exported_light_family(light_type, shape)
        if previous_usd_family and previous_usd_family != current_family:
            return FieldClassification(STATUS_TOPOLOGY, LIGHT_FORM_CHANGED)
        return FieldClassification(
            STATUS_SUPPORTED,
            "supported_light_shape_within_usd_family",
            _shape_attribute_names(light),
        )
    classification = classify_mapped_field(
        field,
        non_render=_NON_RENDER_FIELD_REASONS,
        unsupported=_UNSUPPORTED_FIELD_REASONS,
    )
    if classification is not None:
        return classification
    if field == "shadow_soft_size" and light_type not in {"POINT", "SPOT"}:
        return FieldClassification(STATUS_UNSUPPORTED, "non_applicable_light_radius_field")
    if field == "size_y" and not (light_type == "AREA" and shape in {"RECTANGLE", "ELLIPSE"}):
        return FieldClassification(STATUS_UNSUPPORTED, "non_applicable_area_size_y")

    supported = _supported_field_attributes(light)
    if field in supported:
        return FieldClassification(STATUS_SUPPORTED, "supported_light_value", supported[field])
    return FieldClassification(STATUS_UNSUPPORTED, "unsupported_light_field")


def usd_attribute_values(light_object: Any) -> tuple[UsdAttributeValue, ...]:
    light = _light_data(light_object)
    light_type = _light_type(light)
    if light_type not in {"POINT", "SPOT", "SUN", "AREA"}:
        return ()

    area = blender_light_emitter_area(light_object)
    intensity = blender_energy_to_intensity(_float_attr(light, "energy", 0.0), light_type=light_type)
    if light_type != "SUN":
        intensity = intensity / area
        formula = "120*pi*energy/exported_emitter_area"
    else:
        formula = "120*pi*pi*energy"
    attrs = [
        _attribute(
            "inputs:intensity",
            intensity,
            "energy",
            {
                "conversion_policy": INTENSITY_POLICY_VERSION,
                "formula": formula,
                "source_property": "energy",
                "source_units": "Blender light energy",
                "target_units": "OVRTX UsdLux intensity",
                "measured_light_scale": MEASURED_LIGHT_SCALE,
                "parity_intensity_scale": PARITY_INTENSITY_SCALE,
                "emitter_area": area,
                "normalize_policy": "ovrtx_0_3_ignores_inputs_normalize",
            },
        ),
        _attribute(
            "inputs:color",
            blender_light_final_color(light),
            "color",
            {
                "source_property": "color,use_temperature,temperature",
                "color_temperature_policy": "bake_blender_temperature_color",
            },
        ),
        _attribute(
            "inputs:enableColorTemperature",
            False,
            "use_temperature",
            {
                "source_property": "use_temperature,temperature",
                "color_temperature_policy": "disable_ovrtx_temperature_curve",
            },
        ),
        _attribute(
            "inputs:normalize",
            False,
            "normalize",
            {
                "normalize_policy": "ovrtx_0_3_ignores_inputs_normalize",
            },
        ),
    ]

    if light_type == "AREA":
        if _light_shape(light) in {"DISK", "ELLIPSE"}:
            attrs.append(
                _attribute(
                    "inputs:radius",
                    blender_light_disk_radius(light),
                    "size",
                    {"shape": _light_shape(light), "exported_usd_family": "DiskLight"},
                )
            )
        else:
            size = _float_attr(light, "size", 0.0)
            size_y = _float_attr(light, "size_y", size) if _light_shape(light) == "RECTANGLE" else size
            attrs.append(
                _attribute(
                    "inputs:width",
                    size,
                    "size",
                    {"shape": _light_shape(light), "exported_usd_family": "RectLight"},
                )
            )
            attrs.append(
                _attribute(
                    "inputs:height",
                    size_y,
                    "size_y" if _light_shape(light) == "RECTANGLE" else "size",
                    {"shape": _light_shape(light), "exported_usd_family": "RectLight"},
                )
            )
    elif light_type in {"POINT", "SPOT"}:
        attrs.append(
            _attribute(
                "inputs:radius",
                max(_float_attr(light, "shadow_soft_size", 0.0), MIN_SPHERE_RADIUS),
                "shadow_soft_size",
                {"exported_usd_family": "SphereLight"},
            )
        )
        if light_type == "SPOT":
            attrs.append(
                _attribute(
                    "inputs:shaping:cone:angle",
                    math.degrees(_float_attr(light, "spot_size", 0.0)) / 2.0,
                    "spot_size",
                    {"exported_usd_family": "SphereLight"},
                )
            )
            attrs.append(
                _attribute(
                    "inputs:shaping:cone:softness",
                    _float_attr(light, "spot_blend", 0.0),
                    "spot_blend",
                    {"exported_usd_family": "SphereLight"},
                )
            )
    elif light_type == "SUN":
        attrs.append(
            _attribute(
                "inputs:angle",
                math.degrees(_float_attr(light, "angle", 0.0)) / 2.0,
                "angle",
                {"exported_usd_family": "DistantLight"},
            )
        )
    return tuple(attrs)


def _attribute(
    name: str,
    value: Any,
    blender_property_path: str,
    metadata: Mapping[str, Any] | None = None,
) -> UsdAttributeValue:
    return UsdAttributeValue(
        name,
        value,
        SUPPORTED_USD_ATTRIBUTES[name],
        blender_property_path,
        metadata or {},
    )


def blender_energy_to_intensity(watts: float, *, light_type: str = "AREA") -> float:
    scale = PARITY_INTENSITY_SCALE
    if light_type == "SUN":
        scale *= math.pi
    return float(watts) * scale


def blender_light_disk_radius(light: Any) -> float:
    if _light_shape(light) == "ELLIPSE":
        return (_float_attr(light, "size", 0.0) + _float_attr(light, "size_y", 0.0)) / 4.0
    return _float_attr(light, "size", 0.0) / 2.0


def blender_light_emitter_area(light_object: Any) -> float:
    light = _light_data(light_object)
    light_type = _light_type(light)
    sx, sy = _object_scale_xy(light_object)
    if light_type == "AREA":
        shape = _light_shape(light)
        if shape in {"DISK", "ELLIPSE"}:
            radius = blender_light_disk_radius(light)
            return max(math.pi * (radius * sx) * (radius * sy), MIN_EMITTER_AREA)
        size = _float_attr(light, "size", 0.0)
        size_y = _float_attr(light, "size_y", size) if shape == "RECTANGLE" else size
        return max(size * sx * size_y * sy, MIN_EMITTER_AREA)
    if light_type in {"POINT", "SPOT"}:
        radius = max(_float_attr(light, "shadow_soft_size", 0.0), MIN_SPHERE_RADIUS)
        return 4.0 * math.pi * radius * radius
    return 1.0


def blender_light_final_color(light: Any) -> tuple[float, float, float]:
    color = _float3_attr(light, "color", (1.0, 1.0, 1.0))
    if bool(getattr(light, "use_temperature", False)):
        tint = _float3_attr(light, "temperature_color", (1.0, 1.0, 1.0))
        color = tuple(channel * tint[index] for index, channel in enumerate(color))
    return color


def _supported_field_attributes(light: Any) -> dict[str, tuple[str, ...]]:
    light_type = _light_type(light)
    attrs = {
        field: EDIT_VALUE_ATTRIBUTES_BY_FIELD[field]
        for field in ("energy", "color", "use_temperature", "temperature")
    }
    if light_type in {"POINT", "SPOT"}:
        attrs["shadow_soft_size"] = EDIT_VALUE_ATTRIBUTES_BY_FIELD["shadow_soft_size"]
    if light_type == "SPOT":
        attrs["spot_size"] = EDIT_VALUE_ATTRIBUTES_BY_FIELD["spot_size"]
        attrs["spot_blend"] = EDIT_VALUE_ATTRIBUTES_BY_FIELD["spot_blend"]
    if light_type == "SUN":
        attrs["angle"] = EDIT_VALUE_ATTRIBUTES_BY_FIELD["angle"]
    if light_type == "AREA":
        attrs["size"] = _shape_attribute_names(light)
        if _light_shape(light) in {"RECTANGLE", "ELLIPSE"}:
            attrs["size_y"] = _shape_attribute_names(light)
    return attrs


def _shape_attribute_names(light: Any) -> tuple[str, ...]:
    return ("inputs:radius",) if _light_shape(light) in {"DISK", "ELLIPSE"} else (
        "inputs:width",
        "inputs:height",
    )


def _light_data(light_object: Any) -> Any:
    data = getattr(light_object, "data", None)
    if data is not None and _light_type(data):
        return data
    return light_object


def _light_type(light: Any) -> str:
    return str(getattr(light, "type", "") or "").strip().upper()


def _light_shape(light: Any) -> str:
    return str(getattr(light, "shape", "") or "").strip().upper()


def _object_scale_xy(light_object: Any) -> tuple[float, float]:
    scale = getattr(light_object, "scale", None)
    if scale is None:
        return (1.0, 1.0)
    x = getattr(scale, "x", None)
    y = getattr(scale, "y", None)
    if x is not None and y is not None:
        return (abs(float(x)) or 1.0, abs(float(y)) or 1.0)
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)) and len(scale) >= 2:
        return (abs(float(scale[0])) or 1.0, abs(float(scale[1])) or 1.0)
    return (1.0, 1.0)


def _float_attr(value: Any, name: str, default: float) -> float:
    try:
        return float(getattr(value, name))
    except (TypeError, ValueError, AttributeError):
        return float(default)


def _float3_attr(value: Any, name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = getattr(value, name, default)
    try:
        channels = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return default
    if len(channels) < 3:
        return default
    return (channels[0], channels[1], channels[2])
