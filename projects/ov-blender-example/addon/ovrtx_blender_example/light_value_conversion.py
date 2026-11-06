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
INTENSITY_POLICY_VERSION = "cycles_parity_sun_irradiance_2026_10_14"
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
        "exposure": ("inputs:exposure",),
        "normalize": ("inputs:intensity",),
        "diffuse_factor": ("inputs:diffuse",),
        "specular_factor": ("inputs:specular",),
        "color": ("inputs:color",),
        "use_temperature": ("inputs:color", "inputs:enableColorTemperature"),
        "temperature": ("inputs:color", "inputs:enableColorTemperature"),
        "shadow_soft_size": ("inputs:radius",),
        "spot_size": ("inputs:shaping:cone:angle", "inputs:shaping:cone:softness"),
        "spot_blend": ("inputs:shaping:cone:angle", "inputs:shaping:cone:softness"),
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
    "transmission_factor": "unsupported_light_contribution_factor",
    "volume_factor": "unsupported_light_contribution_factor",
    "use_custom_distance": "unsupported_light_cutoff_distance",
    "cutoff_distance": "unsupported_light_cutoff_distance",
    "use_shadow": "unsupported_light_shadow_toggle",
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


def unsupported_field_values(light_object: Any) -> dict[str, Any]:
    """Bounded scalar snapshot for actionable changed-field diagnostics."""
    light = _light_data(light_object)
    return {field: value for field in _UNSUPPORTED_FIELD_REASONS
            if isinstance(value := getattr(light, field, None), (bool, int, float, str))}


def unsupported_field_changes(light_object: Any, previous: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple({"field": field, "value": value, "previous_value": previous[field],
                  "reason": _UNSUPPORTED_FIELD_REASONS[field]}
                 for field, value in unsupported_field_values(light_object).items()
                 if field in previous and value != previous[field])


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
    node_factor = light_node_tree_factor(light)
    intensity = blender_energy_to_intensity(_float_attr(light, "energy", 0.0), light_type=light_type)
    intensity *= node_factor.strength
    normalize = bool(getattr(light, "normalize", True))
    if light_type != "SUN" and normalize:
        intensity = intensity / area
        formula = "120*pi*energy/exported_emitter_area"
    elif light_type != "SUN":
        formula = "120*pi*energy"
    else:
        # Measured 2026-10-14 (controlled grey receiver, 0.8 albedo): a Cycles
        # sun of strength S lights it to 0.8*S/pi; OVRTX with the former
        # 120*pi^2 scale read 3.19x too bright, 120*pi reads 1.016x.
        formula = "120*pi*energy"
    attrs = [
        _attribute("inputs:diffuse", _float_attr(light, "diffuse_factor", 1.0), "diffuse_factor"),
        _attribute("inputs:specular", _float_attr(light, "specular_factor", 1.0), "specular_factor"),
        _attribute(
            "inputs:exposure",
            _float_attr(light, "exposure", 0.0),
            "exposure",
            {"source_property": "exposure", "source_units": "stops", "target_units": "UsdLux exposure stops"},
        ),
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
                "source_normalize": normalize,
                "normalize_policy": "ovrtx_0_3_ignores_inputs_normalize",
                "light_node_tree": node_factor.diagnostics(),
            },
        ),
        _attribute(
            "inputs:color",
            tuple(channel * node_factor.color[index] for index, channel in enumerate(blender_light_final_color(light))),
            "color",
            {
                "source_property": "color,use_temperature,temperature,node_tree",
                "color_temperature_policy": "bake_blender_temperature_color",
                "light_node_tree": node_factor.diagnostics(),
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
            cone_angle = blender_spot_cone_angle_degrees(_float_attr(light, "spot_size", 0.0), _float_attr(light, "spot_blend", 0.0))
            attrs.append(
                _attribute(
                    "inputs:shaping:cone:angle",
                    cone_angle,
                    "spot_size",
                    {
                        "exported_usd_family": "SphereLight",
                        "blend_policy": "hard_cone_at_blend_midpoint",
                        "source_blend": _float_attr(light, "spot_blend", 0.0),
                    },
                )
            )
            # Measured 2026-10-14: any nonzero OVRTX cone softness attenuates
            # from the cone axis (a default 45 degree / 0.15 blend spot lost
            # about 60% at its centre). Cycles keeps a flat plateau with a
            # smoothstep edge, so the edge is approximated by a hard cone at
            # the blend midpoint and softness stays zero.
            attrs.append(
                _attribute(
                    "inputs:shaping:cone:softness",
                    0.0,
                    "spot_blend",
                    {"exported_usd_family": "SphereLight", "blend_policy": "hard_cone_at_blend_midpoint"},
                )
            )
    elif light_type == "SUN":
        attrs.append(
            _attribute(
                "inputs:angle",
                math.degrees(_float_attr(light, "angle", 0.0)),  # UsdLux angle is the full angular diameter (measured)
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
    return float(watts) * PARITY_INTENSITY_SCALE


def blender_spot_cone_angle_degrees(spot_size: float, spot_blend: float) -> float:
    """Half-angle of the hard cone that matches Cycles' spot edge in energy.

    Cycles multiplies a spot by smoothstep((cos(theta) - cos(half)) /
    ((1 - cos(half)) * blend)), a plateau with an edge that is symmetric about
    cos(half) + (1 - cos(half)) * blend / 2.
    """

    half = max(0.0, float(spot_size)) / 2.0
    blend = min(1.0, max(0.0, float(spot_blend)))
    cos_half = math.cos(half)
    return math.degrees(math.acos(min(1.0, cos_half + 0.5 * (1.0 - cos_half) * blend)))


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


class LightNodeFactor:
    """Constant multipliers a light's node tree applies to energy and colour."""

    __slots__ = ("strength", "color", "status", "reason")

    def __init__(self, strength: float = 1.0, color: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 status: str = "absent", reason: str = "") -> None:
        self.strength = float(strength)
        self.color = tuple(float(channel) for channel in color)
        self.status = status
        self.reason = reason

    def diagnostics(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "strength": self.strength, "color": self.color}


def light_node_tree_factor(light: Any) -> LightNodeFactor:
    """Evaluate the constant part of a light node tree the way Cycles applies it.

    Cycles scales a light by its Light Output's Emission strength and colour
    (Blender 5.2 evaluates the tree whenever the light has one). Supported:
    Emission Strength as a constant or a Quadratic Light Falloff with a
    constant Strength, and Color as a constant or a Blackbody with a constant
    Temperature. Any other graph keeps energy-only conversion and reports why.
    """

    tree = getattr(light, "node_tree", None)
    nodes = list(getattr(tree, "nodes", ()) or ()) if tree is not None else []
    if not nodes:
        return LightNodeFactor()
    approximations: list[str] = []
    outputs = [node for node in nodes if _node_kind(node) == "OUTPUT_LIGHT"]
    active = [node for node in outputs if bool(getattr(node, "is_active_output", True))] or outputs
    if not active:
        return LightNodeFactor(status="unsupported", reason="missing_light_output")
    surface = _linked_from(_socket(active[0], "Surface"))
    if surface is None:
        return LightNodeFactor(status="unsupported", reason="unlinked_light_output")
    emission, _ = surface
    if _node_kind(emission) != "EMISSION":
        return LightNodeFactor(status="unsupported", reason="unsupported_light_node:" + _node_kind(emission))

    strength_source = _linked_from(_socket(emission, "Strength"))
    if strength_source is None:
        strength = _socket_float(_socket(emission, "Strength"), 1.0)
    else:
        falloff, output = strength_source
        if _node_kind(falloff) != "LIGHT_FALLOFF":
            return LightNodeFactor(status="unsupported", reason="unsupported_light_node:" + _node_kind(falloff))
        if output != "Quadratic":
            return LightNodeFactor(status="unsupported", reason="unsupported_light_falloff:" + output)
        if _linked_from(_socket(falloff, "Strength")) is not None:
            return LightNodeFactor(status="unsupported", reason="linked_light_falloff_strength")
        strength = _socket_float(_socket(falloff, "Strength"), 100.0)
        smooth = _socket_float(_socket(falloff, "Smooth"), 0.0)
        if smooth > 1.0e-6:
            # Cycles also multiplies by d^2 / (smooth + d^2), dimming surfaces
            # near the light; OVRTX lights have no such term and UsdLux has no
            # radial equivalent (measured n-04-07: OVRTX/Cycles is 1 + smooth/d^2
            # on a grey receiver, e.g. 2.0 at d=1 m for smooth 1, 1.1 at 3 m).
            approximations.append(f"light_falloff_smooth:{smooth:g}")

    color_source = _linked_from(_socket(emission, "Color"))
    if color_source is None:
        color = _socket_color(_socket(emission, "Color"))
    else:
        blackbody, _ = color_source
        if _node_kind(blackbody) != "BLACKBODY":
            return LightNodeFactor(status="unsupported", reason="unsupported_light_node:" + _node_kind(blackbody))
        if _linked_from(_socket(blackbody, "Temperature")) is not None:
            return LightNodeFactor(status="unsupported", reason="linked_blackbody_temperature")
        color = blackbody_rgb(_socket_float(_socket(blackbody, "Temperature"), 1500.0))
    if approximations:
        return LightNodeFactor(strength, color, status="approximated", reason=",".join(approximations))
    return LightNodeFactor(strength, color, status="applied")


_BLACKBODY_R = ((1.61919106e+03, -2.05010916e-03, 5.02995757e+00), (2.48845471e+03, -1.11330907e-03, 3.22621544e+00),
                (3.34143193e+03, -4.86551192e-04, 1.76486769e+00), (4.09461742e+03, -1.27446582e-04, 7.25731635e-01),
                (4.67028036e+03, 2.91258199e-05, 1.26703442e-01), (4.59509185e+03, 2.87495649e-05, 1.50345020e-01),
                (3.78717450e+03, 9.35907826e-06, 3.99075871e-01))
_BLACKBODY_G = ((-4.88999748e+02, 6.04330754e-04, -7.55807526e-02), (-7.55994277e+02, 3.16730098e-04, 4.78306139e-01),
                (-1.02363977e+03, 1.20223470e-04, 9.36662319e-01), (-1.26571316e+03, 4.87340896e-06, 1.27054498e+00),
                (-1.42529332e+03, -4.01150431e-05, 1.43972784e+00), (-1.17554822e+03, -2.16378048e-05, 1.30408023e+00),
                (-5.00799571e+02, -4.59832026e-06, 1.09098763e+00))
_BLACKBODY_B = ((5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03),
                (2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02),
                (-1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02),
                (-3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01),
                (-1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02),
                (-1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01),
                (6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01))


def blackbody_rgb(kelvin: float) -> tuple[float, float, float]:
    """Blender's blackbody colour (scene-linear Rec.709, luminance-normalised).

    Port of ``blackbody_temperature_to_rec709`` in Blender's
    ``imbuf/intern/colormanagement.cc``; matches Light.temperature_color and
    the Blackbody node under Blender's default (Rec.709 scene-linear) config.
    """

    t = float(kelvin)
    if t >= 12000.0:
        rgb = (0.8262954810464208, 0.9945080501520986, 1.566307710274283)
    elif t < 800.0:
        rgb = (5.413294490189271, -0.20319390035873933, -0.0822535242887164)
    else:
        i = (6 if t >= 6365.0 else 5 if t >= 3315.0 else 4 if t >= 1902.0 else
             3 if t >= 1449.0 else 2 if t >= 1167.0 else 1 if t >= 965.0 else 0)
        r, g, b = _BLACKBODY_R[i], _BLACKBODY_G[i], _BLACKBODY_B[i]
        rgb = (r[0] / t + r[1] * t + r[2], g[0] / t + g[1] * t + g[2], ((b[0] * t + b[1]) * t + b[2]) * t + b[3])
    return tuple(max(0.0, channel) for channel in rgb)  # type: ignore[return-value]


def _node_kind(node: Any) -> str:
    return str(getattr(node, "type", ""))


def _socket(node: Any, name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    getter = getattr(inputs, "get", None)
    return getter(name) if callable(getter) else None


def _linked_from(socket: Any | None) -> tuple[Any, str] | None:
    links = list(getattr(socket, "links", ()) or ()) if socket is not None else []
    if not links:
        return None
    link = links[0]
    node = getattr(link, "from_node", None)
    if bool(getattr(link, "is_muted", False)) or bool(getattr(node, "mute", False)):
        # Muted nodes pass inputs through (or nothing); never evaluate them as active.
        return _MUTED, "muted"
    return node, str(getattr(getattr(link, "from_socket", None), "name", ""))


class _MutedNode:
    type = "MUTED"


_MUTED = _MutedNode()


def _socket_float(socket: Any | None, default: float) -> float:
    try:
        return float(getattr(socket, "default_value", default))
    except (TypeError, ValueError):
        return default


def _socket_color(socket: Any | None) -> tuple[float, float, float]:
    try:
        value = tuple(float(channel) for channel in getattr(socket, "default_value", (1.0, 1.0, 1.0)))
    except (TypeError, ValueError):
        return (1.0, 1.0, 1.0)
    return value[:3] if len(value) >= 3 else (1.0, 1.0, 1.0)


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
        for field in ("energy", "exposure", "normalize", "diffuse_factor", "specular_factor", "color", "use_temperature", "temperature")
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
    matrix = getattr(light_object, "matrix_world", None)
    scale = matrix.to_scale() if matrix is not None and hasattr(matrix, "to_scale") else getattr(light_object, "scale", None)
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
