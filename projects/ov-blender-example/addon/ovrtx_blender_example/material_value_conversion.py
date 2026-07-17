# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender material value edit conversion policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
    float_value,
    node_input,
    socket_is_linked,
)


MATERIAL_GRAPH_CHANGED = "topology_material_graph_changed"
MATERIAL_BINDING_CHANGED = "topology_material_binding_changed"
TEXTURE_CONNECTED_INPUT = "unsupported_material_texture_connected_input"

# Field-name mapping covering both authored shader families: the first
# attribute of each spec targets the UsdPreviewSurface input the shared
# Principled -> UsdPreviewSurface table (``usd_value_edit_support``)
# authors; the remaining attributes target the OpenPBR MaterialX surface
# shader inputs written by the material scene layer.
_PRINCIPLED_VALUE_SPECS = (
    ("Base Color", ("inputs:diffuseColor", "inputs:base_color")),
    ("Roughness", ("inputs:roughness", "inputs:specular_roughness")),
    ("Metallic", ("inputs:metallic", "inputs:base_metalness")),
    ("IOR", ("inputs:ior", "inputs:specular_ior")),
    ("Emission", ("inputs:emissiveColor", "inputs:emission_color", "inputs:emission_luminance")),
    ("Alpha", ("inputs:geometry_opacity",)),
    ("Transmission Weight", ("inputs:transmission_weight", "inputs:transmission_color")),
    ("Specular IOR Level", ("inputs:specular_weight",)),
    ("Specular Tint", ("inputs:specular_color",)),
    ("Anisotropic", ("inputs:specular_roughness_anisotropy",)),
    ("Coat Weight", ("inputs:coat_weight",)),
    ("Coat Roughness", ("inputs:coat_roughness",)),
    ("Coat IOR", ("inputs:coat_ior",)),
    ("Coat Tint", ("inputs:coat_color",)),
    ("Subsurface", (
        "inputs:subsurface_weight",
        "inputs:subsurface_color",
        "inputs:subsurface_radius",
        "inputs:subsurface_radius_scale",
        "inputs:subsurface_scatter_anisotropy",
    )),
)
_EMISSION_LUMINANCE_SCALE = 120.0 * 3.141592653589793 * 3.141592653589793
_EPSILON = 1.0e-6

SUPPORTED_USD_ATTRIBUTES = usd_value_edit_support.MATERIAL_USD_VALUE_TYPES

# Texture-wirable authored value inputs (the intersection of the shared
# value table and the converter's texture wiring table): the interactive
# edit builder compares Blender socket link state against the authored
# shader's connection state on these attributes to classify texture
# connects/disconnects as material-graph topology.
TEXTURE_WIRED_VALUE_ATTRIBUTES = tuple(
    "inputs:" + input_name
    for input_name in usd_value_edit_support.PRINCIPLED_PREVIEW_SURFACE_TEXTURE_INPUTS.values()
    if "inputs:" + input_name in usd_value_edit_support.MATERIAL_USD_VALUE_TYPES
)

EDIT_VALUE_ATTRIBUTES_BY_FIELD = MappingProxyType({
    **{
        f"principled:{input_name}": tuple(usd_attributes)
        for input_name, usd_attributes in _PRINCIPLED_VALUE_SPECS
    },
    "principled:Emission Color": ("inputs:emissiveColor", "inputs:emission_color"),
    "principled:Emission Strength": ("inputs:emissiveColor", "inputs:emission_luminance"),
    "principled:Subsurface Weight": ("inputs:subsurface_weight", "inputs:subsurface_color"),
    "principled:Subsurface Radius": ("inputs:subsurface_radius", "inputs:subsurface_radius_scale"),
    "principled:Subsurface Scale": ("inputs:subsurface_radius", "inputs:subsurface_radius_scale"),
    "principled:Subsurface Anisotropy": ("inputs:subsurface_scatter_anisotropy",),
})
EDIT_VALUE_CONCEPTS = frozenset(
    {
        "material.base_color",
        "material.roughness",
        "material.metallic",
        "material.opacity",
        "material.transmission",
        "material.ior",
        "material.specular",
        "material.anisotropy",
        "material.coat",
        "material.subsurface",
        "material.emission",
    }
)
EDIT_TOPOLOGY_CONCEPTS = frozenset({"material.graph"})
EDIT_TOPOLOGY_KINDS = frozenset()
_GRAPH_TOPOLOGY_INPUTS = frozenset(
    {
        "Base Color",
        "Roughness",
        "Metallic",
        "Alpha",
        "Transmission Weight",
        "Emission Color",
        "Normal",
        "Weight",
        "Subsurface Scale",
    }
)

_TOPOLOGY_FIELD_REASONS = {
    "use_nodes": "topology_material_node_mode_changed",
    "node_tree": MATERIAL_GRAPH_CHANGED,
    "nodes": MATERIAL_GRAPH_CHANGED,
    "links": MATERIAL_GRAPH_CHANGED,
    "texture": "topology_material_texture_changed",
    "image": "topology_material_texture_changed",
    "material_slot": MATERIAL_BINDING_CHANGED,
    "material_binding": MATERIAL_BINDING_CHANGED,
    "name": "topology_material_identifier_changed",
    "name_full": "topology_material_identifier_changed",
    "materialx_openpbr": "topology_materialx_openpbr_graph_changed",
}

_UNSUPPORTED_FIELD_REASONS = {
    "diffuse_color": "unsupported_material_datablock_surface_value",
    "specular_color": "unsupported_material_datablock_surface_value",
    "roughness": "unsupported_material_datablock_surface_value",
    "specular_intensity": "unsupported_material_datablock_surface_value",
    "metallic": "unsupported_material_datablock_surface_value",
    "surface_render_method": "unsupported_material_transparency_setting",
    "blend_method": "unsupported_material_transparency_setting",
    "alpha_threshold": "unsupported_material_transparency_setting",
    "use_transparency_overlap": "unsupported_material_transparency_setting",
    "show_transparent_back": "unsupported_material_transparency_setting",
    "use_transparent_shadow": "unsupported_material_transparency_setting",
    "use_backface_culling": "unsupported_material_culling_setting",
    "use_backface_culling_shadow": "unsupported_material_culling_setting",
    "use_backface_culling_lightprobe_volume": "unsupported_material_culling_setting",
    "use_raytrace_refraction": "unsupported_material_refraction_setting",
    "use_screen_refraction": "unsupported_material_refraction_setting",
    "refraction_depth": "unsupported_material_refraction_setting",
    "use_sss_translucency": "unsupported_material_subsurface_setting",
    "displacement_method": "unsupported_material_displacement_setting",
    "max_vertex_displacement": "unsupported_material_displacement_setting",
    "thickness_mode": "unsupported_material_thickness_setting",
    "use_thickness_from_shadow": "unsupported_material_thickness_setting",
    "volume_intersection_method": "unsupported_material_volume_setting",
    "line_color": "unsupported_material_line_art_setting",
    "line_priority": "unsupported_material_line_art_setting",
}

_UNSUPPORTED_PRINCIPLED_INPUT_REASONS = {
    "Normal": "unsupported_material_vector_input",
    "Tangent": "unsupported_material_vector_input",
    "Coat Normal": "unsupported_material_vector_input",
    "Weight": "unsupported_material_diffuse_lobe",
    "Diffuse Roughness": "unsupported_material_diffuse_lobe",
    "Subsurface IOR": "unsupported_material_subsurface_lobe",
    "Anisotropic Rotation": "unsupported_material_anisotropy_lobe",
    "Sheen Weight": "unsupported_material_sheen_lobe",
    "Sheen Roughness": "unsupported_material_sheen_lobe",
    "Sheen Tint": "unsupported_material_sheen_lobe",
    "Thin Film Thickness": "unsupported_material_thin_film_lobe",
    "Thin Film IOR": "unsupported_material_thin_film_lobe",
}

_NON_RENDER_FIELD_REASONS = {
    **BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    "preview_render_type": "non_runtime_material_editor_field",
    "use_preview_world": "non_runtime_material_editor_field",
    "paint_active_slot": "non_runtime_material_editor_field",
    "paint_clone_slot": "non_runtime_material_editor_field",
    "pass_index": "non_runtime_material_editor_field",
}


def classify_field(material: Any, property_name: str) -> FieldClassification:
    field = str(property_name or "").strip()
    classification = classify_mapped_field(
        field,
        non_render=_NON_RENDER_FIELD_REASONS,
        topology=_TOPOLOGY_FIELD_REASONS,
        unsupported=_UNSUPPORTED_FIELD_REASONS,
    )
    if classification is not None:
        return classification
    if field.startswith("principled:"):
        input_name = field.split(":", 1)[1]
        if input_name in _UNSUPPORTED_PRINCIPLED_INPUT_REASONS:
            return FieldClassification(STATUS_UNSUPPORTED, _UNSUPPORTED_PRINCIPLED_INPUT_REASONS[input_name])
        return _classify_supported_principled_input(material, input_name)
    return FieldClassification(STATUS_UNSUPPORTED, "unsupported_material_field")


def usd_attribute_values(material: Any) -> tuple[UsdAttributeValue, ...]:
    principled = _principled_node(material)
    if principled is None:
        return ()

    attributes: list[UsdAttributeValue] = []
    base_color = _unlinked_input_value(principled, "Base Color")
    base_color_value: tuple[float, float, float] | None = None
    if base_color.available:
        base_color_value = _rgb_tuple(base_color.value)
        attributes.append(
            _attribute(
                "inputs:diffuseColor",
                base_color_value,
                "principled:Base Color",
                ("Principled BSDF.Base Color",),
            )
        )
        attributes.append(
            _attribute(
                "inputs:base_color",
                base_color_value,
                "principled:Base Color",
                ("Principled BSDF.Base Color",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
    for input_name, attributes_by_family in (
        ("Roughness", (("inputs:roughness", "UsdPreviewSurface"), ("inputs:specular_roughness", "ND_open_pbr_surface_surfaceshader"))),
        ("Metallic", (("inputs:metallic", "UsdPreviewSurface"), ("inputs:base_metalness", "ND_open_pbr_surface_surfaceshader"))),
        ("IOR", (("inputs:ior", "UsdPreviewSurface"), ("inputs:specular_ior", "ND_open_pbr_surface_surfaceshader"))),
    ):
        value = _unlinked_input_value(principled, input_name)
        if value.available:
            for usd_attribute, shader_family in attributes_by_family:
                attributes.append(
                    _attribute(
                        usd_attribute,
                        float_value(value.value, 0.0),
                        f"principled:{input_name}",
                        (f"Principled BSDF.{input_name}",),
                        shader_family=shader_family,
                    )
                )
    _append_unlinked_float(
        attributes,
        principled,
        "Alpha",
        "inputs:geometry_opacity",
        default=1.0,
        concept_property="principled:Alpha",
    )
    transmission_weight = _unlinked_input_value(principled, "Transmission Weight")
    if transmission_weight.available:
        weight = float_value(transmission_weight.value, 0.0)
        attributes.append(
            _attribute(
                "inputs:transmission_weight",
                weight,
                "principled:Transmission Weight",
                ("Principled BSDF.Transmission Weight",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
        if weight > _EPSILON and base_color_value is not None and any(abs(channel - 1.0) > _EPSILON for channel in base_color_value):
            attributes.append(
                _attribute(
                    "inputs:transmission_color",
                    base_color_value,
                    "principled:Transmission Weight",
                    ("Principled BSDF.Transmission Weight", "Principled BSDF.Base Color"),
                    shader_family="ND_open_pbr_surface_surfaceshader",
                )
            )
    _append_specular_values(attributes, principled)
    _append_coat_values(attributes, principled)
    _append_subsurface_values(attributes, principled, base_color_value)
    emission_color = _unlinked_input_value(principled, "Emission Color")
    emission_strength = _unlinked_input_value(principled, "Emission Strength")
    if emission_color.available and emission_strength.available:
        color = _rgb_tuple(emission_color.value)
        strength = max(0.0, float_value(emission_strength.value, 1.0))
        attributes.append(
            _attribute(
                _principled_usd_attribute("Emission"),
                tuple(channel * strength for channel in color),
                "principled:Emission",
                ("Principled BSDF.Emission Color", "Principled BSDF.Emission Strength"),
            )
        )
        attributes.append(
            _attribute(
                "inputs:emission_color",
                color,
                "principled:Emission",
                ("Principled BSDF.Emission Color",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
        attributes.append(
            _attribute(
                "inputs:emission_luminance",
                strength * _EMISSION_LUMINANCE_SCALE,
                "principled:Emission",
                ("Principled BSDF.Emission Strength",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
    return tuple(attributes)


def requires_topology_edit(material: Any) -> bool:
    node_tree = getattr(material, "node_tree", None)
    nodes = tuple(getattr(node_tree, "nodes", ()) or ())
    if not nodes:
        return False
    principled = _principled_node(material)
    if principled is None:
        return True
    return any(
        socket_is_linked(socket)
        for socket_name in _GRAPH_TOPOLOGY_INPUTS
        if (socket := node_input(principled, socket_name)) is not None
    )


def texture_wired_input_links(material: Any) -> dict[str, bool] | None:
    """Blender link state per texture-wirable authored value input.

    Returns ``None`` when the material has no Principled BSDF node (the
    authored shader carries defaults; graph divergence cannot be assessed).
    Keys are the authored ``inputs:*`` attribute names shared with the
    topology converter; values are whether the corresponding Principled
    socket currently has a link. The interactive edit builder compares this
    against the authored shader's connection state so texture connects and
    disconnects classify as material-graph topology (task04-02).
    """

    principled = _principled_node(material)
    if principled is None:
        return None
    links: dict[str, bool] = {}
    for socket_name, input_name in usd_value_edit_support.PRINCIPLED_PREVIEW_SURFACE_TEXTURE_INPUTS.items():
        usd_attribute = "inputs:" + input_name
        if usd_attribute not in SUPPORTED_USD_ATTRIBUTES:
            # Texture-only inputs (normal) are never value edits.
            continue
        socket = node_input(principled, socket_name)
        if socket is None:
            continue
        links[usd_attribute] = socket_is_linked(socket)
    return links


@dataclass(frozen=True)
class _InputValue:
    available: bool
    value: Any = None


def _classify_supported_principled_input(material: Any, input_name: str) -> FieldClassification:
    usd_attributes = EDIT_VALUE_ATTRIBUTES_BY_FIELD.get(f"principled:{input_name}")
    if usd_attributes is None:
        return FieldClassification(STATUS_UNSUPPORTED, "unsupported_material_field")
    principled = _principled_node(material)
    if principled is None:
        return FieldClassification(STATUS_TOPOLOGY, "material_without_principled_bsdf")
    if input_name == "Emission":
        sockets = ("Emission Color", "Emission Strength")
    else:
        sockets = (input_name,)
    for socket_name in sockets:
        socket = node_input(principled, socket_name)
        if socket is None:
            return FieldClassification(STATUS_UNSUPPORTED, "unsupported_material_field")
        if socket_is_linked(socket):
            # A value change on a texture-driven socket has no rendered
            # effect (the connection wins): unsupported-for-value-edit and
            # reported via edit records (task04-07). Connecting or
            # disconnecting the texture itself is a graph rewire and stays
            # topology (``node_tree``/``links`` fields above and the
            # builder's authored-connection divergence check).
            return FieldClassification(STATUS_UNSUPPORTED, TEXTURE_CONNECTED_INPUT)
    return FieldClassification(STATUS_SUPPORTED, "supported_material_principled_value", usd_attributes)


def _principled_usd_attribute(input_name: str) -> str:
    return EDIT_VALUE_ATTRIBUTES_BY_FIELD[f"principled:{input_name}"][0]


def _attribute(
    name: str,
    value: Any,
    blender_property_path: str,
    source_fields: tuple[str, ...],
    *,
    shader_family: str = "UsdPreviewSurface",
) -> UsdAttributeValue:
    return UsdAttributeValue(
        name=name,
        value=value,
        value_type=SUPPORTED_USD_ATTRIBUTES[name],
        blender_property_path=blender_property_path,
        metadata={
            "source_fields": source_fields,
            "shader_family": shader_family,
        },
    )


def _append_unlinked_float(
    attributes: list[UsdAttributeValue],
    node: Any,
    input_name: str,
    usd_attribute: str,
    *,
    default: float = 0.0,
    concept_property: str | None = None,
) -> None:
    value = _unlinked_input_value(node, input_name)
    if not value.available:
        return
    attributes.append(
        _attribute(
            usd_attribute,
            float_value(value.value, default),
            concept_property or f"principled:{input_name}",
            (f"Principled BSDF.{input_name}",),
            shader_family="ND_open_pbr_surface_surfaceshader",
        )
    )


def _append_specular_values(attributes: list[UsdAttributeValue], node: Any) -> None:
    level = _unlinked_input_value(node, "Specular IOR Level")
    if level.available:
        attributes.append(
            _attribute(
                "inputs:specular_weight",
                2.0 * float_value(level.value, 0.5),
                "principled:Specular IOR Level",
                ("Principled BSDF.Specular IOR Level",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
    tint = _unlinked_input_value(node, "Specular Tint")
    if tint.available:
        attributes.append(
            _attribute(
                "inputs:specular_color",
                _rgb_tuple(tint.value),
                "principled:Specular Tint",
                ("Principled BSDF.Specular Tint",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
    _append_unlinked_float(
        attributes,
        node,
        "Anisotropic",
        "inputs:specular_roughness_anisotropy",
        concept_property="principled:Anisotropic",
    )


def _append_coat_values(attributes: list[UsdAttributeValue], node: Any) -> None:
    for input_name, usd_attribute, default in (
        ("Coat Weight", "inputs:coat_weight", 0.0),
        ("Coat Roughness", "inputs:coat_roughness", 0.03),
        ("Coat IOR", "inputs:coat_ior", 1.5),
    ):
        _append_unlinked_float(attributes, node, input_name, usd_attribute, default=default)
    tint = _unlinked_input_value(node, "Coat Tint")
    if tint.available:
        attributes.append(
            _attribute(
                "inputs:coat_color",
                _rgb_tuple(tint.value),
                "principled:Coat Tint",
                ("Principled BSDF.Coat Tint",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )


def _append_subsurface_values(
    attributes: list[UsdAttributeValue],
    node: Any,
    base_color: tuple[float, float, float] | None,
) -> None:
    weight = _unlinked_input_value(node, "Subsurface Weight")
    radius = _unlinked_input_value(node, "Subsurface Radius")
    scale = _unlinked_input_value(node, "Subsurface Scale")
    if not (weight.available and radius.available and scale.available):
        return
    effective_radius = tuple(max(0.0, channel * float_value(scale.value, 0.05)) for channel in _rgb_tuple(radius.value))
    max_radius = max(effective_radius)
    if max_radius <= _EPSILON:
        return
    attributes.extend(
        (
            _attribute(
                "inputs:subsurface_weight",
                float_value(weight.value, 0.0),
                "principled:Subsurface Weight",
                ("Principled BSDF.Subsurface Weight",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            ),
            _attribute(
                "inputs:subsurface_radius",
                max_radius,
                "principled:Subsurface Scale",
                ("Principled BSDF.Subsurface Radius", "Principled BSDF.Subsurface Scale"),
                shader_family="ND_open_pbr_surface_surfaceshader",
            ),
            _attribute(
                "inputs:subsurface_radius_scale",
                tuple(channel / max_radius for channel in effective_radius),
                "principled:Subsurface Radius",
                ("Principled BSDF.Subsurface Radius", "Principled BSDF.Subsurface Scale"),
                shader_family="ND_open_pbr_surface_surfaceshader",
            ),
        )
    )
    if base_color is not None:
        attributes.append(
            _attribute(
                "inputs:subsurface_color",
                base_color,
                "principled:Subsurface Weight",
                ("Principled BSDF.Subsurface Weight", "Principled BSDF.Base Color"),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )
    anisotropy = _unlinked_input_value(node, "Subsurface Anisotropy")
    if anisotropy.available:
        attributes.append(
            _attribute(
                "inputs:subsurface_scatter_anisotropy",
                float_value(anisotropy.value, 0.0),
                "principled:Subsurface Anisotropy",
                ("Principled BSDF.Subsurface Anisotropy",),
                shader_family="ND_open_pbr_surface_surfaceshader",
            )
        )


def _principled_node(material: Any) -> Any | None:
    node_tree = getattr(material, "node_tree", None)
    for node in getattr(node_tree, "nodes", ()) or ():
        node_type = str(getattr(node, "type", "") or "")
        node_name = str(getattr(node, "name", "") or "")
        if node_type == "BSDF_PRINCIPLED" or node_name == "Principled BSDF":
            return node
    return None


def _unlinked_input_value(node: Any, name: str) -> _InputValue:
    socket = node_input(node, name)
    if socket is None or socket_is_linked(socket):
        return _InputValue(False)
    return _InputValue(True, getattr(socket, "default_value", None))


def _rgb_tuple(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        return (0.0, 0.0, 0.0)
    try:
        values = list(value)
    except TypeError:
        return (0.0, 0.0, 0.0)
    if len(values) < 3:
        return (0.0, 0.0, 0.0)
    return tuple(max(0.0, float_value(values[index], 0.0)) for index in range(3))
