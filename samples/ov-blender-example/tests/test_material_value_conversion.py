# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import material_value_conversion as conversion  # noqa: E402


class _Socket:
    def __init__(self, value: object, *, linked: bool = False) -> None:
        self.default_value = value
        self.is_linked = linked


class _Node:
    def __init__(self, node_type: str, inputs: dict[str, _Socket]) -> None:
        self.type = node_type
        self.name = "Principled BSDF" if node_type == "BSDF_PRINCIPLED" else node_type
        self.inputs = inputs


def _material(inputs: dict[str, _Socket] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        type="MATERIAL",
        node_tree=SimpleNamespace(
            nodes=[
                _Node(
                    "BSDF_PRINCIPLED",
                    inputs
                    or {
                        "Base Color": _Socket((0.1, 0.2, 0.3, 1.0)),
                        "Roughness": _Socket(0.45),
                        "Metallic": _Socket(0.2),
                        "IOR": _Socket(1.45),
                        "Alpha": _Socket(0.5),
                        "Transmission Weight": _Socket(0.75),
                        "Specular IOR Level": _Socket(0.25),
                        "Specular Tint": _Socket((0.8, 0.9, 1.0, 1.0)),
                        "Anisotropic": _Socket(0.4),
                        "Coat Weight": _Socket(0.3),
                        "Coat Roughness": _Socket(0.2),
                        "Coat IOR": _Socket(1.4),
                        "Coat Tint": _Socket((0.9, 0.8, 0.7, 1.0)),
                        "Subsurface Weight": _Socket(0.6),
                        "Subsurface Radius": _Socket((1.0, 0.4, 0.2)),
                        "Subsurface Scale": _Socket(0.05),
                        "Subsurface Anisotropy": _Socket(0.3),
                        "Emission Color": _Socket((0.1, 0.2, 0.5, 1.0)),
                        "Emission Strength": _Socket(2.0),
                    },
                )
            ]
        ),
    )


def test_usd_attribute_values_emit_research_supported_preview_surface_values() -> None:
    attributes = {attribute.name: attribute for attribute in conversion.usd_attribute_values(_material())}

    assert attributes["inputs:diffuseColor"].value == (0.1, 0.2, 0.3)
    assert attributes["inputs:roughness"].value == 0.45
    assert attributes["inputs:specular_roughness"].value == 0.45
    assert attributes["inputs:metallic"].value == 0.2
    assert attributes["inputs:base_metalness"].value == 0.2
    assert attributes["inputs:ior"].value == 1.45
    assert attributes["inputs:specular_ior"].value == 1.45
    assert attributes["inputs:geometry_opacity"].value == 0.5
    assert attributes["inputs:transmission_weight"].value == 0.75
    assert attributes["inputs:transmission_color"].value == (0.1, 0.2, 0.3)
    assert attributes["inputs:specular_weight"].value == 0.5
    assert attributes["inputs:specular_color"].value == (0.8, 0.9, 1.0)
    assert attributes["inputs:specular_roughness_anisotropy"].value == 0.4
    assert attributes["inputs:coat_weight"].value == 0.3
    assert attributes["inputs:coat_roughness"].value == 0.2
    assert attributes["inputs:coat_ior"].value == 1.4
    assert attributes["inputs:coat_color"].value == (0.9, 0.8, 0.7)
    assert attributes["inputs:subsurface_weight"].value == 0.6
    assert attributes["inputs:subsurface_color"].value == (0.1, 0.2, 0.3)
    assert attributes["inputs:subsurface_radius"].value == 0.05
    assert attributes["inputs:subsurface_radius_scale"].value == pytest.approx((1.0, 0.4, 0.2))
    assert attributes["inputs:subsurface_scatter_anisotropy"].value == 0.3
    assert attributes["inputs:emissiveColor"].value == (0.2, 0.4, 1.0)
    assert attributes["inputs:emission_color"].value == (0.1, 0.2, 0.5)
    assert attributes["inputs:emission_luminance"].value == 2.0 * 120.0 * 3.141592653589793 * 3.141592653589793
    assert set(attributes) == set(conversion.SUPPORTED_USD_ATTRIBUTES)


def test_supported_usd_attributes_share_classifier_usd_attributes() -> None:
    material = _material()
    authoring_properties = {
        "principled:Base Color": ("inputs:diffuseColor", "inputs:base_color"),
        "principled:Roughness": ("inputs:roughness", "inputs:specular_roughness"),
        "principled:Metallic": ("inputs:metallic", "inputs:base_metalness"),
        "principled:IOR": ("inputs:ior", "inputs:specular_ior"),
        "principled:Emission": ("inputs:emissiveColor", "inputs:emission_color", "inputs:emission_luminance"),
        "principled:Alpha": ("inputs:geometry_opacity",),
        "principled:Transmission Weight": ("inputs:transmission_weight", "inputs:transmission_color"),
        "principled:Specular IOR Level": ("inputs:specular_weight",),
        "principled:Specular Tint": ("inputs:specular_color",),
        "principled:Anisotropic": ("inputs:specular_roughness_anisotropy",),
        "principled:Coat Weight": ("inputs:coat_weight",),
        "principled:Coat Roughness": ("inputs:coat_roughness",),
        "principled:Coat IOR": ("inputs:coat_ior",),
        "principled:Coat Tint": ("inputs:coat_color",),
    }

    for property_name, usd_attributes in authoring_properties.items():
        classification = conversion.classify_field(material, property_name)

        assert classification.status == conversion.STATUS_SUPPORTED
        assert classification.usd_attributes == usd_attributes
    classified_attributes = {
        attribute
        for attributes in authoring_properties.values()
        for attribute in attributes
    }
    assert classified_attributes <= set(conversion.SUPPORTED_USD_ATTRIBUTES)


def test_texture_connected_supported_input_is_unsupported_and_not_emitted() -> None:
    # task04-02 clarification: a value change on a texture-driven socket is
    # unsupported-for-value-edit (reported via task04-07), not topology —
    # connecting/disconnecting the texture itself stays topology.
    material = _material(
        {
            "Base Color": _Socket((1.0, 0.0, 0.0, 1.0), linked=True),
            "Roughness": _Socket(0.45),
        }
    )

    classification = conversion.classify_field(material, "principled:Base Color")
    attributes = {attribute.name for attribute in conversion.usd_attribute_values(material)}

    assert classification.status == conversion.STATUS_UNSUPPORTED
    assert classification.reason == conversion.TEXTURE_CONNECTED_INPUT
    assert "inputs:diffuseColor" not in attributes


def test_missing_principled_is_topology() -> None:
    material = SimpleNamespace(type="MATERIAL", node_tree=SimpleNamespace(nodes=[]))

    classification = conversion.classify_field(material, "principled:Roughness")

    assert classification.status == conversion.STATUS_TOPOLOGY
    assert classification.reason == "material_without_principled_bsdf"
    assert conversion.usd_attribute_values(material) == ()


def test_policy_field_mapping_derives_from_shared_converter_table() -> None:
    # task04-02: the value-edit policy and the authored material converter
    # share one Principled -> UsdPreviewSurface name mapping.
    from ovrtx_blender_example import usd_value_edit_support

    shared = usd_value_edit_support.PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS
    # The policy's specs cover every shared field, and each shared field's
    # primary (UsdPreviewSurface) attribute is exactly what the shared table
    # authors. The specs additionally carry the OpenPBR MaterialX shader
    # attributes, which the shared preview-surface table does not describe.
    policy_specs = dict(conversion._PRINCIPLED_VALUE_SPECS)
    for field_name, (input_name, value_type) in shared.items():
        assert policy_specs[field_name][0] == "inputs:" + input_name
        assert conversion.SUPPORTED_USD_ATTRIBUTES["inputs:" + input_name] == value_type
    # Every shared value input is present in the supported-attribute table
    # (a superset that also lists the OpenPBR attributes).
    assert {
        "inputs:" + input_name for input_name, _ in shared.values()
    } <= set(conversion.SUPPORTED_USD_ATTRIBUTES)
    # Every shared value input has an authored default for the converter.
    assert set(usd_value_edit_support.PREVIEW_SURFACE_VALUE_INPUT_DEFAULTS) == {
        spec[0] for spec in shared.values()
    }
    # Texture-wirable value attributes are the intersection of the shared
    # value table and the converter's texture wiring table.
    assert conversion.TEXTURE_WIRED_VALUE_ATTRIBUTES == (
        "inputs:diffuseColor",
        "inputs:emissiveColor",
        "inputs:roughness",
        "inputs:metallic",
    )

def test_texture_wired_input_links_reports_blender_link_state() -> None:
    unlinked = conversion.texture_wired_input_links(_material())
    linked = conversion.texture_wired_input_links(
        _material(
            {
                "Base Color": _Socket((1.0, 0.0, 0.0, 1.0), linked=True),
                "Emission Color": _Socket((0.0, 0.0, 0.0, 1.0)),
                "Roughness": _Socket(0.45),
                "Metallic": _Socket(0.2),
            }
        )
    )
    from types import SimpleNamespace

    no_principled = conversion.texture_wired_input_links(
        SimpleNamespace(type="MATERIAL", node_tree=SimpleNamespace(nodes=[]))
    )

    assert unlinked == {
        "inputs:diffuseColor": False,
        "inputs:emissiveColor": False,
        "inputs:roughness": False,
        "inputs:metallic": False,
    }
    assert linked == {
        "inputs:diffuseColor": True,
        "inputs:emissiveColor": False,
        "inputs:roughness": False,
        "inputs:metallic": False,
    }
    assert no_principled is None


def test_material_field_classification_reports_topology_unsupported_and_non_render_fields() -> None:
    material = _material()

    assert conversion.classify_field(material, "use_nodes").status == conversion.STATUS_TOPOLOGY
    assert conversion.classify_field(material, "diffuse_color").reason == "unsupported_material_datablock_surface_value"
    assert conversion.classify_field(material, "use_fake_user").status == conversion.STATUS_NON_RENDER
