# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import light_value_conversion  # noqa: E402
from ovrtx_blender_example import material_value_conversion  # noqa: E402
from ovrtx_blender_example import materialx_openpbr_conversion  # noqa: E402
from ovrtx_blender_example import usd_opinion_write  # noqa: E402
from ovrtx_blender_example import usd_value_edit_support  # noqa: E402
from ovrtx_blender_example import value_edit_conversion  # noqa: E402
from ovrtx_blender_example import world_dome_conversion  # noqa: E402


def test_default_value_edit_conversion_policies_are_module_shaped() -> None:
    policies = value_edit_conversion.default_value_edit_conversion_policies()

    assert policies.material is material_value_conversion
    assert policies.light is light_value_conversion
    assert policies.world is world_dome_conversion
    assert policies.material.SUPPORTED_USD_ATTRIBUTES["inputs:diffuseColor"] == "Color3f"
    assert policies.light.SUPPORTED_USD_ATTRIBUTES["inputs:intensity"] == "Float"
    assert policies.world.SUPPORTED_USD_ATTRIBUTES["inputs:color"] == "Color3f"


def test_usd_value_edit_support_contract_owns_supported_attribute_types() -> None:
    assert material_value_conversion.SUPPORTED_USD_ATTRIBUTES is usd_value_edit_support.MATERIAL_USD_VALUE_TYPES
    assert light_value_conversion.SUPPORTED_USD_ATTRIBUTES is usd_value_edit_support.LIGHT_USD_VALUE_TYPES
    assert world_dome_conversion.SUPPORTED_USD_ATTRIBUTES is usd_value_edit_support.WORLD_USD_VALUE_TYPES
    assert usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES["inputs:diffuseColor"] == "Color3f"
    assert usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES["inputs:geometry_opacity"] == "Float"
    assert usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES["inputs:emission_luminance"] == "Float"
    assert usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES["inputs:normalize"] == "Bool"
    assert usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES["xformOp:transform"] == "Matrix4d"


def test_declared_edit_value_support_is_writable() -> None:
    policies = value_edit_conversion.default_value_edit_conversion_policies()

    for policy in (policies.material, policies.light, policies.world):
        for field, attributes in policy.EDIT_VALUE_ATTRIBUTES_BY_FIELD.items():
            assert attributes, field
            for attribute in attributes:
                assert attribute in policy.SUPPORTED_USD_ATTRIBUTES, (policy, field, attribute)
                assert (
                    usd_value_edit_support.USD_OPINION_ATTRIBUTE_TYPES[attribute]
                    == policy.SUPPORTED_USD_ATTRIBUTES[attribute]
                )


def test_declared_edit_topology_support_is_writable() -> None:
    policies = value_edit_conversion.default_value_edit_conversion_policies()

    for policy in (policies.material, policies.light, policies.world):
        assert policy.EDIT_TOPOLOGY_KINDS <= usd_opinion_write.SUPPORTED_TOPOLOGY_KINDS


def test_declared_material_export_and_edit_value_concepts_match() -> None:
    assert materialx_openpbr_conversion.EXPORT_VALUE_CONCEPTS == material_value_conversion.EDIT_VALUE_CONCEPTS


def test_declared_material_export_and_edit_concepts_match_across_value_and_topology() -> None:
    export_concepts = (
        materialx_openpbr_conversion.EXPORT_VALUE_CONCEPTS
        | materialx_openpbr_conversion.EXPORT_TOPOLOGY_CONCEPTS
    )
    edit_concepts = (
        material_value_conversion.EDIT_VALUE_CONCEPTS
        | material_value_conversion.EDIT_TOPOLOGY_CONCEPTS
    )

    assert export_concepts == edit_concepts
