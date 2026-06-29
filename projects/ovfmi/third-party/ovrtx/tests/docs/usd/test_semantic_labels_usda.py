# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Validate USDA snippets for semantic label override patterns."""

from pathlib import Path

from pxr import Usd

from conftest import validate_usda

DATA_DIR = Path(__file__).parent / "data"


def test_semantic_label_overrides_usda():
    """Validate SemanticsAPI class/label overs on existing composed prims."""
    usda_path = DATA_DIR / "semantic_label_overrides.usda"
    usda_text = usda_path.read_text()
    validate_usda(usda_text)

    stage = Usd.Stage.Open(str(usda_path))
    assert stage

    for robot_name in ["robot_alpha", "robot_beta"]:
        prim = stage.GetPrimAtPath(f"/World/{robot_name}")
        assert prim.IsValid(), f"missing composed prim /World/{robot_name}"
        api_schemas = prim.GetMetadata("apiSchemas")
        assert list(api_schemas.explicitItems) == ["SemanticsAPI:label", "SemanticsAPI:class"]
        assert prim.GetAttribute("semantic:class:params:semanticType").Get() == "class"
        assert prim.GetAttribute("semantic:class:params:semanticData").Get() == "robot"
        assert prim.GetAttribute("semantic:label:params:semanticType").Get() == "label"
        assert prim.GetAttribute("semantic:label:params:semanticData").Get() == robot_name

    warehouse = stage.GetPrimAtPath("/World/warehouse")
    assert warehouse.IsValid()
    assert warehouse.GetMetadata("apiSchemas") is None
