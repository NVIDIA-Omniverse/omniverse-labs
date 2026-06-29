# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Validate USDA files used in sensor_configuration.rst."""

from pathlib import Path

from conftest import validate_usda

DATA_DIR = Path(__file__).parent / "data"


def test_minimal_render_product():
    """Validate minimal_render_product.usda has expected structure."""
    usda_text = (DATA_DIR / "minimal_render_product.usda").read_text()
    layer = validate_usda(usda_text)

    camera_prim = layer.GetPrimAtPath("/Render/Camera")
    assert camera_prim, "/Render/Camera prim should exist"

    ldr_prim = layer.GetPrimAtPath("/Render/Camera/LdrColor")
    assert ldr_prim, "/Render/Camera/LdrColor prim should exist"


def test_multi_render_product_shared_vars():
    """Validate multi_render_product_shared_vars.usda has expected structure."""
    usda_text = (DATA_DIR / "multi_render_product_shared_vars.usda").read_text()
    layer = validate_usda(usda_text)

    front_prim = layer.GetPrimAtPath("/Render/FrontCamera")
    assert front_prim, "/Render/FrontCamera prim should exist"

    rear_prim = layer.GetPrimAtPath("/Render/RearCamera")
    assert rear_prim, "/Render/RearCamera prim should exist"

    ldr_prim = layer.GetPrimAtPath("/Render/Vars/LdrColor")
    assert ldr_prim, "/Render/Vars/LdrColor prim should exist"
