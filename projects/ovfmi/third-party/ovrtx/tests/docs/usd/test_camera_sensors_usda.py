# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Validate USDA files used in camera_sensors.rst."""

from pathlib import Path

from conftest import validate_usda

DATA_DIR = Path(__file__).parent / "data"


def test_camera_sensor_render_product():
    """Validate camera_sensor_render_product.usda has expected structure."""
    usda_text = (DATA_DIR / "camera_sensor_render_product.usda").read_text()
    layer = validate_usda(usda_text)

    camera_prim = layer.GetPrimAtPath("/Render/Camera")
    assert camera_prim, "/Render/Camera prim should exist"

    ldr_prim = layer.GetPrimAtPath("/Render/Camera/LdrColor")
    assert ldr_prim, "/Render/Camera/LdrColor prim should exist"

    hdr_prim = layer.GetPrimAtPath("/Render/Camera/HdrColor")
    assert hdr_prim, "/Render/Camera/HdrColor prim should exist"
