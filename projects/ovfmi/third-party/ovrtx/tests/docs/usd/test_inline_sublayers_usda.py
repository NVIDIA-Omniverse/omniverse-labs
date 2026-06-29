# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Validate USDA snippets for inline sublayer composition patterns."""

from pathlib import Path

from pxr import Usd

from conftest import validate_usda

DATA_DIR = Path(__file__).parent / "data"


def test_inline_sublayers_camera_renderproduct_usda():
    """Validate inline sublayer root with additional camera/render-product prims."""
    usda_path = DATA_DIR / "inline_sublayers_camera_renderproduct.usda"
    usda_text = usda_path.read_text()
    layer = validate_usda(usda_text)

    assert layer.GetPrimAtPath("/DocsCamera")
    assert layer.GetPrimAtPath("/Render/DocsCamera")
    assert layer.GetPrimAtPath("/Render/DocsCamera/LdrColor")

    stage = Usd.Stage.Open(str(usda_path))
    assert stage
    for prim_path in [
        "/World/Plane",
        "/World/Camera",
        "/Render/Camera",
        "/DocsCamera",
        "/Render/DocsCamera",
        "/Render/DocsCamera/LdrColor",
    ]:
        assert stage.GetPrimAtPath(prim_path).IsValid(), f"missing composed prim {prim_path}"
