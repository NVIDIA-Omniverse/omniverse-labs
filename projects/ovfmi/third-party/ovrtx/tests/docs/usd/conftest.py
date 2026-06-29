# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Shared helpers for USDA validation tests."""

from pxr import Sdf


def validate_usda(usda_text: str) -> Sdf.Layer:
    """Parse a USDA string and return the layer, failing if invalid."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(usda_text), "Failed to parse USDA"
    return layer
