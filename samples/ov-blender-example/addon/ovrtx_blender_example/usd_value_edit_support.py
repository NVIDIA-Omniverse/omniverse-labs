# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-neutral USD value edit support contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


# Principled BSDF -> UsdPreviewSurface name mapping shared by the authored
# material conversion layer and the
# material value-edit policy (``material_value_conversion``): a live value
# edit must target exactly the attribute name the converter authored, so
# both sides read this one table (blender-live-render task04-02).
#
# Keys are the policy's Blender-facing field names. "Emission" is the
# policy's combined field: Blender's "Emission Color" x "Emission Strength"
# folded into one ``emissiveColor`` value. Values are
# ``(preview_surface_input_name, usd_value_type)``.
PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "Base Color": ("diffuseColor", "Color3f"),
        "Roughness": ("roughness", "Float"),
        "Metallic": ("metallic", "Float"),
        "IOR": ("ior", "Float"),
        "Emission": ("emissiveColor", "Color3f"),
    }
)

# Values the authored converter writes when a socket value is unavailable
# (no Principled BSDF node, or the socket is texture-driven): the
# UsdPreviewSurface spec defaults, except ``diffuseColor`` which keeps
# Blender's 0.8 default gray (the converter's historical default).
PREVIEW_SURFACE_VALUE_INPUT_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "diffuseColor": (0.8, 0.8, 0.8),
        "roughness": 0.5,
        "metallic": 0.0,
        "ior": 1.5,
        "emissiveColor": (0.0, 0.0, 0.0),
    }
)

# Principled sockets the authored converter wires textures for
# (``UsdUVTexture`` connections on the authored shader inputs). "Normal" is
# texture-only: a vector input, never a value edit. Texture and image
# changes on these sockets are shader-graph topology, not value edits.
PRINCIPLED_PREVIEW_SURFACE_TEXTURE_INPUTS: Mapping[str, str] = MappingProxyType(
    {
        "Base Color": "diffuseColor",
        "Emission Color": "emissiveColor",
        "Roughness": "roughness",
        "Metallic": "metallic",
        "Normal": "normal",
    }
)

MATERIAL_USD_VALUE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        **{
            "inputs:" + input_name: value_type
            for input_name, value_type in PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS.values()
        },
        "inputs:base_color": "Color3f",
        "inputs:base_metalness": "Float",
        "inputs:geometry_opacity": "Float",
        "inputs:specular_roughness": "Float",
        "inputs:transmission_weight": "Float",
        "inputs:transmission_color": "Color3f",
        "inputs:specular_ior": "Float",
        "inputs:specular_weight": "Float",
        "inputs:specular_color": "Color3f",
        "inputs:specular_roughness_anisotropy": "Float",
        "inputs:coat_weight": "Float",
        "inputs:coat_roughness": "Float",
        "inputs:coat_ior": "Float",
        "inputs:coat_color": "Color3f",
        "inputs:subsurface_weight": "Float",
        "inputs:subsurface_color": "Color3f",
        "inputs:subsurface_radius": "Float",
        "inputs:subsurface_radius_scale": "Color3f",
        "inputs:subsurface_scatter_anisotropy": "Float",
        "inputs:emission_color": "Color3f",
        "inputs:emission_luminance": "Float",
    }
)

LIGHT_USD_VALUE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "inputs:intensity": "Float",
        "inputs:color": "Color3f",
        "inputs:normalize": "Bool",
        "inputs:enableColorTemperature": "Bool",
        "inputs:radius": "Float",
        "inputs:width": "Float",
        "inputs:height": "Float",
        "inputs:shaping:cone:angle": "Float",
        "inputs:shaping:cone:softness": "Float",
        "inputs:angle": "Float",
    }
)

WORLD_USD_VALUE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "inputs:intensity": "Float",
        "inputs:color": "Color3f",
    }
)

# Camera projection/framing values attempted as live updates behind the
# per-session live-honor capability probe (blender-live-render task04-05).
# These are the composed camera's own UsdGeomCamera attribute names (the
# generated presentation authors them), not ``inputs:*`` shader inputs.
CAMERA_USD_VALUE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "focalLength": "Float",
        "horizontalAperture": "Float",
        "verticalAperture": "Float",
        "clippingRange": "Float2",
    }
)

USD_OPINION_ATTRIBUTE_TYPES: Mapping[str, str] = MappingProxyType(
    {
        **MATERIAL_USD_VALUE_TYPES,
        **LIGHT_USD_VALUE_TYPES,
        **WORLD_USD_VALUE_TYPES,
        "xformOp:transform": "Matrix4d",
    }
)
