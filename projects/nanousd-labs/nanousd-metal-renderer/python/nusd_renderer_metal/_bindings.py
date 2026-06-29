# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private Metal backend bindings for the nanousd OVRTX facade."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BINDINGS_PATH = Path(__file__).resolve().parents[1] / "nusd_renderer" / "_bindings.py"
_SPEC = importlib.util.spec_from_file_location("_nanousd_metal_backend_bindings", _BINDINGS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Metal renderer bindings from {_BINDINGS_PATH}")
_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_module)

NuRenderer = _module.NuRenderer
NU_RENDER_RASTER = _module.NU_RENDER_RASTER
NU_RENDER_SHADOW = _module.NU_RENDER_SHADOW
NU_RENDER_RT = _module.NU_RENDER_RT
NU_GS_CAMERA_EQUIRECT = _module.NU_GS_CAMERA_EQUIRECT
NU_GS_CAMERA_FISHEYE = _module.NU_GS_CAMERA_FISHEYE
NU_GS_CAMERA_PINHOLE = _module.NU_GS_CAMERA_PINHOLE
NU_GS_COLOR_LINEAR = _module.NU_GS_COLOR_LINEAR
NU_GS_COLOR_SRGB = _module.NU_GS_COLOR_SRGB
NU_GS_PROXY_AABB = _module.NU_GS_PROXY_AABB
NU_GS_PROXY_ICOSAHEDRON = _module.NU_GS_PROXY_ICOSAHEDRON

__all__ = [
    "NuRenderer",
    "NU_RENDER_RASTER",
    "NU_RENDER_SHADOW",
    "NU_RENDER_RT",
    "NU_GS_CAMERA_EQUIRECT",
    "NU_GS_CAMERA_FISHEYE",
    "NU_GS_CAMERA_PINHOLE",
    "NU_GS_COLOR_LINEAR",
    "NU_GS_COLOR_SRGB",
    "NU_GS_PROXY_AABB",
    "NU_GS_PROXY_ICOSAHEDRON",
]
