# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared dataclasses for Dev Variant Presenter. No ovrtx/pxr imports (safe to import anywhere)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RenderModeName = Literal["RealTimePathTracing", "PathTracing", "Minimal"]
# RT2 token is camelCase, no spaces — verified against the ovrtx assets.


@dataclass(frozen=True)
class VariantChoice:
    """One variant set pinned to one variant on one prim."""
    prim_path: str
    set_name: str
    variant: str


@dataclass(frozen=True)
class VariantSetInfo:
    set_name: str
    prim_path: str
    variants: tuple[str, ...]
    current: str


@dataclass(frozen=True)
class CameraInfo:
    path: str   # e.g. "/World/Cameras/.../Main_Cam_01"
    name: str   # leaf name for display
    animated: bool = False   # time-sampled xform on the prim or any ancestor (turntable rig)


@dataclass(frozen=True)
class StageInfo:
    usd_path: str
    default_prim: str
    up_axis: str
    start_time: float
    end_time: float
    fps: float
    variant_sets: tuple[VariantSetInfo, ...]
    cameras: tuple[CameraInfo, ...]
    meters_per_unit: float = 1.0   # for converting ovrtx pick worldPositionM (meters) -> scene units


@dataclass(frozen=True)
class QualitySpec:
    mode: RenderModeName = "RealTimePathTracing"
    samples_per_pixel: int = 64      # PathTracing convergence target
    max_bounces: int = 4
    resolution: tuple[int, int] = (1920, 1080)


# A Selection is a tuple of VariantChoice. Group by prim for authoring.
Selection = tuple[VariantChoice, ...]
