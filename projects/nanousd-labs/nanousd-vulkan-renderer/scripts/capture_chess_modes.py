#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the chess wrapper USDA in {RT, RASTER, SHADOW} modes and dump PNGs.

Standalone — drives the nusd_renderer ctypes bindings directly so we don't
need a full nanousdview/PySide setup. Mirrors the camera framing the
nanousdview window would produce by reading the wrapper's authored camera
and applying it explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

sys.path.insert(0, str(REPO / "python"))
from nusd_renderer._bindings import (  # noqa: E402
    NuRenderer,
    NU_RENDER_RT,
    NU_RENDER_RASTER,
    NU_RENDER_SHADOW,
)

ASSET_ROOT = Path(
    os.environ.get(
        "USD_WG_ASSETS",
        str(Path.home() / "simready/pixar/usd_wg_assets/full_assets"),
    )
)
WRAPPER_USD = Path(
    os.environ.get(
        "NUSD_CHESS_WRAPPER",
        str(ASSET_ROOT / "OpenChessSet/_chess_capture_wrapper.usda"),
    )
)
ENV_HDR = REPO / "comparisons" / "chess" / "photo_studio_london_hall_1k.hdr"

# From the wrapper USDA: identity rotation, translation (0.3, 0.25, 0.3),
# focalLength=18.147, vertical aperture=11.787 → vertical FOV ≈ 36°.
CAM_EYE = (0.3, 0.25, 0.3)
CAM_TARGET = (0.0, 0.0, 0.0)
CAM_UP = (0.0, 1.0, 0.0)
CAM_FOV_V = 36.0


def render_mode(out_png: Path, mode: int, width: int, height: int) -> None:
    nu = NuRenderer(width=width, height=height, enable_rt=True, enable_materials=True)
    nu.load_usd(str(WRAPPER_USD))
    nu.set_camera_explicit(eye=CAM_EYE, target=CAM_TARGET, up=CAM_UP,
                           fov_degrees=CAM_FOV_V, near_clip=0.001, far_clip=100.0)
    nu.build_accel()
    # Two renders: first is throwaway (lazy alloc / pipeline JIT).
    nu.render(mode=mode)
    nu.render(mode=mode)
    rgba = nu.fetch_pixels()
    Image.fromarray(rgba).save(out_png, optimize=True)
    nu.close()
    print(f"  wrote {out_png} ({width}x{height})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=810)
    ap.add_argument("--modes", nargs="+",
                    default=["rt", "raster", "shadow"],
                    choices=["rt", "raster", "shadow"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "rt": ("chess_vulkan_rt.png", NU_RENDER_RT),
        "raster": ("chess_vulkan_raster.png", NU_RENDER_RASTER),
        "shadow": ("chess_vulkan_shadow.png", NU_RENDER_SHADOW),
    }
    for m in args.modes:
        png_name, mode_id = table[m]
        print(f"chess {m}:")
        render_mode(args.out_dir / png_name, mode_id, args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
