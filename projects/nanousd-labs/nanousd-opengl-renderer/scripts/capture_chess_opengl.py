#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the chess wrapper USDA via the OpenGL OVRTX backend."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parent
sys.path.insert(0, str(WORKSPACE / "nanousdview" / "python"))

from nanousdview._backend import (  # noqa: E402
    VIEW_RENDER_RASTER,
    OvrtxViewportRenderer,
    configure_backend,
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
ENV_HDR = Path(
    os.environ.get(
        "NUSD_CHESS_HDR",
        str(WORKSPACE / "nanousd-vulkan-renderer/tests/correctness/assets/photo_studio_london_hall_1k.hdr"),
    )
)
CAM_EYE = (0.3, 0.25, 0.3)
CAM_TARGET = (0.0, 0.0, 0.0)
CAM_UP = (0.0, 1.0, 0.0)
CAM_FOV_V = 36.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    configure_backend("opengl", WORKSPACE)
    renderer = OvrtxViewportRenderer(width=args.width, height=args.height)
    try:
        renderer.load_stage(str(WRAPPER_USD))
        renderer.set_camera(
            eye=CAM_EYE,
            target=CAM_TARGET,
            up=CAM_UP,
            fov_degrees=CAM_FOV_V,
            near_clip=0.001,
            far_clip=100.0,
        )
        rgba = None
        for _ in range(2):
            rgba = renderer.render_ldr(delta_time=0.0, render_mode=VIEW_RENDER_RASTER)
        Image.fromarray(rgba).save(args.out, optimize=True)
        print(f"  wrote {args.out} ({args.width}x{args.height})")
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
