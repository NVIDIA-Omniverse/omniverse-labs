# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one OpenGL OVRTX frame with optional NUSD_CURVES_SMOKETEST=1."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent
sys.path.insert(0, str(WORKSPACE / "nanousdview" / "python"))

from nanousdview._backend import (  # noqa: E402
    VIEW_RENDER_RASTER,
    OvrtxViewportRenderer,
    configure_backend,
)


def render_once(usd_path: str, smoketest: bool, out_png: str) -> None:
    if smoketest:
        os.environ["NUSD_CURVES_SMOKETEST"] = "1"
    else:
        os.environ.pop("NUSD_CURVES_SMOKETEST", None)

    width, height = 512, 512
    configure_backend("opengl", WORKSPACE)
    renderer = OvrtxViewportRenderer(width=width, height=height)
    try:
        renderer.load_stage(usd_path)
        renderer.set_camera(
            eye=(0.0, 0.0, 4.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=45.0,
            near_clip=0.1,
            far_clip=1000.0,
        )
        img = renderer.render_ldr(delta_time=0.0, render_mode=VIEW_RENDER_RASTER)
    finally:
        renderer.close()

    out = Path(out_png).with_suffix(".ppm")
    with open(out, "wb") as fh:
        fh.write(f"P6\n{width} {height}\n255\n".encode())
        fh.write(img[:, :, :3].astype("uint8").tobytes())
    print(f"wrote {out} ({img.shape})  nonzero_pixels={int((img[:, :, :3].sum(axis=2) > 0).sum())}")


if __name__ == "__main__":
    usd = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "test_pbr_materials.usda")
    mode = sys.argv[2] if len(sys.argv) > 2 else "on"
    render_once(usd, smoketest=(mode == "on"), out_png=f"/tmp/smoke_{mode}")
