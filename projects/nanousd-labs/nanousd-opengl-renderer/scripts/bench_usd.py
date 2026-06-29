#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a USD file through nanousdview's OpenGL OVRTX backend."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent
sys.path.insert(0, str(WORKSPACE / "nanousdview" / "python"))

from nanousdview._backend import (  # noqa: E402
    VIEW_RENDER_RASTER,
    OvrtxViewportRenderer,
    configure_backend,
)


def rss_mb() -> float:
    # macOS: ru_maxrss is bytes; Linux: kilobytes.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd_path")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args()

    print(f"=== {Path(args.usd_path).name} ===")
    rss0 = rss_mb()
    print(f"baseline RSS: {rss0:.0f} MB")

    configure_backend("opengl", WORKSPACE)
    t0 = time.perf_counter()
    renderer = OvrtxViewportRenderer(width=args.width, height=args.height)
    try:
        renderer.load_stage(args.usd_path)
        renderer.set_camera(
            eye=(3.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=45.0,
            near_clip=0.01,
            far_clip=10000.0,
        )
        t_load = time.perf_counter() - t0
        rss_load = rss_mb()
        print(f"load:     {t_load*1000:.1f} ms  ({rss_load - rss0:+.0f} MB -> RSS {rss_load:.0f} MB)")

        renderer.render_ldr(delta_time=0.0, render_mode=VIEW_RENDER_RASTER)

        t_start = time.perf_counter()
        for _ in range(args.frames):
            renderer.render_ldr(delta_time=0.0, render_mode=VIEW_RENDER_RASTER)
        t_render = time.perf_counter() - t_start
        rss_peak = rss_mb()
    finally:
        renderer.close()

    fps = args.frames / t_render if t_render > 0.0 else 0.0
    ms_per_frame = t_render * 1000.0 / max(args.frames, 1)
    print(f"render:   {args.frames} frames in {t_render*1000:.0f} ms  -> {fps:.1f} FPS ({ms_per_frame:.1f} ms/frame)")
    print(f"peak RSS: {rss_peak:.0f} MB  ({rss_peak - rss0:+.0f} MB total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
