#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bench_no_swizzle.py — measure CPU readback time before/after the
per-pixel BGRA→RGBA swizzle is removed.

Compares two paths over the same renderer instance:
  baseline: nu_fetch_pixels(NU_PIXEL_RGBA8) — does CPU swizzle
  new:      nu_fetch_pixels(NU_PIXEL_BGRA8) — pure memcpy

Verifies pixel-byte-identity:
  baseline_rgba[..., [2, 1, 0, 3]] == new_bgra

Usage:
    python test/bench_no_swizzle.py /tmp/grid_tera.usdc
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "python"))

os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
os.environ.setdefault(
    "NUSD_RENDERER_LIB",
    os.path.join(_REPO, "build", "libnusd_renderer.so"),
)

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT  # noqa: E402

# Pixel format constants — must match enum NuPixelFormat in nusd_renderer.h
NU_PIXEL_RGBA8 = 0
NU_PIXEL_BGRA8 = 2


def _percentiles(arr_ms):
    a = np.asarray(arr_ms)
    return {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def main(usd_path: str):
    W, H = 1280, 720

    print(f"USD: {usd_path}", flush=True)
    r = NuRenderer(width=W, height=H, enable_rt=True, enable_materials=True)
    n = r.load_usd(usd_path)
    print(f"loaded {n} meshes", flush=True)

    r.set_camera(
        eye=(94.6, 35.0, -97.0),
        target=(40.0, 9.7, 41.9),
        fov_degrees=42.0,
        near_clip=0.05,
        far_clip=4000.0,
    )
    r.build_accel()

    # Pre-allocate the output buffers used by both paths.
    rgba_buf = np.zeros((H, W, 4), dtype=np.uint8)
    bgra_buf = np.zeros((H, W, 4), dtype=np.uint8)

    # ---- Pixel-identity check ----
    r.render(NU_RENDER_RT)

    res = r._lib.nu_fetch_pixels(
        r._handle, rgba_buf.ctypes.data_as(ctypes.c_void_p), NU_PIXEL_RGBA8
    )
    assert res == 0, f"RGBA fetch failed: {res}"
    res = r._lib.nu_fetch_pixels(
        r._handle, bgra_buf.ctypes.data_as(ctypes.c_void_p), NU_PIXEL_BGRA8
    )
    assert res == 0, f"BGRA fetch failed: {res}"

    expected_bgra = rgba_buf[..., [2, 1, 0, 3]]
    if np.array_equal(expected_bgra, bgra_buf):
        print("PIXEL IDENTITY: PASS (BGRA matches RGBA[..., [2,1,0,3]])")
    else:
        diff = np.abs(expected_bgra.astype(np.int16) - bgra_buf.astype(np.int16)).max()
        print(f"PIXEL IDENTITY: FAIL (max abs diff = {diff})")
        sys.exit(1)

    # ---- Bench: rerender each frame so the swapchain content is refreshed ----
    WARMUP = 5
    STEADY = 30

    rgba_ms = []
    bgra_ms = []

    # Interleave to keep cache state similar between paths.
    for i in range(WARMUP + STEADY):
        # Baseline (RGBA8 with CPU swizzle)
        r.render(NU_RENDER_RT)
        t0 = time.perf_counter()
        r._lib.nu_fetch_pixels(
            r._handle, rgba_buf.ctypes.data_as(ctypes.c_void_p), NU_PIXEL_RGBA8
        )
        t1 = time.perf_counter()
        rgba_ms.append((t1 - t0) * 1000.0)

        # New (BGRA8, memcpy only)
        r.render(NU_RENDER_RT)
        t0 = time.perf_counter()
        r._lib.nu_fetch_pixels(
            r._handle, bgra_buf.ctypes.data_as(ctypes.c_void_p), NU_PIXEL_BGRA8
        )
        t1 = time.perf_counter()
        bgra_ms.append((t1 - t0) * 1000.0)

    rgba_steady = _percentiles(rgba_ms[WARMUP:])
    bgra_steady = _percentiles(bgra_ms[WARMUP:])

    print(f"\nResolution: {W}x{H}, steady N={STEADY}")
    print(f"RGBA8 (swizzle):  median={rgba_steady['median']:.3f} ms  "
          f"min={rgba_steady['min']:.3f}  p90={rgba_steady['p90']:.3f}  "
          f"p99={rgba_steady['p99']:.3f}  max={rgba_steady['max']:.3f}")
    print(f"BGRA8 (memcpy):   median={bgra_steady['median']:.3f} ms  "
          f"min={bgra_steady['min']:.3f}  p90={bgra_steady['p90']:.3f}  "
          f"p99={bgra_steady['p99']:.3f}  max={bgra_steady['max']:.3f}")

    speedup = rgba_steady["median"] / bgra_steady["median"]
    delta = rgba_steady["median"] - bgra_steady["median"]
    print(f"\nspeedup: {speedup:.2f}x   delta: {delta:.3f} ms saved per fetch")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python test/bench_no_swizzle.py <usd_path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
