# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for nu_render_async + nu_fetch_async vs sync render+fetch.

Measures steady-state per-frame ms on /tmp/grid_tera.usdc (12.3 M curves)
at 1280x720. Target: ≤16 ms median (≥ 60 fps).
"""

import os
import sys
import time
import numpy as np

USD_PATH = os.environ.get("BENCH_USD", "/tmp/grid_tera.usdc")
W = int(os.environ.get("BENCH_W", "1280"))
H = int(os.environ.get("BENCH_H", "720"))
N = int(os.environ.get("BENCH_N", "35"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))


def bench():
    from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

    print(f"USD = {USD_PATH}")
    print(f"size = {W}x{H}, frames = {N}, warmup = {WARMUP}")

    r = NuRenderer(width=W, height=H, enable_rt=True)

    nloaded = r.load_usd(USD_PATH)
    print(f"loaded {nloaded} prims, mesh_count = {r.mesh_count}, "
          f"curve_segments = {r._lib.nu_get_curve_segment_count(r._handle)}")

    r.set_camera(eye=(15.0, 15.0, 15.0), target=(0.0, 0.0, 0.0),
                 fov_degrees=45.0, near_clip=0.1, far_clip=1000.0)
    r.build_accel()

    # ---- Sync baseline: nu_render then nu_fetch_pixels ----
    print("\n[sync] render + fetch_pixels")
    sync_times = []
    for i in range(N):
        t0 = time.perf_counter()
        r.render(mode=NU_RENDER_RT)
        buf = r.fetch_pixels()
        sync_times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(sync_times[WARMUP:])
    print(f"  median = {np.median(arr):.2f} ms,  p10 = {np.percentile(arr, 10):.2f},"
          f"  p90 = {np.percentile(arr, 90):.2f}")

    # ---- Async: render_async + fetch_async ----
    print("\n[async] render_async + fetch_async")
    out = np.zeros((H, W, 4), dtype=np.uint8)
    async_times = []
    for i in range(N):
        t0 = time.perf_counter()
        r.render_async()
        r.fetch_async(out)
        async_times.append((time.perf_counter() - t0) * 1000)
    arr2 = np.array(async_times[WARMUP:])
    print(f"  median = {np.median(arr2):.2f} ms,  p10 = {np.percentile(arr2, 10):.2f},"
          f"  p90 = {np.percentile(arr2, 90):.2f}")

    print(f"\n  speedup (sync / async) = {np.median(arr) / np.median(arr2):.1f}x")
    print(f"  async fps              = {1000.0 / np.median(arr2):.1f}")

    # Sanity: output buffer should be non-zero after warmup (frames > 1).
    print(f"  out_pixels nonzero = {bool(np.any(out > 0))} "
          f"(min={out.min()} max={out.max()})")

    # Print per-frame for first 10 frames so the warm-up shape is visible.
    print(f"\n[async] first 10 frame times (ms): "
          f"{['%.2f' % t for t in async_times[:10]]}")
    print(f"[sync]  first 10 frame times (ms): "
          f"{['%.2f' % t for t in sync_times[:10]]}")

    r.close()
    target_ms = 16.0
    median_async = float(np.median(arr2))
    print(f"\n=== TARGET ≤ {target_ms} ms — got {median_async:.2f} ms — "
          f"{'PASS' if median_async <= target_ms else 'FAIL'}")
    return median_async <= target_ms


if __name__ == "__main__":
    ok = bench()
    sys.exit(0 if ok else 1)
