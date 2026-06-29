#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile timing breakdown: zero-copy vs CPU readback, with double-buffer overlap."""

import sys
import os
import time
import math
import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
from nusd_renderer._cuda_interop import CudaVulkanInterop


@wp.kernel
def detile(tiled: wp.array3d(dtype=wp.uint8), output: wp.array4d(dtype=wp.uint8),
           nc: int, tw: int, th: int):
    e, y, x = wp.tid()
    col = e % nc
    row = e // nc
    sx = col * tw + x
    sy = row * th + y
    output[e, y, x, 0] = tiled[sy, sx, 0]
    output[e, y, x, 1] = tiled[sy, sx, 1]
    output[e, y, x, 2] = tiled[sy, sx, 2]
    output[e, y, x, 3] = tiled[sy, sx, 3]


def main():
    wp.init()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    W, H = 100, 100

    r = NuRenderer(width=512, height=512, enable_rt=True)
    v = np.array([[-0.3,-0.3,-0.3],[0.3,-0.3,-0.3],[0.3,0.3,-0.3],[-0.3,0.3,-0.3],
                  [-0.3,-0.3,0.3],[0.3,-0.3,0.3],[0.3,0.3,0.3],[-0.3,0.3,0.3]], dtype=np.float32)
    idx = np.array([0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,2,6,7,2,7,3,0,3,7,0,7,4,1,5,6,1,6,2], dtype=np.uint32)
    r.add_mesh(positions=v, indices=idx, display_color=(0.8, 0.2, 0.2))
    gv = np.array([[-10,-0.5,-10],[10,-0.5,-10],[10,-0.5,10],[-10,-0.5,10]], dtype=np.float32)
    gi = np.array([0,1,2,0,2,3], dtype=np.uint32)
    r.add_mesh(positions=gv, indices=gi, display_color=(0.5, 0.5, 0.5))
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    vps = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        vps[i, 0] = 1; vps[i, 5] = 1; vps[i, 10] = -1; vps[i, 11] = 5; vps[i, 15] = 1
        vps[i, 16] = 1; vps[i, 21] = -1; vps[i, 27] = -1; vps[i, 30] = -1; vps[i, 31] = 0

    nc = math.ceil(math.sqrt(N))
    out = wp.zeros((N, H, W, 4), dtype=wp.uint8, device="cuda:0")

    # ---- 1. Zero-copy (non-overlap, skip_staging) ----
    r.render_tiled(vps, N, W, H, NU_RENDER_RT)
    interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H, skip_staging=True)

    for _ in range(10):
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        tiled = interop.wait_and_get_warp_array()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        wp.synchronize()

    print(f"\n{'='*70}")
    print(f"  Timing breakdown at N={N} ({W}x{H})")
    print(f"{'='*70}")

    times_fence = []
    times_warp = []
    times_launch = []
    times_sync = []
    times_total = []

    for _ in range(20):
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        t0 = time.perf_counter()
        r.wait_tiled_complete()
        t1 = time.perf_counter()
        tiled = interop.wait_and_get_warp_array()
        t2 = time.perf_counter()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        t3 = time.perf_counter()
        wp.synchronize()
        t4 = time.perf_counter()

        times_fence.append((t1 - t0) * 1000)
        times_warp.append((t2 - t1) * 1000)
        times_launch.append((t3 - t2) * 1000)
        times_sync.append((t4 - t3) * 1000)
        times_total.append((t4 - t0) * 1000)

    nonoverlap_ms = np.mean(times_total)
    print(f"\n  Non-overlap (wait current frame):")
    print(f"    fence_wait:   {np.mean(times_fence):.3f}ms")
    print(f"    warp_array:   {np.mean(times_warp):.3f}ms")
    print(f"    kernel_launch:{np.mean(times_launch):.3f}ms")
    print(f"    wp.sync:      {np.mean(times_sync):.3f}ms")
    print(f"    TOTAL:        {nonoverlap_ms:.3f}ms")

    # ---- 2. Zero-copy with double-buffer OVERLAP ----
    # Warmup overlap path
    for _ in range(10):
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        tiled = interop.overlap_get_warp_array()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        wp.synchronize()

    times_fence2 = []
    times_warp2 = []
    times_launch2 = []
    times_sync2 = []
    times_total2 = []

    for _ in range(20):
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        t0 = time.perf_counter()
        r.wait_previous_tiled_complete()
        t1 = time.perf_counter()
        tiled = interop.overlap_get_warp_array()
        t2 = time.perf_counter()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        t3 = time.perf_counter()
        wp.synchronize()
        t4 = time.perf_counter()

        times_fence2.append((t1 - t0) * 1000)
        times_warp2.append((t2 - t1) * 1000)
        times_launch2.append((t3 - t2) * 1000)
        times_sync2.append((t4 - t3) * 1000)
        times_total2.append((t4 - t0) * 1000)

    overlap_ms = np.mean(times_total2)
    print(f"\n  Double-buffer OVERLAP (read previous frame):")
    print(f"    prev_fence:   {np.mean(times_fence2):.3f}ms")
    print(f"    warp_array:   {np.mean(times_warp2):.3f}ms")
    print(f"    kernel_launch:{np.mean(times_launch2):.3f}ms")
    print(f"    wp.sync:      {np.mean(times_sync2):.3f}ms")
    print(f"    TOTAL:        {overlap_ms:.3f}ms")

    # ---- 3. End-to-end step time (render + read + detile) ----
    print(f"\n  End-to-end step time (render + read + detile + sync):")

    # Non-overlap
    times_step = []
    for _ in range(20):
        t0 = time.perf_counter()
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        tiled = interop.wait_and_get_warp_array()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        wp.synchronize()
        t1 = time.perf_counter()
        times_step.append((t1 - t0) * 1000)
    step_nonoverlap = np.mean(times_step)

    # Overlap (includes render in the loop)
    times_step2 = []
    # Seed: submit first frame
    r.render_tiled(vps, N, W, H, NU_RENDER_RT)
    for _ in range(20):
        t0 = time.perf_counter()
        tiled = interop.overlap_get_warp_array()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        wp.synchronize()
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        t1 = time.perf_counter()
        times_step2.append((t1 - t0) * 1000)
    # Drain last frame
    interop.wait_and_get_warp_array()
    step_overlap = np.mean(times_step2)

    # Overlap WITHOUT wp.synchronize (simulates RL pipeline where downstream
    # warp kernels implicitly serialize on the same CUDA stream)
    times_step3 = []
    r.render_tiled(vps, N, W, H, NU_RENDER_RT)
    for _ in range(20):
        t0 = time.perf_counter()
        tiled = interop.overlap_get_warp_array()
        wp.launch(detile, dim=(N, H, W), inputs=[tiled, out, nc, W, H], device="cuda:0")
        # No wp.synchronize() — downstream kernels serialize on same stream
        r.render_tiled(vps, N, W, H, NU_RENDER_RT)
        t1 = time.perf_counter()
        times_step3.append((t1 - t0) * 1000)
    wp.synchronize()  # drain
    interop.wait_and_get_warp_array()
    step_nosync = np.mean(times_step3)

    print(f"    Non-overlap:         {step_nonoverlap:.3f}ms  ({N / step_nonoverlap * 1000:.0f} env-steps/s)")
    print(f"    Overlap+sync:        {step_overlap:.3f}ms  ({N / step_overlap * 1000:.0f} env-steps/s)")
    print(f"    Overlap (no sync):   {step_nosync:.3f}ms  ({N / step_nosync * 1000:.0f} env-steps/s)")
    print(f"    Speedup (overlap):   {step_nonoverlap / step_overlap:.2f}x")
    print(f"    Speedup (no sync):   {step_nonoverlap / step_nosync:.2f}x")

    interop.close()
    r.close()


if __name__ == "__main__":
    main()
