#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tiled rendering benchmark — produces a full performance report.

Measures:
  1. Render-only vs render+fetch latency breakdown
  2. Camera count scaling (1 to 256)
  3. Resolution scaling (16x16 to 512x512)
  4. Kitchen_set scene complexity impact
  5. TLAS rebuild overhead
  6. Memory usage tracking
  7. Sustained throughput over 500 frames
"""

import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT


# ---- Helpers ----

def make_view_matrix(eye, target):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 1, 0], dtype=np.float32)
    r = np.cross(f, up)
    r_len = np.linalg.norm(r)
    if r_len < 1e-6:
        up = np.array([0, 0, 1], dtype=np.float32)
        r = np.cross(f, up)
        r_len = np.linalg.norm(r)
    r /= r_len
    u = np.cross(r, f)
    view = np.zeros(16, dtype=np.float32)
    view[0] = r[0];  view[1] = r[1];  view[2]  = r[2];  view[3]  = -np.dot(r, eye)
    view[4] = u[0];  view[5] = u[1];  view[6]  = u[2];  view[7]  = -np.dot(u, eye)
    view[8] = -f[0]; view[9] = -f[1]; view[10] = -f[2]; view[11] = np.dot(f, eye)
    view[15] = 1.0
    return view

def invert_view(view):
    vi = np.zeros(16, dtype=np.float32)
    vi[0] = view[0]; vi[1] = view[4]; vi[2]  = view[8]
    vi[4] = view[1]; vi[5] = view[5]; vi[6]  = view[9]
    vi[8] = view[2]; vi[9] = view[6]; vi[10] = view[10]
    vi[3]  = -(vi[0]*view[3]  + vi[1]*view[7]  + vi[2]*view[11])
    vi[7]  = -(vi[4]*view[3]  + vi[5]*view[7]  + vi[6]*view[11])
    vi[11] = -(vi[8]*view[3]  + vi[9]*view[7]  + vi[10]*view[11])
    vi[15] = 1.0
    return vi

def make_proj_matrix(fov_deg, aspect, near=0.01, far=10000.0):
    fov = math.radians(fov_deg)
    t = math.tan(fov * 0.5)
    proj = np.zeros(16, dtype=np.float32)
    proj[0]  = 1.0 / (aspect * t)
    proj[5]  = -1.0 / t
    proj[10] = far / (near - far)
    proj[11] = -(far * near) / (far - near)
    proj[14] = -1.0
    return proj

def invert_proj(proj):
    pi = np.zeros(16, dtype=np.float32)
    pi[0]  = 1.0 / proj[0]
    pi[5]  = 1.0 / proj[5]
    pi[11] = 1.0 / proj[14]
    pi[14] = -1.0
    pi[15] = proj[10] / proj[14]
    return pi

def make_vp_inv(eye, target, fov_deg, w, h):
    view = make_view_matrix(eye, target)
    vi = invert_view(view)
    proj = make_proj_matrix(fov_deg, w / h)
    pi = invert_proj(proj)
    return np.concatenate([vi, pi])

def make_box(cx=0, cy=0, cz=0, hx=0.3, hy=0.3, hz=0.3):
    verts = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz], [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz], [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ], dtype=np.float32)
    idx = np.array([
        0,1,2, 0,2,3, 4,6,5, 4,7,6,
        0,4,5, 0,5,1, 2,6,7, 2,7,3,
        0,3,7, 0,7,4, 1,5,6, 1,6,2,
    ], dtype=np.uint32)
    return verts, idx

def orbit_vps(N, radius, center, fov, w, h, y_offset=0.5):
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
    return np.stack([
        make_vp_inv(
            (radius * math.cos(a), y_offset, radius * math.sin(a)),
            center, fov, w, h)
        for a in angles
    ])


def bench_func(func, warmup=5, iterations=50):
    """Benchmark a function, return (mean_ms, std_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std(), arr.min(), arr.max()


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ---- Benchmarks ----

def bench_render_vs_fetch(r, w, h, N):
    """Measure render-only vs render+fetch latency."""
    print_header(f"Render vs Fetch Breakdown ({N} cameras @ {w}x{h})")

    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)

    # Render only
    def render_only():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)

    mean_r, std_r, min_r, max_r = bench_func(render_only)

    # Render + fetch
    def render_and_fetch():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_rf, std_rf, min_rf, max_rf = bench_func(render_and_fetch)

    fetch_ms = mean_rf - mean_r
    total_pixels = N * w * h
    mpix_s = total_pixels / (mean_rf / 1000) / 1e6

    print(f"  Render only:   {mean_r:7.2f} ms (std {std_r:.2f}, min {min_r:.2f}, max {max_r:.2f})")
    print(f"  Render+fetch:  {mean_rf:7.2f} ms (std {std_rf:.2f}, min {min_rf:.2f}, max {max_rf:.2f})")
    print(f"  Fetch overhead: {fetch_ms:6.2f} ms ({100*fetch_ms/mean_rf:.0f}%)")
    print(f"  Throughput:    {mpix_s:7.1f} Mpix/s ({total_pixels/1e3:.0f}K pixels)")
    return mean_r, mean_rf, fetch_ms


def bench_camera_scaling(r, w, h):
    """Sweep camera count from 1 to 256."""
    print_header(f"Camera Count Scaling ({w}x{h} tiles)")
    print(f"  {'N':>5} {'Render ms':>10} {'R+Fetch ms':>11} {'Fetch ms':>9} {'FPS':>7} {'Mpix/s':>8} {'us/cam':>8}")
    print(f"  {'-'*5} {'-'*10} {'-'*11} {'-'*9} {'-'*7} {'-'*8} {'-'*8}")

    for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)

        def render():
            r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)

        def render_fetch():
            r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

        mean_r, _, _, _ = bench_func(render, warmup=3, iterations=20)
        mean_rf, _, _, _ = bench_func(render_fetch, warmup=3, iterations=20)
        fetch_ms = mean_rf - mean_r
        fps = 1000 / mean_rf
        mpix = N * w * h / (mean_rf / 1000) / 1e6
        us_per_cam = mean_rf / N * 1000

        print(f"  {N:5d} {mean_r:10.2f} {mean_rf:11.2f} {fetch_ms:9.2f} {fps:7.1f} {mpix:8.1f} {us_per_cam:8.1f}")


def bench_resolution_scaling(r, N):
    """Sweep resolution at fixed camera count."""
    print_header(f"Resolution Scaling ({N} cameras)")
    print(f"  {'WxH':>10} {'Pixels':>10} {'R+Fetch ms':>11} {'FPS':>7} {'Mpix/s':>8} {'Fetch%':>7}")
    print(f"  {'-'*10} {'-'*10} {'-'*11} {'-'*7} {'-'*8} {'-'*7}")

    for size in [16, 32, 48, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512]:
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, size, size)

        def render():
            r.render_tiled(vps, num_cameras=N, tile_w=size, tile_h=size, mode=NU_RENDER_RT)

        def render_fetch():
            r.render_tiled(vps, num_cameras=N, tile_w=size, tile_h=size, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=size, tile_h=size)

        mean_r, _, _, _ = bench_func(render, warmup=3, iterations=20)
        mean_rf, _, _, _ = bench_func(render_fetch, warmup=3, iterations=20)
        fetch_ms = mean_rf - mean_r
        total_pix = N * size * size
        fps = 1000 / mean_rf
        mpix = total_pix / (mean_rf / 1000) / 1e6
        fetch_pct = 100 * fetch_ms / mean_rf if mean_rf > 0 else 0

        print(f"  {size:4d}x{size:<4d} {total_pix:10d} {mean_rf:11.2f} {fps:7.1f} {mpix:8.1f} {fetch_pct:6.0f}%")


def bench_sustained(r, w, h, N, frames=500):
    """Sustained throughput over many frames."""
    print_header(f"Sustained Throughput ({N} cameras @ {w}x{h}, {frames} frames)")

    vps_base = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)

    # Warmup
    for _ in range(10):
        r.render_tiled(vps_base, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    times = []
    for i in range(frames):
        t0 = time.perf_counter()
        r.render_tiled(vps_base, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    total_pix = N * w * h

    # Percentiles
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    print(f"  Mean:   {arr.mean():.2f} ms ({1000/arr.mean():.0f} FPS)")
    print(f"  Median: {p50:.2f} ms")
    print(f"  P95:    {p95:.2f} ms")
    print(f"  P99:    {p99:.2f} ms")
    print(f"  Min:    {arr.min():.2f} ms")
    print(f"  Max:    {arr.max():.2f} ms")
    print(f"  Std:    {arr.std():.2f} ms")
    print(f"  Mpix/s: {total_pix / (arr.mean()/1000) / 1e6:.1f}")

    # Check for stalls
    stall_threshold = arr.mean() * 3
    stalls = (arr > stall_threshold).sum()
    print(f"  Stalls (>3x mean): {stalls}/{frames}")


def bench_tlas_rebuild(r, w, h, N):
    """Measure TLAS rebuild overhead during tiled rendering."""
    print_header(f"TLAS Rebuild Overhead ({N} cameras @ {w}x{h})")

    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)

    # Without TLAS rebuild
    def no_rebuild():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_no, _, _, _ = bench_func(no_rebuild, warmup=5, iterations=30)

    # With TLAS rebuild (move mesh each frame)
    xform_base = np.eye(4, dtype=np.float32).flatten()

    def with_rebuild():
        xform = xform_base.copy()
        xform[3] += np.random.uniform(-0.1, 0.1)
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_rb, _, _, _ = bench_func(with_rebuild, warmup=5, iterations=30)

    overhead = mean_rb - mean_no
    print(f"  Without TLAS rebuild: {mean_no:.2f} ms")
    print(f"  With TLAS rebuild:    {mean_rb:.2f} ms")
    print(f"  Rebuild overhead:     {overhead:.2f} ms ({100*overhead/mean_no:.0f}%)")


def bench_kitchen_set(w, h, N):
    """Benchmark Kitchen_set tiled rendering."""
    kitchen_path = os.environ.get("NUSD_KITCHEN_SET", os.path.expanduser("~/Kitchen_set/Kitchen_set.usd"))
    if not os.path.exists(kitchen_path):
        print("\n  [SKIP] Kitchen_set not found")
        return

    print_header(f"Kitchen_set ({N} cameras @ {w}x{h})")

    r = NuRenderer(width=max(w, 256), height=max(h, 256), enable_rt=True)
    if not r.rt_available:
        r.close()
        print("  [SKIP] RT not available")
        return

    t0 = time.perf_counter()
    try:
        nmeshes = r.load_usd(kitchen_path)
    except RuntimeError:
        r.close()
        print("  [SKIP] USD backend not available")
        return
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Loaded {nmeshes} meshes in {load_ms:.0f} ms")

    # Initial render to build BLAS/TLAS
    r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
    t0 = time.perf_counter()
    r.render(mode=NU_RENDER_RT)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  BLAS/TLAS build: {build_ms:.0f} ms")

    vps = orbit_vps(N, 200, (70, 50, 30), 60.0, w, h, y_offset=120)

    # Warmup
    for _ in range(5):
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    # Measure render only
    def render_only():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)

    def render_fetch():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_r, _, _, _ = bench_func(render_only, warmup=3, iterations=30)
    mean_rf, std_rf, min_rf, max_rf = bench_func(render_fetch, warmup=3, iterations=30)
    fetch_ms = mean_rf - mean_r
    total_pix = N * w * h
    mpix = total_pix / (mean_rf / 1000) / 1e6

    print(f"\n  Results:")
    print(f"    Render only:   {mean_r:.2f} ms")
    print(f"    Render+fetch:  {mean_rf:.2f} ms (std {std_rf:.2f})")
    print(f"    Fetch:         {fetch_ms:.2f} ms ({100*fetch_ms/mean_rf:.0f}%)")
    print(f"    FPS:           {1000/mean_rf:.0f}")
    print(f"    Mpix/s:        {mpix:.1f}")
    print(f"    GPU memory:    {r.gpu_memory_used / (1024*1024):.0f} MB")

    # Resolution sweep on kitchen
    print(f"\n  Kitchen resolution sweep ({N} cameras):")
    print(f"  {'WxH':>10} {'ms':>8} {'FPS':>7} {'Mpix/s':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
    for sz in [32, 64, 100, 128, 200, 256]:
        vps_sz = orbit_vps(N, 200, (70, 50, 30), 60.0, sz, sz, y_offset=120)
        def rf(s=sz, v=vps_sz):
            r.render_tiled(v, num_cameras=N, tile_w=s, tile_h=s, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=s, tile_h=s)
        m, _, _, _ = bench_func(rf, warmup=3, iterations=15)
        tp = N * sz * sz
        print(f"  {sz:4d}x{sz:<4d} {m:8.2f} {1000/m:7.0f} {tp/(m/1000)/1e6:8.1f}")

    r.close()


def main():
    print("="*70)
    print("  TILED RENDERING COMPREHENSIVE BENCHMARK")
    print("="*70)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Simple scene
    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available:
        print("ERROR: RT not available")
        r.close()
        return

    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    print(f"  GPU memory: {r.gpu_memory_used / (1024*1024):.0f} MB")

    # 1. Render vs fetch breakdown
    bench_render_vs_fetch(r, 100, 100, 32)

    # 2. Camera count scaling
    bench_camera_scaling(r, 64, 64)

    # 3. Resolution scaling
    bench_resolution_scaling(r, 16)

    # 4. TLAS rebuild overhead
    bench_tlas_rebuild(r, 100, 100, 16)

    # 5. Sustained throughput
    bench_sustained(r, 100, 100, 32, frames=500)

    r.close()

    # 6. Kitchen_set benchmarks
    bench_kitchen_set(100, 100, 32)
    bench_kitchen_set(64, 64, 64)

    print_header("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
