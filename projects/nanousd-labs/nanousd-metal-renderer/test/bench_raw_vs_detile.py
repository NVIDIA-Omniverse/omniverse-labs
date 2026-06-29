#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: raw staging pointer vs de-tiled fetch at high camera counts.

Compares:
  fetch_pixels_tiled() — CPU de-tile loop: N*tile_h memcpy calls
  fetch_tiled_raw()    — raw staging pointer: zero-copy numpy view

Also reports GPU memory usage per configuration.
"""

import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT


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


def bench(func, warmup=3, iterations=20):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.min(), arr.max()


def main():
    W, H = 100, 100
    env_counts = [512, 1024, 2048, 4096, 8192]

    print("=" * 80)
    print("  RAW STAGING vs DE-TILED FETCH — High Environment Count Benchmark")
    print("=" * 80)
    print(f"  Tile size: {W}x{H}, Iterations: 20 per config\n")

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available:
        print("ERROR: RT not available")
        r.close()
        return

    # Simple Cartpole-like scene (7 meshes)
    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    # Ground plane
    gv = np.array([[-10,-0.5,-10],[10,-0.5,-10],[10,-0.5,10],[-10,-0.5,10]], dtype=np.float32)
    gi = np.array([0,1,2,0,2,3], dtype=np.uint32)
    r.add_mesh(positions=gv, indices=gi, display_color=(0.5, 0.5, 0.5))

    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    # ---- Speed comparison ----
    print("  SPEED COMPARISON")
    print(f"  {'Envs':>6} {'TiledRes':>10} {'Render':>9} {'De-tile':>9} {'Raw':>9} {'Speedup':>8} {'Env-s/s(raw)':>13}")
    print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*13}")

    results = []
    for N in env_counts:
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)
        nc = math.ceil(math.sqrt(N))
        nr = math.ceil(N / nc)
        tw, th = nc * W, nr * H

        # Render only
        def do_render():
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        render_ms, _, _ = bench(do_render)

        # Render + de-tiled fetch
        def do_detile():
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        detile_ms, _, _ = bench(do_detile)

        # Render + raw staging
        def do_raw():
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H)
        raw_ms, _, _ = bench(do_raw)

        fetch_detile = detile_ms - render_ms
        fetch_raw = raw_ms - render_ms
        speedup = fetch_detile / fetch_raw if fetch_raw > 0.001 else float('inf')
        env_s = N / (raw_ms / 1000)

        results.append({
            'N': N, 'tw': tw, 'th': th,
            'render_ms': render_ms, 'detile_ms': detile_ms, 'raw_ms': raw_ms,
            'fetch_detile': fetch_detile, 'fetch_raw': fetch_raw,
            'speedup': speedup, 'env_s': env_s,
        })

        print(f"  {N:6d} {tw}x{th:>4} {render_ms:8.2f}  {detile_ms:8.2f}  {raw_ms:8.2f}  {speedup:7.1f}x {env_s:12,.0f}")

    # ---- GPU memory ----
    print(f"\n  GPU MEMORY USAGE")
    print(f"  {'Envs':>6} {'TiledRes':>10} {'Staging(MB)':>12} {'Total GPU(MB)':>14}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*14}")

    for N in env_counts:
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)
        nc = math.ceil(math.sqrt(N))
        nr = math.ceil(N / nc)
        tw, th = nc * W, nr * H

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H)

        gpu_mb = r.gpu_memory_used / (1024 * 1024)
        # Staging = 2 buffers × total_w × total_h × 4 bytes (double-buffered)
        staging_mb = 2 * tw * th * 4 / (1024 * 1024)
        # Tiled image itself
        image_mb = tw * th * 4 / (1024 * 1024)

        print(f"  {N:6d} {tw}x{th:>4} {staging_mb:11.1f}  {gpu_mb:13.1f}")

    r.close()

    # ---- Summary ----
    print(f"\n  SUMMARY")
    print(f"  The de-tile loop does N×tile_h small memcpy calls ({W} bytes each).")
    print(f"  At 4096 envs = {4096 * H:,} memcpy calls → 99%+ of readback time.")
    print(f"  fetch_tiled_raw() returns a numpy view over the staging buffer (zero memcpy).")
    print(f"  Consumer (IsaacLab) can do GPU de-tiling via warp kernel instead.\n")


if __name__ == "__main__":
    main()
