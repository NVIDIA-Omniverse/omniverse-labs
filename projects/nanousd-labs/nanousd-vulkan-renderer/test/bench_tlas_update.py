#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLAS update-in-place benchmark.

Measures the performance improvement of TLAS MODE_UPDATE vs the previous
full MODE_BUILD approach. Tests with both simple scenes and Kitchen_set.

Key metrics:
  - TLAS update latency (isolated)
  - Full frame time with TLAS update (render + update + fetch)
  - Sustained throughput over many frames with animated transforms
  - Comparison: update overhead as % of frame time
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
        0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1, 2,6,7, 2,7,3, 0,3,7, 0,7,4, 1,5,6, 1,6,2,
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


def bench_tlas_update_simple():
    """TLAS update benchmark with simple multi-box scene."""
    print_header("TLAS Update: Simple Scene (3 boxes)")

    r = NuRenderer(width=256, height=256, enable_rt=True)
    if not r.rt_available:
        r.close()
        print("  [SKIP] RT not available")
        return

    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    N, w, h = 16, 100, 100
    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)
    xform_base = np.eye(4, dtype=np.float32).flatten()

    # No TLAS update
    def no_update():
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_no, std_no, _, _ = bench_func(no_update, warmup=5, iterations=50)

    # With TLAS update (move mesh each frame)
    frame = [0]
    def with_update():
        frame[0] += 1
        t = frame[0] * 0.02
        xform = xform_base.copy()
        xform[3] = math.sin(t) * 0.5  # oscillate X
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_up, std_up, _, _ = bench_func(with_update, warmup=5, iterations=50)

    # TLAS update only (isolated)
    def tlas_only():
        frame[0] += 1
        t = frame[0] * 0.02
        xform = xform_base.copy()
        xform[3] = math.sin(t) * 0.5
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))

    mean_tlas, std_tlas, min_tlas, max_tlas = bench_func(tlas_only, warmup=5, iterations=100)

    overhead = mean_up - mean_no
    print(f"  Without TLAS update:  {mean_no:.3f} ms (std {std_no:.3f})")
    print(f"  With TLAS update:     {mean_up:.3f} ms (std {std_up:.3f})")
    print(f"  Update overhead:      {overhead:.3f} ms ({100*overhead/mean_no:.1f}%)")
    print(f"  TLAS update isolated: {mean_tlas:.3f} ms (min {min_tlas:.3f}, max {max_tlas:.3f})")
    print(f"  FPS (with update):    {1000/mean_up:.0f}")

    # Sustained 1000-frame animation
    print(f"\n  Sustained 1000-frame animation:")
    times = []
    for i in range(1000):
        t = i * 0.02
        xform = xform_base.copy()
        xform[3] = math.sin(t) * 0.5
        xform[7] = math.cos(t * 0.7) * 0.3

        t0 = time.perf_counter()
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"    Mean:   {arr.mean():.3f} ms ({1000/arr.mean():.0f} FPS)")
    print(f"    Median: {np.median(arr):.3f} ms")
    print(f"    P95:    {np.percentile(arr, 95):.3f} ms")
    print(f"    P99:    {np.percentile(arr, 99):.3f} ms")
    print(f"    Min:    {arr.min():.3f} ms")
    print(f"    Max:    {arr.max():.3f} ms")
    print(f"    Stalls: {np.sum(arr > 3 * arr.mean())}/1000")

    r.close()


def bench_tlas_update_kitchen():
    """TLAS update benchmark with Kitchen_set (1788 instances)."""
    kitchen_path = os.environ.get("NUSD_KITCHEN_SET", os.path.expanduser("~/Kitchen_set/Kitchen_set.usd"))
    if not os.path.exists(kitchen_path):
        print("\n  [SKIP] Kitchen_set not found")
        return

    print_header("TLAS Update: Kitchen_set (1788 instances)")

    r = NuRenderer(width=256, height=256, enable_rt=True)
    if not r.rt_available:
        r.close()
        print("  [SKIP] RT not available")
        return

    try:
        nmeshes = r.load_usd(kitchen_path)
    except RuntimeError:
        r.close()
        print("  [SKIP] USD backend not available")
        return
    print(f"  Loaded {nmeshes} meshes")

    r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
    r.render(mode=NU_RENDER_RT)

    # Test configs: different camera counts and resolutions
    configs = [
        (16, 100, 100),
        (32, 100, 100),
        (64, 64, 64),
        (32, 200, 200),
    ]

    for N, w, h in configs:
        print(f"\n  --- {N} cameras @ {w}x{h} ---")
        vps = orbit_vps(N, 200, (70, 50, 30), 60.0, w, h, y_offset=120)

        # No TLAS update
        def no_update():
            r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

        mean_no, std_no, _, _ = bench_func(no_update, warmup=3, iterations=30)

        # With TLAS update: move first 10 meshes
        xform_base = np.eye(4, dtype=np.float32).flatten()
        mesh_ids = list(range(min(10, nmeshes)))
        frame = [0]

        def with_update():
            frame[0] += 1
            t = frame[0] * 0.02
            xforms = np.tile(xform_base, (len(mesh_ids), 1))
            for j in range(len(mesh_ids)):
                xforms[j, 3] += math.sin(t + j * 0.5) * 5.0  # oscillate X
                xforms[j, 7] += math.cos(t + j * 0.3) * 2.0  # oscillate Y
            r.set_transforms(mesh_ids=mesh_ids, transforms=xforms)
            r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

        mean_up, std_up, _, _ = bench_func(with_update, warmup=3, iterations=30)

        # TLAS update only (isolated)
        def tlas_only():
            frame[0] += 1
            t = frame[0] * 0.02
            xforms = np.tile(xform_base, (len(mesh_ids), 1))
            for j in range(len(mesh_ids)):
                xforms[j, 3] += math.sin(t + j * 0.5) * 5.0
            r.set_transforms(mesh_ids=mesh_ids, transforms=xforms)

        mean_tlas, _, min_tlas, max_tlas = bench_func(tlas_only, warmup=3, iterations=50)

        overhead = mean_up - mean_no
        total_pix = N * w * h
        print(f"    Without update: {mean_no:.3f} ms ({1000/mean_no:.0f} FPS)")
        print(f"    With update:    {mean_up:.3f} ms ({1000/mean_up:.0f} FPS)")
        print(f"    Overhead:       {overhead:.3f} ms ({100*overhead/mean_no:.1f}%)")
        print(f"    TLAS isolated:  {mean_tlas:.3f} ms (min {min_tlas:.3f}, max {max_tlas:.3f})")
        print(f"    Mpix/s:         {total_pix / (mean_up / 1000) / 1e6:.1f}")

    # All-meshes update: move EVERY mesh (worst case for 1788 instances)
    print(f"\n  --- ALL 1788 meshes moving (worst case) ---")
    N, w, h = 32, 100, 100
    vps = orbit_vps(N, 200, (70, 50, 30), 60.0, w, h, y_offset=120)
    all_ids = list(range(nmeshes))
    xform_all = np.tile(xform_base, (nmeshes, 1))
    frame = [0]

    def all_update():
        frame[0] += 1
        t = frame[0] * 0.01
        for j in range(nmeshes):
            xform_all[j, 3] = math.sin(t + j * 0.01) * 2.0
        r.set_transforms(mesh_ids=all_ids, transforms=xform_all)
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    mean_all, std_all, _, _ = bench_func(all_update, warmup=3, iterations=20)

    def all_tlas_only():
        frame[0] += 1
        t = frame[0] * 0.01
        for j in range(nmeshes):
            xform_all[j, 3] = math.sin(t + j * 0.01) * 2.0
        r.set_transforms(mesh_ids=all_ids, transforms=xform_all)

    mean_all_tlas, _, _, _ = bench_func(all_tlas_only, warmup=3, iterations=20)

    print(f"    Full frame:     {mean_all:.3f} ms ({1000/mean_all:.0f} FPS)")
    print(f"    TLAS isolated:  {mean_all_tlas:.3f} ms")
    print(f"    TLAS % of frame: {100*mean_all_tlas/mean_all:.1f}%")

    # Sustained 500-frame animation with 10 moving meshes
    print(f"\n  Sustained 500-frame animation (10 meshes moving):")
    N, w, h = 32, 100, 100
    vps = orbit_vps(N, 200, (70, 50, 30), 60.0, w, h, y_offset=120)
    mesh_ids = list(range(10))
    xforms = np.tile(xform_base, (10, 1))
    times = []

    for i in range(500):
        t = i * 0.02
        for j in range(10):
            xforms[j, 3] = math.sin(t + j * 0.5) * 5.0
            xforms[j, 7] = math.cos(t + j * 0.3) * 2.0

        t0 = time.perf_counter()
        r.set_transforms(mesh_ids=mesh_ids, transforms=xforms)
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"    Mean:   {arr.mean():.3f} ms ({1000/arr.mean():.0f} FPS)")
    print(f"    Median: {np.median(arr):.3f} ms")
    print(f"    P95:    {np.percentile(arr, 95):.3f} ms")
    print(f"    P99:    {np.percentile(arr, 99):.3f} ms")
    print(f"    Min:    {arr.min():.3f} ms")
    print(f"    Max:    {arr.max():.3f} ms")
    print(f"    Stalls: {np.sum(arr > 3 * arr.mean())}/500")

    r.close()


def main():
    print("="*70)
    print("  TLAS UPDATE-IN-PLACE BENCHMARK")
    print("="*70)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR")
    print(f"  Persistent buffers: scratch + staging (zero alloc per frame)")

    bench_tlas_update_simple()
    bench_tlas_update_kitchen()

    print_header("BENCHMARK COMPLETE")
    print(f"\n  Previous baseline (full rebuild): ~0.84 ms per TLAS rebuild (Kitchen_set)")
    print(f"  Compare TLAS isolated times above to see improvement.\n")


if __name__ == "__main__":
    main()
