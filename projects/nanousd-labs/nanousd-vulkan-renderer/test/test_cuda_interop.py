#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test CUDA-Vulkan interop: validate CudaVulkanInterop class end-to-end.

Creates a simple scene, renders via tiled RT, imports the GPU buffer into CUDA,
and verifies pixels match the CPU readback path.
"""

import math
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


def test_interop_basic():
    """Test 1: Create interop, import into CUDA, verify pixels match CPU path."""
    print("=" * 70)
    print("  TEST 1: CudaVulkanInterop basic validation")
    print("=" * 70)

    N = 16
    W, H = 100, 100

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available:
        print("  SKIP: RT not available")
        r.close()
        return False

    # Build a scene
    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    gv = np.array([[-10,-0.5,-10],[10,-0.5,-10],[10,-0.5,10],[-10,-0.5,10]], dtype=np.float32)
    gi = np.array([0,1,2,0,2,3], dtype=np.uint32)
    r.add_mesh(positions=gv, indices=gi, display_color=(0.5, 0.5, 0.5))

    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    # Check interop availability
    print(f"  interop_available: {r.interop_available}")
    if not r.interop_available:
        print("  SKIP: Interop not available (missing VK_KHR_external_memory_fd)")
        r.close()
        return False

    # Do a tiled render first (so interop resources are created)
    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

    # Get CPU reference via raw staging
    cpu_raw = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()
    print(f"  CPU raw shape: {cpu_raw.shape}, first 8 bytes: {cpu_raw.ravel()[:8]}")

    # Now create the interop object
    from nusd_renderer._cuda_interop import CudaVulkanInterop
    interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H)
    print(f"  Interop created: image={interop.image_w}x{interop.image_h}")

    # Render again (this signals the timeline semaphore for CUDA)
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

    # Read via CUDA DtoH
    cuda_pixels = interop.wait_and_copy_to_host()
    print(f"  CUDA pixels shape: {cuda_pixels.shape}, first 8 bytes: {cuda_pixels.ravel()[:8]}")

    # Also get CPU reference for this frame
    cpu_raw2 = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()
    print(f"  CPU raw2 shape: {cpu_raw2.shape}, first 8 bytes: {cpu_raw2.ravel()[:8]}")

    # Compare: CUDA and CPU should both have the same tiled image
    # (same frame, same data, just different readback paths)
    match = np.array_equal(cuda_pixels, cpu_raw2)
    nonzero = np.count_nonzero(cuda_pixels)
    total = cuda_pixels.size

    if match:
        print(f"  PASS: CUDA pixels match CPU raw ({nonzero}/{total} nonzero)")
    else:
        diff = np.abs(cuda_pixels.astype(int) - cpu_raw2.astype(int))
        max_diff = diff.max()
        mean_diff = diff.mean()
        mismatch_pct = (diff > 0).sum() / diff.size * 100
        print(f"  WARN: Pixel mismatch — max_diff={max_diff}, mean={mean_diff:.2f}, "
              f"{mismatch_pct:.1f}% differ")
        if max_diff <= 1:
            print(f"  PASS (within rounding tolerance)")
        else:
            print(f"  FAIL: max diff {max_diff} exceeds tolerance")

    interop.close()
    r.close()
    return True


def test_interop_multi_frame():
    """Test 2: Multiple frames — ensure semaphore tracking stays in sync."""
    print("\n" + "=" * 70)
    print("  TEST 2: Multi-frame semaphore sync")
    print("=" * 70)

    N = 4
    W, H = 80, 80

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available or not r.interop_available:
        print("  SKIP")
        r.close()
        return False

    v, idx = make_box()
    r.add_mesh(positions=v, indices=idx, display_color=(0.8, 0.3, 0.1))
    r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    vps = orbit_vps(N, 2.0, (0, 0, 0), 45.0, W, H)

    # First render to initialize tiled resources
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

    from nusd_renderer._cuda_interop import CudaVulkanInterop
    interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H)

    num_frames = 10
    all_ok = True
    for frame in range(num_frames):
        # Render
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

        # CUDA read
        cuda_px = interop.wait_and_copy_to_host()

        # CPU reference
        cpu_px = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()

        match = np.array_equal(cuda_px, cpu_px)
        if not match:
            diff = np.abs(cuda_px.astype(int) - cpu_px.astype(int)).max()
            if diff > 1:
                print(f"  Frame {frame}: FAIL (max_diff={diff})")
                all_ok = False
            else:
                pass  # rounding OK
        # else: perfect match

    if all_ok:
        print(f"  PASS: {num_frames} frames, all pixels match")
    else:
        print(f"  FAIL: some frames had pixel mismatches")

    interop.close()
    r.close()
    return all_ok


def test_interop_high_envs():
    """Test 3: Higher env counts (512) — verify at scale."""
    print("\n" + "=" * 70)
    print("  TEST 3: High env count (512)")
    print("=" * 70)

    N = 512
    W, H = 100, 100

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available or not r.interop_available:
        print("  SKIP")
        r.close()
        return False

    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        r.add_mesh(positions=v, indices=idx, display_color=(0.7, 0.3, 0.2))
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)

    # Initialize tiled
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

    from nusd_renderer._cuda_interop import CudaVulkanInterop
    interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H)

    # Render + CUDA read
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
    cuda_px = interop.wait_and_copy_to_host()
    cpu_px = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()

    nc = math.ceil(math.sqrt(N))
    nr = math.ceil(N / nc)
    total_w, total_h = nc * W, nr * H

    print(f"  Grid: {nc}x{nr} = {total_w}x{total_h} ({total_w*total_h*4/1024/1024:.1f} MB)")
    print(f"  CUDA shape: {cuda_px.shape}, CPU shape: {cpu_px.shape}")

    match = np.array_equal(cuda_px, cpu_px)
    nonzero_cuda = np.count_nonzero(cuda_px)
    nonzero_cpu = np.count_nonzero(cpu_px)

    if match:
        print(f"  PASS: 512-env pixels match ({nonzero_cuda} nonzero values)")
    else:
        diff = np.abs(cuda_px.astype(int) - cpu_px.astype(int))
        max_diff = diff.max()
        n_differ = (diff > 0).sum()
        print(f"  max_diff={max_diff}, {n_differ} values differ, "
              f"CUDA nonzero={nonzero_cuda}, CPU nonzero={nonzero_cpu}")
        if max_diff <= 1:
            print(f"  PASS (within rounding)")
        else:
            print(f"  FAIL")

    # Quick timing
    import time
    times_cuda = []
    times_cpu = []
    for _ in range(5):
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        t0 = time.perf_counter()
        cuda_px = interop.wait_and_copy_to_host()
        times_cuda.append((time.perf_counter() - t0) * 1000)

    for _ in range(5):
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        t0 = time.perf_counter()
        cpu_px = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H)
        times_cpu.append((time.perf_counter() - t0) * 1000)

    print(f"  CUDA readback: {np.mean(times_cuda):.2f}ms avg")
    print(f"  CPU raw staging: {np.mean(times_cpu):.2f}ms avg")

    interop.close()
    r.close()
    return True


def test_all_in_one():
    """Run all tests within a single NuRenderer to avoid multi-GLFW issues."""
    print("=" * 70)
    print("  ALL-IN-ONE TEST: single renderer, multi-frame + high envs")
    print("=" * 70)

    N = 16
    W, H = 100, 100

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available:
        print("  SKIP: RT not available")
        r.close()
        return

    # Build a scene
    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    gv = np.array([[-10,-0.5,-10],[10,-0.5,-10],[10,-0.5,10],[-10,-0.5,10]], dtype=np.float32)
    gi = np.array([0,1,2,0,2,3], dtype=np.uint32)
    r.add_mesh(positions=gv, indices=gi, display_color=(0.5, 0.5, 0.5))

    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    if not r.interop_available:
        print("  SKIP: Interop not available")
        r.close()
        return

    vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)

    # ---- Test A: Basic validation ----
    print("\n  A) Basic validation (16 cameras)")
    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
    cpu_raw = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()
    print(f"     CPU raw: {cpu_raw.shape}, first 8: {cpu_raw.ravel()[:8]}")

    from nusd_renderer._cuda_interop import CudaVulkanInterop
    interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H)
    print(f"     Interop: {interop.image_w}x{interop.image_h}")

    r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
    cuda_px = interop.wait_and_copy_to_host()
    cpu_raw2 = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()

    match = np.array_equal(cuda_px, cpu_raw2)
    if match:
        print(f"     PASS: CUDA matches CPU ({np.count_nonzero(cuda_px)}/{cuda_px.size} nonzero)")
    else:
        diff = np.abs(cuda_px.astype(int) - cpu_raw2.astype(int))
        print(f"     {'PASS (rounding)' if diff.max() <= 1 else 'FAIL'}: max_diff={diff.max()}")

    # ---- Test B: Multi-frame ----
    print("\n  B) Multi-frame (10 frames, same interop)")
    all_ok = True
    for frame in range(10):
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        cuda_px = interop.wait_and_copy_to_host()
        cpu_px = r.fetch_tiled_raw(num_cameras=N, tile_w=W, tile_h=H).copy()
        diff = np.abs(cuda_px.astype(int) - cpu_px.astype(int)).max()
        if diff > 1:
            print(f"     Frame {frame}: FAIL (max_diff={diff})")
            all_ok = False
    if all_ok:
        print(f"     PASS: 10 frames match")

    interop.close()

    # ---- Test C: High env count ----
    print("\n  C) High env count (512 cameras)")
    N2 = 512
    vps2 = orbit_vps(N2, 3.0, (0, 0, 0), 45.0, W, H)

    r.render_tiled(vps2, num_cameras=N2, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
    interop2 = CudaVulkanInterop(r, num_cameras=N2, tile_w=W, tile_h=H)

    r.render_tiled(vps2, num_cameras=N2, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
    cuda_px2 = interop2.wait_and_copy_to_host()
    cpu_px2 = r.fetch_tiled_raw(num_cameras=N2, tile_w=W, tile_h=H).copy()

    match2 = np.array_equal(cuda_px2, cpu_px2)
    nc = math.ceil(math.sqrt(N2))
    nr = math.ceil(N2 / nc)
    total_w, total_h = nc * W, nr * H
    print(f"     Grid: {nc}x{nr} = {total_w}x{total_h} ({total_w*total_h*4/1024/1024:.1f} MB)")
    if match2:
        print(f"     PASS: {N2}-env pixels match")
    else:
        diff2 = np.abs(cuda_px2.astype(int) - cpu_px2.astype(int))
        print(f"     {'PASS (rounding)' if diff2.max() <= 1 else 'FAIL'}: max_diff={diff2.max()}")

    # Quick timing
    import time
    times_cuda = []
    times_cpu = []
    for _ in range(5):
        r.render_tiled(vps2, num_cameras=N2, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        t0 = time.perf_counter()
        interop2.wait_and_copy_to_host()
        times_cuda.append((time.perf_counter() - t0) * 1000)

    for _ in range(5):
        r.render_tiled(vps2, num_cameras=N2, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        t0 = time.perf_counter()
        r.fetch_tiled_raw(num_cameras=N2, tile_w=W, tile_h=H)
        times_cpu.append((time.perf_counter() - t0) * 1000)

    print(f"     CUDA DtoH: {np.mean(times_cuda):.2f}ms avg")
    print(f"     CPU staging: {np.mean(times_cpu):.2f}ms avg")

    interop2.close()
    r.close()
    print("\n  DONE")


if __name__ == "__main__":
    test_all_in_one()
