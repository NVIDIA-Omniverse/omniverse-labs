#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the warp de-tiling kernel: verifies GPU de-tiling matches CPU de-tiling.

Creates a scene, renders with tiled RT, imports the tiled image via CUDA interop
as a warp array, runs the _detile_kernel on GPU, and compares against the CPU
fetch_pixels_tiled output (which does the same de-tiling on CPU).
"""

import math
import sys
import os
import time
import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import the de-tiling kernel from the IsaacLab renderer module.
# Since that module has IsaacLab deps, we replicate the kernel here directly.
@wp.kernel
def _detile_kernel(
    tiled: wp.array3d(dtype=wp.uint8),
    output: wp.array4d(dtype=wp.uint8),
    num_cols: int,
    tile_w: int,
    tile_h: int,
):
    env, y, x = wp.tid()
    col = env % num_cols
    row = env // num_cols
    src_x = col * tile_w + x
    src_y = row * tile_h + y
    output[env, y, x, 0] = tiled[src_y, src_x, 0]
    output[env, y, x, 1] = tiled[src_y, src_x, 1]
    output[env, y, x, 2] = tiled[src_y, src_x, 2]
    output[env, y, x, 3] = tiled[src_y, src_x, 3]


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


def test_detile_kernel():
    """Test that GPU de-tiling via warp kernel matches CPU fetch_pixels_tiled."""
    print("=" * 70)
    print("  TEST: GPU de-tiling kernel vs CPU fetch_pixels_tiled")
    print("=" * 70)

    wp.init()

    W, H = 100, 100

    r = NuRenderer(width=512, height=512, enable_rt=True)
    if not r.rt_available or not r.interop_available:
        print(f"  SKIP: RT={r.rt_available}, interop={r.interop_available}")
        r.close()
        return

    # Build scene
    for i in range(3):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2)][i]
        r.add_mesh(positions=v, indices=idx, display_color=color)
    gv = np.array([[-10,-0.5,-10],[10,-0.5,-10],[10,-0.5,10],[-10,-0.5,10]], dtype=np.float32)
    gi = np.array([0,1,2,0,2,3], dtype=np.uint32)
    r.add_mesh(positions=gv, indices=gi, display_color=(0.5, 0.5, 0.5))
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    from nusd_renderer._cuda_interop import CudaVulkanInterop

    for N in [4, 16, 64, 512]:
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, W, H)

        # Render and get CPU reference BEFORE creating interop (which disables staging)
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        cpu_detiled = r.fetch_pixels_tiled(N, W, H).copy()

        # Now create interop (enables skip_staging) and render again
        interop = CudaVulkanInterop(r, num_cameras=N, tile_w=W, tile_h=H)
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)

        # GPU path: get tiled warp array, run de-tile kernel
        tiled_warp = interop.wait_and_get_warp_array()

        output_gpu = wp.zeros((N, H, W, 4), dtype=wp.uint8, device="cuda:0")
        num_cols = math.ceil(math.sqrt(N))

        wp.launch(
            _detile_kernel,
            dim=(N, H, W),
            inputs=[tiled_warp, output_gpu, num_cols, W, H],
            device="cuda:0",
        )
        wp.synchronize()

        # Copy GPU result to CPU for comparison
        gpu_detiled = output_gpu.numpy()

        match = np.array_equal(gpu_detiled, cpu_detiled)
        if match:
            nonzero = np.count_nonzero(gpu_detiled)
            print(f"  N={N:4d}: PASS — GPU de-tile matches CPU ({nonzero} nonzero values)")
        else:
            diff = np.abs(gpu_detiled.astype(int) - cpu_detiled.astype(int))
            max_diff = diff.max()
            n_differ = (diff > 0).sum()
            if max_diff <= 1:
                print(f"  N={N:4d}: PASS (rounding) — max_diff={max_diff}, {n_differ} values differ")
            else:
                print(f"  N={N:4d}: FAIL — max_diff={max_diff}, {n_differ} values differ")
                # Show first mismatch location
                where = np.argwhere(diff > 1)
                if len(where):
                    loc = where[0]
                    print(f"           First mismatch at {tuple(loc)}: "
                          f"GPU={gpu_detiled[tuple(loc)]}, CPU={cpu_detiled[tuple(loc)]}")

        # Timing comparison for largest case
        if N == 512:
            print(f"\n  Timing at N={N}:")
            # GPU de-tile path
            times_gpu = []
            for _ in range(10):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
                t0 = time.perf_counter()
                tiled_w = interop.wait_and_get_warp_array()
                wp.launch(_detile_kernel, dim=(N, H, W),
                          inputs=[tiled_w, output_gpu, num_cols, W, H], device="cuda:0")
                wp.synchronize()
                times_gpu.append((time.perf_counter() - t0) * 1000)

            # CPU de-tile path
            times_cpu = []
            for _ in range(10):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
                t0 = time.perf_counter()
                cpu_px = r.fetch_pixels_tiled(N, W, H)
                times_cpu.append((time.perf_counter() - t0) * 1000)

            print(f"    GPU zero-copy + de-tile: {np.mean(times_gpu):.2f}ms avg")
            print(f"    CPU staging + de-tile:   {np.mean(times_cpu):.2f}ms avg")
            print(f"    Speedup: {np.mean(times_cpu) / np.mean(times_gpu):.1f}x")

        interop.close()

    r.close()
    print("\n  DONE")


if __name__ == "__main__":
    test_detile_kernel()
