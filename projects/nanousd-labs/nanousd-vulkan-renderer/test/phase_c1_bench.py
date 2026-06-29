# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase C.1 perf bench — measure FPS overhead vs Phase B.

Runs OFF-mode and ON-mode tiled renders of a textured scene (agibot_scanner)
at 256 cams x 64x64 = 1024x1024 tiled, mirroring Phase B's perf gate. Reports
mean FPS over N frames after a warm-up.

Phase B's commit (802a72d) reported on RTX 5090:
  OFF = 227.5 FPS, ON = 222.9 FPS, ratio 1.021 (~0.09 ms/frame compute cost)

Phase C.1 expectation: ON drops a small additional amount because the compute
pass now does texture sampling (15-20% of rchit work, but only once per pixel
at the deferred-compute stage). We expect ratio between 1.05 and 1.15.
"""
import sys
import os
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'python')))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT


def make_view_matrix(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    f = target - eye; f /= np.linalg.norm(f) + 1e-9
    up = np.asarray(up, np.float32)
    r = np.cross(f, up); rl = np.linalg.norm(r)
    if rl < 1e-6:
        up = np.array([0, 0, 1], np.float32); r = np.cross(f, up); rl = np.linalg.norm(r)
    r /= rl
    u = np.cross(r, f)
    view = np.zeros(16, np.float32)
    view[0] = r[0]; view[1] = r[1]; view[2] = r[2]; view[3] = -np.dot(r, eye)
    view[4] = u[0]; view[5] = u[1]; view[6] = u[2]; view[7] = -np.dot(u, eye)
    view[8] = -f[0]; view[9] = -f[1]; view[10] = -f[2]; view[11] = np.dot(f, eye)
    view[15] = 1.0
    return view


def invert_view(v):
    return np.linalg.inv(v.reshape(4, 4)).reshape(16)


def make_proj(fov_deg, aspect, near=0.1, far=1000.0):
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2)
    proj = np.zeros(16, np.float32)
    proj[0] = f / aspect; proj[5] = f
    proj[10] = (far + near) / (near - far); proj[14] = (2 * far * near) / (near - far)
    proj[11] = -1.0
    return proj


def invert_proj(proj):
    pi = np.zeros(16, np.float32)
    pi[0] = 1.0 / proj[0]; pi[5] = 1.0 / proj[5]; pi[14] = -1.0
    pi[11] = 1.0 / proj[14]; pi[15] = proj[10] / proj[14]
    return pi


def make_vp_inv(eye, target, fov_deg, w, h):
    return np.concatenate([invert_view(make_view_matrix(eye, target)),
                           invert_proj(make_proj(fov_deg, w / h))])


def main():
    repo = Path(__file__).resolve().parents[1]
    SCENE = os.environ.get(
        "NUSD_AGIBOT_SCANNER",
        str(repo / "tests/correctness/assets/agibot_scanner.usda"),
    )
    NUM_CAMS = 256
    TILE_W = TILE_H = 64
    WARMUP = 5
    BENCH = 30

    r = NuRenderer(width=TILE_W, height=TILE_H, enable_rt=True, enable_materials=True)
    if not r.rt_available:
        print("RT not available")
        sys.exit(2)

    r.load_usd(SCENE)
    r.build_accel()

    # Camera ring around the agibot scanner.
    bounds = r.get_scene_bounds()
    if bounds is not None:
        bmin, bmax = bounds
        cx = float((bmin[0] + bmax[0]) * 0.5)
        cy = float((bmin[1] + bmax[1]) * 0.5)
        cz = float((bmin[2] + bmax[2]) * 0.5)
        target = np.array([cx, cy, cz], np.float32)
        diag = float(np.linalg.norm(np.array(bmax) - np.array(bmin)))
        radius = max(diag * 1.5, 0.1)
        elev = cy + diag * 0.3
    else:
        target = np.array([0.0, 0.5, 0.0], np.float32)
        radius = 4.5; elev = 2.0

    vps = []
    import math
    for i in range(NUM_CAMS):
        ang = 2 * math.pi * i / NUM_CAMS
        eye = target + np.array([radius * math.cos(ang),
                                  elev - target[1],
                                  radius * math.sin(ang)], np.float32)
        vps.append(make_vp_inv(eye, target, 45.0, TILE_W, TILE_H))
    vps = np.stack(vps).astype(np.float32)

    def run_bench(deferred):
        r.set_deferred_shade(deferred)
        # Warmup
        for _ in range(WARMUP):
            r.render_tiled(vps, num_cameras=NUM_CAMS, tile_w=TILE_W, tile_h=TILE_H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=NUM_CAMS, tile_w=TILE_W, tile_h=TILE_H)
        # Measure
        t0 = time.perf_counter()
        for _ in range(BENCH):
            r.render_tiled(vps, num_cameras=NUM_CAMS, tile_w=TILE_W, tile_h=TILE_H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=NUM_CAMS, tile_w=TILE_W, tile_h=TILE_H)
        dt = time.perf_counter() - t0
        return BENCH / dt, dt / BENCH * 1000

    fps_off, ms_off = run_bench(False)
    fps_on,  ms_on  = run_bench(True)
    ratio = fps_off / fps_on if fps_on > 0 else 0.0
    overhead_ms = ms_on - ms_off

    print(f"\nPhase C.1 perf bench ({NUM_CAMS} cams x {TILE_W}x{TILE_H}, "
          f"{NUM_CAMS * TILE_W * TILE_H / 1e6:.1f} Mpix/frame):")
    print(f"  OFF: {fps_off:7.2f} FPS, {ms_off:6.3f} ms/frame")
    print(f"  ON : {fps_on:7.2f} FPS, {ms_on:6.3f} ms/frame")
    print(f"  Compute-pass overhead: +{overhead_ms:.3f} ms/frame, "
          f"ratio OFF/ON = {ratio:.3f}")
    pass_perf = ratio < 1.30  # within 30% — Phase C.1 is heavier than Phase B
    print(f"  Within 30% gate (lenient — texture sampling is non-trivial): {pass_perf}")
    r.close()


if __name__ == "__main__":
    main()
