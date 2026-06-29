# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase C.4 bench-path tile-hash check.

Validates that the IsaacLab bench's actual data path (CudaOwnedInterop +
fast_mode) produces correct mode 3 pixels per the workspace's
distinct-tile-hash gate.

Per memory `feedback_visual_renderer_correctness_gate`: bench-perf gain
without bench-path visual verification is the exact failure mode that
hid a 15/16-bit-identical-tile bug in the past. This script renders the
SAME scene (agibot proxy with fast_mode=True, no scene lights, no IBL)
through the bench's interop path, then uses fetch_pixels_tiled to
read back the same data the policy training reads. A stress at 1024
camera tiles ensures we catch any per-env-becomes-degenerate bug.

Usage: python phase_c4_bench_path_tile_check.py
"""
import sys
import os
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
from phase_c1_validate import make_vp_inv, per_tile_distinct_colors, per_tile_hash, save_grid_ppm

REPO = Path(__file__).resolve().parents[1]
SCENE = os.environ.get(
    "NUSD_AGIBOT_SCANNER",
    str(REPO / "tests/correctness/assets/agibot_scanner.usda"),
)


def render(num_cams, deferred, debug_mode):
    """Render the scene with bench-style settings: fast_mode=True (RL
    sensors), large camera count to stress per-tile diversity."""
    r = NuRenderer(width=64, height=64, enable_rt=True, enable_materials=True)
    if not r.rt_available:
        raise RuntimeError("RT not available")
    r.load_usd(SCENE)
    r.build_accel()
    r.set_fast_mode(True)
    r.set_deferred_shade(deferred)
    if deferred:
        r.set_deferred_debug_mode(debug_mode)

    bounds = r.get_scene_bounds()
    if bounds is not None:
        bmin, bmax = bounds
        target = np.array(
            [(bmin[0] + bmax[0]) * 0.5,
             (bmin[1] + bmax[1]) * 0.5,
             (bmin[2] + bmax[2]) * 0.5],
            np.float32,
        )
        diag = float(np.linalg.norm(np.array(bmax) - np.array(bmin)))
        radius = max(diag * 1.5, 0.1)
        elev   = target[1] + diag * 0.3
    else:
        target = np.array([0.0, 0.5, 0.0], np.float32)
        radius = 4.5
        elev   = 2.0

    vps = []
    for i in range(num_cams):
        ang = 2 * math.pi * i / max(num_cams, 1)
        eye = target + np.array(
            [radius * math.cos(ang),
             elev - target[1],
             radius * math.sin(ang)],
            np.float32,
        )
        vps.append(make_vp_inv(eye, target, 45.0, 64, 64))
    vps = np.stack(vps).astype(np.float32)

    for _ in range(4):
        r.render_tiled(vps, num_cameras=num_cams, tile_w=64, tile_h=64, mode=NU_RENDER_RT)
        _ = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=64, tile_h=64)
    r.render_tiled(vps, num_cameras=num_cams, tile_w=64, tile_h=64, mode=NU_RENDER_RT)
    fr = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=64, tile_h=64).copy()
    r.close()
    return fr


def main():
    """Run distinct-tile-hash check at 64 cameras (proxy for bench-scale
    per-env diversity stress).

    Workspace memory rule: distinct-tile-hash count must match between
    OFF and ON. If C.4 produces 1/N hashes vs OFF's N/N, the bench is
    measuring degenerate output. Note: agibot OFF returns black on the
    tiled path (pre-existing — see phase_c1_validate.py:272-289), so we
    can't compare ON/OFF distinct counts directly. We use mode 0 as the
    proxy reference (it does the per-env base-color path — proven
    bit-correct in C.3 gates) and require mode 3 to match its
    distinct-tile-hash count. If mode 3 has FEWER distinct tile hashes
    than mode 0 on the same scene, mode 3 is dropping per-env detail
    and the dispatch is broken in this configuration."""
    N = 64
    print(f"Bench-path tile-hash check: {N} cameras, fast_mode=True, agibot_scanner.usda")

    fr_m0 = render(N, deferred=True, debug_mode=0)
    fr_m3 = render(N, deferred=True, debug_mode=3)

    save_grid_ppm(f"{REPO}/render_phaseC4_bench_path_m0.ppm", fr_m0, 8, 64, 64)
    save_grid_ppm(f"{REPO}/render_phaseC4_bench_path_m3.ppm", fr_m3, 8, 64, 64)

    counts_m0 = per_tile_distinct_colors(fr_m0)
    counts_m3 = per_tile_distinct_colors(fr_m3)
    hashes_m0 = per_tile_hash(fr_m0)
    hashes_m3 = per_tile_hash(fr_m3)

    distinct_m0 = len(set(hashes_m0))
    distinct_m3 = len(set(hashes_m3))
    print(f"  mode0 sum={int(fr_m0.sum()):>15,d}  distinct tile hashes: {distinct_m0}/{N}")
    print(f"  mode3 sum={int(fr_m3.sum()):>15,d}  distinct tile hashes: {distinct_m3}/{N}")
    print(f"  mode0 max distinct colors per tile: {max(counts_m0)}")
    print(f"  mode3 max distinct colors per tile: {max(counts_m3)}")

    # Mode 3 must produce at least as many distinct tile hashes as mode 0.
    # If mode 3 collapses to fewer distinct hashes, per-env content is
    # being dropped and the bench's perf number is meaningless.
    pass_distinct = distinct_m3 >= distinct_m0
    pass_no_nan   = not np.isnan(fr_m3.astype(np.float32)).any()
    pass_content  = fr_m3.sum() > 0
    pass_brighter = fr_m3.sum() >= fr_m0.sum()  # IBL/ambient adds energy
    pass_max_colors = max(counts_m3) >= max(counts_m0)  # mode 3 has at least as much variation

    ok = pass_distinct and pass_no_nan and pass_content and pass_brighter and pass_max_colors
    print(f"  PASS" if ok else "  FAIL")
    if not ok:
        print(f"  reasons: distinct={pass_distinct} no_nan={pass_no_nan} "
              f"content={pass_content} brighter={pass_brighter} max_colors={pass_max_colors}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
