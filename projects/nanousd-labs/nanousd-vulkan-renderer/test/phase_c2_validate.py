# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase C.2 deferred-shading validation.

Verifies the four shader-side gates of the Phase C.2 brief:

  Gate 1 (off byte-identity)  — covered by test_scene_curves on
                                showcase_grid_medium.usdc; sha256 matches
                                Phase C.1 commit cdf2d7f.
  Gate 2 (mode=1 normal viz)  — agibot_scanner.usda renders smooth
                                normal-direction RGB with many distinct
                                colors per tile (>= 50).
  Gate 3 (mode=0 byte ident.) — agibot_scanner.usda mode=0 sha256 matches
                                the C.1-baseline reference render.
  Gate 4 (no-material fallback) — test_cube.usda renders without crashing
                                and without NaN in either mode.

Usage:
  python phase_c2_validate.py [<expected_mode0_sha256>]

If no SHA is provided, the script just emits the captured sums + counts
without failing on Gate 3 (handy for environments where the baseline PPM
doesn't exist).
"""
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
from phase_c1_validate import (
    make_vp_inv,
    save_grid_ppm,
    per_tile_distinct_colors,
    per_tile_hash,
)

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
AGIBOT = str(REPO / "tests/correctness/assets/agibot_scanner.usda")
CUBE = os.environ.get(
    "NUSD_TEST_CUBE",
    str(WORKSPACE / "nanousd-opengl-renderer/test_cube.usda"),
)

# C.1 baseline SHA captured before Phase C.2 changes (mode=0 = C.1 path).
DEFAULT_C1_SHA = "688561f1fbbb32d7602b378584b5418abb730c5f865691c8ccf014ec8036aa76"


def render_scene(usdc_path, num_cams, tile_w, tile_h, debug_mode):
    r = NuRenderer(width=tile_w, height=tile_h, enable_rt=True, enable_materials=True)
    if not r.rt_available:
        raise RuntimeError("RT not available")
    r.load_usd(usdc_path)
    r.build_accel()
    r.set_deferred_shade(True)
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
        vps.append(make_vp_inv(eye, target, 45.0, tile_w, tile_h))
    vps = np.stack(vps).astype(np.float32)

    for _ in range(4):
        r.render_tiled(vps, num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h, mode=NU_RENDER_RT)
        _ = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h)
    r.render_tiled(vps, num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h, mode=NU_RENDER_RT)
    fr = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h).copy()
    r.close()
    return fr


def gate2_normals():
    print("Gate 2: agibot_scanner mode=1 normal viz")
    fr = render_scene(AGIBOT, 16, 64, 64, debug_mode=1)
    OUT = str(REPO / "render_phaseC2_normals.ppm")
    save_grid_ppm(OUT, fr, 4, 64, 64)
    counts = per_tile_distinct_colors(fr)
    hashes = per_tile_hash(fr)
    print(f"  saved {OUT}, sum={fr.sum()}")
    print(f"  per-tile distinct colors: {counts}")
    print(f"  distinct tile hashes: {len(set(hashes))}/16")
    pass_no_nan = not np.isnan(fr.astype(np.float32)).any()
    pass_normals = max(counts) >= 50
    pass_hash = len(set(hashes)) >= 10
    pass_content = fr.sum() > 0
    ok = pass_no_nan and pass_normals and pass_hash and pass_content
    print(f"  PASS" if ok else "  FAIL")
    return ok


def gate3_mode0_byte_identity(expected_sha=DEFAULT_C1_SHA):
    print("Gate 3: agibot_scanner mode=0 byte-identity vs C.1 baseline")
    fr = render_scene(AGIBOT, 16, 64, 64, debug_mode=0)
    OUT = str(REPO / "render_phaseC2_mode0.ppm")
    save_grid_ppm(OUT, fr, 4, 64, 64)
    with open(OUT, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    print(f"  saved {OUT}, sum={fr.sum()}")
    print(f"  got      sha256: {got}")
    print(f"  expected sha256: {expected_sha}")
    ok = got == expected_sha
    print(f"  PASS" if ok else "  FAIL")
    return ok


def gate4_no_material_fallback():
    print("Gate 4: test_cube.usda no-material fallback")
    if not os.path.exists(CUBE):
        print(f"  SKIP: {CUBE} not present")
        return True
    fr0 = render_scene(CUBE, 4, 64, 64, debug_mode=0)
    fr1 = render_scene(CUBE, 4, 64, 64, debug_mode=1)
    save_grid_ppm(
        str(REPO / "render_phaseC2_gate4_mode0.ppm"),
        fr0, 2, 64, 64)
    save_grid_ppm(
        str(REPO / "render_phaseC2_gate4_mode1.ppm"),
        fr1, 2, 64, 64)
    c0 = per_tile_distinct_colors(fr0)
    c1 = per_tile_distinct_colors(fr1)
    print(f"  mode=0 distinct colors per tile: {c0} (sum={fr0.sum()})")
    print(f"  mode=1 distinct colors per tile: {c1} (sum={fr1.sum()})")
    pass_no_nan = (
        not np.isnan(fr0.astype(np.float32)).any()
        and not np.isnan(fr1.astype(np.float32)).any()
    )
    pass_mode0_simple = max(c0) <= 6        # C.1 fallback = per-mesh color
    pass_mode1_variation = max(c1) >= 2     # at least 2 face-normal directions
    ok = pass_no_nan and pass_mode0_simple and pass_mode1_variation \
         and fr0.sum() > 0 and fr1.sum() > 0
    print(f"  PASS" if ok else "  FAIL")
    return ok


def main():
    expected_sha = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_C1_SHA
    g2 = gate2_normals()
    g3 = gate3_mode0_byte_identity(expected_sha)
    g4 = gate4_no_material_fallback()
    print()
    print(f"Gate 2 (normal viz)            : {'PASS' if g2 else 'FAIL'}")
    print(f"Gate 3 (mode=0 byte identity)  : {'PASS' if g3 else 'FAIL'}")
    print(f"Gate 4 (no-material fallback)  : {'PASS' if g4 else 'FAIL'}")
    sys.exit(0 if (g2 and g3 and g4) else 1)


if __name__ == "__main__":
    main()
