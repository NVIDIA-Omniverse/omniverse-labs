#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for nanousd-vulkan-renderer.

Renders a small fixed set of assets in two modes:

  - Vulkan (under test): nanousd_renderer's RT path
  - ovrtx (reference):    external reference renderer, 200 frames, deterministic at fixed seed

For each asset the repo carries TWO golden images that are checked in:

  golden/<asset>_vulkan.png   — what Vulkan rendered when the golden was set
  golden/<asset>_ovrtx.png    — what ovrtx rendered when the golden was set

Each test runs both renderers fresh and compares against both goldens:

  vulkan-vs-vulkan-golden:  tight RMS≤2 — catches *regressions* in our renderer
  ovrtx-vs-ovrtx-golden:    tight RMS≤2 — catches drift in the ovrtx wrapper / driver
  vulkan-vs-ovrtx (current): loose RMS≤TOL_OVRTX — sanity check vs reference truth

Pass/fail is reported per-asset and as a suite total. Returns nonzero on
any failure (so this script can be wired into CI).

To regenerate the goldens (after an intentional change), run:
    python3 run_tests.py --update

Usage:
    python3 run_tests.py             # run + verify
    python3 run_tests.py --update    # rewrite goldens from current renders
    python3 run_tests.py --only chess  # filter to one test
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RENDERER_REPO = HERE.parent.parent
sys.path.insert(0, str(RENDERER_REPO / "comparisons" / "usdimaging_lights"))
sys.path.insert(0, str(RENDERER_REPO / "python"))
import compare_lights  # noqa  (reuses render_ovrtx subprocess pattern)
from nusd_renderer._bindings import (
    NuRenderer, NU_RENDER_RASTER, NU_RENDER_SHADOW, NU_RENDER_RT,
)


_MODE_NAMES = {
    NU_RENDER_RASTER: "raster",
    NU_RENDER_SHADOW: "shadow",
    NU_RENDER_RT:     "rt",
}


def render_vulkan_mode(usd: Path, label: str, w: int, h: int,
                       mode: int) -> Path:
    """Render `usd` via Vulkan in the requested mode.
    Mirrors compare_lights.render_vulkan but with explicit mode selection
    instead of the RT-or-fallback default."""
    cam = compare_lights.parse_camera_from_usda(usd)
    if cam['eye'] is None or cam['fwd'] is None:
        raise RuntimeError(f"No camera in {usd}")
    eye = cam['eye']
    fwd = cam['fwd']
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    fov_v = float(np.degrees(2 * np.arctan(cam['verticalAperture']
                                           / (2 * cam['focalLength']))))

    nu = NuRenderer(width=w, height=h, enable_rt=True, enable_materials=True)
    n = nu.load_usd(str(usd))
    if n == 0:
        nu.close()
        raise RuntimeError(f"loaded 0 meshes from {usd}")
    nu.set_camera(eye=eye, target=target, fov_degrees=fov_v,
                  near_clip=0.01, far_clip=1000.0)
    if mode != NU_RENDER_RASTER:
        try:
            nu.build_accel()
        except Exception:
            pass
    if mode != NU_RENDER_RASTER and not nu.rt_available:
        nu.close()
        raise RuntimeError(f"mode {_MODE_NAMES[mode]} requires RT (hardware unavailable)")
    nu.render(mode=mode)
    nu.render(mode=mode)
    rgba = nu.fetch_pixels()
    out_png = OUTPUT_DIR / f"{label}_vulkan.png"
    Image.fromarray(rgba).save(out_png, optimize=True)
    nu.close()
    return out_png

ASSET_DIR = HERE / "assets"
GOLDEN_DIR = HERE / "golden"
OUTPUT_DIR = HERE / "output"
GOLDEN_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
compare_lights.OUT_DIR = OUTPUT_DIR

# Default thresholds (overridable per-test)
TOL_GOLDEN_RMS_DEFAULT = 2.0      # vulkan-vs-vulkan-golden  (regression detection)
TOL_OVRTX_RMS_DEFAULT  = 60.0     # vulkan-vs-ovrtx (current)

TESTS = [
    # (label, USDA, mode, w, h, ovrtx_frames, tol_golden, tol_ovrtx)
    #
    # Each asset is exercised in all three render modes. The cross-renderer
    # (vs ovrtx) threshold is only meaningful for RT — raster and shadow
    # modes use the renderer's three-point lighting and will diverge
    # heavily from ovrtx's path-traced reference. They're set to 999 so
    # those tests only measure vulkan-vs-vulkan-golden regression.
    #
    # chess: MaterialX standard_surface materials. ovrtx UJITSO fails to
    # compile the multi-node graph and falls back to uniform red, so the
    # cross-renderer threshold is loose for all modes.
    ("chess_king_black",     ASSET_DIR / "chess_king_black.usda",     NU_RENDER_RT,     640, 480, 200, 2.0, 999.0),
    ("chess_king_black",     ASSET_DIR / "chess_king_black.usda",     NU_RENDER_SHADOW, 640, 480, 200, 2.0, 999.0),
    ("chess_king_black",     ASSET_DIR / "chess_king_black.usda",     NU_RENDER_RASTER, 640, 480, 200, 2.0, 999.0),
    # agibot: OmniPBR/MDL.
    ("agibot_scanner",       ASSET_DIR / "agibot_scanner.usda",       NU_RENDER_RT,     640, 480, 60,  2.0, 60.0),
    ("agibot_scanner",       ASSET_DIR / "agibot_scanner.usda",       NU_RENDER_SHADOW, 640, 480, 60,  2.0, 999.0),
    ("agibot_scanner",       ASSET_DIR / "agibot_scanner.usda",       NU_RENDER_RASTER, 640, 480, 60,  2.0, 999.0),
    # material_tray: ~88 RMS to ovrtx is the IBL-exposure divergence
    # (Vulkan brighter for white plastic). Loose threshold until the
    # post-process auto-exposure lands.
    ("agibot_material_tray", ASSET_DIR / "agibot_material_tray.usda", NU_RENDER_RT,     640, 480, 60,  2.0, 100.0),
    ("agibot_material_tray", ASSET_DIR / "agibot_material_tray.usda", NU_RENDER_SHADOW, 640, 480, 60,  2.0, 999.0),
    ("agibot_material_tray", ASSET_DIR / "agibot_material_tray.usda", NU_RENDER_RASTER, 640, 480, 60,  2.0, 999.0),
]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    a = a[..., :3].astype(np.float32)
    b = b[..., :3].astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def run_one(label: str, usd: Path, mode: int, w: int, h: int, frames: int,
            tol_golden: float, tol_ovrtx: float, update: bool) -> dict:
    mode_name = _MODE_NAMES.get(mode, f"mode{mode}")
    full_label = f"{label}_{mode_name}"
    print(f"\n=== {full_label} ===")
    print(f"  USDA: {usd}")
    if not usd.exists():
        return {'label': full_label, 'status': 'SKIP', 'reason': f'missing {usd}'}

    t0 = time.perf_counter()
    try:
        vulk_png = render_vulkan_mode(usd, full_label, w, h, mode)
    except Exception as e:
        return {'label': full_label, 'status': 'FAIL', 'reason': f'vulkan render: {e}'}
    # ovrtx render is shared across modes (it's always path-traced reference).
    # Cache by base label so we only run ovrtx once per asset.
    ovrtx_png = OUTPUT_DIR / f"{label}_ovrtx.png"
    if not ovrtx_png.exists():
        try:
            ovrtx_png = compare_lights.render_ovrtx(usd, label, w, h, frames)
        except Exception as e:
            return {'label': full_label, 'status': 'FAIL', 'reason': f'ovrtx render: {e}'}
    wall_ms = (time.perf_counter() - t0) * 1000

    g_vulk = GOLDEN_DIR / f"{full_label}_vulkan.png"
    g_ovrtx = GOLDEN_DIR / f"{label}_ovrtx.png"  # ovrtx golden shared across modes

    if update:
        Image.open(vulk_png).save(g_vulk, optimize=True)
        # Only refresh the ovrtx golden once per asset (shared across modes).
        if not g_ovrtx.exists() or mode == NU_RENDER_RT:
            Image.open(ovrtx_png).save(g_ovrtx, optimize=True)
            print(f"  UPDATED goldens: {g_vulk.name}, {g_ovrtx.name}")
        else:
            print(f"  UPDATED golden: {g_vulk.name} (ovrtx shared)")
        return {'label': full_label, 'status': 'UPDATE', 'wall_ms': wall_ms}

    cur_vulk = _load_rgb(vulk_png)
    cur_ovrtx = _load_rgb(ovrtx_png)
    rms_cross = _rms(cur_vulk, cur_ovrtx)

    fails = []
    if g_vulk.exists():
        rms_vv = _rms(cur_vulk, _load_rgb(g_vulk))
        status_vv = 'PASS' if rms_vv <= tol_golden else 'FAIL'
        if status_vv == 'FAIL':
            fails.append(f'vulkan-vs-golden RMS={rms_vv:.2f} > {tol_golden}')
        print(f"  vulkan-vs-golden:  RMS {rms_vv:6.2f}  threshold {tol_golden}  {status_vv}")
    else:
        print(f"  vulkan-vs-golden:  no golden (run --update to create)")
        rms_vv = float('nan')
        status_vv = 'NO_GOLDEN'

    if g_ovrtx.exists():
        rms_oo = _rms(cur_ovrtx, _load_rgb(g_ovrtx))
        status_oo = 'PASS' if rms_oo <= tol_golden else 'FAIL'
        if status_oo == 'FAIL':
            fails.append(f'ovrtx-vs-golden RMS={rms_oo:.2f} > {tol_golden}')
        print(f"  ovrtx-vs-golden:   RMS {rms_oo:6.2f}  threshold {tol_golden}  {status_oo}")
    else:
        print(f"  ovrtx-vs-golden:   no golden (run --update to create)")
        rms_oo = float('nan')
        status_oo = 'NO_GOLDEN'

    status_cross = 'PASS' if rms_cross <= tol_ovrtx else 'FAIL'
    if status_cross == 'FAIL':
        fails.append(f'vulkan-vs-ovrtx-current RMS={rms_cross:.2f} > {tol_ovrtx}')
    print(f"  vulkan-vs-ovrtx:   RMS {rms_cross:6.2f}  threshold {tol_ovrtx}  {status_cross}")

    overall = 'FAIL' if fails else 'PASS'
    print(f"  {overall}  ({wall_ms:.0f} ms)")
    return {
        'label': full_label, 'status': overall,
        'rms_vulkan_vs_golden': rms_vv,
        'rms_ovrtx_vs_golden':  rms_oo,
        'rms_vulkan_vs_ovrtx':  rms_cross,
        'failures': fails,
        'wall_ms': wall_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true',
                    help='Rewrite goldens from current renders (then exit)')
    ap.add_argument('--only', help='Filter to tests whose label contains this string')
    args = ap.parse_args()

    results = []
    for label, usd, mode, w, h, frames, tol_g, tol_o in TESTS:
        if args.only and args.only not in f"{label}_{_MODE_NAMES.get(mode, mode)}":
            continue
        results.append(run_one(label, usd, mode, w, h, frames,
                               tol_g, tol_o, args.update))

    summary = {'tests': results}
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    if args.update:
        print(f"\n{len(results)} golden(s) updated.")
        return 0

    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')
    n_skip = sum(1 for r in results if r['status'] == 'SKIP')
    n_nog  = sum(1 for r in results if r['status'] == 'NO_GOLDEN')

    print()
    print("=" * 60)
    print(f"  {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_nog} no golden")
    print("=" * 60)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
