#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VK3DGRT visual regression tests.

Mirrors tests/correctness/run_tests.py but for the splat path. Each test
renders a deterministic Gaussian-splat scene through nu_gs_render, then
checks the resulting RGBA PNG against a checked-in golden using per-pixel
RMS. This catches regressions in:

  - particle SSBO upload / staging layout
  - gs_as_build compute push-constant transpose (Slang column-major vs C
    row-major); a flipped transpose would shift every splat
  - TLAS instance custom index plumbing (a swap would re-order the
    grayscale gradient our rchit paints)
  - rt_image -> swapchain blit inside gpu_gs_render_tile
  - SBT entry order / layout

The current rgen body is the Pragmatist gate (paint by InstanceID); when
Phase 3's K-buffer trace loop replaces it, run with --update once and
re-commit the golden.

Usage:
    python3 run_tests.py             # run + verify
    python3 run_tests.py --update    # rewrite goldens from current renders
    python3 run_tests.py --only grid12  # filter to one test
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RENDERER_REPO = HERE.parent.parent
sys.path.insert(0, str(RENDERER_REPO / "python"))

from nusd_renderer import _bindings as B  # noqa: E402

GOLDEN_DIR = HERE / "golden"
OUTPUT_DIR = HERE / "output"
GOLDEN_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-pixel RMS threshold. Splat path is fully determinate (no jitter, no
# random sampling), so drift across drivers should stay within ~1 LSB.
TOL_RMS = 2.0


def _make_grid12():
    """12 splats in a 4x3 grid (XY plane, z=0) with distinct SH-DC per splat
    so the test image carries enough chromatic variation to catch
    per-particle SSBO indexing / SH layout regressions in addition to the
    geometric smoke check.

    Per-particle SH coefficients are laid out so:
      - column index (0..3) varies the R channel from -1 to +1
      - row index    (0..2) varies the G channel from -1 to +1
      - B channel is constant 0.5
    SH-DC reconstructs as `0.5 + C0 * sh0` per band, so the rendered
    color sweeps a 12-cell color grid.
    """
    cols, rows = 4, 3
    N = cols * rows
    positions = np.array(
        [[x, y, 0.0]
         for y in (-0.4, 0.0, 0.4)
         for x in (-0.6, -0.2, 0.2, 0.6)],
        dtype=np.float32,
    )
    scales = np.full((N, 3), 0.1, dtype=np.float32)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 0] = 1.0  # wxyz identity

    opacities = np.full(N, 0.9, dtype=np.float32)
    sh = np.zeros((N, 1, 3), dtype=np.float32)
    for i in range(N):
        cx = i % cols
        cy = i // cols
        sh[i, 0, 0] = (cx / float(cols - 1)) * 2.0 - 1.0  # R sweep
        sh[i, 0, 1] = (cy / float(rows - 1)) * 2.0 - 1.0  # G sweep
        sh[i, 0, 2] = 0.5
    return positions, scales, orientations, opacities, sh, 0, None  # default camera


def _make_stack32():
    """32 small low-α splats stacked along Z, plus an explicit camera
    pointing down -Z so the central pixel's ray hits all 32 in order.

    With K=16, pass 1 fills the K-buffer with the 16 closest, pass 2
    re-traces from `tMin = lastDist + epsT` to pick up the next 16. This
    is the only fixture that actually exercises the rgen multi-pass
    loop — grid12 has < K overlaps per ray and runs single-pass.

    Opacity is kept low (0.1) so transmittance stays above min_T (0.03)
    after 16 splats: T₁₆ = 0.9¹⁶ ≈ 0.185, T₃₂ ≈ 0.034. SH-DC R sweeps
    from -1 → +1 across the stack so the depth ordering is visible in
    the per-pixel α-blend gradient."""
    N = 32
    z_vals = np.linspace(-0.4, 0.4, N).astype(np.float32)
    positions = np.zeros((N, 3), dtype=np.float32)
    positions[:, 2] = z_vals
    scales = np.full((N, 3), 0.08, dtype=np.float32)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    opacities = np.full(N, 0.1, dtype=np.float32)
    sh = np.zeros((N, 1, 3), dtype=np.float32)
    sh[:, 0, 0] = np.linspace(-1.0, 1.0, N).astype(np.float32)  # R depth-gradient
    sh[:, 0, 1] = 0.5
    sh[:, 0, 2] = 0.5
    # Explicit camera at +Z, looking at origin. Ray through center pixel
    # goes down -Z and hits every splat.
    camera = dict(eye=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0),
                  up=(0.0, 1.0, 0.0), fov_degrees=45.0)
    return positions, scales, orientations, opacities, sh, 0, camera


def _make_viewdep4():
    """4 large splats at the corners of a square in the XY plane, each with
    sh_degree=1 SH coefficients chosen so the splat color *changes with
    view-direction*. Splat A at (-0.4, -0.4, 0) has band-1 X-coefficient
    set; splat B at (+0.4, -0.4, 0) has band-1 Y; etc. The camera is
    pulled in close enough that the per-splat view direction differs
    materially across the image, which propagates to differences in the
    band-1 contribution. Validates the full SH eval path (gs_eval_sh
    branch in rgen)."""
    N = 4
    positions = np.array([
        [-0.4, -0.4, 0.0],
        [+0.4, -0.4, 0.0],
        [-0.4, +0.4, 0.0],
        [+0.4, +0.4, 0.0],
    ], dtype=np.float32)
    scales = np.full((N, 3), 0.18, dtype=np.float32)
    orientations = np.zeros((N, 4), dtype=np.float32)
    orientations[:, 0] = 1.0
    opacities = np.full(N, 0.95, dtype=np.float32)

    # SH layout per particle: (deg+1)^2 = 4 coefficients × 3 channels = 12 floats.
    # Band 0 (DC at offset 0): keep as 0.5 → reconstructs to ~0.64 baseline.
    # Band 1 coefficients (l=1, m=-1,0,+1) at offsets 1, 2, 3:
    #   m=-1 multiplied by -y, m=0 by z, m=+1 by -x.
    # Each splat gets one band-1 coefficient lit on R only, so they
    # carry a distinctive view-dependent red tint.
    sh = np.zeros((N, 4, 3), dtype=np.float32)
    sh[:, 0, :] = 0.5  # DC
    sh[0, 3, 0] = 2.0  # splat 0: -x lobe in R
    sh[1, 3, 0] = -2.0  # splat 1: +x lobe in R
    sh[2, 1, 0] = 2.0  # splat 2: -y lobe in R
    sh[3, 1, 0] = -2.0  # splat 3: +y lobe in R

    camera = dict(eye=(0.0, 0.0, 1.5), target=(0.0, 0.0, 0.0),
                  up=(0.0, 1.0, 0.0), fov_degrees=60.0)
    return positions, scales, orientations, opacities, sh, 1, camera


def _make_real_synth():
    """Round-trips a synthetic 3DGS-format PLY through the
    `tools/ply_to_usd.py` converter (parse path) into nu_gs_set_particles.
    Validates the input contract end-to-end: PLY layout (xyz + nx/y/z +
    f_dc + f_rest + opacity + scale + rot), σ-from-log-σ exp(),
    opacity-from-raw sigmoid, wxyz quaternion normalize, and SH layout
    shuffle (PLY's per-channel rest layout → AOUSD's
    [particle][coeff][channel] flat). When a real PLY drops in (e.g.
    from `references/3dgrut`), swap the synth call for it and the rest
    of the harness reuses unchanged."""
    import sys as _sys
    tools_dir = str(RENDERER_REPO / "tools")
    if tools_dir not in _sys.path:
        _sys.path.insert(0, tools_dir)
    from ply_to_usd import synthesize_ply, parse_3dgs_ply

    synth_path = OUTPUT_DIR / "synth.ply"
    synthesize_ply(synth_path, N=16, sh_degree=1, seed=42)
    scene = parse_3dgs_ply(synth_path)

    camera = dict(eye=(0.0, 0.0, 1.5), target=(0.0, 0.0, 0.0),
                  up=(0.0, 1.0, 0.0), fov_degrees=60.0)
    return (scene.positions, scene.scales, scene.orientations,
            scene.opacities, scene.sh, scene.sh_degree, camera)


_FIXTURES = {
    "grid12":     _make_grid12,
    "stack32":    _make_stack32,
    "viewdep4":   _make_viewdep4,
    "real_synth": _make_real_synth,
}


def _render(name, w, h, proxy=B.NU_GS_PROXY_ICOSAHEDRON):
    # Tests sharing a fixture across proxy kinds register the proxy
    # variant via the test entry; the fixture builder name stays the
    # base name (so we don't have to duplicate _make_grid12 just to
    # toggle the proxy).
    fixture_name = name.split("_aabb")[0] if name.endswith("_aabb") else name
    pos, scl, ori, opa, sh, sh_deg, camera = _FIXTURES[fixture_name]()

    r = B.NuRenderer(width=w, height=h, enable_rt=True, visible=False)
    try:
        r.gs_set_particles(pos, scl, ori, opa, sh, sh_degree=sh_deg)
        if proxy != B.NU_GS_PROXY_ICOSAHEDRON:
            r.gs_set_proxy(proxy)
        if camera is not None:
            r.set_camera_explicit(**camera)
        r.gs_render(cam_id=0)

        import ctypes
        buf = (ctypes.c_uint8 * (w * h * 4))()
        rc = B._Lib.get().nu_fetch_pixels(r._handle, buf, B.NU_PIXEL_RGBA8)
        if rc != B.NU_OK:
            raise RuntimeError(f"nu_fetch_pixels failed: rc={rc}")
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()

        # Iso-opacity depth + normal: returned alongside the color image
        # so tests can guard against regressions in either AOV.
        depth = r.gs_fetch_depth()  if hasattr(r, "gs_fetch_depth")  else None
        normal = r.gs_fetch_normal() if hasattr(r, "gs_fetch_normal") else None
    finally:
        r.close()

    out_png = OUTPUT_DIR / f"{name}.png"
    Image.fromarray(img).save(out_png, optimize=True)
    return img, depth, normal, out_png


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    return float(np.sqrt((diff * diff).mean()))


# (label, width, height, sanity-check minimum number of distinct R values)
# grid12: with K-buffer α-compositing the splats have smooth Gaussian
# falloff, producing >100 unique R values per render. Threshold 50 is a
# conservative "the K-buffer alpha-blend is alive" check — if rahit
# stops firing (e.g., someone re-adds VK_GEOMETRY_OPAQUE_BIT_KHR to the
# unit BLAS, or drops rahit from the hit group), unique-R collapses to
# the SH-DC plateaus (~5) and this catches it before the RMS diff fires.
#
# stack32: 32 splats stacked along Z, viewed head-on. Exercises the
# rgen multi-pass loop because the central ray hits > K=16 splats. If
# multi-pass is broken, the central pixel cluster shows truncated
# composition (transmittance plateau halfway through the stack) rather
# than the smooth color sweep that lands when both passes complete.
# Test entries: (label, w, h, min_unique_R[, proxy_kind]). Proxy defaults
# to icosahedron; the *_aabb variants exercise the procedural-AABB BLAS +
# rint shader path. AABB and icosa give visually similar but not bit-
# identical output (different proxy bounds → different first-hit t-values
# at silhouette edges), so each variant gets its own golden.
TESTS = [
    ("grid12",        256, 256, 50, B.NU_GS_PROXY_ICOSAHEDRON),
    ("stack32",       128, 128, 30, B.NU_GS_PROXY_ICOSAHEDRON),
    ("viewdep4",      256, 256, 50, B.NU_GS_PROXY_ICOSAHEDRON),
    ("grid12_aabb",   256, 256, 50, B.NU_GS_PROXY_AABB),
    # Round-trip validation of tools/ply_to_usd.py converter against a
    # synthetic 3DGS-format PLY (16 particles, sh_degree=1). Catches PLY
    # parse / σ-exp / opacity-sigmoid / SH-layout regressions.
    ("real_synth",    256, 256, 30, B.NU_GS_PROXY_ICOSAHEDRON),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite goldens from current renders")
    ap.add_argument("--only", help="run only the test with this label")
    args = ap.parse_args()

    failures = 0
    ran = 0
    for entry in TESTS:
        label, w, h, min_unique = entry[0], entry[1], entry[2], entry[3]
        proxy = entry[4] if len(entry) > 4 else B.NU_GS_PROXY_ICOSAHEDRON
        if args.only and args.only != label:
            continue
        ran += 1
        print(f"[{label}] rendering {w}x{h} ...", end=" ", flush=True)
        img, depth, normal, out_png = _render(label, w, h, proxy=proxy)

        # Baseline sanity: image must contain enough distinct values to
        # prove InstanceID plumbing is alive (each splat paints a unique
        # gray).
        n_unique = len(np.unique(img[:, :, 0]))
        if n_unique < min_unique:
            print(f"FAIL: only {n_unique} distinct R values "
                  f"(expected >= {min_unique}; pipeline regressed)")
            failures += 1
            continue

        golden = GOLDEN_DIR / f"{label}.png"
        if args.update or not golden.exists():
            existed = golden.exists()
            Image.fromarray(img).save(golden, optimize=True)
            print(f"{'UPDATED' if existed else 'CREATED'} {golden.name}")
            continue

        ref = np.array(Image.open(golden))
        if ref.shape != img.shape:
            print(f"FAIL: golden {ref.shape} vs current {img.shape} mismatch")
            failures += 1
            continue
        rms = _rms(img, ref)
        ok = rms <= TOL_RMS

        # Depth output sanity (plan §7). Cheap invariants: shape, all
        # values are either -1 (sky / never crossed iso) or finite
        # positive t in [tMin, tMax] = [0.001, 10000]. Catches NaN,
        # negative-t-other-than-sky, or buffer-mis-binding regressions.
        depth_msg = ""
        if depth is not None:
            if depth.shape != (h, w):
                print(f"FAIL: depth shape {depth.shape} expected ({h}, {w})")
                failures += 1; continue
            if not np.all(np.isfinite(depth)):
                print("FAIL: depth contains non-finite values")
                failures += 1; continue
            valid = depth > 0
            sky   = depth < 0
            other = ~(valid | sky)
            if other.any():
                print(f"FAIL: depth has {int(other.sum())} pixels neither sky nor positive-t")
                failures += 1; continue
            if valid.any():
                t_max_observed = float(depth[valid].max())
                if t_max_observed > 10000.0 + 1.0:
                    print(f"FAIL: depth max {t_max_observed} outside expected tMax")
                    failures += 1; continue
            depth_msg = f" depth-valid={int(valid.sum())}"

        # Iso-opacity normal sanity. (0, 0, 0) on miss; otherwise expected
        # to be unit length (±1 LSB). NaN / non-unit values flag a
        # regression in the analytic gradient or storage layout.
        normal_msg = ""
        if normal is not None:
            if normal.shape != (h, w, 3):
                print(f"FAIL: normal shape {normal.shape} expected ({h}, {w}, 3)")
                failures += 1; continue
            if not np.all(np.isfinite(normal)):
                print("FAIL: normal contains non-finite values")
                failures += 1; continue
            n_len = np.linalg.norm(normal, axis=-1)
            valid_n = n_len > 0.5  # treat <0.5 as "miss / undefined"
            if valid_n.any():
                len_at_hits = n_len[valid_n]
                if (len_at_hits.min() < 0.999 or len_at_hits.max() > 1.001):
                    print(f"FAIL: normal not unit length at hits "
                          f"(min={len_at_hits.min():.4f}, max={len_at_hits.max():.4f})")
                    failures += 1; continue
            normal_msg = f" normal-valid={int(valid_n.sum())}"

        print(f"RMS={rms:.3f} (tol {TOL_RMS}) "
              f"unique-R={n_unique}{depth_msg}{normal_msg} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    if ran == 0:
        print("No tests ran"); return 2
    if failures:
        print(f"\n{failures}/{ran} test(s) failed")
        return 1
    print(f"\n{ran} test(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
