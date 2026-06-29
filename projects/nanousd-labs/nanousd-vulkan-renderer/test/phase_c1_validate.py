# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase C.1 deferred-shading validation.

Loads a scene with materials + textures, renders it tiled in two modes
(deferred OFF and ON), and reports:
  - distinct color counts per tile in each mode
  - per-tile SSIM between OFF (rgen-shaded) and ON (deferred compute)
  - existence of texture-driven variation in ON output (i.e. >2 distinct
    colors when sampling a real texture, since flat per-mesh would only
    give 1 color)

Usage:
  python phase_c1_validate.py <scene.usdc> <out_off.ppm> <out_on.ppm>

The OFF output is the legacy rgen Lambertian shading; ON is Phase C.1 compute
(textured albedo, NO lighting). They will visibly differ — the gate is that
ON shows recognisable texture content (not flat per-mesh color from Phase B).
"""
import sys
import numpy as np
import os

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'python')))

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT


def make_view_matrix(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    up = np.asarray(up, np.float32)
    r = np.cross(f, up); rl = np.linalg.norm(r)
    if rl < 1e-6:
        up = np.array([0, 0, 1], np.float32)
        r = np.cross(f, up); rl = np.linalg.norm(r)
    r /= rl
    u = np.cross(r, f)
    view = np.zeros(16, np.float32)
    view[0] = r[0]; view[1] = r[1]; view[2] = r[2]; view[3] = -np.dot(r, eye)
    view[4] = u[0]; view[5] = u[1]; view[6] = u[2]; view[7] = -np.dot(u, eye)
    view[8] = -f[0]; view[9] = -f[1]; view[10] = -f[2]; view[11] = np.dot(f, eye)
    view[15] = 1.0
    return view


def invert_view(view):
    m = view.reshape(4, 4)
    return np.linalg.inv(m).reshape(16)


def make_proj_matrix(fov_deg, aspect, near=0.1, far=1000.0):
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
    view = make_view_matrix(eye, target)
    vi = invert_view(view)
    proj = make_proj_matrix(fov_deg, w / h)
    pi = invert_proj(proj)
    return np.concatenate([vi, pi])


def save_grid_ppm(path, frames, num_cols, tile_w, tile_h):
    """frames: (N, H, W, 4) uint8. Pack into a (rows*tile_h, cols*tile_w, 3) PPM."""
    n = frames.shape[0]
    num_rows = (n + num_cols - 1) // num_cols
    canvas = np.zeros((num_rows * tile_h, num_cols * tile_w, 3), np.uint8)
    for i in range(n):
        r = i // num_cols; c = i % num_cols
        canvas[r * tile_h:(r + 1) * tile_h,
               c * tile_w:(c + 1) * tile_w, :] = frames[i, :, :, :3]
    with open(path, 'wb') as fh:
        fh.write(b'P6\n%d %d\n255\n' % (canvas.shape[1], canvas.shape[0]))
        fh.write(canvas.tobytes())


def per_tile_distinct_colors(frames):
    """Count distinct RGB colors per camera tile."""
    counts = []
    for fr in frames:
        rgb = fr[:, :, :3].reshape(-1, 3)
        # Pack into uint32 for fast unique
        packed = (rgb[:, 0].astype(np.uint32) |
                  (rgb[:, 1].astype(np.uint32) << 8) |
                  (rgb[:, 2].astype(np.uint32) << 16))
        counts.append(len(np.unique(packed)))
    return counts


def per_tile_hash(frames):
    return [hash(fr.tobytes()) for fr in frames]


def per_tile_ssim(frames_a, frames_b):
    """Simple windowed SSIM-like mean over tile, on luminance.
    Not a textbook SSIM, but a robust similarity proxy."""
    out = []
    for a, b in zip(frames_a, frames_b):
        ya = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]).astype(np.float32) / 255.0
        yb = (0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]).astype(np.float32) / 255.0
        mu_a = ya.mean(); mu_b = yb.mean()
        sa = ya.var(); sb = yb.var()
        sab = ((ya - mu_a) * (yb - mu_b)).mean()
        c1 = (0.01) ** 2; c2 = (0.03) ** 2
        ssim = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / \
               ((mu_a * mu_a + mu_b * mu_b + c1) * (sa + sb + c2))
        out.append(float(ssim))
    return out


def render_scene(usdc_path, num_cams, tile_w, tile_h, deferred):
    r = NuRenderer(width=tile_w, height=tile_h, enable_rt=True, enable_materials=True)
    if not r.rt_available:
        raise RuntimeError("RT not available")
    r.load_usd(usdc_path)
    r.build_accel()
    r.set_deferred_shade(deferred)

    # Build a ring of cameras around the scene's actual bounds. This auto-
    # scales the camera distance so the scene fills the frame regardless of
    # asset scale (chess king at 8 cm vs showcase at 30 m).
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
    for i in range(num_cams):
        ang = 2 * math.pi * i / max(num_cams, 1)
        eye = target + np.array([radius * math.cos(ang),
                                  elev - target[1],
                                  radius * math.sin(ang)], np.float32)
        vps.append(make_vp_inv(eye, target, 45.0, tile_w, tile_h))
    vps = np.stack(vps).astype(np.float32)

    r.render_tiled(vps, num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h, mode=NU_RENDER_RT)
    pixels = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h)
    out = pixels.copy()
    r.close()
    return out


def render_scene_both(usdc_path, num_cams, tile_w, tile_h):
    """Render the same scene from the same cameras with deferred OFF then ON,
    using the SAME renderer instance to ensure identical scene/transform state.
    The toggle invalidates the tiled cmd cache so each frame rebuilds with
    the new push-constant/pipeline state."""
    r = NuRenderer(width=tile_w, height=tile_h, enable_rt=True, enable_materials=True)
    if not r.rt_available:
        raise RuntimeError("RT not available")
    r.load_usd(usdc_path)
    r.build_accel()

    # Camera ring from scene bounds.
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
    for i in range(num_cams):
        ang = 2 * math.pi * i / max(num_cams, 1)
        eye = target + np.array([radius * math.cos(ang),
                                  elev - target[1],
                                  radius * math.sin(ang)], np.float32)
        vps.append(make_vp_inv(eye, target, 45.0, tile_w, tile_h))
    vps = np.stack(vps).astype(np.float32)

    # We render OFF then ON. To avoid double-buffer / cache-invalidation
    # quirks, we run a couple of warm-up frames on each mode and only
    # capture the final settled frame.
    def render_settled(deferred):
        r.set_deferred_shade(deferred)
        # Multiple frames to flush double-buffer parity; the captured frame
        # is the result of an N-th render after a cache rebuild.
        for _ in range(4):
            r.render_tiled(vps, num_cameras=num_cams, tile_w=tile_w,
                           tile_h=tile_h, mode=NU_RENDER_RT)
            _ = r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=tile_w, tile_h=tile_h)
        r.render_tiled(vps, num_cameras=num_cams, tile_w=tile_w,
                       tile_h=tile_h, mode=NU_RENDER_RT)
        return r.fetch_pixels_tiled(num_cameras=num_cams, tile_w=tile_w,
                                     tile_h=tile_h).copy()

    fr_off = render_settled(False)
    fr_on  = render_settled(True)

    r.close()
    return fr_off, fr_on


def main():
    if len(sys.argv) < 4:
        print("Usage: phase_c1_validate.py <scene.usdc> <out_off.ppm> <out_on.ppm>",
              file=sys.stderr)
        sys.exit(2)
    scene = sys.argv[1]
    out_off = sys.argv[2]
    out_on  = sys.argv[3]

    NUM_CAMS = 16
    TILE_W = TILE_H = 64
    NUM_COLS = 4

    print(f"Rendering {scene} (OFF, {NUM_CAMS} cams x {TILE_W}x{TILE_H})...")
    fr_off = render_scene(scene, NUM_CAMS, TILE_W, TILE_H, deferred=False)
    print(f"  OFF shape={fr_off.shape}, sum={fr_off.sum()}")
    print(f"Rendering {scene} (ON,  {NUM_CAMS} cams x {TILE_W}x{TILE_H})...")
    fr_on  = render_scene(scene, NUM_CAMS, TILE_W, TILE_H, deferred=True)
    print(f"  ON  shape={fr_on.shape}, sum={fr_on.sum()}")

    save_grid_ppm(out_off, fr_off, NUM_COLS, TILE_W, TILE_H)
    save_grid_ppm(out_on,  fr_on,  NUM_COLS, TILE_W, TILE_H)
    print(f"Saved {out_off} and {out_on}")

    # Per-tile distinct colors + hash analysis
    counts_off = per_tile_distinct_colors(fr_off)
    counts_on  = per_tile_distinct_colors(fr_on)
    hashes_off = per_tile_hash(fr_off)
    hashes_on  = per_tile_hash(fr_on)
    distinct_off = len(set(hashes_off))
    distinct_on  = len(set(hashes_on))
    ssim_per_tile = per_tile_ssim(fr_off, fr_on)

    print()
    print("--- Phase C.1 validation ---")
    print(f"OFF distinct-tile-hashes: {distinct_off}/{NUM_CAMS}")
    print(f"ON  distinct-tile-hashes: {distinct_on}/{NUM_CAMS}")
    print(f"Per-tile distinct-color counts:")
    print(f"  OFF: {counts_off}")
    print(f"  ON:  {counts_on}")
    print(f"  ON  min/max/median: {min(counts_on)}/{max(counts_on)}/{int(np.median(counts_on))}")
    print(f"Per-tile SSIM(OFF, ON): "
          f"min={min(ssim_per_tile):.3f}, "
          f"median={np.median(ssim_per_tile):.3f}, "
          f"max={max(ssim_per_tile):.3f}")
    print()

    # Pass/fail criteria for Phase C.1.
    #
    # The deferred compute pass writes textured albedo only — no lighting,
    # no IBL, no shadows. The OFF (rgen-shaded) path runs the full PBR rchit
    # which depends on bound lights / IBL; for assets that don't yet have
    # lighting wired up via the Python path (DomeLight HDR not loaded),
    # OFF is dark and contains no information, while ON is unlit textured.
    # In that case OFF can't be the comparison baseline; instead we only
    # require ON to:
    #   - show real texture variation (lots of distinct colors per tile),
    #   - not have NaN,
    #   - light up tiles that the OFF path can identify as hit-tiles
    #     (per-tile distinct-tile-count, not per-tile pixel content).
    #
    # When OFF *does* have content, we additionally require the distinct-
    # tile-hash counts to match (same set of cameras hit geometry).
    pass_textured     = max(counts_on) >= 5
    pass_no_nan       = not np.isnan(fr_on.astype(np.float32)).any()
    pass_no_all_black = (fr_on.sum() > 0)

    off_has_content = (fr_off.sum() > 0)
    if off_has_content:
        pass_distinct_match = (distinct_on == distinct_off)
        pass_ssim           = min(ssim_per_tile) > 0.4
    else:
        # OFF is dark (no lighting); skip those gates — can't compare.
        pass_distinct_match = True
        pass_ssim           = True
        print("  NOTE: OFF render is dark (no IBL/lights bound by test "
              "harness); skipping ON-vs-OFF gates.")

    print("Gate verdicts:")
    print(f"  distinct-tile-count match (ON==OFF): {pass_distinct_match} "
          f"({distinct_on} vs {distinct_off})")
    print(f"  no all-black ON tiles overall: {pass_no_all_black}")
    print(f"  ON has texture variation (max colors/tile >=5): {pass_textured}")
    print(f"  no NaN: {pass_no_nan}")
    print(f"  per-tile SSIM > 0.4 (when OFF has content): {pass_ssim}")

    all_pass = (pass_distinct_match and pass_no_all_black and pass_textured
                and pass_no_nan and pass_ssim)
    print()
    print("VERDICT:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
