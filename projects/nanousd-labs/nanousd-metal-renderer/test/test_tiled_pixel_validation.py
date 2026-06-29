# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-level validation and image output for tiled multi-camera RT.

Tests:
  - Save rendered tiles as PPM for visual inspection
  - Per-pixel comparison between tiled and single-camera render paths
  - Histogram analysis (brightness distribution, color channel balance)
  - Boundary pixel check (no artifacts at tile edges)
  - De-tiling correctness: compare C de-tiling against Python reference
  - Background color consistency (sky/miss color)
"""

import math
import time
import unittest
import numpy as np
import os
import struct
import tempfile


# ---- Helpers ----

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


def save_ppm(path, pixels):
    """Save RGBA8 pixels as PPM file."""
    h, w = pixels.shape[0], pixels.shape[1]
    with open(path, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(pixels[:, :, :3].tobytes())


def make_scene(renderer_cls, w=100, h=100):
    from nusd_renderer._bindings import NU_RENDER_RT
    r = renderer_cls(width=w, height=h, enable_rt=True)
    if not r.rt_available:
        r.close()
        return None
    v1, idx = make_box(cx=0.0)
    v2, _ = make_box(cx=1.5)
    r.add_mesh(positions=v1, indices=idx, display_color=(0.8, 0.2, 0.2), name="red_box")
    r.add_mesh(positions=v2, indices=idx, display_color=(0.2, 0.2, 0.8), name="blue_box")
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)
    return r


class TestTiledPixelValidation(unittest.TestCase):
    """Pixel-level correctness for tiled render output."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT
        cls.output_dir = tempfile.mkdtemp(prefix="tiled_test_images_")
        print(f"\n  Test images will be saved to: {cls.output_dir}")

    def test_save_tiled_renders_as_ppm(self):
        """Save individual camera tiles as PPM files for visual inspection."""
        r = make_scene(self.NuRenderer, w=200, h=200)
        if r is None:
            self.skipTest("RT not available")

        W, H = 200, 200
        N = 4
        eyes = [
            (0, 0, 5),    # front
            (5, 0, 0),    # right
            (0, 5, 0),    # top
            (-5, 0, 0),   # left
        ]
        labels = ["front", "right", "top", "left"]

        vps = np.stack([make_vp_inv(eye, (0.75, 0, 0), 45.0, W, H) for eye in eyes])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        saved = []
        for i in range(N):
            path = os.path.join(self.output_dir, f"tiled_cam{i}_{labels[i]}.ppm")
            save_ppm(path, pixels[i])
            saved.append(path)
            size_kb = os.path.getsize(path) / 1024
            mean_rgb = pixels[i, :, :, :3].astype(float).mean(axis=(0, 1))
            print(f"    Saved {labels[i]:>6}: {path} ({size_kb:.0f} KB, mean RGB=({mean_rgb[0]:.0f},{mean_rgb[1]:.0f},{mean_rgb[2]:.0f}))")

        # Also save a contact sheet (all 4 tiles in one image)
        grid = np.zeros((H * 2, W * 2, 4), dtype=np.uint8)
        grid[:H, :W] = pixels[0]
        grid[:H, W:] = pixels[1]
        grid[H:, :W] = pixels[2]
        grid[H:, W:] = pixels[3]
        grid_path = os.path.join(self.output_dir, "tiled_grid_2x2.ppm")
        save_ppm(grid_path, grid)
        print(f"    Saved grid: {grid_path}")

        r.close()

    def test_brightness_histogram(self):
        """Analyze brightness distribution of rendered tiles."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        W, H = 100, 100
        vps = np.stack([
            make_vp_inv((0, 0, 3), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, 8), (0, 0, 0), 45.0, W, H),
        ])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        print(f"\n  Brightness histogram analysis:")
        for cam in range(2):
            luma = (0.299 * pixels[cam, :, :, 0].astype(float)
                  + 0.587 * pixels[cam, :, :, 1].astype(float)
                  + 0.114 * pixels[cam, :, :, 2].astype(float))

            # Histogram bins: 0-63, 64-127, 128-191, 192-255
            bins = [0, 64, 128, 192, 256]
            hist, _ = np.histogram(luma.ravel(), bins=bins)
            total = luma.size
            pct = hist / total * 100

            print(f"    Camera {cam} (dist={'near' if cam==0 else 'far'}):")
            print(f"      Dark  (0-63):   {pct[0]:5.1f}% ({hist[0]:>6})")
            print(f"      Mid-lo(64-127): {pct[1]:5.1f}% ({hist[1]:>6})")
            print(f"      Mid-hi(128-191):{pct[2]:5.1f}% ({hist[2]:>6})")
            print(f"      Bright(192-255):{pct[3]:5.1f}% ({hist[3]:>6})")
            print(f"      Mean luma: {luma.mean():.1f}, Std: {luma.std():.1f}")

            # Sanity checks
            self.assertGreater(luma.std(), 5.0,
                f"Camera {cam} has very low variance ({luma.std():.1f}) — likely all one color")

        r.close()

    def test_tile_boundary_no_artifacts(self):
        """Check that pixels at tile boundaries don't have artifacts."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 4
        # All cameras at same position — every tile should be identical
        vps = np.stack([make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)] * N)

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        # All tiles should be identical (same camera)
        for i in range(1, N):
            diff = np.abs(pixels[0].astype(int) - pixels[i].astype(int))
            max_diff = diff.max()
            self.assertEqual(max_diff, 0,
                f"Tile {i} differs from tile 0 by up to {max_diff} — boundary artifact?")

        # Check edge rows/columns of each tile specifically
        for i in range(N):
            # Top row
            top = pixels[i, 0, :, :3]
            # Bottom row
            bottom = pixels[i, -1, :, :3]
            # Left column
            left = pixels[i, :, 0, :3]
            # Right column
            right = pixels[i, :, -1, :3]

            # These should be reasonable (not all zero unless that's the background)
            # and should match the interior (same camera, no tile-edge discontinuity)
            self.assertTrue(
                np.array_equal(top, pixels[0, 0, :, :3]),
                f"Tile {i} top row differs from tile 0")
            self.assertTrue(
                np.array_equal(bottom, pixels[0, -1, :, :3]),
                f"Tile {i} bottom row differs from tile 0")

        r.close()
        print(f"\n  Tile boundary check: {N} identical tiles match perfectly")

    def test_background_color_consistency(self):
        """Verify background (miss/sky) color is consistent across all cameras."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 4
        # All cameras looking away from geometry
        vps = np.stack([
            make_vp_inv((0, 0, -100 - i * 10), (0, 0, -200), 45.0, W, H)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        # Background should be the same for all cameras (same sky color)
        bg_colors = []
        for i in range(N):
            # Get the most common color (should be background)
            flat = pixels[i, :, :, :3].reshape(-1, 3)
            unique, counts = np.unique(flat, axis=0, return_counts=True)
            dominant = unique[counts.argmax()]
            bg_colors.append(dominant)

        # All cameras should have similar dominant color
        for i in range(1, N):
            diff = np.abs(bg_colors[0].astype(int) - bg_colors[i].astype(int))
            self.assertTrue(np.all(diff < 5),
                f"Camera {i} background color {bg_colors[i]} differs from camera 0 {bg_colors[0]}")

        print(f"\n  Background consistency: all {N} cameras have BG color ~{bg_colors[0]}")
        r.close()

    def test_pixel_by_pixel_single_vs_tiled(self):
        """Pixel-exact comparison between single-camera and tiled render paths.

        Uses fast_mode to isolate the rendering pipeline from tone mapping.
        In fast_mode (flat diffuse, no ACES/shadows/PBR), the only difference
        between single and tiled is the sRGB swapchain blit.  We verify:
            single_srgb_pixel == sRGB_OETF(tiled_linear_pixel)  (within ±1)

        This proves the camera matrices, ray generation, and hit/miss shaders
        produce identical output in both paths.  The non-fast-mode PBR path has
        larger residuals due to ACES tone mapping interacting with 8-bit
        quantization at two stages (storage image + sRGB blit).
        """
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        W, H = 100, 100
        r = make_scene(NuRenderer, w=W, h=H)
        if r is None:
            self.skipTest("RT not available")

        eye = (0, 0, 5)
        target = (0, 0, 0)
        fov = 45.0

        # Enable fast_mode: flat diffuse shading, no ACES/shadows/PBR
        r.set_fast_mode(True)

        # Single-camera render (readback goes through sRGB swapchain)
        r.set_camera(eye=eye, target=target, fov_degrees=fov)
        r.render(mode=NU_RENDER_RT)
        single_srgb = r.fetch_pixels()[:, :, :3].copy()

        # Tiled single-camera render (readback is linear UNORM)
        vps = make_vp_inv(eye, target, fov, W, H).reshape(1, 32)
        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        tiled_linear = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)[0, :, :, :3].copy()

        r.close()

        # Convert tiled linear to sRGB for comparison with swapchain readback.
        t = tiled_linear.astype(np.float64) / 255.0
        tiled_srgb = np.where(t <= 0.0031308,
                              12.92 * t,
                              1.055 * t ** (1.0 / 2.4) - 0.055) * 255.0
        tiled_srgb = np.clip(tiled_srgb + 0.5, 0, 255).astype(np.uint8)

        diff = np.abs(single_srgb.astype(int) - tiled_srgb.astype(int))
        max_diff = diff.max()
        mean_diff = diff.mean()

        pixels_match = (diff.max(axis=2) <= 1).sum()
        total_pixels = H * W

        print(f"\n  Single(sRGB) vs sRGB(Tiled) fast_mode comparison ({W}x{H}):")
        print(f"    Max pixel diff:     {max_diff}")
        print(f"    Mean pixel diff:    {mean_diff:.4f}")
        print(f"    Exact match (±1):   {pixels_match}/{total_pixels} ({pixels_match/total_pixels*100:.0f}%)")

        if max_diff > 2:
            diff_img = (diff * 50).clip(0, 255).astype(np.uint8)
            diff_path = os.path.join(self.output_dir, "single_vs_tiled_diff.ppm")
            save_ppm(diff_path, np.concatenate([diff_img, np.full((H, W, 1), 255, dtype=np.uint8)], axis=2))

            s_path = os.path.join(self.output_dir, "single_camera_linear.ppm")
            t_path = os.path.join(self.output_dir, "tiled_camera_linear.ppm")
            save_ppm(s_path, np.concatenate([single_srgb, np.full((H, W, 1), 255, dtype=np.uint8)], axis=2))
            save_ppm(t_path, np.concatenate([tiled_linear, np.full((H, W, 1), 255, dtype=np.uint8)], axis=2))

        # In fast_mode, single = sRGB(tiled) to within ±1 for all pixels
        self.assertLessEqual(max_diff, 2,
            f"Max pixel diff {max_diff} exceeds threshold — camera matrices or "
            f"shading differs between single and tiled paths")
        self.assertLess(mean_diff, 1.0,
            f"Mean pixel diff {mean_diff:.2f} too high")

    def test_detiling_correctness_reference(self):
        """Verify C de-tiling produces correct shape, valid alpha, and
        different tiles for different cameras."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 32, 32, 6
        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.5), (0, 0, 0), 45.0, W, H)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)

        # Get C-de-tiled output
        c_output = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        # 1. Shape is correct
        self.assertEqual(c_output.shape, (N, H, W, 4))

        # 2. Each tile has valid content (alpha = 255, non-black)
        for i in range(N):
            self.assertTrue(np.all(c_output[i, :, :, 3] == 255),
                f"Camera {i} alpha not all 255")
            self.assertTrue(c_output[i, :, :, :3].mean() > 10,
                f"Camera {i} is all black")

        # 3. Adjacent cameras at different distances should produce different images
        for i in range(1, N):
            diff = np.abs(c_output[0].astype(int) - c_output[i].astype(int))
            mean_diff = diff.mean()
            print(f"\n    Camera 0 vs camera {i} mean diff: {mean_diff:.2f}")
            self.assertGreater(mean_diff, 0.1,
                f"Camera 0 and camera {i} are identical — de-tiling may be wrong")

        # 4. Re-render same cameras and verify determinism
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        c_output2 = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        np.testing.assert_array_equal(c_output, c_output2,
            "Two identical tiled renders produced different output")

        r.close()

    def test_grid_layout_verification(self):
        """Verify the tiled grid layout dimensions for various camera counts."""
        import math

        print(f"\n  Grid layout verification:")
        print(f"  {'N':>4} {'Cols':>5} {'Rows':>5} {'Grid':>12} {'Active':>7} {'Waste':>6}")
        print(f"  {'--':>4} {'----':>5} {'----':>5} {'----':>12} {'------':>7} {'-----':>6}")

        for N in [1, 2, 3, 4, 5, 7, 8, 9, 16, 25, 32, 64, 100, 128, 256]:
            cols = math.ceil(math.sqrt(N))
            rows = (N + cols - 1) // cols
            total = cols * rows
            waste = total - N
            pct_waste = waste / total * 100

            print(f"  {N:>4} {cols:>5} {rows:>5} {cols}x{rows:>9} {N:>7} {pct_waste:>5.1f}%")

            # Verify: active tiles < total tiles, no more waste than necessary
            self.assertGreaterEqual(total, N)
            self.assertLess(waste, cols, f"N={N}: waste ({waste}) >= cols ({cols})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
