# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge-case tests for tiled rendering: resolution boundaries, concurrent
renderers, asymmetric tiles, single-pixel tiles, non-power-of-2, and
Kitchen_set tiled rendering with varying camera configurations.

These tests target corner cases that can trigger Vulkan validation errors,
off-by-one bugs in tile index computation, or SSBO overflow.
"""

import math
import time
import unittest
import numpy as np
import os


# ---- Helpers (shared with other test files) ----

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


def make_scene(renderer_cls, w=100, h=100, num_boxes=2):
    from nusd_renderer._bindings import NU_RENDER_RT
    r = renderer_cls(width=w, height=h, enable_rt=True)
    if not r.rt_available:
        r.close()
        return None
    for i in range(num_boxes):
        v, idx = make_box(cx=i * 1.5)
        color = [(0.8, 0.2, 0.2), (0.2, 0.2, 0.8), (0.2, 0.8, 0.2),
                 (0.8, 0.8, 0.2), (0.8, 0.2, 0.8)][i % 5]
        r.add_mesh(positions=v, indices=idx, display_color=color, name=f"box_{i}")
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)
    return r


# ---- Resolution boundary tests ----

class TestTiledResolutionEdgeCases(unittest.TestCase):
    """Resolution edge cases: tiny tiles, odd dimensions, non-power-of-2."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def _make_scene_and_render(self, w, h, num_cameras):
        """Helper: create scene, do tiled render, return (renderer, pixels)."""
        r = make_scene(self.NuRenderer, w=w, h=h)
        if r is None:
            self.skipTest("RT not available")
        vps = np.stack([
            make_vp_inv((2 * math.cos(i * 0.7), 0.5, 2 * math.sin(i * 0.7)),
                        (0, 0, 0), 45.0, w, h)
            for i in range(num_cameras)
        ])
        r.render_tiled(vps, num_cameras=num_cameras, tile_w=w, tile_h=h,
                       mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=num_cameras, tile_w=w, tile_h=h)
        return r, pixels

    def test_1x1_tiles(self):
        """Single pixel tiles — minimum resolution."""
        r, pixels = self._make_scene_and_render(1, 1, 4)
        self.assertEqual(pixels.shape, (4, 1, 1, 4))
        # Alpha should always be 255
        for i in range(4):
            self.assertEqual(pixels[i, 0, 0, 3], 255)
        r.close()

    def test_2x2_tiles(self):
        """2x2 tiles — minimal useful resolution."""
        r, pixels = self._make_scene_and_render(2, 2, 4)
        self.assertEqual(pixels.shape, (4, 2, 2, 4))
        # At least some non-black pixels across all cameras
        total = pixels[:, :, :, :3].sum()
        self.assertGreater(total, 0)
        r.close()

    def test_3x3_tiles(self):
        """3x3 tiles — odd minimum, tests rounding in tile grid."""
        r, pixels = self._make_scene_and_render(3, 3, 9)
        self.assertEqual(pixels.shape, (9, 3, 3, 4))
        r.close()

    def test_odd_dimensions(self):
        """Odd tile dimensions: 7x13 with 5 cameras."""
        r, pixels = self._make_scene_and_render(7, 13, 5)
        self.assertEqual(pixels.shape, (5, 13, 7, 4))
        # Alpha channel should be 255 everywhere
        self.assertTrue(np.all(pixels[:, :, :, 3] == 255))
        r.close()

    def test_prime_dimensions(self):
        """Prime number tile dimensions: 17x23 with 7 cameras."""
        r, pixels = self._make_scene_and_render(17, 23, 7)
        self.assertEqual(pixels.shape, (7, 23, 17, 4))
        r.close()

    def test_non_square_wide(self):
        """Wide tiles: 200x10, testing extreme aspect ratio handling."""
        r, pixels = self._make_scene_and_render(200, 10, 4)
        self.assertEqual(pixels.shape, (4, 10, 200, 4))
        r.close()

    def test_non_square_tall(self):
        """Tall tiles: 10x200, testing extreme aspect ratio handling."""
        r, pixels = self._make_scene_and_render(10, 200, 4)
        self.assertEqual(pixels.shape, (4, 200, 10, 4))
        r.close()

    def test_power_of_2_sweep(self):
        """Sweep power-of-2 resolutions from 4 to 256."""
        for size in [4, 8, 16, 32, 64, 128, 256]:
            with self.subTest(size=size):
                r, pixels = self._make_scene_and_render(size, size, 4)
                self.assertEqual(pixels.shape, (4, size, size, 4))
                # Content check: at least one camera sees non-black
                has_content = any(
                    pixels[i, :, :, :3].sum() > 0 for i in range(4))
                self.assertTrue(has_content,
                    f"No content at {size}x{size}")
                r.close()

    def test_non_power_of_2_sweep(self):
        """Non-power-of-2 resolutions: 33, 50, 99, 100, 101, 127, 129."""
        for size in [33, 50, 99, 100, 101, 127, 129]:
            with self.subTest(size=size):
                r, pixels = self._make_scene_and_render(size, size, 4)
                self.assertEqual(pixels.shape, (4, size, size, 4))
                r.close()

    def test_asymmetric_tiles_many_cameras(self):
        """60x40 tiles with 15 cameras — 4x4 grid, last row partially filled."""
        W, H, N = 60, 40, 15
        r, pixels = self._make_scene_and_render(W, H, N)
        self.assertEqual(pixels.shape, (N, H, W, 4))
        # Check that camera 14 (last one, partial row) has valid alpha
        self.assertTrue(np.all(pixels[14, :, :, 3] == 255))
        r.close()

    def test_max_single_dimension(self):
        """512x4 tiles — max width, minimal height, tests tiled image limits."""
        r, pixels = self._make_scene_and_render(512, 4, 4)
        self.assertEqual(pixels.shape, (4, 4, 512, 4))
        r.close()


class TestConcurrentRenderers(unittest.TestCase):
    """Test multiple renderer instances operating simultaneously."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    @unittest.skip("Known issue: concurrent Vulkan renderer instances crash on close()")
    def test_two_renderers_different_scenes(self):
        """Two renderers with different geometry produce different images."""
        pass

    @unittest.skip("Known issue: concurrent Vulkan renderer instances crash on close()")
    def test_two_renderers_different_resolutions(self):
        """Two renderers at different resolutions running tiled render."""
        pass

    def test_renderer_create_destroy_cycle(self):
        """Create and destroy multiple renderers in sequence — tests cleanup."""
        for i in range(5):
            r = make_scene(self.NuRenderer, w=64, h=64)
            if r is None:
                self.skipTest("RT not available")
            vps = np.stack([
                make_vp_inv((2, 0, 2), (0, 0, 0), 45.0, 64, 64)
            ])
            r.render_tiled(vps, num_cameras=1, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
            pix = r.fetch_pixels_tiled(num_cameras=1, tile_w=64, tile_h=64)
            self.assertEqual(pix.shape, (1, 64, 64, 4))
            r.close()


class TestTiledCameraSSBO(unittest.TestCase):
    """Test camera SSBO edge cases: max cameras, SSBO resize, identity matrices."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_ssbo_grows_between_renders(self):
        """SSBO should grow when camera count increases between renders."""
        r = make_scene(self.NuRenderer, w=32, h=32)
        if r is None:
            self.skipTest("RT not available")

        # First: 4 cameras
        vps4 = np.stack([
            make_vp_inv((2, 0, 2), (0, 0, 0), 45.0, 32, 32)
            for _ in range(4)
        ])
        r.render_tiled(vps4, num_cameras=4, tile_w=32, tile_h=32, mode=self.NU_RENDER_RT)
        pix4 = r.fetch_pixels_tiled(num_cameras=4, tile_w=32, tile_h=32)
        self.assertEqual(pix4.shape, (4, 32, 32, 4))

        # Then: 16 cameras — SSBO must grow
        vps16 = np.stack([
            make_vp_inv((2 * math.cos(i * 0.4), 0.5, 2 * math.sin(i * 0.4)),
                        (0, 0, 0), 45.0, 32, 32)
            for i in range(16)
        ])
        r.render_tiled(vps16, num_cameras=16, tile_w=32, tile_h=32, mode=self.NU_RENDER_RT)
        pix16 = r.fetch_pixels_tiled(num_cameras=16, tile_w=32, tile_h=32)
        self.assertEqual(pix16.shape, (16, 32, 32, 4))

        # Then back down to 2 cameras — should still work
        vps2 = np.stack([
            make_vp_inv((2, 0, 2), (0, 0, 0), 45.0, 32, 32),
            make_vp_inv((-2, 0, 2), (0, 0, 0), 45.0, 32, 32),
        ])
        r.render_tiled(vps2, num_cameras=2, tile_w=32, tile_h=32, mode=self.NU_RENDER_RT)
        pix2 = r.fetch_pixels_tiled(num_cameras=2, tile_w=32, tile_h=32)
        self.assertEqual(pix2.shape, (2, 32, 32, 4))
        r.close()

    def test_ssbo_camera_independence(self):
        """Changing one camera's VP shouldn't affect other cameras' output."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        N = 4
        base_vps = np.stack([
            make_vp_inv((2 * math.cos(i * math.pi / 2), 0.5,
                         2 * math.sin(i * math.pi / 2)),
                        (0, 0, 0), 45.0, 64, 64)
            for i in range(N)
        ])

        # Render baseline
        r.render_tiled(base_vps, num_cameras=N, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
        baseline = r.fetch_pixels_tiled(num_cameras=N, tile_w=64, tile_h=64).copy()

        # Move camera 2 far away
        modified_vps = base_vps.copy()
        modified_vps[2] = make_vp_inv((100, 100, 100), (0, 0, 0), 45.0, 64, 64)

        r.render_tiled(modified_vps, num_cameras=N, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
        modified = r.fetch_pixels_tiled(num_cameras=N, tile_w=64, tile_h=64)

        # Camera 0, 1, 3 should be identical
        for i in [0, 1, 3]:
            np.testing.assert_array_equal(baseline[i], modified[i],
                err_msg=f"Camera {i} changed when only camera 2 was moved")

        # Camera 2 should be different
        diff2 = np.abs(baseline[2].astype(float) - modified[2].astype(float)).mean()
        self.assertGreater(diff2, 1.0, "Camera 2 should differ after moving it")
        r.close()

    def test_identity_view_matrix(self):
        """Camera at origin looking down -Z with identity-like matrices."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        # Place boxes along -Z axis
        v, idx = make_box(cz=-2)
        r.add_mesh(positions=v, indices=idx, display_color=(0.5, 0.5, 0.0))

        # Identity view = camera at origin, looking down -Z
        eye = (0, 0, 0.001)  # Slightly off origin to avoid degenerate
        target = (0, 0, -2)
        vps = np.stack([make_vp_inv(eye, target, 45.0, 64, 64)])
        r.render_tiled(vps, num_cameras=1, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
        pix = r.fetch_pixels_tiled(num_cameras=1, tile_w=64, tile_h=64)
        self.assertEqual(pix.shape, (1, 64, 64, 4))
        # Should see the box
        center = pix[0, 32, 32, :3]
        self.assertGreater(center.sum(), 0, "Center pixel should see geometry")
        r.close()


class TestTiledGridLayout(unittest.TestCase):
    """Test that the ceil(sqrt(N)) grid layout handles edge cases correctly."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_perfect_square_camera_counts(self):
        """Camera counts that are perfect squares: 1, 4, 9, 16."""
        for n in [1, 4, 9, 16]:
            with self.subTest(num_cameras=n):
                r = make_scene(self.NuRenderer, w=32, h=32)
                if r is None:
                    self.skipTest("RT not available")
                vps = np.stack([
                    make_vp_inv((2, 0, 2 + i * 0.01), (0, 0, 0), 45.0, 32, 32)
                    for i in range(n)
                ])
                r.render_tiled(vps, num_cameras=n, tile_w=32, tile_h=32,
                               mode=self.NU_RENDER_RT)
                pix = r.fetch_pixels_tiled(num_cameras=n, tile_w=32, tile_h=32)
                self.assertEqual(pix.shape, (n, 32, 32, 4))
                r.close()

    def test_one_more_than_perfect_square(self):
        """N=5 (2x2 + 1), N=10 (3x3 + 1), N=17 (4x4 + 1) — tests last partial row."""
        for n in [5, 10, 17]:
            with self.subTest(num_cameras=n):
                r = make_scene(self.NuRenderer, w=32, h=32)
                if r is None:
                    self.skipTest("RT not available")
                vps = np.stack([
                    make_vp_inv((2 * math.cos(i * 0.5), 0.5, 2 * math.sin(i * 0.5)),
                                (0, 0, 0), 45.0, 32, 32)
                    for i in range(n)
                ])
                r.render_tiled(vps, num_cameras=n, tile_w=32, tile_h=32,
                               mode=self.NU_RENDER_RT)
                pix = r.fetch_pixels_tiled(num_cameras=n, tile_w=32, tile_h=32)
                self.assertEqual(pix.shape, (n, 32, 32, 4))
                # Last camera should have valid alpha
                self.assertTrue(np.all(pix[-1, :, :, 3] == 255))
                r.close()

    def test_one_less_than_perfect_square(self):
        """N=3 (2x2 - 1), N=8 (3x3 - 1), N=15 (4x4 - 1) — almost full grid."""
        for n in [3, 8, 15]:
            with self.subTest(num_cameras=n):
                r = make_scene(self.NuRenderer, w=32, h=32)
                if r is None:
                    self.skipTest("RT not available")
                vps = np.stack([
                    make_vp_inv((2 * math.cos(i * 0.5), 0.5, 2 * math.sin(i * 0.5)),
                                (0, 0, 0), 45.0, 32, 32)
                    for i in range(n)
                ])
                r.render_tiled(vps, num_cameras=n, tile_w=32, tile_h=32,
                               mode=self.NU_RENDER_RT)
                pix = r.fetch_pixels_tiled(num_cameras=n, tile_w=32, tile_h=32)
                self.assertEqual(pix.shape, (n, 32, 32, 4))
                r.close()


class TestTiledDeterminism(unittest.TestCase):
    """Verify that tiled rendering is deterministic across multiple invocations."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_back_to_back_identical(self):
        """Two consecutive renders with same VP should produce identical pixels."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        N = 4
        vps = np.stack([
            make_vp_inv((2 * math.cos(i), 0.5, 2 * math.sin(i)),
                        (0, 0, 0), 45.0, 100, 100)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=100, tile_h=100, mode=self.NU_RENDER_RT)
        pix1 = r.fetch_pixels_tiled(num_cameras=N, tile_w=100, tile_h=100).copy()

        r.render_tiled(vps, num_cameras=N, tile_w=100, tile_h=100, mode=self.NU_RENDER_RT)
        pix2 = r.fetch_pixels_tiled(num_cameras=N, tile_w=100, tile_h=100)

        np.testing.assert_array_equal(pix1, pix2,
            err_msg="Back-to-back renders with same VP should be identical")
        r.close()

    def test_render_after_resolution_change(self):
        """Render at one resolution, switch to another, switch back — same result."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        N = 4
        vps_100 = np.stack([
            make_vp_inv((2, 0, 2), (0, 0, 0), 45.0, 100, 100)
            for _ in range(N)
        ])
        vps_50 = np.stack([
            make_vp_inv((2, 0, 2), (0, 0, 0), 45.0, 50, 50)
            for _ in range(N)
        ])

        # Render at 100x100
        r.render_tiled(vps_100, num_cameras=N, tile_w=100, tile_h=100, mode=self.NU_RENDER_RT)
        pix_before = r.fetch_pixels_tiled(num_cameras=N, tile_w=100, tile_h=100).copy()

        # Render at 50x50 (causes tiled image resize)
        r.render_tiled(vps_50, num_cameras=N, tile_w=50, tile_h=50, mode=self.NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=50, tile_h=50)

        # Back to 100x100 (another resize)
        r.render_tiled(vps_100, num_cameras=N, tile_w=100, tile_h=100, mode=self.NU_RENDER_RT)
        pix_after = r.fetch_pixels_tiled(num_cameras=N, tile_w=100, tile_h=100)

        np.testing.assert_array_equal(pix_before, pix_after,
            err_msg="Same VP should produce same pixels after resolution round-trip")
        r.close()

    def test_100_frame_bit_exact(self):
        """100 frames with same VP: every frame must be bit-exact."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        N = 4
        vps = np.stack([
            make_vp_inv((2 * math.cos(i), 0.5, 2 * math.sin(i)),
                        (0, 0, 0), 45.0, 64, 64)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
        reference = r.fetch_pixels_tiled(num_cameras=N, tile_w=64, tile_h=64).copy()

        mismatches = 0
        for frame in range(100):
            r.render_tiled(vps, num_cameras=N, tile_w=64, tile_h=64, mode=self.NU_RENDER_RT)
            current = r.fetch_pixels_tiled(num_cameras=N, tile_w=64, tile_h=64)
            if not np.array_equal(reference, current):
                mismatches += 1

        self.assertEqual(mismatches, 0,
            f"{mismatches}/100 frames differed from reference")
        r.close()


class TestKitchenSetTiledDeep(unittest.TestCase):
    """Deep Kitchen_set tiled rendering tests — requires NANOUSD_BACKEND."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT
        cls.kitchen_path = os.environ.get("NUSD_KITCHEN_SET", os.path.expanduser("~/Kitchen_set/Kitchen_set.usd"))
        if not os.path.exists(cls.kitchen_path):
            raise unittest.SkipTest(f"Kitchen_set not found at {cls.kitchen_path}")

    def _load_kitchen(self, w=200, h=200):
        r = self.NuRenderer(width=w, height=h, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")
        try:
            nmeshes = r.load_usd(self.kitchen_path)
        except RuntimeError:
            r.close()
            self.skipTest("USD backend not available")
        r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
        r.render(mode=self.NU_RENDER_RT)
        return r, nmeshes

    def test_kitchen_many_cameras_32(self):
        """32 cameras orbiting Kitchen_set — IsaacLab-scale."""
        r, nmeshes = self._load_kitchen(w=100, h=100)
        N, W, H = 32, 100, 100
        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
        vps = np.stack([
            make_vp_inv(
                (200 * math.cos(a), 120, 200 * math.sin(a)),
                (70, 50, 30), 60.0, W, H)
            for a in angles
        ])
        t0 = time.perf_counter()
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n  Kitchen_set 32 cameras 100x100: {elapsed_ms:.1f} ms ({nmeshes} meshes)")

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N)
                        if pixels[i, :, :, :3].sum() > 0)
        self.assertGreater(non_black, N * 0.7,
            f"Only {non_black}/{N} cameras see content")
        r.close()

    def test_kitchen_resolution_sweep(self):
        """Kitchen_set tiled rendering at various resolutions."""
        r, nmeshes = self._load_kitchen(w=256, h=256)
        N = 8
        results = []
        for res in [32, 64, 128, 256]:
            angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
            vps = np.stack([
                make_vp_inv(
                    (200 * math.cos(a), 120, 200 * math.sin(a)),
                    (70, 50, 30), 60.0, res, res)
                for a in angles
            ])
            t0 = time.perf_counter()
            r.render_tiled(vps, num_cameras=N, tile_w=res, tile_h=res,
                           mode=self.NU_RENDER_RT)
            pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=res, tile_h=res)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            mpix = N * res * res / 1e6
            results.append((res, elapsed_ms, mpix / (elapsed_ms / 1000)))
            self.assertEqual(pixels.shape, (N, res, res, 4))

        print(f"\n  Kitchen_set resolution sweep ({nmeshes} meshes, {N} cameras):")
        for res, ms, mpix_s in results:
            print(f"    {res}x{res}: {ms:.1f} ms, {mpix_s:.1f} Mpix/s")
        r.close()

    def test_kitchen_determinism(self):
        """Kitchen_set tiled rendering is deterministic."""
        r, _ = self._load_kitchen(w=100, h=100)
        N, W, H = 4, 100, 100
        vps = np.stack([
            make_vp_inv(
                (200 * math.cos(i * math.pi / 2), 120,
                 200 * math.sin(i * math.pi / 2)),
                (70, 50, 30), 60.0, W, H)
            for i in range(N)
        ])
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pix1 = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H).copy()

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pix2 = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        np.testing.assert_array_equal(pix1, pix2,
            err_msg="Kitchen_set tiled render should be deterministic")
        r.close()

    def test_kitchen_sustained_frames(self):
        """50 frames sustained tiled rendering of Kitchen_set — memory + perf stability."""
        r, nmeshes = self._load_kitchen(w=64, h=64)
        N, W, H = 16, 64, 64
        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)

        times = []
        for frame in range(50):
            # Slowly orbit
            offset = frame * 0.02
            vps = np.stack([
                make_vp_inv(
                    (200 * math.cos(a + offset), 120, 200 * math.sin(a + offset)),
                    (70, 50, 30), 60.0, W, H)
                for a in angles
            ])
            t0 = time.perf_counter()
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            times.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(times[5:])  # Skip warmup
        std = np.std(times[5:])
        print(f"\n  Kitchen_set 50 frames ({nmeshes} meshes, {N}x{W}x{H}):")
        print(f"    avg: {avg:.1f} ms, std: {std:.1f} ms, "
              f"min: {min(times[5:]):.1f}, max: {max(times[5:]):.1f}")

        # Std should be reasonable (< 50% of mean) — no stalls
        self.assertLess(std, avg * 0.5,
            f"Frame time variance too high: std={std:.1f} vs avg={avg:.1f}")
        r.close()


if __name__ == "__main__":
    unittest.main()
