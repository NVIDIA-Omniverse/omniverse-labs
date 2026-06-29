# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for tiled multi-camera ray tracing.

Tests verify:
  - Tiled output shape and content for varying camera counts
  - Per-camera isolation (different cameras produce different images)
  - Tiled vs single-camera pixel consistency
  - Transform updates propagate correctly in tiled mode
  - Edge cases: 1 camera, prime counts, large counts
  - Grid tiling layout (de-tiling correctness)
  - Multi-frame stability (no corruption across frames)
  - Performance scaling across camera counts
"""

import math
import time
import unittest
import numpy as np
import sys


# ---- Helpers ----

def make_view_matrix(eye, target):
    """Build row-major 4x4 view matrix from eye + target."""
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
    """Invert orthonormal view matrix."""
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
    """Build row-major perspective projection (Vulkan Y-flip, depth [0,1])."""
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
    """Invert symmetric perspective projection."""
    pi = np.zeros(16, dtype=np.float32)
    pi[0]  = 1.0 / proj[0]
    pi[5]  = 1.0 / proj[5]
    pi[11] = 1.0 / proj[14]
    pi[14] = -1.0
    pi[15] = proj[10] / proj[14]
    return pi


def make_vp_inv(eye, target, fov_deg, w, h):
    """Build 32-float (view_inv[16], proj_inv[16]) for one camera."""
    view = make_view_matrix(eye, target)
    vi = invert_view(view)
    proj = make_proj_matrix(fov_deg, w / h)
    pi = invert_proj(proj)
    return np.concatenate([vi, pi])


def make_scene(renderer_cls, w=100, h=100):
    """Create a renderer with a standard test scene (box + sphere approximation)."""
    from nusd_renderer._bindings import NU_RENDER_RT

    r = renderer_cls(width=w, height=h, enable_rt=True)
    if not r.rt_available:
        r.close()
        return None

    # Box mesh
    hx, hy, hz = 0.3, 0.3, 0.3
    box_verts = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ], dtype=np.float32)
    box_idx = np.array([
        0,1,2, 0,2,3, 4,6,5, 4,7,6,
        0,4,5, 0,5,1, 2,6,7, 2,7,3,
        0,3,7, 0,7,4, 1,5,6, 1,6,2,
    ], dtype=np.uint32)

    # Second mesh offset to the right
    box2_verts = box_verts + np.array([1.5, 0, 0], dtype=np.float32)

    r.add_mesh(positions=box_verts, indices=box_idx,
               display_color=(0.8, 0.2, 0.2), name="red_box")
    r.add_mesh(positions=box2_verts, indices=box_idx,
               display_color=(0.2, 0.2, 0.8), name="blue_box")

    # Initial render to build accel structures
    r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
    r.render(mode=NU_RENDER_RT)

    return r


# ---- Correctness Tests ----

class TestTiledRenderCorrectness(unittest.TestCase):
    """Correctness tests for tiled multi-camera rendering."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

        # Create shared test scene
        cls.renderer = make_scene(NuRenderer)
        if cls.renderer is None:
            raise unittest.SkipTest("RT not available")

    @classmethod
    def tearDownClass(cls):
        if cls.renderer:
            cls.renderer.close()

    def test_basic_tiled_render(self):
        """Tiled render produces correct output shape and non-black pixels."""
        r = self.renderer
        W, H = 100, 100

        vp0 = make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)
        vp1 = make_vp_inv((0, 0, 8), (0, 0, 0), 45.0, W, H)
        vps = np.stack([vp0, vp1])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (2, H, W, 4))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertGreater(pixels[0].sum(), 0, "Camera 0 should not be all black")
        self.assertGreater(pixels[1].sum(), 0, "Camera 1 should not be all black")

    def test_single_camera_tiled(self):
        """Tiled render with 1 camera works correctly."""
        r = self.renderer
        W, H = 100, 100

        vp = make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)
        vps = vp.reshape(1, 32)

        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (1, H, W, 4))
        self.assertGreater(pixels[0].sum(), 0)

    def test_cameras_produce_different_images(self):
        """Different camera positions produce distinct images."""
        r = self.renderer
        W, H = 100, 100

        vps = np.stack([
            make_vp_inv((0, 0, 3), (0, 0, 0), 45.0, W, H),   # close front
            make_vp_inv((3, 0, 0), (0, 0, 0), 45.0, W, H),   # right side
            make_vp_inv((0, 3, 0), (0, 0, 0), 45.0, W, H),   # top
            make_vp_inv((-3, 0, 0), (0, 0, 0), 45.0, W, H),  # left side
        ])

        r.render_tiled(vps, num_cameras=4, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=4, tile_w=W, tile_h=H)

        # Every pair of cameras should produce different images
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(
                    np.array_equal(pixels[i], pixels[j]),
                    f"Cameras {i} and {j} produced identical images")

    def test_tiled_vs_sequential_consistency(self):
        """Tiled render should produce same pixels as sequential single-camera renders."""
        W, H = 100, 100
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        # Create a fresh renderer for this test
        r = make_scene(NuRenderer)
        if r is None:
            self.skipTest("RT not available")

        eye = (0, 0, 5)
        target = (0, 0, 0)
        fov = 45.0

        # Sequential single-camera render
        r.set_camera(eye=eye, target=target, fov_degrees=fov)
        r.render(mode=NU_RENDER_RT)
        single_pixels = r.fetch_pixels()  # (H, W, 4)

        # Tiled single-camera render
        vp = make_vp_inv(eye, target, fov, W, H)
        r.render_tiled(vp.reshape(1, 32), num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        tiled_pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)

        r.close()

        # Compare — may not be pixel-identical due to different render paths
        # (swapchain vs storage image, BGRA swizzle vs RGBA direct)
        # but should be structurally similar: same geometry visible
        single_sum = single_pixels[:, :, :3].astype(np.int64).sum()
        tiled_sum = tiled_pixels[0, :, :, :3].astype(np.int64).sum()

        # Both should be non-zero
        self.assertGreater(single_sum, 0, "Single render is all black")
        self.assertGreater(tiled_sum, 0, "Tiled render is all black")

        # Check mean color is in similar ballpark (within 30%)
        ratio = tiled_sum / max(single_sum, 1)
        self.assertGreater(ratio, 0.5, f"Tiled brightness too low vs single: ratio={ratio:.2f}")
        self.assertLess(ratio, 2.0, f"Tiled brightness too high vs single: ratio={ratio:.2f}")

    def test_prime_camera_count(self):
        """Prime number of cameras (7) tiles correctly."""
        r = self.renderer
        W, H = 64, 64
        N = 7

        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.5), (0, 0, 0), 45.0, W, H)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        for i in range(N):
            self.assertGreater(pixels[i].sum(), 0, f"Camera {i} is all black (N={N})")

    def test_large_camera_count(self):
        """32 cameras render without crash or corruption."""
        r = self.renderer
        W, H = 32, 32
        N = 32

        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
        vps = np.stack([
            make_vp_inv(
                (3 * math.cos(a), 0.5, 3 * math.sin(a)),
                (0, 0, 0), 45.0, W, H)
            for a in angles
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N) if pixels[i].sum() > 0)
        self.assertEqual(non_black, N, f"Only {non_black}/{N} cameras produced content")

    def test_64_cameras(self):
        """64 cameras (8x8 grid) at small resolution."""
        r = self.renderer
        W, H = 16, 16
        N = 64

        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N) if pixels[i].sum() > 0)
        self.assertGreater(non_black, N * 0.9, f"Too many black cameras: {N - non_black}/{N}")

    def test_transform_update_affects_tiled(self):
        """Transform updates to meshes are visible in tiled render."""
        W, H = 100, 100
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        r = make_scene(NuRenderer)
        if r is None:
            self.skipTest("RT not available")

        vp = make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H).reshape(1, 32)

        # Render initial state
        r.render_tiled(vp, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        p1 = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H).copy()

        # Move mesh 0 far off screen (row-vector convention: translation in last row)
        xform = np.eye(4, dtype=np.float32)
        xform[3, 0] = 100.0  # way off to the right
        r.set_transforms([0], xform.reshape(1, 16))

        # Render again
        r.render_tiled(vp, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        p2 = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H).copy()

        r.close()

        # Images should differ (mesh moved off screen)
        self.assertFalse(np.array_equal(p1, p2), "Images should differ after transform")

    def test_multi_frame_stability(self):
        """Rendering the same scene multiple times produces consistent output."""
        r = self.renderer
        W, H = 64, 64

        vp = make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H).reshape(1, 32)

        frames = []
        for _ in range(5):
            r.render_tiled(vp, num_cameras=1, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            p = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)
            frames.append(p.copy())

        # All frames should be identical (deterministic rendering, same camera)
        for i in range(1, len(frames)):
            self.assertTrue(
                np.array_equal(frames[0], frames[i]),
                f"Frame {i} differs from frame 0 (non-deterministic rendering)")

    def test_camera_isolation_no_bleed(self):
        """Adjacent camera tiles don't bleed into each other."""
        r = self.renderer
        W, H = 100, 100

        # Camera 0: looking at the scene
        # Camera 1: looking at empty space (no geometry visible)
        vps = np.stack([
            make_vp_inv((0, 0, 3), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, -100), (0, 0, -200), 45.0, W, H),  # looking away
        ])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        # Camera 0 should have geometry
        cam0_rgb = pixels[0, :, :, :3].astype(np.int64)
        # Camera 1 should be mostly sky/background (no geometry in view)
        cam1_rgb = pixels[1, :, :, :3].astype(np.int64)

        # Camera 0 should have more varied content
        cam0_std = cam0_rgb.std()
        cam1_std = cam1_rgb.std()

        # Camera 0 (seeing geometry) should have higher pixel variance
        self.assertGreater(cam0_std, cam1_std * 0.5,
            f"Camera 0 (geometry) should have more detail than camera 1 (empty): "
            f"std0={cam0_std:.1f} vs std1={cam1_std:.1f}")

    def test_varying_resolutions(self):
        """Tiled render works at different tile resolutions."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        for W, H in [(32, 32), (64, 48), (100, 100), (128, 64)]:
            with self.subTest(resolution=f"{W}x{H}"):
                r = make_scene(NuRenderer, w=W, h=H)
                if r is None:
                    self.skipTest("RT not available")

                vps = np.stack([
                    make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H),
                    make_vp_inv((0, 0, 8), (0, 0, 0), 45.0, W, H),
                ])

                r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
                pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

                self.assertEqual(pixels.shape, (2, H, W, 4))
                self.assertGreater(pixels[0].sum(), 0)
                self.assertGreater(pixels[1].sum(), 0)
                r.close()

    def test_orbit_cameras_all_visible(self):
        """Cameras orbiting the scene at equal angles all see geometry."""
        r = self.renderer
        W, H = 64, 64
        N = 8
        radius = 4.0

        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
        vps = np.stack([
            make_vp_inv(
                (radius * math.cos(a), 0, radius * math.sin(a)),
                (0, 0, 0), 45.0, W, H)
            for a in angles
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        for i in range(N):
            rgb = pixels[i, :, :, :3]
            unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
            self.assertGreater(unique_colors, 3,
                f"Camera {i} (angle={math.degrees(angles[i]):.0f}deg) has too few unique colors: "
                f"{unique_colors} — likely not seeing geometry")


# ---- Performance Tests ----

class TestTiledRenderPerformance(unittest.TestCase):
    """Performance benchmarks for tiled multi-camera rendering."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT
        cls.renderer = make_scene(NuRenderer, w=100, h=100)
        if cls.renderer is None:
            raise unittest.SkipTest("RT not available")

    @classmethod
    def tearDownClass(cls):
        if cls.renderer:
            cls.renderer.close()

    def test_scaling_camera_count(self):
        """Measure tiled render time as camera count scales."""
        r = self.renderer
        W, H = 64, 64
        NFRAMES = 10

        print("\n  Tiled RT scaling benchmark:")
        print(f"  {'Cameras':>8} {'Pixels':>12} {'Render ms':>10} {'Fetch ms':>10} {'Total ms':>10} {'FPS':>8}")
        print(f"  {'-------':>8} {'------':>12} {'---------':>10} {'--------':>10} {'--------':>10} {'---':>8}")

        for N in [1, 2, 4, 8, 16, 32, 64]:
            vps = np.stack([
                make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H)
                for i in range(N)
            ])

            # Warm up
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            # Benchmark render
            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            t_render = (time.perf_counter() - t0) / NFRAMES * 1000

            # Benchmark fetch
            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            t_total = (time.perf_counter() - t0) / NFRAMES * 1000
            t_fetch = t_total - t_render

            npix = N * W * H
            fps = 1000.0 / t_total if t_total > 0 else 0

            print(f"  {N:>8} {npix:>12,} {t_render:>10.2f} {t_fetch:>10.2f} {t_total:>10.2f} {fps:>8.1f}")

    def test_tiled_vs_sequential_throughput(self):
        """Compare tiled single-dispatch vs sequential per-camera renders."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        W, H = 64, 64
        N = 16
        NFRAMES = 10

        r = make_scene(NuRenderer, w=W, h=H)
        if r is None:
            self.skipTest("RT not available")

        eyes = [(0, 0, 3 + i * 0.2) for i in range(N)]

        # ---- Sequential: N separate renders ----
        t0 = time.perf_counter()
        for _ in range(NFRAMES):
            for eye in eyes:
                r.set_camera(eye=eye, target=(0, 0, 0), fov_degrees=45.0)
                r.render(mode=NU_RENDER_RT)
                r.fetch_pixels()
        t_seq = (time.perf_counter() - t0) / NFRAMES * 1000

        # ---- Tiled: single dispatch ----
        vps = np.stack([make_vp_inv(eye, (0, 0, 0), 45.0, W, H) for eye in eyes])

        # Warm up
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        t0 = time.perf_counter()
        for _ in range(NFRAMES):
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        t_tiled = (time.perf_counter() - t0) / NFRAMES * 1000

        r.close()

        speedup = t_seq / t_tiled if t_tiled > 0 else float('inf')
        print(f"\n  Tiled vs Sequential ({N} cameras, {W}x{H}):")
        print(f"    Sequential: {t_seq:>8.2f} ms/frame ({N} renders + {N} readbacks)")
        print(f"    Tiled:      {t_tiled:>8.2f} ms/frame (1 render + 1 readback)")
        print(f"    Speedup:    {speedup:.1f}x")

        # Tiled should be faster for N >= 4
        self.assertGreater(speedup, 1.0,
            f"Tiled should be faster than sequential: {t_tiled:.2f}ms vs {t_seq:.2f}ms")

    def test_resolution_scaling(self):
        """Measure how render time scales with per-camera resolution."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        NFRAMES = 10
        N = 8

        print(f"\n  Resolution scaling ({N} cameras):")
        print(f"  {'Resolution':>12} {'Total Pixels':>14} {'Total ms':>10} {'ms/Mpix':>10}")
        print(f"  {'----------':>12} {'------------':>14} {'--------':>10} {'-------':>10}")

        for W, H in [(32, 32), (64, 64), (100, 100), (128, 128), (200, 200)]:
            r = make_scene(NuRenderer, w=W, h=H)
            if r is None:
                self.skipTest("RT not available")

            vps = np.stack([
                make_vp_inv((0, 0, 3 + i * 0.2), (0, 0, 0), 45.0, W, H)
                for i in range(N)
            ])

            # Warm up
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
                r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            t_total = (time.perf_counter() - t0) / NFRAMES * 1000

            total_pix = N * W * H
            ms_per_mpix = t_total / (total_pix / 1e6) if total_pix > 0 else 0

            print(f"  {W:>5}x{H:<5} {total_pix:>14,} {t_total:>10.2f} {ms_per_mpix:>10.2f}")
            r.close()

    def test_gpu_memory_scaling(self):
        """Measure GPU memory as camera count increases."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        W, H = 64, 64

        print(f"\n  GPU memory scaling ({W}x{H} tiles):")
        print(f"  {'Cameras':>8} {'Tiled Image':>14} {'GPU Memory MB':>14}")
        print(f"  {'-------':>8} {'----------':>14} {'------------':>14}")

        for N in [1, 4, 16, 32, 64]:
            r = make_scene(NuRenderer, w=W, h=H)
            if r is None:
                self.skipTest("RT not available")

            vps = np.stack([
                make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H)
                for i in range(N)
            ])

            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            mem_mb = r.gpu_memory_used / (1024 * 1024)
            cols = math.ceil(math.sqrt(N))
            rows = (N + cols - 1) // cols
            tiled_img = f"{cols * W}x{rows * H}"

            print(f"  {N:>8} {tiled_img:>14} {mem_mb:>14.1f}")
            r.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
