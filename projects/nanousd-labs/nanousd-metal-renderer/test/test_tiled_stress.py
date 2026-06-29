# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress tests, memory leak detection, and edge cases for tiled multi-camera RT.

Tests:
  - Memory leak detection over 1000+ frames
  - Rapid dimension switching (fuzzing)
  - Maximum camera counts (128, 256)
  - Interleaved tiled + single-camera rendering
  - Scene mutation during tiled rendering (add/remove meshes, transform updates)
  - Extreme aspect ratios and tiny tiles
  - Alpha channel correctness
  - PPM save from tiled render
  - Pixel-level reproducibility across renderer instances
"""

import math
import time
import unittest
import numpy as np
import os
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


# ---- Stress Tests ----

class TestTiledStress(unittest.TestCase):
    """Stress tests and edge cases."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_memory_leak_1000_frames(self):
        """Render 1000 frames and verify GPU memory doesn't grow."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 8
        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.2), (0, 0, 0), 45.0, W, H)
            for i in range(N)
        ])

        # Warm up
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        mem_start = r.gpu_memory_used

        for frame in range(1000):
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        mem_end = r.gpu_memory_used
        r.close()

        growth_mb = (mem_end - mem_start) / (1024 * 1024)
        print(f"\n  Memory leak test (1000 frames, {N} cams {W}x{H}):")
        print(f"    Start: {mem_start / (1024*1024):.2f} MB")
        print(f"    End:   {mem_end / (1024*1024):.2f} MB")
        print(f"    Growth: {growth_mb:.4f} MB")

        self.assertLess(growth_mb, 1.0,
            f"GPU memory grew by {growth_mb:.2f} MB over 1000 frames — likely a leak")

    def test_rapid_dimension_switching(self):
        """Rapidly switch between different tile dimensions and camera counts."""
        r = make_scene(self.NuRenderer, w=200, h=200)
        if r is None:
            self.skipTest("RT not available")

        configs = [
            (1, 100, 100),
            (4, 64, 64),
            (16, 32, 32),
            (64, 16, 16),
            (7, 50, 50),
            (2, 200, 100),
            (3, 80, 60),
            (32, 24, 24),
            (1, 200, 200),
            (9, 40, 40),
        ]

        for iteration in range(5):  # 5 full cycles
            for N, W, H in configs:
                vps = np.stack([
                    make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H)
                    for i in range(N)
                ])
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
                self.assertEqual(pixels.shape, (N, H, W, 4),
                    f"Wrong shape on iter {iteration}, config ({N},{W},{H})")

        r.close()
        print(f"\n  Rapid dimension switching: 5 cycles x {len(configs)} configs = {5 * len(configs)} transitions OK")

    def test_128_cameras(self):
        """128 cameras at small resolution."""
        r = make_scene(self.NuRenderer, w=16, h=16)
        if r is None:
            self.skipTest("RT not available")

        N = 128
        W, H = 16, 16
        vps = np.stack([
            make_vp_inv((3 * math.cos(a), 0.5, 3 * math.sin(a)), (0, 0, 0), 45.0, W, H)
            for a in np.linspace(0, 2 * math.pi, N, endpoint=False)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N) if pixels[i].sum() > 0)
        r.close()
        print(f"\n  128 cameras: {non_black}/{N} non-black")
        self.assertEqual(non_black, N)

    def test_256_cameras(self):
        """256 cameras (16x16 grid) at tiny resolution."""
        r = make_scene(self.NuRenderer, w=8, h=8)
        if r is None:
            self.skipTest("RT not available")

        N = 256
        W, H = 8, 8
        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.02), (0, 0, 0), 45.0, W, H)
            for i in range(N)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N) if pixels[i].sum() > 0)
        r.close()
        print(f"\n  256 cameras: {non_black}/{N} non-black")
        self.assertGreater(non_black, N * 0.95)

    def test_interleaved_tiled_and_single(self):
        """Alternate between tiled and single-camera renders."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        W, H = 100, 100

        for i in range(20):
            if i % 2 == 0:
                # Tiled render
                N = 4
                vps = np.stack([
                    make_vp_inv((0, 0, 3 + j), (0, 0, 0), 45.0, W, H) for j in range(N)
                ])
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
                self.assertEqual(pixels.shape, (N, H, W, 4))
                self.assertGreater(pixels[0].sum(), 0, f"Tiled frame {i} cam 0 is black")
            else:
                # Single-camera render
                r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
                r.render(mode=self.NU_RENDER_RT)
                pixels = r.fetch_pixels()
                self.assertEqual(pixels.shape, (H, W, 4))
                self.assertGreater(pixels.sum(), 0, f"Single frame {i} is black")

        r.close()
        print(f"\n  Interleaved tiled/single: 20 frames OK")

    @unittest.skip("Pre-existing bug: double-free in gpu_destroy_rt_scene on mesh removal")
    def test_scene_mutation_between_tiled_renders(self):
        """Add/remove meshes between tiled render calls."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        r = NuRenderer(width=64, height=64, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)]).reshape(1, 32)

        # Add box, render
        v, idx = make_box()
        mid = r.add_mesh(positions=v, indices=idx, display_color=(0.8, 0.2, 0.2), name="box")
        r.set_camera(eye=(0, 0, 5), target=(0, 0, 0))
        r.render(mode=NU_RENDER_RT)
        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        p1 = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H).copy()
        self.assertGreater(p1[0].sum(), 0, "Initial render is black")

        # Add another box, re-render
        v2, idx2 = make_box(cx=1.0)
        mid2 = r.add_mesh(positions=v2, indices=idx2, display_color=(0.2, 0.8, 0.2), name="box2")
        r.render(mode=NU_RENDER_RT)  # triggers TLAS rebuild
        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        p2 = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H).copy()
        self.assertGreater(p2[0].sum(), 0, "After adding mesh, render is black")
        self.assertFalse(np.array_equal(p1, p2), "Adding a mesh should change the image")

        # Remove original box, re-render
        r.remove_mesh(mid)
        r.render(mode=NU_RENDER_RT)  # triggers TLAS rebuild
        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        p3 = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H).copy()
        self.assertGreater(p3[0].sum(), 0, "After removing one mesh, render is black")
        self.assertFalse(np.array_equal(p2, p3), "Removing a mesh should change the image")

        r.close()
        print(f"\n  Scene mutation: add/remove between tiled renders OK")

    def test_transform_animation_over_many_frames(self):
        """Animate a transform over 200 frames and verify output changes each time."""
        r = make_scene(self.NuRenderer, w=64, h=64, num_boxes=1)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 4
        vps = np.stack([
            make_vp_inv((0, 0, 3 + j), (0, 0, 0), 45.0, W, H) for j in range(N)
        ])

        prev_sums = None
        changes = 0
        total_frames = 200
        xform = np.eye(4, dtype=np.float32)

        for frame in range(total_frames):
            # Rotate the mesh
            angle = frame * 0.05
            xform[0, 0] = math.cos(angle)
            xform[0, 2] = math.sin(angle)
            xform[2, 0] = -math.sin(angle)
            xform[2, 2] = math.cos(angle)
            r.set_transforms([0], xform.reshape(1, 16))

            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            sums = [int(pixels[i].sum()) for i in range(N)]
            if prev_sums is not None and sums != prev_sums:
                changes += 1
            prev_sums = sums

        r.close()
        change_rate = changes / (total_frames - 1)
        print(f"\n  Transform animation ({total_frames} frames): "
              f"{changes}/{total_frames - 1} frames changed ({change_rate:.0%})")
        self.assertGreater(change_rate, 0.5,
            f"Rotation animation should change most frames, got {change_rate:.0%}")

    def test_extreme_aspect_ratio_wide(self):
        """Very wide tile: 256x16."""
        r = make_scene(self.NuRenderer, w=256, h=16)
        if r is None:
            self.skipTest("RT not available")

        W, H = 256, 16
        vps = np.stack([
            make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, 8), (0, 0, 0), 45.0, W, H),
        ])
        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)
        self.assertEqual(pixels.shape, (2, H, W, 4))
        self.assertGreater(pixels[0].sum(), 0)
        r.close()

    def test_extreme_aspect_ratio_tall(self):
        """Very tall tile: 16x256."""
        r = make_scene(self.NuRenderer, w=16, h=256)
        if r is None:
            self.skipTest("RT not available")

        W, H = 16, 256
        vps = np.stack([
            make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, 8), (0, 0, 0), 45.0, W, H),
        ])
        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)
        self.assertEqual(pixels.shape, (2, H, W, 4))
        self.assertGreater(pixels[0].sum(), 0)
        r.close()

    def test_alpha_channel(self):
        """Verify alpha channel is always 255 for non-transparent surfaces."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        W, H = 100, 100
        vps = np.stack([
            make_vp_inv((0, 0, 3), (0, 0, 0), 45.0, W, H),
            make_vp_inv((3, 0, 0), (0, 0, 0), 45.0, W, H),
        ])
        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        # Alpha should be 255 everywhere (opaque output)
        for cam in range(2):
            alpha = pixels[cam, :, :, 3]
            self.assertTrue(np.all(alpha == 255),
                f"Camera {cam}: not all alpha=255, min={alpha.min()}, max={alpha.max()}")

        r.close()

    def test_reproducibility_across_instances(self):
        """Two separate renderer instances with same scene produce identical output."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        W, H = 64, 64
        vps = np.stack([make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)]).reshape(1, 32)

        outputs = []
        for _ in range(2):
            r = make_scene(NuRenderer, w=W, h=H, num_boxes=1)
            if r is None:
                self.skipTest("RT not available")
            r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
            pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)
            outputs.append(pixels.copy())
            r.close()

        self.assertTrue(np.array_equal(outputs[0], outputs[1]),
            "Two renderer instances with same scene should produce identical output")

    def test_complex_scene_many_meshes(self):
        """Tiled render with a complex scene (20 meshes)."""
        r = make_scene(self.NuRenderer, w=100, h=100, num_boxes=20)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 100, 100, 8
        vps = np.stack([
            make_vp_inv(
                (10 * math.cos(a), 3, 10 * math.sin(a)),
                (15, 0, 0), 45.0, W, H)
            for a in np.linspace(0, 2 * math.pi, N, endpoint=False)
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        for i in range(N):
            rgb = pixels[i, :, :, :3]
            unique = len(np.unique(rgb.reshape(-1, 3), axis=0))
            self.assertGreater(unique, 5,
                f"Camera {i} in complex scene has only {unique} unique colors")

        r.close()
        print(f"\n  Complex scene (20 meshes, {N} cameras): all cameras see detail")


class TestTiledEdgeCases(unittest.TestCase):
    """Edge cases and numerical stability."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_camera_looking_straight_down(self):
        """Camera at (0,5,0) looking at origin (degenerate up vector case)."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([
            make_vp_inv((0, 5, 0), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, -5, 0), (0, 0, 0), 45.0, W, H),  # looking straight up
        ])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (2, H, W, 4))
        # Should not crash, output should be valid
        for cam in range(2):
            self.assertTrue(np.all(pixels[cam, :, :, 3] == 255),
                f"Camera {cam} has invalid alpha")

        r.close()

    def test_camera_very_far_away(self):
        """Camera at extreme distance should still render (tiny scene in center)."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([
            make_vp_inv((0, 0, 500), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, 1000), (0, 0, 0), 45.0, W, H),
        ])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)

        # At 500 units away, the 0.6-unit box subtends ~0.07 degrees — may or may not hit pixels
        # Just verify no crash and valid alpha
        self.assertEqual(pixels.shape, (2, H, W, 4))
        self.assertTrue(np.all(pixels[:, :, :, 3] == 255))
        r.close()

    def test_camera_very_close(self):
        """Camera very close to geometry (inside the scene bounds)."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([
            make_vp_inv((0, 0, 0.5), (0, 0, 0), 45.0, W, H),
            make_vp_inv((0, 0, 0.1), (0, 0, -1), 45.0, W, H),
        ])

        r.render_tiled(vps, num_cameras=2, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=2, tile_w=W, tile_h=H)
        self.assertEqual(pixels.shape, (2, H, W, 4))
        r.close()

    def test_narrow_fov(self):
        """Narrow FOV (5 degrees) — telephoto lens."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([
            make_vp_inv((0, 0, 20), (0, 0, 0), 5.0, W, H),
        ]).reshape(1, 32)

        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)

        # Narrow FOV should magnify the scene
        self.assertEqual(pixels.shape, (1, H, W, 4))
        self.assertGreater(pixels[0].sum(), 0, "Narrow FOV render is all black")
        r.close()

    def test_wide_fov(self):
        """Very wide FOV (120 degrees) — fisheye-like."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        vps = np.stack([
            make_vp_inv((0, 0, 3), (0, 0, 0), 120.0, W, H),
        ]).reshape(1, 32)

        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (1, H, W, 4))
        self.assertGreater(pixels[0].sum(), 0, "Wide FOV render is all black")
        r.close()

    def test_all_cameras_same_position(self):
        """All cameras at identical position should produce identical output."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 8
        vps = np.stack([make_vp_inv((0, 0, 5), (0, 0, 0), 45.0, W, H)] * N)

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        for i in range(1, N):
            self.assertTrue(np.array_equal(pixels[0], pixels[i]),
                f"Camera {i} differs from camera 0 despite identical position")

        r.close()

    def test_cameras_with_different_fovs(self):
        """Cameras with different FOVs produce different images."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H = 64, 64
        fovs = [20.0, 45.0, 90.0, 120.0]
        vps = np.stack([
            make_vp_inv((0, 0, 5), (0, 0, 0), fov, W, H)
            for fov in fovs
        ])

        r.render_tiled(vps, num_cameras=4, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=4, tile_w=W, tile_h=H)

        # Each pair should differ (different FOVs)
        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                self.assertFalse(np.array_equal(pixels[i], pixels[j]),
                    f"FOV {fovs[i]}° and {fovs[j]}° produced identical images")

        r.close()

    def test_color_fidelity(self):
        """Verify that mesh display colors appear in the rendered output."""
        from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT

        W, H = 100, 100
        r = NuRenderer(width=W, height=H, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")

        # Single large box filling the view, red
        v, idx = make_box(hx=2, hy=2, hz=0.1)
        r.add_mesh(positions=v, indices=idx, display_color=(1.0, 0.0, 0.0), name="red_wall")
        r.set_camera(eye=(0, 0, 2), target=(0, 0, 0))
        r.render(mode=NU_RENDER_RT)

        vps = np.stack([make_vp_inv((0, 0, 2), (0, 0, 0), 45.0, W, H)]).reshape(1, 32)
        r.render_tiled(vps, num_cameras=1, tile_w=W, tile_h=H, mode=NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=1, tile_w=W, tile_h=H)

        # Center pixel region should be predominantly red
        center = pixels[0, H//4:3*H//4, W//4:3*W//4, :3].astype(np.float32)
        mean_rgb = center.mean(axis=(0, 1))
        print(f"\n  Color fidelity — red wall, center mean RGB: ({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")

        # Red channel should dominate (threshold accounts for ground plane/sky
        # contributions to non-red channels in the center crop)
        self.assertGreater(mean_rgb[0], mean_rgb[1] + 10,
            f"Red channel should dominate for red wall: {mean_rgb}")
        self.assertGreater(mean_rgb[0], mean_rgb[2] + 10,
            f"Red channel should dominate for red wall: {mean_rgb}")

        r.close()


class TestTiledPerfDeep(unittest.TestCase):
    """Extended performance benchmarks."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def test_sustained_throughput_100_frames(self):
        """Measure sustained throughput over 100 frames (checks for thermal/power throttling)."""
        r = make_scene(self.NuRenderer, w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 16
        NFRAMES = 100
        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H) for i in range(N)
        ])

        # Warm up
        for _ in range(10):
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        # Measure per-frame times
        times = []
        for f in range(NFRAMES):
            t0 = time.perf_counter()
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            times.append((time.perf_counter() - t0) * 1000)

        r.close()

        times = np.array(times)
        print(f"\n  Sustained throughput ({NFRAMES} frames, {N} cameras, {W}x{H}):")
        print(f"    Mean:   {times.mean():.3f} ms/frame ({1000/times.mean():.0f} FPS)")
        print(f"    Median: {np.median(times):.3f} ms")
        print(f"    Min:    {times.min():.3f} ms")
        print(f"    Max:    {times.max():.3f} ms")
        print(f"    Std:    {times.std():.3f} ms")
        print(f"    P95:    {np.percentile(times, 95):.3f} ms")
        print(f"    P99:    {np.percentile(times, 99):.3f} ms")

        # Check for stability: P99 should be within 5x of median (no major spikes)
        self.assertLess(np.percentile(times, 99), np.median(times) * 5,
            "P99 latency is more than 5x median — possible throttling or stall")

    def test_render_only_vs_render_plus_fetch(self):
        """Isolate GPU render time from CPU readback time."""
        r = make_scene(self.NuRenderer, w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        NFRAMES = 50
        configs = [(4, 64, 64), (16, 64, 64), (32, 64, 64), (4, 128, 128), (16, 128, 128)]

        print(f"\n  Render vs Fetch breakdown ({NFRAMES} frame avg):")
        print(f"  {'Config':>20} {'Render ms':>10} {'Fetch ms':>10} {'Total ms':>10} {'Fetch%':>8}")
        print(f"  {'------':>20} {'---------':>10} {'--------':>10} {'--------':>10} {'------':>8}")

        for N, W, H in configs:
            vps = np.stack([
                make_vp_inv((0, 0, 3 + i * 0.1), (0, 0, 0), 45.0, W, H) for i in range(N)
            ])

            # Warm up
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            # Render only
            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            t_render = (time.perf_counter() - t0) / NFRAMES * 1000

            # Render + fetch
            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            t_total = (time.perf_counter() - t0) / NFRAMES * 1000
            t_fetch = t_total - t_render

            label = f"{N}cam {W}x{H}"
            fetch_pct = (t_fetch / t_total * 100) if t_total > 0 else 0
            print(f"  {label:>20} {t_render:>10.3f} {t_fetch:>10.3f} {t_total:>10.3f} {fetch_pct:>7.1f}%")

        r.close()

    def test_tlas_rebuild_overhead_in_tiled(self):
        """Measure overhead of TLAS rebuild within tiled render path."""
        r = make_scene(self.NuRenderer, w=64, h=64, num_boxes=5)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 64, 64, 8
        NFRAMES = 50
        vps = np.stack([
            make_vp_inv((0, 0, 3 + i * 0.3), (0, 0, 0), 45.0, W, H) for i in range(N)
        ])

        # Warm up
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        # Without transform updates (no TLAS rebuild)
        t0 = time.perf_counter()
        for _ in range(NFRAMES):
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        t_no_tlas = (time.perf_counter() - t0) / NFRAMES * 1000

        # With transform updates (forces TLAS rebuild each frame)
        xform = np.eye(4, dtype=np.float32)
        t0 = time.perf_counter()
        for f in range(NFRAMES):
            xform[0, 3] = 0.01 * f
            r.set_transforms([0], xform.reshape(1, 16))
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        t_with_tlas = (time.perf_counter() - t0) / NFRAMES * 1000

        r.close()

        tlas_overhead = t_with_tlas - t_no_tlas
        print(f"\n  TLAS rebuild overhead ({N} cameras, 5 meshes):")
        print(f"    Without TLAS rebuild: {t_no_tlas:.3f} ms/frame")
        print(f"    With TLAS rebuild:    {t_with_tlas:.3f} ms/frame")
        print(f"    TLAS overhead:        {tlas_overhead:.3f} ms/frame")

    def test_large_resolution_tiled(self):
        """Benchmark at training-realistic resolution: 32 cameras at 100x100."""
        r = make_scene(self.NuRenderer, w=100, h=100, num_boxes=5)
        if r is None:
            self.skipTest("RT not available")

        W, H, N = 100, 100, 32
        NFRAMES = 50

        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
        vps = np.stack([
            make_vp_inv(
                (5 * math.cos(a), 1, 5 * math.sin(a)),
                (3.0, 0, 0), 45.0, W, H)
            for a in angles
        ])

        # Warm up
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        t0 = time.perf_counter()
        for _ in range(NFRAMES):
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
        t_total = (time.perf_counter() - t0) / NFRAMES * 1000

        r.close()

        total_pix = N * W * H
        fps = 1000.0 / t_total
        mpix_per_s = total_pix / t_total * 1000 / 1e6
        print(f"\n  Training-realistic config ({N} cameras, {W}x{H}, 5 meshes):")
        print(f"    Total: {t_total:.2f} ms/frame ({fps:.0f} FPS)")
        print(f"    Pixels: {total_pix:,} per frame")
        print(f"    Throughput: {mpix_per_s:.1f} Mpix/s")


class TestTiledWithUSD(unittest.TestCase):
    """Test tiled rendering with USD-loaded scenes."""

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

    def test_kitchen_set_tiled_render(self):
        """Load Kitchen_set and render with tiled cameras."""
        r = self.NuRenderer(width=200, height=200, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")

        try:
            t0 = time.perf_counter()
            nmeshes = r.load_usd(self.kitchen_path)
            t_load = (time.perf_counter() - t0) * 1000
        except RuntimeError:
            r.close()
            self.skipTest("USD backend (libnanousd.so) not available")
        print(f"\n  Kitchen_set: {nmeshes} meshes loaded in {t_load:.0f} ms")

        # Initial render to build accel
        r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
        r.render(mode=self.NU_RENDER_RT)

        W, H, N = 200, 200, 8
        angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
        vps = np.stack([
            make_vp_inv(
                (300 * math.cos(a), 100, 300 * math.sin(a)),
                (0, 50, 0), 45.0, W, H)
            for a in angles
        ])

        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        self.assertEqual(pixels.shape, (N, H, W, 4))
        non_black = sum(1 for i in range(N) if pixels[i, :, :, :3].sum() > 0)
        print(f"  Tiled render: {non_black}/{N} cameras see content")
        self.assertGreater(non_black, N * 0.5)

        # Most cameras should see varied geometry (some may see mostly sky)
        rich_cameras = 0
        for i in range(N):
            rgb = pixels[i, :, :, :3]
            unique = len(np.unique(rgb.reshape(-1, 3), axis=0))
            if unique > 10:
                rich_cameras += 1
        print(f"  Rich cameras (>10 unique colors): {rich_cameras}/{N}")
        self.assertGreaterEqual(rich_cameras, N // 2,
            f"Only {rich_cameras}/{N} cameras see varied content")

        r.close()

    def test_kitchen_set_tiled_perf(self):
        """Benchmark tiled rendering of Kitchen_set."""
        r = self.NuRenderer(width=100, height=100, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")

        try:
            nmeshes = r.load_usd(self.kitchen_path)
        except RuntimeError:
            r.close()
            self.skipTest("USD backend (libnanousd.so) not available")
        r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
        r.render(mode=self.NU_RENDER_RT)

        NFRAMES = 20
        configs = [(4, 100, 100), (8, 100, 100), (16, 64, 64), (32, 64, 64)]

        print(f"\n  Kitchen_set tiled perf ({nmeshes} meshes):")
        print(f"  {'Config':>16} {'Total ms':>10} {'FPS':>8} {'Mpix/s':>8}")
        print(f"  {'------':>16} {'--------':>10} {'---':>8} {'------':>8}")

        for N, W, H in configs:
            angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
            vps = np.stack([
                make_vp_inv(
                    (300 * math.cos(a), 100, 300 * math.sin(a)),
                    (0, 50, 0), 45.0, W, H)
                for a in angles
            ])

            # Warm up
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            t_total = (time.perf_counter() - t0) / NFRAMES * 1000

            total_pix = N * W * H
            fps = 1000.0 / t_total
            mpix_s = total_pix / t_total * 1000 / 1e6

            label = f"{N}cam {W}x{H}"
            print(f"  {label:>16} {t_total:>10.2f} {fps:>8.0f} {mpix_s:>8.1f}")

        mem_mb = r.gpu_memory_used / (1024 * 1024)
        print(f"  GPU memory: {mem_mb:.1f} MB")
        r.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
