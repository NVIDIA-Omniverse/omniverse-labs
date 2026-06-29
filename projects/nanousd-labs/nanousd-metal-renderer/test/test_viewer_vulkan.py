# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nusd_renderer Python bindings and NuRenderer wrapper."""

import unittest
import numpy as np
import sys


class TestNuRenderer(unittest.TestCase):
    """Test the NuRenderer Python wrapper around libnusd_renderer.so."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer

    def test_create_destroy(self):
        """Renderer can be created and destroyed."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        self.assertIsNotNone(r._handle)
        r.close()
        self.assertIsNone(r._handle)

    def test_add_mesh_triangle(self):
        """Add a single triangle mesh."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ], dtype=np.float32)
        normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        mesh_id = r.add_mesh(
            positions=positions,
            indices=indices,
            normals=normals,
            display_color=(0.2, 0.6, 1.0),
            name="triangle",
        )
        self.assertEqual(mesh_id, 0)
        r.close()

    def test_add_multiple_meshes(self):
        """Add and remove multiple meshes."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        id0 = r.add_mesh(positions=positions, indices=indices, name="m0")
        id1 = r.add_mesh(positions=positions, indices=indices, name="m1")
        self.assertEqual(id0, 0)
        self.assertEqual(id1, 1)

        r.remove_mesh(id0)
        # Slot 0 is now free — next add should reuse it
        id2 = r.add_mesh(positions=positions, indices=indices, name="m2")
        self.assertEqual(id2, 0)
        r.close()

    def test_raster_render_and_fetch(self):
        """Render a raster frame and fetch pixels."""
        from nusd_renderer._bindings import NU_RENDER_RASTER

        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        r.add_mesh(
            positions=positions,
            indices=indices,
            normals=normals,
            display_color=(1.0, 0.0, 0.0),
            name="red_tri",
        )
        r.set_camera(eye=(0, 0, 3), target=(0, 0, 0), fov_degrees=45.0)
        r.render(mode=NU_RENDER_RASTER)

        pixels = r.fetch_pixels()
        self.assertEqual(pixels.shape, (240, 320, 4))
        self.assertEqual(pixels.dtype, np.uint8)
        # Alpha channel should be 255 everywhere
        self.assertTrue(np.all(pixels[:, :, 3] == 255))
        # Should have some non-zero color data
        self.assertTrue(np.any(pixels[:, :, :3] > 0))
        r.close()

    def test_rt_render_if_available(self):
        """Render an RT frame if hardware supports it."""
        from nusd_renderer._bindings import NU_RENDER_RT, NU_RENDER_RASTER

        r = self.NuRenderer(width=320, height=240, enable_rt=True)
        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        r.add_mesh(positions=positions, indices=indices, normals=normals, name="tri")
        r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))

        if r.rt_available:
            r.render(mode=NU_RENDER_RT)
        else:
            r.render(mode=NU_RENDER_RASTER)

        pixels = r.fetch_pixels()
        self.assertEqual(pixels.shape, (240, 320, 4))
        self.assertTrue(np.any(pixels[:, :, :3] > 0))
        r.close()

    def test_set_transforms(self):
        """Transform update does not crash."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        mid = r.add_mesh(positions=positions, indices=indices, name="t")
        # Identity + translation
        xform = np.eye(4, dtype=np.float32)
        xform[0, 3] = 1.0  # translate X
        r.set_transforms([mid], xform.reshape(1, 16))
        r.close()

    def test_set_colors(self):
        """Color update does not crash."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        mid = r.add_mesh(positions=positions, indices=indices, name="c")
        r.set_colors([mid], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        r.close()

    def test_gpu_memory_tracking(self):
        """GPU memory counter is non-zero after mesh upload."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        r.add_mesh(positions=positions, indices=indices, normals=normals, name="mem")
        self.assertGreater(r.gpu_memory_used, 0)
        r.close()

    def test_multiple_frames(self):
        """Render multiple frames without crash (tests persistent staging buffer)."""
        from nusd_renderer._bindings import NU_RENDER_RASTER

        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        r.add_mesh(positions=positions, indices=indices, normals=normals, name="multi")
        r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))

        for i in range(5):
            r.render(mode=NU_RENDER_RASTER)
            pixels = r.fetch_pixels()
            self.assertEqual(pixels.shape, (240, 320, 4))
        r.close()

    def test_rt_tlas_update(self):
        """RT render with transform updates triggers TLAS rebuild without crash."""
        from nusd_renderer._bindings import NU_RENDER_RT

        r = self.NuRenderer(width=320, height=240, enable_rt=True)
        if not r.rt_available:
            r.close()
            self.skipTest("RT not available")

        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ], dtype=np.float32)
        normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        mid = r.add_mesh(positions=positions, indices=indices, normals=normals, name="rt_tri")
        r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))

        # Initial RT render
        r.render(mode=NU_RENDER_RT)
        p1 = r.fetch_pixels()
        self.assertEqual(p1.shape, (240, 320, 4))

        # Move mesh and render again (triggers TLAS rebuild).
        # Row-vector convention (USD): translation in last row.
        xform = np.eye(4, dtype=np.float32)
        xform[3, 0] = 0.5  # translate +X
        r.set_transforms([mid], xform.reshape(1, 16))
        r.render(mode=NU_RENDER_RT)
        p2 = r.fetch_pixels()
        self.assertEqual(p2.shape, (240, 320, 4))

        # Frames should differ (mesh moved)
        self.assertFalse(np.array_equal(p1, p2), "RT frames should differ after transform update")
        r.close()

    def test_clear_scene(self):
        """clear_scene removes all meshes."""
        r = self.NuRenderer(width=320, height=240, enable_rt=False)
        positions = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        r.add_mesh(positions=positions, indices=indices, name="a")
        r.add_mesh(positions=positions, indices=indices, name="b")
        r.clear_scene()

        # After clear, next mesh should get id 0
        mid = r.add_mesh(positions=positions, indices=indices, name="c")
        self.assertEqual(mid, 0)
        r.close()


if __name__ == "__main__":
    unittest.main()
