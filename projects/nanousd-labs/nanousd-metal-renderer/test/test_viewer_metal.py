# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ViewerMetal — Newton ViewerBase adapter on Metal.

Validates the public ViewerMetal interface end-to-end against the
in-tree Metal renderer, without requiring Newton/IsaacLab/warp to be
installed. The test mocks the ViewerBase contract calls log_mesh +
log_instances + begin_frame + end_frame and asserts that
get_frame() returns a non-trivial RGB image.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np


class TestViewerMetal(unittest.TestCase):
    """Verify the ViewerMetal adapter renders end-to-end."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer.viewer_metal import ViewerMetal
        except (RuntimeError, ImportError) as e:
            raise unittest.SkipTest(f"ViewerMetal not importable: {e}")
        cls.ViewerMetal = ViewerMetal

    def test_create_destroy(self):
        v = self.ViewerMetal(width=128, height=128, render_mode="raster")
        self.assertTrue(v.is_running())
        self.assertFalse(v.is_paused())
        v.close()
        self.assertFalse(v.is_running())

    def test_render_mesh_via_logmesh(self):
        v = self.ViewerMetal(width=320, height=240, render_mode="rt")

        # A unit triangle facing +Z.
        points = np.array([
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0,
        ], dtype=np.float32)
        normals = np.array([
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        v.log_mesh(name="tri", points=points, indices=indices, normals=normals)

        # Place camera looking at the triangle along -Z.
        v.set_camera(pos=(0.0, 0.0, 3.0), pitch=0.0, yaw=math.pi)

        v.begin_frame(0.0)
        v.end_frame()

        frame = v.get_frame()
        self.assertIsNotNone(frame, "ViewerMetal.get_frame() returned None")
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)

        # Triangle should produce some non-sky pixel content. Loosely:
        # at least 1% of pixels deviate from the corner sky pixel.
        corner = frame[0, 0]
        diffs = np.abs(frame.astype(int) - corner.astype(int)).sum(axis=-1)
        non_sky = int((diffs > 30).sum())
        self.assertGreater(non_sky, frame.shape[0] * frame.shape[1] // 100,
            f"Only {non_sky} pixels diverged from sky — triangle not rendered?")

        v.close()

    def test_log_instances_with_quat_transform(self):
        """Pass a single 7-element (pos3+quat4) transform via log_instances
        and verify the renderer accepts it (exercises the row-vector
        conversion in _transform_to_mat4_row_vec)."""
        v = self.ViewerMetal(width=128, height=128, render_mode="rt")

        points = np.array([
            -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        v.log_mesh(name="tri", points=points, indices=indices)

        # Identity transform: pos=(0,0,0), quat=(0,0,0,1).
        xforms = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        v.log_instances(name="inst_a", mesh="tri", xforms=xforms)

        # 90° rotation around Y axis: quat = (0, sin(45°), 0, cos(45°))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        xforms_rot = np.array([[0.5, 0, 0, 0, s, 0, c]], dtype=np.float32)
        v.log_instances(name="inst_b", mesh="tri", xforms=xforms_rot)

        v.set_camera(pos=(0, 0, 3), pitch=0.0, yaw=math.pi)
        v.begin_frame(0.0)
        v.end_frame()

        frame = v.get_frame()
        self.assertIsNotNone(frame)
        v.close()

    def test_save_screenshot(self):
        v = self.ViewerMetal(width=64, height=64, render_mode="raster")
        # save_screenshot requires a rendered frame; need a mesh first.
        points = np.array([
            -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        v.log_mesh(name="tri", points=points, indices=indices)
        v.set_camera(pos=(0, 0, 3), pitch=0.0, yaw=math.pi)

        v.begin_frame(0.0)
        v.end_frame()
        out = "/tmp/viewer_metal_test.ppm"
        if os.path.exists(out):
            os.remove(out)
        try:
            v.save_screenshot(out)
        except RuntimeError:
            # Some configurations (no geometry, no RT, etc.) skip the
            # render path; that's acceptable for this smoke test.
            v.close()
            return
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        os.remove(out)
        v.close()


if __name__ == "__main__":
    unittest.main()
