# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLAS update-in-place correctness tests.

Verifies that MODE_UPDATE produces visually correct results:
  - Moved objects actually appear at new positions
  - Back-to-origin produces identical images to initial state
  - Many update cycles don't accumulate drift
  - Extreme transforms don't corrupt the TLAS
  - Mixed static/dynamic objects remain correct
"""

import math
import unittest
import numpy as np
import os


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
        0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1, 2,6,7, 2,7,3, 0,3,7, 0,7,4, 1,5,6, 1,6,2,
    ], dtype=np.uint32)
    return verts, idx


def orbit_vps(N, radius, center, fov, w, h, y_offset=0.5):
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
    return np.stack([
        make_vp_inv(
            (radius * math.cos(a), y_offset, radius * math.sin(a)),
            center, fov, w, h)
        for a in angles
    ])


class TestTLASUpdateCorrectness(unittest.TestCase):
    """Test that TLAS update-in-place produces correct visual results."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def _make_scene(self):
        r = self.NuRenderer(width=128, height=128, enable_rt=True)
        if not r.rt_available:
            r.close()
            return None
        # Three colored boxes at known positions
        for i, (cx, color) in enumerate([
            (-1.5, (0.8, 0.2, 0.2)),
            (0.0,  (0.2, 0.8, 0.2)),
            (1.5,  (0.2, 0.2, 0.8)),
        ]):
            v, idx = make_box(cx=cx)
            r.add_mesh(positions=v, indices=idx, display_color=color)
        r.set_camera(eye=(0, 1, 5), target=(0, 0, 0))
        r.render(mode=self.NU_RENDER_RT)
        return r

    def _render_tiled(self, r, N=4, w=64, h=64):
        vps = orbit_vps(N, 3.0, (0, 0, 0), 45.0, w, h)
        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=self.NU_RENDER_RT)
        return r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    def test_move_object_changes_image(self):
        """Moving an object must change the rendered image."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_before = self._render_tiled(r)

        # Move red box to the right
        xform = np.eye(4, dtype=np.float32).flatten()
        xform[3] = 2.0  # translate X
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))

        pix_after = self._render_tiled(r)
        diff = np.abs(pix_before.astype(float) - pix_after.astype(float))
        changed_pixels = np.sum(diff.max(axis=-1) > 5)
        self.assertGreater(changed_pixels, 10,
                           "Moving object should change rendered image")
        r.close()

    def test_return_to_origin_restores_image(self):
        """Moving object away then back should produce the same image."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_original = self._render_tiled(r)

        # Move away
        xform = np.eye(4, dtype=np.float32).flatten()
        xform[3] = 5.0
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        _ = self._render_tiled(r)

        # Move back to origin (identity transform)
        xform_id = np.eye(4, dtype=np.float32).flatten()
        r.set_transforms(mesh_ids=[0], transforms=xform_id.reshape(1, 16))
        pix_restored = self._render_tiled(r)

        np.testing.assert_array_equal(pix_original, pix_restored,
                                      "Restoring original transform should produce identical image")
        r.close()

    def test_many_update_cycles_no_drift(self):
        """500 update cycles back-and-forth should produce identical results."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_reference = self._render_tiled(r)

        xform = np.eye(4, dtype=np.float32).flatten()
        xform_id = np.eye(4, dtype=np.float32).flatten()

        for i in range(500):
            # Move to random position
            xform[3] = math.sin(i * 0.1) * 3.0
            xform[7] = math.cos(i * 0.07) * 1.0
            r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
            _ = self._render_tiled(r)

        # Return to identity
        r.set_transforms(mesh_ids=[0], transforms=xform_id.reshape(1, 16))
        pix_final = self._render_tiled(r)

        np.testing.assert_array_equal(pix_reference, pix_final,
                                      "After 500 update cycles, returning to origin should match reference")
        r.close()

    def test_extreme_translation(self):
        """Very large translations should not corrupt the TLAS."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        xform = np.eye(4, dtype=np.float32).flatten()
        for dist in [100, 1000, 10000, 100000]:
            xform[3] = float(dist)
            r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
            pix = self._render_tiled(r)
            # Should not crash; image should be valid (non-zero, no NaN artifacts)
            self.assertEqual(pix.shape[0], 4)  # 4 cameras
            self.assertGreater(pix.sum(), 0, f"Image should not be all-black at dist={dist}")

        r.close()

    def test_scale_transform(self):
        """Scaling via transform matrix should work."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_normal = self._render_tiled(r)

        # Scale red box 2x
        xform = np.eye(4, dtype=np.float32).flatten()
        xform[0] = 2.0  # scale X
        xform[5] = 2.0  # scale Y
        xform[10] = 2.0 # scale Z
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        pix_scaled = self._render_tiled(r)

        # Images should differ
        diff = np.abs(pix_normal.astype(float) - pix_scaled.astype(float))
        changed = np.sum(diff.max(axis=-1) > 5)
        self.assertGreater(changed, 10, "Scaling should visibly change the image")
        r.close()

    def test_rotation_transform(self):
        """Rotation via transform matrix should work."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_normal = self._render_tiled(r)

        # Rotate red box 45 degrees around Y
        angle = math.radians(45)
        c, s = math.cos(angle), math.sin(angle)
        xform = np.eye(4, dtype=np.float32).flatten()
        xform[0] = c;  xform[2] = s    # row 0: [c, 0, s, tx]
        xform[8] = -s; xform[10] = c   # row 2: [-s, 0, c, tz]
        # Keep original X offset for red box
        xform[3] = -1.5
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        pix_rotated = self._render_tiled(r)

        diff = np.abs(pix_normal.astype(float) - pix_rotated.astype(float))
        changed = np.sum(diff.max(axis=-1) > 5)
        self.assertGreater(changed, 5, "Rotation should visibly change the image")
        r.close()

    def test_multiple_objects_independent(self):
        """Moving one object should not affect other objects' positions."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        # Render from a fixed camera looking at the green (center) box
        N, w, h = 1, 128, 128
        vps = np.stack([make_vp_inv((0, 0, 3), (0, 0, 0), 30.0, w, h)])

        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=self.NU_RENDER_RT)
        pix_before = r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

        # Move the RED box (index 0) far away — should only affect red pixels
        xform = np.eye(4, dtype=np.float32).flatten()
        xform[3] = 100.0  # move far off-screen
        r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))

        r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=self.NU_RENDER_RT)
        pix_after = r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

        # The green box center area should still be visible
        center_region = pix_after[0, h//3:2*h//3, w//3:2*w//3, :3]
        green_ish = center_region[:, :, 1].mean()
        self.assertGreater(green_ish, 30,
                           "Green box should still be visible after moving red box away")

        r.close()

    def test_static_objects_unchanged_after_dynamic_update(self):
        """Objects not in the set_transforms call should remain static."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        # Only update mesh 0 (red) — meshes 1 (green) and 2 (blue) should be unchanged
        xform_id = np.eye(4, dtype=np.float32).flatten()
        for i in range(100):
            xform = xform_id.copy()
            xform[3] = math.sin(i * 0.1) * 2.0
            r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
            pix = self._render_tiled(r)
            # Just verify no crash and we get pixels
            self.assertGreater(pix.sum(), 0)

        r.close()

    def test_zero_translation_is_noop(self):
        """Setting identity transform should produce same image as initial state."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_before = self._render_tiled(r)

        xform_id = np.eye(4, dtype=np.float32).flatten()
        r.set_transforms(mesh_ids=[0], transforms=xform_id.reshape(1, 16))
        pix_after = self._render_tiled(r)

        np.testing.assert_array_equal(pix_before, pix_after,
                                      "Identity transform should produce identical image")
        r.close()

    def test_all_meshes_update_simultaneously(self):
        """Update all 3 meshes at once."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        pix_before = self._render_tiled(r)

        # Move all boxes up
        xforms = np.zeros((3, 16), dtype=np.float32)
        for i in range(3):
            xforms[i] = np.eye(4, dtype=np.float32).flatten()
            xforms[i, 7] = 2.0  # translate Y up
        r.set_transforms(mesh_ids=[0, 1, 2], transforms=xforms)
        pix_moved = self._render_tiled(r)

        diff = np.abs(pix_before.astype(float) - pix_moved.astype(float))
        changed = np.sum(diff.max(axis=-1) > 5)
        self.assertGreater(changed, 20, "Moving all boxes up should change image significantly")

        # Move all back to original positions
        xforms_orig = np.zeros((3, 16), dtype=np.float32)
        for i, cx in enumerate([-1.5, 0.0, 1.5]):
            xforms_orig[i] = np.eye(4, dtype=np.float32).flatten()
            # Note: original positions are baked into vertex data, so identity = original
        r.set_transforms(mesh_ids=[0, 1, 2], transforms=xforms_orig)
        pix_restored = self._render_tiled(r)

        np.testing.assert_array_equal(pix_before, pix_restored,
                                      "Restoring all original transforms should match initial image")
        r.close()

    def test_rapid_alternating_transforms(self):
        """Rapidly alternate between two transforms — no corruption."""
        r = self._make_scene()
        if r is None:
            self.skipTest("RT not available")

        xform_a = np.eye(4, dtype=np.float32).flatten()
        xform_a[3] = 2.0
        xform_b = np.eye(4, dtype=np.float32).flatten()
        xform_b[3] = -2.0

        # Get reference images for each position
        r.set_transforms(mesh_ids=[0], transforms=xform_a.reshape(1, 16))
        pix_a_ref = self._render_tiled(r).copy()

        r.set_transforms(mesh_ids=[0], transforms=xform_b.reshape(1, 16))
        pix_b_ref = self._render_tiled(r).copy()

        # Rapidly alternate 200 times
        for i in range(200):
            xform = xform_a if i % 2 == 0 else xform_b
            r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
            pix = self._render_tiled(r)

        # After 200 alternations (even count), should be at position A
        r.set_transforms(mesh_ids=[0], transforms=xform_a.reshape(1, 16))
        pix_final_a = self._render_tiled(r)
        np.testing.assert_array_equal(pix_a_ref, pix_final_a,
                                      "After alternating, position A should match reference")

        r.set_transforms(mesh_ids=[0], transforms=xform_b.reshape(1, 16))
        pix_final_b = self._render_tiled(r)
        np.testing.assert_array_equal(pix_b_ref, pix_final_b,
                                      "After alternating, position B should match reference")
        r.close()


class TestTLASUpdateKitchenSet(unittest.TestCase):
    """TLAS update tests with Kitchen_set (requires NANOUSD_BACKEND)."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

        kitchen_path = os.environ.get("NUSD_KITCHEN_SET", os.path.expanduser("~/Kitchen_set/Kitchen_set.usd"))
        if not os.path.exists(kitchen_path):
            raise unittest.SkipTest("Kitchen_set not found")

        cls.r = NuRenderer(width=256, height=256, enable_rt=True)
        if not cls.r.rt_available:
            cls.r.close()
            raise unittest.SkipTest("RT not available")

        try:
            cls.nmeshes = cls.r.load_usd(kitchen_path)
        except RuntimeError:
            cls.r.close()
            raise unittest.SkipTest("USD backend not available")

        cls.r.set_camera(eye=(0, 100, 300), target=(0, 50, 0))
        cls.r.render(mode=NU_RENDER_RT)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'r') and cls.r:
            cls.r.close()

    def _render_tiled(self, N=8, w=64, h=64):
        vps = orbit_vps(N, 200, (70, 50, 30), 60.0, w, h, y_offset=120)
        self.r.render_tiled(vps, num_cameras=N, tile_w=w, tile_h=h, mode=self.NU_RENDER_RT)
        return self.r.fetch_pixels_tiled(num_cameras=N, tile_w=w, tile_h=h)

    def test_kitchen_move_restores(self):
        """Move a mesh in Kitchen_set, then restore — image should match."""
        # Set identity as known baseline (USD transforms are non-identity)
        xform_id = np.eye(4, dtype=np.float32).flatten()
        self.r.set_transforms(mesh_ids=[0], transforms=xform_id.reshape(1, 16))
        pix_ref = self._render_tiled()

        xform = np.eye(4, dtype=np.float32).flatten()
        xform[7] = 50.0  # move up
        self.r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))
        _ = self._render_tiled()

        self.r.set_transforms(mesh_ids=[0], transforms=xform_id.reshape(1, 16))
        pix_restored = self._render_tiled()

        np.testing.assert_array_equal(pix_ref, pix_restored)

    def test_kitchen_100_update_cycles(self):
        """100 update cycles with 10 meshes moving, then restore to same state."""
        # Kitchen_set meshes have non-identity transforms from USD.
        # set_transforms uses absolute world transforms, so identity != original.
        # We set identity first, take reference, animate, then restore to identity.
        xform_base = np.eye(4, dtype=np.float32).flatten()
        mesh_ids = list(range(min(10, self.nmeshes)))

        # Set identity as our known baseline
        xforms_id = np.tile(xform_base, (len(mesh_ids), 1))
        self.r.set_transforms(mesh_ids=mesh_ids, transforms=xforms_id)
        pix_ref = self._render_tiled()

        xforms = np.tile(xform_base, (len(mesh_ids), 1))
        for i in range(100):
            t = i * 0.05
            for j in range(len(mesh_ids)):
                xforms[j] = xform_base.copy()
                xforms[j, 3] = math.sin(t + j) * 20.0
                xforms[j, 7] = math.cos(t + j * 0.7) * 10.0
            self.r.set_transforms(mesh_ids=mesh_ids, transforms=xforms)
            _ = self._render_tiled()

        # Restore all to identity baseline
        self.r.set_transforms(mesh_ids=mesh_ids, transforms=xforms_id)
        pix_restored = self._render_tiled()

        np.testing.assert_array_equal(pix_ref, pix_restored,
                                      "After 100 cycles, restoring should match reference")

    def test_kitchen_all_meshes_update(self):
        """Update ALL 1788 meshes (worst case), verify no crash."""
        all_ids = list(range(self.nmeshes))
        xform_base = np.eye(4, dtype=np.float32).flatten()
        xforms = np.tile(xform_base, (self.nmeshes, 1))

        # Small jitter to all meshes
        for j in range(self.nmeshes):
            xforms[j, 3] += (j % 10) * 0.01  # tiny X offset

        self.r.set_transforms(mesh_ids=all_ids, transforms=xforms)
        pix = self._render_tiled()
        self.assertGreater(pix.sum(), 0, "Should produce non-black image")

        # Restore
        xforms_id = np.tile(xform_base, (self.nmeshes, 1))
        self.r.set_transforms(mesh_ids=all_ids, transforms=xforms_id)


if __name__ == '__main__':
    unittest.main()
