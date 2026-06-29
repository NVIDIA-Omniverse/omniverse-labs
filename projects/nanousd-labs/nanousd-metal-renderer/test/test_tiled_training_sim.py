# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-realistic tiled rendering simulation.

Simulates an IsaacLab-style RL training loop:
  - 32 parallel environments at 100x100
  - Camera poses update every frame (orbit around scene)
  - Measure sustained throughput over 500 frames
  - Track frame-time jitter, worst-case latency, memory stability
  - Verify image content changes per frame (not frozen)
"""

import math
import time
import unittest
import numpy as np
import os


# ---- VP matrix helpers (same convention as the renderer) ----

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


class TestTrainingSimulation(unittest.TestCase):
    """Simulate an RL training rendering loop."""

    @classmethod
    def setUpClass(cls):
        try:
            from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT
        except RuntimeError as e:
            raise unittest.SkipTest(f"libnusd_renderer.so not found: {e}")
        cls.NuRenderer = NuRenderer
        cls.NU_RENDER_RT = NU_RENDER_RT

    def _make_training_scene(self, w=100, h=100):
        """Create a scene with multiple objects (like a simple Cartpole env)."""
        r = self.NuRenderer(width=w, height=h, enable_rt=True)
        if not r.rt_available:
            r.close()
            return None

        # Cart (flat box)
        cart_v, cart_i = make_box(cx=0, cy=0, cz=0, hx=0.5, hy=0.1, hz=0.3)
        r.add_mesh(positions=cart_v, indices=cart_i,
                   display_color=(0.2, 0.4, 0.8), name="cart")

        # Pole (tall thin box)
        pole_v, pole_i = make_box(cx=0, cy=0.6, cz=0, hx=0.05, hy=0.5, hz=0.05)
        r.add_mesh(positions=pole_v, indices=pole_i,
                   display_color=(0.8, 0.2, 0.2), name="pole")

        # Ground plane
        ground_v, ground_i = make_box(cx=0, cy=-0.2, cz=0, hx=3, hy=0.05, hz=3)
        r.add_mesh(positions=ground_v, indices=ground_i,
                   display_color=(0.3, 0.5, 0.3), name="ground")

        # Initial render to build accel
        r.set_camera(eye=(0, 1, 3), target=(0, 0.5, 0))
        r.render(mode=self.NU_RENDER_RT)

        return r

    def test_32_envs_500_frames(self):
        """32 parallel environments, 100x100, 500 frames with changing cameras."""
        r = self._make_training_scene(w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        N, W, H = 32, 100, 100
        NFRAMES = 500

        frame_times = []
        render_times = []
        fetch_times = []
        prev_hash = None
        frozen_count = 0

        for frame in range(NFRAMES):
            # Each environment has its camera at a slightly different angle
            # and it orbits slowly
            base_angle = frame * 0.02
            vps = np.stack([
                make_vp_inv(
                    (3 * math.cos(base_angle + i * 0.2),
                     1 + 0.3 * math.sin(frame * 0.05 + i),
                     3 * math.sin(base_angle + i * 0.2)),
                    (0, 0.5, 0), 45.0, W, H)
                for i in range(N)
            ])

            t0 = time.perf_counter()
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            t1 = time.perf_counter()
            pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            t2 = time.perf_counter()

            render_times.append((t1 - t0) * 1000)
            fetch_times.append((t2 - t1) * 1000)
            frame_times.append((t2 - t0) * 1000)

            # Check that output changes between frames
            cur_hash = hash(pixels.tobytes()[:1024])
            if cur_hash == prev_hash:
                frozen_count += 1
            prev_hash = cur_hash

        r.close()

        frame_arr = np.array(frame_times)
        render_arr = np.array(render_times)
        fetch_arr = np.array(fetch_times)

        mean_fps = 1000.0 / frame_arr.mean()
        total_pixels = N * W * H
        mpix_s = total_pixels * mean_fps / 1e6

        print(f"\n  Training simulation: {N} envs, {W}x{H}, {NFRAMES} frames")
        print(f"    Frame time:  mean={frame_arr.mean():.2f} ms, "
              f"p50={np.median(frame_arr):.2f}, p95={np.percentile(frame_arr,95):.2f}, "
              f"p99={np.percentile(frame_arr,99):.2f}, max={frame_arr.max():.2f} ms")
        print(f"    Render time: mean={render_arr.mean():.2f} ms")
        print(f"    Fetch time:  mean={fetch_arr.mean():.2f} ms ({100*fetch_arr.mean()/frame_arr.mean():.0f}% of frame)")
        print(f"    Throughput:  {mean_fps:.0f} FPS, {mpix_s:.1f} Mpix/s")
        print(f"    Frozen frames: {frozen_count}/{NFRAMES}")

        # Assertions
        self.assertLess(frozen_count, NFRAMES * 0.05,
            f"Too many frozen frames: {frozen_count}/{NFRAMES}")
        # Should sustain at least 100 FPS for this simple scene
        self.assertGreater(mean_fps, 100,
            f"Throughput too low: {mean_fps:.0f} FPS")

    def test_64_envs_200_frames(self):
        """64 parallel environments at lower resolution."""
        r = self._make_training_scene(w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        N, W, H = 64, 64, 64
        NFRAMES = 200

        frame_times = []
        for frame in range(NFRAMES):
            base_angle = frame * 0.03
            vps = np.stack([
                make_vp_inv(
                    (3 * math.cos(base_angle + i * 0.1),
                     1 + 0.2 * math.sin(frame * 0.04 + i),
                     3 * math.sin(base_angle + i * 0.1)),
                    (0, 0.5, 0), 45.0, W, H)
                for i in range(N)
            ])

            t0 = time.perf_counter()
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            frame_times.append((time.perf_counter() - t0) * 1000)

        r.close()

        arr = np.array(frame_times)
        mean_fps = 1000.0 / arr.mean()
        mpix_s = N * W * H * mean_fps / 1e6

        print(f"\n  Training simulation: {N} envs, {W}x{H}, {NFRAMES} frames")
        print(f"    Frame: mean={arr.mean():.2f} ms, p95={np.percentile(arr,95):.2f}, max={arr.max():.2f} ms")
        print(f"    Throughput: {mean_fps:.0f} FPS, {mpix_s:.1f} Mpix/s")

        self.assertGreater(mean_fps, 50, f"Throughput too low: {mean_fps:.0f} FPS")

    def test_memory_stability_1000_frames(self):
        """1000 frames, verify no memory growth."""
        r = self._make_training_scene(w=64, h=64)
        if r is None:
            self.skipTest("RT not available")

        N, W, H = 16, 64, 64
        import resource

        # Warm up
        vps = np.stack([make_vp_inv((0,0,3), (0,0,0), 45.0, W, H)] * N)
        r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
        r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        for frame in range(1000):
            angle = frame * 0.01
            vps = np.stack([
                make_vp_inv(
                    (3*math.cos(angle + i), 1, 3*math.sin(angle + i)),
                    (0, 0, 0), 45.0, W, H)
                for i in range(N)
            ])
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

        mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        growth = mem_end - mem_start

        r.close()

        print(f"\n  Memory stability (1000 frames, {N} cams {W}x{H}):")
        print(f"    Start: {mem_start:.1f} MB, End: {mem_end:.1f} MB, Growth: {growth:.2f} MB")

        self.assertLess(growth, 50.0,
            f"Memory grew by {growth:.1f} MB over 1000 frames")

    def test_transform_animation_training(self):
        """Animate mesh transforms (cart sliding) while rendering tiled.

        Uses larger movements (cart translating +-1.5 units) to ensure
        clearly visible per-frame changes.
        """
        r = self._make_training_scene(w=100, h=100)
        if r is None:
            self.skipTest("RT not available")

        N, W, H = 4, 100, 100
        NFRAMES = 100

        prev_pixels = None
        change_count = 0

        for frame in range(NFRAMES):
            # Slide the cart (mesh 0) back and forth by 1.5 units.
            # Row-vector convention: translation in last row.
            dx = 1.5 * math.sin(frame * 0.2)
            xform = np.array([
                1,  0, 0, 0,
                0,  1, 0, 0,
                0,  0, 1, 0,
                dx, 0, 0, 1
            ], dtype=np.float32)
            r.set_transforms(mesh_ids=[0], transforms=xform.reshape(1, 16))

            vps = np.stack([
                make_vp_inv(
                    (3*math.cos(i*1.6), 1.5, 3*math.sin(i*1.6)),
                    (0, 0.5, 0), 45.0, W, H)
                for i in range(N)
            ])

            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            pixels = r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            if prev_pixels is not None:
                # Count pixels that changed by more than 5 values
                changed = np.any(np.abs(pixels.astype(int) - prev_pixels.astype(int)) > 5, axis=3)
                pct = 100 * changed.sum() / changed.size
                if pct > 0.1:  # at least 0.1% of pixels changed
                    change_count += 1
            prev_pixels = pixels.copy()

        r.close()

        pct_changed = 100 * change_count / (NFRAMES - 1)
        print(f"\n  Transform animation: {change_count}/{NFRAMES-1} frames changed ({pct_changed:.0f}%)")

        # Most frames should show visual change from cart animation
        self.assertGreater(pct_changed, 50,
            f"Only {pct_changed:.0f}% of frames changed — TLAS rebuild may not work with tiled rendering")

    def test_scaling_comparison(self):
        """Compare throughput across different env counts and resolutions."""
        r = self._make_training_scene(w=200, h=200)
        if r is None:
            self.skipTest("RT not available")

        configs = [
            (8,  100, 100),
            (16, 100, 100),
            (32, 100, 100),
            (64,  64,  64),
            (128, 32,  32),
        ]

        NFRAMES = 50
        print(f"\n  Scaling comparison ({NFRAMES} frames each):")
        print(f"  {'Config':>18} {'Frame ms':>10} {'FPS':>8} {'Mpix/s':>8}")
        print(f"  {'------':>18} {'--------':>10} {'---':>8} {'------':>8}")

        for N, W, H in configs:
            vps = np.stack([
                make_vp_inv(
                    (3*math.cos(i*0.2), 1, 3*math.sin(i*0.2)),
                    (0, 0.5, 0), 45.0, W, H)
                for i in range(N)
            ])

            # Warm up
            r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
            r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)

            t0 = time.perf_counter()
            for _ in range(NFRAMES):
                r.render_tiled(vps, num_cameras=N, tile_w=W, tile_h=H, mode=self.NU_RENDER_RT)
                r.fetch_pixels_tiled(num_cameras=N, tile_w=W, tile_h=H)
            mean_ms = (time.perf_counter() - t0) / NFRAMES * 1000

            fps = 1000.0 / mean_ms
            mpix = N * W * H * fps / 1e6
            print(f"  {f'{N}x{W}x{H}':>18} {mean_ms:>10.2f} {fps:>8.0f} {mpix:>8.1f}")

        r.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
