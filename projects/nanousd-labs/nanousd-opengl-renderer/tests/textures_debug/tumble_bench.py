# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tumble benchmark — render N frames orbiting the warehouse and report
per-frame timings. Used to profile interactive frame rate at a fixed
window size.

Usage:
    python tests/textures_debug/tumble_bench.py [usd] [W H] [frames]
"""

from __future__ import annotations
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))

from nusd_gles._bindings import GlesViewer


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def main():
    usd = sys.argv[1] if len(sys.argv) > 1 else \
        os.environ.get(
            "NUSD_ISAAC_WAREHOUSE",
            str(Path.home() / "assets/isaac/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"),
        )
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    nframes = int(sys.argv[4]) if len(sys.argv) > 4 else 240

    print(f"Loading {usd} at {width}x{height}")
    t0 = time.perf_counter()
    v = GlesViewer(usd, width, height)
    bounds = v.get_scene_bounds()
    t1 = time.perf_counter()
    print(f"Load+bounds: {(t1 - t0) * 1000:.1f} ms")

    if bounds is None:
        print("No bounds")
        return 1
    mn, mx = np.array(bounds[0], dtype=np.float32), np.array(bounds[1], dtype=np.float32)
    center = (mn + mx) * 0.5
    sx, sy, sz = mx - mn
    target = np.array([center[0], mn[1] + min(1.6, 0.5 * sy), center[2]], dtype=np.float32)
    distance = max(min(0.12 * max(sx, sz), 12.0), 5.0)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    proj = perspective(60.0, width / height, 0.1, max(sx, sy, sz) * 4.0)
    out = np.empty((height, width, 4), dtype=np.uint8)

    # Warmup — first frame includes shader compile / texture upload paths.
    yaw = 0.0
    pitch = 0.3
    eye = target + np.array([
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    ], dtype=np.float32) * distance
    v.render(width, height, look_at(eye, target, up), proj, eye, out=out)

    # Tumble.
    print(f"Tumbling {nframes} frames...")
    times_ms = np.empty(nframes, dtype=np.float64)
    t_start = time.perf_counter()
    for i in range(nframes):
        yaw = 2.0 * math.pi * i / nframes
        eye = target + np.array([
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ], dtype=np.float32) * distance
        view = look_at(eye, target, up)
        t_a = time.perf_counter()
        v.render(width, height, view, proj, eye, out=out)
        t_b = time.perf_counter()
        times_ms[i] = (t_b - t_a) * 1000.0
    t_end = time.perf_counter()

    v.close()

    wall_ms = (t_end - t_start) * 1000.0
    fps_overall = nframes / (wall_ms / 1000.0)
    p50 = np.percentile(times_ms, 50)
    p90 = np.percentile(times_ms, 90)
    p99 = np.percentile(times_ms, 99)
    mn_ms = times_ms.min()
    mx_ms = times_ms.max()
    print(f"\n=== {nframes} frames @ {width}x{height} ===")
    print(f"  wall      : {wall_ms:.1f} ms  ({fps_overall:.1f} fps overall)")
    print(f"  per-frame : min={mn_ms:.2f}  p50={p50:.2f}  p90={p90:.2f}  p99={p99:.2f}  max={mx_ms:.2f}  ms")
    print(f"  fps proxy : p50={1000.0/p50:.1f}  p90={1000.0/p90:.1f}  p99={1000.0/p99:.1f}")

    # Save a sample frame from the middle.
    dump = REPO / "tests/textures_debug/dumps/tumble_frame.ppm"
    dump.parent.mkdir(exist_ok=True)
    with open(dump, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(out[..., :3].tobytes())
    print(f"  sample    : {dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
