# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-grained readback timing."""

import time
import ctypes
import numpy as np


def bench():
    from nusd_renderer._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_RT, _Lib

    W, H = 1920, 1080
    r = NuRenderer(width=W, height=H, enable_rt=True)

    positions = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
    normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    r.add_mesh(positions=positions, indices=indices, normals=normals, name="t")
    r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))

    lib = _Lib.get()

    # Warm up
    r.render(mode=NU_RENDER_RASTER)

    # Time just the render
    t0 = time.perf_counter()
    r.render(mode=NU_RENDER_RASTER)
    t_render = (time.perf_counter() - t0) * 1000

    # Time just the fetch
    buf = np.empty((H, W, 4), dtype=np.uint8)
    t0 = time.perf_counter()
    res = lib.nu_fetch_pixels(r._handle, buf.ctypes.data_as(ctypes.c_void_p), 0)
    t_fetch = (time.perf_counter() - t0) * 1000

    print(f"Render:  {t_render:.2f} ms")
    print(f"Fetch:   {t_fetch:.2f} ms")
    print(f"Total:   {t_render + t_fetch:.2f} ms")

    # Multiple fetches (staging buffer cached)
    times = []
    for i in range(5):
        r.render(mode=NU_RENDER_RASTER)
        t0 = time.perf_counter()
        lib.nu_fetch_pixels(r._handle, buf.ctypes.data_as(ctypes.c_void_p), 0)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\n5x fetch times: {[f'{t:.1f}' for t in times]} ms")
    print(f"Average:  {sum(times)/len(times):.1f} ms")

    # Try smaller resolution
    r.close()
    r2 = NuRenderer(width=320, height=240, enable_rt=False)
    r2.add_mesh(positions=positions, indices=indices, normals=normals, name="t2")
    r2.set_camera(eye=(0, 0, 3), target=(0, 0, 0))
    r2.render(mode=NU_RENDER_RASTER)

    buf2 = np.empty((240, 320, 4), dtype=np.uint8)
    times2 = []
    for i in range(5):
        r2.render(mode=NU_RENDER_RASTER)
        t0 = time.perf_counter()
        lib.nu_fetch_pixels(r2._handle, buf2.ctypes.data_as(ctypes.c_void_p), 0)
        times2.append((time.perf_counter() - t0) * 1000)
    print(f"\n320x240 fetch times: {[f'{t:.1f}' for t in times2]} ms")
    print(f"Average:  {sum(times2)/len(times2):.1f} ms")

    r2.close()


if __name__ == "__main__":
    bench()
