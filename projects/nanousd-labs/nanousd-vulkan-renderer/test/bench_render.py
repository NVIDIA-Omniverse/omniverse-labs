# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark rendering performance for nusd_renderer."""

import time
import sys
import os
import numpy as np


def bench_programmatic(width=1920, height=1080, nframes=20):
    """Benchmark with a programmatic triangle scene."""
    from nusd_renderer._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_RT

    print(f"Programmatic benchmark ({width}x{height}, {nframes} frames)")

    r = NuRenderer(width=width, height=height, enable_rt=True)

    # Add a simple mesh
    positions = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    normals = np.array([[0, 0, 1]] * 3, dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)

    mid = r.add_mesh(positions=positions, indices=indices, normals=normals, name="tri")
    r.set_camera(eye=(0, 0, 3), target=(0, 0, 0))

    # Warm up
    r.render(mode=NU_RENDER_RASTER)
    r.fetch_pixels()

    # Raster benchmark
    t0 = time.perf_counter()
    for i in range(nframes):
        r.render(mode=NU_RENDER_RASTER)
    t_raster_render = (time.perf_counter() - t0) / nframes * 1000

    t0 = time.perf_counter()
    for i in range(nframes):
        r.render(mode=NU_RENDER_RASTER)
        r.fetch_pixels()
    t_raster_total = (time.perf_counter() - t0) / nframes * 1000

    # RT benchmark
    if r.rt_available:
        # Warm up RT
        r.render(mode=NU_RENDER_RT)
        r.fetch_pixels()

        t0 = time.perf_counter()
        for i in range(nframes):
            r.render(mode=NU_RENDER_RT)
        t_rt_render = (time.perf_counter() - t0) / nframes * 1000

        t0 = time.perf_counter()
        for i in range(nframes):
            r.render(mode=NU_RENDER_RT)
            r.fetch_pixels()
        t_rt_total = (time.perf_counter() - t0) / nframes * 1000

    # RT + TLAS update
    if r.rt_available:
        xform = np.eye(4, dtype=np.float32)
        t0 = time.perf_counter()
        for i in range(nframes):
            xform[0, 3] = float(i) * 0.01
            r.set_transforms([mid], xform.reshape(1, 16))
            r.render(mode=NU_RENDER_RT)
            r.fetch_pixels()
        t_rt_tlas = (time.perf_counter() - t0) / nframes * 1000

    print(f"  Raster render only:         {t_raster_render:7.2f} ms/frame")
    print(f"  Raster render + readback:   {t_raster_total:7.2f} ms/frame")
    if r.rt_available:
        print(f"  RT render only:             {t_rt_render:7.2f} ms/frame")
        print(f"  RT render + readback:       {t_rt_total:7.2f} ms/frame")
        print(f"  RT + TLAS update + readback:{t_rt_tlas:7.2f} ms/frame")
    print(f"  Readback overhead (raster): {t_raster_total - t_raster_render:7.2f} ms")
    if r.rt_available:
        print(f"  Readback overhead (RT):     {t_rt_total - t_rt_render:7.2f} ms")
        print(f"  TLAS update overhead:       {t_rt_tlas - t_rt_total:7.2f} ms")
    print(f"  GPU memory:                 {r.gpu_memory_used / (1024*1024):.1f} MB")
    r.close()


def bench_usd(usd_path, width=1920, height=1080, nframes=10):
    """Benchmark with a USD file."""
    from nusd_renderer._bindings import NuRenderer, NU_RENDER_RASTER, NU_RENDER_RT

    print(f"\nUSD benchmark: {usd_path} ({width}x{height}, {nframes} frames)")

    r = NuRenderer(width=width, height=height, enable_rt=True)

    t0 = time.perf_counter()
    nmeshes = r.load_usd(usd_path)
    t_load = (time.perf_counter() - t0) * 1000
    print(f"  Scene load:                 {t_load:7.0f} ms ({nmeshes} meshes)")

    # Warm up
    r.render(mode=NU_RENDER_RASTER)
    r.fetch_pixels()

    t0 = time.perf_counter()
    for i in range(nframes):
        r.render(mode=NU_RENDER_RASTER)
        r.fetch_pixels()
    t_raster = (time.perf_counter() - t0) / nframes * 1000

    if r.rt_available:
        r.render(mode=NU_RENDER_RT)
        r.fetch_pixels()

        t0 = time.perf_counter()
        for i in range(nframes):
            r.render(mode=NU_RENDER_RT)
            r.fetch_pixels()
        t_rt = (time.perf_counter() - t0) / nframes * 1000

    print(f"  Raster + readback:          {t_raster:7.2f} ms/frame ({1000/t_raster:.0f} FPS)")
    if r.rt_available:
        print(f"  RT + readback:              {t_rt:7.2f} ms/frame ({1000/t_rt:.0f} FPS)")
    print(f"  GPU memory:                 {r.gpu_memory_used / (1024*1024):.1f} MB")
    r.close()


if __name__ == "__main__":
    bench_programmatic()

    usd_path = os.environ.get("NUSD_KITCHEN_SET", os.path.expanduser("~/Kitchen_set/Kitchen_set.usd"))
    import os
    if os.path.exists(usd_path):
        bench_usd(usd_path)
    else:
        print(f"\nSkipping USD benchmark: {usd_path} not found")
