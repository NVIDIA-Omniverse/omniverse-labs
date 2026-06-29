#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bench_cuda_readback.py — verify and benchmark the new CUDA readback path.

Compares nu_fetch_pixels (baseline; CPU swizzle + host memcpy) against
nu_fetch_pixels_cuda (CUDA imports the exportable Vulkan buffer and copies
DtoH straight into the destination numpy array).

Verifies:
  1. Pixel-identity (after BGRA→RGBA swizzle on the CUDA side)
  2. Frame-time delta

Usage:
    python tools/bench_cuda_readback.py /tmp/grid_tera.usdc
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

# Allow override via env vars (so we run against the worktree build, not
# whatever is in the parent directory).
_DEFAULT_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_DEFAULT_PYTHON, "python"))

os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

# Force the worktree's library, not the shared submodule's.
os.environ.setdefault(
    "NUSD_RENDERER_LIB",
    os.path.join(_DEFAULT_PYTHON, "build", "libnusd_renderer.so"),
)

from nusd_renderer._bindings import NuRenderer, NU_RENDER_RT  # noqa: E402


# ------------ Minimal CUDA driver-API ctypes wrapper ------------

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64
CUexternalMemory = ctypes.c_void_p
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1


class CU_EXT_MEM_HANDLE_DESC(ctypes.Structure):
    class _HandleUnion(ctypes.Union):
        class _Fd(ctypes.Structure):
            _fields_ = [("fd", ctypes.c_int)]

        class _Win32(ctypes.Structure):
            _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]

        _fields_ = [("fd", _Fd), ("win32", _Win32)]

    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", _HandleUnion),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CU_EXT_MEM_BUFFER_DESC(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


def _load_cuda():
    import ctypes.util

    path = ctypes.util.find_library("cuda")
    if not path:
        for p in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1"]:
            if os.path.exists(p):
                path = p
                break
    if not path:
        raise RuntimeError("libcuda.so not found")
    lib = ctypes.CDLL(path)
    lib.cuInit.restype = CUresult
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuCtxGetCurrent.restype = CUresult
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cuCtxCreate_v2.restype = CUresult
    lib.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    lib.cuImportExternalMemory.restype = CUresult
    lib.cuImportExternalMemory.argtypes = [
        ctypes.POINTER(CUexternalMemory),
        ctypes.POINTER(CU_EXT_MEM_HANDLE_DESC),
    ]
    lib.cuExternalMemoryGetMappedBuffer.restype = CUresult
    lib.cuExternalMemoryGetMappedBuffer.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        CUexternalMemory,
        ctypes.POINTER(CU_EXT_MEM_BUFFER_DESC),
    ]
    lib.cuDestroyExternalMemory.restype = CUresult
    lib.cuDestroyExternalMemory.argtypes = [CUexternalMemory]
    lib.cuMemcpyDtoH_v2.restype = CUresult
    lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
    return lib


def _check(lib, res, fn):
    if res != 0:
        raise RuntimeError(f"CUDA driver: {fn} returned {res}")


def main(usd_path: str):
    W, H = 1280, 720

    print(f"USD: {usd_path}", flush=True)
    r = NuRenderer(width=W, height=H, enable_rt=True, enable_materials=True)

    print(f"interop_available={r.interop_available}", flush=True)
    if not r.interop_available:
        print("interop unavailable — aborting", file=sys.stderr)
        sys.exit(2)

    t0 = time.perf_counter()
    n = r.load_usd(usd_path)
    print(f"load_ms = {(time.perf_counter() - t0) * 1000:.1f} (n={n})", flush=True)

    r.set_camera(
        eye=(94.6, 35.0, -97.0),
        target=(40.0, 9.7, 41.9),
        fov_degrees=42.0,
        near_clip=0.05,
        far_clip=4000.0,
    )

    t0 = time.perf_counter()
    r.build_accel()
    print(f"build_ms = {(time.perf_counter() - t0) * 1000:.1f}", flush=True)

    # ---- Set up CUDA & import the interop buffer ONCE (after first render) ----

    cuda = _load_cuda()
    _check(cuda, cuda.cuInit(0), "cuInit")
    ctx = ctypes.c_void_p()
    res = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if res != 0 or not ctx.value:
        new_ctx = ctypes.c_void_p()
        _check(cuda, cuda.cuCtxCreate_v2(ctypes.byref(new_ctx), 0, 0), "cuCtxCreate")

    # First render → triggers buffer creation + fd export
    r.render(NU_RENDER_RT)
    info = r.fetch_pixels_cuda()
    print(
        f"fetch_pixels_cuda: fd={info['mem_fd']} size={info['size']} "
        f"{info['width']}x{info['height']} fmt={info['format']}",
        flush=True,
    )
    assert info["width"] == W and info["height"] == H

    desc = CU_EXT_MEM_HANDLE_DESC()
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
    desc.handle.fd.fd = info["mem_fd"]
    desc.size = info["size"]
    desc.flags = 0
    ext_mem = CUexternalMemory()
    _check(
        cuda,
        cuda.cuImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(desc)),
        "cuImportExternalMemory",
    )

    bdesc = CU_EXT_MEM_BUFFER_DESC()
    bdesc.offset = 0
    bdesc.size = info["size"]
    bdesc.flags = 0
    dptr = CUdeviceptr(0)
    _check(
        cuda,
        cuda.cuExternalMemoryGetMappedBuffer(ctypes.byref(dptr), ext_mem, ctypes.byref(bdesc)),
        "cuExternalMemoryGetMappedBuffer",
    )

    # ---- Pixel-identity check ----
    # baseline returns RGBA8; CUDA path returns BGRA8 → swizzle on the host.
    bgra_buf = np.zeros((H, W, 4), dtype=np.uint8)
    payload_bytes = W * H * 4
    rgba_baseline = r.fetch_pixels()
    # The render call has already happened; the cuda buffer was overwritten by
    # fetch_pixels_cuda above — but fetch_pixels copied from swapchain too,
    # so both have the same source. Verify equality.
    _check(
        cuda,
        cuda.cuMemcpyDtoH_v2(bgra_buf.ctypes.data_as(ctypes.c_void_p), dptr, payload_bytes),
        "cuMemcpyDtoH",
    )
    rgba_from_cuda = bgra_buf[..., [2, 1, 0, 3]]  # BGRA → RGBA
    if np.array_equal(rgba_baseline, rgba_from_cuda):
        print("PIXEL-IDENTICAL: ✓ baseline RGBA == CUDA-imported BGRA[swizzled]", flush=True)
    else:
        diff = (rgba_baseline.astype(int) - rgba_from_cuda.astype(int))
        n_diff = int(np.count_nonzero(diff))
        print(
            f"PIXEL MISMATCH: {n_diff} bytes differ "
            f"max_abs_diff={int(np.abs(diff).max())}",
            flush=True,
        )
        # Save diagnostic crops
        np.save("/tmp/cuda_baseline.npy", rgba_baseline)
        np.save("/tmp/cuda_imported.npy", rgba_from_cuda)
        sys.exit(1)

    # ---- Steady-state benchmark, both paths ----
    print(">>> NVTX:warmup", flush=True)
    for _ in range(3):
        r.render(NU_RENDER_RT)
        r.fetch_pixels()

    print(">>> NVTX:steady_30 (baseline: render + fetch_pixels)", flush=True)
    base_times = []
    for _ in range(30):
        t = time.perf_counter()
        r.render(NU_RENDER_RT)
        r.fetch_pixels()
        base_times.append((time.perf_counter() - t) * 1000.0)
    base_med = float(np.median(base_times))
    print(f"baseline steady_median_ms = {base_med:.2f}", flush=True)

    # CUDA path: render + fetch_pixels_cuda + cuMemcpyDtoH into preallocated host buf
    cuda_buf = np.zeros((H, W, 4), dtype=np.uint8)  # BGRA8

    # Warmup the CUDA path (a few extra calls to settle)
    for _ in range(3):
        r.render(NU_RENDER_RT)
        r.fetch_pixels_cuda()
        _check(
            cuda,
            cuda.cuMemcpyDtoH_v2(cuda_buf.ctypes.data_as(ctypes.c_void_p), dptr, payload_bytes),
            "cuMemcpyDtoH",
        )

    print(">>> NVTX:steady_30 (cuda: render + fetch_pixels_cuda + cuMemcpyDtoH)", flush=True)
    cuda_times = []
    for _ in range(30):
        t = time.perf_counter()
        r.render(NU_RENDER_RT)
        r.fetch_pixels_cuda()
        _check(
            cuda,
            cuda.cuMemcpyDtoH_v2(cuda_buf.ctypes.data_as(ctypes.c_void_p), dptr, payload_bytes),
            "cuMemcpyDtoH",
        )
        cuda_times.append((time.perf_counter() - t) * 1000.0)
    cuda_med = float(np.median(cuda_times))
    print(f"cuda     steady_median_ms = {cuda_med:.2f}", flush=True)

    print(
        f"\nDelta: {base_med - cuda_med:+.2f} ms ({100.0 * (base_med - cuda_med) / base_med:+.1f}%)",
        flush=True,
    )
    print(f"baseline p5/p95 = {np.percentile(base_times, 5):.2f} / {np.percentile(base_times, 95):.2f}")
    print(f"cuda     p5/p95 = {np.percentile(cuda_times, 5):.2f} / {np.percentile(cuda_times, 95):.2f}")

    # Cleanup CUDA
    cuda.cuDestroyExternalMemory(ext_mem)
    r.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bench_cuda_readback.py <USD path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
