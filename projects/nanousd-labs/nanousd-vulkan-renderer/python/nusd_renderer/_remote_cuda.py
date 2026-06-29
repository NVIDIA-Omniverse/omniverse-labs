# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CUDA driver bindings for the subprocess renderer (host + client).

Centralizes the ctypes surface so `_remote_host.py` and `_remote.py` use the
same definitions. Mirrors the bindings used in
`scripts/subprocess_renderer_poc/{parent,child}.py` Phase 0 POC, but lives in
a permanent location and adds the event/memcpy primitives Phase 1 needs.
"""

from __future__ import annotations

import ctypes


libcuda = ctypes.CDLL("libcuda.so.1")
libcudart = ctypes.CDLL("libcudart.so.12")

CUresult = ctypes.c_int
CUcontext = ctypes.c_void_p
CUdevice = ctypes.c_int
CUdeviceptr = ctypes.c_uint64
CUstream = ctypes.c_void_p
CUevent = ctypes.c_void_p

# All IPC handle structs are 64 bytes on Linux.
IPC_HANDLE_SIZE = 64


class CUipcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * IPC_HANDLE_SIZE)]


class CUipcEventHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * IPC_HANDLE_SIZE)]


CU_EVENT_INTERPROCESS = 0x4
CU_EVENT_DISABLE_TIMING = 0x2

CU_DEVICE_ATTRIBUTE_MIG_MODE = 100  # noqa: F841 (some drivers don't expose this)
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20

CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1


def _check(name: str, rc: int) -> None:
    if rc != 0:
        es = ctypes.c_char_p()
        libcuda.cuGetErrorString(rc, ctypes.byref(es))
        msg = es.value.decode() if es.value else "?"
        raise RuntimeError(f"{name} -> {rc} ({msg})")


# ---- Function signatures ----

libcuda.cuInit.argtypes = [ctypes.c_uint]
libcuda.cuInit.restype = CUresult

libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
libcuda.cuDeviceGet.restype = CUresult

libcuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice]
libcuda.cuDeviceGetAttribute.restype = CUresult

libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
libcuda.cuCtxCreate_v2.restype = CUresult

libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
libcuda.cuCtxGetCurrent.restype = CUresult

# Memory.
libcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
libcuda.cuMemAlloc_v2.restype = CUresult

libcuda.cuMemFree_v2.argtypes = [CUdeviceptr]
libcuda.cuMemFree_v2.restype = CUresult

libcuda.cuMemsetD8Async.argtypes = [CUdeviceptr, ctypes.c_byte, ctypes.c_size_t, CUstream]
libcuda.cuMemsetD8Async.restype = CUresult

libcuda.cuMemcpyDtoDAsync_v2.argtypes = [CUdeviceptr, CUdeviceptr, ctypes.c_size_t, CUstream]
libcuda.cuMemcpyDtoDAsync_v2.restype = CUresult

libcuda.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
libcuda.cuMemGetInfo_v2.restype = CUresult

# Streams.
libcuda.cuStreamCreate.argtypes = [ctypes.POINTER(CUstream), ctypes.c_uint]
libcuda.cuStreamCreate.restype = CUresult

libcuda.cuStreamSynchronize.argtypes = [CUstream]
libcuda.cuStreamSynchronize.restype = CUresult

libcuda.cuStreamWaitEvent.argtypes = [CUstream, CUevent, ctypes.c_uint]
libcuda.cuStreamWaitEvent.restype = CUresult

# Events.
libcuda.cuEventCreate.argtypes = [ctypes.POINTER(CUevent), ctypes.c_uint]
libcuda.cuEventCreate.restype = CUresult

libcuda.cuEventDestroy_v2.argtypes = [CUevent]
libcuda.cuEventDestroy_v2.restype = CUresult

libcuda.cuEventRecord.argtypes = [CUevent, CUstream]
libcuda.cuEventRecord.restype = CUresult

libcuda.cuEventQuery.argtypes = [CUevent]
libcuda.cuEventQuery.restype = CUresult

# IPC.
libcuda.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(CUipcMemHandle), CUdeviceptr]
libcuda.cuIpcGetMemHandle.restype = CUresult

libcuda.cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(CUdeviceptr), CUipcMemHandle, ctypes.c_uint]
libcuda.cuIpcOpenMemHandle_v2.restype = CUresult

libcuda.cuIpcCloseMemHandle.argtypes = [CUdeviceptr]
libcuda.cuIpcCloseMemHandle.restype = CUresult

libcuda.cuIpcGetEventHandle.argtypes = [ctypes.POINTER(CUipcEventHandle), CUevent]
libcuda.cuIpcGetEventHandle.restype = CUresult

libcuda.cuIpcOpenEventHandle.argtypes = [ctypes.POINTER(CUevent), CUipcEventHandle]
libcuda.cuIpcOpenEventHandle.restype = CUresult

# Error.
libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
libcuda.cuGetErrorString.restype = CUresult


# ---- High-level helpers ----

def init_cuda_context() -> tuple[CUdevice, CUcontext]:
    """Init CUDA, create primary context on device 0. Call once per process."""
    _check("cuInit", libcuda.cuInit(0))
    dev = CUdevice()
    _check("cuDeviceGet", libcuda.cuDeviceGet(ctypes.byref(dev), 0))
    ctx = CUcontext()
    _check("cuCtxCreate_v2", libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
    return dev, ctx


def is_mig_enabled(device: CUdevice) -> bool:
    """Return True if MIG mode is enabled on the device. cuIpcGetMemHandle
    returns NOT_SUPPORTED under MIG, so callers should refuse subprocess mode.
    """
    val = ctypes.c_int(0)
    rc = libcuda.cuDeviceGetAttribute(ctypes.byref(val), CU_DEVICE_ATTRIBUTE_MIG_MODE, device)
    if rc != 0:
        # Attribute not supported on this driver — probably means no MIG.
        return False
    return val.value != 0


def free_vram_bytes() -> tuple[int, int]:
    """Return (free, total) device memory in bytes for the current context."""
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    _check("cuMemGetInfo_v2", libcuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)))
    return free.value, total.value


def alloc_ipc_buffer(size_bytes: int) -> tuple[CUdeviceptr, CUipcMemHandle]:
    """Allocate a CUDA buffer and export its IPC mem handle."""
    dptr = CUdeviceptr()
    _check("cuMemAlloc_v2", libcuda.cuMemAlloc_v2(ctypes.byref(dptr), size_bytes))
    handle = CUipcMemHandle()
    _check("cuIpcGetMemHandle", libcuda.cuIpcGetMemHandle(ctypes.byref(handle), dptr))
    return dptr, handle


def open_ipc_buffer(handle_bytes: bytes) -> CUdeviceptr:
    """Import an IPC mem handle into this process's context."""
    if len(handle_bytes) != IPC_HANDLE_SIZE:
        raise ValueError(f"handle_bytes must be {IPC_HANDLE_SIZE} bytes")
    handle = CUipcMemHandle()
    ctypes.memmove(handle.reserved, handle_bytes, IPC_HANDLE_SIZE)
    dptr = CUdeviceptr()
    _check(
        "cuIpcOpenMemHandle_v2",
        libcuda.cuIpcOpenMemHandle_v2(ctypes.byref(dptr), handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
    )
    return dptr


def create_ipc_event() -> tuple[CUevent, CUipcEventHandle]:
    """Create an INTERPROCESS+DISABLE_TIMING event and export its IPC handle."""
    ev = CUevent()
    flags = CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING
    _check("cuEventCreate", libcuda.cuEventCreate(ctypes.byref(ev), flags))
    handle = CUipcEventHandle()
    _check("cuIpcGetEventHandle", libcuda.cuIpcGetEventHandle(ctypes.byref(handle), ev))
    return ev, handle


def open_ipc_event(handle_bytes: bytes) -> CUevent:
    """Import an IPC event handle."""
    if len(handle_bytes) != IPC_HANDLE_SIZE:
        raise ValueError(f"handle_bytes must be {IPC_HANDLE_SIZE} bytes")
    handle = CUipcEventHandle()
    ctypes.memmove(handle.reserved, handle_bytes, IPC_HANDLE_SIZE)
    ev = CUevent()
    _check("cuIpcOpenEventHandle", libcuda.cuIpcOpenEventHandle(ctypes.byref(ev), handle))
    return ev


def handle_to_bytes(handle) -> bytes:
    """Convert a CUipcMemHandle or CUipcEventHandle to bytes for transport."""
    return bytes(handle.reserved)
