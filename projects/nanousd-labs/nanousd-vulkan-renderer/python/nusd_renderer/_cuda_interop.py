# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-Vulkan interop: import Vulkan render buffers into CUDA for zero-copy access.

Uses the CUDA driver API (libcuda.so) via ctypes to:
  1. Import two Vulkan linear buffers (double-buffered) as CUDA external memory
  2. Map each as a CUdeviceptr (linear RGBA8 buffer on GPU)
  3. Synchronize via Vulkan fence (nu_wait_tiled_complete) before CUDA reads,
     OR (when ``use_semaphore_sync=True``) via an imported Vulkan timeline
     semaphore using ``cuWaitExternalSemaphoresAsync`` on the torch CUDA stream.

Double-buffered: Vulkan writes to buffer[N%2] while CUDA reads buffer[(N-1)%2].
This overlaps RT dispatch with CUDA de-tiling, hiding most of the RT latency.
The first frame returns zeros (no previous frame available yet).

Synchronization modes:

- Fence (default): The Vulkan command buffer that copies the tiled image into
  the interop buffer also submits with a VkFence. ``nu_wait_tiled_complete()``
  waits on this fence from the CPU, guaranteeing the interop buffer is ready
  before CUDA reads it.

- Semaphore (``use_semaphore_sync=True``): The same submission additionally
  signals an exported timeline semaphore. We import that semaphore into CUDA
  and wait on it asynchronously on the active torch CUDA stream — no host
  block, so the wait can overlap with subsequent CPU work. Required for the
  PR 1b wait-reordering optimization.

Usage:
    from nusd_renderer._cuda_interop import CudaVulkanInterop

    mu = NuRenderer(...)
    # ... render_tiled() ...
    interop = CudaVulkanInterop(mu, num_cameras=512, tile_w=100, tile_h=100)
    # After each render_tiled():
    pixels = interop.wait_and_copy_to_host()     # numpy on CPU
    # or:
    warp_arr = interop.wait_and_get_warp_array()  # GPU-resident, zero-copy
"""

import ctypes
import ctypes.util
import os

import numpy as np

# ---- CUDA Driver API types and constants ----

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64  # unsigned long long on 64-bit
CUexternalMemory = ctypes.c_void_p
CUexternalSemaphore = ctypes.c_void_p
CUstream = ctypes.c_void_p

# External memory handle types
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1

# External semaphore handle types — values from cuda.h.
# Timeline semaphore exported as an opaque POSIX fd is type 9.
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9

# CUDA VMM API constants (cuMemCreate / cuMemMap path, used by Hybrid C
# inverted-ownership interop).
CU_MEM_ALLOCATION_TYPE_PINNED = 1            # CUmemAllocationType
CU_MEM_LOCATION_TYPE_DEVICE = 1              # CUmemLocationType
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1 # CUmemAllocationHandleType
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3       # CUmemAccess_flags
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

CUmemGenericAllocationHandle = ctypes.c_uint64


class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(ctypes.Structure):
    """CUDA_EXTERNAL_MEMORY_HANDLE_DESC — describes how to import external memory."""

    class _HandleUnion(ctypes.Union):
        class _FdHandle(ctypes.Structure):
            _fields_ = [("fd", ctypes.c_int)]

        class _Win32Handle(ctypes.Structure):
            _fields_ = [
                ("handle", ctypes.c_void_p),
                ("name", ctypes.c_void_p),
            ]

        _fields_ = [
            ("fd", _FdHandle),
            ("win32", _Win32Handle),
        ]

    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", _HandleUnion),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(ctypes.Structure):
    """Describes a linear buffer mapping from external memory."""

    _fields_ = [
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(ctypes.Structure):
    """CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC — describes how to import an external semaphore.

    Mirrors the layout of the cuda.h struct of the same name. The handle union
    only includes the fd and win32 variants (the largest, two pointers = 16
    bytes); the omitted ``nvSciSyncObj`` pointer fits inside the same 16 bytes.
    """

    class _HandleUnion(ctypes.Union):
        class _Win32Handle(ctypes.Structure):
            _fields_ = [
                ("handle", ctypes.c_void_p),
                ("name", ctypes.c_void_p),
            ]

        _fields_ = [
            ("fd", ctypes.c_int),
            ("win32", _Win32Handle),
        ]

    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", _HandleUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CUmemLocation(ctypes.Structure):
    """CUDA memory location descriptor."""

    _fields_ = [
        ("type", ctypes.c_uint),  # CUmemLocationType
        ("id", ctypes.c_int),
    ]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    """Properties for cuMemCreate (VMM API)."""

    _fields_ = [
        ("type", ctypes.c_uint),                  # CUmemAllocationType
        ("requestedHandleTypes", ctypes.c_uint),  # CUmemAllocationHandleType
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    """Access descriptor for cuMemSetAccess."""

    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_uint),  # CUmemAccess_flags
    ]


class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(ctypes.Structure):
    """CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS — wait parameters for cuWaitExternalSemaphoresAsync.

    Mirrors cuda.h field-by-field; ctypes natural alignment produces the 144-byte
    layout the driver expects (inner ``params`` is 72 bytes, then 4-byte ``flags``,
    then 16 * 4 bytes ``reserved``).
    """

    class _Params(ctypes.Structure):
        class _Fence(ctypes.Structure):
            _fields_ = [("value", ctypes.c_uint64)]

        class _NvSciSync(ctypes.Union):
            _fields_ = [
                ("fence", ctypes.c_void_p),
                ("reserved", ctypes.c_uint64),
            ]

        class _KeyedMutex(ctypes.Structure):
            _fields_ = [
                ("key", ctypes.c_uint64),
                ("timeoutMs", ctypes.c_uint),
            ]

        _fields_ = [
            ("fence", _Fence),
            ("nvSciSync", _NvSciSync),
            ("keyedMutex", _KeyedMutex),
            ("reserved", ctypes.c_uint * 10),
        ]

    _fields_ = [
        ("params", _Params),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


def _load_cuda_driver():
    """Load libcuda.so and bind the functions we need."""
    path = ctypes.util.find_library("cuda")
    if not path:
        # Try common locations
        for p in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1"]:
            if os.path.exists(p):
                path = p
                break
    if not path:
        raise RuntimeError("libcuda.so not found — CUDA driver not installed")

    lib = ctypes.CDLL(path)

    # cuInit
    lib.cuInit.restype = CUresult
    lib.cuInit.argtypes = [ctypes.c_uint]

    # cuImportExternalMemory
    lib.cuImportExternalMemory.restype = CUresult
    lib.cuImportExternalMemory.argtypes = [
        ctypes.POINTER(CUexternalMemory),
        ctypes.POINTER(CUDA_EXTERNAL_MEMORY_HANDLE_DESC),
    ]

    # cuExternalMemoryGetMappedBuffer
    lib.cuExternalMemoryGetMappedBuffer.restype = CUresult
    lib.cuExternalMemoryGetMappedBuffer.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        CUexternalMemory,
        ctypes.POINTER(CUDA_EXTERNAL_MEMORY_BUFFER_DESC),
    ]

    # cuMemcpyDtoH
    lib.cuMemcpyDtoH_v2.restype = CUresult
    lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]

    # cuDestroyExternalMemory
    lib.cuDestroyExternalMemory.restype = CUresult
    lib.cuDestroyExternalMemory.argtypes = [CUexternalMemory]

    # cuImportExternalSemaphore
    lib.cuImportExternalSemaphore.restype = CUresult
    lib.cuImportExternalSemaphore.argtypes = [
        ctypes.POINTER(CUexternalSemaphore),
        ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC),
    ]

    # cuWaitExternalSemaphoresAsync
    lib.cuWaitExternalSemaphoresAsync.restype = CUresult
    lib.cuWaitExternalSemaphoresAsync.argtypes = [
        ctypes.POINTER(CUexternalSemaphore),
        ctypes.POINTER(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS),
        ctypes.c_uint,
        CUstream,
    ]

    # cuDestroyExternalSemaphore
    lib.cuDestroyExternalSemaphore.restype = CUresult
    lib.cuDestroyExternalSemaphore.argtypes = [CUexternalSemaphore]

    # cuCtxGetCurrent (to check CUDA is initialized)
    lib.cuCtxGetCurrent.restype = CUresult
    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    # ---- VMM API (used by CudaOwnedInterop / Hybrid C path) ----

    lib.cuMemGetAllocationGranularity.restype = CUresult
    lib.cuMemGetAllocationGranularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(CUmemAllocationProp),
        ctypes.c_uint,  # CUmemAllocationGranularity_flags
    ]

    lib.cuMemCreate.restype = CUresult
    lib.cuMemCreate.argtypes = [
        ctypes.POINTER(CUmemGenericAllocationHandle),
        ctypes.c_size_t,
        ctypes.POINTER(CUmemAllocationProp),
        ctypes.c_uint64,  # flags (must be 0)
    ]

    lib.cuMemAddressReserve.restype = CUresult
    lib.cuMemAddressReserve.argtypes = [
        ctypes.POINTER(CUdeviceptr),
        ctypes.c_size_t,
        ctypes.c_size_t,
        CUdeviceptr,
        ctypes.c_uint64,
    ]

    lib.cuMemMap.restype = CUresult
    lib.cuMemMap.argtypes = [
        CUdeviceptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        CUmemGenericAllocationHandle,
        ctypes.c_uint64,
    ]

    lib.cuMemSetAccess.restype = CUresult
    lib.cuMemSetAccess.argtypes = [
        CUdeviceptr,
        ctypes.c_size_t,
        ctypes.POINTER(CUmemAccessDesc),
        ctypes.c_size_t,
    ]

    lib.cuMemExportToShareableHandle.restype = CUresult
    lib.cuMemExportToShareableHandle.argtypes = [
        ctypes.c_void_p,                  # void* shareableHandle (POSIX fd: int*)
        CUmemGenericAllocationHandle,
        ctypes.c_uint,                    # CUmemAllocationHandleType
        ctypes.c_uint64,                  # flags (must be 0)
    ]

    lib.cuMemUnmap.restype = CUresult
    lib.cuMemUnmap.argtypes = [CUdeviceptr, ctypes.c_size_t]

    lib.cuMemAddressFree.restype = CUresult
    lib.cuMemAddressFree.argtypes = [CUdeviceptr, ctypes.c_size_t]

    lib.cuMemRelease.restype = CUresult
    lib.cuMemRelease.argtypes = [CUmemGenericAllocationHandle]

    return lib


def _check(lib, result, func_name):
    if result != 0:
        raise RuntimeError(f"CUDA driver API error: {func_name} returned {result}")


def _import_buffer(cuda_lib, mem_fd, mem_size):
    """Import a single Vulkan buffer fd into CUDA. Returns (CUexternalMemory, CUdeviceptr)."""
    ext_mem = CUexternalMemory()

    mem_desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
    mem_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
    mem_desc.handle.fd.fd = mem_fd
    mem_desc.size = mem_size
    mem_desc.flags = 0
    _check(
        cuda_lib,
        cuda_lib.cuImportExternalMemory(ctypes.byref(ext_mem), ctypes.byref(mem_desc)),
        "cuImportExternalMemory",
    )
    # fd is consumed by CUDA — do not close it

    buf_desc = CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    buf_desc.offset = 0
    buf_desc.size = mem_size
    buf_desc.flags = 0

    dptr = CUdeviceptr(0)
    _check(
        cuda_lib,
        cuda_lib.cuExternalMemoryGetMappedBuffer(
            ctypes.byref(dptr), ext_mem, ctypes.byref(buf_desc),
        ),
        "cuExternalMemoryGetMappedBuffer",
    )
    return ext_mem, dptr


class CudaVulkanInterop:
    """Zero-copy CUDA-Vulkan interop for the tiled render target.

    Double-buffered: imports two Vulkan linear interop buffers into CUDA.
    Vulkan writes to buf[N%2] each frame while CUDA reads buf[(N-1)%2].
    This overlaps RT dispatch with CUDA de-tiling for ~2x throughput.

    The first call to wait_and_get_warp_array() returns zeros (no previous
    frame), which is acceptable for RL training warmup.

    Synchronization defaults to the Vulkan fence (via nu_wait_tiled_complete);
    the fence wait blocks until the READ buffer is ready — but since it's the
    previous frame, it's usually already done. When ``use_semaphore_sync=True``
    the renderer instead imports the per-render timeline semaphore into CUDA
    and issues an async wait on the active torch CUDA stream, removing the
    host block.
    """

    def __init__(
        self,
        renderer,
        num_cameras,
        tile_w,
        tile_h,
        skip_staging=False,
        use_semaphore_sync=False,
    ):
        """Import Vulkan interop buffers into CUDA.

        Args:
            renderer: NuRenderer instance with interop_available == True.
            num_cameras: Number of cameras in the tiled grid.
            tile_w: Per-camera width.
            tile_h: Per-camera height.
            skip_staging: If True, disable CPU staging buffer copy for faster
                rendering. Saves ~80MB/frame of GPU bandwidth at 2048 envs.
                When enabled, fetch_pixels_tiled / fetch_tiled_raw will not work.
            use_semaphore_sync: If True, import the Vulkan timeline semaphore
                into CUDA and use ``cuWaitExternalSemaphoresAsync`` on the
                active torch CUDA stream instead of the host ``vkWaitForFences``
                call. Required for PR 1b (wait reordering). Default False
                preserves the existing fence-based behaviour.
        """
        self._renderer = renderer
        self._skip_staging = skip_staging
        self._num_cameras = num_cameras
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cuda = _load_cuda_driver()
        self._ext_mem = [CUexternalMemory(), CUexternalMemory()]
        self._imported_dptr = [CUdeviceptr(0), CUdeviceptr(0)]
        self._frame_count = 0
        self._use_semaphore = bool(use_semaphore_sync)
        # Imported timeline semaphore (only when _use_semaphore is True). The
        # local _timeline_value tracks the number of waits we have issued; it
        # is incremented once per wait so the wait value matches what the C
        # side has signalled (or will signal next, in overlap mode — see
        # wait_tiled_complete_on_stream / wait_previous_tiled_complete_on_stream).
        self._ext_sem = CUexternalSemaphore()
        self._timeline_value = 0

        # Ensure CUDA is initialized with a valid context
        _check(self._cuda, self._cuda.cuInit(0), "cuInit")
        ctx = ctypes.c_void_p()
        res = self._cuda.cuCtxGetCurrent(ctypes.byref(ctx))
        if res != 0 or not ctx.value:
            # No context — create one on device 0
            self._cuda.cuCtxCreate_v2.restype = CUresult
            self._cuda.cuCtxCreate_v2.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int
            ]
            new_ctx = ctypes.c_void_p()
            _check(self._cuda, self._cuda.cuCtxCreate_v2(
                ctypes.byref(new_ctx), 0, 0), "cuCtxCreate")

        # Get interop info from Vulkan (exports one-time-use fds)
        info = renderer.get_cuda_interop_info(num_cameras, tile_w, tile_h)
        self._image_w = info["image_w"]
        self._image_h = info["image_h"]

        if self._use_semaphore:
            # Import the timeline semaphore into CUDA. The fd is consumed by
            # the import; do not close it. Subsequent waits on this CUexternalSemaphore
            # will block the torch CUDA stream until the chosen timeline value
            # has been signalled by Vulkan.
            sem_desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
            sem_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
            sem_desc.handle.fd = info["sem_fd"]
            sem_desc.flags = 0
            _check(
                self._cuda,
                self._cuda.cuImportExternalSemaphore(
                    ctypes.byref(self._ext_sem), ctypes.byref(sem_desc)
                ),
                "cuImportExternalSemaphore",
            )
            # The local counter tracks the number of waits issued. It starts
            # at zero regardless of how many renders have already happened
            # before init — each wait_*_on_stream call increments it once and
            # matches the corresponding C-side signal value.
            self._timeline_value = 0
        else:
            # Fence-based sync — semaphore fd is unused, close it.
            os.close(info["sem_fd"])

        # Import both double-buffered interop buffers into CUDA
        for i in range(2):
            self._ext_mem[i], self._imported_dptr[i] = _import_buffer(
                self._cuda, info["mem_fds"][i], info["mem_size"]
            )

        # Skip CPU staging copy — CUDA is the only consumer.
        if self._skip_staging:
            renderer.set_skip_staging(True)

    @property
    def image_w(self):
        return self._image_w

    @property
    def image_h(self):
        return self._image_h

    @property
    def frame_count(self):
        return self._frame_count

    def _stream_wait_for_value(self, stream_handle, value):
        """Issue an async wait on ``stream_handle`` for timeline value ``value``."""
        params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
        params.params.fence.value = ctypes.c_uint64(value).value
        params.flags = 0
        _check(
            self._cuda,
            self._cuda.cuWaitExternalSemaphoresAsync(
                ctypes.byref(self._ext_sem),
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream_handle)),
            ),
            "cuWaitExternalSemaphoresAsync",
        )

    def wait_tiled_complete_on_stream(self, stream_handle: int):
        """Wait on the torch CUDA stream for the most recent ``render_tiled()``.

        Async equivalent of ``self._renderer.wait_tiled_complete()``. Each call
        increments the local timeline counter and waits for that value, which
        matches the C-side signal pattern (one increment per render_tiled).
        """
        self._timeline_value += 1
        self._stream_wait_for_value(stream_handle, self._timeline_value)

    def wait_previous_tiled_complete_on_stream(self, stream_handle: int):
        """Wait on the torch CUDA stream for the PREVIOUS ``render_tiled()``.

        Async equivalent of ``self._renderer.wait_previous_tiled_complete()``.
        Used by the overlap path: after render N is submitted, CUDA only needs
        the buffer from render N-1 (the one it is about to read). The first
        call has no previous frame and skips the wait — matching the existing
        "first frame returns zeros" contract.
        """
        self._timeline_value += 1
        if self._timeline_value > 1:
            self._stream_wait_for_value(stream_handle, self._timeline_value - 1)

    def _torch_stream_handle(self):
        """Return the active torch CUDA stream as an int handle."""
        import torch

        return int(torch.cuda.current_stream().cuda_stream)

    def wait_and_copy_to_host(self):
        """Wait for Vulkan render, copy the read buffer to CPU numpy array.

        Returns:
            numpy array of shape (image_h, image_w, 4), dtype uint8.
        """
        # Always use the host fence here — the cuMemcpyDtoH below is a
        # synchronous host copy that does not respect stream-side waits
        # queued via cuWaitExternalSemaphoresAsync.
        self._renderer.wait_tiled_complete()
        read_idx = self._renderer.get_interop_read_idx()
        self._frame_count += 1

        buf_size = self._image_w * self._image_h * 4
        host_buf = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)
        _check(
            self._cuda,
            self._cuda.cuMemcpyDtoH_v2(
                host_buf.ctypes.data_as(ctypes.c_void_p),
                self._imported_dptr[read_idx],
                buf_size,
            ),
            "cuMemcpyDtoH",
        )
        return host_buf

    def wait_and_get_device_ptr(self):
        """Wait for current frame's Vulkan render, return its device pointer.

        Blocks until the most recent render_tiled() completes, then returns
        that frame's interop buffer. No overlap — use for correctness tests.

        Returns:
            (CUdeviceptr, image_w, image_h) — linear RGBA8.
        """
        if self._use_semaphore:
            self.wait_tiled_complete_on_stream(self._torch_stream_handle())
        else:
            self._renderer.wait_tiled_complete()
        read_idx = self._renderer.get_interop_read_idx()
        self._frame_count += 1
        return self._imported_dptr[read_idx], self._image_w, self._image_h

    def wait_and_get_warp_array(self):
        """Wait for current frame and return GPU-resident warp array.

        Returns:
            warp.array of shape (image_h, image_w, 4), dtype=wp.uint8, on GPU.
        """
        import warp as wp

        dptr, w, h = self.wait_and_get_device_ptr()
        return wp.array(
            dtype=wp.uint8,
            shape=(h, w, 4),
            ptr=dptr.value,
            capacity=w * h * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,  # Vulkan owns the memory
        )

    def overlap_get_device_ptr(self):
        """Get PREVIOUS frame's device pointer (double-buffered overlap).

        Call AFTER render_tiled() for frame N. Waits on frame N-1's fence
        (typically instant since N-1 was submitted before N), then returns
        frame N-1's interop buffer. Frame N continues rendering on GPU.

        When direct buffer write is active (SSBO path), both frames write to
        buf[0], so true overlap isn't possible — falls back to synchronous
        wait on the current frame's fence.

        First call returns buffer 0's pointer (uninitialized — zeros).
        Caller should discard the first frame's output.

        Returns:
            (CUdeviceptr, image_w, image_h) — linear RGBA8 from previous frame.
        """
        if self._use_semaphore:
            self.wait_previous_tiled_complete_on_stream(self._torch_stream_handle())
        else:
            self._renderer.wait_previous_tiled_complete()
        prev_idx = self._renderer.get_interop_prev_idx()
        self._frame_count += 1
        return self._imported_dptr[prev_idx], self._image_w, self._image_h

    def overlap_get_warp_array(self):
        """Get PREVIOUS frame as GPU-resident warp array (double-buffered overlap).

        Same as overlap_get_device_ptr() but wrapped as a warp array.
        First frame returns zeros (acceptable for RL training warmup).

        Returns:
            warp.array of shape (image_h, image_w, 4), dtype=wp.uint8, on GPU.
        """
        import warp as wp

        dptr, w, h = self.overlap_get_device_ptr()
        return wp.array(
            dtype=wp.uint8,
            shape=(h, w, 4),
            ptr=dptr.value,
            capacity=w * h * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,  # Vulkan owns the memory
        )

    def wait_and_get_per_env_array(self):
        """Wait for current frame, return as [num_envs, H, W, 4] warp array.

        Requires per_env_layout to be enabled on the renderer. The SSBO already
        contains pixels in [env, H, W, 4] contiguous layout, so no de-tiling
        kernel is needed.

        Returns:
            warp.array of shape (num_envs, tile_h, tile_w, 4), dtype=wp.uint8.
        """
        import warp as wp

        dptr, w, h = self.wait_and_get_device_ptr()
        n = self._num_cameras
        tw, th = self._tile_w, self._tile_h
        return wp.array(
            dtype=wp.uint8,
            shape=(n, th, tw, 4),
            ptr=dptr.value,
            capacity=n * th * tw * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def overlap_get_per_env_array(self):
        """Get PREVIOUS frame as [num_envs, H, W, 4] warp array (overlapped).

        Requires per_env_layout to be enabled on the renderer. No de-tiling.
        First frame returns zeros.

        Returns:
            warp.array of shape (num_envs, tile_h, tile_w, 4), dtype=wp.uint8.
        """
        import warp as wp

        dptr, w, h = self.overlap_get_device_ptr()
        n = self._num_cameras
        tw, th = self._tile_w, self._tile_h
        return wp.array(
            dtype=wp.uint8,
            shape=(n, th, tw, 4),
            ptr=dptr.value,
            capacity=n * th * tw * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def close(self):
        """Release CUDA resources."""
        if self._skip_staging:
            self._renderer.set_skip_staging(False)
        for i in range(2):
            self._imported_dptr[i] = CUdeviceptr(0)
            if self._ext_mem[i]:
                self._cuda.cuDestroyExternalMemory(self._ext_mem[i])
                self._ext_mem[i] = CUexternalMemory()
        if self._use_semaphore and self._ext_sem:
            self._cuda.cuDestroyExternalSemaphore(self._ext_sem)
            self._ext_sem = CUexternalSemaphore()
            self._use_semaphore = False

    def __del__(self):
        self.close()


class CudaOwnedInterop:
    """Hybrid C — caller-owned memory for the tiled render target.

    The trainer allocates the double-buffered output via the CUDA VMM API
    (cuMemCreate + cuMemExportToShareableHandle, POSIX_FILE_DESCRIPTOR) and
    hands the resulting fds to the renderer. The renderer creates VkBuffers
    with VK_KHR_external_memory_fd, imports each fd as VkDeviceMemory, and
    writes pixels directly into the trainer-owned memory. The trainer reads
    pixels via its own ``cuMemMap``-backed CUdeviceptr — there is no
    ``cuImportExternalMemory`` call in the trainer's CUDA context.

    This sidesteps the slow-mode in cudaStreamSynchronize that the production
    interop path triggers when the trainer's CUDA context holds active imports
    of Vulkan-allocated memory.

    Mirrors :class:`CudaVulkanInterop`'s public surface so callers can swap
    via a single env-var flag (``NUSD_INVERTED_INTEROP=1``).

    Lifecycle: caller owns the underlying CUDA allocation. ``close()`` calls
    cuMemUnmap + cuMemAddressFree + cuMemRelease after the renderer's
    VkDeviceMemory is freed. The renderer must be torn down (or have a fresh
    set_external_output_buffers call with new fds) before close() runs.
    """

    def __init__(
        self,
        renderer,
        num_cameras,
        tile_w,
        tile_h,
        *,
        use_semaphore_sync=False,
    ):
        """Allocate caller-owned CUDA buffers, hand them to the renderer.

        Args:
            renderer: NuRenderer with interop_available == True and a recent
                libnusd_renderer.so (must export
                ``nu_set_external_output_buffers``).
            num_cameras: Number of cameras in the tiled grid.
            tile_w: Per-camera width.
            tile_h: Per-camera height.
            use_semaphore_sync: If True, import the renderer's timeline
                semaphore via cuImportExternalSemaphore and use
                cuWaitExternalSemaphoresAsync on the trainer's torch CUDA
                stream instead of host-blocking on a Vulkan fence. The
                semaphore import is cheap and is NOT part of the slow-mode
                memory trigger; it stays the same as in the original interop.
        """
        self._renderer = renderer
        self._num_cameras = num_cameras
        self._tile_w = tile_w
        self._tile_h = tile_h
        self._cuda = _load_cuda_driver()

        # Layout matches what nu_render_tiled produces in per-env-SSBO mode:
        # one logical 1-D ramp of [num_envs, tile_h, tile_w, 4] bytes, but the
        # underlying allocation is sized via the same ceil-sqrt grid as
        # nu_get_cuda_interop_info (so total_w * total_h * 4 bytes).
        import math
        num_cols = int(math.ceil(math.sqrt(float(num_cameras))))
        num_rows = (num_cameras + num_cols - 1) // num_cols
        self._image_w = num_cols * tile_w
        self._image_h = num_rows * tile_h
        self._logical_size = self._image_w * self._image_h * 4

        # CUDA bookkeeping
        self._handles = [CUmemGenericAllocationHandle(0),
                         CUmemGenericAllocationHandle(0)]
        self._dptrs = [CUdeviceptr(0), CUdeviceptr(0)]
        self._aligned_size = 0
        self._frame_count = 0
        self._closed = False

        self._use_semaphore = bool(use_semaphore_sync)
        self._ext_sem = CUexternalSemaphore()
        self._timeline_value = 0

        # Make sure CUDA is initialised with a current context — required
        # before any VMM API call.
        _check(self._cuda, self._cuda.cuInit(0), "cuInit")
        ctx = ctypes.c_void_p()
        res = self._cuda.cuCtxGetCurrent(ctypes.byref(ctx))
        if res != 0 or not ctx.value:
            self._cuda.cuCtxCreate_v2.restype = CUresult
            self._cuda.cuCtxCreate_v2.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int,
            ]
            new_ctx = ctypes.c_void_p()
            _check(self._cuda, self._cuda.cuCtxCreate_v2(
                ctypes.byref(new_ctx), 0, 0), "cuCtxCreate")

        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0  # device 0; matches gpu_index default
        prop.win32HandleMetaData = None

        granularity = ctypes.c_size_t(0)
        _check(
            self._cuda,
            self._cuda.cuMemGetAllocationGranularity(
                ctypes.byref(granularity), ctypes.byref(prop),
                CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        gran = max(granularity.value, 64 * 1024)  # also satisfy Vulkan req.alignment
        aligned = ((self._logical_size + gran - 1) // gran) * gran
        self._aligned_size = aligned

        fds = [-1, -1]
        try:
            for i in range(2):
                _check(
                    self._cuda,
                    self._cuda.cuMemCreate(
                        ctypes.byref(self._handles[i]),
                        ctypes.c_size_t(aligned),
                        ctypes.byref(prop),
                        ctypes.c_uint64(0),
                    ),
                    f"cuMemCreate[{i}]",
                )
                _check(
                    self._cuda,
                    self._cuda.cuMemAddressReserve(
                        ctypes.byref(self._dptrs[i]),
                        ctypes.c_size_t(aligned),
                        ctypes.c_size_t(0),
                        CUdeviceptr(0),
                        ctypes.c_uint64(0),
                    ),
                    f"cuMemAddressReserve[{i}]",
                )
                _check(
                    self._cuda,
                    self._cuda.cuMemMap(
                        self._dptrs[i],
                        ctypes.c_size_t(aligned),
                        ctypes.c_size_t(0),
                        self._handles[i],
                        ctypes.c_uint64(0),
                    ),
                    f"cuMemMap[{i}]",
                )

                access = CUmemAccessDesc()
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = 0
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(
                    self._cuda,
                    self._cuda.cuMemSetAccess(
                        self._dptrs[i],
                        ctypes.c_size_t(aligned),
                        ctypes.byref(access),
                        ctypes.c_size_t(1),
                    ),
                    f"cuMemSetAccess[{i}]",
                )

                fd = ctypes.c_int(-1)
                _check(
                    self._cuda,
                    self._cuda.cuMemExportToShareableHandle(
                        ctypes.byref(fd),
                        self._handles[i],
                        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                        ctypes.c_uint64(0),
                    ),
                    f"cuMemExportToShareableHandle[{i}]",
                )
                fds[i] = fd.value

            # Hand both fds to the renderer. On success the renderer takes
            # ownership (vkAllocateMemory consumes them); on failure we still
            # own the fds and must close them.
            try:
                renderer.set_external_output_buffers(
                    fds, aligned, num_cameras, tile_w, tile_h,
                )
                fds = [-1, -1]  # consumed
            except Exception:
                for fd in fds:
                    if fd >= 0:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                fds = [-1, -1]
                raise

            # Engage the direct-SSBO write path: with interop_buf[0] now
            # populated by our import, set_skip_staging triggers
            # direct_write_active so the raygen shader writes pixels to
            # binding 9 (= our caller-owned memory) and skips the
            # vkCmdCopyImageToBuffer staging step. Without this the renderer
            # writes to its tiled image and never touches our buffer.
            renderer.set_skip_staging(True)

        except Exception:
            self.close()
            raise

        # Optional async sync via the renderer's timeline semaphore.
        if self._use_semaphore:
            sem_info = renderer.get_external_timeline_semaphore_fd()
            sem_desc = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
            sem_desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
            sem_desc.handle.fd = sem_info["sem_fd"]
            sem_desc.flags = 0
            _check(
                self._cuda,
                self._cuda.cuImportExternalSemaphore(
                    ctypes.byref(self._ext_sem), ctypes.byref(sem_desc)
                ),
                "cuImportExternalSemaphore",
            )
            self._timeline_value = 0

    # ---- Properties --------------------------------------------------------

    @property
    def image_w(self):
        return self._image_w

    @property
    def image_h(self):
        return self._image_h

    @property
    def frame_count(self):
        return self._frame_count

    # ---- Internal helpers --------------------------------------------------

    def _stream_wait_for_value(self, stream_handle, value):
        params = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
        params.params.fence.value = ctypes.c_uint64(value).value
        params.flags = 0
        _check(
            self._cuda,
            self._cuda.cuWaitExternalSemaphoresAsync(
                ctypes.byref(self._ext_sem),
                ctypes.byref(params),
                1,
                ctypes.c_void_p(int(stream_handle)),
            ),
            "cuWaitExternalSemaphoresAsync",
        )

    def _torch_stream_handle(self):
        import torch
        return int(torch.cuda.current_stream().cuda_stream)

    # ---- Sync API (matches CudaVulkanInterop) ------------------------------

    def wait_tiled_complete_on_stream(self, stream_handle: int):
        """Async wait on ``stream_handle`` for the most recent render_tiled."""
        self._timeline_value += 1
        self._stream_wait_for_value(stream_handle, self._timeline_value)

    def wait_previous_tiled_complete_on_stream(self, stream_handle: int):
        """Async wait on ``stream_handle`` for the PREVIOUS render_tiled."""
        self._timeline_value += 1
        if self._timeline_value > 1:
            self._stream_wait_for_value(stream_handle, self._timeline_value - 1)

    # ---- Pixel access (mirrors CudaVulkanInterop) --------------------------

    def wait_and_get_device_ptr(self):
        """Block until the most recent render_tiled completes, return its dptr.

        Currently single-buffer-only: returns ``dptrs[0]`` regardless of the
        renderer's ``interop_write_idx``. The renderer's set-A/set-B alternation
        for double-buffered overlap has a bug under inverted memory ownership
        where slot-1 writes don't reach the trainer (set A → buf[0] always
        works, set B → buf[1] only works when set B is bound on every frame).
        Until that's fixed, the trainer reads dptrs[0] and the renderer's
        alternation is effectively harmless: every other frame's writes go to
        dptrs[1] and are dropped, but the most recent writes to dptrs[0] are
        always visible after wait_tiled_complete. Sync-latency comparisons are
        unaffected.
        """
        if self._use_semaphore:
            self.wait_tiled_complete_on_stream(self._torch_stream_handle())
        else:
            self._renderer.wait_tiled_complete()
        self._frame_count += 1
        return self._dptrs[0], self._image_w, self._image_h

    def wait_and_get_warp_array(self):
        """Block; return the most recent render's pixels as a warp array (RGBA8)."""
        import warp as wp
        dptr, w, h = self.wait_and_get_device_ptr()
        return wp.array(
            dtype=wp.uint8,
            shape=(h, w, 4),
            ptr=dptr.value,
            capacity=w * h * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def overlap_get_device_ptr(self):
        """Wait for PREVIOUS render and return its dptr.

        Single-buffer-only fallback: same as :meth:`wait_and_get_device_ptr`,
        which means no producer/consumer overlap. See the comment in
        ``wait_and_get_device_ptr`` for the reason.
        """
        return self.wait_and_get_device_ptr()

    def overlap_get_warp_array(self):
        """Same as overlap_get_device_ptr but wrapped as a warp array."""
        import warp as wp
        dptr, w, h = self.overlap_get_device_ptr()
        return wp.array(
            dtype=wp.uint8,
            shape=(h, w, 4),
            ptr=dptr.value,
            capacity=w * h * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def wait_and_get_per_env_array(self):
        """Block; return current render as [num_envs, tile_h, tile_w, 4] warp.

        Requires per_env_layout enabled on the renderer.
        """
        import warp as wp
        dptr, w, h = self.wait_and_get_device_ptr()
        n = self._num_cameras
        tw, th = self._tile_w, self._tile_h
        return wp.array(
            dtype=wp.uint8,
            shape=(n, th, tw, 4),
            ptr=dptr.value,
            capacity=n * th * tw * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def overlap_get_per_env_array(self):
        """Get PREVIOUS render as [num_envs, tile_h, tile_w, 4] warp array."""
        import warp as wp
        dptr, w, h = self.overlap_get_device_ptr()
        n = self._num_cameras
        tw, th = self._tile_w, self._tile_h
        return wp.array(
            dtype=wp.uint8,
            shape=(n, th, tw, 4),
            ptr=dptr.value,
            capacity=n * th * tw * 4,
            device="cuda:0",
            copy=False,
            deleter=lambda p, s: None,
        )

    def wait_and_copy_to_host(self):
        """Block; copy the read buffer to a host numpy array (RGBA8)."""
        self._renderer.wait_tiled_complete()
        self._frame_count += 1
        host_buf = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)
        _check(
            self._cuda,
            self._cuda.cuMemcpyDtoH_v2(
                host_buf.ctypes.data_as(ctypes.c_void_p),
                self._dptrs[0],
                self._image_w * self._image_h * 4,
            ),
            "cuMemcpyDtoH",
        )
        return host_buf

    # ---- Cleanup -----------------------------------------------------------

    def close(self):
        """Release all VMM resources. Idempotent."""
        if self._closed:
            return
        self._closed = True

        # Drop the imported semaphore first (cheap, just dec-ref).
        if self._use_semaphore and self._ext_sem and self._ext_sem.value:
            try:
                self._cuda.cuDestroyExternalSemaphore(self._ext_sem)
            except Exception:
                pass
            self._ext_sem = CUexternalSemaphore()
            self._use_semaphore = False

        # Free CUDA-owned memory in reverse of cuMemMap/AddressReserve/Create.
        for i in range(2):
            if self._dptrs[i].value:
                try:
                    self._cuda.cuMemUnmap(self._dptrs[i],
                                          ctypes.c_size_t(self._aligned_size))
                except Exception:
                    pass
                try:
                    self._cuda.cuMemAddressFree(self._dptrs[i],
                                                ctypes.c_size_t(self._aligned_size))
                except Exception:
                    pass
                self._dptrs[i] = CUdeviceptr(0)
            if self._handles[i].value:
                try:
                    self._cuda.cuMemRelease(self._handles[i])
                except Exception:
                    pass
                self._handles[i] = CUmemGenericAllocationHandle(0)

    def __del__(self):
        self.close()
