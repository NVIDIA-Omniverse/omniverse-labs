# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-side proxy for the subprocess renderer.

`RemoteRenderer` mirrors the private low-level renderer surface used by older
training experiments. New callers should enter through `ovrtx.Renderer`.

This module also defines `RemoteCudaInterop`, mirroring
`CudaVulkanInterop`. The trainer's `_init_cuda_interop` ends up calling
the remote variant, which routes to the subprocess via the same RPC
socket.

Phase 1 milestone shape: the body of `RemoteRenderer.render_tiled()`
sends a RENDER RPC and returns; downstream consumers (analogous to
`overlap_get_per_env_array`) wait on the IPC pixel-ready event using
`torch.cuda.current_stream().wait_event(...)` — no host-side sync.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import time
from typing import Any

from . import _remote_cuda as cu
from ._remote_cuda import CUdeviceptr, libcuda, _check
from . import _remote_protocol as proto
from ._remote_protocol import Op, send_message, recv_message


class RemoteRendererError(RuntimeError):
    """Raised when the subprocess renderer dies, hangs, or returns a bad reply."""


class RemoteRenderer:
    """Proxy for `NuRenderer` that runs the real renderer in a child process.

    Constructed from the in-process renderer's __init__. Forks the child
    immediately, performs INIT handshake, opens the IPC pixel buffer +
    pixel-ready event into the trainer's CUDA context.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_envs: int,
        enable_rt: bool = True,
        enable_materials: bool = False,
        canned_scene: bool = False,
    ):
        self._width = width
        self._height = height
        self._num_envs = num_envs

        # Use subprocess.Popen + exec rather than bare os.fork(). The trainer
        # has already initialised CUDA (torch / warp), and a forked child
        # inherits a corrupted CUDA driver state — `cuInit` returns rc=3 in
        # the child without execve. Going through Popen guarantees a fresh
        # address space.
        import subprocess

        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        # Pass `b`'s fd to the child via `pass_fds`. The fd number in the
        # child's address space comes from `pass_fds=[b.fileno()]` — the
        # child sees a different number than the parent's `b.fileno()`,
        # so we tell it via argv.
        b.set_inheritable(True)
        cmd = [
            sys.executable,
            "-m",
            "nusd_renderer._remote_host",
            str(b.fileno()),
        ]
        env = os.environ.copy()
        # Ensure the renderer's Python package is importable in the child.
        # PYTHONPATH from the parent should already include this if the
        # parent itself imported nusd_renderer.
        self._proc = subprocess.Popen(
            cmd,
            pass_fds=[b.fileno()],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # stderr/stdout go to the per-PID log via dup2 in the child
            stderr=subprocess.DEVNULL,
            env=env,
            close_fds=True,
        )
        self._proc_pid = self._proc.pid
        b.close()  # parent doesn't need its end
        self._sock = a
        self._sock.setblocking(True)
        # Generous initial timeout to cover Vulkan device init + pipeline-cache load.
        self._sock.settimeout(180.0)

        log_path = f"/tmp/nusd_renderer_{self._proc_pid}.log"
        sys.stderr.write(f"[RemoteRenderer] spawned child pid={self._proc_pid}, log: {log_path}\n")
        sys.stderr.flush()

        # Init CUDA in the trainer if it isn't already. torch.cuda.init() or
        # any prior CUDA op would have done this; bare-Python callers (e.g. the
        # factory smoke test) need us to be defensive.
        try:
            cu.libcuda.cuInit(0)
            ctx_ptr = cu.CUcontext()
            rc = cu.libcuda.cuCtxGetCurrent(ctypes.byref(ctx_ptr))
            if rc != 0 or not ctx_ptr.value:
                # No current context — bring up our own primary context.
                cu.init_cuda_context()
        except Exception:
            pass

        # Send INIT.
        send_message(self._sock, Op.INIT, {
            "width": width,
            "height": height,
            "num_envs": num_envs,
            "enable_rt": enable_rt,
            "enable_materials": enable_materials,
            "canned_scene": canned_scene,
        })

        # Receive INIT_OK.
        op, payload, _fds = recv_message(self._sock)
        if op != Op.INIT_OK:
            raise RemoteRendererError(f"expected INIT_OK, got {op}")

        # Open IPC handles in trainer's context.
        self._pixel_dptrs = [cu.open_ipc_buffer(h) for h in payload["ipc_pixel_handles"]]
        self._pixel_event = cu.open_ipc_event(payload["ipc_pixel_event_handle"])
        self._pixel_size_bytes = payload["pixel_size_bytes"]
        self._num_buffers = payload["num_buffers"]
        self._frame_seq = 0

        sys.stderr.write(f"[RemoteRenderer] INIT_OK: {self._num_buffers} buffers × {self._pixel_size_bytes // 1024} KB\n")
        sys.stderr.flush()

        # Drop timeout to a per-frame budget.
        self._sock.settimeout(5.0)

    # ---- NuRenderer-compatible API surface (Phase 1 minimum) ----

    @property
    def interop_available(self) -> bool:
        """Always True for the remote variant — the child manages the actual
        CUDA-Vulkan interop, the trainer just sees IPC-mapped pixels.
        """
        return True

    def render_tiled(
        self,
        vp_inv_bytes: bytes,
        body_q_bytes: bytes,
        # mode: int = 0,
    ) -> None:
        """Send a RENDER RPC. Returns immediately after the RPC; the trainer
        waits on `self._pixel_event` via `wait_event` before reading pixels.
        """
        self._frame_seq += 1
        send_message(self._sock, Op.RENDER, {
            "frame_seq": self._frame_seq,
            "vp_inv_bytes": vp_inv_bytes,
            "body_q_bytes": body_q_bytes,
        })
        op, payload, _ = recv_message(self._sock)
        if op != Op.RENDER_OK:
            raise RemoteRendererError(f"expected RENDER_OK, got {op}")
        self._last_buffer_idx = payload["buffer_idx"]

    def wait_pixels_ready_on_stream(self, stream_handle: int) -> None:
        """Issue an in-stream wait on the pixel-ready event; no host block."""
        rc = libcuda.cuStreamWaitEvent(ctypes.c_void_p(int(stream_handle)), self._pixel_event, 0)
        if rc != 0:
            raise RemoteRendererError(f"cuStreamWaitEvent failed: rc={rc}")

    def get_pixel_dptr(self, buffer_idx: int | None = None) -> int:
        """Return the trainer-side device pointer for the most recent frame's
        IPC pixel buffer. Wrap as `wp.array(ptr=..., copy=False)` or as a
        torch tensor via `from_dlpack`.
        """
        if buffer_idx is None:
            buffer_idx = getattr(self, "_last_buffer_idx", 0)
        return self._pixel_dptrs[buffer_idx].value

    # ---- Generic NuRenderer method passthrough via __getattr__ ----

    # Methods we explicitly own — anything else falls through to RPC_CALL.
    _OWN_METHODS = frozenset({
        "render_tiled",
        "wait_pixels_ready_on_stream",
        "get_pixel_dptr",
        "close",
        "interop_available",
    })

    def __getattr__(self, name: str):
        # Note: __getattr__ is only invoked when the attribute is NOT found
        # by normal lookup. Properties + methods defined on the class are
        # caught before this. Anything truly missing routes to RPC_CALL.
        if name.startswith("_") or name in type(self)._OWN_METHODS:
            raise AttributeError(name)
        sock = self.__dict__.get("_sock")
        if sock is None:
            raise AttributeError(f"RemoteRenderer is closed; can't proxy '{name}'")

        def _proxy(*args, **kwargs):
            send_message(sock, Op.RPC_CALL, {
                "method": name,
                "args": list(args),
                "kwargs": dict(kwargs),
            })
            op, payload, _ = recv_message(sock)
            if op != Op.RPC_CALL_OK:
                raise RemoteRendererError(f"expected RPC_CALL_OK, got {op}")
            if "exception" in payload:
                raise RemoteRendererError(
                    f"child raised {payload['exception']}\nchild traceback:\n{payload['traceback']}"
                )
            return payload["result"]

        return _proxy

    def close(self) -> None:
        """Send SHUTDOWN, wait for child to exit cleanly."""
        if self._sock is None:
            return
        try:
            send_message(self._sock, Op.SHUTDOWN, None)
            try:
                op, _, _ = recv_message(self._sock)  # consume SHUTDOWN ack
            except (EOFError, BrokenPipeError):
                pass
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = None

        # Reap via subprocess.Popen.
        if hasattr(self, "_proc") and self._proc is not None:
            try:
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RemoteCudaInterop:
    """Trainer-side proxy mirroring `CudaVulkanInterop`.

    The real renderer's `CudaVulkanInterop` lives in the child. From the
    trainer's perspective, we just need to:
      - present a `wait_and_get_per_env_array() / overlap_get_per_env_array()`
        method that returns a `wp.array` view over the IPC pixel buffer,
      - have the wait happen in-stream (via `cuStreamWaitEvent` on the
        IPC pixel-ready event), so the trainer's torch context never blocks
        on a host-side fence.

    Created from `RemoteRenderer` rather than via the original
    `CudaVulkanInterop.__init__` constructor — `inner.py._init_cuda_interop`
    branches on `isinstance(self._nu, RemoteRenderer)` to decide which to
    instantiate.
    """

    def __init__(self, remote_renderer: RemoteRenderer):
        self._remote = remote_renderer
        self._tile_w = remote_renderer._width
        self._tile_h = remote_renderer._height
        self._num_cameras = remote_renderer._num_envs

    @property
    def image_w(self) -> int:
        return self._tile_w * self._num_cameras  # tiled-grid layout

    @property
    def image_h(self) -> int:
        return self._tile_h

    def wait_and_get_device_ptr(self):
        """Synchronous wait on the most-recent pixel-ready event; return ptr."""
        # In-stream wait — no host block.
        torch_stream = self._torch_stream_handle()
        self._remote.wait_pixels_ready_on_stream(torch_stream)
        # Return ptr of the buffer the child wrote (last_buffer_idx).
        return CUdeviceptr(self._remote.get_pixel_dptr()), self.image_w, self.image_h

    def overlap_get_device_ptr(self):
        """Same as wait_and_get_device_ptr for the remote variant.

        The child already double-buffers; from the trainer's view, the
        IPC mapping is just "the most recent frame the child finished."
        """
        return self.wait_and_get_device_ptr()

    def wait_and_get_per_env_array(self):
        import warp as wp

        dptr, _, _ = self.wait_and_get_device_ptr()
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
        return self.wait_and_get_per_env_array()

    def wait_and_get_warp_array(self):
        """Tiled-grid (image_h, image_w, 4) view — for code paths that
        consume the pre-detile layout. The remote renderer always uses the
        per_env_layout, so this is mostly for back-compat."""
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

    def overlap_get_warp_array(self):
        return self.wait_and_get_warp_array()

    def wait_render_done(self):
        """Compatibility shim — the remote renderer signals via cuda event,
        already used by wait_and_get_*. No-op here."""
        pass

    def _torch_stream_handle(self) -> int:
        import torch
        return int(torch.cuda.current_stream().cuda_stream)

    def close(self):
        # Nothing to free — RemoteRenderer.close() handles all IPC handles.
        pass
