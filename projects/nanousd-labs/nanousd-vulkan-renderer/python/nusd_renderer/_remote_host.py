# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Child process entry point for the subprocess renderer.

Architecture (see PHASE1_PLAN.md):
    - Forked by `_remote.RemoteRenderer.__init__` in the trainer process.
    - Receives the RPC socket as fd 3 (parent uses socketpair + dup2).
    - Initialises CUDA + Vulkan + the real NuRenderer.
    - Allocates `cuMemAlloc`-backed double-buffered IPC pixel buffers.
    - Creates an INTERPROCESS CUDA event for pixel-ready signaling.
    - Sends INIT_OK with the IPC handles to the parent.
    - Runs an RPC loop: per RENDER, receive body_q + vp_inv bytes, run
      the warp transform kernel + Vulkan dispatch, D2D-copy pixels into
      the IPC export buffer, record the event, send RENDER_OK.

Lifecycle hardening (per ops-critic agent):
    - PDEATHSIG: child dies if parent dies.
    - SIGTERM handler: clean Vulkan/CUDA teardown then _exit(0).
    - MIG detection: refuses to start under MIG (cuIpcGetMemHandle
      returns NOT_SUPPORTED).
    - VRAM check at INIT.
    - Per-PID log file at /tmp/nusd_renderer_<pid>.log.

This is a Python module rather than a script. A standalone
`python -m nusd_renderer._remote_host` entrypoint is provided for debugging
the private low-level backend.
"""

from __future__ import annotations

import ctypes
import os
import signal
import socket
import struct
import sys
import time
import traceback
from typing import Any

# All CUDA helpers + signature bindings.
from . import _remote_cuda as cu
from ._remote_cuda import (
    CUdeviceptr,
    CUstream,
    libcuda,
    _check,
)
from . import _remote_protocol as proto
from ._remote_protocol import Op, send_message, recv_message


# ---- Linux helpers ----

PR_SET_PDEATHSIG = 1
SIGKILL = 9
SIGTERM = 15


def _set_pdeathsig(sig: int = SIGTERM) -> None:
    """Cause the kernel to send `sig` to this process when its parent dies.

    Important: must be called in child after fork, before any CUDA/Vulkan
    init. Without this, an orphaned renderer leaks GPU memory until manual
    cleanup.
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    rc = libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG, {sig}) failed: {os.strerror(err)}")
    # Defensive: the parent might already be dead by the time we call this.
    if os.getppid() == 1:
        os.kill(os.getpid(), sig)


# ---- Host state ----

class HostState:
    def __init__(self):
        self.config: dict | None = None
        self.cuda_dev = None
        self.cuda_ctx = None
        self.nu_renderer = None  # NuRenderer
        self.cuda_stream: CUstream | None = None
        # Double-buffered IPC pixel buffers (cuMemAlloc).
        self.ipc_pixel_dptrs: list[CUdeviceptr] = []
        self.ipc_pixel_handles: list[bytes] = []
        # Pixel-ready event (one shared, signaled per frame).
        self.ipc_pixel_event = None
        self.ipc_pixel_event_handle: bytes | None = None
        # The Vulkan-imported pixel buffer + interop wrapper (set up after
        # the parent calls SET_TRANSFORM_LAYOUT or equivalent scene ready).
        self.vk_imported_dptrs: list[CUdeviceptr] = []
        self.cuda_interop = None  # CudaVulkanInterop (in child's context)
        # True once a scene is loaded and the real render path is wired up.
        self.scene_ready = False
        # Frame counter and stats.
        self.frame_seq = 0
        self.last_ping_t = 0.0


# ---- Lifecycle handlers ----

_state: HostState | None = None


def _shutdown_handler(signum, frame):
    """Catch SIGTERM/SIGINT, do clean teardown, exit."""
    sys.stderr.write(f"[remote_host] received signal {signum}, shutting down\n")
    sys.stderr.flush()
    if _state is not None:
        try:
            _teardown(_state)
        except Exception:
            traceback.print_exc()
    os._exit(0)


def _teardown(state: HostState) -> None:
    if state.nu_renderer is not None:
        try:
            state.nu_renderer.close()
        except Exception:
            traceback.print_exc()
        state.nu_renderer = None
    for dptr in state.ipc_pixel_dptrs:
        try:
            libcuda.cuMemFree_v2(dptr)
        except Exception:
            pass
    state.ipc_pixel_dptrs.clear()
    if state.ipc_pixel_event is not None:
        try:
            libcuda.cuEventDestroy_v2(state.ipc_pixel_event)
        except Exception:
            pass
        state.ipc_pixel_event = None


# ---- Init handshake ----

def _do_init(state: HostState, sock: socket.socket, payload: dict) -> None:
    """Handle INIT message: bring up CUDA, NuRenderer, IPC buffers; reply INIT_OK."""
    state.config = payload
    width = payload["width"]
    height = payload["height"]
    num_envs = payload["num_envs"]
    enable_rt = payload.get("enable_rt", True)
    enable_materials = payload.get("enable_materials", False)

    sys.stderr.write(f"[remote_host] INIT: {width}x{height} num_envs={num_envs} rt={enable_rt}\n")
    sys.stderr.flush()

    # 1. CUDA context (must happen after fork, never inherited).
    state.cuda_dev, state.cuda_ctx = cu.init_cuda_context()

    # 2. Hardening checks.
    if cu.is_mig_enabled(state.cuda_dev):
        raise RuntimeError("MIG mode is enabled on this device — cuIpcGetMemHandle is not supported under MIG. Refusing subprocess mode; trainer should fall back to in-process renderer.")

    free, total = cu.free_vram_bytes()
    sys.stderr.write(f"[remote_host] VRAM: {free / 1e9:.2f} / {total / 1e9:.2f} GB free\n")
    sys.stderr.flush()
    if free < 2 * 1024 * 1024 * 1024:
        raise RuntimeError(f"insufficient VRAM: {free / 1e9:.2f} GB free, need >= 2 GB")

    # 3. NuRenderer instance.
    from ._bindings import NuRenderer

    state.nu_renderer = NuRenderer(
        width=width,
        height=height,
        enable_rt=enable_rt,
        enable_materials=enable_materials,
    )

    # 4. Allocate double-buffered IPC pixel buffers (cuMemAlloc, IPC-shareable).
    pixel_size_bytes = num_envs * width * height * 4  # RGBA8 per env
    n_buffers = 2
    state.ipc_pixel_dptrs = []
    state.ipc_pixel_handles = []
    for _ in range(n_buffers):
        dptr, h = cu.alloc_ipc_buffer(pixel_size_bytes)
        # Zero-init so first read returns black, not garbage.
        s = CUstream()
        _check("cuStreamCreate", libcuda.cuStreamCreate(ctypes.byref(s), 1))
        _check("cuMemsetD8Async", libcuda.cuMemsetD8Async(dptr, 0, pixel_size_bytes, s))
        _check("cuStreamSynchronize", libcuda.cuStreamSynchronize(s))
        state.ipc_pixel_dptrs.append(dptr)
        state.ipc_pixel_handles.append(cu.handle_to_bytes(h))

    # 5. Pixel-ready event.
    state.ipc_pixel_event, ev_handle = cu.create_ipc_event()
    state.ipc_pixel_event_handle = cu.handle_to_bytes(ev_handle)

    # 6. Render-side stream for the D2D copy.
    state.cuda_stream = CUstream()
    _check("cuStreamCreate", libcuda.cuStreamCreate(ctypes.byref(state.cuda_stream), 1))

    # 7. Optional canned scene + interop setup. The trainer can opt in via
    #    payload["canned_scene"]=True. This sets up a minimal scene
    #    (one mesh with `num_envs` instances), brings up CudaVulkanInterop in
    #    the child's context, and flips state.scene_ready so the real
    #    render path runs instead of the cuMemset placeholder.
    if payload.get("canned_scene", False):
        _setup_canned_scene_and_interop(state)

    # 8. Send INIT_OK with handles.
    init_ok = {
        "child_pid": os.getpid(),
        "ipc_pixel_handles": state.ipc_pixel_handles,
        "ipc_pixel_event_handle": state.ipc_pixel_event_handle,
        "pixel_size_bytes": pixel_size_bytes,
        "num_buffers": n_buffers,
        "width": width,
        "height": height,
        "num_envs": num_envs,
        "scene_ready": state.scene_ready,
    }
    send_message(sock, Op.INIT_OK, init_ok)
    sys.stderr.write(f"[remote_host] INIT_OK sent (pid={os.getpid()}, {n_buffers} buffers × {pixel_size_bytes // 1024} KB)\n")
    sys.stderr.flush()


# ---- Main RPC loop ----

def _serve_loop(state: HostState, sock: socket.socket) -> None:
    """Receive RPCs forever; dispatch to handlers."""
    while True:
        try:
            op, payload, _fds = recv_message(sock)
        except (EOFError, BrokenPipeError):
            sys.stderr.write("[remote_host] parent socket closed; exiting\n")
            sys.stderr.flush()
            break

        if op == Op.INIT:
            _do_init(state, sock, payload)
        elif op == Op.PING:
            send_message(sock, Op.PONG, {"t_ns": time.perf_counter_ns()})
            state.last_ping_t = time.perf_counter()
        elif op == Op.RENDER:
            _do_render(state, sock, payload)
        elif op == Op.RPC_CALL:
            _do_rpc_call(state, sock, payload)
        elif op == Op.SHUTDOWN:
            sys.stderr.write("[remote_host] SHUTDOWN received\n")
            sys.stderr.flush()
            send_message(sock, Op.SHUTDOWN, None)
            break
        else:
            sys.stderr.write(f"[remote_host] unknown op {op}\n")
            sys.stderr.flush()


def _setup_canned_scene_and_interop(state: HostState) -> None:
    """Set up a minimal self-contained scene + CudaVulkanInterop in the child.

    Used for end-to-end perf validation without needing the parent to ship
    real scene data. Adds one triangle mesh, `num_envs` instances arranged
    in a row, configures camera, builds accel, brings up CudaVulkanInterop.

    After this, state.scene_ready=True and the real render path runs.
    """
    import numpy as np

    n = state.config["num_envs"]

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    mesh_id = state.nu_renderer.add_mesh(positions=positions, indices=indices)

    for i in range(n):
        xform = np.eye(4, dtype=np.float32)
        xform[0, 3] = i * 2.0
        state.nu_renderer.add_mesh_instance(prototype_mesh_id=mesh_id, transform=xform)

    state.nu_renderer.set_camera(
        eye=(n / 2.0, -10.0, 5.0),
        target=(n / 2.0, 0.0, 0.0),
        fov_degrees=60.0,
    )

    state.nu_renderer.build_accel()

    # Drive one render to set up the renderer's tiled SSBO + interop buffer.
    state.nu_renderer.set_per_env_layout(True)

    # Use a 1×1 vp_inv as a dummy; the real path takes a (num_envs, 32) array.
    # Actually we need a proper (num_envs, 32) here; the renderer will fault
    # otherwise. Build a unit-VP per env.
    vp_inv = np.zeros((n, 32), dtype=np.float32)
    for i in range(n):
        # Identity 4x4 in slot [0..16) (view_inv) and [16..32) (proj_inv).
        for r in range(4):
            vp_inv[i, r * 4 + r] = 1.0
            vp_inv[i, 16 + r * 4 + r] = 1.0

    # Trigger a render so the tiled image buffer + per-env layout exist.
    from ._bindings import NU_RENDER_RT
    state.nu_renderer.render_tiled(vp_inv, n, state.config["width"], state.config["height"], NU_RENDER_RT)

    # Wait for completion before attempting interop setup.
    if hasattr(state.nu_renderer, "wait_tiled_complete"):
        state.nu_renderer.wait_tiled_complete()

    # Bring up CudaVulkanInterop in the child's context.
    from ._cuda_interop import CudaVulkanInterop
    state.cuda_interop = CudaVulkanInterop(
        state.nu_renderer,
        n,
        state.config["width"],
        state.config["height"],
        skip_staging=True,
        use_semaphore_sync=False,
    )

    # Cache the canned-scene vp_inv so we can re-render efficiently.
    state.canned_vp_inv = vp_inv

    state.scene_ready = True
    sys.stderr.write(f"[remote_host] canned scene ready: {n} instances, interop set up\n")
    sys.stderr.flush()


def _do_rpc_call(state: HostState, sock: socket.socket, payload: dict) -> None:
    """Generic NuRenderer method passthrough.

    Payload: {"method": str, "args": [...], "kwargs": {...}}.
    Invokes `getattr(state.nu_renderer, method)(*args, **kwargs)` and replies
    with {"result": ...} or {"exception": str, "traceback": str}.

    Used for the long tail of NuRenderer methods (add_mesh, set_camera,
    build_accel, etc.) where adding a dedicated RPC op for each is overkill.
    """
    method_name = payload["method"]
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})

    try:
        if state.nu_renderer is None:
            raise RuntimeError("NuRenderer not yet initialized")
        method = getattr(state.nu_renderer, method_name, None)
        if method is None:
            raise AttributeError(f"NuRenderer has no method '{method_name}'")
        result = method(*args, **kwargs)
        # Some NuRenderer methods return numpy arrays or scalars; both pickle.
        # CUDA pointers (CUdeviceptr) don't pickle as device pointers but as
        # integer values, which is what callers want.
        send_message(sock, Op.RPC_CALL_OK, {"result": result})
    except Exception as e:
        traceback_str = traceback.format_exc()
        sys.stderr.write(f"[remote_host] RPC_CALL '{method_name}' raised: {e}\n{traceback_str}\n")
        sys.stderr.flush()
        send_message(sock, Op.RPC_CALL_OK, {
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback_str,
        })


def _do_render(state: HostState, sock: socket.socket, payload: dict) -> None:
    """Handle a RENDER RPC.

    Production path:
      1. Optional vp_inv update via `nu_set_camera_explicit` or by writing
         the renderer's vp_inv buffer.
      2. Optional warp transform-kernel launch (skipped here when the parent
         already pushed transforms via `set_transforms`).
      3. `nu_render_tiled` — the Vulkan dispatch.
      4. Wait for the renderer's interop fence (host vkWaitForFences inside
         the renderer's existing path).
      5. `cuMemcpyDtoDAsync` from the renderer's Vulkan-imported buffer into
         the IPC export buffer.
      6. `cuEventRecord` on the pixel-ready event (parent waits via
         `cuStreamWaitEvent`, no host sync).

    Phase 1 milestone — this implementation keeps the cuMemset placeholder
    when the renderer hasn't been set up with a scene (i.e. the parent
    didn't call `LOAD_USD` / `SET_TRANSFORM_LAYOUT` first). A subsequent
    PR wires the real path through.
    """
    state.frame_seq += 1
    seq = payload["frame_seq"]
    buffer_idx = seq % len(state.ipc_pixel_dptrs)

    t0 = time.perf_counter_ns()

    if state.scene_ready:
        # Real render path: parent pushed scene + transforms; just dispatch.
        _real_render_into_ipc_buffer(state, buffer_idx, payload)
    else:
        # Placeholder: write a per-frame pattern so the IPC plumbing can be
        # validated independently of NuRenderer scene setup.
        pattern = (seq & 0xFF)
        pixel_size = state.config["num_envs"] * state.config["width"] * state.config["height"] * 4
        _check(
            "cuMemsetD8Async",
            libcuda.cuMemsetD8Async(state.ipc_pixel_dptrs[buffer_idx], pattern, pixel_size, state.cuda_stream),
        )

    _check("cuEventRecord", libcuda.cuEventRecord(state.ipc_pixel_event, state.cuda_stream))

    elapsed_us = (time.perf_counter_ns() - t0) / 1000

    send_message(sock, Op.RENDER_OK, {
        "frame_seq": seq,
        "buffer_idx": buffer_idx,
        "render_us": int(elapsed_us),
    })


def _real_render_into_ipc_buffer(state: HostState, buffer_idx: int, payload: dict) -> None:
    """Drive the real NuRenderer for one frame, copy result into the IPC buf.

    Uses state.canned_vp_inv if no payload vp_inv was provided (canned-scene
    mode). For real-IsaacLab-integration mode, the parent ships vp_inv (and
    transforms) per-frame.
    """
    import numpy as np

    n = state.config["num_envs"]

    vp_inv_bytes = payload.get("vp_inv_bytes", b"") or b""
    if vp_inv_bytes:
        vp_inv = np.frombuffer(vp_inv_bytes, dtype=np.float32)
        if vp_inv.size == n * 32:
            vp_inv = vp_inv.reshape(n, 32).copy()  # copy: numpy views into bytes are read-only
        else:
            raise RuntimeError(f"vp_inv size mismatch: got {vp_inv.size}, expected {n * 32}")
    else:
        vp_inv = state.canned_vp_inv

    # Dispatch the Vulkan render.
    from ._bindings import NU_RENDER_RT
    state.nu_renderer.render_tiled(
        vp_inv,
        n,
        state.config["width"],
        state.config["height"],
        NU_RENDER_RT,
    )

    # Copy from the renderer's Vulkan-imported buffer into our IPC export.
    # The interop's overlap_get_per_env_array waits on the previous frame
    # (always already done), avoiding a host fence wait. We use
    # wait_and_get_device_ptr for the synchronous path here — simpler,
    # production-correct.
    src_dptr, w, h = state.cuda_interop.wait_and_get_device_ptr()
    pixel_size = n * state.config["width"] * state.config["height"] * 4
    _check(
        "cuMemcpyDtoDAsync",
        libcuda.cuMemcpyDtoDAsync_v2(
            state.ipc_pixel_dptrs[buffer_idx],
            src_dptr,
            pixel_size,
            state.cuda_stream,
        ),
    )


# ---- Entry point ----

def serve(rpc_fd: int) -> None:
    """Run the host loop on the given socket fd.

    Expected to be called from a subprocess.Popen-spawned child where the
    parent passed `rpc_fd` via `pass_fds=[fd]` and the fd-number-in-child
    came in via argv.
    """
    global _state

    # Redirect stdout/stderr to a per-PID log file (per ops-critic).
    log_path = f"/tmp/nusd_renderer_{os.getpid()}.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    _state = HostState()

    # Hardening: die when parent dies.
    try:
        _set_pdeathsig(SIGTERM)
    except Exception as e:
        sys.stderr.write(f"[remote_host] WARNING: prctl(PDEATHSIG) failed: {e}\n")
        sys.stderr.flush()

    # Signal handlers.
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # Convert fd → socket object.
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=rpc_fd)

    sys.stderr.write(f"[remote_host] serving on rpc_fd={rpc_fd} pid={os.getpid()}\n")
    sys.stderr.flush()

    try:
        _serve_loop(_state, sock)
    except Exception as e:
        sys.stderr.write(f"[remote_host] EXCEPTION in serve loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
    finally:
        _teardown(_state)
        sock.close()


def main():
    """Entrypoint for `python -m nusd_renderer._remote_host`. Reads RPC fd
    from argv[1] (set by the parent's exec)."""
    if len(sys.argv) < 2:
        print("usage: _remote_host RPC_FD", file=sys.stderr)
        sys.exit(2)
    rpc_fd = int(sys.argv[1])
    serve(rpc_fd)


if __name__ == "__main__":
    main()
