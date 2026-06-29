# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RPC protocol for the subprocess renderer.

Wire format:
    Each message is `uint32 length (LE) || uint8 op_id || bytes payload`.
    Payload is `pickle.dumps(dict_or_None)`. SCM_RIGHTS-passed fds travel
    out-of-band on the same socket via `socket.sendmsg(ancdata=...)`.

We control both ends, run as the same UID, on the same node — pickle is
fine. msgpack/protobuf would buy nothing.

Op IDs are stable enums; do not renumber. Append-only.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
from enum import IntEnum
from typing import Any


class Op(IntEnum):
    # Initialization handshake.
    INIT = 1            # parent → child: render config (width, height, num_envs, etc.)
    INIT_OK = 2         # child → parent: ipc handles + ready signal
    # Scene mutation (one-time at setup).
    LOAD_USD = 3        # parent → child: usd_path
    LOAD_USD_OK = 4
    SET_TRANSFORM_LAYOUT = 5  # parent → child: shape descriptors for warp transform kernel
    SET_TRANSFORM_LAYOUT_OK = 6
    # Hot loop.
    RENDER = 10         # parent → child: frame_seq + body_q bytes + vp_inv bytes
    RENDER_OK = 11      # child → parent: frame_seq + telemetry
    # Generic method passthrough — invoke arbitrary NuRenderer methods on
    # the child. Used for set_camera, add_mesh, build_accel, etc.
    RPC_CALL = 12       # parent → child: {"method": str, "args": list, "kwargs": dict}
    RPC_CALL_OK = 13    # child → parent: {"result": Any} or {"exception": str, "traceback": str}
    # Lifecycle.
    PING = 20           # bidirectional heartbeat
    PONG = 21
    LOG = 30            # child → parent: stderr/stderr re-emit (severity-bridged)
    SHUTDOWN = 99       # parent → child


HEARTBEAT_INTERVAL_S = 0.5
HEARTBEAT_TIMEOUT_S = 2.0
WAIT_EVENT_TIMEOUT_FRAMES = 5  # multiplier of p99 frame time before declaring stuck


# ---- Wire helpers ----

def _send_all(sock: socket.socket, data: bytes) -> None:
    sent = 0
    while sent < len(data):
        n = sock.send(data[sent:])
        if n == 0:
            raise BrokenPipeError("socket closed mid-send")
        sent += n


def _recv_all(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError(f"socket closed; expected {remaining} more bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, op: Op, payload: Any = None, fds: list[int] | None = None) -> None:
    """Send one message; optionally pass file descriptors via SCM_RIGHTS."""
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("<IB", len(body), int(op))
    full = header + body
    if fds:
        ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack(f"{len(fds)}i", *fds))]
        sock.sendmsg([full], ancdata)
    else:
        _send_all(sock, full)


def recv_message(sock: socket.socket, recv_fds: int = 0) -> tuple[Op, Any, list[int]]:
    """Receive one message; if recv_fds > 0, accept ancillary SCM_RIGHTS data."""
    if recv_fds > 0:
        # Recvmsg the header + as much body as fits in one read; finish the
        # body via plain recv. SCM_RIGHTS only ever rides the first packet.
        ancbufsize = socket.CMSG_SPACE(recv_fds * struct.calcsize("i"))
        msg, ancdata, _, _ = sock.recvmsg(5 + 4096, ancbufsize)
        if len(msg) < 5:
            raise EOFError("short header")
        length, op_id = struct.unpack("<IB", msg[:5])
        body_part = msg[5:]
        body_remaining = length - len(body_part)
        if body_remaining > 0:
            body = body_part + _recv_all(sock, body_remaining)
        else:
            body = body_part[:length]

        fds: list[int] = []
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                n = len(cmsg_data) // struct.calcsize("i")
                fds.extend(struct.unpack(f"{n}i", cmsg_data))
        payload = pickle.loads(body)
        return Op(op_id), payload, fds

    header = _recv_all(sock, 5)
    length, op_id = struct.unpack("<IB", header)
    body = _recv_all(sock, length) if length else b""
    payload = pickle.loads(body) if body else None
    return Op(op_id), payload, []


# ---- Payload schemas (informal — payloads are dicts) ----

# INIT (parent → child):
#   {
#     "width": int, "height": int,
#     "num_envs": int,
#     "enable_rt": bool,
#     "enable_materials": bool,
#     "log_path": str,        # /tmp/nusd_renderer_<pid>.log
#     "parent_pid": int,
#   }
#
# INIT_OK (child → parent):
#   {
#     "child_pid": int,
#     "ipc_pixel_handle": bytes(64),   # cuIpcMemHandle
#     "ipc_pixels_ready_event": bytes(64),  # cuIpcEventHandle
#     "pixel_size_bytes": int,
#     "num_buffers": int,    # double-buffered = 2
#   }
#
# RENDER (parent → child):
#   {
#     "frame_seq": int,
#     "body_q_bytes": bytes,   # raw float32 transformf array (or warp array bytes)
#     "vp_inv_bytes": bytes,   # raw float32 4x4 matrices
#     "buffer_idx": int,       # which double-buffered slot to write into
#   }
#
# RENDER_OK (child → parent):
#   {
#     "frame_seq": int,
#     "buffer_idx": int,       # slot the parent should read from
#     "render_us": int,        # GPU render time
#     "copy_us": int,          # D2D copy time
#   }
