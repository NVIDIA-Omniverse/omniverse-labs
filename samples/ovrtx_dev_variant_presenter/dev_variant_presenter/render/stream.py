# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ovstream WebRTC wrapper: RGBA8 (CUDA) -> BGRA8 swizzle -> stream_video, with a
last-frame heartbeat. Render-thread owned; callbacks fire on ovstream SDK threads
(kept short).

ovstream 0.4.5 still documents `VideoFrame.from_cuda_array` as accepting **only**
BGRA8 uint8 (H,W,4) — confirmed against the installed package docstring and against
another ovstream 0.4.x consumer (also pinned to 0.4.5, still swizzles). The
RGBA→BGRA Warp kernel therefore remains required, not a leftover 0.3 habit.

ovstream.initialize()/shutdown() are PROCESS-GLOBAL and must run at most once each per
process. Cycling them — which any per-reload streamer rebuild does — degrades the NVENC
encoder context and the stream goes silently black, recoverable only by a full process
restart. So global init happens once; dropping a stale client recreates ONLY the
ovstream.Server (rebuild_server), never the global context.

`submit_last` re-submits the previously encoded frame. It is not a redundant safety net:
this app deliberately stops rendering when nothing changes (idle GPU throttle) and blocks
the render loop for seconds at a time during a reload-path variant switch (USD_LOCK +
build_composite + populate_usd). WebRTC drops a client after roughly 7s without media, so
both paths pump `submit_last` to keep the session alive without burning GPU on identical
frames. Callers: the idle throttle and the reopen/batch heartbeats in `render.runtime`.
"""
from __future__ import annotations

import warp as wp
import ovstream

_OVSTREAM_INITED = False   # process-global: ovstream.initialize() runs exactly once


@wp.kernel
def _swap_rb(buf: wp.array3d(dtype=wp.uint8)):
    x, y = wp.tid()
    r = buf[y, x, 0]
    b = buf[y, x, 2]
    buf[y, x, 0] = b
    buf[y, x, 2] = r


class Streamer:
    def __init__(self, width: int, height: int, signal_port: int,
                 on_input=None, on_connection=None):
        self.width = width
        self.height = height
        self.signal_port = signal_port
        self._on_input = on_input
        self._on_connection = on_connection
        self._server = None
        self._buf = None
        self._stream = None
        self._event = None
        self._has_frame = False
        self._initialized = False

    def start(self) -> None:
        global _OVSTREAM_INITED
        wp.init()
        self._alloc_buffer(self.width, self.height)
        if not _OVSTREAM_INITED:
            ovstream.initialize()   # process-global, ONCE — see module docstring
            _OVSTREAM_INITED = True
        self._initialized = True
        self._start_server()

    def _alloc_buffer(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self._buf = wp.zeros((height, width, 4), dtype=wp.uint8, device="cuda:0")
        self._stream = wp.get_stream("cuda:0")
        self._event = wp.Event(device="cuda:0")
        self._has_frame = False

    def _start_server(self) -> None:
        s = ovstream.Server(ovstream.ServerType.WEBRTC)
        if self._on_connection is not None:
            s.on_connection = self._on_connection
        if self._on_input is not None:
            s.on_input = self._on_input
        cfg = ovstream.ServerConfig(width=self.width, height=self.height)
        cfg.webrtc_signal_port = self.signal_port
        # Pin the advertised WebRTC media candidate to loopback. This is a localhost-only
        # viewer; left unset, ovstream auto-detects an IP from the host's interfaces, and on a
        # machine with many adapters (Wi-Fi/Ethernet/Bluetooth + link-local 169.254.*) it picks
        # one the browser's ICE can't pair with (observed: media bound to 127.0.0.238) -> the
        # signaling channel connects but video never flows (black viewport) -> the frontend
        # dry-pipe watchdog escalates to /api/restart -> exit-43 death spiral. Auto-detection is
        # non-deterministic across runs, which is why the stream worked some launches and not
        # others. (For remote access this would need the host's reachable LAN IP instead.)
        cfg.webrtc_public_ip = "127.0.0.1"
        s.start(cfg)
        self._server = s

    def rebuild_server(self, width: int | None = None, height: int | None = None) -> None:
        """Drop the held (possibly half-open / ghost) client by recreating ONLY the
        ovstream.Server — NOT the process-global ovstream context. Optionally resize the
        frame buffer first (resolution change). This is the safe, repeatable eviction:
        cycling ovstream.initialize()/shutdown() instead wedges the encoder black."""
        try:
            if self._server is not None:
                self._server.stop()
                self._server.close()
        except Exception:  # noqa: BLE001
            pass
        self._server = None
        if width is not None and height is not None and (width, height) != (self.width, self.height):
            self._alloc_buffer(width, height)
        self._start_server()

    @property
    def is_client_connected(self) -> bool:
        return bool(self._server is not None and getattr(self._server, "is_client_connected", False))

    def submit(self, rgba_cuda) -> None:
        """rgba_cuda: a warp uint8 (H,W,4) array (LdrColor mapped to CUDA). Copies in,
        swizzles to BGRA in the persistent buffer, streams, and remembers it."""
        wp.copy(self._buf, rgba_cuda)
        wp.launch(_swap_rb, dim=(self.width, self.height), inputs=[self._buf], device="cuda:0")
        self._stream.record_event(self._event)
        self._has_frame = True
        self._send()

    def submit_last(self) -> None:
        """Re-stream the last BGRA buffer to keep the WebRTC heartbeat alive (~7s window)."""
        if self._has_frame:
            self._send()

    def _send(self) -> None:
        try:
            vf = ovstream.VideoFrame.from_cuda_array(
                self._buf,
                sync=ovstream.CudaSync(stream=self._stream.cuda_stream,
                                       wait_event=self._event.cuda_event))
            self._server.stream_video(vf)
        except ovstream.OvstreamError:
            pass  # transient disconnect race / no client — drop, don't crash the loop

    def stop(self) -> None:
        """Full teardown INCLUDING the process-global shutdown — only at process/render-loop
        exit. Mid-session client eviction or resize uses rebuild_server (no global cycle)."""
        global _OVSTREAM_INITED
        try:
            if self._server is not None:
                self._server.stop()
                self._server.close()
        except Exception:
            pass
        if self._initialized:
            ovstream.shutdown()
            _OVSTREAM_INITED = False
            self._initialized = False
