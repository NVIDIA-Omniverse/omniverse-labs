# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime settings: ports, fixed stream resolution, viewer prim paths."""
from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    control_port: int = 8080
    signal_port: int = 49100
    stream_resolution: tuple[int, int] = (1280, 720)
    render_product_path: str = "/Render/Viewport"
    viewer_camera_path: str = "/Viewer/Camera"


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Best-effort check that `port` can be bound on `host` right now. TOCTOU-racy by
    nature (something could grab it before the real server binds), but fine for picking
    a dev port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(preferred: int, host: str = "127.0.0.1", scan: int = 25) -> int:
    """`preferred` if free; else the next free port within `scan` above it; else an
    OS-assigned ephemeral port. Lets Dev Variant Presenter coexist with other local web apps
    without manual port juggling. The scan keeps the chosen port near the default
    (friendly URLs) before falling back to a random high port."""
    for port in range(preferred, preferred + scan + 1):
        if _port_is_free(port, host):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
