# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket

from dev_variant_presenter.config import _port_is_free, find_free_port


def test_find_free_port_returns_preferred_when_free():
    # grab then release an ephemeral port; find_free_port should hand that same one back
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    assert find_free_port(p) == p


def test_find_free_port_skips_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        s.listen()
        chosen = find_free_port(busy)        # busy is held open for the whole scan
        assert chosen != busy
        assert _port_is_free(chosen)
