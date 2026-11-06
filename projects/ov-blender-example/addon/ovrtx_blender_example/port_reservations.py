# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explain a refused bind by naming the OS port reservation that caused it.

Windows hands whole port blocks to WinNAT / Hyper-V and to dynamic-port
consumers, and a bind inside one of those blocks fails with ``WinError 10013``
(``WSAEACCES``) — the same errno an ordinary permission problem gives. A worker
that dies on such a bind surfaces here as
``ovphysx_grpc_server exited before reporting SERVING``, which says nothing about
why and sends the reader looking at the worker.

This module turns that into a sentence naming the reserved range. It is
diagnosis only: nothing here changes which port is used, and a failure to
diagnose never becomes a failure to run.

The reservations move. ``netsh int ipv4 show excludedportrange protocol=tcp`` is
read live rather than baked into a constant, because the ranges are reassigned
across reboots and differ per machine.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess

#: `netsh` prints one "<start> <end>" pair per line, sometimes with a trailing
#: `*` marking an administered exclusion.
_RANGE_RE = re.compile(r"^\s*(\d{1,5})\s+(\d{1,5})\s*\*?\s*$")


def split_address(address: str) -> tuple[str, int]:
    """Split ``host:port``; port 0 means "not parseable as a TCP endpoint"."""
    host, _, port = str(address or "").rpartition(":")
    try:
        return host, int(port)
    except ValueError:
        return host, 0


def excluded_ranges() -> list[tuple[int, int]]:
    """TCP port ranges the OS has reserved. Empty when unknown or non-Windows."""
    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            ["netsh", "interface", "ipv4", "show", "excludedportrange", "protocol=tcp"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    ranges = []
    for line in completed.stdout.splitlines():
        match = _RANGE_RE.match(line)
        if match:
            ranges.append((int(match.group(1)), int(match.group(2))))
    return ranges


def reserving_range(port: int) -> tuple[int, int] | None:
    for low, high in excluded_ranges():
        if low <= port <= high:
            return (low, high)
    return None


def bindable(address: str) -> bool:
    """Whether *address* can be bound right now.

    Uses no ``SO_REUSEADDR``: the question is whether this exact endpoint is
    available to a fresh listener, which is what the worker will attempt.
    """
    host, port = split_address(address)
    if not port:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host or "127.0.0.1", port))
        return True
    except OSError:
        return False


def explain(address: str) -> str:
    """A sentence naming why *address* cannot be bound, or "" if it can.

    Returning "" for "no idea" as well as for "it is fine" is deliberate: this
    text is appended to an existing error, and a speculative explanation is worse
    than none.
    """
    host, port = split_address(address)
    if not port or bindable(address):
        return ""

    reserved = reserving_range(port)
    if reserved is not None:
        low, high = reserved
        return (
            f"Port {port} is inside the OS-reserved TCP range {low}-{high}, so no "
            f"process can bind {address} (Windows reports WinError 10013 / "
            f"WSAEACCES, which looks like a permission error). List the ranges with "
            f"`netsh interface ipv4 show excludedportrange protocol=tcp` and pick a "
            f"port outside them, then set BOTH "
            f"OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS and "
            f"OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND -- the worker command bakes "
            f"the port, so changing only the address leaves the worker listening "
            f"elsewhere."
        )
    return (
        f"{address} could not be bound. It is not inside any OS-reserved TCP range, "
        f"so the likely cause is another process already listening on port {port}."
    )
