#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared command helpers for graph-provided validation commands."""

from __future__ import annotations

import os
import shlex
import stat
import sys
from pathlib import Path


PYTHON_PLACEHOLDER = "{python}"


def split_command(command: str) -> list[str]:
    argv = shlex.split(command, posix=os.name != "nt")
    return [sys.executable if arg == PYTHON_PLACEHOLDER else arg for arg in argv]


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(arg) if os.name != "nt" else arg for arg in argv)


def ensure_executable(argv: list[str]) -> None:
    """Add owner+group+other execute bits to argv[0] if it is a regular file lacking exec permission.

    Generated adapter and dump-layer .exe artifacts can land with mode -rw-r--r-- depending on
    the umask of the regen step. Without this, subprocess.run raises PermissionError before
    the program ever starts. Idempotent and safe to call on any command: non-existent files
    and already-executable files are left untouched.
    """
    if not argv:
        return
    try:
        path = Path(argv[0])
    except (TypeError, ValueError):
        return
    try:
        if not path.is_file():
            return
        current = path.stat().st_mode
        desired = current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if current != desired:
            path.chmod(desired)
    except OSError:
        # Best-effort; if we can't chmod, let subprocess.run surface the underlying error.
        pass
