# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform command-line tokenization helpers."""

from __future__ import annotations

import os
import shlex


def split_command(value: str, *, windows: bool | None = None) -> list[str]:
    """Split a command while removing quotes retained by ``shlex`` on Windows.

    ``subprocess.list2cmdline`` quotes arguments containing spaces. In non-POSIX
    mode ``shlex.split`` keeps those surrounding quotes, which makes filesystem
    checks treat them as literal filename characters.
    """

    if windows is None:
        windows = os.name == "nt"
    parts = shlex.split(value, posix=not windows)
    if not windows:
        return parts
    return [_strip_matching_quotes(part) for part in parts]


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__ = ["split_command"]
