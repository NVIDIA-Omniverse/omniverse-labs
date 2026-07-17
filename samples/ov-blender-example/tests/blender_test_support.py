# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared executable discovery for headless Blender tests."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


_NATIVE_BLENDER_COMMANDS = {
    "darwin": ("/Applications/Blender.app/Contents/MacOS/Blender",),
    "linux": ("/snap/blender/current/blender",),
    "win32": (r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",),
}


def blender_executable() -> Path | None:
    configured = os.environ.get("BLENDER_COMMAND")
    if configured is not None:
        executable = shutil.which(configured)
        if executable is None:
            raise ValueError(
                f"BLENDER_COMMAND does not identify an executable: {configured!r}"
            )
        return Path(executable).absolute()

    for command in ("blender", *_NATIVE_BLENDER_COMMANDS.get(sys.platform, ())):
        if executable := shutil.which(command):
            return Path(executable).absolute()
    return None
