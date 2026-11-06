# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover the root-owned external OVStream runtime and selected interpreter."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import platform
import shlex
import subprocess
from typing import Sequence
DEFAULT_OVRTX_NATIVE_CLIENT_MODULE = ""
DEFAULT_OVRTX_PORT = ""
DEFAULT_OVPHYSX_ADDRESS = ""
@dataclass(frozen=True)
class BundledRuntimeDefaults:
    """Current request metadata; empty native slots carry no loader/selector behavior."""
    root: Path
    platform_id: str
    worker_command: str = ""
    native_client_path: str = ""
    ovphysx_worker_command: str = ""
    ovphysx_native_client_path: str = ""

def addon_root() -> Path:
    configured = os.environ.get("OV_BLENDER_EXAMPLE_BUNDLED_ADDON_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]

def public_sdk_package_root() -> Path | None:
    """Installed package marker selects SDK mode even before setup completes."""
    package = Path(__file__).resolve().parent
    for root in (package, package.parent, addon_root()):
        if (root / 'public-sdk.json').is_file():
            return root
    return None

def public_sdk_python() -> str | None:
    configured = os.environ.get('OV_BLENDER_PUBLIC_SDK_PYTHON', '').strip()
    if configured:
        return configured
    if public_sdk_package_root() is not None:
        from . import public_sdk_install
        return public_sdk_install.ready_python()
    return None

def public_sdk_worker_script() -> Path:
    package = public_sdk_package_root()
    if package is not None:
        return package / 'runtime' / 'public_sdk_worker.py'
    return Path(__file__).resolve().parents[2] / 'runtime' / 'public_sdk_worker.py'

def current_platform_id() -> str:
    machine = platform.machine().lower()
    if os.name == "posix" and platform.system().lower() == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x64"
    if os.name == "posix" and platform.system().lower() == "linux" and machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    if os.name == "nt" and platform.system().lower() == "windows" and machine in {"x86_64", "amd64"}:
        return "windows-x64"
    return ""

def serialize_command(args: Sequence[str], *, platform_id: str | None = None) -> str:
    values = [str(arg) for arg in args]
    platform_id = platform_id or current_platform_id()
    if platform_id == "windows-x64":
        return subprocess.list2cmdline(values)
    return shlex.join(values)

def parse_command(value: str, *, platform_id: str | None = None) -> list[str]:
    platform_id = platform_id or current_platform_id()
    if not platform_id:
        platform_id = "windows-x64" if os.name == "nt" else "linux-x64"
    if platform_id != "windows-x64":
        return shlex.split(value)

    args: list[str] = []
    index = 0
    while index < len(value):
        while index < len(value) and value[index] in " \t":
            index += 1
        if index == len(value):
            break
        arg: list[str] = []
        quoted = False
        while index < len(value) and (quoted or value[index] not in " \t"):
            if value[index] == "\\":
                start = index
                while index < len(value) and value[index] == "\\":
                    index += 1
                backslashes = index - start
                if index < len(value) and value[index] == '"':
                    arg.extend("\\" * (backslashes // 2))
                    if backslashes % 2:
                        arg.append('"')
                    else:
                        quoted = not quoted
                    index += 1
                else:
                    arg.extend("\\" * backslashes)
                continue
            if value[index] == '"':
                quoted = not quoted
            else:
                arg.append(value[index])
            index += 1
        args.append("".join(arg))
    return args

def public_sdk_selected() -> bool:
    return True


def defaults(*, root: Path | None = None, **_unused) -> BundledRuntimeDefaults:
    python = public_sdk_python()
    source = public_sdk_worker_script().parent.parent
    command = serialize_command([python, str(public_sdk_worker_script())]) if python else ""
    return BundledRuntimeDefaults(root=source, platform_id=current_platform_id(),
        worker_command=command, ovphysx_worker_command=command)
