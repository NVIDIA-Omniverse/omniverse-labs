# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled runtime discovery for release zips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import shlex
import subprocess
from typing import Sequence


DEFAULT_OVRTX_PORT = "50051"
DEFAULT_OVRTX_NATIVE_CLIENT_MODULE = "ovrtx_bridge_client"
DEFAULT_OVPHYSX_ADDRESS = "127.0.0.1:50094"


@dataclass(frozen=True)
class BundledRuntimeDefaults:
    root: Path
    platform_id: str
    worker_command: str = ""
    native_client_path: str = ""
    ovphysx_server: str = ""
    ovphysx_worker_command: str = ""
    ovphysx_native_client_path: str = ""
    ovphysx_root: str = ""
    ovphysx_bridge_runtime_root: str = ""
    ovruntime_root: str = ""


def addon_root() -> Path:
    configured = os.environ.get("OV_BLENDER_EXAMPLE_BUNDLED_ADDON_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def current_platform_id() -> str:
    machine = platform.machine().lower()
    if os.name == "posix" and platform.system().lower() == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x64"
    if os.name == "posix" and platform.system().lower() == "linux" and machine in {"aarch64", "arm64"}:
        return "linux-aarch64"
    if os.name == "nt" and platform.system().lower() == "windows" and machine in {"x86_64", "amd64"}:
        return "windows-x64"
    return ""


def _installed_runtime_root(platform_id: str) -> Path | None:
    if platform_id not in {"linux-x64", "linux-aarch64", "windows-x64"}:
        return None
    try:
        import bpy  # type: ignore

        from . import ADDON_PREFERENCES_ID
        from . import runtime_store

        storage_root = bpy.utils.extension_path_user(ADDON_PREFERENCES_ID, path="", create=False)
    except Exception:
        return None
    if not storage_root:
        return None
    current_root = runtime_store.paths(Path(storage_root), platform_id).current_root
    if runtime_store.read_install_record(current_root) is None:
        return None
    return current_root


def defaults(
    *,
    root: Path | None = None,
    ovrtx_port: str = DEFAULT_OVRTX_PORT,
    ovphysx_address: str = DEFAULT_OVPHYSX_ADDRESS,
    ovphysx_device: str = "cpu",
) -> BundledRuntimeDefaults:
    platform_id = current_platform_id()
    if root is None:
        root = _installed_runtime_root(platform_id)
    root = (root or addon_root()).expanduser().resolve()
    if platform_id not in {"linux-x64", "linux-aarch64", "windows-x64"}:
        return BundledRuntimeDefaults(root=root, platform_id=platform_id)

    native_dir = root / "native" / "client"
    runtime_dir = root / "runtime"
    executable_suffix = ".exe" if platform_id == "windows-x64" else ""
    worker_package_root = runtime_dir / "ovrtx-bridge-server"
    worker = worker_package_root / "bin" / f"ovrtx-bridge-server{executable_suffix}"
    ovphysx_bridge_runtime_root = runtime_dir / "ovphysx-bridge-server"
    ovphysx_server = ovphysx_bridge_runtime_root / "bin" / f"ovphysx_grpc_server{executable_suffix}"
    ovphysx_root = ovphysx_bridge_runtime_root / "private" / "ovphysx-runtime"
    ovruntime_root = ovphysx_root

    worker_command = ""
    if worker.is_file() and worker_package_root.is_dir():
        worker_command = serialize_command(
            [
                str(worker),
                "--address",
                "127.0.0.1",
                "--port",
                str(ovrtx_port),
                "--package-root",
                str(worker_package_root),
            ],
            platform_id=platform_id,
        )

    ovphysx_worker_command = ""
    if ovphysx_server.is_file():
        ovphysx_worker_command = serialize_command(
            [
                str(ovphysx_server),
                "--listen",
                str(ovphysx_address),
                "--device",
                str(ovphysx_device),
            ],
            platform_id=platform_id,
        )

    native_client_path = str(native_dir) if native_dir.is_dir() else ""
    return BundledRuntimeDefaults(
        root=root,
        platform_id=platform_id,
        worker_command=worker_command,
        native_client_path=native_client_path,
        ovphysx_server=str(ovphysx_server) if ovphysx_server.is_file() else "",
        ovphysx_worker_command=ovphysx_worker_command,
        ovphysx_native_client_path=native_client_path,
        ovphysx_root=str(ovphysx_root) if ovphysx_root.is_dir() else "",
        ovphysx_bridge_runtime_root=str(ovphysx_bridge_runtime_root) if ovphysx_bridge_runtime_root.is_dir() else "",
        ovruntime_root=str(ovruntime_root) if ovruntime_root.is_dir() else "",
    )


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


__all__ = [
    "BundledRuntimeDefaults",
    "DEFAULT_OVPHYSX_ADDRESS",
    "DEFAULT_OVRTX_PORT",
    "addon_root",
    "current_platform_id",
    "defaults",
    "parse_command",
    "serialize_command",
]
