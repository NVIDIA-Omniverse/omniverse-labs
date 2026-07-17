# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prerequisite checks for the OVRTX Blender add-on."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from . import bundled_runtime
from .command_line import split_command


_MIN_NVIDIA_DRIVER = {"linux-x64": "570.158.01", "windows-x64": "573.39"}
_PROC_NVIDIA_VERSION = Path("/proc/driver/nvidia/version")
_PACKAGE_ROOT_MARKERS = ("plugins", "mdl", "usd_plugins")

@dataclass(frozen=True)
class PreflightCheck:
    key: str
    label: str
    ok: bool
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "ok": self.ok,
            "message": self.message,
        }


def check_addon_prerequisites(
    *,
    worker_command: str = "",
    native_client_path: str = "",
    native_client_module: str = bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE,
    runtime_root: Path | None = None,
) -> list[PreflightCheck]:
    """Return actionable status for external prerequisites not bundled in the zip.

    The active Blender scene is the render input, so no user-selected scene
    path is required or checked here.
    """

    bundle = bundled_runtime.defaults(root=runtime_root) if runtime_root is not None else bundled_runtime.defaults()
    if not worker_command.strip():
        worker_command = bundle.worker_command
    if not native_client_path.strip():
        native_client_path = bundle.native_client_path

    checks = [
        _driver_version_check(),
        _worker_command_check(worker_command),
        _worker_package_root_check(worker_command),
        _native_client_path_check(native_client_path),
        _module_check(native_client_module, native_client_path),
    ]
    return checks


def preflight_summary(checks: list[PreflightCheck]) -> dict[str, object]:
    blockers = [check.as_dict() for check in checks if not check.ok]
    return {
        "status": "pass" if not blockers else "blocked",
        "checks": [check.as_dict() for check in checks],
        "blockers": blockers,
    }


def _detect_nvidia_driver_version() -> str:
    try:
        if _PROC_NVIDIA_VERSION.is_file():
            match = re.search(
                r"Kernel Module\s+([0-9][0-9.]*)",
                _PROC_NVIDIA_VERSION.read_text(encoding="utf-8", errors="replace"),
            )
            if match:
                return match.group(1)
    except OSError:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return ""
    kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "timeout": 10,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def _driver_version_check() -> PreflightCheck:
    detected = _detect_nvidia_driver_version()
    if not detected:
        return PreflightCheck("nvidia_driver", "NVIDIA driver", True, "No NVIDIA driver detected; skipping check.")
    minimum = _MIN_NVIDIA_DRIVER.get(bundled_runtime.current_platform_id(), "")
    if not minimum:
        return PreflightCheck("nvidia_driver", "NVIDIA driver", True, detected)
    # Platform-wide floor; add GPU-generation detection only when the add-on has reliable GPU facts.
    if _version_key(detected) < _version_key(minimum):
        return PreflightCheck(
            "nvidia_driver",
            "NVIDIA driver",
            False,
            f"Driver {detected} is older than the supported minimum {minimum}.",
        )
    return PreflightCheck("nvidia_driver", "NVIDIA driver", True, detected)
def _worker_command_check(value: str) -> PreflightCheck:
    if not value.strip():
        return PreflightCheck(
            "worker_command",
            "OVRTX worker",
            False,
            "Set the ovrtx-bridge-server command.",
        )
    return PreflightCheck("worker_command", "OVRTX worker", True, "configured")


def _worker_package_root_check(value: str) -> PreflightCheck:
    parts, error = _worker_command_parts(value)
    if error:
        return PreflightCheck(
            "worker_package_root",
            "OVRTX package root",
            False,
            "Set the worker command with --package-root.",
        )
    package_root = _option_value(parts, "--package-root")
    if not package_root:
        return PreflightCheck(
            "worker_package_root",
            "OVRTX package root",
            False,
            "Worker command must include --package-root.",
        )
    path = Path(package_root).expanduser()
    if not path.is_dir():
        return PreflightCheck(
            "worker_package_root",
            "OVRTX package root",
            False,
            f"Missing directory: {path}",
        )
    missing = [marker for marker in _PACKAGE_ROOT_MARKERS if not (path / marker).is_dir()]
    if missing:
        return PreflightCheck(
            "worker_package_root",
            "OVRTX package root",
            False,
            f"Not an OVRTX package root (missing {', '.join(missing)}): {path}",
        )
    return PreflightCheck("worker_package_root", "OVRTX package root", True, str(path))


def _native_client_path_check(value: str) -> PreflightCheck:
    if not value.strip():
        return PreflightCheck(
            "native_client_path",
            "Native client path",
            False,
            "Set the built native client output path.",
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        return PreflightCheck(
            "native_client_path",
            "Native client path",
            False,
            f"Missing directory: {path}",
        )
    return PreflightCheck("native_client_path", "Native client path", True, str(path))


def _module_check(value: str, native_client_path: str = "") -> PreflightCheck:
    module = value.strip()
    if not module:
        return PreflightCheck(
            "native_client_module",
            "Native client",
            False,
            "Set the native client module name.",
        )
    extra_paths = [str(Path(native_client_path).expanduser())] if native_client_path.strip() else []
    try:
        native_module = _import_module(module, extra_paths)
    except (ImportError, AttributeError, ValueError) as exc:
        return PreflightCheck("native_client_module", "Native client", False, f"Import check failed: {exc}")
    if native_module is None:
        return PreflightCheck("native_client_module", "Native client", False, f"Module not importable: {module}")
    try:
        from .ovrtx_runtime_client import _bind_render_native_client

        _bind_render_native_client(native_module, native_module.Client)
    except Exception as exc:
        return PreflightCheck("native_client_module", "Native client", False, f"Native client surface invalid: {exc}")
    return PreflightCheck("native_client_module", "Native client", True, module)


def ensure_native_client_path(value: str) -> None:
    path = str(Path(value).expanduser())
    if value.strip() and Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def _import_module(module: str, extra_paths: list[str]) -> object | None:
    search_paths = [path for path in extra_paths if path and Path(path).is_dir()]
    if not search_paths:
        if importlib.util.find_spec(module) is None:
            return None
        return importlib.import_module(module)
    original = list(sys.path)
    try:
        for path in reversed(search_paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        if importlib.util.find_spec(module) is None:
            return None
        return importlib.import_module(module)
    finally:
        sys.path[:] = original


def _worker_command_parts(value: str, *, windows: bool | None = None) -> tuple[list[str], str]:
    if not value.strip():
        return [], "Set the ovrtx-bridge-server command."
    try:
        parts = split_command(value, windows=windows)
    except ValueError as exc:
        return [], f"Invalid command: {exc}"
    if not parts:
        return [], "Set the ovrtx-bridge-server command."
    return parts, ""


def _option_value(parts: list[str], name: str) -> str:
    for index, part in enumerate(parts):
        if part == name and index + 1 < len(parts):
            return parts[index + 1]
        prefix = f"{name}="
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return ""


__all__ = [
    "PreflightCheck",
    "check_addon_prerequisites",
    "ensure_native_client_path",
    "preflight_summary",
]
