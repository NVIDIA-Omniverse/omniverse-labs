# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver and external OVStream availability checks."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
from . import bundled_runtime
# Warp 1.18 wheels are CUDA 13.4 builds: CUDA 13 needs an R580 driver.
_MIN_NVIDIA_DRIVER = {"linux-x64": "580.65.06", "windows-x64": "580.88"}
_PROC_NVIDIA_VERSION = Path("/proc/driver/nvidia/version")
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

def check_addon_prerequisites(**_unused) -> list[PreflightCheck]:
    configured = bundled_runtime.public_sdk_python()
    python = Path(configured).expanduser() if configured else None
    if python is None:
        from . import public_sdk_install
        state = public_sdk_install.status()
        message = str(state.get("error") or state.get("message") or "Rendering setup is pending")
    else:
        message = "Ready; renderer and physics start when needed" if python.is_file() else "Rendering environment is unavailable; retry setup"
    worker = bundled_runtime.public_sdk_worker_script()
    return [_driver_version_check(),
        PreflightCheck("public_sdk_python", "Rendering setup", python is not None and python.is_file(), message),
        PreflightCheck("public_sdk_worker", "Rendering components", worker.is_file(),
            "Available" if worker.is_file() else "Incomplete add-on package; reinstall the ZIP")]
