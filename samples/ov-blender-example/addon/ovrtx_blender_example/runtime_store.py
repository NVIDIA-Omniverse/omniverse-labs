# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed runtime bundle storage for split extension releases."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

from .runtime_manifest import RuntimeManifest


INSTALL_RECORD_NAME = "installed-runtime.json"


def _filesystem_path(path: Path) -> str:
    absolute = os.path.abspath(str(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


@dataclass(frozen=True)
class RuntimeStorePaths:
    platform_root: Path
    current_root: Path
    staging_root: Path
    download_root: Path


@dataclass(frozen=True)
class RuntimeStatus:
    state: str
    current_root: Path
    message: str
    installed_manifest_sha256: str = ""


def paths(storage_root: Path, platform_id: str) -> RuntimeStorePaths:
    platform_root = storage_root.expanduser().resolve() / "runtimes" / platform_id
    return RuntimeStorePaths(
        platform_root=platform_root,
        current_root=platform_root / "current",
        staging_root=platform_root / ".current.staging",
        download_root=platform_root / ".downloads",
    )


def status(storage_root: Path, platform_id: str, manifest_sha256: str) -> RuntimeStatus:
    store_paths = paths(storage_root, platform_id)
    record = read_install_record(store_paths.current_root)
    if record is None:
        return RuntimeStatus("missing", store_paths.current_root, "Runtime is not installed.")
    installed_sha = str(record.get("manifest_sha256", "") or "")
    if installed_sha != manifest_sha256:
        return RuntimeStatus(
            "mismatch",
            store_paths.current_root,
            "Installed runtime does not match this extension.",
            installed_manifest_sha256=installed_sha,
        )
    return RuntimeStatus(
        "ready",
        store_paths.current_root,
        "Runtime is installed.",
        installed_manifest_sha256=installed_sha,
    )


def _missing_layout_entry(current_root: Path, manifest: RuntimeManifest) -> str | None:
    for component in manifest.components:
        for target in component.targets:
            target_path = current_root / target.target.rstrip("/")
            if not target_path.exists():
                return f"Installed runtime target is missing: {target_path.relative_to(current_root)}"
        for executable in component.executables:
            executable_path = current_root / executable
            if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
                return f"Installed runtime executable is missing or not executable: {executable}"
    return None


def verify(storage_root: Path, manifest: RuntimeManifest) -> RuntimeStatus:
    runtime_status = status(storage_root, manifest.platform, manifest.sha256)
    if runtime_status.state != "ready":
        return runtime_status
    current_root = runtime_status.current_root
    missing = _missing_layout_entry(current_root, manifest)
    if missing is not None:
        return RuntimeStatus(
            "broken",
            current_root,
            missing,
            installed_manifest_sha256=runtime_status.installed_manifest_sha256,
        )
    return RuntimeStatus(
        "ready",
        current_root,
        "Runtime is verified.",
        installed_manifest_sha256=runtime_status.installed_manifest_sha256,
    )


def read_install_record(current_root: Path) -> Mapping[str, Any] | None:
    record_path = current_root / INSTALL_RECORD_NAME
    if not record_path.is_file():
        return None
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def remove_runtime(storage_root: Path, platform_id: str) -> None:
    store_paths = paths(storage_root, platform_id)
    for root in (
        store_paths.current_root,
        store_paths.staging_root,
        store_paths.download_root,
    ):
        try:
            shutil.rmtree(_filesystem_path(root))
        except FileNotFoundError:
            pass


__all__ = [
    "INSTALL_RECORD_NAME",
    "RuntimeStatus",
    "RuntimeStorePaths",
    "paths",
    "read_install_record",
    "remove_runtime",
    "status",
    "verify",
]
