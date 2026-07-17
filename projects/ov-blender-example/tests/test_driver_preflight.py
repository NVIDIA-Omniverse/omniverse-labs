# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import bundled_runtime, preflight  # noqa: E402


def test_driver_preflight_blocks_below_floor(monkeypatch) -> None:
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")
    monkeypatch.setattr(preflight, "_detect_nvidia_driver_version", lambda: "570.158.00")

    check = preflight._driver_version_check()

    assert not check.ok
    assert "570.158.01" in check.message


def test_driver_preflight_passes_at_or_above_floor(monkeypatch) -> None:
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "windows-x64")
    monkeypatch.setattr(preflight, "_detect_nvidia_driver_version", lambda: "573.39")
    assert preflight._driver_version_check().ok
    monkeypatch.setattr(preflight, "_detect_nvidia_driver_version", lambda: "595.97")
    assert preflight._driver_version_check().ok


def test_driver_preflight_soft_passes_when_driver_is_undetected(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_detect_nvidia_driver_version", lambda: "")
    assert preflight._driver_version_check().ok


def test_driver_detection_reads_proc_version(tmp_path: Path, monkeypatch) -> None:
    proc = tmp_path / "version"
    proc.write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  777.66.55  Tue Jan 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "_PROC_NVIDIA_VERSION", proc)

    assert preflight._detect_nvidia_driver_version() == "777.66.55"
