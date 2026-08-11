# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import bundled_runtime  # noqa: E402
from ovrtx_blender_example.command_line import split_command  # noqa: E402


def _bundle_root(tmp_path: Path) -> Path:
    root = tmp_path / "addon"
    (root / "native" / "client").mkdir(parents=True)
    (root / "runtime" / "ovrtx-bridge-server").mkdir(parents=True)
    (root / "runtime" / "ovphysx-bridge-server" / "private" / "ovphysx-runtime").mkdir(parents=True)
    (root / "runtime" / "ovrtx-bridge-server" / "bin").mkdir()
    (root / "runtime" / "ovphysx-bridge-server" / "bin").mkdir()
    (root / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "runtime" / "ovphysx-bridge-server" / "bin" / "ovphysx_grpc_server").write_text("#!/bin/sh\n", encoding="utf-8")
    return root


def test_bundled_runtime_defaults_resolve_linux_x64_layout(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    defaults = bundled_runtime.defaults(root=root, ovrtx_port="50151", ovphysx_address="127.0.0.1:50194")

    assert defaults.platform_id == "linux-x64"
    assert defaults.native_client_path == str(root / "native" / "client")
    assert defaults.ovphysx_native_client_path == str(root / "native" / "client")
    assert defaults.ovphysx_root == str(root / "runtime" / "ovphysx-bridge-server" / "private" / "ovphysx-runtime")
    assert defaults.ovphysx_bridge_runtime_root == str(root / "runtime" / "ovphysx-bridge-server")
    assert defaults.ovruntime_root == str(root / "runtime" / "ovphysx-bridge-server" / "private" / "ovphysx-runtime")


def test_bundled_runtime_defaults_resolve_linux_aarch64_layout(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-aarch64")

    defaults = bundled_runtime.defaults(root=root)

    assert defaults.native_client_path == str(root / "native" / "client")
    assert str(root / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server") in defaults.worker_command


def test_bundled_runtime_defaults_prefer_installed_runtime(tmp_path: Path, monkeypatch) -> None:
    installed = _bundle_root(tmp_path / "installed")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")
    monkeypatch.setattr(bundled_runtime, "_installed_runtime_root", lambda _platform_id: installed)

    defaults = bundled_runtime.defaults()

    assert str(installed / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server") in defaults.worker_command
    assert defaults.native_client_path == str(installed / "native" / "client")


def test_bundled_runtime_defaults_fall_back_to_addon_root(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path / "addon")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")
    monkeypatch.setattr(bundled_runtime, "_installed_runtime_root", lambda _platform_id: None)
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: root)

    defaults = bundled_runtime.defaults()

    assert str(root / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server") in defaults.worker_command


def test_bundled_runtime_explicit_root_wins_over_installed_runtime(tmp_path: Path, monkeypatch) -> None:
    installed = _bundle_root(tmp_path / "installed")
    explicit = _bundle_root(tmp_path / "explicit")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")
    monkeypatch.setattr(bundled_runtime, "_installed_runtime_root", lambda _platform_id: installed)

    defaults = bundled_runtime.defaults(root=explicit)

    assert str(explicit / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server") in defaults.worker_command


def test_bundled_runtime_defaults_resolve_windows_x64_layout(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path / "Blender Foundation")
    (root / "runtime" / "ovrtx-bridge-server" / "bin" / "ovrtx-bridge-server.exe").write_text("", encoding="utf-8")
    (root / "runtime" / "ovphysx-bridge-server" / "bin" / "ovphysx_grpc_server.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "windows-x64")

    defaults = bundled_runtime.defaults(root=root, ovrtx_port="50151", ovphysx_address="127.0.0.1:50194")

    assert defaults.platform_id == "windows-x64"
    assert defaults.native_client_path == str(root / "native" / "client")
    assert defaults.ovphysx_native_client_path == str(root / "native" / "client")


def test_split_windows_command_round_trips_list2cmdline_paths_with_spaces() -> None:
    arguments = [
        r"C:\Program Files\OVRTX\ovrtx-bridge-server.exe",
        "--package-root",
        r"C:\ProgramData\Blender Foundation\runtime\ovrtx-bridge-server",
    ]

    assert split_command(subprocess.list2cmdline(arguments), windows=True) == arguments


def test_addon_root_can_use_release_zip_extract_for_validation(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_BUNDLED_ADDON_ROOT", str(root))

    assert bundled_runtime.addon_root() == root.resolve()


def test_bundled_runtime_ignores_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    root = _bundle_root(tmp_path)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-arm64")

    defaults = bundled_runtime.defaults(root=root)

    assert defaults.platform_id == "linux-arm64"
    assert defaults.native_client_path == ""
