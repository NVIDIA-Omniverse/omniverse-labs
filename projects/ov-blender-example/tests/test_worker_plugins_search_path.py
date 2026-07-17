# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker plugins DLL search-path mitigation (runtime measurements: Windows
worker exits 0xC0000135 unless the staged ``plugins`` directory is on the
child process DLL search path)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovrtx_runtime_client import (  # noqa: E402
    apply_worker_runtime_environment,
    worker_plugins_search_path_from_worker_command,
    worker_runtime_environment_evidence,
)


def _worker_command(package_root: Path) -> str:
    return (
        f'"{package_root.parent / "bin" / "ovrtx-bridge-server.exe"}" '
        f'--address 127.0.0.1 --port 50051 --package-root "{package_root}"'
    )


def _package_root_with_plugins(tmp_path: Path) -> Path:
    package_root = tmp_path / "runtime" / "ovrtx-bridge-server"
    (package_root / "plugins").mkdir(parents=True)
    return package_root


def test_plugins_search_path_resolves_from_the_package_root(tmp_path: Path) -> None:
    package_root = _package_root_with_plugins(tmp_path)
    resolved = worker_plugins_search_path_from_worker_command(
        _worker_command(package_root)
    )
    assert resolved == str(package_root / "plugins")


def test_plugins_search_path_is_empty_without_a_plugins_directory(tmp_path: Path) -> None:
    package_root = tmp_path / "runtime" / "ovrtx-bridge-server"
    package_root.mkdir(parents=True)
    assert (
        worker_plugins_search_path_from_worker_command(_worker_command(package_root))
        == ""
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL search-path mitigation")
def test_apply_worker_runtime_environment_prepends_plugins_to_path(
    tmp_path: Path,
) -> None:
    package_root = _package_root_with_plugins(tmp_path)
    plugins = str(package_root / "plugins")
    env: dict[str, str] = {"PATH": r"C:\existing"}

    previous = apply_worker_runtime_environment(env, _worker_command(package_root))

    assert env["PATH"].split(os.pathsep)[0] == plugins
    assert r"C:\existing" in env["PATH"].split(os.pathsep)
    assert previous["PATH"] == r"C:\existing"
    # Re-applying is idempotent: no duplicate entry, no re-recorded previous.
    again = apply_worker_runtime_environment(env, _worker_command(package_root))
    assert env["PATH"].split(os.pathsep).count(plugins) == 1
    assert "PATH" not in again


def test_worker_environment_evidence_reports_the_plugins_path(tmp_path: Path) -> None:
    package_root = _package_root_with_plugins(tmp_path)
    evidence = worker_runtime_environment_evidence(_worker_command(package_root))
    assert evidence["worker_plugins_search_path"] == str(package_root / "plugins")
    assert evidence["worker_plugins_search_path_configured"] is True
