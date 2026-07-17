# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovrtx_runtime_client import _restore_environment, _sanitize_worker_environment  # noqa: E402


def test_sanitize_worker_environment_strips_snap_then_restores(monkeypatch) -> None:
    original = os.pathsep.join(["/snap/blender/123/lib", "/usr/lib"])
    monkeypatch.setenv("LD_LIBRARY_PATH", original)

    previous = _sanitize_worker_environment()

    assert os.environ["LD_LIBRARY_PATH"] == "/usr/lib"
    _restore_environment(previous)
    assert os.environ["LD_LIBRARY_PATH"] == original


def test_sanitize_worker_environment_is_noop_without_snap(monkeypatch) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")

    assert _sanitize_worker_environment() == {}
    assert os.environ["LD_LIBRARY_PATH"] == "/usr/lib"


def test_sanitize_worker_environment_is_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    assert _sanitize_worker_environment() == {}
    assert "LD_LIBRARY_PATH" not in os.environ
