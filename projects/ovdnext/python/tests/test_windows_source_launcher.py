# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Windows import preflight: no desktop or listener is started."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "ci/run_ovui_windows.py"
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader")


@pytest.fixture
def runtime():
    native = os.environ.get("OVDNEXT_TEST_NATIVE_ROOT")
    if not native:
        pytest.skip("Set OVDNEXT_TEST_NATIVE_ROOT for the real built runtime")
    return ["--native-root", native]


def invoke(arguments, tmp_path):
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"):
        environment.pop(key, None)
    environment['PATH'] = os.pathsep.join((str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']))
    return subprocess.run([sys.executable, str(LAUNCHER), *arguments],
                          cwd=tmp_path, env=environment, capture_output=True,
                          text=True, timeout=30)


def test_import_preflight_resolves_native_and_cuda_without_qt(runtime, tmp_path):
    result = invoke([*runtime, "--check"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["application_imported"] is True
    assert report["ovui_version"] == "0.2.0"
    assert Path(report["native_module"]).is_file()
    assert not (tmp_path / "imgui.ini").exists()


def test_missing_cuda_dependency_is_reported_before_import(runtime, tmp_path):
    result = invoke([*runtime, "--cuda-bin", str(tmp_path), "--check"], tmp_path)
    assert result.returncode != 0
    assert "cudart64_12.dll" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_native_build_is_reported_before_import(runtime, tmp_path):
    result = invoke([*runtime, "--native-root", str(tmp_path), "--check"], tmp_path)
    assert result.returncode != 0
    assert "_pvd_native" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_does_not_require_native_dependencies(tmp_path):
    result = invoke(["--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--native-root" in result.stdout
