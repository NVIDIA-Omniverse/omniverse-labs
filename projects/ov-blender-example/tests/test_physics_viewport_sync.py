# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native acceptance: pixels, Blender objects, physics and rewind agree."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_real_blender_physics_viewport_sync(tmp_path):
    runtime = os.environ.get("OVRTX_SYNC_RUNTIME_ROOT")
    if not runtime:
        pytest.skip("set OVRTX_SYNC_RUNTIME_ROOT to a materialized full runtime")
    blender = blender_executable()
    assert blender is not None, "set BLENDER_COMMAND to the matching Blender executable"
    script = Path(__file__).resolve().parents[1] / "scripts/run_physics_viewport_sync_probe.py"
    completed = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(script), "--", "--runtime-root", runtime, "--output", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
    )
    (tmp_path / "blender.log").write_text(completed.stdout, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed"
    checks = {item["label"]: item for item in result["checks"]}
    assert checks["rewound"]["timeline_reset"]
    assert checks["rewound"]["simulation_time_ns"] == 0
    assert checks["orbited"]["simulation_time_ns"] == 0
    assert checks["reentered_camera"]["simulation_time_ns"] == checks["settled"]["simulation_time_ns"]
    assert checks["reentered_camera"]["blender_z"] == pytest.approx(checks["settled"]["blender_z"], abs=1e-5)
    assert checks["replayed"]["blender_z"] == pytest.approx(checks["settled"]["blender_z"], abs=1e-5)
    assert all(item["max_pixel_error"] < 3 for item in checks.values())
