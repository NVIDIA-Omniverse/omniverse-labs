# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Smoke tests for the public Python sensor examples."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PUBLIC_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SENSOR_EXAMPLES = PUBLIC_ROOT / "examples" / "python" / "sensors"
RENDERING_ROOT = PUBLIC_ROOT.parents[1]
LOCAL_OVRTX_BINARY_DIR = RENDERING_ROOT / "_build" / "linux-x86_64" / "release"


def run_python_example(example_name: str, tmp_path: Path) -> str:
    """Run a public Python sensor example headless and return combined output."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to run public Python examples")

    example_dir = PYTHON_SENSOR_EXAMPLES / example_name
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / f"{example_name}-venv")

    if LOCAL_OVRTX_BINARY_DIR.exists():
        existing_library_path = env.get("LD_LIBRARY_PATH")
        if existing_library_path:
            env["LD_LIBRARY_PATH"] = f"{LOCAL_OVRTX_BINARY_DIR}:{existing_library_path}"
        else:
            env["LD_LIBRARY_PATH"] = str(LOCAL_OVRTX_BINARY_DIR)

    result = subprocess.run(
        [uv, "run", "--frozen", "main.py", "--no-rr"],
        cwd=example_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def test_python_lidar_example_reports_valid_pointcloud(tmp_path):
    """Run the lidar example and sanity-check the printed PointCloud summary."""
    output = run_python_example("lidar", tmp_path)

    match = re.search(
        r"valid points=(\d+), mean intensity=([-+0-9.eE]+), "
        r"max time offset=(\d+) ns",
        output,
    )
    assert match, output

    valid_points = int(match.group(1))
    mean_intensity = float(match.group(2))
    max_time_offset_ns = int(match.group(3))

    assert valid_points > 1000
    assert math.isfinite(mean_intensity)
    assert mean_intensity >= 0.0
    assert max_time_offset_ns > 0


def test_python_radar_example_reports_moving_detections(tmp_path):
    """Run the radar example and sanity-check per-step radial velocity output."""
    output = run_python_example("radar", tmp_path)

    step_matches = re.findall(
        r"step (\d+): valid points=(\d+), "
        r"radial velocity min/max=\[([-+0-9.eE]+), ([-+0-9.eE]+)\] m/s",
        output,
    )
    assert len(step_matches) == 10, output

    max_abs_radial_velocity = 0.0
    for step, valid_points, min_velocity, max_velocity in step_matches:
        assert 1 <= int(step) <= 10
        assert int(valid_points) > 0

        min_velocity = float(min_velocity)
        max_velocity = float(max_velocity)
        assert math.isfinite(min_velocity)
        assert math.isfinite(max_velocity)
        max_abs_radial_velocity = max(
            max_abs_radial_velocity,
            abs(min_velocity),
            abs(max_velocity),
        )

    assert max_abs_radial_velocity > 0.1
