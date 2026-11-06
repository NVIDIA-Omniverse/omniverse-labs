# SPDX-License-Identifier: Apache-2.0
"""Run automatic duplicate identity repair coverage in real Blender."""
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_duplicated_physics_objects_are_reidentified(tmp_path):
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender duplicate identity coverage")
    fixture = Path(__file__).with_name("blender_physics_duplicate_identity.py")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(fixture), "--", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "PHYSICS_DUPLICATE_IDENTITY_OK" in result.stdout
