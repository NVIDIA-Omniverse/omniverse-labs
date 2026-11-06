# SPDX-License-Identifier: Apache-2.0
"""Run restore/evaluation regression coverage in real Blender."""
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_restored_physics_transforms_do_not_queue_structural_edits():
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender restore coverage")
    fixture = Path(__file__).with_name("blender_physics_restore_snapshot.py")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(fixture)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert "PHYSICS_RESTORE_SNAPSHOT_OK" in result.stdout
