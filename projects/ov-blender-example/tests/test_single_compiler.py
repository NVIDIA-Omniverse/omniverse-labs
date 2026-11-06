# SPDX-License-Identifier: Apache-2.0
"""Run single-compiler scale, identity and visibility coverage in real Blender."""
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_owner_compiler_handles_scale_identity_and_visibility(tmp_path):
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender single-compiler coverage")
    fixture = Path(__file__).with_name("blender_single_compiler.py")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(fixture), "--", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "SINGLE_COMPILER_PASS" in result.stdout
