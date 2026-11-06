# SPDX-License-Identifier: Apache-2.0
"""Run restricted-context generation coverage in a real Blender window."""
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_initial_generation_does_not_write_ids_in_render_callbacks():
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender restricted-generation coverage")
    fixture = Path(__file__).with_name("blender_physics_restricted_generation.py")
    result = subprocess.run(
        # ID writes are only disallowed while drawing, so this needs a window.
        [str(blender), "--factory-startup", "--python-exit-code", "17",
         "--python", str(fixture)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert "PHYSICS_RESTRICTED_GENERATION_OK" in result.stdout
