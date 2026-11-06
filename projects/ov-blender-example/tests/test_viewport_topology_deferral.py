# SPDX-License-Identifier: Apache-2.0
"""Run view_draw topology deferral coverage in a real Blender window."""
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


def test_topology_edits_never_reconcile_in_view_draw():
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender viewport deferral coverage")
    fixture = Path(__file__).with_name("blender_viewport_topology_deferral.py")
    # view_draw only runs for a drawn window, so this needs a GUI session.
    result = subprocess.run(
        [str(blender), "--factory-startup", "--python-exit-code", "17", "--python", str(fixture)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "VIEWPORT_TOPOLOGY_DEFERRAL_OK" in result.stdout
