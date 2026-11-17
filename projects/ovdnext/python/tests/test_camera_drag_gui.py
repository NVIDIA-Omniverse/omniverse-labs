# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera input regression through the application's real workspace widgets."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not (os.environ.get("DISPLAY") or os.environ.get("OMNIUI_HEADLESS") == "1"),
                    reason="Requires an off-screen OVUI backend or graphics display")
def test_camera_drag_retains_ownership_across_workspace_panels(tmp_path):
    script = Path(__file__).with_name("camera_drag_gui.py")
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=tmp_path,
        env=dict(os.environ, OVUIINSPECT_ENABLED="0"), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "camera drag ownership: dark, light passed" in result.stdout
    assert not (tmp_path / "imgui.ini").exists()
