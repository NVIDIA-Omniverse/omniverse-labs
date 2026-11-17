# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout checks exercise the actual widget engine when a display is available."""
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest


@pytest.mark.skipif(not GUI_AVAILABLE,reason="Requires an OVUI graphics display")
def test_native_object_rows_recover_names_after_panel_resize():
    root=Path(__file__).resolve().parents[2]
    environment=dict(os.environ,OVUIINSPECT_ENABLED="0")
    result=subprocess.run([sys.executable,str(root / "python/tests/object_browser_layout_gui.py")],
        cwd=root,env=environment,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert "Native tree resize checks passed" in result.stdout,result.stdout+result.stderr
