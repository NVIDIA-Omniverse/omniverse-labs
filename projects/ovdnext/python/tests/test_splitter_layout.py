# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native widget layout must apply resize before unrelated capture updates."""
import os
from pathlib import Path
import subprocess
import sys
from gui_runtime import GUI_AVAILABLE
import pytest


@pytest.mark.skipif(not GUI_AVAILABLE,reason="Requires an OVUI graphics display")
def test_both_sibling_orders_resize_and_persist_after_real_layout():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/"python/tests/splitter_layout_gui.py")],
        cwd=root,env=dict(os.environ,OVUIINSPECT_ENABLED="0"),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"cancelled": true' in result.stdout
