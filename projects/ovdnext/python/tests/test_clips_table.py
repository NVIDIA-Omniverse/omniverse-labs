# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native clip table checks run separately from actual Inspector input flows."""
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display')
def test_native_clip_table_rows_scoped_menus_and_retained_layout():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'python/tests/clips_table_gui.py')],cwd=root,
        env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native clip table checks passed' in result.stdout
