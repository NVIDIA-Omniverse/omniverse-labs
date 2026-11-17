# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display-dependent delivery of command modifiers and real dialog keypad keys."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='Requires an OVUI graphics display')
def test_native_exact_modifiers_and_keypad_dialog_delivery(tmp_path):
    root=Path(__file__).resolve().parents[2]
    caller_file=tmp_path/'caller-owned.txt';caller_file.write_text('unchanged')
    assert not (root/'imgui.ini').exists()
    result=subprocess.run([sys.executable,str(root/'python/tests/exact_shortcuts_gui.py')],
        cwd=tmp_path,env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native exact modifier and keypad delivery passed' in result.stdout
    assert caller_file.read_text()=='unchanged'
    assert not (tmp_path/'imgui.ini').exists() and not (root/'imgui.ini').exists()
