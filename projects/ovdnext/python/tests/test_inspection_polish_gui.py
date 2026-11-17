# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys

import pytest
from gui_runtime import GUI_AVAILABLE, run_gui_process


@pytest.mark.skipif(not GUI_AVAILABLE, reason='Requires an OVUI graphics display')
def test_contact_and_compact_shape_inspection_in_real_application():
    root = Path(__file__).resolve().parents[2]
    result = run_gui_process([sys.executable, str(root / 'python/tests/inspection_polish_gui.py')],
        cwd=root, env=dict(os.environ, OVUIINSPECT_ENABLED='0'), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Inspection GUI passed' in result.stdout
