# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained property rows exercise actual OVUI frame replacement and layout."""
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest


@pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires an OVUI graphics display")
def test_native_property_rows_refresh_and_reflow_without_rebuilding_unchanged_values():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "python/tests/retained_properties_layout_gui.py")],
        cwd=root, env=dict(os.environ, OVUIINSPECT_ENABLED="0"), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native retained property checks passed" in result.stdout
