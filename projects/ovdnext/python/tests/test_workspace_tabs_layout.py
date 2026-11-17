# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check real OVUI page ownership, clipping, scrolling and deferred replacement."""
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display or hidden Windows opt-in')
def test_bottom_workspace_owns_messages_and_restores_its_extent():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'python/tests/activity_panel_gui.py')],
        cwd=root,env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native bottom workspace checks passed' in result.stdout


@pytest.mark.skipif(not os.environ.get('DISPLAY'),reason='Requires an OVUI graphics display')
def test_workspace_tabs_own_hidden_results_and_close_only_generated_pages(tmp_path):
    root=Path(__file__).resolve().parents[2]
    caller_file=tmp_path/'caller-owned.txt';caller_file.write_text('unchanged')
    assert not (root/'imgui.ini').exists()
    result=subprocess.run([sys.executable,str(root/'python/tests/workspace_tabs_layout_gui.py')],
        cwd=tmp_path,env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native workspace tab checks passed' in result.stdout
    assert caller_file.read_text()=='unchanged'
    assert not (tmp_path/'imgui.ini').exists() and not (root/'imgui.ini').exists()


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display or hidden Windows opt-in')
def test_analysis_controls_survive_frames_without_retaining_stale_results():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'python/tests/analysis_controls_gui.py')],
        cwd=root,env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native analysis control checks passed' in result.stdout


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display or hidden Windows opt-in')
def test_profile_graphs_keep_recorded_identity_and_distinct_thread_values():
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'python/tests/profile_graph_identity_gui.py')],
        cwd=root,env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native profile graph identity checks passed' in result.stdout
