# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent marker hit testing plus the actual GUI subprocess gate."""
from types import SimpleNamespace
from pathlib import Path
import os
import sys
import pytest
from gui_runtime import run_gui_process


def test_marker_input_cycles_overlaps_without_a_playhead_model():
    from ovd_ui.widgets.finding_track import FindingTrack
    selected = []
    track = FindingTrack.__new__(FindingTrack)
    track.widget = SimpleNamespace(computed_width=300, screen_position_x=10, visible=True)
    track.visual = SimpleNamespace(rebuild=lambda: None)
    track.on_select = selected.append
    track._key = None;track._selected = None
    markers = [SimpleNamespace(set_digest="a", finding_id="f", role=role, frame=2) for role in ("right", "left")]
    assert track.update((0, 4), (0, 4), markers)
    key = track._key
    assert not track.update((0, 4), (0, 4), markers)
    assert track._key == key
    track.press(160, 0, 0, 0);track.press(160, 0, 0, 0)
    assert [m.role for m in selected] == ["left", "right"]
    track.press(-500, 0, 0, 0)
    assert len(selected) == 2
    paints=[];track.visual.rebuild=lambda:paints.append(True)
    track.restyle()
    assert len(paints)==1
    track.update((0,4),(0,4),markers)
    assert len(paints)==1  # Styling does not invalidate the geometry cache.


GUI_AVAILABLE = bool(os.environ.get("DISPLAY")) or (sys.platform=="win32" and
    (os.environ.get("OVD_UI_HIDDEN_GUI_TESTS")=="1" or bool(os.environ.get("CI") and os.environ.get("GLCONTEXT_WIN_LIBGL"))))


@pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires Linux DISPLAY or owned Windows GUI opt-in/CI Mesa")
def test_actual_evidence_import_markers_roles_go_and_clips(tmp_path):
    root=Path(__file__).resolve().parents[2]
    environment=dict(os.environ,OVUIINSPECT_ENABLED="0",LOCALAPPDATA=str(tmp_path/"local"),APPDATA=str(tmp_path/"config"),
        XDG_DATA_HOME=str(tmp_path/"data"),XDG_CONFIG_HOME=str(tmp_path/"config"),XDG_CACHE_HOME=str(tmp_path/"cache"))
    if sys.platform=="win32":environment["OVD_UI_HIDDEN_GUI_TESTS"]="1"
    result=run_gui_process([sys.executable,"-B",str(root/"python/tests/evidence_gui.py"),"--verify"],
        cwd=tmp_path,env=environment,text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert "Evidence GUI gestures passed" in result.stdout
