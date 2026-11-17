# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame mapping and playback policy at the timeline/model boundary."""
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import omni.ui as ui
from gui_runtime import GUI_AVAILABLE
import pytest

from ovd_ui.playback import FrameMode, Playback
from ovd_ui.widgets.timeline import FrameScale, SoftGlassTimeline


def timeline(model, *, first=0, last=100, width=420, origin=100):
    # Exercise production input methods without starting a graphics device.
    widget = SoftGlassTimeline.__new__(SoftGlassTimeline)
    widget.model, widget._min, widget._max = model, first, last
    widget._domain, widget._dragging = None, False
    widget.widget = SimpleNamespace(enabled=True, computed_width=width, screen_position_x=origin)
    widget.visual = SimpleNamespace(rebuild=lambda: None)
    return widget


@pytest.mark.parametrize("first,last,width", [(0, 2, 600), (10, 25, 71),
    (0, 100000, 920), (2**60, 2**60+1000, 820), (9, 9, 100)])
def test_integer_mapping_uses_inset_endpoints_and_roundtrips(first, last, width):
    scale = FrameScale(first, last, width)
    for frame in (first, (first+last)//2, last):
        assert scale.frame(scale.position(frame)) == frame
    assert scale.frame(-1000) == first
    assert scale.frame(width+1000) == last
    ticks = scale.ticks()
    assert all(first <= tick <= last for tick in ticks)
    assert all(type(tick) is int for tick in ticks)
    assert len(set(b-a for a, b in zip(ticks, ticks[1:]))) <= 1


def test_click_drag_clamps_clip_and_current_width_with_outside_release():
    model = ui.SimpleIntModel(50)
    ruler = timeline(model, first=20, last=80)
    ruler.set_domain(0, 100)
    ruler.press(100+210, 30, 0, 0)
    assert model.as_int == 50
    ruler.move(-1000, 30, 0)
    assert model.as_int == 20
    ruler.widget.computed_width = 820  # Resize during the same captured drag.
    ruler.move(100+610, 30, 0)
    assert model.as_int == 75
    ruler.move(2000, 30, 0)
    assert model.as_int == 80
    ruler.release(-1000, 30, 0, 0)
    assert not ruler._dragging and model.as_int == 80
    ruler.move(100, 30, 0)
    assert model.as_int == 80


def test_bounds_updates_do_not_change_published_model_and_invalid_clip_cannot_seek():
    model = ui.SimpleIntModel(15)
    ruler = timeline(model)
    changes = []
    model.add_value_changed_fn(lambda value: changes.append(value.as_int))
    ruler.min, ruler.max = 40, 70
    ruler.set_domain(0, 100)
    assert model.as_int == 15 and changes == []
    ruler.min = 120
    ruler.press(310, 20, 0, 0)
    assert model.as_int == 15
    ruler.set_domain()
    ruler.min, ruler.max = 20, 60
    assert ruler._scale().first == 20 and ruler._scale().last == 60
    for bad in (True, 1.5, float("nan"), "3"):
        with pytest.raises(TypeError): ruler.min = bad
    with pytest.raises(ValueError): ruler.set_domain(100, 10)


def test_zero_width_disabled_right_click_and_nonfinite_input_do_not_seek():
    model = ui.SimpleIntModel(50)
    ruler = timeline(model, width=0)
    ruler.press(100, 10, 0, 0)
    assert model.as_int == 50
    ruler.widget.computed_width = 420
    ruler.enabled = False
    ruler.press(100, 10, 0, 0)
    assert not ruler._dragging
    ruler.enabled = True
    ruler.press(100, 10, 1, 0)
    assert not ruler._dragging
    ruler.press(float("nan"), 10, 0, 0)
    assert model.as_int == 50


def test_application_parity_and_rejected_seek_are_authoritative():
    playback = Playback()
    playback.reset(101, ovd=True)
    playback.clip.start, playback.clip.stop = 20, 61
    model = ui.SimpleIntModel(playback.seek(20))
    ruler = timeline(model, first=20, last=60, width=820)
    ruler.set_domain(0, 100)
    requested = []
    suppress = False
    ready = True
    def application_changed(value):
        nonlocal suppress
        if suppress:
            return
        requested.append(value.as_int)
        actual = playback.seek(value.as_int) if ready else playback.frame
        suppress = True
        model.set_value(actual)
        suppress = False
    model.add_value_changed_fn(application_changed)
    x = 100 + ruler._scale().position(41)
    ruler.press(x, 20, 0, 0)
    assert requested == [41] and playback.frame == model.as_int == 42
    ruler.release(x, 20, 0, 0)
    assert requested == [41]  # Release never re-requests an unsnapped frame.
    playback.set_mode(FrameMode.PRE)
    model.set_value(playback.frame)
    ruler.press(100+ruler._scale().position(44), 20, 0, 0)
    assert playback.frame == model.as_int == 45
    ready = False  # An unpublished live packet leaves the old view selected.
    ruler.move(100+ruler._scale().position(57), 20, 0)
    assert playback.frame == model.as_int == 45


def test_drag_one_frame_clip_never_requests_outside_capture():
    model = ui.SimpleIntModel(7)
    ruler = timeline(model, first=7, last=7)
    for x in (-1000, 100, 310, 2000):
        ruler.press(x, 0, 0, 0)
        ruler.move(x+100, 0, 0)
        ruler.release(x, 0, 0, 0)
        assert model.as_int == 7


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a real OVUI graphics display")
def test_actual_ovui_layout_shrinks_and_expands_without_changing_the_model(tmp_path):
    root = Path(__file__).resolve().parents[2]
    caller_file = tmp_path / "caller-owned.txt"
    caller_file.write_text("unchanged")
    assert not (root / "imgui.ini").exists()
    environment = dict(os.environ, OVUIINSPECT_ENABLED="0")
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "python"), environment.get("PYTHONPATH", "")))
    result = subprocess.run([sys.executable, str(root / "python/tests/timeline_gui.py"), "--verify"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Timeline layout checks passed" in result.stdout, result.stdout + result.stderr
    assert caller_file.read_text() == "unchanged"
    assert not (tmp_path / "imgui.ini").exists() and not (root / "imgui.ini").exists()


@pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires an OVUI graphics display or hidden Windows opt-in")
def test_application_ruler_survives_frame_caption_and_digit_changes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, OVUIINSPECT_ENABLED="0",
        XDG_DATA_HOME=str(tmp_path / "data"), XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, str(root / "python/tests/transport_layout_gui.py")],
        cwd=root, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Application transport layout checks passed" in result.stdout
