# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A loaded model is insufficient: recorded curves and table text must rasterize."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from gui_runtime import GUI_AVAILABLE, run_gui_process

pytestmark = pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires an OVUI display or hidden Windows opt-in")
ROOT = Path(__file__).resolve().parents[2]


def render_panel(tmp_path, panel, theme, height=800):
    output = tmp_path / "rendered"
    result = run_gui_process([sys.executable, str(ROOT / "python/tests/populated_panels_gui.py"),
        "--panel", panel, "--theme", theme, "--height", str(height), "--output", str(output)],
        cwd=tmp_path, env=dict(os.environ, OVUIINSPECT_ENABLED="0", LOCALAPPDATA=str(tmp_path / "prefs"),
                              XDG_DATA_HOME=str(tmp_path / "prefs")),
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "panel.json").read_text())
    if sys.platform == "win32" and os.environ.get("GLCONTEXT_WIN_LIBGL"):
        assert "llvmpipe" in report["graphics"]["GL_RENDERER"].lower(), report["graphics"]
    return np.asarray(Image.open(output / "panel.png").convert("RGB")).astype(int), report


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_recorded_graph_curves_are_visible_in_the_canvas(tmp_path, theme):
    pixels, report = render_panel(tmp_path, "graph", theme)
    x, y, width, height = report["canvas"]
    assert width > 100 and height > 40, report
    assert report["frames"] == 249
    red, green, blue = np.moveaxis(pixels[y+2:y+height-2, x+2:x+width-2], 2, 0)
    # The two measured curves span almost all frames. Legend text, axes,
    # gridlines, a nonempty model or a lone point cannot satisfy this check.
    for mask in ((green > red + 20) & (green > blue + 20),
                 (red > green + 20) & (green > blue + 20)):
        assert np.count_nonzero(np.any(mask, axis=0)) > width * .7, "Recorded curve missing from screenshot"


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("height", [800, 600])
def test_populated_recordings_show_a_readable_row(tmp_path, theme, height):
    pixels, report = render_panel(tmp_path, "recordings", theme, height)
    assert [row[3] for row in report["rows"]] == ["Latest iteration.ovd", "Baseline.ovd"]
    x, y, width, outer_height = report["outer"]
    _, header_y, _, header_height = report["header"]
    row_y = header_y + header_height
    visible_height = min(28, y + outer_height - row_y)
    assert visible_height >= 20, f"Populated recordings have no readable row: {report}"
    # Inset excludes borders; the first timestamp must contain real glyphs.
    cell = pixels[row_y+3:row_y+visible_height-3, x+5:x+150]
    ink = np.all(cell > 150, axis=2) if theme == "dark" else np.all(cell < 120, axis=2)
    assert np.count_nonzero(ink) > 30, "Recording row text was not rendered"


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_compact_recordings_remain_reachable_after_scroll_and_resize(tmp_path, theme):
    result = run_gui_process([sys.executable, str(ROOT / "python/tests/compact_recordings_gui.py"),
                              "--theme", theme], cwd=tmp_path,
                             env=dict(os.environ, OVUIINSPECT_ENABLED="0"),
                             capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Compact rows, scrolling, selection and resize checks passed" in result.stdout
