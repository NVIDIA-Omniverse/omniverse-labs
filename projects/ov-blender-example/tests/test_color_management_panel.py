# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color Management (view-settings) controls in the OVRTX Render panel.

Spec render-quality-color-controls, task02-03: the OVRTX Render panel exposes
Blender's own ``scene.view_settings`` display transform — View Transform, Look,
Exposure, Gamma — plus the presentation-mode selector (task02-01), by drawing
Blender's properties directly with no duplicated add-on properties and no value
mirroring.

Two lanes:

* Plain (non-Blender) source-level checks on the panel ``draw`` method: it
  props the presentation-mode selector and the four ``scene.view_settings``
  controls against the view-settings data-block (not against the add-on
  settings group), labels the section as a Blender-owned display transform,
  and wraps the view-settings body in a column whose ``enabled`` flag is the
  resolved-mode gating hook.
* A headless Blender draw smoke test that registers the add-on properties and
  panel, drives the real ``draw`` method with a recording layout, and asserts
  the four view-settings controls are drawn against ``scene.view_settings``
  (the same object Blender's stock Color Management panel edits — no add-on
  copies). Skips when no Blender executable is available.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import color_presentation  # noqa: E402


UI_SOURCE = (ROOT / "addon" / "ovrtx_blender_example" / "ui.py").read_text(
    encoding="utf-8"
)

# The four Blender ``scene.view_settings`` controls, in the stock
# RENDER_PT_color_management draw order (View Transform, Look, then Exposure,
# Gamma).
VIEW_SETTINGS_PROPS_IN_ORDER = (
    "view_transform",
    "look",
    "exposure",
    "gamma",
)


def _draw_source() -> str:
    """Source text of the ``OVRTXEXAMPLE_PT_render_settings.draw`` method."""

    marker = "def draw(self, context: Any) -> None:"
    start = UI_SOURCE.index(marker)
    end = UI_SOURCE.index("\nelse:", start)
    return UI_SOURCE[start:end]


# --- Source-level checks (plain lane) --------------------------------------


def test_draw_shows_all_four_view_settings_controls_in_order() -> None:
    draw = _draw_source()
    positions = []
    for name in VIEW_SETTINGS_PROPS_IN_ORDER:
        needle = f'view, "{name}"'
        assert needle in draw, name
        positions.append(draw.index(needle))
    # View Transform, Look, Exposure, Gamma in the stock layout order.
    assert positions == sorted(positions)


def test_view_settings_drawn_against_blender_view_settings_not_addon_copies() -> None:
    # No add-on-owned copies: the four controls draw the ``view`` data-block
    # (``context.scene.view_settings``), never the add-on ``settings`` group.
    draw = _draw_source()
    assert "context.scene.view_settings" in draw
    for name in VIEW_SETTINGS_PROPS_IN_ORDER:
        assert f'settings, "{name}"' not in draw, name


def test_section_labeled_as_blender_display_transform() -> None:
    # The section is labeled as a Blender-owned display transform, not an
    # OVRTX post-grade and not scene compensation.
    draw = _draw_source()
    assert 'text="Color Management"' in draw
    assert "Blender display transform" in draw


def test_view_settings_body_wrapped_in_enabled_gated_column() -> None:
    # The gating hook: the view-settings body lives in a column whose
    # ``enabled`` flag is driven by the resolved presentation mode. task03-01
    # keys this off ``presentation_from_scene`` (which folds in the env
    # override, the UI enum, and fail-closed handling) via
    # ``color_control_gating`` — never the raw property value.
    draw = _draw_source()
    assert "presentation_from_scene" in draw
    assert "color_control_gating" in draw
    assert ".enabled = gating[" in draw


# --- Headless Blender draw smoke test --------------------------------------
_DRIVER = """
import json
import sys
import traceback

result = {"errors": [], "steps": []}
output_path = sys.argv[sys.argv.index("--") + 1]

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import properties, ui

    bpy.ops.wm.read_homefile(use_empty=True)
    properties.register()
    ui.register()
    result["steps"].append("registered")

    panel_cls = bpy.types.OVRTXEXAMPLE_PT_render_settings
    result["panel_registered"] = panel_cls is not None

    # Stub the session-status projection so the draw smoke test does not depend
    # on live engine/native state; the panel draws a static session box from it.
    ui.viewport_session_status = lambda *a, **k: {
        "status": "stopped",
        "label": "Stopped",
        "hint": "",
        "logs": {"status": "stdout", "log_dir": ""},
    }

    scene = bpy.context.scene
    view_settings = scene.view_settings

    class RecordingLayout:
        def __init__(self, rec):
            self._rec = rec
        def prop(self, data, name, **kw):
            # Record whether the drawn property targets Blender's own
            # view_settings data-block. bpy_struct ``==`` compares the
            # underlying RNA pointer, so a re-access of scene.view_settings
            # (a distinct Python wrapper) still matches — while the add-on
            # settings group does not, proving there is no add-on-owned copy.
            self._rec.append({
                "name": name,
                "on_view_settings": bool(data == view_settings),
            })
            return self
        def operator(self, *a, **kw):
            return self
        def label(self, *a, **kw):
            return self
        def row(self, **kw):
            return self
        def box(self):
            return self
        def column(self, **kw):
            return self

    class FakePanel:
        pass

    props_drawn = []
    panel = FakePanel()
    panel.layout = RecordingLayout(props_drawn)
    panel_cls.draw(panel, bpy.context)
    result["props_drawn"] = props_drawn
    result["steps"].append("drawn")

    ui.unregister()
    properties.unregister()
    result["steps"].append("unregistered")
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_panel_draws_view_settings_controls_headless(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless panel draw test")

    driver = tmp_path / "color_mgmt_panel_draw_driver.py"
    driver.write_text(
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon"))),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    completed = subprocess.run(
        (
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(driver),
            "--",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["errors"] == []
    assert result["steps"] == ["registered", "drawn", "unregistered"]
    assert result["panel_registered"] is True

    props = result["props_drawn"]
    by_name = {entry["name"]: entry for entry in props}

    # The presentation-mode selector is drawn (against the add-on settings).
    assert "color_presentation_mode" in by_name
    assert by_name["color_presentation_mode"]["on_view_settings"] is False

    # Every view-settings control is drawn against Blender's own
    # scene.view_settings data-block (no add-on-owned copies).
    drawn_view_props = [
        entry["name"] for entry in props if entry["on_view_settings"]
    ]
    for name in VIEW_SETTINGS_PROPS_IN_ORDER:
        assert name in by_name, name
        assert by_name[name]["on_view_settings"] is True, name
    assert drawn_view_props == list(VIEW_SETTINGS_PROPS_IN_ORDER)
