# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTPT quality controls in the OVRTX Render properties panel.

Spec render-quality-color-controls, task01-02: the OVRTX Render panel draws the
four documented RTPT quality controls (Max Bounces, Max Specular and
Transmission Bounces, Max Volume Bounces, Firefly Filter) as one contiguous
quality section directly after the Minimum/Maximum Samples rows, and shows no
render-mode selector or dead path-tracer-only control.

Two lanes:

* Plain (non-Blender) source-level checks on the panel ``draw`` method: it
  props exactly the four documented RTPT scene properties, in order, and
  references neither a render-mode attribute nor a path-tracer-only attribute
  namespace.
* A headless Blender draw smoke test that registers the add-on properties and
  panel, drives the real ``draw`` method with a recording layout, and asserts
  the emitted controls include the four quality properties right after the
  samples rows. Skips when no Blender executable is available.
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

from ovrtx_blender_example import properties  # noqa: E402


UI_SOURCE = (ROOT / "addon" / "ovrtx_blender_example" / "ui.py").read_text(
    encoding="utf-8"
)

# The four documented RTPT scene-property names, in the documented UI order.
RTPT_PROPS_IN_ORDER = (
    "rtpt_max_bounces",
    "rtpt_max_specular_and_transmission_bounces",
    "rtpt_max_volume_bounces",
    "rtpt_firefly_filter_enabled",
)


def _draw_source() -> str:
    """Source text of the ``OVRTXEXAMPLE_PT_render_settings.draw`` method.

    Sliced from the module source so the checks run in the plain pytest lane
    (where ``bpy`` is unavailable and the panel class collapses to ``None``).
    """

    marker = "def draw(self, context: Any) -> None:"
    start = UI_SOURCE.index(marker)
    # The draw method is the last member of the panel class, immediately
    # followed by the module-level ``else:`` that nulls the classes.
    end = UI_SOURCE.index("\nelse:", start)
    return UI_SOURCE[start:end]


# --- Source-level checks (plain lane) --------------------------------------


def test_documented_props_match_the_scene_property_group() -> None:
    # The panel draws exactly the four properties owned by task01-01's mapping;
    # this ties the UI order list to the single source of truth.
    assert set(RTPT_PROPS_IN_ORDER) == set(properties.RTPT_RENDER_SETTINGS)


def test_draw_props_all_four_rtpt_controls() -> None:
    draw = _draw_source()
    for name in RTPT_PROPS_IN_ORDER:
        assert f'settings, "{name}"' in draw, name


def test_draw_orders_quality_controls_after_samples() -> None:
    draw = _draw_source()
    positions = [draw.index(f'"{name}"') for name in ("min_samples", "max_samples", *RTPT_PROPS_IN_ORDER)]
    # min/max samples first, then the four RTPT controls in documented order.
    assert positions == sorted(positions)


def test_draw_shows_no_render_mode_selector() -> None:
    # Negative assertion: the panel never props a render-mode attribute.
    assert "rendermode" not in _draw_source().lower()


def test_draw_shows_no_path_tracer_only_control() -> None:
    # Negative assertion: no PT-only (``omni:rtx:pt:``) control is drawn.
    assert "omni:rtx:pt:" not in _draw_source()


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

    class RecordingLayout:
        def __init__(self, rec):
            self._rec = rec
        def prop(self, data, name, **kw):
            self._rec.append(name)
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


def test_panel_draws_quality_controls_headless(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless panel draw test")

    driver = tmp_path / "rtpt_panel_draw_driver.py"
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
    # Every documented RTPT control was drawn.
    for name in RTPT_PROPS_IN_ORDER:
        assert name in props, name
    # They form one contiguous quality section, in order, right after the
    # Maximum Samples row.
    max_idx = props.index("max_samples")
    assert props[max_idx + 1 : max_idx + 1 + len(RTPT_PROPS_IN_ORDER)] == list(
        RTPT_PROPS_IN_ORDER
    )
