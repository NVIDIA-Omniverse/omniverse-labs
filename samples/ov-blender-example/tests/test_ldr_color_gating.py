# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LDR color-control gating in the OVRTX Render panel.

Spec render-quality-color-controls, task03-01: in LDR display-passthrough mode
OVRTX owns the display encoding, so Blender's View Transform / Look / Exposure /
Gamma controls must not be presented as effective. They stay visible (for
discoverability) but disabled, with a short explanation. Scene-linear HDR
enables them. A fail-closed scene-linear selection (``status == unavailable``)
is gated as LDR with its ``unavailable_reason`` surfaced. Gating keys off the
resolved mode from ``color_presentation.presentation_from_scene`` — never the
raw property value.

Three lanes:

* Pure-Python unit tests on the ``color_control_gating`` helper covering each
  resolved mode: LDR, scene-linear (enabled), and scene-linear fail-closed.
* Plain (non-Blender) source-level checks on the panel ``draw`` method: the
  gating is keyed off ``presentation_from_scene`` and the view-settings column
  ``enabled`` flag comes from ``color_control_gating``.
* A headless Blender draw test that drives the real ``draw`` for each resolved
  mode and asserts the view-settings column's ``enabled`` state and the
  presence/absence of the gating explanation. Skips when no Blender is found.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import color_presentation, ui  # noqa: E402


UI_SOURCE = (ROOT / "addon" / "ovrtx_blender_example" / "ui.py").read_text(
    encoding="utf-8"
)


def _draw_source() -> str:
    marker = "def draw(self, context: Any) -> None:"
    start = UI_SOURCE.index(marker)
    end = UI_SOURCE.index("\nelse:", start)
    return UI_SOURCE[start:end]


# --- color_control_gating unit tests (each resolved mode) ------------------


def test_ldr_passthrough_disables_controls_with_explanation() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        None, requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    )
    gating = ui.color_control_gating(diagnostics)
    assert gating["enabled"] is False
    # The disable-with-explanation copy names OVRTX as the display-encoding
    # owner in LDR passthrough.
    assert gating["explanation"]
    assert gating["explanation"][0] == ui.LDR_COLOR_GATING_EXPLANATION
    assert "OVRTX owns the display encoding" in ui.LDR_COLOR_GATING_EXPLANATION
    # No fail-closed reason line in the plain-LDR case.
    assert len(gating["explanation"]) == 1


def test_scene_linear_available_enables_controls() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        None,
        requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
        hdr_readback_available=True,
    )
    gating = ui.color_control_gating(diagnostics)
    assert gating["enabled"] is True
    assert gating["explanation"] == []


def test_scene_linear_unknown_readback_still_enables_controls() -> None:
    # At UI draw time HdrColor readback availability is unknown (None); that
    # does not fail closed, so a scene-linear selection is presented enabled.
    diagnostics = color_presentation.presentation_from_scene(
        None, requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR
    )
    gating = ui.color_control_gating(diagnostics)
    assert gating["enabled"] is True
    assert gating["explanation"] == []


def test_scene_linear_failclosed_gates_as_ldr_with_reason() -> None:
    # Scene-linear requested but the runtime cannot read back HdrColor: the
    # controls are gated as LDR (disabled) and the unavailable reason is
    # surfaced so the artist knows why the selection is not in effect.
    diagnostics = color_presentation.presentation_from_scene(
        None,
        requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
        hdr_readback_available=False,
    )
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    gating = ui.color_control_gating(diagnostics)
    assert gating["enabled"] is False
    assert gating["explanation"][0] == ui.LDR_COLOR_GATING_EXPLANATION
    # The fail-closed reason is surfaced as a second line, mentioning HdrColor.
    assert len(gating["explanation"]) == 2
    assert "HdrColor" in gating["explanation"][1]


def test_env_ocio_failclosed_gates_as_ldr_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reserved OCIO env seam resolves to an unavailable status at draw
    # time; it, too, gates as LDR and surfaces its reason generically.
    monkeypatch.setenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, "ocio_baked")
    diagnostics = color_presentation.presentation_from_scene(None)
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    gating = ui.color_control_gating(diagnostics)
    assert gating["enabled"] is False
    assert gating["explanation"][0] == ui.LDR_COLOR_GATING_EXPLANATION
    assert len(gating["explanation"]) == 2
    # Unknown reason codes fall back to the raw code (nothing swallowed).
    assert diagnostics["unavailable_reason"] in gating["explanation"][1]


def test_gating_ignores_raw_enum_when_env_forces_ldr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with the scene enum set to scene-linear, an env override to LDR
    # disables the controls — gating keys off the resolved mode, not the raw
    # property value.
    from types import SimpleNamespace

    scene = SimpleNamespace(
        ovrtx_example=SimpleNamespace(
            color_presentation_mode=color_presentation.MODE_SCENE_LINEAR_HDR
        ),
        view_settings=SimpleNamespace(
            view_transform="", look="", exposure=0.0, gamma=1.0
        ),
        display_settings=SimpleNamespace(display_device=""),
    )
    monkeypatch.setenv(
        color_presentation.ENV_COLOR_PRESENTATION_MODE,
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    )
    gating = ui.color_control_gating(
        color_presentation.presentation_from_scene(scene)
    )
    assert gating["enabled"] is False


# --- Source-level checks (plain lane) --------------------------------------


def test_draw_gates_off_presentation_from_scene() -> None:
    draw = _draw_source()
    # Gating is derived from presentation_from_scene (resolved mode), not the
    # raw enum, and mapped through color_control_gating.
    assert "presentation_from_scene(context.scene)" in draw
    assert "color_control_gating(" in draw
    assert "view_col.enabled = gating[" in draw
    # The disabled state renders its explanation.
    assert 'gating["explanation"]' in draw


# --- Headless Blender draw test (each resolved mode) -----------------------
_DRIVER = """
import json
import os
import sys
import traceback

result = {"errors": [], "steps": [], "scenarios": {}}
output_path = sys.argv[sys.argv.index("--") + 1]

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import color_presentation, properties, ui

    bpy.ops.wm.read_homefile(use_empty=True)
    properties.register()
    ui.register()
    result["steps"].append("registered")

    panel_cls = bpy.types.OVRTXEXAMPLE_PT_render_settings

    ui.viewport_session_status = lambda *a, **k: {
        "status": "stopped",
        "label": "Stopped",
        "hint": "",
        "logs": {"status": "stdout", "log_dir": ""},
    }

    class RecordingLayout:
        def __init__(self, labels, columns):
            self.enabled = True
            self.props = []
            self._labels = labels
            self._columns = columns
        def prop(self, data, name, **kw):
            self.props.append(name)
            return self
        def operator(self, *a, **kw):
            return self
        def label(self, text="", **kw):
            self._labels.append(text)
            return self
        def row(self, **kw):
            return self
        def box(self):
            return self
        def column(self, **kw):
            col = RecordingLayout(self._labels, self._columns)
            self._columns.append(col)
            return col

    class FakePanel:
        pass

    def draw_once():
        labels = []
        columns = []
        panel = FakePanel()
        panel.layout = RecordingLayout(labels, columns)
        panel_cls.draw(panel, bpy.context)
        view_cols = [c for c in columns if "view_transform" in c.props]
        assert len(view_cols) == 1, "expected exactly one view-settings column"
        return {
            "view_col_enabled": bool(view_cols[0].enabled),
            "labels": labels,
        }

    scene = bpy.context.scene
    env = color_presentation.ENV_COLOR_PRESENTATION_MODE

    # LDR passthrough (default enum, no env): disabled + LDR explanation.
    os.environ.pop(env, None)
    scene.ovrtx_example.color_presentation_mode = (
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    )
    result["scenarios"]["ldr"] = draw_once()

    # Scene-linear HDR: enabled, no gating explanation.
    scene.ovrtx_example.color_presentation_mode = (
        color_presentation.MODE_SCENE_LINEAR_HDR
    )
    result["scenarios"]["scene_linear"] = draw_once()

    # Fail-closed (reserved OCIO env seam resolves unavailable at draw time):
    # gated as LDR with a surfaced reason.
    os.environ[env] = "ocio_baked"
    try:
        result["scenarios"]["fail_closed"] = draw_once()
    finally:
        os.environ.pop(env, None)

    result["steps"].append("drawn")

    ui.unregister()
    properties.unregister()
    result["steps"].append("unregistered")
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_panel_gating_each_resolved_mode_headless(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless panel draw test")

    driver = tmp_path / "ldr_gating_draw_driver.py"
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

    scenarios = result["scenarios"]

    # LDR passthrough: controls disabled, LDR explanation shown.
    ldr = scenarios["ldr"]
    assert ldr["view_col_enabled"] is False
    assert any("OVRTX owns the display encoding" in text for text in ldr["labels"])

    # Scene-linear HDR: controls enabled, no gating explanation.
    scene_linear = scenarios["scene_linear"]
    assert scene_linear["view_col_enabled"] is True
    assert not any(
        "OVRTX owns the display encoding" in text for text in scene_linear["labels"]
    )

    # Fail-closed: gated as LDR (disabled) with a surfaced reason.
    fail_closed = scenarios["fail_closed"]
    assert fail_closed["view_col_enabled"] is False
    assert any(
        "OVRTX owns the display encoding" in text for text in fail_closed["labels"]
    )
    assert any("unavailable" in text.lower() for text in fail_closed["labels"])
