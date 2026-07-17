# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless stock-panel poll verification (spec blender-live-render, 03-02).

Task 03-01 proved COMPAT_ENGINES membership flips at register/unregister
(``tests/test_stock_panel_compat.py``). This module goes one step further: it
boots headless Blender, registers the add-on, sets the scene render engine to
``OVRTX_EXAMPLE`` for real, and verifies visibility per representative panel
group the way the properties editor would:

- ``poll()`` is called with the real headless context for panels whose poll
  reads only ``context.engine``/``context.scene`` (world, render output).
- ``COMPAT_ENGINES`` membership is asserted for panels whose poll needs
  properties-editor context members (``context.object`` / ``context.light`` /
  ``context.camera`` / ``context.material``) that a background context does
  not provide; the poll call -- not the test -- is skipped for those, per the
  task clarification.
- Excluded panels get a companion spot check: membership stays absent, and
  the engine-only-poll excluded panels actually poll False with the engine
  active.

Representative pins are reused from task 03-01's audit-backed tables so the
two modules cannot drift apart silently. A pinned panel disappearing in a
future Blender bump is a legitimate failure: the audit needs refreshing.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "addon"))

import test_stock_panel_compat as compat_pins  # noqa: E402

from ovrtx_blender_example.engine import ENGINE_ID  # noqa: E402


# Module-level skip: everything here needs a real Blender executable.
pytestmark = pytest.mark.skipif(
    blender_executable() is None,
    reason=(
        "Blender executable not available for headless panel poll "
        "verification (install Blender 5.1 or set BLENDER_COMMAND)"
    ),
)


# Panels whose poll() reads only context.engine / context.scene (audited in
# Blender 5.1.2 bl_ui sources), so the real headless context is enough:
#   WORLD_PT_context_world.poll   -> context.engine in COMPAT_ENGINES
#   RenderOutputButtonsPanel.poll -> context.engine in COMPAT_ENGINES
POLL_VERIFIED_PANELS = {
    "world": compat_pins.INCLUDED_PANELS["world"],  # WORLD_PT_context_world
    "render_output": compat_pins.INCLUDED_PANELS["render_output"],  # RENDER_PT_format
}

# Panels whose poll() needs properties-editor context members that a
# background (headless) context does not resolve. The poll call is skipped
# for these and COMPAT_ENGINES membership is asserted instead:
#   OBJECT_PT_visibility.poll -> context.object   (editor context member)
#   DataButtonsPanel.poll     -> context.light    (properties-editor member)
#   CameraButtonsPanel.poll   -> context.camera   (properties-editor member)
#   MaterialButtonsPanel.poll -> context.material (properties-editor member)
MEMBERSHIP_ONLY_PANELS = {
    "object": compat_pins.INCLUDED_PANELS["object"],  # OBJECT_PT_visibility
    "light_data": compat_pins.INCLUDED_PANELS["light_data"],  # DATA_PT_EEVEE_light
    "camera_data": compat_pins.INCLUDED_PANELS["camera_data"],  # DATA_PT_lens
    "material": compat_pins.INCLUDED_PANELS["material"],  # MATERIAL_PT_custom_props
}

# Extra-inclusion panels from 03-01 whose stock compat sets lack
# BLENDER_RENDER entirely ({'BLENDER_EEVEE', 'BLENDER_WORKBENCH'}); they gain
# the engine ID via STOCK_PANEL_COMPAT_EXTRA_INCLUSIONS. Their polls need
# context.material / context.camera, so membership only.
EXTRA_INCLUSION_PANELS = {
    "material_slots": compat_pins.INCLUDED_PANELS["material_slots_extra_inclusion"],
    "camera_dof": compat_pins.INCLUDED_PANELS["camera_dof_extra_inclusion"],
    "camera_dof_aperture": "DATA_PT_camera_dof_aperture",  # sibling of the 03-01 pin
    "light_shadow": compat_pins.INCLUDED_PANELS["light_shadow_extra_inclusion"],
}

# Excluded-panel spot checks (not exhaustive; the curated list lives in
# ui.STOCK_PANEL_COMPAT_EXCLUSIONS and is only spot-checked, per the spec).
EXCLUDED_MEMBERSHIP_PANELS = compat_pins.EXCLUDED_PANELS

# Excluded panels whose poll() is engine-only (RenderButtonsPanel /
# RenderFreestyleButtonsPanel), so with OVRTX_EXAMPLE active their poll must
# actually return False -- proving exclusion at the poll level, not just the
# membership level.
EXCLUDED_POLL_PANELS = ("RENDER_PT_freestyle", "RENDER_PT_gpencil")


def test_pins_reuse_task03_01_audit_tables() -> None:
    """The six spec groups map 1:1 onto the 03-01 audit pins."""
    spec_groups = (
        "object",
        "light_data",
        "camera_data",
        "world",
        "material",
        "render_output",
    )
    pinned = {**POLL_VERIFIED_PANELS, **MEMBERSHIP_ONLY_PANELS}
    assert set(pinned) == set(spec_groups)
    # Excluded poll spot checks must come from the curated exclusion list.
    from ovrtx_blender_example import ui

    for name in EXCLUDED_POLL_PANELS:
        assert name in ui.STOCK_PANEL_COMPAT_EXCLUSIONS, name


_DRIVER = """
import json
import sys
import traceback

result = {"errors": []}
output_path = sys.argv[sys.argv.index("--") + 1]

POLL_PANELS = __POLL_PANELS__
MEMBERSHIP_PANELS = __MEMBERSHIP_PANELS__
EXCLUDED_MEMBERSHIP = __EXCLUDED_MEMBERSHIP__
EXCLUDED_POLL = __EXCLUDED_POLL__

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    import ovrtx_blender_example as addon
    from ovrtx_blender_example.engine import ENGINE_ID

    addon.register()

    # The behavior under test: the scene engine is OVRTX_EXAMPLE for real,
    # so poll() sees it via context.engine exactly as the properties editor
    # would.
    bpy.context.scene.render.engine = ENGINE_ID
    result["scene_engine"] = bpy.context.scene.render.engine
    result["context_engine"] = getattr(bpy.context, "engine", None)

    def inspect(name, call_poll):
        cls = getattr(bpy.types, name, None)
        if cls is None:
            return {"exists": False}
        entry = {"exists": True}
        compat = getattr(cls, "COMPAT_ENGINES", None)
        entry["member"] = None if compat is None else (ENGINE_ID in compat)
        if call_poll:
            try:
                entry["poll"] = bool(cls.poll(bpy.context))
            except Exception:
                entry["poll_error"] = traceback.format_exc()
        return entry

    result["included"] = {}
    for name in POLL_PANELS:
        result["included"][name] = inspect(name, call_poll=True)
    for name in MEMBERSHIP_PANELS:
        result["included"][name] = inspect(name, call_poll=False)

    result["excluded"] = {}
    for name in EXCLUDED_MEMBERSHIP:
        result["excluded"][name] = inspect(name, call_poll=False)
    for name in EXCLUDED_POLL:
        result["excluded"][name] = inspect(name, call_poll=True)

    addon.unregister()
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


@pytest.fixture(scope="module")
def poll_probe(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the headless Blender poll probe once and share the parsed result."""
    blender = blender_executable()
    assert blender is not None  # pytestmark guards this

    tmp_path = tmp_path_factory.mktemp("stock_panel_poll")
    membership_panels = sorted(
        set(MEMBERSHIP_ONLY_PANELS.values()) | set(EXTRA_INCLUSION_PANELS.values())
    )
    driver_source = (
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon")))
        .replace("__POLL_PANELS__", repr(sorted(POLL_VERIFIED_PANELS.values())))
        .replace("__MEMBERSHIP_PANELS__", repr(membership_panels))
        .replace("__EXCLUDED_MEMBERSHIP__", repr(list(EXCLUDED_MEMBERSHIP_PANELS)))
        .replace("__EXCLUDED_POLL__", repr(list(EXCLUDED_POLL_PANELS)))
    )
    driver = tmp_path / "stock_panel_poll_driver.py"
    driver.write_text(driver_source, encoding="utf-8")
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
    return result


def test_stock_panels_poll_visible_with_engine_active(poll_probe: dict) -> None:
    # The engine really was active; context.engine is what stock polls read.
    assert poll_probe["scene_engine"] == ENGINE_ID
    assert poll_probe["context_engine"] == ENGINE_ID

    # Every pinned panel must exist in this Blender build; a disappearance
    # means the 03-01 audit needs refreshing.
    missing = [
        name for name, entry in poll_probe["included"].items() if not entry["exists"]
    ]
    assert missing == []

    # Membership holds for every representative (all six groups + extras).
    for name, entry in poll_probe["included"].items():
        assert entry["member"] is True, name

    # Engine/scene-only polls pass with the real headless context.
    for group, name in POLL_VERIFIED_PANELS.items():
        entry = poll_probe["included"][name]
        assert "poll_error" not in entry, (group, name, entry.get("poll_error"))
        assert entry["poll"] is True, (group, name)


def test_excluded_panels_stay_excluded_with_engine_active(poll_probe: dict) -> None:
    missing = [
        name for name, entry in poll_probe["excluded"].items() if not entry["exists"]
    ]
    assert missing == []

    # Spot check: curated exclusions never gain the engine ID.
    for name in EXCLUDED_MEMBERSHIP_PANELS:
        assert poll_probe["excluded"][name]["member"] is False, name

    # And the engine-only-poll exclusions really poll invisible while the
    # engine is active.
    for name in EXCLUDED_POLL_PANELS:
        entry = poll_probe["excluded"][name]
        assert "poll_error" not in entry, (name, entry.get("poll_error"))
        assert entry["poll"] is False, name
