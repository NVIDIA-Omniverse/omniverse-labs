# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stock panel COMPAT_ENGINES registration (spec blender-live-render, 03-01).

Fast tests cover the pure inclusion rule (Cycles-pattern membership minus the
curated exclusion list, plus the extra-inclusion list). The headless Blender
test proves the real wiring: after ``addon.register()`` representative panels
per included group report ``OVRTX_EXAMPLE`` in ``COMPAT_ENGINES``, excluded
panels do not, double register/unregister is idempotent, and
``addon.unregister()`` restores the audited baseline sets exactly.
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

from ovrtx_blender_example import ui  # noqa: E402
from ovrtx_blender_example.engine import ENGINE_ID  # noqa: E402


# One representative panel per included group (audited on Blender 5.1.2).
INCLUDED_PANELS = {
    "object": "OBJECT_PT_visibility",
    "light_data": "DATA_PT_EEVEE_light",
    "light_shadow_extra_inclusion": "DATA_PT_EEVEE_light_shadow",
    "light_beam_shape": "DATA_PT_spot",
    "camera_data": "DATA_PT_lens",
    "camera_dof_extra_inclusion": "DATA_PT_camera_dof",
    "world": "WORLD_PT_context_world",
    "material": "MATERIAL_PT_custom_props",
    "material_slots_extra_inclusion": "EEVEE_MATERIAL_PT_context_material",
    "render_output": "RENDER_PT_format",
    "render_dimensions": "RENDER_PT_frame_range",
    "color_management": "RENDER_PT_color_management",
    "view_layer_basics": "VIEWLAYER_PT_layer",
}

# Spot checks of the curated exclusion list (not exhaustive).
EXCLUDED_PANELS = (
    "RENDER_PT_freestyle",
    "VIEWLAYER_PT_freestyle",
    "MATERIAL_PT_freestyle_line",
    "RENDER_PT_gpencil",
    "DATA_PT_lightprobe",
    "EEVEE_MATERIAL_PT_viewport_settings",
    "DATA_PT_preview",
    "DATA_PT_light",
)


# --- Pure inclusion rule -----------------------------------------------------


def test_blender_render_member_is_included() -> None:
    assert ui.stock_panel_included(
        "DATA_PT_lens", {"BLENDER_RENDER", "BLENDER_EEVEE", "CYCLES"}
    )


def test_non_member_is_excluded() -> None:
    assert not ui.stock_panel_included("EEVEE_MATERIAL_PT_surface", {"BLENDER_EEVEE"})


def test_missing_compat_engines_is_excluded() -> None:
    assert not ui.stock_panel_included("OBJECT_PT_transform", None)


def test_curated_exclusions_override_membership() -> None:
    for name in EXCLUDED_PANELS:
        assert name in ui.STOCK_PANEL_COMPAT_EXCLUSIONS
        assert not ui.stock_panel_included(name, {"BLENDER_RENDER"})


def test_extra_inclusions_do_not_need_blender_render() -> None:
    assert ui.stock_panel_included(
        "EEVEE_MATERIAL_PT_context_material",
        {"BLENDER_EEVEE", "BLENDER_WORKBENCH"},
    )
    assert ui.stock_panel_included(
        "DATA_PT_camera_dof", {"BLENDER_EEVEE", "BLENDER_WORKBENCH"}
    )
    assert ui.stock_panel_included("DATA_PT_EEVEE_light", {"BLENDER_EEVEE"})
    assert ui.stock_panel_included("DATA_PT_EEVEE_light_shadow", {"BLENDER_EEVEE"})


def test_exclusions_and_extra_inclusions_are_disjoint() -> None:
    assert not (
        ui.STOCK_PANEL_COMPAT_EXCLUSIONS & ui.STOCK_PANEL_COMPAT_EXTRA_INCLUSIONS
    )


def test_own_render_panel_is_not_part_of_the_stock_enumeration() -> None:
    # The add-on's own panel has COMPAT_ENGINES == {ENGINE_ID}; the stock
    # enumeration must never discard the engine ID from it at unregister.
    assert not ui.stock_panel_included("OVRTXEXAMPLE_PT_render_settings", {ENGINE_ID})


# --- Headless Blender: real register/unregister ------------------------------

_DRIVER = """
import json
import sys
import traceback

result = {"errors": []}
output_path = sys.argv[sys.argv.index("--") + 1]

INCLUDED = __INCLUDED__
EXCLUDED = __EXCLUDED__

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    import ovrtx_blender_example as addon
    from ovrtx_blender_example import ui
    from ovrtx_blender_example.engine import ENGINE_ID, OvrtxExampleRenderEngine

    check_names = sorted(set(INCLUDED) | set(EXCLUDED))

    def membership():
        out = {}
        for name in check_names:
            cls = getattr(bpy.types, name, None)
            compat = getattr(cls, "COMPAT_ENGINES", None) if cls else None
            out[name] = None if compat is None else (ENGINE_ID in compat)
        return out

    def compat_snapshot():
        out = {}
        for name in check_names:
            cls = getattr(bpy.types, name, None)
            compat = getattr(cls, "COMPAT_ENGINES", None) if cls else None
            out[name] = None if compat is None else sorted(compat)
        return out

    result["panels_exist"] = {
        name: getattr(bpy.types, name, None) is not None for name in check_names
    }
    baseline = compat_snapshot()
    result["baseline_has_engine"] = membership()

    addon.register()
    result["after_register"] = membership()
    result["uses_standard_shading_nodes"] = (
        OvrtxExampleRenderEngine.bl_use_shading_nodes_custom is False
    )
    own_panel_compat = sorted(bpy.types.OVRTXEXAMPLE_PT_render_settings.COMPAT_ENGINES)

    # Idempotency: a second registration pass is a no-op on set membership
    # and enumerates the same panel count both ways.
    result["second_register_count"] = ui.register_stock_panel_compat()
    result["after_double_register"] = membership()

    # A stray unregister pass must not strip the add-on's own panel.
    result["first_unregister_count"] = ui.unregister_stock_panel_compat()
    result["own_panel_after_stock_unregister"] = sorted(
        bpy.types.OVRTXEXAMPLE_PT_render_settings.COMPAT_ENGINES
    )
    result["own_panel_at_register"] = own_panel_compat
    result["reregister_count"] = ui.register_stock_panel_compat()

    addon.unregister()
    result["after_unregister"] = membership()
    result["baseline_restored"] = compat_snapshot() == baseline

    # Double unregister of the compat helper is safe after full unregister.
    ui.unregister_stock_panel_compat()
    result["after_double_unregister"] = membership()
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_stock_panel_compat_register_unregister_headless(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless panel-compat test")

    driver_source = (
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon")))
        .replace("__INCLUDED__", repr(sorted(INCLUDED_PANELS.values())))
        .replace("__EXCLUDED__", repr(list(EXCLUDED_PANELS)))
    )
    driver = tmp_path / "stock_panel_compat_driver.py"
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
    assert result["uses_standard_shading_nodes"] is True

    # Every pinned representative and excluded panel must exist in this
    # Blender build; a disappearance means the audit needs refreshing.
    missing = [name for name, exists in result["panels_exist"].items() if not exists]
    assert missing == []

    # Baseline: the engine ID is nowhere before registration.
    assert all(value is not True for value in result["baseline_has_engine"].values())

    for group, name in INCLUDED_PANELS.items():
        assert result["after_register"][name] is True, (group, name)
    for name in EXCLUDED_PANELS:
        assert result["after_register"][name] is False, name

    # Idempotent double register: same membership, same enumeration size.
    assert result["after_double_register"] == result["after_register"]
    assert result["second_register_count"] == result["first_unregister_count"]
    assert result["second_register_count"] == result["reregister_count"]
    # The full stock enumeration is far larger than the pinned spot checks.
    assert result["second_register_count"] > 100

    # A stock-compat unregister pass never touches the add-on's own panel.
    assert result["own_panel_after_stock_unregister"] == result["own_panel_at_register"]

    # Unregister restores stock visibility rules exactly, and repeating the
    # discard pass is safe.
    for name in result["after_unregister"]:
        assert result["after_unregister"][name] is not True, name
    assert result["baseline_restored"] is True
    assert result["after_double_unregister"] == result["after_unregister"]
