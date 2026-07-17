# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTPT quality scene properties and their documented runtime mapping.

Spec render-quality-color-controls, task01-01: four scene properties on
``scene.ovrtx_example`` back the documented RTPT quality attributes, with
defaults equal to the documented runtime defaults so out-of-the-box output is
unchanged. ``RTPT_RENDER_SETTINGS`` is the single source of truth for property
name -> (attribute name, dtype, default) and is importable without ``bpy``.

The mapping assertions run in the plain (non-Blender) pytest lane. Property
registration, defaults, and ``.blend`` save/load persistence are verified by a
headless Blender driver when an executable is available; otherwise that test
skips.
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


# --- Documented contract (OVRTX render-settings skill + UI-remap decision) --
# The UI presents Cycles-like Max Bounces semantics: UI 0 = direct lighting
# only, and the add-on adds +2 to reach the OVRTX wire value (the worker counts
# the primary camera ray, so wire 2 = direct lighting). The two sub-caps are
# 0-based sub-budgets (UI == wire, offset 0). The documented runtime (wire)
# defaults are unchanged: 3 / 3 / 15 / true.
#
# name | attribute | type | ui_default | wire_default | offset
_DOCUMENTED = {
    "rtpt_max_bounces": ("omni:rtx:rtpt:maxBounces", "int32", 1, 3, 2),
    "rtpt_max_specular_and_transmission_bounces": (
        "omni:rtx:rtpt:maxSpecularAndTransmissionBounces",
        "int32",
        3,
        3,
        0,
    ),
    "rtpt_max_volume_bounces": ("omni:rtx:rtpt:maxVolumeBounces", "int32", 15, 15, 0),
    "rtpt_firefly_filter_enabled": (
        "omni:rtx:rtpt:fireflyFilter:enabled",
        "bool",
        True,
        True,
        0,
    ),
}


def test_mapping_has_exactly_the_four_documented_settings() -> None:
    assert set(properties.RTPT_RENDER_SETTINGS) == set(_DOCUMENTED)


def test_mapping_matches_documented_attributes_dtypes_and_defaults() -> None:
    for name, (attribute, dtype, ui_default, wire_default, offset) in _DOCUMENTED.items():
        spec = properties.RTPT_RENDER_SETTINGS[name]
        assert spec.attribute == attribute, name
        assert spec.dtype == dtype, name
        # The property (UI) default is the artist-facing value.
        assert spec.default == ui_default, name
        # bool defaults must be an actual bool, not an int alias.
        assert type(spec.default) is type(ui_default), name
        assert spec.offset == offset, name
        # The documented runtime (wire) default is unchanged.
        assert spec.wire_default == wire_default, name


def test_to_wire_and_from_wire_round_trip_with_the_camera_ray_offset() -> None:
    bounces = properties.RTPT_RENDER_SETTINGS["rtpt_max_bounces"]
    # UI 0 = direct lighting only (wire 2); UI 1 = one indirect bounce (wire 3).
    assert bounces.to_wire(0) == 2
    assert bounces.to_wire(1) == 3
    assert bounces.to_wire(10) == 12
    assert bounces.from_wire(2) == 0
    assert bounces.from_wire(bounces.to_wire(7)) == 7

    volume = properties.RTPT_RENDER_SETTINGS["rtpt_max_volume_bounces"]
    # 0-based sub-budget: UI value passes straight through to the wire.
    assert volume.to_wire(0) == 0
    assert volume.to_wire(15) == 15
    assert volume.from_wire(15) == 15

    firefly = properties.RTPT_RENDER_SETTINGS["rtpt_firefly_filter_enabled"]
    assert firefly.to_wire(True) is True
    assert firefly.from_wire(False) is False


def test_max_bounces_ui_hard_max_keeps_wire_within_documented_range() -> None:
    spec = properties.RTPT_RENDER_SETTINGS["rtpt_max_bounces"]
    assert properties.RTPT_MAX_BOUNCES_UI_MAX == 126
    assert spec.to_wire(properties.RTPT_MAX_BOUNCES_UI_MAX) == properties.RTPT_BOUNCE_MAX


def test_mapping_importable_without_bpy() -> None:
    # The mapping lives outside the ``if bpy is not None`` guard, so it must be
    # populated even when Blender is unavailable in this process.
    assert properties.RTPT_RENDER_SETTINGS
    assert isinstance(
        next(iter(properties.RTPT_RENDER_SETTINGS.values())),
        properties.RtptSettingSpec,
    )


# --- Headless Blender registration / persistence --------------------------
_DRIVER = """
import json
import sys
import traceback

result = {"errors": [], "steps": []}
output_path = sys.argv[sys.argv.index("--") + 1]
blend_path = sys.argv[sys.argv.index("--") + 2]

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import properties

    # Custom (non-runtime) defaults so save/load persistence is observable
    # against values that differ from the shipped documented defaults.
    CUSTOM = {
        "rtpt_max_bounces": 7,
        "rtpt_max_specular_and_transmission_bounces": 9,
        "rtpt_max_volume_bounces": 21,
        "rtpt_firefly_filter_enabled": False,
    }

    bpy.ops.wm.read_homefile(use_empty=True)
    properties.register()
    result["steps"].append("registered")

    scene = bpy.context.scene
    settings = scene.ovrtx_example

    # Registration + documented defaults.
    result["registered_defaults"] = {
        name: getattr(settings, name) for name in properties.RTPT_RENDER_SETTINGS
    }

    # Author custom values and save.
    for name, value in CUSTOM.items():
        setattr(settings, name, value)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    result["steps"].append("saved")

    # Reload from disk into a clean session and read back.
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    reloaded = bpy.context.scene.ovrtx_example
    result["reloaded_values"] = {name: getattr(reloaded, name) for name in CUSTOM}
    result["steps"].append("reloaded")

    properties.unregister()
    result["steps"].append("unregistered")
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_properties_register_default_and_persist_through_blend(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless property regression")

    driver = tmp_path / "rtpt_properties_driver.py"
    driver.write_text(
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon"))),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    blend_path = tmp_path / "rtpt_properties.blend"

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
            str(blend_path),
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
    assert result["steps"] == ["registered", "saved", "reloaded", "unregistered"]

    # Registered with the documented runtime defaults.
    assert result["registered_defaults"] == {
        name: spec.default for name, spec in properties.RTPT_RENDER_SETTINGS.items()
    }

    # Values survived a full .blend save/load round trip.
    assert result["reloaded_values"] == {
        "rtpt_max_bounces": 7,
        "rtpt_max_specular_and_transmission_bounces": 9,
        "rtpt_max_volume_bounces": 21,
        "rtpt_firefly_filter_enabled": False,
    }
