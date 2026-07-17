# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared pure-Python curve-width/validation helper.

These run without bpy or pxr (the module's core contract): the live runtime,
the fixture-prep exporter, and headless CI all rely on identical width and
validity math.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import curve_widths as cw  # noqa: E402


# --- particle widths ---------------------------------------------------------
def test_particle_width_range_blunt_tip_floor():
    # tip below the 5% floor; blunt (not close) raises it to 20% of root.
    root, tip = cw.particle_width_range(0.01, 1.0, 0.04, use_close_tip=False)
    assert root == 0.01
    assert math.isclose(tip, 0.002, rel_tol=1e-6)  # 20% blunt floor


def test_particle_width_range_close_tip_allows_thinner():
    # Same inputs, but close tip only floors to 5% of root, so it stays thinner.
    root, tip = cw.particle_width_range(0.01, 1.0, 0.04, use_close_tip=True)
    assert math.isclose(tip, 0.0005, rel_tol=1e-6)  # 5% close floor


def test_particle_width_range_zero_tip_radius_defaults():
    # A tip_radius that reads back as 0 is treated as the 0.2 default (mirrors
    # the fixture-prep `or default` guard), not a literal zero.
    _, tip = cw.particle_width_range(0.01, 1.0, 0.0, use_close_tip=True)
    assert math.isclose(tip, 0.002, rel_tol=1e-6)  # 0.01 * 0.2


def test_particle_width_range_defaults_guard_zero_none():
    assert cw.particle_width_range(0, None, None) == (0.005, 0.001)


def test_hair_sample_count_from_render_step():
    assert cw.hair_sample_count(5) == 33  # 2**5 + 1, native resolution
    assert cw.hair_sample_count(0, hair_step=2) == 5  # falls back to hair_step
    assert cw.hair_sample_count(10, cap=64) == 64  # capped for USD size
    assert cw.hair_sample_count(0, 0, minimum=3) >= 3  # clamps to floor


def test_fill_hair_widths_tapers_root_to_tip():
    widths = cw.fill_hair_widths([4], root_width=0.01, tip_width=0.002)
    assert len(widths) == 4
    assert math.isclose(widths[0], 0.01, rel_tol=1e-6)
    assert math.isclose(widths[-1], 0.002, rel_tol=1e-6)
    # Monotonically decreasing along the strand.
    assert all(widths[i] >= widths[i + 1] for i in range(len(widths) - 1))


def test_fill_hair_widths_length_matches_sum_and_multi_strand():
    counts = [3, 5, 2]
    widths = cw.fill_hair_widths(counts, 0.02, 0.004)
    assert len(widths) == sum(counts)
    # Each strand starts at root width.
    assert math.isclose(widths[0], 0.02, rel_tol=1e-6)
    assert math.isclose(widths[3], 0.02, rel_tol=1e-6)
    assert math.isclose(widths[8], 0.02, rel_tol=1e-6)


def test_fill_hair_widths_single_point_strand():
    widths = cw.fill_hair_widths([1], 0.01, 0.002)
    assert list(widths) == [np.float32(0.01)]


# --- physical width rule (#83) ----------------------------------------------
def test_physical_curve_width_generalizes_57_relation():
    # width = 2 * bevel_depth * radius
    assert math.isclose(cw.physical_curve_width(0.0008, 1.0), 0.0016, rel_tol=1e-9)
    assert math.isclose(cw.physical_curve_width(0.0008, 5.0), 0.008, rel_tol=1e-9)


def test_width_is_finite_positive():
    assert cw.width_is_finite_positive(0.001)
    assert not cw.width_is_finite_positive(0.0)
    assert not cw.width_is_finite_positive(-0.001)
    assert not cw.width_is_finite_positive(float("nan"))
    assert not cw.width_is_finite_positive(float("inf"))


def test_width_is_implausible_flags_meter_scale_blob():
    # Classroom-style: authored 8.74 m vs physical ~0.0016 m reference.
    assert cw.width_is_implausible(8.74, 0.0016)
    # A width within tolerance is plausible.
    assert not cw.width_is_implausible(0.0016, 0.0016)
    assert not cw.width_is_implausible(0.05, 0.0016)  # 31x, under 100x default


def test_width_is_implausible_flags_nonfinite_regardless_of_reference():
    assert cw.width_is_implausible(float("nan"), 0.0016)
    assert cw.width_is_implausible(-1.0, 0.0016)
    # No usable reference -> only finite/positive gate applies.
    assert not cw.width_is_implausible(2.0, 0.0)


# --- BasisCurves validity ----------------------------------------------------
def test_points_match_counts():
    assert cw.points_match_counts(10, [4, 6])
    assert not cw.points_match_counts(9, [4, 6])


def test_curve_count_valid_linear():
    assert cw.curve_count_valid(2, "linear", "catmullRom", "nonperiodic")
    assert not cw.curve_count_valid(1, "linear", "catmullRom", "nonperiodic")
    assert cw.curve_count_valid(3, "linear", "catmullRom", "periodic")
    assert not cw.curve_count_valid(2, "linear", "catmullRom", "periodic")


def test_curve_count_valid_cubic_catmullrom_nonperiodic():
    # vstep 1 -> vc >= 4.
    assert not cw.curve_count_valid(3, "cubic", "catmullRom", "nonperiodic")
    assert cw.curve_count_valid(4, "cubic", "catmullRom", "nonperiodic")
    assert cw.curve_count_valid(12, "cubic", "catmullRom", "nonperiodic")


def test_curve_count_valid_cubic_bezier_vstep3():
    # bezier nonperiodic: (vc - 4) % 3 == 0 -> 4, 7, 10 ...
    assert cw.curve_count_valid(4, "cubic", "bezier", "nonperiodic")
    assert cw.curve_count_valid(7, "cubic", "bezier", "nonperiodic")
    assert not cw.curve_count_valid(6, "cubic", "bezier", "nonperiodic")


def test_curve_count_valid_pinned_allows_short():
    assert cw.curve_count_valid(2, "cubic", "catmullRom", "pinned")
    assert not cw.curve_count_valid(1, "cubic", "catmullRom", "pinned")


def test_min_points_for_cubic():
    assert cw.min_points_for_cubic("catmullRom", "nonperiodic") == 4
    assert cw.min_points_for_cubic("catmullRom", "pinned") == 2


def test_widths_length_valid_interpolations():
    assert cw.widths_length_valid("constant", 1, 10, 2)
    assert not cw.widths_length_valid("constant", 2, 10, 2)
    assert cw.widths_length_valid("uniform", 2, 10, 2)
    assert cw.widths_length_valid("vertex", 10, 10, 2)
    assert not cw.widths_length_valid("vertex", 2, 10, 2)  # the silent-blob bug
    assert cw.widths_length_valid("varying", 8, 10, 2, num_segments=6)


def test_invalid_and_short_strand_indices():
    counts = [4, 3, 12, 1]
    # For cubic catmullRom nonperiodic, 3 and 1 are invalid.
    assert cw.invalid_curve_indices(counts) == [1, 3]
    assert cw.short_strand_indices(counts) == [1, 3]
    # Pinned wrap rescues vc>=2, so only the single-point strand is short.
    assert cw.short_strand_indices(counts, "catmullRom", "pinned") == [3]


def test_module_imports_without_bpy_or_pxr(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "bpy", None)
    monkeypatch.setitem(sys.modules, "pxr", None)
    importlib.reload(cw)
