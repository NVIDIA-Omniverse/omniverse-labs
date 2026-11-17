# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qt integer oracle and deterministic keyboard cursor/repeat ownership."""
from itertools import product
from pathlib import Path
import shutil
import subprocess

import pytest

from ovd_ui.graph_keyboard import KeyboardCrosshair, accepted_rectangle, pixel
from reference_data import load


@pytest.fixture(scope="module")
def qt_oracle():
    return {tuple(r["input"]): r["expected"] for r in load("graph_rectangles")}


def test_shared_rectangle_matches_actual_qt_qwt_inclusive_arithmetic(qt_oracle):
    deltas = (-12, -11, -10, -2, -1, 0, 1, 2, 10, 11, 12)
    cases = [(x, y, x + dx, y + dy) for x, y in ((0, 0), (-11, -7), (11, 12), (-1, 0),
             (.5, -.5), (-.5, .5), (.49, -.49), (.51, -.51))
             for dx, dy in product(deltas, repeat=2)]
    assert len(qt_oracle) == len(cases) == 968
    for case in cases:
        row = qt_oracle[case]
        expected = None if row is None else tuple(row)
        assert accepted_rectangle(case[:2], case[2:]) == expected, case


def test_pixel_thresholds_keep_inclusive_and_reverse_adjacent_rules():
    assert accepted_rectangle((0, 0), (0, 0)) is None
    assert accepted_rectangle((0, 0), (1, 0)) == (-5, -5, 5, 5)
    assert accepted_rectangle((1, 0), (0, 0)) is None
    assert accepted_rectangle((0, 0), (0, 1)) == (-5, -5, 5, 5)
    assert accepted_rectangle((-5, -5), (6, 6)) == (-5, -5, 6, 6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -(1 << 29), 1 << 29])
def test_invalid_pixel_input_is_rejected(value):
    with pytest.raises(ValueError): pixel(value)


def crosshair(width=100, height=80):
    cursor = KeyboardCrosshair()
    cursor.configure(width, height)
    return cursor


def test_first_position_is_raw_center_or_last_physical_pointer():
    cursor = crosshair()
    assert cursor.arrow(262, 0) and cursor.position == (50, 39)
    cursor.release(262)
    cursor.cancel()
    assert not cursor.pointer(12.2, 23.7)
    assert cursor.begin("owned view") and cursor.position == cursor.anchor == (12, 24)
    assert cursor.origin == "owned view"


def test_repeat_delay_five_pixel_steps_release_and_no_stall_catchup():
    cursor = crosshair(500, 400)
    cursor.pointer(10, 20)
    assert cursor.arrow(262, 10) and cursor.position == (11, 20)
    assert not cursor.arrow(262, 10.1)  # Duplicate press cannot become a repeat.
    assert not cursor.repeat(10.399)
    assert cursor.repeat(10.4) and cursor.position == (16, 20)
    assert cursor.repeat(10.450001) and cursor.position == (21, 20)
    assert cursor.repeat(20) and cursor.position == (26, 20)
    assert not cursor.repeat(20.001)
    cursor.release(265)
    assert cursor.held == 262
    cursor.release(262)
    assert not cursor.repeat(100) and cursor.position == (26, 20)


@pytest.mark.parametrize("key,expected", [(262, (9, 4)), (263, (0, 4)), (264, (4, 9)), (265, (4, 0))])
def test_repeat_clamps_to_last_canvas_pixel(key, expected):
    cursor = crosshair(10, 10)
    cursor.arrow(key, 0)
    cursor.repeat(1)
    cursor.repeat(2)
    assert cursor.position == expected


def test_pointer_notifications_do_not_erase_keyboard_position_but_motion_takes_over():
    cursor = crosshair()
    cursor.pointer(20, 25)
    cursor.begin("origin")
    cursor.arrow(262, 0)
    assert not cursor.pointer(20, 25) and cursor.position == (21, 25)
    assert cursor.pointer(31, 32) and cursor.position == (31, 32)
    assert cursor.active and cursor.anchor == (20, 25) and cursor.held is None
    rectangle, origin = cursor.finish()
    assert rectangle == accepted_rectangle((20, 25), (31, 32)) and origin == "origin"
    assert not cursor.active
    assert cursor.pointer(33, 35) and not cursor.visible


def test_resize_including_fractional_extent_change_cancels_every_pending_state():
    cursor = crosshair(100.1, 80)
    cursor.pointer(10, 15)
    cursor.begin("old")
    cursor.arrow(264, 0)
    assert cursor.configure(100.9, 80)
    assert cursor.position is cursor.last_pointer is cursor.anchor is cursor.origin is cursor.held is None
    assert not cursor.visible and not cursor.configure(100.9, 80)
    assert cursor.configure(0, 80) and cursor.size is None
    assert not cursor.begin("unavailable")


def test_rejected_point_and_escape_do_not_return_a_zoom_rectangle():
    cursor = crosshair()
    cursor.begin("view")
    assert cursor.finish() == (None, "view")
    cursor.begin("view")
    cursor.arrow(264, 0)
    assert cursor.cancel() and cursor.finish() is None and not cursor.repeat(10)
