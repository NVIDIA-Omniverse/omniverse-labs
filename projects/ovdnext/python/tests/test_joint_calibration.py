# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned geometry calibration against retained, independent original Qt queues."""
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import _pvd_native
from _pvd_native import Session

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.joint_scale import (JointCalibration, resolve_joint_base, _upper_median, _extent_votes,
                               validate_joint_scale, joint_scale_from_preferences)
from ovd_ui.render_settings import RenderingSettings, DisplayMode
from ovd_ui.settings import Preferences
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads(gzip.decompress((Path(__file__).parent / "data/qt_joint_calibration.json.gz").read_bytes()))


def test_master_rejects_joint_arm_overflow_without_changing_the_calibrated_request():
    calibration=JointCalibration(1,1,1,1,1e37,'captured-tolerance',0)
    assert calibration.request(2,ovd=True,master_scale=2)=={'joint_scale':2.,'joint_base_length':1e37}
    with pytest.raises(ValueError):calibration.request(2,ovd=True,master_scale=20)
    assert calibration.base_length==1e37


@pytest.mark.parametrize('ovd',[False,True])
def test_master_checks_raw_sdk_joint_arm_even_with_a_small_ovd_calibration(ovd):
    calibration=JointCalibration(1,1,1,1,.1,'captured-tolerance',0)
    with pytest.raises(ValueError):calibration.request(2,ovd=ovd,master_scale=2e38)
    with pytest.raises(ValueError):validate_joint_scale(2,master_scale=2e38)


@pytest.fixture(scope="session")
def calibration_writer(tmp_path_factory):
    return executable("pvd_joint_calibration_fixture", "PVD_JOINT_CALIBRATION_WRITER")


@pytest.fixture
def calibration_capture(calibration_writer, tmp_path):
    def make(mode):
        path = tmp_path / (mode + ".ovd")
        subprocess.run([str(calibration_writer), str(path), mode], check=True)
        return path, Session(str(path), str(tmp_path))
    return make


def resolve(view, **kwargs):
    return resolve_joint_base(view, PvdDataAdapter(view), RenderingSettings(), **kwargs)


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["capture"])
def test_real_owned_rows_match_original_qt_queue_upper_median(case, tmp_path):
    source = Session(str(ROOT / "test/data" / case["capture"]), str(tmp_path))
    from capture_semantics import populated_scene
    scene_id=populated_scene(source,case['samples'][0]['frame']).id
    for sample in case["samples"]:
        view = source.presentation(scene_id, sample["frame"], tolerance_speed=None)
        bounds = np.array([row["queue_world_bounds"] for row in sample["records"]], dtype="f4").reshape(-1, 2, 3)
        # Independent oracle computes directly from original submitted bounds;
        # it does not use the resolver's masks, weights, or median algorithm.
        expected = sorted(float(np.float32(2) * np.max((b[1] - b[0]) * np.float32(.5))) for b in bounds)
        hidden, selected = case['hidden'], case['selected']
        if case['capture'] == 'multi_geom.ovd':
            from capture_semantics import multi_geom_roles
            roles = multi_geom_roles(view)
            geometry = view.geometry_snapshot()
            assert set(geometry.kinds[geometry.object_ids == roles.plane]) == {4}
            assert geometry.kinds[geometry.object_ids == roles.box].tolist() == [2]
            hidden, selected = [roles.plane], [roles.box]
        result = resolve(view, hidden=hidden, selected=selected)
        assert result.vote_count == len(expected)
        assert result.base_length == expected[len(expected) // 2]
        assert result.source == "solid-geometry" and result.matches(view)
        assert len(view.geometry_snapshot().kinds) >= (1000 if "ue4" in case["capture"] else 8)


@pytest.mark.parametrize("values,weights", [([4, 1], [1, 1]), ([0., -0., 4., 2.], [1]*4),
    ([3., 1., 2.], [3, 2, 1]), ([1.e-38, 1.e30, 0., 1.], [2, 1, 1, 1]), ([], [])])
def test_radix_rank_matches_independent_expanded_median(values, weights):
    data = np.asarray(values, dtype="f4");data[data == 0] = 0
    weights = np.asarray(weights, dtype="u8")
    expected = sorted(float(value) for value, weight in zip(data, weights) for _ in range(int(weight)))
    result, count = _upper_median(lambda: iter(((data[:2], weights[:2]), (data[2:], weights[2:]))))
    assert count == len(expected)
    assert result == (expected[len(expected)//2] if expected else None)


def test_extent_arithmetic_rejects_invalid_and_overflow_and_normalizes_zero():
    bounds = np.array([[[-0., 0, 0], [0., 0, 0]], [[2, 0, 0], [1, 0, 0]],
        [[-3e38]*3, [3e38]*3], [[0, 0, 0], [float("nan"), 1, 1]], [[-2]*3, [2]*3]], dtype="f4")
    sizes, valid = _extent_votes(bounds)
    assert valid.tolist() == [True, False, False, False, True]
    assert sizes[0].view("u4") == 0 and sizes[-1] == 4


@pytest.mark.parametrize("mode", ["empty", "plane", "zero", "overflow", "surface-empty-first"])
def test_native_empty_degenerate_and_first_empty_surface_use_own_tolerance(calibration_capture, mode):
    _, source = calibration_capture(mode)
    first = source.presentation(frame=1, tolerance_speed=None)
    result = resolve(first)
    assert result.base_length == first.tolerance_length == 7
    assert result.source == "captured-tolerance"
    if mode == "surface-empty-first":
        # A later valid mesh really exists; Qt stops before it at the first
        # positive, empty mesh reference rather than using native row order.
        assert len(first.geometry_snapshot().meshes) == 1
    later = source.presentation(frame=2, tolerance_speed=None)
    assert resolve(later).base_length == later.tolerance_length == 9
    assert result.base_length == 7 and result.matches(later)  # ordinary frames share calibration scope


def test_deformable_first_surface_has_one_vote_per_triangle(calibration_capture, monkeypatch):
    _, source = calibration_capture("surface-two")
    view = source.presentation(frame=1, tolerance_speed=None)
    assert len(view.geometry_snapshot().meshes) == 2
    import ovd_ui.joint_scale as module
    monkeypatch.setattr(module, "TRIANGLE_CHUNK", 1)  # crosses chunk boundaries in every radix pass
    result = resolve(view)
    assert result.vote_count == 2 and result.base_length == 2
    hidden = [int(view.geometry_snapshot().object_ids[0])]
    assert resolve(view, hidden=hidden).base_length == 7


def test_visibility_and_retained_base_are_owned_not_mutable(calibration_capture):
    _, source = calibration_capture("box")
    view = source.presentation(frame=1, tolerance_speed=None)
    first = resolve(view)
    assert first.vote_count == 1 and first.base_length == 4
    actor = int(view.geometry_snapshot().object_ids[0])
    assert resolve(view, hidden=[actor]).base_length == 7
    later = source.presentation(frame=2, tolerance_speed=None)
    assert resolve(later).base_length == 16
    del source, later
    assert first.base_length == 4 and resolve(view) == first


@pytest.mark.parametrize("value", [True, None, "1", -1, 2.01, float("nan"), float("inf"), 10**1000])
def test_invalid_raw_preferences_fall_back_without_writing(tmp_path, value):
    preferences = Preferences(tmp_path / "preferences.json")
    preferences.values["joint_scale"] = value
    assert joint_scale_from_preferences(preferences) == 1
    assert not preferences.path.exists()
    with pytest.raises(ValueError):validate_joint_scale(value)


def test_foreign_or_incomplete_adapter_cannot_calibrate(calibration_capture):
    _, source = calibration_capture("box")
    first = source.presentation(frame=1)
    second = source.presentation(frame=2)
    with pytest.raises(ValueError):resolve_joint_base(second, PvdDataAdapter(first))
