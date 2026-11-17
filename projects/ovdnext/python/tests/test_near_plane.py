# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The logarithmic control follows original Qt settings and keeps exact drafts."""
import math
from types import SimpleNamespace

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.camera import Camera, CameraController, near_plane_from_slider, near_plane_slider_position
from ovd_ui.widgets.camera_view import CameraView
from reference_data import load


@pytest.fixture(scope="module")
def qt_near():
    return load("near_plane")


def test_all_slider_positions_and_saturation_match_original_qt(qt_near):
    for item in qt_near["positions"]:
        assert near_plane_from_slider(item["position"]) == pytest.approx(item["near"], rel=2e-6)
    for item in qt_near["values"]:
        assert near_plane_slider_position(item["near"]) == item["position"]
    assert near_plane_from_slider(1) == .0001
    assert near_plane_from_slider(100) == 1000


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf])
def test_invalid_precise_value_has_no_slider_location(value):
    with pytest.raises(ValueError): near_plane_slider_position(value)


@pytest.fixture
def editor():
    changes = []
    control = CameraController(Camera(near=.0123))
    view = CameraView(control, lambda: changes.append(control.camera.near),
        **{name: lambda: None for name in ("frame_all", "frame_selected", "center_selected", "track_selected", "reset")})
    view.near = SimpleNamespace(model=ui.SimpleFloatModel(.0123))
    view.near_slider = SimpleNamespace(model=ui.SimpleIntModel(1, min=1, max=100))
    view.near.model.add_value_changed_fn(view._near_field_changed)
    view.near_slider.model.add_value_changed_fn(view._near_slider_changed)
    view.speed = SimpleNamespace(model=ui.SimpleFloatModel(2.))
    view.fov = SimpleNamespace(model=ui.SimpleFloatModel(75.))
    view.handed = SimpleNamespace(model=ui.SimpleBoolModel(False))
    for key in ("source", "mode", "up"):
        model = ui.SimpleIntModel(0)
        setattr(view, key, SimpleNamespace(model=SimpleNamespace(get_item_value_model=lambda value=model: value)))
    view.coordinates = [SimpleNamespace(model=ui.SimpleFloatModel(0.)) for _ in range(3)]
    view.status = SimpleNamespace(text="")
    view._sync_near(control.camera.near)
    return view, control, changes


def test_slider_updates_projection_without_committing_other_drafts_or_stopping_tracking(editor):
    view, control, changes = editor
    control.tracking_id = 55
    control.captured = True
    position = control.camera.position.copy()
    projection = control.camera.projection(800, 600)
    for value in range(1, 101):
        view.near_slider.model.set_value(value)
        assert view.near_slider.model.as_int == value
        assert control.camera.near == pytest.approx(near_plane_from_slider(value))
        assert view.near.model.as_float == pytest.approx(control.camera.near)
        assert control.camera.projection(800, 600)[2, 3] == pytest.approx(-control.camera.near)
    assert len(changes) == 100
    assert control.tracking_id == 55 and control.captured
    assert control.camera.fov == 60 and view.fov.model.as_float == 75
    assert control.speed_scale == 1 and view.speed.model.as_float == 2
    np.testing.assert_array_equal(control.camera.position, position)
    after = control.camera.projection(800, 600)
    after[2, 3] = projection[2, 3]
    np.testing.assert_array_equal(after, projection)


@pytest.mark.parametrize("near", [.01234567, .000001, 10000.])
def test_precise_entry_and_sync_do_not_quantize_camera(editor, near):
    view, control, changes = editor
    original = control.camera.near
    view.near.model.set_value(near)
    assert control.camera.near == original and not changes
    assert view.near_slider.model.as_int == near_plane_slider_position(view.near.model.as_float)
    view._apply()
    assert control.camera.near == pytest.approx(near)
    assert len(changes) == 1
    exact = control.camera.near
    view._sync()
    assert control.camera.near == exact and len(changes) == 1


def test_invalid_direct_draft_keeps_previous_projection(editor):
    view, control, changes = editor
    before = control.camera.projection(800, 600)
    view.near.model.set_value(0.)
    view._apply()
    assert "positive finite" in view.status.text and not changes
    np.testing.assert_array_equal(control.camera.projection(800, 600), before)


def test_direct_slider_integer_edits_cannot_escape_original_range(editor):
    view, control, changes = editor
    view.near_slider.model.set_value(-50)
    assert view.near_slider.model.as_int == 1 and control.camera.near == .0001
    view.near_slider.model.set_value(500)
    assert view.near_slider.model.as_int == 100 and control.camera.near == 1000
    assert len(changes) == 2
