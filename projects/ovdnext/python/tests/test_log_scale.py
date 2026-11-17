# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logarithmic drag positions leave accepted multiplier models authoritative."""
import gc
import math
import weakref

import omni.ui as ui
import pytest

from ovd_ui.widgets.log_scale import LogScaleSlider


@pytest.fixture
def make_control():
    controls = []
    def make(model):
        control = LogScaleSlider(model)
        controls.append(control)
        return control
    yield make
    for control in controls:
        control.destroy()


@pytest.mark.parametrize("position, expected", [
    (0., .05), (.25, .22360679774997896), (.5, 1.),
    (.75, 4.47213595499958), (1., 20.),
])
def test_drag_maps_geometric_range_with_neutral_center(make_control, position, expected):
    model = ui.SimpleFloatModel(2.)
    control = make_control(model)
    control.position_model.set_value(position)
    assert model.as_float == pytest.approx(expected)


@pytest.mark.parametrize("position", [.496, .499, .501, .504])
def test_drag_snaps_within_three_percent_of_neutral(make_control, position):
    model = ui.SimpleFloatModel(2.)
    control = make_control(model)
    control.position_model.set_value(position)
    assert model.as_float == 1.
    assert control.position_model.as_float == .5


@pytest.mark.parametrize("position, expected", [
    (.49, .9418449208830277), (.51, 1.0617459178549786),
    (-1., .05), (2., 20.),
])
def test_drag_outside_snap_window_is_continuous_and_bounded(make_control, position, expected):
    model = ui.SimpleFloatModel(2.)
    control = make_control(model)
    control.position_model.set_value(position)
    assert model.as_float == pytest.approx(expected)
    assert 0 <= control.position_model.as_float <= 1


@pytest.mark.parametrize("value", [3.1517279, .0001, 100., 0., 1.02])
def test_construction_paint_and_numeric_entry_preserve_exact_values(make_control, value):
    model = ui.SimpleFloatModel(value)
    changes = []
    model.add_value_changed_fn(lambda m: changes.append(m.as_float))
    control = make_control(model)
    control.update()
    assert model.as_float == value and changes == []
    assert control.model is model and control.field.model is model
    assert not control.control.number.visible
    control.field.model.set_value(3.1517279)
    assert model.as_float == 3.1517279


@pytest.mark.parametrize("value, position", [
    (.0001, 0.), (.22360679774997896, .25), (1., .5),
    (4.47213595499958, .75), (100., 1.), (0., 0.),
])
def test_external_model_changes_refresh_position_without_rewriting_value(make_control, value, position):
    model = ui.SimpleFloatModel(2.)
    control = make_control(model)
    model.set_value(value)
    assert model.as_float == value
    assert control.position_model.as_float == pytest.approx(position)


@pytest.mark.parametrize("subscribe_first", [True, False])
def test_rejected_drag_and_numeric_entry_restore_accepted_position(make_control, subscribe_first):
    model = ui.SimpleFloatModel(1.)
    def validate(m):
        if not math.isfinite(m.as_float) or not 0 <= m.as_float <= 5:
            m.set_value(1.)
    if subscribe_first:
        model.add_value_changed_fn(validate)
    control = make_control(model)
    if not subscribe_first:
        model.add_value_changed_fn(validate)
    control.position_model.set_value(1.)
    assert model.as_float == 1. and control.position_model.as_float == .5
    control.field.model.set_value(float("nan"))
    assert model.as_float == 1. and control.position_model.as_float == .5


def test_destroy_detaches_both_model_directions_and_releases_owner():
    control = LogScaleSlider(ui.SimpleFloatModel(1.))
    model, position = control.model, control.position_model
    control.destroy()
    control.destroy()
    assert control.closed and control.control.closed
    model.set_value(2.)
    assert position.as_float == .5
    position.set_value(1.)
    assert model.as_float == 2.
    control.update()
    reference = weakref.ref(control)
    del control
    gc.collect()
    assert reference() is None
