# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep native models and their validation authoritative beneath slider paint."""
import gc
import weakref

import omni.ui as ui
import pytest

from ovd_ui.widgets.float_slider import SoftGlassFloatSlider


def test_construction_and_unlaid_paint_preserve_off_step_value():
    model = ui.SimpleFloatModel(1.013)
    changes = []
    model.add_value_changed_fn(lambda m: changes.append(m.as_float))
    control = SoftGlassFloatSlider(model, min=0, max=2, step=.02, precision=2)
    assert control.slider.model is model
    control.update()  # No computed layout yet; must not touch the model.
    control.destroy()
    assert changes == [] and model.as_float == 1.013


def test_unspecified_step_and_precision_retain_native_defaults():
    model = ui.SimpleFloatModel(73.12345)
    original = ui.FloatSlider(model, min=0, max=100)
    control = SoftGlassFloatSlider(model, min=0, max=100, suffix="%")
    assert (control.slider.step, control.slider.precision) == (original.step, original.precision)
    control.destroy()


def test_rejected_value_restores_same_model_and_destroy_removes_observers():
    model = ui.SimpleFloatModel(1.013)
    control = SoftGlassFloatSlider(model, min=0, max=2)
    def validate(m):
        if not 0 <= m.as_float <= 2:
            m.set_value(1.013)
    validation = model.add_value_changed_fn(validate)
    model.set_value(3.)
    assert model.as_float == 1.013 and control.pending
    control.destroy()
    control.destroy()
    model.set_value(1.5)
    assert not control.pending and not control.numeric
    reference = weakref.ref(control)
    del control
    gc.collect()
    assert reference() is None
    model.remove_value_changed_fn(validation)


@pytest.mark.parametrize("bounds", [(0, 0), (2, 1), (0, float("inf")), (float("nan"), 2)])
def test_invalid_bounds_fail_before_subscribing(bounds):
    with pytest.raises(ValueError, match="bounds"):
        SoftGlassFloatSlider(ui.SimpleFloatModel(1.), min=bounds[0], max=bounds[1])
