# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A logarithmic drag control with an authoritative multiplier field."""
import math
import weakref

import omni.ui as ui

from ..design_tokens import LAYOUT
from .float_slider import SoftGlassFloatSlider


class LogScaleSlider:
    def __init__(self, model, *, tooltip="", height=28, identifier=""):
        self.model = model
        self.closed = self._syncing = False
        self.position_model = ui.SimpleFloatModel()
        self._sync()
        help_text = (tooltip + "\n" if tooltip else "") + (
            "Drag: 0.05x to 20x; center: 1x. Enter an exact multiplier in the field."
            "\nCtrl-click on the slider edits its normalized position (0 to 1).")
        with ui.HStack(height=height, spacing=LAYOUT["slider_gap"]) as self.widget:
            self.control = SoftGlassFloatSlider(self.position_model, min=0, max=1,
                step=.001, precision=4, height=height, tooltip=help_text,
                identifier=identifier)
            self.control.number.visible = False
            self.field = ui.FloatField(model, width=88, height=height, precision=8,
                name="mono", tooltip=tooltip, explicit_hover=True,
                identifier=f"{identifier}_value" if identifier else "")
            ui.Label("x", width=10, name="mono_small")
        reference = weakref.ref(self)
        def changed(_):
            owner = reference()
            if owner is not None:
                owner._sync()
        def moved(_):
            owner = reference()
            if owner is not None:
                owner._move()
        self._subscriptions = (model.add_value_changed_fn(changed),
                               self.position_model.add_value_changed_fn(moved))

    def _sync(self):
        if self.closed:
            return
        value = self.model.as_float
        position = (math.log(value, 20.) + 1.) / 2. if value > 0 else 0.
        self._syncing = True
        try:
            self.position_model.set_value(max(0., min(1., position)))
        finally:
            self._syncing = False

    def _move(self):
        if self.closed or self._syncing:
            return
        position = max(0., min(1., self.position_model.as_float))
        value = 20. ** (2. * position - 1.)
        self.model.set_value(1. if abs(value - 1.) < .03 else value)
        # The caller may reject the proposal and restore its accepted value.
        self._sync()

    def update(self):
        if not self.closed:
            self.control.update()

    def destroy(self):
        if self.closed:
            return
        self.closed = True
        changed, moved = self._subscriptions
        self.model.remove_value_changed_fn(changed)
        self.position_model.remove_value_changed_fn(moved)
        self.control.destroy()
