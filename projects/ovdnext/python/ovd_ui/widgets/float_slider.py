# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Soft Glass paint around the public native FloatSlider input control.

Call update() between completed UI frames. Model/size notifications only queue
paint, so fixed decorative widths cannot invalidate layout during widget draw.
The model, drag quantization and temporary numeric editor remain native OVUI.
"""
import math
import weakref

import omni.ui as ui

from .. import style
from ..design_tokens import LAYOUT, font_pixels


class SoftGlassFloatSlider:
    def __init__(self, model, *, min=0., max=1., step=None, precision=None,
                 display_precision=2, suffix="", height=28, tooltip="",
                 identifier=""):
        self.model = model
        self.minimum, self.maximum = float(min), float(max)
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum)
                and self.maximum > self.minimum):
            raise ValueError("Slider bounds must be finite and increasing")
        self.display_precision, self.suffix = display_precision, suffix
        self.pending, self.closed = True, False
        self.pointer_drag = self.numeric = False
        native_height = 24
        thumb = LAYOUT["slider_thumb"]
        track = LAYOUT["slider_track"]
        with ui.HStack(height=height, spacing=LAYOUT["slider_gap"]) as self.widget:
            with ui.VStack(spacing=0):
                ui.Spacer()
                with ui.ZStack(height=native_height):
                    # Independent clipping prevents stale paint widths imposing
                    # a minimum when the native control's parent shrinks.
                    with ui.Frame(height=native_height, horizontal_clipping=True,
                                  vertical_clipping=True):
                        with ui.ZStack():
                            with ui.Placer(offset_x=6, offset_y=(native_height-track)/2,
                                           width=1, height=track, stable_size=True,
                                           opaque_for_mouse_events=False) as self.track_host:
                                ui.Rectangle(opaque_for_mouse_events=False, style={
                                    "Rectangle": {"background_color": style.cl.pvd_elevated,
                                                  "border_radius": 2}})
                            with ui.Placer(offset_x=6, offset_y=(native_height-track)/2,
                                           width=1, height=track, stable_size=True,
                                           opaque_for_mouse_events=False) as self.fill_host:
                                ui.Rectangle(opaque_for_mouse_events=False, style={
                                    "Rectangle": {"background_color": style.cl.pvd_accent,
                                                  "border_radius": 2}})
                            with ui.Placer(offset_x=0, offset_y=(native_height-thumb)/2,
                                           width=thumb, height=thumb, stable_size=True,
                                           opaque_for_mouse_events=False) as self.thumb_host:
                                ui.Circle(radius=thumb/2, size_policy=ui.CircleSizePolicy.FIXED,
                                    opaque_for_mouse_events=False, style={"Circle": {
                                        "background_color": style.cl.pvd_text, "border_width": 0}})
                    with ui.Frame(height=native_height, horizontal_clipping=True,
                                  vertical_clipping=True):
                        options = {"min": min, "max": max}
                        if step is not None:
                            options["step"] = step
                        if precision is not None:
                            options["precision"] = precision
                        self.slider = ui.FloatSlider(model, **options, height=native_height,
                            format=" ", style=self._slider_style(False), explicit_hover=True, tooltip=tooltip,
                            identifier=identifier, opaque_for_mouse_events=False)
                ui.Spacer()
            self.number = ui.Label("", name="mono_small", width=LAYOUT["slider_number"],
                alignment=ui.Alignment.RIGHT_CENTER, style={"Label": {"color": style.cl.pvd_secondary}})
        self.slider.set_computed_content_size_changed_fn(self._changed)
        # These observe native input; opaque=False avoids a second InvisibleButton
        # stealing activation before FloatSlider draws its own native item.
        self.slider.set_mouse_pressed_fn(self._pointer_pressed)
        self.slider.set_mouse_released_fn(self._pointer_released)
        reference = weakref.ref(self)
        def changed(_):
            owner = reference()
            if owner is not None:
                owner._changed()
        def begin(_):
            owner = reference()
            if owner is not None:
                owner._set_numeric(not owner.pointer_drag)
        def end(_):
            owner = reference()
            if owner is not None:
                owner._set_numeric(False)
        self._subscriptions = (model.add_value_changed_fn(changed),
                               model.add_begin_edit_fn(begin), model.add_end_edit_fn(end))

    @staticmethod
    def _slider_style(numeric):
        return {"Slider": {"draw_mode": ui.SliderDrawMode.HANDLE,
            "background_color": style.cl.pvd_field if numeric else 0,
            "secondary_color": 0, "secondary_selected_color": 0, "border_width": 0,
            "border_radius": 8, "color": style.cl.pvd_text,
            "font": style.url.pvd_font_regular, "font_size": font_pixels(12), "padding": 4}}

    def _set_numeric(self, numeric):
        if self.closed or self.numeric == numeric:
            return
        self.numeric = numeric
        # Empty format restores the native formatter, including precision rules
        # for Ctrl-click, keyboard entry and native Enter/Escape handling.
        self.slider.format = "" if numeric else " "
        self.slider.style = self._slider_style(numeric)

    def _pointer_pressed(self, x, y, button, modifiers):
        if not self.closed and button == 0:
            self.pointer_drag = not bool(modifiers & 2)  # Public GLFW Ctrl bit.
            self._set_numeric(not self.pointer_drag)

    def _pointer_released(self, x, y, button, modifiers):
        if not self.closed and button == 0:
            self.pointer_drag = False

    def _changed(self):
        if not self.closed:
            self.pending = True

    def update(self):
        if self.closed or not self.pending:
            return
        width = self.slider.computed_content_width
        if width <= 12:
            return
        value = self.model.as_float
        ratio = ((value-self.minimum)/(self.maximum-self.minimum)) if math.isfinite(value) else 0.
        ratio = max(0., min(1., ratio))
        # Pinned standalone FloatSlider: 8px native grab + 2px end padding.
        # Only paint is clamped. Caller validation remains authoritative.
        span = width-12
        self.track_host.width = ui.Pixel(span)
        self.fill_host.width = ui.Pixel(span*ratio)
        self.fill_host.visible = ratio > 0
        self.thumb_host.offset_x = ui.Pixel(6+span*ratio-LAYOUT["slider_thumb"]/2)
        self.number.text = f"{value:.{self.display_precision}f}{self.suffix}"
        self.pending = False

    def destroy(self):
        """Detach observers without changing the caller-owned model."""
        if self.closed:
            return
        self.closed = True
        changed, begin, end = self._subscriptions
        self.model.remove_value_changed_fn(changed)
        self.model.remove_begin_edit_fn(begin)
        self.model.remove_end_edit_fn(end)
        self.slider.set_computed_content_size_changed_fn(None)
        self.slider.set_mouse_pressed_fn(None)
        self.slider.set_mouse_released_fn(None)
        self.pointer_drag = self.pending = False
