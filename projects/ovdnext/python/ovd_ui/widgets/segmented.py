# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact mutually exclusive choices with the Soft Glass tab treatment."""
import omni.ui as ui


class SegmentedControl:
    def __init__(self, choices, *, value=0, on_changed=None, height=36):
        """Choices are ``(integer value, label, width)`` tuples."""
        self.model=ui.SimpleIntModel(value)
        self.buttons={}
        self.frame=ui.ZStack(width=sum(width for _,_,width in choices)+6+2*(len(choices)-1),height=height)
        with self.frame:
            ui.Rectangle(name="subtle")
            with ui.VStack(spacing=0):
                ui.Spacer(height=3)
                with ui.HStack(spacing=2):
                    ui.Spacer(width=1)
                    for identity,label,width in choices:
                        self.buttons[identity]=ui.Button(label,width=width,
                            name="tab_active" if identity==value else "tab",
                            clicked_fn=lambda i=identity:self.model.set_value(i))
                    ui.Spacer(width=1)
                ui.Spacer(height=3)
        self._callback=on_changed
        self.model.add_value_changed_fn(self._changed)

    def _changed(self, model):
        for identity,button in self.buttons.items():
            button.name="tab_active" if identity==model.as_int else "tab"
        if self._callback:self._callback(model.as_int)

    @property
    def enabled(self): return self.frame.enabled

    @enabled.setter
    def enabled(self, value):
        self.frame.enabled=value
        for button in self.buttons.values():button.enabled=value
