# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection controls stay visually distinct from commands and tab navigation."""
import omni.ui as ui
from .style import BG, MUTED, rgba


class RadioOption:
    """Native focus/click target with a vector radio indicator and aligned label."""
    def __init__(self, title, selected, callback, **kwargs):
        enabled = kwargs.pop('enabled', True)
        with ui.ZStack():
            self.button = ui.RadioButton(text=title, clicked_fn=callback,
                style_type_name_override='Button',
                style={'Button.Label': {'color': 0}, 'Button.Label::Option': {'color': 0},
                       'Button.Label::OptionSelected': {'color': 0},
                       'Button.Label:disabled': {'color': 0},
                       'Button.Label::Option:disabled': {'color': 0},
                       'Button.Label::OptionSelected:disabled': {'color': 0}}, **kwargs)
            with ui.HStack(spacing=7, opaque_for_mouse_events=False):
                ui.Spacer(width=5)
                with ui.VStack(width=17):
                    ui.Spacer()
                    with ui.ZStack(height=17):
                        self.ring = ui.Circle(width=17,height=17,style={'background_color': BG,
                            'border_color': MUTED,'border_width':1.5})
                        with ui.Placer(offset_x=5,offset_y=5):
                            self.dot = ui.Circle(width=7,height=7,
                                style={'background_color':rgba(172,221,244)})
                    ui.Spacer()
                self.label = ui.Label(title,name='OptionLabel',word_wrap=True)
                ui.Spacer(width=4)
        self.enabled = enabled
        set_option(self, title, selected)

    def __getattr__(self, key):
        return getattr(self.button, key)

    @property
    def enabled(self):
        return self.button.enabled

    @enabled.setter
    def enabled(self, value):
        self.button.enabled = value
        self.label.enabled = value

    @property
    def checked(self):
        return self.button.checked

    @property
    def name(self):
        return self.button.name


def set_option(widget, title, selected):
    state = (title, bool(selected))
    if widget.__dict__.get('_presented') == state and widget.button.checked == bool(selected):
        return
    widget._presented = state
    widget.button.checked = bool(selected)
    widget.button.name = 'OptionSelected' if selected else 'Option'
    widget.button.text = title
    widget.label.text = title
    widget.label.name = 'SelectedOptionLabel' if selected else 'OptionLabel'
    widget.dot.visible = bool(selected)


def option(title, selected, callback, **kwargs):
    return RadioOption(title, selected, callback, **kwargs)


def toggle(lab, title, getter, setter):
    model = ui.SimpleBoolModel(bool(getter()))
    with ui.HStack(height=30, spacing=8):
        widget = ui.CheckBox(model, width=24)
        lab._label(title, 14).set_mouse_pressed_fn(
            lambda x, y, button, mods: model.set_value(not model.as_bool)
            if button == 0 and widget.enabled else None)
    model.add_value_changed_fn(lambda m: setter(m.as_bool) if m.as_bool != bool(getter()) else None)
    lab.toggle_models.append((model, getter))
    return widget
