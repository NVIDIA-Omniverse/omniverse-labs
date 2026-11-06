# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def rgba(r, g, b, a=255):
    return r | (g << 8) | (b << 16) | (a << 24)


INK = rgba(231, 237, 239)
MUTED = rgba(146, 162, 173)
GREEN = rgba(165, 225, 85)
BG = rgba(15, 22, 29)
CARD = rgba(25, 35, 44)
STYLE = {
    "Window": {"background_color": BG, "border_width": 0},
    "Label": {"color": INK, "font_size": 16},
    "Button": {"background_color": rgba(36, 49, 59), "border_radius": 5, "margin": 0, "padding": 10},
    "Button:hovered": {"background_color": rgba(54, 72, 83)},
    "Button:pressed": {"background_color": rgba(69, 89, 99)},
    "Button.Label": {"color": INK, "font_size": 15},
    "Button::Utility": {"background_color": BG, "border_width": 0, "padding": 4},
    "Button::Utility:hovered": {"background_color": CARD},
    "Button.Label::Utility": {"color": MUTED, "font_size": 13},
    "Button::Info": {"background_color": BG, "border_width": 0, "border_radius": 7, "padding": 7},
    "Button::Info:hovered": {"background_color": CARD},
    "Button::InfoActive": {"background_color": CARD, "border_width": 0, "border_radius": 7, "padding": 7},
    "Button::Info:focused": {"border_color": INK, "border_width": 1},
    "Button::InfoActive:focused": {"border_color": INK, "border_width": 1},
    "Button::Utility:focused": {"border_color": INK, "border_width": 1},
    "Button::Primary": {"background_color": GREEN, "border_radius": 5},
    "Button::Primary:hovered": {"background_color": rgba(188, 244, 114)},
    "Button.Label::Primary": {"color": rgba(18, 30, 20), "font_size": 16},
    "Button::Busy": {"background_color": rgba(31,65,83), "border_radius": 5},
    "Button.Label::Busy": {"color": rgba(151,221,255), "font_size": 15},
    "Button::Busy:disabled": {"background_color": rgba(31,65,83)},
    "Button.Label::Busy:disabled": {"color": rgba(151,221,255)},
    "Button::Selected": {"background_color": rgba(46, 74, 92), "border_color": rgba(125, 197, 225), "border_width": 1, "border_radius": 5},
    "Button.Label::Selected": {"color": rgba(212, 239, 252), "font_size": 15},
    "Button:focused": {"border_color": rgba(212, 239, 252), "border_width": 2},
    "Button:disabled": {"background_color": rgba(27, 36, 44)},
    "Button.Label:disabled": {"color": rgba(128, 145, 155)},
    "Button::RoomNav": {"background_color": BG, "border_width": 0, "padding": 7},
    "Button::CurrentRoom": {"background_color": CARD, "border_color": rgba(125,197,225), "border_width": 1, "padding": 7},
    "Button::SettingsCog": {"background_color": BG, "border_width": 0},
    "Button::Tab": {"background_color": BG, "border_width": 0, "border_radius": 0, "padding": 4},
    "Button::ActiveTab": {"background_color": BG, "border_width": 0, "border_radius": 0, "padding": 4},
    "Button.Label::Tab": {"color": MUTED, "font_size": 14},
    "Button.Label::ActiveTab": {"color": INK, "font_size": 14},
    "Button::Option": {"background_color": BG, "border_width": 0, "border_radius": 0, "padding": 4},
    "Button::OptionSelected": {"background_color": BG, "border_width": 0, "border_radius": 0, "padding": 4},
    "Button::OptionSelected:checked": {"background_color": BG},
    "Button::Option:hovered": {"background_color": CARD},
    "Button::OptionSelected:hovered": {"background_color": CARD},
    "Button::Option:disabled": {"background_color": BG, "border_width": 0},
    "Button::OptionSelected:disabled": {"background_color": BG, "border_width": 0},
    "Button.Label::Option": {"color": MUTED, "font_size": 14},
    "Button.Label::OptionSelected": {"color": rgba(172,221,244), "font_size": 14},
    "RadioButton": {"background_color": BG, "border_width": 0, "border_radius": 0, "padding": 4},
    "RadioButton:hovered": {"background_color": CARD},
    "RadioButton::OptionSelected": {"background_color": BG, "border_width": 0},
    "RadioButton.Label": {"color": MUTED, "font_size": 14},
    "RadioButton.Label::OptionSelected": {"color": rgba(172, 221, 244), "font_size": 14},
    "RadioButton:focused": {"border_width": 1, "border_color": INK},
    "RadioButton.Label:disabled": {"color": rgba(128, 145, 155)},
    "Label::OptionLabel": {"color": MUTED, "font_size": 14},
    "Label::SelectedOptionLabel": {"color": rgba(172,221,244), "font_size": 14},
    "Label::OptionLabel:disabled": {"color": rgba(128,145,155)},
    "Label::SelectedOptionLabel:disabled": {"color": rgba(128,145,155)},
    "ScrollingFrame": {"background_color": BG},
    "ScrollBar": {"background_color": BG, "secondary_color": rgba(52, 67, 77), "border_radius": 4},
    "Separator": {"color": rgba(49, 64, 75)},
}


def scaled_style(scale):
    style = {name: dict(values) for name, values in STYLE.items()}
    for values in style.values():
        if 'font_size' in values:
            values['font_size'] *= scale
    return style
