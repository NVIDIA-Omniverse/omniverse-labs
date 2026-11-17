# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single style startup; OVUI color shades propagate to all panels."""
import omni.ui as ui
from types import SimpleNamespace
from omni.ui import color as cl
from omni.ui import constant as fl
from omni.ui import url
from ovui_widgets.common.style import palette as _ovui_palette
from .design_tokens import COLORS, FONTS, LAYOUT, font_pixels

for name, (dark, light) in COLORS.items():
    cl.shade(cl(dark), light=cl(light), name=f"pvd_{name}")
for name, path in FONTS.items():
    setattr(url, f"pvd_font_{name}", str(path))

layout = SimpleNamespace(**{f"pvd_{name}": value for name, value in LAYOUT.items()})
for name, value in vars(layout).items():
    setattr(fl, name, value)

STYLES = {
    "Window": {"background_color": cl.pvd_background, "color": cl.pvd_text,
               "border_color": cl.pvd_border, "padding": fl.pvd_zero},
    "Rectangle": {"background_color": cl.pvd_surface},
    "Rectangle::island": {"background_color": cl.pvd_island,
                          "border_radius": fl.pvd_island_radius,
                          "border_color": cl.pvd_border, "border_width": fl.pvd_border_width,
                          # Current standalone shadow draws a square apron.
                          # Keep the CSS token, omit the observed artifact.
                          "shadow_color": cl.pvd_transparent, "shadow_thickness": fl.pvd_zero,
                          "shadow_offset_x": fl.pvd_zero, "shadow_offset_y": fl.pvd_shadow_offset_y},
    "Rectangle::raised": {"background_color": cl.pvd_elevated, "border_radius": fl.pvd_field_radius},
    "Rectangle::workspace_rail": {"background_color": cl.pvd_workspace_rail,
                                   "border_radius": fl.pvd_workspace_rail_radius},
    "Rectangle::field": {"background_color": cl.pvd_field, "border_radius": fl.pvd_field_radius},
    "Rectangle::hierarchy_search": {"background_color": cl.pvd_field,
                                     "border_radius": fl.pvd_field_radius,
                                     "border_color": cl.pvd_border, "border_width": fl.pvd_border_width},
    "Rectangle::hierarchy_search:selected": {"border_color": cl.pvd_accent},
    "Circle::hierarchy_search": {"background_color": cl.pvd_transparent,
                                  "border_color": cl.pvd_muted, "border_width": fl.pvd_search_icon_stroke},
    "Rectangle::subtle": {"background_color": cl.pvd_subtle, "border_radius": fl.pvd_field_radius},
    "Rectangle::splitter": {"background_color": cl.pvd_accent, "border_radius": fl.pvd_border_width},
    "Rectangle::transparent": {"background_color": cl.pvd_transparent},
    "Rectangle::object_selection_middle": {"background_color": cl.pvd_transparent,
                                            "border_radius": fl.pvd_zero},
    "Rectangle::object_selection_cap": {"background_color": cl.pvd_transparent,
                                         "border_radius": fl.pvd_row_radius},
    "Rectangle::object_selection_middle:selected": {"background_color": cl.pvd_accent_surface},
    "Rectangle::object_selection_cap:selected": {"background_color": cl.pvd_accent_surface},
    "Rectangle::profile_lane": {"background_color": cl.pvd_field},
    "Rectangle::profile_scope": {"background_color": cl.pvd_muted},
    "Rectangle::profile_selected": {"background_color": cl.pvd_accent},
    "Rectangle::profile_range": {"background_color": cl.pvd_profile_range},
    "Rectangle::profile_marker": {"background_color": cl.pvd_warning},
    "Rectangle::message_error": {"background_color": cl.pvd_error_surface},
    "Rectangle::message_warning": {"background_color": cl.pvd_warning_surface},
    "Rectangle::message_info": {"background_color": cl.pvd_transparent},
    "Rectangle::message_error:selected": {"background_color": cl.pvd_accent_surface},
    "Rectangle::message_warning:selected": {"background_color": cl.pvd_accent_surface},
    "Rectangle::message_info:selected": {"background_color": cl.pvd_accent_surface},
    "Label": {"color": cl.pvd_text, "font_size": fl.pvd_font},
    "Label::muted": {"color": cl.pvd_muted, "font_size": fl.pvd_small_font},
    "Label::hierarchy_search": {"color": cl.pvd_muted, "font_size": fl.pvd_font},
    "Label::profile_text": {"color": cl.pvd_field, "font_size": fl.pvd_small_font},
    "Label::brand": {"color": cl.pvd_badge_text, "font_size": fl.pvd_font, "font": url.pvd_font_semibold},
    "Label::title": {"font_size": fl.pvd_heading_font, "font": url.pvd_font_medium},
    "Label::filename": {"font_size": fl.pvd_font, "font": url.pvd_font_medium},
    "Label::accent": {"color": cl.pvd_link},
    "Label::secondary": {"color": cl.pvd_secondary},
    "Label::eyebrow": {"color": cl.pvd_muted, "font_size": fl.pvd_small_font},
    "Label::mono": {"font": url.pvd_font_mono, "font_size": fl.pvd_font},
    "Label::mono_small": {"font": url.pvd_font_mono, "font_size": fl.pvd_small_font},
    "Label::warning": {"color": cl.pvd_warning, "font_size": fl.pvd_small_font},
    "Label::error": {"color": cl.pvd_error, "font_size": fl.pvd_small_font},
    "Label::message_error": {"color": cl.pvd_error},
    "Label::message_warning": {"color": cl.pvd_warning},
    "Label::recording_displayed": {"font": url.pvd_font_semibold, "font_size": fl.pvd_font},
    "Rectangle::recording_row": {"background_color": cl.pvd_transparent},
    "Rectangle::recording_alternate": {"background_color": cl.pvd_elevated},
    "Rectangle::recording_row:selected": {"background_color": cl.pvd_accent_surface},
    "Rectangle::recording_alternate:selected": {"background_color": cl.pvd_accent_surface},
    "Label::section": {"font_size": fl.pvd_section_font, "font": url.pvd_font_medium},
    "Label::object_name": {"color": cl.pvd_text, "font_size": fl.pvd_font},
    "Label::object_metadata": {"color": cl.pvd_muted, "font_size": fl.pvd_small_font},
    "Label::object_name:selected": {"color": cl.pvd_selected_text},
    "Triangle::object_chevron": {"background_color": cl.pvd_muted},
    "Button": {"background_color": cl.pvd_elevated, "border_radius": fl.pvd_radius,
               "padding": fl.pvd_compact_padding, "margin": fl.pvd_zero},
    "Button:hovered": {"background_color": cl.pvd_hover},
    "Button:pressed": {"background_color": cl.pvd_accent_surface},
    "Button:disabled": {"background_color": cl.pvd_surface},
    "Button:checked": {"background_color": cl.pvd_accent_surface},
    "Button:selected": {"background_color": cl.pvd_accent_surface},
    "Button.Label": {"color": cl.pvd_text, "font_size": fl.pvd_font, "font": url.pvd_font_medium},
    "Button.Image": {"color": cl.pvd_text},
    "Button.Image:disabled": {"color": cl.pvd_disabled},
    "Button.Label:disabled": {"color": cl.pvd_disabled},
    "Button::primary": {"background_color": cl.pvd_accent},
    "Button.Label::primary": {"color": cl.pvd_on_accent, "font": url.pvd_font_semibold},
    "Button.Image::primary": {"color": cl.pvd_on_accent},
    "Button.Image::primary:disabled": {"color": cl.pvd_disabled},
    "Button::primary:hovered": {"background_color": cl.pvd_accent},
    "Button::primary:pressed": {"background_color": cl.pvd_accent_surface},
    "Button::ghost": {"background_color": cl.pvd_transparent},
    "Button.Label::ghost": {"color": cl.pvd_secondary},
    "Button.Image::ghost": {"color": cl.pvd_secondary},
    "Button::ghost:hovered": {"background_color": cl.pvd_hover},
    "Button::link": {"background_color": cl.pvd_transparent, "padding": fl.pvd_zero},
    "Button.Label::link": {"color": cl.pvd_link, "font_size": fl.pvd_small_font,
                            "font": url.pvd_font_regular},
    "Button::link:hovered": {"background_color": cl.pvd_hover},
    "Button::quiet": {"background_color": cl.pvd_transparent},
    "Button.Label::quiet": {"color": cl.pvd_muted},
    "Button.Image::quiet": {"color": cl.pvd_muted},
    "Button::quiet:hovered": {"background_color": cl.pvd_hover},
    "Button::outline": {"background_color": cl.pvd_transparent,
                        "border_color": cl.pvd_outline, "border_width": fl.pvd_border_width},
    "Button.Image::outline": {"color": cl.pvd_secondary},
    "Button::workspace_frame": {"background_color": cl.pvd_island_source,
                                 "border_radius": fl.pvd_field_radius,
                                 "border_color": cl.pvd_border, "border_width": fl.pvd_border_width},
    "Button.Label::workspace_frame": {"color": cl.pvd_secondary, "font": url.pvd_font_regular},
    "Button::workspace_frame:hovered": {"border_color": cl.pvd_outline},
    "Button::workspace_frame:pressed": {"background_color": cl.pvd_accent_surface},
    "Button.Label::workspace_frame:disabled": {"color": cl.pvd_disabled},
    "Button::workspace_detach": {"background_color": cl.pvd_workspace_detach,
                                  "border_color": cl.pvd_workspace_detach_border,
                                  "border_width": fl.pvd_border_width},
    "Button.Label::workspace_detach": {"font": url.pvd_font_regular},
    "Button::workspace_detach:hovered": {"border_color": cl.pvd_outline},
    "Button::workspace_detach:pressed": {"background_color": cl.pvd_accent_surface},
    "Button::tab": {"background_color": cl.pvd_transparent, "border_radius": fl.pvd_tab_radius},
    "Button.Label::tab": {"color": cl.pvd_muted},
    "Button.Image::tab": {"color": cl.pvd_accent},
    "Button::tab_active": {"background_color": cl.pvd_elevated, "border_radius": fl.pvd_tab_radius},
    "Button.Label::tab_active": {"color": cl.pvd_text},
    "Button::workspace_tab": {"background_color": cl.pvd_transparent,
                               "border_radius": fl.pvd_workspace_tab_radius},
    "Button::workspace_tab:disabled": {"background_color": cl.pvd_transparent},
    "Button.Label::workspace_tab": {"color": cl.pvd_muted, "font": url.pvd_font_regular},
    "Button.Image::workspace_tab": {"color": cl.pvd_accent},
    "Button::workspace_tab_active": {"background_color": cl.pvd_workspace_tab_active,
                                      "border_radius": fl.pvd_workspace_tab_radius},
    "Button.Label::workspace_tab_active": {"color": cl.pvd_text, "font": url.pvd_font_regular},
    "Button::badge": {"background_color": cl.pvd_accent_surface},
    "Button.Label::badge": {"color": cl.pvd_badge_text, "font": url.pvd_font_semibold},
    "Button.Image::badge": {"color": cl.pvd_badge_text},
    "Button.Image::accent": {"color": cl.pvd_link},
    "Field": {"background_color": cl.pvd_field, "color": cl.pvd_text,
              "border_radius": fl.pvd_field_radius, "padding": fl.pvd_compact_padding,
              "border_color": cl.pvd_border, "border_width": fl.pvd_border_width,
              "font_size": fl.pvd_font},
    "Field:focused": {"border_color": cl.pvd_accent},
    "Field:disabled": {"color": cl.pvd_disabled},
    "Field::mono": {"font": url.pvd_font_mono, "font_size": fl.pvd_font},
    "Field::hierarchy_search": {"background_color": cl.pvd_transparent,
                                 "border_width": fl.pvd_zero},
    "ComboBox": {"background_color": cl.pvd_field, "color": cl.pvd_text,
                 "secondary_color": cl.pvd_elevated,
                 "secondary_background_color": cl.pvd_surface,
                 "selected_color": cl.pvd_accent_surface,
                 "border_color": cl.pvd_border, "border_width": fl.pvd_zero,
                 "border_radius": fl.pvd_radius, "font_size": fl.pvd_font},
    "ComboBox:hovered": {"border_color": cl.pvd_muted},
    "ComboBox:pressed": {"border_color": cl.pvd_accent},
    "ComboBox:disabled": {"color": cl.pvd_disabled},
    "ComboBox.Label": {"color": cl.pvd_text, "font_size": fl.pvd_font},
    "CheckBox": {"background_color": cl.pvd_field, "color": cl.pvd_accent,
                 "border_color": cl.pvd_border, "border_width": fl.pvd_border_width,
                 "border_radius": fl.pvd_radius},
    "CheckBox:hovered": {"border_color": cl.pvd_accent},
    "CheckBox:disabled": {"color": cl.pvd_disabled},
    "Slider": {"background_color": cl.pvd_elevated, "color": cl.pvd_accent,
               "secondary_color": cl.pvd_accent_surface, "border_radius": fl.pvd_radius},
    "Slider.Label": {"color": cl.pvd_text},
    "TreeView": {"background_color": cl.pvd_surface, "background_selected_color": cl.pvd_accent_surface,
                 "secondary_color": cl.pvd_border},
    "TreeView.Item": {"color": cl.pvd_text, "font_size": fl.pvd_font},
    "TreeView.Item:selected": {"color": cl.pvd_selected_text},
    "TreeView.Header": {"background_color": cl.pvd_elevated, "color": cl.pvd_muted,
                        "font_size": fl.pvd_small_font},
    "TreeView:selected": {"background_color": cl.pvd_accent_surface},
    "TreeView::objects": {"secondary_color": cl.pvd_transparent},
    "TreeView::objects:selected": {"background_color": cl.pvd_transparent},
    "ScrollingFrame": {"background_color": cl.pvd_transparent, "secondary_color": cl.pvd_outline,
                       "scrollbar_size": fl.pvd_scrollbar},
    "CollapsableFrame": {"background_color": cl.pvd_surface, "secondary_color": cl.pvd_elevated,
                         "color": cl.pvd_text, "padding": fl.pvd_spacing},
    "Plot": {"background_color": cl.pvd_field, "color": cl.pvd_accent,
             "secondary_color": cl.pvd_text, "selected_color": cl.pvd_text,
             "border_color": cl.pvd_border, "border_width": fl.pvd_border_width,
             "background_selected_color": cl.pvd_surface, "border_radius": fl.pvd_radius},
    "Separator": {"color": cl.pvd_border},
    "Menu.Window": {"background_color": cl.pvd_surface, "border_radius": fl.pvd_radius,
                    "border_color": cl.pvd_border, "border_width": fl.pvd_border_width},
    "Menu.Item": {"color": cl.pvd_text, "margin_height": fl.pvd_spacing,
                  "font_size": fl.pvd_font, "background_selected_color": cl.pvd_elevated},
    "Menu.Item:hovered": {"background_color": cl.pvd_elevated,
                          "background_selected_color": cl.pvd_elevated},
    # MenuItem's hit surface retains the legacy native type; its text uses
    # Menu.Item. Both public selectors are needed for the same hover color.
    "MenuItem": {"background_selected_color": cl.pvd_elevated},
    "MenuItem:hovered": {"background_selected_color": cl.pvd_elevated},
    "Menu.Item:disabled": {"color": cl.pvd_disabled},
    "Menu.Separator": {"color": cl.pvd_border},
    "MenuBar": {"background_color": cl.pvd_background, "color": cl.pvd_text},
    "MenuBar.Item": {"color": cl.pvd_text, "font_size": fl.pvd_font,
                     "background_selected_color": cl.pvd_elevated},
    "MenuBar.Item:hovered": {"background_color": cl.pvd_elevated,
                             "background_selected_color": cl.pvd_elevated},
    "Tooltip": {"background_color": cl.pvd_elevated, "color": cl.pvd_text,
                "font_size": fl.pvd_small_font},
    # Window resolves this width into its native border; Rectangle::dialog owns
    # the themed outline, avoiding a second dark rim around the rounded surface.
    "Dialog": {"background_color": cl.pvd_surface, "color": cl.pvd_text,
               # Native modal dimming otherwise uses a gray wash in both themes.
               "secondary_background_color": cl.pvd_shadow,
               "border_color": cl.pvd_border, "border_width": fl.pvd_zero,
               "border_radius": fl.pvd_island_radius, "padding": fl.pvd_zero},
    "Dialog.SectionTitle": {"color": cl.pvd_text, "font_size": fl.pvd_section_font},
    "Dialog.Message": {"color": cl.pvd_text, "font_size": fl.pvd_font},
    "Rectangle::dialog": {"background_color": cl.pvd_surface,
                          "border_color": cl.pvd_border, "border_width": fl.pvd_border_width,
                          "border_radius": fl.pvd_island_radius},
}

# An empty selector is OVUI's *global override*, not a CSS-like inherited base:
# using it for fonts or colors overrides named variants, including primary
# labels and monospace values. Set actual type defaults instead.
for _text_type in ("Label", "Field", "ComboBox", "ComboBox.Label", "Slider.Label",
                   "TreeView.Item", "TreeView.Header", "Menu.Item", "MenuBar.Item",
                   "Tooltip", "Dialog.SectionTitle", "Dialog.Message", "CollapsableFrame"):
    STYLES[_text_type].setdefault("font", url.pvd_font_regular)
STYLES["Dialog.SectionTitle"]["font"] = url.pvd_font_medium

for _selector, _properties in STYLES.items():
    if "font_size" in _properties:
        _css_token = str(_properties["font_size"])
        _face = "mono" if str(_properties.get("font", "")).startswith("pvd_font_mono") else "regular"
        _raster_token = f"{_css_token}_{_face}_pixels"
        setattr(fl, _raster_token, font_pixels(LAYOUT[_css_token.removeprefix("pvd_")], _face))
        _properties["font_size"] = getattr(fl, _raster_token)

# The shared OVUI confirmation service uses semantic button categories.
# They do not inherit Button, so provide complete styles for each category.
for _button_type, _variant in (("OKButton", "primary"), ("CancelButton", "quiet")):
    STYLES[_button_type] = {**STYLES["Button"], **STYLES[f"Button::{_variant}"]}
    STYLES[f"{_button_type}.Label"] = {
        **STYLES["Button.Label"], **STYLES.get(f"Button.Label::{_variant}", {})}
    for _state in ("hovered", "pressed", "disabled"):
        STYLES[f"{_button_type}:{_state}"] = STYLES[f"Button:{_state}"]


def startup(theme):
    for path in FONTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required Soft Glass font is missing: {path}")
    ui.style.default = STYLES
    set_theme(theme)


def set_theme(theme):
    cl.set_shade("light" if theme == "light" else "default")


def style_dialog(window):
    """Apply dialog defaults to the frame that actually paints OVUI chrome.

    Some released OVUI wheels ignore Window's style_type_name_override kwarg;
    the owned frame is the documented styling surface used by Window itself.
    """
    window.frame.style_type_name_override = "Dialog"
    window.frame.set_style({"Dialog": STYLES["Dialog"]})
    window.padding_x = layout.pvd_zero
    window.padding_y = layout.pvd_zero
