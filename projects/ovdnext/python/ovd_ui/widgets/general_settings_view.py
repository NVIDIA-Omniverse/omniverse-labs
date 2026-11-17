# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shared draft for rendering, property display and profile appearance."""
from dataclasses import replace
import omni.ui as ui

from ..array_settings import ArrayDisplaySettings
from ..camera import normalize_key
from ..formatting import ValueFormat
from ..general_settings import GeneralSettings, integer_text
from ..palette import color_hex, parse_color
from ..render_settings import RenderingSettings
from ..style import layout as fl, style_dialog
from .render_settings_view import RenderingSettingsView


class GeneralSettingsView(RenderingSettingsView):
    SECTIONS = (("rendering", "Rendering"), ("properties", "Property display"), ("profile", "Profile"))
    FORMATS = ("auto", "fixed", "scientific")

    def __init__(self, preferences, on_apply, *, sample_info=None):
        self.accepted = GeneralSettings.from_preferences(preferences)
        super().__init__(self.accepted.rendering, on_apply, preferences=preferences, sample_info=sample_info)
        self.property_controls = {}
        self.sections, self.tabs = {}, {}
        self.section = "rendering"
        self._syncing_color = False

    def show(self, section="rendering"):
        opening = self.window is None or not self.window.visible
        if self.window is None:
            self._build()
        if opening:
            self.accepted = GeneralSettings.from_preferences(self.preferences)
            self.settings = self.accepted.rendering
            self._sync()
        self._select_section(section)
        self.window.visible = True
        self.window.focus()

    def _select_section(self, section):
        if section not in self.sections:
            section = "rendering"
        self.section = section
        for key, panel in self.sections.items():
            panel.visible = key == section
            self.tabs[key].name = "tab_active" if key == section else "tab"

    def set_settings(self, settings, *, fields=()):
        """External toolbar changes update their fields without losing drafts."""
        super().set_settings(settings, fields=fields)
        self.accepted = replace(self.accepted, rendering=settings)

    def set_profile_color(self, color):
        parse_color(color)
        if color == self.accepted.profile_function_color:
            return
        self.accepted = replace(self.accepted, profile_function_color=color)
        if hasattr(self, "profile_color"):
            self.profile_color.model.set_value(color)

    def _fill(self, draft):
        self._write_controls(draft.rendering, fields=self.controls)
        values = {"float_format": self.FORMATS.index(draft.value_format.float_format),
                  "float_precision": str(draft.value_format.precision),
                  "show_hex": draft.value_format.show_hex,
                  "array_group_size": str(draft.arrays.group_size),
                  "max_array_to_load": str(draft.arrays.max_array_to_load)}
        for key, value in values.items():
            model = self.property_controls[key].model
            if hasattr(model, "get_item_value_model"):
                model = model.get_item_value_model()
            model.set_value(value)
        self.profile_color.model.set_value(draft.profile_function_color)

    def _sync(self):
        self._fill(self.accepted)
        self._status_error = False
        self.status.text = "Changes stay in this draft until you apply them."

    def refresh(self):
        if self.window is not None and self.window.visible and self.sample_info:
            self.attachment_status.text = self.sample_info()

    def _defaults(self):
        self._fill(GeneralSettings())
        self._status_error = False
        self.status.text = "Default values are in the draft. Apply to save them."

    def _candidate(self):
        controls = self.property_controls
        values = self._read_values()
        mode = controls["float_format"].model.get_item_value_model().as_int
        if mode not in range(len(self.FORMATS)):
            raise ValueError("Choose Auto, Fixed or Scientific float format")
        return GeneralSettings(
            rendering=RenderingSettings(**values),
            arrays=ArrayDisplaySettings(
                group_size=integer_text(controls["array_group_size"].model.as_string, "Group size"),
                max_array_to_load=integer_text(controls["max_array_to_load"].model.as_string, "Array expansion limit")),
            value_format=ValueFormat(
                integer_text(controls["float_precision"].model.as_string, "Float precision"),
                self.FORMATS[mode], controls["show_hex"].model.as_bool),
            profile_function_color=self.profile_color.model.as_string)

    def _apply(self, defaults=False):
        if defaults:
            self._defaults()
            return False
        try:
            candidate = self._candidate()
            candidate.save(self.preferences)
        except (ValueError, OSError) as error:
            self._status_error = True
            self.status.text = str(error)
            return False
        self.accepted = candidate
        self.settings = candidate.rendering
        self.on_apply(candidate)
        self._sync()
        self.status.text = "Settings saved."
        return True

    def close(self):
        if self.window is not None:
            self.window.visible = False

    def key(self, key, modifiers, pressed):
        if pressed and self.window.focused and modifiers & 15 == 0:
            if normalize_key(key) == 256:
                self.close()

    def _property_row(self, label, key, *, choices=None, boolean=False, tooltip=""):
        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
            ui.Label(label, width=210, explicit_hover=True, tooltip=tooltip)
            if choices:
                control = ui.ComboBox(0, *choices, explicit_hover=True, tooltip=tooltip)
            elif boolean:
                control = ui.CheckBox(width=24, explicit_hover=True, tooltip=tooltip)
                ui.Spacer()
            else:
                control = ui.StringField(name="mono", explicit_hover=True, tooltip=tooltip)
            self.property_controls[key] = control

    def _properties(self):
        self._property_row("Float format", "float_format", choices=("Auto", "Fixed", "Scientific"))
        self._property_row("Precision (0 to 25)", "float_precision",
            tooltip="Digits after the decimal point. Auto uses scientific notation for large recorded values.")
        self._property_row("Hexadecimal integers", "show_hex", boolean=True)
        ui.Spacer(height=fl.pvd_spacing)
        ui.Label("Array display", name="section", height=fl.pvd_control)
        self._property_row("Group size (10 to 9999)", "array_group_size",
            tooltip="Items per group in the browser and property inspector. Large groups show one page at a time.")
        self._property_row("Expansion limit (0 to 99999)", "max_array_to_load",
            tooltip="An array can expand only when its full element count is within this limit. Zero suppresses every nonempty property array.")
        ui.Label("Larger arrays keep their recorded count. Object relationships remain available in the browser.",
            name="muted", word_wrap=True, height=fl.pvd_row * 2)

    def _color_text_changed(self, model):
        if self._syncing_color:
            return
        try:
            rgba = parse_color(model.as_string)
        except ValueError:
            return
        self._syncing_color = True
        try:
            for item, value in zip(self.profile_picker.model.get_item_children(), rgba):
                self.profile_picker.model.get_item_value_model(item).set_value(value)
        finally:
            self._syncing_color = False

    def _color_picker_changed(self, _model):
        if self._syncing_color:
            return
        model = self.profile_picker.model
        rgba = [model.get_item_value_model(item).as_float for item in model.get_item_children()]
        self._syncing_color = True
        try:
            self.profile_color.model.set_value(color_hex(rgba))
        finally:
            self._syncing_color = False

    def _profile(self):
        ui.Label("Profile function color", name="section", height=fl.pvd_control)
        with ui.HStack(height=fl.pvd_control * 1.5, spacing=fl.pvd_spacing):
            self.profile_picker = ui.ColorWidget(width=fl.pvd_control * 1.5, height=fl.pvd_control)
            self.profile_color = ui.StringField(name="mono", height=fl.pvd_control,
                explicit_hover=True, tooltip="RGB/ARGB hex value or SVG color name")
        self.profile_color.model.add_value_changed_fn(self._color_text_changed)
        for item in self.profile_picker.model.get_item_children():
            self.profile_picker.model.get_item_value_model(item).add_value_changed_fn(self._color_picker_changed)
        ui.Label("Applies to loaded and future profiler views. Actor and gizmo colors are edited in Scene palette.",
            name="muted", word_wrap=True, height=fl.pvd_row * 2)

    def _build(self):
        self.window = ui.Window("General settings", width=700, height=680,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self.key)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_spacing)
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label("General settings", name="title")
                            ui.Button("Close", name="quiet", width=70, clicked_fn=self.close)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            for key, label in self.SECTIONS:
                                self.tabs[key] = ui.Button(label, name="tab",
                                    clicked_fn=lambda section=key: self._select_section(section))
                        with ui.ZStack():
                            for key, _label in self.SECTIONS:
                                panel = ui.ScrollingFrame(visible=key == self.section,
                                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF)
                                self.sections[key] = panel
                                with panel:
                                    with ui.VStack(height=0, spacing=fl.pvd_spacing):
                                        if key == "rendering":
                                            self._rendering_rows()
                                            self.attachment_status = ui.Label("", name="muted",
                                                height=fl.pvd_row * 2, word_wrap=True)
                                        elif key == "properties":
                                            self._properties()
                                        else:
                                            self._profile()
                        self.status = ui.Label("", name="muted", height=fl.pvd_row * 2, word_wrap=True)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Button("Restore defaults", name="outline", width=150, clicked_fn=self._defaults,
                                explicit_hover=True, tooltip="Stages defaults for all three sections. Apply saves them; Cancel discards the draft.")
                            ui.Spacer()
                            ui.Button("Cancel", width=90, clicked_fn=self.close)
                            ui.Button("Apply", name="primary", width=110, clicked_fn=self._apply)
                        ui.Spacer(height=fl.pvd_spacing)
                    ui.Spacer(width=fl.pvd_padding)
