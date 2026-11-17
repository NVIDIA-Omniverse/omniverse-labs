# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable palette editor using OVUI's native color picker and value models."""
import re

import omni.ui as ui

from ..palette import (ACTOR_KEYS, GIZMO_KEYS, MAGENTA, PROFILE_FUNCTION_DEFAULT,
                       UNUSED_ACTOR_KEYS, color_hex, parse_color)
from ..style import layout as fl, style_dialog
from .file_browser import CaptureFileDialog


class PaletteView:
    def __init__(self, palette, preferences, *, on_changed=None):
        self.palette, self.preferences = palette, preferences
        self.on_changed = on_changed
        self.window = None
        self.file_dialog = None
        self.section = "actors"
        self.pending = {}
        self.controls = {}
        self.tabs = {}
        self._syncing = False
        self._revision = -1

    @property
    def visible(self):
        return bool(self.window and self.window.visible)

    def show(self):
        if self.window is None:
            self._build()
        self.window.visible = True
        self.refresh(force=True)

    def hide(self):
        if self.window:
            self.window.visible = False

    def destroy(self):
        if self.file_dialog:
            self.file_dialog.destroy()
        if self.window:
            self.window.destroy()
            self.window = None

    def _value(self, section, key):
        if (section, key) in self.pending:
            return self.pending[section, key]
        if section == "profile":
            return self.preferences.get("ProfileFunctionColor", PROFILE_FUNCTION_DEFAULT)
        return self.palette.raw(section, key)

    def sync_profile_color(self, value):
        """Refresh an accepted profile color without replacing a palette draft."""
        identity = ("profile", "ProfileFunctionColor")
        if identity not in self.pending:
            self._sync_control(identity, value)

    def _sync_control(self, identity, value):
        control = self.controls.get(identity)
        if control is None:
            return
        field, picker = control
        self._syncing = True
        try:
            field.model.set_value(value)
            try:
                rgba = parse_color(value)
            except ValueError:
                rgba = MAGENTA
            for item, component in zip(picker.model.get_item_children(), rgba):
                picker.model.get_item_value_model(item).set_value(component)
        finally:
            self._syncing = False

    def _edit_text(self, identity, value):
        if self._syncing:
            return
        self.pending[identity] = value
        self._sync_control(identity, value)
        self._status()

    def _edit_picker(self, identity, model):
        if self._syncing:
            return
        rgba = [model.get_item_value_model(item).as_float for item in model.get_item_children()]
        value = color_hex(rgba)
        self.pending[identity] = value
        # The native picker changes its channel value models directly. Updating
        # those models here would feed changes back into the active color drag.
        self._syncing = True
        try:
            self.controls[identity][0].model.set_value(value)
        finally:
            self._syncing = False
        self._status()

    def _status(self, message=None):
        if self.window is None:
            return
        invalid = []
        for identity, value in self.pending.items():
            try:
                parse_color(value)
            except ValueError:
                invalid.append(".".join(identity))
        self.apply_button.enabled = bool(self.pending) and not invalid
        self.status.text = message or (
            f"Invalid color: {', '.join(invalid)}" if invalid else
            f"{len(self.pending)} edited colors · Apply saves your changes" if self.pending else
            f"{len(self.palette.issues)} missing or invalid colors render magenta" if self.palette.issues else
            "Palette is up to date · Changes to its JSON file reload automatically")

    def refresh(self, *, force=False):
        self.palette.poll(force=force)
        if self.window is None or (not force and self._revision == self.palette.revision):
            return
        self._revision = self.palette.revision
        for identity in self.controls:
            self._sync_control(identity, self._value(*identity))
        self._status(self.palette.load_error or None)

    def apply(self):
        try:
            for value in self.pending.values():
                parse_color(value)
            colors = {identity: value for identity, value in self.pending.items() if identity[0] != "profile"}
            if colors:
                self.palette.edit(colors)
            if ("profile", "ProfileFunctionColor") in self.pending:
                self.preferences.set("ProfileFunctionColor", self.pending["profile", "ProfileFunctionColor"])
            self.pending.clear()
            self.refresh(force=True)
            if self.on_changed:
                self.on_changed()
        except (OSError, ValueError, TypeError) as error:
            self._status(f"Could not save changes: {error}")

    def discard(self):
        self.pending.clear()
        self.refresh(force=True)

    def _defaults(self):
        try:
            if self.section == "profile":
                self.pending["profile", "ProfileFunctionColor"] = PROFILE_FUNCTION_DEFAULT
            else:
                self.pending.update(self.palette.default_changes())
            self.refresh(force=True)
        except (OSError, ValueError, KeyError) as error:
            self._status(f"Could not load defaults: {error}")

    def _import(self):
        if self.file_dialog is None:
            self.file_dialog = CaptureFileDialog(self._import_selected, title="Import color palette",
                extensions=(".json",), initial_path=self.palette.path.parent)
        self.file_dialog.show()

    def _import_selected(self, path):
        self.palette.import_file(path)
        self.pending.clear()
        self.refresh(force=True)
        if self.on_changed:
            self.on_changed()

    def _choose_section(self, section):
        self.section = section
        for name, button in self.tabs.items():
            button.name = "primary" if name == section else ""
        self.controls.clear()
        self.body.rebuild()

    def _rows(self):
        keys = ACTOR_KEYS if self.section == "actors" else GIZMO_KEYS if self.section == "gizmos" else ("ProfileFunctionColor",)
        with ui.VStack(height=0, spacing=fl.pvd_spacing):
            if self.section == "actors":
                explanation = "Native actor colors follow the reference viewer, including its sleeping-kinematic rule. Stored unused keys remain editable for compatibility."
            elif self.section == "gizmos":
                explanation = "Center-of-mass color follows the reference palette. Other generated gizmo colors extend its unused settings; recorded custom and query colors stay intact."
            else:
                explanation = "Profile function color is stored separately in the OvUI preferences. It does not change colors.json."
            ui.Label(explanation, name="muted", word_wrap=True, height=fl.pvd_row * 3)
            for key in keys:
                identity = (self.section, key)
                with ui.HStack(height=fl.pvd_control * 1.5, spacing=fl.pvd_spacing):
                    picker = ui.ColorWidget(width=fl.pvd_control * 1.5, height=fl.pvd_control)
                    with ui.VStack(spacing=fl.pvd_zero):
                        title = "Profile function color" if self.section == "profile" else re.sub(r"(?<!^)(?=[A-Z])", " ", key).capitalize()
                        ui.Label(title, height=fl.pvd_row)
                        detail = "Stored key · unused in reference renderer" if self.section == "actors" and key in UNUSED_ACTOR_KEYS else key
                        ui.Label(detail, name="muted", height=fl.pvd_row)
                    field = ui.StringField(width=fl.pvd_button_width * 1.7, height=fl.pvd_control,
                        explicit_hover=True, tooltip="RGB/ARGB hex value or SVG color name")
                    self.controls[identity] = field, picker
                    field.model.add_value_changed_fn(lambda model, entry=identity: self._edit_text(entry, model.as_string))
                    for item in picker.model.get_item_children():
                        picker.model.get_item_value_model(item).add_value_changed_fn(
                            lambda _, entry=identity, model=picker.model: self._edit_picker(entry, model))
                    self._sync_control(identity, self._value(*identity))
                ui.Separator(height=fl.pvd_border_width)

    def _build(self):
        self.window = ui.Window("Scene palette", width=fl.pvd_dialog_width, height=fl.pvd_dialog_height,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_spacing)
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label("Scene palette", name="section")
                            ui.Button("Close", name="quiet", width=fl.pvd_button_width, clicked_fn=self.hide)
                        ui.Label(str(self.palette.path), name="muted", elided_text=True,
                            explicit_hover=True, tooltip=str(self.palette.path), height=fl.pvd_row)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            for label, section in (("Actors", "actors"), ("Gizmos", "gizmos"), ("Profile", "profile")):
                                self.tabs[section] = ui.Button(label, name="primary" if section == self.section else "",
                                    clicked_fn=lambda value=section: self._choose_section(value))
                        with ui.ScrollingFrame():
                            self.body = ui.Frame(build_fn=self._rows)
                        self.status = ui.Label("", name="muted", word_wrap=True, height=fl.pvd_row * 2)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Button("Import JSON", clicked_fn=self._import)
                            ui.Button("Defaults", clicked_fn=self._defaults)
                            ui.Button("Reload", clicked_fn=self.discard)
                            self.apply_button = ui.Button("Apply", name="primary", clicked_fn=self.apply, enabled=False)
                        ui.Spacer(height=fl.pvd_spacing)
                    ui.Spacer(width=fl.pvd_padding)
