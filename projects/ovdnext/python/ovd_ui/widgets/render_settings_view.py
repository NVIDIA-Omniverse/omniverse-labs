# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable rendering settings panel; the owner applies immutable settings."""
import omni.ui as ui

from ..render_settings import RenderingSettings
from ..style import layout as fl, style_dialog


class RenderingSettingsView:
    MODES = ("None", "All", "Selected", "Unselected")

    def __init__(self, settings, on_apply, *, preferences=None, sample_info=None):
        self.settings, self.on_apply = settings, on_apply
        self.preferences, self.sample_info = preferences, sample_info
        self.window = None
        self.controls = {}
        self._status_error = False

    def destroy(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def show(self):
        if self.window is None:
            self._build()
        self.window.visible = True
        self._sync()
        self.window.focus()

    def set_settings(self, settings, *, fields=()):
        """Adopt saved settings, optionally updating named controls only.

        External toolbar changes can synchronize their fields in an open panel
        without discarding other drafts or clearing its validation message.
        An unbuilt panel simply retains the settings for its next ``show()``.
        """
        self.settings = settings
        self._write_controls(settings, fields=fields)

    def _write_controls(self, settings, *, fields):
        """Fill controls without accepting a draft or changing saved settings."""
        for key in fields:
            control = self.controls.get(key)
            if control is None:
                continue
            value = getattr(settings, key)
            if key == "background_mode":
                value = ("workspace", "reference", "custom").index(value)
            model = control.model
            if hasattr(model, "get_item_value_model"):
                model = model.get_item_value_model()
            model.set_value(value)

    def _sync(self):
        self.set_settings(self.settings, fields=self.controls)
        self._status_error = False
        self.status.text = self.sample_info() if self.sample_info else "Changes apply to the viewport and object picking."

    def refresh(self):
        """Update actual attachment status after a draw without clearing errors/drafts."""
        if self.window is not None and self.window.visible and not self._status_error and self.sample_info:
            self.status.text = self.sample_info()

    def _apply(self, defaults=False):
        try:
            values = self._read_values()
            candidate = RenderingSettings() if defaults else RenderingSettings(**values)
            if self.preferences is not None:
                self.preferences.set("rendering", candidate.to_preferences())
            self.on_apply(candidate)
            self.settings = candidate
            self._sync()
        except (ValueError, OSError) as error:
            self._status_error = True
            self.status.text = str(error)

    def _read_values(self):
        values = {}
        for key, control in self.controls.items():
            model = control.model
            if hasattr(model, "get_item_value_model"):
                values[key] = model.get_item_value_model().as_int
            elif key in ("anti_aliasing", "show_particles", "draw_orbit_center"):
                values[key] = model.as_bool
            elif key == "background_color":
                values[key] = model.as_string
            else:
                values[key] = model.as_float
        mode = values["background_mode"]
        if mode not in range(3):
            raise ValueError("Choose Workspace theme, Reference gray or Custom background")
        values["background_mode"] = ("workspace", "reference", "custom")[mode]
        return values

    def _row(self, label, key, choices=None, boolean=False, tooltip=""):
        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
            ui.Label(label, width=180, explicit_hover=True, tooltip=tooltip)
            if choices:
                control = ui.ComboBox(0, *choices, explicit_hover=True, tooltip=tooltip)
            elif boolean:
                control = ui.CheckBox(width=24, explicit_hover=True, tooltip=tooltip)
                ui.Spacer()
            elif key == "background_color":
                control = ui.StringField(name="mono", explicit_hover=True, tooltip=tooltip)
            else:
                control = ui.FloatField(name="mono", explicit_hover=True, tooltip=tooltip)
            self.controls[key] = control

    def _rendering_rows(self):
        self._row("Anti-aliasing", "anti_aliasing", boolean=True,
            tooltip="Requests up to 8 MSAA samples. The status reports the actual hardware attachment count.")
        self._row("Face culling", "face_culling", ("None", "Normal", "Inverse"),
            tooltip="Chirality follows the viewport camera. Recorded heightfields remain two-sided.")
        self._row("Geometry scale", "geometry_scale", tooltip="Scales world geometry, overlays and pick coordinates. Use Zoom scene after changing scale.")
        self._row("Recorded planes", "planes", ("Auto", "None", "Basis (no synthetic plane)", "Scene"))
        self._row("Plane extent scale", "plane_scale", tooltip="OVUI addition: scales the visible finite patch of an infinite recorded plane.")
        self._row("Particles", "show_particles", boolean=True)
        self._row("Particle size", "particle_size", tooltip="0–10 world units for legacy particles. Recorded OVD particles retain their physical radius above 0.001; zero hides them.")
        self._row("Sleeping objects", "sleeping_objects", self.MODES)
        self._row("Trigger shapes", "trigger_shapes", ("None", "All", "Unselected (legacy Selected)", "Selected (legacy Unselected)"),
            tooltip="The original renderer's Selected/Unselected trigger condition is inverted. These labels show the actual result while preserving its stored mode values.")
        self._row("Wireframe", "wireframe", self.MODES)
        self._row("Camera orbit center", "draw_orbit_center", boolean=True,
            tooltip="Draw a white wire sphere at the camera pivot. Its size follows the current scene and Gizmo size; it cannot be selected.")
        self._row("Background", "background_mode", ("Workspace theme", "Reference gray", "Custom"))
        self._row("Custom background", "background_color", tooltip="RGB/ARGB hex or SVG color name. Viewport background is independent of the scene actor palette.")
        ui.Label("Recorded object colors stay in Scene palette. Workspace surfaces follow the UI theme.",
            name="muted", height=fl.pvd_row * 2, word_wrap=True)

    def _build(self):
        self.window = ui.Window("Rendering settings", width=660, height=730,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_spacing)
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label("Rendering settings", name="section")
                            ui.Button("Close", name="quiet", width=70,
                                clicked_fn=lambda: setattr(self.window, "visible", False))
                        self._rendering_rows()
                        self.status = ui.Label("", name="muted", height=fl.pvd_row * 2, word_wrap=True)
                        ui.Spacer()
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Button("Restore defaults", name="outline", clicked_fn=lambda: self._apply(True))
                            ui.Spacer()
                            ui.Button("Apply", name="primary", width=110, clicked_fn=self._apply)
                        ui.Spacer(height=fl.pvd_spacing)
                    ui.Spacer(width=fl.pvd_padding)
