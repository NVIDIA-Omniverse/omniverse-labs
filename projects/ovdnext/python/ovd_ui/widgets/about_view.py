# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Soft Glass About/help surface with immutable copyable package provenance."""
import omni.ui as ui

from ..app_info import PRODUCT_NAME, MANUAL_STATUS, load_about_info
from ..camera import normalize_key
from ..clipboard import copy_text, ClipboardError
from ..desktop import SupportComposeRequest, DesktopError
from ..style import layout as fl, style_dialog


ABOUT_WIDTH = 688
ABOUT_BASE_HEIGHT = 348
ABOUT_KEY_WIDTH = 130
ABOUT_ACTION_WIDTH = 148


class AboutView:
    def __init__(self, info=None, *, on_workspace_guide=None, on_error=None,
                 on_copy_text=copy_text, compose_request=SupportComposeRequest):
        self.info = info if info is not None else load_about_info()
        self.on_workspace_guide, self.on_error = on_workspace_guide, on_error
        self.on_copy_text, self.compose_request = on_copy_text, compose_request
        self.window = self.status_label = self.support_button = None
        self.status = ""
        self.request = None
        self._closed = False

    def show(self):
        if self._closed:
            return
        if self.window is None:
            self._build()
        self.window.visible = True
        self.window.focus()

    def hide(self):
        if self.window is not None:
            self.window.visible = False

    def key(self, key, modifiers, pressed):
        if pressed and normalize_key(key) == 256:
            self.hide()

    def _set_status(self, text, *, error=False):
        self.status = text
        if self.status_label is not None:
            self.status_label.text = text
        if error and self.on_error:
            self.on_error(text)

    def copy_info(self):
        if self._closed:
            return False
        try:
            self.on_copy_text(self.info.copy_text)
        except (ClipboardError, ValueError, OSError) as error:
            self._set_status(str(error), error=True)
            return False
        self._set_status("Displayed information copied.")
        return True

    def show_guide(self):
        if not self._closed and self.on_workspace_guide:
            self.hide()
            self.on_workspace_guide()

    def report_issue(self):
        if self._closed or self.request is not None:
            return False
        try:
            self.request = self.compose_request()
        except DesktopError as error:
            self._set_status(str(error), error=True)
            return False
        if self.support_button is not None:
            self.support_button.enabled = False
        self._set_status("Opening your email application. No message has been sent.")
        return True

    def refresh(self):
        if self._closed or self.request is None:
            return
        try:
            if not self.request.poll():
                return
            self._set_status("Compose request handed to your desktop. No message has been sent.")
        except DesktopError as error:
            self._set_status(str(error), error=True)
        self.request.close()
        self.request = None
        if self.support_button is not None:
            self.support_button.enabled = True

    def destroy(self):
        self._closed = True
        if self.request is not None:
            self.request.close()
            self.request = None
        if self.window is not None:
            self.window.destroy()
        self.window = self.status_label = self.support_button = None
        self.on_workspace_guide = self.on_error = None

    def _build(self):
        self.window = ui.Window("About and help", width=ABOUT_WIDTH,
            height=ABOUT_BASE_HEIGHT + len(self.info.rows) * (fl.pvd_control + fl.pvd_spacing),
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_DOCKING |
                  ui.WINDOW_FLAGS_NO_RESIZE)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self.key)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_padding)
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label(PRODUCT_NAME, name="title")
                            ui.Button("Close", name="quiet", width=fl.pvd_button_width, clicked_fn=self.hide)
                        ui.Label("About and help", name="muted", height=fl.pvd_control)
                        for label, value in self.info.rows:
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label(label + ":", name="muted", width=ABOUT_KEY_WIDTH)
                                ui.Label(value, name="mono_small", explicit_hover=True, tooltip=value, elided_text=True)
                        ui.Label(self.info.provenance_status, name="muted", height=fl.pvd_control)
                        ui.Spacer(height=fl.pvd_spacing)
                        ui.Label("Help and support", name="section", height=fl.pvd_control)
                        ui.Label(MANUAL_STATUS, name="muted", height=fl.pvd_row, word_wrap=True)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Button("Workspace guide", name="outline", width=ABOUT_ACTION_WIDTH,
                                enabled=self.on_workspace_guide is not None, clicked_fn=self.show_guide)
                            self.support_button = ui.Button("Report issue", name="outline", width=ABOUT_ACTION_WIDTH,
                                explicit_hover=True, tooltip="Open an email draft to pvdsupport@nvidia.com with subject PVD Issue.",
                                clicked_fn=self.report_issue)
                            ui.Spacer()
                            ui.Button("Copy info", name="primary", width=fl.pvd_button_width, clicked_fn=self.copy_info)
                        self.status_label = ui.Label(self.status, name="muted", height=fl.pvd_row * 2, word_wrap=True)
                        ui.Spacer(height=fl.pvd_padding)
                    ui.Spacer(width=fl.pvd_padding)
