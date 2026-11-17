# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded native-frame navigation using the same playback rules as the ruler."""
from copy import deepcopy
import re

import omni.ui as ui

from ..camera import normalize_key
from ..style import layout as fl, style_dialog


def frame_target(text, playback):
    """Resolve a displayed native frame without changing the supplied playhead."""
    clip = playback.clip
    if clip is None or clip.stop <= clip.start:
        raise ValueError("Open a capture with published frames first.")
    text = str(text).strip()
    if not re.fullmatch(r"\+?[0-9]+", text):
        raise ValueError("Enter a whole frame number.")
    digits = text.lstrip("+").lstrip("0") or "0"
    maximum = str(clip.last)
    if len(digits) > len(maximum) or (len(digits) == len(maximum) and digits > maximum):
        raise ValueError(f"Choose a frame from {clip.start:,} to {clip.last:,}.")
    value = int(digits)
    if value < clip.start:
        raise ValueError(f"Choose a frame from {clip.start:,} to {clip.last:,}.")
    requested = deepcopy(playback)
    frame = requested.seek(value)
    detail = f"Frame {frame:,}"
    if playback.ovd:
        detail += f" · {requested.label}"
    return frame, detail


class GoToFrameDialog:
    """Retained modal; the owner validates the capture/clip before accepting."""

    def __init__(self, on_apply):
        self.on_apply = on_apply
        self.window = None
        self.playback = self.context = None
        self._closed = False

    def _build(self):
        self.window = ui.Window("Go to frame", width=440, height=258,
            flags=ui.WINDOW_FLAGS_MODAL | ui.WINDOW_FLAGS_NO_TITLE_BAR |
                  ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_RESIZE)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self.key)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_padding)
                        ui.Label("Go to frame", name="title", height=fl.pvd_control)
                        self.range_label = ui.Label("", name="muted", height=fl.pvd_row,
                                                   elided_text=True)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Label("Frame number", width=fl.pvd_button_width)
                            self.model = ui.SimpleStringModel("")
                            self.field = ui.StringField(self.model, name="mono",
                                explicit_hover=True, tooltip="Native frame number, matching the timeline and Frame field.")
                        self.preview = ui.Label("", name="muted", height=fl.pvd_row,
                                                elided_text=True)
                        ui.Spacer()
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Spacer()
                            self.go = ui.Button("Go to frame", name="primary",
                                width=fl.pvd_button_width, clicked_fn=self.apply)
                            ui.Button("Cancel", width=fl.pvd_button_width, clicked_fn=self.close)
                        ui.Spacer(height=fl.pvd_padding)
                    ui.Spacer(width=fl.pvd_padding)
        self.model.add_value_changed_fn(lambda _: self.validate())

    def show(self, playback, context):
        if self._closed:
            return
        if self.window is None:
            self._build()
        self.playback, self.context = deepcopy(playback), context
        clip = self.playback.clip
        self.range_label.text = f"{clip.name} · Frames {clip.start:,} to {clip.last:,}"
        self.model.set_value(str(playback.frame))
        self.validate()
        self.window.visible = True
        self.window.focus()
        self.field.focus_keyboard()

    def validate(self):
        try:
            _, detail = frame_target(self.model.as_string, self.playback)
            self.preview.text = detail
            self.go.enabled = True
            return True
        except ValueError as error:
            self.preview.text = str(error)
            self.go.enabled = False
            return False

    def apply(self):
        if self._closed or not self.validate():
            return False
        frame, _ = frame_target(self.model.as_string, self.playback)
        try:
            self.on_apply(frame, self.context)
        except (ValueError, RuntimeError) as error:
            self.preview.text = str(error)
            return False
        self.close()
        return True

    def key(self, key, modifiers, pressed):
        if not pressed:
            return
        key = normalize_key(key)
        if key == 256:
            self.close()
        elif key in (257,335) and not modifiers & 15:
            self.apply()

    def close(self):
        if self.window is not None:
            self.window.visible = False

    def destroy(self):
        self._closed = True
        self.on_apply = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
