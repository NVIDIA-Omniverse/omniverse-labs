# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent evidence input surface; it never owns a playhead model."""
import omni.ui as ui
from omni.ui import color as cl
from .timeline import FrameScale, SoftGlassTimeline


def marker_key(marker):
    return marker.set_digest, marker.finding_id, marker.role, marker.frame


class FindingTrack:
    def __init__(self, on_select, color_for=None):
        self.on_select, self.color_for = on_select, color_for or (lambda _: cl.pvd_accent)
        self._key = self._selected = None
        self._domain, self._clip, self._markers, self.builds = (0, 1), (0, 1), (), 0
        with ui.Frame(height=12, visible=False, horizontal_clipping=True, vertical_clipping=True) as self.widget:
            with ui.ZStack():
                self.visual = ui.Frame(build_fn=self._build)
                self.input = ui.InvisibleButton(identifier="evidence_track", explicit_hover=True,
                    tooltip="Recorded evidence points only. Click selects details; repeated clicks cycle overlaps.",
                    mouse_pressed_fn=self.press)
        self.widget.set_computed_content_size_changed_fn(lambda *_: self.update(self._domain, self._clip, self._markers))

    def update(self, domain, clip, markers):
        self._domain, self._clip = tuple(domain), tuple(clip)
        self._markers = tuple(sorted((m for m in markers if domain[0] <= m.frame <= domain[1] and clip[0] <= m.frame <= clip[1]), key=marker_key))
        self.widget.visible = bool(self._markers)
        key = self._domain, self._clip, tuple(marker_key(m) for m in self._markers), self.widget.computed_width
        if key == self._key:
            return False
        self._key = key;self.visual.rebuild()
        return True

    def press(self, x, _y, button, _modifiers):
        if button != 0 or not self._markers:
            return
        scale = FrameScale(*self._domain, self.widget.computed_width)
        local = x-self.widget.screen_position_x
        hits = sorted((m for m in self._markers if abs(scale.position(m.frame)-local) <= 6),
                      key=lambda m: (abs(scale.position(m.frame)-local), marker_key(m)))
        if not hits:
            return
        tokens = [marker_key(m) for m in hits]
        index = (tokens.index(self._selected)+1) % len(hits) if self._selected in tokens else 0
        self._selected = tokens[index]
        self.on_select(hits[index]);self.visual.rebuild()

    def _build(self):
        if not hasattr(self, "input"):
            return
        self.builds += 1
        scale = FrameScale(*self._domain, self.widget.computed_width)
        with ui.ZStack():
            ui.Rectangle(style={"background_color": cl.pvd_field, "border_radius": 3})
            for marker in self._markers:
                color = self.color_for(marker)
                SoftGlassTimeline._at(max(0, scale.position(marker.frame)-2), 2, 5, 8,
                    lambda c=color: ui.Rectangle(style={"background_color": c, "border_radius": 2}))

    def restyle(self):
        self.visual.rebuild()

    def destroy(self):
        self.widget = None
