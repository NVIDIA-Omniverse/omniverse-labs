# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Soft Glass frame ruler, sharing the application's integer playhead model.

``min`` and ``max`` are inclusive seek/clip bounds, as on ``ui.IntSlider``.
``set_domain(first, last)`` optionally fixes the displayed capture bounds. Neither
bounds nor model notifications seek implicitly; only a left click/drag writes the
model. The application's existing callback remains responsible for parity,
published-frame availability, and restoring an unapplied seek.
"""
from dataclasses import dataclass
import math
import operator
import weakref

import omni.ui as ui
from omni.ui import color as cl

from ..design_tokens import FONTS, font_pixels


_CLIP_BORDER = cl.shade(cl("#61CB7C66"), light=cl("#25984D66"))
_CLIP_FILL = cl.shade(cl("#61CB7C29"), light=cl("#25984D1F"))


def _integer(value):
    if isinstance(value, bool):
        raise TypeError("Frame bounds must be integers")
    return operator.index(value)


@dataclass(frozen=True)
class FrameScale:
    """Integer capture coordinates mapped into an inset, finite pixel track."""
    first: int
    last: int
    width: float
    inset: float = 10.

    @property
    def left(self):
        return min(self.inset, max(0., self.width) / 2)

    @property
    def right(self):
        return max(self.left, max(0., self.width) - self.left)

    def position(self, frame):
        if self.last <= self.first:
            return (self.left + self.right) / 2
        fraction = (frame - self.first) / (self.last - self.first)
        return self.left + max(0., min(1., fraction)) * (self.right - self.left)

    def frame(self, x):
        if self.right <= self.left or self.last <= self.first:
            return self.first
        fraction = max(0., min(1., (x - self.left) / (self.right - self.left)))
        # Relative arithmetic keeps large native frame IDs out of floating point.
        return self.first + math.floor(fraction * (self.last - self.first) + .5)

    def ticks(self):
        """Uniform integer steps with enough room for each decimal label."""
        span = self.last - self.first
        if span <= 0:
            return (self.first,)
        label_width = max(len(str(self.first)), len(str(self.last))) * 6 + 16
        count = max(1, int((self.right - self.left) / max(54, label_width)))
        wanted = max(1, math.ceil(span / count))
        decade = 10 ** (len(str(wanted)) - 1)
        step = next(n * decade for n in (1, 2, 5, 10) if n * decade >= wanted)
        first = -(-self.first // step) * step
        return tuple(range(first, self.last + 1, step)) or (self.first,)


class SoftGlassTimeline:
    """A clipped 44px ruler; the input surface survives visual rebuilds.

    No application update-loop hook is required. Model notifications rebuild
    only the playhead; bounds and size changes rebuild the ruler. Keep this
    object alive alongside its model;
    ``destroy()`` removes the model callback when replacing a workspace.
    """
    def __init__(self, model, *, min=0, max=1, height=44, width=None,
                 identifier="frame_timeline"):
        self.model = model
        self._min, self._max = _integer(min), _integer(max)
        self._domain = None
        self._dragging = False
        self._size = (0., 0.)
        self._built_layout = None
        self._painted_current = None
        self._marker = self._badge = self._badge_label = None
        options = {"height": height, "horizontal_clipping": True, "vertical_clipping": True}
        if width is not None:
            options["width"] = width
        with ui.Frame(**options) as self.widget:
            with ui.ZStack():
                ui.Rectangle(style={"background_color": cl.pvd_field,
                    "border_color": cl.pvd_border, "border_width": 1,
                    "border_radius": 11})
                # Exactly one clipped Frame owns both paint and input. It keeps
                # stale visual widths from imposing a minimum on the next layout.
                # Nested clipped containers would cover the sibling input window.
                self.visual = ui.Frame(build_fn=self._build)
                self.input = ui.InvisibleButton(identifier=identifier,
                    tooltip="Drag to seek within the active clip", explicit_hover=True,
                    mouse_pressed_fn=self.press, mouse_moved_fn=self.move,
                    mouse_released_fn=self.release)
        self.widget.set_computed_content_size_changed_fn(self._resized)
        # The native model must not keep an otherwise destroyed workspace alive.
        reference = weakref.ref(self)
        def changed(_model):
            owner = reference()
            if owner is not None:
                owner._redraw()
        self._subscription = model.add_value_changed_fn(changed)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        value = _integer(value)
        if value != self._min:
            self._min = value
            self._redraw()

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        value = _integer(value)
        if value != self._max:
            self._max = value
            self._redraw()

    @property
    def enabled(self):
        return self.widget.enabled

    @enabled.setter
    def enabled(self, value):
        self.widget.enabled = bool(value)
        if not value:
            self._dragging = False
        self._redraw()

    @property
    def width(self):
        return self.widget.width

    @width.setter
    def width(self, value):
        self.widget.width = value
        # Explicit child width changes need the visual Frame's layout cache
        # invalidated; a native window resize already invalidates the ancestors.
        self._built_layout = None
        self._redraw()

    def set_domain(self, first=None, last=None):
        """Set inclusive capture bounds, or call without arguments to follow clip.

        The caller converts native exclusive frame_count to last=frame_count-1.
        A temporarily out-of-domain active clip is intersected for display/input.
        """
        domain = None if first is None and last is None else (_integer(first), _integer(last))
        if domain is not None and domain[1] < domain[0]:
            raise ValueError("Timeline domain last must be at least first")
        if domain != self._domain:
            self._domain = domain
            self._redraw()

    def _scale(self):
        first, last = self._domain or (self._min, max(self._min, self._max))
        return FrameScale(first, last, max(0., self.widget.computed_width))

    def _clip(self, scale):
        first, last = max(scale.first, self._min), min(scale.last, self._max)
        return (first, last) if first <= last else None

    def _seek(self, screen_x):
        scale = self._scale()
        clip = self._clip(scale)
        if not self.enabled or clip is None or scale.width <= 0 or not math.isfinite(screen_x):
            return
        frame = max(clip[0], min(clip[1], scale.frame(screen_x - self.widget.screen_position_x)))
        if frame != self.model.as_int:
            self.model.set_value(frame)

    def press(self, x, _y, button, _modifiers):
        if button == 0 and self.enabled:
            self._dragging = True
            self._seek(x)

    def move(self, x, _y, *_):
        if self._dragging:
            self._seek(x)

    def release(self, _x, _y, button, *_):
        if button == 0:
            # Do not seek again here: parity snapping may have adjusted the
            # model synchronously, and a release is not another frame request.
            self._dragging = False

    def _resized(self):
        size = (self.widget.computed_width, self.widget.computed_height)
        if size != self._size:
            self._size = size
            self._redraw()

    def _redraw(self):
        layout = getattr(self, "_built_layout", None)
        if layout is not None and layout == self._layout():
            if self.model.as_int != self._painted_current:
                # Populate before the next draw. A lazy Frame build draws its
                # previous canvas once more; direct offset mutations during a
                # mouse callback can lose their pending layout invalidation.
                scale, height, _clip, enabled = layout
                with self.cursor:
                    self._build_cursor(scale, height, cl.pvd_accent if enabled else cl.pvd_disabled)
        else:
            self._built_layout = None
            self.visual.rebuild()

    def _layout(self):
        scale = self._scale()
        return scale, max(0., self.widget.computed_height), self._clip(scale), self.enabled

    @staticmethod
    def _at(x, y, width, height, build):
        # Fixed spacer rows place only content inside an already clipped Frame.
        with ui.VStack(spacing=0):
            ui.Spacer(height=max(0, y))
            with ui.HStack(height=height, spacing=0):
                ui.Spacer(width=max(0, x))
                with ui.ZStack(width=max(0, width), height=height):
                    build()
                ui.Spacer()
            ui.Spacer()

    def _label(self, text, color, alignment=ui.Alignment.CENTER):
        return ui.Label(text, alignment=alignment, elided_text=True,
            style={"color": color, "font": str(FONTS["mono"]),
                   "font_size": font_pixels(10, "mono")})

    def _build(self):
        # The initial lazy build can run before the input surface is laid out.
        if not hasattr(self, "input"):
            return
        layout = self._layout()
        scale, height, clip, enabled = layout
        width = scale.width
        if width <= 1 or height <= 1:
            return
        accent = cl.pvd_accent if enabled else cl.pvd_disabled
        with ui.ZStack(width=width, height=height) as self._ruler:
            for tick in scale.ticks():
                x = scale.position(tick)
                self._at(x, 1, 1, height-2,
                    lambda: ui.Rectangle(style={"background_color": cl.pvd_border}))
                label_width = min(width-2, max(16, len(str(tick))*6+4))
                label_x = max(1, min(width-label_width-1, x-label_width/2))
                self._at(label_x, height-16, label_width, 14,
                    lambda tick=tick: self._label(str(tick), cl.pvd_muted))
            if clip is not None:
                left, right = scale.position(clip[0]), scale.position(clip[1])
                band_width = max(2, right-left)
                self._at(min(left, width-band_width), 12, band_width, max(2, height-24),
                    lambda: ui.Rectangle(style={"background_color": _CLIP_FILL,
                        "border_color": _CLIP_BORDER, "border_width": 1,
                        "border_radius": 6}))
            with ui.Frame() as self.cursor:
                self._build_cursor(scale, height, accent)
        self._built_layout = layout

    def _build_cursor(self, scale, height, accent):
        current = self.model.as_int
        self._painted_current = current
        self._marker = self._badge = self._badge_label = None
        width, x = scale.width, scale.position(current)
        with ui.ZStack(width=width, height=height):
            # An empty replacement clears the previous canvas while preserving
            # the published model. Only a user gesture clamps to the clip.
            if not scale.first <= current <= scale.last:
                return
            # Fixed-size Placer overlays keep offsets out of parent layout.
            # Constructing the small canvas handles both before-draw model
            # publication and mouse callbacks during draw on public OVUI.
            with ui.Placer(width=2, height=height-2, offset_y=1,
                           offset_x=max(1, min(width-3, x-1)),
                           stable_size=True, draggable=False) as self._marker:
                self._marker_shape = ui.Rectangle(style={"background_color": accent})
            badge_width = min(max(18, len(str(current))*6+12), max(0, width-4))
            with ui.Placer(width=badge_width, height=16, offset_y=1,
                           offset_x=max(2, min(width-badge_width-2, x-badge_width/2)),
                           stable_size=True, draggable=False) as self._badge:
                with ui.ZStack() as self._badge_content:
                    ui.Rectangle(style={"background_color": accent, "border_radius": 6})
                    # Preserve the oracle's two lower rounded corners.
                    with ui.VStack(spacing=0):
                        ui.Rectangle(height=6, style={"background_color": accent, "border_radius": 0})
                        ui.Spacer()
                    self._badge_label = self._label(str(current), cl.pvd_on_accent)

    def destroy(self):
        self._dragging = False
        if self._subscription is not None:
            self.model.remove_value_changed_fn(self._subscription)
            self._subscription = None

    def get_inspector_state(self):
        scale = self._scale()
        return {"frame": self.model.as_int, "clip": self._clip(scale),
                "domain": [scale.first, scale.last], "ticks": list(scale.ticks()),
                "width": scale.width, "height": self.widget.computed_height,
                "dragging": self._dragging, "enabled": self.enabled}
