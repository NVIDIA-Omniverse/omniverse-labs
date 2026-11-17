# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pixel-based graph selection without changing the operating-system pointer.

The rectangle arithmetic matches the vendored QwtPlotZoomer::accept and Qt5
integer QRect rules. Keyboard repeat is deliberately application-owned: one
pixel initially, then five after 400ms and every 50ms, quantized by UI polling.
A stalled UI produces at most one repeat per poll, never a catch-up jump.
"""
from __future__ import annotations

import math


def pixel(value: float) -> int:
    """Qt QPointF::toPoint rounding, in bounded logical canvas coordinates."""
    if not math.isfinite(value) or abs(value) > 1 << 28:
        raise ValueError("Graph pixel coordinates must be finite and bounded")
    # Qt5 qRound resolves negative half ties upward too (-0.5 becomes 0).
    return math.floor(value + .5)


def _half(value: int) -> int:
    return value // 2 if value >= 0 else -((-value) // 2)


def accepted_rectangle(first, last):
    """Return integer (left,top,right,bottom), or None for a rejected point.

Preserve Qt's reverse-adjacent zero-width case, inclusive endpoints, truncating
center, and minimum 11-pixel size. Expansion is not clipped to the canvas;
Qwt also permits the corrected rectangle to extend past its input endpoints.
"""
    left, top = map(pixel, first)
    right, bottom = map(pixel, last)
    if right < left - 1: left, right = right, left
    if bottom < top - 1: top, bottom = bottom, top
    width, height = right - left + 1, bottom - top + 1
    if width < 2 and height < 2: return None
    center_x, center_y = _half(left + right), _half(top + bottom)
    width, height = max(11, width), max(11, height)
    left, top = center_x - _half(width - 1), center_y - _half(height - 1)
    return left, top, left + width - 1, top + height - 1


class KeyboardCrosshair:
    """Owned two-dimensional cursor and selection, independent of snapped samples."""
    ARROWS = {262: (1, 0), 263: (-1, 0), 264: (0, 1), 265: (0, -1)}
    REPEAT_DELAY = .4
    REPEAT_INTERVAL = .05

    def __init__(self):
        self.size = None
        self.extent = None
        self.position = self.last_pointer = None
        self.anchor = self.origin = None
        self.held = self.next_repeat = None
        self.visible = False

    @property
    def active(self): return self.anchor is not None

    def configure(self, width, height):
        if not all(math.isfinite(v) and 1 <= v <= 1 << 28 for v in (width, height)):
            changed = self.size is not None or self.visible
            self.cancel(clear_pointer=True)
            self.size = None
            self.extent = None
            return changed
        size = (int(width), int(height))
        if (width, height) == self.extent: return False
        self.cancel(clear_pointer=True)
        self.size = size
        self.extent = (width, height)
        return True

    def cancel(self, *, clear_pointer=False):
        changed = self.visible or self.active or self.held is not None
        self.visible = False
        self.anchor = self.origin = None
        self.stop_repeat()
        if clear_pointer: self.position = self.last_pointer = None
        return changed

    def stop_repeat(self):
        self.held = self.next_repeat = None

    def release(self, key):
        if key == self.held: self.stop_repeat()

    def pointer(self, x, y):
        """A changed physical pointer takes over an active selection endpoint.

Unchanged passive-hover notifications never overwrite keyboard motion. An
ordinary mouse press cancels the keyboard selection in the widget adapter.
"""
        if self.size is None or not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return False
        point = self._clamp((pixel(x), pixel(y)))
        if point == self.last_pointer: return False
        self.last_pointer = point
        if not self.visible: return False
        self.position = point
        self.stop_repeat()
        if not self.active: self.visible = False
        return True

    def _clamp(self, point):
        return tuple(max(0, min(value, extent - 1)) for value, extent in zip(point, self.size))

    def activate(self):
        if self.size is None: return False
        if not self.visible:
            self.position = self.last_pointer or tuple((extent - 1) // 2 for extent in self.size)
        self.visible = True
        return True

    def arrow(self, key, now):
        if key not in self.ARROWS or self.held == key or not self.activate(): return False
        self.held, self.next_repeat = key, now + self.REPEAT_DELAY
        self._step(1)
        return True

    def _step(self, amount):
        dx, dy = self.ARROWS[self.held]
        self.position = self._clamp((self.position[0] + dx * amount, self.position[1] + dy * amount))

    def repeat(self, now):
        if self.held is None or now < self.next_repeat: return False
        self._step(5)
        deadline = self.next_repeat + self.REPEAT_INTERVAL
        self.next_repeat = deadline if deadline > now else now + self.REPEAT_INTERVAL
        return True

    def begin(self, origin):
        if not self.activate(): return False
        self.stop_repeat()
        self.anchor, self.origin = self.position, origin
        return True

    def finish(self):
        if not self.active: return None
        result = accepted_rectangle(self.anchor, self.position), self.origin
        self.anchor = self.origin = None
        self.stop_repeat()
        return result
