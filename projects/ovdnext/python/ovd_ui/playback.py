# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame and clip state, independent of widgets and the native engine.

Capture frame counts and clip stops are exclusive. Frame zero is the native
initial state; the viewer plays from frame one. OVD records odd pre-simulation
and even post-simulation frames. A parity peek never changes the playback mode.
These conventions match PVDNext/mainwindow.cpp; boundary operations additionally
clamp to published frames, including trimmed clips and incomplete live steps.
"""
from dataclasses import dataclass
from enum import IntEnum


class FrameMode(IntEnum):
    FULL = 0
    PRE = 1
    POST = 2


@dataclass
class Clip:
    name: str
    start: int
    stop: int
    frame: int
    live_tail: bool = False

    @property
    def last(self):
        return max(self.start, self.stop - 1)


class Playback:
    def __init__(self):
        self.clips = []
        self.index = 0
        self.ovd = False
        self.mode = FrameMode.FULL
        self.direction = 0
        self.fps = 60.

    @property
    def clip(self):
        return self.clips[self.index] if self.clips else None

    @property
    def frame(self):
        return self.clip.frame if self.clip else 0

    @property
    def playing(self):
        return self.direction != 0

    @property
    def step_size(self):
        return 2 if self.ovd and self.mode != FrameMode.FULL else 1

    @property
    def label(self):
        if not self.ovd:
            return str(self.frame)
        return f"{(self.frame + 1) // 2} {'Pre' if self.frame % 2 else 'Post'}"

    def reset(self, frame_count=0, ovd=False):
        self.ovd = ovd
        self.mode = FrameMode.POST if ovd else FrameMode.FULL
        self.direction = 0
        self.index = 0
        start = min(1, max(0, int(frame_count) - 1))
        self.clips = [Clip("Capture", start, int(frame_count), start, live_tail=True)] if frame_count else []
        return self.first()

    def switch_scene(self, frame_count, ovd=False):
        """Keep capture clips; extend the active clip to the new scene's end.

        Qt retains the list/current row on scene changes. Bound every retained
        range to the destination timeline, and start at the first valid frame
        of the active range rather than publishing frame one outside a trim.
        """
        count=max(0,int(frame_count))
        if not self.clips or not count or self.ovd!=ovd:
            return self.reset(count,ovd)
        self.pause()
        for clip in self.clips:
            clip.live_tail=False
            clip.start=min(clip.start,count-1)
            clip.stop=max(clip.start+1,min(clip.stop,count))
            clip.frame=max(clip.start,min(clip.frame,clip.last))
        self.clip.stop=count
        self.clip.live_tail=True
        return self.first()

    def _bounds(self):
        """First and last matching frames; fallback for a partial sim step."""
        clip = self.clip
        if not clip:
            return 0, 0
        first, last = clip.start, clip.last
        if self.step_size == 2:
            parity = 1 if self.mode == FrameMode.PRE else 0
            first += (parity - first) % 2
            last -= (last - parity) % 2
            if first > last:
                return clip.start, clip.last
        return first, last

    def seek(self, frame, *, snap=True):
        if not self.clip:
            return 0
        clip = self.clip
        frame = max(clip.start, min(int(frame), clip.last))
        if snap and self.step_size == 2:
            first, last = self._bounds()
            parity = 1 if self.mode == FrameMode.PRE else 0
            if frame % 2 != parity:
                frame = frame + 1 if frame + 1 <= last else frame - 1
            frame = max(first, min(frame, last))
        clip.frame = frame
        return frame

    def set_mode(self, mode):
        self.mode = FrameMode(mode) if self.ovd else FrameMode.FULL
        return self.seek(self.frame)

    def peek_parity(self):
        if not self.ovd or not self.clip:
            return self.frame
        target = self.frame + (1 if self.frame % 2 else -1)
        if self.clip.start <= target < self.clip.stop:
            return self.seek(target, snap=False)
        return self.frame

    def _canonical(self):
        frame = self.frame
        if self.step_size == 2:
            if self.mode == FrameMode.PRE and frame % 2 == 0:
                frame -= 1
            elif self.mode == FrameMode.POST and frame % 2:
                frame += 1
        first, last = self._bounds()
        return max(first, min(frame, last))

    def step(self, direction=1):
        first, last = self._bounds()
        target = self._canonical() + self.step_size * (1 if direction > 0 else -1)
        return self.seek(max(first, min(target, last)), snap=False)

    def first(self):
        return self.seek(self._bounds()[0], snap=False)

    def last(self):
        return self.seek(self._bounds()[1], snap=False)

    def toggle(self, direction=1):
        if self.clip:
            direction = 1 if direction > 0 else -1
            self.direction = 0 if self.direction == direction else direction

    def pause(self):
        self.direction = 0

    def tick(self):
        if self.playing:
            before = self.frame
            self.step(self.direction)
            if self.frame == before or self.frame == self._bounds()[self.direction > 0]:
                self.pause()
        return self.frame

    def extend(self, frame_count, *, follow=False):
        """Extend the primary live clip only after native publication."""
        if not self.clips:
            self.reset(frame_count, self.ovd)
        elif frame_count > self.clips[0].stop:
            self.clips[0].stop = frame_count
        if follow:
            self.index = 0
            return self.last()
        return self.frame

    def select(self, index):
        if not 0 <= index < len(self.clips):
            raise IndexError("No such clip")
        self.pause()
        self.index = index
        return self.seek(self.frame)

    def split(self):
        clip = self.clip
        if not clip or clip.frame >= clip.last:
            return False
        split = clip.frame + 1
        self.clips[self.index] = Clip(clip.name + "_1", clip.start, split, clip.start)
        self.clips.append(Clip(clip.name + "_2", split, clip.stop, split, live_tail=clip.live_tail))
        self.pause()
        self.first()
        return True

    def trim_start(self):
        if self.clip:
            self.clip.start = self.frame

    def trim_end(self):
        # Legacy uses exclusive stop = playhead (the displayed end is stop-1).
        if self.clip and self.frame > self.clip.start:
            self.clip.stop = self.frame
            self.clip.live_tail = False
            self.seek(self.frame)

    def rename(self, name):
        if self.clip and name.strip():
            self.clip.name = name.strip()

    def delete(self, index=None):
        if len(self.clips) <= 1:
            return False
        index = self.index if index is None else index
        if not 0 <= index < len(self.clips):
            raise IndexError("No such clip")
        del self.clips[index]
        self.index = 0 if index == self.index else self.index - (index < self.index)
        self.pause()
        self.seek(self.frame)
        return True
