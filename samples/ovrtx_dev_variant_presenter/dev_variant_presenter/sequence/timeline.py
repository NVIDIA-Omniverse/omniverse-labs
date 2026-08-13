# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeline (NLE) pure logic — data model, validation, State(t), frame sampling, presets.

PURE: imports only dataclasses/typing + dev_variant_presenter.models. NEVER ovrtx/pxr/renderer
(a test asserts this). The browser mirrors this in JS so scrubbing is instant client-side;
the server is authoritative for render. The render thread consumes `state_at` + `frame_times`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dev_variant_presenter.models import Selection, VariantChoice

TrackKind = Literal["variant_set", "camera"]
DEFAULT_CLIP_S: float = 2.0


class TimelineError(ValueError):
    """Raised when a Timeline violates its invariants."""


@dataclass(frozen=True)
class Clip:
    value: str          # variant name (variant_set track) or camera prim path (camera track)
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def to_dict(self) -> dict:
        return {"value": self.value, "start_s": self.start_s, "duration_s": self.duration_s}

    @staticmethod
    def from_dict(d: dict) -> "Clip":
        return Clip(d["value"], float(d["start_s"]), float(d["duration_s"]))


@dataclass(frozen=True)
class Track:
    kind: TrackKind
    set_name: str | None
    prim_path: str | None
    clips: tuple[Clip, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "set_name": self.set_name, "prim_path": self.prim_path,
                "clips": [c.to_dict() for c in self.clips]}

    @staticmethod
    def from_dict(d: dict) -> "Track":
        return Track(d["kind"], d.get("set_name"), d.get("prim_path"),
                     tuple(Clip.from_dict(c) for c in d.get("clips", ())))


@dataclass(frozen=True)
class Timeline:
    duration_s: float
    fps: float
    tracks: tuple[Track, ...]

    def to_dict(self) -> dict:
        return {"duration_s": self.duration_s, "fps": self.fps,
                "tracks": [t.to_dict() for t in self.tracks]}

    @staticmethod
    def from_dict(d: dict) -> "Timeline":
        return Timeline(float(d["duration_s"]), float(d["fps"]),
                        tuple(Track.from_dict(t) for t in d.get("tracks", ())))


def validate(timeline: Timeline) -> None:
    """Raise TimelineError if invariants are violated: each variant set has at most one
    track, clips have positive duration and do not overlap within a track."""
    seen_sets: set[str] = set()
    for track in timeline.tracks:
        if track.kind == "variant_set":
            if track.set_name is None:
                raise TimelineError("variant_set track requires set_name")
            if track.set_name in seen_sets:
                raise TimelineError(f"more than one track targets variant set '{track.set_name}'")
            seen_sets.add(track.set_name)
        prev_end = None
        for clip in sorted(track.clips, key=lambda c: c.start_s):
            if clip.duration_s <= 0:
                raise TimelineError(f"clip '{clip.value}' has non-positive duration {clip.duration_s}")
            if prev_end is not None and clip.start_s < prev_end - 1e-9:
                raise TimelineError(f"overlapping clips on track '{track.set_name or track.kind}'")
            prev_end = clip.end_s


def _value_at(track: Track, t: float) -> str | None:
    """Clip covering t, else the previous clip's value (a gap holds the last value), else None."""
    held: str | None = None
    for clip in sorted(track.clips, key=lambda c: c.start_s):
        if clip.start_s <= t + 1e-9:
            held = clip.value
            if t < clip.end_s - 1e-9:
                return clip.value
        else:
            break
    return held


def state_at(timeline: Timeline, t: float, base_selection: Selection) -> tuple[Selection, str | None]:
    """Compose the full Selection + active camera path at time t. variant_set track →
    clip/gap/base; untracked sets pinned to base; camera track → active camera path.
    BEFORE the first camera clip the first clip's camera already governs — falling back
    to 'whatever camera is live in the viewport' rendered surprise lead-ins."""
    track_values: dict[str, str] = {}
    camera_path: str | None = None
    for track in timeline.tracks:
        if track.kind == "camera":
            camera_path = _value_at(track, t)
            if camera_path is None and track.clips:
                camera_path = min(track.clips, key=lambda c: c.start_s).value
            continue
        v = _value_at(track, t)
        if v is not None and track.set_name is not None:
            track_values[track.set_name] = v
    composed = [VariantChoice(b.prim_path, b.set_name, track_values.get(b.set_name, b.variant))
                for b in base_selection]
    return tuple(composed), camera_path


def camera_clip_at(timeline: Timeline, t: float) -> tuple[str, float] | None:
    """(camera_path, clip_start_s) of the camera clip active (or HELD through a gap,
    mirroring _value_at) at time t. Before the first camera clip, the FIRST clip
    answers (its start clamps the clip-relative time to 0 — lead-in holds frame 0)."""
    found: tuple[str, float] | None = None
    for track in timeline.tracks:
        if track.kind != "camera":
            continue
        clips = sorted(track.clips, key=lambda c: c.start_s)
        if clips and found is None:
            found = (clips[0].value, clips[0].start_s)   # lead-in: first clip governs
        for clip in clips:
            if clip.start_s <= t + 1e-9:
                found = (clip.value, clip.start_s)
            else:
                break
    return found


def loop_stage_time(rel_seconds: float, stage_fps: float, start: float, end: float) -> float:
    """Map seconds-into-a-clip to a stage timecode, looping over the stage's authored
    range (a 360 turntable rig loops seamlessly; clips longer than the animation wrap)."""
    span = end - start + 1.0
    if span <= 1.0 or stage_fps <= 0:
        return start
    return start + (max(0.0, rel_seconds) * stage_fps) % span


def frame_times(timeline: Timeline) -> list[float]:
    """Sample times [0, duration) at fps (round to nearest frame count)."""
    n = round(timeline.duration_s * timeline.fps)
    return [i / timeline.fps for i in range(n)]


def fill_track_all_variants(set_info, clip_s: float = DEFAULT_CLIP_S) -> Track:
    """One sequential clip per variant of a set."""
    clips = tuple(Clip(v, i * clip_s, clip_s) for i, v in enumerate(set_info.variants))
    return Track("variant_set", set_info.set_name, set_info.prim_path, clips)


def make_slideshow(sets, clip_s: float = DEFAULT_CLIP_S) -> Timeline:
    """One track stepping through every (set, variant). Render-only sequence (not validate()-able
    — a single track holds clips for multiple sets)."""
    clips, t = [], 0.0
    prim_path = sets[0].prim_path if sets else None
    set_name = sets[0].set_name if sets else None
    for s in sets:
        for v in s.variants:
            clips.append(Clip(v, t, clip_s)); t += clip_s
    return Timeline(t, 30.0, (Track("variant_set", set_name, prim_path, tuple(clips)),))


def make_mixer(sets, clip_s: float = DEFAULT_CLIP_S) -> Timeline:
    """Parallel tracks (one per set) in lockstep; shorter tracks hold their final value."""
    steps = max((len(s.variants) for s in sets), default=0)
    tracks = []
    for s in sets:
        raw = [Clip(s.variants[min(i, len(s.variants) - 1)], i * clip_s, clip_s) for i in range(steps)]
        merged: list[Clip] = []
        for c in raw:
            if merged and merged[-1].value == c.value and abs(merged[-1].end_s - c.start_s) < 1e-9:
                last = merged[-1]
                merged[-1] = Clip(last.value, last.start_s, last.duration_s + c.duration_s)
            else:
                merged.append(c)
        tracks.append(Track("variant_set", s.set_name, s.prim_path, tuple(merged)))
    return Timeline(steps * clip_s, 30.0, tuple(tracks))
