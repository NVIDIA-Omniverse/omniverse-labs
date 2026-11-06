# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hold live-PhysX timeline playback to the frame OVRTX last presented.

The OVRTX image shows the timeline frame it was submitted for, a few frames
behind ``frame_current``; keyframed objects and overlays are drawn at
``frame_current``. While a live-physics OVRTX viewport presents, this
``frame_change_pre`` handler keeps a playback step from moving more than one
frame past the presented frame: it puts ``frame_current`` back before the
frame is evaluated, and Blender's next playback step tries again.

The hold never applies under Sync to Audio, to backward steps (rewind, loop
wrap, reverse play), to jumps, to scrubbing or outside playback. Without a new
presented frame for ``TIMEOUT_S`` it records a stall and lets playback run
until the renderer presents again.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

try:
    from bpy.app.handlers import persistent  # type: ignore
except ModuleNotFoundError:
    def persistent(function: Any) -> Any:
        return function

#: Longest wait for a new presented frame before playback advances anyway.
TIMEOUT_S = 2.0

_clock = time.monotonic
# Scene uid -> the frame this handler last let through.
_last_frame: dict[int, int] = {}
# Scene uid -> (presented frame, monotonic time it was first seen while held).
_waiting: dict[int, tuple[int, float]] = {}
# Scene uid -> presented frame of a stall; the hold resumes once it changes.
_stalled: dict[int, int] = {}
_counts = {"held_steps": 0, "admitted_steps": 0, "stalls": 0}
_last_stall: dict[str, Any] | None = None


def _bpy() -> Any:
    try:
        import bpy  # type: ignore
    except ModuleNotFoundError:
        return None
    return bpy


def _playback_state() -> tuple[bool, bool]:
    """(timeline playing, user scrubbing) across the open windows."""
    bpy = _bpy()
    playing = scrubbing = False
    windows = getattr(getattr(getattr(bpy, "context", None), "window_manager", None), "windows", ())
    for window in windows:
        screen = getattr(window, "screen", None)
        playing = playing or bool(getattr(screen, "is_animation_playing", False))
        scrubbing = scrubbing or bool(getattr(screen, "is_scrubbing", False))
    return playing, scrubbing


def _presented_frame(scene: Any) -> int | None:
    """The live-physics timeline frame OVRTX last presented, or None."""
    from . import scene_generation_sessions
    if scene_generation_sessions.presentation_stopped(scene):
        return None
    from . import engine
    return engine.presented_physics_timeline_frame(scene)


def _max_step(scene: Any) -> int:
    """Largest forward change still treated as a playback step, not a jump."""
    if getattr(scene, "sync_mode", "NONE") != "FRAME_DROP":
        return 1
    render = getattr(scene, "render", None)
    fps = float(getattr(render, "fps", 24) or 24) / float(getattr(render, "fps_base", 1.0) or 1.0)
    return max(1, math.ceil(fps))


def enabled(scene: Any) -> bool:
    return bool(getattr(getattr(scene, "ov", None), "sync_playback_to_ovrtx_frames", False))


def target_frame(scene: Any, previous: int | None, current: int) -> int:
    """The frame playback may show now; ``current`` unless the hold applies."""
    uid = int(getattr(scene, "session_uid", 0) or 0)
    playing, scrubbing = _playback_state()
    step = current - previous if previous is not None else 0
    if (not enabled(scene) or not playing or scrubbing
            or getattr(scene, "sync_mode", "NONE") == "AUDIO_SYNC"
            or not 0 < step <= _max_step(scene)):
        _waiting.pop(uid, None)
        return current
    presented = _presented_frame(scene)
    if presented is None:
        _waiting.pop(uid, None)
        _stalled.pop(uid, None)
        return current
    if _stalled.get(uid, presented) != presented:
        del _stalled[uid]
    if current <= presented + 1 or uid in _stalled:
        _waiting.pop(uid, None)
        _counts["admitted_steps"] += 1
        return current
    now = _clock()
    seen = _waiting.get(uid)
    if seen is None or seen[0] != presented:
        seen = _waiting[uid] = (presented, now)
    if now - seen[1] >= TIMEOUT_S:
        _record_stall(uid, current, presented, now - seen[1])
        return current
    _counts["held_steps"] += 1
    # A Frame Dropping step may skip frames; it still stops one past the image.
    return min(current, max(previous, presented + 1))


def _record_stall(uid: int, frame: int, presented: int, waited_s: float) -> None:
    global _last_stall
    _waiting.pop(uid, None)
    _stalled[uid] = presented
    _counts["stalls"] += 1
    _last_stall = {"scene_uid": uid, "frame": frame, "presented_frame": presented,
                   "waited_s": round(waited_s, 3), "timeout_s": TIMEOUT_S}


@persistent
def frame_change_pre(scene: Any, *_unused: Any) -> None:
    """Put a held playback step back before Blender evaluates the frame."""
    if scene is None or threading.current_thread() is not threading.main_thread():
        return
    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid <= 0:
        return
    current = int(scene.frame_current)
    frame = target_frame(scene, _last_frame.get(uid), current)
    if frame != current:
        scene.frame_current = frame
    _last_frame[uid] = frame


def reset() -> None:
    global _last_stall
    _last_frame.clear()
    _waiting.clear()
    _stalled.clear()
    _counts.update(held_steps=0, admitted_steps=0, stalls=0)
    _last_stall = None


def diagnostics() -> dict[str, Any]:
    return {"timeout_s": TIMEOUT_S, **_counts, "holding_scene_uids": sorted(_waiting),
            "stalled_scene_uids": sorted(_stalled), "last_stall": _last_stall}
