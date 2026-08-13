# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest

from dev_variant_presenter.models import VariantChoice, VariantSetInfo
from dev_variant_presenter.sequence.timeline import (
    DEFAULT_CLIP_S, Clip, Timeline, TimelineError, Track,
    fill_track_all_variants, frame_times, make_mixer, make_slideshow,
    state_at, validate,
)


# ---- model + round-trip ----
def test_timeline_module_does_not_import_ovrtx():
    mod = importlib.import_module("dev_variant_presenter.sequence.timeline")
    text = open(mod.__file__, "r", encoding="utf-8").read()
    assert "import ovrtx" not in text and "from ovrtx" not in text and "import pxr" not in text


def test_dataclasses_are_frozen_and_typed():
    clip = Clip(value="Sakura", start_s=2.0, duration_s=3.0)
    track = Track(kind="variant_set", set_name="Carpaint", prim_path="/World/Looks", clips=(clip,))
    tl = Timeline(duration_s=10.0, fps=30.0, tracks=(track,))
    assert tl.fps == 30.0 and track.kind == "variant_set"
    with pytest.raises(Exception):
        clip.value = "Noir"


def test_to_dict_from_dict_round_trip():
    import json
    tl = Timeline(12.0, 24.0, (
        Track("variant_set", "Carpaint", "/World/Looks",
              (Clip("Sakura", 0.0, 4.0), Clip("Noir", 4.0, 4.0))),
        Track("camera", None, None, (Clip("/World/Cameras/Main_Cam_01", 0.0, 12.0),)),
    ))
    d = tl.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert Timeline.from_dict(d) == tl


def test_from_dict_defaults_optional_track_fields():
    d = {"duration_s": 5.0, "fps": 30.0,
         "tracks": [{"kind": "camera", "clips": [{"value": "/Cam", "start_s": 0.0, "duration_s": 5.0}]}]}
    tl = Timeline.from_dict(d)
    assert tl.tracks[0].set_name is None and tl.tracks[0].prim_path is None


# ---- validate ----
def _set_track(*clips):
    return Track("variant_set", "Carpaint", "/World/Looks", tuple(clips))


def test_validate_passes_for_sequential_clips():
    validate(Timeline(8.0, 30.0, (_set_track(Clip("A", 0.0, 4.0), Clip("B", 4.0, 4.0)),)))


def test_validate_raises_on_overlap():
    with pytest.raises(TimelineError):
        validate(Timeline(8.0, 30.0, (_set_track(Clip("A", 0.0, 5.0), Clip("B", 4.0, 4.0)),)))


def test_validate_raises_on_two_tracks_for_same_set():
    with pytest.raises(TimelineError):
        validate(Timeline(8.0, 30.0, (_set_track(Clip("A", 0.0, 4.0)), _set_track(Clip("B", 4.0, 4.0)))))


def test_validate_allows_distinct_sets_plus_camera():
    validate(Timeline(8.0, 30.0, (
        Track("variant_set", "Carpaint", "/World/Looks", (Clip("A", 0.0, 8.0),)),
        Track("variant_set", "Doors", "/World/Doors", (Clip("Open", 0.0, 8.0),)),
        Track("camera", None, None, (Clip("/Cam", 0.0, 8.0),)),
    )))


# ---- state_at ----
def _base():
    return (
        VariantChoice("/World/Looks", "Carpaint", "Noir"),
        VariantChoice("/World/Looks", "Wheel_Colors", "Chrome"),
        VariantChoice("/World/Doors", "Doors", "Closed"),
    )


def _v(selection, set_name):
    return next((c.variant for c in selection if c.set_name == set_name), None)


def test_clip_under_playhead_selects_variant():
    tl = Timeline(8.0, 30.0, (Track("variant_set", "Carpaint", "/World/Looks",
                 (Clip("Sakura", 0.0, 4.0), Clip("Sky", 4.0, 4.0))),))
    assert _v(state_at(tl, 1.0, _base())[0], "Carpaint") == "Sakura"
    assert _v(state_at(tl, 5.0, _base())[0], "Carpaint") == "Sky"


def test_gap_holds_previous():
    tl = Timeline(10.0, 30.0, (Track("variant_set", "Carpaint", "/World/Looks",
                 (Clip("Sakura", 0.0, 3.0), Clip("Sky", 6.0, 3.0))),))
    assert _v(state_at(tl, 4.5, _base())[0], "Carpaint") == "Sakura"


def test_before_first_clip_falls_back_to_base():
    tl = Timeline(10.0, 30.0, (Track("variant_set", "Carpaint", "/World/Looks",
                 (Clip("Sakura", 2.0, 3.0),)),))
    assert _v(state_at(tl, 0.5, _base())[0], "Carpaint") == "Noir"


def test_untracked_set_pinned_to_base():
    tl = Timeline(8.0, 30.0, (Track("variant_set", "Carpaint", "/World/Looks",
                 (Clip("Sakura", 0.0, 8.0),)),))
    sel, _ = state_at(tl, 1.0, _base())
    assert _v(sel, "Carpaint") == "Sakura" and _v(sel, "Wheel_Colors") == "Chrome"
    assert _v(sel, "Doors") == "Closed" and len(sel) == 3


def test_camera_track_returns_active_camera():
    tl = Timeline(10.0, 30.0, (Track("camera", None, None,
                 (Clip("/World/Cameras/Main_Cam_01", 0.0, 5.0),
                  Clip("/World/Cameras/Closeup_A", 5.0, 5.0))),))
    assert state_at(tl, 1.0, _base())[1] == "/World/Cameras/Main_Cam_01"
    assert state_at(tl, 6.0, _base())[1] == "/World/Cameras/Closeup_A"
    assert state_at(Timeline(5.0, 30.0, ()), 1.0, _base())[1] is None


# ---- frame_times ----
def test_frame_times_count_and_spacing():
    times = frame_times(Timeline(2.0, 30.0, ()))
    assert len(times) == 60 and times[0] == 0.0
    assert abs(times[1] - 1.0 / 30.0) < 1e-9


def test_frame_times_rounds_duration():
    assert len(frame_times(Timeline(1.05, 24.0, ()))) == 25  # round(25.2)


def test_frame_times_empty_for_zero_duration():
    assert frame_times(Timeline(0.0, 30.0, ())) == []


# ---- presets ----
def _carpaint():
    return VariantSetInfo("Carpaint", "/World/Looks", ("Noir", "Sakura", "Sky"), "Noir")


def _doors():
    return VariantSetInfo("Doors", "/World/Doors", ("Closed", "Open"), "Closed")


def test_fill_track_n_sequential_clips():
    track = fill_track_all_variants(_carpaint(), clip_s=2.0)
    assert track.set_name == "Carpaint" and track.prim_path == "/World/Looks"
    assert [c.value for c in track.clips] == ["Noir", "Sakura", "Sky"]
    assert [c.start_s for c in track.clips] == [0.0, 2.0, 4.0]
    assert all(c.duration_s == 2.0 for c in track.clips)


def test_fill_track_default_duration():
    assert all(c.duration_s == DEFAULT_CLIP_S for c in fill_track_all_variants(_carpaint()).clips)


def test_slideshow_one_track_sequential_across_sets():
    tl = make_slideshow([_carpaint(), _doors()], clip_s=1.5)
    assert len(tl.tracks) == 1
    clips = tl.tracks[0].clips
    assert len(clips) == 5
    assert [c.start_s for c in clips] == [0.0, 1.5, 3.0, 4.5, 6.0]
    assert abs(tl.duration_s - 7.5) < 1e-9


def test_mixer_parallel_tracks_lockstep():
    tl = make_mixer([_carpaint(), _doors()], clip_s=2.0)
    assert len(tl.tracks) == 2
    validate(tl)
    cp = next(t for t in tl.tracks if t.set_name == "Carpaint")
    dr = next(t for t in tl.tracks if t.set_name == "Doors")
    assert [c.value for c in cp.clips] == ["Noir", "Sakura", "Sky"]
    assert dr.clips[-1].end_s >= cp.clips[-1].end_s - 1e-9  # shorter track holds last value
    assert abs(tl.duration_s - 6.0) < 1e-9  # 3 steps * 2.0
