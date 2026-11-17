# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from ovd_ui.playback import Playback, FrameMode


def test_ovd_parity_peek_and_direction_keep_same_sim_step():
    playback = Playback()
    assert playback.reset(217, ovd=True) == 2
    playback.set_mode(FrameMode.PRE)
    playback.seek(211)
    assert playback.peek_parity() == 212
    assert playback.mode == FrameMode.PRE
    assert playback.step(-1) == 209
    playback.seek(211)
    playback.peek_parity()
    assert playback.step(1) == 213
    playback.set_mode(FrameMode.POST)
    playback.seek(212)
    assert playback.peek_parity() == 211
    assert playback.step(-1) == 210


def test_forward_reverse_stop_at_clip_boundaries_without_loop():
    playback = Playback()
    playback.reset(8)
    playback.seek(3)
    playback.trim_start()
    playback.toggle()
    assert [playback.tick() for _ in range(6)] == [4, 5, 6, 7, 7, 7]
    assert not playback.playing
    playback.toggle(-1)
    assert [playback.tick() for _ in range(6)] == [6, 5, 4, 3, 3, 3]
    assert not playback.playing


def test_clip_ranges_split_trim_and_last_clip_protection():
    playback = Playback()
    playback.reset(11)
    playback.seek(5)
    assert playback.split()
    assert [(c.start, c.stop) for c in playback.clips] == [(1, 6), (6, 11)]
    assert playback.select(1) == 6
    playback.seek(9)
    playback.trim_end()
    assert playback.clip.last == 8
    assert playback.frame == 8
    playback.rename("  Landing  ")
    assert playback.clip.name == "Landing"
    assert playback.delete()
    assert playback.clip.name == "Capture_1"
    assert not playback.delete()


def test_partial_live_steps_and_legacy_capture_modes():
    playback = Playback()
    assert playback.reset(2, ovd=True) == 1
    assert playback.peek_parity() == 1
    assert playback.extend(3, follow=True) == 2
    assert playback.extend(4, follow=True) == 2
    assert playback.extend(5, follow=True) == 4
    playback.reset(2, ovd=False)
    playback.set_mode(FrameMode.POST)
    assert playback.mode == FrameMode.FULL
    assert playback.peek_parity() == 1
    playback.reset()
    assert playback.tick() == playback.step() == playback.first() == playback.last() == 0


def test_shorter_scene_keeps_named_clips_with_valid_ranges_and_stops_playback():
    playback=Playback()
    playback.reset(20)
    playback.seek(10)
    playback.split()
    playback.select(1)
    playback.seek(16)
    playback.trim_start()
    playback.rename('Late events')
    playback.toggle()
    assert playback.switch_scene(6)==5
    assert playback.index==1 and not playback.playing
    assert [(c.name,c.start,c.stop,c.frame) for c in playback.clips]==[
        ('Capture_1',1,6,1),('Late events',5,6,5)]
    assert playback.step()==playback.step(-1)==5
    assert playback.switch_scene(25)==5
    assert [(c.start,c.stop) for c in playback.clips]==[(1,6),(5,25)]
    assert playback.select(0)==1


def test_scene_change_retains_frame_mode_with_bounded_initial_snapshot_fallback():
    playback=Playback()
    playback.reset(12,ovd=True)
    playback.set_mode(FrameMode.PRE)
    playback.seek(5)
    playback.split()
    playback.select(1)
    assert playback.switch_scene(16,ovd=True)==7
    assert playback.mode==FrameMode.PRE and playback.index==1
    assert [(c.start,c.stop) for c in playback.clips]==[(1,6),(6,16)]
    assert playback.switch_scene(1,ovd=True)==0
    assert all((c.start,c.stop,c.frame)==(0,1,0) for c in playback.clips)
    assert playback.switch_scene(0,ovd=True)==0 and not playback.clips
    assert playback.switch_scene(4,ovd=False)==1
    assert playback.mode==FrameMode.FULL and len(playback.clips)==1
