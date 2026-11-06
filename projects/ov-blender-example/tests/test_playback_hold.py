# SPDX-License-Identifier: Apache-2.0
"""Live-PhysX playback waits for the OVRTX image of the frame it shows (H6)."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example  # noqa: E402
from ovrtx_blender_example import playback_hold as hold  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from test_scheduler_thread_pose_mirror import _load_engine_with_fake_bpy, _tick_result  # noqa: E402

ADDON = ROOT / "addon" / "ovrtx_blender_example"


class Scene(SimpleNamespace):
    def __init__(self, frame=10, sync_mode="NONE", enabled=True):
        super().__init__(session_uid=7, frame_current=frame, sync_mode=sync_mode,
                         render=SimpleNamespace(fps=30, fps_base=1.0),
                         ov=SimpleNamespace(sync_playback_to_ovrtx_frames=enabled))


@pytest.fixture
def live(monkeypatch):
    """Playback on, one live-physics viewport presenting ``state.presented``."""
    hold.reset()
    state = SimpleNamespace(presented=9, playing=True, scrubbing=False, now=100.0)
    monkeypatch.setattr(hold, "_playback_state", lambda: (state.playing, state.scrubbing))
    monkeypatch.setattr(hold, "_presented_frame", lambda _scene: state.presented)
    monkeypatch.setattr(hold, "_clock", lambda: state.now)
    yield state
    hold.reset()


def step(scene, to):
    """One Blender playback step: frame_current moves, then frame_change_pre runs."""
    scene.frame_current = to
    hold.frame_change_pre(scene)
    return scene.frame_current


def test_playback_hold_waits_for_presented_timeline_frame(live):
    scene = Scene(frame=10)
    hold.frame_change_pre(scene)  # first observation: nothing to compare yet
    assert step(scene, 11) == 10  # presented 9: 11 would lead by 2
    assert step(scene, 11) == 10
    live.presented = 10
    assert step(scene, 11) == 11  # lead 1 once frame 10 is presented
    assert step(scene, 12) == 11
    assert hold.diagnostics()["held_steps"] == 3
    # Frame Dropping skips frames but still stops one past the image.
    scene.sync_mode = "FRAME_DROP"
    live.presented = 11
    assert step(scene, 15) == 12


def test_playback_hold_never_blocks_scrub_or_rewind(live):
    scene = Scene(frame=10)
    hold.frame_change_pre(scene)
    live.scrubbing = True
    assert step(scene, 14) == 14  # scrub forward
    live.scrubbing = False
    assert step(scene, 3) == 3  # rewind / jump to start / loop wrap
    assert step(scene, 2) == 2  # reverse playback
    assert step(scene, 40) == 40  # jump to end during playback
    live.playing = False  # stopped: frame stepping and scripts
    assert step(scene, 41) == 41
    assert hold.diagnostics()["held_steps"] == 0


def test_playback_hold_released_on_retire(monkeypatch):
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(ovrtx_blender_example, "engine", module, raising=False)
    monkeypatch.setattr(module._DISPLAY_OVERLAY, "discard_owner", lambda _owner: None)
    scene = Scene(frame=10)
    viewport = module.OvrtxExampleRenderEngine()
    viewport._viewport_scene = scene
    viewport._presented_frame = SimpleNamespace(runtime_tick_result=_tick_result(module),
                                                runtime_tick_timeline=(True, 6, 1))
    monkeypatch.setattr(module, "_live_viewport_engines", lambda: [viewport])
    monkeypatch.setattr(sessions, "presentation_stopped", lambda _scene: False)
    monkeypatch.setattr(hold, "_playback_state", lambda: (True, False))
    hold.reset()
    hold.frame_change_pre(scene)
    assert step(scene, 11) == 10  # held by the committed presentation (frame 6)
    viewport._retire_physics_presentation()
    assert module.presented_physics_timeline_frame(scene) is None
    assert step(scene, 11) == 11
    # A stopped presentation (stop button, save, baked playback) releases it too.
    viewport._presented_frame = SimpleNamespace(runtime_tick_result=_tick_result(module),
                                                runtime_tick_timeline=(True, 6, 1))
    assert step(scene, 12) == 11
    monkeypatch.setattr(sessions, "presentation_stopped", lambda _scene: True)
    assert step(scene, 12) == 12
    hold.reset()


def test_playback_hold_on_by_default(live):
    tree = ast.parse((ADDON / "authoring_properties.py").read_text(encoding="utf-8"))
    group = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "OvSceneAuthoringProperties")
    field = next(n for n in group.body if isinstance(n, ast.AnnAssign) and n.target.id == "sync_playback_to_ovrtx_frames")
    keywords = {k.arg: k.value for k in field.annotation.keywords}
    assert field.annotation.func.id == "BoolProperty"
    assert ast.literal_eval(keywords["default"]) is True
    assert ast.literal_eval(keywords["name"]) == "Sync Playback to OVRTX Frames"
    assert f"after {hold.TIMEOUT_S:g} s without a new image" in ast.literal_eval(keywords["description"])
    assert '"sync_playback_to_ovrtx_frames"' in (ADDON / "physics_ui.py").read_text(encoding="utf-8")
    # Not physics authoring: toggling it never restarts the simulation.
    from ovrtx_blender_example import physics_edit_snapshot
    assert "sync_playback_to_ovrtx_frames" in physics_edit_snapshot.NON_PHYSICS_SCENE_SETTINGS
    scene = Scene(frame=10, enabled=False)
    hold.frame_change_pre(scene)
    assert step(scene, 11) == 11  # the user turned it off


def test_playback_hold_inactive_without_live_physx(monkeypatch):
    module = _load_engine_with_fake_bpy(monkeypatch)
    scene = Scene(frame=10)
    viewport = module.OvrtxExampleRenderEngine()
    viewport._viewport_scene = scene
    monkeypatch.setattr(module, "_live_viewport_engines", lambda: [viewport])
    assert module.presented_physics_timeline_frame(scene) is None  # nothing presented yet
    physics_free = module.RuntimeTickResult(status=module.RuntimeTickStatus.NOT_ENABLED, enabled=False,
                                            should_request_redraw=False)
    viewport._presented_frame = SimpleNamespace(runtime_tick_result=physics_free, runtime_tick_timeline=(True, 3, 1))
    assert module.presented_physics_timeline_frame(scene) is None  # ordinary OVRTX viewport
    viewport._presented_frame = SimpleNamespace(runtime_tick_result=_tick_result(module),
                                                runtime_tick_timeline=(False, 3, 1))
    assert module.presented_physics_timeline_frame(scene) is None  # timeline controls off
    viewport._presented_frame = SimpleNamespace(runtime_tick_result=_tick_result(module),
                                                runtime_tick_timeline=(True, 3, 1))
    assert module.presented_physics_timeline_frame(SimpleNamespace(session_uid=8)) is None
    assert module.presented_physics_timeline_frame(scene) == 3
    # Without a live-physics presentation, playback is ordinary Blender playback.
    hold.reset()
    monkeypatch.setattr(hold, "_playback_state", lambda: (True, False))
    monkeypatch.setattr(hold, "_presented_frame", lambda _scene: None)
    hold.frame_change_pre(scene)
    assert step(scene, 11) == 11 and step(scene, 12) == 12
    assert hold.diagnostics()["held_steps"] == 0
    hold.reset()


def test_playback_hold_not_applied_under_sync_to_audio(live):
    scene = Scene(frame=10, sync_mode="AUDIO_SYNC")
    hold.frame_change_pre(scene)
    assert step(scene, 11) == 11 and step(scene, 13) == 13
    scene.sync_mode = "NONE"
    assert step(scene, 14) == 13  # Play Every Frame holds (presented 9)


def test_playback_hold_times_out_on_render_stall(live):
    scene = Scene(frame=10)
    hold.frame_change_pre(scene)
    assert step(scene, 11) == 10
    live.now += hold.TIMEOUT_S - 0.25
    assert step(scene, 11) == 10
    live.now += 0.25
    assert step(scene, 11) == 11  # no new image for TIMEOUT_S: playback advances
    diagnostics = hold.diagnostics()
    assert diagnostics["stalls"] == 1
    assert diagnostics["last_stall"] == {"scene_uid": 7, "frame": 11, "presented_frame": 9,
                                         "waited_s": hold.TIMEOUT_S, "timeout_s": hold.TIMEOUT_S}
    assert step(scene, 12) == 12 and step(scene, 13) == 13  # stalled renderer never freezes playback
    live.presented = 12  # the renderer presents again: the hold resumes
    assert step(scene, 14) == 13
    live.now += hold.TIMEOUT_S / 2
    live.presented = 13
    assert step(scene, 14) == 14
    assert step(scene, 15) == 14
    live.now += hold.TIMEOUT_S * 0.9  # the wait restarts with each new image
    assert step(scene, 15) == 14
    assert hold.diagnostics()["stalls"] == 1
