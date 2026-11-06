# SPDX-License-Identifier: Apache-2.0
"""Saving restores authored state, then the live physics viewport presents again."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402

SCENE = SimpleNamespace(session_uid=91, render=SimpleNamespace(engine="OVRTX_EXAMPLE"))
OTHER = SimpleNamespace(session_uid=92, render=SimpleNamespace(engine="OVRTX_EXAMPLE"))
# Blender passes the file path to save handlers, not a scene.
SAVED_PATH = "C:/projects/physics.blend"


def runtime(calls, key):
    return SimpleNamespace(
        playback_lock=SimpleNamespace(clear=lambda **kw: calls.append(("restore", key))),
        _wake_presentations=lambda: calls.append(("wake", key)))


@pytest.fixture
def state(monkeypatch):
    calls = []
    monkeypatch.setattr(sessions, "_runtimes", {91: runtime(calls, 91), 92: runtime(calls, 92)})
    monkeypatch.setattr(sessions, "_presentation_stopped", set())
    monkeypatch.setattr(sessions, "_save_stopped_presentations", set())
    monkeypatch.setattr(sessions, "_bake_capture_owner", None)
    monkeypatch.setattr(sessions, "_baked_playback_paused", set())
    monkeypatch.setattr(sessions, "_baked_active", lambda _scene: False)
    monkeypatch.setattr(sessions, "bpy", None)
    return calls


def test_save_post_resumes_the_presentation_the_save_stopped(state):
    sessions.save_physics_presentation(SAVED_PATH)
    assert ("restore", 91) in state and sessions.presentation_stopped(SCENE)
    epoch = sessions._presentation_epoch
    sessions.resume_saved_physics_presentation(SAVED_PATH)
    assert not sessions.presentation_stopped(SCENE) and not sessions.presentation_stopped(OTHER)
    assert {("wake", 91), ("wake", 92)} <= set(state)
    assert sessions._presentation_epoch > epoch
    assert sessions._save_stopped_presentations == set()


def test_failed_save_also_resumes(state):
    sessions.save_physics_presentation(SAVED_PATH)
    sessions.resume_saved_physics_presentation(SAVED_PATH)  # save_post_fail runs the same handler
    assert not sessions.presentation_stopped(SCENE)


def test_stop_owned_before_the_save_stays_stopped(state):
    sessions._presentation_stopped.add(91)  # e.g. the stop button restored authoring
    sessions.save_physics_presentation(SAVED_PATH)
    sessions.resume_saved_physics_presentation(SAVED_PATH)
    assert sessions.presentation_stopped(SCENE)
    assert not sessions.presentation_stopped(OTHER)
    assert ("wake", 91) not in state and ("wake", 92) in state


def test_baked_playback_pause_stays_stopped(state):
    sessions.save_physics_presentation(SAVED_PATH)
    sessions._baked_playback_paused.add(92)
    sessions.resume_saved_physics_presentation(SAVED_PATH)
    assert not sessions.presentation_stopped(SCENE)
    assert 92 in sessions._presentation_stopped and ("wake", 92) not in state


def test_bake_capture_owner_short_circuits_resume(state, monkeypatch):
    sessions._presentation_stopped.add(91)
    sessions._save_stopped_presentations.add(91)
    monkeypatch.setattr(sessions, "_bake_capture_owner", object())
    epoch = sessions._presentation_epoch
    sessions.resume_saved_physics_presentation(SAVED_PATH)
    assert 91 in sessions._presentation_stopped and not state
    assert sessions._presentation_epoch == epoch


def test_resume_without_a_save_does_nothing(state):
    epoch = sessions._presentation_epoch
    sessions.resume_saved_physics_presentation(SAVED_PATH)
    assert sessions._presentation_epoch == epoch and not state


def test_save_post_handlers_are_registered_and_removed(monkeypatch):
    class Timers:
        def __init__(self):
            self.registered = set()

        def is_registered(self, fn):
            return fn in self.registered

        def register(self, fn, **_kwargs):
            self.registered.add(fn)

        def unregister(self, fn):
            self.registered.discard(fn)

    names = ("load_pre", "load_post", "animation_playback_pre", "animation_playback_post",
             "save_pre", "save_post", "save_post_fail", "undo_pre", "redo_pre", "frame_change_post")
    blender = SimpleNamespace(app=SimpleNamespace(
        handlers=SimpleNamespace(**{name: [] for name in names}), timers=Timers()))
    monkeypatch.setattr(sessions, "_start_preparation_worker", lambda: None)
    monkeypatch.setattr(sessions, "close", lambda **_kwargs: None)
    sessions.register_handlers(blender)
    assert sessions.save_physics_presentation in blender.app.handlers.save_pre
    assert sessions.resume_saved_physics_presentation in blender.app.handlers.save_post
    assert sessions.resume_saved_physics_presentation in blender.app.handlers.save_post_fail
    sessions.unregister_handlers(blender)
    assert not blender.app.handlers.save_post and not blender.app.handlers.save_post_fail


def test_viewport_presents_the_held_physics_frame_after_save(monkeypatch):
    """Without a resume, every frame after the save is refused and the view stays gray."""
    from test_scheduler_thread_pose_mirror import _load_engine_with_fake_bpy, _tick_result

    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(ovrtx_blender_example, "engine", module, raising=False)
    scene = SimpleNamespace(session_uid=41)
    viewport = module.OvrtxExampleRenderEngine()
    viewport._viewport_scene = scene
    frame = SimpleNamespace(runtime_tick_result=_tick_result(module),
                            runtime_tick_timeline=(False, 1, 1), publication_index=7)
    viewport._frame_slot = SimpleNamespace(peek_latest=lambda: frame)
    monkeypatch.setattr(module, "_live_viewport_engines", lambda: [viewport])
    monkeypatch.setattr(module._DISPLAY_OVERLAY, "discard_work", lambda _owner: None)
    monkeypatch.setattr(sessions, "current_presentation_context", lambda _scene: (5, sessions._presentation_epoch, ()))
    monkeypatch.setattr(module.operator_state, "prepare_runtime_pose_mirror",
                        lambda *args, **kwargs: ({"poses_by_path": {"/World/Cube": "held"}}, {"enabled": True}))
    from ovrtx_blender_example import physics_pose_transaction
    monkeypatch.setattr(physics_pose_transaction, "apply",
                        lambda *args, **kwargs: {"enabled": True, "status": "applied"})
    calls = []
    monkeypatch.setattr(sessions, "_runtimes", {41: runtime(calls, 41)})
    monkeypatch.setattr(sessions, "_presentation_stopped", set())
    monkeypatch.setattr(sessions, "_save_stopped_presentations", set())
    monkeypatch.setattr(sessions, "_bake_capture_owner", None)
    monkeypatch.setattr(sessions, "_baked_playback_paused", set())
    monkeypatch.setattr(sessions, "_baked_active", lambda _scene: False)
    monkeypatch.setattr(sessions, "bpy", SimpleNamespace(
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: None))))

    # The held sample is presented before the save.
    viewport._queue_frame_presentation(frame)
    module.bpy.app.timers.run_pending()
    assert viewport._frame_for_presentation(frame) is frame

    sessions.save_physics_presentation(SAVED_PATH)
    assert viewport._frame_for_presentation(frame) is None  # authored state while saving
    # The stranded state: further publications are refused while stopped.
    viewport._queue_frame_presentation(frame)
    module.bpy.app.timers.run_pending()
    assert viewport._frame_for_presentation(frame) is None

    sessions.resume_saved_physics_presentation(SAVED_PATH)
    module.bpy.app.timers.run_pending()
    assert ("wake", 41) in calls
    assert viewport._frame_for_presentation(frame) is frame
    assert viewport._commit_frame_pose_mirror(frame)
