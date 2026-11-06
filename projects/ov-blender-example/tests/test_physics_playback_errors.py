# SPDX-License-Identifier: Apache-2.0
"""Rejected physics rebuilds are reported to the user, not only to diagnostics."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from ovrtx_blender_example import user_messages  # noqa: E402

SCENE = SimpleNamespace(session_uid=91, render=SimpleNamespace(engine="OVRTX_EXAMPLE"))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    user_messages.default_bus().reset()
    monkeypatch.setattr(sessions, "_playback_errors", {})
    monkeypatch.setattr(sessions, "_presentation_stopped", set())
    monkeypatch.setattr(sessions, "_baked_active", lambda _scene: False)
    yield
    user_messages.default_bus().reset()


def messages():
    return [(m.level, m.text) for m in user_messages.default_bus().take_pending()]


def test_stop_holds_presented_sample_but_save_restores_authored_state(monkeypatch):
    calls = []
    runtime = SimpleNamespace(playback_lock=SimpleNamespace(clear=lambda **kw: calls.append('restore')),
                              _wake_presentations=lambda: calls.append('wake'))
    monkeypatch.setattr(sessions, '_runtimes', {91: runtime})
    monkeypatch.setattr(sessions, '_bake_capture_owner', None)
    monkeypatch.setattr(sessions, '_baked_playback_paused', set())
    monkeypatch.setattr(sessions, 'bpy', None)
    epoch = sessions._presentation_epoch
    sessions.hold_physics_presentation(SCENE)
    assert calls == ['wake'] and not sessions.presentation_stopped(SCENE)
    assert sessions._presentation_epoch > epoch
    sessions.save_physics_presentation(SCENE)
    assert calls == ['wake', 'restore'] and sessions.presentation_stopped(SCENE)


def play(monkeypatch, error=None):
    def demand(_scene, **_kwargs):
        if error is not None:
            raise error
        sessions._clear_playback_error(SCENE.session_uid)
    monkeypatch.setattr(sessions, "demand_physics_playback", demand)
    sessions.animation_playback_pre(SCENE)


def test_rejected_rebuild_is_reported_once_and_shown_in_status(monkeypatch):
    error = ValueError("Duplicate Object identity; explicitly repair duplicates")
    play(monkeypatch, error)
    reported = messages()
    assert len(reported) == 1 and reported[0][0] == "ERROR"
    assert "Physics not updated: ValueError: Duplicate Object identity" in reported[0][1]
    assert "previous simulation is still playing" in reported[0][1]
    assert "Duplicate Object identity" in sessions.physics_playback_error(SCENE)
    play(monkeypatch, error)
    assert messages() == [], "Unchanged error was reported again"


def test_success_clears_status_and_recurrence_is_reported_again(monkeypatch):
    error = ValueError("Endpoint has no enabled authored OV body")
    play(monkeypatch, error)
    messages()
    play(monkeypatch)
    assert sessions.physics_playback_error(SCENE) == ""
    assert [level for level, _ in messages()] == ["INFO"]
    play(monkeypatch)
    assert messages() == [], "Success without a prior error is silent"
    play(monkeypatch, error)
    assert [level for level, _ in messages()] == ["ERROR"]


def test_distinct_errors_are_each_reported(monkeypatch):
    play(monkeypatch, ValueError("first"))
    play(monkeypatch, ValueError("second"))
    assert [text.split(": ", 2)[2].split(";")[0] for _, text in messages()] == ["first", "second"]


def test_blender_playback_rejection_is_reported(tmp_path):
    import subprocess
    from blender_test_support import blender_executable
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run Blender playback-error coverage")
    fixture = Path(__file__).with_name("blender_physics_playback_errors.py")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(fixture), "--", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "PHYSICS_PLAYBACK_ERRORS_OK" in result.stdout
