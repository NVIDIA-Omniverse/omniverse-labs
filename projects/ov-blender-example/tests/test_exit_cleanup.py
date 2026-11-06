"""Quit cleanup runs before Blender destroys RNA and native plugin state."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine, scene_generation_sessions, worker_process_reaper, public_sdk_install, render_sequence


@pytest.mark.parametrize("stopped", [True, False])
def test_exit_closes_current_owners_before_process_backstop(monkeypatch, stopped):
    events = []
    monkeypatch.setattr(engine, "stop_viewport_sessions_for_unregister",
        lambda: events.append("viewport") or stopped)
    monkeypatch.setattr(scene_generation_sessions, "close", lambda: events.append("scene"))
    monkeypatch.setattr(render_sequence, "shutdown", lambda: events.append("sequence"))
    monkeypatch.setattr(public_sdk_install, "close", lambda: events.append("install"))
    monkeypatch.setattr(worker_process_reaper, "reap", lambda: events.append("reap"))
    addon._exit_pre()
    assert events == ["sequence", "install", "viewport", "scene", "reap"]


def test_reaper_targets_explicit_owner_and_not_legacy_executable(monkeypatch):
    killed=[]
    monkeypatch.setattr(worker_process_reaper, "_worker_basenames", set())
    monkeypatch.setattr(worker_process_reaper, "_explicit_pids", set())
    monkeypatch.setattr(worker_process_reaper, "_windows_child_worker_pids", lambda names, own: set())
    monkeypatch.setattr(worker_process_reaper, "_terminate_pid", lambda pid: killed.append(pid) or True)
    worker_process_reaper.register_worker_pid(4242)
    assert worker_process_reaper.reap() == 1
    assert killed == [4242]
    assert not worker_process_reaper._DEFAULT_WORKER_BASENAMES
