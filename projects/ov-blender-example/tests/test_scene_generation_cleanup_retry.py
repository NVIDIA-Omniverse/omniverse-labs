# SPDX-License-Identifier: Apache-2.0
"""A file-load cleanup wait cannot permanently poison generation timers."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import scene_generation_sessions as sessions


@pytest.fixture
def waiting_scene(monkeypatch):
    callbacks, builds, repairs, updates = [], [], [], []
    scene = SimpleNamespace(session_uid=101, update_tag=lambda: updates.append(101))
    blender = SimpleNamespace(context=SimpleNamespace(scene=scene),
        app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn, **kw: callbacks.append(fn))))
    task = {'result': {'status': 'closing'}}
    monkeypatch.setattr(sessions, 'bpy', blender)
    monkeypatch.setattr(sessions, '_CLOSE_TASK', task)
    monkeypatch.setattr(sessions, '_viewport_generation_timers', set())
    monkeypatch.setattr(sessions, '_viewport_generation_errors', {})
    monkeypatch.setattr(sessions, '_initial_generation_revision', 0)
    monkeypatch.setattr(sessions, '_initial_generation_diagnostics', {'status': 'unavailable'})
    monkeypatch.setattr(sessions, 'repair_duplicate_identities', lambda scene: repairs.append(scene))
    generation = SimpleNamespace(number=0, digest='after-cleanup')
    monkeypatch.setattr(sessions, 'generation_for_scene', lambda scene: builds.append(scene) or generation)
    return SimpleNamespace(scene=scene, blender=blender, task=task, callbacks=callbacks,
        builds=builds, repairs=repairs, updates=updates, generation=generation)


def test_viewport_wait_retries_without_latching_or_duplicate_timers(waiting_scene):
    case = waiting_scene
    sessions._schedule_viewport_generation(case.scene, 101)
    timer = case.callbacks[0]
    assert timer() == 0.1
    assert timer() == 0.1
    assert not case.repairs and not case.builds
    assert sessions._viewport_generation_errors == {}
    sessions._schedule_viewport_generation(case.scene, 101)
    assert len(case.callbacks) == 1
    case.task['result']['status'] = 'closed'
    assert timer() is None
    assert case.builds == [case.scene]
    assert case.updates == [101]
    assert not sessions._viewport_generation_timers


def test_initial_generation_wait_retries_and_becomes_ready(waiting_scene):
    case = waiting_scene
    assert sessions.schedule_initial_generation(case.blender)
    timer = case.callbacks[0]
    assert timer() == 0.1
    assert sessions._initial_generation_diagnostics['status'] == 'scheduled'
    assert sessions._initial_generation_diagnostics['error'] == ''
    assert not case.repairs and not case.builds
    assert not sessions.schedule_initial_generation(case.blender)
    case.task['result']['status'] = 'closed'
    assert timer() is None
    assert sessions._initial_generation_diagnostics['status'] == 'ready'
    assert case.builds == [case.scene]


@pytest.mark.parametrize('initial', [False, True])
def test_a_second_file_load_cancels_waiting_generation(waiting_scene, initial):
    case = waiting_scene
    if initial:
        sessions.schedule_initial_generation(case.blender)
    else:
        sessions._schedule_viewport_generation(case.scene, 101)
    timer = case.callbacks[0]
    assert timer() == 0.1
    sessions._viewport_generation_timers.clear()
    sessions._invalidate_initial_generation()
    case.task['result']['status'] = 'closed'
    assert timer() is None
    assert not case.builds


@pytest.mark.parametrize('initial', [False, True])
def test_failed_cleanup_stays_terminal(waiting_scene, initial):
    case = waiting_scene
    case.task['result']['status'] = 'failed'
    if initial:
        sessions.schedule_initial_generation(case.blender)
    else:
        sessions._schedule_viewport_generation(case.scene, 101)
    assert case.callbacks[0]() is None
    if initial:
        assert sessions._initial_generation_diagnostics['status'] == 'failed'
    else:
        assert 'cleanup failed' in sessions._viewport_generation_errors[101]
        assert not sessions._viewport_generation_timers
    assert not case.builds
