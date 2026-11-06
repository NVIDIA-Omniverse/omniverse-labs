"""SDK-free concurrency controls for shared scene physics preparation."""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
import threading

import pytest

from ovrtx_blender_example import scene_generation_sessions as sessions


def harness(monkeypatch):
    transport = threading.RLock()
    attempted = threading.Event()
    finished = threading.Event()
    replayed = threading.Event()
    generation = SimpleNamespace(number=4)
    physics = SimpleNamespace(active_generation=None, last_error='', calls=[], stops=[])

    def activate(value, **kwargs):
        physics.calls.append(value)
        physics.active_generation = value.number
        return True

    def deactivate():
        physics.stops.append(physics.active_generation)
        physics.active_generation = None
        return 'stopped'

    physics.activate, physics.deactivate = activate, deactivate

    class Controller:
        @contextmanager
        def _serialized_transport(self):
            attempted.set()
            with transport:
                try:
                    yield
                finally:
                    finished.set()

    owner = SimpleNamespace(current_generation=generation, pending_generation=None,
                            retained_values_for=lambda _: ((), (), ()))
    runtime = SimpleNamespace(viewport_ids=set(), lifecycle_status='open',
        reuse_blocked=False, _preparation_lock=threading.RLock(), owner=owner,
        preparation_transport_finished=finished,
        _generation_number=None, scheduler=SimpleNamespace(), ovphysx=physics,
        close=lambda: True, replay_retained_values=lambda _: replayed.set())
    runtime.ovrtx = SimpleNamespace(controller=Controller(), active_generation=None,
        update_request=lambda _: None, last_ensure_result=None, last_error='')
    runtime.activate = MethodType(sessions.AuthoringGenerationRuntime.activate, runtime)
    runtime.activate_blocking = MethodType(sessions.AuthoringGenerationRuntime.activate_blocking, runtime)
    runtime._activate_ovrtx = lambda *_: True

    def finish(value, *_):
        runtime._generation_number = value.number
        runtime.ovrtx.active_generation = value.number
        return True

    runtime._finish_activation = finish
    for name, value in dict(_owners={12: owner}, _runtimes={12: runtime}, _dirty={},
            _camera_capture_owners={}, _preparation_pending={12: 1},
            _preparation_revisions={12: 1}, _preparation_condition=threading.Condition(),
            _preparation_stopping=False, _preparation_paused=False,
            _baked_drain_requests={}, _bake_capture_owner=None).items():
        monkeypatch.setattr(sessions, name, value)
    monkeypatch.setattr(sessions, 'generation_requires_physics', lambda _: True)
    return runtime, generation, transport, attempted, replayed


def stop_worker(worker):
    with sessions._preparation_condition:
        sessions._preparation_stopping = True
        sessions._preparation_condition.notify_all()
    worker.join(2)
    assert not worker.is_alive()


def test_background_and_viewport_activation_cannot_overlap_or_replace_same_owner(monkeypatch):
    runtime, generation, _, attempted, replayed = harness(monkeypatch)
    first_entered, release, overlapped = threading.Event(), threading.Event(), threading.Event()
    original_activate = runtime.ovphysx.activate
    errors = []

    def activate(value, **kwargs):
        if first_entered.is_set():
            overlapped.set()
        else:
            first_entered.set()
            assert release.wait(2), 'test did not release initial physics activation'
        return original_activate(value, **kwargs)

    runtime.ovphysx.activate = activate
    worker = threading.Thread(target=sessions._preparation_loop, daemon=True)

    def viewport_activate():
        try:
            runtime.activate_blocking(generation, object())
        except BaseException as exc:
            errors.append(exc)

    viewport = threading.Thread(target=viewport_activate, daemon=True)
    try:
        worker.start()
        assert first_entered.wait(2)
        attempted.clear()
        viewport.start()
        assert attempted.wait(2), 'viewport did not reach shared transport admission'
        assert not overlapped.wait(0.2), 'two callers entered physics activation concurrently'
    finally:
        release.set()
        viewport.join(2)
        stop_worker(worker)
    assert not viewport.is_alive()
    assert not errors
    assert replayed.is_set()
    assert runtime.ovphysx.calls == [generation]
    assert runtime.ovphysx.stops == []
    assert runtime.ovphysx.active_generation == generation.number


def test_viewport_preserves_physics_already_prepared_for_same_generation(monkeypatch):
    runtime, generation, *_ = harness(monkeypatch)
    runtime.ovphysx.active_generation = generation.number
    runtime.activate_blocking(generation, object())
    assert runtime.ovphysx.calls == []
    assert runtime.ovphysx.stops == []
    assert runtime.ovphysx.active_generation == generation.number


@pytest.mark.parametrize('requires_physics,active,baked,loads,stops', [
    (True, 3, False, 1, 0),
    (False, 3, False, 0, 1),
    (False, None, False, 0, 0),
    (True, 3, True, 0, 0),
])
def test_activation_preserves_replacement_removal_and_baked_ownership(
        monkeypatch, requires_physics, active, baked, loads, stops):
    runtime, generation, *_ = harness(monkeypatch)
    monkeypatch.setattr(sessions, 'generation_requires_physics', lambda _: requires_physics)
    runtime.ovphysx.active_generation = active
    runtime.scheduler.baked_playback_paused = baked
    runtime.activate_blocking(generation, object())
    assert len(runtime.ovphysx.calls) == loads
    assert len(runtime.ovphysx.stops) == stops
    if baked:
        assert runtime.ovphysx.active_generation == active


@pytest.mark.parametrize('change', ['generation', 'pending', 'dirty', 'closing',
                                   'runtime', 'owner', 'viewport', 'paused',
                                   'camera', 'reuse_blocked'])
def test_waiting_preparation_rechecks_ownership_before_physics_load(monkeypatch, change):
    runtime, generation, transport, attempted, replayed = harness(monkeypatch)
    worker = threading.Thread(target=sessions._preparation_loop, daemon=True)
    try:
        with transport:
            worker.start()
            assert attempted.wait(2)
            if change == 'generation': runtime.owner.current_generation = SimpleNamespace(number=5)
            elif change == 'pending': runtime.owner.pending_generation = SimpleNamespace(number=5)
            elif change == 'dirty': sessions._dirty[12] = True
            elif change == 'closing': runtime.lifecycle_status = 'closing'
            elif change == 'runtime': sessions._runtimes.pop(12)
            elif change == 'owner': sessions._owners[12] = SimpleNamespace(current_generation=generation)
            elif change == 'viewport': runtime.viewport_ids.add(1)
            elif change == 'paused': sessions._preparation_paused = True
            elif change == 'camera': sessions._camera_capture_owners[12] = object()
            elif change == 'reuse_blocked': runtime.reuse_blocked = True
        assert runtime.preparation_transport_finished.wait(2)
    finally:
        # Stop only after the worker inspected changed ownership; stopping
        # itself must not make the negative controls pass by cancelling earlier.
        stop_worker(worker)
    assert runtime.ovphysx.calls == []
    assert not replayed.is_set()
