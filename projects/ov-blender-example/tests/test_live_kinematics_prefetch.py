# SPDX-License-Identifier: Apache-2.0
"""Kinematic prefetch: slice staleness and cancellable main-thread waits."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example.engine as engine_module  # noqa: E402
from ovrtx_blender_example import main_thread_waits  # noqa: E402
from ovrtx_blender_example import physics_live_kinematics as plk  # noqa: E402

TS, F0, FPS = 1_000_000_000 // 60, 1, 24.0


def test_key_edit_between_slices_discards_earlier_slices():
    anim, calls = {"x": 1.0}, [0]
    source = plk.LiveKinematicSource(-77, object())

    def on_main(callback):
        calls[0] += 1
        if calls[0] == 2:  # The user keys between slice 1 and slice 2.
            anim["x"] = 5.0
            plk.note_animation_edit()
        return callback(None)

    source._on_main = on_main
    source._evaluate_steps = lambda scene, steps, *a: (
        {s: (anim["x"], s) for s in steps[:10]}, plk._animation_epoch[0])
    try:
        source.prefetch(range(1, 21), TS, F0, FPS)
        with pytest.raises(plk.KinematicSampleNotReady):
            source.sample(1, TS, F0, FPS)
        source.prefetch(range(1, 21), TS, F0, FPS)
        assert [source.sample(s, TS, F0, FPS)[0] for s in range(1, 21)] == [5.0] * 20
    finally:
        plk.forget_scene(-77)


def test_stopping_owner_cancels_a_main_thread_wait(monkeypatch):
    queued = []
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(
        app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn, first_interval: queued.append(fn)))))
    ran, outcome = [], {}

    def worker():
        try:
            engine_module._run_on_main_thread(lambda: ran.append(True), timeout=30)
        except main_thread_waits.MainThreadWaitCancelled as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    while not queued:
        thread.join(0.01)
    # The main thread stops the owner instead of servicing the queued call.
    assert main_thread_waits.cancel_waits_of(thread, "stopping") == 1
    thread.join(5)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), main_thread_waits.MainThreadWaitCancelled)
    queued[0]()  # The late timer must not run a cancelled call.
    assert ran == []
    assert main_thread_waits._waits == {}


def test_completed_main_thread_call_is_not_cancelled(monkeypatch):
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(
        app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn, first_interval: fn()))))
    results = []
    thread = threading.Thread(target=lambda: results.append(engine_module._run_on_main_thread(lambda: 7)))
    thread.start()
    thread.join(5)
    assert results == [7]
    assert main_thread_waits.cancel_waits_of(thread, "stopping") == 0


def test_cancellation_is_sticky_for_waits_started_later(monkeypatch):
    queued = []
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(
        app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn, first_interval: queued.append(fn)))))
    cancelled, outcome = threading.Event(), {}

    def worker():
        cancelled.wait(5)  # Between prefetch slices when the owner stops.
        try:
            engine_module._run_on_main_thread(lambda: None, timeout=30)
        except main_thread_waits.MainThreadWaitCancelled as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert main_thread_waits.cancel_waits_of(thread, "stopping") == 0
    cancelled.set()
    thread.join(5)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), main_thread_waits.MainThreadWaitCancelled)
    assert queued == []


def test_physics_snapshot_admits_render_init_on_the_job_thread():
    from ovrtx_blender_example import physics_edit_snapshot

    errors = []

    def job():
        for admitted in (False, True):
            try:
                if admitted:
                    with physics_edit_snapshot.render_init_capture():
                        physics_edit_snapshot.capture(SimpleNamespace())
                else:
                    physics_edit_snapshot.capture(SimpleNamespace())
            except Exception as exc:  # noqa: BLE001 - classified below
                errors.append((admitted, "main thread" in str(exc)))

    thread = threading.Thread(target=job)
    thread.start()
    thread.join(5)
    assert (False, True) in errors
    assert (True, True) not in errors
