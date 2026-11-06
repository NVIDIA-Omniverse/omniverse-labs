from dataclasses import replace

import pytest

from test_latest_view_render_loop import _Harness, _Client, _snapshot, RuntimeTickResult, RuntimeTickStatus
from ovrtx_blender_example.shared_stage_composition import BodyPose


def exhausted(tmp_path, monkeypatch, depth):
    monkeypatch.setenv('OVRTX_RENDER_PIPELINE_DEPTH', str(depth))
    monkeypatch.setenv('OVRTX_RENDER_PIPELINE_POLL_BUDGET_MS', '0')
    client = _Client()
    client.async_supported = True
    client.worker_owned = True
    client.camera_successor_deferral = True
    harness = _Harness(monkeypatch, tmp_path, client)
    tick = RuntimeTickResult(status=RuntimeTickStatus.NOOP, enabled=True, generation=2,
        step_count=10, simulation_time_ns=100,
        physics_pose_set=(BodyPose('/Body', (1., 2., 3.), (0., 0., 0., 1.)),))
    state = [tick]
    harness.scheduler.tick_viewport = lambda *args, **kwargs: state[0]
    harness.loop._adopt_snapshot(_snapshot(timeline_controls_enabled=True,
        timeline_playing=True, timeline_frame=74, max_samples=16))
    for _ in range(40):
        harness.loop._iterate()
        if harness.loop._completed_samples == 16: break
    assert harness.loop._completed_samples == 16
    assert not harness.loop._pending_samples
    return harness, state


@pytest.mark.parametrize('depth', [1, 2])
@pytest.mark.parametrize('transition', ['generation', 'stop_same_generation', 'stop_new_generation', 'rewind', 'reset'])
def test_exhausted_budget_gets_one_fresh_refinement_after_ownership_transition(tmp_path, monkeypatch, depth, transition):
    harness, state = exhausted(tmp_path, monkeypatch, depth)
    loop = harness.loop
    old = harness.slot.frames()[-1]
    before = len(harness.slot.frames())
    try:
        if transition in ('generation', 'stop_new_generation'):
            state[0] = replace(state[0], generation=3)
        changes = {'timeline_playing': False} if transition.startswith('stop') else (
            {'timeline_frame': 1} if transition == 'rewind' else
            {'simulation_reset_token': 1} if transition == 'reset' else {})
        loop._adopt_snapshot(replace(loop._snapshot, **changes))
        for _ in range(40):
            loop._iterate()
            if len(harness.slot.frames()) > before and loop._completed_samples == 16: break
        new = harness.slot.frames()[before:]
        assert [frame.completed_samples for frame in new] == list(range(1, 17))
        assert all(frame.runtime_tick_result is state[0] for frame in new)
        assert all(frame.render_result is not old.render_result for frame in new)
        assert all(frame.generation == state[0].generation for frame in new)
        assert loop._take_timeout() is None
        assert not loop._work_due_without_new_snapshot()
        for _ in range(3): loop._iterate()
        assert len(harness.slot.frames()) == before + 16
        assert not loop._timeline_transition_pending
    finally:
        loop._cancel_pending_samples()
        if loop._pipeline_loop: loop._pipeline_loop.close()


def test_failed_tick_preserves_pending_stop_transition(tmp_path, monkeypatch):
    harness, state = exhausted(tmp_path, monkeypatch, 1)
    loop = harness.loop
    before = len(harness.slot.frames())
    loop._adopt_snapshot(replace(loop._snapshot, timeline_playing=False))
    state[0] = replace(state[0], status=RuntimeTickStatus.FAILED)
    loop._iterate()
    assert loop._timeline_transition_pending
    assert loop._completed_samples == 16
    assert all(frame.status != 'frame' for frame in harness.slot.frames()[before:])


@pytest.mark.parametrize('stop', [False, True])
def test_busy_default_generation_does_not_acquire_or_consume_transition(tmp_path, monkeypatch, stop):
    harness, state = exhausted(tmp_path, monkeypatch, 1)
    loop = harness.loop
    valid = state[0]
    before = len(harness.slot.frames())
    if stop:
        loop._adopt_snapshot(replace(loop._snapshot, timeline_playing=False))
    state[0] = RuntimeTickResult(status=RuntimeTickStatus.BUSY, enabled=True)
    loop._iterate()
    assert loop._generation == valid.generation
    assert loop._completed_samples == 16
    assert len(harness.slot.frames()) == before
    assert loop._timeline_transition_pending is stop
    state[0] = replace(valid, generation=valid.generation if stop else valid.generation + 1)
    loop._iterate()
    assert len(harness.slot.frames()) == before + 1
    assert harness.slot.frames()[-1].runtime_tick_result is state[0]
    assert loop._completed_samples == 1
    assert not loop._timeline_transition_pending


def test_forward_timeline_motion_does_not_reset_settled_budget(tmp_path, monkeypatch):
    harness, state = exhausted(tmp_path, monkeypatch, 1)
    loop = harness.loop
    before = len(harness.slot.frames())
    loop._adopt_snapshot(replace(loop._snapshot, timeline_frame=75))
    loop._iterate()
    assert not loop._timeline_transition_pending
    assert loop._completed_samples == 16
    assert len(harness.slot.frames()) == before


def test_native_cancellation_fence_does_not_consume_stop_transition(tmp_path, monkeypatch):
    from test_timeline_async_pose_binding import moving_harness, delayed_native_cancel
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    native = delayed_native_cancel(harness)
    try:
        loop._adopt_snapshot(replace(loop._snapshot, timeline_playing=False))
        loop._iterate()
        assert loop._timeline_transition_pending and len(calls) == 1
        native.complete = True
        loop._iterate()
        assert not loop._timeline_transition_pending and len(calls) == 2
    finally:
        native.complete = True
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()
