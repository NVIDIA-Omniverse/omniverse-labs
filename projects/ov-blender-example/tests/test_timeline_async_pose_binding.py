"""Submission pose identity and bounded asynchronous timeline acquisition."""
from dataclasses import replace
import pytest
from test_latest_view_render_loop import _warm_public_pipeline, RuntimeTickResult, RuntimeTickStatus


def moving_harness(tmp_path, monkeypatch):
    harness = _warm_public_pipeline(tmp_path, monkeypatch, timeline_controls_enabled=True)
    loop = harness.loop
    loop._cancel_pending_samples()
    loop._pipeline_cancelled_at_monotonic_ns = None
    loop._adopt_snapshot(replace(loop._snapshot, timeline_playing=True, timeline_frame=7))
    calls = []
    def tick(request, **kwargs):
        calls.append(request)
        return RuntimeTickResult(status=RuntimeTickStatus.STEPPED, enabled=True,
            generation=17, step_count=len(calls), should_reset_refinement=True,
            presentation_revision=len(calls), applied_revision=len(calls))
    harness.scheduler.tick_viewport = tick
    harness.handoffs = []
    loop._tick_result_sink = lambda tick, request: harness.handoffs.append((tick, request))
    loop._iterate()
    assert len(loop._pending_samples) == 1
    return harness, calls


def test_live_physics_stage_freezes_until_owned_pixels_and_keeps_submission_tick(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        pending = loop._pending_samples[0]
        count = len(harness.slot.frames())
        for frame in (8, 9, 10):
            loop._adopt_snapshot(replace(loop._snapshot, timeline_frame=frame))
            loop._iterate()
        assert len(calls) == 1  # producer may continue; renderer stage must not
        assert len(loop._pending_samples) == 1
        assert len(harness.slot.frames()) == count
        pending.remaining = 0
        loop._iterate()
        result = harness.slot.frames()[-1]
        assert result.runtime_tick_result.step_count == 1
        assert result.runtime_tick_result.generation == 17
        assert result.runtime_tick_timeline[1] == 7
        assert result.runtime_tick_playing is True
        assert harness.handoffs[-1][1].timeline_frame == 7
        assert harness.handoffs[-1][0] is result.runtime_tick_result
        assert len(loop._pending_samples) == 0  # never prearm old pose refinement
        loop._iterate()
        assert len(calls) == 2
        assert len(loop._pending_samples) == 1
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


@pytest.mark.parametrize('change', [{'timeline_playing': False}, {'timeline_frame': 1},
                                  {'timeline_start': 2}, {'camera_matrix': None},
                                  {'simulation_reset_token': 1}, {'timeline_controls_enabled': False},
                                  {'timeline_end': 200}])
def test_stop_rewind_and_view_identity_cancel_submitted_pose(tmp_path, monkeypatch, change):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        pending = loop._pending_samples[0]
        loop._adopt_snapshot(replace(loop._snapshot, **change))
        assert not loop._pending_samples
        assert not loop._pending_sample_contexts
        assert all(t.cancelled for t in pending.tickets.values())
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


@pytest.mark.parametrize('physics_enabled', [False, True])
@pytest.mark.parametrize('override', [None, '1', '2'])
def test_default_public_timeline_async_honors_explicit_synchronous_override(tmp_path, monkeypatch, override, physics_enabled):
    expected_async = override == '2' or (override is None and physics_enabled)
    from test_latest_view_render_loop import _Harness, _Client, _snapshot
    if override is None:
        monkeypatch.delenv('OVRTX_RENDER_PIPELINE_DEPTH', raising=False)
    else:
        monkeypatch.setenv('OVRTX_RENDER_PIPELINE_DEPTH', override)
    client = _Client(); client.async_supported = True
    client.worker_owned = True; client.camera_successor_deferral = True
    client.poll_resolve_after = 100
    harness = _Harness(monkeypatch, tmp_path, client)
    harness.scheduler.tick_viewport = lambda *args, **kwargs: RuntimeTickResult(
        status=RuntimeTickStatus.NOOP, enabled=physics_enabled)
    loop = harness.loop
    try:
        loop._adopt_snapshot(_snapshot(timeline_playing=True, max_samples=4))
        loop._iterate(); loop._iterate()
        assert bool(client.begin_render_calls) == expected_async
        assert len(loop._pending_samples) <= 1
    finally:
        loop._cancel_pending_samples()
        if loop._pipeline_loop is not None: loop._pipeline_loop.close()


def test_pose_timestamp_only_attaches_to_matching_generation_and_step():
    from types import SimpleNamespace
    from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest
    controller = SimpleNamespace(completed=False, step_count=12, current_simulation_time_ns=123,
        composition_generation=9, physics_pose_set=lambda now: (),
        last_update={'step_count':12,'composition_generation':9,'pose_produced_monotonic_ns':555})
    scheduler=RuntimeScheduler()
    def result():
        return scheduler._result_from_stage(None, RuntimeTickStatus.STEPPED,
            RuntimeTickRequest(input_usd_path='scene.usda'), timeline_reset=False, controller=controller)
    assert result().physics_pose_produced_monotonic_ns == 555
    controller.last_update['composition_generation']=8
    assert result().physics_pose_produced_monotonic_ns == 0
    controller.last_update=None
    assert result().physics_pose_produced_monotonic_ns == 0


def test_second_viewport_cancels_live_ticket_and_uses_specific_sync_fallback(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        pending=loop._pending_samples[0]
        monkeypatch.setattr(harness.controller, '_has_shared_presentations', lambda: True)
        loop._iterate()
        assert all(t.cancelled for t in pending.tickets.values())
        assert not loop._pending_samples
        assert loop.diagnostics()['timeline_acquisition_mode'] == 'sync_shared_presentations'
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_async_read_failure_cancels_ticket_and_retains_restoration_handoff(tmp_path, monkeypatch):
    from test_latest_view_render_loop import RenderClientError, FRAME_STATUS_FAILED
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        pending=loop._pending_samples[0]
        def fail(*args, **kwargs): raise RenderClientError('read failed')
        monkeypatch.setattr(harness.controller, 'poll_render', fail)
        loop._iterate()
        assert all(t.cancelled for t in pending.tickets.values())
        assert not loop._pending_samples
        assert harness.slot.frames()[-1].status == FRAME_STATUS_FAILED
        assert harness.handoffs[-1][0].step_count == 1
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_session_epoch_change_discards_live_submission_before_new_tick(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop=harness.loop
    try:
        pending=loop._pending_samples[0]
        harness.controller._session_revision += 1
        loop._iterate()
        assert all(t.cancelled for t in pending.tickets.values())
        assert id(pending) not in loop._pending_sample_contexts
        assert harness.slot.frames()[-1].runtime_tick_result.step_count == 2
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def delayed_native_cancel(harness):
    """Cancellation obsoletes pixels immediately but native work stays alive."""
    from types import SimpleNamespace
    state = SimpleNamespace(complete=False, polls=0)
    client = harness.client
    client.has_pending_render_cancellations = False
    client.pending_native_render_count = 1
    pending = harness.loop._pending_samples[0]
    for ticket in pending.tickets.values():
        previous = ticket.cancel
        def cancel(previous=previous):
            previous()
            client.has_pending_render_cancellations = not state.complete
        ticket.cancel = cancel
    def drain():
        if not client.has_pending_render_cancellations: return True
        state.polls += 1
        if not state.complete: return False
        client.has_pending_render_cancellations = False
        client.pending_native_render_count = 0
        return True
    client.drain_render_cancellations = drain
    client.cancel_pending_renders = lambda: [ticket.cancel() for ticket in pending.tickets.values()]
    return state


@pytest.mark.parametrize('change', [{'timeline_playing': False}, {'timeline_frame': 1}, {'simulation_reset_token': 1}])
def test_stop_and_rewind_keep_stage_fenced_after_logical_ticket_cancellation(tmp_path, monkeypatch, change):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    native = delayed_native_cancel(harness)
    try:
        publications = len(harness.slot.frames())
        loop._adopt_snapshot(replace(loop._snapshot, **change))
        assert not loop._pending_samples
        for _ in range(3): loop._iterate()
        assert len(calls) == 1 and len(harness.slot.frames()) == publications
        assert loop.diagnostics()['render_queue_depth'] == 1
        assert loop.diagnostics()['render_cancellation_fence']
        assert loop._take_timeout() == 0 and loop._work_due_without_new_snapshot()
        native.complete = True
        loop._iterate()
        assert len(calls) == 2
        assert harness.slot.frames()[-1].runtime_tick_result.step_count == 2
        assert not loop.diagnostics()['render_cancellation_fence']
    finally:
        native.complete = True
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_shared_viewport_cannot_mutate_before_displaced_native_ticket_completes(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    native = delayed_native_cancel(harness)
    try:
        for _ in range(2):
            with harness.controller._serialized_transport(presentation_key='other-viewport') as (_, deferred):
                assert deferred
        assert len(calls) == 1
        assert harness.client.has_pending_render_cancellations
        native.complete = True
        with harness.controller._serialized_transport(presentation_key='other-viewport') as (changed, deferred):
            assert changed and not deferred
    finally:
        native.complete = True
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_session_ensure_waits_for_native_cancellation_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    native = delayed_native_cancel(harness)
    ensured = []
    loop._lifecycle = SimpleNamespace(ensure_session=lambda request: ensured.append(request) or harness.scheduler)
    loop._ensure_pending = True
    loop._pending_ensure_reason = 'scene_composition_changed'
    try:
        assert loop._run_pending_ensure({}) is False
        assert not ensured and len(calls) == 1
        native.complete = True
        assert loop._run_pending_ensure({}) is True
        assert len(ensured) == 1
    finally:
        native.complete = True
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()


def test_exclusive_job_waits_without_dequeue_or_stage_mutation(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    native = delayed_native_cancel(harness)
    ran = []
    future = loop.call(lambda: ran.append('stage mutation'), label='native fence regression')
    try:
        assert loop._run_pending_jobs() is False
        assert not ran and not future.done()
        native.complete = True
        assert loop._run_pending_jobs() is True
        assert ran == ['stage mutation'] and future.done()
    finally:
        native.complete = True
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()
