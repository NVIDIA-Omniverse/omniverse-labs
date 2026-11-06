from dataclasses import replace
from types import SimpleNamespace

import pytest

from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest, RuntimeTickResult, RuntimeTickStatus
from ovrtx_blender_example.shared_stage_composition import BodyPose
from test_timeline_async_pose_binding import moving_harness


def accepted():
    scheduler = RuntimeScheduler()
    scheduler._controller = object()
    port = SimpleNamespace(client=SimpleNamespace(runtime=SimpleNamespace(epoch=1)), simulation_id='render')
    request = RuntimeTickRequest(input_usd_path='scene')
    tick = RuntimeTickResult(status=RuntimeTickStatus.STEPPED, enabled=True,
        generation=3, step_count=12, simulation_time_ns=100,
        physics_pose_set=(BodyPose('/Body', (1., 2., 3.), (0., 0., 0., 1.)),),
        physics_pose_produced_monotonic_ns=12345)
    scheduler._last_pose_projection = {'result': {'native_pose_route': 'native_inputs', 'transform_count': 1}}
    scheduler._retain_pose_provenance(tick, request, port, True)
    return scheduler, port, request, replace(tick, status=RuntimeTickStatus.REUSED_LATEST,
        physics_pose_produced_monotonic_ns=0)


@pytest.mark.parametrize('camera_write', [False, True])
def test_same_pose_retains_original_receipt_without_claiming_a_new_projection(camera_write):
    scheduler, port, request, tick = accepted()
    if camera_write:
        tick = replace(tick, values_written=True, update={'update_result': {
            'values_written': True, 'result': {'camera_updated': True}}})
    result = scheduler._retain_pose_provenance(tick, request, port, False)
    assert result.physics_pose_produced_monotonic_ns == 12345
    receipt = result.update['update_result']['result']
    assert receipt['native_pose_route'] == 'native_inputs'
    assert receipt['pose_projection_reused'] is True
    assert receipt['current_update_result'] == ({'camera_updated': True} if camera_write else {})
    assert result.values_written is camera_write
    assert result.step_count == tick.step_count and result.status == RuntimeTickStatus.REUSED_LATEST
    assert scheduler._pose_projection_count == 0


def test_public_tick_boundary_keeps_accepted_metadata_across_refinement(monkeypatch):
    scheduler, port, request, reused = accepted()
    scheduler._accepted_pose_provenance = None
    calls = []
    def tick(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            scheduler._pose_projection_count += 1
            return replace(reused, physics_pose_produced_monotonic_ns=12345,
                should_reset_refinement=True, values_written=True)
        return reused
    monkeypatch.setattr(scheduler, '_tick_viewport', tick)
    first = scheduler.tick_viewport(request, ovrtx_updates=port)
    second = scheduler.tick_viewport(request, ovrtx_updates=port)
    assert first.physics_pose_produced_monotonic_ns == second.physics_pose_produced_monotonic_ns == 12345
    assert first.applied_revision == second.applied_revision
    assert second.update['update_result']['result']['pose_projection_reused'] is True
    assert not second.values_written
    assert scheduler._pose_projection_count == 1


@pytest.mark.parametrize('transition', ['same_session_reload', 'population', 'recovery', 'failed'])
def test_direct_projection_token_invalidates_native_receipt(transition):
    scheduler, port, request, tick = accepted()
    state = {'token': (id(port.client), 100, 1, 1)}
    port.client.direct_projection_token = lambda simulation_id: state['token']
    original = replace(tick, physics_pose_produced_monotonic_ns=12345)
    scheduler._retain_pose_provenance(original, request, port, True)
    stable = scheduler._retain_pose_provenance(tick, request, port, False)
    assert stable.physics_pose_produced_monotonic_ns == 12345
    if transition == 'same_session_reload': state['token'] = (id(port.client), 101, 1, 1)
    elif transition == 'population': state['token'] = (id(port.client), 100, 1, 2)
    else: state['token'] = None
    invalidated = scheduler._retain_pose_provenance(tick, request, port, False)
    assert invalidated.physics_pose_produced_monotonic_ns == 0
    assert not invalidated.update
    if transition in ('recovery', 'failed'):
        assert scheduler._accepted_pose_provenance is None
        state['token'] = (id(port.client), 100, 1, 1)
        assert scheduler._retain_pose_provenance(tick, request, port, False).physics_pose_produced_monotonic_ns == 0


@pytest.mark.parametrize('change', ['controller', 'client', 'epoch', 'simulation', 'scene', 'reset_token',
    'generation', 'step_count', 'simulation_time_ns', 'poses', 'reset', 'failed', 'disabled'])
def test_provenance_never_crosses_pose_or_owner_boundary(change):
    scheduler, port, request, tick = accepted()
    if change == 'controller': scheduler._controller = object()
    elif change == 'client': port.client = object()
    elif change == 'epoch': port.client.runtime.epoch += 1
    elif change == 'simulation': port.simulation_id = 'replacement'
    elif change == 'scene': request = replace(request, input_usd_path='other')
    elif change == 'reset_token': request = replace(request, simulation_reset_token=1)
    elif change in ('generation', 'step_count', 'simulation_time_ns'):
        tick = replace(tick, **{change: getattr(tick, change) + 1})
    elif change == 'poses': tick = replace(tick, physics_pose_set=(BodyPose('/Body', (2., 2., 3.), (0., 0., 0., 1.)),))
    elif change == 'reset': tick = replace(tick, timeline_reset=True)
    elif change == 'failed': tick = replace(tick, status=RuntimeTickStatus.FAILED)
    elif change == 'disabled': tick = replace(tick, enabled=False)
    result = scheduler._retain_pose_provenance(tick, request, port, False)
    assert result.physics_pose_produced_monotonic_ns == 0
    assert not result.update


def test_async_update_spans_stay_with_each_submitted_pose_and_render_boundary(tmp_path, monkeypatch):
    harness, calls = moving_harness(tmp_path, monkeypatch)
    loop = harness.loop
    try:
        prior = None
        for step in (1, 2):
            pending = loop._pending_samples[0]
            submitted_marks = dict(loop._pending_sample_update_marks[id(pending)])
            assert submitted_marks['runtime_update_completed_monotonic_ns'] >= submitted_marks['runtime_update_started_monotonic_ns']
            assert submitted_marks != prior
            pending.remaining = 0
            loop._iterate()
            shown = harness.slot.frames()[-1]
            assert shown.runtime_tick_result.step_count == step
            assert shown.timing_marks['render_call_started_monotonic_ns'] == pending.native_started_ns
            for key, value in submitted_marks.items(): assert shown.timing_marks[key] == value
            assert shown.timing_marks['render_call_completed_monotonic_ns'] >= pending.native_started_ns
            assert not loop._pending_sample_update_marks
            prior = submitted_marks
            if step == 1: loop._iterate()
    finally:
        loop._cancel_pending_samples()
        loop._pipeline_loop.close()
