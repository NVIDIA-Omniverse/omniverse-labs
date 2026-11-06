"""Negative controls for moving physics continuity and uncertain-step evidence."""
from copy import deepcopy
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
import asyncio

import pytest

from physics_live_recovery_checks import assert_live_progress, assert_uncertain_not_replayed, assert_fresh_frame, RecentNativeFrames, survival_boundary, assert_renderer_not_reloaded
from test_physics_continuity_checks import snapshot


def moving_pair():
    before = snapshot()
    before.update(timestep_ns=10000000, step_count=8, controller_time_ns=80000000,
                  producer_activity=dict(playback_active=True, thread_alive=True, stop_requested=False),
                  producer={'producer_step_count': 9, 'producer_simulation_time_ns': 90000000,
                            'in_flight': True, 'producer_completed': False})
    after = deepcopy(before)
    after.update(step_count=18, controller_time_ns=180000000, backend_time_ns=200000000,
                 producer={'producer_step_count': 19, 'producer_simulation_time_ns': 190000000,
                           'in_flight': True, 'producer_completed': False})
    after['states'][0]['simulation_time_ns'] = 200000000
    after['states'][0]['translate']['x'] += .025
    return before, after


def test_live_progress_accepts_independent_presentation_lag():
    before, after = moving_pair()
    assert assert_live_progress(before, after)['fixed_steps'] == 10


@pytest.mark.parametrize('fault', ['owner', 'stage', 'reset', 'stalled', 'frozen_producer',
                                   'wrong_velocity', 'jump', 'missing', 'nan', 'time', 'norm', 'paused', 'stopped'])
def test_invalid_continuity_rejected(fault):
    before, after = moving_pair()
    if fault == 'owner': after['pid'] += 1
    elif fault == 'stage': after['stage_identity'] += 1
    elif fault == 'reset': after['controller_time_ns'] = 0
    elif fault == 'stalled': after = deepcopy(before)
    elif fault == 'frozen_producer': after['producer'] = deepcopy(before['producer'])
    elif fault == 'wrong_velocity': after['states'][0]['linear_velocity']['x'] = .5
    elif fault == 'jump': after['states'][0]['translate']['x'] += .1
    elif fault == 'missing': after['states'] = []
    elif fault == 'nan': after['states'][0]['translate']['x'] = float('nan')
    elif fault == 'time': after['backend_time_ns'] = 1
    elif fault == 'norm': after['states'][0]['orient']['r'] = 2
    elif fault == 'paused': after['producer_activity']['playback_active'] = False
    elif fault == 'stopped': after['producer_activity']['stop_requested'] = True
    with pytest.raises(AssertionError):
        assert_live_progress(before, after)


@pytest.mark.parametrize('fault', ['pid', 'epoch', 'restarts', 'request_counts', 'uncertain_operation'])
def test_uncertain_step_cannot_replay_or_replace(fault):
    before = dict(pid=1, launcher_pid=2, epoch=1, restarts=0,
                  request_counts={'load': 1, 'step': 5}, uncertain_operation='step')
    after = deepcopy(before)
    assert_uncertain_not_replayed(before, after)
    if fault == 'request_counts': after[fault]['step'] += 1
    elif fault == 'uncertain_operation': after[fault] = ''
    else: after[fault] += 1
    with pytest.raises(AssertionError):
        assert_uncertain_not_replayed(before, after)


@pytest.mark.parametrize('fault', ['held_redraw', 'old_submission', 'old_copy', 'new_owner', 'new_stage'])
def test_renderer_survival_requires_new_native_submission_copy_and_exact_draw(fault):
    native = renderer_native()
    demand = dict(armed_at=10., submission_sequence=20, copy_sequence=19, renderer=native)
    proof = dict(submission=dict(started=11., sequence=21, pid=10, epoch=1),
                 copy_sequence=20, copied_at=12., native=deepcopy(native))
    assert_fresh_frame(demand, proof)
    if fault == 'held_redraw': proof['submission']['started'] = 9.
    elif fault == 'old_submission': proof['submission']['sequence'] = 20
    elif fault == 'old_copy': proof['copy_sequence'] = 19
    elif fault == 'new_owner': proof['native']['runtime_pid'] = 11
    else: proof['native']['native_stage_id'] = 6
    with pytest.raises(AssertionError):
        assert_fresh_frame(demand, proof)


def native_result():
    from ovrtx_blender_example.ovrtx_runtime_client import RenderResult
    return RenderResult(width=1, height=1, rgba8=memoryview(bytearray([1, 2, 3, 255])).toreadonly(),
                        completed_samples=1, session_completed_samples=1, simulation_time_ns=10,
                        native_timings={'runtime_pid': 10}, product_id='/render/product')


def renderer_native():
    return dict(runtime_pid=10, runtime_epoch=1, simulation_id='scene', native_stage_id=5,
                stage_ordinal=9, scene_path='scene.usda', worker_load_requests=1)


def test_native_frame_proofs_are_bounded_and_use_actual_carrier_identity():
    frames = RecentNativeFrames(limit=2)
    first, second, third = native_result(), native_result(), native_result()
    frames.record(first, 'first')
    assert frames.proof_for(first) == 'first'
    assert frames.proof_for(second) is None
    frames.record(second, 'second')
    frames.record(third, 'third')
    assert frames.proof_for(first) is None
    assert frames.proof_for(second) == 'second' and frames.proof_for(third) == 'third'


@pytest.mark.parametrize('field', RecentNativeFrames._FIELDS)
def test_same_carrier_with_changed_frame_identity_is_rejected(field):
    frames = RecentNativeFrames()
    result = native_result()
    frames.record(result, 'proof')
    value = getattr(result, field)
    changed = value + 1 if isinstance(value, int) else value + '-changed'
    assert frames.proof_for(replace(result, **{field: changed})) is None


def test_equal_distinct_copy_or_timing_and_mutable_or_empty_planes_rejected():
    frames = RecentNativeFrames()
    result = native_result()
    frames.record(result, 'proof')
    assert frames.proof_for(replace(result, rgba8=bytes(result.rgba8))) is None
    assert frames.proof_for(replace(result, native_timings=dict(result.native_timings))) is None
    for plane in (b'', bytearray([1, 2, 3, 255]), memoryview(bytearray([1, 2, 3, 255]))):
        invalid = replace(result, rgba8=plane)
        frames.record(invalid, 'invalid')
        assert frames.proof_for(invalid) is None


@pytest.mark.parametrize('pipeline', [False, True])
def test_actual_render_loop_replacements_preserve_exact_copy_proof(pipeline):
    from ovrtx_blender_example.viewport_render_thread import LatestViewRenderLoop
    frames = RecentNativeFrames()
    original = native_result()
    native = renderer_native()
    proof = dict(submission=dict(started=11., sequence=21, pid=10, epoch=1),
                 copy_sequence=20, copied_at=12., native=native)
    frames.record(original, proof)
    loop = LatestViewRenderLoop(
        mailbox=None, frame_slot=None, scheduler=None,
        request_for_snapshot=lambda snapshot: None,
        controller=SimpleNamespace(render=lambda *a, **k: original),
    )
    loop._snapshot = SimpleNamespace(max_samples=1, written_monotonic_ns=1, timing_marks={})
    loop._request = object()
    marks = {}
    if pipeline:
        loop._pipeline_loop = asyncio.new_event_loop()
        loop._pending_samples = deque([SimpleNamespace(native_started_ns=1)])
        loop._pending_sample_snapshots = {id(loop._pending_samples[0]): loop._snapshot}
        loop._pending_sample_contexts = {id(loop._pending_samples[0]): (None, loop._request)}
        loop._pipeline_depth = 2
        async def poll(pending): return original
        loop._poll_pending_async = poll
        try:
            wrapped = loop._acquire_sample_pipelined(marks)
            assert loop._resolved_sample_snapshot is loop._snapshot
            assert loop._pending_sample_snapshots == {}
        finally:
            loop._pipeline_loop.close()
    else:
        wrapped = loop._acquire_sample_sync(marks)
    assert wrapped is not original  # The actual production replacement caused live-v3 failure.
    # Engine draw also changes only the displayed sample count.
    drawn = replace(wrapped, completed_samples=7)
    assert frames.proof_for(drawn) is proof
    demand = dict(armed_at=10., submission_sequence=20, copy_sequence=19, renderer=native)
    assert_fresh_frame(demand, frames.proof_for(drawn))
    # Correctly associated old native copies still cannot satisfy a later demand.
    with pytest.raises(AssertionError):
        assert_fresh_frame(dict(demand, armed_at=12.), frames.proof_for(drawn))


def boundary_pair():
    restarted = renderer_native()
    baseline = dict(copy_sequence=40, copied_at=19.,
                    submission=dict(started=18., sequence=40, pid=10, epoch=1),
                    native=dict(restarted, stage_ordinal=15))
    demand = survival_boundary(restarted, baseline, armed_at=20.,
                               submission_sequence=41, copy_sequence=40)
    fresh = dict(copy_sequence=42, copied_at=22.,
                 submission=dict(started=21., sequence=42, pid=10, epoch=1),
                 native=dict(baseline['native']))
    return restarted, baseline, demand, fresh


def test_content_revision_can_advance_before_but_not_after_survival_boundary():
    restarted, baseline, demand, fresh = boundary_pair()
    assert restarted['stage_ordinal'] == 9
    assert demand['renderer']['stage_ordinal'] == 15
    assert demand['baseline_copy'] is baseline
    assert_fresh_frame(demand, fresh)
    fresh['native']['stage_ordinal'] = 16
    with pytest.raises(AssertionError):
        assert_fresh_frame(demand, fresh)


@pytest.mark.parametrize('fault', ['owner', 'epoch', 'stage', 'path', 'reload', 'regressed', 'after_arm', 'old_copy', 'future_submit'])
def test_boundary_rejects_wrong_owner_reload_or_invalid_copy(fault):
    restarted, baseline, _, _ = boundary_pair()
    fields = dict(owner='runtime_pid', epoch='runtime_epoch', stage='native_stage_id',
                  path='scene_path', reload='worker_load_requests')
    if fault in fields:
        key = fields[fault]
        baseline['native'][key] = 'other' if fault == 'path' else baseline['native'][key] + 1
    elif fault == 'regressed': baseline['native']['stage_ordinal'] = 8
    elif fault == 'after_arm': baseline['copied_at'] = 21.
    elif fault == 'old_copy': baseline['copy_sequence'] = 39
    else: baseline['submission']['sequence'] = 42
    with pytest.raises(AssertionError):
        survival_boundary(restarted, baseline, armed_at=20., submission_sequence=41, copy_sequence=40)


@pytest.mark.parametrize('field', ['pid', 'epoch', 'launcher_pid', 'restarts', 'load'])
def test_renderer_diagnostics_reject_hidden_replacement_or_reload(field):
    before = dict(pid=10, epoch=1, launcher_pid=2, restarts=0, request_counts={'load': 1, 'render': 3})
    after = deepcopy(before)
    after['request_counts']['render'] += 1
    assert_renderer_not_reloaded(before, after)
    if field == 'load': after['request_counts']['load'] += 1
    else: after[field] += 1
    with pytest.raises(AssertionError):
        assert_renderer_not_reloaded(before, after)
