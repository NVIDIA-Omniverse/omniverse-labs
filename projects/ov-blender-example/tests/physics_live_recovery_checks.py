"""Acceptance for uninterrupted native producer telemetry, without pausing it."""
import math
from collections import deque

RENDER_OWNER_FIELDS = ('runtime_pid', 'runtime_epoch', 'simulation_id',
                       'native_stage_id', 'scene_path', 'worker_load_requests')


class RecentNativeFrames:
    """Retain the exact owned copy across sample-count-only result wrappers.

    The render loop and engine replace RenderResult to update completed_samples.
    They retain the actual read-only pixel carrier and native timings object.
    Strong references prevent recycled ids; equal pixel contents are insufficient.
    """
    _FIELDS = ('width', 'height', 'session_completed_samples', 'simulation_time_ns',
               'render_output_simulation_time_ns', 'frame_format', 'frame_color_mode',
               'render_var', 'rgba8_origin', 'product_id')

    def __init__(self, limit=32):
        self.frames = deque(maxlen=limit)

    def record(self, result, proof):
        self.frames.append((result, proof))

    def latest_proof(self):
        return self.frames[-1][1] if self.frames else None

    def proof_for(self, result):
        plane = result.rgba8
        if not plane or not (isinstance(plane, bytes) or
                             isinstance(plane, memoryview) and plane.readonly):
            return None
        for observed, proof in reversed(tuple(self.frames)):
            if (observed.rgba8 is plane and observed.native_timings is result.native_timings
                    and all(getattr(observed, key) == getattr(result, key) for key in self._FIELDS)
                    and observed.linear_rgba16f is result.linear_rgba16f
                    and observed.presentation_rgba32f is result.presentation_rgba32f
                    and observed.semantic_capture is result.semantic_capture):
                return proof
        return None


OWNER_FIELDS = ('object_ids', 'pid', 'launcher_pid', 'epoch', 'session',
                'stage_identity', 'physics_identity', 'runtime_identity',
                'cpu_only_policy', 'physics_generation', 'expected_paths', 'timestep_ns')


def validate_live(row):
    assert row['pid'] > 0 and row['epoch'] > 0 and row['session']
    assert row['loaded'] and row['stage_identity'] and row['physics_identity']
    assert row['cpu_only_policy'] is True and row['backend_time_ns'] > 0
    assert row['producer_activity'] == dict(playback_active=True, thread_alive=True, stop_requested=False)
    assert 0 <= row['controller_time_ns'] <= row['backend_time_ns']
    assert 0 <= row['producer']['producer_simulation_time_ns'] <= row['backend_time_ns']
    assert row['controller_time_ns'] == row['step_count'] * row['timestep_ns']
    assert row['producer']['producer_simulation_time_ns'] == row['producer']['producer_step_count'] * row['timestep_ns']
    assert len(row['states']) == len(row['expected_paths']) > 0
    assert {s['prim_path'] for s in row['states']} == set(row['expected_paths'])
    for state in row['states']:
        assert state['simulation_time_ns'] == row['backend_time_ns']
        for field, axes in (('translate', 'xyz'), ('orient', 'ijkr'),
                            ('linear_velocity', 'xyz'), ('angular_velocity', 'xyz')):
            assert state[field]['found'] is True
            assert all(math.isfinite(state[field][axis]) for axis in axes)
        assert abs(math.hypot(*(state['orient'][a] for a in 'ijkr')) - 1) < 1e-4


def assert_live_progress(before, after):
    validate_live(before)
    validate_live(after)
    for key in OWNER_FIELDS:
        assert before[key] == after[key], ('physics owner replaced', key)
    assert after['backend_time_ns'] > before['backend_time_ns']
    assert after['producer']['producer_step_count'] > before['producer']['producer_step_count']
    assert after['step_count'] >= before['step_count']
    dt = (after['backend_time_ns'] - before['backend_time_ns']) / 1e9
    steps = (after['backend_time_ns'] - before['backend_time_ns']) // before['timestep_ns']
    old = {s['prim_path']: s for s in before['states']}
    errors = []
    # Native float32 positions near x=100 have one-ULP spacing 2**-17 m.
    # Bound cumulative rounding by one ULP per actual fixed step plus readback.
    tolerance = 2e-5 + steps * 2**-17
    assert tolerance < .02, ('fixture duration exceeded numerical scope', tolerance)
    for state in after['states']:
        start = old[state['prim_path']]
        for sample in (start, state):
            assert abs(sample['linear_velocity']['x'] - .25) < 1e-5
            assert all(abs(sample['linear_velocity'][a]) < 1e-5 for a in 'yz')
            assert all(abs(sample['angular_velocity'][a]) < 1e-5 for a in 'xyz')
        assert state['translate']['x'] > start['translate']['x']
        errors.append(abs(state['translate']['x'] - start['translate']['x'] - .25 * dt))
        errors.extend(abs(state['translate'][a] - start['translate'][a]) for a in 'yz')
        assert abs(sum(state['orient'][a] * start['orient'][a] for a in 'ijkr')) > 1 - 1e-5
    assert max(errors) <= tolerance, (errors, tolerance)
    return dict(elapsed_simulation_seconds=dt, fixed_steps=steps,
                maximum_position_error_m=max(errors), position_tolerance_m=tolerance)


def assert_uncertain_not_replayed(before, after):
    assert before['uncertain_operation'] == after['uncertain_operation'] == 'step'
    for key in ('pid', 'epoch', 'launcher_pid', 'restarts', 'request_counts'):
        assert before[key] == after[key], ('uncertain step replay or owner replacement', key)
    assert after['restarts'] == 0


def assert_fresh_frame(demand, proof):
    assert proof['submission']['started'] > demand['armed_at']
    assert proof['submission']['sequence'] > demand['submission_sequence']
    assert proof['copy_sequence'] > demand['copy_sequence']
    assert proof['copied_at'] >= proof['submission']['started']
    for key in (*RENDER_OWNER_FIELDS, 'stage_ordinal'):
        assert proof['native'][key] == demand['renderer'][key]
    assert proof['submission']['pid'] == proof['native']['runtime_pid']
    assert proof['submission']['epoch'] == proof['native']['runtime_epoch']


def survival_boundary(restarted, latest, *, armed_at, submission_sequence, copy_sequence):
    """Bind the content revision after uncertainty, not the first restart frame.

    Native ordinal advances on accepted pose/camera writes during continued Play.
    Owner identity remains fixed; after this boundary ordinal must remain exact.
    """
    assert latest is not None and latest['submission'] is not None
    assert latest['copied_at'] <= armed_at
    assert latest['copy_sequence'] == copy_sequence
    assert latest['submission']['sequence'] <= submission_sequence
    native = latest['native']
    for key in RENDER_OWNER_FIELDS:
        assert native[key] == restarted[key], ('renderer replaced before demand', key)
    assert native['stage_ordinal'] >= restarted['stage_ordinal']
    assert latest['submission']['pid'] == native['runtime_pid']
    assert latest['submission']['epoch'] == native['runtime_epoch']
    return dict(armed_at=armed_at, submission_sequence=submission_sequence,
                copy_sequence=copy_sequence, renderer=dict(native), baseline_copy=latest)


def assert_renderer_not_reloaded(before, after):
    for key in ('pid', 'epoch', 'launcher_pid', 'restarts'):
        assert before[key] == after[key], ('renderer owner replaced', key)
    assert before['request_counts'].get('load', 0) == after['request_counts'].get('load', 0)
