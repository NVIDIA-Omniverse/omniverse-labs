"""Pure acceptance checks for the actual renderer/physics continuity probe."""
import math


def validate_snapshot(row):
    assert row['pid'] > 0 and row['epoch'] > 0 and row['session']
    assert row['loaded'] and row['stage_identity'] and row['physics_identity']
    assert row['cpu_only_policy'] is True
    assert row['backend_time_ns'] == row['controller_time_ns'] > 0
    assert row['step_count'] > 0
    states = row['states']
    assert len(states) == len(row['expected_paths']) > 0
    assert {s['prim_path'] for s in states} == set(row['expected_paths'])
    for state in states:
        assert state['simulation_time_ns'] == row['backend_time_ns']
        for field, axes in (('translate', 'xyz'), ('orient', 'ijkr'),
                            ('linear_velocity', 'xyz'), ('angular_velocity', 'xyz')):
            assert state[field]['found'] is True
            assert all(math.isfinite(state[field][axis]) for axis in axes)
        assert abs(math.hypot(*(state['orient'][axis] for axis in 'ijkr')) - 1) < 1e-4


def assert_preserved(before, after):
    validate_snapshot(before)
    validate_snapshot(after)
    for key in ('object_ids', 'pid', 'launcher_pid', 'epoch', 'session',
                'stage_identity', 'physics_identity', 'runtime_identity',
                'cpu_only_policy', 'physics_generation', 'expected_paths',
                'step_count', 'controller_time_ns', 'backend_time_ns', 'states'):
        assert before[key] == after[key], ('Physics changed during renderer recovery', key)


def assert_progressed(before, after):
    validate_snapshot(before)
    validate_snapshot(after)
    for key in ('object_ids', 'pid', 'launcher_pid', 'epoch', 'session',
                'stage_identity', 'physics_identity', 'runtime_identity',
                'physics_generation', 'expected_paths'):
        assert before[key] == after[key], ('Physics owner changed on subsequent scrub', key)
    assert after['step_count'] > before['step_count']
    dt = (after['backend_time_ns'] - before['backend_time_ns']) / 1e9
    assert dt > 0
    old = {s['prim_path']: s for s in before['states']}
    errors = []
    for state in after['states']:
        start = old[state['prim_path']]
        assert abs(start['linear_velocity']['x'] - .25) < 1e-5
        delta = state['translate']['x'] - start['translate']['x']
        assert delta > 0
        errors.append(abs(delta - .25 * dt))
        for axis in 'yz':
            errors.append(abs(state['translate'][axis] - start['translate'][axis]))
    # Native float32 world positions at x=100; this bounds accumulated rounding.
    assert max(errors) < 2e-4, errors
    return {'elapsed_simulation_seconds': dt, 'maximum_position_error_m': max(errors),
            'position_tolerance_m': 2e-4}
