"""Negative controls for recovery evidence: unchanged clocks alone are insufficient."""
from copy import deepcopy
import pytest
from physics_continuity_checks import assert_preserved, assert_progressed, validate_snapshot


def snapshot():
    return dict(object_ids=[1, 2, 3, 4], pid=10, launcher_pid=11, epoch=1, session='a',
                loaded=True, stage_identity=20, physics_identity=21, runtime_identity={'hash': 'x'},
                cpu_only_policy=True, physics_generation=1, expected_paths=['/Body'],
                step_count=12, backend_time_ns=100000000, controller_time_ns=100000000,
                states=[dict(prim_path='/Body', simulation_time_ns=100000000,
                             translate=dict(found=True, x=100., y=0., z=1.),
                             orient=dict(found=True, i=0., j=0., k=0., r=1.),
                             linear_velocity=dict(found=True, x=.25, y=0., z=0.),
                             angular_velocity=dict(found=True, x=0., y=0., z=0.))])


@pytest.mark.parametrize('field,value', [('pid', 12), ('epoch', 2), ('session', 'b'),
    ('stage_identity', 22), ('physics_identity', 23), ('object_ids', [2, 3, 4, 5]),
    ('step_count', 13), ('physics_generation', 2)])
def test_changed_owner_or_solver_rejected(field, value):
    before = snapshot()
    after = deepcopy(before)
    after[field] = value
    with pytest.raises(AssertionError):
        assert_preserved(before, after)


def test_raw_state_change_rejected_even_if_clocks_unchanged():
    before = snapshot()
    after = deepcopy(before)
    after['states'][0]['linear_velocity']['x'] = .5
    with pytest.raises(AssertionError):
        assert_preserved(before, after)


@pytest.mark.parametrize('fault', ['missing', 'nan', 'norm', 'timestamp', 'device'])
def test_invalid_read_rejected(fault):
    value = snapshot()
    if fault == 'missing': value['states'] = []
    if fault == 'nan': value['states'][0]['translate']['x'] = float('nan')
    if fault == 'norm': value['states'][0]['orient']['r'] = 2
    if fault == 'timestamp': value['backend_time_ns'] += 1
    if fault == 'device': value['cpu_only_policy'] = False
    with pytest.raises(AssertionError): validate_snapshot(value)


def test_preserved_and_same_owner_progression():
    before = snapshot()
    assert_preserved(before, deepcopy(before))
    after = deepcopy(before)
    after.update(step_count=24, backend_time_ns=200000000, controller_time_ns=200000000)
    after['states'][0]['simulation_time_ns'] = 200000000
    after['states'][0]['translate']['x'] += .025
    assert assert_progressed(before, after)['maximum_position_error_m'] < 1e-10
    after['states'][0]['translate']['x'] += .1
    with pytest.raises(AssertionError): assert_progressed(before, after)
