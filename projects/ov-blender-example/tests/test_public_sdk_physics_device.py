"""Load must honor effective native device policy, including sticky CPU mode."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import public_sdk_backend as backend


@pytest.mark.parametrize('bad', [None, 'sparse', 'array', 'unknown', 'missing', 'duplicate'])
def test_native_state_columns_use_one_snapshot_and_validate_emitted_groups(monkeypatch, bad):
    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, 'ovphysx.types', SimpleNamespace(
        SimObjectType=SimpleNamespace(RIGID_BODY=1), ObjectScope=SimpleNamespace(ALL=2)))
    rows = [('orientation', [0, 0, 0, 1]), ('angularVelocity', [0, 0, 0]),
            ('position', [1, 2, 3]), ('linearVelocity', [4, 5, 6])]
    groups = [SimpleNamespace(attribute=name, is_delete=False, is_array=False,
              index_map=None, prim_index_map=None, prim_list=['/Body'],
              tensors=[SimpleNamespace(numpy=lambda values=values: np.asarray([values]))])
              for name, values in rows]
    if bad == 'sparse': groups[0].index_map = [0]
    if bad == 'array': groups[0].is_array = True
    if bad == 'unknown': groups[0].attribute = 'positions'
    if bad == 'missing': groups.pop()
    if bad == 'duplicate': groups.append(groups[0])
    calls = []
    def read_sync(kind, attributes, *, scope):
        calls.append((kind, attributes, scope))
        return nullcontext(SimpleNamespace(groups=groups))
    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker.stage = object()
    worker.physx = SimpleNamespace(read_sync=read_sync)
    worker.ovstage = SimpleNamespace(PathDictionary=lambda stage: nullcontext(
        SimpleNamespace(token_to_string=lambda token: token, get_path_strings=lambda paths: paths)))
    if bad:
        with pytest.raises((NotImplementedError, ValueError)):
            worker._state_columns()
    else:
        result = worker._state_columns()
        assert result['translate']['/Body'] == [1, 2, 3]
        assert result['orient']['/Body'] == [0, 0, 0, 1]
        assert result['linear_velocity']['/Body'] == [4, 5, 6]
    assert calls == [(1, ['position', 'orientation', 'linearVelocity', 'angularVelocity'], 2)]
    assert worker.native_state_read_sessions == 1


def worker_with_policy(monkeypatch, effective, *, query_error=None):
    calls = []

    class PhysX:
        @staticmethod
        def set_cpu_mode(value):
            calls.append(('set', value))

        @staticmethod
        def get_cpu_mode():
            calls.append(('get',))
            if query_error:
                raise query_error
            return effective

        def __init__(self):
            calls.append(('create-physics',))

        def attach_ovstage(self, stage, **kwargs):
            calls.append(('attach',))
            return 'attach-task'

        def wait_task(self, task):
            assert task == 'attach-task'

        def wait_all(self):
            pass

        def warmup(self):
            calls.append(('warmup',))

    class Stage:
        def __init__(self, name):
            calls.append(('create-stage',))

        def advance_write_floor(self, ordinal):
            return SimpleNamespace(wait=lambda: None)

    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker.physx = worker.stage = None
    worker.ovphysx = SimpleNamespace(PhysX=PhysX)
    worker.ovstage = SimpleNamespace(Stage=Stage, PopulationDomain=SimpleNamespace(ALL=1),
        population=SimpleNamespace(open_usd=lambda *args, **kwargs: calls.append(('populate',))))
    worker.runtime_identity = lambda: {'adapter_sha256': 'test-identity'}
    monkeypatch.setattr(backend, '_versions', lambda: {})
    return worker, calls


@pytest.mark.parametrize('requested,effective', [('gpu', True), ('cpu', False)])
@pytest.mark.parametrize('override', [None, '1'])
def test_rejects_effective_mismatch_before_native_objects(monkeypatch, requested, effective, override):
    # No environment flag is necessary: process-wide sticky mode is also read.
    if override is None:
        monkeypatch.delenv('OVPHYSX_DISABLE_GPU', raising=False)
    else:
        monkeypatch.setenv('OVPHYSX_DISABLE_GPU', override)
    worker, calls = worker_with_policy(monkeypatch, effective)
    with pytest.raises(RuntimeError, match=f'Requested {requested.upper()} physics conflicts with effective'):
        worker.dispatch('load', {'scene_path': 'unused.usda', 'device': requested})
    assert calls == [('set', requested == 'cpu'), ('get',)]
    assert worker.physx is None and worker.stage is None
    assert worker.dispatch('health', {})['loaded'] is False


@pytest.mark.parametrize('requested,effective', [('gpu', False), ('cpu', True)])
def test_matched_policy_loads_and_reports_native_mode(monkeypatch, requested, effective):
    worker, calls = worker_with_policy(monkeypatch, effective)
    result = worker.dispatch('load', {'scene_path': 'scene.usda', 'device': requested})
    assert calls[:3] == [('set', requested == 'cpu'), ('get',), ('create-physics',)]
    assert result['loaded'] is True and result['cpu_only_policy'] is effective
    assert ('attach',) in calls
    assert (('warmup',) in calls) is (requested == 'gpu')


def test_policy_query_failure_creates_no_simulation(monkeypatch):
    worker, calls = worker_with_policy(monkeypatch, False, query_error=RuntimeError('native query failed'))
    with pytest.raises(RuntimeError, match='native query failed'):
        worker.dispatch('load', {'scene_path': 'unused.usda', 'device': 'gpu'})
    assert calls == [('set', False), ('get',)]
    assert worker.physx is None and worker.stage is None


@pytest.mark.parametrize('steps', [1, 2, 8])
def test_publication_extracts_state_once_after_final_substep(steps):
    from contextlib import nullcontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
    from ovrtx_blender_example.direct_sdk_client import PublicSdkPhysicsClient
    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker.time_ns = 0; worker.body_prims = {'/Body'}
    calls = []
    worker.physx = SimpleNamespace(step_sync=lambda dt: calls.append(('step', dt)))
    vectors = {'position': [1, 2, 3], 'orientation': [0, 0, 0, 1],
               'linearVelocity': [0, 0, 0], 'angularVelocity': [0, 0, 0]}
    def read_columns():
        calls.append(('snapshot',))
        worker.native_state_read_sessions = getattr(worker, 'native_state_read_sessions', 0) + 1
        return {key: {'/Body': vectors[name]} for key, name in
                (('translate', 'position'), ('orient', 'orientation'),
                 ('linear_velocity', 'linearVelocity'), ('angular_velocity', 'angularVelocity'))}
    worker._state_columns = read_columns
    client = PublicSdkPhysicsClient.__new__(PublicSdkPhysicsClient)
    client.config = SimpleNamespace(body_prims=('/Body',))
    client.runtime = SimpleNamespace(transaction=nullcontext, request=worker.dispatch)
    states, result = client.advance_and_read_body_states(0, steps, 10000000)
    assert len([row for row in calls if row[0] == 'step']) == steps
    assert len([row for row in calls if row[0] == 'snapshot']) == 1
    assert result['publication_native_state_reads'] == 1
    assert worker.state_read_calls == 1
    assert result['publication_state_reads'] == 1
    assert result['publication_substeps'] == steps
    assert result['step_count'] == steps and result['simulation_time_ns'] == steps * 10000000
    assert states[0]['prim_path'] == '/Body'
    if steps:
        assert result['state_read_calls'] == 1 and result['state_read_ms'] >= 0


def test_step_invalid_state_option_rejected_before_native_mutation():
    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker.physx = SimpleNamespace(step_sync=lambda dt: pytest.fail('must validate before mutation'))
    with pytest.raises(ValueError, match='boolean'):
        worker.dispatch('step', {'dt_ns': 1000, 'read_states': 'false'})


def test_step_legacy_caller_receives_states_and_native_failure_is_not_retried():
    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker.time_ns = 0
    worker.physx = SimpleNamespace(step_sync=lambda dt: None)
    worker.read = lambda: ['legacy-state']
    assert worker.dispatch('step', {'dt_ns': 1000})['states'] == ['legacy-state']
    calls = []
    def fail(dt):
        calls.append(dt)
        raise RuntimeError('uncertain step')
    worker.physx.step_sync = fail
    with pytest.raises(RuntimeError, match='uncertain'):
        worker.dispatch('step', {'dt_ns': 1000, 'read_states': False})
    assert len(calls) == 1 and worker.time_ns == 1000


def test_publication_aggregates_native_substep_timing():
    from contextlib import nullcontext
    from ovrtx_blender_example.direct_sdk_client import PublicSdkPhysicsClient
    client = PublicSdkPhysicsClient.__new__(PublicSdkPhysicsClient)
    client.config = SimpleNamespace(body_prims=('/Body',))
    calls = []
    def request(op, params):
        calls.append((op, params))
        return {'step_ms': 3.0} if op == 'step' else {'states': [], 'state_read_ms': 2.0}
    client.runtime = SimpleNamespace(transaction=nullcontext, request=request)
    _, report = client.advance_and_read_body_states(4, 3, 10000000)
    assert report['step_ms'] == 9.0 and report['read_ms'] == 2.0
    assert report['total_ms'] >= 0
    assert [op for op, _ in calls] == ['step', 'step', 'step', 'read']
    assert all(params['read_states'] is False for op, params in calls if op == 'step')
