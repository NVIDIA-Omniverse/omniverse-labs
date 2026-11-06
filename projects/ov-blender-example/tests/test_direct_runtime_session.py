# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))
from ovrtx_blender_example import direct_runtime_session
from ovrtx_blender_example.direct_runtime_session import (
    FATAL_DEVICE_ERROR_PATTERNS, RuntimeFatalDeviceError, RuntimeSession, RuntimeSessionError,
    RuntimeTransportError, RuntimeWorkerError,
)


def test_unicode_explicit_venv_refused_before_runtime_creation(tmp_path, monkeypatch):
    # An installed runtime is always in an ASCII folder; an explicit external
    # interpreter in a non-ASCII folder cannot be remapped and is refused.
    from ovrtx_blender_example import native_paths
    monkeypatch.setattr(native_paths, '_windows', lambda windows: True)
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='non-ASCII characters'):
        RuntimeSession(tmp_path/'venv \u03a9/Scripts/python.exe',tmp_path/'worker.py','renderer',output)
    assert not output.exists()


WORKER = "import sys\nsys.path.insert(0, " + repr(str(ROOT / 'runtime')) + ")\n" + r'''
import argparse, json, os, pathlib, socket, time
from public_sdk_worker import receive_control_socket
p = argparse.ArgumentParser()
p.add_argument('--kind'); p.add_argument('--fd', type=int); p.add_argument('--output-root')
p.add_argument('--socket-share-stdin', action='store_true')
p.add_argument('--bootstrap-timeout', type=float, default=15)
a = p.parse_args()
connection = receive_control_socket(a.fd, shared_stdin=a.socket_share_stdin, timeout_s=a.bootstrap_timeout)
reader = connection.makefile('rb')
state = {'events': [], 'values': {}}
print('native diagnostic on stdout', flush=True)
for line in reader:
    req = json.loads(line); op = req['op']; params = req['params']
    if params.get('crash_once'):
        marker = pathlib.Path(params['crash_once'])
        if not marker.exists():
            marker.touch(); os._exit(9)
    if params.get('crash_always'):
        os._exit(9)
    if params.get('reset_once'):
        marker = pathlib.Path(params['reset_once'])
        if not marker.exists():
            # Close the control socket first and exit only later, like a
            # native crash whose process teardown outlives its connection.
            marker.touch(); connection.shutdown(socket.SHUT_RDWR); connection.close()
            time.sleep(params['exit_after']); os._exit(9)
    device_error = None
    if params.get('device_error_once'):
        marker = pathlib.Path(params['device_error_once'])
        if not marker.exists():
            marker.touch(); device_error = params['device_error']
    if params.get('sleep'):
        time.sleep(params['sleep'])
    if params.get('bad_id'):
        req['id'] += 1
    if params.get('stale_epoch'):
        req['epoch'] -= 1
    if params.get('reject'):
        response = {'id': req['id'], 'ok': False, 'error': 'rejected'}
    elif device_error:
        response = {'id': req['id'], 'ok': False, 'error': device_error}
    else:
        if op == 'load':
            state = {'events': [], 'values': {}, 'scene': params['scene_path']}
        elif op == 'update_scene' and not params.get('fail_scene'):
            state = {'events': [], 'values': {}, 'scene': params['scene_path']}
        elif op in ('update', 'retarget_camera'):
            for v in params['values']:
                state['values'][v['prim_path'] + ':' + v['attribute_name']] = v['value']
        elif op == 'material_graph_update':
            roots = [update['material_path'] for update in params['updates']]
            state.setdefault('graphs', {}).update({update['material_path']: update['layer_body'] for update in params['updates']})
            state['values'] = {key: value for key, value in state['values'].items()
                               if not any(key.startswith(root + '/') for root in roots)}
        elif op == 'instance_graph_update':
            state.setdefault('instances', {}).update({str(update['owner_id']): update['layer_body'] for update in params['updates']})
        elif op == 'resize':
            state['size'] = [params['width'], params['height']]
        elif op == 'unload':
            state = {'events': [], 'values': {}}
        state['events'].append(op)
        result = dict(state)
        if op == 'update_scene':
            result.update(scene_updated=not params.get('fail_scene'), rollback_succeeded=params.get('rollback', True))
            if params.get('error'):
                result['error'] = params['error']
        if op in ('update', 'retarget_camera'):
            result['updated_count'] = params.get('ack_count', len(params['values']))
            if 'derived' in params:
                result['accepted_derived_values'] = params['derived']
                if params.get('derived_nan'):
                    result['accepted_derived_values'][0]['value'][0][0] = float('nan')
                for row in params['derived'] if isinstance(params['derived'], list) else ():
                    if isinstance(row, dict) and 'prim_path' in row and 'attribute_name' in row:
                        state['values'][row['prim_path'] + ':' + row['attribute_name']] = row.get('value')
        if op == 'material_graph_update':
            result.update(material_graph_updated=True, material_count=len(params['updates']))
        if op == 'instance_graph_update':
            result.update(instance_graph_updated=True, owner_count=params.get('ack_count', len(params['updates'])))
        if op == 'retarget_camera':
            result['camera_retargeted'] = params.get('ack_camera', True)
        response = {'id': req['id'], 'epoch': req['epoch'], 'ok': True, 'result': result}
    response.update(pid=os.getpid(), session=req['session'])
    if params.get('wrong_pid'): response['pid'] = os.getppid()
    if params.get('wrong_session'): response['session'] = '0' * 64
    if params.get('duplicate_hello'): response = dict(protocol=2, session=req['session'], epoch=req['epoch'], pid=os.getpid(), capabilities=dict(control=1, owned_frames=1))
    response.setdefault('epoch', req['epoch'])
    connection.sendall(json.dumps(response).encode() + b'\n')
'''


@pytest.fixture
def session(tmp_path):
    worker = tmp_path / 'worker.py'
    worker.write_text(WORKER)
    instance = RuntimeSession(sys.executable, worker, 'renderer', tmp_path / 'renderer', timeout_s=2)
    # Respawn backoff is covered by test_restart_backoff_applied.
    instance.restart_backoff_s = (0.0, 0.0)
    yield instance
    instance.close()


def value(number):
    return {'prim_path': '/World/A', 'attribute_name': 'omni:xform', 'value_type': 'Matrix4d', 'value': number}


def test_slow_native_teardown_exits_naturally(tmp_path):
    worker = tmp_path / 'slow_teardown.py'
    worker.write_text(WORKER + '\ntime.sleep(6)\n')
    instance = RuntimeSession(sys.executable, worker, 'renderer', tmp_path / 'renderer')
    instance.request('load', {'scene_path': '/accepted.usda'})
    instance.close()
    assert instance.returncode == 0
    assert instance.diagnostics()['closed']


def test_renderer_child_does_not_inherit_blender_ocio(tmp_path, monkeypatch):
    monkeypatch.setenv('OCIO', str(tmp_path/'blender-display-config.ocio'))
    monkeypatch.setenv('OVRTX_TEST_RETAINED_ENV', 'retained')
    worker = tmp_path/'environment-worker.py'
    worker.write_text(WORKER.replace("state = {'events': [], 'values': {}}",
        "state = {'events': [], 'values': {}, 'ocio': os.environ.get('OCIO'), "
        "'retained_env': os.environ.get('OVRTX_TEST_RETAINED_ENV')}"))
    with RuntimeSession(sys.executable, worker, 'renderer', tmp_path/'renderer', timeout_s=2) as runtime:
        result = runtime.request('state', {})
        assert result['ocio'] is None
        assert result['retained_env'] == 'retained'
    assert os.environ['OCIO'] == str(tmp_path/'blender-display-config.ocio')


def test_failed_scene_restoration_preserves_native_diagnostic(session):
    session.request('load', {'scene_path': '/scene.usda'})
    detail = 'Scene update failed (missing attribute); predecessor restoration failed (invalid ordinal)'
    with pytest.raises(RuntimeSessionError, match='missing attribute') as caught:
        session.request('update_scene', {'scene_path': '/new.usda', 'fail_scene': True,
                                        'rollback': False, 'error': detail})
    assert detail in str(caught.value)
    with pytest.raises(RuntimeSessionError, match='unknown'):
        session.request('render', {})


def kill_owned(session):
    session._process.kill()
    deadline = time.monotonic() + 2
    while session.returncode is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert session.returncode is not None


@pytest.mark.parametrize('forgery', ['wrong_pid', 'wrong_session', 'duplicate_hello'])
def test_control_response_authentication_rejects_forgery(session, forgery):
    with pytest.raises(RuntimeTransportError):
        session.request('health', {forgery: True})
    assert session.diagnostics()['transport_failed']


def test_recovery_restores_acknowledged_scene_latest_values_and_size(session):
    session.request('load', {'scene_path': '/scene.usda'})
    first = value(1)
    session.request('update', {'values': [first]})
    first['value'] = 999  # Caller mutation must not change retained data.
    latest = value(2)
    session.request('update', {'values': [latest]})
    latest['value'] = 888
    session.request('resize', {'width': 32, 'height': 16})
    old_pid, old_epoch = session.pid, session.epoch
    kill_owned(session)
    result = session.request('render', {'samples': 1})
    assert result['scene'] == '/scene.usda'
    assert result['values'] == {'/World/A:omni:xform': 2}
    assert result['size'] == [32, 16]
    assert result['events'] == ['load', 'update', 'resize', 'render']
    assert result['_runtime_epoch'] == old_epoch + 1
    assert result['_runtime_pid'] != old_pid
    assert session.diagnostics()['retained_values'] == 1
    assert (session.output_root / 'epoch-2' / 'worker.log').read_text().strip() == 'native diagnostic on stdout'


def test_renderer_eof_during_render_is_retried_only_after_confirmed_exit(session, tmp_path):
    session.request('load', {'scene_path': '/scene.usda'})
    result = session.request('render', {'crash_once': str(tmp_path / 'crashed')})
    assert result['_runtime_epoch'] == 2
    assert result['events'] == ['load', 'render']


@pytest.mark.parametrize('operation', ['next_raw_chunk', 'release_frame', 'cancel_raw_transfer'])
@pytest.mark.parametrize('death_boundary', ['before_send', 'after_send'])
def test_interrupted_transfer_operation_never_replays_into_new_epoch(session, tmp_path, operation, death_boundary):
    session.request('load', {'scene_path': '/scene.usda'})
    old_epoch = session.epoch
    params = {'copy_token': 'old-held-token'}
    if death_boundary == 'before_send':
        kill_owned(session)
    else:
        params['crash_once'] = str(tmp_path / 'crashed')
    with pytest.raises(RuntimeTransportError):
        session.request(operation, params)
    assert session.epoch == old_epoch
    assert session.diagnostics()['restarts'] == 0
    session.recover()
    result = session.request('render', {})
    assert result['_runtime_epoch'] == old_epoch + 1
    assert result['events'] == ['load', 'render']


def test_ambiguous_update_never_replayed_and_requires_explicit_recovery(session, tmp_path):
    session.request('load', {'scene_path': '/scene.usda'})
    session.request('update', {'values': [value(1)]})
    with pytest.raises(RuntimeTransportError):
        session.request('update', {'values': [value(2)], 'crash_once': str(tmp_path / 'crashed')})
    with pytest.raises(RuntimeSessionError, match='Outcome'):
        session.request('render')
    assert session.epoch == 1
    session.recover()
    assert session.request('render')['values']['/World/A:omni:xform'] == 1


def test_rejected_load_preserves_last_accepted_scene(session):
    session.request('load', {'scene_path': '/accepted.usda'})
    with pytest.raises(RuntimeWorkerError, match='rejected'):
        session.request('load', {'scene_path': '/rejected.usda', 'reject': True})
    kill_owned(session)
    assert session.request('render')['scene'] == '/accepted.usda'


def test_unload_clears_replay_state(session):
    session.request('load', {'scene_path': '/scene.usda'})
    session.request('update', {'values': [value(1)]})
    session.request('unload')
    kill_owned(session)
    result = session.request('render')
    assert 'scene' not in result
    assert result['values'] == {}
    assert result['events'] == ['render']


def test_renderer_failure_does_not_touch_independent_physics(session, tmp_path):
    with RuntimeSession(sys.executable, session.worker_script, 'physics', tmp_path / 'physics') as physics:
        physics.request('load', {'scene_path': '/physics.usda'})
        physics_pid = physics.pid
        session.request('load', {'scene_path': '/scene.usda'})
        kill_owned(session)
        session.request('render')
        assert physics.request('step')['events'] == ['load', 'step']
        assert physics.pid == physics_pid
        kill_owned(physics)
        with pytest.raises(RuntimeTransportError):
            physics.request('step')
        with pytest.raises(RuntimeSessionError, match='Physics state'):
            physics.recover()
        assert physics.epoch == 1


def test_hung_live_renderer_does_not_restart_automatically(session):
    session.request('load', {'scene_path': '/scene.usda'})
    session.timeout_s = .05
    with pytest.raises(RuntimeTransportError):
        session.request('render', {'sleep': 3})
    assert session.returncode is None
    assert session.epoch == 1
    with pytest.raises(RuntimeTransportError, match='alive'):
        session.request('render')
    session.timeout_s = 2
    session.recover()
    assert session.request('render')['scene'] == '/scene.usda'


def test_retry_budget_is_bounded(session):
    session.request('load', {'scene_path': '/scene.usda'})
    for _ in range(2):
        kill_owned(session)
        session.request('render')
    kill_owned(session)
    with pytest.raises(RuntimeSessionError, match='budget exhausted'):
        session.request('render')
    assert session.epoch == 3


@pytest.mark.parametrize('bad', ['bad_id', 'stale_epoch'])
def test_invalid_response_identity_poisoned_transport(session, bad):
    session.request('load', {'scene_path': '/scene.usda'})
    with pytest.raises(RuntimeTransportError):
        session.request('render', {bad: True})
    with pytest.raises(RuntimeTransportError, match='alive'):
        session.request('render')


def test_transaction_holds_worker_until_shared_frame_copy_is_complete(session):
    session.request('load', {'scene_path': '/scene.usda'})
    entered = threading.Event()
    finished = threading.Event()
    def close():
        entered.set()
        session.close()
        finished.set()
    with session.transaction():
        result = session.request('render')
        thread = threading.Thread(target=close)
        thread.start()
        assert entered.wait(1)
        assert not finished.wait(.05)
        assert result['_runtime_pid'] == session.pid
        assert session.returncode is None
    thread.join(3)
    assert finished.is_set()
    assert session.returncode is not None
    session.close()
    with pytest.raises(RuntimeSessionError, match='closed'):
        session.request('render')


def test_large_payload_and_invalid_updates_fail_before_spawning(session):
    with pytest.raises(ValueError, match='16 MiB'):
        session.request('load', {'scene_path': 'a' * (16 * 1024 * 1024)})
    with pytest.raises(ValueError, match='attribute_name'):
        session.request('update', {'values': [{'prim_path': '/A'}]})
    assert session.pid is None


def test_physics_loss_during_step_is_never_retried(session, tmp_path):
    with RuntimeSession(sys.executable, session.worker_script, 'physics', tmp_path / 'physics') as physics:
        physics.request('load', {'scene_path': '/physics.usda'})
        with pytest.raises(RuntimeTransportError):
            physics.request('step', {'crash_once': str(tmp_path / 'physics-crashed')})
        assert physics.epoch == 1
        assert physics.diagnostics()['uncertain_operation'] == 'step'
        with pytest.raises(RuntimeSessionError, match='Outcome'):
            physics.request('step')


@pytest.mark.parametrize('ack_count', [0, None, True, 2])
def test_incomplete_update_ack_preserves_prior_replay_state(session, ack_count):
    session.request('load', {'scene_path': '/accepted.usda'})
    session.request('update', {'values': [value(1)]})
    with pytest.raises(RuntimeSessionError, match='acknowledge every update'):
        session.request('update', {'values': [value(2)], 'ack_count': ack_count})
    assert session.diagnostics()['uncertain_operation'] == 'update'
    assert session.diagnostics()['has_scene']
    with pytest.raises(RuntimeSessionError, match='Outcome'):
        session.request('render')
    session.recover()
    result = session.request('render')
    assert result['scene'] == '/accepted.usda'
    assert result['values'] == {'/World/A:omni:xform': 1}
    assert result['events'] == ['load', 'update', 'render']


def test_generation_replaces_restart_snapshot_after_complete_ack(session):
    session.request('load', {'scene_path': '/original.usda'})
    session.request('update', {'values': [value(1)]})
    session.request('update_scene', {'scene_path': '/next.usda'})
    session.request('update', {'values': [value(2)]})
    kill_owned(session)
    result = session.request('render')
    assert result['scene'] == '/next.usda'
    assert result['values'] == {'/World/A:omni:xform': 2}


@pytest.mark.parametrize('rollback', [True, False])
def test_failed_generation_preserves_accepted_restart_snapshot(session, rollback):
    session.request('load', {'scene_path': '/original.usda'})
    session.request('update', {'values': [value(1)]})
    with pytest.raises(RuntimeSessionError):
        session.request('update_scene', {'scene_path': '/bad.usda', 'fail_scene': True, 'rollback': rollback})
    if not rollback:
        with pytest.raises(RuntimeSessionError, match='Outcome'):
            session.request('render')
    session.recover()
    result = session.request('render')
    assert result['scene'] == '/original.usda'
    assert result['values'] == {'/World/A:omni:xform': 1}


@pytest.mark.parametrize('operation', ['load', 'step', 'write_poses', 'write_velocities', 'write_masses', 'unload'])
def test_physics_mutation_exception_blocks_continuation_and_recovery(session, tmp_path, operation):
    with RuntimeSession(sys.executable, session.worker_script, 'physics', tmp_path / 'physics') as physics:
        physics.request('load', {'scene_path': '/physics.usda'})
        with pytest.raises(RuntimeWorkerError, match='rejected'):
            physics.request(operation, {'reject': True})
        assert physics.diagnostics()['uncertain_operation'] == operation
        with pytest.raises(RuntimeSessionError, match='Outcome'):
            physics.request('step')
        with pytest.raises(RuntimeSessionError, match='Physics state'):
            physics.recover()
        assert physics.epoch == 1


def test_physics_read_rejection_does_not_invent_uncertain_mutation(session, tmp_path):
    with RuntimeSession(sys.executable, session.worker_script, 'physics', tmp_path / 'physics') as physics:
        physics.request('load', {'scene_path': '/physics.usda'})
        with pytest.raises(RuntimeWorkerError, match='rejected'):
            physics.request('read', {'reject': True})
        assert not physics.diagnostics()['uncertain_operation']
        assert physics.request('step')['events'] == ['load', 'step']


@pytest.mark.parametrize('operation,params', [
    ('update', {'values': [value(2)]}),
    ('resize', {'width': 64, 'height': 48}),
    ('update_scene', {'scene_path': '/new.usda'}),
])
def test_unsent_edit_recovers_confirmed_dead_renderer_and_retains_new_ack(session, operation, params):
    session.request('load', {'scene_path': '/old.usda'})
    session.request('update', {'values': [value(1)]})
    session.request('resize', {'width': 32, 'height': 16})
    old_pid = session.pid
    kill_owned(session)
    result = session.request(operation, params)
    assert result['_runtime_epoch'] == 2
    assert result['_runtime_pid'] != old_pid
    # A scene replacement clears fake-worker events, so request counts also
    # prove the preceding acknowledged scene/value/size were replayed first.
    counts = session.diagnostics()['request_counts']
    assert counts['load'] == 2
    assert counts['update'] == 2 + (operation == 'update')
    assert counts['resize'] == 2 + (operation == 'resize')
    if operation == 'update_scene':
        assert result['scene'] == '/new.usda'
        assert result['events'] == ['update_scene']
    else:
        assert result['scene'] == '/old.usda'
        assert result['events'] == ['load', 'update', 'resize', operation]
    # The new edit is now acknowledged state and survives a later recovery.
    kill_owned(session)
    recovered = session.request('render')
    if operation == 'update':
        assert recovered['values']['/World/A:omni:xform'] == 2
    elif operation == 'resize':
        assert recovered['size'] == [64, 48]
    else:
        assert recovered['scene'] == '/new.usda'


@pytest.mark.parametrize('operation,params', [
    ('load', {'scene_path': '/new.usda'}), ('unload', {}),
    ('health', {}), ('unknown_operation', {}),
])
def test_confirmed_dead_renderer_does_not_expand_lifecycle_recovery(session, operation, params):
    session.request('load', {'scene_path': '/old.usda'})
    kill_owned(session)
    with pytest.raises(RuntimeTransportError, match='worker exited'):
        session.request(operation, params)
    assert session.epoch == 1


@pytest.mark.parametrize('operation,params', [
    ('update', {'values': [value(2)]}),
    ('resize', {'width': 64, 'height': 48}),
    ('update_scene', {'scene_path': '/new.usda'}),
])
def test_unsent_edit_requires_acknowledged_scene(session, operation, params):
    session.request('health')
    kill_owned(session)
    with pytest.raises(RuntimeTransportError, match='worker exited'):
        session.request(operation, params)
    assert session.epoch == 1


@pytest.mark.parametrize('operation,params', [
    ('resize', {'width': 64, 'height': 48}),
    ('update_scene', {'scene_path': '/new.usda'}),
])
def test_edit_crash_after_send_is_not_automatically_retried(session, tmp_path, operation, params):
    session.request('load', {'scene_path': '/old.usda'})
    with pytest.raises(RuntimeTransportError):
        session.request(operation, {**params, 'crash_once': str(tmp_path / 'crashed')})
    assert session.epoch == 1
    assert session.diagnostics()['uncertain_operation'] == operation
    with pytest.raises(RuntimeSessionError, match='Outcome'):
        session.request(operation, params)


def test_unsent_renderer_edit_policy_never_recovers_physics(session, tmp_path):
    with RuntimeSession(sys.executable, session.worker_script, 'physics', tmp_path / 'physics') as physics:
        physics.request('load', {'scene_path': '/physics.usda'})
        kill_owned(physics)
        with pytest.raises(RuntimeTransportError, match='worker exited'):
            physics.request('update', {'values': [value(1)]})
        assert physics.epoch == 1


def test_failed_process_stop_does_not_falsely_complete_later_close(session, monkeypatch):
    session.request('load', {'scene_path': '/accepted.usda'})
    original_stop = session._stop_process
    attempts = []

    def stop():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError('process reap not confirmed')
        original_stop()

    monkeypatch.setattr(session, '_stop_process', stop)
    with pytest.raises(RuntimeError, match='reap not confirmed'):
        session.close()
    assert not session.diagnostics()['closed']
    assert session.diagnostics()['has_scene']
    session.close()
    assert len(attempts) == 2
    assert session.diagnostics()['closed']
    assert session.returncode is not None


@pytest.mark.parametrize('response', [{'ack_count': 0}, {'ack_camera': False}])
def test_camera_retarget_incomplete_ack_blocks_capture_and_preserves_ledger(session, response):
    session.request('load', {'scene_path': 'old.usda'})
    session.request('update', {'values': [value(1)]})
    with pytest.raises(RuntimeSessionError):
        session.request('retarget_camera', {'values': [value(2)], **response})
    assert session.diagnostics()['uncertain_operation'] == 'retarget_camera'
    assert session._values[('/World/A', 'omni:xform')]['value'] == 1


def pose_row(path, x):
    matrix = [[float(i == j) for j in range(4)] for i in range(4)]
    matrix[3][0] = x
    return dict(prim_path=path, attribute_name='omni:xform', value_type='Matrix4d', value=matrix)


def test_inherited_explicit_descendant_is_acknowledged_and_recovered(session):
    session.request('load', {'scene_path': '/scene.usda'})
    child = pose_row('/World/A/Child', 3)
    session.request('update', {'values': [child]})
    derived = pose_row('/World/A/Child', 12)
    receipt = session.request('update', {'values': [pose_row('/World/A', 10)], 'derived': [derived]})
    assert receipt['updated_count'] == 1
    assert session._values[('/World/A/Child', 'omni:xform')] == derived
    session.recover()
    restored = session.request('render')
    assert restored['values']['/World/A/Child:omni:xform'] == derived['value']


@pytest.mark.parametrize('forgery', ['unretained', 'not_descendant', 'bad_type', 'nonfinite', 'duplicate', 'extra_field', 'not_list'])
def test_forged_derived_receipt_preserves_acknowledged_ledger_and_blocks_work(session, forgery):
    import copy
    session.request('load', {'scene_path': '/scene.usda'})
    original = pose_row('/World/A/Child', 3)
    session.request('update', {'values': [original]})
    derived = pose_row('/World/A/Child', 12)
    rows = [derived]
    if forgery == 'unretained': derived['prim_path'] = '/World/A/Other'
    elif forgery == 'not_descendant': derived['prim_path'] = '/World/A/Child'; parent = '/Else'
    elif forgery == 'bad_type': derived['value_type'] = 'Float'
    elif forgery == 'nonfinite': derived['value'][0][0] = float('nan')
    elif forgery == 'duplicate': rows.append(copy.deepcopy(derived))
    elif forgery == 'extra_field': derived['unexpected'] = True
    elif forgery == 'not_list': rows = {'invalid': True}
    # NaN cannot be sent as a valid request; inject it in the worker response.
    if forgery == 'nonfinite':
        rows = [pose_row('/World/A/Child', 12)]
    params = {'values': [pose_row(parent if forgery == 'not_descendant' else '/World/A', 10)], 'derived': rows}
    if forgery == 'nonfinite': params['derived_nan'] = True
    with pytest.raises(RuntimeSessionError, match='derived update receipt'):
        session.request('update', params)
    assert session._values == {('/World/A/Child', 'omni:xform'): original}
    assert session.diagnostics()['uncertain_operation'] == 'update'
    with pytest.raises(RuntimeSessionError, match='unknown'):
        session.request('render')


def test_partial_ack_with_derived_values_never_adopts_either_state(session):
    session.request('load', {'scene_path': '/scene.usda'})
    original = pose_row('/World/A/Child', 3)
    session.request('update', {'values': [original]})
    with pytest.raises(RuntimeSessionError):
        session.request('update', {'values': [pose_row('/World/A', 10)], 'ack_count': 0,
                                   'derived': [pose_row('/World/A/Child', 12)]})
    assert session._values == {('/World/A/Child', 'omni:xform'): original}


def test_derived_receipt_cannot_override_explicitly_requested_descendant(session):
    session.request('load', {'scene_path': '/scene.usda'})
    original = pose_row('/World/A/Child', 3)
    session.request('update', {'values': [original]})
    with pytest.raises(RuntimeSessionError, match='derived update receipt'):
        session.request('update', {'values': [pose_row('/World/A', 10), pose_row('/World/A/Child', 20)],
                                   'derived': [pose_row('/World/A/Child', 12)]})
    assert session._values == {('/World/A/Child', 'omni:xform'): original}


def test_camera_retarget_rebases_source_used_for_renderer_recovery(session):
    session.request('load',{'scene_path':'camera-a.usda'})
    session.request('retarget_camera',{'values':[value(2)],'source_scene_path':'camera-b.usda'})
    session.recover()
    state=session.request('state',{})
    assert state['scene']=='camera-b.usda'


def test_recovery_compiles_latest_material_graph_before_shader_constants(session):
    session.request('load', {'scene_path': '/scene.usda'})
    def graph(body):
        return {'updates': [{'material_path': '/M', 'layer_body': '#usda 1.0\n' + body}]}
    session.request('material_graph_update', graph('def Material "M" {}\n'))
    session.request('update', {'values': [{'prim_path': '/M/S', 'attribute_name': 'inputs:roughness',
                                          'value_type': 'Float', 'value': .9}]})
    newest = graph('def Material "M" {\n string comment = "latest"\n}\n')
    session.request('material_graph_update', newest)
    session.request('update', {'values': [{'prim_path': '/M/S', 'attribute_name': 'inputs:roughness',
                                          'value_type': 'Float', 'value': .2}]})
    session._process.kill(); session._process.wait()
    result = session.request('render', {})
    assert result['events'] == ['load', 'material_graph_update', 'update', 'render']
    assert result['graphs']['/M'] == newest['updates'][0]['layer_body']
    assert result['values']['/M/S:inputs:roughness'] == .2
    assert session.diagnostics()['retained_material_graphs'] == 1


def test_recovery_replays_current_instance_graph_before_owner_values(session):
    session.request('load', {'scene_path': '/scene.usda'})
    graph = dict(owner_id=7, projection_path='/World/_DirectGeometry/Object_7',
        hidden_source_path='/World/Original', layer_body='#usda 1.0\ndef Xform "World" {}\n')
    session.request('instance_graph_update', {'updates': [graph]})
    graph['layer_body'] = '#usda 1.0\ndef Xform "World" { string label = "latest" }\n'
    session.request('instance_graph_update', {'updates': [graph]})
    row = {**value(3), 'prim_path': graph['projection_path']}
    session.request('update', {'values': [row]})
    session.recover()
    result = session.request('state', {})
    assert result['events'] == ['load', 'instance_graph_update', 'update', 'state']
    assert result['instances']['7'] == graph['layer_body']
    assert session.diagnostics()['retained_instance_graphs'] == 1


def test_partial_instance_ack_keeps_accepted_graph_and_requires_recovery(session):
    session.request('load', {'scene_path': '/scene.usda'})
    graph = dict(owner_id=7, projection_path='/World/_DirectGeometry/Object_7',
        hidden_source_path='/World/Original', layer_body='#usda 1.0\n')
    with pytest.raises(RuntimeSessionError, match='not acknowledged'):
        session.request('instance_graph_update', {'updates': [graph], 'ack_count': 0})
    assert not session._instance_graphs
    assert session._uncertain_operation == 'instance_graph_update'


def test_scoped_scene_acceptance_retains_only_unaffected_recovery_values(session):
    session._retain('load', {'scene_path': 'old'})
    rows = [{**value(1), 'prim_path': '/A'}, {**value(2), 'prim_path': '/B'},
            {**value(3), 'prim_path': '/B', 'attribute_name': 'visibility'}]
    session._retain('update', {'values': rows})
    session._retain('update_scene', {'scene_path': 'new', 'population_changed_prim_paths': ['/A']},
        result={'population_update': 'scoped_overlay'})
    assert list(session._values.values()) == [rows[1]]
    session._retain('update_scene', {'scene_path': 'newest', 'population_changed_prim_paths': ['/']},
        result={'population_update': 'scoped_overlay'})
    assert not session._values


def test_reset_with_slow_exit_recovers_once(session, tmp_path):
    session.request('load', {'scene_path': '/scene.usda'})
    old_pid, old_process = session.pid, session._process
    # EOF while the worker is still alive; it would exit only 2 s later.
    result = session.request('render', {'reset_once': str(tmp_path / 'reset'), 'exit_after': 2})
    assert result['events'] == ['load', 'render']
    assert result['_runtime_epoch'] == 2
    assert result['_runtime_pid'] != old_pid
    diagnostics = session.diagnostics()
    assert diagnostics['restarts'] == 1
    assert [row['cause'] for row in diagnostics['recoveries']] == ['worker_lost']
    recovery = diagnostics['last_recovery']
    assert recovery['recovered'] and recovery['op'] == 'render' and recovery['failed_pid'] == old_pid
    assert recovery['worker_log'] == str(session.output_root / 'epoch-1' / 'worker.log')
    # The lost worker was stopped through its Job, not waited for.
    stop = diagnostics['stops'][0]
    assert stop['pid'] == old_pid and not stop['waited'] and old_process.poll() is not None
    assert len(diagnostics['stops']) == 1
    assert session.request('render')['_runtime_epoch'] == 2


@pytest.mark.parametrize('pattern', FATAL_DEVICE_ERROR_PATTERNS)
def test_fatal_device_error_reply_poisons_and_recovers(session, tmp_path, pattern):
    session.request('load', {'scene_path': '/scene.usda'})
    old_pid, old_process = session.pid, session._process
    error = f'RuntimeError: renderer render failed: {pattern}'
    result = session.request('render', {'device_error_once': str(tmp_path / 'device'), 'device_error': error})
    assert result['events'] == ['load', 'render']
    assert result['_runtime_epoch'] == 2
    assert result['_runtime_pid'] != old_pid
    diagnostics = session.diagnostics()
    assert diagnostics['restarts'] == 1
    assert not diagnostics['poisoned']
    recovery = diagnostics['last_recovery']
    assert recovery['cause'] == 'fatal_device_error' and recovery['recovered']
    assert error in recovery['error']
    # The poisoned worker answered, but was stopped rather than reused.
    stop = diagnostics['stops'][0]
    assert stop['pid'] == old_pid and not stop['waited'] and old_process.poll() is not None


def test_fatal_device_error_outside_safe_operation_poisons_without_retry(session, tmp_path):
    session.request('load', {'scene_path': '/scene.usda'})
    session.request('update', {'values': [value(1)]})
    with pytest.raises(RuntimeFatalDeviceError, match='DEVICE_LOST'):
        session.request('update', {'values': [value(2)], 'device_error_once': str(tmp_path / 'device'),
                                   'device_error': 'VK_ERROR_DEVICE_LOST'})
    diagnostics = session.diagnostics()
    assert diagnostics['poisoned'] and diagnostics['returncode'] is not None
    assert diagnostics['restarts'] == 0 and session.epoch == 1
    # The rejected edit is never replayed; the next render restores acked state.
    result = session.request('render')
    assert result['_runtime_epoch'] == 2
    assert result['values']['/World/A:omni:xform'] == 1


def test_non_fatal_error_reply_unchanged(session):
    session.request('load', {'scene_path': '/scene.usda'})
    pid = session.pid
    with pytest.raises(RuntimeWorkerError, match='rejected') as caught:
        session.request('render', {'reject': True})
    assert not isinstance(caught.value, RuntimeFatalDeviceError)
    diagnostics = session.diagnostics()
    assert diagnostics['restarts'] == 0 and diagnostics['recoveries'] == []
    assert not diagnostics['poisoned'] and not diagnostics['transport_failed']
    assert diagnostics['returncode'] is None
    assert session.request('render')['_runtime_pid'] == pid


def test_first_load_crash_recovers_once(session, tmp_path):
    result = session.request('load', {'scene_path': '/scene.usda', 'crash_once': str(tmp_path / 'crashed')})
    assert result['scene'] == '/scene.usda'
    assert result['events'] == ['load']
    assert result['_runtime_epoch'] == 2
    diagnostics = session.diagnostics()
    assert diagnostics['restarts'] == 1
    assert diagnostics['last_recovery']['op'] == 'load'
    assert diagnostics['last_recovery']['cause'] in {'worker_lost', 'worker_exited'}
    assert session.request('render')['scene'] == '/scene.usda'
    # Once only: a load that also crashes its replacement is reported.
    with RuntimeSession(sys.executable, session.worker_script, 'renderer', tmp_path / 'again', timeout_s=2) as again:
        again.restart_backoff_s = (0.0, 0.0)
        with pytest.raises(RuntimeTransportError):
            again.request('load', {'scene_path': '/scene.usda', 'crash_always': True})
        assert again.epoch == 2
        assert again.diagnostics()['restarts'] == 1
        assert again.diagnostics()['uncertain_operation'] == 'load'


def test_second_load_crash_not_retried(session, tmp_path):
    session.request('load', {'scene_path': '/first.usda'})
    with pytest.raises(RuntimeTransportError):
        session.request('load', {'scene_path': '/second.usda', 'crash_once': str(tmp_path / 'crashed')})
    diagnostics = session.diagnostics()
    assert session.epoch == 1
    assert diagnostics['restarts'] == 0 and diagnostics['recoveries'] == []
    assert diagnostics['uncertain_operation'] == 'load'


def test_restart_backoff_applied(session, monkeypatch):
    assert direct_runtime_session.RESTART_BACKOFF_S == (1.0, 3.0)
    session.restart_backoff_s = direct_runtime_session.RESTART_BACKOFF_S
    delays = []
    real_sleep = time.sleep
    # Record the respawn backoff without waiting for it; short polls still sleep.
    monkeypatch.setattr(direct_runtime_session.time, 'sleep',
                        lambda seconds: delays.append(seconds) if seconds >= .5 else real_sleep(seconds))
    session.request('load', {'scene_path': '/scene.usda'})
    session.recover()
    assert delays == []  # An explicit recover() does not back off.
    session._restarts = 0
    for _ in range(2):
        kill_owned(session)
        session.request('render')
    assert delays == [1.0, 3.0]
    assert [row['backoff_s'] for row in session.diagnostics()['recoveries']] == [0.0, 1.0, 3.0]


def test_restart_budget_resets_after_successes(session):
    session.request('load', {'scene_path': '/scene.usda'})
    for _ in range(2):
        kill_owned(session)
        session.request('render')
    # The recovering render is the first of 20 consecutive successes.
    for _ in range(18):
        session.request('render')
    assert session.diagnostics()['restarts'] == 2
    session.request('render')
    assert session.diagnostics()['restarts'] == 0
    assert session.diagnostics()['budget_resets'] == 1
    for _ in range(2):
        kill_owned(session)
        session.request('render')
    assert session.diagnostics()['restarts'] == 2
    # Ten minutes without a failure also restore the budget.
    session._last_failure_at -= session.budget_reset_s
    kill_owned(session)
    assert session.request('render')['_runtime_epoch'] == 6
    assert session.diagnostics()['restarts'] == 1
    assert session.diagnostics()['budget_resets'] == 2


def test_exhausted_recovery_keeps_worker_failure_classification(session, tmp_path):
    session.max_restarts = 0
    session.request('load', {'scene_path': '/scene.usda'})
    with pytest.raises(RuntimeFatalDeviceError, match='budget exhausted'):
        session.request('render', {'device_error_once': str(tmp_path / 'device'), 'device_error': 'CUDA error 700'})
    session.max_restarts = 1
    session.request('render')  # Recovers the poisoned worker; the budget is now spent.
    with pytest.raises(RuntimeTransportError, match='budget exhausted'):
        session.request('render', {'reset_once': str(tmp_path / 'reset'), 'exit_after': 2})
