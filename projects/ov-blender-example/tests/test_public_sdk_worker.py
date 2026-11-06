# SPDX-License-Identifier: Apache-2.0
"""Control replies remain complete under socket backpressure."""
import importlib.util
import json
from pathlib import Path
import socket
import threading
import io
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))

spec = importlib.util.spec_from_file_location(
    'public_sdk_worker_test', Path(__file__).parents[1] / 'runtime/public_sdk_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def receive_reply(response):
    sender, receiver = socket.socketpair()
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    sender.settimeout(5)
    receiver.settimeout(5)
    chunks = []
    errors = []

    def consume():
        try:
            while not chunks or not chunks[-1].endswith(b'\n'):
                part = receiver.recv(2048)
                if not part:
                    raise AssertionError('Truncated reply')
                chunks.append(part)
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        worker.send_response(sender, response)
        reader.join(timeout=6)
        assert not reader.is_alive()
        assert not errors
        return json.loads(b''.join(chunks))
    finally:
        sender.close()
        receiver.close()


def test_large_reply_is_complete_with_small_socket_buffer():
    response = {'id': 7, 'epoch': 2, 'ok': True, 'result': {'data': 'abc' * 700000}}
    assert receive_reply(response) == response


def test_oversized_reply_returns_bounded_error_with_request_identity():
    response = {'id': 9, 'epoch': 3, 'ok': True, 'result': 'x' * worker.MAX_MESSAGE}
    result = receive_reply(response)
    assert result['id'] == 9 and result['epoch'] == 3
    assert result['ok'] is False
    assert 'exceeds 16 MiB' in result['error']


def test_windows_shared_socket_is_acknowledged_before_native_imports(monkeypatch):
    shared = b'only-on-authenticated-bootstrap'
    from ovrtx_blender_example.frame_protocol import offer
    challenge = {'application': offer(), 'protocol': 2, 'session': 'a' * 64, 'epoch': 3, 'port': 12345}
    data = io.BytesIO(json.dumps(challenge).encode() + b'\n')
    calls = []

    class Connection:
        def settimeout(self, value): calls.append(('timeout', value))
        def sendall(self, value): calls.append(('send', json.loads(value)))
        def close(self): calls.append(('close',))

    class Bootstrap(Connection):
        remaining = io.BytesIO(struct.pack('!I', len(shared)) + shared)
        def recv(self, count): return self.remaining.read(min(3, count))

    connection = Connection()
    bootstrap = Bootstrap()
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=data, close=lambda: calls.append(('stdin_closed',))))
    monkeypatch.setattr(worker.socket, 'create_connection', lambda address, timeout: bootstrap)
    monkeypatch.setattr(worker.socket, 'fromshare', lambda value: calls.append(('shared', value)) or connection, raising=False)
    assert worker.receive_control_socket(shared_stdin=True, timeout_s=1) is connection
    hello = [value for kind, *values in calls for value in values if kind == 'send'][0]
    assert hello['pid'] == worker.os.getpid()
    assert hello['session'] == challenge['session'] and hello['epoch'] == 3
    assert calls.index(('shared', shared)) > calls.index(('send', hello))
    assert ('send', {'ready': True, 'session': challenge['session'], 'epoch': 3, 'pid': worker.os.getpid()}) in calls


@pytest.mark.parametrize('payload', [b'', b'wrong', b'OVBC1\0\0',
    b'OVBC1' + struct.pack('!I', 0), b'OVBC1' + struct.pack('!I', 4097),
    b'OVBC1' + struct.pack('!I', 10) + b'short'])
def test_invalid_socket_bootstrap_is_rejected_before_fromshare(monkeypatch, payload):
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(payload)))
    monkeypatch.setattr(worker.socket, 'fromshare', lambda value: pytest.fail('invalid bootstrap reached socket import'), raising=False)
    with pytest.raises(ValueError):
        worker.receive_control_socket(shared_stdin=True, timeout_s=1)


def test_stalled_bootstrap_exits_without_native_sdk_imports(tmp_path):
    process = subprocess.Popen([sys.executable, str(Path(worker.__file__)),
        '--kind', 'renderer', '--socket-share-stdin', '--bootstrap-timeout', '.2',
        '--output-root', str(tmp_path)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.wait(timeout=5) == 124
        output, errors = process.communicate()
        assert b'public_sdk_backend' not in output + errors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@pytest.mark.parametrize('kind', ['renderer', 'physics'])
@pytest.mark.parametrize('matching_lock', [True, False])
def test_real_worker_private_control_keeps_native_stdout_out_of_replies(tmp_path, monkeypatch, kind, matching_lock):
    from ovrtx_blender_example.direct_runtime_session import RuntimeSession
    package = tmp_path / 'addon/ovrtx_blender_example'
    package.mkdir(parents=True)
    (package / 'frame_protocol.py').write_bytes((Path(__file__).parents[1] / 'addon/ovrtx_blender_example/frame_protocol.py').read_bytes())
    tmp_path = tmp_path / 'runtime'
    tmp_path.mkdir()
    script = tmp_path / 'public_sdk_worker.py'
    script.write_bytes(Path(worker.__file__).read_bytes())
    (tmp_path / 'runtime_identity.py').write_text("def verified_identity(): return {'fixture': True}\n")
    injected = tmp_path / 'foreign-python'; injected.mkdir()
    marker = tmp_path / 'foreign-imported'
    (injected / 'sitecustomize.py').write_text('from pathlib import Path; Path(' + repr(str(marker)) + ').touch()\n')
    (injected / 'public_sdk_backend.py').write_text("raise RuntimeError('foreign SDK import')\n")
    monkeypatch.setenv('PYTHONPATH', str(injected))
    monkeypatch.setenv('PYTHONHOME', str(injected))
    monkeypatch.setenv('VIRTUAL_ENV', str(injected))
    (tmp_path / 'public_sdk_backend.py').write_text('''
print('simulated native log', flush=True)
class RendererBackend:
    def __init__(self, output_root): pass
    def dispatch(self, op, params): return {'operation': op, 'payload': params}
PhysicsBackend = RendererBackend
''')
    physics_identity = {'backend': 'public_sdk', 'adapter_sha256': 'a' * 64,
                        'native_sha256': {'native.dll': 'b' * 64},
                        'runtime_lock': {'fixture': matching_lock}}
    (tmp_path / 'incremental_physics_backend.py').write_text(
        'from public_sdk_backend import RendererBackend\n'
        'class IncrementalPhysicsBackend(RendererBackend):\n'
        '    def dispatch(self, op, params):\n'
        '        result = super().dispatch(op, params)\n'
        '        result["runtime_identity"] = ' + repr(physics_identity) + '\n'
        '        return result\n')
    runtime = RuntimeSession(sys.executable, script, kind, tmp_path / 'runtime', timeout_s=5)
    try:
        if kind == 'physics' and not matching_lock:
            with pytest.raises(RuntimeError, match='does not match the verified runtime lock'):
                runtime.request('health', {'message': 'roundtrip'})
        else:
            result = runtime.request('health', {'message': 'roundtrip'})
            assert result['payload'] == {'message': 'roundtrip'}
            assert result['operation'] == 'health'
            assert result['runtime_identity'] == (physics_identity if kind == 'physics' else {'fixture': True})
            if kind == 'physics':
                assert result['runtime_lock'] == {'fixture': True}
        assert not marker.exists()
    finally:
        runtime.close()
    assert runtime.returncode == 0
    assert 'simulated native log' in (runtime.output_root / 'epoch-1/worker.log').read_text()
