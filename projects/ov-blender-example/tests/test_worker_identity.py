# SPDX-License-Identifier: Apache-2.0
"""Private bootstrap and OS ownership; no renderer/GPU work is submitted."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example.direct_runtime_session import RuntimeSession, RuntimeTransportError
from ovrtx_blender_example.frame_protocol import offer, CAPABILITIES
from ovrtx_blender_example.owned_process_tree import OwnedProcessTree


@pytest.fixture(scope='module')
def interpreters(tmp_path_factory):
    root = tmp_path_factory.mktemp('worker paths with spaces')
    venv = root / 'virtual environment'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(venv)], check=True)
    executable = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    return [sys.executable, str(executable)]


@pytest.mark.parametrize('index', [0, 1], ids=['standalone', 'venv'])
def test_actual_public_worker_first_control_and_cleanup(interpreters, index, tmp_path):
    # Actual Windows interpreter/venv + production bootstrap with explicit
    # fixture backend/policy modules; real locked-SDK probe is separate.
    package = tmp_path / 'addon/ovrtx_blender_example'
    package.mkdir(parents=True)
    (package / 'frame_protocol.py').write_bytes((ROOT / 'addon/ovrtx_blender_example/frame_protocol.py').read_bytes())
    tmp_path = tmp_path / 'runtime'
    tmp_path.mkdir()
    script = tmp_path / 'public_sdk_worker.py'
    script.write_bytes((ROOT / 'runtime/public_sdk_worker.py').read_bytes())
    (tmp_path / 'runtime_identity.py').write_text("def verified_identity(): return {'fixture': True}\n")
    (tmp_path / 'public_sdk_backend.py').write_text("import os\nclass RendererBackend:\n def __init__(self, output_root): pass\n def dispatch(self, op, params): return {'pid':os.getpid(), 'loaded':False, 'versions':{'ovstream':'fixture'}}\n")
    runtime = RuntimeSession(interpreters[index], script,
                             'renderer', tmp_path / 'output with spaces', timeout_s=30)
    owner = None
    try:
        result = runtime.request('health')
        assert result['pid'] == runtime.pid == result['_runtime_pid']
        assert result['loaded'] is False
        assert result['versions']['ovstream']
        assert runtime._owner.contains(runtime.pid)
        assert runtime._owner.contains(runtime.launcher_pid)
        if os.name == 'nt':
            assert (runtime.pid != runtime.launcher_pid) is (index == 1)
        assert runtime.diagnostics()['capabilities']['control'] == 1
        owner = runtime._owner
        evidence = {'python': interpreters[index], 'pid': runtime.pid,
                    'launcher_pid': runtime.launcher_pid, 'result': result}
    finally:
        runtime.close()
    assert runtime.returncode is not None
    assert owner.job is None and owner.worker_handle is None
    (tmp_path / 'identity-evidence.json').write_text(json.dumps(evidence, indent=2))


@pytest.mark.parametrize('mutation', ['session', 'epoch', 'pid', 'capability', 'boolean_capability', 'protocol', 'extra', 'duplicate'])
def test_hello_identity_cannot_be_forged_or_repeated(tmp_path, mutation):
    runtime = RuntimeSession(sys.executable, __file__, 'renderer', tmp_path)
    runtime._session_nonce = 'a' * 64; runtime._epoch = 5
    authenticated = []
    runtime._owner = SimpleNamespace(contains=lambda pid: type(pid) is int and pid == 123,
                                    authenticate=authenticated.append)
    hello = {'protocol': 2, 'session': 'a' * 64, 'epoch': 5, 'pid': 123,
             'application': offer(), 'capabilities': dict(CAPABILITIES)}
    if mutation == 'session': hello['session'] = 'b' * 64
    if mutation == 'epoch': hello['epoch'] = 4
    if mutation == 'pid': hello['pid'] = os.getpid()
    if mutation == 'capability': hello['capabilities']['control'] = 900
    if mutation == 'boolean_capability': hello['capabilities']['control'] = True
    if mutation == 'protocol': hello['protocol'] = True
    if mutation == 'extra': hello['unexpected'] = True
    if mutation == 'duplicate': runtime._accept_hello(hello)
    with pytest.raises(ValueError): runtime._accept_hello(hello)
    assert authenticated == ([123] if mutation == 'duplicate' else [])


@pytest.mark.parametrize('mode', ['timeout', 'early_exit', 'malformed', 'stale', 'wrong_peer', 'oversize'])
def test_real_failed_bootstrap_reaps_owned_tree(tmp_path, mode):
    script = tmp_path / 'bad worker.py'
    marker = tmp_path / 'descendant.pid'
    script.write_text('''import json, os, socket, sys, time, subprocess, pathlib
challenge = json.loads(sys.stdin.buffer.readline())
descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
staged = pathlib.Path(''' + repr(str(marker) + '.tmp') + ''')
staged.write_text(str(descendant.pid))
os.replace(staged, ''' + repr(str(marker)) + ''')
mode = ''' + repr(mode) + '''
if mode == 'early_exit': sys.exit(17)
if mode == 'timeout': time.sleep(20)
s = socket.create_connection(('127.0.0.1', challenge['port']))
hello = dict(protocol=2, session=challenge['session'], epoch=challenge['epoch'], pid=os.getpid(), capabilities=challenge['application']['required'], application=challenge['application'])
if mode == 'stale': hello['session'] = '0' * 64
if mode == 'wrong_peer': hello['pid'] = os.getppid()
payload = b'not-json\\n' if mode == 'malformed' else (b'x'*4097 if mode == 'oversize' else json.dumps(hello).encode()+b'\\n')
s.sendall(payload)
time.sleep(20)
''')
    # The deadline spans the fake worker's own startup and descendant spawn,
    # which can exceed a fraction of a second on hosted CI runners. The marker
    # is published atomically, so a reap mid-write cannot leave it partial.
    runtime = RuntimeSession(sys.executable, script, 'renderer', tmp_path / 'out', timeout_s=2)
    started = time.monotonic()
    with pytest.raises(RuntimeTransportError, match='bootstrap'):
        runtime.request('health')
    assert time.monotonic() - started < 5
    assert runtime._process.poll() is not None
    assert runtime._owner is None and runtime._socket is None
    assert marker.is_file()
    if os.name == 'nt':
        from ovrtx_blender_example.win32_shm import process_exists
        assert not process_exists(int(marker.read_text()))
    else:
        # A reparented zombie is terminated; init owns reaping it. Never treat
        # kill(pid, 0) alone as proof that the descendant is still executing.
        stat = Path('/proc') / marker.read_text() / 'stat'
        assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    runtime.close()


@pytest.mark.parametrize('failure', ['share_error', 'share_oversize'])
@pytest.mark.skipif(os.name != 'nt', reason='Windows socket duplication contract')
def test_windows_socket_duplication_failure_reaps_authenticated_worker(tmp_path, monkeypatch, failure):
    from test_direct_runtime_session import WORKER
    import ovrtx_blender_example.direct_runtime_session as module
    script = tmp_path / 'worker.py'; script.write_text(WORKER)
    runtime = RuntimeSession(sys.executable, script, 'renderer', tmp_path / 'out', timeout_s=2)
    original_pair = module.socket.socketpair
    class Child:
        def __init__(self, endpoint): self.endpoint = endpoint
        def share(self, pid):
            assert pid == runtime.identity.pid
            if failure == 'share_error': raise OSError('injected duplication failure')
            return b'x' * 4097
        def close(self): self.endpoint.close()
    def pair(*args):
        parent, child = original_pair(*args)
        return parent, Child(child)
    monkeypatch.setattr(module.socket, 'socketpair', pair)
    with pytest.raises(RuntimeTransportError, match='bootstrap'):
        runtime.request('health')
    assert runtime.returncode is not None and runtime._owner is None
    runtime.close()


def test_job_creation_failure_closes_both_private_socket_endpoints(tmp_path, monkeypatch):
    import ovrtx_blender_example.direct_runtime_session as module
    import ovrtx_blender_example.owned_process_tree as ownership
    sockets = socket.socketpair(socket.AF_INET if os.name == 'nt' else socket.AF_UNIX)
    monkeypatch.setattr(module.socket, 'socketpair', lambda *args: sockets)
    def fail(): raise OSError('injected job creation failure')
    monkeypatch.setattr(ownership, 'OwnedProcessTree', fail)
    runtime = RuntimeSession(sys.executable, __file__, 'renderer', tmp_path, timeout_s=1)
    with pytest.raises(RuntimeTransportError, match='job creation failure'):
        runtime.request('health')
    assert all(endpoint.fileno() == -1 for endpoint in sockets)
    assert runtime.pid is None
    runtime.close()


def test_job_reaps_descendant_after_launcher_exits_and_keeps_unrelated_process(tmp_path):
    if os.name != 'nt':
        pytest.skip('Windows process Job execution; POSIX contracts below')
    marker = tmp_path / 'descendant.json'
    code = 'import os,time,pathlib; pathlib.Path(' + repr(str(marker)) + ').write_text(str(os.getpid())); time.sleep(60)'
    launcher = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c",' + repr(code) + '])'
    owner = OwnedProcessTree()
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        process = owner.spawn([sys.executable, '-c', launcher], creationflags=subprocess.CREATE_NO_WINDOW)
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
        pid = int(marker.read_text())
        owner.authenticate(pid)
        assert owner.poll_worker() is None and process.returncode == 0
        assert not owner.contains(unrelated.pid)
        owner.stop()
        assert owner.poll_worker() is not None
        assert unrelated.poll() is None
    finally:
        if owner.job: owner.stop()
        unrelated.kill(); unrelated.wait(timeout=5)


def test_posix_group_ownership_contract_without_linux_execution(monkeypatch):
    import ovrtx_blender_example.owned_process_tree as module
    # Instantiate the platform-neutral state directly. No POSIX host/process.
    owner = object.__new__(OwnedProcessTree)
    owner.windows = False
    owner.job = None; owner.worker_handle = None; owner.worker_returncode = None
    waited = []; killed = []
    owner.process = SimpleNamespace(pid=123, wait=lambda timeout: waited.append(timeout))
    monkeypatch.setattr(module.os, 'getpgid', lambda pid: {123: 123, 124: 123, 900: 900}[pid], raising=False)
    def killpg(pid, sig):
        killed.append((pid, sig))
        if sig == 0: raise ProcessLookupError()
    monkeypatch.setattr(module.os, 'killpg', killpg, raising=False)
    monkeypatch.setattr(module.signal, 'SIGKILL', 9, raising=False)
    assert owner.contains(123) and owner.contains(124) and not owner.contains(900)
    owner.stop()
    assert killed == [(123, 9), (123, 0)] and waited == [3]


def test_posix_group_teardown_requires_confirmation(monkeypatch):
    import ovrtx_blender_example.owned_process_tree as module
    owner = object.__new__(OwnedProcessTree)
    owner.windows = False; owner.job = None
    owner.process = SimpleNamespace(pid=123, wait=lambda timeout: None)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: None, raising=False)
    monkeypatch.setattr(module.signal, 'SIGKILL', 9, raising=False)
    clock = iter([0, 4])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    with pytest.raises(RuntimeError, match='not confirmed'): owner.stop()


def test_session_preserves_worker_failure_after_successful_launcher_exit(tmp_path):
    runtime = RuntimeSession(sys.executable, __file__, 'renderer', tmp_path)
    runtime._worker_pid = 555
    runtime._process = SimpleNamespace(pid=444, poll=lambda: 0)
    runtime._owner = SimpleNamespace(stop=lambda: None, poll_worker=lambda: 37)
    runtime.close()
    assert runtime.returncode == 37
    assert runtime.diagnostics()['returncode'] == 37
    assert runtime.launcher_pid == 444 and runtime.pid == 555


def test_restart_changes_nonce_and_authenticated_pid(tmp_path):
    from test_direct_runtime_session import WORKER
    script = tmp_path / 'worker.py'; script.write_text(WORKER)
    with RuntimeSession(sys.executable, script, 'renderer', tmp_path / 'out') as runtime:
        runtime.request('load', {'scene_path': '/test.usda'})
        old = (runtime.pid, runtime._session_nonce, runtime.epoch)
        runtime.recover()
        assert runtime.pid != old[0] and runtime._session_nonce != old[1]
        assert runtime.epoch == old[2] + 1
        assert runtime.request('render')['_runtime_pid'] == runtime.pid


@pytest.mark.parametrize('field', ['pid', 'epoch'])
def test_cleanup_refuses_foreign_registry_even_in_owned_directory(tmp_path, monkeypatch, field):
    from ovrtx_blender_example import ovstream_shm_cleanup as cleanup, win32_shm
    monkeypatch.setattr(win32_shm, 'process_exists', lambda pid: False)
    record = {'schema': 1, 'pid': 12345, 'epoch': 'authenticated', 'streams': ['blender-' + 'a' * 32]}
    record[field] = 54321 if field == 'pid' else 'old-session'
    registry = tmp_path / 'ovstream-resources.json'; registry.write_text(json.dumps(record))
    unrelated = tmp_path / 'unrelated'; unrelated.write_bytes(b'preserve')
    with pytest.raises(ValueError):
        cleanup.cleanup_registry(registry, expected_pid=12345, expected_epoch='authenticated')
    assert json.loads(registry.read_text()) == record and unrelated.read_bytes() == b'preserve'


def test_owned_windows_cleanup_uses_kernel_lifetime_without_unlink(tmp_path, monkeypatch):
    from ovrtx_blender_example import ovstream_shm_cleanup as cleanup, win32_shm
    monkeypatch.setattr(win32_shm, 'process_exists', lambda pid: False)
    registry = tmp_path / 'ovstream-resources.json'
    registry.write_text(json.dumps({'schema': 1, 'pid': 12345, 'epoch': 'authenticated', 'streams': ['blender-' + 'a' * 32]}))
    monkeypatch.setattr(Path, 'unlink', lambda *args, **kwargs: pytest.fail('Windows cleanup must not unlink arbitrary files'))
    assert cleanup.cleanup_registry(registry, expected_pid=12345, expected_epoch='authenticated') == []
