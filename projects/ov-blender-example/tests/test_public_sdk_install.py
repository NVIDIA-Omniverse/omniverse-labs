# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import public_sdk_install as install
from ovrtx_blender_example import public_sdk_sidecar as sidecar
from ovrtx_blender_example import public_sdk_lock as policy

LOCK_PATH = Path(__file__).resolve().parents[1] / 'runtime/public-sdk-lock.json'
PINS = policy.load_lock(LOCK_PATH)['versions']


@pytest.fixture
def manager(tmp_path, monkeypatch):
    requirements = tmp_path / 'requirements.txt'
    requirements.with_name(policy.LOCK_NAME).write_bytes(LOCK_PATH.read_bytes())
    requirements.write_text(''.join(f'{name}=={version}\n' for name, version in PINS.items()))
    monkeypatch.setattr(install.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20 * 1024**3))
    instance = install.Installer()
    instance.configure(tmp_path / 'storage', sys.executable, requirements)
    yield instance
    instance.close()
    if instance._thread:
        instance._thread.join(6)
        assert not instance._thread.is_alive()


def finish(manager):
    manager._thread.join(5)
    assert not manager._thread.is_alive()
    return manager.status()


def test_non_ascii_windows_store_maps_to_an_ascii_root_made_on_install(manager, tmp_path, monkeypatch):
    from ovrtx_blender_example import native_paths
    monkeypatch.setattr(native_paths, '_windows', lambda windows: True)
    monkeypatch.setenv('ProgramData', str(tmp_path/'programdata'))
    monkeypatch.setenv('SystemDrive', str(tmp_path/'system'))
    store = tmp_path/'relocated \u03a9'
    manager.configure(store, manager._config[1], manager._config[2])
    planned = tmp_path/'programdata/ovrtx-blender'/native_paths.mapped_name(store)
    # Configuring only plans the root; nothing is written before an install starts.
    assert manager._config[0] == planned and manager.status()['state'] == 'unconfigured'
    assert not store.exists() and not planned.exists()
    monkeypatch.setattr(manager, '_install', lambda config, source='', **kwargs: None)
    assert manager.start(source='local.zip') is True
    manager._thread.join(5)
    assert planned.is_dir() and json.loads((store/'native-root.json').read_text())['native_root'] == str(planned)
    assert manager.ovrtx_cache_root().parent.parent == planned


def test_install_refused_only_when_no_ascii_runtime_folder_can_be_made(manager, tmp_path, monkeypatch):
    from ovrtx_blender_example import native_paths
    monkeypatch.setattr(native_paths, '_windows', lambda windows: True)
    monkeypatch.setenv('ProgramData', str(tmp_path/'program \u03a9'))
    monkeypatch.setenv('SystemDrive', str(tmp_path/'system \u03a9'))
    store = tmp_path/'relocated \u03a9'
    manager.configure(store, manager._config[1], manager._config[2])
    assert manager.start(source='local.zip') is False and manager._thread is None
    assert manager.status()['state'] == 'failed' and 'no ASCII folder' in manager.status()['error']
    assert not store.exists()


def fake_commands(manager, monkeypatch, failure=None):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if '-m' in command and 'venv' in command:
            python = install._environment_python(Path(command[-1]))
            python.parent.mkdir(parents=True)
            python.write_text('test interpreter')
        elif command[1] == '-c':
            if 'run_name="__main__"' in command[2]:
                return  # locked materializer gets independent executable coverage
            if failure:
                raise RuntimeError(failure)
            # Only inspect the verification program; never import an SDK here.
            tree = ast.parse(command[2])
            paths = [node.args[0].value for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'Path']
            Path(paths[0]).write_text(json.dumps(PINS))
    monkeypatch.setattr(manager, '_run', run)
    return commands


def fake_local(manager, monkeypatch):
    root = manager._config[0]
    def materialize(source, storage, requirements, *, cancelled, progress):
        assert source == 'chosen.zip'
        assert not cancelled()
        directory = root / 'sidecars' / ('a' * 32)
        python = directory / 'python' / 'python.exe'
        python.parent.mkdir(parents=True)
        python.write_text('test interpreter')
        manifest = {'python':'python/python.exe', 'site_packages':'python/lib/site-packages', 'files':{
            'python/python.exe':hashlib.sha256(python.read_bytes()).hexdigest()}}
        (directory/'sidecar.json').write_text(json.dumps(manifest))
        progress('Verified package', 1, 1)
        return {'root': str(directory), 'python': str(python), 'manifest': {}}
    monkeypatch.setattr(sidecar, 'materialize_local', materialize)
    # This fixture isolates installer publication; locked manifest validation
    # has executable coverage in test_public_sdk_sidecar.
    monkeypatch.setattr(sidecar, 'validate_manifest', lambda manifest, requirements: manifest)


def test_offline_does_not_create_storage_or_launch(manager):
    assert not manager.start(online_access=False)
    assert manager.status()['state'] == 'offline'
    assert not manager._config[0].exists()


def test_auto_verifies_then_publishes_immutable_receipt(manager, monkeypatch):
    commands = fake_commands(manager, monkeypatch)
    assert manager.start(online_access=True)
    assert finish(manager)['state'] == 'ready'
    assert len(commands) == 3
    assert 'materialize_sdk.py' in commands[1][2]
    assert '"install","--cache"' in commands[1][2]
    assert 'inspect_installed' in commands[2][2]
    receipt = json.loads((manager._config[0] / 'ready.json').read_text())
    assert receipt['versions'] == PINS
    assert receipt['python'] == manager.ready_python()
    assert Path(manager.ready_python()).is_file()
    assert not manager.start()
    restored = install.Installer()
    restored.configure(manager._config[0], sys.executable, manager._config[2])
    assert restored.ready_python() == manager.ready_python()


def test_local_package_works_offline_without_venv_or_pip(manager, monkeypatch):
    fake_local(manager, monkeypatch)
    commands = fake_commands(manager, monkeypatch)
    assert manager.start(online_access=False, source='chosen.zip')
    assert finish(manager)['state'] == 'ready'
    assert len(commands) == 1 and commands[0][1] == '-c'
    receipt = json.loads((manager._config[0] / 'ready.json').read_text())
    assert receipt['source_kind'] == 'local_package'
    restored = install.Installer()
    restored.configure(manager._config[0], sys.executable, manager._config[2])
    assert restored.ready_python() == manager.ready_python()


@pytest.mark.parametrize('local', (False, True))
def test_restart_reaches_ready_without_hashing_until_inputs_change(manager, monkeypatch, local):
    if local:
        fake_local(manager, monkeypatch)
    fake_commands(manager, monkeypatch)
    assert manager.start(online_access=not local, source='chosen.zip' if local else '')
    assert finish(manager)['state'] == 'ready'
    receipt = json.loads((manager._config[0] / 'ready.json').read_text())
    assert receipt['input_stamp']['requirements'] is not None
    with monkeypatch.context() as startup:
        def no_hashing(*_args, **_kwargs):
            raise AssertionError('runtime inputs hashed at startup')
        startup.setattr(install, '_pins', no_hashing)
        startup.setattr(sidecar, 'installed_python', no_hashing)
        restored = install.Installer()
        restored.configure(manager._config[0], sys.executable, manager._config[2])
        assert restored.ready_python() == manager.ready_python()
    # A changed add-on input is a runtime change: verify once, then record it.
    requirements = manager._config[2]
    original = requirements.read_text()
    requirements.write_text(original + '# changed\n')
    changed = install.Installer()
    changed.configure(manager._config[0], sys.executable, requirements)
    assert changed.ready_python() is None  # pins digest changed: needs setup
    requirements.write_text(original)
    reverified = install.Installer()
    reverified.configure(manager._config[0], sys.executable, requirements)
    assert reverified.ready_python() == manager.ready_python()
    with monkeypatch.context() as startup:
        startup.setattr(install, '_pins', no_hashing)
        again = install.Installer()
        again.configure(manager._config[0], sys.executable, requirements)
        assert again.ready_python() == manager.ready_python()


def test_failed_local_update_preserves_previous_ready_runtime(manager, monkeypatch):
    fake_commands(manager, monkeypatch)
    manager.start(); assert finish(manager)['state'] == 'ready'
    previous = manager.ready_python()
    receipt = (manager._config[0] / 'ready.json').read_bytes()
    fake_local(manager, monkeypatch)
    fake_commands(manager, monkeypatch, failure='SDK import failed')
    assert manager.start(online_access=False, source='chosen.zip')
    assert finish(manager)['state'] == 'failed'
    assert manager.ready_python() == previous
    assert (manager._config[0] / 'ready.json').read_bytes() == receipt
    assert Path(previous).is_file()
    assert not (manager._config[0] / 'sidecars' / ('a' * 32)).exists()


def test_failed_verification_never_publishes_and_retry_uses_new_candidate(manager, monkeypatch):
    commands = fake_commands(manager, monkeypatch, failure='bad version')
    manager.start(); assert finish(manager)['state'] == 'failed'
    old = Path(commands[0][-1])
    assert not old.exists()
    assert not (manager._config[0] / 'ready.json').exists()
    commands = fake_commands(manager, monkeypatch)
    manager.start(); assert finish(manager)['state'] == 'ready'
    assert Path(commands[0][-1]) != old


def test_low_disk_fails_before_launch(manager, monkeypatch):
    monkeypatch.setattr(install.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024))
    commands = fake_commands(manager, monkeypatch)
    manager.start()
    assert finish(manager)['state'] == 'failed'
    assert '8 GiB' in manager.status()['error']
    assert commands == []


def test_single_flight_cancel_before_commit_and_retry(manager, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    commands = fake_commands(manager, monkeypatch)
    run = manager._run
    def blocked(command, **kwargs):
        run(command, **kwargs)
        if command[1] == '-c':
            entered.set(); assert release.wait(3)
    monkeypatch.setattr(manager, '_run', blocked)
    manager.start(); assert entered.wait(3)
    assert not manager.start()
    manager.configure(manager._config[0], sys.executable, manager._config[2])
    start = time.monotonic(); manager.cancel()
    assert time.monotonic() - start < .1
    release.set()
    assert finish(manager)['state'] == 'cancelled'
    assert not (manager._config[0] / 'ready.json').exists()
    assert not Path(commands[0][-1]).exists()
    fake_commands(manager, monkeypatch)
    manager.start(); assert finish(manager)['state'] == 'ready'


@pytest.mark.parametrize('change', ['pins', 'path', 'missing', 'schema'])
def test_receipt_revalidation_rejects_stale_or_outside_candidates(manager, monkeypatch, change):
    fake_commands(manager, monkeypatch)
    manager.start(); finish(manager)
    receipt_path = manager._config[0] / 'ready.json'
    receipt = json.loads(receipt_path.read_text())
    if change == 'pins': receipt['versions']['ovrtx'] = 'wrong'
    if change == 'path': receipt['environment_relative'] = '../outside'
    if change == 'schema': receipt['schema'] = 99
    if change == 'missing': Path(receipt['python']).unlink()
    receipt_path.write_text(json.dumps(receipt))
    manager.configure(manager._config[0], sys.executable, manager._config[2])
    assert manager.ready_python() is None


def test_real_owned_process_cancel_reaped_and_sdk_free(manager, tmp_path, monkeypatch):
    launched = []
    original = install.subprocess.Popen
    def popen(*args, **kwargs):
        process = original(*args, **kwargs); launched.append(process); return process
    monkeypatch.setattr(install.subprocess, 'Popen', popen)
    result = []
    with (tmp_path / 'child.log').open('wb') as log:
        def target():
            try:
                manager._run([sys.executable, '-c', 'import time; time.sleep(60)'], log=log, timeout_s=10)
            except install._Cancelled:
                result.append('cancelled')
        thread = threading.Thread(target=target); thread.start()
        deadline = time.monotonic() + 3
        while not launched and time.monotonic() < deadline:
            time.sleep(.01)
        assert launched
        manager.close(); thread.join(5)
    assert not thread.is_alive()
    assert launched[0].poll() is not None
    assert result == ['cancelled']
    assert manager._cleanup_safe


def test_windows_gate_runs_code_only_after_release(tmp_path):
    target = tmp_path / 'entered'
    command = [sys.executable, '-c', install._GATE, '-c',
               f'from pathlib import Path; Path({str(target)!r}).write_text("ok")']
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        time.sleep(.1)
        assert not target.exists()
        process.stdin.write(b'G'); process.stdin.close()
        assert process.wait(timeout=5) == 0
    assert target.read_text() == 'ok'


@pytest.mark.parametrize('failure', ['query', 'close', 'none'])
def test_windows_setup_cleanup_confirms_job_and_never_loses_guard(manager,failure):
    calls=[]
    class Process:
        stdin=None
        returncode=None
        def poll(self):return self.returncode
        def kill(self):calls.append('kill');self.returncode=125
        def wait(self,timeout):calls.append('wait');return self.returncode
    class Job:
        def pin_members(self):calls.append('pin_members')
        def wait_members(self,deadline):calls.append('wait_members')
        def active(self):
            if failure=='query':raise OSError('Job query denied')
            return 0
        def terminate(self):calls.append('terminate')
        def close(self):
            calls.append('close')
            if failure=='close':raise OSError('Job close denied')
    if failure=='none':
        manager._finish_windows_setup(Job(),Process(),None)
        assert manager._cleanup_safe
    else:
        with pytest.raises(RuntimeError,match='cleanup errors'):
            manager._finish_windows_setup(Job(),Process(),None)
        assert not manager._cleanup_safe
    assert 'close' in calls and 'wait' in calls and 'kill' in calls


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX signal escalation contract')
def test_unconfirmed_teardown_blocks_retry_and_preserves_candidate(manager, monkeypatch, tmp_path):
    class Unreapable:
        pid = 987654
        stdin = None
        def poll(self): return None
        def wait(self, timeout): raise subprocess.TimeoutExpired('owned test child', timeout)
    monkeypatch.setattr(install, '_WINDOWS', False)
    monkeypatch.setattr(install.subprocess, 'Popen', lambda *a, **k: Unreapable())
    monkeypatch.setattr(install.os, 'killpg', lambda *args: None)
    with (tmp_path / 'failure.log').open('wb') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            manager._run([sys.executable, '-c', 'pass'], log=log, timeout_s=-1)
    assert not manager._cleanup_safe
    assert not manager.start()
    assert 'restart Blender' in manager.status()['error']


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX signal escalation contract')
def test_forced_kill_confirmed_allows_retry(manager, monkeypatch, tmp_path):
    class Stubborn:
        pid = 987654
        stdin = None
        calls = 0
        def poll(self): return None
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired('owned test child', timeout)
            return -9
    monkeypatch.setattr(install, '_WINDOWS', False)
    monkeypatch.setattr(install.subprocess, 'Popen', lambda *a, **k: Stubborn())
    monkeypatch.setattr(install.os, 'killpg', lambda *args: None)
    with (tmp_path / 'failure.log').open('wb') as log:
        with pytest.raises(TimeoutError):
            manager._run([sys.executable, '-c', 'pass'], log=log, timeout_s=-1)
    assert manager._cleanup_safe


def test_changed_requirements_during_setup_never_publish(manager, monkeypatch):
    fake_commands(manager, monkeypatch)
    original = manager._run
    def changed(command, **kwargs):
        original(command, **kwargs)
        if command[1] == '-c':
            manager._config[2].write_text(manager._config[2].read_text() + '# changed\n')
    monkeypatch.setattr(manager, '_run', changed)
    manager.start()
    assert finish(manager)['state'] == 'failed'
    assert not (manager._config[0] / 'ready.json').exists()


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process-group descendant cleanup')
def test_cancel_kills_descendant_even_when_parent_exits_first(manager, tmp_path):
    child_pid = tmp_path / 'child.pid'
    child_code = ('import os,signal,time; from pathlib import Path; '
                  'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  f'Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)')
    parent_code = ('import subprocess,sys,time; '
                   f'subprocess.Popen([sys.executable,"-c",{child_code!r}]); time.sleep(60)')
    errors = []
    with (tmp_path / 'tree.log').open('wb') as log:
        def target():
            try:
                manager._run([sys.executable, '-c', parent_code], log=log, timeout_s=10)
            except install._Cancelled:
                pass
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=target); thread.start()
        deadline = time.monotonic() + 3
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert child_pid.exists()
        manager.cancel(); thread.join(5)
        assert not thread.is_alive()
    assert not errors
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            # Linux reports ESRCH when the process exits between open and read.
            break
        if state == 'Z':
            break
        time.sleep(.01)
    else:
        pytest.fail('Owned nested setup process survived cancellation')
