# SPDX-License-Identifier: Apache-2.0
"""Executable lock, rollback and portable-layout contracts (no Linux execution)."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import public_sdk_lock as policy

LOCK = Path(__file__).resolve().parents[1] / 'runtime/public-sdk-lock.json'


@pytest.fixture
def lock():
    return policy.load_lock(LOCK)


def test_requirements_and_complete_identity_have_no_sil_dependency(lock):
    assert policy.versions(LOCK.with_name('requirements-public-sdk.txt')) == lock['versions']
    assert lock['versions']['warp-lang'] == '1.18.0'
    for target in ('windows-x64', 'linux-x64'):
        identity = policy.lock_identity(lock, target)
        assert set(identity['artifacts']) == policy.REQUIRED
        assert identity['python']['abi'] == 'cp313'
        assert 'sil' not in json.dumps(identity).lower()
        for artifact in lock['platforms'][target]['artifacts'].values():
            # Both platforms must be installable: materialization checks the
            # reviewed member inventory as well as the complete wheel hash.
            inventory = artifact.get('files_sha256', '')
            assert len(inventory) == 64 and all(c in '0123456789abcdef' for c in inventory)


@pytest.mark.parametrize('target', ['windows-x64', 'linux-x64'])
@pytest.mark.parametrize('bad', ['version', 'sha256', 'platform', 'interpreter', 'missing'])
def test_invalid_tuple_refused(tmp_path, lock, target, bad):
    row = lock['platforms'][target]['artifacts']['numpy']
    if bad == 'version': row['version'] = '999'
    if bad == 'sha256': row['sha256'] = 'xyz'
    if bad == 'platform': row['filename'] = row['filename'].replace('win_amd64', 'arm64').replace('x86_64', 'arm64')
    if bad == 'interpreter': row['filename'] = row['filename'].replace('cp313', 'cp312')
    if bad == 'missing': del lock['platforms'][target]['artifacts']['warp-lang']
    path = tmp_path / 'lock.json'; path.write_text(json.dumps(lock))
    with pytest.raises(ValueError): policy.load_lock(path)


def test_interpreter_mismatch_is_rejected_before_acquisition(lock, monkeypatch):
    monkeypatch.setattr(policy, 'interpreter_identity', lambda: dict(lock['python'], minor=12))
    with pytest.raises(ValueError, match='CPython 3.13'): policy.validate_interpreter(lock)


@pytest.mark.parametrize('target', ['windows-x64', 'linux-x64'])
def test_two_fresh_resolutions_identical_and_partial_download_never_activates(tmp_path, lock, monkeypatch, target):
    payload = b'fixture wheel bytes'
    for row in lock['platforms'][target]['artifacts'].values():
        row['sha256'] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(policy, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    first = policy.acquire(lock, tmp_path / 'first', target)
    second = policy.acquire(lock, tmp_path / 'second', target)
    assert {n: policy.digest(p) for n, p in first.items()} == {n: policy.digest(p) for n, p in second.items()}
    previous = tmp_path / 'ready.json'; previous.write_bytes(b'previous runtime')
    monkeypatch.setattr(policy, 'urlopen', lambda *a, **kw: io.BytesIO(b'truncated'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        policy.acquire(lock, tmp_path / 'failed', target)
    assert previous.read_bytes() == b'previous runtime'
    assert not list((tmp_path / 'failed').iterdir())
    next(iter(first.values())).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Cached .* checksum mismatch'):
        policy.acquire(lock, tmp_path / 'first', target)


def test_cancellation_retains_only_verified_downloads(tmp_path, lock, monkeypatch):
    monkeypatch.setattr(policy, 'urlopen', lambda *a, **kw: io.BytesIO(b'partial'))
    with pytest.raises(RuntimeError, match='cancelled'):
        policy.acquire(lock, tmp_path, 'windows-x64', cancelled=lambda: True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('target,suffix', [('windows-x64', '.dll'), ('linux-x64', '.so')])
def test_installed_hash_inventory_and_native_transitive_tampering(tmp_path, lock, monkeypatch, target, suffix):
    monkeypatch.setattr(policy, 'host_platform', lambda: target)
    monkeypatch.setattr(policy, 'interpreter_identity', lambda: lock['python'])
    monkeypatch.setattr(policy.sysconfig, 'get_path', lambda name: str(tmp_path / 'site'))
    monkeypatch.setattr(policy.importlib.metadata, 'distribution', lambda name: SimpleNamespace(
        version=lock['versions'][name], locate_file=lambda _: tmp_path / 'site'))
    monkeypatch.setattr(policy.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(tmp_path / 'site' / name / '__init__.py')))
    packages = {}
    for name, row in lock['platforms'][target]['artifacts'].items():
        module = policy.IMPORT_NAMES.get(name, name)
        files = {f'{name}/native{suffix}': hashlib.sha256(name.encode()).hexdigest(),
                 f'{module}/__init__.py': hashlib.sha256(b'').hexdigest()}
        path = tmp_path / 'site' / next(iter(files)); path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        module_path = tmp_path / 'site' / module / '__init__.py'
        module_path.parent.mkdir(parents=True, exist_ok=True); module_path.write_bytes(b'')
        row['files_sha256'] = hashlib.sha256(policy.canonical(files)).hexdigest()
        packages[name] = files
    receipt = tmp_path / policy.RECEIPT_NAME
    receipt.write_text(json.dumps(dict(identity=policy.lock_identity(lock), packages=packages)))
    assert policy.inspect_installed(lock, receipt=receipt)['native_count'] == len(packages)
    hook = tmp_path / 'site/injected.pth'; hook.write_text('import foreign_code')
    with pytest.raises(ValueError, match='startup hooks'):
        policy.inspect_installed(lock, receipt=receipt)
    hook.unlink()
    with monkeypatch.context() as changed:
        changed.setattr(policy.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(tmp_path / 'sil-fallback' / name / '__init__.py')))
        with pytest.raises(ValueError, match='outside its locked inventory'):
            policy.inspect_installed(lock, receipt=receipt)
    native = tmp_path / 'site/numpy' / ('native' + suffix)
    native.write_bytes(b'wrong native ABI')
    with pytest.raises(ValueError, match='checksum mismatch'):
        policy.inspect_installed(lock, receipt=receipt, verify_files=True)
    native.write_bytes(b'numpy')
    # Worker startup never hashes runtime files: it uses install-time records.
    with monkeypatch.context() as startup:
        def no_hashing(_path):
            raise AssertionError('runtime file hashed at worker startup')
        startup.setattr(policy, 'digest', no_hashing)
        receipt_data = json.loads(receipt.read_text())
        receipt.write_text(json.dumps(dict(receipt_data, python_executable_sha256='e' * 64)))
        identity = policy.inspect_installed(lock, receipt=receipt)
        assert identity['python_executable_sha256'] == 'e' * 64
        native.rename(native.with_name('moved'))
        with pytest.raises(ValueError, match='Runtime file missing'):
            policy.inspect_installed(lock, receipt=receipt)
        native.with_name('moved').rename(native)
        receipt.write_text(json.dumps(receipt_data))
    (tmp_path / 'site' / ('injected' + suffix)).write_bytes(b'foreign loader library')
    with pytest.raises(ValueError, match='Untracked native'): policy.inspect_installed(lock, receipt=receipt)


def test_installed_inventory_cannot_be_rewritten_to_bless_tampering(tmp_path, lock, monkeypatch):
    monkeypatch.setattr(policy, 'interpreter_identity', lambda: lock['python'])
    receipt = tmp_path / policy.RECEIPT_NAME
    packages = {name: {} for name in policy.REQUIRED}
    receipt.write_text(json.dumps(dict(identity=policy.lock_identity(lock), packages=packages)))
    monkeypatch.setattr(policy.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version=lock['versions'][name]))
    with pytest.raises(ValueError, match='inventory was altered'): policy.inspect_installed(lock, receipt=receipt)


def test_installed_wrong_version_never_accepted(tmp_path, lock, monkeypatch):
    monkeypatch.setattr(policy, 'interpreter_identity', lambda: lock['python'])
    monkeypatch.setattr(policy.sysconfig, 'get_path', lambda name: str(tmp_path / 'site'))
    receipt = tmp_path / policy.RECEIPT_NAME
    for row in lock['platforms'][policy.host_platform()]['artifacts'].values():
        row['files_sha256'] = hashlib.sha256(policy.canonical({})).hexdigest()
    receipt.write_text(json.dumps(dict(identity=policy.lock_identity(lock), packages={n: {} for n in policy.REQUIRED})))
    monkeypatch.setattr(policy.importlib.metadata, 'distribution', lambda name: SimpleNamespace(version='0.0'))
    with pytest.raises(ValueError, match='version mismatch'): policy.inspect_installed(lock, receipt=receipt)


def test_all_wheel_native_transitives_are_hashed(tmp_path):
    wheel = tmp_path / 'fixture.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('package/transitive/library.dll', b'ABI dependency')
        archive.writestr('package/__init__.py', b'')
        archive.writestr('package.dist-info/RECORD', b'installer mutable')
    files = policy.wheel_files(wheel)
    assert files['package/transitive/library.dll'] == hashlib.sha256(b'ABI dependency').hexdigest()
    assert not any(p.endswith('RECORD') for p in files)


def test_linux_qualification_explicitly_deferred(lock):
    assert lock['platforms']['linux-x64']['qualification'] == 'deferred'


def test_installer_runs_offline_hash_enforcing_install():
    source = (LOCK.parent / 'materialize_sdk.py').read_text()
    for flag in ('--no-deps', '--no-index', '--require-hashes'):
        assert flag in source
