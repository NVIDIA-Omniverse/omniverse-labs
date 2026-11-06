"""Native admission identities include the platform's real package binaries.

Worker start never hashes files: native hashes come from the install-time
integrity receipt and the adapter identity from its build-time record.
"""
import hashlib
import json
from pathlib import Path
import platform as host_platform
import sys
from types import SimpleNamespace

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
sys.path.insert(0, str(RUNTIME))
import public_sdk_backend as backend
import runtime_identity
import physics_adapter_identity
from physics_adapter_identity import adapter_identity, recorded_identity


@pytest.fixture(autouse=True)
def isolated_lock(monkeypatch):
    monkeypatch.setattr(runtime_identity, 'verified_identity', lambda: {'lock_sha256': 'c' * 64})


def native_packages(tmp_path, worker, extension, *, windows_extras=False):
    """Create package trees in a fake site and return the matching receipt rows."""
    site = tmp_path / 'site'
    packages = {}
    for name in ('ovphysx', 'ovstage'):
        root = site / name
        (root / 'plugins').mkdir(parents=True)
        (root / '__init__.py').write_text('')
        (root / 'plugins' / ('native' + extension)).write_bytes(name.encode())
        (root / 'ignored.txt').write_bytes(b'not native')
        rows = {f'{name}/__init__.py': hashlib.sha256(b'').hexdigest(),
                f'{name}/ignored.txt': hashlib.sha256(b'not native').hexdigest(),
                f'{name}/plugins/native{extension}': hashlib.sha256(name.encode()).hexdigest(),
                # Outside the package directory: never part of its native roster.
                f'{name}-1.0.dist-info/METADATA': 'd' * 64}
        if windows_extras:
            (root / 'binding.PYD').write_bytes(b'python binding')
            (root / 'unrelated.so').write_bytes(b'not Windows native')
            rows[f'{name}/binding.PYD'] = hashlib.sha256(b'python binding').hexdigest()
            rows[f'{name}/unrelated.so'] = hashlib.sha256(b'not Windows native').hexdigest()
        packages[name] = rows
        setattr(worker, name, SimpleNamespace(__file__=str(root / '__init__.py'), __name__=name))
    return packages


def write_receipt(prefix, packages):
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / 'public-sdk-integrity.json').write_text(json.dumps({'identity': {}, 'packages': packages}))


def new_worker():
    worker = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    worker._runtime_identity = None
    return worker


@pytest.fixture
def no_file_hashing(monkeypatch):
    """Fail if worker identity reads file contents for hashing."""
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Worker start must not hash files')
    monkeypatch.setattr(hashlib, 'file_digest', forbidden)
    original_open = Path.open
    def guarded_open(path, mode='r', *args, **kwargs):
        if mode.startswith('r') and path.suffix.lower() in ('.dll', '.pyd', '.so'):
            forbidden(path)
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: forbidden(path))
    monkeypatch.setattr(physics_adapter_identity, 'adapter_identity', forbidden)
    return calls


@pytest.mark.parametrize('platform,extension,architecture', [
    ('linux', '.so', 'x86_64'), ('win32', '.dll', 'AMD64')])
def test_identity_uses_install_receipt_with_stable_relative_keys(tmp_path, monkeypatch, no_file_hashing,
                                                                 platform, extension, architecture):
    worker = new_worker()
    packages = native_packages(tmp_path, worker, extension, windows_extras=platform == 'win32')
    write_receipt(tmp_path / 'prefix', packages)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'prefix'))
    # Simulate the complete identity. On Windows, asking the real platform
    # module for machine() while sys.platform is temporarily "linux" caches an
    # empty machine string (Windows has no os.uname), poisoning later tests.
    monkeypatch.setattr(host_platform, '_uname_cache', None)
    monkeypatch.setattr(host_platform, 'machine', lambda: architecture)
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(backend.importlib.metadata, 'version', lambda name: 'test-version')
    result = worker.runtime_identity()
    assert no_file_hashing == []
    assert result['platform'] == platform
    assert result['architecture'] == architecture
    assert host_platform._uname_cache is None
    for name in ('ovphysx', 'ovstage'):
        assert result['native_sha256'][f'{name}/plugins/native{extension}'] == hashlib.sha256(name.encode()).hexdigest()
    assert len(result['native_sha256']) == (4 if platform == 'win32' else 2)
    if platform == 'win32':
        assert result['native_sha256']['ovphysx/binding.PYD'] == hashlib.sha256(b'python binding').hexdigest()
    assert all('\\' not in key for key in result['native_sha256'])
    assert (result['adapter_sha256'], result['adapter_components']) == recorded_identity()
    assert result['runtime_lock'] == {'lock_sha256': 'c' * 64}
    assert set(result['adapter_components']) == {
        'public_sdk_backend.py', 'incremental_physics_backend.py',
        'inline_physics_geometry.py', 'renderer_values.py', 'physics_adapter_identity.py',
        'runtime_identity.py', 'public_sdk_worker.py'}


def test_recorded_adapter_identity_matches_sources():
    """The shipped record cannot go stale: it must equal a fresh recomputation."""
    assert recorded_identity() == adapter_identity()
    record = RUNTIME / physics_adapter_identity.RECORD_NAME
    assert record.read_bytes() == physics_adapter_identity.record_bytes()


@pytest.mark.parametrize('corruption', ['digest', 'component', 'roster', 'schema', 'missing'])
def test_invalid_adapter_record_fails_closed(tmp_path, corruption):
    record = json.loads((RUNTIME / physics_adapter_identity.RECORD_NAME).read_text())
    if corruption == 'digest':
        record['adapter_sha256'] = '0' * 64
    elif corruption == 'component':
        record['adapter_components']['renderer_values.py'] = '0' * 64
    elif corruption == 'roster':
        record['adapter_components'].pop('renderer_values.py')
    elif corruption == 'schema':
        record['schema'] = 2
    if corruption != 'missing':
        (tmp_path / physics_adapter_identity.RECORD_NAME).write_text(json.dumps(record))
    with pytest.raises((RuntimeError, OSError)):
        recorded_identity(tmp_path)


def test_effective_adapter_changes_invalidate_identity_but_rendering_does_not(tmp_path):
    root = Path(backend.__file__).parent
    _, components = adapter_identity()
    for name in components:
        (tmp_path / name).write_bytes((root / name).read_bytes())
    original = adapter_identity(tmp_path)
    path = tmp_path / 'public_sdk_backend.py'
    path.write_text(path.read_text() + '\nclass RendererOnly: pass\n')
    assert adapter_identity(tmp_path) == original
    for name in components:
        path = tmp_path / name
        saved = path.read_text()
        if name == 'public_sdk_backend.py':
            changed = saved.replace('class PhysicsBackend:', 'class PhysicsBackend:\n    physics_revision = 2')
        else:
            changed = saved + '\nphysics_revision = 2\n'
        path.write_text(changed)
        assert adapter_identity(tmp_path)[0] != original[0], name
        path.write_text(saved)


def test_missing_one_native_package_fails_closed(tmp_path, monkeypatch, no_file_hashing):
    worker = new_worker()
    packages = {}
    for name in ('ovphysx', 'ovstage'):
        root = tmp_path / 'site' / name
        root.mkdir(parents=True)
        packages[name] = {}
        if name == 'ovphysx':
            (root / 'native.dll').write_bytes(b'present')
            packages[name]['ovphysx/native.dll'] = hashlib.sha256(b'present').hexdigest()
        setattr(worker, name, SimpleNamespace(__file__=str(root / '__init__.py'), __name__=name))
    write_receipt(tmp_path / 'prefix', packages)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'prefix'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(RuntimeError, match='No native binaries found for ovstage'):
        worker.runtime_identity()
    assert worker._runtime_identity is None


@pytest.mark.parametrize('change', ['untracked_native', 'missing_native', 'no_receipt', 'package_absent'])
def test_native_roster_must_match_install_receipt(tmp_path, monkeypatch, no_file_hashing, change):
    worker = new_worker()
    packages = native_packages(tmp_path, worker, '.dll')
    ovphysx = tmp_path / 'site' / 'ovphysx'
    if change == 'untracked_native':
        (ovphysx / 'plugins' / 'injected.dll').write_bytes(b'not installed')
    elif change == 'missing_native':
        packages['ovphysx']['ovphysx/plugins/removed.dll'] = 'e' * 64
    elif change == 'package_absent':
        packages.pop('ovstage')
    if change != 'no_receipt':
        write_receipt(tmp_path / 'prefix', packages)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'prefix'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(RuntimeError, match='receipt'):
        worker.runtime_identity()
    assert worker._runtime_identity is None
