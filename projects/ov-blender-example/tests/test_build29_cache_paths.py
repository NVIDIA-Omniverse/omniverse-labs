# SPDX-License-Identifier: Apache-2.0
"""Build 29 follow-up: a persistent DerivedDataCache and stale worker output roots cleaned."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'addon'))
import public_sdk_backend
from public_sdk_backend import RendererBackend
from ovrtx_blender_example import direct_sdk_client as client
from test_public_sdk_install import manager, fake_local, fake_commands, finish  # noqa: F401 - fixtures

VERSION = '0.6.0.1791947691'


class ReachedNativeInitialization(Exception):
    pass


def _load(output_root, monkeypatch):
    """Run ``load`` up to native initialization; return the renderer configuration."""
    configurations = []
    def initialize(config):
        configurations.append(config)
        raise ReachedNativeInitialization()
    owner = RendererBackend.__new__(RendererBackend)
    owner.output_root = output_root
    owner.active_cuda_gpus = '0'
    owner.unload = lambda: None
    owner.rtx = SimpleNamespace(RendererConfig=lambda **kw: kw, Renderer=initialize)
    monkeypatch.setattr(public_sdk_backend, '_ovrtx_version', lambda: VERSION)
    with pytest.raises(ReachedNativeInitialization):
        owner.dispatch('load', {'sensor_paths': ['/Render/Camera']})
    return configurations[0]


def test_derived_data_cache_path_stable_across_epochs(tmp_path, monkeypatch):
    cache_root = tmp_path/'settings'/'ovrtx-sdk'/'cache'/('ovrtx-'+VERSION)
    monkeypatch.setenv('OV_BLENDER_OVRTX_CACHE_ROOT', str(cache_root))
    monkeypatch.setenv('UJITSO_DATASTORE_CONFIG', 'grpcdns://foreign-cache:1234')
    persistent = (cache_root.parent/('ujitso-'+VERSION)).resolve()
    expected = 'local://'+persistent.as_posix()
    worker = tmp_path/'temp'/'ov-public-renderer-1-0123456789'
    for epoch in ('epoch-1', 'epoch-2'):
        configuration = _load(worker/epoch, monkeypatch)
        assert configuration['datastore_cache'] == expected
        assert os.environ['UJITSO_DATASTORE_CONFIG'] == expected
        assert configuration['cache_root_path'] == str(cache_root.resolve())
        assert not (worker/epoch/'cache'/'DerivedDataCache').exists()
    # A second Blender process (another output root) shares the same folder.
    assert _load(tmp_path/'temp'/'ov-public-renderer-2-0123456789'/'epoch-1', monkeypatch)['datastore_cache'] == expected
    assert persistent.is_dir() and persistent.parent == cache_root.parent.resolve()


def test_derived_data_cache_falls_back_without_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv('OV_BLENDER_OVRTX_CACHE_ROOT', raising=False)
    output = tmp_path/'ov-public-renderer-1-0123456789'/'epoch-1'
    configuration = _load(output, monkeypatch)
    expected = 'local://'+(output/'cache'/'DerivedDataCache').resolve().as_posix()
    assert configuration['datastore_cache'] == expected
    assert os.environ['UJITSO_DATASTORE_CONFIG'] == expected
    assert (output/'cache'/'DerivedDataCache').is_dir()
    assert not list(tmp_path.glob('ujitso-*'))
    # A blank value is the same as none.
    monkeypatch.setenv('OV_BLENDER_OVRTX_CACHE_ROOT', '  ')
    assert _load(output, monkeypatch)['datastore_cache'] == expected


def _root(base, name, files):
    root = base/name
    root.mkdir(exist_ok=True)
    for relative, data in files.items():
        (root/relative).parent.mkdir(parents=True, exist_ok=True)
        (root/relative).write_bytes(data)
    return root


def test_stale_output_roots_removed_only_when_owner_dead_and_marked(tmp_path, monkeypatch):
    dead, alive = 4242, 4343
    epoch = {'epoch-1/worker.log': b'log', 'epoch-1/cache/DerivedDataCache/largecachedata_0': b'x'*1000}
    # A new session writes the marker before the worker starts.
    marked_dead = tmp_path/f'ov-public-renderer-{dead}-0123456789'
    client.write_output_root_marker(marked_dead, 'renderer', pid=dead)
    _root(tmp_path, marked_dead.name, epoch)
    marked_alive = tmp_path/f'ov-public-renderer-{alive}-0123456789'
    client.write_output_root_marker(marked_alive, 'renderer', pid=alive)
    _root(tmp_path, marked_alive.name, epoch)
    # Roots from add-on versions before the marker: accepted-scene + epoch-N with worker.log.
    legacy_dead = _root(tmp_path, f'ov-public-renderer-{dead}-aaaaaaaaaa',
                        {**epoch, 'accepted-scene/scene.usda': b'#usda 1.0\n'})
    legacy_physics = _root(tmp_path, f'ov-public-physics-{dead}-bbbbbbbbbb', {'epoch-1/worker.log': b'log'})
    legacy_alive = _root(tmp_path, f'ov-public-renderer-{alive}-aaaaaaaaaa', epoch)
    # Not recognisably the add-on's: kept even though the PID is dead.
    kept = [
        _root(tmp_path, f'ov-public-renderer-{dead}-cccccccccc', {'epoch-1/worker.log': b'log', 'notes.txt': b'user'}),
        _root(tmp_path, f'ov-public-renderer-{dead}-dddddddddd', {'epoch-1/ovrtx.log': b'no worker log'}),
        _root(tmp_path, f'ov-public-renderer-{dead}-eeeeeeeeee', {'data.bin': b'user'}),
        _root(tmp_path, f'ov-public-renderer-{dead}-0123', epoch),  # wrong name
        _root(tmp_path, f'ov-public-viewer-{dead}-0123456789', epoch),
        _root(tmp_path, f'ov-public-renderer-{dead}-ffffffffff', {}),  # empty
    ]
    foreign_marker = tmp_path/f'ov-public-renderer-{dead}-9999999999'
    client.write_output_root_marker(foreign_marker, 'renderer', pid=alive)  # names another process
    kept.append(foreign_marker)
    bad_marker = _root(tmp_path, f'ov-public-renderer-{dead}-8888888888',
                       {**epoch, client.OUTPUT_ROOT_MARKER: b'not json'})
    kept.append(bad_marker)
    own = tmp_path/f'ov-public-renderer-{os.getpid()}-7777777777'
    client.write_output_root_marker(own, 'renderer')
    kept.append(own)
    unrelated = _root(tmp_path, 'other-folder', {'epoch-1/worker.log': b'log'})

    expected_bytes = sum(f.stat().st_size for p in (marked_dead, legacy_dead, legacy_physics)
                         for f in p.rglob('*') if f.is_file())
    asked = []
    def running(pid):
        asked.append(pid)
        return pid == alive
    # The default per-user %TEMP%: roots without a host (legacy layout) are this host's.
    removed, freed = client.remove_stale_output_roots(tmp_path, running=running, legacy_roots=True)

    assert sorted(removed) == sorted(p.name for p in (marked_dead, legacy_dead, legacy_physics))
    assert freed == expected_bytes
    for path in (marked_dead, legacy_dead, legacy_physics):
        assert not path.exists()
    for path in (marked_alive, legacy_alive, *kept, unrelated):
        assert path.is_dir(), path
    assert (marked_alive/'epoch-1/cache/DerivedDataCache/largecachedata_0').is_file()
    assert set(asked) <= {dead, alive}
    # Running it again changes nothing.
    assert client.remove_stale_output_roots(tmp_path, running=running, legacy_roots=True) == ([], 0)


def test_stale_output_root_in_use_is_kept_with_its_marker(tmp_path, monkeypatch):
    root = tmp_path/'ov-public-renderer-4242-0123456789'
    _root(tmp_path, root.name, {'epoch-1/worker.log': b'log', 'accepted-scene/a.usda': b'a'})
    real = client.shutil.rmtree
    def busy(path, *args, **kwargs):
        if Path(path).name == 'epoch-1':
            raise PermissionError('in use')
        return real(path, *args, **kwargs)
    monkeypatch.setattr(client.shutil, 'rmtree', busy)
    assert client.remove_stale_output_roots(tmp_path, running=lambda pid: False, legacy_roots=True) == ([], 0)
    # The legacy root was adopted: its marker stays, so the next start retries it.
    assert json.loads((root/client.OUTPUT_ROOT_MARKER).read_text())['pid'] == 4242
    monkeypatch.setattr(client.shutil, 'rmtree', real)
    # The adopted marker names this host, so the retry needs no legacy permission.
    assert client.remove_stale_output_roots(tmp_path, running=lambda pid: False)[0] == [root.name]
    assert not root.exists()


def _start_session(monkeypatch):
    """Start a renderer session with a fake worker; wait for its cleanup; return its root."""
    sessions = []
    class Session:
        def __init__(self, **kwargs):
            sessions.append(kwargs)
    import ovrtx_blender_example.direct_runtime_session as runtime_session
    import ovrtx_blender_example.bundled_runtime as bundled
    import ovrtx_blender_example.session_lifecycle as lifecycle
    monkeypatch.setattr(runtime_session, 'RuntimeSession', Session)
    monkeypatch.setattr(bundled, 'public_sdk_python', lambda: str(Path(sys.executable)))
    monkeypatch.setattr(client, 'worker_script', lambda: Path(__file__))
    monkeypatch.setattr(client, '_worker_environment', lambda: {})
    monkeypatch.setattr(lifecycle, 'pid_running', lambda pid: pid == os.getpid())
    client._session('renderer')
    with client._cleanup_running:  # held until the background cleanup finishes
        pass
    return Path(sessions[0]['output_root'])


def test_session_marks_its_root_and_starts_cleanup(tmp_path, monkeypatch):
    stale = tmp_path/'ov-public-renderer-4242-0123456789'
    client.write_output_root_marker(stale, 'renderer', pid=4242)
    monkeypatch.setenv(client.OUTPUT_ENV, str(tmp_path))
    root = _start_session(monkeypatch)
    record = json.loads((root/client.OUTPUT_ROOT_MARKER).read_text())
    assert record == dict(schema=2, application='ovrtx_blender_example', kind='renderer', pid=os.getpid(),
                          host=client.socket.gethostname())
    assert not stale.exists() and root.is_dir()


_EPOCH = {'epoch-1/worker.log': b'log', 'accepted-scene/scene.usda': b'#usda 1.0\n'}


def _edit_marker(root, **changes):
    marker = root/client.OUTPUT_ROOT_MARKER
    record = {**json.loads(marker.read_text()), **changes}
    marker.write_text(json.dumps({key: value for key, value in record.items() if value is not None}))


def test_stale_root_of_other_host_kept(tmp_path, monkeypatch):
    # OV_BLENDER_PUBLIC_SDK_OUTPUT may be a folder other machines share. Their PIDs
    # cannot be checked here, so their roots are never judged dead.
    this_host = tmp_path/'ov-public-renderer-4242-0123456789'
    client.write_output_root_marker(this_host, 'renderer', pid=4242)
    _root(tmp_path, this_host.name, _EPOCH)
    other_host = tmp_path/'ov-public-renderer-4242-aaaaaaaaaa'
    client.write_output_root_marker(other_host, 'renderer', pid=4242)
    _root(tmp_path, other_host.name, _EPOCH)
    _edit_marker(other_host, host=client.socket.gethostname() + '-other')
    no_host = tmp_path/'ov-public-physics-4242-bbbbbbbbbb'  # schema 2 without its host
    client.write_output_root_marker(no_host, 'physics', pid=4242)
    _root(tmp_path, no_host.name, _EPOCH)
    _edit_marker(no_host, host=None)
    assert json.loads((this_host/client.OUTPUT_ROOT_MARKER).read_text())['host'] == client.socket.gethostname()

    # Also with legacy roots allowed (the default %TEMP%): a marker's host is binding.
    for legacy_roots in (False, True):
        removed, _ = client.remove_stale_output_roots(tmp_path, running=lambda pid: False,
                                                      legacy_roots=legacy_roots)
        assert removed == ([] if legacy_roots else [this_host.name])
    assert not this_host.exists()
    for root in (other_host, no_host):
        assert (root/'epoch-1/worker.log').is_file() and (root/client.OUTPUT_ROOT_MARKER).is_file()

    # Through a session on the shared folder: the other host's root survives.
    monkeypatch.setenv(client.OUTPUT_ENV, str(tmp_path))
    own = _start_session(monkeypatch)
    assert own.parent == tmp_path and own.is_dir() and other_host.is_dir() and no_host.is_dir()


def test_legacy_root_kept_under_custom_output_base(tmp_path, monkeypatch):
    def roots(base):
        base.mkdir()
        legacy = _root(base, 'ov-public-renderer-4242-aaaaaaaaaa', _EPOCH)  # before the marker
        first_schema = _root(base, 'ov-public-renderer-4242-bbbbbbbbbb', {  # marker without a host
            **_EPOCH, client.OUTPUT_ROOT_MARKER: json.dumps(dict(
                schema=1, application='ovrtx_blender_example', kind='renderer', pid=4242)).encode()})
        marked = base/'ov-public-renderer-4242-cccccccccc'
        client.write_output_root_marker(marked, 'renderer', pid=4242)
        _root(base, marked.name, _EPOCH)
        return legacy, first_schema, marked

    # OV_BLENDER_PUBLIC_SDK_OUTPUT set: the host of a root without one is unknown.
    custom, temp = tmp_path/'custom', tmp_path/'temp'
    legacy, first_schema, marked = roots(custom)
    monkeypatch.setenv(client.OUTPUT_ENV, str(custom))
    monkeypatch.setattr(client.tempfile, 'gettempdir', lambda: str(temp))
    own = _start_session(monkeypatch)
    assert own.parent == custom and not marked.exists()
    for root in (legacy, first_schema):
        assert (root/'epoch-1/worker.log').is_file() and (root/'accepted-scene/scene.usda').is_file()
    assert not (legacy/client.OUTPUT_ROOT_MARKER).exists()  # not adopted
    assert json.loads((first_schema/client.OUTPUT_ROOT_MARKER).read_text())['schema'] == 1
    # A direct call keeps them too: its default is the safe one.
    assert client.remove_stale_output_roots(custom, running=lambda pid: False) == ([], 0)

    # No override: the default per-user %TEMP% is this host's, so they are removed.
    monkeypatch.delenv(client.OUTPUT_ENV)
    legacy_temp, first_schema_temp, marked_temp = roots(temp)
    own = _start_session(monkeypatch)
    assert own.parent == temp and own.is_dir()
    assert not legacy_temp.exists() and not first_schema_temp.exists() and not marked_temp.exists()
    assert legacy.is_dir() and first_schema.is_dir()


def _make_link(kind, link, target):
    if kind == 'junction':
        if sys.platform != 'win32':
            pytest.skip('junctions are Windows-only')
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f'symlinks unavailable: {error}')


def _remove_link(link):
    if link.is_symlink() or sys.platform != 'win32':
        os.unlink(link)
    else:
        os.rmdir(link)  # a junction; its target is untouched


@pytest.mark.parametrize('kind', ['junction', 'symlink'])
@pytest.mark.parametrize('where', ['link', 'epoch-1/link', 'epoch-1/cache/link'])
def test_root_with_link_or_junction_kept(tmp_path, kind, where):
    target = tmp_path/'target'
    (target/'keep').mkdir(parents=True)
    (target/'keep/data.bin').write_bytes(b'user data')
    root = tmp_path/'ov-public-renderer-4242-0123456789'
    client.write_output_root_marker(root, 'renderer', pid=4242)
    _root(tmp_path, root.name, {'epoch-1/worker.log': b'log', 'epoch-1/cache/DerivedDataCache/x': b'x'})
    deletable = tmp_path/'ov-public-renderer-4242-aaaaaaaaaa'  # the same root without the link
    client.write_output_root_marker(deletable, 'renderer', pid=4242)
    _root(tmp_path, deletable.name, {'epoch-1/worker.log': b'log', 'epoch-1/cache/DerivedDataCache/x': b'x'})
    link = root/where
    link.parent.mkdir(parents=True, exist_ok=True)
    _make_link(kind, link, target)
    assert client._is_link(link)

    for legacy_roots in (False, True):
        removed, _ = client.remove_stale_output_roots(tmp_path, running=lambda pid: False,
                                                      legacy_roots=legacy_roots)
        assert removed == ([] if legacy_roots else [deletable.name])
    # The root, its marker, its files and the link survive; so does the link target.
    assert (root/client.OUTPUT_ROOT_MARKER).is_file() and (root/'epoch-1/worker.log').is_file()
    assert (root/'epoch-1/cache/DerivedDataCache/x').is_file()
    assert client._is_link(link) and (link/'keep/data.bin').read_bytes() == b'user data'
    assert sorted(path.name for path in target.rglob('*')) == ['data.bin', 'keep']

    # A root that is itself a link is refused too, and its target kept. Without the
    # link inside it, the first root is now removed.
    _remove_link(link)
    linked_root = tmp_path/'ov-public-renderer-4242-bbbbbbbbbb'
    client.write_output_root_marker(target, 'renderer', pid=4242)
    _make_link(kind, linked_root, target)
    assert client.remove_stale_output_roots(tmp_path, running=lambda pid: False)[0] == [root.name]
    assert client._is_link(linked_root) and (target/'keep/data.bin').read_bytes() == b'user data'
    assert (target/client.OUTPUT_ROOT_MARKER).is_file()
    _remove_link(linked_root)  # so pytest's temp cleanup never walks into the target twice


def test_sidecar_remove_accepts_persistent_ujitso_cache(manager, monkeypatch):  # noqa: F811
    from ovrtx_blender_example import public_sdk_sidecar as sidecar
    fake_commands(manager, monkeypatch)
    manager.start(); assert finish(manager)['state'] == 'ready'
    cache_root = manager.ovrtx_cache_root()
    ujitso = cache_root.parent/('ujitso-'+cache_root.name.removeprefix('ovrtx-'))
    files = {'cacheindex.dat': b'i'*64, 'cachelru.dat': b'l'*16, 'smallcachedata_0': b's'*4096,
             'largecachedata_0': b'L'*(1 << 20), 'app_instance_lock': b'',
             'fork-1/cacheindex.dat': b'f'*8}
    for relative, data in files.items():
        (ujitso/relative).parent.mkdir(parents=True, exist_ok=True)
        (ujitso/relative).write_bytes(data)
    (cache_root/'shadercache').mkdir(parents=True)
    # Upgrade: a local package replaces the runtime; the cache is outside every candidate.
    fake_local(manager, monkeypatch)
    fake_commands(manager, monkeypatch)
    assert manager.start(online_access=False, source='chosen.zip')
    assert finish(manager)['state'] == 'ready'
    root, _, requirements, _, _ = manager._config
    candidate = root/'sidecars'/('a'*32)
    sidecar.installed_python(candidate, root, requirements, verify_files=True)
    # Remove: verified against the add-on's pin, like test_public_sdk_remove's fixture.
    import hashlib
    package = root.parent/'package'; (package/'runtime').mkdir(parents=True)
    for source in (requirements, requirements.with_name('public-sdk-lock.json')):
        (package/'runtime'/source.name).write_bytes(source.read_bytes())
    (package/'public-sdk.json').write_text(json.dumps({'sidecar_manifest_sha256':
        hashlib.sha256((candidate/'sidecar.json').read_bytes()).hexdigest()}))
    manager.configure(root, sys.executable, package/'runtime'/requirements.name)
    assert manager.ready_python()
    assert manager.remove()
    status = finish(manager)
    assert status['state'] == 'unconfigured', status
    assert not candidate.exists() and not (root/'ready.json').exists()
    # The persistent caches are the profile's, not the runtime payload's: kept as they were.
    assert {p.relative_to(ujitso).as_posix(): p.read_bytes() for p in ujitso.rglob('*') if p.is_file()} == files
