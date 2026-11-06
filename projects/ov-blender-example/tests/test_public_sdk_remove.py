# SPDX-License-Identifier: Apache-2.0
"""Clean packaged removal: ownership, transactional boundary and active owners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import public_sdk_install as install
from ovrtx_blender_example import public_sdk_sidecar as sidecar
from test_public_sdk_install import manager, fake_local, fake_commands, finish


@pytest.fixture
def installed(manager, monkeypatch):
    fake_local(manager, monkeypatch)
    fake_commands(manager, monkeypatch)
    manager.start(source='chosen.zip')
    assert finish(manager)['state'] == 'ready'
    root, _, requirements, _, _ = manager._config
    candidate = root/'sidecars'/('a'*32)
    marker = requirements.parent.parent/'public-sdk.json'
    # Fixture requirements are at tmp root; give the marker its own package
    # directory to avoid touching another parametrized fixture's pin.
    package = root.parent/'package'; (package/'runtime').mkdir(parents=True)
    for source in (requirements, requirements.with_name('public-sdk-lock.json')):
        (package/'runtime'/source.name).write_bytes(source.read_bytes())
    requirements = package/'runtime'/requirements.name
    (package/'public-sdk.json').write_text(json.dumps({'sidecar_manifest_sha256':
        hashlib.sha256((candidate/'sidecar.json').read_bytes()).hexdigest()}))
    manager.configure(root, sys.executable, requirements)
    assert manager.ready_python()
    return manager, root, candidate


def test_remove_managed_candidate_preserves_siblings_logs_and_external_files(installed):
    manager, root, candidate = installed
    sibling = root/'sidecars'/('b'*32); sibling.mkdir(); (sibling/'user').write_bytes(b'keep')
    outside = root.parent/'external-python.exe'; outside.write_bytes(b'external')
    logs = {p.name:p.read_bytes() for p in (root/'logs').iterdir()}
    assert manager.remove()
    assert finish(manager)['state'] == 'unconfigured'
    assert manager.ready_python() is None and not (root/'ready.json').exists()
    assert not candidate.exists()
    assert (sibling/'user').read_bytes() == b'keep' and outside.read_bytes() == b'external'
    assert {p.name:p.read_bytes() for p in (root/'logs').iterdir()} == logs


@pytest.mark.parametrize('kind', ['unknown', 'tamper', 'pin', 'external', 'unconfirmed', 'active'])
def test_removal_refusals_do_not_mutate_selection_or_candidate(installed, monkeypatch, kind):
    manager, root, candidate = installed
    if kind == 'unknown': (candidate/'user.txt').write_bytes(b'user')
    if kind == 'tamper': Path(manager.ready_python()).write_bytes(b'changed')
    if kind == 'pin': (manager._config[2].parent.parent/'public-sdk.json').write_text('{}')
    if kind == 'external': monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PYTHON', str(candidate/'python/python.exe'))
    if kind == 'unconfirmed': manager._cleanup_safe = False
    if kind == 'active': manager._thread = SimpleNamespace(is_alive=lambda: True, join=lambda *_: None)
    before = {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    if kind in {'external', 'unconfirmed', 'active'}:
        with pytest.raises(RuntimeError): manager.remove()
        if kind == 'active': manager._thread = None
    else:
        manager.remove(); assert finish(manager)['state'] == 'failed'
    assert {p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


def test_receipt_deactivation_failure_restores_original_selection(installed, monkeypatch):
    manager, root, candidate = installed
    receipt = (root/'ready.json').read_bytes(); python = manager.ready_python()
    rename = install.os.rename
    def fail(source, destination):
        if Path(source) == root/'ready.json': raise PermissionError('receipt denied')
        return rename(source, destination)
    monkeypatch.setattr(install.os, 'rename', fail)
    manager.remove(); state = finish(manager)
    assert state['state'] == 'failed' and 'receipt denied' in state['error']
    assert manager.ready_python() == python and Path(python).is_file()
    assert (root/'ready.json').read_bytes() == receipt and candidate.exists()
    assert not list((root/'sidecars').glob('removed-*'))


def test_delete_failure_keeps_disabled_candidate_and_reports_partial_failure(installed, monkeypatch):
    manager, root, candidate = installed
    unlink = Path.unlink
    def fail(path, *args, **kwargs):
        if path.name == 'python.exe': raise PermissionError('file denied')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail)
    manager.remove(); state = finish(manager)
    assert state['state'] == 'failed' and 'file denied' in state['error']
    assert 'remaining files:' in state['error'] and manager.ready_python() is None
    assert not (root/'ready.json').exists()
    assert len(list((root/'sidecars').glob('removed-*'))) == 1


def test_runtime_selection_and_setup_are_excluded_while_removal_validates(installed, monkeypatch):
    from ovrtx_blender_example import bundled_runtime
    manager, root, candidate = installed
    entered, release = threading.Event(), threading.Event()
    read = manager._read_receipt
    def blocked(config):
        entered.set()
        assert release.wait(3)
        return read(config)
    monkeypatch.setattr(manager, '_read_receipt', blocked)
    monkeypatch.setattr(install, 'ready_python', manager.ready_python)
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PYTHON', raising=False)
    try:
        manager.remove(); assert entered.wait(3)
        assert (root/'ready.json').is_file()
        assert bundled_runtime.public_sdk_python() is None
        assert not manager.start(source='chosen.zip')
        with pytest.raises(RuntimeError): manager.remove()
    finally:
        release.set()
        if manager._thread: manager._thread.join(5)
    assert manager.status()['state'] == 'unconfigured'


def test_empty_unknown_directory_is_retained_and_reported(installed):
    manager, root, candidate = installed
    (candidate/'user-empty').mkdir()
    manager.remove(); state = finish(manager)
    assert state['state'] == 'unconfigured'
    assert (Path(state['retained_path'])/'user-empty').is_dir()
    assert 'additional entries were preserved' in state['message']


def test_second_manifest_read_cannot_expand_deletion_ownership(installed, monkeypatch):
    manager, root, candidate = installed
    ready = (root/'ready.json').read_bytes()
    read = manager._read_receipt
    def replace_manifest(config):
        python = read(config)
        (candidate/'user.txt').write_bytes(b'not owned')
        path = candidate/'sidecar.json'
        manifest = json.loads(path.read_bytes())
        manifest['files']['user.txt'] = hashlib.sha256(b'not owned').hexdigest()
        path.write_text(json.dumps(manifest))
        return python
    monkeypatch.setattr(manager, '_read_receipt', replace_manifest)
    manager.remove(); state = finish(manager)
    assert state['state'] == 'failed' and 'manifest changed' in state['error']
    assert (root/'ready.json').read_bytes() == ready
    assert (candidate/'user.txt').read_bytes() == b'not owned'
    assert (candidate/'python/python.exe').is_file()
    assert not list((root/'sidecars').glob('removed-*'))


def test_directory_permission_error_is_not_reported_as_complete(installed, monkeypatch):
    manager, root, candidate = installed
    rmdir = Path.rmdir
    def denied(path, *args, **kwargs):
        if path.name == 'python': raise PermissionError('directory denied')
        return rmdir(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'rmdir', denied)
    manager.remove(); state = finish(manager)
    assert state['state'] == 'failed' and 'directory denied' in state['error']
    assert manager.ready_python() is None


@pytest.mark.parametrize('cause', ['external', 'installing', 'render', 'viewport', 'scene', 'unknown-scene'])
def test_ui_refuses_unconfirmed_or_active_owners_before_removal(monkeypatch, cause):
    import ovrtx_blender_example as addon
    from ovrtx_blender_example import engine, scene_generation_sessions as scenes
    called = []
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PYTHON', raising=False)
    if cause == 'external': monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PYTHON', 'external')
    monkeypatch.setattr(install, 'status', lambda: {'state':'installing' if cause == 'installing' else 'ready'})
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', cause == 'render')
    monkeypatch.setattr(engine, 'stop_viewport_sessions_for_unregister', lambda: cause != 'viewport')
    monkeypatch.setattr(scenes, 'close', lambda: None)
    monkeypatch.setattr(scenes, 'close_status', lambda: {'status':'failed' if cause == 'scene' else 'unknown' if cause == 'unknown-scene' else 'closed'})
    monkeypatch.setattr(install, 'remove', lambda: called.append('remove'))
    with pytest.raises(RuntimeError): addon._remove_public_sdk_runtime()
    assert called == []


def test_ui_stops_owners_before_starting_background_removal(monkeypatch):
    import ovrtx_blender_example as addon
    from ovrtx_blender_example import engine, scene_generation_sessions as scenes
    events = []
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PYTHON', raising=False)
    monkeypatch.setattr(install, 'status', lambda: {'state':'ready'})
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', False)
    monkeypatch.setattr(engine, 'stop_viewport_sessions_for_unregister', lambda: events.append('viewport') or True)
    monkeypatch.setattr(scenes, 'close', lambda: events.append('scene'))
    monkeypatch.setattr(scenes, 'close_status', lambda: {'status':'closed'})
    monkeypatch.setattr(install, 'remove', lambda: events.append('remove'))
    monkeypatch.setattr(addon, 'bpy', SimpleNamespace(app=SimpleNamespace(timers=None)))
    addon._remove_public_sdk_runtime()
    assert events == ['viewport', 'scene', 'remove']


@pytest.mark.parametrize('failed', [False, True])
def test_public_override_reset_preserves_external_environment(monkeypatch, tmp_path, failed):
    import ovrtx_blender_example as addon
    external = tmp_path/'external/python.exe'; external.parent.mkdir(); external.write_bytes(b'owned by user')
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PYTHON', str(external))
    events = []
    def close():
        events.append('close')
        if failed: raise RuntimeError('owner still live')
    monkeypatch.setattr(addon, '_close_public_sdk_runtime_owners', close)
    monkeypatch.setattr(addon, '_start_public_sdk_setup', lambda **kw: events.append(('configure installed', kw)))
    monkeypatch.setattr(addon, '_bump_runtime_generation', lambda: events.append('invalidate selection'))
    if failed:
        with pytest.raises(RuntimeError): addon._reset_public_sdk_override()
        assert install.os.environ['OV_BLENDER_PUBLIC_SDK_PYTHON'] == str(external)
        assert events == ['close']
    else:
        addon._reset_public_sdk_override()
        assert 'OV_BLENDER_PUBLIC_SDK_PYTHON' not in install.os.environ
        assert events == ['close', ('configure installed', {'configure_only':True}), 'invalidate selection']
    assert external.read_bytes() == b'owned by user'


@pytest.mark.parametrize('mode', ['automatic', 'local_package'])
def test_reset_configuration_never_launches_or_downloads_missing_runtime(monkeypatch, tmp_path, mode):
    import ovrtx_blender_example as addon
    package=tmp_path/'package'; package.mkdir()
    (package/'public-sdk.json').write_text(json.dumps({'schema':1,'backend':'public_sdk','install_mode':mode}))
    events=[]
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PYTHON', raising=False)
    monkeypatch.setattr(addon.bundled_runtime, 'public_sdk_package_root', lambda: package)
    monkeypatch.setattr(addon, '_public_sdk_storage_root', lambda: tmp_path/'storage')
    monkeypatch.setattr(addon, '_bundled_python_path', lambda: Path(sys.executable))
    monkeypatch.setattr(addon, 'bpy', SimpleNamespace(app=SimpleNamespace(timers=None)))
    monkeypatch.setattr(install, 'configure', lambda *args: events.append('configure'))
    monkeypatch.setattr(install, 'start', lambda **kw: pytest.fail('Reset must not install/download/execute Python'))
    assert addon._start_public_sdk_setup(configure_only=True) is False
    assert events == ['configure']
