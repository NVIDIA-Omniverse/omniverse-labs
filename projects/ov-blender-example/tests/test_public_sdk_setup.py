# SPDX-License-Identifier: Apache-2.0
"""Portable source-checkout setup keeps SDKs outside Blender and isolates state."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import public_sdk_environment as env
import setup_public_sdk as setup


def test_version_mismatch_does_not_import_native_sdks(monkeypatch):
    monkeypatch.setattr(env.sys, 'version_info', (3, 13, 0))
    monkeypatch.setattr(env.importlib.metadata, 'version', lambda name: 'wrong')
    def forbidden(name):
        raise AssertionError('Do not import an incompatible SDK')
    monkeypatch.setattr(env.importlib, 'import_module', forbidden)
    result = env.inspect_environment()
    assert not result['ok'] and not result['gpu_render_validated']
    assert any('ovrtx: expected' in error for error in result['errors'])


def test_environment_has_private_paths_external_python_and_explicit_gpu(tmp_path, monkeypatch):
    python = tmp_path / 'Python with spaces' / 'python.exe'
    python.parent.mkdir(); python.touch()
    admission = tmp_path / 'local-admission.json'; admission.write_text('{}')
    monkeypatch.setenv('OVRTX_ACTIVE_CUDA_GPUS', '3')
    monkeypatch.setenv('PYTHONPATH', 'existing')
    config = {'python': str(python), 'output': str(tmp_path / 'output with spaces'),
              'gpu': 1, 'physics_admission': str(admission)}
    actual = env.worker_environment(config)
    assert actual['OV_BLENDER_PUBLIC_SDK_PYTHON'] == str(python)
    assert actual['OV_BLENDER_PUBLIC_SDK_GPU'] == '1'
    assert 'OVRTX_ACTIVE_CUDA_GPUS' not in actual
    assert actual['PYTHONPATH'].endswith(os.pathsep + 'existing')
    assert actual['OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION'] == str(admission)
    for key in ('BLENDER_USER_CONFIG', 'BLENDER_USER_SCRIPTS'):
        assert Path(actual[key]).is_relative_to(tmp_path)
        assert Path(actual[key]).is_dir()


@pytest.mark.parametrize('version', ['Blender 5.1.0', 'Blender 5.0.0', 'Blender 4.5.0', 'unknown'])
def test_wrong_blender_version_is_actionable(tmp_path, monkeypatch, version):
    executable = tmp_path / 'blender.exe'; executable.touch()
    monkeypatch.setattr(env.subprocess, 'run', lambda *args, **kw: SimpleNamespace(stdout=version))
    with pytest.raises(ValueError, match='5.2 or newer'):
        env.check_blender(str(executable))


@pytest.mark.parametrize('returncode', [0, 1])
def test_existing_sdk_setup_never_installs_and_preserves_config_on_failed_check(tmp_path, monkeypatch, returncode):
    executable = tmp_path / 'python.exe'; executable.touch()
    config = tmp_path / 'config.json'; config.write_text('{"previous": true}')
    monkeypatch.setattr(setup, 'check_blender', lambda value: tmp_path / 'blender.exe')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=returncode, stdout='SDK result', stderr='')
    monkeypatch.setattr(setup.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['setup', '--sdk-python', str(executable), '--blender', 'unused',
        '--config', str(config), '--output', str(tmp_path / 'output')])
    if returncode:
        with pytest.raises(SystemExit, match='preflight failed'):
            setup.main()
        assert json.loads(config.read_text()) == {'previous': True}
    else:
        setup.main()
        result = env.read_config(config)
        assert result['python'] == str(executable)
        assert result['physics_admission'].endswith('physics-admission.json')
    assert len(commands) == 1 and commands[0][1].endswith('public_sdk_environment.py')


@pytest.mark.parametrize('system,architecture,valid', [
    ('linux', 'aarch64', False), ('linux', 'arm64', False),
    ('linux', 'x86_64', True), ('win32', 'AMD64', True),
    ('win32', 'ARM64', False), ('darwin', 'arm64', False)])
def test_sdk_preflight_admits_only_supported_native_architectures(monkeypatch, system, architecture, valid):
    monkeypatch.setattr(env.sys, 'version_info', (3, 13, 0))
    monkeypatch.setattr(env.sys, 'platform', system)
    monkeypatch.setattr(env.platform, 'machine', lambda: architecture)
    monkeypatch.setattr(env.importlib.metadata, 'version', lambda name: env.requirements()[name])
    monkeypatch.setattr(env.public_sdk_lock, 'inspect_installed',
                        lambda lock, verify_files: {'verified': verify_files})
    loaded = []
    monkeypatch.setattr(env.importlib, 'import_module', lambda name: loaded.append(name))
    result = env.inspect_environment()
    assert result['ok'] == valid
    assert result['architecture'] == architecture.lower()
    assert bool(loaded) == valid
    if valid:
        assert result['runtime_identity'] == {'verified': True}  # build preflight hashes


@pytest.mark.parametrize('version', ['Blender 5.2.2', 'Blender 5.3.0 Alpha', 'Blender 6.0.0'])
def test_supported_blender_versions_remain_explicit(tmp_path, monkeypatch, version):
    executable = tmp_path/'blender'; executable.touch()
    monkeypatch.setattr(env.subprocess, 'run', lambda *args, **kw: SimpleNamespace(stdout=version))
    assert env.check_blender(str(executable)) == executable
