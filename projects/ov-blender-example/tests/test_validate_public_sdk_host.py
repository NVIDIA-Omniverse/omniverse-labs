"""Host orchestration must promote normally gated evidence before admission."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('physics_host_validation_test', SCRIPTS / 'validate_public_sdk_host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.mark.parametrize('failure', [None, 'physics-production', 'mass-production-gpu', 'physics-promote', 'unpromoted-output'])
def test_admission_requires_full_normal_gate_runs_and_successful_promotion(tmp_path, monkeypatch, failure):
    output = tmp_path / 'run'
    destination = tmp_path / 'existing-admission.json'
    previous = '{"schema_version":1,"measurements":[]}'
    destination.write_text(previous)
    config = {'blender': 'blender', 'output': str(tmp_path), 'physics_admission': str(destination)}
    monkeypatch.setattr(host, 'read_config', lambda path: config)
    monkeypatch.setattr(host, 'check_blender', lambda path: path)
    monkeypatch.setattr(host, 'worker_environment', lambda *args: {'OV_BLENDER_PUBLIC_SDK_PYTHON': 'worker-python'})
    monkeypatch.setattr(sys, 'argv', ['validate_public_sdk_host.py', '--physics-only', '--output', str(output)])
    calls = []
    measured_identity = {'fixture': 'exact measured native/adapter identity'}

    def run(command, *, env, stdout, stderr, timeout):
        label = Path(stdout.name).stem
        calls.append((label, list(command), dict(env)))
        if failure == label:
            return subprocess.CompletedProcess(command, 23)
        if label in ('physics-record', 'physics-promote'):
            path = Path(command[command.index('--destination') + 1])
            status = 'production' if label == 'physics-promote' and failure != 'unpromoted-output' else 'candidate'
            path.write_text(json.dumps({'schema_version': 1, 'measurements': [
                {'identity': measured_identity, 'status': status}]}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(host.subprocess, 'run', run)
    if failure:
        with pytest.raises(RuntimeError):
            host.main()
    else:
        host.main()
    report = json.loads((output / 'result.json').read_text())
    labels = [label for label, _, _ in calls]
    if failure:
        assert report['status'] == 'failed' and report['physics_admitted'] is False
        if failure != 'unpromoted-output':
            assert destination.read_text() == previous
        if failure in ('physics-production', 'mass-production-gpu'):
            assert 'physics-promote' not in labels
    else:
        assert report['status'] == 'passed' and report['physics_admitted'] is True
        assert json.loads(destination.read_text())['measurements'][0]['status'] == 'production'
        assert labels == ['environment', 'physics-measure', 'mass-measure-cpu', 'mass-measure-gpu',
                          'physics-record', 'physics-production', 'mass-production-cpu',
                          'mass-production-gpu', 'physics-promote']
        promote = calls[-1][1]
        assert promote[promote.index('--evidence-root') + 1] == str(output / 'physics-measure')
        assert promote[promote.index('--production-evidence-root') + 1] == str(output / 'physics-production')
        for flag, prefix in (('--live-mass-evidence', 'mass-measure-'),
                             ('--production-live-mass-evidence', 'mass-production-')):
            index = promote.index(flag)
            assert promote[index + 1:index + 3] == [str(output / (prefix + device)) for device in ('cpu', 'gpu')]
        assert '--measure-unadmitted' in calls[1][1]
        for label, command, env in calls:
            if label in ('physics-production', 'mass-production-cpu', 'mass-production-gpu'):
                assert '--measure-unadmitted' not in command
                assert env['OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION'] == str(output / 'physics-candidate.json')
