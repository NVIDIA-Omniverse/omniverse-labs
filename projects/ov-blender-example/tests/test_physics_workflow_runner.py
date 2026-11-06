"""Workflow orchestration rejects unqualified admission and builds valid CLI."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_physics_workflows.py'
spec = importlib.util.spec_from_file_location('workflow_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('case', runner.CASES)
def test_commands_require_production_and_correct_device(case, tmp_path):
    command = runner.command_for(Path('blender.exe'), tmp_path, case, tmp_path / 'out', tmp_path / 'mixed.blend')
    args = command[command.index('--') + 1:]
    assert '--public-sdk' in args and '--measure-unadmitted' not in args
    assert args[args.index('--output') + 1] == str(tmp_path / 'out')
    _, needs_blend, cpu_implicit = runner.CASES[case]
    assert ('--blend' in args) == needs_blend
    assert ('--device' in args) != cpu_implicit
    if not cpu_implicit:
        assert args[args.index('--device') + 1] == 'cpu'


def test_blend_required_before_launch(tmp_path):
    with pytest.raises(ValueError, match='requires --blend'):
        runner.command_for(Path('blender.exe'), tmp_path, 'bake_capture', tmp_path, None)


def test_gpu_kinematic_selection_preserves_mixed_joint_restrictions(tmp_path):
    command = runner.command_for(Path('blender.exe'), tmp_path, 'kinematic_capture', tmp_path, None, 'gpu')
    assert command[command.index('--device') + 1] == 'gpu'
    with pytest.raises(ValueError, match='requires CPU'):
        runner.command_for(Path('blender.exe'), tmp_path, 'bone_presentation', tmp_path, tmp_path / 'x.blend', 'gpu')


def test_modal_bake_uses_real_window_and_background_cases_do_not(tmp_path):
    command = runner.command_for(Path('blender.exe'), tmp_path, 'bake_ui', tmp_path, tmp_path / 'x.blend')
    assert '--background' not in command
    command = runner.command_for(Path('blender.exe'), tmp_path, 'bake_capture', tmp_path, tmp_path / 'x.blend')
    assert '--background' in command


@pytest.mark.parametrize('offline', [False, True])
def test_environment_clears_policy_and_loader_overrides(monkeypatch, tmp_path, offline):
    for key in ('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', 'OVPHYSX_DISABLE_GPU',
                'OVPHYSX_LIBRARY_PATH', 'PYTHONPATH', 'BLENDER_USER_SCRIPTS'):
        monkeypatch.setenv(key, 'foreign')
    env = runner.environment(tmp_path, tmp_path / 'profile', offline=offline)
    assert 'OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION' not in env
    assert 'OVPHYSX_DISABLE_GPU' not in env and 'OVPHYSX_LIBRARY_PATH' not in env
    assert 'PYTHONPATH' not in env
    assert ('OV_BLENDER_PUBLIC_SDK_PYTHON' in env) != offline
    assert Path(env['BLENDER_USER_SCRIPTS']).is_dir()


@pytest.mark.parametrize('fault', ['measurement', 'override', 'candidate', 'wrong_manifest', 'empty'])
def test_admission_rejects_incomplete_or_nondefault_proof(tmp_path, fault):
    manifest = tmp_path / 'shipped.json'
    row = dict(measurement_only=False, local_manifest_override=None, device='cpu',
               admission_checks=[dict(status='production', local_override=False, manifest_path=str(manifest))])
    if fault == 'measurement': row['measurement_only'] = True
    elif fault == 'override': row['local_manifest_override'] = str(tmp_path / 'candidate.json')
    elif fault == 'candidate': row['admission_checks'][0]['status'] = 'candidate'
    elif fault == 'wrong_manifest': row['admission_checks'][0]['manifest_path'] = str(tmp_path / 'other.json')
    else: row['admission_checks'] = []
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        runner.verify_admission(path, manifest)


def test_default_production_proof(tmp_path):
    manifest = tmp_path / 'shipped.json'
    row = dict(measurement_only=False, local_manifest_override=None, device='cpu',
               admission_checks=[dict(status='production', local_override=False, manifest_path=str(manifest))])
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps(row))
    assert runner.verify_admission(path, manifest) == row


def viewport_result(directory):
    directory.mkdir(parents=True, exist_ok=True)
    checks = []
    for label in ('initial', 'settled', 'reentered_camera', 'rewound', 'orbited', 'replayed'):
        raw = directory / (label + '.npy')
        raw.write_bytes(('retained raw bytes: ' + label).encode())
        checks.append(dict(label=label, blender_z=.25, simulation_time_ns=(0 if label in
            ('initial', 'rewound', 'orbited') else 2_000_000_000),
            timeline_reset=label == 'rewound', max_pixel_error=2.999,
            rgba_path=raw.name, rgba_sha256=runner.digest(raw)))
    return dict(status='passed', checks=checks, vendor_sdk_imports=[], cleanup_errors=[],
        runtimes_after_shutdown=[dict(kind=kind, closed=True, returncode=0,
                                     cleanup_errors=[], has_scene=False) for kind in ('renderer', 'physics')],
        resources_after_shutdown={'owned-stream': {'mapping': False, 'wake': False, 'pipe': False}})


def test_viewport_accepts_original_outer_limits_and_hashes(tmp_path):
    result = viewport_result(tmp_path)
    result['checks'][2]['blender_z'] += 0.999e-5
    result['checks'][5]['blender_z'] -= 0.999e-5
    assert set(runner.validate_viewport_result(result)) == runner.VIEWPORT_LABELS
    assert len(runner.verify_viewport_artifacts(result, tmp_path)) == 6


@pytest.mark.parametrize('fault', ['status', 'missing', 'duplicate', 'extra', 'rewind_reset',
    'rewind_time', 'orbit_time', 'reentry_time', 'reentry_pose', 'replay_pose', 'pixel_boundary',
    'nan_pixels', 'nan_pose', 'sdk_import', 'missing_sdk_proof', 'cleanup_error',
    'open_runtime', 'bad_exit', 'retained_scene', 'runtime_cleanup_error', 'missing_physics',
    'missing_resources', 'empty_resource_state', 'live_mapping'])
def test_viewport_rejects_false_pass_and_missing_evidence(tmp_path, fault):
    result = viewport_result(tmp_path)
    checks = {row['label']: row for row in result['checks']}
    if fault == 'status': result['status'] = 'failed'
    elif fault == 'missing': result['checks'].pop()
    elif fault == 'duplicate': result['checks'][-1] = deepcopy(result['checks'][0])
    elif fault == 'extra': result['checks'].append(deepcopy(result['checks'][0]))
    elif fault == 'rewind_reset': checks['rewound']['timeline_reset'] = False
    elif fault == 'rewind_time': checks['rewound']['simulation_time_ns'] = 1
    elif fault == 'orbit_time': checks['orbited']['simulation_time_ns'] = 1
    elif fault == 'reentry_time': checks['reentered_camera']['simulation_time_ns'] += 1
    elif fault == 'reentry_pose': checks['reentered_camera']['blender_z'] += 1.001e-5
    elif fault == 'replay_pose': checks['replayed']['blender_z'] += 1.001e-5
    elif fault == 'pixel_boundary': checks['initial']['max_pixel_error'] = 3
    elif fault == 'nan_pixels': checks['initial']['max_pixel_error'] = float('nan')
    elif fault == 'nan_pose': checks['settled']['blender_z'] = float('nan')
    elif fault == 'sdk_import': result['vendor_sdk_imports'] = ['ovphysx']
    elif fault == 'missing_sdk_proof': result.pop('vendor_sdk_imports')
    elif fault == 'cleanup_error': result['cleanup_errors'] = ['close failed']
    elif fault == 'open_runtime': result['runtimes_after_shutdown'][0]['closed'] = False
    elif fault == 'bad_exit': result['runtimes_after_shutdown'][0]['returncode'] = 17
    elif fault == 'retained_scene': result['runtimes_after_shutdown'][0]['has_scene'] = True
    elif fault == 'runtime_cleanup_error': result['runtimes_after_shutdown'][0]['cleanup_errors'] = ['failed']
    elif fault == 'missing_physics': result['runtimes_after_shutdown'].pop()
    elif fault == 'missing_resources': result['resources_after_shutdown'] = {}
    elif fault == 'empty_resource_state': result['resources_after_shutdown']['owned-stream'] = {}
    else: result['resources_after_shutdown']['owned-stream']['mapping'] = True
    with pytest.raises((AssertionError, KeyError)):
        runner.validate_viewport_result(result)


@pytest.mark.parametrize('fault', ['tamper', 'missing_file', 'escape', 'absolute', 'duplicate_raw'])
def test_viewport_raw_evidence_rejects_mismatch_or_ambiguous_ownership(tmp_path, fault):
    result = viewport_result(tmp_path)
    row = result['checks'][0]
    if fault == 'tamper': (tmp_path / row['rgba_path']).write_bytes(b'changed')
    elif fault == 'missing_file': (tmp_path / row['rgba_path']).unlink()
    elif fault == 'escape': row['rgba_path'] = '../outside.npy'
    elif fault == 'absolute': row['rgba_path'] = str((tmp_path / row['rgba_path']).resolve())
    else:
        result['checks'][-1]['rgba_path'] = row['rgba_path']
        result['checks'][-1]['rgba_sha256'] = row['rgba_sha256']
    with pytest.raises((AssertionError, ValueError)):
        runner.verify_viewport_artifacts(result, tmp_path)


def test_raw_and_preview_outputs_are_in_artifact_inventory(tmp_path):
    for suffix in ('.npy', '.npz', '.raw', '.bin', '.rgba', '.png', '.exr', '.json', '.jsonl', '.blend', '.usda'):
        (tmp_path / ('output' + suffix)).write_bytes(suffix.encode())
    (tmp_path / 'run.log').write_text('diagnostic log')
    hashes = runner.artifact_hashes(tmp_path)
    assert len(hashes) == 11 and 'run.log' not in hashes
    assert all(value == runner.digest(tmp_path / name) for name, value in hashes.items())


def test_runner_rejects_viewport_false_success_and_retains_raw_hashes(tmp_path, monkeypatch):
    # This is the Windows runner's pure mocked control-flow test. Process
    # execution is replaced below; allow it to run on either test host.
    monkeypatch.setattr(runner.sys, 'platform', 'win32')
    for relative in ('_build/lts-validation-5.2.1/install/blender-5.2.1-windows-x64/blender.exe',
                     '_build/blender-ovrtx-ovstream/runtime-lock/env/Scripts/python.exe'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'no process launched')
    output = tmp_path / 'evidence'
    def fake_process(command, env, target, timeout):
        result = viewport_result(target)
        result['checks'][-1]['blender_z'] += .001
        (target / 'result.json').write_text(json.dumps(result))
        return {'exit_code': 0, 'job_active_after_exit': 0}
    monkeypatch.setattr(runner, 'run_process', fake_process)
    monkeypatch.setattr(runner, 'verify_admission', lambda *args: {'independent': 'tested separately'})
    monkeypatch.setattr(runner.sys, 'argv', [str(SCRIPT), '--repo', str(tmp_path), '--output', str(output),
                                          '--cases', 'viewport_sync'])
    with pytest.raises(AssertionError):
        runner.main()
    receipt = json.loads((output / 'suite.json').read_text())
    assert receipt['status'] == 'failed'
    assert len([name for name in receipt['cells'][0]['artifacts'] if name.endswith('.npy')]) == 6
    assert 'status' not in receipt['cells'][0]  # Never records a passed cell.
