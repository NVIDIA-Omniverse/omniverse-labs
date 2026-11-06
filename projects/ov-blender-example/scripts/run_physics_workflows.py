"""Serial, production-admitted Windows physics workflow qualification.

Each cell runs in a fresh stock Blender profile. Offline cells remove the SDK
selection entirely. Existing fixture assertions remain the acceptance oracle.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import time


CASES = {
    'viewport_sync': ('../scripts/run_physics_viewport_sync_probe.py', False, True),
    'generation_workflows': ('blender_physics_generation_workflows.py', False, False),
    'kinematic_scheduler': ('blender_kinematic_scheduler.py', False, False),
    'kinematic_scheduler_extended': ('blender_kinematic_scheduler.py', False, False),
    'kinematic_capture': ('blender_kinematic_capture.py', False, False),
    'bone_presentation': ('blender_bone_presentation.py', True, False),
    'bone_presentation_inherit_none': ('blender_bone_presentation.py', True, False),
    'bake_capture': ('blender_bake_capture.py', True, True),
    'bake_capture_lifecycle': ('blender_bake_capture_lifecycle.py', True, True),
    'bake_capture_concurrent': ('blender_bake_capture_concurrent.py', True, True),
    'bake_actions_native': ('blender_bake_actions_native.py', True, False),
    'baked_playback_gate': ('blender_baked_playback_gate.py', True, True),
    'bake_ui': ('blender_bake_ui.py', True, True),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


VIEWPORT_LABELS = frozenset(('initial', 'settled', 'reentered_camera', 'rewound', 'orbited', 'replayed'))
ARTIFACT_SUFFIXES = frozenset(('.json', '.jsonl', '.blend', '.npy', '.npz', '.bin', '.raw',
                              '.rgba', '.png', '.exr', '.usd', '.usda', '.usdc'))


def artifact_hashes(target):
    return {str(path.relative_to(target)): digest(path) for path in target.rglob('*')
            if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES}


def validate_viewport_result(result):
    """Preserve the original test_physics_viewport_sync outer acceptance oracle.

    pytest.approx(abs=1e-5) retains its default relative tolerance of 1e-6.
    The six labels must be unique; dictionary construction must not hide a
    duplicated/missing case. Extra identity/matrix evidence is not a substitute
    for these original numeric assertions.
    """
    assert result['status'] == 'passed', result
    rows = result['checks']
    assert len(rows) == len(VIEWPORT_LABELS), rows
    checks = {row['label']: row for row in rows}
    assert set(checks) == VIEWPORT_LABELS, checks
    for row in rows:
        assert math.isfinite(row['blender_z']), row
        assert math.isfinite(row['max_pixel_error']) and row['max_pixel_error'] < 3, row
    assert checks['rewound']['timeline_reset'], checks['rewound']
    assert checks['rewound']['simulation_time_ns'] == 0, checks['rewound']
    assert checks['orbited']['simulation_time_ns'] == 0, checks['orbited']
    assert checks['reentered_camera']['simulation_time_ns'] == checks['settled']['simulation_time_ns']
    expected_z = checks['settled']['blender_z']
    for label in ('reentered_camera', 'replayed'):
        # Match pytest.approx's asymmetric expected-value relative tolerance.
        assert abs(checks[label]['blender_z'] - expected_z) <= max(1e-5, 1e-6 * abs(expected_z)), checks[label]

    # These fields are emitted from actual Blender module enumeration and
    # retained RuntimeSession diagnostics by the public viewport fixture.
    assert result['vendor_sdk_imports'] == [], result['vendor_sdk_imports']
    assert result['cleanup_errors'] == [], result['cleanup_errors']
    runtimes = result['runtimes_after_shutdown']
    assert {'renderer', 'physics'} <= {row['kind'] for row in runtimes}, runtimes
    for runtime in runtimes:
        assert runtime['closed'] is True and runtime['returncode'] == 0, runtime
        assert runtime['cleanup_errors'] == [] and runtime['has_scene'] is False, runtime
    resources = result['resources_after_shutdown']
    assert resources, 'Missing observed shared-frame resource cleanup'
    for stream, state in resources.items():
        assert state and all(value is False for value in state.values()), (stream, state)
    return checks


def verify_viewport_artifacts(result, target):
    """Bind each accepted pixel case to its retained raw array, inside output."""
    target = target.resolve()
    verified = {}
    raw_paths = set()
    for row in result['checks']:
        relative = Path(row['rgba_path'])
        assert not relative.is_absolute(), relative
        path = (target / relative).resolve()
        path.relative_to(target)
        assert path.suffix.lower() == '.npy' and path.is_file(), path
        actual = digest(path)
        assert actual == row['rgba_sha256'], (row['label'], actual, row['rgba_sha256'])
        raw_paths.add(path)
        verified[str(relative)] = actual
        if row.get('colors_path'):
            colors = Path(row['colors_path'])
            assert not colors.is_absolute(), colors
            colors = (target / colors).resolve()
            colors.relative_to(target)
            assert colors.suffix.lower() == '.npy' and colors.is_file(), colors
            verified[str(colors.relative_to(target))] = digest(colors)
    assert len(raw_paths) == len(VIEWPORT_LABELS), 'Cases must retain distinct raw frame files'
    return verified


def command_for(blender, source, case, output, blend, device='cpu'):
    filename, needs_blend, cpu_implicit = CASES[case]
    if device == 'gpu' and not case.startswith('kinematic_'):
        raise ValueError('This mixed-joint workflow fixture requires CPU; retain admitted GPU restrictions')
    command = [str(blender), *([] if case == 'bake_ui' else ['--background']), '--factory-startup', '--threads', '2',
               '--python-exit-code', '17', '--python', str(source / 'tests' / filename),
               '--', '--public-sdk', '--output', str(output)]
    if not cpu_implicit:
        command += ['--device', device]
    if needs_blend:
        if blend is None:
            raise ValueError(f'{case} requires --blend from the admitted mixed Object/bone fixture')
        command += ['--blend', str(blend)]
    if case.endswith('_extended'):
        command.append('--extended')
    if case.endswith('_inherit_none'):
        command.append('--inherit-none')
    return command


def environment(repo, output, *, offline=False):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(('OV_BLENDER', 'OVRTX_', 'OV_OUTPUT_', 'OVPHYSX_', 'OVSTAGE_',
                           'OVSTREAM_', 'OMNI_')) or key in (
                'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PXR_PLUGINPATH_NAME'):
            env.pop(key)
    for key in ('BLENDER_USER_CONFIG', 'BLENDER_USER_SCRIPTS', 'BLENDER_USER_DATAFILES', 'TEMP', 'TMP'):
        directory = output / 'environment' / key
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0')
    if not offline:
        env.update(OV_BLENDER_PUBLIC_SDK_PYTHON=str(repo / '_build/blender-ovrtx-ovstream/runtime-lock/env/Scripts/python.exe'),
                   OV_BLENDER_PUBLIC_SDK_OUTPUT=str(output / 'workers'),
                   OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR=str(output / 'generations'))
    return env


def verify_admission(path, manifest, device='cpu'):
    row = json.loads(path.read_text())
    assert row['measurement_only'] is False and not row['local_manifest_override'], row
    assert row['device'] == device and row['admission_checks'], row
    for check in row['admission_checks']:
        assert check['status'] == 'production' and check['local_override'] is False, check
        assert Path(check['manifest_path']).resolve() == manifest.resolve(), check
    return row


def run_process(command, env, output, timeout):
    from ovrtx_blender_example.public_sdk_install import _WindowsJob
    job = _WindowsJob()
    row = dict(command=command, environment={k: v for k, v in env.items() if os.environ.get(k) != v})
    started = time.monotonic()
    try:
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                job.assign(process)
            except BaseException:
                process.kill()
                process.wait(timeout=15)
                raise
            row['blender_pid'] = process.pid
            (output / 'process.json').write_text(json.dumps(row, indent=2) + '\n')
            try:
                row['exit_code'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row['watchdog_timeout'] = timeout
                job.terminate()
                row['exit_code'] = process.wait(timeout=15)
        row['seconds'] = time.monotonic() - started
        deadline = time.monotonic() + 15
        while True:
            row['job_active_after_exit'] = job_active_count(job)
            if not row['job_active_after_exit'] or time.monotonic() >= deadline:
                break
            time.sleep(.1)
        return row
    finally:
        job.close()
        (output / 'process.json').write_text(json.dumps(row, indent=2) + '\n')


def job_active_count(job):
    import ctypes as c
    from ctypes import wintypes as w
    class Accounting(c.Structure):
        _fields_ = [('user', c.c_int64), ('kernel', c.c_int64),
                    ('period_user', c.c_int64), ('period_kernel', c.c_int64),
                    ('faults', w.DWORD), ('total', w.DWORD),
                    ('active', w.DWORD), ('terminated', w.DWORD)]
    query = job.api.QueryInformationJobObject
    query.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p]
    query.restype = w.BOOL
    accounting = Accounting()
    if not query(job.handle, 1, c.byref(accounting), c.sizeof(accounting), None):
        raise c.WinError(c.get_last_error())
    return accounting.active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--blend', type=Path)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    assert sys.platform == 'win32'
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'addon'))
    blender = repo / '_build/lts-validation-5.2.1/install/blender-5.2.1-windows-x64/blender.exe'
    manifest = source / 'addon/ovrtx_blender_example/public_sdk_physics_admission.json'
    frozen = [*source.rglob('*.py'), source / 'runtime/public-sdk-lock.json', manifest, blender,
              repo / '_build/blender-ovrtx-ovstream/runtime-lock/env/Scripts/python.exe']
    if args.blend:
        args.blend = args.blend.resolve()
        frozen.append(args.blend)
    before = {str(path): digest(path) for path in frozen}
    commands = {case: command_for(blender, source, case, output / case, args.blend, args.device) for case in args.cases}
    receipt = dict(status='prepared', commands=commands, source_before=before,
                   host=dict(platform=platform.platform(), orchestrator_python=sys.version,
                             baseline_environment_receipt='docs/graphs/blender-ovrtx-ovstream/preflight.md'),
                   requested_device=args.device,
                   measurement_only=False, local_manifest_override=None, cells=[])
    destination = output / 'suite.json'
    def persist():
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
    persist()
    if args.preflight_only:
        receipt['status'] = 'preflight-only; no Blender or native process launched'
        persist()
        return 0
    try:
        for case, command in commands.items():
            target = output / case
            target.mkdir()
            print('START', case, flush=True)
            row = run_process(command, environment(repo, target), target, args.timeout)
            row['case'] = case
            # Failed cells retain the same raw artifact inventory as successes.
            row['artifacts'] = artifact_hashes(target)
            receipt['cells'].append(row)
            persist()
            assert row['exit_code'] == 0 and 'watchdog_timeout' not in row, row
            assert row['job_active_after_exit'] == 0, row
            row['admission'] = verify_admission(target / 'runtime-identity.json', manifest, args.device)
            result_path = target / ('result.json' if case == 'viewport_sync' else 'results.json')
            row['results'] = json.loads(result_path.read_text())
            if case == 'viewport_sync':
                validate_viewport_result(row['results'])
                row['verified_raw_frames'] = verify_viewport_artifacts(row['results'], target)
            if case == 'bake_ui':
                assert row['results']['passed'] is True, row['results']
            if case == 'bake_actions_native':
                offline = target / 'offline'
                offline.mkdir()
                offline_command = [str(blender), '--background', '--factory-startup', '--threads', '2',
                                   '--python-exit-code', '17', '--python', str(source / 'tests/blender_bake_offline.py'),
                                   '--', '--directory', str(target)]
                row['offline'] = run_process(offline_command, environment(repo, offline, offline=True), offline, 180)
                assert row['offline']['exit_code'] == 0 and 'watchdog_timeout' not in row['offline']
                assert row['offline']['job_active_after_exit'] == 0
            row['artifacts'] = artifact_hashes(target)
            row['status'] = 'passed'
            persist()
            print('PASS', case, flush=True)
        receipt['status'] = 'passed'
    except BaseException:
        receipt['status'] = 'failed'
        raise
    finally:
        receipt['source_changed'] = [path for path, value in before.items() if digest(Path(path)) != value]
        if receipt['source_changed']:
            receipt['status'] = 'failed: source changed'
        persist()
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
