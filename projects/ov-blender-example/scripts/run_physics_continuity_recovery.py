# SPDX-License-Identifier: Apache-2.0
"""Scene-owned physics continuity during actual Windows renderer recovery, with source and owned-process receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from validate_render_outputs import process_running
from run_physics_workflows import job_active_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=420)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--mode', choices=('paused', 'live', 'live-control'), default='paused')
    args = parser.parse_args()
    assert sys.platform == 'win32'
    repo, out = args.repo.resolve(), args.output.resolve()
    out.relative_to(repo)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'addon'))
    from ovrtx_blender_example.public_sdk_install import _WindowsJob
    from run_ovstream_raw_frame_smoke import resource_state
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(('OV_BLENDER', 'OVRTX_', 'OV_OUTPUT_')) or key.startswith(('OVPHYSX_', 'OVSTAGE_', 'OVSTREAM_', 'OMNI_')) or key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PXR_PLUGINPATH_NAME'):
            env.pop(key)
    for key in ('BLENDER_USER_CONFIG', 'BLENDER_USER_SCRIPTS', 'BLENDER_USER_DATAFILES', 'TEMP', 'TMP'):
        directory = out / 'environment' / key
        directory.mkdir(parents=True)
        env[key] = str(directory)
    env.update(OV_BLENDER_PUBLIC_SDK_PYTHON=str(repo / '_build/blender-ovrtx-ovstream/runtime-lock/env/Scripts/python.exe'),
               OV_BLENDER_PUBLIC_SDK_OUTPUT=str(out),
               OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR=str(out / 'generations'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0')
    blender = repo / '_build/lts-validation-5.2.1/install/blender-5.2.1-windows-x64/blender.exe'
    env['OV_BLENDER_PHYSICS_LIVE_CONTROL'] = '1' if args.mode == 'live-control' else '0'
    fixture = source / 'tests' / ('blender_physics_live_recovery.py' if args.mode in {'live', 'live-control'}
                                else 'blender_physics_continuity_recovery.py')
    command = [str(blender), '--factory-startup', '--threads', '2', '--python-exit-code', '17', '--python', str(fixture)]
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*.py') if '__pycache__' not in p.parts}
    frozen_paths = [source / 'runtime/public-sdk-lock.json',
                    source / 'addon/ovrtx_blender_example/public_sdk_physics_admission.json',
                    blender, Path(env['OV_BLENDER_PUBLIC_SDK_PYTHON'])]
    frozen_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    receipt = dict(command=command, source_before=hashes, environment_overrides={k: v for k, v in env.items() if os.environ.get(k) != v})
    receipt['frozen_dependencies'] = frozen_hashes
    path = out / 'receipt.json'
    assert all(p.is_file() for p in frozen_paths)
    assert not env.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION')
    receipt['measurement_only'] = False
    receipt['local_manifest_override'] = None
    for target in (fixture, source / 'tests/physics_continuity_checks.py', Path(__file__)):
        compile(target.read_text(encoding='utf-8'), str(target), 'exec')
    if args.preflight_only:
        receipt['status'] = 'preflight-only; no native process launched'
        path.write_text(json.dumps(receipt, indent=2) + '\n')
        return 0
    started = time.monotonic()
    job = _WindowsJob()
    try:
        with (out / 'blender.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                job.assign(process)
            except Exception:
                process.kill()
                process.wait(timeout=15)
                raise
            receipt['blender_pid'] = process.pid
            path.write_text(json.dumps(receipt, indent=2) + '\n')
            try:
                receipt['exit_code'] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                receipt['watchdog_timeout'] = args.timeout
                job.terminate()
                receipt['exit_code'] = process.wait(timeout=15)
        receipt['elapsed_seconds'] = time.monotonic() - started
        result_path = out / 'recovery/result.json'
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        receipt['result'] = result
        receipt['source_changed'] = [key for key, value in hashes.items()
                                     if hashlib.sha256((source / key).read_bytes()).hexdigest() != value]
        receipt['dependencies_changed'] = [key for key, value in frozen_hashes.items()
                                           if hashlib.sha256(Path(key).read_bytes()).hexdigest() != value]
        pids = set()
        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {'pid', 'runtime_pid', 'launcher_pid', 'producer_pid'} and type(item) is int and item > 0:
                        pids.add(item)
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        collect(result)
        trace_path = out / 'recovery/physics-operations.jsonl'
        if trace_path.is_file():
            for line in trace_path.read_text(encoding='utf-8').splitlines():
                collect(json.loads(line))
        deadline = time.monotonic() + 15
        while True:
            alive = [pid for pid in pids if process_running(pid)]
            if not alive or time.monotonic() >= deadline:
                break
            time.sleep(.1)
        receipt['observed_pids'] = sorted(pids)
        receipt['alive_after_exit'] = alive
        receipt['resources_after_exit'] = {stream: resource_state(stream) for stream in result.get('streams', [])}
        deadline = time.monotonic() + 15
        while True:
            active = job_active_count(job)
            if active == 0 or time.monotonic() >= deadline:
                break
            time.sleep(.1)
        receipt['job_active_before_close'] = active
        receipt['artifacts'] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (out / 'recovery').rglob('*') if p.is_file()}
        path.write_text(json.dumps(receipt, indent=2) + '\n')
        expected_status = 'diagnostic-complete' if args.mode == 'live-control' else 'passed'
        passed = (receipt['exit_code'] == 0 and result.get('status') == expected_status and active == 0
                  and not receipt['source_changed'] and not alive
                  and not receipt['dependencies_changed']
                  and not any(any(state.values()) for state in receipt['resources_after_exit'].values()))
        return 0 if passed else 1
    finally:
        job.close()


if __name__ == '__main__':
    raise SystemExit(main())
