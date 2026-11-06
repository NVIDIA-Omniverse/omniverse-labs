#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run native public-SDK gates on the target host, serially with bounded outputs.

--physics measures the platform's exact binaries and enables expanded physics
only after measurement AND production acceptance. No Linux evidence is reused
as proof for Windows DLLs. Logs remain local under the requested output folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from public_sdk_environment import PUBLIC, DEFAULT_CONFIG, check_blender, read_config, worker_environment, resolve_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--physics', action='store_true', help='Measure and admit CPU/GPU physics on this host')
    parser.add_argument('--physics-only', action='store_true', help='Run only the physics acceptance sequence')
    args = parser.parse_args()
    config = read_config(args.config)
    consumer = None
    if not args.physics_only:
        if not config.get('consumer_python'):
            parser.error('Run setup_public_sdk_consumer.py --environment <new-directory> --config ' + str(args.config))
        consumer = str(resolve_executable(config['consumer_python']))
    blender = str(check_blender(config['blender']))
    root = (args.output or Path(config['output']) / ('validation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).absolute()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh output directory to preserve prior measurements')
    root.mkdir(parents=True, exist_ok=True)
    env = worker_environment(config, root / 'workers')
    python = env['OV_BLENDER_PUBLIC_SDK_PYTHON']
    report = {'status': 'running', 'steps': [], 'physics_admitted': False,
              'viewport_ui_validated': False, 'output': str(root)}
    def run(label, command, timeout=900):
        print(f'START {label}', flush=True)
        started = time.monotonic()
        with (root / (label + '.log')).open('w') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout)
        report['steps'].append({'name': label, 'exit_code': process.returncode,
                                'seconds': time.monotonic() - started})
        (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        if process.returncode:
            raise RuntimeError(f'{label} failed; inspect {root / (label + ".log")}')
        print(f'PASS {label}', flush=True)
    try:
        run('environment', [python, str(PUBLIC / 'scripts/public_sdk_environment.py')], 180)
        if not args.physics_only:
            run('consumer-environment', [consumer, '-I', str(PUBLIC / 'scripts/setup_public_sdk_consumer.py'), '--verify'], 180)
            run('first-frame', [consumer, str(PUBLIC / 'scripts/run_public_sdk_first_frame_smoke.py'),
                '--python', python, '--gpu', str(config.get('gpu', 0)), '--output', str(root / 'first-frame')])
            run('raw-shm', [python, str(PUBLIC / 'scripts/run_ovstream_raw_frame_smoke.py'),
                '--device', f'cuda:{config.get("gpu", 0)}', '--output', str(root / 'raw-shm.json')])
            run('async-render', [consumer, str(PUBLIC / 'scripts/run_public_sdk_async_smoke.py'),
                '--python', python, '--gpu', str(config.get('gpu', 0)), '--output', str(root / 'async')])
            metadata = root / 'stage-metadata.json'
            metadata.write_text(json.dumps({'upAxis': 'Y', 'metersPerUnit': 1, 'defaultPrim': 'World'}))
            run('renderer-recovery', [consumer, str(PUBLIC / 'scripts/run_public_sdk_recovery_smoke.py'),
                '--python', python, '--gpu', str(config.get('gpu', 0)),
                '--scene', str(PUBLIC / 'tests/fixtures/public_sdk_render.usda'),
                '--stage-metadata', str(metadata), '--render-var', 'HdrColor',
                '--output', str(root / 'recovery')])
            run('blender-animation', [blender, '--background', '--factory-startup',
                '--python-exit-code', '17', '--python', str(PUBLIC / 'scripts/probe_public_sdk_blender.py'),
                '--', '--output', str(root / 'animation'), '--frames', '3', '--animation'])
        if args.physics or args.physics_only:
            suite = [python, str(PUBLIC / 'scripts/validate_public_sdk_physics.py'), '--blender', blender]
            run('physics-measure', suite + ['--output', str(root / 'physics-measure'), '--measure-unadmitted'], 12000)
            for device in ('cpu', 'gpu'):
                run('mass-measure-' + device, [blender, '--background', '--factory-startup',
                    '--python-exit-code', '17', '--python', str(PUBLIC / 'tests/blender_public_physics_generation_reuse.py'),
                    '--', '--output', str(root / ('mass-measure-' + device)), '--device', device,
                    '--measure-unadmitted'])
            candidate = root / 'physics-candidate.json'
            run('physics-record', [python, str(PUBLIC / 'scripts/record_public_sdk_physics_admission.py'),
                '--evidence-root', str(root / 'physics-measure'), '--destination', str(candidate),
                '--live-mass-evidence', str(root / 'mass-measure-cpu'), str(root / 'mass-measure-gpu')])
            env['OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION'] = str(candidate)
            run('physics-production', suite + ['--output', str(root / 'physics-production')], 12000)
            for device in ('cpu', 'gpu'):
                run('mass-production-' + device, [blender, '--background', '--factory-startup',
                    '--python-exit-code', '17', '--python', str(PUBLIC / 'tests/blender_public_physics_generation_reuse.py'),
                    '--', '--output', str(root / ('mass-production-' + device)), '--device', device])
            destination = Path(config['physics_admission'])
            # Candidate selection bootstraps qualification only. Promote through
            # the recorder, which verifies both normally gated evidence sets and
            # atomically retains the prior destination if any proof is missing.
            run('physics-promote', [python, str(PUBLIC / 'scripts/record_public_sdk_physics_admission.py'),
                '--evidence-root', str(root / 'physics-measure'), '--destination', str(destination),
                '--live-mass-evidence', str(root / 'mass-measure-cpu'), str(root / 'mass-measure-gpu'),
                '--production-evidence-root', str(root / 'physics-production'),
                '--production-live-mass-evidence', str(root / 'mass-production-cpu'), str(root / 'mass-production-gpu')])
            measured = json.loads(candidate.read_text())['measurements']
            promoted = json.loads(destination.read_text())['measurements']
            if not measured or any(not any(row.get('identity') == proof.get('identity')
                    and row.get('status') == 'production' for row in promoted) for proof in measured):
                raise RuntimeError('Physics recorder did not produce production admission for the measured identity')
            env['OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION'] = str(destination)
            report.update(physics_admitted=True, physics_admission=str(destination))
        report['status'] = 'passed'
        print('PUBLIC_SDK_HOST_VALIDATION_PASS', root, flush=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
