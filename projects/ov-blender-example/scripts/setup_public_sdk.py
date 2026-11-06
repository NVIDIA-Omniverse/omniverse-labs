#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create or verify a separate public SDK environment; never modify Blender Python."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import uuid

from public_sdk_environment import PUBLIC, DEFAULT_CONFIG, check_blender, resolve_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', required=True, help='Blender 5.2 or newer executable')
    parser.add_argument('--sdk-python', help='Use an existing SDK environment; do not install packages')
    parser.add_argument('--environment', type=Path, default=PUBLIC.parent / 'out/public-sdk-venv')
    parser.add_argument('--output', type=Path, default=PUBLIC.parent / 'out/public-sdk')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    blender = check_blender(args.blender)
    if args.sdk_python:
        python = resolve_executable(args.sdk_python)
    else:
        if sys.version_info[:2] != (3, 13):
            parser.error('Run with 64-bit Python 3.13, e.g. py -3.13 setup_public_sdk.py ...')
        environment = args.environment.expanduser().absolute() / ('env-' + uuid.uuid4().hex)
        environment.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(environment.parent).free < 8 * 1024**3:
            parser.error('At least 8 GiB free space is required for SDK installation and extraction')
        if environment.exists() and not (environment / 'pyvenv.cfg').is_file():
            parser.error('Environment directory already exists and is not a virtual environment')
        temporary_root = environment.parent / 'tmp'
        temporary_root.mkdir(parents=True, exist_ok=True)
        for name in ('TEMP', 'TMP', 'TMPDIR'):
            os.environ[name] = str(temporary_root)
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        try:
            subprocess.run([str(python), '-I', str(PUBLIC / 'runtime/materialize_sdk.py'),
                            'install', '--cache', str(environment.parent / 'cache')], check=True)
        except BaseException:
            shutil.rmtree(environment)
            raise
    probe = subprocess.run([str(python), str(PUBLIC / 'scripts/public_sdk_environment.py')],
                           capture_output=True, text=True, timeout=180)
    if probe.returncode:
        print(probe.stdout, end=''); print(probe.stderr, file=sys.stderr, end='')
        raise SystemExit('Public SDK preflight failed; configuration was not replaced')
    output = args.output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'sdk-preflight.log').write_text(probe.stdout + probe.stderr)
    config = {'schema': 1, 'python': str(python), 'blender': str(blender),
              'output': str(output), 'gpu': args.gpu,
              'physics_admission': str(output / 'physics-admission.json')}
    args.config.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.config.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, indent=2) + '\n'); temporary.replace(args.config)
    print(f'SDK environment ready. Local configuration: {args.config.absolute()}')
    print('Next: launch_public_sdk_blender.py, or validate_public_sdk_host.py --physics')
    print('Imports passed; native GPU rendering and Windows physics admission need the host validation.')


if __name__ == '__main__':
    main()
