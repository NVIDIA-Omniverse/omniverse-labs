#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: taken from the consumer/test environment (scripts/requirements-public-sdk-consumer.txt), never the shipped SDK runtime.
"""Create/verify the SDK-free Windows diagnostic consumer, separate from SDK Python."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import venv

REQUIREMENTS = Path(__file__).with_name('requirements-public-sdk-consumer.txt')
VERSIONS = {'usd-core': '25.11', 'numpy': '2.3.3', 'pillow': '11.3.0'}
VENDORS = ('ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp')


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def inspect():
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 13) or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('This consumer lock requires Windows x64 CPython 3.13')
    if any(importlib.util.find_spec(name) is not None for name in VENDORS):
        raise RuntimeError('Diagnostic consumer must not contain NVIDIA SDK packages')
    files = {}
    for name, expected in VERSIONS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {distribution.version}')
        for item in distribution.files or ():
            path = Path(distribution.locate_file(item))
            if path.is_file() and path.suffix != '.pyc' and str(item) != 'RECORD':
                files[str(path.absolute())] = sha256(path)
    from pxr import Sdf, Usd, UsdUtils
    from PIL import Image
    Image.new('RGB', (1, 1), (20, 40, 60)).tobytes()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/Consumer')
    UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), lambda value: value)
    if set(VENDORS) & sys.modules.keys():
        raise RuntimeError('NVIDIA SDK imported in diagnostic consumer')
    return dict(python=sys.executable, python_sha256=sha256(sys.executable),
                versions=VERSIONS, requirements_sha256=sha256(REQUIREMENTS), files=files,
                native_sha256={path: value for path, value in files.items() if Path(path).suffix.lower() in ('.pyd', '.dll')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path)
    parser.add_argument('--config', type=Path, help='Add consumer_python to an existing SDK configuration')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--record', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    receipt = Path(sys.prefix) / 'consumer-identity.json'
    if args.record or args.verify:
        identity = inspect()
        if args.record:
            receipt.write_text(json.dumps(identity, indent=2) + '\n')
        elif json.loads(receipt.read_text()) != identity:
            raise RuntimeError('Consumer files differ from the recorded hash-verified installation')
        print(json.dumps(dict(status='passed', python=sys.executable, receipt=str(receipt),
                              versions=VERSIONS, native_files=len(identity['native_sha256']))))
        return
    if not args.environment:
        parser.error('--environment is required for creation')
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 13):
        parser.error('Run with Windows x64 CPython 3.13')
    environment = args.environment.absolute()
    if environment.exists():
        parser.error('Use a new environment directory; existing environments are never modified')
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / 'Scripts/python.exe'
    subprocess.run([str(python), '-m', 'pip', 'install', '--require-hashes', '--only-binary=:all:',
                    '--no-deps', '-r', str(REQUIREMENTS)], check=True)
    subprocess.run([str(python), '-I', str(Path(__file__).resolve()), '--record'], check=True)
    if args.config:
        config = json.loads(args.config.read_text())
        if config.get('schema') != 1 or not config.get('python'):
            raise ValueError('Expected an existing setup_public_sdk configuration')
        config['consumer_python'] = str(python)
        temporary = args.config.with_suffix('.tmp')
        temporary.write_text(json.dumps(config, indent=2) + '\n')
        temporary.replace(args.config)
    print(f'Consumer ready: {python}')


if __name__ == '__main__':
    main()
