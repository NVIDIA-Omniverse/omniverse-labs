# SPDX-License-Identifier: Apache-2.0
"""External-only locked install/inspection entry point; no online resolver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'addon' if (SOURCE / 'addon').is_dir() else SOURCE))
from ovrtx_blender_example import public_sdk_lock as policy


def install(lock, cache):
    policy.validate_interpreter(lock)
    temporary_root = Path(cache).absolute().parent / 'install-tmp'
    temporary_root.mkdir(parents=True, exist_ok=True)
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[name] = str(temporary_root)
    artifacts = policy.acquire(lock, cache)
    receipt = policy.build_integrity(lock, artifacts)
    cache = Path(cache)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir=cache, delete=False) as stream:
        stream.write(policy.install_requirements(lock, artifacts))
        requirements = Path(stream.name)
    try:
        subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
            'install', '--no-deps', '--no-index', '--no-cache-dir', '--no-compile', '--require-hashes',
            '-r', str(requirements)], check=True)
        subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', 'check'], check=True)
    finally:
        requirements.unlink(missing_ok=True)
    # Record the interpreter hash now: workers read it instead of hashing at startup.
    receipt['python_executable_sha256'] = policy.digest(getattr(sys, '_base_executable', sys.executable))
    (Path(sys.prefix) / policy.RECEIPT_NAME).write_text(json.dumps(receipt, sort_keys=True) + '\n')
    return policy.inspect_installed(lock, verify_files=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'verify', 'resolve'))
    parser.add_argument('--lock', type=Path, default=SOURCE / 'runtime' / policy.LOCK_NAME)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    lock = policy.load_lock(args.lock)
    policy.validate_interpreter(lock)
    if args.action in ('install', 'resolve') and args.cache is None:
        parser.error('--cache is required; use an explicitly owned runtime cache')
    if args.action == 'install':
        result = install(lock, args.cache)
    elif args.action == 'resolve':
        artifacts = policy.acquire(lock, args.cache)
        policy.build_integrity(lock, artifacts)
        result = policy.lock_identity(lock)
    else:
        result = policy.inspect_installed(lock, verify_files=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
