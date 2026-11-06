#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Portable setup/launch helpers. NVIDIA imports occur only in external Python."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

PUBLIC = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PUBLIC.parent / 'out' / 'public-sdk-config.json'
sys.path.insert(0, str(PUBLIC / 'addon'))
from ovrtx_blender_example import public_sdk_lock


def requirements():
    return public_sdk_lock.versions(PUBLIC / 'runtime/requirements-public-sdk.txt')


# OVStream 0.5 links the CUDA driver (nvcuda.dll / libcuda.so.1) when it loads, so it
# cannot import on a host without an NVIDIA driver, such as a GPU-less packaging runner.
# Its files are still verified against the lock inventory above; only the import is
# deferred to the add-on's driver preflight on the artist's machine.
DRIVER_LINKED_IMPORTS = frozenset({'ovstream'})


def cuda_driver_available():
    import ctypes
    try:
        ctypes.CDLL('nvcuda.dll' if sys.platform == 'win32' else 'libcuda.so.1')
    except OSError:
        return False
    return True


def inspect_environment():
    errors, installed = [], {}
    identity = None
    try:
        identity = public_sdk_lock.inspect_installed(public_sdk_lock.load_lock(PUBLIC / 'runtime/public-sdk-lock.json'), verify_files=True)
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as exc:
        errors.append(str(exc))
    if sys.version_info[:2] != (3, 13) or struct.calcsize('P') != 8:
        errors.append('Use 64-bit Python 3.13 for the pinned public SDK environment')
    architecture = platform.machine().lower()
    if not ((sys.platform == 'linux' and architecture in ('amd64', 'x86_64'))
            or (sys.platform == 'win32' and architecture in ('amd64', 'x86_64'))):
        errors.append('This locked adapter supports Linux x64 and Windows x64 only')
    for name, expected in requirements().items():
        try:
            installed[name] = importlib.metadata.version(name)
            if installed[name] != expected:
                errors.append(f'{name}: expected {expected}, found {installed[name]}')
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'{name}: missing; install requirements-public-sdk.txt in external Python')
    deferred = []
    if not errors:
        for name in ('ovstage', 'ovrtx', 'ovphysx', 'ovstream'):
            try:
                importlib.import_module(name)
            except Exception as exc:
                message = f'{name} import: {type(exc).__name__}: {exc}'
                if name in DRIVER_LINKED_IMPORTS and isinstance(exc, OSError) and not cuda_driver_available():
                    deferred.append(message)
                else:
                    errors.append(message)
    return {'ok': not errors, 'platform': sys.platform, 'architecture': architecture, 'python': sys.executable,
            'python_version': sys.version.split()[0], 'versions': installed,
            'errors': errors, 'deferred_imports': deferred, 'runtime_identity': identity,
            'gpu_render_validated': False}


def resolve_executable(value):
    expanded = Path(value).expanduser()
    candidate = expanded if expanded.is_file() else Path(shutil.which(str(value)) or str(expanded))
    if not candidate.is_file():
        raise ValueError(f'Executable not found: {value}')
    # Resolving a venv Python symlink would lose pyvenv.cfg.
    return candidate.absolute()


def check_blender(value):
    blender = resolve_executable(value)
    result = subprocess.run([str(blender), '--version'], capture_output=True, text=True,
                            timeout=30, check=True)
    match = re.search(r'Blender (\d+)\.(\d+)', result.stdout)
    if not match or tuple(map(int, match.groups())) < (5, 2):
        raise ValueError('Select Blender 5.2 or newer explicitly with --blender')
    return blender


def worker_environment(config, output=None):
    root = Path(output or config['output']).expanduser().absolute()
    env = dict(os.environ)
    # An explicitly selected public SDK must not inherit SIL selection or a
    # different GPU from an unrelated shell/session.
    env.pop('OVRTX_ACTIVE_CUDA_GPUS', None)
    env.pop('OV_BLENDER_EXAMPLE_BACKEND', None)
    env.update(OV_BLENDER_PUBLIC_SDK_PYTHON=str(resolve_executable(config['python'])),
               OV_BLENDER_PUBLIC_SDK_OUTPUT=str(root),
               OV_BLENDER_PUBLIC_SDK_GPU=str(config.get('gpu', 0)),
               BLENDER_USER_CONFIG=str(root / 'blender-config'),
               BLENDER_USER_SCRIPTS=str(root / 'blender-scripts'),
               OV_BLENDER_EXAMPLE_SCENE_GENERATION_DIR=str(root / 'scene-generations'),
               OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR=str(root / 'render-composition'))
    env['PYTHONPATH'] = str(PUBLIC / 'addon') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    admission = config.get('physics_admission')
    if admission and Path(admission).is_file():
        env['OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION'] = str(Path(admission).absolute())
    for key in ('BLENDER_USER_CONFIG', 'BLENDER_USER_SCRIPTS', 'OV_BLENDER_PUBLIC_SDK_OUTPUT'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 1 or not all(config.get(k) for k in ('python', 'blender', 'output')):
        raise ValueError('Run setup_public_sdk.py to create a valid local configuration')
    return config


if __name__ == '__main__':
    report = inspect_environment()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)
