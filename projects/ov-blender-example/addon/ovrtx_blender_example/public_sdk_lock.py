# SPDX-License-Identifier: Apache-2.0
"""Canonical, stdlib-only SDK lock policy shared by Blender and external tools.

Only the external interpreter calls ``inspect_installed``. No SDK is imported by
this module. Artifact hashes cover bundled native dependencies as well as Python.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import sys
import sysconfig
from urllib.request import urlopen
import zipfile

LOCK_NAME = 'public-sdk-lock.json'
RECEIPT_NAME = 'public-sdk-integrity.json'
REQUIRED = frozenset(('ovrtx', 'ovstage', 'ovphysx', 'ovstream', 'warp-lang', 'numpy', 'packaging',
                      'physx-usd-schemas'))
IMPORT_NAMES = {'warp-lang': 'warp', 'physx-usd-schemas': 'physx_usd_schemas'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def host_platform():
    if platform.machine().lower() not in ('amd64', 'x86_64') or sys.platform not in ('win32', 'linux'):
        raise ValueError('Locked runtime supports Windows x64 and Linux x64 only')
    return 'windows-x64' if sys.platform == 'win32' else 'linux-x64'


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if lock.get('schema') != 1 or set(lock.get('versions', {})) != REQUIRED:
        raise ValueError('Runtime lock schema or complete dependency tuple is invalid')
    if lock.get('python') != dict(implementation='cpython', major=3, minor=13, abi='cp313', bits=64):
        raise ValueError('Runtime lock requires the reviewed CPython 3.13 x64 ABI')
    if set(lock.get('platforms', {})) != {'windows-x64', 'linux-x64'}:
        raise ValueError('Runtime lock must describe both portable platform layouts')
    for target, entry in lock['platforms'].items():
        if set(entry.get('artifacts', {})) != REQUIRED:
            raise ValueError('Runtime lock is missing transitive dependencies')
        for name, row in entry['artifacts'].items():
            filename = row['filename']
            tag = 'win_amd64' if target == 'windows-x64' else 'x86_64'
            if (row['version'] != lock['versions'][name] or not filename.endswith('.whl')
                    or Path(filename).name != filename or '\\' in filename
                    or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                    or (tag not in filename and not filename.endswith('-none-any.whl'))
                    or ('-cp' in filename and '-cp313-cp313-' not in filename)
                    or not row['url'].startswith(('https://files.pythonhosted.org/', 'https://pypi.nvidia.com/'))):
                raise ValueError(f'Invalid locked version/hash/platform/interpreter artifact: {name}')
    return lock


def lock_identity(lock, target=None):
    target = target or host_platform()
    artifacts = lock['platforms'][target]['artifacts']
    return dict(schema=1, platform=target, python=lock['python'], versions=lock['versions'],
                lock_sha256=hashlib.sha256(canonical(lock)).hexdigest(),
                artifacts={name: row['sha256'] for name, row in artifacts.items()})


def versions(requirements):
    requirements = Path(requirements)
    lock = load_lock(requirements.with_name(LOCK_NAME))
    parsed = dict(line.split('==', 1) for line in requirements.read_text().splitlines()
                  if line and not line.startswith('#'))
    if parsed != lock['versions']:
        raise ValueError('Generated requirements disagree with canonical public-sdk-lock.json; regenerate them')
    return dict(lock['versions'])


def interpreter_identity():
    return dict(implementation=sys.implementation.name, major=sys.version_info[0],
                minor=sys.version_info[1], abi=f'cp{sys.version_info[0]}{sys.version_info[1]}',
                bits=struct.calcsize('P') * 8)


def validate_interpreter(lock):
    if interpreter_identity() != lock['python'] or sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise ValueError('Runtime requires CPython 3.13 x64 (cp313); select a matching external Python')


def wheel_files(path):
    """Hash actual wheel bytes, including every bundled native transitive library."""
    files = {}
    with zipfile.ZipFile(path) as wheel:
        for row in wheel.infolist():
            if row.is_dir():
                continue
            name = row.filename
            if name.startswith('/') or '\\' in name or ':' in name or '..' in PurePosixPath(name).parts:
                raise ValueError('Unsafe wheel member')
            if name in files:
                raise ValueError('Duplicate wheel member')
            # pip necessarily rewrites RECORD and generates launchers. All other
            # source/metadata/native bytes are stable and checked after install.
            if name.endswith('.dist-info/RECORD'):
                continue
            if '.data/' in name:
                raise ValueError('Wheel .data relocation needs explicit lock support')
            with wheel.open(row) as stream:
                files[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return files


def acquire(lock, cache, target=None, *, cancelled=lambda: False):
    """Download exact URLs atomically; a partial file never becomes an artifact."""
    target = target or host_platform()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, row in lock['platforms'][target]['artifacts'].items():
        destination = cache / row['filename']
        if destination.exists():
            if digest(destination) != row['sha256']:
                raise ValueError(f'Cached {name} checksum mismatch; remove only {destination} and retry. Prior runtime retained.')
        else:
            partial = destination.with_suffix('.whl.partial')
            try:
                with urlopen(row['url'], timeout=60) as source, partial.open('wb') as output:
                    while block := source.read(1024 * 1024):
                        if cancelled():
                            raise RuntimeError('Runtime download cancelled; prior runtime retained')
                        output.write(block)
                if digest(partial) != row['sha256']:
                    raise ValueError(f'Downloaded {name} checksum mismatch; retry. Prior runtime retained.')
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        result[name] = destination
    return result


def install_requirements(lock, artifacts):
    return ''.join(f'{Path(artifacts[name]).absolute().as_uri()} --hash=sha256:{row["sha256"]}\n'
                   for name, row in lock['platforms'][host_platform()]['artifacts'].items())


def build_integrity(lock, artifacts):
    manifests = {}
    for name, path in artifacts.items():
        row = lock['platforms'][host_platform()]['artifacts'][name]
        if digest(path) != row['sha256']:
            raise ValueError(f'{name} artifact changed before installation')
        files = wheel_files(path)
        if hashlib.sha256(canonical(files)).hexdigest() != row.get('files_sha256'):
            raise ValueError(f'{name} file inventory differs from reviewed lock')
        manifests[name] = files
    return dict(identity=lock_identity(lock), packages=manifests)


def validate_integrity_record(lock, record, target=None):
    target = target or host_platform()
    if record.get('identity') != lock_identity(lock, target) or set(record.get('packages', {})) != REQUIRED:
        raise ValueError('Installed runtime identity differs from the complete locked tuple; reinstall')
    for name, files in record['packages'].items():
        row = lock['platforms'][target]['artifacts'][name]
        if hashlib.sha256(canonical(files)).hexdigest() != row.get('files_sha256'):
            raise ValueError(f'{name} installed inventory was altered; reinstall')
    return record['packages']


def inspect_installed(lock, *, receipt=None, verify_files=False):
    """Check ABI, versions, and trusted wheel file inventories before activation.

    ``verify_files=True`` hashes every installed byte; only runtime
    installation, change or package build passes it. The default serves
    worker startup: the locked inventory must be complete, and hashes come
    from the records written when the runtime was installed.
    """
    validate_interpreter(lock)
    identity = lock_identity(lock)
    receipt = Path(receipt or (Path(sys.prefix) / RECEIPT_NAME))
    record = json.loads(receipt.read_text())
    validate_integrity_record(lock, record)
    site = Path(sysconfig.get_path('purelib')).resolve()
    if any(site.glob('*.pth')) or any((site / name).exists() for name in ('sitecustomize.py', 'usercustomize.py')):
        raise ValueError('Runtime has untracked Python startup hooks; rebuild the isolated environment')
    native = {}
    checks = []
    for name, row in lock['platforms'][host_platform()]['artifacts'].items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != row['version']:
            raise ValueError(f'{name} version mismatch; reinstall the complete tuple')
        files = record['packages'][name]
        if Path(distribution.locate_file('')).resolve() != site:
            raise ValueError(f'{name} metadata resolves outside the locked environment')
        module = importlib.util.find_spec(IMPORT_NAMES.get(name, name))
        try:
            origin = Path(module.origin).resolve().relative_to(site).as_posix() if module else None
        except (ValueError, TypeError):
            origin = None
        if origin not in files:
            raise ValueError(f'{name} import resolves outside its locked inventory; clear PYTHONPATH/user site overrides')
        if hashlib.sha256(canonical(files)).hexdigest() != row.get('files_sha256'):
            raise ValueError(f'{name} installed inventory was altered; reinstall')
        for relative, expected in files.items():
            path = (site / relative).resolve()
            path.relative_to(site)
            checks.append((relative, path, expected))
            if path.suffix.lower() in ('.dll', '.pyd', '.so') or '.so.' in path.name:
                native[relative] = expected
    if verify_files:
        # Installation only: bound CPU and memory while hashing every byte.
        # Resolve paths and validate metadata/origins above before starting
        # workers; join all workers (including on failure) before returning.
        def verify_file(check):
            relative, path, expected = check
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'Runtime file checksum mismatch: {relative}; reinstall. Prior runtime retained.')

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix='runtime-integrity') as pool:
            for _ in pool.map(verify_file, checks):
                pass
    else:
        for relative, path, _expected in checks:
            if not path.is_file():
                raise ValueError(f'Runtime file missing: {relative}; reinstall. Prior runtime retained.')
    observed = {p.relative_to(site).as_posix() for p in site.rglob('*') if p.is_file()
                and (p.suffix.lower() in ('.dll', '.pyd', '.so') or '.so.' in p.name)}
    if observed != set(native):
        raise ValueError('Untracked native libraries in runtime; rebuild the isolated environment')
    executable = getattr(sys, '_base_executable', sys.executable)
    python_sha256 = digest(executable) if verify_files else recorded_executable_sha256(record, executable)
    return dict(identity, native_sha256=hashlib.sha256(canonical(native)).hexdigest(), native_count=len(native),
                python_version=sys.version, python_executable_sha256=python_sha256)


def recorded_executable_sha256(record, executable):
    """The interpreter hash recorded at installation, never computed at startup.

    Automatic installs record it in the integrity receipt. Packaged sidecars
    carry it in their install-validated ``sidecar.json`` file roster. Returns
    ``None`` when no record covers this interpreter.
    """
    if isinstance(record.get('python_executable_sha256'), str):
        return record['python_executable_sha256']
    manifest_path = Path(sys.prefix).parent / 'sidecar.json'
    try:
        manifest = json.loads(manifest_path.read_text())
        relative = manifest['python']
        if (manifest_path.parent / relative).resolve() != Path(executable).resolve():
            return None
        value = manifest['files'][relative]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) else None
