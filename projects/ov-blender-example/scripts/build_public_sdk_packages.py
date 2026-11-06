#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a small installable add-on ZIP and a separate platform SDK sidecar.

Developer packaging only. Artists install the add-on and select sidecar.zip in
Preferences. Public SDK binaries are never copied into the add-on or source Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / 'addon'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovrtx_blender_example.public_sdk_sidecar import expected_versions, platform_id
from ovrtx_blender_example import public_sdk_lock
import runtime_licenses
# The add-on's own license and notices travel inside the add-on ZIP.
ADDON_LICENSE_FILES = ('LICENSE', 'THIRD_PARTY_NOTICES.md')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def archive_tree(root, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Use a new output path; packages are immutable')
    temporary = output.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(output.suffix + '.sha256').write_text(digest(output) + '  ' + output.name + '\n', encoding='utf-8')
    print(json.dumps(str(output)), flush=True)


def copy_source(source, destination):
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Add-on source contains symlinks')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))


def is_native_payload(path):
    name = path.name.lower()
    if (path.suffix.lower() in ('.dll', '.pyd', '.exe', '.whl', '.dylib', '.a', '.lib')
            or re.search(r'\.so(?:\.|$)', name)):
        return True
    if path.is_file():
        with path.open('rb') as stream:
            magic = stream.read(4)
        return magic.startswith((b'\x7fELF', b'MZ')) or magic in (
            b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe')
    return False


def check_physics_adapter_record(runtime):
    """Ship only a physics adapter identity record that matches its sources.

    Workers read this record instead of hashing sources at every start, so a
    stale record would misidentify the adapter used for physics admission.
    """
    # run_path compiles without writing bytecode into the staged archive tree.
    identity = runpy.run_path(str(runtime / 'physics_adapter_identity.py'),
                              run_name='physics_adapter_identity_build')
    record = runtime / identity['RECORD_NAME']
    # A CRLF checkout of the LF record is the same record.
    if (not record.is_file()
            or record.read_bytes().replace(b'\r\n', b'\n') != identity['record_bytes'](runtime)):
        raise ValueError('runtime/physics_adapter_identity.json is missing or stale; '
                         'run runtime/physics_adapter_identity.py --write and requalify physics')


def build_addon(output, automatic=False, *, sidecar_manifest_sha256=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='public-sdk-addon-', dir=output.parent) as temporary:
        stage = Path(temporary)
        for path in (PUBLIC / 'addon').iterdir():
            if path.is_file() and path.suffix in ('.py', '.toml'):
                shutil.copy2(path, stage / path.name)
        copy_source(PUBLIC / 'addon/ovrtx_blender_example', stage / 'ovrtx_blender_example')
        copy_source(PUBLIC / 'runtime', stage / 'runtime')
        for name in ADDON_LICENSE_FILES:
            shutil.copy2(PUBLIC / name, stage / name)
        check_physics_adapter_record(stage / 'runtime')
        marker = {'schema': 1, 'backend': 'public_sdk',
                  'install_mode': 'automatic' if automatic else 'local_package'}
        if sidecar_manifest_sha256 is not None:
            if not re.fullmatch('[0-9a-f]{64}', sidecar_manifest_sha256):
                raise ValueError('Invalid paired sidecar manifest pin')
            marker['sidecar_manifest_sha256'] = sidecar_manifest_sha256
        (stage / 'public-sdk.json').write_text(json.dumps(marker, indent=2) + '\n')
        manifest = stage / 'blender_manifest.toml'
        text = manifest.read_text()
        text = text.replace('Connect to a managed local ovrtx worker over loopback gRPC',
                            'Use local public SDK workers; optionally download SDK packages')
        manifest.write_text(text)
        # The worker/runtime source is small; there must be no SDK binaries in
        # this archive, even if a developer's runtime directory was polluted.
        forbidden = [p for p in stage.rglob('*') if is_native_payload(p)]
        if forbidden:
            raise ValueError('Add-on source contains native SDK binaries')
        archive_tree(stage, output)


def clean_environment():
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    temporary = PUBLIC.parent / 'out/package-tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    return env


def query_python(python, code):
    result = subprocess.run([str(python), '-I', '-c', code], env=clean_environment(),
                            check=True, capture_output=True, text=True, timeout=180)
    return json.loads(result.stdout)


def existing_relative(path, root):
    """Use on-disk spelling in the archive even if Windows sysconfig differs."""
    current = Path(root)
    for part in Path(path).relative_to(root).parts:
        matches = [child for child in current.iterdir() if child.name.casefold() == part.casefold()]
        exact = [child for child in matches if child.name == part]
        if len(exact) == 1:
            current = exact[0]
        elif len(matches) == 1:
            current = matches[0]
        else:
            raise ValueError('Cannot determine unambiguous archived path spelling')
    return current.relative_to(root).as_posix()


def build_sidecar(output, base_python, sdk_python=None, *, unpacked=False, blender_license=None):
    if output.exists():
        raise ValueError('Use a new output path; packages are immutable')
    layout = query_python(base_python, 'import sys,sysconfig,json; print(json.dumps(dict(prefix=sys.base_prefix, executable=sys._base_executable, version=list(sys.version_info[:2]), purelib=sysconfig.get_path("purelib"))))')
    if layout['version'] != [3, 13]:
        raise ValueError('Sidecar base Python must be 3.13')
    base = Path(layout['prefix']).resolve()
    if base in (Path('/usr'), Path('/usr/local'), Path('/')):
        raise ValueError('Use a self-contained Python installation, such as Blender bundled Python 3.13')
    # Blender's <version>/python sits two levels below the release's license folder.
    blender_license = Path(blender_license or base.parents[1] / 'license/license.md')
    if not blender_license.is_file():
        raise ValueError('Base Python must come from a Blender release that ships license/license.md')
    executable = Path(layout['executable']).absolute().relative_to(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 12 * 1024**3:
        raise ValueError('Sidecar packaging needs 12 GiB free for staging and archive')
    with tempfile.TemporaryDirectory(prefix='public-sdk-sidecar-', dir=output.parent) as temporary:
        stage = Path(temporary); target = stage / 'python'
        print('Copying standalone Python runtime', flush=True)
        # SDK use can leave renderer shader caches beside the interpreter.
        # They are generated, host-specific state, not Python/runtime inputs.
        ignored = shutil.ignore_patterns('site-packages', '__pycache__', '*.pyc', '*.pyo')
        def ignore_python(directory, names):
            result = set(ignored(directory, names))
            if Path(directory) == base / 'bin':
                result.add('cache')
            return result
        shutil.copytree(base, target, ignore=ignore_python)
        python = target / executable
        copied = query_python(python, 'import sys,sysconfig,json; print(json.dumps(dict(prefix=sys.prefix,purelib=sysconfig.get_path("purelib"))))')
        if Path(copied['prefix']).resolve() != target.resolve():
            raise ValueError('Copied Python did not relocate; refusing a non-portable sidecar')
        packages = Path(copied['purelib'])
        packages.resolve().relative_to(target.resolve())
        packages.mkdir(parents=True, exist_ok=True)
        if sdk_python:
            subprocess.run([str(sdk_python), '-I', str(PUBLIC / 'runtime/materialize_sdk.py'), 'verify'],
                           env=clean_environment(), check=True, timeout=300)
            print('Copying only selected public SDK distributions and their runtime dependencies', flush=True)
            listing = query_python(sdk_python, '''import importlib.metadata as m,json,sysconfig
from pathlib import Path
root=Path(sysconfig.get_path('purelib')).resolve(); result={}
for name in ('ovrtx','ovphysx','ovstage','ovstream','numpy','warp-lang','packaging','physx-usd-schemas'):
 d=m.distribution(name); files=[]
 for f in d.files or ():
  p=Path(d.locate_file(f)).absolute()
  try: relative=p.relative_to(root)
  except ValueError: continue
  if '..' in relative.parts or '__pycache__' in relative.parts or p.suffix in ('.pyc','.pyo'): continue
  if p.is_file(): files.append([str(p),relative.as_posix()])
 result[name]={'version':d.version,'files':files}
print(json.dumps(result))''')
            for name, expected in expected_versions(PUBLIC / 'runtime/requirements-public-sdk.txt').items():
                if listing[name]['version'] != expected:
                    raise ValueError(f'Existing SDK has wrong {name} version')
            for row in listing.values():
                for source, relative in row['files']:
                    destination = packages / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            sdk_prefix = query_python(sdk_python, 'import sys,json; print(json.dumps(sys.prefix))')
            shutil.copy2(Path(sdk_prefix) / public_sdk_lock.RECEIPT_NAME, target / public_sdk_lock.RECEIPT_NAME)
        else:
            subprocess.run([str(python), '-m', 'ensurepip'], env=clean_environment(), check=True)
            subprocess.run([str(python), '-I', str(PUBLIC / 'runtime/materialize_sdk.py'),
                'install', '--cache', str(output.parent / 'sdk-wheel-cache')], env=clean_environment(), check=True)
        subprocess.run([str(python), str(PUBLIC / 'scripts/public_sdk_environment.py')],
                       env=clean_environment(), check=True, timeout=180)
        for cache in stage.rglob('__pycache__'):
            shutil.rmtree(cache)
        # pip installs the SDK wheels above; the shipped runtime never runs it.
        runtime_licenses.remove_pip(target)
        print('Collecting runtime licenses', flush=True)
        runtime_licenses.build_licenses(stage, packages, blender_license,
                                        query_python(python, runtime_licenses.PROBE_CODE), platform=platform_id())
        print('Hashing sidecar files', flush=True)
        files = {path.relative_to(stage).as_posix(): digest(path) for path in sorted(stage.rglob('*')) if path.is_file()}
        manifest = {'schema': 1, 'backend': 'public_sdk', 'platform': platform_id(),
                    'runtime_identity': public_sdk_lock.lock_identity(public_sdk_lock.load_lock(PUBLIC / 'runtime/public-sdk-lock.json')),
                    'runtime_integrity': json.loads((target / public_sdk_lock.RECEIPT_NAME).read_text()),
                    'site_packages': existing_relative(packages, stage),
                    'python': python.relative_to(stage).as_posix(),
                    'requirements': expected_versions(PUBLIC / 'runtime/requirements-public-sdk.txt'), 'files': files}
        (stage / 'sidecar.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if unpacked:
            stage.rename(output)
            print(output, flush=True)
        else:
            print('Writing sidecar ZIP', flush=True)
            archive_tree(stage, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('addon', 'sidecar'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--automatic', action='store_true', help='Add-on auto-download option; default is local sidecar selection')
    parser.add_argument('--base-python', default=getattr(sys, '_base_executable', sys.executable))
    parser.add_argument('--sdk-python', help='Reuse public SDK packages from an existing environment instead of downloading')
    parser.add_argument('--unpacked', action='store_true', help='Retain a verified standalone sidecar directory for local installation')
    parser.add_argument('--blender-license', type=Path, help="Blender release license/license.md; default is beside the base Python's Blender")
    args = parser.parse_args()
    if args.kind == 'addon':
        build_addon(args.output.absolute(), args.automatic)
    else:
        build_sidecar(args.output.absolute(), args.base_python, args.sdk_python, unpacked=args.unpacked,
                      blender_license=args.blender_license)


if __name__ == '__main__':
    main()
