# SPDX-License-Identifier: Apache-2.0
"""Runtime license collection over synthetic staged files; never runs a runtime."""
import io
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/runtime_licenses.py'
spec = importlib.util.spec_from_file_location('runtime_licenses_test', SCRIPT)
licenses = importlib.util.module_from_spec(spec); spec.loader.exec_module(licenses)

PROBE = dict(python='3.13.13', openssl='OpenSSL 3.5.6 7 Apr 2026', sqlite='3.51.3', zlib='1.3.1',
             expat='expat_2.7.5', libmpdec='2.5.1', mimalloc=True)
SECTIONS = (
    ('Apache License 2.0', 'Apache-2.0', [('OpenSSL', '3.5.6')]),
    ('BSD 2-Clause "Simplified" License', 'BSD-2-Clause', [('Other', '1.0')]),
    ('BSD Zero Clause License', '0BSD', [('XZ (LZMA)', '5.2.5')]),
    ('bzip2 and libbzip2 License v1.0.6', 'bzip2-1.0.6', [('Bzip2', '1.0.8')]),
    ('MIT', 'MIT', [('Expat', '2_7_5'), ('libffi', '3.5.2')]),
    ('Python License 2.0', 'Python-2.0', [('Python', '3.13.13')]),
    ('Zlib License', 'Zlib', [('Zlib', '1.3.1')]),
)


def blender_license(path, *, drop=None):
    text = ['# Blender Third-Party Licenses\n']
    for title, spdx, rows in SECTIONS:
        text += [f'## [{title}](https://spdx.org/licenses/{spdx}.html)\n', '| Library | Version | Copyright |',
                 '| ------- | ------- | --------- |']
        text += [f'| [{name}](https://example.invalid) | {version} | `Copyright {name}` |' for name, version in rows if name != drop]
        text += ['<details>', f'<summary>{title}</summary>', '', f'{spdx} license text', '</details>', '']
    text += ['## [Public Domain](https://spdx.org/licenses/CC0-1.0.html)', '',
             'Public domain projects have no specific license or requirements in regards to copyright enforcement.', '',
             '| Library | Version |', '| ------- | ------- |', '| [Sqlite](https://www.sqlite.org) | 3.51.3 |', '']
    path.write_text('\n'.join(text), encoding='utf-8')
    return path


def dist(site, name, version, license, files=(), record=(), license_files=()):
    info = site / f'{name.replace("-", "_")}-{version}.dist-info'
    info.mkdir(parents=True)
    header = f'Metadata-Version: 2.4\nName: {name}\nVersion: {version}\nLicense-Expression: {license}\n'
    header += ''.join(f'License-File: {entry}\n' for entry in license_files)
    (info / 'METADATA').write_text(header, encoding='utf-8')
    for relative, data in files:
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (info / 'RECORD').write_text(''.join(f'{r},sha256=x,1\n' for r in record), encoding='utf-8')
    return info


@pytest.fixture
def stage(tmp_path):
    site = tmp_path / 'stage/python/lib/site-packages'
    site.mkdir(parents=True)
    return tmp_path / 'stage', site


def test_every_component_has_exact_version_and_license_text(stage, tmp_path):
    root, site = stage
    nvidia = b'NVIDIA proprietary license\n'
    dist(site, 'ovrtx', '0.6.0.1', 'LicenseRef-NvidiaProprietary', license_files=['LICENSE'],
         files=[('ovrtx-0.6.0.1.dist-info/licenses/LICENSE', nvidia),
                ('ovrtx/THIRD-PARTY-NOTICES.txt', b'COMPONENTS\n\nComponent | Version(s) | SPDX\n\n'
                 b'abseil | unspecified | Apache-2.0\ncurl | 8.19.0 | LicenseRef-curl\n\nSHARED TEXTS\n')],
         record=['ovrtx/THIRD-PARTY-NOTICES.txt', 'ovrtx/core.py'])
    dist(site, 'ovstage', '0.3.0.1', 'LicenseRef-NvidiaProprietary', license_files=['LICENSE'],
         files=[('ovstage-0.3.0.1.dist-info/licenses/LICENSE', nvidia.replace(b'\n', b'\r\n'))])
    dist(site, 'ovstream', '0.5.7', 'LicenseRef-NvidiaProprietary',
         files=[('ovstream/THIRD_PARTY_NOTICES.md', b'# notices\n')], record=['ovstream/THIRD_PARTY_NOTICES.md'])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('zlib-LICENSE.txt', 'zlib text')
        z.writestr('a/Tracy-LICENSE.txt', 'tracy one')
        z.writestr('a/tracy-LICENSE.txt', 'tracy two')
        z.writestr('b/Tracy-LICENSE.txt', 'same')
        z.writestr('b/tracy-LICENSE.txt', 'same')
    dist(site, 'ovphysx', '0.7.0', 'LicenseRef-NVIDIA-Omniverse', license_files=['LICENSE.txt', 'ovphysx-LICENSES.zip'],
         files=[('ovphysx-0.7.0.dist-info/licenses/LICENSE.txt', b'omniverse'),
                ('ovphysx-0.7.0.dist-info/licenses/ovphysx-LICENSES.zip', archive.getvalue())])
    manifest = licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), PROBE, platform='linux-x64')
    rows = {row['name']: row for row in manifest['components']}
    assert {'Python', 'OpenSSL', 'SQLite', 'zlib', 'Expat', 'libffi', 'bzip2', 'XZ Utils (liblzma)', 'libmpdec', 'mimalloc'} <= set(rows)
    assert (rows['Python']['version'], rows['OpenSSL']['version'], rows['Expat']['version']) == ('3.13.13', '3.5.6', '2.7.5')
    assert (rows['libmpdec']['version'], rows['mimalloc']['version']) == ('2.5.1', '2.1.2')
    assert rows['XZ Utils (liblzma)']['license_files'] == ['python-runtime/XZ-Utils-liblzma-5.2.5/LICENSE.txt']
    assert rows['ovstream']['license_text_from'] in ('ovrtx', 'ovstage')
    folder = root / 'licenses'
    unpacked = folder / 'python-packages/ovphysx-0.7.0/ovphysx-0.7.0.dist-info/licenses/ovphysx-LICENSES'
    assert (unpacked / 'zlib-LICENSE.txt').read_text() == 'zlib text'
    # Names differing only in case keep both texts, or one copy when identical, on any file system.
    assert {(p.name, p.read_text()) for p in (unpacked / 'a').iterdir()} == {('Tracy-LICENSE.txt', 'tracy one'), ('tracy-LICENSE-2.txt', 'tracy two')}
    assert [(p.name, p.read_text()) for p in (unpacked / 'b').iterdir()] == [('Tracy-LICENSE.txt', 'same')]
    libmpdec = (folder / rows['libmpdec']['license_files'][0]).read_text()
    assert libmpdec.startswith('Copyright (c) 2008-2020 Stefan Krah. All rights reserved.\n') and 'SUCH DAMAGE.' in libmpdec
    mimalloc = (folder / rows['mimalloc']['license_files'][0]).read_text()
    assert mimalloc.startswith('Copyright (c) 2018-2021 Microsoft Corporation, Daan Leijen\n') and 'SOFTWARE.' in mimalloc
    assert 'asyncio' not in libmpdec + mimalloc and 'W3C' not in libmpdec
    incorporated = [p for p in rows['Python']['license_files'] if p.endswith('cpython-3.13.13-license.rst')]
    assert 'Licenses and Acknowledgements for Incorporated Software' in (folder / incorporated[0]).read_text(encoding='utf-8')
    for row in manifest['components']:
        assert row['version'] and row['license'] and row['license_files']
        assert all((folder / path).is_file() for path in row['license_files'])
    bundled = {row['name']: row for row in manifest['bundled_components']}
    assert bundled['curl']['version'] == '8.19.0' and bundled['abseil']['version'] is None
    assert bundled['abseil']['bundled_in'] == 'ovrtx 0.6.0.1' and manifest['bundled_without_supplier_version'] == 1
    readme = (folder / 'README.md').read_text(encoding='utf-8')
    assert '| ovrtx | 0.6.0.1 | LicenseRef-NvidiaProprietary |' in readme and 'not stated by supplier' in readme
    names = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    assert licenses.validate(names, lambda name: (root / name).read_bytes())['platform'] == 'linux-x64'


def test_distribution_without_its_own_license_fails(stage, tmp_path):
    root, site = stage
    dist(site, 'thing', '1.0', 'MIT', files=[('thing/THIRD_PARTY_NOTICES.md', b'x')], record=['thing/THIRD_PARTY_NOTICES.md'])
    with pytest.raises(ValueError, match='thing 1.0 ships without its own license text'):
        licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), PROBE, platform='linux-x64')


def test_license_reference_is_not_borrowed_when_carriers_disagree(stage, tmp_path):
    root, site = stage
    for name, text in (('a', b'one'), ('b', b'two')):
        dist(site, name, '1.0', 'LicenseRef-Vendor', license_files=['LICENSE'],
             files=[(f'{name}-1.0.dist-info/licenses/LICENSE', text)])
    dist(site, 'c', '1.0', 'LicenseRef-Vendor')
    with pytest.raises(ValueError, match='c 1.0 ships without'):
        licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), PROBE, platform='linux-x64')


@pytest.mark.parametrize('field,value', [('openssl', 'OpenSSL 3.5.7 1 Jan 2027'), ('python', '3.13.14'), ('expat', 'expat_2.7.6')])
def test_interpreter_must_match_blender_license_versions(stage, tmp_path, field, value):
    root, site = stage
    with pytest.raises(ValueError, match='differs from Blender license file'):
        licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), dict(PROBE, **{field: value}), platform='linux-x64')


def test_missing_python_library_in_blender_license_fails(stage, tmp_path):
    root, site = stage
    with pytest.raises(ValueError, match='lacks Python libraries'):
        licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md', drop='libffi'), PROBE, platform='linux-x64')


def test_cpython_license_document_must_match_its_reviewed_digest(tmp_path, monkeypatch):
    name, _ = licenses.CPYTHON_LICENSE['3.13.13']
    (tmp_path / name).write_bytes((licenses.LICENSE_DATA / name).read_bytes() + b'\nedited\n')
    monkeypatch.setattr(licenses, 'LICENSE_DATA', tmp_path)
    with pytest.raises(ValueError, match='differs from the reviewed'):
        licenses.cpython_license('3.13.13')
    with pytest.raises(ValueError, match='No reviewed license document for CPython 3.14.0'):
        licenses.cpython_license('3.14.0')


def test_mimalloc_listed_only_when_compiled_in(stage, tmp_path):
    root, site = stage
    manifest = licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), dict(PROBE, mimalloc=False), platform='windows-x64')
    assert 'mimalloc' not in {row['name'] for row in manifest['components']}


def test_pip_distribution_is_refused(stage, tmp_path):
    root, site = stage
    dist(site, 'pip', '26.0.1', 'MIT', license_files=['LICENSE.txt'], files=[('pip-26.0.1.dist-info/licenses/LICENSE.txt', b'mit')])
    with pytest.raises(ValueError, match='pip must not be shipped'):
        licenses.build_licenses(root, site, blender_license(tmp_path / 'license.md'), PROBE, platform='linux-x64')


@pytest.mark.parametrize('layout', ['posix', 'windows'])
def test_remove_pip_deletes_pip_launchers_and_ensurepip_only(tmp_path, layout):
    python = tmp_path / 'python'
    if layout == 'posix':
        paths = ['bin/python3.13', 'bin/pip3', 'bin/pip3.13', 'bin/f2py', 'lib/python3.13/ensurepip/__init__.py',
                 'lib/python3.13/ensurepip/_bundled/pip-26.0.1-py3-none-any.whl', 'lib/python3.13/site-packages/pip/__init__.py',
                 'lib/python3.13/site-packages/pip-26.0.1.dist-info/METADATA', 'lib/python3.13/site-packages/vendor/pip/keep.py',
                 'lib/python3.13/venv/__init__.py']
    else:
        paths = ['bin/python.exe', 'Scripts/pip3.exe', 'Scripts/pip3.13.exe', 'Scripts/f2py.exe', 'Lib/ensurepip/__init__.py',
                 'Lib/site-packages/pip/__init__.py', 'Lib/site-packages/pip-26.0.1.dist-info/METADATA', 'Lib/venv/__init__.py']
    for path in paths:
        (python / path).parent.mkdir(parents=True, exist_ok=True)
        (python / path).write_text('x')
    licenses.remove_pip(python)
    left = {p.relative_to(python).as_posix() for p in python.rglob('*') if p.is_file()}
    expected = {'bin/python3.13', 'bin/f2py', 'lib/python3.13/site-packages/vendor/pip/keep.py', 'lib/python3.13/venv/__init__.py'} \
        if layout == 'posix' else {'bin/python.exe', 'Scripts/f2py.exe', 'Lib/venv/__init__.py'}
    assert left == expected


@pytest.mark.parametrize('name', ['python/Lib/site-packages/pip/__init__.py', 'python/bin/pip3',
                                  'python/lib/python3.13/ensurepip/__init__.py', 'python/Scripts/pip.exe'])
def test_validate_rejects_shipped_pip(name):
    with pytest.raises(ValueError, match='ships pip'):
        licenses.validate({name, 'licenses/components.json'}, lambda _: b'{}')


def test_validate_rejects_missing_manifest_or_license_file():
    with pytest.raises(ValueError, match='lacks its licenses manifest'):
        licenses.validate({'python/bin/python3.13'}, lambda _: b'')
    manifest = json.dumps(dict(platform='linux-x64', components=[dict(name='x', version='1', license='MIT', license_files=['a/LICENSE'])]))
    with pytest.raises(ValueError, match='license file missing'):
        licenses.validate({'licenses/components.json'}, lambda _: manifest.encode())
    incomplete = json.dumps(dict(platform='linux-x64', components=[dict(name='x', version='', license='MIT', license_files=['a'])]))
    with pytest.raises(ValueError, match='incomplete for x'):
        licenses.validate({'licenses/components.json', 'licenses/a'}, lambda _: incomplete.encode())
    with pytest.raises(ValueError, match='differ only in case'):
        licenses.validate({'licenses/components.json', 'licenses/A', 'licenses/a'}, lambda _: manifest.encode())
