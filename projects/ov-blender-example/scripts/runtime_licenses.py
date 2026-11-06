#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect the license and exact version of everything a runtime sidecar ships.

Inputs are the staged files themselves, never a lockfile or hand-kept list:
Python distributions are read from their installed metadata and RECORD, and
the interpreter's own libraries come from the license file of the Blender
release whose Python was copied, cross-checked against the running interpreter.
Every shipped component must have a version and at least one license text.
"""
from __future__ import annotations

from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import textwrap
import zipfile

FOLDER = 'licenses'
MANIFEST = 'components.json'
# File names that hold license or notice text; code files with such names are not notices.
NOTICE_NAME = re.compile(r'(?i)licen[cs]e|copying|notice|copyright')
CODE_SUFFIXES = {'.py', '.pyi', '.pyc', '.pyd', '.so', '.dll', '.h', '.hpp', '.c', '.cpp', '.json', '.toml'}
TEXT_SUFFIXES = {'', '.txt', '.md', '.rst', '.html', '.apache', '.bsd', '.mit', '.1'}
# Blender's license file names the libraries its bundled Python was built with.
BLENDER_PYTHON_LIBRARIES = {'Python': 'Python', 'OpenSSL': 'OpenSSL', 'Sqlite': 'SQLite', 'Zlib': 'zlib',
                            'Expat': 'Expat', 'libffi': 'libffi', 'Bzip2': 'bzip2', 'XZ (LZMA)': 'XZ Utils (liblzma)'}
# Interpreter probe key for each library whose version Python reports at run time.
PROBED = {'Python': 'python', 'OpenSSL': 'openssl', 'SQLite': 'sqlite', 'zlib': 'zlib', 'Expat': 'expat'}
# CPython's Doc/license.rst for the exact release, with its incorporated-software notices.
LICENSE_DATA = Path(__file__).resolve().with_name('license_data')
CPYTHON_LICENSE = {'3.13.13': ('cpython-3.13.13-license.rst',
                               '62f2c9c2c75d511170eb464ad5f83b78cc1f37eb2eb49c2846c9aa6c4557ee99')}
# Libraries vendored inside CPython's own source tree, identified by that source release;
# their license texts are the matching sections of CPython's license document.
CPYTHON_VENDORED = {
    '3.13.13': (
        dict(name='libmpdec', license='BSD-2-Clause', probe='libmpdec'),
        # Include/internal/mimalloc/mimalloc.h of 3.13.13: MI_MALLOC_VERSION 212.
        dict(name='mimalloc', license='MIT', version='2.1.2', probe='mimalloc'),
    ),
}
PROBE_CODE = '''import json,os,subprocess,sys,ssl,sqlite3,zlib,pyexpat,decimal
mimalloc=subprocess.run([sys.executable,'-I','-c','pass'],env=dict(os.environ,PYTHONMALLOC='mimalloc'),capture_output=True).returncode==0
print(json.dumps(dict(python='%d.%d.%d'%sys.version_info[:3],openssl=ssl.OPENSSL_VERSION,sqlite=sqlite3.sqlite_version,
 zlib=zlib.ZLIB_RUNTIME_VERSION,expat=pyexpat.EXPAT_VERSION,libmpdec=decimal.__libmpdec_version__,mimalloc=mimalloc)))'''
# pip, its metadata, its launchers, and ensurepip (which carries a pip wheel).
PIP_PATH = re.compile(r'(?i)(^|/)(site-packages/(pip|pip-[^/]+\.dist-info)|(lib/python3\.\d+|Lib)/ensurepip)(/|$)'
                      r'|(^|/)(bin|Scripts)/pip(\d+(\.\d+)?)?(\.exe)?$')


def remove_pip(python_root):
    """Delete pip from a copied Python after the build no longer needs it."""
    python_root = Path(python_root)
    removed = []
    for path in sorted(python_root.rglob('*'), key=lambda p: len(p.parts)):
        relative = path.relative_to(python_root).as_posix()
        if path.exists() and PIP_PATH.search(relative):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(relative)
    if any(PIP_PATH.search(p.relative_to(python_root).as_posix()) for p in python_root.rglob('*')):
        raise ValueError('pip is still present in the runtime')
    return removed


def _plain(version):
    """Normalize spellings such as 2_7_5, 'expat_2.7.5' and 'OpenSSL 3.5.6 7 Apr 2026'."""
    match = re.search(r'\d+(?:[._]\d+)+[a-z]?', str(version))
    return match.group(0).replace('_', '.') if match else str(version)


def blender_python_libraries(license_md):
    """Return ({name: row}, {spdx: text}) for Python's libraries from Blender's license.md."""
    text = Path(license_md).read_text(encoding='utf-8')
    sections = re.split(r'(?m)^## ', text)[1:]
    rows, texts = {}, {}
    for section in sections:
        heading = re.match(r'\[([^\]]+)\]\(([^)]+)\)', section)
        if not heading:
            continue
        title, url = heading.groups()
        spdx = re.search(r'spdx\.org/licenses/([^/]+?)\.html', url)
        details = re.search(r'<details>\s*<summary>[^<]*</summary>\n(.*?)</details>', section, re.S)
        if details:
            body = details.group(1).strip() + '\n'
        else:
            statement = re.search(r'\n\n([^|\n<][^\n]+)\n', section)
            body = (title + '\n\n' + statement.group(1) + '\n') if statement else ''
        if spdx and body:
            texts[spdx.group(1)] = body
        for line in section.splitlines():
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')] if line.startswith('| ') else []
            if len(cells) < 2 or set(cells[1]) <= set('-'):
                continue
            link = re.match(r'\[([^\]]+)\]\(([^)]*)\)', cells[0])
            name, homepage = (link.group(1), link.group(2)) if link else (cells[0], '')
            if name in BLENDER_PYTHON_LIBRARIES:
                rows[BLENDER_PYTHON_LIBRARIES[name]] = dict(
                    name=BLENDER_PYTHON_LIBRARIES[name], version=_plain(cells[1]), homepage=homepage,
                    license='LicenseRef-Public-Domain' if title == 'Public Domain' else (spdx.group(1) if spdx else title),
                    copyright=cells[2].strip('`') if len(cells) > 2 else '', text=body)
    missing = set(BLENDER_PYTHON_LIBRARIES.values()) - set(rows)
    if missing:
        raise ValueError(f'Blender license file lacks Python libraries: {sorted(missing)}')
    if any(not row['text'].strip() for row in rows.values()):
        raise ValueError('Blender license file lacks a license text for a Python library')
    return rows, texts


def cpython_license(version):
    """CPython's license document for this release, checked against its reviewed digest."""
    if version not in CPYTHON_LICENSE:
        raise ValueError(f'No reviewed license document for CPython {version}')
    name, expected = CPYTHON_LICENSE[version]
    data = (LICENSE_DATA / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'{name} differs from the reviewed CPython license document')
    return name, data.decode('utf-8')


def rst_notice(document, title):
    """The literal notice block of one 'Licenses and Acknowledgements' section."""
    section = re.search(rf'(?m)^{re.escape(title)}\n-+\n(.*?)(?=^\S[^\n]*\n[-=]{{3,}}\n|^\.\. _|\Z)', document, re.S)
    block = re.search(r'::\n\n((?:   .*\n|\n)+)', section.group(1)) if section else None
    if not block:
        raise ValueError(f'CPython license document has no {title} notice')
    return textwrap.dedent(block.group(1)).strip() + '\n'


def _slug(name):
    """A folder name without spaces or punctuation that some unzip tools mangle."""
    return re.sub(r'-{2,}', '-', re.sub(r'[^A-Za-z0-9.+_-]+', '-', name)).strip('-')


def _safe(name):
    parts = PurePosixPath(name.replace('\\', '/')).parts
    if not parts or name.startswith('/') or any(p in ('', '.', '..') or ':' in p for p in parts):
        raise ValueError(f'Unsafe license archive member: {name}')
    return PurePosixPath(*parts)


def _store(data, name, destination, written):
    """Write one license file; expand license archives so every text is readable."""
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    _store(archive.read(member), member.filename, destination / (name[:-4]), written)
        return
    target = destination / _safe(name)
    # Supplier archives can hold names that differ only in case (Tracy-LICENSE.txt and
    # tracy-LICENSE.txt); the runtime must unpack on Windows, so neither may overwrite the other.
    taken = {str(path).casefold(): path for path in written}
    if str(target).casefold() in taken:
        if taken[str(target).casefold()].read_bytes() == data:
            return
        count = 2
        while str(target.with_stem(f'{target.stem}-{count}')).casefold() in taken:
            count += 1
        target = target.with_stem(f'{target.stem}-{count}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    written.append(target)


def _is_notice(relative):
    path = PurePosixPath(relative)
    if '__pycache__' in path.parts or not NOTICE_NAME.search(path.name):
        return False
    suffix = path.suffix.lower()
    return suffix == '.zip' or (suffix not in CODE_SUFFIXES and (suffix in TEXT_SUFFIXES or path.name.isupper()))


def _distribution(dist_info):
    metadata = Parser().parsestr(dist_info.joinpath('METADATA').read_text(encoding='utf-8'))
    license = metadata.get('License-Expression') or ''
    if not license:
        declared = (metadata.get('License') or '').strip()
        classifiers = [c.rsplit('::', 1)[-1].strip() for c in metadata.get_all('Classifier') or ()
                       if c.startswith('License ::')]
        # Classifiers do not say whether several licenses apply together or as a choice; keep their words.
        license = (declared if declared and '\n' not in declared and len(declared) < 80
                   else 'PyPI classifiers: ' + '; '.join(classifiers) if classifiers else '')
    files = set()
    for entry in metadata.get_all('License-File') or ():
        for candidate in (dist_info / 'licenses' / entry, dist_info / entry):
            if candidate.is_file():
                files.add(candidate)
                break
    record = dist_info / 'RECORD'
    for line in record.read_text(encoding='utf-8').splitlines() if record.is_file() else ():
        relative = line.rsplit(',', 2)[0]
        if relative and not relative.startswith('..') and _is_notice(relative):
            path = dist_info.parent / relative
            if path.is_file():
                files.add(path)
    return metadata['Name'], metadata['Version'], license.strip(), sorted(files)


def _supplier_rows(path, bundled_in):
    """Parse 'Component | Version(s) | License' tables in supplier notice files."""
    rows, active = [], False
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return rows
    for line in lines:
        cells = [cell.strip() for cell in line.split(' | ')]
        if re.match(r'(?i)^component\s*\|\s*version', line.strip()):
            active = True
        elif active and len(cells) == 3 and cells[0]:
            version = cells[1] if cells[1].lower() not in ('', 'unspecified', 'unknown', 'n/a') else None
            rows.append(dict(name=re.sub(r'\s+', ' ', cells[0]), version=version, license=cells[2],
                             bundled_in=bundled_in, declared_in=path.name))
        elif active and rows and line.strip():
            active = False
    return rows


def build_licenses(stage, site_packages, license_md, probe, *, platform):
    """Write stage/licenses with copied texts, components.json and README.md."""
    stage, site_packages = Path(stage), Path(site_packages)
    folder = stage / FOLDER
    if folder.exists():
        raise ValueError('Runtime already has a licenses folder')
    folder.mkdir()
    components, bundled = [], []

    libraries, spdx_texts = blender_python_libraries(license_md)
    for key, field in PROBED.items():
        if _plain(probe[field]) != libraries[key]['version']:
            raise ValueError(f'{key} {probe[field]} differs from Blender license file version {libraries[key]["version"]}')
    python_version = libraries['Python']['version']
    if python_version not in CPYTHON_VENDORED:
        raise ValueError(f'No reviewed vendored-library record for CPython {python_version}')
    document_name, document = cpython_license(python_version)
    for row in CPYTHON_VENDORED[python_version]:
        if row['probe'] == 'mimalloc' and not probe.get('mimalloc'):
            continue
        version = row.get('version') or _plain(probe[row['probe']])
        # The notice block already begins with the component's copyright line.
        libraries[row['name']] = dict(name=row['name'], version=version, license=row['license'],
                                      copyright='', homepage='', text=rst_notice(document, row['name']),
                                      bundled_in=f'Python {python_version}')
    for row in libraries.values():
        text = row.get('text') or spdx_texts.get(row['license'])
        if not text:
            raise ValueError(f'No license text for {row["name"]} ({row["license"]})')
        target = folder / 'python-runtime' / _slug(f'{row["name"]}-{row["version"]}') / 'LICENSE.txt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((row['copyright'] + '\n\n' if row['copyright'] else '') + text, encoding='utf-8')
        files = [target]
        if row['name'] == 'Python':
            # CPython's own notices for the software it incorporates ship beside its license.
            files.append(target.with_name(document_name))
            files[-1].write_text(document, encoding='utf-8')
        components.append(dict(name=row['name'], version=row['version'], license=row['license'],
                               kind='python-runtime', shipped_in='python', homepage=row['homepage'],
                               bundled_in=row.get('bundled_in', ''),
                               license_files=[path.relative_to(folder).as_posix() for path in files]))

    distributions = []
    for dist_info in sorted(site_packages.glob('*.dist-info')):
        name, version, license, files = _distribution(dist_info)
        if name.lower() == 'pip':
            raise ValueError('pip must not be shipped in the runtime')
        if not version:
            raise ValueError(f'{name} has no version')
        # The package's own license sits in its metadata folder or its top-level package.
        own = [f for f in files if (f.parent in (dist_info, dist_info / 'licenses') or f.parent.parent == site_packages)
               and re.match(r'(?i)(licen[cs]e|copying)', f.name) and f.suffix.lower() != '.zip']
        distributions.append((name, version, license, files, own, dist_info))
    texts = {}
    # Wheels built on Windows may carry the same license text with CRLF line endings.
    text = lambda path: path.read_bytes().replace(b'\r\n', b'\n')
    for name, version, license, files, own, dist_info in distributions:
        if len({text(f) for f in own}) == 1:
            texts.setdefault(license, set()).add((text(own[0]), name))
    for name, version, license, files, own, dist_info in distributions:
        destination = folder / 'python-packages' / _slug(f'{name}-{version}')
        written, borrowed = [], None
        for path in files:
            _store(path.read_bytes(), path.relative_to(site_packages).as_posix(), destination, written)
            if 'notice' in path.name.lower() and path.suffix.lower() != '.zip':
                bundled += _supplier_rows(path, f'{name} {version}')
        if not own:
            # A distribution that declares a license reference without its text gets
            # it only when every other shipped carrier of that reference agrees on it.
            donors = texts.get(license, set())
            if license.startswith('LicenseRef-') and len({data for data, _ in donors}) == 1:
                data, borrowed = sorted(donors)[0]
                _store(data, 'LICENSE', destination, written)
            else:
                raise ValueError(f'{name} {version} ships without its own license text')
        row = dict(name=name, version=version, license=license, kind='python-package',
                   shipped_in=dist_info.parent.relative_to(stage).as_posix(),
                   license_files=sorted(p.relative_to(folder).as_posix() for p in written))
        if borrowed:
            row['license_text_from'] = borrowed
        components.append(row)

    unversioned = sum(1 for row in bundled if row['version'] is None)
    manifest = dict(schema=1, platform=platform, components=components, bundled_components=bundled,
                    bundled_without_supplier_version=unversioned,
                    sources=dict(python_runtime=f'Blender license/license.md (Python {python_version})',
                                 python_packages='installed distribution METADATA and RECORD'))
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (folder / 'README.md').write_text(readme(manifest), encoding='utf-8')
    return manifest


def readme(manifest):
    lines = ['# Runtime third-party licenses', '',
             f'Components shipped in the {manifest["platform"]} runtime, each with its exact version and license.',
             'License texts are in the folders named in the last column; `components.json` holds the same data.', '',
             '| Component | Version | License | Shipped in | License files |', '| --- | --- | --- | --- | --- |']
    for row in manifest['components']:
        where = row['bundled_in'] or row['shipped_in'] if row['kind'] == 'python-runtime' else row['shipped_in']
        folder = row['license_files'][0].split('/')[:2]
        lines.append(f'| {row["name"]} | {row["version"]} | {row["license"]} | {where} | `{"/".join(folder)}/` |')
    if manifest['bundled_components']:
        lines += ['', '## Components bundled inside the packages above', '',
                  'As declared in each supplier\'s notice file. "not stated by supplier" marks a component '
                  'whose supplier gives no version; it is identified by the exact package version that carries it.', '',
                  '| Component | Version | License | Bundled in |', '| --- | --- | --- | --- |']
        for row in manifest['bundled_components']:
            lines.append(f'| {row["name"]} | {row["version"] or "not stated by supplier"} | {row["license"]} | {row["bundled_in"]} |')
    return '\n'.join(lines) + '\n'


def validate(names, read):
    """Check a packaged runtime: a licenses manifest whose files exist, and no pip."""
    names = set(names)
    if any(PIP_PATH.search(name) for name in names):
        raise ValueError('Runtime ships pip')
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError('Runtime has paths that differ only in case')
    if f'{FOLDER}/{MANIFEST}' not in names:
        raise ValueError('Runtime lacks its licenses manifest')
    manifest = json.loads(read(f'{FOLDER}/{MANIFEST}'))
    for row in manifest['components']:
        if not row.get('version') or not row.get('license') or not row.get('license_files'):
            raise ValueError(f'Runtime license record incomplete for {row.get("name")}')
        for path in row['license_files']:
            if f'{FOLDER}/{path}' not in names:
                raise ValueError(f'Runtime license file missing: {path}')
    return manifest
