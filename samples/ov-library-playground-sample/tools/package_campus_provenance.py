# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundle retained Poly Haven provenance with the unchanged R48 campus artwork."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = 'c2f715cf7cbe4842f53be845d3074e766355c4343bfdefb4d70b2d669a1c6aa8'
SOURCE_MANIFEST_SHA256 = '93e3dca2968deae0a1b077c0c4a3b889e4c99d972f35d472d2841c76b81ed581'
PREFIX = 'nvidia-campus-r48-wayfinding-landscape/'
RECORDS = ('ATTRIBUTION.md', 'CC0-1.0.txt', 'texture-derivations.json',
           'oak_veneer_01/provenance.json', 'forest_floor/provenance.json',
           'rocky_terrain_03/provenance.json')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf-8')


def build(source, output):
    if output.exists():
        raise FileExistsError('Choose a new archive path; existing archives are preserved.')
    if file_hash(source) != SOURCE_SHA256:
        raise ValueError('Expected the original pinned R48 campus archive.')
    additions = {'provenance/polyhaven/' + name:
                 (ROOT/'data/campus-provenance/polyhaven'/name).read_bytes()
                 for name in RECORDS}
    with zipfile.ZipFile(source) as original:
        manifest_data = original.read(PREFIX + 'manifest.json')
        if digest(manifest_data) != SOURCE_MANIFEST_SHA256:
            raise ValueError('The original campus manifest does not match its pin.')
        manifest = json.loads(manifest_data)
        records = {entry['path']: entry for entry in manifest['files']}
        members = {entry.filename for entry in original.infolist() if not entry.is_dir()}
        if members != {PREFIX + name for name in records} | {PREFIX + 'manifest.json'}:
            raise ValueError('Archive and manifest contents differ.')
        if records.keys() & additions.keys():
            raise ValueError('Provenance would replace an existing campus file.')
        # Keep the exact original manifest as historical evidence. Its hashes
        # describe the original package; the root manifest describes this one.
        additions['provenance/r48-original-manifest.json'] = manifest_data
        readme = original.read(PREFIX + 'README.txt').decode('utf-8').rstrip()
        readme += ('\n\nPoly Haven texture provenance\n'
                   'The provenance/polyhaven/ folder contains the original source records,\n'
                   'CC0-1.0 license, attribution and texture adaptation details for Oak\n'
                   'Veneer 01, Forest Floor and Rocky Terrain 03. Keep these files with\n'
                   'the campus when copying or redistributing it. They cover the named\n'
                   'textures; the other artwork retains its existing terms.\n')
        additions['README.txt'] = readme.encode('utf-8')
        updated = dict(manifest)
        updated['packaging_revision'] = 'polyhaven-provenance-2026-11-03'
        updated['source_archive_sha256'] = SOURCE_SHA256
        updated_records = dict(records)
        for name, data in additions.items():
            updated_records[name] = {'path': name, 'size': len(data), 'sha256': digest(data)}
        updated['files'] = [updated_records[name] for name in sorted(updated_records)]
        additions['manifest.json'] = json_bytes(updated)
        output.parent.mkdir(parents=True, exist_ok=True)
        print('Repacking the verified campus with Poly Haven provenance...', flush=True)
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as packed:
            for entry in original.infolist():
                relative = entry.filename.removeprefix(PREFIX)
                if entry.is_dir() or relative in additions:
                    continue
                with original.open(entry) as src, packed.open(entry, 'w', force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            for name, data in sorted(additions.items()):
                entry = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 11, 3, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                packed.writestr(entry, data, compresslevel=6)
    print('Verifying every packaged file...', flush=True)
    with zipfile.ZipFile(output) as packed:
        expected = {PREFIX + name for name in updated_records} | {PREFIX + 'manifest.json'}
        names = packed.namelist()
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError('The packaged inventory differs from its manifest.')
        for name, record in updated_records.items():
            with packed.open(PREFIX + name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != record['sha256'] or packed.getinfo(PREFIX + name).file_size != record['size']:
                raise ValueError('Packaged file verification failed: ' + name)
    return {'filename': output.name, 'bytes': output.stat().st_size,
            'sha256': file_hash(output), 'prefix': PREFIX, 'manifest': 'manifest.json',
            'manifest_sha256': digest(additions['manifest.json']),
            'source_archive_sha256': SOURCE_SHA256,
            'original_files_preserved': len(records) - 1,
            'original_readme_extended': True, 'manifest_files': len(updated_records),
            'provenance_records': list(RECORDS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = build(args.source.resolve(), args.output.resolve())
    if args.report:
        args.report.write_bytes(json_bytes(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
