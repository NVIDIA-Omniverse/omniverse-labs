# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repackage the reviewed assets with portable provenance and unchanged scene data.

Run with the sample's USD-enabled Python. Existing archives are never overwritten.
Only the reviewed provenance fields may change; every other authored USD field
and every untouched archive member must compare equal before packaging succeeds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from pxr import Sdf

CAMPUS_SHA = 'f4571f611335e10a1bec90bebb182e0e42442d1154ecfe5bf0e61673cb1d5225'
EXHIBITS_SHA = '61f21815e83c6f9cefde613df14881562b9b5e5b1b54f4136d15379c6ee12a83'
LOCAL = re.compile(r'[A-Za-z]:[/\\]Users[/\\]|/home/|/Users/', re.I)
RAW_LOCAL = re.compile(rb'[A-Za-z]:[/\\]Users[/\\]|/home/|/Users/', re.I)
REVISION = 'portable-provenance-2026-11-05'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf-8')


def specs(layer):
    paths = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    return {str(path): spec for path in paths
            if (spec := layer.pseudoRoot if path == Sdf.Path.absoluteRootPath
                else layer.GetObjectAtPath(path)) is not None}


def strings(value):
    if isinstance(value, Sdf.AssetPath):
        yield value.path
    elif isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, (list, tuple)) or type(value).__name__ in ('StringArray', 'TokenArray', 'AssetPathArray'):
        for item in value:
            yield from strings(item)
    elif type(value).__name__.endswith('ListOp'):
        yield str(value)


def local_fields(layer):
    result = []
    for path, spec in specs(layer).items():
        for key in spec.ListInfoKeys():
            if key == 'default' and isinstance(spec, Sdf.AttributeSpec):
                if str(spec.typeName) not in ('asset', 'asset[]', 'string', 'string[]', 'token', 'token[]'):
                    continue
            if any(LOCAL.search(text) for text in strings(spec.GetInfo(key))):
                result.append((path, key))
    return result


def rewrite_usd(source, destination, kind, revision=None):
    original = Sdf.Layer.FindOrOpen(str(source))
    if not original:
        raise ValueError('Cannot open USD: ' + source.name)
    rewritten = Sdf.Layer.CreateNew(str(destination))
    rewritten.TransferContent(original)
    found = local_fields(original)
    changes = {}
    for path, key in found:
        spec = rewritten.GetObjectAtPath(path)
        if kind == 'campus' and path.endswith('.userProperties:pavilion_geometry_source') and key == 'default':
            value = spec.default.replace('\\', '/')
            prefix = re.match(r'/home/[^/]+/ov-library-lab/', value)
            if prefix:
                replacement = value[prefix.end():]
            elif re.fullmatch(r'[A-Za-z]:/Users/[^/]+/kit/projects/ov-library-lab/authoring/.+', value):
                replacement = 'authoring/' + value.split('/ov-library-lab/authoring/', 1)[1]
            else:
                raise ValueError('Unexpected campus provenance prefix')
            if replacement.startswith('/') or '..' in Path(replacement).parts:
                raise ValueError('Invalid portable authoring identifier')
        elif kind == 'exhibits' and path == '/Asset/Axis/Model' and key == 'assetInfo':
            value = dict(spec.GetInfo(key))
            identifier = value.get('identifier')
            if not isinstance(identifier, Sdf.AssetPath):
                raise ValueError('Expected an assetInfo identifier')
            marker = '/simready-sources/' + revision + '/'
            if marker not in identifier.path:
                raise ValueError('Unexpected SimReady provenance revision')
            upstream = identifier.path.split(marker, 1)[1]
            if '..' in Path(upstream).parts or upstream.startswith('/'):
                raise ValueError('Invalid upstream provenance path')
            replacement = dict(value, identifier=Sdf.AssetPath(
                'https://huggingface.co/datasets/nvidia/simready-assets/resolve/' + revision + '/' + upstream))
        else:
            raise ValueError('Unreviewed workstation path field: ' + path + ' / ' + key)
        changes[path, key] = replacement
        spec.SetInfo(key, replacement)
    if not found:
        raise ValueError('Expected the reviewed provenance fields in ' + source.name)
    rewritten.Save()
    # Reopen the freshly written crate; this also drops unreachable strings left
    # by historical in-place edits to the original crate's string table.
    rewritten.Reload()
    before, after = specs(original), specs(rewritten)
    if before.keys() != after.keys():
        raise ValueError('USD spec inventory changed')
    checked = 0
    for path, old in before.items():
        new = after[path]
        if set(old.ListInfoKeys()) != set(new.ListInfoKeys()):
            raise ValueError('Authored field inventory changed: ' + path)
        for key in old.ListInfoKeys():
            expected = changes.get((path, key), old.GetInfo(key))
            if new.GetInfo(key) != expected:
                raise ValueError('Unexpected USD field change: ' + path + ' / ' + key)
            checked += 1
    if original.GetExternalReferences() != rewritten.GetExternalReferences() or local_fields(rewritten):
        raise ValueError('Dependency change or remaining authored workstation path')
    with destination.open('rb') as stream:
        carry = b''
        while chunk := stream.read(4 * 1024 * 1024):
            if RAW_LOCAL.search(carry + chunk):
                raise ValueError('Remaining printable workstation path in rewritten crate')
            carry = chunk[-32:]
    return {'file': source.name, 'specs_compared': len(before), 'fields_compared': checked,
            'provenance_fields_updated': len(changes), 'other_fields_identical': True,
            'composition_references_unchanged': True, 'workstation_paths_remaining': 0}


def package(source, output, scratch, kind):
    expected = CAMPUS_SHA if kind == 'campus' else EXHIBITS_SHA
    if sha(source) != expected:
        raise ValueError('Input archive differs from the reviewed snapshot: ' + source.name)
    if output.exists():
        raise FileExistsError(output)
    prefix = 'nvidia-campus-r48-wayfinding-landscape/' if kind == 'campus' else 'simready/'
    manifest_name = 'manifest.json' if kind == 'campus' else 'release-manifest.json'
    updates, checks = {}, []
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read(prefix + manifest_name))
        records = {record['path']: dict(record) for record in manifest['files']}
        index = json.loads(archive.read(prefix + 'index.json')) if kind == 'exhibits' else None
        targets = ['source/pavilion.usdc'] if kind == 'campus' else sorted(v['file'] for v in index['assets'].values())
        if len(targets) != (1 if kind == 'campus' else 8):
            raise ValueError('Unexpected reviewed USD inventory')
        for number, relative in enumerate(targets):
            original = scratch / (str(number) + '-original.usdc')
            portable = scratch / (str(number) + '-portable.usdc')
            with archive.open(prefix + relative) as src, original.open('xb') as dst:
                shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            record = records[relative]
            if sha(original) != record['sha256']:
                raise ValueError('Input USD differs from its manifest')
            print('Rewriting provenance and comparing USD fields: ' + relative, flush=True)
            check = rewrite_usd(original, portable, kind, index['revision'] if index else None)
            check['file'] = relative
            checks.append(check)
            updates[relative] = portable
            print('Verified: ' + relative, flush=True)
        if index:
            for item in index['assets'].values():
                item['bytes'] = updates[item['file']].stat().st_size
            updated_index = scratch / 'index.json'
            updated_index.write_bytes(encoded(index))
            updates['index.json'] = updated_index
        for relative, file in updates.items():
            records[relative]['sha256'] = sha(file)
            records[relative]['size' if kind == 'campus' else 'bytes'] = file.stat().st_size
        manifest['files'] = [records[name] for name in sorted(records)]
        manifest['packaging_revision'] = REVISION
        manifest['previous_archive_sha256'] = expected
        manifest_bytes = encoded(manifest)
        expected_members = {prefix + name for name in records} | {prefix + manifest_name}
        members = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(members) != len(expected_members) or set(members) != expected_members:
            raise ValueError('Archive members differ from manifest')
        print('Packaging ' + output.name, flush=True)
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
            for entry in archive.infolist():
                relative = entry.filename.removeprefix(prefix)
                if entry.is_dir() or relative == manifest_name:
                    continue
                if relative in updates:
                    with updates[relative].open('rb') as src, packed.open(entry, 'w', force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
                else:
                    with archive.open(entry) as src, packed.open(entry, 'w', force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
            entry = zipfile.ZipInfo(prefix + manifest_name, date_time=(2026, 11, 5, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            packed.writestr(entry, manifest_bytes)
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != expected_members:
            raise ValueError('Output inventory changed')
        for relative, record in records.items():
            entry = archive.getinfo(prefix + relative)
            with archive.open(entry) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != record['sha256'] or entry.file_size != record.get('size', record.get('bytes')):
                raise ValueError('Output manifest verification failed: ' + relative)
    return {'filename': output.name, 'bytes': output.stat().st_size, 'sha256': sha(output),
            'prefix': prefix, 'manifest': manifest_name,
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'manifest_files': len(records), 'previous_archive_sha256': expected,
            'updated_files': sorted(updates), 'usd_checks': checks,
            'all_other_members_byte_identical': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campus', type=Path, required=True)
    parser.add_argument('--exhibits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for kind, source, filename in (
        ('exhibits', args.exhibits, 'OV-Library-Lab-prepared-exhibits-portable.zip'),
        ('campus', args.campus, 'nvidia-campus-r48-portable-provenance.zip'),
    ):
        scratch = args.output / ('work-' + kind)
        scratch.mkdir()
        report[kind] = package(source, args.output / filename, scratch, kind)
        (args.output / 'portability-report.json').write_bytes(encoded(report))
        print('VERIFIED ' + filename, flush=True)


if __name__ == '__main__':
    main()
