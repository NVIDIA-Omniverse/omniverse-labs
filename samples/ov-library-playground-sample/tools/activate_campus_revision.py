# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and locally activate a separately pinned authored campus revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(revision):
    spec=json.loads((ROOT/'config/campus-revisions.json').read_text(encoding='utf-8'))['revisions'][revision]
    directory=(ROOT/'data/campus'/spec['directory']).resolve()
    if not directory.is_relative_to((ROOT/'data/campus').resolve()):
        raise ValueError('Revision escapes the campus directory')
    manifest=directory/'manifest.json'
    if not spec['manifest_sha256'] or digest(manifest)!=spec['manifest_sha256']:
        raise ValueError('Revision manifest does not match its reviewed pin')
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if data['revision']!=revision or data['layout']!=spec['layout']:
        raise ValueError('Scene and navigation layout disagree')
    for item in data['files']:
        path=(directory/item['path']).resolve()
        if not path.is_relative_to(directory) or path.is_symlink():
            raise ValueError('Unsafe scene dependency')
        if path.stat().st_size!=item['size'] or digest(path)!=item['sha256']:
            raise ValueError('Scene dependency changed: '+item['path'])
    return len(data['files'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision',help='Registered revision, or r45 to restore the original')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.revision=='r45':
        from scene_assets import verify_install
        old_pin=ROOT/'config/scene-release-r45.json'
        pin=json.loads((old_pin if old_pin.is_file() else ROOT/'config/scene-release.json').read_text())['campus']
        verify_install(ROOT/'data/campus/nvidia-campus-r45-portable',pin)
        count=140
    else:
        count=verify(args.revision)
    if not args.verify_only:
        target=ROOT/'data/campus/active-revision.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps({'revision':args.revision},indent=2)+'\n',encoding='utf-8')
        os.replace(temporary,target)
    print(('Verified' if args.verify_only else 'Activated'),args.revision,count,'files',flush=True)


if __name__=='__main__':
    main()
