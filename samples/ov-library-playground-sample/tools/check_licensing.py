# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check license delivery and NVIDIA code headers without native SDK imports."""
import json
from pathlib import Path
import sys

CODE_SUFFIXES = {'.py', '.ps1', '.bat', '.cjs', '.js', '.yml', '.yaml'}
COPYRIGHT = 'SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
LICENSE_ID = 'SPDX-License-Identifier: Apache-2.0'
REQUIRED = ('LICENSE', 'NOTICE', 'CONTRIBUTING.md', 'THIRD_PARTY_NOTICES.md', 'docs/LICENSING.md')


def validate(root):
    root = Path(root)
    errors = []
    policy = root/'config/public-files.json'
    if policy.is_file():
        config = json.loads(policy.read_text(encoding='utf-8'))
        files = {n: n for n in config['files']} | config['overrides']
    else:
        config = json.loads((root/'config/sample-files.json').read_text(encoding='utf-8'))
        files = {n: n for n in config['files']}
    for name in REQUIRED:
        if name not in files or not (root/files[name]).is_file():
            errors.append('Missing public license/notice: '+name)
    for name, source in files.items():
        scene_override = name.startswith('data/scene-overrides/') and Path(name).suffix == '.usda'
        if Path(name).suffix not in CODE_SUFFIXES and not scene_override:
            continue
        if not scene_override and not name.startswith(('source/', 'scripts/', 'tools/', '.github/')):
            continue
        first = '\n'.join((root/source).read_text(encoding='utf-8-sig').splitlines()[:8])
        if scene_override and not first.startswith('#usda 1.0\n'):
            errors.append('Missing USD ASCII magic header: '+name)
        if COPYRIGHT not in first or LICENSE_ID not in first:
            errors.append('Missing NVIDIA software SPDX header: '+name)
    if (root/'LICENSE').is_file():
        text = (root/'LICENSE').read_text(encoding='utf-8')
        for marker in ('Apache License', 'Version 2.0, January 2004', 'CC-BY-4.0'):
            if marker not in text:
                errors.append('Incomplete source/documentation license text: '+marker)
    return errors


if __name__ == '__main__':
    issues = validate(Path(__file__).resolve().parents[1])
    if issues:
        print('\n'.join(issues))
        sys.exit(1)
    print('PASS: public NVIDIA software headers and license/notice delivery; approval is separate.')
