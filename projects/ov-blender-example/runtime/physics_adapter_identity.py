# SPDX-License-Identifier: Apache-2.0
"""Identity of the complete effective physics adapter, independent of rendering.

``adapter_identity`` hashes the adapter sources. It runs only when the record
is generated or checked (development, tests, package build), never at worker
start. Workers read ``physics_adapter_identity.json`` via ``recorded_identity``.
Regenerate the record after changing a component:
``python runtime/physics_adapter_identity.py --write``.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

RECORD_NAME = 'physics_adapter_identity.json'
COMPONENTS = ('public_sdk_backend.py', 'incremental_physics_backend.py',
              'inline_physics_geometry.py', 'renderer_values.py',
              'physics_adapter_identity.py', 'runtime_identity.py',
              'public_sdk_worker.py')


def _combined(components):
    return hashlib.sha256(json.dumps(components, sort_keys=True).encode('utf-8')).hexdigest()


def adapter_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).parent
    components = {}
    for filename in COMPONENTS:
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        if filename == 'public_sdk_backend.py':
            tree = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'PhysicsBackend')
        components[filename] = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode('utf-8')).hexdigest()
    return _combined(components), components


def record_bytes(root=None):
    digest, components = adapter_identity(root)
    record = {'schema': 1, 'adapter_sha256': digest, 'adapter_components': components}
    return (json.dumps(record, indent=2, sort_keys=True) + '\n').encode('utf-8')


def recorded_identity(root=None):
    """Return the recorded (digest, components) without reading any source."""
    root = Path(root) if root is not None else Path(__file__).parent
    record = json.loads((root / RECORD_NAME).read_text(encoding='utf-8'))
    if not isinstance(record, dict):
        record = {}
    components = record.get('adapter_components')
    digest = record.get('adapter_sha256')
    if (record.get('schema') != 1 or not isinstance(components, dict)
            or set(components) != set(COMPONENTS)
            or not all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                       for value in (digest, *components.values()))
            or _combined(components) != digest):
        raise RuntimeError(f'Invalid physics adapter identity record: {RECORD_NAME}')
    return digest, dict(components)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Write or check the recorded physics adapter identity')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    expected = record_bytes(args.root)
    path = args.root / RECORD_NAME
    if args.write:
        path.write_bytes(expected)
    elif not path.is_file() or path.read_bytes() != expected:
        print(f'{path} is stale; run with --write', file=sys.stderr)
        return 1
    print(json.loads(expected)['adapter_sha256'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
