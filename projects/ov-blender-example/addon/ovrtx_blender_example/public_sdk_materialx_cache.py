# SPDX-License-Identifier: Apache-2.0
"""Bounded locked Windows/Linux MaterialX pairs, not artifact provenance.

The vendor format uses domain-separated FNV128 to detect inconsistent pairs.
These are noncryptographic consistency checks; MDL is generated shading code.
Every shipped byte remains governed by the external sidecar manifest instead.
"""
import re

MAX_METADATA_BYTES = 256 * 1024
MAX_MDL_BYTES = 4 * 1024**2
_HEX = r'[0-9a-f]{32}'
_MODULE = r'omni_mtlx_v2_' + _HEX
_BASIS = 0x6c62272e07bb014262b821756295c58d
_PRIME = 0x1000000000000000000013b
_MASK = (1 << 128) - 1


def _fnv(data, seed=_BASIS):
    for byte in data:
        seed = ((seed ^ byte) * _PRIME) & _MASK
    return seed


def persistent_digest(domain, data):
    seed = _fnv(domain.encode('ascii'))
    seed = _fnv(len(data).to_bytes(8, 'little'), seed)
    seed = _fnv(data, seed)
    # Carbonite prints its low 64-bit word first, then the high word.
    return f'{seed & ((1 << 64) - 1):016x}{seed >> 64:016x}'


def cache_kind(relative):
    if re.fullmatch(r'\.metadata/' + _HEX + r'\.metadata', relative):
        return 'metadata', MAX_METADATA_BYTES
    if re.fullmatch(_MODULE + r'/materialx\.mdl', relative):
        return 'mdl', MAX_MDL_BYTES
    return None, 0


def inspect_cache(relative, data):
    """Parse only the exact measured version, with bounded text/record counts."""
    kind, limit = cache_kind(relative)
    if kind is None or not data or len(data) > limit:
        raise ValueError('Unexpected MaterialX cache path or size')
    if b'\x00' in data or any(byte < 32 and byte not in (9, 10, 13) for byte in data):
        raise ValueError('Non-text bytes in MaterialX generated cache')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('Invalid MaterialX cache text') from error
    if kind == 'mdl':
        if not text.startswith('mdl 1.10;\n'):
            raise ValueError('Unexpected generated MaterialX MDL version/header')
        return dict(kind=kind, module=relative.split('/')[0],
                    mdl_digest=persistent_digest('omni.materialx.persistent-mdl.v2', data))
    # Serializer emits canonical LF text. Reject alternate whitespace, duplicate
    # rows, malformed fields, oversized counts and trailing records explicitly.
    if '\r' in text or not text.endswith('\n'):
        raise ValueError('Unexpected MaterialX metadata serialization')
    lines = text.splitlines()
    if len(lines) < 8 or lines[0] != 'rtx.materialx.metadata.v2':
        raise ValueError('Unexpected MaterialX metadata schema')
    fields = {}
    for line, label, pattern in zip(lines[1:6], ('lookup', 'generator', 'module', 'mdl', 'metadata'),
                                    (_HEX, _HEX, _MODULE, _HEX, _HEX)):
        match = re.fullmatch(label + ' (' + pattern + ')', line)
        if not match:
            raise ValueError('Invalid MaterialX metadata identity field')
        fields[label] = match.group(1)
    if relative != '.metadata/' + fields['lookup'] + '.metadata':
        raise ValueError('MaterialX lookup filename disagrees with metadata')
    cursor = 6
    for plural, singular, width in (('nodes', 'node', 2), ('ports', 'port', 3)):
        if cursor >= len(lines):
            raise ValueError('Missing MaterialX metadata records')
        count = re.fullmatch(plural + r' (0|[1-9][0-9]{0,4})', lines[cursor])
        if not count or int(count.group(1)) > 4096:
            raise ValueError('MaterialX metadata record count exceeds bound')
        number = int(count.group(1)); cursor += 1
        if number > len(lines) - cursor:
            raise ValueError('Truncated MaterialX metadata records')
        records = []
        for line in lines[cursor:cursor + number]:
            pieces = line.split(' ')
            if len(pieces) != width + 1 or pieces[0] != singular:
                raise ValueError('Invalid MaterialX metadata record')
            decoded = []
            for piece in pieces[1:]:
                if not re.fullmatch(r'(?:[0-9a-f]{2}){1,4096}', piece):
                    raise ValueError('Invalid MaterialX encoded identifier')
                value = bytes.fromhex(piece)
                if any(byte < 32 or byte == 127 for byte in value):
                    raise ValueError('Control byte in MaterialX identifier')
                decoded.append(value)
            records.append(tuple(decoded))
        if records != sorted(records) or len(records) != len(set(records)):
            raise ValueError('Noncanonical MaterialX metadata records')
        cursor += number
    if cursor != len(lines):
        raise ValueError('Trailing MaterialX metadata records')
    body = ('\n'.join(lines[6:]) + '\n').encode('utf-8')
    if fields['metadata'] != persistent_digest('omni.materialx.persistent-metadata.v2', body):
        raise ValueError('MaterialX metadata digest mismatch')
    return dict(kind=kind, module=fields['module'], mdl_digest=fields['mdl'])


def validate_pairs(records):
    modules = {row['module']: row['mdl_digest'] for row in records.values() if row['kind'] == 'mdl'}
    referenced = set()
    for row in records.values():
        if row['kind'] == 'metadata':
            if modules.get(row['module']) != row['mdl_digest']:
                raise ValueError('Missing or inconsistent MaterialX metadata/MDL pair')
            referenced.add(row['module'])
    if referenced != set(modules):
        raise ValueError('Unpaired generated MaterialX MDL')
