# SPDX-License-Identifier: Apache-2.0
"""Explicit USD-value to public OVStage tensor layouts for renderer edits."""
from __future__ import annotations

import re
import numpy as np


def prepare_value(value):
    """Validate the complete logical value before any stage write is enqueued."""
    path = value.get('prim_path')
    name = value.get('attribute_name', value.get('attribute'))
    if not isinstance(path, str) or not path.startswith('/') or path == '/':
        raise ValueError('Update requires an absolute non-root prim path')
    if not isinstance(name, str) or not name:
        raise ValueError('Update requires an attribute name')
    raw = value['value']
    kind = str(value.get('value_type', '')).lower()
    array = kind.endswith('array') or kind.endswith('[]')
    base = kind.removesuffix('array').removesuffix('[]')
    if not kind:
        base = 'bool' if isinstance(raw, bool) else 'int' if isinstance(raw, int) else 'token' if isinstance(raw, str) else 'float'
    roles = {'color': 'COLOR', 'point': 'POINT', 'vector': 'VECTOR', 'normal': 'NORMAL', 'texcoord': 'TEXTURE_COORDINATE'}
    role = next((v for k, v in roles.items() if base.startswith(k)), 'NONE')
    if base == 'relationship':
        if not isinstance(raw, (list, tuple)) or not all(
                isinstance(v, str) and v.startswith('/') and v != '/' for v in raw):
            raise ValueError('Relationship requires an ordered list of absolute target paths')
        return path, name, base, True, 'RELATIONSHIP_PATH_ID', tuple(raw), 1
    if base in ('token', 'string', 'asset'):
        if base == 'string' and array:
            raise ValueError('String-array edits require an explicit string-column encoding')
        items = raw if array else [raw]
        if not isinstance(items, (list, tuple)) or not all(isinstance(v, str) for v in items):
            raise ValueError(f'{kind} requires string values')
        return path, name, base, array, role, tuple(items), 1
    if 'matrix' in base:
        width, dtype, role = 16, np.float64, 'MATRIX'
        if base not in ('matrix4d', 'matrix4f'):
            raise ValueError(f'Unsupported matrix type: {kind}')
    elif base.startswith('quat'):
        if base not in ('quatd', 'quatf', 'quath'):
            raise ValueError(f'Unsupported quaternion type: {kind}')
        width, dtype, role = 4, {'quatd': np.float64, 'quatf': np.float32, 'quath': np.float16}[base], 'QUATERNION'
    else:
        match = re.fullmatch(r'(?:float|double|half|int|uint|color|point|vector|normal|texcoord)([234])?[fdh]?', base)
        if base == 'bool':
            width, dtype = 1, np.bool_
        elif match:
            width = int(match.group(1) or 1)
            dtype = np.float64 if base.startswith('double') or base.endswith('d') else np.float16 if base.startswith('half') or base.endswith('h') else np.uint32 if base.startswith('uint') else np.int32 if base.startswith('int') else np.float32
        else:
            raise ValueError(f'Unsupported renderer value type: {kind}')
    numeric = np.asarray(raw)
    if numeric.dtype.kind not in 'biuf' or not np.isfinite(numeric).all():
        raise ValueError('Update requires finite numeric values')
    if array:
        expected = (numeric.size // width, width) if width > 1 else (numeric.size,)
        if numeric.size % width or (numeric.size and width > 1 and tuple(numeric.shape) != expected):
            raise ValueError(f'{kind} requires an array of {width}-component elements')
    elif numeric.size != width:
        raise ValueError(f'{kind} requires exactly {width} component(s)')
    if np.dtype(dtype).kind in 'iu':
        limits = np.iinfo(dtype)
        if ((numeric < limits.min) | (numeric > limits.max) | (numeric != np.floor(numeric))).any():
            raise ValueError(f'{kind} values exceed the integer representation')
    if dtype == np.bool_ and not np.isin(numeric, [False, True]).all():
        raise ValueError('Bool values must be true or false')
    data = np.asarray(raw, dtype=dtype).reshape((-1, width) if array else (1, width))
    if width == 1:
        data = data.reshape(-1)
    if not np.isfinite(data).all():
        raise ValueError('Update overflows its declared value type')
    # omni:xform is the native hierarchy INPUT. Fabric worldMatrix is a
    # derived read-only output; writing it does not propagate to child meshes.
    return path, name, base, array, role, data, width


def tensor_value(prepared, ovs, paths, existing_dtype=None, *, existing_semantic=None,
                 existing_is_array=None, existing_width=None):
    """Construct lane-aware array tensors; preserve existing schema storage."""
    path, name, base, array, role, data, width = prepared
    semantic = getattr(ovs.AttributeSemantic, role)
    if name == 'extent' and array:
        # Rendering population stores USD float3[2] bounds as a fixed double6,
        # unlike ordinary variable-length vec3 arrays. Keep callers USD-typed
        # and adapt only this documented native representation here.
        if width != 3 or np.asarray(data).shape != (2, 3) or np.asarray(data).dtype.kind != 'f':
            raise ValueError(f'{path}.extent requires two floating-point vec3 bounds')
        if np.any(data[0] > data[1]):
            raise ValueError(f'{path}.extent minimum must not exceed maximum')
        data = np.asarray(data, dtype=np.float64).reshape(1, 6)
        width, array, semantic = 6, False, ovs.AttributeSemantic.NONE
    elif base == 'relationship':
        data = np.asarray([paths.intern_path(v) for v in data], dtype=np.uint64)
    elif base == 'token':
        data = np.asarray([paths.intern_token(v) for v in data], dtype=np.uint64)
        # Reserved metadata uses token IDs but requires semantic NONE in the
        # installed native API; stamping TOKEN_ID is explicitly rejected.
        if name in {'usd-prim-type', 'usd-schemas'}:
            if array != (name == 'usd-schemas'):
                raise ValueError(f'{name} has an invalid reserved metadata kind')
            semantic = ovs.AttributeSemantic.NONE
        else:
            semantic = ovs.AttributeSemantic.TOKEN_ID
    elif base == 'asset':
        # Authored and resolved token IDs are separate lanes in public OVStage.
        data = np.asarray([[paths.intern_token(v), paths.intern_token(v)] for v in data], dtype=np.uint64)
        width, semantic = 2, ovs.AttributeSemantic.ASSET_PATH_ID
    elif base == 'string':
        if array:
            raise ValueError('String-array edits require an explicit string-column encoding')
        data = np.frombuffer(data[0].encode('utf-8'), dtype=np.uint8).copy()
        array, width, semantic = True, 1, ovs.AttributeSemantic.STRING
    if existing_dtype is not None:
        destination = np.dtype(existing_dtype)
        if base in ('token', 'asset', 'string', 'relationship'):
            if destination != data.dtype:
                raise ValueError(f"{path}.{name}: ID/string storage type differs from the existing column")
        elif data.dtype.kind == 'f':
            if destination.kind != 'f':
                raise ValueError(f"{path}.{name}: floating value cannot replace an integer or Boolean column")
            limit = np.finfo(destination).max
            if np.any(np.abs(data) > limit):
                raise ValueError(f"{path}.{name}: value overflows the existing column")
        elif data.dtype.kind in 'iu':
            if destination.kind not in 'iu':
                raise ValueError(f"{path}.{name}: integer value requires an integer column")
            bounds = np.iinfo(destination)
            if any(int(v) < bounds.min or int(v) > bounds.max for v in data.flat):
                raise ValueError(f"{path}.{name}: integer overflows the existing column")
        elif data.dtype.kind == 'b' and destination.kind not in 'biu':
            raise ValueError(f"{path}.{name}: Boolean value requires Boolean/integer storage")
        data = data.astype(destination, copy=False)
    if existing_is_array is not None and array != existing_is_array:
        raise ValueError(f"{path}.{name}: fixed/array kind differs from the existing column")
    if existing_width is not None and width != existing_width:
        raise ValueError(f"{path}.{name}: tuple width differs from the existing column")
    if existing_semantic is not None:
        if base in ('token', 'asset', 'string', 'relationship') and int(existing_semantic) != int(semantic):
            raise ValueError(f"{path}.{name}: ID storage semantic differs from the existing column")
        # Logical Blender colors can target existing VECTOR columns.
        # Existing roles are immutable; retain the schema read from OVStage.
        semantic = existing_semantic
    if data.dtype.kind in 'biuf' and not np.isfinite(data).all():
        raise ValueError(f"{path}.{name}: value overflows the existing column")
    if array and width > 1:
        code = ovs.DLDataTypeCode.kDLFloat if data.dtype.kind == 'f' else ovs.DLDataTypeCode.kDLUInt if data.dtype.kind == 'u' else ovs.DLDataTypeCode.kDLInt
        data = ovs.make_dltensor(data, dtype=ovs.DLDataType(code, data.dtype.itemsize * 8, width), shape=[len(data)])
    return data, array, semantic


def tensor_width(tensor):
    """Components per element of an OVStage read tensor.

    OVStage 0.2 encoded the tuple width in ``dtype.lanes`` with a flat shape;
    0.3 reads use scalar lanes and typed trailing dimensions such as (N, 3) or
    (N, 4, 4). The product covers both layouts.
    """
    width = int(tensor.dtype.lanes)
    for extent in getattr(tensor, 'shape_tuple', ())[1:]:
        width *= int(extent)
    return width
