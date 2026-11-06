# SPDX-License-Identifier: Apache-2.0
"""Typed updates validate whole logical values before native mutation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from renderer_values import prepare_value, tensor_value


def value(kind, raw, attribute='inputs:base_color'):
    return {'prim_path': '/World/Shader', 'attribute_name': attribute,
            'value_type': kind, 'value': raw}


@pytest.mark.parametrize('kind,raw', [('Float', float('nan')), ('Int', 2**40),
                                    ('Int', .5), ('Bool', 2), ('Color3f', [1,2]),
                                    ('Float2Array', [1,2,3,4])])
def test_invalid_values_are_rejected_before_native_writes(kind, raw):
    with pytest.raises(ValueError):
        prepare_value(value(kind, raw))


def test_transform_updates_target_native_hierarchy_input_not_derived_world_matrix():
    result = prepare_value(value('Matrix4d', np.eye(4).tolist(), 'omni:xform'))
    assert result[1] == 'omni:xform'
    assert result[5].dtype == np.float64
    assert result[5].shape == (1,16)


def test_uv_array_uses_explicit_two_lane_tensor():
    calls = []
    ovs = SimpleNamespace(AttributeSemantic=SimpleNamespace(NONE=0),
        DLDataTypeCode=SimpleNamespace(kDLFloat=2),
        DLDataType=lambda *args: args,
        make_dltensor=lambda data, **kwargs: calls.append((data, kwargs)) or 'tensor')
    tensor, is_array, semantic = tensor_value(prepare_value(value('Float2Array', [[0,1],[1,0]], 'primvars:st')), ovs, None)
    assert (tensor, is_array, semantic) == ('tensor', True, 0)
    assert calls[0][1] == {'dtype': (2,32,2), 'shape': [2]}


def _semantics():
    return SimpleNamespace(AttributeSemantic=SimpleNamespace(NONE=0, COLOR=8, VECTOR=6,
                           STRING=13, TOKEN_ID=2, ASSET_PATH_ID=1))


def test_existing_light_vector_role_is_preserved_for_color_value():
    data, is_array, semantic = tensor_value(prepare_value(value('Color3f',[.1,.2,.8])),
        _semantics(), None, np.float32, existing_semantic=6, existing_is_array=False, existing_width=3)
    assert semantic == 6 and not is_array
    assert np.allclose(data, [[.1,.2,.8]])


@pytest.mark.parametrize('kind,raw,dtype', [('Float',1.75,np.int32), ('Int',257,np.uint8),
    ('Int',-1,np.uint8), ('Double',1e100,np.float32), ('Float',1,np.bool_)])
def test_existing_storage_rejects_lossy_type_or_overflow_before_cast(kind,raw,dtype):
    with pytest.raises(ValueError):
        tensor_value(prepare_value(value(kind,raw)),_semantics(),None,dtype)


def test_float_width_conversion_and_boolean_byte_storage_are_allowed():
    data, _, _ = tensor_value(prepare_value(value('Double',.25)),_semantics(),None,np.float32)
    assert data.dtype == np.float32 and data[0] == .25
    data, _, _ = tensor_value(prepare_value(value('Bool',True)),_semantics(),None,np.uint8)
    assert data.dtype == np.uint8 and data[0] == 1


@pytest.mark.parametrize('array,semantic,width', [(False,13,1),(True,0,1),(True,13,2)])
def test_string_encoding_cannot_bypass_existing_schema_validation(array,semantic,width):
    with pytest.raises(ValueError):
        tensor_value(prepare_value(value('String','hello')),_semantics(),None,np.uint8,
                     existing_is_array=array,existing_semantic=semantic,existing_width=width)


@pytest.mark.parametrize('targets', [[], ['/Looks/A'], ['/Looks/B', '/Looks/A']])
def test_relationships_preserve_empty_and_multiple_ordered_targets(targets):
    ovs = SimpleNamespace(AttributeSemantic=SimpleNamespace(RELATIONSHIP_PATH_ID=14))
    paths = SimpleNamespace(intern_path=lambda path: {'/Looks/A': 31, '/Looks/B': 47}[path])
    data, is_array, semantic = tensor_value(
        prepare_value(value('Relationship', targets, 'material:binding')), ovs, paths,
        np.uint64, existing_semantic=14, existing_is_array=True, existing_width=1)
    assert data.dtype == np.uint64 and data.tolist() == [paths.intern_path(p) for p in targets]
    assert is_array and semantic == 14


@pytest.mark.parametrize('targets', [['relative'], ['/'], ['/Looks/A', 12], '/Looks/A'])
def test_invalid_relationship_target_never_reaches_native_write(targets):
    with pytest.raises(ValueError, match='ordered list'):
        prepare_value(value('Relationship', targets, 'material:binding'))


def test_usd_extent_encodes_native_fixed_double6_bounds():
    bounds = [[-1, -2, -3], [4, 5, 6]]
    prepared = prepare_value(value('Float3Array', bounds, 'extent'))
    data, is_array, semantic = tensor_value(prepared, _semantics(), None, np.float64,
        existing_semantic=0, existing_is_array=False, existing_width=6)
    assert data.dtype == np.float64 and data.shape == (1, 6)
    assert data.tolist() == [[-1, -2, -3, 4, 5, 6]]
    assert not is_array and semantic == 0


@pytest.mark.parametrize('bounds', [[], [[0, 0, 0]], [[0, 0, 0]] * 3,
                                   [[1, 0, 0], [0, 1, 1]]])
def test_extent_rejects_malformed_or_inverted_bounds(bounds):
    prepared = prepare_value(value('Float3Array', bounds, 'extent'))
    with pytest.raises(ValueError, match='extent'):
        tensor_value(prepared, _semantics(), None)


def test_extent_does_not_weaken_existing_native_schema_check():
    prepared = prepare_value(value('Float3Array', [[0, 0, 0], [1, 1, 1]], 'extent'))
    with pytest.raises(ValueError, match='fixed/array'):
        tensor_value(prepared, _semantics(), None, np.float32,
            existing_semantic=0, existing_is_array=True, existing_width=3)


def test_mesh_primvar_preserves_native_role_and_interpolation_storage_width():
    calls = []
    ovs = SimpleNamespace(AttributeSemantic=SimpleNamespace(NONE=0, TEXTURE_COORDINATE=16),
        DLDataTypeCode=SimpleNamespace(kDLFloat=2), DLDataType=lambda *args: args,
        make_dltensor=lambda data, **kwargs: calls.append((data, kwargs)) or 'tensor')
    prepared = prepare_value(value('TexCoord2fArray', [[0, 1], [1, 0]], 'primvars:st'))
    data, array, semantic = tensor_value(prepared, ovs, None, np.float32,
        existing_semantic=6, existing_is_array=True, existing_width=2)
    assert data == 'tensor' and array and semantic == 6
    assert calls[0][1] == {'dtype': (2, 32, 2), 'shape': [2]}
    with pytest.raises(ValueError, match='tuple width'):
        tensor_value(prepared, ovs, None, np.float32,
            existing_semantic=6, existing_is_array=True, existing_width=3)


@pytest.mark.parametrize('name,kind,raw,is_array', [
    ('usd-prim-type', 'Token', 'Mesh', False),
    ('usd-schemas', 'TokenArray', ['MaterialBindingAPI'], True),
    ('usd-schemas', 'TokenArray', [], True),
])
def test_reserved_metadata_uses_canonical_none_semantic(name, kind, raw, is_array):
    paths = SimpleNamespace(intern_token=lambda token: 123)
    data, actual_array, semantic = tensor_value(prepare_value(value(kind, raw, name)),
        _semantics(), paths)
    assert data.dtype == np.uint64 and actual_array == is_array and semantic == 0


@pytest.mark.parametrize('name,kind,raw', [
    ('usd-prim-type', 'TokenArray', ['Mesh']),
    ('usd-schemas', 'Token', 'MaterialBindingAPI'),
])
def test_reserved_metadata_rejects_wrong_kind_before_native_mutation(name, kind, raw):
    with pytest.raises(ValueError, match='reserved metadata kind'):
        tensor_value(prepare_value(value(kind, raw, name)), _semantics(),
                     SimpleNamespace(intern_token=lambda token: 123))


@pytest.mark.parametrize('kind,dtype', [('QuathArray', np.float16),
                                      ('QuatfArray', np.float32), ('QuatdArray', np.float64)])
def test_quaternion_storage_preserves_declared_precision_and_xyzw_order(kind, dtype):
    prepared = prepare_value(value(kind, [[0, 0, 0, 1]], 'orientations'))
    assert prepared[5].dtype == dtype
    assert prepared[5].tolist() == [[0, 0, 0, 1]]
