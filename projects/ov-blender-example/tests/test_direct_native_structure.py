# SPDX-License-Identifier: Apache-2.0
"""Structural probes validate complete packets and preserve native ownership."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'runtime'))
import direct_native_structure as structure
from public_sdk_backend import RendererBackend


def fixture(monkeypatch):
    events = []
    schema = {'usd-prim-type': dict(dtype=np.dtype('uint64'), semantic=2, is_array=False,
                                  width=1, shape=[1], preview=[1])}
    columns = {'/World': schema, '/World/Mesh': schema, '/World/Material': schema,
               '/World/Shader': schema}
    def read(backend, paths, path, names=None):
        events.append(('read', path))
        return columns.get(path, {})
    monkeypatch.setattr(structure, '_columns', read)
    paths = SimpleNamespace(intern_token=lambda name: 1, intern_path=lambda name: 2,
                            create_path_list_from_strings=lambda names: nullcontext(names))
    ovs = SimpleNamespace(PathDictionary=lambda stage: nullcontext(paths),
        AttributeSemantic=SimpleNamespace(NONE=0, TOKEN_ID=2, RELATIONSHIP_PATH_ID=4,
                                         CONNECTION_PATH_ID=12, MATRIX=5),
        Scope=SimpleNamespace(ALL=0), DLDataTypeCode=SimpleNamespace(kDLUInt=1),
        DLDataType=lambda *args: args,
        make_dltensor=lambda data, **kwargs: (data, kwargs))
    def op(name, *args):
        events.append((name, *args))
        return SimpleNamespace(wait=lambda: events.append(('wait', name)))
    backend = RendererBackend.__new__(RendererBackend)
    backend.ovstage = ovs; backend.np = np; backend.ordinal = 4; backend.native_stage_id = 'retained'
    backend.pending = {}; backend.simulation_time_ns = 0; backend.sample_count = 8
    backend.native_pose_fanout = object(); backend.desired_values = {}
    backend.stage = SimpleNamespace(clone_async=lambda *args: op('clone', *args),
        query_from_path_list=lambda paths: nullcontext(paths),
        delete_attributes=lambda *args: op('delete', *args),
        advance_write_floor=lambda *args: op('floor', *args))
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal: events.append(('pull', ordinal)),
                                       reset=lambda **kwargs: events.append(('reset',)))
    backend._write_phase = lambda paths, rows: events.extend(('write', row[0], row[1], row[3]) for row in rows)
    return backend, events, columns


def packet(**kwargs):
    return dict(experimental=True, **kwargs)


def test_clone_and_delete_are_one_published_ordinal_with_native_stage_retained(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    from native_pose_fanout import NativePoseFanout
    backend.native_pose_fanout = NativePoseFanout(backend.stage, backend.ovstage, render_only=True)
    backend.native_pose_fanout.synthetic_identities = {'/World/Mesh/Child': ('Mesh', False),
                                                       '/World/Other/Child': ('Mesh', False)}
    result = structure.apply(backend, packet(
        clones=[dict(source='/World/Mesh', targets=['/World/Copy'])], deletes=['/World/Mesh']))
    assert result['native_stage_id'] == 'retained' and result['ordinal'] == 5
    assert not result['production_enabled'] and result['source_projection_stale']
    assert [event[0] for event in events if event[0] != 'read'] == [
        'clone', 'wait', 'delete', 'wait', 'floor', 'wait', 'pull', 'reset']
    # The retained stage keeps untouched synthetic world inputs; deleted ones go.
    assert backend.native_pose_fanout.synthetic_identities == {'/World/Other/Child': ('Mesh', False)}
    assert backend.sample_count == 0


def test_create_metadata_and_reset_boundary_precede_numeric_values(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    result = structure.apply(backend, packet(creates=[dict(prim_path='/World/New', prim_type='Xform',
        schemas=[], values=[dict(attribute_name='omni:xform', value_type='Matrix4d', value=np.eye(4).tolist()),
                            dict(attribute_name='omni:resetXformStack', value_type='Bool', value=True)])]))
    assert [event[2] for event in events if event[0] == 'write'] == [
        'usd-prim-type', 'usd-schemas', 'omni:resetXformStack', 'omni:xform']
    assert result['create_count'] == 1 and result['written_attributes'] == 4


@pytest.mark.parametrize('change', [
    {'experimental': False},
    {'creates': [dict(prim_path='/World/Mesh', prim_type='Mesh')]},
    {'clones': [dict(source='/World/Missing', targets=['/World/Copy'])]},
    {'clones': [dict(source='/World/Mesh', targets=['/World/Mesh'])]},
    {'creates': [dict(prim_path='/World/New', prim_type='Shader')],
     'values': [dict(prim_path='/World/Shader', attribute_name='inputs:x', value_type='Float', value=float('nan'))]},
])
def test_invalid_packet_cannot_mutate_prefix(monkeypatch, change):
    backend, events, columns = fixture(monkeypatch)
    with pytest.raises(ValueError):
        structure.apply(backend, {**packet(), **change})
    assert backend.ordinal == 4
    assert all(event[0] == 'read' for event in events)


def test_connection_preserves_native_pairs_and_rejects_regular_float_column(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    rows = [dict(prim_path='/World/Shader', attribute_name='inputs:surfaceConnection', value_type='Connection',
                 value=[dict(prim_path='/World/OtherShader', attribute_name='outputs:surface')])]
    columns['/World/Shader'] = {**columns['/World/Shader'], 'inputs:surfaceConnection': dict(
        dtype=np.dtype('float32'), semantic=0, is_array=False, width=1)}
    with pytest.raises(ValueError, match='not a native connection'):
        structure.apply(backend, packet(values=rows))
    assert backend.ordinal == 4
    columns['/World/Shader']['inputs:surfaceConnection'] = dict(
        dtype=np.dtype('uint64'), semantic=12, is_array=True, width=2)
    written = []
    backend._write_phase = lambda paths, rows: written.extend(rows)
    structure.apply(backend, packet(values=rows))
    data, layout = written[0][2]
    assert data.tolist() == [[2, 1]] and layout['dtype'] == (1, 64, 2)
    assert written[0][3:] == (True, 12)


def test_failed_native_structure_requires_reload_before_rendering(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    def fail(*args):
        raise RuntimeError('Native clone failed')
    backend.stage.clone_async = fail
    with pytest.raises(RuntimeError, match='clone failed'):
        structure.apply(backend, packet(clones=[dict(source='/World/Mesh', targets=['/World/Copy'])]))
    assert backend.experimental_structure_uncertain
    with pytest.raises(RuntimeError, match='reload'):
        backend.dispatch('render', {})
    assert not any(event[0] == 'floor' for event in events)


def test_missing_target_in_later_create_is_validated_before_first_create(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    with pytest.raises(ValueError, match='target does not exist'):
        structure.apply(backend, packet(creates=[dict(prim_path='/World/New', prim_type='Mesh')],
            values=[dict(prim_path='/Missing', attribute_name='points', value_type='Float', value=1)]))
    assert backend.ordinal == 4 and all(event[0] == 'read' for event in events)


def test_column_discovery_uses_exact_path_filter_not_unevaluated_path_list():
    events = []
    query = SimpleNamespace(wait=lambda: events.append('wait'),
                            result=lambda: SimpleNamespace(attributes=[], total_prim_count=0))
    def discover(*, filter):
        events.append(filter)
        return nullcontext(query)
    ovs = SimpleNamespace(Filter=lambda predicates: predicates,
        Predicate=lambda *args: args, FilterOp=SimpleNamespace(IN='in'))
    backend = SimpleNamespace(ovstage=ovs, stage=SimpleNamespace(query=discover))
    assert structure._columns(backend, None, '/World/Shader') == {}
    assert events == [[('usd-path', 'in', ['/World/Shader'])], 'wait']


def test_cloning_does_not_invalidate_original_sources_accepted_values(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    key = ('/World/Mesh', 'material:binding')
    backend.desired_values = {key: {'value': ['/World/Material']}}
    structure.apply(backend, packet(clones=[dict(source='/World/Mesh', targets=['/World/Copy'])]))
    assert backend.desired_values == {key: {'value': ['/World/Material']}}


def test_unvalidated_point_instancer_create_is_rejected_before_native_calls(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    with pytest.raises(ValueError, match='PointInstancer creation is unsupported'):
        structure.apply(backend, packet(creates=[dict(prim_path='/World/Instances', prim_type='PointInstancer')]))
    assert events == [] and backend.ordinal == 4


def test_inspection_decodes_native_asset_authored_resolved_string_pair(monkeypatch):
    backend, events, columns = fixture(monkeypatch)
    columns['/World/Shader'] = {'inputs:file': dict(dtype=np.dtype('uint64'), semantic=1,
        is_array=False, width=2, shape=[2], preview=[11, 12])}
    paths = SimpleNamespace(token_to_string=lambda token: {11: './checker.png',12: '/assets/checker.png'}[token])
    backend.ovstage.PathDictionary = lambda stage: nullcontext(paths)
    result = structure.inspect(backend, {'prim_paths': ['/World/Shader'], 'attributes': ['inputs:file']})
    assert result['columns']['/World/Shader']['inputs:file']['asset_paths'] == [
        dict(authored='./checker.png', resolved='/assets/checker.png')]
    assert all(event[0] == 'read' for event in events)
