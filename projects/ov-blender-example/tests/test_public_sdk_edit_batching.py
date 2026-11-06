# SPDX-License-Identifier: Apache-2.0
"""Grouped native edits preserve validation, ordering and operation ownership."""
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))
from public_sdk_backend import RendererBackend
from native_pose_fanout import FanoutPlan


def backend_fixture(*, wait_failure=None):
    events = []
    closed = []
    active = set()
    backend = RendererBackend.__new__(RendererBackend)
    @contextmanager
    def query(paths):
        path = paths[0]
        events.append(("query", path))
        active.add(path)
        try:
            yield path
        finally:
            active.remove(path)
            closed.append(path)
            events.append(("close", path))
    def read(path, tokens, ordinal):
        events.append(("read", path, tuple(tokens)))
        groups = [SimpleNamespace(is_delete=False, attribute="usd-prim-type", prim_count=1)]
        return nullcontext(SimpleNamespace(wait=lambda: None, groups=lambda: groups))
    def write(path, descriptors, ordinal):
        names = tuple(item.attribute for item in descriptors)
        events.append(("submit", path, names))
        def wait():
            assert path in active
            events.append(("wait", path, names))
            if path == wait_failure:
                raise RuntimeError("native apply failed")
        return SimpleNamespace(wait=wait)
    backend.stage = SimpleNamespace(query_from_path_list=query, read_attributes=read,
        release_group=lambda group: None, write_attributes=write,
        advance_write_floor=lambda *args: events.append(("floor",)) or SimpleNamespace(wait=lambda: None))
    paths = SimpleNamespace(intern_token=lambda name: name,
        create_path_list_from_strings=lambda paths: nullcontext(paths))
    backend.ovstage = SimpleNamespace(PathDictionary=lambda stage: nullcontext(paths),
        OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal), Scope=SimpleNamespace(ALL=0),
        WriteDesc=lambda **kwargs: SimpleNamespace(**kwargs), AttributeSemantic=SimpleNamespace(NONE=0,MATRIX=5))
    backend.np=np; backend.ordinal=4; backend.desired_values={}; backend.sample_count=0
    backend.simulation_time_ns=0
    backend.renderer=SimpleNamespace(update_from_stage=lambda ordinal: events.append(("renderer",)), reset=lambda **kw: None)
    plan=FanoutPlan()
    backend.native_pose_fanout=SimpleNamespace(prepare=lambda *args, **kwargs: plan,accepted=lambda plan: None)
    # Schema read queries are released before writes start; test ownership only
    # for the later phase queries.
    original_read=backend.stage.read_attributes
    def read_reset(*args):
        closed.clear()
        return original_read(*args)
    backend.stage.read_attributes=read_reset
    return backend, events, closed


def row(path,name,value,kind="Float"):
    return dict(prim_path=path,attribute_name=name,value=value,value_type=kind)


def test_properties_share_schema_read_and_native_copy_operation():
    backend,events,closed=backend_fixture()
    backend.update([row("/Light","inputs:intensity",2),row("/Light","inputs:exposure",3)])
    assert len([event for event in events if event[0]=="read"])==1
    assert [event for event in events if event[0]=="submit"]==[("submit","/Light",("inputs:intensity","inputs:exposure"))]
    assert events.index(("wait","/Light",("inputs:intensity","inputs:exposure"))) < events.index(("floor",))


def test_non_transform_edit_skips_existing_pose_helper_and_reports_actual_schema_work():
    backend, events, closed = backend_fixture()
    def unwanted(*args, **kwargs):
        raise AssertionError('Non-transform edits must not reconcile hierarchy')
    backend.native_pose_fanout = SimpleNamespace(prepare=unwanted, accepted=unwanted)
    result = backend.update([row('/Light', 'inputs:intensity', 3),
                             row('/Light', 'inputs:exposure', 2)])
    assert result['native_schema_read_count'] == 1
    assert result['native_schema_attribute_count'] == 3
    assert result['native_update_timings']['total_ms'] >= result['native_update_timings']['schema_read_encode_ms']


def test_non_transform_edit_does_not_create_a_pose_helper():
    backend, events, closed = backend_fixture()
    backend.native_pose_fanout = None
    backend.update([row('/Light', 'inputs:intensity', 3)])
    assert backend.native_pose_fanout is None


@pytest.mark.parametrize('name,value,kind', [
    ('faceVertexIndices', [0, 2, 1], 'IntArray'),
    ('faceVertexCounts', [3], 'IntArray'),
    ('material:binding', ['/Looks/A', '/Looks/B'], 'Relationship'),
    ('material:binding', [], 'Relationship'),
    ('resolution', [64, 32], 'Int2'),
])
def test_geometry_binding_and_product_changes_reset_only_after_native_publication(name, value, kind):
    backend, events, closed = backend_fixture()
    backend.renderer.reset = lambda **kwargs: events.append(('reset',))
    # Native IDs are interned by the worker, never sent as client-chosen IDs.
    backend.ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID = 14
    original_paths = backend.ovstage.PathDictionary
    @contextmanager
    def paths(stage):
        with original_paths(stage) as dictionary:
            dictionary.intern_path = lambda path: {'/Looks/A': 1, '/Looks/B': 2}[path]
            yield dictionary
    backend.ovstage.PathDictionary = paths
    result = backend.update([row('/Mesh', name, value, kind)])
    assert events.index(('floor',)) < events.index(('renderer',)) < events.index(('reset',))
    assert result['updated_count'] == 1 and not result['native_transform_auto_invalidation']


def test_boundaries_complete_before_any_matrix_submission():
    backend,events,closed=backend_fixture()
    # Allow the same path in a later phase: the wait still precedes that phase's
    # query release, independently of the schema-read scope.
    # Query owners are held for each phase by the backend.
    values=[item for path in ("/A","/B") for item in (
        row(path,"omni:xform",np.eye(4).tolist(),"Matrix4d"),
        row(path,"omni:resetXformStack",True,"Bool"))]
    backend.update(values)
    boundary_waits=[i for i,event in enumerate(events) if event[0]=="wait" and event[2]==("omni:resetXformStack",)]
    matrix_submits=[i for i,event in enumerate(events) if event[0]=="submit" and event[2]==("omni:xform",)]
    assert max(boundary_waits)<min(matrix_submits)


def test_invalid_suffix_never_submits_valid_prefix():
    backend,events,closed=backend_fixture()
    with pytest.raises(ValueError):
        backend.update([row("/A","inputs:exposure",1),row("/B","inputs:exposure",float("nan"))])
    assert not any(event[0]=="submit" for event in events)
    assert backend.ordinal==4 and backend.desired_values=={}


def test_failed_operation_drains_peers_before_queries_release_and_never_commits_floor():
    backend,events,closed=backend_fixture(wait_failure="/A")
    with pytest.raises(RuntimeError,match="native apply failed"):
        backend.update([row("/A","inputs:exposure",1),row("/B","inputs:exposure",2)])
    assert [event[1] for event in events if event[0]=="wait"]==["/A","/B"]
    assert not any(event[0] in {"floor","renderer"} for event in events)
    assert backend.desired_values=={}
    # Both schema queries closed earlier; the final close of each belongs to
    # native apply and follows draining all peer operations.
    last_wait=max(i for i,event in enumerate(events) if event[0]=="wait")
    assert all(max(i for i,event in enumerate(events) if event==("close",path))>last_wait for path in ("/A","/B"))


def test_enqueue_failure_still_drains_previously_submitted_operation():
    backend,events,closed=backend_fixture()
    original=backend.stage.write_attributes
    def enqueue(path, descriptors, ordinal):
        if path=="/B":
            raise RuntimeError("native enqueue failed")
        return original(path,descriptors,ordinal)
    backend.stage.write_attributes=enqueue
    with pytest.raises(RuntimeError,match="native enqueue failed"):
        backend.update([row("/A","inputs:exposure",1),row("/B","inputs:exposure",2)])
    assert [event[1] for event in events if event[0]=="wait"]==["/A"]
    assert not any(event[0]=="floor" for event in events)


def test_duplicate_columns_keep_sequential_last_write_semantics():
    backend,events,closed=backend_fixture()
    def write(path,name,ordinal,data,**kwargs):
        events.append(("single",float(np.asarray(data).reshape(-1)[0])))
        return SimpleNamespace(wait=lambda:events.append(("single_wait",)))
    backend.stage.write_attribute=write
    backend.update([row("/A","inputs:exposure",1),row("/A","inputs:exposure",2)])
    assert [event for event in events if event[0]=="single"]==[("single",1),("single",2)]
    assert not any(event[0]=="submit" for event in events)
    assert backend.desired_values[("/A","inputs:exposure")]["value"]==2


def test_native_schema_invalid_suffix_rejects_entire_prevalidated_batch():
    backend,events,closed=backend_fixture()
    def read(path,tokens,ordinal):
        groups=[SimpleNamespace(is_delete=False,attribute="usd-prim-type",prim_count=1)]
        if path=="/B":
            groups.append(SimpleNamespace(is_delete=False,attribute="inputs:exposure",prim_count=1,
                array=lambda index:np.asarray([1],dtype=np.int32),raw=SimpleNamespace(semantic=0),
                is_array=False,tensor=lambda index:SimpleNamespace(dtype=SimpleNamespace(lanes=1))))
        return nullcontext(SimpleNamespace(wait=lambda:None,groups=lambda:groups))
    backend.stage.read_attributes=read
    with pytest.raises(ValueError):
        backend.update([row("/A","inputs:exposure",1),row("/B","inputs:exposure",1.5)])
    assert not any(event[0]=="submit" for event in events)
    assert backend.ordinal==4


def test_cancelled_sample_is_not_a_native_fence_and_cannot_admit_stage_writes():
    backend,events,closed=backend_fixture()
    backend.pending={'sample':{'cancelled':True,'operation':SimpleNamespace(wait=lambda **kwargs:None)}}
    with pytest.raises(ValueError,match='completed native samples'):
        backend.update([row('/Light','inputs:intensity',2)])
    assert backend.ordinal==4 and not events
    assert 'sample' in backend.pending


def test_finished_cancelled_sample_is_reaped_before_stage_mutation():
    backend,events,closed=backend_fixture()
    backend.pending={'sample':{'cancelled':True,'products':object()}}
    backend.update([row('/Light','inputs:intensity',2)])
    assert not backend.pending and backend.ordinal==5
    assert any(event[0]=='submit' for event in events)


def test_live_uncollected_sample_blocks_stage_mutation_without_cancellation():
    backend,events,closed=backend_fixture()
    backend.pending={'sample':{'cancelled':False}}
    with pytest.raises(ValueError,match='completed native samples'):
        backend.update([row('/Light','inputs:intensity',2)])
    assert backend.pending['sample']['cancelled'] is False and not events


def test_same_filename_population_update_invalidates_hierarchy_before_value_replay(tmp_path):
    backend=RendererBackend.__new__(RendererBackend)
    scene=tmp_path/'scene.usda';scene.write_text('#usda 1.0\n')
    backend.scene_config={'scene_path':str(scene),'stage_metadata':{},
                          'sensor_paths':['/Render/Beauty'],'render_var':'LdrColor'}
    backend.desired_values={};backend.ordinal=4;backend.pending={}
    backend.native_stage_id='persistent-stage';backend.simulation_time_ns=0
    backend.native_pose_fanout=object()
    backend.population_owner=SimpleNamespace(path=scene,ordinal=4,overlays=[],snapshot_inline_sources=lambda:{})
    def replace(path,*,ordinal):
        backend.population_owner.ordinal=ordinal
        return {}
    backend.population_owner.replace=replace
    backend.stage=SimpleNamespace(advance_write_floor=lambda ordinal:SimpleNamespace(wait=lambda:None))
    backend.renderer=SimpleNamespace(update_from_stage=lambda ordinal:None,reset=lambda **kwargs:None)
    backend.pose_population=SimpleNamespace(accepted_generation_replaced=lambda **kwargs:None)
    def configure(params):
        assert backend.native_pose_fanout is None
    backend._apply_scene_config=configure
    result=backend.update_scene(dict(backend.scene_config))
    assert result['scene_updated'] and result['native_stage_id']=='persistent-stage'
    assert backend.native_pose_fanout is None


@pytest.mark.parametrize('mode, changed, expected_paths', [
    ('scoped_overlay', ['/World/Cam'], ['/World/Mesh', '/World/Mat/Shader']),
    ('scoped_overlay', ['/World/Mat'], ['/World/Mesh', '/World/Cam']),
    ('scoped_overlay', ['/World'], []),
    ('scoped_overlay', [], ['/World/Mesh', '/World/Cam', '/World/Mat/Shader']),
    ('scoped_overlay', None, []),
    ('replacement', [], []),
])
def test_population_update_worker_cache_matches_effective_invalidation(tmp_path, mode, changed, expected_paths):
    backend = RendererBackend.__new__(RendererBackend)
    scene = tmp_path/'scene.usda'; scene.write_text('#usda 1.0\n')
    patch = tmp_path/'patch.usdc'; patch.write_bytes(b'patch')
    backend.scene_config = {'scene_path': str(scene), 'stage_metadata': {},
                            'sensor_paths': ['/Render/Beauty'], 'render_var': 'LdrColor'}
    old_values = {(p, 'omni:xform'): {'prim_path': p, 'value': i}
                  for i, p in enumerate(['/World/Mesh', '/World/Cam', '/World/Mat/Shader'])}
    backend.desired_values = dict(old_values)
    backend.ordinal = 4; backend.pending = {}; backend.native_stage_id = 'same-stage'
    backend.simulation_time_ns = 0; backend.native_pose_fanout = object()
    backend.population_owner = SimpleNamespace(path=scene, ordinal=4, overlays=[], snapshot_inline_sources=lambda:{})
    def apply(*args, ordinal, **kwargs):
        backend.population_owner.ordinal = ordinal
        return {'population_update': mode}
    backend.population_owner.replace = apply
    backend.population_owner.apply_delta = apply
    backend.stage = SimpleNamespace(advance_write_floor=lambda ordinal: SimpleNamespace(wait=lambda:None))
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal:None, reset=lambda **kwargs:None)
    backend.pose_population = SimpleNamespace(accepted_generation_replaced=lambda **kwargs:None)
    expected = {key:value for key,value in old_values.items() if key[0] in expected_paths}
    def configure(params):
        assert backend.desired_values == expected  # Preserved before any later value replay.
    backend._apply_scene_config = configure
    params = dict(backend.scene_config, population_delta={'path':str(patch), 'predecessor_path':str(scene)})
    if changed is not None:
        params['population_changed_prim_paths'] = changed
    result = backend.update_scene(params)
    assert result['scene_updated'], result
    assert backend.desired_values == expected


@pytest.mark.parametrize('mode, changed, carried', [
    ('scoped_overlay', ['/World/__ovrtx'], True),
    ('scoped_overlay', ['/World/Body'], False),
    ('replacement', ['/World/__ovrtx'], False),
])
def test_scoped_overlay_keeps_synthetic_fanout_identities_outside_changed_paths(tmp_path, mode, changed, carried):
    from native_pose_fanout import NativePoseFanout
    backend = RendererBackend.__new__(RendererBackend)
    scene = tmp_path/'scene.usda'; scene.write_text('#usda 1.0\n')
    patch = tmp_path/'patch.usdc'; patch.write_bytes(b'patch')
    backend.scene_config = {'scene_path': str(scene), 'stage_metadata': {},
                            'sensor_paths': ['/Render/Beauty'], 'render_var': 'LdrColor'}
    backend.desired_values = {}
    backend.ordinal = 4; backend.pending = {}; backend.native_stage_id = 'same-stage'
    backend.simulation_time_ns = 0
    backend.stage = SimpleNamespace(advance_write_floor=lambda ordinal: SimpleNamespace(wait=lambda: None))
    previous = NativePoseFanout(backend.stage, SimpleNamespace(), render_only=True)
    previous.synthetic_identities = {'/World/Body/Mesh': ('Mesh', False)}
    backend.native_pose_fanout = previous
    backend.population_owner = SimpleNamespace(path=scene, ordinal=4, overlays=[], snapshot_inline_sources=lambda: {})
    def apply(*args, ordinal, **kwargs):
        backend.population_owner.ordinal = ordinal
        return {'population_update': mode}
    backend.population_owner.replace = apply
    backend.population_owner.apply_delta = apply
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal: None, reset=lambda **kwargs: None)
    backend.pose_population = SimpleNamespace(accepted_generation_replaced=lambda **kwargs: None)
    backend._apply_scene_config = lambda params: None
    result = backend.update_scene(dict(backend.scene_config, population_changed_prim_paths=changed,
        population_delta={'path': str(patch), 'predecessor_path': str(scene)}))
    assert result['scene_updated'], result
    if carried:
        successor = backend.native_pose_fanout
        assert successor is not previous and successor.stage is backend.stage
        assert successor.synthetic_identities == {'/World/Body/Mesh': ('Mesh', False)}
    else:
        assert backend.native_pose_fanout is None


@pytest.mark.parametrize('caller_target', [False, True])
def test_untyped_target_only_admitted_for_proven_derived_fanout(caller_target):
    backend, events, closed = backend_fixture()
    structural = '/Parent/Container'
    plan = FanoutPlan(structural_paths={structural}, auto_invalidation_safe=True)
    plan.extra_rows = [row(structural, 'omni:xform', np.eye(4).tolist(), 'Matrix4d'),
                       row(structural, 'omni:resetXformStack', True, 'Bool')]
    backend.native_pose_fanout.prepare = lambda *args, **kwargs: plan
    original = backend.stage.read_attributes
    backend.stage.read_attributes = lambda path, tokens, ordinal: (
        nullcontext(SimpleNamespace(wait=lambda: None, groups=lambda: []))
        if path == structural else original(path, tokens, ordinal))
    values = [row('/Parent', 'omni:xform', np.eye(4).tolist(), 'Matrix4d')]
    if caller_target:
        values.append(row(structural, 'omni:resetXformStack', True, 'Bool'))
        with pytest.raises(ValueError, match='existing typed USD prim'):
            backend.update(values)
        assert not any(event[0] == 'submit' for event in events)
    else:
        backend.update(values)
        assert any(event[0] == 'submit' and event[1] == structural for event in events)
