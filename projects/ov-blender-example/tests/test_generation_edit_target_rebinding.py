# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queued authoring values follow sparse namespaces at native consumption."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example.scene_generation_sessions import AuthoringGenerationRuntime
from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath
from ovrtx_blender_example.interactive_edit_planner import EditIntent, EditShape, DataAuthority
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from ovrtx_blender_example.view_update_stream import ViewUpdateStream
from ovrtx_blender_example.runtime_scheduler import RuntimeTickResult, RuntimeTickStatus
import pytest
from ovrtx_blender_example import scene_generation_sessions as sessions


def intent(uid=7, path='/World/Old/Cube', value=1):
    return EditIntent(shape=EditShape.VALUE, data_authority=DataAuthority.VIEW,
        blender_property_path='matrix_world', usd_prim_path=path,
        usd_property_path=path+'.xformOp:transform',
        provenance={'blender_session_uid':uid,'blender_id_kind':'OBJECT'},
        value=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[value,0,0,1]])


def runtime():
    r=AuthoringGenerationRuntime.__new__(AuthoringGenerationRuntime)
    # This namespace fixture has no compiled physics roots; activation still
    # consumes the same materialization seam as a real SceneGeneration.
    generic_stage = str(Path(__file__).parent / 'fixtures' / 'public_sdk_render.usda')
    r._native_view_generation=SimpleNamespace(materialize_usd=lambda: generic_stage, blender_prim_paths={
        BlenderId('OBJECT',7):BlenderPrimPath('Cube','MESH','/World/New/Cube','/World/New/Cube/Mesh'),
        BlenderId('OBJECT',8):BlenderPrimPath('Light','LIGHT','/World/New/Light','/World/New/Light/Area',data_session_uid=18),
    })
    return r


class Port:
    def __init__(self): self.transforms=[];self.attributes=[]
    def update_transforms(self,values):
        assert all(v.prim_path.startswith('/World/New/') for v in values)
        self.transforms.extend(values)
        return OvrtxValueUpdateResult(len(values),16666667)
    def update_attribute_values(self,values):
        assert all(v.prim_path.startswith('/World/New/') for v in values)
        self.attributes.extend(values)
        return OvrtxValueUpdateResult(len(values),16666667)


def test_old_and_new_queued_paths_coalesce_by_current_identity_with_latest_value():
    r=runtime();stream=ViewUpdateStream(target_resolver=r._resolve_native_view_target)
    stream.queue(intent(value=1))
    stream.queue(intent(path='/World/New/Cube',value=2))
    port=Port();result=stream.apply_pending(port)
    assert result['values_written'] is True
    assert len(port.transforms)==1
    assert port.transforms[0].prim_path=='/World/New/Cube'
    assert port.transforms[0].matrix[3][0]==2
    assert not stream.has_pending


def test_light_data_update_rebinds_schema_path_using_verified_data_identity():
    r=runtime()
    edit=replace(intent(uid=18,path='/World/Old/Light/Area'),
        blender_property_path='energy',usd_property_path='/World/Old/Light/Area.inputs:intensity',
        provenance={'blender_session_uid':18,'blender_id_kind':'LIGHT'},value=200)
    stream=ViewUpdateStream(target_resolver=r._resolve_native_view_target)
    stream.queue(edit);port=Port();result=stream.apply_pending(port)
    assert result['values_written'] is True
    assert len(port.attributes)==1
    assert port.attributes[0].prim_path=='/World/New/Light/Area'
    assert port.attributes[0].attribute=='inputs:intensity'


def test_removed_authoring_target_is_reported_without_poisoning_remaining_values():
    r=runtime();stream=ViewUpdateStream(target_resolver=r._resolve_native_view_target)
    stream.queue(intent(uid=999,path='/World/Old/Deleted'));stream.queue(intent(value=3))
    port=Port();result=stream.apply_pending(port)
    assert len(port.transforms)==1
    assert not result.get('failed',False)
    assert any(u.get('skipped_reason')=='scene_generation_target_removed' for u in result['updates'])
    assert not stream.has_pending


def test_target_resolver_failure_is_atomic_and_preserves_pending_values():
    def resolver(edit):
        if edit.provenance['blender_session_uid']==999: raise ValueError('ambiguous target')
        return edit
    stream=ViewUpdateStream(target_resolver=resolver)
    stream.queue(intent(path='/World/New/Cube'));stream.queue(intent(uid=999,path='/World/Old/Deleted'))
    port=Port();result=stream.apply_pending(port)
    assert result['failed'] is True
    assert 'view_target_resolution_error' in result['skipped_reason']
    assert not port.transforms and not port.attributes
    assert stream.has_pending


def test_composition_camera_values_are_not_rewritten_as_authoring_objects():
    r=runtime()
    edit=replace(intent(),provenance={},blender_property_path='viewport_camera_matrix')
    assert r._resolve_native_view_target(edit) is edit


def test_native_binding_can_return_to_predecessor_after_rejected_candidate():
    r=runtime();edit=intent()
    assert r._resolve_native_view_target(edit).usd_prim_path=='/World/New/Cube'
    r._native_view_generation=SimpleNamespace(blender_prim_paths={
        BlenderId('OBJECT',7):BlenderPrimPath('Cube','MESH','/World/Old/Cube','/World/Old/Cube/Mesh')})
    assert r._resolve_native_view_target(edit).usd_prim_path=='/World/Old/Cube'


def test_interleaved_old_and_new_target_aliases_keep_global_latest_submission():
    r=runtime();stream=ViewUpdateStream(target_resolver=r._resolve_native_view_target)
    stream.queue(intent(value=1))
    stream.queue(intent(path='/World/New/Cube',value=2))
    stream.queue(intent(value=3))
    port=Port();stream.apply_pending(port)
    assert len(port.transforms)==1
    assert port.transforms[0].matrix[3][0]==3


def test_composition_world_owner_is_not_treated_as_deleted_authoring_object():
    r=runtime();r._native_view_generation.world_session_uid=27
    r._native_material_edit_resolver=lambda generation: SimpleNamespace(
        resolve_world_dome=lambda: SimpleNamespace(value=SimpleNamespace(prim_path='/World/StudioDome')))
    edit=replace(intent(uid=27),blender_property_path='world_dome.intensity',usd_prim_path='/World/StudioDome',
        usd_property_path='/World/StudioDome.inputs:intensity',
        provenance={'blender_session_uid':27,'blender_id_kind':'WORLD'},value=2)
    assert r._resolve_native_view_target(edit).usd_prim_path=='/World/StudioDome'
    assert r._resolve_native_view_target(replace(edit,provenance={'blender_session_uid':28,'blender_id_kind':'WORLD'})) is None


def test_material_queue_rebinds_to_native_shader_instead_of_stock_material_path():
    r=runtime()
    r._native_view_generation.blender_prim_paths[BlenderId('MATERIAL',37)]=BlenderPrimPath(
        'Orange','MATERIAL','/World/New/_materials/Orange','/World/New/_materials/Orange')
    calls=[]
    def resolve(identity,**kwargs):
        calls.append((identity.session_uid,kwargs))
        return SimpleNamespace(value=SimpleNamespace(prim_path='/OVRTX_Materials/Orange/OpenPBR'))
    r._native_material_edit_resolver=lambda generation: SimpleNamespace(resolve_material=resolve)
    edit=replace(intent(uid=37),blender_property_path='Base Color',usd_prim_path='/World/Old/_materials/Orange',
        usd_property_path='/World/Old/_materials/Orange.inputs:base_color',
        provenance={'blender_session_uid':37,'blender_id_kind':'MATERIAL'},value=[.1,.2,.3])
    resolved=r._resolve_native_view_target(edit)
    assert resolved.usd_property_path=='/OVRTX_Materials/Orange/OpenPBR.inputs:base_color'
    assert calls==[(37,{'usd_attribute':'inputs:base_color','property_name':'Base Color'})]


@pytest.mark.parametrize('changed',[False,True])
def test_same_generation_replays_values_only_when_composition_is_repopulated(changed):
    r=runtime();r._generation_number=7
    transforms=(object(),);attributes=(object(),);events=[]
    r.owner=SimpleNamespace(retained_values_for=lambda generation:(transforms,attributes,()))
    r.ovrtx=SimpleNamespace(active_generation=7,composition=SimpleNamespace(composed_scene_path='before'),
        request=None,last_ensure_result=SimpleNamespace(session_started=False),update_request=lambda request:None,last_error='')
    def ensure():
        r.ovrtx.composition=SimpleNamespace(composed_scene_path='after' if changed else 'before')
        return True
    r.ovrtx.ensure_request=ensure
    r.scheduler=SimpleNamespace(has_pending_sim_updates=False,baked_playback_paused=False,note_applied_content=lambda:None)
    r._activate_ovrtx=lambda generation,t,a:events.append((t,a)) or True
    r._finish_activation=lambda *args:True
    r.activate(SimpleNamespace(number=7),SimpleNamespace())
    assert events==([(transforms,attributes)] if changed else [])


def test_native_material_snapshot_replays_current_binding_and_discards_deleted_identity():
    r=runtime();generation=r._native_view_generation
    generation.blender_prim_paths[BlenderId('MATERIAL',37)]=BlenderPrimPath('Orange','MATERIAL','/Stock/Orange','/Stock/Orange')
    edit=replace(intent(uid=37,path='/PBR/Old'),blender_property_path='Base Color',
        usd_property_path='/PBR/Old.inputs:base_color',provenance={'blender_session_uid':37,'blender_id_kind':'MATERIAL'},value=[.1,.8,.2])
    r._retained_material_edits={(37,'inputs:base_color'):edit,(99,'inputs:base_color'):replace(edit,provenance={'blender_session_uid':99,'blender_id_kind':'MATERIAL'})}
    r._native_material_edit_resolver=lambda generation:SimpleNamespace(resolve_material=lambda *args,**kwargs:SimpleNamespace(value=SimpleNamespace(prim_path='/PBR/Current')))
    replay=[]
    def values(port,transforms,attributes):
        replay.extend(attributes)
        return RuntimeTickResult(status=RuntimeTickStatus.NOOP,enabled=True)
    r.scheduler=SimpleNamespace(pending_view_targets=lambda:(frozenset(),frozenset()),has_pending_view_updates=False,replay_retained_values=values)
    r.ovrtx=SimpleNamespace(activate=lambda generation:True,controller=SimpleNamespace(apply_runtime_updates=lambda callback:callback(None,None)),last_error='')
    assert r._activate_ovrtx(generation,(),())
    assert len(replay)==1 and replay[0].prim_path=='/PBR/Current' and replay[0].value==[.1,.8,.2]
    assert (99,'inputs:base_color') not in r._retained_material_edits


def test_candidate_deletion_failure_preserves_material_for_predecessor_replay():
    r = runtime()
    predecessor = r._native_view_generation
    predecessor.blender_prim_paths[BlenderId('MATERIAL', 37)] = BlenderPrimPath(
        'Orange', 'MATERIAL', '/Stock/Orange', '/Stock/Orange')
    candidate = SimpleNamespace(blender_prim_paths={})
    r.owner = SimpleNamespace(current_generation=predecessor, pending_generation=candidate)
    edit = replace(intent(uid=37, path='/PBR/Old'), blender_property_path='Base Color',
        usd_property_path='/PBR/Old.inputs:base_color',
        provenance={'blender_session_uid': 37, 'blender_id_kind': 'MATERIAL'},
        value=[.1, .8, .2])
    key = (37, 'inputs:base_color')
    r._retained_material_edits = {key: edit}
    r._native_material_edit_resolver = lambda generation: SimpleNamespace(
        resolve_material=lambda *args, **kwargs: SimpleNamespace(
            value=SimpleNamespace(prim_path='/PBR/Restored')))
    replay = []

    def replay_values(port, transforms, attributes):
        replay.extend(attributes)
        return RuntimeTickResult(status=RuntimeTickStatus.NOOP, enabled=True)

    r.scheduler = SimpleNamespace(
        pending_view_targets=lambda: (frozenset(), frozenset()),
        has_pending_view_updates=True, replay_retained_values=replay_values)
    r.ovrtx = SimpleNamespace(activate=lambda generation: True, last_error='',
        controller=SimpleNamespace(apply_runtime_updates=lambda callback: callback(None, None)))
    r._apply_pending_view_values = lambda: RuntimeTickResult(
        status=RuntimeTickStatus.FAILED, enabled=True, skipped_reason='injected candidate failure')
    assert not r._activate_ovrtx(candidate, (), ())
    assert key in r._retained_material_edits
    r.scheduler.has_pending_view_updates = False
    assert r._activate_ovrtx(predecessor, (), ())
    assert replay[-1].prim_path == '/PBR/Restored'
    assert replay[-1].value == [.1, .8, .2]
    # A successful candidate still leaves predecessor values until acceptance.
    assert r._activate_ovrtx(candidate, (), ())
    assert key in r._retained_material_edits
    r.owner.current_generation = candidate
    r.owner.pending_generation = None
    assert r._activate_ovrtx(candidate, (), ())
    assert not r._retained_material_edits


@pytest.mark.parametrize('accepted,uid,expected',[(True,27,True),(False,27,False),(True,28,False)])
def test_accepted_world_graph_group_schedules_only_verified_world_owner(monkeypatch,accepted,uid,expected):
    generation=SimpleNamespace(blender_prim_paths={BlenderId('WORLD',27):BlenderPrimPath('World','WORLD','/World/env_light','/World/env_light')})
    owner=SimpleNamespace(current_generation=generation)
    scene=SimpleNamespace(session_uid=101)
    edit=replace(intent(uid=uid,path='/World/env_light'),shape=EditShape.TOPOLOGY,blender_property_path='node_tree',
        provenance={'blender_id_kind':'WORLD','blender_session_uid':uid})
    result=SimpleNamespace(accepted=accepted,plan=SimpleNamespace(impact=SimpleNamespace(scene_generation_replacement_requested=True)))
    runtime=SimpleNamespace(viewport_ids={'owned'},submit_edit_group=lambda group:(result,))
    monkeypatch.setattr(sessions,'_owners',{101:owner})
    monkeypatch.setattr(sessions,'resolve_current_scene_edit_group',lambda *args:(edit,))
    monkeypatch.setattr(sessions,'_runtime_for_owner',lambda *args:runtime)
    dirty=[]
    monkeypatch.setattr(sessions,'mark_scene_dirty',lambda *args,**kwargs:dirty.append((args,kwargs)) or True)
    assert sessions.submit_current_scene_edit_group(scene,(edit,),{})==(result,)
    assert bool(dirty)==expected
    if expected:assert dirty==[((scene,{BlenderId('WORLD',27)}),{'defer_world_reconciliation':True})]
