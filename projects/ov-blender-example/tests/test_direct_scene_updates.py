"""Focused coordinator acknowledgment and consumer-switch regressions."""
from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example import direct_scene_updates as direct
from ovrtx_blender_example import scene_generation_sessions as sessions


def fixture():
    state=direct.DirectState((1,'abc'),None,(),{}, {},'accepted-materials',('material-layer','light-layer'),scene_uid=42)
    state.values[('/M','inputs:roughness')]=dict(prim_path='/M',attribute_name='inputs:roughness',value_type='Float',value=.7)
    state.pending_materials='next-materials';state.pending_lights='next-lights';state.revision=2
    state.dirty={'material'}
    return NS(_direct_state=state),NS(number=1,digest='abc'),state


def test_successful_native_ack_adopts_current_contract():
    owner,generation,state=fixture();written=[]
    port=NS(update_attribute_values=lambda values:(written.extend(values) or NS(updated_count=len(values))))
    controller=NS(_active=lambda:(NS(direct_projection_token=lambda _:(1,1,1,0)),'zoo'),apply_runtime_updates=lambda operation:operation(port,False))
    receipt=direct.apply(owner,generation,controller)
    assert receipt['revision']==2 and len(written)==1
    assert state.materials=='next-materials' and state.lights=='next-lights'
    assert state.applied_revision==2 and state.pending_materials is None
    # Current projection is retained for later population, never cleared as history.
    assert state.dirty=={'material'} and len(state.values)==1
    second=direct.apply(owner,generation,controller)
    assert second['submitted_value_count']==0 and len(written)==1


def test_incomplete_native_ack_preserves_pending_projection():
    owner,generation,state=fixture()
    controller=NS(_active=lambda:(NS(direct_projection_token=lambda _:(1,1,1,0)),'zoo'),apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=0)),False))
    with pytest.raises(RuntimeError,match='incomplete'):
        direct.apply(owner,generation,controller)
    assert state.materials=='accepted-materials' and state.pending_materials=='next-materials'
    assert state.applied_revision==-1


def test_other_consumers_require_current_projection_and_no_old_layers(monkeypatch):
    owner,generation,state=fixture()
    monkeypatch.setattr(sessions,'_camera_capture_owners',{})
    monkeypatch.setattr(sessions,'_dirty',{})
    assert direct.presentation(owner,generation) is None
    direct.require_projection_for_other_consumer(owner,NS(session_uid=42))
    assert sessions._dirty[42]=={'material'}
    assert direct.fallback_ids(owner,{'mesh'})=={'material','mesh'}


def test_capture_owner_can_reuse_bounded_presentation(monkeypatch):
    owner,generation,state=fixture()
    monkeypatch.setattr(sessions,'_camera_capture_owners',{42:object()})
    monkeypatch.setattr(sessions,'_dirty',{})
    assert direct.presentation(owner,generation)==('material-layer','light-layer')
    direct.require_projection_for_other_consumer(owner,NS(session_uid=42))
    assert sessions._dirty=={}


def test_worker_changes_during_native_write_preserve_pending_contracts():
    owner,generation,state=fixture()
    state.pending_visibility='next-visibility'
    state.ack.changed(('/M','inputs:roughness'))
    token=[(1,1,1,0)]
    client=NS(direct_projection_token=lambda _:token[0])
    def submit(operation):
        result=operation(NS(update_attribute_values=lambda values:NS(updated_count=len(values))),False)
        token[0]=(1,1,2,0)
        return result
    controller=NS(_active=lambda:(client,'zoo'),apply_runtime_updates=submit)
    with pytest.raises(RuntimeError,match='projection_ack_uncertain'):
        direct.apply(owner,generation,controller)
    assert state.materials=='accepted-materials'
    assert state.pending_materials=='next-materials' and state.pending_lights=='next-lights'
    assert state.pending_visibility=='next-visibility' and state.applied_revision==-1
    assert state.ack.pending and state.ack.accepted_token is None
    controller.apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=len(values))),False)
    assert direct.apply(owner,generation,controller)['submitted_value_count']==1
    assert state.applied_revision==2 and state.pending_visibility is None


def test_unknown_native_token_cannot_acknowledge_contracts():
    owner,generation,state=fixture()
    controller=NS(_active=lambda:(NS(direct_projection_token=lambda _:None),'zoo'),
        apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=len(values))),False))
    with pytest.raises(RuntimeError,match='projection_ack_uncertain'):
        direct.apply(owner,generation,controller)
    assert state.applied_revision==-1 and state.pending_materials=='next-materials'


@pytest.mark.parametrize('category', ['MATERIAL', 'WORLD', 'VISIBILITY'])
def test_unacknowledged_a_b_a_edits_coalesce_to_current_contract(monkeypatch, category):
    import sys
    from ovrtx_blender_example import scene_generation as generation_module
    from ovrtx_blender_example import material_direct_updates, direct_light_updates, direct_visibility_updates
    key = generation_module.BlenderId('OBJECT' if category == 'VISIBILITY' else category, 7)
    original = False if category == 'VISIBILITY' else .3
    current = [original]
    baseline = NS(value=original, handles={key: object()}, users={}, states={key: object()})
    empty_materials = NS(effective_slots={}, mesh_users={}, object_handles={})
    state = direct.DirectState((1, 'abc'), None, (), {}, {},
        baseline if category == 'MATERIAL' else empty_materials, (), scene_uid=42)
    scene = NS(session_uid=42, objects=[])
    generation = NS(number=1, digest='abc', scene_object_count=0, blender_prim_paths={}, empty_evaluated_mesh_proofs={})
    if category == 'WORLD':
        state.lights = baseline
    if category == 'VISIBILITY':
        obj = NS(name='Mesh', type='MESH', parent=None, data=NS(session_uid=10),
                 hide_render=False, hide_viewport=False, visible_shadow=True, instance_type='NONE')
        scene.objects = [obj]
        state.objects = {key: obj}
        state.controls[key] = (0, 10, 'MESH', False, False, True, 'NONE', ())
        state.visibility = baseline
        generation.scene_object_count = 1
        generation.blender_prim_paths[key] = NS(blender_id_name='Mesh', schema_path='/M/Shape', object_path='/M')
    monkeypatch.setitem(sys.modules, 'bpy', NS(context=NS(evaluated_depsgraph_get=lambda: object())))
    monkeypatch.setattr(sessions, '_camera_capture_owners', {42: object()})
    monkeypatch.setattr(generation_module, '_proven_non_geometry_objects', lambda *args: {key})
    monkeypatch.setattr(generation_module, '_empty_evaluated_mesh_proofs', lambda *args: {})
    monkeypatch.setattr(material_direct_updates, 'changed_effective_slots', lambda *args: frozenset())
    attribute = 'visibility' if category == 'VISIBILITY' else 'inputs:intensity' if category == 'WORLD' else 'inputs:roughness'
    def compiler(_scene, _generation, _ids, contract, **kwargs):
        value = ('invisible' if current[0] else 'inherited') if category == 'VISIBILITY' else current[0]
        rows = () if current[0] == contract.value else (dict(prim_path='/M', attribute_name=attribute,
            value_type='Token' if category == 'VISIBILITY' else 'Float', value=value),)
        return NS(fallback=(), values=rows, next_contract=NS(**{**vars(contract), 'value': current[0]}))
    module = material_direct_updates if category == 'MATERIAL' else direct_light_updates if category == 'WORLD' else direct_visibility_updates
    monkeypatch.setattr(module, 'plan', compiler)
    owner = NS(_direct_state=state)
    provenance = {key: NS(geometry=False, transform=False, shading=True)}
    current[0] = True if category == 'VISIBILITY' else .8
    assert direct.plan(owner, scene, generation, {key}, provenance)
    current[0] = original
    assert direct.plan(owner, scene, generation, {key}, provenance)
    expected = 'inherited' if category == 'VISIBILITY' else original
    assert state.values[('/M', attribute)]['value'] == expected
    field = 'visibility' if category == 'VISIBILITY' else 'lights' if category == 'WORLD' else 'materials'
    assert getattr(state, field) is baseline
    assert getattr(state, 'pending_' + field).value == original
    written = []
    port = NS(update_attribute_values=lambda values: (written.extend(values) or NS(updated_count=len(values))))
    controller = NS(_active=lambda: (NS(direct_projection_token=lambda _: (1,1,1,0)), 'zoo'),
                    apply_runtime_updates=lambda operation: operation(port, False))
    direct.apply(owner, generation, controller)
    assert len(written) == 1 and written[0].value == expected
    assert getattr(state, field).value == original
    assert getattr(state, 'pending_' + field) is None


def _compiler_test_environment(monkeypatch):
    import sys
    from ovrtx_blender_example import scene_generation as sg, material_direct_updates
    monkeypatch.setitem(sys.modules, 'bpy', NS(context=NS(evaluated_depsgraph_get=lambda: object())))
    monkeypatch.setattr(sessions, '_camera_capture_owners', {42: object()})
    monkeypatch.setattr(sg, '_proven_non_geometry_objects', lambda nominated,generation,affected,objects,provenance: set(objects))
    monkeypatch.setattr(sg, '_empty_evaluated_mesh_proofs', lambda *args: {})
    monkeypatch.setattr(material_direct_updates, 'changed_effective_slots', lambda *args: frozenset())
    return sg


def _leaf_environment(monkeypatch):
    import numpy as np
    from ovrtx_blender_example import direct_visibility_updates as visibility
    sg=_compiler_test_environment(monkeypatch)
    key=sg.BlenderId('OBJECT',7)
    obj=NS(session_uid=7,name='Mesh',name_full='Mesh',type='MESH',parent=None,
        data=NS(session_uid=10),hide_render=False,hide_viewport=False,visible_shadow=True,
        instance_type='NONE',modifiers=())
    obj.evaluated_get=lambda graph:NS(matrix_world=np.eye(4),hide_render=obj.hide_render)
    signature=('Mesh',10,'data',(),(),0,False,False,())
    monkeypatch.setattr(sg,'_mesh_shading_inputs',lambda o:(*signature[:6],o.hide_render,o.hide_viewport,signature[8]))
    state=direct.DirectState((1,'abc'),None,(),{key:obj},{10:{key}},NS(),(),scene_uid=42)
    state.controls[key]=(0,10,'MESH',False,False,True,'NONE',())
    state.visibility=visibility.VisibilityContract((1,'abc'),{key:visibility.State('/Mesh','inherited',visibility._controls(obj,None),True)},{key:obj})
    generation=NS(number=1,digest='abc',scene_object_count=1,blender_prim_paths={key:NS(blender_id_name='Mesh',schema_path='/Mesh',object_path='/Mesh')},mesh_shading_inputs={key:signature},empty_evaluated_mesh_proofs={})
    class NoSceneScan:
        def __len__(self):return 1
        def __iter__(self):raise AssertionError('scene scan')
    return sg,key,obj,state,generation,NS(session_uid=42,objects=NoSceneScan())


@pytest.mark.parametrize('field', ['matte', 'holdout'])
def test_custom_render_opinion_cannot_be_acknowledged_as_geometry_noop(monkeypatch, field):
    sg,key,obj,state,generation,scene=_leaf_environment(monkeypatch)
    obj.ov=NS(rendering=NS(matte='INHERIT',holdout='INHERIT'))
    setattr(obj.ov.rendering,field,'ON')
    accepted=state.controls[key]
    assert not direct.plan(NS(_direct_state=state),scene,generation,{key},
        {key:sg.SceneUpdateProvenance(False,False,True)})
    assert state.diagnostics['fallback_reasons']==('object_structure_or_visibility_changed',)
    assert state.controls[key]==accepted and not state.values


def test_hide_geometry_notification_does_not_read_mesh_or_scan_scene(monkeypatch):
    from ovrtx_blender_example import direct_mesh_updates
    sg,key,obj,state,generation,scene=_leaf_environment(monkeypatch)
    monkeypatch.setattr(sg,'_proven_non_geometry_objects',lambda *a:set())
    monkeypatch.setattr(direct_mesh_updates,'evaluated_mesh',lambda *a:pytest.fail('visibility extracted geometry'))
    obj.hide_viewport=True
    owner=NS(_direct_state=state)
    assert direct.plan(owner,scene,generation,{key},{key:sg.SceneUpdateProvenance(True,False,False)})
    assert state.values[('/Mesh','visibility')]['value']=='invisible'
    assert state.diagnostics['visibility_only_count']==1
    assert state.controls[key][4] is False


def test_visibility_proof_does_not_swallow_a_simultaneous_mesh_edit(monkeypatch):
    sg,key,obj,state,generation,scene=_leaf_environment(monkeypatch)
    obj.hide_render=True;mesh=sg.BlenderId('MESH',10)
    notes={key:sg.SceneUpdateProvenance(True,False,False),mesh:sg.SceneUpdateProvenance(True,False,False)}
    assert direct.visibility_only_owners(state,generation,{key},{key,mesh},notes)==set()
    assert direct.visibility_only_owners(state,generation,{key},{key},{key:sg.SceneUpdateProvenance()})==set()
    obj.modifiers=[NS(type='BOOLEAN')]
    assert direct.visibility_only_owners(state,generation,{key},{key},notes)==set()


def test_mixed_unknown_owner_preserves_visibility_geometry_exclusion(monkeypatch):
    sg,key,obj,state,generation,scene=_leaf_environment(monkeypatch)
    obj.hide_render=True
    assert not direct.plan(NS(_direct_state=state),scene,generation,{key,sg.BlenderId('MESH',99)},
        {key:sg.SceneUpdateProvenance(True,False,False)})
    assert state.visibility_only_ids=={key}
    assert state.diagnostics['fallback_reasons']==('unmapped_data_owner:MESH:99',)


def test_mesh_swap_uses_existing_prim_and_repairs_lookup_only_on_ack(monkeypatch):
    from contextlib import contextmanager
    from ovrtx_blender_example import direct_mesh_updates as mesh_updates
    sg,key,obj,state,generation,scene=_leaf_environment(monkeypatch)
    obj.data=NS(session_uid=20)
    seen=[]
    @contextmanager
    def mesh(*args):
        seen.append(key);yield NS()
    monkeypatch.setattr(mesh_updates,'evaluated_mesh',mesh)
    state.mesh_contracts[key]=NS(subsets=())
    patch=NS(supported=True)
    monkeypatch.setattr(mesh_updates,'build_mesh_patch',lambda *a:patch)
    monkeypatch.setattr(mesh_updates,'wire_values',lambda *a:[dict(prim_path='/Mesh',attribute_name='points',value_type='Point3fArray',value=[[1,2,3]])])
    owner=NS(_direct_state=state)
    assert direct.plan(owner,scene,generation,{key,sg.BlenderId('MESH',20)},
        {key:sg.SceneUpdateProvenance(True,False,False)}),state.diagnostics
    assert seen==[key] and state.diagnostics['mesh_swap_count']==1
    assert state.data_users=={10:{key}} and state.controls[key][1]==10
    port=NS(update_attribute_values=lambda values:NS(updated_count=0))
    controller=NS(_active=lambda:(NS(direct_projection_token=lambda _:(1,1,1,0)),'zoo'),
        apply_runtime_updates=lambda operation:operation(port,False))
    with pytest.raises(RuntimeError):direct.apply(owner,generation,controller)
    assert state.data_users=={10:{key}}
    port.update_attribute_values=lambda values:NS(updated_count=len(values))
    direct.apply(owner,generation,controller)
    assert state.data_users=={20:{key}} and state.controls[key][1]==20
    assert state.visibility.states[key].controls[1]==20


@pytest.mark.parametrize('failure', ['graph', 'values', 'epoch'])
def test_pending_material_graph_retires_old_values_and_survives_failed_ack(monkeypatch, failure):
    from ovrtx_blender_example import material_direct_updates, material_graph_updates
    sg=_compiler_test_environment(monkeypatch)
    key=sg.BlenderId('MATERIAL', 7)
    accepted=NS(label='old');pending=NS(label='new')
    state=direct.DirectState((1,'abc'),None,(),{}, {},accepted,(),scene_uid=42)
    stale=('/Materials/M/OldShader','inputs:roughness')
    outside=('/Other','inputs:intensity')
    for target in (stale,outside):
        state.values[target]=dict(prim_path=target[0],attribute_name=target[1],value_type='Float',value=.7)
        state.ack.changed(target)
    graph=dict(material_path='/Materials/M',layer_body='#usda 1.0\n')
    monkeypatch.setattr(material_direct_updates,'plan',lambda *a,**kw:NS(fallback=('graph_changed',)))
    monkeypatch.setattr(material_graph_updates,'plan',lambda *a,**kw:NS(fallback=(),updates=(graph,),next_contract=pending))
    owner=NS(_direct_state=state);generation=NS(number=1,digest='abc',scene_object_count=0)
    assert direct.plan(owner,NS(session_uid=42,objects=[]),generation,{key},{})
    assert stale not in state.values and stale not in state.ack.pending
    assert state.pending_graphs=={'/Materials/M':graph} and state.pending_materials is pending
    token=[(1,1,1,0)];graphs=[];writes=[]
    def graph_write(_simulation, updates):
        if failure=='graph':raise RuntimeError('graph rejected')
        graphs.extend(updates)
    def values_write(values):
        writes.extend(values)
        if failure=='epoch':token[0]=(1,1,2,0)
        return NS(updated_count=0 if failure=='values' else len(values))
    client=NS(direct_projection_token=lambda _:token[0],update_material_graphs=graph_write)
    controller=NS(_active=lambda:(client,'zoo'),apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=values_write),False))
    with pytest.raises(RuntimeError):direct.apply(owner,generation,controller)
    assert state.materials is accepted and state.pending_materials is pending
    assert state.pending_graphs=={'/Materials/M':graph} and state.applied_revision==-1
    client.update_material_graphs=lambda _simulation,updates:graphs.extend(updates)
    controller.apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=len(values))),False)
    direct.apply(owner,generation,controller)
    assert state.materials is pending and not state.pending_graphs
    assert all(value.prim_path=='/Other' for value in writes)


@pytest.mark.parametrize('reverse_before_ack', [False, True])
def test_reparent_visibility_controls_and_child_index_advance_only_on_ack(monkeypatch, reverse_before_ack):
    import numpy as np
    from ovrtx_blender_example import direct_structure_updates as structure, direct_visibility_updates as visibility
    sg=_compiler_test_environment(monkeypatch)
    def obj(uid,name,kind,parent=None):
        value=NS(session_uid=uid,name=name,type=kind,parent=parent,data=NS(session_uid=uid+100) if kind=='MESH' else None,
            hide_render=False,hide_viewport=False,visible_shadow=True,instance_type='NONE',rigid_body=None,rigid_body_constraint=None)
        value.evaluated_get=lambda graph:NS(matrix_world=np.eye(4),hide_render=value.hide_render)
        return value
    old=obj(1,'Old','EMPTY');new=obj(2,'New','EMPTY');mesh=obj(3,'Mesh','MESH',old);child=obj(4,'Child','MESH',mesh)
    objects={sg.BlenderId('OBJECT',o.session_uid):o for o in (old,new,mesh,child)}
    old_id,new_id,mesh_id,child_id=(sg.BlenderId('OBJECT',i) for i in (1,2,3,4))
    children={old_id:{mesh_id},mesh_id:{child_id}}
    materials=NS()
    state=direct.DirectState((1,'abc'),None,(),objects,{},materials,(),scene_uid=42,children=children)
    generation=NS(number=1,digest='abc',scene_object_count=4,blender_prim_paths={k:NS(blender_id_name=o.name,object_path='/'+o.name,schema_path='/'+o.name) for k,o in objects.items()},empty_evaluated_mesh_proofs={})
    state.controls={k:(o.parent.session_uid if o.parent else 0,o.data.session_uid if o.data else 0,o.type,False,False,True,'NONE',()) for k,o in objects.items()}
    state.structure=structure.StructureContract((1,'abc'),objects,{k:'/'+o.name for k,o in objects.items()},
        {old_id:None,new_id:None,mesh_id:old_id,child_id:mesh_id},children,
        {k:structure._shape_controls(o) for k,o in objects.items()},frozenset(objects))
    states={k:visibility.State('/'+objects[k].name,'inherited',visibility._controls(objects[k],None),True) for k in (mesh_id,child_id)}
    state.visibility=visibility.VisibilityContract((1,'abc'),states,{k:objects[k] for k in states})
    owner=NS(_direct_state=state);scene=NS(session_uid=42,objects=list(objects.values()))
    provenance={mesh_id:NS(geometry=False,transform=True,shading=False)}
    mesh.parent=new
    assert direct.plan(owner,scene,generation,{mesh_id},provenance),state.diagnostics
    if reverse_before_ack:
        mesh.parent=old
        assert direct.plan(owner,scene,generation,{mesh_id},provenance),state.diagnostics
    expected_id=old_id if reverse_before_ack else new_id
    expected_uid=1 if reverse_before_ack else 2
    assert state.structure.parents[mesh_id]==old_id and children[old_id]=={mesh_id}
    assert state.controls[mesh_id][0]==1 and state.visibility.states[mesh_id].controls[0]==1
    assert state.pending_visibility.states[mesh_id].controls[0]==expected_uid
    controller=NS(_active=lambda:(NS(direct_projection_token=lambda _:(1,1,1,0)),'zoo'),
        apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=0)),False))
    with pytest.raises(RuntimeError):direct.apply(owner,generation,controller)
    assert state.structure.parents[mesh_id]==old_id and children[old_id]=={mesh_id}
    assert state.pending_structure is not None and state.controls[mesh_id][0]==1
    controller.apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda values:NS(updated_count=len(values))),False)
    direct.apply(owner,generation,controller)
    assert state.structure.parents[mesh_id]==expected_id
    assert mesh_id in children[expected_id]
    assert state.controls[mesh_id][0]==expected_uid and state.visibility.states[mesh_id].controls[0]==expected_uid
    if not reverse_before_ack:assert mesh_id not in children[old_id]
    assert state.pending_structure is None


@pytest.mark.parametrize('category', ['MATERIAL', 'WORLD', 'LIGHT', 'CAMERA', 'MESH', 'EMPTY'])
def test_physics_scene_routes_only_visual_consumers_directly(monkeypatch,category):
    from ovrtx_blender_example import material_direct_updates, direct_light_updates, direct_camera_updates
    sg=_compiler_test_environment(monkeypatch)
    kind=category if category in {'MATERIAL','WORLD'} else 'OBJECT'
    key=sg.BlenderId(kind,7)
    state=direct.DirectState((1,'abc'),None,(),{}, {},NS(),(),scene_uid=42,has_physics=True)
    scene=NS(session_uid=42,objects=[])
    generation=NS(number=1,digest='abc',scene_object_count=0,blender_prim_paths={},empty_evaluated_mesh_proofs={})
    state.lights=NS(handles={key:object()},users={})
    state.cameras=NS()
    result=lambda *a,**kw:NS(fallback=(),values=(),next_contract=a[3])
    monkeypatch.setattr(material_direct_updates,'plan',result)
    monkeypatch.setattr(direct_light_updates,'plan',result)
    monkeypatch.setattr(direct_camera_updates,'plan',result)
    if kind=='OBJECT':
        obj=NS(name=category,type=category,parent=None,data=None,hide_render=False,hide_viewport=False,visible_shadow=True,instance_type='NONE')
        state.objects[key]=obj;scene.objects=[obj];generation.scene_object_count=1
        generation.blender_prim_paths[key]=NS(blender_id_name=category,object_path='/'+category,schema_path='/'+category)
        state.controls[key]=(0,0,category,False,False,True,'NONE',())
    assert direct.plan(NS(_direct_state=state),scene,generation,{key},{key:NS(geometry=False,transform=False,shading=True)}) is (category not in {'MESH','EMPTY'})
    if category in {'MESH','EMPTY'}:
        assert state.diagnostics['fallback_reasons']==('affected_physics_consumer_requires_population',)


def test_physics_gate_includes_geometry_descendants_of_light(monkeypatch):
    from ovrtx_blender_example import direct_light_updates
    sg=_compiler_test_environment(monkeypatch)
    light_id=sg.BlenderId('OBJECT',1);mesh_id=sg.BlenderId('OBJECT',2)
    light=NS(session_uid=1,name='Light',type='LIGHT',parent=None,data=None,hide_render=False,hide_viewport=False,visible_shadow=True,instance_type='NONE')
    mesh=NS(session_uid=2,name='Mesh',type='MESH',parent=light,data=None,hide_render=False,hide_viewport=False,visible_shadow=True,instance_type='NONE')
    state=direct.DirectState((1,'abc'),None,(),{light_id:light,mesh_id:mesh},{},NS(),(),scene_uid=42,has_physics=True,children={light_id:{mesh_id}},lights=NS(handles={},users={}))
    state.controls={light_id:(0,0,'LIGHT',False,False,True,'NONE',()),mesh_id:(1,0,'MESH',False,False,True,'NONE',())}
    generation=NS(number=1,digest='abc',scene_object_count=2,blender_prim_paths={light_id:NS(blender_id_name='Light',object_path='/Light',schema_path='/Light'),mesh_id:NS(blender_id_name='Mesh',object_path='/Mesh',schema_path='/Mesh')},empty_evaluated_mesh_proofs={})
    scene=NS(session_uid=42,objects=[light,mesh])
    monkeypatch.setattr(direct,'_pose',lambda *a:())
    monkeypatch.setattr(direct_light_updates,'plan',lambda *a,**kw:NS(fallback=(),values=(),next_contract=a[3]))
    assert not direct.plan(NS(_direct_state=state),scene,generation,{light_id},{light_id:NS(geometry=False,transform=True,shading=False)})
    assert 'affected_physics_consumer_requires_population' in state.diagnostics['fallback_reasons']


def test_bootstrap_physics_detection_includes_native_collision_schema(monkeypatch,tmp_path):
    from ovrtx_blender_example import material_direct_updates,direct_light_updates,direct_camera_updates,direct_visibility_updates,direct_structure_updates
    path=tmp_path/'native-physics.usda'
    path.write_text('#usda 1.0\ndef Cube "NativeCollider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}\n')
    monkeypatch.setattr(material_direct_updates,'capture_composed_contract',lambda *a,**kw:NS())
    monkeypatch.setattr(direct_light_updates,'capture_contract',lambda *a,**kw:None)
    monkeypatch.setattr(direct_camera_updates,'capture_contract',lambda *a,**kw:None)
    monkeypatch.setattr(direct_visibility_updates,'capture_contract',lambda *a,**kw:None)
    monkeypatch.setattr(direct_structure_updates,'capture_contract',lambda *a,**kw:None)
    state=direct.initialize(NS(),NS(session_uid=42,objects=[]),NS(number=1,digest='abc',usd_path=str(path)),NS(material_scene_layer=None,light_scene_layer=None))
    assert state.has_physics


def test_instance_source_adopts_mapping_only_on_ack_and_replays_only_on_recovery():
    from ovrtx_blender_example.scene_generation import BlenderId
    owner,generation,state=fixture();key=BlenderId('OBJECT',12)
    source=NS(projection_path='/World/_DirectGeometry/Object_12',hidden_source_path='/World/GN',
        layer_text='#usda 1.0\n',mapping=NS(object_path='/World/_DirectGeometry/Object_12'),reference_identities=())
    state.pending_instances[key]=source;state.mesh_contracts[key]='old shape'
    token=[(1,1,1,0)];calls=[];fail=[True]
    def submit(_id,updates):
        calls.append(updates)
        if fail[0]:raise RuntimeError('unacknowledged graph')
    client=NS(direct_projection_token=lambda _:token[0],update_instance_graphs=submit)
    controller=NS(_active=lambda:(client,'zoo'),apply_runtime_updates=lambda operation:operation(NS(update_attribute_values=lambda rows:NS(updated_count=len(rows))),False))
    with pytest.raises(RuntimeError,match='unacknowledged'):direct.apply(owner,generation,controller)
    assert key in state.pending_instances and key not in state.mapping_overrides
    fail[0]=False;direct.apply(owner,generation,controller)
    assert state.mapping_overrides[key] is source.mapping and not state.pending_instances and key not in state.mesh_contracts
    direct.apply(owner,generation,controller);assert len(calls)==2
    token[0]=(2,1,1,0);direct.apply(owner,generation,controller)
    assert len(calls)==3 and calls[-1][0]['owner_id']==12
