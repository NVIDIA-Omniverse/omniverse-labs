# SPDX-License-Identifier: Apache-2.0
"""Effective material users must survive sparse USD namespace replacement."""
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example import scene_generation as g
from ovrtx_blender_example import scene_generation_sessions as sessions


def fixture():
    shared=N(session_uid=41); unused=N(session_uid=42)
    a=N(session_uid=10,name_full='DataUser',type='MESH',parent=None,
        data=N(session_uid=20,materials=(shared,)),material_slots=(N(material=shared,link='DATA'),))
    b=N(session_uid=11,name_full='ObjectUser',type='MESH',parent=None,
        data=N(session_uid=21,materials=(unused,)),material_slots=(N(material=shared,link='OBJECT'),))
    c=N(session_uid=12,name_full='Other',type='MESH',parent=None,
        data=N(session_uid=22,materials=(unused,)),material_slots=(N(material=unused,link='DATA'),))
    mappings={g.BlenderId('OBJECT',obj.session_uid):g.BlenderPrimPath(
        obj.name_full,'MESH','/World/'+obj.name_full,'/World/'+obj.name_full+'/Mesh',obj.data.session_uid)
        for obj in (a,b,c)}
    for material,name in ((shared,'Shared'),(unused,'Other')):
        path='/World/_materials/'+name
        mappings[g.BlenderId('MATERIAL',material.session_uid)]=g.BlenderPrimPath(name,'MATERIAL',path,path)
    predecessor=N(blender_prim_paths=mappings,world_session_uid=0)
    return N(session_uid=5,objects=(a,b,c),world=None),predecessor,shared,unused


@pytest.mark.parametrize('kind',['OBJECT','MATERIAL'])
def test_shared_object_override_closure_keeps_every_active_usd_binding(tmp_path,kind):
    from pxr import Sdf,Usd,UsdShade
    scene,predecessor,shared,unused=fixture()
    affected=g.BlenderId(kind,10 if kind=='OBJECT' else shared.session_uid)
    objects,removed,materials=g._sparse_object_closure(scene,predecessor,(affected,))
    assert objects==(scene.objects[:1] if kind=='OBJECT' else scene.objects[:2]) and not removed
    assert materials==(set() if kind=='OBJECT' else {g.BlenderId('MATERIAL',41)})
    def stock(path,objects):
        stage=Usd.Stage.CreateNew(str(path));stage.DefinePrim('/World','Xform')
        mappings={}
        for obj in objects:
            identity=g.BlenderId('OBJECT',obj.session_uid);mapping=predecessor.blender_prim_paths[identity]
            prim=stage.DefinePrim(mapping.object_path,'Xform')
            prim.CreateAttribute(g._OBJECT_UID_ATTRIBUTE,Sdf.ValueTypeNames.Int).Set(obj.session_uid)
            mesh=stage.DefinePrim(mapping.schema_path,'Mesh');mappings[identity]=mapping
            material=g.effective_object_materials(obj)[0]
            material_id=g.BlenderId('MATERIAL',material.session_uid)
            material_mapping=predecessor.blender_prim_paths[material_id]
            mat=UsdShade.Material.Define(stage,material_mapping.object_path)
            mat.GetPrim().CreateAttribute(g._MATERIAL_UID_ATTRIBUTE,Sdf.ValueTypeNames.Int).Set(material.session_uid)
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
            mappings[material_id]=material_mapping
        stage.GetRootLayer().Save()
        return mappings
    base=tmp_path/'base.usda';stock(base,scene.objects)
    selected=tmp_path/'selected.usda';selected_mappings=stock(selected,objects)
    delta=tmp_path/'delta.usda'
    rewritten,replaced,deleted=g._write_scene_topology_delta(selected,delta,1,selected_mappings,
        predecessor,removed,materials)
    mappings=g._updated_generation_mappings(predecessor,rewritten,removed,materials)
    root=Sdf.Layer.CreateNew(str(tmp_path/'composed.usda'))
    root.subLayerPaths=[str(delta),str(base)];root.Save()
    g._validate_composed_generation(tmp_path/'composed.usda',mappings,replaced,deleted)
    stage=Usd.Stage.Open(root)
    expected=('/World/_materials/Shared' if kind=='OBJECT' else '/World/__ovrtx/generation_000001/_materials/Shared')
    for obj in objects:
        mapping=mappings[g.BlenderId('OBJECT',obj.session_uid)]
        assert stage.GetPrimAtPath(mapping.schema_path).GetRelationship('material:binding').GetTargets()==[Sdf.Path(expected)]
    assert stage.GetPrimAtPath('/World/_materials/Shared').IsActive()==(kind=='OBJECT')
    if kind=='OBJECT':
        assert stage.GetPrimAtPath('/World/ObjectUser/Mesh').GetRelationship('material:binding').GetTargets()==[Sdf.Path(expected)]
    assert stage.GetPrimAtPath('/World/Other/Mesh').GetRelationship('material:binding').GetTargets()==[Sdf.Path('/World/_materials/Other')]


def test_effective_slots_reachability_preserves_none_and_data_fallback(monkeypatch):
    scene,accepted,shared,_=fixture()
    monkeypatch.setattr(sessions,'_owners',{5:N(current_generation=accepted,pending_generation=None)})
    assert sessions.topology_identity_changes(scene)==set()
    scene.objects[0].material_slots=(N(material=None),)
    assert g.effective_object_materials(scene.objects[0])==(None,)
    assert sessions.topology_identity_changes(scene)==set() # still used by OBJECT-linked user
    scene.objects[1].material_slots=()
    assert sessions.topology_identity_changes(scene)=={g.BlenderId('MATERIAL',41)}
    del scene.objects[0].material_slots
    assert g.effective_object_materials(scene.objects[0])==(shared,)
    assert sessions.topology_identity_changes(scene)==set()


def test_export_identity_tags_effective_override_and_restores_after_failure():
    class ID(dict):
        def __init__(self,uid):super().__init__();self.session_uid=uid
    base,override=ID(41),ID(42)
    data=ID(20);data.materials=(base,)
    obj=ID(10);obj.type='MESH';obj.data=data;obj.material_slots=(N(material=override),)
    with pytest.raises(RuntimeError,match='export failure'):
        with g._temporary_export_identities(N(objects=(obj,))):
            assert override[g._IDENTITY_PROPERTIES['MATERIAL']]==42
            assert not base
            raise RuntimeError('export failure')
    assert not override and not obj and not data and not base


@pytest.mark.parametrize("coalesced",[False,True])
def test_nested_instance_material_users_and_old_shared_path_survive_sparse_export(tmp_path,monkeypatch,coalesced):
    from pxr import Sdf,Usd,UsdShade
    scene,predecessor,shared,unused=fixture()
    shared.name_full='Same name';shared.library=None;unused.name_full='Other'
    linked=N(session_uid=43,name_full='Same name',library=N(filepath='fixture.blend'))
    child=N(session_uid=90,type='MESH',data=N(materials=(linked,)),material_slots=(N(material=linked),))
    nested=N(session_uid=91,type='EMPTY',instance_collection=N(objects=(child,),children=()))
    collection=N(objects=(nested,),children=())
    owner=N(session_uid=13,type='EMPTY',name_full='Fixture',parent=None,instance_collection=collection)
    # Cycles must not expand forever or create independent child mappings.
    collection.objects=(nested,owner)
    scene.objects=(*scene.objects,owner)
    owner_id=g.BlenderId('OBJECT',13)
    predecessor.blender_prim_paths[owner_id]=g.BlenderPrimPath('Fixture','EMPTY','/World/Fixture','/World/Fixture')
    base=tmp_path/'instance-base.usda';stage=Usd.Stage.CreateNew(str(base))
    for identity,mapping in predecessor.blender_prim_paths.items():
        prim=stage.DefinePrim(mapping.object_path,'Material' if identity.kind=='MATERIAL' else 'Xform')
        prim.CreateAttribute(g._MATERIAL_UID_ATTRIBUTE if identity.kind=='MATERIAL' else g._OBJECT_UID_ATTRIBUTE,
                             Sdf.ValueTypeNames.Int).Set(identity.session_uid)
        if identity.kind=='OBJECT' and identity!=owner_id:stage.DefinePrim(mapping.schema_path,'Mesh')
    old_mat=UsdShade.Material(stage.GetPrimAtPath('/World/_materials/Shared'))
    child_path='/World/Fixture/Nested/LinkedMesh'
    UsdShade.MaterialBindingAPI.Apply(stage.DefinePrim(child_path,'Mesh')).Bind(old_mat)
    for obj in scene.objects[:2]:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/'+obj.name_full+'/Mesh')).Bind(old_mat)
    stage.GetRootLayer().Save();predecessor.usd_path=str(base)
    assert g.object_export_materials(owner)==(linked,)
    objects,removed,materials=g._sparse_object_closure(scene,predecessor,(g.BlenderId('MATERIAL',41),))
    assert objects==(scene.objects[0],scene.objects[1],owner)
    assert materials=={g.BlenderId('MATERIAL',41)} and not removed
    # New stock export distinguishes same names by UID; both child references
    # and ordinary bindings must move together into the candidate namespace.
    selected=tmp_path/'instance-selected.usda';stock=Usd.Stage.CreateNew(str(selected))
    selected_mappings={identity:mapping for identity,mapping in predecessor.blender_prim_paths.items()
                       if identity in {g.BlenderId('OBJECT',obj.session_uid) for obj in objects}|materials}
    linked_path='/World/_materials/Shared' if coalesced else '/World/_materials/Shared_001'
    if not coalesced:
        selected_mappings[g.BlenderId('MATERIAL',43)]=g.BlenderPrimPath('Same name','MATERIAL',linked_path,linked_path)
    for identity,mapping in selected_mappings.items():
        prim=stock.DefinePrim(mapping.object_path,'Material' if identity.kind=='MATERIAL' else 'Xform')
        prim.CreateAttribute(g._MATERIAL_UID_ATTRIBUTE if identity.kind=='MATERIAL' else g._OBJECT_UID_ATTRIBUTE,
                             Sdf.ValueTypeNames.Int).Set(identity.session_uid)
        if identity.kind=='MATERIAL':prim.CreateAttribute('userProperties:blender:data_name',Sdf.ValueTypeNames.String).Set('Same name')
    for obj in objects[:2]:
        mesh=stock.DefinePrim('/World/'+obj.name_full+'/Mesh','Mesh')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(UsdShade.Material(stock.GetPrimAtPath('/World/_materials/Shared')))
    UsdShade.MaterialBindingAPI.Apply(stock.DefinePrim(child_path,'Mesh')).Bind(UsdShade.Material(stock.GetPrimAtPath(linked_path)))
    stock.GetRootLayer().Save()
    # Strict material mapping remains UID-based despite equal Blender names.
    monkeypatch.setattr(g,'_blender_module_provider',lambda:N(data=N(materials=(shared,linked,unused))))
    validated=g._validated_blender_prim_paths(scene,selected)
    assert validated[g.BlenderId('MATERIAL',41)].object_path=='/World/_materials/Shared'
    if coalesced:assert g.BlenderId('MATERIAL',43) not in validated
    else:assert validated[g.BlenderId('MATERIAL',43)].object_path==linked_path
    assert g.BlenderId('OBJECT',90) not in validated and g.BlenderId('OBJECT',91) not in validated
    delta=tmp_path/'instance-delta.usda'
    rewritten,replaced,deleted=g._write_scene_topology_delta(selected,delta,1,validated,predecessor,removed,materials)
    mappings=g._updated_generation_mappings(predecessor,rewritten,removed,materials)
    root=Sdf.Layer.CreateNew(str(tmp_path/'instance-composed.usda'));root.subLayerPaths=[str(delta),str(base)];root.Save()
    g._validate_composed_generation(tmp_path/'instance-composed.usda',mappings,replaced,deleted)
    composed=Usd.Stage.Open(root)
    moved='/World/__ovrtx/generation_000001/Fixture/Nested/LinkedMesh'
    assert composed.GetPrimAtPath(moved).GetRelationship('material:binding').GetTargets()==[
        Sdf.Path('/World/__ovrtx/generation_000001'+linked_path[len('/World'):])]


def test_linked_instance_material_identity_tag_and_reachability_restore(monkeypatch):
    class ID(dict):
        def __init__(self,uid):super().__init__();self.session_uid=uid
    linked=ID(43);linked.library=N(filepath='fixture.blend')
    child=N(session_uid=90,type='MESH',data=N(materials=(linked,)))
    owner=ID(13);owner.type='EMPTY';owner.instance_collection=N(session_uid=70,objects=(child,),children=())
    scene=N(session_uid=5,objects=(owner,),world=None)
    mapped={g.BlenderId('OBJECT',13):g.BlenderPrimPath('Fixture','EMPTY','/World/Fixture','/World/Fixture'),
            g.BlenderId('MATERIAL',43):g.BlenderPrimPath('Nickel','MATERIAL','/World/_materials/Nickel','/World/_materials/Nickel')}
    accepted=N(blender_prim_paths=mapped,world_session_uid=0)
    monkeypatch.setattr(sessions,'_owners',{5:N(current_generation=accepted,pending_generation=None)})
    assert sessions.topology_identity_changes(scene)==set()
    with pytest.raises(RuntimeError,match='export failure'):
        with g._temporary_export_identities(scene):
            assert linked[g._IDENTITY_PROPERTIES['MATERIAL']]==43
            assert not hasattr(child,'ovrtx_object_session_uid')
            raise RuntimeError('export failure')
    assert not linked and not owner
    owner.instance_collection=None
    assert sessions.topology_identity_changes(scene)=={g.BlenderId('MATERIAL',43)}


def test_geometry_edit_does_not_flood_through_shared_material_graph(monkeypatch):
    scene,predecessor,shared,other=fixture()
    # Middle object uses two materials, connecting otherwise unrelated users.
    scene.objects[1].material_slots=(N(material=shared),N(material=other))
    def unexpected_scan(*args):
        pytest.fail("geometry-only update traversed accepted material users")
    monkeypatch.setattr(g,'_accepted_material_binding_users',unexpected_scan)
    objects,removed,materials=g._sparse_object_closure(scene,predecessor,(g.BlenderId('OBJECT',10),))
    assert objects==scene.objects[:1] and not removed and not materials


def test_material_edit_does_not_promote_other_unchanged_slots_to_dirty():
    scene,predecessor,shared,other=fixture()
    scene.objects[1].material_slots=(N(material=shared),N(material=other))
    objects,removed,materials=g._sparse_object_closure(scene,predecessor,(g.BlenderId('MATERIAL',41),))
    assert objects==scene.objects[:2] and not removed
    assert materials=={g.BlenderId('MATERIAL',41)}


def test_repeated_sparse_edit_keeps_generated_namespace_assemblies_separate(monkeypatch):
    scene,predecessor,shared,other=fixture()
    predecessor.blender_prim_paths={identity:g.BlenderPrimPath(mapping.blender_id_name,
        mapping.blender_id_type,mapping.object_path.replace('/World/','/World/__ovrtx/generation_000001/'),
        mapping.schema_path.replace('/World/','/World/__ovrtx/generation_000001/'),mapping.data_session_uid)
        for identity,mapping in predecessor.blender_prim_paths.items()}
    objects,removed,materials=g._sparse_object_closure(scene,predecessor,(g.BlenderId('OBJECT',10),))
    assert objects==scene.objects[:1] and not removed and not materials
    assert g._world_child_path('/World/__ovrtx/generation_000001/DataUser/Mesh')=='/World/__ovrtx/generation_000001/DataUser'
