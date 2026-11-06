from types import SimpleNamespace as NS
from dataclasses import replace
from ovrtx_blender_example import direct_visibility_updates as direct


def fixture():
    obj = NS(type='MESH', parent=None, data=NS(session_uid=2), instance_type='NONE',
             hide_render=False, hide_viewport=False)
    obj.visible_get = lambda **kw: True
    evaluated = NS(hide_render=False)
    obj.evaluated_get = lambda graph: evaluated
    state = direct.State('/World/Box/Mesh', 'inherited', direct._controls(obj, None), True)
    contract = direct.VisibilityContract((1,'abc'), {'object':state}, {'object':obj})
    return NS(), NS(number=1,digest='abc'), obj, evaluated, contract


def test_evaluated_driver_hide_show_and_ack():
    scene, generation, obj, evaluated, contract = fixture()
    evaluated.hide_render = True
    result = direct.plan(scene,generation,{'object'},contract,depsgraph=object())
    assert not result.fallback
    assert result.values == ({'prim_path':'/World/Box/Mesh','attribute_name':'visibility','value_type':'Token','value':'invisible'},)
    assert contract.states['object'].token == 'inherited'
    assert not direct.plan(scene,generation,{'object'},result.next_contract,depsgraph=object()).values
    evaluated.hide_render = False
    shown = direct.plan(scene,generation,{'object'},result.next_contract,depsgraph=object())
    assert shown.values[0]['value'] == 'inherited'


def test_render_disabled_collection_cannot_be_unhidden_by_object():
    scene,generation,obj,evaluated,contract = fixture()
    state=replace(contract.states['object'],render_enabled=False,token='invisible')
    contract=replace(contract,states={'object':state})
    assert not direct.plan(scene,generation,{'object'},contract).values


def test_structure_or_unknown_target_fallback_atomic():
    scene,generation,obj,evaluated,contract=fixture()
    obj.hide_render=True
    result=direct.plan(scene,generation,{'object','unknown'},contract)
    assert not result.values and result.next_contract is contract
    assert result.fallback == ('visibility_target_not_supported',)
    obj.hide_viewport=True
    hidden=direct.plan(scene,generation,{'object'},contract)
    assert not hidden.fallback and hidden.values[0]['value']=='invisible'
    obj.hide_viewport=False;obj.data=NS(session_uid=9)
    assert direct.plan(scene,generation,{'object'},contract).fallback


def test_viewport_eye_hide_and_show_use_tokens_without_scene_access():
    scene,generation,obj,evaluated,contract=fixture()
    eye=[False];obj.hide_get=lambda **kwargs:eye[0]
    eye[0]=True
    hidden=direct.plan(scene,generation,{'object'},contract)
    assert not hidden.fallback and hidden.values[0]['value']=='invisible'
    assert contract.states['object'].token=='inherited'
    eye[0]=False
    shown=direct.plan(scene,generation,{'object'},hidden.next_contract)
    assert not shown.fallback and shown.values[0]['value']=='inherited'


def test_no_scene_access_for_targeted_plan_and_generation_guard():
    scene,generation,obj,evaluated,contract=fixture()
    class NoScan:
        def __getattr__(self,name):raise AssertionError('scene access: '+name)
    assert not direct.plan(NoScan(),generation,{'object'},contract).fallback
    generation.number=2
    assert direct.plan(scene,generation,{'object'},contract).fallback == ('visibility_generation_changed',)


def test_capture_existing_leaf_only(monkeypatch):
    scene,generation,obj,evaluated,contract=fixture()
    obj.session_uid=1
    scene.objects=[obj];scene.view_layers=[]
    generation.blender_prim_paths={'object':NS(schema_path='/World/Box/Mesh')}
    monkeypatch.setattr(direct,'_identity',lambda obj:'object')
    monkeypatch.setattr(direct.object_visibility,'scene_view_layer',lambda *a,**kw:None)
    prim=NS(GetTypeName=lambda:'Mesh',GetAttribute=lambda name:NS(Get=lambda:'inherited'))
    stage=NS(GetPrimAtPath=lambda path:prim)
    captured=direct.capture_contract(stage,scene,generation)
    assert captured.states['object'].path=='/World/Box/Mesh'
    obj.instance_type='COLLECTION'
    assert not direct.capture_contract(stage,scene,generation).states
