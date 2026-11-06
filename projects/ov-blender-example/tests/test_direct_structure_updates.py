from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example import direct_structure_updates as direct


def obj(name,uid,parent=None,kind='EMPTY',x=0):
    value=NS(name=name,session_uid=uid,parent=parent,type=kind,data=None,instance_type='NONE',
             rigid_body=None,rigid_body_constraint=None)
    value.matrix_world=[[1.,0.,0.,float(x)],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    value.evaluated_get=lambda graph:value
    return value


def fixture(monkeypatch):
    monkeypatch.setattr(direct,'_identity',lambda value:value.name)
    a=obj('A',1);b=obj('B',2,x=5);child=obj('Child',3,a,'MESH',2);leaf=obj('Leaf',4,child,'MESH',3)
    spare=obj('Spare',5,None,'MESH',50)
    objects={value.name:value for value in [a,b,child,leaf,spare]}
    mappings={name:NS(object_path='/World/'+name,schema_path='/World/'+name+('/Mesh' if value.type=='MESH' else '')) for name,value in objects.items()}
    generation=NS(number=1,digest='abc',blender_prim_paths=mappings)
    stage=NS(GetPrimAtPath=lambda path:NS(GetTypeName=lambda:'Mesh' if path.endswith('/Mesh') else 'Xform'))
    contract=direct.capture_contract(stage,generation,objects)
    return generation,contract,a,b,child,leaf,spare


def test_reparent_preserves_prim_identity_and_acks_only_changed_edges(monkeypatch):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    child.parent=b;child.matrix_world[0][3]=7.;leaf.matrix_world[0][3]=8.
    plan=direct.plan_reparent(generation,{'Child'},contract,depsgraph=object())
    assert not plan.fallback and plan.affected=={'Child','Leaf'}
    assert contract.parents['Child']=='A' and contract.children['A']=={'Child'}
    poses={row['prim_path']:row['value'] for row in plan.values if row['attribute_name']=='omni:xform'}
    assert poses['/World/Child'][3][0]==7 and poses['/World/Leaf'][3][0]==8
    assert plan.parent_session_uids=={'Child':2}
    direct.acknowledge(contract,plan)
    assert contract.parents['Child']=='B' and contract.children['B']=={'Child'}
    assert not contract.children['A']
    assert direct.descendant_closure(contract.children,{'B'})=={'B','Child','Leaf'}
    assert direct.descendant_closure(contract.children,{'A'})=={'A'}
    with pytest.raises(ValueError):direct.acknowledge(contract,plan)


def test_coalesced_a_b_a_still_emits_reversal_before_ack(monkeypatch):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    child.parent=b
    first=direct.plan_reparent(generation,{'Child'},contract,depsgraph=object())
    child.parent=a
    reverse=direct.plan_reparent(generation,set(),contract,depsgraph=object(),pending=first)
    assert reverse.parent_changes==( ('Child','A','A'), ) and len(reverse.values)==4
    direct.acknowledge(contract,reverse)
    assert contract.parents['Child']=='A' and contract.children['A']=={'Child'}
    assert not contract.children.get('B')


def test_nominated_closure_never_evaluates_unrelated_objects(monkeypatch):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    def fail(graph):raise AssertionError('unrelated evaluation')
    a.evaluated_get=b.evaluated_get=spare.evaluated_get=fail
    child.parent=b
    assert not direct.plan_reparent(generation,{'Child'},contract,depsgraph=object()).fallback


@pytest.mark.parametrize('mutation,reason',[
    (lambda child:setattr(child,'name','Renamed'),'structure_schema_or_namespace_changed'),
    (lambda child:setattr(child,'instance_type','COLLECTION'),'structure_schema_or_namespace_changed'),
    (lambda child:setattr(child,'rigid_body',object()),'structure_schema_or_namespace_changed'),
])
def test_not_a_rename_instances_or_physics_recook(monkeypatch,mutation,reason):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    child.parent=b;mutation(child)
    assert direct.plan_reparent(generation,{'Child'},contract,depsgraph=object()).fallback==(reason,)


def test_existing_physics_descendant_is_not_claimed_supported(monkeypatch):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    leaf.rigid_body=object();contract.controls['Leaf']=direct._shape_controls(leaf)
    child.parent=b
    assert direct.plan_reparent(generation,{'Child'},contract,depsgraph=object()).fallback==('structure_physics_requires_population',)


def test_parent_cycles_and_generation_changes_fail_without_mutation(monkeypatch):
    generation,contract,a,b,child,leaf,spare=fixture(monkeypatch)
    child.parent=leaf
    assert direct.plan_reparent(generation,{'Child'},contract,depsgraph=object()).fallback==('structure_parent_cycle',)
    generation.number=2
    assert direct.plan_reparent(generation,{'Child'},contract,depsgraph=object()).fallback==('structure_generation_changed',)
    assert contract.parents['Child']=='A'
