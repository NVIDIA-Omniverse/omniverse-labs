from dataclasses import replace
from types import SimpleNamespace as NS
import math
import pytest
from ovrtx_blender_example import direct_light_updates as direct
from ovrtx_blender_example import light_value_conversion as conversion
from ovrtx_blender_example import world_dome_conversion as worlds

class Matrix(list):
    def to_scale(self):return (2.,3.,1.)

def fixture():
    data=NS(type='AREA',shape='RECTANGLE',energy=10.,size=2.,size_y=3.)
    matrix=Matrix([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    obj=NS(type='LIGHT',data=data,matrix_world=matrix,scale=(1.,1.,1.))
    obj.evaluated_get=lambda graph:obj
    values=direct._light_values(obj)+(('xformOp:transform','Matrix4d',direct._pose(obj)),)
    schema=frozenset(name for name,typ,val in values)
    state=direct.State('/L/Shape','RectLight','AREA_RECT',schema,values,conversion.unsupported_field_values(obj),'/L',frozenset({'xformOp:transform'}))
    contract=direct.LightContract((1,'abc'),{'object':state},{'object':obj},{'light':('object',)})
    return NS(world=None),NS(number=1,digest='abc'),contract,obj

def test_energy_size_pose_and_ack():
    scene,generation,contract,obj=fixture()
    obj.data.energy=30.;obj.data.size=4.;obj.matrix_world[0][3]=7.
    result=direct.plan(scene,generation,{'light'},contract,depsgraph=object())
    assert not result.fallback
    rows={r['attribute_name']:r for r in result.values}
    assert rows['inputs:intensity']['value']==pytest.approx(30*120*math.pi/(4*3*2*3))
    assert rows['inputs:width']['value']==4.
    assert rows['omni:xform']['value'][3][0]==7.
    assert rows['omni:xform']['prim_path']=='/L'
    assert direct.plan(scene,generation,{'light'},result.next_contract).values==()
    assert contract.states['object'].values!=result.next_contract.states['object'].values

def test_schema_and_unsupported_batch_fail_closed():
    scene,generation,contract,obj=fixture();obj.data.shape='DISK'
    assert direct.plan(scene,generation,{'object'},contract).fallback==('light_form_changed',)
    obj.data.shape='RECTANGLE';obj.data.energy=12
    bad=replace(contract,states={'object':replace(contract.states['object'],schema=frozenset())})
    result=direct.plan(scene,generation,{'object'},bad)
    assert result.fallback and not result.values and result.next_contract is bad

def test_world_resource_rows_and_graph_fallback(monkeypatch):
    world=NS(use_nodes=False,color=(1,1,1))
    monkeypatch.setattr(direct,'_world_values',lambda w:(('inputs:intensity','Float',w.color[0]),('inputs:texture:file','Asset',w.path)))
    world.path='/a.hdr'
    state=direct.State('/World/StudioDome','DomeLight','WORLD',frozenset({'inputs:intensity','inputs:texture:file'}),direct._world_values(world))
    contract=direct.LightContract((1,'abc'),{'world':state},{'world':world},{'tree':('world',)},'world',('flat',))
    scene=NS(world=world);generation=NS(number=1,digest='abc');world.path='/b.hdr'
    result=direct.plan(scene,generation,{'tree'},contract)
    assert not result.fallback and result.values[0]['value']=='/b.hdr'
    assert result.handled==frozenset({'tree'})
    scene.world=NS()
    assert direct.plan(scene,generation,{'tree'},result.next_contract).fallback==('world_assignment_changed',)

def test_dome_rotation_half_turn_and_inverse():
    # Unrotated Blender worlds need a half turn to face the native dome like Cycles.
    m=worlds.environment_rotation_matrix(0)
    assert [m[i][j] for i in range(4) for j in range(4)]==pytest.approx(
        [-1.,0.,0.,0., 0.,-1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.])
    m=worlds.environment_rotation_matrix(math.pi/2)
    assert m[0][1]==pytest.approx(1.) and m[1][0]==pytest.approx(-1.)
    assert m[0][0]==pytest.approx(0.,abs=1e-12) and m[2][2]==1.
