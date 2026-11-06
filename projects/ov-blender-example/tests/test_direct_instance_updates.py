from types import SimpleNamespace as NS
import numpy as np
import pytest
from pxr import Usd,UsdGeom,Sdf
from ovrtx_blender_example import direct_instance_updates as direct
from ovrtx_blender_example.scene_generation import BlenderId
from test_direct_mesh_updates import quad,Items

class Attributes(list):
    def __getitem__(self,key):
        if isinstance(key,str):return next(a for a in self if a.name==key)
        return super().__getitem__(key)
    def get(self,key):return next((a for a in self if a.name==key),None)


def fixture(*,mixed=True):
    matrices=[np.eye(4),np.eye(4)];matrices[1][0,3]=4.
    attrs=Attributes([NS(name='.reference_index',data=Items([NS(value=0),NS(value=0)])),
                      NS(name='instance_transform',data=Items([NS(value=m) for m in matrices]))])
    proto=NS(mesh=quad(),instances_pointcloud=lambda:None)
    geometry=NS(mesh=quad() if mixed else None,instances_pointcloud=lambda:NS(attributes=attrs),instance_references=lambda:[proto])
    evaluated=NS(matrix_world=np.eye(4),evaluated_geometry=lambda:geometry)
    obj=NS(name='GN',name_full='GN',session_uid=1,type='MESH',data=NS(session_uid=2),children=[],evaluated_get=lambda graph:evaluated)
    generation=NS(blender_prim_paths={BlenderId('MATERIAL',42):NS(schema_path='/World/Looks/Existing'),
                                     BlenderId('OBJECT',1):NS(object_path='/World/GN',schema_path='/World/GN/Mesh')})
    return obj,generation,geometry,proto,attrs


def stage_from(source):
    assert not source.fallback,source.fallback
    layer=Sdf.Layer.CreateAnonymous(".usda");assert layer.ImportFromString(source.layer_text)
    return Usd.Stage.Open(layer)


@pytest.mark.parametrize('mixed',[True,False])
def test_current_evaluated_prototypes_uvs_and_mixed_components(mixed):
    obj,generation,geometry,proto,attrs=fixture(mixed=mixed)
    proto.mesh.vertices[1].co=(9.,0.,0.)
    source=direct.compile_owner(obj,object(),generation)
    stage=stage_from(source);root=source.projection_path
    assert source.instance_count==2 and source.prototype_count==1
    assert bool(stage.GetPrimAtPath(root+'/RealizedMesh'))==mixed
    instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath(root+'/Instances'))
    assert list(instancer.GetPositionsAttr().Get())[1][0]==4.
    prototype=UsdGeom.Mesh(stage.GetPrimAtPath(root+'/Instances/Prototype_0'))
    assert prototype.GetPointsAttr().Get()[1][0]==9.
    assert prototype.GetNormalsInterpolation()=='faceVarying'
    assert prototype.GetPrim().GetAttribute('primvars:st').Get()
    assert prototype.GetPrim().GetRelationship('material:binding').GetTargets()==['/World/Looks/Existing']
    assert stage.GetPrimAtPath('/World/GN').GetAttribute('visibility').Get()=='invisible'
    assert generation.blender_prim_paths[BlenderId('OBJECT',1)].object_path=='/World/GN'
    assert source.mapping.object_path==root


def test_no_global_depsgraph_access_and_current_mesh_object_reference():
    obj,generation,geometry,proto,attrs=fixture()
    called=[]
    evaluated=NS(type='MESH',evaluated_geometry=lambda:proto)
    reference=NS(type='MESH',session_uid=17,data=NS(session_uid=18),evaluated_get=lambda graph:(called.append(graph) or evaluated))
    geometry.instance_references=lambda:[reference]
    class Graph:
        def __getattr__(self,name):raise AssertionError('global graph access '+name)
    graph=Graph();source=direct.compile_owner(obj,graph,generation)
    assert not source.fallback and called==[graph]


def test_shear_and_custom_instance_attributes_are_not_silently_dropped():
    obj,generation,geometry,proto,attrs=fixture()
    attrs['instance_transform'].data[0].value[0,1]=.4
    assert direct.compile_owner(obj,None,generation).fallback==('instance_transform_has_shear',)
    attrs['instance_transform'].data[0].value=np.eye(4)
    attrs.append(NS(name='Tint',data=[]))
    assert direct.compile_owner(obj,None,generation).fallback==('instance_attribute_requires_population:Tint',)


def test_negative_scale_trs_preserved():
    m=np.eye(4);m[0,0]=-2;m[1,1]=3;m[2,2]=4;m[:3,3]=[1,2,3]
    pos,quat,scale=direct._trs(m)
    assert pos==(1.,2.,3.) and scale==(-2.,3.,4.) and quat.GetReal()==pytest.approx(1.)


def test_missing_material_and_nested_instances_return_clear_fallback():
    obj,generation,geometry,proto,attrs=fixture(mixed=False)
    generation.blender_prim_paths.pop(BlenderId('MATERIAL',42))
    assert direct.compile_owner(obj,None,generation).fallback==('instance_material_not_populated',)
    proto.instances_pointcloud=lambda:object()
    assert direct.compile_owner(obj,None,generation).fallback==('nested_instances_require_population',)


def test_projection_packet_supersedes_only_current_owner_and_hides_original():
    obj,generation,geometry,proto,attrs=fixture()
    source=direct.compile_owner(obj,None,generation)
    current={(source.projection_path+'/Instances/proto0','points'):object(),('/Unrelated','inputs:intensity'):object()}
    packet=direct.packet_for_source(source,current_values=current)['inline_population']
    assert packet['owned_roots']==[source.projection_path,'/World/GN']
    targets={(item['prim_path'],item['attribute_name']) for item in packet['superseded_attributes']}
    assert targets=={(source.projection_path+'/Instances/proto0','points'),('/World/GN','visibility')}
    assert len(current)==2


def test_empty_instances_removes_renderable_projection_without_empty_prototypes():
    obj,generation,geometry,proto,attrs=fixture(mixed=False)
    attrs['.reference_index'].data=Items()
    attrs['instance_transform'].data=Items()
    source=direct.compile_owner(obj,None,generation)
    stage=stage_from(source)
    assert source.instance_count==source.prototype_count==0
    assert not stage.GetPrimAtPath(source.projection_path+'/Instances')
    assert stage.GetPrimAtPath('/World/GN').GetAttribute('visibility').Get()=='invisible'


def test_real_blender_mesh_line_topology_attribute_does_not_block_instances():
    obj,generation,geometry,proto,attrs=fixture(mixed=False)
    attrs.append(NS(name='.edge_verts',data_type='INT32_2D',domain='POINT',data=Items([NS(value=(0,1)),NS(value=(0,1))])))
    source=direct.compile_owner(obj,None,generation)
    assert not source.fallback and source.instance_count==2
    stage=stage_from(source)
    assert not stage.GetPrimAtPath(source.projection_path+'/Instances').GetAttribute('primvars:.edge_verts')
