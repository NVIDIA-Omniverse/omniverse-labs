from types import SimpleNamespace as NS
import numpy as np
import pytest
from addon.ovrtx_blender_example.direct_mesh_updates import MeshContract, PrimvarContract, SubsetContract, build_mesh_patch, extract_object_patch, contract_from_usd, DirectMeshUnsupported, wire_values

class Items(list):
    def foreach_get(self, key, output):
        output[:] = np.asarray([getattr(v,key) for v in self]).reshape(-1)
class Named(Items):
    def get(self,name): return next((v for v in self if v.name==name),None)
def quad():
    return NS(materials=[NS(session_uid=42)], vertices=Items([NS(co=v) for v in [(0,0,0),(2,0,0),(2,2,0),(0,2,0)]]), loops=Items([NS(vertex_index=i) for i in range(4)]), polygons=Items([NS(material_index=0)]), loop_triangles=Items([NS(loops=(0,1,2),polygon_index=0),NS(loops=(0,2,3),polygon_index=0)]), corner_normals=Items([NS(vector=v) for v in [(0,0,1),(0,1,0),(0,0,1),(1,0,0)]]), uv_layers=Named([NS(name='UVMap',active_render=True,data=Items([NS(uv=v) for v in [(0,0),(1,0),(1,1),(0,1)]]))]), attributes=Named(),calc_loop_triangles=lambda:None)
def contract(**kw): return MeshContract('/World/Mesh',material_slots=(42,),**kw)
def rows(p):
    assert p.supported,p.reason
    return {(v['prim_path'],v['attribute_name']):v['value'] for v in p.values}

def test_complete_bundle_preserves_corner_normals_uvs_units():
    c=contract(unit_scale=.01,primvars=(PrimvarContract('primvars:st','UVMap','TexCoord2fArray','faceVarying',True,'uv'),))
    p=build_mesh_patch(quad(),c);r=rows(p)
    np.testing.assert_array_equal(r[c.path,'faceVertexIndices'],[0,1,2,0,2,3])
    np.testing.assert_allclose(r[c.path,'extent'],[[0,0,0],[.02,.02,0]])
    np.testing.assert_array_equal(r[c.path,'normals'][[1,5]],[[0,1,0],[1,0,0]])
    np.testing.assert_array_equal(r[c.path,'primvars:st'][[1,5]],[[1,0],[0,1]])
    np.testing.assert_array_equal(r[c.path,'primvars:st:indices'],range(6))
    assert p.triangle_count==2 and isinstance(wire_values(p)[0]['value'],list)

def test_subset_indices_and_parent_invalidation_travel_with_topology():
    m=quad();m.materials.append(NS(session_uid=43));m.polygons.append(NS(material_index=1));m.loop_triangles[1].polygon_index=1
    c=MeshContract('/World/Mesh',material_slots=(42,43),subsets=(SubsetContract('/World/Mesh/A',(0,)),SubsetContract('/World/Mesh/B',(1,))))
    patch=build_mesh_patch(m,c);r=rows(patch)
    np.testing.assert_array_equal(r['/World/Mesh/A','indices'],[0])
    np.testing.assert_array_equal(r['/World/Mesh/B','indices'],[1])
    m.loop_triangles.append(NS(loops=(1,2,3),polygon_index=1))
    np.testing.assert_array_equal(rows(build_mesh_patch(m,c))['/World/Mesh/B','indices'],[1,2])


def test_new_slot_and_new_subset_reject_complete_patch():
    m=quad();m.materials.append(NS(session_uid=43));p=build_mesh_patch(m,contract())
    assert p.reason=='evaluated_material_slots_changed' and not p.values
    m.polygons.append(NS(material_index=1));m.loop_triangles[1].polygon_index=1
    p=build_mesh_patch(m,MeshContract('/World/Mesh',material_slots=(42,43)))
    assert p.reason=='new_material_subset_required' and not p.values

def test_named_face_attribute_rgb_preserved():
    m=quad();m.attributes.append(NS(name='Tint',data_type='FLOAT_COLOR',domain='FACE',data=Items([NS(color=(.1,.2,.3,.9))])))
    c=contract(primvars=(PrimvarContract('primvars:Tint','Tint','Color3fArray','faceVarying'),))
    np.testing.assert_allclose(rows(build_mesh_patch(m,c))[c.path,'primvars:Tint'],np.tile([.1,.2,.3],(6,1)))

def test_mixed_nodes_component_rejected():
    evaluated=NS(type='MESH',evaluated_geometry=lambda:NS(mesh=quad(),curves=object()))
    assert extract_object_patch(NS(evaluated_get=lambda _:evaluated),None,contract()).reason=='evaluated_curves_component_requires_population'

def test_mesh_only_geometry_nodes_uses_final_geometry_without_copy():
    e=NS(type='MESH',modifiers=[NS(type='NODES')],evaluated_geometry=lambda:NS(mesh=quad()))
    assert extract_object_patch(NS(evaluated_get=lambda _:e),None,contract()).supported

def test_temporary_mesh_released_on_fallback():
    clear=[];e=NS(type='MESH',modifiers=[],to_mesh=lambda **_:quad(),to_mesh_clear=lambda:clear.append(1))
    c=contract(primvars=(PrimvarContract('primvars:Missing','Missing','FloatArray','uniform'),))
    assert extract_object_patch(NS(evaluated_get=lambda _:e),None,c).reason=='missing_mesh_attribute:Missing'
    assert clear==[1]

def test_nonfinite_positions_rejected():
    m=quad();m.vertices[0].co=(float('nan'),0,0)
    assert build_mesh_patch(m,contract()).reason=='nonfinite_mesh_column:points'

def test_initial_usd_contract_indexed_uv():
    from pxr import Usd,UsdGeom,Sdf
    s=Usd.Stage.CreateInMemory();m=UsdGeom.Mesh.Define(s,'/World/Mesh');m.CreateSubdivisionSchemeAttr('none');m.CreateNormalsAttr([(0,0,1)]*6);m.SetNormalsInterpolation('faceVarying')
    pv=UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying');pv.Set([(0,0),(1,0),(1,1),(0,1)]);pv.SetIndices([0,1,2,0,2,3])
    c=contract_from_usd(m.GetPrim(),quad())
    assert c.primvars==(PrimvarContract('primvars:st','UVMap','texCoord2f[]','faceVarying',True,'uv'),)
    assert build_mesh_patch(quad(),c).supported

def test_unknown_primvar_cannot_remain_stale_on_topology_change():
    from pxr import Usd,UsdGeom,Sdf
    s=Usd.Stage.CreateInMemory();m=UsdGeom.Mesh.Define(s,'/M');m.CreateSubdivisionSchemeAttr('none');m.CreateNormalsAttr([(0,0,1)]*6)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('other',Sdf.ValueTypeNames.FloatArray,'vertex').Set([0,1,2,3])
    with pytest.raises(DirectMeshUnsupported,match='unmapped_primvar:other'): contract_from_usd(m.GetPrim(),quad())

def test_geometry_set_owner_survives_contract_and_mesh_extraction():
    from addon.ovrtx_blender_example.direct_mesh_updates import evaluated_mesh
    released=[]
    mesh=quad()
    original_calc=mesh.calc_loop_triangles
    def calc():
        assert not released, 'borrowed Mesh outlived its GeometrySet owner'
        original_calc()
    mesh.calc_loop_triangles=calc
    class GeometryOwner:
        def __init__(self): self.mesh=mesh
        def __del__(self): released.append(True)
    e=NS(type='MESH',evaluated_geometry=GeometryOwner)
    obj=NS(evaluated_get=lambda _:e)
    with evaluated_mesh(obj,None) as borrowed:
        assert not released
        assert build_mesh_patch(borrowed,contract()).supported
    assert released==[True]
    released.clear()
    assert extract_object_patch(obj,None,contract()).supported
    assert released==[True]


def test_evaluated_material_identity_comes_from_original_id():
    from addon.ovrtx_blender_example.direct_mesh_updates import material_slot_ids
    mesh=quad()
    mesh.materials=[NS(session_uid=0,original=NS(session_uid=42))]
    assert material_slot_ids(mesh)==(42,)
    assert build_mesh_patch(mesh,contract()).supported
    mesh.materials[0].original.session_uid=43
    assert build_mesh_patch(mesh,contract()).reason=='evaluated_material_slots_changed'


def test_constant_scalar_renderer_primvar_is_preserved_by_geometry_patch():
    from pxr import Usd,UsdGeom,Sdf
    s=Usd.Stage.CreateInMemory();m=UsdGeom.Mesh.Define(s,'/M')
    m.CreateSubdivisionSchemeAttr('none');m.CreateNormalsAttr([(0,0,1)]*6);m.SetNormalsInterpolation('faceVarying')
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('doNotCastShadows',Sdf.ValueTypeNames.Bool,'constant').Set(True)
    c=contract_from_usd(m.GetPrim(),quad())
    assert not c.primvars
    patch=build_mesh_patch(quad(),c)
    assert patch.supported and all(row['attribute_name']!='primvars:doNotCastShadows' for row in patch.values)


def test_unknown_scalar_constant_cannot_be_silently_preserved():
    from pxr import Usd,UsdGeom,Sdf
    s=Usd.Stage.CreateInMemory();m=UsdGeom.Mesh.Define(s,'/M')
    m.CreateSubdivisionSchemeAttr('none');m.CreateNormalsAttr([(0,0,1)]*6)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('gn_authored_value',Sdf.ValueTypeNames.Float,'constant').Set(0.5)
    with pytest.raises(DirectMeshUnsupported,match='unsupported_primvar_type:gn_authored_value:float'):
        contract_from_usd(m.GetPrim(),quad())


def test_live_contract_captures_material_subset_dependency():
    from pxr import Usd,UsdGeom
    s=Usd.Stage.CreateInMemory();m=UsdGeom.Mesh.Define(s,'/M')
    m.CreateSubdivisionSchemeAttr('none');m.CreateNormalsAttr([(0,0,1)]*6)
    sub=UsdGeom.Subset.Define(s,'/M/Red');sub.CreateElementTypeAttr('face');sub.CreateFamilyNameAttr('materialBind');sub.CreateIndicesAttr([0,1])
    sub.GetPrim().CreateRelationship('material:binding').SetTargets(['/Looks/Red'])
    contract=contract_from_usd(m.GetPrim(),quad(),material_paths=('/Looks/Red',))
    assert contract.subsets==(SubsetContract('/M/Red',(0,)),)


@pytest.mark.parametrize("has_mesh", [True, False])
def test_real_geometryset_instances_accessor_never_silently_drops_instances(has_mesh):
    geometry=NS(mesh=quad() if has_mesh else None,instances_pointcloud=lambda:object(),instance_references=lambda:[])
    evaluated=NS(type='MESH',evaluated_geometry=lambda:geometry)
    patch=extract_object_patch(NS(evaluated_get=lambda _:evaluated),None,contract())
    assert not patch.supported and patch.reason=='evaluated_instances_component_requires_population'


def test_real_geometryset_no_instances_accessor_keeps_mesh_route():
    geometry=NS(mesh=quad(),instances_pointcloud=lambda:None,instance_references=lambda:[])
    evaluated=NS(type='MESH',evaluated_geometry=lambda:geometry)
    assert extract_object_patch(NS(evaluated_get=lambda _:evaluated),None,contract()).supported
