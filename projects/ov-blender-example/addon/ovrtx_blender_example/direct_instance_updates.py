# SPDX-License-Identifier: Apache-2.0
"""Candidate scoped GeometrySet -> current inline mesh/PointInstancer source.

Visits only a nominated evaluated owner and its direct prototype references.
Copies current evaluated mesh data, so stale bootstrap USD is never used as a
prototype. The coordinator retains one current contribution per live owner,
batches changed contributions without merging prior source text, and adopts local path
mapping overrides only after acknowledgment. This module never edits generation
mappings, Blender data, a file, or a live renderer.
"""
from dataclasses import dataclass
import numpy as np
from . import direct_mesh_updates as meshes

class InstanceUnsupported(ValueError):
    pass

@dataclass(frozen=True)
class InstanceSource:
    owner_identity: object = None
    projection_path: str = ''
    hidden_source_path: str = ''
    layer_text: str = ''
    mapping: object = None
    instance_count: int = 0
    prototype_count: int = 0
    fallback: tuple = ()
    reference_identities: tuple = ()


def _read(data, member, width, dtype):
    result=np.empty(len(data)*width,dtype=dtype)
    data.foreach_get(member,result)
    return result.reshape((-1,width)) if width>1 else result


def _mesh(stage,path,mesh,generation):
    from pxr import UsdGeom,Sdf,Tf,Vt,UsdShade
    from .scene_generation import blender_id
    primvars=[];names=set();uv_sources=set()
    layers=list(mesh.uv_layers)
    for layer in layers:
        name='st' if bool(getattr(layer,'active_render',False)) else Tf.MakeValidIdentifier(layer.name)
        if name in names:raise InstanceUnsupported('instance_prototype_primvar_name_collision')
        names.add(name);uv_sources.add(layer.name)
        primvars.append(meshes.PrimvarContract('primvars:'+name,layer.name,'TexCoord2fArray','faceVarying',False,'uv'))
    kinds={'FLOAT':'FloatArray','INT':'IntArray','BOOLEAN':'BoolArray',
           'FLOAT_VECTOR':'Float3Array','FLOAT2':'Float2Array','FLOAT_COLOR':'Color3fArray','BYTE_COLOR':'Color3fArray'}
    builtin={'position','material_index','sharp_face','sharp_edge'}
    for attr in mesh.attributes:
        if attr.name.startswith('.') or attr.name in builtin or attr.name in uv_sources:continue
        if attr.domain not in {'POINT','CORNER','FACE'} or attr.data_type not in kinds:
            raise InstanceUnsupported('instance_prototype_attribute_requires_population:'+attr.name)
        name=Tf.MakeValidIdentifier(attr.name)
        if name in names:raise InstanceUnsupported('instance_prototype_primvar_name_collision')
        names.add(name)
        primvars.append(meshes.PrimvarContract('primvars:'+name,attr.name,kinds[attr.data_type],
            {'POINT':'vertex','CORNER':'faceVarying','FACE':'uniform'}[attr.domain]))
    materials=[]
    for material in mesh.materials:
        if material is None:
            materials.append(None);continue
        original=getattr(material,'original',None) or material
        mapping=generation.blender_prim_paths.get(blender_id(original,'MATERIAL'))
        if mapping is None:raise InstanceUnsupported('instance_material_not_populated')
        materials.append(str(mapping.schema_path))
    subsets=tuple(meshes.SubsetContract(path+'/Material_'+str(index),(index,)) for index in range(len(materials))) if len(materials)>1 else ()
    contract=meshes.MeshContract(path,primvars=tuple(primvars),subsets=subsets,material_slots=meshes.material_slot_ids(mesh))
    patch=meshes.build_mesh_patch(mesh,contract)
    if not patch.supported:raise InstanceUnsupported(patch.reason)
    usd_mesh=UsdGeom.Mesh.Define(stage,path)
    usd_mesh.CreateSubdivisionSchemeAttr('none');usd_mesh.SetNormalsInterpolation('faceVarying')
    for subset in subsets:
        child=UsdGeom.Subset.Define(stage,subset.path)
        child.CreateElementTypeAttr('face');child.CreateFamilyNameAttr('materialBind')
    for row in patch.values:
        target=stage.GetPrimAtPath(row['prim_path'])
        typ=getattr(Sdf.ValueTypeNames,row['value_type'])
        target.CreateAttribute(row['attribute_name'],typ).Set(row['value'].tolist())
    for pv in primvars:
        usd_mesh.GetPrim().GetAttribute(pv.usd_name).SetMetadata('interpolation',pv.interpolation)
    if subsets:
        UsdGeom.Subset.SetFamilyType(usd_mesh,'materialBind','partition')
        for index,subset in enumerate(subsets):
            if materials[index]:UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(subset.path)).GetDirectBindingRel().SetTargets([materials[index]])
    elif materials and materials[0]:
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).GetDirectBindingRel().SetTargets([materials[0]])


def _trs(matrix):
    from pxr import Gf
    m=np.asarray(matrix,dtype=np.float64)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-6):
        raise InstanceUnsupported('instance_transform_not_finite_affine')
    scale=np.linalg.norm(m[:3,:3],axis=0)
    if np.any(scale<1e-10):raise InstanceUnsupported('instance_transform_degenerate')
    rotation=m[:3,:3]/scale
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5):
        raise InstanceUnsupported('instance_transform_has_shear')
    if np.linalg.det(rotation)<0:
        scale[0]*=-1;rotation[:,0]*=-1
    rot4=np.eye(4);rot4[:3,:3]=rotation
    quat=Gf.Matrix4d(*rot4.T.reshape(-1).tolist()).ExtractRotationQuat()
    return tuple(m[:3,3]),Gf.Quatf(float(quat.GetReal()),Gf.Vec3f(*quat.GetImaginary())),tuple(scale)


def _prototype(stage,path,reference,depsgraph,generation):
    if getattr(reference,'type',None)=='MESH' and callable(getattr(reference,'evaluated_get',None)):
        with meshes.evaluated_mesh(reference,depsgraph) as mesh:_mesh(stage,path,mesh,generation)
        return
    if hasattr(reference,'mesh'):
        instances=getattr(reference,'instances_pointcloud',None)
        if callable(instances) and instances() is not None:
            raise InstanceUnsupported('nested_instances_require_population')
        if any(getattr(reference,name,None) is not None for name in ('curves','pointcloud','volume','grease_pencil')):
            raise InstanceUnsupported('prototype_non_mesh_component_requires_population')
        if reference.mesh is not None:
            _mesh(stage,path,reference.mesh,generation);return
    raise InstanceUnsupported('instance_reference_requires_mesh_geometry')


def compile_owner(obj,depsgraph,generation,*,projection_path=None,render_visible=True,allow_mesh_only=False):
    """Compile the current nominated owner; references are evaluated once each.

    Current mixed realized mesh + mesh instances and instances-only output are
    supported. Collections, nested instance references, custom per-instance
    attributes and shear explicitly fall back until their contracts are proven.
    """
    from pxr import Usd,UsdGeom,Gf,Vt,Sdf
    from .scene_generation import blender_id,BlenderPrimPath
    original=getattr(obj,'original',None) or obj
    identity=blender_id(original,'OBJECT')
    path=projection_path or '/World/_DirectGeometry/Object_'+str(identity.session_uid)
    try:
        if getattr(original,'rigid_body',None) or getattr(original,'rigid_body_constraint',None):
            raise InstanceUnsupported('instance_physics_requires_population')
        if getattr(original,'children',()):raise InstanceUnsupported('instance_owner_children_require_population')
        if not Sdf.Path(path).IsPrimPath() or not path.startswith('/World/'):
            raise InstanceUnsupported('instance_projection_path_invalid')
        evaluated=original.evaluated_get(depsgraph)
        render_visible = render_visible and not bool(getattr(evaluated,'hide_render',False)) and not bool(getattr(original,'hide_viewport',False))
        accessor=getattr(evaluated,'evaluated_geometry',None)
        if not callable(accessor):raise InstanceUnsupported('geometry_set_api_unavailable')
        geometry=accessor() # Retain owner through all borrowed mesh/attribute reads.
        if any(getattr(geometry,name,None) is not None for name in ('curves','pointcloud','volume','grease_pencil')):
            raise InstanceUnsupported('owner_non_mesh_component_requires_population')
        get_instances=getattr(geometry,'instances_pointcloud',None)
        get_references=getattr(geometry,'instance_references',None)
        if not callable(get_instances) or not callable(get_references):
            raise InstanceUnsupported('geometry_set_instance_api_unavailable')
        pointcloud=get_instances()
        if pointcloud is None:
            if not allow_mesh_only:raise InstanceUnsupported('owner_has_no_instance_component')
            attributes={};indices=np.empty(0,dtype=np.int32);transforms=[];references=[]
        else:
            attributes=pointcloud.attributes
            for attr in attributes:
                # Instance on Points can carry the source Mesh Line's internal
                # edge endpoint storage onto the instance point domain. It is
                # topology bookkeeping, not instance shading or placement.
                if attr.name not in {'instance_transform','.reference_index','id','position','radius','.edge_verts'}:
                    raise InstanceUnsupported('instance_attribute_requires_population:'+attr.name)
            indices=_read(attributes['.reference_index'].data,'value',1,np.int32)
            transforms=attributes['instance_transform'].data
            if len(transforms)!=len(indices):raise InstanceUnsupported('instance_array_lengths_differ')
            references=get_references()
        reference_identities=set()
        for reference in references:
            if getattr(reference,'type',None)=='MESH' and callable(getattr(reference,'evaluated_get',None)):
                ref_original=getattr(reference,'original',None) or reference
                reference_identities.add(blender_id(ref_original,'OBJECT'))
                if getattr(ref_original,'data',None) is not None:
                    reference_identities.add(blender_id(ref_original.data,'MESH'))
        if len(indices) and (indices.min()<0 or indices.max()>=len(references)):
            raise InstanceUnsupported('instance_reference_index_out_of_range')
        # None references have no geometry. Omit their points rather than
        # manufacturing an empty prototype that the renderer may reject.
        used=sorted({int(index) for index in indices if references[int(index)] is not None})
        remap={old:new for new,old in enumerate(used)}
        keep=[i for i,index in enumerate(indices) if int(index) in remap]
        stage=Usd.Stage.CreateInMemory()
        root=UsdGeom.Xform.Define(stage,path)
        pose=Gf.Matrix4d(*np.asarray(evaluated.matrix_world,dtype=float).T.reshape(-1).tolist())
        root.AddTransformOp().Set(pose);root.SetResetXformStack(True)
        root.CreateVisibilityAttr('inherited' if render_visible else 'invisible')
        if geometry.mesh is not None and len(geometry.mesh.vertices):_mesh(stage,path+'/RealizedMesh',geometry.mesh,generation)
        if used:
            instancer=UsdGeom.PointInstancer.Define(stage,path+'/Instances')
            prototype_paths=[]
            for index in used:
                prototype_path=path+'/Instances/Prototype_'+str(remap[index])
                _prototype(stage,prototype_path,references[index],depsgraph,generation)
                prototype_paths.append(prototype_path)
            positions=[];quats=[];scales=[]
            for index in keep:
                position,quat,scale=_trs(transforms[index].value)
                positions.append(Gf.Vec3f(*position));quats.append(quat);scales.append(Gf.Vec3f(*scale))
            instancer.CreatePrototypesRel().SetTargets(prototype_paths)
            instancer.CreateProtoIndicesAttr([remap[int(indices[index])] for index in keep])
            instancer.CreatePositionsAttr(positions);instancer.CreateScalesAttr(scales)
            # Float orientation avoids ambiguous half-bit readback in installed SDK.
            instancer.GetPrim().CreateAttribute('orientationsf',Sdf.ValueTypeNames.QuatfArray).Set(quats)
            ids=attributes.get('id')
            if ids is not None:
                current_ids=_read(ids.data,'value',1,np.int64)
                instancer.CreateIdsAttr([int(current_ids[index]) for index in keep])
        previous=generation.blender_prim_paths.get(identity)
        hidden=str(previous.object_path) if previous is not None else ''
        if hidden and (path==hidden or path.startswith(hidden+'/') or hidden.startswith(path+'/')):
            raise InstanceUnsupported('instance_projection_overlaps_original_owner')
        if hidden:
            UsdGeom.Imageable(stage.OverridePrim(hidden)).CreateVisibilityAttr('invisible')
        mapping=BlenderPrimPath(original.name,str(original.type),path,path,int(getattr(getattr(original,'data',None),'session_uid',0)))
        return InstanceSource(identity,path,hidden,stage.GetRootLayer().ExportToString(),mapping,len(keep),len(used),reference_identities=tuple(sorted(reference_identities)))
    except (InstanceUnsupported,meshes.DirectMeshUnsupported,ValueError,TypeError,AttributeError,KeyError) as error:
        return InstanceSource(owner_identity=identity,projection_path=path,fallback=(str(error),))


def packet_for_source(source, *, current_values=()):
    """Single-owner zoo transport; coordinator merges current live contributions.

    current_values supplies existing `(prim_path, attribute_name)` keys, ideally
    from the coordinator's owner index. Current evaluated source supersedes
    native overrides within the new projection, including SDK prototype paths.
    """
    if source.fallback or not source.layer_text:
        raise ValueError('Cannot submit a rejected instance source')
    root=source.projection_path
    superseded={target for target in current_values if target[0]==root or target[0].startswith(root+'/')}
    roots=[root]
    if source.hidden_source_path:
        roots.append(source.hidden_source_path)
        superseded.add((source.hidden_source_path,'visibility'))
    return dict(experimental=True,inline_population=dict(usda=source.layer_text,owned_roots=roots,
        superseded_attributes=[dict(prim_path=path,attribute_name=name) for path,name in sorted(superseded)]))
