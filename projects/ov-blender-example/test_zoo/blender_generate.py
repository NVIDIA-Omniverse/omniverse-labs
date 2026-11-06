# SPDX-License-Identifier: Apache-2.0
"""Background Blender5.2 author/export/edit zoo, using the actual add-on compiler."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import uuid

import bpy
from mathutils import Vector
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

sys.path.insert(0,str(Path(__file__).parent))
from catalog import BY_ID


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--addon-source',type=Path,required=True)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--edits',action='store_true')
    return p.parse_args(sys.argv[sys.argv.index('--')+1:])


def material(name='ZooMaterial',color=(.45,.08,.025,1)):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    p=mat.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=color
    p.inputs['Roughness'].default_value=.3
    mat.diffuse_color=color
    return mat,p


def mesh(kind='cube',name='Subject'):
    if kind=='sphere': bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12)
    elif kind=='torus': bpy.ops.mesh.primitive_torus_add(major_segments=24,minor_segments=12)
    elif kind=='ngon': bpy.ops.mesh.primitive_circle_add(vertices=7,fill_type='NGON')
    else: bpy.ops.mesh.primitive_cube_add(size=1.5)
    obj=bpy.context.object;obj.name=name;obj.location.z=1.1
    mat,p=material(name+'Material');obj.data.materials.append(mat)
    return obj,p


def point_at(obj,point):
    obj.rotation_euler=(Vector(point)-obj.location).to_track_quat('-Z','Y').to_euler()


def base():
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.resolution_x=512;scene.render.resolution_y=384;scene.render.resolution_percentage=100
    scene.frame_start=1;scene.frame_end=24;scene.render.fps=24
    scene.world=bpy.data.worlds.new('ZooWorld');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.08,.08,.08,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3
    bpy.ops.object.camera_add(location=(5,-7,4.3));camera=bpy.context.object;camera.name='ZooCamera'
    camera.data.lens=48;point_at(camera,(0,0,1));scene.camera=camera
    bpy.ops.object.light_add(type='AREA',location=(1,-3,5));light=bpy.context.object;light.name='Key'
    light.data.energy=850;light.data.shape='DISK';light.data.size=4;point_at(light,(0,0,1))
    bpy.ops.mesh.primitive_plane_add(size=20);floor=bpy.context.object;floor.name='Floor'
    floor.data.materials.append(material('FloorMaterial',(.22,.25,.3,1))[0])
    return scene


PRINCIPLED={
 'base_color':('Base Color',(.025,.6,.15,1)), 'roughness':('Roughness',.8),
 'metallic':('Metallic',1.),'alpha':('Alpha',.3),'transmission':('Transmission Weight',1.),
 'ior':('IOR',2.),'specular':('Specular IOR Level',.9),'anisotropy':('Anisotropic',.8),
 'coat':('Coat Weight',1.),'coat_roughness':('Coat Roughness',.5),'coat_ior':('Coat IOR',2.),
 'coat_tint':('Coat Tint',(.05,.3,1,1)),'subsurface':('Subsurface Weight',.7),
 'subsurface_radius':('Subsurface Radius',(1,.4,.2)),'sheen':('Sheen Weight',1.),
 'sheen_tint':('Sheen Tint',(.8,.15,.6,1)), 'emission':('Emission Color',(.1,.6,1,1)),
 'emission_strength':('Emission Strength',3.),'thin_film':('Thin Film Thickness',500.)}


def image_map(out,name='Checker',normal=False):
    image=bpy.data.images.new(name,width=64,height=64,alpha=True)
    data=[]
    for y in range(64):
        for x in range(64):
            check=((x//8+y//8)%2)
            data.extend((.5,.5,1.,1.) if normal else ((.85,.08,.015,1) if check else (.03,.2,.8,.2)))
    image.pixels.foreach_set(data)
    image.filepath_raw=str(out/(name+'.png'));image.file_format='PNG';image.save()
    return image


def geo_nodes(obj,variant):
    group=bpy.data.node_groups.new('ZooGeometry','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    n,l=group.nodes,group.links;inp=n.new('NodeGroupInput');output=n.new('NodeGroupOutput')
    modifier=obj.modifiers.new('ZooNodes','NODES');modifier.node_group=group
    source=inp.outputs['Geometry']
    if variant in {'instances','realized_instances'}:
        grid=n.new('GeometryNodeMeshGrid');grid.inputs['Size X'].default_value=3;grid.inputs['Size Y'].default_value=3
        grid.inputs['Vertices X'].default_value=3;grid.inputs['Vertices Y'].default_value=3
        inst=n.new('GeometryNodeInstanceOnPoints');l.new(grid.outputs['Mesh'],inst.inputs['Points'])
        l.new(source,inst.inputs['Instance']);inst.inputs['Scale'].default_value=(.22,.22,.22)
        source=inst.outputs['Instances']
        if variant=='realized_instances':
            real=n.new('GeometryNodeRealizeInstances');l.new(source,real.inputs['Geometry']);source=real.outputs['Geometry']
    elif variant=='grid':
        grid=n.new('GeometryNodeMeshGrid');grid.inputs['Vertices X'].default_value=8;grid.inputs['Vertices Y'].default_value=8
        source=grid.outputs['Mesh']
    elif variant=='curve_to_mesh':
        line=n.new('GeometryNodeCurvePrimitiveLine');line.inputs['End'].default_value=(0,0,2)
        circle=n.new('GeometryNodeCurvePrimitiveCircle');circle.inputs['Radius'].default_value=.2
        sweep=n.new('GeometryNodeCurveToMesh');l.new(line.outputs['Curve'],sweep.inputs['Curve']);l.new(circle.outputs['Curve'],sweep.inputs['Profile Curve']);source=sweep.outputs['Mesh']
    elif variant=='points_to_vertices':
        points=n.new('GeometryNodeMeshToPoints');l.new(source,points.inputs['Mesh'])
        vertices=n.new('GeometryNodePointsToVertices');l.new(points.outputs['Points'],vertices.inputs['Points']);source=vertices.outputs['Mesh']
    elif variant=='named_attribute':
        attr=n.new('GeometryNodeStoreNamedAttribute');attr.data_type='FLOAT';attr.domain='POINT'
        attr.inputs['Name'].default_value='zoo_weight';attr.inputs['Value'].default_value=.3
        l.new(source,attr.inputs['Geometry']);source=attr.outputs['Geometry']
    elif variant=='material_assignment':
        assign=n.new('GeometryNodeSetMaterial');assign.inputs['Material'].default_value=obj.data.materials[0]
        l.new(source,assign.inputs['Geometry']);source=assign.outputs['Geometry']
    transform=n.new('GeometryNodeTransform');transform.name='EditTransform'
    l.new(source,transform.inputs['Geometry']);l.new(transform.outputs['Geometry'],output.inputs['Geometry'])
    if variant=='animated_field':
        transform.inputs['Translation'].default_value=(0,0,0);transform.inputs['Translation'].keyframe_insert('default_value',frame=1)
        transform.inputs['Translation'].default_value=(1,0,.5);transform.inputs['Translation'].keyframe_insert('default_value',frame=24)
    return group


def build(case,out):
    scene=base();family,variant=case.family,case.variant
    obj,p=mesh('sphere' if family in {'material','texture','openpbr_direct'} else variant if family=='mesh' else 'cube')
    if family=='mesh':
        if variant=='loose_edges':
            obj.data.clear_geometry()
            obj.data.from_pydata([(0,0,0),(1,0,0),(1,1,0)],[(0,1),(1,2)],[])
        elif variant=='negative_scale':obj.scale.x=-1.2
        elif variant=='nonuniform_parent':
            parent=bpy.data.objects.new('Parent',None);scene.collection.objects.link(parent);parent.scale=(2,1,.7);obj.parent=parent
        elif variant=='multi_material':
            obj.data.materials.append(material('SecondMaterial',(.02,.4,.8,1))[0])
            for face in obj.data.polygons: face.material_index=face.index%2
    elif family=='modifier':
        types={'bevel':'BEVEL','subdivision':'SUBSURF','mirror':'MIRROR','array':'ARRAY','solidify':'SOLIDIFY',
               'boolean':'BOOLEAN','triangulate':'TRIANGULATE','decimate':'DECIMATE','weld':'WELD','displacement':'DISPLACE'}
        mod=obj.modifiers.new('ZooModifier',types[variant])
        if variant=='bevel':mod.width=.2;mod.segments=3
        elif variant=='subdivision':mod.levels=2;mod.render_levels=2
        elif variant=='array':mod.count=3;mod.relative_offset_displace=(1.2,0,0)
        elif variant=='solidify':mod.thickness=.2
        elif variant=='decimate':mod.ratio=.5
        elif variant=='boolean':
            cutter,_=mesh('sphere','Cutter');cutter.location=(.6,0,1.4);mod.object=cutter;cutter.hide_render=True
        elif variant=='displacement':
            mod.texture=bpy.data.textures.new('DisplaceClouds',type='CLOUDS');mod.strength=.3
    elif family=='modifier_stack':
        if variant=='array_bevel':
            obj.modifiers.new('ZooArray','ARRAY').count=3
            bevel=obj.modifiers.new('ZooBevel','BEVEL');bevel.width=.15;bevel.segments=3
        elif variant=='subdivision_displacement':
            sub=obj.modifiers.new('ZooSubdivision','SUBSURF');sub.levels=sub.render_levels=2
            displace=obj.modifiers.new('ZooDisplacement','DISPLACE');displace.strength=.3
            displace.texture=bpy.data.textures.new('StackNoise',type='CLOUDS')
        else:
            obj.modifiers.new('ZooMirror','MIRROR');obj.modifiers.new('ZooSolidify','SOLIDIFY').thickness=.15
    elif family=='instances':
        if variant=='linked_mesh':
            for i in range(3):
                copy=obj.copy();copy.data=obj.data;scene.collection.objects.link(copy);copy.location=(i*1.7-1.7,0,1)
        else:
            collection=bpy.data.collections.new('InstanceSource');scene.collection.children.link(collection)
            for col in list(obj.users_collection):col.objects.unlink(obj)
            collection.objects.link(obj)
            if variant=='nested_collection':
                outer=bpy.data.collections.new('NestedSource');scene.collection.children.link(outer)
                nested=bpy.data.objects.new('NestedInstance',None);nested.instance_type='COLLECTION';nested.instance_collection=collection;outer.objects.link(nested);collection=outer
            for i in range(3):
                inst=bpy.data.objects.new('Instance'+str(i),None);inst.instance_type='COLLECTION';inst.instance_collection=collection
                inst.location=(i*1.7-1.7,1.7,0);scene.collection.objects.link(inst)
    elif family=='geometry_nodes':geo_nodes(obj,variant)
    elif family=='curve':
        bpy.data.objects.remove(obj,do_unlink=True)
        if variant=='text':
            data=bpy.data.curves.new('ZooText','FONT');data.body='OVStage';data.extrude=.08
        elif variant=='hair_curves':
            data=bpy.data.hair_curves.new('ZooHair');data.add_curves([4,4,4])
            position=data.attributes['position']
            for i,item in enumerate(position.data):item.vector=((i//4-.8)*.4,math.sin(i)*.1,(i%4)*.4)
        else:
            data=bpy.data.curves.new('ZooCurve','CURVE');data.dimensions='3D';data.bevel_depth=.08;data.bevel_resolution=3
            spline=data.splines.new({'bezier':'BEZIER','poly':'POLY','nurbs':'NURBS'}[variant])
            points=spline.bezier_points if variant=='bezier' else spline.points;points.add(3)
            for i,point in enumerate(points):
                point.co=(i*.7-1,math.sin(i),.4*i) if variant=='bezier' else (i*.7-1,math.sin(i),.4*i,1)
                if variant=='bezier':point.handle_left_type=point.handle_right_type='AUTO'
        obj=bpy.data.objects.new('Subject',data);scene.collection.objects.link(obj);obj.location.z=.5
        data.materials.append(material()[0])
    elif family=='material':
        if variant in PRINCIPLED:
            socket,value=PRINCIPLED[variant];p.inputs[socket].default_value=value
            if variant.startswith('coat'):p.inputs['Coat Weight'].default_value=1
            if variant.startswith('subsurface'):p.inputs['Subsurface Weight'].default_value=.7
            if variant.startswith('sheen'):p.inputs['Sheen Weight'].default_value=1
            if variant.startswith('emission'):p.inputs['Emission Strength'].default_value=3
        else:
            tree=obj.active_material.node_tree;n,l=tree.nodes,tree.links;output=n.get('Material Output')
            types={'diffuse':'ShaderNodeBsdfDiffuse','glass':'ShaderNodeBsdfGlass','transparent':'ShaderNodeBsdfTransparent','principled_hair':'ShaderNodeBsdfHairPrincipled'}
            if variant in types:l.new(n.new(types[variant]).outputs[0],output.inputs['Surface'])
            else:
                other=n.new('ShaderNodeEmission');other.inputs['Color'].default_value=(.02,.2,1,1)
                mixer=n.new('ShaderNodeMixShader' if variant=='mix_shader' else 'ShaderNodeAddShader')
                l.new(p.outputs[0],mixer.inputs[-2]);l.new(other.outputs[0],mixer.inputs[-1]);l.new(mixer.outputs[0],output.inputs['Surface'])
    elif family=='texture':
        tree=obj.active_material.node_tree;n,l=tree.nodes,tree.links
        image=image_map(out,normal=variant in {'normal_map','coat_normal'})
        tex=n.new('ShaderNodeTexImage');tex.image=image;tex.name='EditTexture'
        if variant in {'image_linear','normal_map','coat_normal','bump','roughness','metallic','alpha','displacement'}:image.colorspace_settings.name='Non-Color'
        if variant=='packed_image':image.pack()
        coords=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeMapping');mapping.name='EditMapping'
        l.new(coords.outputs['UV'],mapping.inputs['Vector']);l.new(mapping.outputs['Vector'],tex.inputs['Vector'])
        source=tex.outputs['Color'];target=p.inputs['Base Color']
        if variant in {'uv_second_set','udim'}:
            uv=obj.data.uv_layers.new(name='SecondUV')
            for item in uv.data:item.uv*=2
            node=n.new('ShaderNodeUVMap');node.uv_map='SecondUV';l.new(node.outputs[0],mapping.inputs['Vector'])
            if variant=='udim':
                image.filepath_raw=str(out/'UDIM.1001.png');image.save()
                second=image_map(out,'UDIM.1002');second.colorspace_settings.name='sRGB'
                tiled=bpy.data.images.load(str(out/'UDIM.1001.png'),check_existing=False)
                tiled.source='TILED';tiled.filepath=str(out/'UDIM.<UDIM>.png')
                if 1002 not in [tile.number for tile in tiled.tiles]:tiled.tiles.new(1002)
                tiled.reload();tex.image=tiled
        elif variant in {'normal_map','coat_normal'}:
            normal=n.new('ShaderNodeNormalMap');l.new(source,normal.inputs['Color']);source=normal.outputs['Normal'];target=p.inputs['Coat Normal' if variant=='coat_normal' else 'Normal']
        elif variant=='bump':
            bump=n.new('ShaderNodeBump');l.new(source,bump.inputs['Height']);source=bump.outputs['Normal'];target=p.inputs['Normal']
        elif variant in {'roughness','metallic','alpha','emission'}:
            target=p.inputs[{'roughness':'Roughness','metallic':'Metallic','alpha':'Alpha','emission':'Emission Color'}[variant]]
            if variant=='emission':p.inputs['Emission Strength'].default_value=3
        elif variant=='displacement':
            displacement=n.new('ShaderNodeDisplacement');l.new(source,displacement.inputs['Height']);source=displacement.outputs[0];target=n.get('Material Output').inputs['Displacement']
        elif variant in {'noise','checker','gradient'}:
            proc=n.new({'noise':'ShaderNodeTexNoise','checker':'ShaderNodeTexChecker','gradient':'ShaderNodeTexGradient'}[variant]);source=proc.outputs[0]
        elif variant=='color_ramp':
            ramp=n.new('ShaderNodeValToRGB');l.new(source,ramp.inputs[0]);source=ramp.outputs['Color']
        elif variant=='mix_color':
            mix=n.new('ShaderNodeMixRGB');mix.inputs[0].default_value=.5;l.new(source,mix.inputs[1]);source=mix.outputs[0]
        l.new(source,target)
    elif family=='primvar':
        if variant in {'point_color','corner_color'}:
            attr=obj.data.color_attributes.new(name='zoo_color',type='FLOAT_COLOR',domain='POINT' if variant=='point_color' else 'CORNER')
            for item in attr.data:item.color=(.1,.7,.3,1)
            node=obj.active_material.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='zoo_color'
            obj.active_material.node_tree.links.new(node.outputs['Color'],p.inputs['Base Color'])
        elif variant in {'point_float','face_float'}:
            attr=obj.data.attributes.new('zoo_weight','FLOAT','POINT' if variant=='point_float' else 'FACE')
            for i,item in enumerate(attr.data):item.value=i/len(attr.data)
        elif variant=='material_index':
            obj.data.materials.append(material('FaceMaterial',(.02,.6,.7,1))[0])
            for face in obj.data.polygons:face.material_index=face.index%2
    elif family=='light':
        light=bpy.data.objects['Key'];light.data.type=variant.upper() if variant in {'point','spot','sun'} else 'AREA'
        if variant.startswith('area_'):light.data.shape=variant.removeprefix('area_').upper()
        if variant=='spot':light.data.spot_size=.8;light.data.spot_blend=.3
        if variant=='sun':light.data.energy=3
        if variant=='temperature':
            if hasattr(light.data,'use_temperature'):light.data.use_temperature=True;light.data.temperature=3500
            else:raise NotImplementedError('This Blender build lacks light temperature controls')
    elif family=='world':
        n=scene.world.node_tree.nodes;l=scene.world.node_tree.links;background=n['Background']
        if variant=='background_strength':background.inputs['Strength'].default_value=2
        elif variant=='background_color':background.inputs['Color'].default_value=(.1,.4,.9,1)
        elif variant=='procedural_sky':l.new(n.new('ShaderNodeTexSky').outputs['Color'],background.inputs['Color'])
        else:
            environment=n.new('ShaderNodeTexEnvironment')
            if variant=='hdri_studio':
                asset=Path(os.environ.get('OV_TEST_ZOO_ASSET_ROOT',str(out.parent.parent/'assets')))/'studio_small_09_1k.hdr'
                if not asset.is_file():raise NotImplementedError('Fetch the bounded CC0 studio HDRI with fetch_assets.py first')
                environment.image=bpy.data.images.load(str(asset))
            else:environment.image=image_map(out,'Environment')
            l.new(environment.outputs['Color'],background.inputs['Color'])
    elif family=='camera':
        camera=scene.camera.data
        if variant=='orthographic':camera.type='ORTHO';camera.ortho_scale=6
        elif variant=='focal_length':camera.lens=80
        elif variant=='shift':camera.shift_x=.2;camera.shift_y=.1
        elif variant=='clipping':camera.clip_start=.5;camera.clip_end=15
        elif variant=='depth_of_field':camera.dof.use_dof=True;camera.dof.focus_object=obj;camera.dof.aperture_fstop=1.4
        elif variant=='panoramic':camera.type='PANO'
    elif family=='animation':
        if variant=='shape_key':
            obj.shape_key_add(name='Basis');key=obj.shape_key_add(name='ZooShape')
            for point in key.data:point.co.z+=.5*point.co.x
            key.value=0;key.keyframe_insert('value',frame=1);key.value=1;key.keyframe_insert('value',frame=24)
        elif variant=='material_keyframe':
            p.inputs['Base Color'].keyframe_insert('default_value',frame=1);p.inputs['Base Color'].default_value=(.02,.8,.1,1);p.inputs['Base Color'].keyframe_insert('default_value',frame=24)
        elif variant=='constraint':
            target=bpy.data.objects.new('AnimatedTarget',None);scene.collection.objects.link(target)
            target.location=(0,0,1);target.keyframe_insert('location',frame=1);target.location.x=2;target.keyframe_insert('location',frame=24)
            constraint=obj.constraints.new('COPY_LOCATION');constraint.target=target
        elif variant=='driver':
            obj.driver_add('location',0).driver.expression='frame / 12.0 - 1.0'
        elif variant=='armature':
            bpy.ops.object.armature_add();rig=bpy.context.object;rig.name='ZooRig'
            group=obj.vertex_groups.new(name='Bone');group.add(list(range(len(obj.data.vertices))),1,'REPLACE')
            mod=obj.modifiers.new('ZooArmature','ARMATURE');mod.object=rig
            bone=rig.pose.bones[0];bone.rotation_mode='XYZ';bone.keyframe_insert('rotation_euler',frame=1)
            bone.rotation_euler.y=.7;bone.keyframe_insert('rotation_euler',frame=24)
        elif variant=='topology':
            mod=obj.modifiers.new('AnimatedTopology','BUILD');mod.frame_start=-4;mod.frame_duration=28
        else:
            obj.keyframe_insert('location',frame=1);obj.location.x=1.5;obj.keyframe_insert('location',frame=24)
    elif family=='visibility':
        if variant=='object_hidden':obj.hide_render=True
        elif variant in {'collection_hidden','view_layer_exclude'}:
            col=bpy.data.collections.new('VisibilityCollection');scene.collection.children.link(col)
            for old in list(obj.users_collection):old.objects.unlink(obj)
            col.objects.link(obj)
            if variant=='collection_hidden':col.hide_render=True
            else:bpy.context.view_layer.layer_collection.children[col.name].exclude=True
        elif variant=='camera_ray':obj.visible_camera=False
        else:obj.is_holdout=True
    elif family=='physics':
        from ovrtx_blender_example.physics_endpoints import initialize_owner,initialize_joint
        from ovrtx_blender_example.physics_authoring import apply_anchor
        scene.ov.physics_scene.schema_opinion='APPLY';initialize_owner(obj)
        body=obj.ov.physics;body.rigid_body.schema_opinion='APPLY'
        body.rigid_body.role={'static':'STATIC','kinematic':'KINEMATIC'}.get(variant,'DYNAMIC');body.rigid_body.mass_kg=1
        body.rigid_body.enabled=variant!='disabled'
        collider=body.colliders.add();collider.name=collider.persistent_id=uuid.uuid5(uuid.NAMESPACE_URL,case.id).hex;collider.schema_opinion='APPLY'
        collider.shape={'analytic_sphere':'ANALYTIC_SPHERE','analytic_box':'BOX','capsule':'CAPSULE'}.get(variant,'CONVEX_HULL')
        if variant.startswith('joint_'):
            handle=bpy.data.objects.new('ZooJoint',None);scene.collection.objects.link(handle);handle.location=(0,0,2)
            joint=initialize_joint(handle);joint.enabled=True;joint.type=variant.removeprefix('joint_').upper();joint.axis='Z'
            joint.endpoint_a.kind='WORLD';joint.endpoint_b.object=obj
            bpy.context.view_layer.update();apply_anchor(handle,scene,bpy.context.evaluated_depsgraph_get())
    elif family=='volume':
        source=obj;source.name='VolumeSource';source.hide_render=True
        data=bpy.data.volumes.new('ZooVolume');volume=bpy.data.objects.new('Subject',data);scene.collection.objects.link(volume)
        mod=volume.modifiers.new('MeshToVolume','MESH_TO_VOLUME');mod.object=source;mod.voxel_amount=24;mod.density=.6
        mat=bpy.data.materials.new('VolumeMaterial');mat.use_nodes=True;n=mat.node_tree.nodes;l=mat.node_tree.links
        node=n.new('ShaderNodeVolumePrincipled');node.inputs['Density'].default_value=.4;l.new(node.outputs[0],n['Material Output'].inputs['Volume']);data.materials.append(mat)
        obj=volume
        if variant=='volume_to_mesh':
            obj,_=mesh('cube','Isosurface');mod=obj.modifiers.new('VolumeToMesh','VOLUME_TO_MESH');mod.object=volume;mod.threshold=.1
            volume.hide_render=True
    elif family=='render':
        if variant=='exposure':scene.view_settings.exposure=2
        elif variant in {'agx','standard'}:scene.view_settings.view_transform='AgX' if variant=='agx' else 'Standard'
        elif variant=='transparent_film':scene.render.film_transparent=True
    elif family=='challenge':
        n=obj.active_material.node_tree.nodes;l=obj.active_material.node_tree.links
        if variant=='volume_shader':
            volume=n.new('ShaderNodeVolumePrincipled');volume.inputs['Density'].default_value=.4
            l.new(volume.outputs[0],n['Material Output'].inputs['Volume'])
        elif variant=='shader_to_rgb':
            node=n.new('ShaderNodeShaderToRGB');emission=n.new('ShaderNodeEmission')
            l.new(p.outputs[0],node.inputs[0]);l.new(node.outputs[0],emission.inputs[0]);l.new(emission.outputs[0],n['Material Output'].inputs['Surface'])
        elif variant=='osl_script':n.new('ShaderNodeScript')
        elif variant in {'geometry_simulation_zone','geometry_bake_node'}:
            group=geo_nodes(obj,'transform')
            output_node=next(node for node in group.nodes if node.type=='GROUP_OUTPUT')
            source=group.nodes['EditTransform'].outputs['Geometry']
            if variant=='geometry_bake_node':
                bake=group.nodes.new('GeometryNodeBake')
                if not bake.bake_items:bake.bake_items.new('GEOMETRY','Geometry')
                group.links.new(source,bake.inputs['Geometry']);group.links.new(bake.outputs['Geometry'],output_node.inputs['Geometry'])
            else:
                output=group.nodes.new('GeometryNodeSimulationOutput');entry=group.nodes.new('GeometryNodeSimulationInput');entry.pair_with_output(output)
                group.links.new(source,entry.inputs['Geometry']);group.links.new(entry.outputs['Geometry'],output.inputs['Geometry']);group.links.new(output.outputs['Geometry'],output_node.inputs['Geometry'])
            obj['zoo_cache_status']='Connected graph; disk bake not performed'
        else:raise NotImplementedError('Dedicated workflow fixture pending; do not substitute mesh rendering for '+variant)
    scene.frame_set(1);bpy.context.view_layer.update()
    return obj


def snapshot():
    def attribute_digest(attribute):
        values=[]
        for item in attribute.data:
            if hasattr(item,'value'):values.append(item.value)
            elif hasattr(item,'vector'):values.append(tuple(item.vector))
            elif hasattr(item,'color'):values.append(tuple(item.color))
        return hashlib.sha256(repr(values).encode()).hexdigest()
    dg=bpy.context.evaluated_depsgraph_get();objects=[]
    for obj in bpy.context.scene.objects:
        row={'name':obj.name,'type':obj.type,'matrix':[list(r) for r in obj.matrix_world],
             'hide_render':obj.hide_render,'modifiers':[m.type for m in obj.modifiers]}
        if obj.type=='MESH':
            evaluated=obj.evaluated_get(dg);data=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
            try:
                row.update(vertices=len(data.vertices),polygons=len(data.polygons),
                    attributes=[{'name':a.name,'domain':a.domain,'type':a.data_type,
                        'value_digest':attribute_digest(a)} for a in data.attributes],
                    point_digest=hashlib.sha256(repr([tuple(v.co) for v in data.vertices]).encode()).hexdigest())
            finally:evaluated.to_mesh_clear()
        objects.append(row)
    materials=[]
    for mat in bpy.data.materials:
        if not mat.users or not mat.use_nodes:continue
        nodes=[]
        for node in mat.node_tree.nodes:
            inputs={}
            for socket in node.inputs:
                if hasattr(socket,'default_value'):
                    value=socket.default_value
                    if isinstance(value,(int,float,str,bool)):inputs[socket.name]=value
                    elif hasattr(value,'__len__'):inputs[socket.name]=list(value)
            nodes.append({'type':node.bl_idname,'inputs':inputs})
        materials.append({'name':mat.name,'nodes':nodes,
            'links':[(l.from_node.name,l.from_socket.name,l.to_node.name,l.to_socket.name) for l in mat.node_tree.links]})
    scene=bpy.context.scene
    return {'frame':scene.frame_current,'objects':objects,'materials':materials,
            'lights':[{'name':o.name,'type':o.data.type,'energy':o.data.energy,'color':list(o.data.color)} for o in scene.objects if o.type=='LIGHT'],
            'camera':{'type':scene.camera.data.type,'lens':scene.camera.data.lens,'ortho_scale':scene.camera.data.ortho_scale},
            'exposure':scene.view_settings.exposure,'size':[scene.render.resolution_x,scene.render.resolution_y],
            'instances':sum(1 for i in dg.object_instances if i.is_instance)}


def edit(kind,obj,case):
    scene=bpy.context.scene
    if kind=='translate':obj.location.x+=.6
    elif kind=='frame':
        for frame in range(1,25):scene.frame_set(frame);bpy.context.view_layer.update()
    elif kind=='topology' and obj.type=='MESH':
        import bmesh
        bm=bmesh.new();bm.from_mesh(obj.data)
        bmesh.ops.subdivide_edges(bm,edges=list(bm.edges)[:4],cuts=1,use_grid_fill=True)
        bm.to_mesh(obj.data);bm.free();obj.data.update()
    elif kind=='delete_add':
        clone=obj.copy();clone.data=obj.data.copy();scene.collection.objects.link(clone);clone.name='AddedSubject';clone.location.x+=1.5
        bpy.data.objects.remove(obj,do_unlink=True);obj=clone
    elif kind=='modifier':
        mod=obj.modifiers[0]
        if hasattr(mod,'width'):mod.width*=2
        elif hasattr(mod,'count'):mod.count+=1
        elif hasattr(mod,'levels'):mod.levels+=1;mod.render_levels=mod.levels
        else:mod.show_render=not mod.show_render;mod.show_viewport=mod.show_render
    elif kind=='instance_count':
        clone=obj.copy();scene.collection.objects.link(clone);clone.location.y+=2
    elif kind=='node_input':obj.modifiers['ZooNodes'].node_group.nodes['EditTransform'].inputs['Scale'].default_value=(1.6,.7,1.2)
    elif kind=='curve_radius':
        if hasattr(obj.data,'bevel_depth'):obj.data.bevel_depth=.2
        else:obj.scale*=1.3
    elif kind in {'material_value','material_graph'}:
        tree=obj.active_material.node_tree;p=tree.nodes.get('Principled BSDF')
        if kind=='material_value':p.inputs['Base Color'].default_value=(.02,.65,.2,1);p.inputs['Roughness'].default_value=.75
        else:
            emission=tree.nodes.new('ShaderNodeEmission');emission.inputs[0].default_value=(.05,.15,1,1)
            tree.links.new(emission.outputs[0],tree.nodes['Material Output'].inputs['Surface'])
    elif kind=='texture_mapping':obj.active_material.node_tree.nodes['EditMapping'].inputs['Scale'].default_value=(3,2,1)
    elif kind=='primvar_value':
        if case.variant=='uv_edit':
            for item in obj.data.uv_layers.active.data:item.uv.x+=.3
        elif case.variant=='normal_edit':
            for face in obj.data.polygons:face.use_smooth=True
        elif case.variant=='material_index':
            for face in obj.data.polygons:face.material_index=1-face.material_index
        else:
            attr=obj.data.attributes.get('zoo_color') or obj.data.attributes.get('zoo_weight')
            for item in attr.data:
                if hasattr(item,'color'):item.color=(.8,.05,.3,1)
                else:item.value=1-item.value
    elif kind in {'light_energy','light_color'}:
        light=bpy.data.objects['Key'].data
        if kind=='light_energy':light.energy*=.2
        else:light.color=(.1,.5,1)
    elif kind=='world_value':scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=1.2
    elif kind=='camera_value':scene.camera.data.lens*=.7;scene.camera.data.ortho_scale*=.7
    elif kind=='visibility':obj.hide_render=not obj.hide_render;obj.visible_camera=True;obj.is_holdout=False
    elif kind=='physics_value':obj.ov.physics.rigid_body.mass_kg=2
    elif kind=='presentation':scene.view_settings.exposure+=1
    elif kind=='resize':scene.render.resolution_x=640;scene.render.resolution_y=360
    elif kind=='openpbr_value':obj['openpbr_edit']=1
    elif kind=='volume_value':
        if obj.type=='VOLUME':obj.modifiers['MeshToVolume'].density=.2
        else:obj.modifiers['VolumeToMesh'].threshold=.4
    bpy.context.view_layer.update()
    return obj


def export(scene,out,step,case):
    from ovrtx_blender_example.scene_generation import SceneGenerationOwner,blender_id
    from ovrtx_blender_example.materialx_openpbr_conversion import scene_layer_from_materials
    from ovrtx_blender_example.light_scene_layer import scene_layer_from_lights
    from ovrtx_blender_example.ovrtx_scene_composition import compose
    from ovrtx_blender_example.render_requests import camera_projection_from_scene_camera
    owner=SceneGenerationOwner(out/step/'generation')
    generation=owner.replace(scene)
    stage=Usd.Stage.Open(generation.usd_path)
    counts={}
    for prim in stage.Traverse():counts[str(prim.GetTypeName())]=counts.get(str(prim.GetTypeName()),0)+1
    mats=tuple(m for m in bpy.data.materials if m.users and m.use_nodes)
    conversion=scene_layer_from_materials(mats,str(generation.usd_path))
    camera=generation.blender_prim_paths[blender_id(scene.camera,'OBJECT')].schema_path
    result={'stage':str(generation.usd_path),'prim_types':counts,'generation_digest':generation.digest,
            'material_conversion_status':str(conversion.status.value),
            'material_conversion_error':conversion.error_reason,'material_diagnostics':conversion.diagnostics,
            'camera_path':camera,'native_status':'not_run'}
    source=str(generation.usd_path)
    material_layer=conversion.value
    if case.family=='openpbr_direct':
        from openpbr import author_overlay
        source=str(out/step/'direct-openpbr.usda')
        subjects=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'Subject' in str(p.GetPath())]
        if not subjects:raise AssertionError('No exported subject mesh for direct OpenPBR binding')
        result['direct_openpbr_parameters']=author_overlay(generation.usd_path,source,subjects,case.variant,
            any(o.get('openpbr_edit',0) for o in scene.objects))
        result['material_route']='direct_usd_not_principled_conversion';material_layer=None
    if conversion.error_reason is None:
        comp=compose(source_scene_path=source,camera_prim_path=camera,
            sensor_paths=('/Render/Zoo',),width=scene.render.resolution_x,height=scene.render.resolution_y,
            camera_projection=(camera_projection_from_scene_camera(
                scene,None,scene.render.resolution_x,scene.render.resolution_y)
                if case.family=='camera' else None),material_scene_layer=material_layer,
            light_scene_layer=scene_layer_from_lights(tuple(o for o in scene.objects if o.type=='LIGHT'),str(generation.usd_path)),
            generate_scene_presentation=True)
        result['composed_scene']=comp.composed_scene_path;result['sensor_paths']=['/Render/Zoo']
        composed=Usd.Stage.Open(comp.composed_scene_path)
        result['render_product_modes']={path:composed.GetPrimAtPath(path).GetAttribute('omni:rtx:rendermode').Get()
            for path in result['sensor_paths']}
        result['stage_metadata']={key:composed.GetMetadata(key) for key in
            ('upAxis','metersPerUnit','timeCodesPerSecond','framesPerSecond','startTimeCode','endTimeCode','defaultPrim','renderSettingsPrimPath')
            if composed.HasAuthoredMetadata(key)}
    return result


def main():
    options=args();case=BY_ID[options.case];out=options.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR']=str(out/'composition')
    sys.path.insert(0,str(options.addon_source.resolve()/'addon'))
    from ovrtx_blender_example import authoring_properties,properties
    authoring_properties.register();properties.register()
    random.seed(3317)
    report={'schema':1,'case':case.id,'case_spec':asdict(case),'blender':bpy.app.version_string,
            'addon_source':str(options.addon_source.resolve()),'native_status':'not_run','steps':[],
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        obj=build(case,out)
        bpy.context.scene['zoo_case']=case.id
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'),compress=True)
        for index,operation in enumerate(('baseline',)+ (case.edits if options.edits else ())):
            if operation!='baseline':obj=edit(operation,obj,case)
            state=snapshot();exported=export(bpy.context.scene,out,f'{index:02d}-{operation}',case)
            digest=hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()
            report['steps'].append({'operation':operation,'blender':state,'usd':exported,'state_digest':digest,
                'blender_state_changed':None if not report['steps'] else digest!=report['steps'][-1]['state_digest']})
        if options.preview:
            scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH'
            scene.display.shading.light='STUDIO';scene.display.shading.color_type='MATERIAL'
            scene.render.filepath=str(out/'blender-authoring-preview.png');bpy.ops.render.render(write_still=True)
            report['preview_kind']='Blender Workbench authoring preview; not OVRTX evidence'
        report['status']='conversion_rejected' if any(s['usd']['material_conversion_error'] for s in report['steps']) else 'export_passed'
    except NotImplementedError as error:
        report.update(status='blocked',error=str(error),error_class=type(error).__name__)
    except Exception as error:
        report.update(status='failed',error=str(error),error_class=type(error).__name__,traceback=traceback.format_exc())
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('ZOO_RESULT '+json.dumps({'case':case.id,'status':report['status']}),flush=True)


if __name__=='__main__':main()
