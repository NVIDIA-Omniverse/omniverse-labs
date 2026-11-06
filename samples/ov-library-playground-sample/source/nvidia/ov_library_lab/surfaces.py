# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PBR surface maps and small manufactured edge radii, in metre units."""
import math
from pxr import Gf, Sdf, UsdGeom, UsdShade


def textured(g, name, asset, roughness=1., tint=(1.,1.,1.), bias=(0.,0.,0.), normal_strength=1.):
    material=g.materials[name]
    surface=UsdShade.Shader(g.stage.GetPrimAtPath(material.GetPath().AppendChild('Surface')))
    reader=UsdShade.Shader.Define(g.stage,material.GetPath().AppendChild('UV'))
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    for source,target,kind in (('Diffuse','diffuseColor','rgb'),('Rough','roughness','r'),('nor_gl','normal','rgb')):
        tex=UsdShade.Shader.Define(g.stage,material.GetPath().AppendChild(source))
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(g.data_dir/'surfaces'/f'{asset}_{source}.jpg'))
        tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if source=='Diffuse' else 'raw')
        tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        scale=(*tint,1.) if source=='Diffuse' else (roughness,roughness,roughness,1.)
        if source=='Diffuse':
            tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias,0.))
        if source=='nor_gl':
            scale=(2*normal_strength,2*normal_strength,2.,1.)
            tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-normal_strength,-normal_strength,-1.,0.))
        tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        value_type=Sdf.ValueTypeNames.Float if kind=='r' else Sdf.ValueTypeNames.Normal3f if target=='normal' else Sdf.ValueTypeNames.Color3f
        surface.CreateInput(target,value_type).ConnectToSource(tex.ConnectableAPI(),kind)


def box_mesh(stage,path,size,radius,uv_metres=1.):
    """Closed rounded box with smooth normals and dimensioned face UVs.

    Points remain in a unit box so existing transform/clearance controls retain
    their meaning. Each edge lies inside the exact original equipment envelope.
    """
    half=[s/2 for s in size]
    radius=min(radius,min(size)*.18)
    points,normals,uvs,counts,indices=[],[],[],[],[]
    for axis in range(3):
        u,v=(axis+1)%3,(axis+2)%3
        def steps(h):
            if radius<=0:
                return [-h,h]
            return [-h,-h+radius*.35,-h+radius,h-radius,h-radius*.35,h]
        us,vs=steps(half[u]),steps(half[v])
        for sign in (-1,1):
            start=len(points)
            for y in vs:
                for x in us:
                    p=[0.,0.,0.]
                    p[axis],p[u],p[v]=sign*half[axis],x,y
                    core=[max(-half[i]+radius,min(half[i]-radius,p[i])) for i in range(3)]
                    delta=[p[i]-core[i] for i in range(3)]
                    length=math.sqrt(sum(d*d for d in delta))
                    normal=[d/length for d in delta] if radius else [sign if i==axis else 0 for i in range(3)]
                    rounded=[core[i]+normal[i]*radius for i in range(3)] if radius else p
                    points.append(Gf.Vec3f(*(rounded[i]/size[i] for i in range(3))))
                    # Inverse transpose of the scale restores this physical normal.
                    normals.append(Gf.Vec3f(*(normal[i]*size[i] for i in range(3))).GetNormalized())
                    uvs.append((x/uv_metres,y/uv_metres))
            for j in range(len(vs)-1):
                for i in range(len(us)-1):
                    a=start+j*len(us)+i
                    face=[a,a+1,a+1+len(us),a+len(us)]
                    indices.extend(face if sign>0 else reversed(face))
                    counts.append(4)
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(-.5),Gf.Vec3f(.5)])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(uvs)
    return mesh


def install(g):
    textured(g,'floor','concrete_floor_02',.55,tint=(.18,.19,.20),bias=(.24,.255,.27),normal_strength=.22)
    textured(g,'wood','plywood',1.)
    textured(g,'carton','cardboard',1.,tint=(.70,.64,.55),normal_strength=.18)


def cylinder_mesh(stage,path,radius,height,axis='Y',segments=128):
    """Explicit smooth sides avoid renderer-dependent primitive faceting."""
    points,normals,indices,counts=[],[],[],[]
    def orient(x,y,z):
        return (z,x,y) if axis=='X' else (y,z,x) if axis=='Y' else (x,y,z)
    for z in (-height/2,height/2):
        for i in range(segments):
            a=i*math.tau/segments
            points.append(Gf.Vec3f(*orient(radius*math.cos(a),radius*math.sin(a),z)))
            normals.append(Gf.Vec3f(*orient(math.cos(a),math.sin(a),0)))
    for i in range(segments):
        j=(i+1)%segments
        indices.extend((i,j,j+segments,i+segments)); counts.append(4)
    for sign in (-1,1):
        start=len(points)
        for i in range(segments):
            a=i*math.tau/segments
            points.append(Gf.Vec3f(*orient(radius*math.cos(a),radius*math.sin(a),sign*height/2)))
            normals.append(Gf.Vec3f(*orient(0,0,sign)))
        face=list(range(start,start+segments))
        indices.extend(face if sign>0 else reversed(face)); counts.append(segments)
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(points)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation('vertex')
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    return mesh
