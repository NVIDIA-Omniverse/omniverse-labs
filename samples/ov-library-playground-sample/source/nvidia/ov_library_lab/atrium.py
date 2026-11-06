# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A walkable NVIDIA sculpture court built from the official vector outlines."""
import json
import math
from pxr import Gf, UsdGeom


def ring(g, path, radius, y, mat, width=.035, center_z=1.):
    curve = UsdGeom.BasisCurves.Define(g.stage,path)
    curve.CreateTypeAttr('linear')
    curve.CreateWrapAttr('periodic')
    curve.CreateCurveVertexCountsAttr([128])
    curve.CreatePointsAttr([Gf.Vec3f(radius*math.cos(i*math.tau/128),y,center_z+radius*math.sin(i*math.tau/128)) for i in range(128)])
    curve.CreateWidthsAttr([width])
    curve.SetWidthsInterpolation('constant')
    g.bind(curve.GetPrim(),mat)
    return curve


def mesh(g, path, points, faces, material):
    obj=UsdGeom.Mesh.Define(g.stage,path)
    obj.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    obj.CreateFaceVertexCountsAttr([len(face) for face in faces])
    obj.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    obj.CreateSubdivisionSchemeAttr('none')
    obj.CreateDoubleSidedAttr(True)
    g.bind(obj.GetPrim(),material)
    return obj


def extrude(g, name, data, width, y, depth, face_material):
    prefix='/World/Atrium/Logo/'+name
    for n,poly in enumerate(data['polygons']):
        p=[(x*width,h*width+y) for x,h in poly['points']]
        count=len(p)
        mesh(g,f'{prefix}/Front{n}',[(x,h,depth/2) for x,h in p],poly['triangles'],face_material)
        mesh(g,f'{prefix}/Back{n}',[(x,h,-depth/2) for x,h in p],[list(reversed(f)) for f in poly['triangles']],face_material)
        points=[(x,h,z) for z in (depth/2,-depth/2) for x,h in p]
        faces=[]
        start=0
        for end in poly['ring_ends']:
            for i in range(start,end):
                j=start if i==end-1 else i+1
                faces.append([i,i+count,j+count,j])
            start=end
        mesh(g,f'{prefix}/Edges{n}',points,faces,'brand_edge' if name=='Eye' else 'silver')


def build(g):
    g.materials['brand_green']=g.material('brand_green',(.181,.485,.001),.32,.22,.10)
    g.materials['brand_edge']=g.material('brand_edge',(.07,.20,.004),.65,.26)
    g.materials['brand_light']=g.material('brand_light',(.31,.73,.015),.1,.3,1.7)
    g.materials['sculpture_base']=g.material('sculpture_base',(.022,.029,.035),.78,.28)
    UsdGeom.Xform.Define(g.stage,'/World/Atrium')
    g.cylinder('/World/Atrium/Plinth',(0,.16,1),2.,.32,'sculpture_base')
    g.cylinder('/World/Atrium/Top',(0,.33,1),1.94,.04,'black')
    g.obstacles.append((-2.,2.,-1.,3.))
    ring(g,'/World/Atrium/PlinthRim',1.965,.345,'brand_light',.035)
    ring(g,'/World/Atrium/FloorHalo',2.38,.024,'brand_light',.027)
    ring(g,'/World/Atrium/FloorOuter',2.52,.017,'silver',.012)
    ring(g,'/World/Atrium/CeilingHalo',2.65,6.9,'brand_light',.07)
    ring(g,'/World/Atrium/CeilingHousing',2.7,6.94,'black',.13)
    data=json.loads((g.data_dir/'brand/logo-contours.json').read_text())
    with g.factory.assembly('/World/Atrium/Logo',(0,.38,1),yaw=-12.):
        extrude(g,'Eye',data['eye'],3.2,.90,.34,'brand_green')
        extrude(g,'Wordmark',data['wordmark'],2.78,.12,.18,'ivory')
        for i,x in enumerate((-.64,.64)):
            g.box(f'/World/Atrium/Logo/Mount{i}',(x,.47,-.21),(.075,.94,.075),'sculpture_base')
    g.rect_light('SculptureKey',(-3.,5.4,5.),(0,1.8,1),(1.,.98,.92),3200.,3.,2.)
    g.rect_light('SculptureRim',(2.8,4.8,-1.8),(0,1.8,1),(.72,.91,1.),2200.,2.,3.)
