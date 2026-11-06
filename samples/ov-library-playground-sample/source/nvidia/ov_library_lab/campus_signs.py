# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architectural wayfinding: raised letters and metal plaques at the atrium."""
import json
import math
from pxr import Gf, Sdf, UsdGeom

# The R45 gallery passages bend. These are their *atrium-side* portals,
# measured from the artist's timber reveals, not the exhibit coordinates.
# Portal center for approach cameras, plaque center, surface rotation in degrees.
ENTRANCES = {
    'rendering': ((-5.75,4.80,-9.69),(-1.22,1.75,-9.72),0.),
    'scene': ((5.75,4.80,-9.69),(1.22,1.75,-9.72),0.),
    'physics': ((-10.69,4.80,-4.96),(-10.72,1.75,-.18),90.),
    'interface': ((-10.69,4.80,6.67),(-10.72,1.75,2.18),90.),
    'storage': ((10.69,4.80,6.67),(10.72,1.75,2.18),-90.),
    'streaming': ((-5.90,4.80,11.69),(-1.22,1.75,11.72),180.),
    'newton': ((5.90,4.80,11.69),(1.22,1.75,11.72),180.),
}
from .extensions import value
from .campus_assets import CAMPUS_TEN_GALLERIES, CAMPUS_REFINED, CAMPUS_VISITOR_POLISH
ENTRANCES.update(value('ENTRANCES', {}))
if CAMPUS_TEN_GALLERIES:
    ENTRANCES.update({
        'rendering':((-7.2,4.80,-9.69),(-3.6,1.75,-9.45),0.),
        'scene':((7.2,4.80,-9.69),(10.48,1.75,-9.32),0.),
        'streaming':((-7.2,4.80,11.69),(-10.48,1.75,11.32),180.),
        'package':((0.,4.80,11.69),(-3.6,1.75,11.45),180.),
        'newton':((7.2,4.80,11.69),(3.6,1.75,11.45),180.),
    })
    ENTRANCES.update(value('TEN_ENTRANCES', {}))
    if CAMPUS_VISITOR_POLISH:
        ENTRANCES['rendering'] = ((0.,4.80,-9.69),(3.6,1.75,-9.45),0.)
        ENTRANCES.update(value('FLAGSHIP_ENTRANCES', {}))
    if CAMPUS_REFINED:
        # Use the perpendicular inside face of the corner pier, clear of the
        # return wall. A complete plaque fits this host without intersecting it.
        ENTRANCES['scene'] = (ENTRANCES['scene'][0], (10.69,1.75,-8.55), -90.)
        ENTRANCES['streaming'] = (ENTRANCES['streaming'][0], (-10.69,1.75,10.55), 90.)
    from .layout import RESERVED_GALLERIES
    reserved_entrances={
        'future_north': (((-7.2,4.80,-9.69),(-3.6,1.75,-9.45),0.) if CAMPUS_VISITOR_POLISH
                         else ((0.,4.80,-9.69),(3.6,1.75,-9.45),0.)),
        'future_east':((10.69,4.80,-4.96),(10.72,1.75,-.18),-90.),
    }
    ENTRANCES.update({key:reserved_entrances[key] for key in RESERVED_GALLERIES})


def letters(g,path,data,height,max_width,y=0.,z=0.,depth=.008,x_offset=0.):
    """Extruded, non-emissive glyphs; preserve type proportions and counters."""
    x0,y0,x1,y1=data['bounds']
    scale=min(height/(y1-y0),max_width/(x1-x0))
    points=[]; faces=[]
    for polygon in data['polygons']:
        outline=[((x-(x0+x1)/2)*scale,(h-(y0+y1)/2)*scale+y) for x,h in polygon['points']]
        base=len(points); n=len(outline)
        points.extend((x+x_offset,h,z+d) for d in (0.,depth) for x,h in outline)
        for a,b,c in polygon['triangles']:
            # Earcut winding is normalized explicitly for outward surfaces.
            pa,pb,pc=outline[a],outline[b],outline[c]
            if (pb[0]-pa[0])*(pc[1]-pa[1])-(pb[1]-pa[1])*(pc[0]-pa[0])<0:b,c=c,b
            faces.extend(([base+a+n,base+b+n,base+c+n],[base+c,base+b,base+a]))
        start=0
        for end in polygon['ends']:
            for i in range(start,end):
                j=start if i==end-1 else i+1
                faces.append([base+i,base+j,base+j+n,base+i+n])
            start=end
    mesh=UsdGeom.Mesh.Define(g.stage,path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    g.bind(mesh.GetPrim(),'sign_ink')


def entrance_direction(key):
    """Project the actual portal onto its plaque's horizontal reading axis."""
    portal,plaque,yaw=ENTRANCES[key]
    angle=math.radians(yaw)
    horizontal=(portal[0]-plaque[0])*math.cos(angle)-(portal[2]-plaque[2])*math.sin(angle)
    return 1 if horizontal>0 else -1


def direction_arrow(direction):
    # A single extruded pictogram, using the same metal as the raised lettering.
    points=[(-.15,0),(-.02,.12),(-.02,.047),(.15,.047),(.15,-.047),(-.02,-.047),(-.02,-.12)]
    return {'bounds':[-.15,-.12,.15,.12], 'polygons':[{
        'points':[(x*(-direction),y)for x,y in points], 'ends':[7],
        'triangles':[(0,1,2),(0,2,5),(0,5,6),(2,3,4),(2,4,5)]}]}


def build(g):
    outlines=json.loads((g.data_dir/'wayfinding/lettering.json').read_text())
    for name,color,metal,roughness in (
        ('sign_ink',(.018,.021,.022),.65,.36),
        ('sign_metal',(.43,.38,.29),.9,.42),
        ('sign_face',(.70,.68,.61),.15,.48),
        ('sign_green',(.18,.42,.014),.15,.40),
    ):
        g.materials[name]=g.material(name,color,metal,roughness)
    g.active_room=None
    for key,(_,plaque,yaw) in ENTRANCES.items():
        text_key='future' if key.startswith('future_') else key
        root='/World/CampusWayfinding/'+key
        with g.factory.assembly(root+'/Plaque',plaque,yaw=yaw) as assembly:
            if CAMPUS_TEN_GALLERIES and (yaw in (0.,180.) or (CAMPUS_REFINED and key in ('scene','streaming'))):
                # Fit the actual structural pier; scale the complete sign so
                # lettering proportions and all mounting details stay intact.
                scale=.60 if CAMPUS_REFINED and key in ('scene','streaming') else .48 if abs(plaque[0])>10 else .72
                assembly.AddScaleOp().Set(Gf.Vec3f(scale))
            # Offset fixings, beveled metal perimeter and a satin painted face.
            fixing=.25 if CAMPUS_TEN_GALLERIES and yaw in (0.,180.) else .53
            for i,x in enumerate((-fixing,fixing)):
                g.cylinder(root+f'/Plaque/Standoff{i}',(x,0,.009),.035,.018,'sign_metal',axis='Z')
            g.box(root+'/Plaque/Body',(0,0,.034),(1.72,.70,.034),'sign_metal')
            g.box(root+'/Plaque/Face',(0,0,.053),(1.685,.665,.008),'sign_face')
            if CAMPUS_VISITOR_POLISH:
                direction=entrance_direction(key)
                # A separate, smaller footer keeps both text rows centered on
                # the plaque, independent of which way its doorway lies.
                letters(g,root+'/Plaque/Arrow',direction_arrow(direction),.11,.18,
                        y=-.205,z=.058,depth=.006)
                assembly.GetPrim().CreateAttribute('userProperties:entranceDirection',Sdf.ValueTypeNames.Token).Set('right'if direction>0 else 'left')
            letters(g,root+'/Plaque/Name',outlines[text_key],.16,1.43,
                    y=.14 if CAMPUS_VISITOR_POLISH else .09,z=.058,depth=.006)
            # Empty release galleries retain the same physical wayfinding,
            # without advertising or loading an unavailable demonstration.
            coming_soon=key.startswith('future_') or (CAMPUS_VISITOR_POLISH and key == 'graph')
            caption='coming_soon'if coming_soon and text_key!='future' else text_key+'_caption'
            letters(g,root+'/Plaque/Caption',outlines[caption],.073,1.40,
                    y=-.05 if CAMPUS_VISITOR_POLISH else -.13,z=.058,depth=.0015)
            if coming_soon:
                assembly.GetPrim().CreateAttribute('userProperties:availability',Sdf.ValueTypeNames.Token).Set('coming-soon')
            inlay_y=-.291 if CAMPUS_VISITOR_POLISH else -.266
            g.box(root+'/Plaque/Inlay',(0,inlay_y,.059),(1.40,.013,.003),'sign_green')


def physics_label(g,side,pos):
    """Small satin-metal comparison plate, bolted to the conveyor frame."""
    outlines=json.loads((g.data_dir/'wayfinding/lettering.json').read_text())
    key='physics_static' if side=='Frozen' else 'physics_live'
    root='/World/PhysicsLabels/'+side
    with g.factory.assembly(root,pos):
        # The upper brackets meet the front frame at y=.83; the plate sits
        # below the belt and in front of its cable tray, clear of the motor.
        for i,x in enumerate((-.59,.59)):
            g.box(root+f'/Bracket{i}',(x,.165,-.155),(.05,.09,.25),'sign_metal')
            g.box(root+f'/Tab{i}',(x,.04,-.022),(.05,.25,.025),'sign_metal')
        g.box(root+'/Body',(0,0,0),(1.70,.46,.028),'sign_metal')
        g.box(root+'/Face',(0,0,.017),(1.666,.426,.008),'sign_face')
        letters(g,root+'/Name',outlines[key],.123,1.45,y=.073,z=.022,depth=.003)
        letters(g,root+'/Caption',outlines[key+'_caption'],.061,1.43,y=-.092,z=.022,depth=.0015)
        g.box(root+'/Inlay',(0,-.177,.024),(1.41,.009,.003),'sign_green')
        for i,x in enumerate((-.78,.78)):
            for j,y in enumerate((-.16,.16)):
                g.cylinder(root+f'/Fixing{i}{j}',(x,y,.025),.012,.008,'sign_metal',axis='Z')
