# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose the artist's campus with the live library exhibits, without editing it."""
import json
import math
from collections import defaultdict
from pxr import Sdf, UsdGeom

from .campus_assets import CAMPUS_SCENE, CAMPUS_CACHE, CAMPUS_HIDDEN_ROOMS
from .content import ROOMS
from .layout import GALLERIES, COMPANIONS, PLACEMENTS


def expose(prim):
    """Physical camera exposure for the campus's authored daylight/luminaires."""
    prim.AddAppliedSchema('OmniRtxCameraExposureAPI_1')
    for name,value in [('exposure:time', .02), ('exposure:responsivity', 1.1026709),
                       ('exposure:fStop', 5.), ('exposure:iso', 100.)]:
        prim.CreateAttribute(name,Sdf.ValueTypeNames.Float).Set(value)


class Walls:
    """Circle/segment clearance, spatially indexed for constant-time walking."""
    def __init__(self, segments):
        self.cells = defaultdict(list)
        for segment in segments:
            ax,az,bx,bz = segment
            for x in range(math.floor(min(ax,bx)/2), math.floor(max(ax,bx)/2)+1):
                for z in range(math.floor(min(az,bz)/2), math.floor(max(az,bz)/2)+1):
                    self.cells[x,z].append(segment)

    def __call__(self,x,z,r):
        for cx in range(math.floor((x-r)/2),math.floor((x+r)/2)+1):
            for cz in range(math.floor((z-r)/2),math.floor((z+r)/2)+1):
                for ax,az,bx,bz in self.cells.get((cx,cz),()):
                    dx,dz = bx-ax,bz-az
                    t = max(0.,min(1.,((x-ax)*dx+(z-az)*dz)/(dx*dx+dz*dz)))
                    if (x-ax-t*dx)**2+(z-az-t*dz)**2 < r*r:
                        return True
        return False


def build(g):
    cache = CAMPUS_CACHE/'navigation.json'
    if not cache.is_file():
        raise RuntimeError('Campus walking data is missing. Run tools/prepare_campus.py once.')
    g.campus_blocker = Walls(json.loads(cache.read_text())['segments'])
    # Only the environment is referenced: the eight saved demonstrations in
    # campus-lab.usdc are snapshots and must never shadow our live library data.
    root = UsdGeom.Xform.Define(g.stage,'/World/Environment')
    optimized = CAMPUS_CACHE/'environment.usda'
    source,path = (optimized,'/Environment') if optimized.is_file() else (CAMPUS_SCENE,'/World/Environment')
    root.GetPrim().GetReferences().AddReference(str(source.resolve()),path)
    # The artist environment includes static information boards. Keep the
    # original artwork intact, but hide boards for unavailable demonstrations
    # in this composed stage. Room walls, frames and landscaping stay in place.
    absent_boards = tuple('_r44_board_'+key+'_' for key in ('clash','graph','spatial') if key not in ROOMS)
    for prim in root.GetPrim().GetChildren():
        if any(marker in prim.GetName() for marker in absent_boards) and prim.IsA(UsdGeom.Xform):
            UsdGeom.Imageable(prim).MakeInvisible()
    from .campus_assets import CAMPUS_REVISION
    if CAMPUS_REVISION == 'r48-wayfinding-landscape':
        for name in ('r51-entrance-floor-infills.usda', 'r50-ovui-pendant-clearance.usda',
                     'r49-gallery-light-mounts.usda'):
            correction = g.data_dir/'scene-overrides'/name
            if correction.is_file():
                g.stage.GetRootLayer().subLayerPaths.append(str(correction.resolve()))
    for name,color,metal,rough,emission in (
        ('panel',(.40,.44,.46),0.,.62,0.), ('acoustic',(.055,.068,.077),0.,.85,0.),
        ('luminaire',(1.,.96,.86),0.,.35,3.), ('aluminum',(.61,.63,.65),1.,.32,0.),
        ('stone',(.15,.17,.18),0.,.35,0.)):
        g.materials[name]=g.material(name,color,metal,rough,emission)
    for key,room in ROOMS.items():
        if key in CAMPUS_HIDDEN_ROOMS:
            continue
        g.active_room = key
        x = room['center']
        floor = g.box('/World/Building/'+key+'/Floor',(x,-.2,-4.4),(15.5,.4,11.2),'floor',collider=True)
        floor.CreateVisibilityAttr().Set('invisible')
        if key in COMPANIONS and COMPANIONS[key] not in CAMPUS_HIDDEN_ROOMS:
            companion=COMPANIONS[key]
            g.box('/World/CampusLabels/'+companion,(x,3.25,-8.30),(4.5,.80,.08),'acoustic')
            g.sign(companion+'_entry',(x,3.25,-8.245),4.3,.62,companion+'_entry.png')
        # Explanations live in the experiment UI. Avoid presentation boards
        # floating across the artist's architecture and daylight sightlines.
    g.active_room = None
    from .campus_signs import build as build_signs
    build_signs(g)
