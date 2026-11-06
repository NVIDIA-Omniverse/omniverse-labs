# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architectural millwork and practical luminaires for the industrial studio."""
from .content import ROOMS
from .layout import HALF_WIDTH, ATRIUM_WIDTH, ATRIUM_DEPTH, COLUMNS, EMPTY_BAYS


def build(g):
    for name,color,metal,rough,emission in (
        ('panel',(.40,.44,.46),0.,.62,0.),
        ('acoustic',(.055,.068,.077),0.,.85,0.),
        ('luminaire',(1.,.96,.86),0.,.35,3.),
        ('aluminum',(.61,.63,.65),1.,.32,0.),
        ('stone',(.15,.17,.18),0.,.35,0.)):
        g.materials[name]=g.material(name,color,metal,rough,emission)
    for key,room in ROOMS.items():
        g.active_room=key
        x=room['center']
        prefix='/World/Architecture/'+key
        # Individual wall panels, shadow gaps and a durable lower kick rail.
        for i in range(7):
            g.box(prefix+f'/Panel{i}',(x+(i-3)*1.51,2.2,-9.93),(1.48,3.08,.06),'panel')
        g.box(prefix+'/Kick',(x,.32,-9.70),(10.7,.18,.09),'aluminum')
        g.box(prefix+'/SignBack',(x,3.95,-9.87),(7.35,1.78,.045),'acoustic')
        for side in (-1,1):
            suffix='L' if side<0 else 'R'
            for j in range(14):
                g.box(prefix+f'/Acoustic{suffix}_{j}',(x+side*5.33,3.18,-8.7+j*.48),(.08,2.05,.075),'acoustic')
            g.box(prefix+f'/LowerRail{suffix}',(x+side*5.30,.36,-4.4),(.065,.22,10.7),'aluminum')
            g.box(prefix+f'/DoorJamb{suffix}',(x+side*3.17,1.96,1.30),(.13,3.92,.30),'aluminum')
        # Three long softboxes create broad, readable reflections on equipment.
        for j,z in enumerate((-7.8,-4.7,-1.6)):
            g.box(prefix+f'/LightHousing{j}',(x,5.12,z),(8.4,.14,.52),'acoustic')
            g.box(prefix+f'/LightLens{j}',(x,5.04,z),(8.12,.016,.36),'luminaire')
        g.box(prefix+'/ThresholdPlate',(x,.008,1.25),(6.2,.012,.54),'aluminum')
    g.active_room=None
    # A coffered ceiling and continuous light slots give the atrium depth.
    ribs = COLUMNS*2+1
    for i in range(ribs):
        x = -HALF_WIDTH+2.5+i*(ATRIUM_WIDTH-5)/(ribs-1)
        g.box(f'/World/Architecture/LobbyRib{i}',(x,7.16,1),(.11,.36,ATRIUM_DEPTH-.3),'acoustic')
    for i,z in enumerate((-7.7,-3.35,1.,5.35,9.7)):
        g.box(f'/World/Architecture/LobbyBeam{i}',(0,7.15,z),(ATRIUM_WIDTH-.3,.38,.11),'acoustic')
        for side in (-1,1):
            suffix='L' if side<0 else 'R'
            g.box(f'/World/Architecture/LobbyLens{i}_{suffix}',(side*(HALF_WIDTH-4.2),7.28,z),(4.7,.03,1.5),'luminaire')
    # Seat against the wall piers between the two doors on each side.
    # The door openings end at z=-1.35 and begin again at z=3.35.
    for i,x in enumerate((-HALF_WIDTH+.6,HALF_WIDTH-.6)):
        g.box(f'/World/Architecture/Bench{i}',(x,.42,1.),(.72,.18,2.1),'wood',obstacle=True)
        for j,dz in enumerate((-.75,.75)):
            g.box(f'/World/Architecture/BenchLeg{i}_{j}',(x,.18,1.+dz),(.62,.36,.09),'black')
    # Unoccupied bays are finished wall piers, ready to become future rooms.
    for i,(x,z) in enumerate(EMPTY_BAYS):
        inset = -.14 if z > 0 else .14
        for j in range(7):
            g.box(f'/World/Architecture/Expansion{i}/Panel{j}',(x+(j-3)*1.51,2.5,z+inset),(1.48,4.8,.06),'panel')
