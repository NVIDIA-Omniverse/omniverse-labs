# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expandable, full-size room bays; widening never moves the front wall."""
import math
from .extensions import value, call
from .campus_assets import CAMPUS_ENABLED, CAMPUS_HIDDEN_ROOMS, CAMPUS_TEN_GALLERIES, CAMPUS_REFINED, CAMPUS_VISITOR_POLISH

# A single visitor lens keeps walking continuous across room boundaries.
WALK_FOCAL_LENGTH = 17.

BAY_WIDTH = 11.
FRONT_Z, REAR_Z = -10., 12.
FRONT_BAYS = value('FRONT_BAYS', ('rendering', 'scene', 'newton', 'package'))
REAR_BAYS = value('REAR_BAYS', ('streaming', None, None, None))
SIDE_BAYS = value('SIDE_BAYS', {'left': ('physics', 'interface'), 'right': ('storage',)})
COLUMNS = max(len(FRONT_BAYS), len(REAR_BAYS))
ATRIUM_WIDTH = COLUMNS*BAY_WIDTH
HALF_WIDTH = ATRIUM_WIDTH/2
ATRIUM_DEPTH = REAR_Z-FRONT_Z
ATRIUM_CENTER_Z = (FRONT_Z+REAR_Z)/2
LEGACY_ORIGINS = {'physics': -11., 'scene': 11.}


def bay_x(index):
    return (index-(COLUMNS-1)/2)*BAY_WIDTH


# Door centre x/z, walking direction into the room, exhibit-local x origin.
PLACEMENTS = {}
EMPTY_BAYS = []
for bays, z, yaw in ((FRONT_BAYS, FRONT_Z, 0.), (REAR_BAYS, REAR_Z, math.pi)):
    for i in range(COLUMNS):
        key = bays[i] if i < len(bays) else None
        if key:
            PLACEMENTS[key] = (bay_x(i), z, yaw, LEGACY_ORIGINS.get(key, 0.))
        else:
            EMPTY_BAYS.append((bay_x(i), z))
for side, bays in SIDE_BAYS.items():
    sign = -1 if side == 'left' else 1
    for i, key in enumerate(bays):
        PLACEMENTS[key] = (sign*HALF_WIDTH, FRONT_Z+BAY_WIDTH*(i+.5), sign*math.pi/2,
                           LEGACY_ORIGINS.get(key, 0.))
LOBBY_POSE = (0., 10.4, 0., .09)

# Campus gallery anchors. Optional exhibits contribute their own placements.
GALLERIES = {
    'rendering': (-9., -18., 0.), 'scene': (9., -18., 0.),
    'physics': (-18., -9., -math.pi/2), 'interface': (-18., 9., -math.pi/2),
    'storage': (18., 9., math.pi/2),
    'streaming': (-9., 18., math.pi), 'newton': (9., 18., math.pi),
}
GALLERIES.update(value('GALLERIES', {}))
COMPANIONS = {'storage': 'package', **value('COMPANIONS', {})}
RESERVED_GALLERIES = {}
if CAMPUS_TEN_GALLERIES:
    GALLERIES.update(rendering=(-18.,-18.,0.), scene=(18.,-18.,0.),
                     streaming=(-18.,18.,math.pi),newton=(18.,18.,math.pi),
                     package=(0.,18.,math.pi))
    GALLERIES.update(value('TEN_GALLERIES', {}))
    if CAMPUS_VISITOR_POLISH:
        # The flagship renderer faces the visitor across the atrium. Move
        # complete live exhibits, preserving their local scale and cameras.
        GALLERIES['rendering'] = (0., -18., 0.)
        GALLERIES.update(value('FLAGSHIP_GALLERIES', {}))
    COMPANIONS = {}
    # The same building can be used by a source edition with fewer integrations.
    # These are architectural reservations, not mock demonstrations.
    RESERVED_GALLERIES = {key:anchor for key,anchor in {
        'future_north':(-18. if CAMPUS_VISITOR_POLISH else 0.,-18.,0.),
        'future_east':(18.,-9.,math.pi/2)
    }.items() if anchor not in GALLERIES.values()}
if CAMPUS_ENABLED:
    exhibit_buffer = .4 if CAMPUS_REFINED else 2.
    PLACEMENTS = {
        key: (x+exhibit_buffer*math.sin(yaw), z-exhibit_buffer*math.cos(yaw), yaw, LEGACY_ORIGINS.get(key, 0.))
        for key, (x, z, yaw) in GALLERIES.items()
    }
    if CAMPUS_TEN_GALLERIES:
        PLACEMENTS.update(value('TEN_CAMPUS_PLACEMENTS', {}))
        if CAMPUS_REFINED:
            # Keep comfortable standing space beside the return-wall arcade.
            nx,nz,ny,no = PLACEMENTS['newton']
            PLACEMENTS['newton'] = (nx-.4,nz,ny,no)
    else:
        PLACEMENTS.update(package=(26., 9., math.pi/2, 0.))
        PLACEMENTS.update(value('CAMPUS_PLACEMENTS', {}))
    LOBBY_POSE = (-3.5, 7.8, -.10, .025)
    if CAMPUS_TEN_GALLERIES:
        # Look toward the new left portal at the same initial distance from
        # the facing atrium wall; the guide remains beside the sculpture.
        LOBBY_POSE = (-3.5, 7.8, -.205, .025)
    if CAMPUS_VISITOR_POLISH:
        LOBBY_POSE = (0., 7.8, 0., .025)


def world(key, u, v):
    x, z, yaw, _ = PLACEMENTS[key]
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * u - s * v, z + s * u + c * v


def local(key, x, z):
    cx, cz, yaw, _ = PLACEMENTS[key]
    c, s = math.cos(yaw), math.sin(yaw)
    return c * (x-cx) + s * (z-cz), -s * (x-cx) + c * (z-cz)


def pose(key, distance=None):
    if CAMPUS_ENABLED and key == 'newton' and distance is None:
        x,z = world(key,0.,-.2)
        return x,z,PLACEMENTS[key][2],-.015
    custom = call('pose', key, distance, world)
    if custom is not None:
        return custom
    x, z = world(key, 0., -1.65 if distance is None else distance)
    return x, z, PLACEMENTS[key][2], .015


def contains(x, z, radius=0.):
    if CAMPUS_ENABLED:
        # Detailed wall and furniture clearance comes from the supplied mesh
        # silhouettes. This envelope keeps visitors on the campus ground floor.
        if -18+radius < x < 18-radius and -18+radius < z < 18-radius:
            return True
        if CAMPUS_TEN_GALLERIES:
            # Fanned links extend laterally past the old square envelope.
            # Mesh-derived walls still enforce the actual corridor boundaries.
            if -26 < x < 26 and -18 < z < 18:
                return True
        for key, (cx, cz, yaw) in {**GALLERIES,**RESERVED_GALLERIES}.items():
            c,s = math.cos(yaw),math.sin(yaw)
            u,v = c*(x-cx)+s*(z-cz),-s*(x-cx)+c*(z-cz)
            depth = 22. if key == 'scene' else 20. if key in ('physics','rendering','streaming',*value('INDUSTRIAL_ROOMS', ())) else 18.
            if CAMPUS_TEN_GALLERIES:
                depth = 15.2
            if -7.75+radius < u < 7.75-radius and -depth+radius < v < 1.:
                return True
        return False
    if -HALF_WIDTH + radius < x < HALF_WIDTH - radius and FRONT_Z + radius < z < REAR_Z - radius:
        return True
    # Extend each room's navigable area through the doorway into the atrium.
    return any(-5.5 + radius < (p := local(key, x, z))[0] < 5.5 - radius
               and -11.2 + radius < p[1] < 1.0 for key in PLACEMENTS)


def room_at(x, z):
    if CAMPUS_ENABLED:
        # A small interaction zone, not a second gallery or a deep-room split.
        custom = call('room_at', x, z, local)
        if custom:
            return custom
        for key,(cx,cz,yaw) in GALLERIES.items():
            c,s = math.cos(yaw),math.sin(yaw)
            u,v = c*(x-cx)+s*(z-cz),-s*(x-cx)+c*(z-cz)
            if -7.9 < u < 7.9 and -22. < v < -.2:
                boundary = {'physics': -11., 'scene': -11.5, 'storage': -9.}.get(key)
                companion = COMPANIONS.get(key)
                return companion if companion and boundary is not None and v < boundary and companion not in CAMPUS_HIDDEN_ROOMS else key
        return 'lobby'
    for key in PLACEMENTS:
        u, v = local(key, x, z)
        if -5.5 < u < 5.5 and -11.2 < v < -.2:
            return key
    return "lobby"
