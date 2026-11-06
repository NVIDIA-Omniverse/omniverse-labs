# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Referenced, attributed automotive asset and its material controls."""
from pxr import Gf, Sdf, UsdGeom
import json


class Automotive:
    def __init__(self, g):
        self.g, self.stage = g, g.stage
        self.roughness, self.metallic, self.clearcoat = .22, .32, 1.
        self.color = 'Carmine'
        old = g.active_room
        g.active_room = 'rendering'
        # The procedural robot cell previously had an implicit, untyped parent.
        # A typed imageable parent makes visibility propagate to every child.
        UsdGeom.Xform.Define(g.stage, '/World/Rendering')
        self.root = UsdGeom.Xform.Define(g.stage, '/World/CarDisplay')
        g.box('/World/CarDisplay/Plinth', (0, .06, -5.4), (7.5, .12, 5.8), 'black', obstacle=True)
        g.factory.boundary('/World/CarDisplay/Boundary', 0, -5.4, 7.7, 6.)
        # Rotated wheel AABBs extend below the real tire geometry. Use actual
        # transformed vertex bounds so the tires rest on the platform.
        self.proportions = json.loads((g.data_dir/'models/car/proportions.json').read_text())
        contact_y = self.proportions['converted_usd_min_m'][1]
        with g.factory.assembly('/World/CarDisplay/Turntable', (0, .12-contact_y, -5.4), yaw=55.) as mount:
            self.rotation = mount.GetPrim().GetAttribute('xformOp:rotateY')
            mount.AddScaleOp().Set(Gf.Vec3f(1.))
            asset = self.stage.DefinePrim('/World/CarDisplay/Turntable/Car')
            asset.GetReferences().AddReference(str(g.data_dir/'models/car/car.usdc'))
        g.placard('CarLabel', (-3.15, 1.05, -6.4), 2., .70, 'car_label.png', floor_y=.12)
        self.fill = g.rect_light('CarFill', (3.1, 3.1, -4.), (0, .8, -5.4), (.78, .87, 1.), 1600., 3., 2.)
        self._refine_materials()
        self.paint(self.color)
        self.fill.MakeInvisible()
        self.set_roughness(self.roughness)
        self.set_metallic(self.metallic)
        self.set_clearcoat(self.clearcoat)
        self.root.MakeInvisible()
        if not g.campus:
            UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Signs/CarLabel')).MakeInvisible()
            UsdGeom.Xform.Define(self.stage,'/World/Placards/CarLabel').MakeInvisible()
        g.active_room = old

    def shader(self, material):
        return self.stage.GetPrimAtPath('/World/CarDisplay/Turntable/Car/Looks/'+material+'/'+material)

    def _refine_materials(self):
        # Preserve source texture/UV connections. The glTF MDL's normal lookup
        # exposes a strength input; paint microtexture should not read as dents.
        for name, strength in (('Paint_1_Carmine', .12), ('Paint_2_Carmine', .12),
                               ('Tireside', .45), ('Tiretread', .65)):
            normal = self.stage.GetPrimAtPath('/World/CarDisplay/Turntable/Car/Looks/'+name+'/normalTex')
            if normal:
                normal.CreateAttribute('inputs:normal_scale_factor', Sdf.ValueTypeNames.Float).Set(strength)
        for name in ('Paint_1_Carmine', 'Paint_2_Carmine'):
            self.shader(name).CreateAttribute('inputs:clearcoat_roughness_factor', Sdf.ValueTypeNames.Float).Set(.09)
        self.shader('Tireside').GetAttribute('inputs:roughness_factor').Set(.76)
        self.shader('Tiretread').GetAttribute('inputs:roughness_factor').Set(.84)
        self.shader('Glass').GetAttribute('inputs:roughness_factor').Set(.025)
        self.shader('Glass').CreateAttribute('inputs:ior', Sdf.ValueTypeNames.Float).Set(1.5)

    def finish(self, name):
        roughness, metal, coat = {'Gloss': (.22, .32, 1.), 'Satin': (.48, .18, .35),
                                  'Matte': (.78, .08, .0)}[name]
        self.set_roughness(roughness)
        self.set_metallic(metal)
        self.set_clearcoat(coat)

    def set_roughness(self, value):
        self.roughness = float(value)
        for name in ('Paint_1_Carmine', 'Paint_2_Carmine'):
            self.shader(name).GetAttribute('inputs:roughness_factor').Set(self.roughness)

    def set_metallic(self, value):
        self.metallic = float(value)
        for name in ('Paint_1_Carmine', 'Paint_2_Carmine'):
            self.shader(name).GetAttribute('inputs:metallic_factor').Set(self.metallic)

    def set_clearcoat(self, value):
        self.clearcoat = float(value)
        for name in ('Paint_1_Carmine', 'Paint_2_Carmine'):
            self.shader(name).GetAttribute('inputs:clearcoat_factor').Set(self.clearcoat)

    def paint(self, name):
        self.color = name
        color = {'Carmine':(.666,.012,.018), 'Ocean':(.018,.16,.42), 'Pearl':(.68,.71,.74)}[name]
        self.shader('Paint_1_Carmine').GetAttribute('inputs:base_color_factor').Set(Gf.Vec3f(*color))

    def set_visible(self, visible):
        (self.root.MakeVisible if visible else self.root.MakeInvisible)()
        (self.fill.MakeVisible if visible else self.fill.MakeInvisible)()
        for path in ('/World/Rendering', '/World/Signs/MaterialLabels'):
            obj = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
            if obj:
                (obj.MakeInvisible if visible else obj.MakeVisible)()
        label = UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Signs/CarLabel'))
        if label:
            (label.MakeVisible if visible else label.MakeInvisible)()
        stand=UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Placards/CarLabel'))
        if stand:
            (stand.MakeVisible if visible else stand.MakeInvisible)()
