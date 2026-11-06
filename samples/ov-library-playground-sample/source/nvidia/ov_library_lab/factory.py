# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable warehouse equipment and SimReady references in metre-based local frames."""
import math
from contextlib import contextmanager

from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

from .layout import PLACEMENTS
from .physics_schema import apply


class FactoryAssets:
    def __init__(self, gallery):
        self.g = gallery
        from .simready_assets import SimReadyAssets
        self.simready = SimReadyAssets(gallery)
        for name, color, metal, rough in [
            ("carton", (.52, .31, .13), 0., .88),
            ("tape", (.78, .59, .31), 0., .48),
            ("paper", (.88, .88, .81), 0., .9),
            ("wood", (.37, .22, .105), 0., .85),
            ("safety", (.78, .46, .018), .0, .4),
            ("steel", (.52, .55, .58), 1., .28),
            ("rack_blue", (.032, .105, .18), .0, .32),
            ("rack_orange", (.72, .13, .018), .0, .38),
            ("belt", (.035, .055, .065), 0., .85),
            ("rubber", (.018, .022, .026), 0., .78),
            ("galvanized", (.42, .46, .49), 1., .44),
            ("sensor_lens", (.12, .025, .012), .0, .12),
        ]:
            gallery.materials[name] = gallery.material(name, color, metal, rough)

    @contextmanager
    def assembly(self, path, pos, yaw=0.):
        g = self.g
        room = g.active_room
        root = UsdGeom.Xform.Define(g.stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(*g._position(pos)))
        root.AddRotateYOp().Set(yaw - (math.degrees(PLACEMENTS[room][2]) if room else 0.))
        g.active_room = None
        try:
            yield root
        finally:
            g.active_room = room

    def carton(self, path, pos, size=(.7, .6, .65), dynamic=False, material="carton"):
        g = self.g
        w, h, d = size
        with self.assembly(path, pos) as root:
            if self.simready.assets:
                variant='printed_carton' if sum(map(ord,path))%3==0 else 'carton'
                self.simready.reference(path+'/Visual',variant,size=size)
                # Invisible analytic collision geometry keeps the 2 kg test load
                # stable; the visible corrugated shell follows the same body.
                body=g.box(path+'/CorrugatedBox',(0,0,0),size,material,collider=dynamic)
                body.MakeInvisible()
            else:
                body = g.box(path+"/CorrugatedBox", (0, 0, 0), size, material, collider=dynamic)
                g.box(path+"/TopTape", (0, h/2+.00015, 0), (.065, .0002, d), "tape")
                g.box(path+"/FrontTape", (0, 0, d/2+.00015), (.065, h, .0002), "tape")
                g.box(path+"/ShippingLabel", (-w*.24, h*.02, d/2+.00035), (w*.32, h*.48, .0003), "paper")
                for i in range(9):
                    g.box(path+f"/Barcode{i}", (-w*.37+i*w*.031, h*.01, d/2+.0006), (w*(.009 if i%3 else .019), h*.19, .0001), "black")
                for i in range(2):
                    g.box(path+f"/LabelLine{i}", (-w*.24, h*(.18+i*.05), d/2+.0006), (w*.23, h*.014, .0001), "black")
            if dynamic:
                UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
                mass=UsdPhysics.MassAPI.Apply(root.GetPrim())
                mass.CreateMassAttr(2.)
                # Uniform 2 kg educational test load, in kg*m^2. The visible
                # corrugated shell is not a claim about a measured customer's SKU.
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f((h*h+d*d)/6,(w*w+d*d)/6,(w*w+h*h)/6))
                UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(g.bounce_material, materialPurpose="physics")
                apply(root.GetPrim(), "PhysxRigidBodyAPI", {
                    "physxRigidBody:solverPositionIterationCount": ("int", 12),
                    "physxRigidBody:solverVelocityIterationCount": ("int", 4),
                    "physxRigidBody:maxDepenetrationVelocity": ("float", .5),
                    "physxRigidBody:angularDamping": ("float", .35)})
                apply(body.GetPrim(), "PhysxCollisionAPI", {
                    "physxCollision:contactOffset": ("float", .003),
                    "physxCollision:restOffset": ("float", 0.)})
        return root

    def pallet(self, path, pos, size=(1.5, 1.25)):
        g = self.g
        w, d = size
        with self.assembly(path, pos) as root:
            if self.simready.assets:
                self.simready.reference(path+'/Visual','pallet',size=(w,.16,d))
                return root
            for i in range(3):
                g.box(path+f"/Runner{i}", ((i-1)*w*.38, .095, 0), (w*.12, .19, d), "wood")
            for i in range(6):
                g.box(path+f"/Slat{i}", (0, .23, -d*.43+i*d*.172), (w, .08, d*.14), "wood")
        return root

    def conveyor(self, path, pos, length=3.7, depth=1.6, height=1., yaw=0., collider=False):
        """Rollers run across local Z; material flows along local X."""
        if self.simready.assets:
            return self.simready_conveyor(path,pos,length,depth,height,yaw,collider)
        g = self.g
        with self.assembly(path, pos, yaw) as root:
            deck = g.box(path+"/Bed", (0, height-.13, 0), (length, .22, depth), "belt")
            if collider:
                contact = g.box(path+"/ContactSurface", (0, height+.025, 0), (length, .10, depth-.12), "belt", collider=True)
                # A visible, continuous receiving belt matches the solver's
                # contact surface; the rollers below support that belt.
                UsdShade.MaterialBindingAPI.Apply(contact.GetPrim()).Bind(g.bounce_material, materialPurpose="physics")
                apply(contact.GetPrim(), "PhysxCollisionAPI", {"physxCollision:contactOffset": ("float", .003)})
            for i, x in enumerate((-length*.4, length*.4)):
                for j, z in enumerate((-depth*.38, depth*.38)):
                    g.box(path+f"/Leg{i}{j}", (x, height*.43, z), (.13, height*.86, .13), "steel")
                    g.box(path+f"/Foot{i}{j}", (x, .06, z), (.3, .1, .28), "black")
                g.box(path+f"/Brace{i}", (x, .35, 0), (.12, .13, depth), "steel")
            for i, z in enumerate((-depth/2, depth/2)):
                g.box(path+f"/Frame{i}", (0, height-.02, z), (length+.12, .32, .09), "rack_blue", collider=collider)
                for j in range(6):
                    g.cylinder(path+f'/FrameBolt{i}_{j}',(-length*.43+j*length*.172,height-.02,z+math.copysign(.052,z)),.014,.014,'steel',axis='Z')
            for i in range(max(5, round(length/.22))):
                count = max(5, round(length/.22))
                g.cylinder(path+f"/Roller{i}", (-length*.46+i*length*.92/(count-1), height, 0), .075, depth-.08, "steel",axis='Z')
            g.cylinder(path+"/DriveMotor", (length*.32, height-.33, depth*.59), .16, .48, "blue",axis='X')
            for i in range(5):
                g.box(path+f"/MotorFin{i}", (length*.19+i*.07, height-.33, depth*.59), (.026, .35, .4), "steel")
            # Service-side details stay attached to the actual assembly, so
            # Clash includes them in mesh queries and preserves their materials.
            z = depth*.5+.07
            g.box(path+'/CableTray', (0,height-.29,z), (length*.72,.10,.10), 'galvanized')
            g.cylinder(path+'/CableLoom', (0,height-.225,z), .018, length*.65, 'rubber', axis='X')
            for i, x in enumerate((-length*.3,length*.3)):
                g.box(path+f'/TrayBracket{i}', (x,height-.22,z-.03), (.035,.21,.15), 'galvanized')
                g.box(path+f'/SensorBracket{i}', (x,height+.035,z-.025), (.09,.16,.035), 'galvanized')
                g.box(path+f'/Photoeye{i}', (x,height+.07,z-.055), (.075,.055,.07), 'black')
                g.cylinder(path+f'/SensorLens{i}', (x,height+.07,z-.095), .015,.012,'sensor_lens',axis='Z')
            g.box(path+'/DriveGuard', (length*.33,height-.36,depth*.59+.225), (.62,.36,.025),'rack_blue')
            for i in range(6):
                g.box(path+f'/GuardVent{i}', (length*.20+i*.052,height-.36,depth*.59+.24), (.019,.20,.009),'rubber')
        return root

    def rack(self, path, pos, width=2.6, height=3.0, shelves=2, stock_levels=2):
        g = self.g
        with self.assembly(path, pos) as root:
            if self.simready.assets:
                scale=width/self.simready.assets['rack']['size'][0]
                self.simready.reference(path+'/Visual','rack',scale=scale)
                # Source shelf elevations are inspected at authoring time.
                for n,y in enumerate((.506,2.382)):
                    if n>=stock_levels:continue
                    for j in range(3):
                        self.carton(path+f'/Stock{n}{j}',((j-1)*width*.25,y*scale+.35,0),(.64,.68,.9))
                return root
            for i, x in enumerate((-width/2, width/2)):
                for j, z in enumerate((-.6, .6)):
                    g.box(path+f"/Upright{i}{j}", (x, height/2, z), (.1, height, .12), "rack_blue")
                    g.box(path+f"/Guard{i}{j}", (x, .23, z), (.21, .46, .22), "safety")
                for n in range(4):
                    g.box(path+f"/Cross{i}{n}", (x, .35+n*.7, 0), (.08, .07, 1.2), "steel")
            for n in range(shelves):
                y = .38+n*1.12
                g.box(path+f"/Shelf{n}", (0, y, 0), (width, .09, 1.3), "steel")
                for j, z in enumerate((-.65, .65)):
                    g.box(path+f"/Beam{n}{j}", (0, y-.08, z), (width+.15, .18, .12), "rack_orange")
                if n>=stock_levels:continue
                for j in range(3):
                    self.carton(path+f"/Stock{n}{j}", ((j-1)*.73, y+.39, 0), (.64, .68, .9))
        return root

    def boundary(self, path, x, z, width, depth):
        g = self.g
        for i, sx in enumerate((-1, 1)):
            g.box(path+f"/Side{i}", (x+sx*width/2, .012, z), (.055, .018, depth), "safety")
            g.box(path+f"/End{i}", (x, .012, z+sx*depth/2), (width, .018, .055), "safety")

    def bollard(self, path, pos):
        with self.assembly(path, pos):
            self.g.cylinder(path+"/Post", (0, .48, 0), .09, .96, "safety")
            self.g.cylinder(path+"/Stripe", (0, .65, 0), .094, .16, "black")
            self.g.box(path+"/Foot", (0, .04, 0), (.32, .08, .32), "black")

    def stacklight(self, path, pos):
        with self.assembly(path, pos):
            self.g.cylinder(path+"/Mast", (0, .22, 0), .028, .44, "steel")
            for i, mat in enumerate(("green", "safety", "rack_orange")):
                self.g.cylinder(path+f"/Lamp{i}", (0, .48+i*.13, 0), .075, .11, mat)

    def amr(self, path, pos):
        g = self.g
        with self.assembly(path, pos) as root:
            if self.simready.assets:
                self.simready.reference(path+'/Visual','amr',instance=True)
                self.simready.reference(path+'/Load','amr_load',instance=True)
            else:
                g.box(path+"/Chassis", (0, .38, 0), (1.6, .42, 1.2), "blue")
                g.box(path+"/Bumper", (0, .23, 0), (1.69, .15, 1.29), "black")
                g.box(path+"/Deck", (0, .63, 0), (1.52, .12, 1.18), "steel")
                for i, x in enumerate((-.52, .52)):
                    for j, z in enumerate((-.61, .61)):
                        g.cylinder(path+f"/Wheel{i}{j}", (x, .22, z), .22, .12, "black",axis='Z')
                        outer=z+math.copysign(.068,z)
                        g.cylinder(path+f'/Hub{i}{j}',(x,.22,outer),.125,.014,'aluminum',axis='Z')
                        g.cylinder(path+f'/Axle{i}{j}',(x,.22,outer+math.copysign(.01,z)),.045,.012,'steel',axis='Z')
                g.cylinder(path+"/Lidar", (.57, .78, 0), .13, .19, "black")
                g.cylinder(path+"/LidarBand", (.57, .79, 0), .134, .05, "scene_glow")
                for i, x in enumerate((-.65, .65)):
                    g.box(path+f"/Status{i}", (x, .41, .607), (.19, .055, .012), "scene_glow")
                with self.assembly(path+"/Load", (0, .72, 0)):
                    self.pallet(path+"/Load/Pallet", (0, 0, 0), (1.35, 1.05))
                    for i in range(2):
                        self.carton(path+f"/Load/Carton{i}", ((i-.5)*.64, .62, 0), (.6, .7, .85))
        return root

    def simready_conveyor(self,path,pos,length,depth,height,yaw,collider):
        """Repeat complete roller sections at uniform scale; don't squash rollers."""
        g=self.g
        asset=self.simready.assets['conveyor']
        lanes=max(1,round(depth/asset['size'][2]))
        count=max(1,round(length/(asset['size'][0]*depth/lanes/asset['size'][2])))
        scale=min(length/count/asset['size'][0],depth/lanes/asset['size'][2])
        offset=height+.075-asset['deck_height']*scale
        with self.assembly(path,pos,yaw) as root:
            for lane in range(lanes):
                z=(lane-(lanes-1)/2)*depth/lanes
                for n in range(count):
                    x=(n-(count-1)/2)*length/count
                    self.simready.reference(path+f'/Section{lane}_{n}','conveyor',scale=scale,
                                            pos=(x,offset,z),instance=False)
            # A fabricated bridge supports the source foot plates. Legs sit
            # at the ends of the span, leaving a real loaded route underneath.
            from pxr import Sdf
            for lane in range(lanes):
                lane_z=(lane-(lanes-1)/2)*depth/lanes
                for side,sign in enumerate((-1,1)):
                    z=lane_z+sign*depth/lanes*.43
                    beam=path+f'/Bearer{lane}_{side}'
                    g.box(beam+'Web',(0,offset-.075,z),(length,.13,.04),'galvanized')
                    for n,y in enumerate((offset-.014,offset-.137)):
                        g.box(beam+f'Flange{n}',(0,y,z),(length,.028,.20),'galvanized')
                    for end,sign_x in enumerate((-1,1)):
                        x=sign_x*(length/2-.20)
                        top=offset-.15
                        name=f'{lane}_{side}_{end}'
                        support=g.box(path+'/Support'+name,(x,top/2,z),(.105,max(.02,top),.105),'steel')
                        support.GetPrim().CreateAttribute('lab:supportTop',Sdf.ValueTypeNames.Double).Set(top)
                        support.GetPrim().CreateAttribute('lab:supportXZ',Sdf.ValueTypeNames.Double2).Set((x,z))
                        for label,y,size,material in (
                            ('BasePlate',.018,(.29,.036,.29),'galvanized'),
                            ('BaseSleeve',.10,(.14,.20,.14),'rack_blue')):
                            ground=g.box(path+'/'+label+name,(x,y,z),size,material)
                            ground.GetPrim().CreateAttribute('lab:groundY',Sdf.ValueTypeNames.Double).Set(y)
            if collider:
                contact=g.box(path+'/ContactSurface',(0,height+.025,0),(length,.10,depth-.12),'belt',collider=True)
                contact.MakeInvisible()
                UsdShade.MaterialBindingAPI.Apply(contact.GetPrim()).Bind(g.bounce_material,materialPurpose='physics')
                apply(contact.GetPrim(),'PhysxCollisionAPI',{'physxCollision:contactOffset':('float',.003)})
        return root
