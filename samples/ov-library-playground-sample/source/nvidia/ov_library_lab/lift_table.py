# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dimensioned scissor-table visualization driven by the ovui shared model.

This is kinematic geometry, not a hydraulic or load-capacity simulation.
Fixed left pivots and right rollers preserve the platform's vertical path.
"""
import math
from pxr import Gf, UsdGeom


class LiftTable:
    arm_length=3.25
    lower_y=.42
    left_x=-1.65

    def __init__(self,g):
        self.g=g
        self.links=[]
        self.pins=[]
        self.actuators=[]
        p='/World/Interface/Mechanism'
        with g.factory.assembly(p,(0,0,-5.5)):
            # Structural channels, fixed pivot brackets, and captive roller tracks.
            for side,z in enumerate((-.79,.79)):
                for suffix,dz in (('Inner',-.105),('Outer',.105)):
                    g.box(p+f'/Base{side}{suffix}',(0,.34,z+dz),(3.9,.17,.055),'rack_blue')
                g.box(p+f'/Track{side}',(0,.3025,z),(3.85,.045,.20),'steel')
                g.box(p+f'/Bracket{side}',(self.left_x,.435,z),(.23,.24,.27),'rack_blue')
                for branch in range(2):
                    path=p+f'/Arm{side}_{branch}'
                    with g.factory.assembly(path,(0,0,0)) as arm:
                        g.box(path+'/Beam',(0,0,0),(self.arm_length,.15,.10),'rack_orange')
                    rotation=arm.AddRotateZOp()
                    self.links.append((side,branch,arm.GetPrim().GetAttribute('xformOp:translate'),rotation))
                for label in ('BottomLeft','BottomRight','TopLeft','TopRight','Center'):
                    pin=g.cylinder(p+f'/Pin{side}{label}',(0,0,z),.095,.30,'steel',axis='Z')
                    self.pins.append((z,label,pin.GetPrim().GetAttribute('xformOp:translate')))
            for i,x in enumerate((-1.82,1.82)):
                g.box(p+f'/Crossmember{i}',(x,.32,0),(.16,.16,1.82),'rack_blue')
                for j,z in enumerate((-.96,.96)):
                    g.box(p+f'/Mount{i}_{j}',(x,.255,z),(.34,.07,.32),'steel')
                    g.cylinder(p+f'/Anchor{i}_{j}',(x,.308,z),.025,.04,'black')
            # Two telescoping cylinders connect the base to the rising arms.
            g.cylinder(p+'/CylinderBaseShaft',(.9,.38,0),.065,1.82,'steel',axis='Z')
            crosshead=g.cylinder(p+'/HydraulicCrosshead',(0,.5,0),.065,1.82,'steel',axis='Z')
            self.crosshead=crosshead.GetPrim().GetAttribute('xformOp:translate')
            for i,z in enumerate((-.43,.43)):
                path=p+f'/Hydraulic{i}'
                with g.factory.assembly(path,(.9,.38,z)) as actuator:
                    g.cylinder(path+'/Barrel',(.355,0,0),.078,.71,'rack_blue',axis='X')
                    g.cylinder(path+'/Gland',(.69,0,0),.088,.07,'steel',axis='X')
                    rod=g.cylinder(path+'/Rod',(.8,0,0),.043,1.,'silver',axis='X')
                    rod_scale=rod.AddScaleOp()
                    tip=g.cylinder(path+'/Clevis',(1.,0,0),.09,.20,'steel',axis='Z')
                    g.cylinder(path+'/BasePin',(0,0,0),.09,.20,'steel',axis='Z')
                self.actuators.append((actuator.AddRotateZOp(),rod.GetPrim().GetAttribute('xformOp:translate'),rod_scale,tip.GetPrim().GetAttribute('xformOp:translate')))
            g.box(p+'/PowerPack',(2.35,.56,-.40),(.62,.59,.75),'rack_blue')
            g.cylinder(p+'/PumpMotor',(2.35,.98,-.40),.19,.36,'steel')
            g.box(p+'/ElectricalBox',(2.35,.67,.06),(.46,.34,.16),'ivory')
            for j,z in enumerate((-.51,-.33)):
                g.cylinder(p+f'/Hose{j}',(1.63,.29,z),.022,1.45,'black',axis='X')

        with g.factory.assembly('/World/Interface/Lift',(0,0,-5.5)) as self.platform:
            p='/World/Interface/Lift'
            g.box(p+'/Platform',(0,0,0),(3.90,.22,2.12),'steel')
            for i,z in enumerate((-1.055,1.055)):
                g.box(p+f'/SafetyEdge{i}',(0,-.025,z),(3.9,.16,.035),'safety')
                # The end roller remains on this captive underside channel.
                g.box(p+f'/Guide{i}',(0,-.22,z*.75),(3.8,.18,.08),'rack_blue')
            g.factory.pallet(p+'/Pallet',(0,.11,0),(2.3,1.5))
            for i in range(3):
                g.factory.carton(p+f'/Carton{i}',((i-1)*.7,.74,0),(.64,.72,1.05))
            self.tag=g.box(p+'/InspectionTag',(0,.77,.54),(.26,.28,.02),'blue')

    def set_height(self,height):
        top=height-.18
        rise=top-self.lower_y
        span=math.sqrt(self.arm_length**2-rise**2)
        middle_x=self.left_x+span/2
        middle_y=(top+self.lower_y)/2
        angle=math.degrees(math.atan2(rise,span))
        for side,branch,translate,rotate in self.links:
            z=(-.79,.79)[side]+(-.065 if branch==0 else .065)
            translate.Set(Gf.Vec3d(middle_x,middle_y,z))
            rotate.Set(angle if branch==0 else -angle)
        for z,label,translate in self.pins:
            x=middle_x if label=='Center' else self.left_x+(span if label.endswith('Right') else 0.)
            y=middle_y if label=='Center' else top if label.startswith('Top') else self.lower_y
            translate.Set(Gf.Vec3d(x,y,z))
        # Attach at 55% of the rising arm; the cylinder stroke follows that pin.
        dx=self.left_x+.55*span-.9
        dy=self.lower_y+.55*rise-.38
        self.crosshead.Set(Gf.Vec3d(dx+.9,dy+.38,0))
        length=math.hypot(dx,dy)
        for rotate,translate,scale,tip in self.actuators:
            rotate.Set(math.degrees(math.atan2(dy,dx)))
            translate.Set(Gf.Vec3d((.65+length)/2,0,0))
            scale.Set(Gf.Vec3f(length-.65,1.,1.))
            tip.Set(Gf.Vec3d(length,0,0))
        self.platform.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*self.g.room_position('interface',(0,height,-5.5))))
