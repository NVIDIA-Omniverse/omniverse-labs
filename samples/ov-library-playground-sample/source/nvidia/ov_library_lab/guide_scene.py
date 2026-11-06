# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The supplied hovering avatar, animated through authored USD transforms.

No locomotion/character solver is claimed: the asset has no legs or skeleton.
The subtle hover and rigid-part gestures are host animation, rendered by ovrtx
through the same ovstage update path as the rest of the lab.
"""
import math
from pxr import Gf, Usd, UsdGeom
from .guide_motion import GuideMotion, HOVER_HEIGHT, gesture


class GuideScene:
    def __init__(self, stage, asset):
        self.stage = stage
        self.motion = GuideMotion()
        self.root = UsdGeom.Xform.Define(stage,'/World/Atrium/Guide')
        self.translate = self.root.AddTranslateOp()
        self.rotate = self.root.AddRotateYOp()
        attitude = UsdGeom.Xform.Define(stage,self.root.GetPath().AppendChild('Attitude'))
        self.attitude = attitude.AddRotateXYZOp()
        model = UsdGeom.Xform.Define(stage,attitude.GetPath().AppendChild('Avatar'))
        model.GetPrim().GetReferences().AddReference(str(asset.resolve()))
        model.AddRotateXOp().Set(-90.)  # supplied Z-up, lab Y-up; face local +Z
        model.AddScaleOp().Set(Gf.Vec3f(1.65))
        self.parts = {}
        for name in ('Head','Helmet','Ear_Left','Ear_Right','Flashlight','Arm_Left','Arm_Right','Clipboard'):
            prim = stage.GetPrimAtPath(model.GetPath().AppendChild(name))
            if not prim:
                raise ValueError('Guide asset is missing expected rigid part: '+name)
            self.parts[name] = UsdGeom.Xformable(prim).AddTransformOp(opSuffix='guidePose')
        # Publish the referenced asset's final mesh transforms explicitly.
        # Ancestor-only writes reach ovstage, but the pinned renderer retains
        # this asset's original mesh poses. World-space mesh updates avoid
        # relying on that ancestor propagation, without editing mesh points.
        self.mesh_poses = []
        for prim in Usd.PrimRange(model.GetPrim()):
            if not prim.IsA(UsdGeom.Mesh):continue
            xform = UsdGeom.Xformable(prim)
            local = xform.GetLocalTransformation()
            reset = xform.GetResetXformStack()
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(opSuffix='guideWorld')
            xform.SetResetXformStack(True)
            self.mesh_poses.append((op,local,None if reset else prim.GetParent()))
        self.pitch = self.look = self.arm = self.board = self.writing = 0.
        self.elapsed = 0.
        self.pose_writes = 0
        self.transform_writes = 0
        self._last_values = {}
        self.apply_pose(0.)

    def set_pose_value(self, op, value):
        # Quiet head/arm poses must not create USD notices and native scene
        # writes on every hover tick. The changed transforms remain batched.
        key = id(op)
        if self._last_values.get(key) == value:return
        op.Set(value)
        self._last_values[key] = value
        self.transform_writes += 1

    @property
    def bubble_anchor(self):
        # UI follows the visitor's camera, but never the hover/gesture pose.
        return self.motion.x,HOVER_HEIGHT+1.,self.motion.z

    @staticmethod
    def pivot(point, xyz):
        rotation = (Gf.Rotation(Gf.Vec3d(1,0,0),xyz[0]) *
                    Gf.Rotation(Gf.Vec3d(0,1,0),xyz[1]) *
                    Gf.Rotation(Gf.Vec3d(0,0,1),xyz[2]))
        return (Gf.Matrix4d().SetTranslate(-Gf.Vec3d(*point)) *
                Gf.Matrix4d().SetRotate(rotation) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*point)))

    def apply_pose(self, dt):
        m,t = self.motion,self.motion.clock
        speaking = m.state == 'talking'
        # All parts ease into a pose; the clipboard follows the same wrist-side
        # shoulder transform as its supporting hand, so it never drifts away.
        blend = 1.-math.exp(-max(0.,dt)*3.8)
        pitch = 12.*m.clipboard + (4.*gesture(t%9.,1.,3.,.65) if speaking else 0.)
        look = -3.*m.clipboard
        arm = 1.8*math.sin(t*.9) if speaking else 0.
        board = -16.*m.clipboard
        self.pitch += (pitch-self.pitch)*blend
        self.look += (look-self.look)*blend
        self.arm += (arm-self.arm)*blend
        self.board += (board-self.board)*blend
        self.writing += (m.writing-self.writing)*blend
        for name in ('pitch','look','arm','board','writing'):
            if abs(getattr(self,name)) < .0001:setattr(self,name,0.)
        self.set_pose_value(self.translate,Gf.Vec3d(m.x,HOVER_HEIGHT+.016*math.sin(t*1.05)+.003*math.sin(t*.53),m.z))
        self.set_pose_value(self.rotate,math.degrees(m.yaw))
        self.set_pose_value(self.attitude,Gf.Vec3f(0.,0.,.12*math.sin(t*.65)))
        head = self.pivot((.012,.025,.365),(self.pitch,0.,self.look))
        for name in ('Head','Helmet','Ear_Left','Ear_Right','Flashlight'):
            self.set_pose_value(self.parts[name],head)
        # The export has rigid arms, so use a shoulder reach and small strokes
        # near the clipboard edge, with no invented finger articulation.
        shoulder = Gf.Vec3d(.175,.015,.25)
        reach = Gf.Rotation(Gf.Vec3d(.08,-.13,-.16),Gf.Vec3d(-.165,-.19,.045))
        reach = Gf.Rotation(Gf.Slerp(self.writing,Gf.Quatd(1.),reach.GetQuat()))
        stroke = self.writing*1.3*math.sin(t*8.8)
        hand = (Gf.Matrix4d().SetTranslate(-shoulder) * Gf.Matrix4d().SetRotate(reach) *
                Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),stroke+self.arm)) *
                Gf.Matrix4d().SetTranslate(shoulder))
        self.set_pose_value(self.parts['Arm_Left'],hand)
        held = self.pivot((-.12,.01,.30),(self.board,0.,0.))
        self.set_pose_value(self.parts['Arm_Right'],held)
        self.set_pose_value(self.parts['Clipboard'],held)
        cache = UsdGeom.XformCache()
        for op,local,parent in self.mesh_poses:
            world = local*cache.GetLocalToWorldTransform(parent) if parent else local
            self.set_pose_value(op,world)
        self.pose_writes += 1

    def update(self, dt, visitor, talking=False, frozen=False):
        self.motion.update(dt,visitor,talking,frozen)
        self.elapsed += max(0.,dt)
        if self.elapsed >= 1/30:
            self.apply_pose(self.elapsed)
            self.elapsed = 0.

    def reset(self):
        self.motion.reset()
        self.pitch = self.look = self.arm = self.board = self.writing = self.elapsed = 0.
        self.apply_pose(0.)
