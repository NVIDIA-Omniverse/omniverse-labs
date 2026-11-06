# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A moving mobile-robot exhibit for native RTX LiDAR sensing."""
import math
from pxr import Gf, Sdf, Usd, UsdGeom
from .lidar_sensor import LidarReadout, MOUNT_PATH, SENSOR_PATH


class LidarRoom:
    def __init__(self, g):
        self.g, self.stage = g, g.stage
        self.visible = False
        self.moving = True
        self.elapsed = 0.
        self.offset = -1.8
        self.target_near = False
        self.wheels = []
        self._last_offset = None
        self.root = UsdGeom.Xform.Define(self.stage, '/World/LidarDisplay')
        old = g.active_room
        g.active_room = 'rendering'
        g.box('/World/LidarDisplay/Platform', (0,.035,-5.4), (7.5,.07,5.8), 'black')
        g.factory.boundary('/World/LidarDisplay/Boundary', 0,-5.4,7.7,6.)
        self.robot = g.factory.amr('/World/LidarDisplay/Robot', (self.offset,.07,-4.5))
        UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/LidarDisplay/Robot/Load')).MakeInvisible()
        # Retain the supplied robot geometry and materials. Only this exhibit's
        # reference is editable, allowing its existing wheels to roll visibly.
        asset = self.stage.GetPrimAtPath('/World/LidarDisplay/Robot/Visual/Asset')
        if asset:
            asset.SetInstanceable(False)
            geometry = self.stage.GetPrimAtPath(str(asset.GetPath())+'/Axis/Model/Geometry')
            for pivot in geometry.GetChildren():
                if pivot.GetName().startswith('wheel_'):
                    for mesh in pivot.GetChildren():
                        if mesh.IsA(UsdGeom.Mesh):
                            xform = UsdGeom.Xformable(mesh)
                            xform.ClearXformOpOrder()
                            self.wheels.append((xform.AddTransformOp(opSuffix='rolling'),pivot))
                            xform.SetResetXformStack(True)
        # Compact bracket and sensor head sit just ahead of the trolley frame.
        g.active_room = None
        g.cylinder('/World/LidarDisplay/Robot/SensorMast', (.70,.65,0), .026,.72,'steel')
        g.box('/World/LidarDisplay/Robot/SensorMount', (.70,.30,0), (.18,.045,.20),'aluminum')
        g.box('/World/LidarDisplay/Robot/SensorBracket', (.70,1.01,0), (.42,.055,.15), 'aluminum')
        g.cylinder('/World/LidarDisplay/Robot/SensorBase', (.86,1.065,0), .10,.07,'steel')
        # Leave the optical aperture open: an opaque cylinder around the ray
        # origin would physically block every return before the near range.
        g.cylinder('/World/LidarDisplay/Robot/SensorHousing', (.86,1.085,0), .093,.025,'black')
        g.cylinder('/World/LidarDisplay/Robot/SensorCap', (.86,1.19,0), .10,.025,'aluminum')
        g.cylinder('/World/LidarDisplay/Robot/SensorStatus', (.86,1.198,0), .101,.006,'scene_glow')
        # Opposing compact camera bodies are fixed to the underside of the
        # sensor bracket. Optical origins sit in front of the visible lenses.
        for name, x, direction in (('Front',.865,1), ('Rear',.535,-1)):
            base='/World/LidarDisplay/Robot/'+name+'Camera'
            g.box(base+'/Housing',(x,.94,0),(.10,.105,.13),'black')
            g.cylinder(base+'/LensRing',(x+direction*.055,.94,0),.036,.016,'steel',axis='X')
            g.cylinder(base+'/Lens',(x+direction*.065,.94,0),.029,.004,'black',axis='X')
        g.active_room = 'rendering'
        g.factory.rack('/World/LidarDisplay/Rack', (-2.,.07,-7.65),2.5,2.5,2)
        with g.factory.assembly('/World/LidarDisplay/Target', (2.25,.07,-7.3)) as target:
            self.target = target
            g.factory.pallet('/World/LidarDisplay/Target/Pallet',(0,0,0),(1.2,1.))
            for level in range(3):
                for col in range(2):
                    g.factory.carton(f'/World/LidarDisplay/Target/Carton{level}_{col}',
                                     ((col-.5)*.58,.16+.25+level*.50,0),(.56,.50,.8))
        g.active_room = old
        # This small, moving exhibit uses editable references. Refreshing a
        # target's shared instance prototype can disturb static cartons in
        # other instances. Keep the geometry/material files shared on disk,
        # but give these few display props independent live transform state.
        for prim in Usd.PrimRange(self.root.GetPrim()):
            if prim.IsInstance():prim.SetInstanceable(False)
        # OmniLidar is a native schema that usd-core need not recognize as
        # Xformable. Read the robot attachment's pose from a standard USD Xform.
        UsdGeom.Xform.Define(self.stage,MOUNT_PATH).AddTranslateOp().Set(Gf.Vec3d(.86,1.12,0.))
        self.sensor_pose=UsdGeom.Xform.Define(self.stage,SENSOR_PATH).AddTransformOp(opSuffix='sensorPose')
        from .robot_cameras import author
        self.camera_poses=author(self.stage,g.campus)
        self.camera_pixels={}
        self.camera_capture={}
        self.readout = LidarReadout(g.room_position('rendering',(0,0,-5.4)))
        self._prepare_display_poses()
        self.set_visible(False)
        self.advance(0.)
        # Every rendered moving mesh already has an explicit world pose.
        # Sending the static asset hierarchy again on each scan is redundant.
        moving = {str(self.robot.GetPath()), str(self.target.GetPath()),
                  str(self.sensor_pose.GetAttr().GetPrim().GetPath())}
        moving.update(str(op.GetAttr().GetPrim().GetPath()) for op, _ in self.camera_poses.values())
        moving.update(str(op.GetAttr().GetPrim().GetPath()) for op, _ in self.wheels)
        moving.update(str(op.GetAttr().GetPrim().GetPath())
                      for poses in self.display_poses.values() for op, _ in poses)
        self.scan_paths = sorted(moving)
        self._scan_prims = [self.stage.GetPrimAtPath(path) for path in self.scan_paths]

    def _prepare_display_poses(self):
        # The pinned renderer can cache inherited mesh poses even after the
        # parent's canonical transform changes. Give every moving display mesh
        # an explicit world pose, as the fleet already does for its wheels.
        # Source geometry and materials stay untouched. Wheel rotation is
        # handled separately around the original asset's axle frames.
        wheels={op.GetAttr().GetPrim().GetPath() for op,_ in self.wheels}
        self.display_poses={}
        for root in (self.robot.GetPrim(),self.target.GetPrim()):
            cache=UsdGeom.XformCache()
            inverse=cache.GetLocalToWorldTransform(root).GetInverse()
            meshes=[(prim,cache.GetLocalToWorldTransform(prim)*inverse)
                    for prim in Usd.PrimRange(root)
                    if UsdGeom.Boundable(prim) and prim.GetPath() not in wheels]
            poses=[]
            for prim,rest in meshes:
                xform=UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                op=xform.AddTransformOp(opSuffix='displayPose')
                xform.SetResetXformStack(True)
                poses.append((op,rest))
            self.display_poses[str(root.GetPath())]=poses
            self._update_display_poses(root)

    def _update_display_poses(self,root):
        world=UsdGeom.XformCache().GetLocalToWorldTransform(root)
        with Sdf.ChangeBlock():
            for op,rest in self.display_poses[str(root.GetPath())]:
                matrix=rest*world
                if op.Get()!=matrix:op.Set(matrix)

    def scan_state(self):
        cache=UsdGeom.XformCache()
        # Publish effective poses for every moving component. The native
        # sensor-only pipeline must not depend on a color-camera traversal to
        # invalidate inherited mesh transforms after a parent moves.
        return {'paths':self.scan_paths,
                'matrices':[[list(row) for row in cache.GetLocalToWorldTransform(prim)]
                            for prim in self._scan_prims], 'reset':[True]*len(self.scan_paths)}

    def set_visible(self, visible):
        if visible and not self.visible:
            self.readout.clear(self.sensor_position())
            self.camera_pixels.clear()
            self.camera_capture.clear()
        self.visible = visible
        (self.root.MakeVisible if visible else self.root.MakeInvisible)()

    def toggle_target(self):
        self.target_near = not self.target_near
        self.target.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*self.g.room_position(
            'rendering', (2.25,.07,-6.1 if self.target_near else -7.3))))
        self._update_display_poses(self.target.GetPrim())

    def advance(self, dt):
        if dt and self.visible and self.moving:
            self.elapsed += dt
        self.offset = -1.8*math.cos(self.elapsed*.42)
        if self.offset == self._last_offset:
            return
        self._last_offset = self.offset
        op = self.robot.GetPrim().GetAttribute('xformOp:translate')
        position = Gf.Vec3d(*self.g.room_position('rendering',(self.offset,.07,-4.5)))
        if op.Get()!=position:op.Set(position)
        self._update_display_poses(self.robot.GetPrim())
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0),math.degrees((self.offset+1.8)/.1)))
        cache = UsdGeom.XformCache()
        self.sensor_pose.Set(cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(MOUNT_PATH)))
        robot_world=cache.GetLocalToWorldTransform(self.robot.GetPrim())
        for camera, local in self.camera_poses.values():
            camera.Set(local*robot_world)
        matrices = [rotation*cache.GetLocalToWorldTransform(pivot) for _,pivot in self.wheels]
        with Sdf.ChangeBlock():
            for (wheel,_),matrix in zip(self.wheels,matrices):
                if wheel.Get()!=matrix:wheel.Set(matrix)

    def sensor_position(self):
        return UsdGeom.Xformable(self.stage.GetPrimAtPath(MOUNT_PATH)).ComputeLocalToWorldTransform(0).ExtractTranslation()

    def reset(self):
        self.elapsed=0.
        self.moving=True
        if self.target_near:self.toggle_target()
        self.readout.color='distance'
        self.readout.view='3d'
        self.advance(0.)
