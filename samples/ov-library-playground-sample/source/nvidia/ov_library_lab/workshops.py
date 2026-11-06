# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Richer factory experiments: cameras, fleet instances, revisions and streaming."""
import math
from pathlib import Path
import time

from pxr import Gf, Sdf, Usd, UsdGeom


class Workshops:
    def __init__(self, gallery):
        self.g, self.stage = gallery, gallery.stage
        self.fleet_count = 1
        self.fleet = []
        self.wheel_ops = []
        self.wheel_parents = []
        self.fleet_asset_ops = []
        self.wheel_angle = None
        self.light_angle, self.light_size, self.light_power = -40., 2., 3500.
        self.emitter_type = 'Rectangle'
        self.emitters = {'Rectangle': gallery.key_light}
        from pxr import UsdLux
        for name, kind in (('Disk', UsdLux.DiskLight), ('Sphere', UsdLux.SphereLight)):
            light = kind.Define(self.stage, '/World/Inspection'+name+'Light')
            light.AddTransformOp().Set(Gf.Matrix4d(1))
            light.CreateIntensityAttr(3500.)
            light.CreateColorAttr(Gf.Vec3f(1., .77, .48))
            light.CreateRadiusAttr(1.)
            light.MakeInvisible()
            self.emitters[name] = light
        self.camera_zoom = 32.
        self.render_subject = 'lidar'
        self.robot_roughness = .24
        self.snapshot_status = 'Capture the inspection camera to a PNG.'
        self.ui_widgets = {'jog': True, 'quality': True, 'light': True}
        self.lift_cycle = False
        self.lift_direction = 1.
        self.ui_events = 0
        self.stream_offset, self.stream_target = -2.3, -2.3
        self.stream_loaded = True
        self.stream_paused = False
        self.stream_fps = 5
        self.stream_status = 'Connect to see the scene camera on a receiving client.'
        self._prepare_fleet_wheels()
        self._fleet()
        self._prepare_fleet_poses()
        self._prepare_wheel_motion()
        self.update_fleet()
        self._inspection_camera()
        from .automotive import Automotive
        self.car = Automotive(gallery)
        from .lidar_room import LidarRoom
        self.lidar = LidarRoom(gallery)
        self.set_subject('lidar')
        self._stream_cell()
        gallery.active_room = None

    def _prepare_fleet_wheels(self):
        if not self.g.factory.simready.assets:
            return
        # A shared, editable asset definition gives the fleet its own wheel
        # animation without changing the asset file or other rooms' robots.
        source = UsdGeom.Xform.Define(self.stage, '/World/FleetAsset')
        info = self.g.factory.simready.assets['amr']
        source.GetPrim().GetReferences().AddReference(str((self.g.factory.simready.directory/info['file']).resolve()))
        source.MakeInvisible()
        instance = self.stage.GetPrimAtPath('/World/Shared/AMR/Visual/Asset')
        instance.GetReferences().ClearReferences()
        instance.GetReferences().AddInternalReference(source.GetPath())
        # Keep the static robot body instanced. Animated geometry inside its
        # prototype can remain frozen in RTX despite successful matrix writes;
        # the visible rolling wheels are separate editable meshes below.
        UsdGeom.Imageable(instance).MakeVisible()
        geometry = self.stage.GetPrimAtPath('/World/FleetAsset/Axis/Model/Geometry')
        for prim in geometry.GetChildren():
            if prim.GetName().startswith('wheel_'):
                UsdGeom.Imageable(prim).MakeInvisible()

    def _prepare_fleet_poses(self):
        if not self.g.factory.simready.assets:
            return
        # Snapshot all local offsets before overriding the shared source: clone
        # references otherwise inherit the newly authored op during this loop.
        assets = []
        for robot in (self.g.shared, *self.fleet):
            for suffix in ('/Visual/Asset', '/Load/Asset'):
                prim = self.stage.GetPrimAtPath(str(robot.GetPath())+suffix)
                assets.append((prim, UsdGeom.Xformable(prim).GetLocalTransformation()))
        for prim, rest in assets:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(opSuffix='fleetPose')
            xform.SetResetXformStack(True)
            self.fleet_asset_ops.append((op, rest, prim.GetParent()))

    def _prepare_wheel_motion(self):
        if not self.g.factory.simready.assets:
            return
        source_geometry = self.stage.GetPrimAtPath('/World/FleetAsset/Axis/Model/Geometry')
        for robot in (self.g.shared, *self.fleet):
            for source in source_geometry.GetChildren():
                if not source.GetName().startswith('wheel_'):
                    continue
                wheel = self.stage.DefinePrim(str(robot.GetPath())+'/Wheels/'+source.GetName())
                wheel.GetReferences().AddInternalReference(source.GetPath())
                wheel.SetInstanceable(False)
                # Override the source wheel's hidden state without making a
                # deliberately hidden fleet ancestor visible as a side effect.
                UsdGeom.Imageable(wheel).GetVisibilityAttr().Set('inherited')
                # The original hidden wheel provides the axle's frame inside
                # the unchanged, shared robot body.
                pivot = self.stage.GetPrimAtPath(str(robot.GetPath())+
                    '/Visual/Asset/Axis/Model/Geometry/'+source.GetName())
                for mesh in wheel.GetChildren():
                    if not mesh.IsA(UsdGeom.Mesh):
                        continue
                    # These meshes have axle-centred coordinates. Publish their
                    # effective world poses independently of the static shared
                    # prototype. Reset is authored before population, using
                    # ovstage's omni:xform + omni:resetXformStack contract.
                    xform = UsdGeom.Xformable(mesh)
                    xform.ClearXformOpOrder()
                    self.wheel_ops.append(xform.AddTransformOp(opSuffix='travel'))
                    xform.SetResetXformStack(True)
                    self.wheel_parents.append(pivot)

    def _fleet(self):
        g = self.g
        for i in range(1, 6):
            path = f'/World/Shared/Fleet/AMR{i+1:02}'
            prim = self.stage.DefinePrim(path)
            prim.GetReferences().AddInternalReference('/World/Shared/AMR')
            # The static body and cargo instance their geometry.
            # Keep this assembly reference editable: RTX does not propagate
            # parent visibility edits through a second, nested instance level.
            # Wheel meshes stay editable so their motion reaches RTX.
            prim.SetInstanceable(not bool(g.factory.simready.assets))
            self.fleet.append(UsdGeom.Xform(prim))
            self.fleet[-1].MakeInvisible()

    def set_fleet(self, count):
        self.fleet_count = count
        for i, robot in enumerate(self.fleet, 1):
            (robot.MakeVisible if i < count else robot.MakeInvisible)()
        self.g.revision += 1
        self.update_fleet()

    def update_fleet(self):
        g = self.g
        with Sdf.ChangeBlock():
            if g.factory.simready.assets:
                # Spread the taller loaded trolleys across the exhibit so each
                # instance stays readable and inside the route during travel.
                if self.fleet_count == 1:
                    slots = ((0., -5.7),)
                elif self.fleet_count == 3:
                    slots = ((-2.1, -5.2), (0., -6.4), (2.1, -5.2))
                else:
                    slots = tuple((x, z) for z in (-4.65, -6.7) for x in (-2.1, 0., 2.1))
                travel = g.shared_offset * (1. if self.fleet_count == 1 else .4)
                for robot, (x, z) in zip((g.shared, *self.fleet), slots):
                    robot.GetPrim().GetAttribute('xformOp:translate').Set(
                        Gf.Vec3d(*g.room_position('scene', (11 + x + travel, g.shared_floor_y, z))))
            else:
                if self.fleet_count == 6:
                    g.shared_translate.Set(Gf.Vec3d(*g.room_position('scene', (11+g.shared_offset*.4-1.9, g.shared_floor_y, -5.7))))
                for i, robot in enumerate(self.fleet, 1):
                    lane = (-7.5, -3.9, -5.7, -7.5, -3.9)[i-1]
                    offset = g.shared_offset if self.fleet_count < 6 else g.shared_offset*.4+(-1.9 if i < 3 else 1.9)
                    robot.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*g.room_position('scene', (11+offset, g.shared_floor_y, lane))))
        # Read the completed hierarchy after the translation change block.
        cache = UsdGeom.XformCache()
        matrices = [rest*cache.GetLocalToWorldTransform(parent)
                    for _, rest, parent in self.fleet_asset_ops]
        with Sdf.ChangeBlock():
            for (op, _, _), matrix in zip(self.fleet_asset_ops, matrices):
                if op.Get() != matrix:
                    op.Set(matrix)
        self.wheel_angle = math.degrees(g.shared_offset*(1. if self.fleet_count == 1 else .4)/.1)
        self._set_wheel_angle(self.wheel_angle)

    def _set_wheel_angle(self, angle):
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), angle))
        cache = UsdGeom.XformCache()
        matrices = [rotation*cache.GetLocalToWorldTransform(p) for p in self.wheel_parents]
        with Sdf.ChangeBlock():
            for op, matrix in zip(self.wheel_ops, matrices):
                if op.Get() != matrix:
                    op.Set(matrix)

    def _inspection_camera(self):
        g = self.g
        self.inspection_camera = g.make_camera('/World/InspectionCamera', g.room_position('rendering', (2.1, 2.8, -2.1)),
                                               g.room_position('rendering', (0, 1.95, -5.5)), self.camera_zoom)

    def set_camera_zoom(self, value):
        self.camera_zoom = float(value)
        self.inspection_camera.GetFocalLengthAttr().Set(self.camera_zoom)

    def camera_angle(self, side):
        car = self.render_subject == 'car'
        distance = 4.6 if car else 2.1
        eye = self.g.room_position('rendering', (distance if side else -distance, 2.8, -1.8 if car else -2.1))
        target = self.g.room_position('rendering', (0, .9 if car else 1.95, -5.5))
        self.inspection_camera.GetPrim().GetAttribute('xformOp:transform').Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse())

    def set_light(self, angle=None, size=None, power=None):
        if angle is not None:
            self.light_angle = float(angle)
        if size is not None:
            self.light_size = float(size)
        if power is not None:
            self.light_power = float(power)
        g = self.g
        theta = math.radians(self.light_angle)
        eye = g.room_position('rendering', (math.sin(theta)*3.4, 4.1, -5.5+math.cos(theta)*3.4))
        target = g.room_position('rendering', (0, .9 if self.render_subject == 'car' else 1.9, -5.5))
        g.key_light.GetPrim().GetAttribute('xformOp:transform').Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 1, 0)).GetInverse())
        if self.emitter_type == 'Rectangle':
            g.key_light.GetWidthAttr().Set(self.light_size)
            g.key_light.GetHeightAttr().Set(self.light_size)
        else:
            g.key_light.GetRadiusAttr().Set(self.light_size/2)
        if g.key_light.GetIntensityAttr().Get() > 0:
            g.key_light.GetIntensityAttr().Set(self.light_power)

    def set_emitter(self, name):
        old = self.g.key_light
        intensity, color = old.GetIntensityAttr().Get(), old.GetColorAttr().Get()
        old.MakeInvisible()
        self.emitter_type = name
        self.g.key_light = self.emitters[name]
        self.g.key_light.MakeVisible()
        self.g.key_light.GetIntensityAttr().Set(intensity)
        self.g.key_light.GetColorAttr().Set(color)
        self.set_light()

    def set_metallic(self, value):
        if self.render_subject == 'car':
            self.car.set_metallic(value)
        else:
            self.stage.GetPrimAtPath('/World/Looks/inspection_alloy/Surface').GetAttribute('inputs:metallic').Set(float(value))

    def set_subject(self, subject):
        self.render_subject = subject
        self.car.set_visible(subject == 'car')
        self.lidar.set_visible(subject == 'lidar')
        if subject == 'lidar':
            UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Rendering')).MakeInvisible()
            label = UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Signs/MaterialLabels'))
            if label:label.MakeInvisible()
        self.set_camera_zoom(28. if subject == 'car' else 32.)
        self.camera_angle(True)
        self.set_light()

    def _stream_cell(self):
        g = self.g
        g.active_room = 'streaming'
        g.box('/World/Streaming/Route', (0, .04, -5.5), (8.8, .08, 2.8), 'black', obstacle=True)
        g.factory.boundary('/World/Streaming/Safety', 0, -5.5, 9.2, 3.3)
        for i in range(15):
            g.box(f'/World/Streaming/RouteMark{i}', (-3.9+i*.55, .09, -4.55), (.24, .02, .065), 'streaming_glow')
        self.stream_floor_y = .08
        self.stream_robot = g.factory.amr('/World/Streaming/AMR', (-2.3, self.stream_floor_y, -5.5))
        for i, x in enumerate((-3., 3.)):
            g.factory.rack(f'/World/Streaming/Rack{i}', (x, 0, -8.5), 2.7, 2.8, 2)
        g.box('/World/Streaming/CameraPost', (3.5, 1.6, -3.6), (.1, 3.2, .1), 'steel')
        g.box('/World/Streaming/Camera', (3.5, 3.2, -3.6), (.32, .23, .38), 'black')
        g.factory.stacklight('/World/Streaming/Status', (-3.7, 0, -3.65))
        self.stream_camera = g.make_camera('/World/StreamCamera', g.room_position('streaming', (4.1, 4., -1.8)),
                                           g.room_position('streaming', (0, 1., -5.8)), 21.)

    def stream_command(self, command):
        if command == 'dispatch':
            self.stream_target = 2.3 if self.stream_target < 0 else -2.3
        elif command == 'cargo':
            self.stream_loaded = not self.stream_loaded
            cargo = UsdGeom.Imageable(self.stage.GetPrimAtPath('/World/Streaming/AMR/Load'))
            (cargo.MakeVisible if self.stream_loaded else cargo.MakeInvisible)()

    def advance(self, dt):
        difference = self.stream_target-self.stream_offset
        if difference and dt:
            self.stream_offset += math.copysign(min(abs(difference), dt*1.2), difference)
            self.stream_robot.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*self.g.room_position('streaming', (self.stream_offset, self.stream_floor_y, -5.5))))
        if self.lift_cycle and dt:
            value = self.g.extra.lift_value + dt*.22*self.lift_direction
            if value >= .9 or value <= .1:
                self.lift_direction *= -1.
            self.g.extra.set_lift(max(.1, min(.9, value)))

    def shutdown(self):
        self.lift_cycle = False
