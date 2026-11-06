# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVRTX LiDAR configuration and presentation of measured returns."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdRender

MOUNT_PATH = '/World/LidarDisplay/Robot/SensorFrame'
SENSOR_PATH = '/World/LabLidarSensor'
PRODUCT = '/Render/Lidar'
VARIABLE = PRODUCT+'/PointCloud'


def author(stage):
    sensor = stage.DefinePrim(SENSOR_PATH, 'OmniLidar')
    sensor.AddAppliedSchema('OmniSensorGenericLidarCoreAPI')
    # Publish the visible robot mount's world pose to a root-level sensor, as
    # in the SDK example. The typed mount also supports USD-only authoring tools.
    xform=UsdGeom.Xformable(sensor)
    xform.AddTransformOp(opSuffix='sensorPose').Set(Gf.Matrix4d(1))
    # Full snapshots make the spatial relationships legible while the visitor
    # pauses or moves the robot. This is an illustrative profile, not a claim
    # to model a manufacturer's timing, calibration or motion distortion.
    for name, kind, value in (
        ('elementsCoordsType', Sdf.ValueTypeNames.Token, 'CARTESIAN'),
        ('outputFrameOfReference', Sdf.ValueTypeNames.Token, 'WORLD'),
        ('includeInvalidPoints', Sdf.ValueTypeNames.Bool, False),
        ('instantLidar', Sdf.ValueTypeNames.Bool, True),
        ('rotPatternFiringResDeg', Sdf.ValueTypeNames.Float, 1.),
        ('farRangeM', Sdf.ValueTypeNames.Float, 12.),
        ('nearRangeM', Sdf.ValueTypeNames.Float, .3),
        ('maxReturns', Sdf.ValueTypeNames.UInt, 1),
    ):
        sensor.CreateAttribute('omni:sensor:Core:'+name, kind).Set(value)
    sensor.CreateAttribute('omni:sensor:frameRate', Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(30,1))
    sensor.CreateAttribute('omni:sensor:startTime', Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(0,1))
    product = UsdRender.Product.Define(stage, PRODUCT)
    product.CreateCameraRel().SetTargets([sensor.GetPath()])
    var = UsdRender.Var.Define(stage, VARIABLE)
    var.CreateSourceNameAttr('PointCloud')
    var.GetPrim().CreateAttribute('channels', Sdf.ValueTypeNames.TokenArray).Set(
        ['Coordinates', 'Intensity', 'Counts', 'TimeOffsetNs'])
    product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
    return product


class LidarReadout:
    def __init__(self, center):
        from PIL import ImageFont
        self.font = ImageFont.load_default(size=20)
        self.center = np.asarray(center)
        self.points = np.empty((0,3), np.float32)
        self.intensity = np.empty(0, np.float32)
        self.offsets = np.empty(0, np.uint32)
        self.origin = np.zeros(3)
        self.native_origin = None
        self.color = 'distance'
        self.view = '3d'
        self.frames = 0
        self.capacity = 0
        self.pixels = np.zeros((360,640,4), np.uint8)
        self.pixels[:,:,:3] = (12,21,30)
        self.pixels[:,:,3] = 255

    def read(self, frame, device, draw=True):
        with frame.render_vars[VARIABLE].map(device=device) as cloud:
            count = int(np.from_dlpack(cloud['Counts'])[0])
            coordinates = np.from_dlpack(cloud['Coordinates'])
            self.capacity = coordinates.shape[1]
            if not 0 <= count <= self.capacity:
                raise RuntimeError('OVRTX returned an invalid LiDAR point count')
            # Only Counts entries are measurements; the remaining tensor is
            # unused capacity and must never be displayed as sensor returns.
            self.points = coordinates[:, :count].T.copy()
            self.intensity = np.from_dlpack(cloud['Intensity'])[:count].copy()
            self.offsets = np.from_dlpack(cloud['TimeOffsetNs'])[:count].copy()
            self.native_origin = np.from_dlpack(cloud.params['frameEndPosM']).copy()
        # Use the native scan's pose for coloring and its marker, rather than
        # the robot's latest pose (which may have moved during GPU processing).
        self.origin = self.native_origin.copy()
        self.frames += 1
        return self.draw() if draw else None

    def clear(self, origin):
        self.points=np.empty((0,3),np.float32)
        self.intensity=np.empty(0,np.float32)
        self.offsets=np.empty(0,np.uint32)
        self.origin=np.asarray(origin)
        self.native_origin=None
        self.frames=0
        self.draw()

    def draw(self):
        from PIL import Image, ImageDraw
        font = self.font
        image = Image.new('RGBA', (640,360), (12,21,30,255))
        draw = ImageDraw.Draw(image)
        # This projection only draws the library's measured XYZ coordinates.
        # No scene geometry, substitute rays or synthetic returns are sampled.
        def project(points):
            p = np.asarray(points)-self.center
            if self.view == 'top':
                return np.column_stack((320+p[:,0]*32, 185+p[:,2]*32))
            return np.column_stack((320+(p[:,0]*.86-p[:,2]*.5)*30,
                                    245+(p[:,0]*.22+p[:,2]*.38-p[:,1]*.90)*30))
        for step in range(-6,7,2):
            for a,b in (((step,0,-6),(step,0,6)), ((-6,0,step),(6,0,step))):
                projected = project(np.asarray([a,b])+self.center)
                draw.line([tuple(p) for p in projected], fill=(31,47,58), width=1)
        pixels = np.array(image)
        if len(self.points):
            valid = np.isfinite(self.points).all(axis=1)
            points = self.points[valid]
            points = points[np.max(np.abs(points-self.center),axis=1)<14.]
            xy = np.rint(project(points)).astype(np.int32)
            keep = (xy[:,0]>1)&(xy[:,0]<638)&(xy[:,1]>24)&(xy[:,1]<338)
            xy, points = xy[keep], points[keep]
            values = (points[:,1]/3.) if self.color == 'height' else np.linalg.norm(points-self.origin,axis=1)/12.
            values = np.clip(values,0,1)
            colors = np.column_stack((52+203*values, 212-49*values, 255-196*values)).astype(np.uint8)
            for dx,dy in ((0,0),(1,0),(0,1)):
                pixels[xy[:,1]+dy,xy[:,0]+dx,:3] = colors
        image=Image.fromarray(pixels);draw=ImageDraw.Draw(image)
        x,y = project(self.origin[None,:])[0]
        draw.ellipse((x-5,y-5,x+5,y+5), fill=(255,255,255), outline=(12,21,30), width=2)
        draw.text((14,8), 'MEASURED POINT CLOUD   /   '+('TOP VIEW' if self.view=='top' else '3D VIEW'), fill=(204,222,232),font=font)
        draw.text((14,339), '2 m grid / white = sensor', fill=(163,184,198),font=font)
        draw.text((355,339), 'CYAN near > AMBER far' if self.color=='distance' else 'CYAN floor > AMBER high', fill=(163,184,198),font=font)
        self.pixels=np.array(image)
        return self.pixels
