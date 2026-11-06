# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two mounted, calibrated fisheye render products in the native sensor scene."""
from pxr import Gf, Sdf, UsdGeom, UsdRender

RESOLUTION = (480, 270)
# The robot's fixed forward direction is +X. It drives backwards on the return
# leg; camera names describe their mounting, not instantaneous travel direction.
CAMERAS = {
    'front': ('/World/RobotFrontCamera', '/Render/RobotFront', (.934, .94, 0.), (1., 0., 0.)),
    'rear': ('/World/RobotRearCamera', '/Render/RobotRear', (.466, .94, 0.), (-1., 0., 0.)),
}


def author(stage, campus=False):
    poses = {}
    for name, (path, product_path, eye, direction) in CAMERAS.items():
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(18.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.025, 250.))
        if campus:
            from .campus import expose
            expose(camera.GetPrim())
        prim = camera.GetPrim()
        prim.AddAppliedSchema('OmniLensDistortionOpenCvFisheyeAPI')
        prim.CreateAttribute('omni:lensdistortion:model', Sdf.ValueTypeNames.Token).Set('opencvFisheye')
        for key, value in (('fx',165.), ('fy',165.), ('cx',240.), ('cy',135.),
                           ('k1',0.), ('k2',0.), ('k3',0.), ('k4',0.)):
            prim.CreateAttribute('omni:lensdistortion:opencvFisheye:'+key, Sdf.ValueTypeNames.Float).Set(value)
        prim.CreateAttribute('omni:lensdistortion:opencvFisheye:imageSize', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*RESOLUTION))
        local = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*eye)+Gf.Vec3d(*direction), Gf.Vec3d(0,1,0)).GetInverse()
        poses[name] = (camera.AddTransformOp(opSuffix='mountedPose'), local)
        product = UsdRender.Product.Define(stage, product_path)
        product.CreateCameraRel().SetTargets([path])
        product.CreateResolutionAttr(Gf.Vec2i(*RESOLUTION))
        settings = product.GetPrim()
        settings.AddAppliedSchema('OmniRtxDebugSettingsAPI_1')
        settings.CreateAttribute('omni:rtx:rendermode', Sdf.ValueTypeNames.Token).Set('RealTimePathTracing')
        settings.CreateAttribute('omni:rtx:rtpt:maxBounces', Sdf.ValueTypeNames.UInt).Set(4)
        settings.CreateAttribute('omni:rtx:rtpt:extraSpecularAndTransmissiveBounces', Sdf.ValueTypeNames.UInt).Set(4)
        var = UsdRender.Var.Define(stage, product_path+'/LdrColor')
        var.CreateSourceNameAttr('LdrColor')
        product.CreateOrderedVarsRel().SetTargets([var.GetPath()])
    return poses
