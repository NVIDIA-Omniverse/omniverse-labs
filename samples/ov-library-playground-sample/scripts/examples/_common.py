# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small shared scene for runnable examples; no gallery or ovui window."""
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source'))
from nvidia.ov_library_lab.bootstrap import configure
configure()
from nvidia.ov_library_lab.native_runtime import NativeRuntime


def run_example(callback):
    # The pinned Windows native stack needs the graphics host initialized before
    # attaching ovstage. Use the same standalone ovui host, with no visible window.
    os.environ['OMNIUI_HEADLESS'] = '1'
    os.environ['OMNIUI_BACKEND'] = 'vulkan'
    import asyncio
    import omni.ui as ui
    async def run():
        pump = asyncio.create_task(ui.run_async())
        try:
            await ui.next_frame()
            callback()
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
    ui.init('Native library example', width=640, height=420)
    try:
        asyncio.run(run())
    finally:
        ui.shutdown()


def scene():
    # Initialize RTX's USD integration before importing/authoring USD objects.
    runtime = NativeRuntime()
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Gf, Sdf, Plug
    Plug.Registry().RegisterPlugins(str(runtime.ovphysx.codeless_schema_root()))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Y')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, '/World')
    for path, position, dimensions in (('/World/Floor', (0, -.1, 0), (8, .2, 8)),
                                       ('/World/Carton', (0, 3, 0), (.6, .6, .6))):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    body = stage.GetPrimAtPath('/World/Carton')
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(2.)
    physics = UsdPhysics.Scene.Define(stage, '/World/Physics')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
    physics.CreateGravityMagnitudeAttr(9.81)
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(500.)
    key = UsdLux.RectLight.Define(stage, '/World/Key')
    key.CreateIntensityAttr(2500.)
    key.CreateWidthAttr(3.)
    key.CreateHeightAttr(3.)
    key.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(2, 6, 3), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 1, 0)).GetInverse())
    material = UsdShade.Material.Define(stage, '/World/Material')
    shader = UsdShade.Shader.Define(stage, '/World/Material/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.05, .45, .7))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.2)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(body).Bind(material)
    for name in ('TourCamera', 'InspectionCamera', 'ObserverCamera', 'StreamCamera'):
        camera = UsdGeom.Camera.Define(stage, '/World/'+name)
        camera.CreateFocalLengthAttr(32.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 100.))
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(5, 4, 7), Gf.Vec3d(0, 1.5, 0), Gf.Vec3d(0, 1, 0)).GetInverse())
    runtime.populate(stage)
    runtime.set_quality((640, 420), 4)
    return runtime, stage


def output_folder():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'.ov_library_lab_runtime/example-output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args.output
