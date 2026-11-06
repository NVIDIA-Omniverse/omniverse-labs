# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render native LiDAR and two fisheye cameras from one scene revision."""
import numpy as np
from _common import output_folder, run_example


def main():
    import ovrtx, ovstage
    output = output_folder()
    renderer = ovrtx.Renderer(ovrtx.RendererConfig(motion_bvh=ovrtx.MotionBvh.ENABLE))
    stage = ovstage.Stage('lab.lidar.example')
    renderer.attach_ovstage(stage)
    try:
        from pxr import Usd, UsdGeom, UsdLux, Gf
        from nvidia.ov_library_lab.lidar_sensor import author, PRODUCT, VARIABLE, SENSOR_PATH
        from nvidia.ov_library_lab.robot_cameras import author as author_cameras, CAMERAS
        from PIL import Image
        usd = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(usd,'Y')
        UsdGeom.SetStageMetersPerUnit(usd,1.)
        UsdGeom.Xform.Define(usd,'/World')
        author(usd)  # OmniLidar + PointCloud with XYZ, intensity, counts and time.
        for pose,local in author_cameras(usd).values():
            pose.Set(local)  # A moving robot uses local * robot_to_world here.
        usd.GetPrimAtPath(SENSOR_PATH).GetAttribute('xformOp:transform:sensorPose').Set(
            Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(.86,1.12,0)))
        ground = UsdGeom.Cube.Define(usd,'/World/Ground')
        ground.AddScaleOp().Set(Gf.Vec3f(10,.01,10))
        target = UsdGeom.Cube.Define(usd,'/World/Target')
        target.AddTranslateOp().Set(Gf.Vec3d(0,1,-4))
        for name,x in (('Ahead',4),('Behind',-4)):
            target=UsdGeom.Cube.Define(usd,'/World/'+name)
            target.AddTranslateOp().Set(Gf.Vec3d(x,1,0))
        UsdLux.DomeLight.Define(usd,'/World/Light').CreateIntensityAttr(600.)
        ovstage.population.open_usd_from_string(stage,usd.GetRootLayer().ExportToString(),ordinal=1)
        stage.advance_write_floor(1).wait()
        for _ in range(4):
            # One step, one scene ordinal, three sensor render products.
            results = renderer.step(render_products={PRODUCT,*(c[1] for c in CAMERAS.values())},delta_time=1/60,ordinal=1)
        with results[PRODUCT].frames[-1].render_vars[VARIABLE].map(device=ovrtx.Device.CPU) as cloud:
            # Capacity can exceed the number of valid returns. Counts is the
            # authority; unused coordinates must not become displayed points.
            count = int(np.from_dlpack(cloud['Counts'])[0])
            xyz = np.from_dlpack(cloud['Coordinates'])[:,:count].T.copy()
            intensity = np.from_dlpack(cloud['Intensity'])[:count].copy()
            times = np.from_dlpack(cloud['TimeOffsetNs'])[:count].copy()
        assert count > 0, 'No returns; inspect the sensor pose and native RTX log.'
        np.savez(output/'lidar-points.npz',xyz_metres=xyz,intensity=intensity,time_offset_ns=times)
        for name,(_,product,_,_) in CAMERAS.items():
            with results[product].frames[-1].render_vars[product+'/LdrColor'].map(device=ovrtx.Device.CPU) as image:
                pixels=np.from_dlpack(image).copy()
            del image  # Release the native mapping; pixels owns its copy.
            Image.fromarray(pixels).save(output/('fisheye-'+name+'.png'))
        print(f'Saved {count:,} valid native LiDAR returns in world coordinates.')
        print('Saved front and rear native fisheye images from the same scene ordinal.')
        del cloud, results
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


if __name__ == '__main__':
    run_example(main)
