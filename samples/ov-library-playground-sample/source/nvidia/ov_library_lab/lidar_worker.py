# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native OVRTX LiDAR; never enables sensor BVH in the walking renderer."""
import base64
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    def send(value):
        protocol.write(json.dumps(value,allow_nan=False)+'\n');protocol.flush()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from nvidia.ov_library_lab.bootstrap import configure
    configure()
    import numpy as np
    import ovrtx, ovstage
    renderer = stage = paths = query = handle = None
    try:
        # Renderer initialization registers the sensor schemas before population.
        renderer = ovrtx.Renderer(ovrtx.RendererConfig(motion_bvh=ovrtx.MotionBvh.ENABLE))
        send({'ready':True,'pid':os.getpid()})
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('close'):break
            try:
                from pxr import Sdf, Usd, UsdGeom
                from nvidia.ov_library_lab.lidar_sensor import author, LidarReadout, SENSOR_PATH, PRODUCT, VARIABLE
                from nvidia.ov_library_lab.robot_cameras import CAMERAS, RESOLUTION
                if request.get('initialize'):
                    print('LIDAR_PROGRESS: Composing the authored room and campus',file=sys.stderr,flush=True)
                    # Identical authored campus/room geometry and materials;
                    # exclude unrelated live experiments from the sensor scene.
                    source=Sdf.Layer.FindOrOpen(request['scene'])
                    print('LIDAR_PROGRESS: Snapshot opened; copying sensor-scene roots',file=sys.stderr,flush=True)
                    usd=Usd.Stage.CreateInMemory()
                    UsdGeom.SetStageUpAxis(usd,'Y');UsdGeom.SetStageMetersPerUnit(usd,1.)
                    UsdGeom.Xform.Define(usd,'/World')
                    # Copy authored specs, retaining source references and native
                    # instancing. Flattening the entire campus is unnecessary.
                    usd.DefinePrim('/Render')
                    for root in ('/World/Environment','/World/Building','/World/Looks', '/World/Lights',
                                 '/World/LidarDisplay', *(c[0] for c in CAMERAS.values()),
                                 *(c[1] for c in CAMERAS.values())):
                        if source.GetPrimAtPath(root):Sdf.CopySpec(source,root,usd.GetRootLayer(),root)
                    usd.GetRootLayer().subLayerPaths=list(source.subLayerPaths)
                    print('LIDAR_PROGRESS: Authoring sensor products',file=sys.stderr,flush=True)
                    author(usd)
                    UsdGeom.Imageable(usd.GetPrimAtPath('/World/LidarDisplay')).MakeVisible()
                    stage=ovstage.Stage('lab-lidar')
                    print('LIDAR_PROGRESS: Attaching the native sensor scene',file=sys.stderr,flush=True)
                    renderer.attach_ovstage(stage)
                    print('LIDAR_PROGRESS: Populating the native sensor scene',file=sys.stderr,flush=True)
                    ovstage.population.open_usd_from_string(stage,usd.GetRootLayer().ExportToString(),ordinal=1)
                    stage.advance_write_floor(1).wait()
                    paths=ovstage.PathDictionary(stage)
                    handle=paths.create_path_list_from_strings(request['paths'])
                    query=stage.query_from_path_list(handle)
                    readout=LidarReadout(request['center'])
                    ordinal=1
                    print('LIDAR_PROGRESS: Preparing the first native scan',file=sys.stderr,flush=True)
                ordinal+=1
                stage.write_attribute(query,'omni:xform',ordinal,np.asarray(request['matrices'],dtype=np.float64),
                                      is_array=False,semantic=ovstage.AttributeSemantic.MATRIX).wait()
                stage.write_attribute(query,'omni:resetXformStack',ordinal,np.asarray(request['reset'],dtype=np.bool_),is_array=False).wait()
                stage.advance_write_floor(ordinal).wait()
                # Only scan on request; idle time never accumulates stale scans.
                # All three products consume this exact pose/scene ordinal.
                outputs=renderer.step(render_products={PRODUCT,*(c[1] for c in CAMERAS.values())},
                                      delta_time=1/30,ordinal=ordinal)
                frames=outputs[PRODUCT].frames
                readout.read(frames[-1],ovrtx.Device.CPU,draw=False)
                encode=lambda array:base64.b64encode(np.ascontiguousarray(array).tobytes()).decode('ascii')
                cameras={}
                for name,(path,product,_,_) in CAMERAS.items():
                    with outputs[product].frames[-1].render_vars[product+'/LdrColor'].map(device=ovrtx.Device.CPU) as image:
                        pixels=np.from_dlpack(image).copy()
                    del image  # The copied pixels no longer need the mapping.
                    if pixels.shape != (RESOLUTION[1],RESOLUTION[0],4) or pixels.dtype != np.uint8:
                        raise RuntimeError('Unexpected native fisheye image shape or format')
                    cameras[name]={'pixels':encode(pixels),'resolution':list(RESOLUTION),
                                   'camera_to_world':request['matrices'][request['paths'].index(path)]}
                send({'id':request['id'],'points':encode(readout.points),'intensity':encode(readout.intensity),
                      'offsets':encode(readout.offsets),'origin':readout.native_origin.tolist(),
                      'capacity':readout.capacity,'frame_count':max(len(p.frames) for p in outputs.values()),
                      'ordinal':ordinal,'cameras':cameras})
                del outputs,frames
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                send({'id':request['id'],'error':str(exc)})
    finally:
        if query:query.release().wait()
        if handle is not None:paths.destroy_path_list(handle)
        if paths:paths.destroy()
        if renderer and stage:renderer.detach_ovstage()
        if stage:stage.destroy()
        if renderer:renderer.destroy()


if __name__=='__main__':main()
