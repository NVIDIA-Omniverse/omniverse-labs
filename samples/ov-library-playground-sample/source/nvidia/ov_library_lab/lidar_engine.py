# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One outstanding scan, native worker isolation, and responsive UI delivery."""
import base64
from pathlib import Path
import tempfile
import time
import numpy as np
from .library_worker import LibraryWorker


class LidarEngine:
    def __init__(self, room, runtime_root):
        self.room=room
        self.runtime_root=Path(runtime_root)
        self.worker=None
        self.scene_file=None
        self.pending=False
        self.ready=False
        self.error=None
        self.active=False
        self.max_frames_per_step=0
        self.next_scan=0.
        self.start()

    def start(self):
        self.close()
        self.error=None;self.ready=False;self.active=False;self.next_scan=0.
        self.room.readout.clear(self.room.sensor_position())
        self.room.camera_pixels.clear();self.room.camera_capture.clear()
        self.started=time.monotonic()
        try:
            with tempfile.NamedTemporaryFile(prefix='lidar-scene-',suffix='.usda',dir=self.runtime_root,delete=False) as file:
                self.scene_file=Path(file.name)
            self.room.stage.GetRootLayer().Export(str(self.scene_file))
            self.worker=LibraryWorker(self.runtime_root,'lidar',environment='libraries')
            self.worker.submit(initialize=True,scene=str(self.scene_file),
                               center=self.room.readout.center.tolist(),**self.room.scan_state())
            self.pending=True
        except Exception as exc:
            # This optional exhibit must not prevent entry to the rest of the lab.
            self.error=str(exc) or type(exc).__name__
            self.close()

    def poll(self):
        if not self.worker:return None
        result=self.worker.poll()
        if result is None and self.pending and time.monotonic()-self.started>(180. if self.ready else 600.):
            self.error='Sensor initialization or scan timed out. Retry the sensor.'
            self.pending=False
            self.worker.close()
        if not result:return None
        self.pending=False
        if result.get('error'):
            self.error=result['error'];return None
        self.ready=True
        cloud=self.room.readout
        decode=lambda name,dtype:np.frombuffer(base64.b64decode(result[name]),dtype=dtype).copy()
        cloud.points=decode('points',np.float32).reshape(-1,3)
        cloud.intensity=decode('intensity',np.float32)
        cloud.offsets=decode('offsets',np.uint32)
        cloud.capacity=result['capacity']
        cloud.native_origin=np.asarray(result['origin'])
        cloud.origin=cloud.native_origin.copy()
        cloud.frames+=1
        from .robot_cameras import RESOLUTION
        images={}
        for name in ('front','rear'):
            camera=result['cameras'][name]
            if camera['resolution']!=list(RESOLUTION):
                self.error='Unexpected robot camera resolution';return None
            images[name]=np.frombuffer(base64.b64decode(camera['pixels']),dtype=np.uint8).reshape(RESOLUTION[1],RESOLUTION[0],4).copy()
        self.room.camera_pixels=images
        self.room.camera_capture={'ordinal':result['ordinal'],
            'poses':{name:camera['camera_to_world'] for name,camera in result['cameras'].items()}}
        self.max_frames_per_step=max(self.max_frames_per_step,result['frame_count'])
        return cloud.draw()

    def update(self, active, synchronous=False):
        self.active=bool(active)
        pixels=self.poll()
        if not active or not self.ready or self.error:return pixels
        if pixels is not None and not synchronous:
            # Present the completed set before starting more sensor GPU work.
            # The next UI tick may submit; never build a queue of stale scans.
            return pixels
        if synchronous:
            # Used only by deterministic validation/capture, never desktop input.
            while self.pending:
                self.poll();time.sleep(.001)
            if self.error:return pixels
        if not self.pending and (synchronous or time.monotonic()>=self.next_scan):
            self.worker.submit(**self.room.scan_state())
            self.pending=True;self.started=time.monotonic();self.next_scan=self.started+.10
        if synchronous:
            while self.pending:
                result=self.poll()
                if result is not None:pixels=result
                if self.pending:time.sleep(.001)
        return pixels

    def close(self):
        if self.worker:
            self.worker.close();self.worker=None
        self.pending=False
        if self.scene_file:
            self.scene_file.unlink(missing_ok=True);self.scene_file=None
