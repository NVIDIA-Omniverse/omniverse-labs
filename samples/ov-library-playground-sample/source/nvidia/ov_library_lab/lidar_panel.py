# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visitor controls for the robot's native RTX LiDAR measurement."""
import omni.ui as ui
import time
from .native_view import NativeView
from .style import MUTED


def build(lab, accent):
    room = lab.gallery.workshops.lidar
    cloud = room.readout
    lab._label('Live LiDAR and front/rear fisheye views from the robot.',14,MUTED)
    lab.observer_view = NativeView(lab,'Lidar',height=180)
    lab.observer_view.update(cloud.draw())
    lab.robot_camera_views={}
    with ui.HStack(height=128,spacing=8):
        for name,label in (('front','FRONT / FISHEYE'),('rear','REAR / FISHEYE')):
            with ui.VStack(spacing=4):
                lab._label(label,11,accent,height=20)
                view=NativeView(lab,'RobotCamera_'+name,height=104)
                lab.robot_camera_views[name]=view
                if name in room.camera_pixels:view.update(room.camera_pixels[name])
    lab._section('TRY IT / ALL THREE VIEWS RESPOND')
    with ui.HStack(height=38,spacing=7):
        lab.lidar_motion = ui.Button('Pause robot' if room.moving else 'Resume robot',name='Primary',
            clicked_fn=lambda:setattr(room,'moving',not room.moving))
        lab.lidar_target = ui.Button('Move target closer',clicked_fn=room.toggle_target)
    lab._label("Pause the route, then move the pallet. Compare its camera appearance with the LiDAR points it returns.",14,MUTED)
    lab.stats = lab._label('',13,accent)
    lab.lidar_choices = {}
    def display(attribute, value):
        setattr(cloud,attribute,value)
        lab.observer_view.update(cloud.draw())
    with ui.CollapsableFrame('Point-cloud display / optional',collapsed=True) as lab.lidar_display_options:
        with ui.VStack(spacing=5):
            with ui.HStack(height=33,spacing=6):
                for value,label in (('distance','Distance'),('height','Height')):
                    lab.lidar_choices[value] = lab.choice(label,lambda:cloud.color,value,lambda v=value:display('color',v))
            with ui.HStack(height=33,spacing=6):
                for value,label in (('3d','3D'),('top','Top')):
                    lab.lidar_choices[value] = lab.choice(label,lambda:cloud.view,value,lambda v=value:display('view',v))
    def retry():
        lab.lidar_engine.start()
        lab.observer_view.update(cloud.draw())
    lab.lidar_retry = ui.Button('Retry sensors',height=32,visible=False,clicked_fn=retry)


def refresh(lab):
    room=lab.gallery.workshops.lidar
    cloud=room.readout
    engine=lab.lidar_engine
    lab.lidar_motion.text='Pause robot' if room.moving else 'Resume robot'
    lab.lidar_target.text='Move target back' if room.target_near else 'Move target closer'
    lab.lidar_retry.visible=bool(engine.error)
    if engine.error:
        lab.stats.text='The robot sensors stopped. Select Retry sensors to reconnect. You can keep exploring the lab.'
        return
    if not engine.ready:
        elapsed = max(0, int(time.monotonic()-getattr(engine, 'started', time.monotonic())))
        lab.stats.text = (f'Preparing LiDAR and cameras / {elapsed//60}:{elapsed%60:02d}\n'
                         'Shader preparation can take a moment.')
        return
    lab.stats.text=(f'{len(cloud.points):,} valid returns  |  12 m sensor range\n'
                    f"{'Robot moving' if room.moving else 'Robot paused / sensors live'}  |  LiDAR + 2 cameras") if cloud.frames else 'Preparing the robot sensor views...'
