# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera actions and navigation settings using the shared camera controller."""
import math

import numpy as np
import omni.ui as ui

from ..camera import MovementMode, near_plane_from_slider, near_plane_slider_position
from ..visual_reset import NEAR, NEAR_SLIDER
from ..style import layout as fl, style_dialog


class CameraView:
    UP_AXES=((0,1,0),(0,0,1),(1,0,0),(0,-1,0),(0,0,-1),(-1,0,0))

    def __init__(self,controller,on_changed,*,frame_all,frame_selected,center_selected,track_selected,reset):
        self.controller,self.on_changed=controller,on_changed
        self.actions=(("Zoom scene",frame_all),("Zoom selection",frame_selected),
                      ("Center selection",center_selected),("Track selection",track_selected),("Reset",reset))
        self.window=None
        self._syncing_near=False
        self._status_error=False

    def destroy(self):
        if self.window:self.window.destroy();self.window=None

    def show(self):
        if self.window is None:self._build()
        self.window.visible=True
        self._sync()

    def _sync(self):
        control=self.controller;camera=control.camera
        self.mode.model.get_item_value_model().set_value(int(control.mode))
        self.source.model.get_item_value_model().set_value(int(control.captured))
        self.fov.model.set_value(camera.fov)
        self._sync_near(camera.near)
        self.speed.model.set_value(control.speed_scale)
        self.handed.model.set_value(camera.left_handed)
        index=int(np.argmax(np.asarray(self.UP_AXES)@camera.world_up))
        self.up.model.get_item_value_model().set_value(index)
        for field,value in zip(self.coordinates,camera.mirror(camera.target)):field.model.set_value(float(value))
        self._status_error=False
        self.refresh_tracking()

    def refresh_tracking(self):
        """Reflect controller changes without replacing unfinished field drafts."""
        if self.window is None:return  # Destroyed windows retain Python child wrappers.
        control=self.controller
        if hasattr(self,"track_button"):
            label="Stop tracking" if control.tracking_id is not None else "Track selection"
            if self.track_button.text!=label:self.track_button.text=label
        if not hasattr(self,"status") or self._status_error:return
        status=(f"Recorded cameras: {len(control.available_cameras)} · Movement: {control.effective_speed:g} units/s"
                + (f" · Tracking {control.tracking_id}" if control.tracking_id is not None else ""))
        if self.status.text!=status:self.status.text=status

    def sync_visual_reset(self):
        """Update only accepted reset fields; keep all other drafts/errors."""
        if self.window is None:return
        control=self.controller;camera=control.camera
        self.mode.model.get_item_value_model().set_value(int(control.mode))
        self.up.model.get_item_value_model().set_value(0)
        self.handed.model.set_value(camera.left_handed)
        self.speed.model.set_value(control.speed_scale)
        self._sync_near(camera.near)
        self.refresh_tracking()

    def _action(self,callback):
        callback();self._sync()

    def _sync_near(self,value,position=None):
        # Inverting the measured original F32 default in Python doubles falls
        # just below57. Keep its observed control position on first open too.
        if position is None and value==NEAR:position=NEAR_SLIDER
        self._syncing_near=True
        try:
            self.near.model.set_value(value)
            self.near_slider.model.set_value(near_plane_slider_position(value) if position is None else position)
        finally:self._syncing_near=False

    def _near_field_changed(self,model):
        if self._syncing_near:return
        try:position=near_plane_slider_position(model.as_float)
        except ValueError:return  # Apply settings reports invalid numeric drafts.
        self._syncing_near=True
        try:self.near_slider.model.set_value(position)
        finally:self._syncing_near=False

    def _near_slider_changed(self,model):
        if self._syncing_near:return
        value=near_plane_from_slider(model.as_int)
        self.controller.camera.near=value
        self._sync_near(value,model.as_int)
        # Match Qt's live projection update without applying unrelated drafts
        # or changing the camera's recorded/tracking/navigation state.
        self.on_changed()

    def _apply(self):
        control=self.controller;camera=control.camera
        try:
            speed,near,fov=self.speed.model.as_float,self.near.model.as_float,self.fov.model.as_float
            if not all(math.isfinite(v) and v>0 for v in (speed,near,fov)):
                raise ValueError("Speed, near plane and field of view must be positive finite numbers.")
            captured=bool(self.source.model.get_item_value_model().as_int)
            if captured and not control.available_cameras:
                raise ValueError("This frame has no recorded Camera with Origin, Target and Up values.")
            control.manual_change()
            control.mode=MovementMode(self.mode.model.get_item_value_model().as_int)
            control.speed_scale=max(.001,min(1000.,speed))
            camera.near=near;camera.set_fov(fov)
            camera.set_world_up(self.UP_AXES[self.up.model.get_item_value_model().as_int])
            camera.left_handed=self.handed.model.as_bool
            control.captured=captured
            self.on_changed();self._sync()
        except ValueError as error:self._status_error=True;self.status.text=str(error)

    def _look_at(self):
        try:
            self.controller.manual_change()
            self.controller.camera.center_at([field.model.as_float for field in self.coordinates])
            self.on_changed();self._sync()
        except ValueError as error:self._status_error=True;self.status.text=str(error)

    def _build(self):
        self.window=ui.Window("Viewport camera",width=620,height=690,
            flags=ui.WINDOW_FLAGS_NO_DOCKING|ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_spacing)
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label("Viewport camera",name="section")
                            ui.Button("Close",name="quiet",width=80,clicked_fn=lambda:setattr(self.window,"visible",False))
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            for label,action in self.actions[:3]:
                                ui.Button(label,clicked_fn=lambda callback=action:self._action(callback))
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            for label,action in self.actions[3:]:
                                button=ui.Button(label,clicked_fn=lambda callback=action:self._action(callback),
                                    explicit_hover=True, tooltip=("Track the selected dynamic body. While tracking, this stops the current target; invoke again to track the current selection."
                                             if label=="Track selection" else ""))
                                if label=="Track selection":self.track_button=button
                        for title,attribute,choices in (("Movement","mode",("First person","Third person / world up")),
                                ("Camera","source",("Free camera","Recorded camera")),
                                ("World up","up",("+Y","+Z","+X","−Y","−Z","−X"))):
                            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                                ui.Label(title,width=160)
                                setattr(self,attribute,ui.ComboBox(0,*choices))
                        for title,attribute in (("Speed scale","speed"),("Field of view (degrees)","fov")):
                            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                                ui.Label(title,width=160)
                                setattr(self,attribute,ui.FloatField())
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            ui.Label("Near plane",width=160)
                            self.near_slider=ui.IntSlider(ui.SimpleIntModel(1,min=1,max=100),min=1,max=100,
                                identifier="camera_near_slider",
                                style={"Slider":{"color":0x00000000}},
                                explicit_hover=True, tooltip="Logarithmic near clipping distance: 0.0001–1000. Drag to update the viewport immediately.")
                            self.near=ui.FloatField(width=120,precision=8,
                                identifier="camera_near_value",
                                explicit_hover=True, tooltip="Precise positive finite distance. Apply settings commits typed values; the slider covers 0.0001–1000.")
                            self.near_slider.model.add_value_changed_fn(self._near_slider_changed)
                            self.near.model.add_value_changed_fn(self._near_field_changed)
                        with ui.HStack(height=fl.pvd_row,spacing=fl.pvd_spacing):
                            ui.Spacer(width=160)
                            ui.Label("0.0001",name="muted")
                            ui.Label("1000",name="muted",alignment=ui.Alignment.RIGHT_CENTER)
                            ui.Spacer(width=120)
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            self.handed=ui.CheckBox(width=24)
                            ui.Label("Left-handed scene")
                            ui.Button("Apply settings",name="primary",width=140,clicked_fn=self._apply)
                        ui.Separator(height=1)
                        ui.Label("Look at coordinates · Moves the orbit center",height=fl.pvd_row)
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            self.coordinates=[]
                            for axis in "XYZ":
                                ui.Label(axis,width=15)
                                self.coordinates.append(ui.FloatField())
                            ui.Button("Look at",width=85,clicked_fn=self._look_at)
                        ui.Label("WASD move · Q/E down/up · Arrows or IJKL look · Shift doubles speed",name="muted",height=fl.pvd_row)
                        ui.Label("Left drag orbits · Right drag looks · Middle drag pans · Wheel zooms",name="muted",height=fl.pvd_row)
                        ui.Label("Shift + wheel changes speed · +/− changes FOV · F frames selection",name="muted",height=fl.pvd_row)
                        self.status=ui.Label("",name="muted",height=fl.pvd_row*2,word_wrap=True)
                        ui.Spacer()
                    ui.Spacer(width=fl.pvd_padding)
