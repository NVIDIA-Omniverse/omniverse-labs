# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clips, capture messages and recordings beside the central analysis pages."""
import math

import omni.ui as ui

from .splitter import SplitGrip
from .workspace_tabs import WorkspaceTabs


TABS=(("clips","Clips"),("messages","Messages"),("recordings","Recordings"))


class ActivityPanel(WorkspaceTabs):
    def __init__(self,window,on_activate,on_dispose,settings,*,on_save=None):
        self.settings=settings
        self.on_save=on_save if on_save is not None else settings.set
        saved=settings.get("soft_glass_activity_height",240)
        self.saved_height=float(saved) if isinstance(saved,(int,float)) and not isinstance(saved,bool) and math.isfinite(saved) else 240.
        self.saved_height=max(160,min(420,self.saved_height))
        self.expanded=settings.get("soft_glass_activity_open",False) is True
        self.pending_expanded=None
        initial="clips"
        self.container=ui.VStack(height=self.saved_height if self.expanded else 40,spacing=0)
        with self.container:
            self.grip_frame=ui.Frame(height=8,visible=self.expanded)
            super().__init__(window,on_activate,on_dispose,tabs=TABS,initial=initial)
        self.deck.visible=self.expanded
        self.pages['clips'].enabled=self.pages['recordings'].enabled=True
        self.pages['messages'].reason='Open an OVD capture to read its messages'
        with self.grip_frame:
            self.grip=SplitGrip(self.container,self._resized,axis="y",direction=-1,minimum=160,maximum=420)
        if self.expanded:self.pending=self.active

    def _build_bar(self):
        with ui.HStack(height=40,spacing=4):
            super()._build_bar()
            ui.Button("Hide" if self.expanded else "Show",width=48,height=28,name="workspace_tab",
                explicit_hover=True,tooltip="Collapse the bottom workspace" if self.expanded else "Expand the bottom workspace",
                clicked_fn=lambda:self.set_expanded(not self.expanded))

    def select(self,key):
        selected=super().select(key)
        if selected:self.pending_expanded=True
        return selected

    def set_expanded(self,expanded):
        self.pending_expanded=bool(expanded)
        if expanded and self.pages[self.active].view is None:self.pending=self.active

    def _resized(self,height):
        self.saved_height=height
        self.on_save("soft_glass_activity_height",height)

    def available(self,key,enabled,reason=""):
        page=self.pages[key]
        page.enabled,page.reason=enabled,reason
        button=self.buttons.get(key)
        if button is not None:button.enabled,button.tooltip=enabled,page.tooltip
        if not enabled and self.active==key:
            self.pending="clips"

    def poll(self):
        # Leave at least 220 logical pixels for the central workspace.
        self.grip.maximum=max(160,min(420,float(self.window.height)-420))
        self.grip.minimum=min(self.grip.maximum,max(160,getattr(self.pages[self.active].host,"minimum_height",160)))
        self.grip.update()
        if self.pending_expanded is not None:
            self.expanded,self.pending_expanded=self.pending_expanded,None
            self.on_save("soft_glass_activity_open",self.expanded)
            if not self.expanded:self.blur()
            self.grip_frame.visible=self.deck.visible=self.expanded
            self.bar.rebuild()
        height=max(self.grip.minimum,min(self.grip.maximum,self.saved_height)) if self.expanded else 40
        if self.grip.anchor is None and not self.grip.awaiting_layout and float(self.container.height)!=height:
            self.container.height=ui.Pixel(height)
        super().poll()

    def get_inspector_state(self):
        return {**super().get_inspector_state(),"expanded":self.expanded,
                "height":self.container.computed_height,"saved_height":self.saved_height,
                "divider_active":self.grip.anchor is not None}

    def destroy(self):
        self.grip.destroy()
        super().destroy()
