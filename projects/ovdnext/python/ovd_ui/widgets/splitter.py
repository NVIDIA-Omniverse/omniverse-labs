# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared OVUI resize grip; call update() between completed UI frames.

Mouse callbacks run during widget drawing. Mutating a later sibling's width
there can lose its layout invalidation when that sibling is subsequently drawn.
Queue/coalesce input instead, then persist the actual width after layout.
"""
import omni.ui as ui
from ..style import layout as fl


class SplitGrip:
    def __init__(self,target,on_resize_end,*,direction=1,
                 minimum=fl.pvd_panel_min_width,maximum=fl.pvd_panel_max_width,axis="x"):
        if axis not in ("x","y"):raise ValueError("Splitter axis must be x or y")
        self.axis=axis
        self.target,self.on_resize_end=target,on_resize_end
        self.direction,self.minimum,self.maximum=direction,minimum,maximum
        self.anchor=None
        self.hovered=False
        self.pending_width=None
        self.awaiting_layout=False
        self.persist_pending=False
        self.closed=False
        extent={"width" if axis=="x" else "height":fl.pvd_splitter_hit_width}
        rule_extent={"width" if axis=="x" else "height":fl.pvd_splitter_rule_width}
        with ui.ZStack(**extent):
            with (ui.HStack if axis=="x" else ui.VStack)(spacing=0,opaque_for_mouse_events=False):
                ui.Spacer(opaque_for_mouse_events=False)
                self.rule=ui.Rectangle(name="splitter",**rule_extent,
                                       visible=False,opaque_for_mouse_events=False)
                ui.Spacer(opaque_for_mouse_events=False)
            self.widget=ui.InvisibleButton(tooltip="Drag to resize the panel",explicit_hover=True,
                mouse_pressed_fn=self.press,mouse_moved_fn=self.move,mouse_released_fn=self.release,
                mouse_hovered_fn=self.hover)

    def hover(self, hovered):
        if self.closed:return
        self.hovered=hovered
        self.rule.visible=bool(hovered or self.anchor)

    def press(self,x,y,button,modifiers):
        if not self.closed and button==0:
            # A new gesture starts from the currently shown panel. Cancel a
            # prior release still queued in this same UI frame, rather than
            # applying it after recording a different visible-width anchor.
            self.pending_width=None
            self.persist_pending=False
            self.anchor=(x,self.target.computed_width) if self.axis=="x" else (y,self.target.computed_height)
            self.rule.visible=True

    def move(self,x,y,*_):
        if not self.closed and self.anchor:
            origin,width=self.anchor
            position=x if self.axis=="x" else y
            desired=max(self.minimum,min(self.maximum,width+(position-origin)*self.direction))
            self.pending_width=desired

    def release(self,x,y,button,*_):
        if not self.closed and button==0 and self.anchor:
            self.move(x,y)
            self.anchor=None
            self.rule.visible=self.hovered
            self.persist_pending=True

    def update(self):
        """Apply outside draw; a later call observes the resulting UI layout."""
        if self.closed:return
        if self.pending_width is not None:
            setattr(self.target,"width" if self.axis=="x" else "height",ui.Pixel(self.pending_width))
            self.pending_width=None
            self.awaiting_layout=True
        elif self.awaiting_layout:
            self.awaiting_layout=False
            if self.persist_pending:
                self.persist_pending=False
                self.on_resize_end(int(self.target.computed_width if self.axis=="x" else self.target.computed_height))

    def destroy(self):
        """Cancel uncommitted input; retained callbacks cannot touch old widgets."""
        if self.closed:return
        self.closed=True
        self.anchor=self.pending_width=None
        self.awaiting_layout=self.persist_pending=False
        self.rule.visible=False
        self.target=self.on_resize_end=self.rule=self.widget=None
