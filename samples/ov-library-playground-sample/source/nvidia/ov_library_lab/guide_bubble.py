# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Screen placement of the guide's ovui speech bubbles, in logical points."""
import math


def project(point, camera, viewport, aspect):
    """Project through the presented camera, respecting image letterboxing."""
    x,z,yaw,pitch,height,focal,aperture = camera
    vx,vy,vw,vh = viewport
    width = min(vw,vh*aspect)
    height_px = width/aspect
    left,top = vx+(vw-width)/2,vy+(vh-height_px)/2
    dx,dy,dz = point[0]-x,point[1]-height,point[2]-z
    sy,cy,sp,cp = math.sin(yaw),math.cos(yaw),math.sin(pitch),math.cos(pitch)
    depth = dx*sy*cp+dy*sp-dz*cy*cp
    if depth <= .05 or width <= 0:return None
    scale = width*focal/(aperture*depth)
    px = left+width/2+(dx*cy+dz*sy)*scale
    py = top+height_px/2+ (dx*sy*sp-dy*cp-dz*cy*sp)*scale
    if not (left < px < left+width and top < py < top+height_px):return None
    return px,py,scale,(left,top,width,height_px)


def placement(anchor, width, height, app_width, app_height):
    """Choose the clearer side; keep the bubble and its tail on screen."""
    px,py,scale,rect = anchor
    left,top,vw,vh = rect
    # The tip points to the edge of the guide's head, not its face.
    radius = .38*scale
    margin = 12.
    right = px+radius+width+margin <= app_width
    x = px+radius if right else px-radius-width
    x = max(margin,min(app_width-width-margin,x))
    y = max(top+8,min(top+vh-height-8,py-height*.38))
    y = max(margin,min(app_height-height-margin,y))
    tail_y = max(18,min(height-42,py-y-12))
    return round(x),round(y),right,round(tail_y)


class SpeechBubble:
    def __init__(self, name, style, width, height):
        import omni.ui as ui
        from .style import BG, GREEN
        flags = (ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE |
                 ui.WINDOW_FLAGS_NO_MOVE | ui.WINDOW_FLAGS_NO_SCROLLBAR |
                 ui.WINDOW_FLAGS_NO_BACKGROUND | ui.WINDOW_FLAGS_NO_DOCKING |
                 ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
        self.window = ui.Window(name,width=width,height=height,visible=False,
                               flags=flags,padding_x=0,padding_y=0)
        self.window.frame.style = style
        self.layout = None
        with self.window.frame:
            with ui.ZStack():
                self.body = ui.Placer(offset_x=18,width=width-18,height=height)
                with self.body:
                    with ui.ZStack():
                        ui.Rectangle(style={'background_color':BG,'border_radius':12,
                            'border_color':GREEN,'border_width':1})
                        with ui.HStack():
                            ui.Spacer(width=16)
                            with ui.VStack():
                                ui.Spacer(height=12)
                                self.content = ui.Frame()
                                ui.Spacer(height=12)
                            ui.Spacer(width=16)
                self.tail = ui.Placer(offset_x=0,offset_y=25,width=20,height=24)
                with self.tail:
                    self.triangle = ui.Triangle(alignment=ui.Alignment.LEFT_CENTER,
                        style={'background_color':GREEN})

    def place(self, anchor, width, height, app_width, app_height):
        import omni.ui as ui
        x,y,right,tail_y = placement(anchor,width,height,app_width,app_height)
        layout = (x,y,right,tail_y,width,height)
        if layout != self.layout:
            self.layout = layout
            self.window.position_x,self.window.position_y = x,y
            self.window.width,self.window.height = width,height
            self.body.offset_x = 18 if right else 0
            self.body.width,self.body.height = ui.Pixel(width-18),ui.Pixel(height)
            self.tail.offset_x = 0 if right else width-20
            self.tail.offset_y = tail_y
            self.triangle.alignment = ui.Alignment.LEFT_CENTER if right else ui.Alignment.RIGHT_CENTER

    def destroy(self):
        self.window.destroy()
