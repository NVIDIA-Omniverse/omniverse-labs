# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Screen-sized OVUI controls anchored to equipment, without Kit viewport APIs."""
import omni.ui as ui
from .guide_bubble import project
from .style import BG, INK, MUTED, GREEN, scaled_style


class EquipmentCard:
    """Compact inspector with an attachment stem, contained in the viewport."""
    def __init__(self, scale):
        self.width, self.height = 310*scale, 218*scale
        self.stem_height = 24*scale
        flags = (ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE |
                 ui.WINDOW_FLAGS_NO_MOVE | ui.WINDOW_FLAGS_NO_SCROLLBAR |
                 ui.WINDOW_FLAGS_NO_BACKGROUND | ui.WINDOW_FLAGS_NO_DOCKING |
                 ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
        self.window=ui.Window('Equipment controls',width=self.width,height=self.height,
                              visible=False,flags=flags,padding_x=0,padding_y=0)
        self.window.frame.style=scaled_style(scale)
        self.layout=None
        with self.window.frame:
            with ui.VStack(spacing=0):
                with ui.ZStack(height=self.height-self.stem_height):
                    ui.Rectangle(style={'background_color':BG,'border_radius':7,
                                        'border_color':MUTED,'border_width':1})
                    with ui.HStack():
                        ui.Spacer(width=14*scale)
                        with ui.VStack():
                            ui.Spacer(height=10*scale)
                            self.content=ui.Frame()
                            ui.Spacer(height=10*scale)
                        ui.Spacer(width=14*scale)
                with ui.ZStack(height=self.stem_height):
                    self.stem=ui.Placer(offset_x=self.width/2,width=2,height=self.stem_height)
                    with self.stem:
                        ui.Rectangle(style={'background_color':GREEN})
                    self.tip=ui.Placer(offset_x=self.width/2-3,offset_y=self.stem_height-6,width=8,height=8)
                    with self.tip:
                        ui.Circle(style={'background_color':GREEN})

    def place(self, anchor):
        px,py,_,(left,top,width,height)=anchor
        # Attach to a fixed mast-height point above the load's entire travel.
        # Fork motion never bobs the control under the pointer.
        x=max(left+8,min(left+width-self.width-8,px-self.width/2))
        y=py-self.height
        if y<top+8 or y+self.height>top+height-8 or width<self.width+16:
            return False
        layout=(round(x),round(y),round(px-x))
        if layout!=self.layout:
            self.layout=layout
            self.window.position_x,self.window.position_y=layout[:2]
            self.stem.offset_x=layout[2]
            self.tip.offset_x=layout[2]-3
        return True

    def destroy(self):
        self.window.destroy()


class EquipmentPanel:
    def __init__(self, lab):
        self.lab=lab
        self.bubble=None  # Stable lifetime contract for close/focus checks.
        self.scale=None
        self.height_label=None

    def hide(self):
        self.lab.panel_modes['interface']='operate'
        self.lab.request_panel()
        self.lab._clear_input()
        if self.bubble:self.bubble.window.visible=False

    def update(self):
        lab,view=self.lab,self.lab.viewport
        visible=(lab.current_room=='interface' and lab.panel_modes.get('interface')=='anchored'
                 and not getattr(lab,'showing_room_info',False)
                 and view.camera_pose and not (lab.settings_window and lab.settings_window.visible))
        anchor=None
        if visible:
            w=view.widget
            rect=(w.screen_position_x,w.screen_position_y,w.computed_width,w.computed_height)
            size=view.presented_size or (1280,836)
            anchor=project(lab.gallery.room_position('interface',(0,2.65,-5.5)),view.camera_pose,rect,size[0]/size[1])
        if not anchor:
            if self.bubble:self.bubble.window.visible=False
            return
        scale=lab.settings.text_scale
        if self.scale!=scale:
            if self.bubble:self.bubble.destroy()
            self.bubble=EquipmentCard(scale)
            self.scale=scale
            window=self.bubble.window
            window.set_focused_changed_fn(lambda focused:lab._clear_input() if focused else None)
            window.set_key_pressed_fn(lambda key,mods,pressed:self.hide() if key==256 and pressed else lab._on_ui_key(key,mods,pressed))
            with self.bubble.content:
                with ui.VStack(spacing=7*scale):
                    with ui.HStack(height=25*scale):
                        ui.Label('Pallet stacker' if lab.gallery.factory.simready.assets else 'Lift table',style={'font_size':17*scale,'color':INK})
                        self.close_button=ui.Button('Close',name='Utility',width=48*scale,clicked_fn=self.hide)
                    with ui.HStack(height=20*scale):
                        ui.Label('FORK HEIGHT',style={'font_size':11*scale,'color':MUTED})
                        self.height_label=ui.Label('',alignment=ui.Alignment.RIGHT_CENTER,width=55*scale,
                                                   style={'font_size':14*scale,'color':GREEN})
                    self.slider=ui.FloatSlider(lab.fork_model,min=0.,max=1.,height=28*scale,format='%.2f')
                    with ui.HStack(height=32*scale,spacing=7*scale):
                        self.lower_button=ui.Button('Lower',clicked_fn=lambda:lab.fork_model.set_value(.15))
                        self.raise_button=ui.Button('Raise',clicked_fn=lambda:lab.fork_model.set_value(.85))
                    ui.Label('Linked to the sidebar controls',height=18*scale,
                             style={'font_size':11*scale,'color':MUTED})
        self.height_label.text=f'{lab.fork_model.as_float*100:.0f}%'
        self.bubble.window.visible=self.bubble.place(anchor)

    def close(self):
        if self.bubble:
            self.bubble.destroy()
            self.bubble=None
            self.scale=None
