# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact, scene-anchored ovui conversation for the atrium guide."""
import math
import omni.ui as ui
from .guide_scene import GuideScene
from .guide_bubble import SpeechBubble, project
from .guide_content import Conversation
from .keys import Key
from .style import GREEN, MUTED, scaled_style


class GuideView:
    def __init__(self, lab, asset):
        self.lab = lab
        self.scene = GuideScene(lab.stage,asset)
        self.window = self.prompt = None
        self.prompt_bubble = self.dialog_bubble = None
        self.pending = None
        self.near = False
        self.conversation = Conversation(lab.settings.allowed)
        self.pending_choice = None
        self.choice_buttons = {}
        self.anchor = None

    @property
    def active(self):
        return self.window is not None and self.window.visible

    @property
    def captures_input(self):
        return self.active or self.pending == 'open'

    def dialog_style(self):
        scale = self.lab.settings.text_scale
        style = scaled_style(scale)
        for name,size,color in (('Heading',14,GREEN),('AnswerTitle',18,None),
                                ('Answer',16,MUTED),('Caption',12,MUTED)):
            style['Label::'+name] = {'font_size':size*scale}
            if color is not None:style['Label::'+name]['color'] = color
        return style

    def build_prompt(self):
        scale = self.lab.settings.text_scale
        self.prompt_bubble = SpeechBubble('Meet the lab guide',self.dialog_style(),276*scale,122*scale)
        self.prompt = self.prompt_bubble.window
        self.prompt.set_key_pressed_fn(self.lab._on_ui_key)
        with self.prompt_bubble.content:
            with ui.VStack(spacing=6):
                ui.Label('YOUR LAB GUIDE',height=20*scale,name='Heading')
                ui.Label('Curious where to begin?',height=22*scale,name='Answer')
                self.prompt_button = ui.Button('Talk / E',height=36*scale,name='Primary',clicked_fn=self.request_open)

    def available(self):
        return (self.lab.current_room == 'lobby' and self.scene.motion.in_reach(self.lab.walker) and
                not (self.lab.settings_window and self.lab.settings_window.visible) and
                not (self.lab.overlay and self.lab.overlay.active))

    def request_open(self):
        if self.available() and not self.active:
            self.pending = 'open'
            self.lab._clear_input()

    def request_close(self):
        self.pending = 'close'

    def key(self, key, pressed):
        if self.captures_input:
            if pressed and key == Key.ESCAPE:self.request_close()
            self.lab._clear_input()
            return True
        if pressed and key == Key.E and self.available():
            self.request_open()
            return True
        return False

    def _open(self):
        if not self.available():return
        if self.dialog_bubble:self.dialog_bubble.destroy()
        self.lab._clear_input()
        self.lab.navigation_focus = False
        self.prompt.visible = False
        scale = self.lab.settings.text_scale
        self.dialog_bubble = SpeechBubble('Lab guide',self.dialog_style(),410*scale,450*scale)
        self.window = self.dialog_bubble.window
        self.window.flags &= ~ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING
        self.window.visible = True
        self.window.set_key_pressed_fn(self.lab._on_ui_key)
        self.conversation.home()
        self.pending_choice = None
        self.dialog_bubble.content.set_build_fn(self.build_conversation)

    def build_conversation(self):
        scale = self.lab.settings.text_scale
        current = self.conversation.current
        self.choice_buttons = {}
        with self.dialog_bubble.content:
            with ui.VStack(spacing=8):
                with ui.HStack(height=24*scale):
                    ui.Label('YOUR LAB GUIDE',name='Heading')
                    self.close_button = ui.Button('×',width=26*scale,clicked_fn=self.request_close,tooltip='Close / Esc')
                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=10):
                        self.title = ui.Label(current.title,height=0,word_wrap=True,name='AnswerTitle')
                        self.reply = ui.Label(current.text,height=0,word_wrap=True,name='Answer')
                        ui.Spacer(height=4)
                ui.Separator(height=1)
                if current.choices:
                    ui.Label('CHOOSE A TOPIC',height=16*scale,name='Caption')
                    columns = 2
                    with ui.VStack(height=0,spacing=4*scale):
                        for index in range(0,len(current.choices),columns):
                            with ui.HStack(height=32*scale,spacing=6*scale):
                                for choice in current.choices[index:index+columns]:
                                    self.choice_buttons[choice.target] = ui.Button(choice.label,
                                        clicked_fn=lambda target=choice.target:self.choose(target))
                                if len(current.choices[index:index+columns]) < columns:ui.Spacer()
                with ui.HStack(height=30*scale,spacing=6*scale):
                    self.back_button = ui.Button('Back',enabled=bool(self.conversation.history),
                        clicked_fn=lambda:self.choose('back'))
                    self.home_button = ui.Button('Main menu',enabled=self.conversation.topic != 'home',
                        clicked_fn=lambda:self.choose('home'))
                ui.Label('Esc to close · Click the scene to walk away',height=20*scale,name='Caption')

    def choose(self, target):
        # Rebuild on the next update, after the button's release callback.
        if not self.active:return
        if self.pending_choice is None:self.pending_choice = target
        self.lab._clear_input()

    def close(self):
        had_conversation = self.window is not None
        self.pending = None
        self.pending_choice = None
        self.choice_buttons = {}
        if self.dialog_bubble:self.dialog_bubble.destroy()
        self.dialog_bubble = self.window = None
        self.lab._clear_input()
        self.lab.navigation_focus = True
        if had_conversation and self.lab.window:
            self.lab.window.focus()

    def shutdown(self):
        self.close()
        if self.prompt_bubble:self.prompt_bubble.destroy()
        self.prompt_bubble = self.prompt = None

    def reset(self):
        self.close()
        self.near = False
        self.scene.reset()

    def update(self, dt):
        if self.pending:
            action,self.pending = self.pending,None
            self._open() if action == 'open' else self.close()
        if self.active and self.pending_choice is not None:
            target,self.pending_choice = self.pending_choice,None
            if target == 'back':self.conversation.back()
            elif target == 'home':self.conversation.home()
            else:self.conversation.choose(target)
            self.dialog_bubble.content.rebuild()
        if self.window and not self.window.visible:self.close()
        if self.active and (self.lab.current_room != 'lobby' or
                math.dist((self.scene.motion.x,self.scene.motion.z),(self.lab.walker.x,self.lab.walker.z)) > 3.3):
            self.close()
        if self.active:self.lab._clear_input()
        frozen = bool(self.lab.settings_window and self.lab.settings_window.visible)
        self.scene.update(dt,(self.lab.walker.x,self.lab.walker.z),self.active,frozen)

    def update_anchor(self):
        """Follow the camera of the displayed RTX frame, not one still rendering."""
        self.near = self.available()
        view = self.lab.viewport
        self.anchor = None
        if self.lab.current_room == 'lobby' and view.camera_pose:
            widget = view.widget
            rect = (widget.screen_position_x,widget.screen_position_y,widget.computed_width,widget.computed_height)
            size = view.presented_size or (1280,836)
            self.anchor = project(self.scene.bubble_anchor,view.camera_pose,rect,size[0]/size[1])
        if self.active and self.anchor is None:self.close()
        if self.prompt:self.prompt.visible = self.near and not self.active and self.anchor is not None
        bubble = self.dialog_bubble if self.active else self.prompt_bubble if self.prompt and self.prompt.visible else None
        if bubble and self.anchor:
            scale = self.lab.settings.text_scale
            aw,ah = ui.Workspace.get_main_window_width(),ui.Workspace.get_main_window_height()
            width = min((410 if self.active else 276)*scale,aw-24)
            height = min((450 if self.active else 122)*scale,ah-24)
            bubble.place(self.anchor,width,height,aw,ah)
