# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared gallery presentation and interactions, independent of Kit."""
import asyncio
import math
import webbrowser
import omni.ui as ui
from .style import *
from .content import ROOMS, CATALOG_URL
from .extensions import call, value
from .layout import LOBBY_POSE, pose
from .campus_assets import CAMPUS_HIDDEN_ROOMS
from .experience import APP_NAME


class GalleryPresentation:

    def _build_header(self, width):
        # Workspace dimensions are logical points; native monitor DPI scales them.
        entries = [('lobby', 'Atrium')] + [(key, room['name']) for key, room in ROOMS.items() if key not in CAMPUS_HIDDEN_ROOMS]
        scale = self.settings.text_scale
        sizes = {key: max(68, math.ceil(len(title)*8.1*scale+22)) for key,title in entries}
        buttons_width = sum(sizes.values())+6*(len(entries)-1)
        rows, row, used = [], [], 0
        for entry in entries:
            space = sizes[entry[0]]+(6 if row else 0)
            if row and used+space > width-48:
                rows.append(row)
                row, used, space = [], 0, sizes[entry[0]]
            row.append(entry)
            used += space
        rows.append(row)
        brand_width = 330
        wide = width >= buttons_width+brand_width+138
        mode = (wide, scale, tuple(tuple(key for key,_ in row) for row in rows))
        if mode == self.header_mode:
            return
        self.header_mode = mode
        self.header.height = ui.Pixel(62 if wide else 50+44*len(rows))
        self.header.clear()
        self.room_buttons = {}

        def brand():
            ui.Label(APP_NAME, width=brand_width, height=38,
                style={'font_size': 20, 'color': GREEN})

        def buttons(items):
            with ui.HStack(height=38, spacing=6):
                for key, title in items:
                    self.room_buttons[key] = ui.Button(title, width=sizes[key],
                        name='CurrentRoom' if key == self.current_room else 'RoomNav',
                        clicked_fn=lambda k=key: self.request_room(k))

        def settings():
            # Draw the cog directly: it stays crisp at monitor DPI and does not
            # depend on a font containing a settings symbol.
            with ui.ZStack(width=42, height=38):
                self.settings_button = ui.Button('', tooltip='Settings / F1',
                    name='SettingsCog', clicked_fn=self.open_settings)
                for i in range(8):
                    angle = i*math.pi/4
                    with ui.Placer(offset_x=19+9.5*math.cos(angle), offset_y=17+9.5*math.sin(angle)):
                        ui.Rectangle(width=4,height=4,style={'background_color':INK,'border_radius':1})
                with ui.Placer(offset_x=12.5,offset_y=10.5):
                    ui.Circle(width=17,height=17,style={'background_color':INK})
                with ui.Placer(offset_x=17,offset_y=15):
                    ui.Circle(width=8,height=8,style={'background_color':BG})

        with self.header:
            with ui.HStack(spacing=0):
                ui.Spacer(width=24)
                with ui.VStack(spacing=6):
                    ui.Spacer()
                    with ui.HStack(height=38,spacing=16):
                        brand()
                        if wide:
                            with ui.Frame(width=buttons_width):
                                buttons(entries)
                        ui.Spacer()
                        settings()
                    if not wide:
                        for row in rows:
                            buttons(row)
                    ui.Spacer()
                ui.Spacer(width=24)

    def _label(self, text, size=16, color=INK, height=0):
        scale = self.settings.text_scale
        return ui.Label(text, word_wrap=True, height=height*scale, style={"font_size": size*scale, "color": color})

    def choice(self, label, getter, value, callback):
        from .controls import option
        button = option(label, getter() == value, lambda: callback() if getter() != value else None)
        self.choice_buttons.append((button, label, getter, value))
        return button

    def toggle(self, label, getter, setter):
        from .controls import toggle
        return toggle(self, label, getter, setter)

    def _section(self, title):
        self._label(title, 11, MUTED, height=19)

    def _build_panel(self):
        self.panel_dirty = False
        self.choice_buttons = []
        self.toggle_models = []
        self.comparison_buttons = {}
        if self.snapshot_task and not self.snapshot_task.done():
            self.snapshot_task.cancel()
        if self.stream_view:
            if self.pending_action and getattr(self.pending_action[1], '__self__', None) is self.stream_view:
                self.pending_action = None
            self.stream_view.close()
            self.stream_view = None
        self.models = {}
        self.tab_buttons = {}
        self.subject_buttons = {}
        self.finish_buttons = {}
        self.detail = None
        self.light_button = None
        if self.observer_view:
            self.observer_view.destroy()
            self.observer_view = None
        for view in getattr(self,'robot_camera_views',{}).values():
            view.destroy()
        self.robot_camera_views={}
        self.stats = None
        self.extended_ui = {}
        self.storage_controls = {}
        self.stream_controls = {}
        call('prepare_panel', self)
        self.panel.clear()
        with self.panel:
            with ui.HStack():
                ui.Spacer(width=23)
                with ui.VStack(spacing=9):
                    ui.Spacer(height=23)
                    if self.current_room == "lobby":
                        self._build_lobby()
                    else:
                        self._build_lesson(self.current_room)
                    ui.Spacer(height=20)
                ui.Spacer(width=23)

    def _build_lobby(self):
        self._label("WELCOME TO THE LAB", 12, GREEN)
        self._label("Understand it.\nThen try it.", 32)
        self._label("Explore the Omniverse Libraries, try their features, and see how they can solve your use cases.", 16, MUTED)
        ui.Separator(height=1)
        self._label('Walk into a room to begin.', 19)
        self._label('The controls for each experiment appear here. Use the library names above to jump straight to a room.', 14, MUTED)
        self._section('GETTING AROUND')
        self._label('W A S D  /  walk\nDrag the scene  /  look around\nArrow keys  /  look around\nShift  /  walk faster\nHome  /  return to the atrium', 17)
        self._label('Click the scene to resume walking after using a control. If you get stuck, press Home.', 14, MUTED)
        if self.gallery.campus:
            self._label(call('campus_description') or 'Explore the galleries around the atrium.', 14, MUTED)
        if self.guide:
            self._section('MEET YOUR GUIDE')
            self._label('Visit the robot beside the NVIDIA sculpture for an introduction to the lab and its libraries.', 14, MUTED)
        ui.Button("Library catalog", height=34, clicked_fn=lambda: webbrowser.open(CATALOG_URL))

    def _build_lesson(self, key):
        from .panels import build
        build(self, key)
        if self.gallery.campus:
            from .layout import COMPANIONS
            companions = {**COMPANIONS, **{b:a for a,b in COMPANIONS.items()},
                          **value('ROOM_COMPANIONS', {})}
            if key in companions and companions[key] in ROOMS and self.settings.allowed(companions[key]):
                other = companions[key]
                ui.Separator(height=1)
                self._label('Also in this gallery: '+ROOMS[other]['name'], 14, MUTED)
                ui.Button('Explore '+ROOMS[other]['name'], height=34,
                          clicked_fn=lambda: self.request_room(other))

    def request_panel(self):
        """UI callbacks request work; the update loop owns widget destruction."""
        self.panel_dirty = True

    def request_room(self, key):
        self.last_failed_action = None
        if self.pending_action:
            self.pending_action = None
            self.notify('Pending action canceled while changing rooms.')
        self.pending_room = key

    def go_to(self, key):
        if key not in ("lobby", *ROOMS) or key in CAMPUS_HIDDEN_ROOMS:
            raise ValueError("Unknown room: " + key)
        if self.guide:self.guide.close()
        self.keys.clear()
        self.navigation_focus = True
        self.dragging = False
        self.last_mouse = None
        self.walker.x, self.walker.z, self.walker.yaw, self.walker.pitch = LOBBY_POSE if key == "lobby" else pose(key)
        if key == 'rendering' and self.gallery.workshops.render_subject in ('car','lidar'):
            self.walker.pitch = -.14
        self.gallery.set_camera(self.walker.x, self.walker.z, self.walker.yaw, self.walker.pitch)
        self._enter_room(key)

    def _enter_room(self, key):
        if key!=self.current_room:
            self.room_info_key=None
            self.showing_room_info=False
        call('enter_room', self, key)
        self.current_room = key
        for room_key, button in self.room_buttons.items():
            button.name = 'CurrentRoom' if room_key == key else 'RoomNav'
        # Entering an exhibit changes only the controls, never the visitor's
        # projection: an automatic lens change looks like a lurch while walking.
        self._build_panel()
        self.panel_scroll.scroll_y = 0

    def _on_scene_focus(self, focused):
        # ovui dispatches key-up only to the focused window. Forget any held
        # keys before another window can consume their release.
        if not focused:
            self._clear_input()

    def _clear_input(self, *_):
        self.keys.clear()
        self.dragging = False
        self.last_mouse = None
        if self.overlay:
            self.overlay.clear_input()

    def _mouse_down(self, x, y, button, modifier):
        if self.guide and self.guide.captures_input:
            if button == 0:self.guide.request_close()
            return
        if button == 0:
            self.navigation_focus = True
            self.dragging = True
            self.last_mouse = (x, y)

    def _mouse_up(self, *_):
        self.dragging = False
        self.last_mouse = None

    def _mouse_move(self, x, y, *_):
        if self.dragging and self.last_mouse:
            sensitivity = .004*self.settings.mouse_sensitivity
            self.walker.yaw += (x-self.last_mouse[0])*sensitivity
            self.walker.pitch = max(-1.05, min(.95, self.walker.pitch-(y-self.last_mouse[1])*sensitivity))
            self.last_mouse = (x, y)

    def _mouse_hover(self, hovered):
        if not hovered:
            self._mouse_up()

    def set_drop_height(self, high):
        from .physics_workshop import configure
        self.high_drop = bool(high)
        configure(self, 'drop')

    def set_gravity(self, moon):
        self.reset_physics()
        self.moon = moon
        self.gallery.gravity_attr.Set(1.62 if moon else 9.81)

    def toggle_light(self):
        self.key_on = not self.key_on
        self.gallery.key_light.GetIntensityAttr().Set(self.gallery.workshops.light_power if self.key_on else 0.)
        if self.current_room == "rendering" and self.light_button:
            self.light_button.text = "Key light ON" if self.key_on else "Key light OFF"

    def toggle_light_color(self):
        from pxr import Gf
        self.warm_light = not self.warm_light
        self.gallery.key_light.GetColorAttr().Set(Gf.Vec3f(*( (1., .77, .48) if self.warm_light else (.35, .70, 1.) )))

    def set_roughness(self, value):
        self.roughness = value
        w = self.gallery.workshops
        if w.render_subject == 'car':
            w.car.set_roughness(value)
        else:
            w.robot_roughness = value
            self.gallery.set_roughness(value)

    def set_render_subject(self, subject):
        w = self.gallery.workshops
        w.set_subject(subject)
        self.roughness = w.car.roughness if subject == 'car' else w.robot_roughness
        if self.current_room == 'rendering':
            self.walker.pitch = -.14 if subject in ('car','lidar') else .015
        self.request_panel()

    def move_shared(self):
        self.gallery.set_shared(offset=-2.4 if self.gallery.shared_target > 0 else 2.4)

    def recolor_shared(self):
        self.gallery.set_shared(toggle_load=True)

    def toggle_spin(self):
        self.spin = not self.spin
