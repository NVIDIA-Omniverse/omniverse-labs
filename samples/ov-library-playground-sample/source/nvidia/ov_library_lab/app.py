# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Omniverse Libraries Playground: ovui host, ovstage scene, ovrtx frames."""
import asyncio
import os
from pathlib import Path
import traceback
import time
from .bootstrap import configure
configure()
import omni.ui as ui
from .style import STYLE, BG, GREEN, MUTED, rgba, scaled_style
from .content import ROOMS
from .presentation import GalleryPresentation
from .native_runtime import NativeRuntime
from .native_view import NativeView
from .keys import Key
from .experience import APP_NAME, ExperienceSettings, PROFILES
from .extensions import call
from .visitor import VisitorControls


class LibraryLab(VisitorControls, GalleryPresentation):
    def __init__(self, runtime_root, profile='public', quality=None):
        self.runtime_root = Path(runtime_root)
        self.settings = ExperienceSettings(runtime_root, profile, quality)
        self.settings_window = None
        self.settings_requested = False
        self.pending_action = self.last_failed_action = None
        self.activity = None
        self.activity_message = ''
        self.activity_until = 0.
        self.reference_pixels = None
        self.physics_results = {}
        self.comparison_run = None
        self.choice_buttons = []
        self.toggle_models = []
        self.comparison_buttons = {}
        self.sync_render = False
        self.pending_camera_pose = None
        self.presented_times = []
        self.restart_requested = False
        self.ready = False
        self.error = None
        self.window = None
        self.viewport = None
        self.observer_view = None
        self.gallery = None
        self.current_room = "lobby"
        self.dragging = False
        self.navigation_focus = True
        self.last_mouse = None
        self.keys = set()
        self.running = False
        self.sim_started = False
        self.sim_time = 0.
        self.accumulator = 0.
        self.spin = False
        self.high_drop = False
        self.roughness = .24
        self.key_on = True
        self.warm_light = True
        self.moon = False
        self.physics_mode = 'drop'
        self.physics_speed = 1.
        self.friction = .38
        self.carton_mass = 2.
        self.panel_modes = {}
        self.pending_room = None
        self.panel_dirty = False
        self.models = {}
        self.tab_buttons = {}
        self.subject_buttons = {}
        self.finish_buttons = {}
        self.stream_view = None
        self.stream_transport = None
        self.stream_error = None
        self.snapshot_task = None
        self.detail = None
        self.light_button = None
        self.panel = None
        self.stats = None
        self.overlay = None
        self.guide = None
        from .equipment_panel import EquipmentPanel
        self.equipment_panel = EquipmentPanel(self)
        self.exit_requested = False

    def request_exit(self):
        self.exit_requested = True

    async def start(self, progress=None):
        from .pacing import configure_desktop_presentation
        self.nonblocking_desktop = configure_desktop_presentation()
        async def report(message):
            print('LAB_PROGRESS: '+message, flush=True)
            if progress:
                progress(message)
            await ui.next_frame()
        await report('1 / 5  Initializing RTX and the scene libraries')
        # ovrtx must initialize its USD integration before creating the USD scene.
        self.native = NativeRuntime()
        await report('2 / 5  Building the gallery and loading its assets')
        from pxr import Usd, Plug
        Plug.Registry().RegisterPlugins(str(self.native.ovphysx.codeless_schema_root()))
        from .scene import Gallery
        from .navigation import Walker
        from .native_storage import RackStorage
        self.stage = Usd.Stage.CreateInMemory()
        self.gallery = Gallery(self.stage, Path(__file__).resolve().parents[3]/'data')
        import json
        from .campus_assets import CAMPUS_REVISION, CAMPUS_LAYOUT
        from .layout import GALLERIES
        print('LAB_SCENE: '+json.dumps({
            'profile': self.settings.profile,
            'revision': CAMPUS_REVISION if self.gallery.campus else 'classic',
            'layout': CAMPUS_LAYOUT if self.gallery.campus else 'procedural',
            'available_libraries': [key for key in ROOMS if self.settings.allowed(key)],
            'rendering_gallery': GALLERIES['rendering'] if self.gallery.campus else None,
        }), flush=True)
        if self.settings.allowed('newton'):
            self.gallery.newton_room.attach(self.runtime_root)
        if self.settings.allowed('package'):
            self.gallery.package_room.attach(self.runtime_root)
        call('start', self)
        self.roughness = self.gallery.workshops.car.roughness
        self.gallery.extra.snapshot_path = self.runtime_root/'storage/revisions/saved-asset.usda'
        self.storage = RackStorage(self.gallery.extra)
        await report('3 / 5  Opening your saved revisions')
        await self.storage.open()
        self.walker = Walker(self.gallery.obstacles)
        from .layout import LOBBY_POSE
        self.walker.x, self.walker.z, self.walker.yaw, self.walker.pitch = LOBBY_POSE
        # Warm up the same view the visitor will see; do not flash the authored
        # camera pose before the first interactive frame.
        self.gallery.set_camera(*LOBBY_POSE)
        if self.gallery.campus_blocker:
            self.walker.dynamic_blockers.append(self.gallery.campus_blocker)
        guide_asset = self.runtime_root/'assets/lab-guide/GTCB_Avatar.usda'
        if guide_asset.is_file():
            from .guide_view import GuideView
            self.guide = GuideView(self,guide_asset)
            self.walker.dynamic_blockers.append(self.guide.scene.motion.blocks)
        await report('4 / 5  Preparing the native simulation and rendering scene')
        self.native.populate(self.stage)
        self.native.prepare_physics()
        # Initialize the local transport while loading, then reuse it across
        # visits instead of allocating CUDA/SHM resources at a room boundary.
        from .stream_transport import LocalStream
        try:
            self.stream_transport = LocalStream()
        except Exception as exc:
            self.stream_error = str(exc)
        self.native.set_quality(*self.settings.render_spec())
        await report('5 / 5  Preparing the first RTX image; shaders may take a few minutes')
        self.native.begin_render({'Main', 'Fleet'})
        first_images = self.native.poll_render()
        last_second, next_log = -1, 30
        while first_images is None:
            await ui.next_frame()
            elapsed = int(time.perf_counter()-self.native.render_started)
            if elapsed != last_second:
                last_second = elapsed
                message = (f'5 / 5  Preparing the first RTX image / {elapsed//60}:{elapsed%60:02d} elapsed\n'
                           'Loading materials and preparing shaders. First launch may take several minutes; later launches reuse the cache.')
                if progress:
                    progress(message)
                if elapsed >= next_log:
                    print('LAB_PROGRESS: '+message.replace('\n', ' / '), flush=True)
                    next_log = elapsed+30
            first_images = self.native.poll_render()
        # Prepare the isolated sensor renderer before live frame submission.
        # A fresh Windows runtime stalled in native stage attachment when this
        # initialization overlapped the main renderer's interactive loop.
        await report('5 / 5  Preparing robot LiDAR and fisheye cameras')
        from .lidar_engine import LidarEngine
        self.lidar_engine = LidarEngine(self.gallery.workshops.lidar,self.runtime_root)
        last_second = -1
        while not self.lidar_engine.ready and not self.lidar_engine.error:
            await ui.next_frame()
            self.lidar_engine.poll()
            elapsed = int(time.monotonic()-self.lidar_engine.started)
            if elapsed != last_second:
                last_second = elapsed
                if progress:
                    progress(f'5 / 5  Preparing robot LiDAR and fisheye cameras / {elapsed//60}:{elapsed%60:02d} elapsed')
        self.gallery.workshops.lidar.readout.clear(self.gallery.workshops.lidar.sensor_position())
        self._build_window()
        self.go_to('lobby')
        self.viewport.update(first_images['Main'])
        self.loading_frame.visible = False
        self.ready = True
        print('LAB_READY: native libraries ready', flush=True)

    def _build_window(self):
        # This full-screen background must accept walking/mouse-look focus
        # without covering passive floating UI such as the guide invitation.
        # ovui passes flags directly to ImGui; the pinned release does not
        # export ImGuiWindowFlags_NoBringToFrontOnFocus (1 << 13) to Python.
        no_bring_to_front = getattr(ui, 'WINDOW_FLAGS_NO_BRING_TO_FRONT_ON_FOCUS', 1 << 13)
        flags = (ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE |
                 ui.WINDOW_FLAGS_NO_MOVE | ui.WINDOW_FLAGS_NO_COLLAPSE |
                 ui.WINDOW_FLAGS_NO_SCROLLBAR | no_bring_to_front)
        self.window = ui.Window(APP_NAME, width=1600, height=980, position_x=0, position_y=0,
                                flags=flags, padding_x=0, padding_y=0, fill_app_window=True)
        self.window.frame.style = scaled_style(self.settings.text_scale)
        # ovui window callbacks use GLFW key codes, also used by the arcade.
        self.window.set_key_pressed_fn(self._on_ui_key)
        self.window.set_focused_changed_fn(self._on_scene_focus)
        with self.window.frame:
            with ui.VStack(spacing=0):
                self.header_mode = None
                self.header = ui.Frame(height=76)
                self._build_header(ui.Workspace.get_main_window_width() or 1600)
                # Separator's menu-backed layout retains its previous width
                # in standalone ovui. A plain rule can shrink with the window.
                ui.Rectangle(height=1, style={'background_color': rgba(50, 66, 76)})
                with ui.HStack(spacing=0):
                    with ui.VStack(spacing=0):
                        with ui.ZStack():
                            self.viewport = NativeView(self,
                                mouse_pressed_fn=self._mouse_down, mouse_released_fn=self._mouse_up,
                                mouse_moved_fn=self._mouse_move, mouse_hovered_fn=self._mouse_hover)
                            if self.overlay:self.overlay.build_prompt()
                            if self.guide:self.guide.build_prompt()
                            self.loading_frame = ui.Frame()
                            with self.loading_frame:
                                with ui.ZStack():
                                    ui.Rectangle(style={"background_color": BG})
                                    with ui.VStack(spacing=16):
                                        ui.Spacer()
                                        ui.Label("Preparing your gallery", alignment=ui.Alignment.CENTER, height=40, style={"font_size": 30, "color": GREEN})
                                        ui.Label("RTX is getting ready. Shader preparation may take a few minutes.", alignment=ui.Alignment.CENTER, height=30, style={"color": MUTED})
                                        ui.Label("You can explore the lessons while the scene loads.", alignment=ui.Alignment.CENTER, height=25, style={"color": MUTED})
                                        ui.Spacer()
                        with ui.HStack(height=26, spacing=0):
                            self.footer_left_inset = ui.Spacer(width=0)
                            self.walking_hint = ui.Label("WASD  Walk  |  Drag / arrow keys  Look  |  Shift  Faster",
                                width=0, alignment=ui.Alignment.LEFT_CENTER,
                                style={"font_size": 14, "color": MUTED})
                            ui.Spacer(width=16)
                            self.activity = ui.Label('', elided_text=True,
                                alignment=ui.Alignment.RIGHT_CENTER,
                                style={"font_size": 13, "color": GREEN})
                            self.footer_right_inset = ui.Spacer(width=0)
                    with ui.VStack(width=410, spacing=0) as self.sidebar:
                        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as self.panel_scroll:
                            self.panel = ui.Frame()

    def _on_ui_key(self, key, modifiers, pressed):
        # A release can arrive in a different window from its press. Always
        # discard it, even when a dialog or navigation focus consumes input.
        if not pressed:
            self.keys.discard(key)
        if self.guide and self.guide.key(key,pressed):
            return
        inspector = call('input_window', self) if self.gallery else None
        if inspector and inspector.visible:
            if pressed and key == Key.ESCAPE:
                inspector.visible = False
            self._clear_input()
            return
        if self.overlay and self.overlay.key(key, pressed):
            return
        if pressed and key == 290:  # F1
            self.open_settings()
            return
        if self.settings_window and self.settings_window.visible:
            if key == Key.ESCAPE:
                self.settings_window.visible = False
            self._clear_input()
            return
        if pressed and key == 268:  # Home
            self.request_room('lobby')
            return
        if key == Key.ESCAPE or key == 258:  # Tab moves focus into the UI.
            # Escape may already have closed the guide on key-down. Its
            # release must not undo the navigation focus restored by close().
            if pressed:
                self._clear_input()
                self.navigation_focus = False
        elif not self.navigation_focus:
            return
        elif pressed:
            self.keys.add(key)
        elif not pressed:
            self.keys.discard(key)

    def toggle_physics(self):
        if not self.sim_started:
            self.native.start_physics(self.gallery.dynamic_paths)
            self.sim_started = True
            self.native.step_physics(1)
            if self.physics_mode == 'slide':
                self.native.pulse(self.gallery.dynamic_paths, 1.15)
        self.running = not self.running
        if self.current_room == 'physics':
            self.release_button.text = 'Pause' if self.running else 'Run experiment'

    def reset_physics(self):
        self.running = False
        self.sim_started = False
        self.sim_time = self.accumulator = 0.
        self.comparison_run = None
        self.native.reset_physics()
        if self.current_room == 'physics' and hasattr(self, 'release_button'):
            self.release_button.text = 'Run experiment'

    def step_physics(self, steps=1):
        self.native.step_physics(steps*2)
        self.sim_time += steps/60

    def update(self, dt):
        if not self.ready:
            return
        self.viewport.update_cursor()
        width, height = ui.Workspace.get_main_window_width(), ui.Workspace.get_main_window_height()
        if width <= 0 or height <= 0:
            self._clear_input()
            return
        if self.settings_requested:
            self.settings_requested = False
            self._build_settings()
        if self.overlay and self.overlay.update():
            return
        self.process_action()
        if self.pending_room is not None:
            key, self.pending_room = self.pending_room, None
            self.go_to(key)
        elif self.panel_dirty:
            self._build_panel()
        dt = min(float(dt), .1)
        if self.guide:
            self.guide.update(dt)
        if width > 0 and height > 0:
            self._build_header(width)
            if abs(self.window.width-width) > 1:
                self.window.width = width
            if abs(self.window.height-height) > 1:
                self.window.height = height
        K = Key
        moving = bool(self.keys & {K.W, K.S, K.A, K.D, K.LEFT, K.RIGHT, K.UP, K.DOWN}) or self.dragging
        self.walker.move(int(K.W in self.keys)-int(K.S in self.keys), int(K.D in self.keys)-int(K.A in self.keys), dt, K.LEFT_SHIFT in self.keys)
        self.walker.yaw += (int(K.RIGHT in self.keys)-int(K.LEFT in self.keys))*dt*1.3
        self.walker.pitch = max(-1.05, min(.95, self.walker.pitch+(int(K.UP in self.keys)-int(K.DOWN in self.keys))*dt))
        self.gallery.set_camera(self.walker.x, self.walker.z, self.walker.yaw, self.walker.pitch)
        room = self.walker.room()
        if room != self.current_room:
            self._enter_room(room)
        if self.running:
            self.accumulator += dt*self.physics_speed
            steps = min(6, int(self.accumulator*60))
            if steps:
                if self.comparison_run:
                    steps = min(steps, max(0, round((2.-self.sim_time)*60)))
                if steps:
                    self.step_physics(steps)
                    self.accumulator -= steps/60
                self.finish_comparison()
        self.gallery.advance_shared(dt, self.spin)
        call('poll', self)
        self.gallery.workshops.advance(dt)
        self.gallery.workshops.lidar.advance(dt if self.current_room == 'rendering' else 0.)
        if self.settings.allowed('newton'):
            self.gallery.newton_room.advance(dt if self.current_room == 'newton' else 0.)
        if self.settings.allowed('package'):
            self.gallery.package_room.advance()
        if self.current_room in ('newton','package'):
            from .extended_panels import refresh as refresh_extended
            refresh_extended(self)
        call('advance', self, dt)
        if self.current_room == 'interface' and hasattr(self, 'lift_model'):
            value = self.gallery.extra.lift_value
            if abs(self.lift_model.as_float-value) > .0001:
                self.lift_model.set_value(value)
        if self.stats:
            if self.current_room == "physics":
                if self.physics_mode == 'stack' and hasattr(self, 'stack_nudge'):
                    self.stack_nudge.enabled = self.sim_started and not self.pending_action
                for friction, button in self.comparison_buttons.items():
                    active = self.comparison_run and self.comparison_run['friction'] == friction
                    button.enabled = self.comparison_run is None
                    next_friction = .65 if .05 in self.physics_results and .65 not in self.physics_results else .05
                    button.name = 'Busy' if active else 'Primary' if friction == next_friction else ''
                    button.text = 'Measuring...' if active else ('Run low friction' if friction == .05 else 'Run high friction')
                y = self.gallery.position(self.gallery.dynamic_paths[0])[1]
                self.stats.text = f"{self.physics_mode.title()}  |  {'Running' if self.running else 'Ready / paused'}  |  {self.physics_speed:.1f}x\nCarton centre {y:.2f} m  |  friction {self.friction:.2f}\nGravity {1.62 if self.moon else 9.81:.2f} m/s²  |  {self.carton_mass:g} kg each"
            elif self.current_room == "rendering":
                w = self.gallery.workshops
                metal = w.car.metallic if w.render_subject == 'car' else self.stage.GetPrimAtPath('/World/Looks/inspection_alloy/Surface').GetAttribute('inputs:metallic').Get()
                for key, value in (('roughness', self.roughness), ('metallic', metal), ('clearcoat', w.car.clearcoat), ('light_angle', w.light_angle), ('light_size', w.light_size), ('zoom', w.camera_zoom)):
                    if key in self.models and abs(self.models[key].as_float-value) > .0001:
                        self.models[key].set_value(value)
                if w.render_subject == 'lidar':
                    from .lidar_panel import refresh
                    refresh(self)
                elif self.panel_modes.get('rendering') == 'camera':
                    self.stats.text = self.gallery.workshops.snapshot_status
                else:
                    self.stats.text = f"Roughness {self.roughness:.2f}  |  {'Warm' if self.warm_light else 'Cool'}  |  Key {'on' if self.key_on else 'off'}"
            elif self.current_room == "scene":
                g = self.gallery
                fleet_moving = abs(g.shared_target-g.shared_offset) > .01
                location = 'In transit' if fleet_moving else 'Station B' if g.shared_offset > 0 else 'Station A'
                self.stats.text = f"{g.workshops.fleet_count} AMR(s)  |  {location}  |  {'Loaded' if g.shared_loaded else 'Empty'}\nShared revision {g.revision}  |  auto shuttle {'on' if self.spin else 'off'}"
            elif self.current_room == "storage":
                from .storage_panel import refresh as refresh_storage
                refresh_storage(self)
            elif self.current_room == "interface":
                extra = self.gallery.extra
                self.stats.text = f"Fork height {extra.lift_value*100:.0f}%  |  Work light {'on' if extra.work_light_on else 'off'}\nPallet QC: {('Awaiting inspection', 'Hold for review', 'Approved')[extra.part_color]}"
                self.stats.style = {"font_size": 13, "color": (rgba(84, 200, 247), rgba(255, 196, 65), GREEN)[extra.part_color]}

            elif self.current_room == 'streaming':
                view = self.stream_view
                connected = bool(view and view.transport)
                for key in ('pause','dispatch','cargo'):
                    self.stream_controls[key].enabled = connected
                self.stream_controls['connect'].enabled = not connected and not self.pending_action
                self.stream_controls['connect'].text = 'Camera connected' if connected else 'Connecting...' if self.pending_action else 'Retry connection'
                self.stream_controls['connect'].name = '' if connected else 'Busy' if self.pending_action else 'Primary'
                self.stream_hint.text = 'Connected. Commands below control the robot through ovstream.' if connected else 'Connect the camera feed to enable client commands.'
                if view and view.transport:
                    t = view.transport
                    age = time.monotonic()-t.last_received if t.last_received else 0.
                    self.stats.text = f"{'FEED PAUSED' if self.gallery.workshops.stream_paused else 'CONNECTED / LOCAL SHM'}  |  {t.received} frames\nFrame age {age:.1f}s  |  target {self.gallery.workshops.stream_fps} fps\nCommands received {view.command_count} / {view.last_command}"
                elif view and view.error:
                    self.stats.text = 'Connection unavailable: '+view.error[:160]
        call('refresh', self)
        products = {'Main'}
        lidar_active=self.observer_view and self.observer_view.updates_enabled and self.observer_view.product=='Lidar'
        lidar_pixels=self.lidar_engine.update(lidar_active,self.sync_render)
        if lidar_active and lidar_pixels is not None:
            self.observer_view.update(lidar_pixels)
            for name,view in self.robot_camera_views.items():
                pixels=self.gallery.workshops.lidar.camera_pixels.get(name)
                if pixels is not None:view.update(pixels)
        if self.observer_view and self.observer_view.updates_enabled and not lidar_active:
            products.add(self.observer_view.product)
        if self.stream_view and not getattr(self,'showing_room_info',False):
            products.add('Stream')
        if self.sync_render:
            self.native.set_quality(*self.settings.render_spec(moving))
            images = self.native.render_all(products)
            self.pending_camera_pose = self._camera_pose_for_overlay()
        else:
            images = self.native.poll_render(gpu_presenters={'Main': self.viewport.present_gpu} if self.viewport.gpu_enabled else None)
        if images:
            if self.observer_view and self.observer_view.product in images:
                self.observer_view.update(images[self.observer_view.product])
            if 'Main' in images:
                self.viewport.camera_pose = self.pending_camera_pose
                if images['Main'] is not None:
                    self.viewport.update(images['Main'])
                self.presented_times.append(time.perf_counter())
                self.presented_times = self.presented_times[-600:]
                self.loading_frame.visible = False
            if self.stream_view and 'Stream' in images:
                self.stream_view.update(images['Stream'])
        if not self.sync_render and self.native.pending_render is None:
            self.native.set_quality(*self.settings.render_spec(moving))
            self.pending_camera_pose = self._camera_pose_for_overlay()
            self.native.begin_render(products)
        if self.guide:self.guide.update_anchor()
        self.equipment_panel.update()
        for button, label, getter, value in self.choice_buttons:
            from .controls import set_option
            set_option(button, label, getter() == value)
        for model, getter in self.toggle_models:
            value = bool(getter())
            if model.as_bool != value:
                model.set_value(value)
        if self.comparison_run:
            self.activity.text = f"{'Measuring' if self.running else 'Paused'} carton travel / {self.sim_time:.1f} of 2 simulated seconds"
        elif self.pending_action:
            self.activity.text = self.pending_action[0]+'...'
        elif time.monotonic() < self.activity_until:
            self.activity.text = self.activity_message
        elif self.native.pending_render and time.perf_counter()-self.native.render_started > .7:
            self.activity.text = 'Preparing RTX view... You can continue using the controls.'
        else:
            self.activity.text = self.settings.quality+' quality  /  '+('Moving view' if moving else 'Ready to explore')
        if self.activity.tooltip != self.activity.text:
            self.activity.tooltip = self.activity.text
        self._align_viewport_footer()

    def _align_viewport_footer(self):
        # Align the single footer row to both edges of the displayed RTX image,
        # including horizontal letterboxing after a window resize.
        view = self.viewport
        w, h = view.widget.computed_width, view.widget.computed_height
        size = view.presented_size
        geometry = (w, h, size)
        if not size or geometry == getattr(self, '_footer_geometry', None):
            return
        self._footer_geometry = geometry
        fit = min(w/size[0], h/size[1])
        inset = max(0, (w-size[0]*fit)/2)
        self.footer_left_inset.width = ui.Pixel(inset)
        self.footer_right_inset.width = ui.Pixel(inset)

    def _camera_pose_for_overlay(self):
        w,camera = self.walker,self.gallery.camera
        return (w.x,w.z,w.yaw,w.pitch,1.85,camera.GetFocalLengthAttr().Get(),camera.GetHorizontalApertureAttr().Get())

    async def close(self):
        self.ready = False
        if getattr(self,'lidar_engine',None):self.lidar_engine.close()
        if self.viewport:
            self.viewport.close_cursor()
        self.equipment_panel.close()
        if self.guide:
            self.guide.shutdown()
        if self.overlay:
            self.overlay.shutdown()
        if self.stream_view:
            self.stream_view.close()
        if self.stream_transport:
            self.stream_transport.close()
            self.stream_transport = None
        if hasattr(self, 'storage') and self.storage.task:
            await asyncio.gather(self.storage.task, return_exceptions=True)
        if self.snapshot_task:
            self.snapshot_task.cancel()
            await asyncio.gather(self.snapshot_task, return_exceptions=True)
        if self.settings_window:
            self.settings_window.destroy()
        if getattr(self, 'example_window', None):
            self.example_window.destroy()
        if self.gallery:
            self.gallery.newton_room.close()
            self.gallery.package_room.close()
            call('close', self)
            self.gallery.workshops.shutdown()
        if hasattr(self, 'native'):
            self.native.close()
        if self.window:
            self.window.destroy()


async def run(runtime_root, verify=False, inspect=False, profile='public', quality=None):
    lab = LibraryLab(runtime_root, profile, quality)
    loading = ui.Window('Preparing '+APP_NAME, width=640, height=220)
    with loading.frame:
        with ui.VStack(spacing=12):
            ui.Label('Preparing the Omniverse Libraries', height=50, alignment=ui.Alignment.CENTER)
            progress = ui.Label('RTX startup can take a few minutes on the first launch.', word_wrap=True, alignment=ui.Alignment.CENTER)
    await ui.next_frame()
    try:
        await lab.start(lambda message: setattr(progress, 'text', message))
        loading.destroy()
        loading = None
        inspector=None
        if inspect:
            import ovuiinspect as inspector
            inspector.attach_application(lab)
        if verify:
            from .native_checks import run_checks
            await run_checks(lab)
            return
        while not lab.exit_requested:
            frame = await ui.next_frame()
            if inspector:
                from omni.ui import _ui
                inspector.drain_pending(_ui,application=lab)
            lab.update(frame.dt)
    except asyncio.CancelledError:
        pass
    except Exception:
        if verify:
            raise
        lab.report_error('Starting or updating the lab')
        if lab.window:
            lab.window.visible = False
        if loading:
            loading.visible = False
        failure = ui.Window('The lab needs to restart', width=640, height=300)
        failure.frame.style = STYLE
        with failure.frame:
            with ui.VStack(spacing=14):
                ui.Label('The lab could not continue', height=40, style={'font_size': 24})
                ui.Label('Your saved files are preserved. Restart to recover the native libraries.\nIf this repeats, choose Performance quality in Settings and check the runtime logs.', word_wrap=True, height=95)
                def restart():
                    lab.restart_requested = True
                    lab.request_exit()
                with ui.HStack(height=40, spacing=12):
                    ui.Button('Restart the lab', name='Primary', clicked_fn=restart)
                    ui.Button('Close', clicked_fn=lab.request_exit)
        while not lab.exit_requested and failure.visible:
            await ui.next_frame()
        failure.destroy()
    finally:
        await lab.close()
    return 75 if lab.restart_requested else 0


async def host(runtime_root, verify=False, inspect=False, profile='public', quality=None):
    # Own cancellation so closing the OS window releases RTX, PhysX, streams and
    # the arcade before ovui tears down its native windowing backend.
    pump=asyncio.create_task(ui.run_async())
    application=asyncio.create_task(run(runtime_root,verify,inspect,profile,quality))
    try:
        await asyncio.wait((pump,application),return_when=asyncio.FIRST_COMPLETED)
        if not application.done():
            application.cancel()
        return await application
    finally:
        pump.cancel()
        await asyncio.gather(pump,return_exceptions=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, default=Path(__file__).resolve().parents[3]/'.ov_library_lab_runtime')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--inspect', action='store_true', help='Enable the optional official ovui inspector on localhost.')
    parser.add_argument('--profile', choices=PROFILES, default='public')
    parser.add_argument('--quality', choices=('Performance', 'Balanced', 'Studio'))
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1000)
    args = parser.parse_args()
    os.environ.setdefault('OVRTX_SKIP_USD_CHECK','1')
    # A 120 Hz desktop pump competes with RTX for the laptop GPU and delays
    # texture handoff. 60 Hz kept input responsive and delivered more images.
    # Offscreen capture has no desktop compositor and retains its own pacing.
    headless = os.environ.get('OMNIUI_HEADLESS','').lower() in ('1','true','yes','on')
    ui.init(APP_NAME, width=max(800, args.width), height=max(600, args.height), max_fps=120 if headless else 60)
    try:
        from .pacing import precise_timers, run_event_loop
        with precise_timers():
            return run_event_loop(host(args.runtime_root,args.verify,args.inspect,args.profile,args.quality))
    finally:
        ui.shutdown()


if __name__ == '__main__':
    main()
