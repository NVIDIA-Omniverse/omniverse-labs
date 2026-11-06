# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visitor controls, recoverable actions and session reset."""
import time
from pathlib import Path
import traceback

import omni.ui as ui
from .experience import QUALITIES
from .distribution import identity
from .extensions import call, value
from .style import STYLE, MUTED, GREEN, scaled_style


class VisitorControls:
    def notify(self, message, seconds=6):
        self.activity_message = message
        self.activity_until = time.monotonic()+seconds
        if getattr(self, 'activity', None):
            self.activity.text = message

    def queue_action(self, label, callback):
        if self.pending_action:
            self.notify('Finishing '+self.pending_action[0].lower()+'...')
            return
        self._clear_input()
        self.pending_action = (label, callback, 2)
        self.notify(label+'...', 180)

    def process_action(self):
        if not self.pending_action:
            return
        label, callback, frames = self.pending_action
        if frames:
            self.pending_action = (label, callback, frames-1)
            return
        self.pending_action = None
        try:
            callback()
            if self.comparison_run:
                self.notify('Measuring carton travel over 2 simulated seconds...', 180)
            else:
                self.notify(label+' complete.')
            self.last_failed_action = None
        except Exception:
            self.last_failed_action = (label, callback)
            self.report_error(label)

    def report_error(self, context):
        text = traceback.format_exc()
        folder = self.runtime_root/'logs'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'visitor-errors.log').open('a', encoding='utf-8') as handle:
            handle.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+context+'\n'+text+'\n')
        print(text, flush=True)
        self.notify(context+' failed. Open Settings for Retry or diagnostics.', 180)

    def retry_action(self):
        if self.last_failed_action:
            self.queue_action(*self.last_failed_action)
        else:
            self.notify('There is no failed action to retry.')

    def set_preference(self, key, value):
        setattr(self.settings, key, value)
        try:
            self.settings.save()
        except OSError:
            self.report_error('Saving preferences')
        if key == 'text_scale':
            self.window.frame.style = scaled_style(value)
            self.request_panel()
        self.notify(f'{key.replace("_", " ").title()}: {value}')

    def open_settings(self):
        if self.guide:self.guide.close()
        self._clear_input()
        self.settings_requested = True

    def _build_settings(self):
        from .experience import VERSION
        if self.settings_window:
            self.settings_window.destroy()
        width = min(550, max(400, ui.Workspace.get_main_window_width()-60))
        height = min(630, max(480, ui.Workspace.get_main_window_height()-90))
        window = self.settings_window = ui.Window('Playground settings', width=width, height=height,
            position_x=max(20, (ui.Workspace.get_main_window_width()-width)/2),
            position_y=max(40, (ui.Workspace.get_main_window_height()-height)/2))
        window.frame.style = scaled_style(self.settings.text_scale)
        window.set_key_pressed_fn(self._on_ui_key)
        with window.frame:
            with ui.ScrollingFrame():
              with ui.VStack(spacing=12):
                self._label('Make yourself comfortable', 24)
                self._label(f'{self.settings.profile.title()} preview / {VERSION}', 12, MUTED)
                self._label('Build: '+identity()['build_id'], 12, MUTED)
                self._label('Rendering quality', 14, MUTED)
                model = ui.ComboBox(list(QUALITIES).index(self.settings.quality), *QUALITIES, height=32).model
                model.add_item_changed_fn(lambda m, _: self.set_preference('quality', list(QUALITIES)[m.get_item_value_model().as_int]))
                self._label('Text size / in addition to your monitor scaling', 14, MUTED)
                with ui.HStack(height=34, spacing=8):
                    ui.Button('Standard', clicked_fn=lambda: self.set_preference('text_scale', 1.))
                    ui.Button('Larger', clicked_fn=lambda: self.set_preference('text_scale', 1.15))
                    ui.Button('Largest', clicked_fn=lambda: self.set_preference('text_scale', 1.25))
                self._label('Mouse look sensitivity', 14, MUTED)
                sensitivity = ui.SimpleFloatModel(self.settings.mouse_sensitivity)
                ui.FloatSlider(sensitivity, min=.3, max=1.5, height=28)
                sensitivity.add_value_changed_fn(lambda m: self.set_preference('mouse_sensitivity', m.as_float))
                self._label('Keyboard: WASD walk, arrows look, Shift faster.\nF1 settings, Home atrium, Esc release movement.\nTab focuses controls; click the scene to resume walking.', 14, MUTED)
                with ui.HStack(height=36, spacing=8):
                    ui.Button('Retry last action', clicked_fn=self.retry_action)
                    ui.Button('Save diagnostics', clicked_fn=self.export_diagnostics)
                self._label('Diagnostics stay on this computer. Review them before sharing.', 12, MUTED)
                ui.Button('Return to atrium / recover position', height=36, clicked_fn=lambda: (setattr(window, 'visible', False), self.request_room('lobby')))
                ui.Button('Start over', height=34,
                    clicked_fn=lambda: (setattr(window,'visible',False), self.queue_action('Resetting the lab',self.start_over)))
                self._label('Reset the experiments and return to the atrium. Saved files are preserved.', 12, MUTED)
                ui.Button('Back to exploring', height=38, name='Primary', clicked_fn=lambda: setattr(window, 'visible', False))

    def export_diagnostics(self):
        import json
        from .experience import VERSION
        from importlib.metadata import version, PackageNotFoundError
        packages = {}
        for name in ('ovrtx', 'ovphysx', 'ovstage', 'ovui', 'ovstorage', 'ovstream', *value('DIAGNOSTIC_PACKAGES', ())):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = 'missing'
        folder = self.runtime_root/'diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/('lab-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
        path.write_text(json.dumps({'app': VERSION, 'profile': self.settings.profile, 'distribution': identity(),
            'packages': packages, 'quality': self.settings.quality, 'room': self.current_room,
            'render_ms': self.native.last_render_ms, 'rendered_frames': self.native.frames,
            'display': 'GPU image upload' if self.viewport.gpu_enabled else 'CPU pixels',
            'gpu_presented_frames': self.viewport.gpu_frames,
            'presented_resolution': self.viewport.presented_size,
            'optional_exhibits': call('diagnostics', self) or {},
            'message': self.activity_message}, indent=2), encoding='utf-8')
        self.notify('Diagnostics saved in the runtime diagnostics folder.')
        return path

    def reset_room(self, key=None):
        from pxr import Gf
        from .physics_workshop import configure
        key = key or self.current_room
        g, e, w = self.gallery, self.gallery.extra, self.gallery.workshops
        if key == 'physics':
            self.moon = self.high_drop = False
            g.gravity_attr.Set(9.81)
            self.physics_speed = 1.
            self.comparison_run = None
            self.physics_results.clear()
            configure(self, 'drop', .38)
            from .physics_inspection import show, set_mass
            show(g, False)
            set_mass(self, 2.)
            self.panel_modes['physics'] = 'drop'
        elif key == 'rendering':
            w.lidar.reset()
            self.native.sensor_output = 'LdrColor'
            self.native.sensor_fisheye = False
            w.set_emitter('Rectangle')
            self.set_render_subject('lidar')
            w.car.paint('Carmine')
            w.car.finish('Gloss')
            self.roughness = w.car.roughness
            self.key_on = self.warm_light = True
            g.key_light.GetIntensityAttr().Set(3500.)
            g.key_light.GetColorAttr().Set(Gf.Vec3f(1., .77, .48))
            w.set_light(angle=-40., size=2., power=3500.)
            w.set_camera_zoom(32.)
            w.camera_angle(True)
            self.reference_pixels = None
            self.panel_modes['rendering'] = 'surface'
        elif key == 'scene':
            self.spin = False
            g.shared_speed = 1.
            g.shared_offset = g.shared_target = -2.4
            if not g.shared_loaded:
                g.set_shared(toggle_load=True)
            w.set_fleet(1)
            g.shared_translate.Set(Gf.Vec3d(*g.room_position('scene', (8.6, g.shared_floor_y, -5.7))))
        elif key == 'newton':
            g.newton_room.reset()
        elif key == 'package':
            g.package_room.reset()
        elif key == 'storage':
            if self.storage.task and not self.storage.task.done():
                raise RuntimeError('Wait for the revision operation to finish, then reset.')
            e.set_asset_variant(0, remember=False)
            e.undo_stack.clear()
            e.redo_stack.clear()
            e.storage_status = 'Working design reset. Saved revisions are preserved.'
            self.storage.guided_saved = self.storage.guided_restored = False
        elif key == 'interface':
            w.lift_cycle = False
            w.lift_direction = 1.
            w.ui_widgets = {'jog': True, 'quality': True, 'light': True}
            e.set_lift(.25)
            e.set_quality(0)
            if not e.work_light_on:
                e.toggle_work_light()
            self.panel_modes['interface'] = 'operate'
        elif key == 'streaming':
            w.stream_paused = False
            w.stream_fps = 5
            w.stream_target = -2.3
            if not w.stream_loaded:
                w.stream_command('cargo')
        else:
            call('reset_room', self, key)
        self.request_panel()

    def start_over(self):
        from .content import ROOMS
        # This resets experiments, never files written by the visitor.
        if self.storage.task and not self.storage.task.done():
            self.notify('Wait for the revision operation to finish, then start over.')
            return
        if self.gallery.package_room.busy:
            self.notify('Wait for the package operation to finish, then start over.')
            return
        for key in ROOMS:
            if self.settings.allowed(key):
                self.reset_room(key)
        if self.guide:self.guide.reset()
        self.go_to('lobby')
        self.notify('Back at the beginning. Saved revisions and images are preserved.')

    def compare_friction(self, friction):
        from .physics_workshop import configure
        configure(self, 'slide', friction)
        self.panel_modes['physics'] = 'slide'
        self.physics_speed = 1.
        self.comparison_run = {'friction': friction, 'start': None}
        self.toggle_physics()
        self.comparison_run['start'] = {p: self.gallery.position(p) for p in self.gallery.dynamic_paths}
        self.request_panel()

    def finish_comparison(self):
        run = self.comparison_run
        if not run or self.sim_time < 2.-1e-8:
            return
        import math
        distances = [math.dist((start[0], start[2]),
            (self.gallery.position(p)[0], self.gallery.position(p)[2])) for p, start in run['start'].items()]
        self.physics_results[run['friction']] = sum(distances)/len(distances)
        self.comparison_run = None
        self.running = False
        self.notify('Comparison complete: mean carton travel measured after 2 simulated seconds.')
        self.request_panel()

    def remember_view(self):
        if self.observer_view and self.observer_view.pixels is not None:
            self.reference_pixels = self.observer_view.pixels.copy()
            self.request_panel()
            self.notify('Inspection view saved for comparison. Now change the light or material.')
        else:
            self.notify('The inspection camera is still preparing. Try again when its image appears.')
