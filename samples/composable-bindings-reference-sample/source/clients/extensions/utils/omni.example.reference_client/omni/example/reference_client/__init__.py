# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only Kit responsibilities: loading a bound USD scene, widgets and UI updates."""
import asyncio
import json
import omni.ext
import omni.kit.app
import omni.ui as ui
import omni.usd
import carb.settings
from pathlib import Path
from pxr import Gf, Usd, Tf
from omni.ui import scene as sc
from .gauges import MachineGauges, ViewportHud
from .navigation import ViewportNavigation
from .timeline import TimelinePanel
from .scene_colors import SceneColors
from composable_bindings.reference.playback import PlaybackController
from composable_bindings.reference.client import Client
from composable_bindings.reference.config import source_key
from composable_bindings.reference.scene_bindings import source_ids, match_bindings

ACTIVE = None  # Local diagnostics/test hook; not an integration API.
COLORS = {'alarm': 0xff5555ee, 'normal': 0xff77cc77, 'unknown': 0xffaaaaaa}


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        global ACTIVE
        self.dock_task = None
        self.scene_task = None
        self.client = None
        self.playback = self.timeline = None
        self.ext_id = ext_id
        self.window = ui.Window('Composable Bindings - Shared Python Client', width=410, height=560)
        self.viewport = self.scene_view = self.navigation = None
        self.paths = {}
        self.authored = []
        self.unmatched = []
        self.overlays = {}
        self.gauge_widgets = {}
        self.hud = self.hud_frame = None
        self.notice = None
        self.stage_dirty = True
        self.selection_dirty = True
        self.selection_subscription = None
        self.camera_signature = None
        self.scene_colors = None
        self.color_layout = None
        self.widgets = {}
        self.revision = -1
        self.layout = None
        settings = carb.settings.get_settings()
        settings.set('/app/window/title', 'Composable Bindings - Default Reference')
        path = settings.get('/exts/omni.example.reference_client/clientConfig')
        view = settings.get('/exts/omni.example.reference_client/viewConfig')
        try:
            self.client = Client(path, view or None).start()
            self.playback = PlaybackController(self.client,path).start()
            self.timeline = TimelinePanel(self.playback)
        except Exception as error:
            with self.window.frame:
                ui.Label('Start run-services.sh first.\n' + str(error), word_wrap=True)
            return
        self.selection_subscription = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
            self._selection_event, name='Reference local selection')
        self.scene_created = False
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._update, name='Reference client presentation')
        # Geometry and source-issued IDs are already authored in the USD file.
        root = Path(__file__).resolve().parents[8]
        stage_file = settings.get('/exts/omni.example.reference_client/stageFile') or str(root/'assets/production-line.usda')
        self.scene_task = asyncio.ensure_future(self._load_scene(stage_file))
        ACTIVE = self

    async def _load_scene(self, stage_file):
        try:
            context = omni.usd.get_context()
            success, error = await context.open_stage_async(str(stage_file))
            if not success:
                raise RuntimeError('Cannot open sample USD: ' + str(error))
            stage = context.get_stage()
            self.scene_colors = SceneColors(stage)
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport and stage.GetPrimAtPath('/World/Camera'):
                viewport.camera_path = '/World/Camera'
            self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._stage_changed, stage)
            self.scene_created = True
            self.stage_dirty = self.selection_dirty = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with self.window.frame:
                ui.Label('Scene loading failed: ' + str(error), word_wrap=True)

    def _selection_event(self, event):
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self.selection_dirty = True

    def _update_selection(self, stage):
        # Selection is local Kit state; never sent to the service or persisted.
        selected_roots = set()
        if stage:
            for path in omni.usd.get_context().get_selection().get_selected_prim_paths():
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    continue
                root = prim.GetPath()
                while prim and not prim.IsPseudoRoot():
                    if source_ids(prim.GetAssetInfo()):
                        root = prim.GetPath()
                        break
                    prim = prim.GetParent()
                selected_roots.add(root)
        # USD parent selection highlights descendants; mirror that for gauges.
        # Resolving a picked mesh first also makes selecting the line's base
        # equivalent to selecting its bound line assembly.
        for overlay in self.overlays.values():
            overlay.set_selected(any(overlay.prim.GetPath().HasPrefix(root) for root in selected_roots))
        self.selection_dirty = False

    def _stage_changed(self, notice, sender):
        if self.scene_colors and self.scene_colors.writing:
            return  # Color-only edits cannot change bindings or bounds.
        # Notice only marks dirty; resolve metadata and bounds on the next UI update.
        paths = list(notice.GetResyncedPaths()) + list(notice.GetChangedInfoOnlyPaths())
        if any(not str(p).startswith(('/World/Camera', '/World/Light')) for p in paths):
            self.stage_dirty = True

    def _attach_viewport(self):
        from omni.kit.viewport.utility import get_active_viewport_window
        self.viewport = get_active_viewport_window()
        if not self.viewport: return False
        with self.viewport.get_frame(self.ext_id):
            with ui.ZStack():
                self.scene_view = sc.SceneView()
                self.hud_frame = ui.Frame()
        self.viewport.viewport_api.add_scene_view(self.scene_view)
        self.dock_task = asyncio.ensure_future(self._dock())
        self.navigation = ViewportNavigation(self.viewport, self.ext_id, lambda: [path for paths in self.paths.values() for path in paths], self._overlay_points)
        return True

    async def _dock(self):
        for _ in range(3): await omni.kit.app.get_app().next_update_async()
        target = ui.Workspace.get_window(self.viewport.name)
        if target:
            self.window.dock_in(target, ui.DockPosition.RIGHT, .28)
            self.timeline.window.dock_in(target, ui.DockPosition.BOTTOM, .21)
            # File loading is asynchronous. Fit only after the bound stage and
            # its gauge layout exist, then allow docking/projection to settle.
            while not self.scene_created or not self.widgets:
                await omni.kit.app.get_app().next_update_async()
            for _ in range(30): await omni.kit.app.get_app().next_update_async()
            # Camera matrices can still describe the previous stage until the
            # renderer delivers frames (especially during first GPU startup).
            await self.viewport.viewport_api.wait_for_rendered_frames(2)
            self.navigation.frame_visible()

    def _overlay_points(self, paths):
        from .gauges import STEP
        world = self.viewport.viewport_api.view.GetInverse()
        right = world.TransformDir(Gf.Vec3d(1,0,0)).GetNormalized()
        up = world.TransformDir(Gf.Vec3d(0,1,0)).GetNormalized()
        points = []
        for overlay in self.overlays.values():
            path = str(overlay.prim.GetPath())
            if any(path == p or path.startswith(p + '/') or p.startswith(path + '/') for p in paths):
                # Reserve the first addition's headroom at startup; F fits larger stacks too.
                top = (max(3, len(overlay.gauges))-1)*STEP + .65
                for x in (-.7,.7):
                    for y in (-.6,top): points.append(overlay.anchor + overlay.scale*(x*right + y*up))
        return points

    def _reload(self):
        try:
            self.client.reload_view()
            self.layout = None
        except Exception as error:
            self.status.text = 'Invalid view configuration: ' + str(error)

    def _select(self, path):
        omni.usd.get_context().get_selection().set_selected_prim_paths([path], True)

    def _build(self, snapshot):
        self.widgets = {}
        paths = self.paths
        with self.window.frame:
            with ui.VStack(spacing=8):
                self.status = ui.Label(snapshot['status'], height=25)
                ui.Label('Configuration updates automatically', height=24)
                with ui.ScrollingFrame():
                    with ui.VStack(height=0, spacing=12):
                        for row in snapshot['rows']:
                            binding = source_key(row['binding'])
                            with ui.CollapsableFrame(row['binding']['label']):
                                with ui.VStack(height=0, spacing=5):
                                    ui.Label(row['binding']['source'] + ' / ' + row['binding']['id'], height=22)
                                    for path in paths.get(binding, []):
                                        ui.Button('Select ' + path, height=24,
                                                  clicked_fn=lambda p=path: self._select(p))
                                    with ui.HStack(height=20, spacing=12):
                                        ui.Label('Metric', width=ui.Fraction(1))
                                        ui.Label('Value', width=114, alignment=ui.Alignment.RIGHT_CENTER)
                                        ui.Label('Status', width=72)
                                    for display in row['displays']:
                                        spec = display['spec']
                                        with ui.HStack(height=24, spacing=12):
                                            title = ui.Label(spec['title'], width=ui.Fraction(1), elided_text=True)
                                            value = ui.Label('', width=114, alignment=ui.Alignment.RIGHT_CENTER)
                                            state = ui.Label('', width=72, alignment=ui.Alignment.LEFT_CENTER)
                                        self.widgets[(binding, spec['id'])] = title, value, state
                        for entry in self.unmatched:
                            with ui.CollapsableFrame('UNMATCHED: ' + entry['source'] + ' / ' + entry['id']):
                                with ui.VStack(height=0, spacing=5):
                                    ui.Label('Source ID is not in the active catalog', height=24,
                                             style={'color': 0xff55bbff})
                                    ui.Button('Select ' + entry['path'], height=24,
                                              clicked_fn=lambda p=entry['path']: self._select(p))
        self.scene_view.scene.clear()
        self.overlays = {}
        stage = omni.usd.get_context().get_stage()
        self.gauge_widgets = {}
        hud_items = []
        for row in snapshot['rows']:
            binding = source_key(row['binding'])
            world = [d for d in row['displays'] if d['spec'].get('placement','world') == 'world']
            hud_items.extend((binding,d) for d in row['displays'] if d['spec'].get('placement') == 'hud-top-left')
            if world:
                entry = next((e for e in snapshot['scene'] if source_key(e) == binding), {})
                for path in paths[binding]:
                    overlay = MachineGauges(self.scene_view, stage.GetPrimAtPath(path), world,
                        entry.get('gauge_offset',[0,0,0]), entry.get('gauge_scale',1.))
                    self.overlays[binding, path] = overlay
                    for key, gauge in overlay.gauges.items():
                        self.gauge_widgets.setdefault((binding, key), []).append(gauge)
        self.hud = ViewportHud(self.hud_frame,hud_items)
        for key, gauge in self.hud.gauges.items():
            self.gauge_widgets.setdefault(key, []).append(gauge)
        self.selection_dirty = True
        self.camera_signature = None

    def _update(self, _event):
        if not self.scene_view and not self._attach_viewport(): return
        snapshot = self.playback.snapshot()
        self.timeline.update(snapshot['timeline'])
        changed = snapshot['revision'] != self.revision
        if not self.scene_created: return
        stage = omni.usd.get_context().get_stage()
        structure_changed = self.stage_dirty
        if self.stage_dirty:
            self.authored = [(str(prim.GetPath()), prim.GetAssetInfo()) for prim in stage.Traverse()] if stage else []
            for overlay in self.overlays.values(): overlay.update_anchor()
            self.stage_dirty = False
            self.selection_dirty = True
            changed = True
        if changed:
            self.paths, self.unmatched = match_bindings(self.authored, [row['binding'] for row in snapshot['rows']])
            self.revision = snapshot['revision']
            snapshot = dict(snapshot, rows=[r for r in snapshot['rows'] if source_key(r['binding']) in self.paths])
            color_layout = json.dumps({'paths': [(key, value) for key, value in sorted(self.paths.items())],
                                       'owners': [source_key(entry) for entry in snapshot['scene'] if 'display_color' in entry]}, sort_keys=True)
            if self.scene_colors and (structure_changed or color_layout != self.color_layout):
                self.scene_colors.configure(self.paths, snapshot['scene'])
                self.color_layout = color_layout
            layout = [(self.paths[source_key(r['binding'])], r['binding'], [d['spec'] for d in r['displays']]) for r in snapshot['rows']]
            # Full specs include title/range/precision edits, not only kind/id.
            gauge_scene = [{key: value for key, value in entry.items() if key != 'display_color'} for entry in snapshot['scene']]
            signature = json.dumps({'rows': layout, 'scene': gauge_scene, 'unmatched': self.unmatched}, sort_keys=True)
            if signature != self.layout:
                self._build(snapshot)
                self.layout = signature
            self.status.text = snapshot['status'] + (f'; {len(self.unmatched)} UNMATCHED source IDs' if self.unmatched else '')
            if self.scene_colors:
                self.scene_colors.update(snapshot['rows'], stale=snapshot.get('stale', False))
                if self.scene_colors.diagnostics:
                    self.status.text += '; ' + '; '.join(self.scene_colors.diagnostics)
            for row in snapshot['rows']:
                binding = source_key(row['binding'])
                for display in row['displays']:
                    key = (binding, display['spec']['id'])
                    _title, value, state = self.widgets[key]
                    value.text = display['text']
                    state.text = display['state'].upper()
                    value.style = state.style = {'color': COLORS[display['state']]}
                    for gauge in self.gauge_widgets.get(key, []):
                        gauge.update(display)
        if self.selection_dirty:
            self._update_selection(stage)
        viewport = self.viewport.viewport_api
        camera_signature = (tuple(v for row in viewport.world_to_ndc for v in row), tuple(viewport.resolution))
        if camera_signature != self.camera_signature:
            self.camera_signature = camera_signature
            for overlay in self.overlays.values(): overlay.resize(viewport)

    def on_shutdown(self):
        global ACTIVE
        ACTIVE = None
        if self.scene_task: self.scene_task.cancel()
        self.scene_task = None
        if self.dock_task: self.dock_task.cancel()
        self.dock_task = None
        self.subscription = None
        self.selection_subscription = None
        if self.playback: self.playback.stop()
        self.playback = None
        if self.timeline: self.timeline.close()
        self.timeline = None
        if self.client:
            self.client.stop()
            self.client = None
        if self.notice: self.notice.Revoke()
        self.notice = None
        if self.scene_colors: self.scene_colors.close()
        self.scene_colors = None
        if self.navigation: self.navigation.close()
        self.navigation = None
        if self.scene_view:
            self.viewport.viewport_api.remove_scene_view(self.scene_view)
            self.scene_view.scene.clear()
            self.viewport.get_frame(self.ext_id).clear()
        self.overlays.clear()
        self.gauge_widgets.clear()
        self.hud = self.hud_frame = None
        self.scene_view = self.viewport = None
        self.widgets.clear()
        if self.window: self.window.destroy()
        self.window = None
