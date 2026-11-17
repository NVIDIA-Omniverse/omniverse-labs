# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-application component catalogue and concise workflow reference."""
from dataclasses import dataclass

import omni.ui as ui

from ..style import layout as fl, style_dialog
from ..camera import Camera, CameraController
from ..playback import Playback
from ..render_settings import RenderingSettings
from .camera_view import CameraView
from .icon_button import IconButton
from .render_settings_view import RenderingSettingsView
from .segmented import SegmentedControl
from .timeline import SoftGlassTimeline
from .goto_frame import GoToFrameDialog


@dataclass(frozen=True)
class Component:
    name: str
    purpose: str
    implementation: str


COMPONENTS = (
    Component("Capture browser", "Find recordings, navigate folders, filter formats, and validate open/save targets.", "CaptureFileDialog · OVUI TreeView and confirmation service"),
    Component("Object explorer", "Browse the scene hierarchy and follow selected objects into the inspector.", "PVD item models · OVUI TreeView"),
    Component("Property inspector", "Read captured values, class information, and object references.", "PVD data adapter · OVUI fields and labels"),
    Component("Physics viewport", "Inspect captured shapes, meshes and debug geometry, then pick an object to see its details.", "Native geometry adapter · OVUI image provider"),
    Component("Playback controls", "Step or jump to a frame, choose pre- or post-simulation frames, and work with clips.", "Playback · GoToFrameDialog · native presentation requests"),
    Component("Frame ruler", "Drag the playhead across the recording while staying within the active clip.", "SoftGlassTimeline · retained integer playhead model"),
    Component("Icon actions", "Recognize transport and workspace actions through clear, themed icons.", "IconButton · native button and public RasterImageProvider"),
    Component("Segmented choices", "Switch between related modes and see which choice is active.", "SegmentedControl · shared integer model"),
    Component("Rendering settings", "Adjust anti-aliasing, visible objects, planes, particles, and the viewport background.", "RenderingSettingsView · validated RenderingSettings"),
    Component("General settings", "Stage rendering, property display, array limits and profile color together; apply or discard the draft.", "GeneralSettingsView · atomic Preferences.set_many"),
    Component("Viewport camera", "Frame the scene or selection, adjust navigation, and choose a recorded camera when available.", "CameraView · CameraController"),
    Component("Profiler", "Select recorded events, compare context and thread task lanes, mark durations, and plot event history.", "ProfileView · native profile snapshots and bounded analysis jobs"),
    Component("Memory tables", "Compare recorded frame allocation activity and outstanding allocation totals; filter, sort and export rows.", "MemoryView · native memory parser and reusable diagnostic table"),
    Component("Error stream", "Inspect recorded legacy error codes, complete messages and source locations.", "ErrorStreamView · native structured records and CSV export"),
    Component("Measurement graphs", "Explore property, frame and profile histories with captured gaps and CSV export.", "PlotView · private native analysis sessions and OVUI plots"),
    Component("Resizable panels", "Adjust the browser and property panel widths to fit the capture being investigated.", "SplitGrip · retained OVUI input surface and persisted widths"),
    Component("Scene palette", "Edit actor, generated gizmo, and profile function colors; reload JSON changes while inspecting a capture.", "PaletteView · native OVUI color pickers and atomic palette storage"),
    Component("Shared controls", "Consistent fields, selectors, buttons, menus, dialogs, and feedback in both themes.", "Central shaded palette · OVUI default styles"),
)


class WidgetCatalogue:
    """Retained, theme-aware help/catalogue window; no native-session dependency.

    The control preview is explicitly local to this window. Workflow callbacks
    are supplied by the host application, so the catalogue never imports it.
    """
    def __init__(self, *, on_toggle_theme=None, on_open_capture=None, version="Development"):
        self.on_toggle_theme = on_toggle_theme
        self.on_open_capture = on_open_capture
        self.version = str(version)
        self.window = None
        self.preview_timeline = self.render_preview = self.camera_preview = None
        self.goto_preview = None
        self._preview_subscription = None
        self._preview_guard = False
        self.preview_playback = Playback()
        self.preview_playback.reset(241, ovd=True)
        self.preview_playback.clip.start, self.preview_playback.clip.stop = 40, 201
        self.preview_playback.seek(120)
        self.preview_settings = RenderingSettings()
        self.preview_camera = CameraController(Camera())

    @property
    def visible(self):
        return bool(self.window and self.window.visible)

    def show(self):
        if self.window is None:
            self._build()
        self.window.visible = True
        self.window.focus()

    def hide(self):
        if self.window:
            self.window.visible = False
        for preview in (self.render_preview, self.camera_preview, self.goto_preview):
            if preview is not None and preview.window is not None:
                preview.window.visible = False

    def destroy(self):
        for preview in (self.preview_timeline, self.render_preview, self.camera_preview, self.goto_preview):
            if preview is not None:
                preview.destroy()
        self.preview_timeline = self.render_preview = self.camera_preview = None
        self.goto_preview = None
        if self._preview_subscription is not None:
            self.preview_model.remove_value_changed_fn(self._preview_subscription)
            self._preview_subscription = None
        if self.window:
            self.window.destroy()
            self.window = None

    def _open_capture(self):
        self.hide()
        if self.on_open_capture:
            self.on_open_capture()

    def _preview_seek(self, model):
        if self._preview_guard:
            return
        self._preview_frame(self.preview_playback.seek(model.as_int))

    def _preview_frame(self, frame):
        self._preview_guard = True
        try:
            self.preview_model.set_value(frame)
        finally:
            self._preview_guard = False
        self.preview_frame_label.text = f"Frame {frame} · {self.preview_playback.label} · Clip 40 to 200"

    def _preview_mode(self, mode):
        self._preview_frame(self.preview_playback.set_mode(mode))

    def _preview_step(self, action):
        self._preview_frame(action())

    def _preview_rendering_changed(self, settings):
        self.preview_settings = settings
        self.preview_status.text = (f"Sample rendering choices applied · Geometry scale {settings.geometry_scale:g} · "
                                    f"Particle size {settings.particle_size:g}")

    def _show_render_preview(self):
        if self.render_preview is None:
            self.render_preview = RenderingSettingsView(self.preview_settings, self._preview_rendering_changed,
                sample_info=lambda: "Sample settings only. Open a recording to see the viewport result.")
        self.render_preview.show()

    def _show_goto_preview(self):
        if self.goto_preview is None:
            self.goto_preview = GoToFrameDialog(lambda frame, _:
                self._preview_frame(self.preview_playback.seek(frame, snap=False)))
        self.goto_preview.show(self.preview_playback, None)

    def _preview_camera_changed(self):
        camera = self.preview_camera.camera
        self.preview_status.text = f"Sample camera · {camera.fov:g}° field of view · {self.preview_camera.effective_speed:g} units/s"

    def _preview_camera_frame(self):
        self.preview_camera.camera.frame_bounds(((-4, -2, -3), (4, 2, 3)))
        self._preview_camera_changed()

    def _preview_selection_required(self):
        self.preview_status.text = "Selection actions become available when you select an object in a recording."
        if self.camera_preview is not None and self.camera_preview.window is not None:
            self.camera_preview.window.visible = False
        self.window.focus()

    def _preview_camera_reset(self):
        self.preview_camera.camera.reset()
        self._preview_camera_changed()

    def _show_camera_preview(self):
        if self.camera_preview is None:
            self.camera_preview = CameraView(self.preview_camera, self._preview_camera_changed,
                frame_all=self._preview_camera_frame, frame_selected=self._preview_selection_required,
                center_selected=self._preview_selection_required, track_selected=self._preview_selection_required,
                reset=self._preview_camera_reset)
        self.camera_preview.show()
        self.camera_preview.window.focus()

    def _switch_theme(self):
        if self.on_toggle_theme:
            self.on_toggle_theme()
        self.window.focus()

    def _build_preview(self):
        ui.Label("Try the controls", name="section", height=fl.pvd_control)
        ui.Label("This sample stays inside the guide. Your recording and saved preferences are unchanged.",
                 name="muted", word_wrap=True, height=fl.pvd_row)
        self.preview_model = ui.SimpleIntModel(self.preview_playback.frame)
        with ui.HStack(height=36, spacing=fl.pvd_spacing):
            self.preview_icons = []
            for icon, tooltip, action in (("first", "First frame in the sample clip", self.preview_playback.first),
                    ("previous", "Previous sample frame", lambda: self.preview_playback.step(-1)),
                    ("next", "Next sample frame", lambda: self.preview_playback.step(1)),
                    ("last", "Last frame in the sample clip", self.preview_playback.last)):
                self.preview_icons.append(IconButton(icon, width=32, height=32, name="ghost", tooltip=tooltip,
                    clicked_fn=lambda action=action: self._preview_step(action)))
            self.preview_modes = SegmentedControl(((0, "All", 42), (1, "Pre", 44), (2, "Post", 48)),
                value=int(self.preview_playback.mode), height=36, on_changed=self._preview_mode)
            ui.Spacer()
            ui.Label("Sample frames · 0 to 240", name="muted", width=180)
        self.preview_timeline = SoftGlassTimeline(self.preview_model, min=40, max=200, height=44,
            identifier="catalogue_sample_timeline")
        self.preview_timeline.set_domain(0, 240)
        self.preview_frame_label = ui.Label("", name="mono_small", height=fl.pvd_row)
        self._preview_subscription = self.preview_model.add_value_changed_fn(self._preview_seek)
        self._preview_frame(self.preview_playback.frame)
        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
            ui.Button("Rendering settings", name="outline", width=170, clicked_fn=self._show_render_preview)
            ui.Button("Camera controls", name="outline", width=150, clicked_fn=self._show_camera_preview)
            ui.Button("Go to frame", name="outline", width=fl.pvd_button_width, clicked_fn=self._show_goto_preview)
            ui.Spacer()
        self.preview_status = ui.Label("Drag the ruler, step through frames, or explore sample settings.",
            name="muted", word_wrap=True, height=fl.pvd_row)

    def _build(self):
        self.window = ui.Window("PVD workspace guide", width=fl.pvd_dialog_width,
            height=fl.pvd_dialog_height,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(lambda key, mod, pressed:
                                      self.hide() if key == 256 and not pressed else None)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.VStack(spacing=fl.pvd_spacing):
                    ui.Spacer(height=fl.pvd_spacing)
                    with ui.HStack():
                        ui.Spacer(width=fl.pvd_padding)
                        with ui.VStack(spacing=fl.pvd_spacing):
                            with ui.HStack(height=fl.pvd_control):
                                ui.Label("PVD on OVUI", name="brand")
                                ui.Button("Close", name="quiet", width=fl.pvd_button_width, clicked_fn=self.hide)
                            ui.Label(f"Workspace guide and component catalogue · {self.version}",
                                     name="muted", height=fl.pvd_row)
                            with ui.ScrollingFrame():
                                with ui.VStack(height=0, spacing=fl.pvd_padding):
                                    ui.Label("Inspect the recording, understand the frame.", name="section", height=fl.pvd_control)
                                    ui.Label("Open a capture, choose its scene, select an object, then move through the timeline to inspect how its values and geometry change.",
                                             word_wrap=True, height=fl.pvd_control * 2)
                                    self._build_preview()
                                    ui.Label(f"Workspace components · {len(COMPONENTS)} families", name="section", height=fl.pvd_control)
                                    for component in COMPONENTS:
                                        with ui.VStack(height=0, spacing=fl.pvd_spacing):
                                            ui.Label(component.name, name="section", height=fl.pvd_row)
                                            ui.Label(component.purpose, word_wrap=True, height=fl.pvd_row * 2)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label("Recording tools · View, compare, and navigate", name="muted")
                                if self.on_toggle_theme:
                                    ui.Button("Switch theme", width=fl.pvd_button_width, clicked_fn=self._switch_theme)
                                if self.on_open_capture:
                                    ui.Button("Open capture", name="primary", width=fl.pvd_button_width * 1.4, clicked_fn=self._open_capture)
                        ui.Spacer(width=fl.pvd_padding)
                    ui.Spacer(height=fl.pvd_spacing)
