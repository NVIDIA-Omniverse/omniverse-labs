# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Python/OVUI workspace over the original C++ PVD engine."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
import asyncio
import tempfile
import time
import numpy as np

import omni.ui as ui
from ovui_widgets.common.selection import SelectionBus

from .adapters import PvdDataAdapter
from .models import ObjectTreeModel
from .renderer import ViewportRenderer
from .overlays import SELECTION_OVERLAYS
from .camera import CameraController, normalize_key
from .playback import Playback, FrameMode
from .interaction import InteractionState, TransparencyMode
from .row_actions import RowActionState
from .widgets.properties import PropertyInspector
from .widgets.object_browser import ObjectBrowser
from .widgets.splitter import SplitGrip
from .widgets.workspace_tabs import WorkspaceTabs, page_surface
from .widgets.activity_panel import ActivityPanel
from .widgets.recordings_view import RecordingsView
from .widgets.clips_view import ClipsView
from .widgets.float_slider import SoftGlassFloatSlider
from .widgets.log_scale import LogScaleSlider
from .widgets.profile_view import ProfileView
from .palette import ColorPalette,PROFILE_FUNCTION_DEFAULT
from .widgets.palette_view import PaletteView
from .widgets.diagnostic_tables import MemoryView,ErrorStreamView
from .error_stream import ErrorRequest,ErrorTasks
from .profiling import DiagnosticsTasks,DiagnosticsRequest,ProfileHistoryRequest,thread_name
from .exports import ExportRequest,export_capture
from .analysis import AnalysisTasks, HistoryRequest, can_plot_property
from .analysis_availability import AnalysisAvailability, AvailabilityContext
from .widgets.analysis_views import PlotView, MessagesView
from .workflow import CaptureWorkflow, wrap_connection_tooltip
from .desktop import OpenFolderRequest, DesktopError
from .ingestion import CaptureOpener, first_dropped_capture, submit_capture
from .lifecycle import run_standalone
from .widgets.file_browser import CaptureFileDialog
from .widgets.catalogue import WidgetCatalogue
from .widgets.about_view import AboutView
from .widgets.capture_selection import CaptureSelectionPanel
from .widgets.icon_button import icon_button, icon_url
from .widgets.segmented import SegmentedControl
from .widgets.timeline import SoftGlassTimeline
from .widgets.finding_track import FindingTrack
from .widgets.evidence import EvidenceView
from .evidence_binding import DisplayContext, EvidenceController, validate_marker_snapshot
from .ingestion import _identity
from .widgets.goto_frame import GoToFrameDialog
from .render_settings import DisplayMode
from .visual_reset import VisualReset, GIZMO_SCALE, RENDER_FIELDS, SELECTION_CATEGORIES
from .joint_scale import (JOINT_SCALE_DEFAULT, joint_scale_from_preferences,
                          validate_joint_scale, resolve_joint_base)
from .viewport_settings import (ViewportDisplaySettings, PREFERENCE_KEY as VIEWPORT_PREFERENCE_KEY,
                                MODE_FIELDS, TOGGLE_FIELDS, SCALE_FIELDS, OWNED_MASK, gizmo_value)
from .widgets.general_settings_view import GeneralSettingsView
from . import style, __version__
from .style import layout as fl
from .design_tokens import STAGE_GRID_RGBA, rgba as design_rgba


@contextmanager
def island_surface(padding=16):
    """Soft Glass panel surface with shared radius, padding and typography."""
    with ui.ZStack():
        ui.Rectangle(name="island",opaque_for_mouse_events=True)
        with ui.HStack():
            ui.Spacer(width=padding)
            with ui.VStack(spacing=0):
                ui.Spacer(height=padding)
                with ui.VStack(spacing=8):
                    yield
                ui.Spacer(height=padding)
            ui.Spacer(width=padding)


@contextmanager
def centered_slot(width,height):
    """Center a fixed-height control in the surrounding toolbar row."""
    with ui.VStack(width=width,spacing=0) as slot:
        ui.Spacer()
        with ui.Frame(height=height):yield slot
        ui.Spacer()


@contextmanager
def workspace_tab_rail():
    """The workspace rail has its own surface; dialog tabs use shared styles."""
    padding=fl.pvd_workspace_rail_padding
    with ui.ZStack():
        ui.Rectangle(name="workspace_rail",opaque_for_mouse_events=True)
        with ui.HStack(spacing=0):
            ui.Spacer(width=padding)
            with ui.VStack(spacing=0):
                ui.Spacer(height=padding)
                with ui.HStack(height=fl.pvd_workspace_rail_height-2*padding,
                               spacing=fl.pvd_workspace_rail_gap):yield
                ui.Spacer(height=padding)
            ui.Spacer(width=padding)


@contextmanager
def dialog_surface(window, title):
    """Themed OVUI window body shared by capture and analysis tools."""
    window.flags |= ui.WINDOW_FLAGS_NO_TITLE_BAR
    style.style_dialog(window)
    with window.frame:
        with ui.ZStack():
            ui.Rectangle(name="dialog")
            with ui.VStack(spacing=fl.pvd_spacing):
                with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                    ui.Spacer(width=fl.pvd_padding)
                    ui.Label(title,name="section")
                    ui.Button("Close",name="quiet",width=80,clicked_fn=lambda:setattr(window,"visible",False))
                    ui.Spacer(width=fl.pvd_padding)
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        yield
                    ui.Spacer(width=fl.pvd_padding)
                ui.Spacer(height=fl.pvd_padding)


class Application:
    def __init__(self, capture=None, width=1600, height=1000, theme=None, live=None):
        self.capture = capture
        self.capture_data_path = capture
        self.workflow = CaptureWorkflow()
        self.capture_selection_view = CaptureSelectionPanel(self.workflow.capture_selection)
        self.recordings_folder_request = None
        self.recordings_folder_status = ""
        self.width, self.height = width, height
        self.theme = theme or self.workflow.settings.get("theme", "dark")
        self.live_endpoint = live
        self.playback = Playback()
        self.analysis_tasks = AnalysisTasks()
        self.diagnostics_tasks=DiagnosticsTasks()
        self.error_tasks=ErrorTasks()
        self.analysis_views = []
        self.palette=ColorPalette()
        self.palette_view=PaletteView(self.palette,self.workflow.settings,on_changed=self._palette_changed)
        self.render_settings_view=GeneralSettingsView(self.workflow.settings,self._general_settings_changed,
            sample_info=lambda:self.renderer.sample_info() if self.renderer else "")
        self.render_settings=self.render_settings_view.accepted.rendering
        self.syncing_render_controls=False
        self.interaction=InteractionState()
        self.row_actions=RowActionState()
        self.replay_tree_rows=False
        self.interaction_replay=False
        self.selection_guard = False
        self.selected_record = None
        self.frame_summary = False
        self.find_inspection = None
        self.goto_frame_dialog = GoToFrameDialog(self._goto_frame)
        self.property_inspector=PropertyInspector(
            lambda identity:self.selection.publish([f"/objects/{identity}"],source="property"),
            self.plot_property,self.workflow.settings,on_request=self._request_presentation,
            on_error=self._report_error,on_summary=self.show_frame_summary)
        self.property_inspector.set_array_settings(self.render_settings_view.accepted.arrays)
        self.selection_history = []
        self.history_index = -1
        self.dialogs = []
        self.last_poll = 0.
        self.session = self.adapter = None
        self.presentation=None
        self.presentation_dirty=False
        self.pending_presentation=None
        self.pending_display=None
        self.pending_playback=None
        self.refreshing=False
        self.renderer = None
        self.navigation = None
        self.viewport_hovered = False
        self.viewport_keyboard_focus = False
        self.object_keyboard_focus = False
        self.viewport_menu=None
        self.viewport_menu_context=None
        self.viewport_menu_generation=0
        self.last_camera = 0.
        viewport=ViewportDisplaySettings.from_preferences(self.workflow.settings)
        self.overlay_mask=viewport.mask
        self.overlay_modes=viewport.modes
        self.gizmo_scale=viewport.gizmo_scale
        self.master_scale=viewport.master_scale
        self.overlay_scales=viewport.scales
        self.interaction.set_transparency(viewport.transparency_mode,viewport.transparency_amount)
        self.joint_scale=joint_scale_from_preferences(self.workflow.settings)
        self.joint_calibration=None
        self.tolerance_speed=1.
        self.scenes = []
        self.scene_index = 0
        self.scene_id = 0
        self.frame = 1
        self.playing = False
        self.direction = 1
        self.last_play = 0.
        self.dirty = True
        self.running = True
        self.status = "Open a capture to inspect its physics data."
        self.selection = SelectionBus()
        self.selection_subscription = self.selection.subscribe(self._selection_changed)
        self.object_model = ObjectTreeModel()
        self.object_model.set_array_settings(self.render_settings_view.accepted.arrays)
        self.capture_opener = CaptureOpener()
        self.analysis_availability = AnalysisAvailability()
        self.evidence_controller = EvidenceController()
        self.pending_evidence = None
        self.evidence_view = EvidenceView(self.evidence_controller,dialog_surface,on_import=self.show_import_evidence,
            on_go=self._go_evidence,on_scene=self._select_evidence_scene,on_clips=self.show_clips,on_changed=self._sync_evidence)
        self.analysis_menu_items = {}
        self.export_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="pvd-export")
        self.pending_export=None
        self.pending = None
        self.load_generation = 0
        self.pending_generation = 0
        self.load_dialog = None
        self.error_message = ""
        self.temporary = tempfile.TemporaryDirectory(prefix="ovd-ui-")
        self.inspector = None
        self.suppress_frame_callback = False
        self.mouse_anchor = None
        self.run_error = None
        self._close_started = False
        self._close_error = None
        self.closed = False

    def request_exit(self):
        self._close_viewport_menu()
        self.running = False

    def _report_error(self, message):
        self.error_message = self.status = str(message)
        self.status_label.text = self.status

    def _save_layout_preference(self, key, value):
        try:
            self.workflow.settings.set(key, value)
        except (OSError, ValueError) as error:
            self._report_error(f"Could not save layout preferences: {error}")

    def get_inspector_state(self):
        return {"capture":str(self.capture),"frame":self.frame,"theme":self.theme,
                "workspace":self.workspace.get_inspector_state() if hasattr(self,"workspace") else None,
                "activity":self.activity.get_inspector_state() if hasattr(self,"activity") else None,
                "recordings":self.recordings_view.get_inspector_state() if hasattr(self,"recordings_view") else None,
                "clips":self.clips_view.get_inspector_state() if hasattr(self,"clips_view") else None,
                "analyses":[view.get_inspector_state() for view in self.analysis_views
                            if hasattr(view,"get_inspector_state")],
                "instances":len(self.adapter.instances) if self.adapter else 0,
                "selected":self.selection.get_snapshot().paths(),"status":self.status,
                "selected_rows":[row.key for row in self.row_actions.selected],
                "tagged_objects":len(self.interaction.transparent),
                "renderer":self.renderer.device if self.renderer else None,
                "draw_objects":self.renderer.instance_count if self.renderer else 0,
                "draw_calls":self.renderer.draw_count if self.renderer else 0,
                "overlays":self.overlay_mask,
                "overlay_modes":{str(bit):self._overlay_mode(bit).name for bit,_ in SELECTION_OVERLAYS},
                "joint_scale":{"accepted":self.joint_scale,
                    "applied":self.presentation.joint_scale if self.presentation else None,
                    "base":self.presentation.joint_base_length if self.presentation else None,
                    "calibration":vars(self.joint_calibration) if self.joint_calibration else None},
                "camera":self.navigation.state() if self.navigation else None,
                "camera_input":{"hovered":self.viewport_hovered,"focused":self.window.focused,
                                "owns_keyboard":self.viewport_keyboard_focus,
                                "last_key":getattr(self,"last_camera_key",None)},
                "presentation":{"generation":self.presentation.generation,"serial":self.presentation.serial,
                                "frame":self.presentation.frame,"scene_id":self.presentation.scene_id,
                                "applied":self.presentation.request_applied} if self.presentation else None,
                "transparency":self.interaction.transparency_mode.name,
                "hidden":len(self.interaction.hidden),
                "layout":{"browser_width":self.browser_column.computed_width if hasattr(self,"browser_column") else 0,
                          "inspector_width":self.inspector_column.computed_width if hasattr(self,"inspector_column") else 0},
                "playback":{"mode":self.playback.mode.name,"playing":self.playback.playing,
                            "clip":self.playback.clip.name if self.playback.clip else None,
                            "range":[self.playback.clip.start,self.playback.clip.stop] if self.playback.clip else None},
                "live":{"port":self.workflow.port,"accept":self.workflow.accept_connections,
                        "follow":self.workflow.follow_live,"keep":self.workflow.keep_recordings,
                        "recordings":len(self.workflow.recordings.entries)}}

    def load(self, path):
        self.cancel_load(announce=False)
        source=Path(path).expanduser().resolve()
        self.load_generation+=1
        self.pending_generation=self.load_generation
        self.status=f"Loading {source.name}…"
        self.status_label.text=self.status
        self.playback.pause()
        self.pending=self.capture_opener.submit(source)
        self._show_load_progress(source)

    def _show_load_progress(self, source):
        self.load_dialog=ui.Window(f"Opening capture##pvd-load-{self.load_generation}",
            width=500,height=260,flags=ui.WINDOW_FLAGS_MODAL|ui.WINDOW_FLAGS_NO_SCROLLBAR)
        with dialog_surface(self.load_dialog,"Opening capture"):
            ui.Label(source.name,height=26,elided_text=True)
            self.load_stage=ui.Label("Preparing capture",name="muted",height=28)
            self.load_progress=ui.ProgressBar(height=8,visible=False,style={"ProgressBar":{
                "background_color":style.cl.pvd_field,"color":style.cl.pvd_accent,
                "secondary_color":style.cl.pvd_text,
                "border_radius":fl.pvd_field_radius,"font_size":fl.pvd_small_font}})
            self.load_detail=ui.Label("",name="muted",height=24)
            with ui.HStack(height=fl.pvd_control):
                ui.Spacer()
                ui.Button("Cancel",width=fl.pvd_button_width,clicked_fn=self.cancel_load)
        self.load_dialog.set_key_pressed_fn(
            lambda key,*_:self.cancel_load() if normalize_key(key)==256 else None)

    def _hide_load_progress(self):
        if self.load_dialog is not None:
            self.load_dialog.visible=False
            self.load_dialog.destroy()
            self.load_dialog=None

    def cancel_load(self, *, announce=True):
        job=self.pending
        if job is not None:
            job.cancel()
            self.pending=None
            self.load_generation+=1
        self._hide_load_progress()
        if job is not None and announce:
            self.status="Capture opening cancelled."
            self.status_label.text=self.status

    def _poll_load_progress(self):
        if not self.pending or self.load_dialog is None:
            return
        if not self.load_dialog.visible:
            self.cancel_load()
            return
        progress=self.pending.progress
        self.load_stage.text=progress.stage
        self.load_progress.visible=progress.fraction is not None
        self.load_progress.model.set_value(progress.fraction or 0.)
        self.load_detail.text=(f"{progress.processed:,} / {progress.total:,} bytes" if progress.total>1
            else "Cancel takes effect after this stage" if not progress.interruptible else "")

    def _drop_capture(self, event):
        path=first_dropped_capture(getattr(event,"mime_data",""))
        if path is not None:self.load(path)

    def _capture_payload(self):
        """Data for native readers; capture itself retains original archive provenance."""
        if self.session is self.workflow.display_session:
            return getattr(self.workflow,"display_data_path",None) or self.capture
        return getattr(self,"capture_data_path",None) or self.capture

    def _finish_load(self):
        job=self.pending
        self.pending=None
        # The header Close button may hide the dialog on the same frame the
        # worker finishes; honor that cancellation before taking its result.
        if self.load_dialog is not None and not self.load_dialog.visible:
            job.cancel()
        self._hide_load_progress()
        if self.pending_generation!=self.load_generation:
            job.cancel()
            return
        try:
            loaded=job.take_result()
        except CancelledError:
            self.status="Capture opening cancelled."
            self.status_label.text=self.status
            return
        except Exception as error:
            self.status=f"Unable to open capture: {error}"
            self.status_label.text=self.status
            return
        self.workflow.adopt_loaded(loaded)

    def _display(self, path, session):
        initial=None
        calibration=None
        new_playback=Playback()
        if session is not None:
            initial=session.presentation(tolerance_speed=None)
            if initial.ready and initial.request_applied:
                scenes=initial.scenes()
                scene=next((s for s in scenes if s.id==initial.scene_id),scenes[0])
                frame=new_playback.reset(scene.frame_count,initial.info().ovd)
                if initial.info().live:frame=self._live_display_frame(scene,initial.info(),new_playback.mode)
                initial=session.presentation(scene.id,frame,tolerance_speed=None)
            if initial.ready and initial.request_applied:
                from .presentation import default_scene_index
                scenes=initial.scenes()
                scene=scenes[default_scene_index(scenes,initial.geometry_snapshot())]
                frame=new_playback.reset(scene.frame_count,initial.info().ovd)
                if initial.info().live:frame=self._live_display_frame(scene,initial.info(),new_playback.mode)
                initial=session.presentation(scene.id,frame,tolerance_speed=None)
            if initial.ready and initial.request_applied:
                # Resolve only from this capture at its chosen scene/frame;
                # prior camera calibration cannot leak into an empty capture.
                calibration=resolve_joint_base(initial,PvdDataAdapter(initial),getattr(self,"render_settings",None))
                initial=session.presentation(initial.scene_id,initial.frame,overlay_mask=self.overlay_mask,
                    gizmo_scale=self.gizmo_scale,family_scales=self._viewport_settings().scales,
                    master_scale=getattr(self,"master_scale",1.),tolerance_speed=None,
                    **calibration.request(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT),ovd=initial.info().ovd,
                                          master_scale=getattr(self,"master_scale",1.)))
            if (not initial.ready or not initial.request_applied or not calibration.matches(initial) or
                    calibration.frame!=initial.frame):
                self.pending_display=(path,session)
                return False
        self.pending_display=None
        self._close_viewport_menu()
        self.mouse_anchor=None
        self.capture,self.session=path,session
        self.joint_calibration=calibration
        self.capture_data_path=(getattr(self.workflow,"display_data_path",None)
            if session is self.workflow.display_session else path) or path
        self.presentation=None
        self.presentation_dirty=False
        self.pending_presentation=None
        self.pending_playback=None
        self.pending_evidence=None
        self.adapter=None
        self.find_inspection=None
        self.frame_summary=False
        self.playback=new_playback
        self.interaction.begin_capture(getattr(session,"capture_key",id(session)))
        self.row_actions.sync(getattr(session,"capture_key",id(session)))
        self.object_browser.close_context_menu()
        self.selected_record=None
        self.selection_guard=True
        self.tree.clear_selection()
        self.selection_guard=False
        self.selection.clear()
        self.selection_history.clear()
        self.history_index=-1
        self.object_model.clear()
        self.object_browser.refresh()
        self.renderer.clear()
        self.scenes=[]
        self.scene_id=self.scene_index=self.frame=0
        if session is None:
            self.file_label.text="Physics viewport"
            self.summary_label.text="No capture loaded"
            self.viewport_info.text="Waiting for a capture"
            self.property_frame.rebuild()
            self.scene_frame.rebuild()
            self._sync_transport()
            self.dirty=True
            return True
        self.file_label.text=path.name if path else "Live capture"
        self._refresh(True,view=initial)
        self.scene_frame.rebuild()
        self.suppress_frame_callback=True
        self.mode_combo.model.set_value(int(self.playback.mode))
        self.suppress_frame_callback=False
        for item in self.object_model.roots:
            self.tree.set_expanded(item,True,False)
        return True

    @staticmethod
    def _live_display_frame(scene,info,mode):
        # A resampling segment can restart below the older timeline maximum.
        # Apply parity inside the completed prefix, never snap into its future.
        latest=max(0,min(scene.frame_count-1,scene.frame_count-1-max(0,info.live_max_frame-info.live_current_frame)))
        published=Playback()
        published.reset(latest+1,info.ovd)
        published.mode=mode
        return published.seek(latest)

    def _request_presentation(self):
        self.presentation_dirty=True

    def _inspection_ids(self):
        # An explicit Find choice can refer to an enumerated object outside its
        # lifetime. Keep that inspection intent separate from alive selection.
        target=getattr(self,"find_inspection",None)
        if target is not None:
            capture_key,identity=target
            if (self.session is not None and self.adapter is not None
                    and capture_key==getattr(self.session,"capture_key",id(self.session))
                    and identity in self.adapter.instances and identity in self.interaction.sticky):
                return [identity]
            self.find_inspection=None
        return [int(path.rsplit("/",1)[-1]) for path in self.selection.get_snapshot().paths()]

    def _find_object(self,identity):
        instance=self.adapter.instances.get(identity) if self.adapter is not None else None
        if instance is None:return False
        self.find_inspection=(getattr(self.session,"capture_key",id(self.session)),identity) if not instance.alive else None
        self.selection.publish([f"/objects/{identity}"],source="find")
        return True

    def _refresh(self, reset_camera=False, *, frame=None, scene_id=None, view=None, evidence_marker=None):
        if not self.session:
            return False
        if evidence_marker is None and getattr(self,"pending_evidence",None):self._cancel_evidence_go()
        if frame is not None or scene_id is not None:
            self.pending_presentation=(self.scene_id if scene_id is None else scene_id,self.frame if frame is None else frame)
        target_scene,target_frame=self.pending_presentation or (self.scene_id,self.frame)
        identities=self._inspection_ids()
        inspect_ids=identities if len(identities)==1 else []
        if getattr(self,"navigation",None) and self.navigation.tracking_id is not None:
            inspect_ids=list(dict.fromkeys([*inspect_ids,self.navigation.tracking_id]))
        reads=self.property_inspector.read_requests(self.adapter,identities if len(identities)==1 else [])
        record=self.selected_record
        from .array_layout import inspection_requests
        record_reads,records=inspection_requests(record)
        reads=list(dict.fromkeys([*reads,*record_reads]))
        request=dict(inspect_ids=inspect_ids,read_requests=reads,record_requests=records,
            overlay_mask=self.overlay_mask,gizmo_scale=self.gizmo_scale,
            master_scale=getattr(self,"master_scale",1.),
            family_scales=self._viewport_settings().scales,tolerance_speed=None,
            allow_unavailable_requests=True)
        if evidence_marker is not None:
            checks=evidence_marker.reads
            request["inspect_ids"]=list(dict.fromkeys([*inspect_ids,evidence_marker.instance_id,*[r.instance_id for r in checks]]))
            request["read_requests"]=list(dict.fromkeys([*reads,*[(r.instance_id,r.property_id,0,r.field_count*(4 if r.type_code==8 else 8)) for r in checks]]))
            request["include_evidence"]=True
        calibration=getattr(self,"joint_calibration",None)
        if calibration is not None:
            request.update(calibration.request(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT),
                           ovd=(view if view is not None else self.presentation).info().ovd,
                           master_scale=getattr(self,"master_scale",1.)))
        def publish(scene, frame):
            self.presentation_dirty=True
            try:
                return self.session.presentation(scene,frame,**request)
            except ValueError as error:
                if not request["read_requests"] or str(error)!="Presentation metadata/inspector data exceeds 128 MiB":raise
                # A later complete native frame may grow its core metadata.
                # Retry exactly once without optional windows, including the
                # calibrated scene/segment initialization publication below.
                self.property_inspector.reject_read_requests(request["read_requests"],
                    "This array window exceeds the available property data budget.")
                request["read_requests"]=[]
                return self.session.presentation(scene,frame,**request)
        if view is None:
            view=publish(target_scene,target_frame)
        self.presentation_dirty=not view.request_applied
        if not view.ready or not view.request_applied:
            self._sync_transport()
            return False
        calibration=getattr(self,"joint_calibration",None)
        if calibration is None or not calibration.matches(view):
            # Scene/segment adoption is an initialization boundary. Ordinary
            # seek/selection refreshes retain the base and make one publication.
            same_namespace=(calibration is not None and calibration.capture_key==view.capture_key and
                            calibration.segment_epoch==view.segment_epoch)
            calibration=resolve_joint_base(view,PvdDataAdapter(view),getattr(self,"render_settings",None),
                selected=self.interaction.selected if same_namespace else (),
                hidden=self.interaction.hidden if same_namespace else ())
            request.update(calibration.request(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT),ovd=view.info().ovd,
                                              master_scale=getattr(self,"master_scale",1.)))
            view=publish(view.scene_id,view.frame)
            self.presentation_dirty=not view.request_applied
            if (not view.ready or not view.request_applied or not calibration.matches(view) or
                    calibration.frame!=view.frame):
                self.presentation_dirty=True
                self._sync_transport()
                return False
        if evidence_marker is not None:
            reason=validate_marker_snapshot(view,evidence_marker)
            if reason:raise ValueError(reason)
        self.joint_calibration=calibration
        self.pending_presentation=None
        self.refreshing=True
        try:
            previous_clip_rows=((self.playback.index,[(c.name,c.start,c.stop) for c in self.playback.clips])
                                if hasattr(self,"clips_view") else None)
            if self.pending_playback is not None:
                self.playback=self.pending_playback
                self.pending_playback=None
            previous_view=self.presentation
            previous_scene=next((s for s in self.scenes if s.id==self.scene_id),None)
            self.presentation=view
            self.tolerance_speed=view.tolerance_speed
            scene_changed=self.scene_id!=view.scene_id
            previous_scenes=[(s.id,s.name) for s in self.scenes]
            self.scenes=view.scenes()
            self.scene_id=view.scene_id
            self.scene_index=next(i for i,s in enumerate(self.scenes) if s.id==view.scene_id)
            scene=self.scenes[self.scene_index]
            same_capture=previous_view is not None and previous_view.capture_key==view.capture_key
            displayed_segment_changed=(same_capture and previous_scene is not None and
                previous_scene.recording_segment!=scene.recording_segment)
            self.frame=view.frame
            if getattr(self,"viewport_menu_context",None) not in (None,self._viewport_context()):
                self._close_viewport_menu()
            if scene_changed:
                mode=self.playback.mode
                if same_capture and not displayed_segment_changed:self.playback.switch_scene(scene.frame_count,view.info().ovd)
                else:
                    self.playback.reset(scene.frame_count,view.info().ovd)
                    if same_capture:self.playback.mode=mode
            if displayed_segment_changed:
                self.interaction.clear_selection(reset_history=True)
                self.row_actions.select(())
                self.selection.clear()
                self.selection_history.clear()
                self.history_index=-1
                self.selected_record=self.find_inspection=None
                self.frame_summary=False
                self.object_browser.close_context_menu()
                self.mouse_anchor=None
                if getattr(self,"navigation",None):
                    self.navigation.stop()
                    self.navigation.manual_change()
                    self.navigation.captured=False
            live=self.session is self.workflow.live_session
            if live:
                # Ownership survives scrub publications and Playback deepcopies.
                # End-trim releases it; split transfers it to the resulting tail.
                for index,clip in enumerate(self.playback.clips):
                    if clip.live_tail and (index==0 or self.workflow.follow_live):
                        clip.stop=scene.frame_count
            if self.playback.clip and not self.playback.clip.start<=self.frame<=self.playback.clip.last:
                covering=next((i for i in reversed(range(len(self.playback.clips)))
                    if self.playback.clips[i].start<=self.frame<self.playback.clips[i].stop),None)
                if covering is None:
                    from .playback import Clip
                    self.playback.clips.append(Clip("Published capture",0,scene.frame_count,self.frame,live_tail=live))
                    covering=len(self.playback.clips)-1
                self.playback.index=covering
            self.playback.seek(self.frame,snap=False)
            self.renderer.scene_id=scene.instance_id
            if scene_changed or previous_scenes!=[(s.id,s.name) for s in self.scenes]:self.scene_frame.rebuild()
            if self.adapter is None:self.adapter=PvdDataAdapter(view)
            else:
                self.adapter.session=view
                self.adapter.reload()
            if self.palette is not None:
                class_names={identity:cls.name for identity,cls in self.adapter.classes.items()}
                if self.renderer.actor_class_names!=class_names:
                    self.renderer.set_palette(self.palette,class_names)
            self.selection_guard=True
            try:
                self.object_model.populate(self.adapter)
                self.row_actions.sync(self.object_model.capture_key,self.object_model.segment_epoch)
                self.object_browser.refresh()
            finally:
                self.selection_guard=False
            snapshot=view.geometry_snapshot()
            if getattr(self,"navigation",None):
                self.navigation.apply_frame(view,self.adapter,reset=reset_camera)
            if snapshot is not None:
                self.renderer.set_snapshot(snapshot,reset_camera)
            native_children,native_owners=self.adapter.ownership_graph()
            children=dict(native_children)
            owners=dict(native_owners)
            if snapshot is not None:
                for actor,shape,geometry,scene in zip(snapshot.object_ids,snapshot.shape_ids,snapshot.geometry_ids,snapshot.scene_ids):
                    actor,shape,geometry,scene=int(actor),int(shape),int(geometry),int(scene)
                    # Share small immutable native adjacencies. Index large
                    # ones once, and copy only memberships that need additions.
                    if shape>0:
                        for parent,child in ((actor,shape),(shape,geometry)):
                            if child<=0:continue
                            members=children.get(parent,())
                            if isinstance(members,tuple) and len(members)>8:
                                members=children[parent]=set(members)
                            if child not in members:
                                if not isinstance(members,set):members=children[parent]=set(members)
                                members.add(child)
                    if scene>0:
                        for identity in (actor,shape,geometry):
                            previous=owners.get(identity)
                            if identity>0 and (previous is None or scene not in previous):
                                owners[identity]=(previous or frozenset())|{scene}
            query_ids=[record.query_id for record in self.adapter.query_records()]
            self.interaction.update_frame([*self.adapter.instances,*query_ids],
                [*[i.id for i in self.adapter.instances.values() if i.alive],*query_ids],
                segment_epoch=view.segment_epoch,children=children,owners=owners,
                scene_ids=[scene.instance_id for scene in self.scenes])
            self._interaction_action(lambda:None)
            restored=[int(path.rsplit("/",1)[-1]) for path in self.selection.get_snapshot().paths()]
            if len(restored)==1 and restored[0] not in view.inspected_ids:
                # Seeking into an object's lifetime can restore sticky selection
                # after capture requests were collected from the previous frame.
                self._request_presentation()
            self.renderer.set_overlays(view.overlay_snapshot())
            scene_count=len(self.scenes)
            self.summary_label.text=f"OVUI workspace · {scene_count} scene{'s' if scene_count!=1 else ''} · {view.info().instance_count:,} objects"
            self.status=f"{self.capture.name if self.capture else 'Live capture'} · {len(self.adapter.instances):,} objects"
            if snapshot is not None and snapshot.diagnostics:
                self.status+=" · "+"; ".join(snapshot.diagnostics[:2])
            self.status_label.text=self.status
            self._sync_transport()
            self.property_frame.rebuild()
            if previous_clip_rows is not None and previous_clip_rows!=(
                    self.playback.index,[(c.name,c.start,c.stop) for c in self.playback.clips]):
                self._refresh_clips()
            self.dirty=True
            return True
        finally:
            self.refreshing=False

    def _sync_analysis_availability(self, *, request=False):
        controller=getattr(self,"analysis_availability",None)
        if controller is None:return
        context=AvailabilityContext.from_display(self._capture_payload(),self.presentation,self.playback.clip)
        previous=controller.context
        controller.sync(context)
        new_capture=context is not None and (previous is None or previous.capture_identity!=context.capture_identity)
        if request or new_capture:controller.request()
        self._update_analysis_controls()

    def _update_analysis_controls(self):
        controller=self.analysis_availability
        for name,action in (("messages_button","messages"),("frame_graph_button","frame_timing")):
            button=getattr(self,name,None)
            if button is not None:
                state=controller.state(action)
                button.enabled,button.tooltip=state.enabled,state.reason
        for action,(item,key) in getattr(self,"analysis_menu_items",{}).items():
            context=controller.context
            if context is None or context.key!=key:
                item.enabled=False
                item.tooltip="The displayed capture or frame changed. Reopen View."
            else:
                state=controller.state(action)
                item.enabled,item.tooltip=state.enabled,state.reason

    def _guard_analysis(self, action, key=None):
        if not hasattr(self,"analysis_availability"):
            self.analysis_availability=AnalysisAvailability()
        self._sync_analysis_availability()
        controller=self.analysis_availability
        if key is not None and (controller.context is None or controller.context.key!=key):
            self.status_label.text="The displayed capture or frame changed. Reopen View."
            return False
        state=controller.state(action)
        if not state.enabled:
            controller.request()
            state=controller.state(action)
            self._update_analysis_controls()
            self.status_label.text=state.reason
        return state.enabled

    def _invoke_analysis_menu(self, action, key, callback):
        self._sync_analysis_availability()
        context=self.analysis_availability.context
        if key is None or context is None or context.key!=key:
            self.status_label.text="The displayed capture or frame changed. Reopen View."
            return
        if self._guard_analysis(action,key):callback()

    def _evidence_context(self):
        view=self.presentation
        if self.session is None or view is None or self.session.capture_key!=view.capture_key:return DisplayContext()
        proof=getattr(self.workflow,"verified_payload",None) if self.session is self.workflow.display_session else None
        clip=self.playback.clip
        bounds=(self.playback.index,clip.start,clip.stop,int(self.playback.mode)) if clip else ()
        return DisplayContext(view.capture_key,view.segment_epoch,proof,view.info().ovd,view.info().live,self.scene_id,bounds)

    def show_evidence(self):
        self._sync_evidence();self.evidence_view.show()

    def show_import_evidence(self):
        directory=self.capture.parent if self.capture else Path.home()
        def selected(path):
            self.evidence_controller.import_file(path,self._evidence_context());self.show_evidence()
        dialog=CaptureFileDialog(selected,title="Import evidence JSON",initial_path=directory,
                                 extensions=(".json",),filter_label="Evidence JSON")
        self.dialogs.append(dialog);dialog.show()

    def _select_evidence_scene(self):
        marker=self.evidence_controller.selected_marker
        if marker is None:return
        index=next((i for i,s in enumerate(self.scenes) if s.id==marker.scene_id),None)
        if index is not None:self.scene_combo.model.get_item_value_model().set_value(index)

    def _evidence_action_key(self):
        controller=self.evidence_controller;context=self._evidence_context()
        return (context.binding_key,context.scene_id,context.clip,controller.revision,controller.selected_set,
                controller.selection,tuple(self.selection.get_snapshot().paths()))

    def _evidence_go_reason(self, marker=None, key=None):
        controller=self.evidence_controller;context=self._evidence_context()
        if key is not None and key!=self._evidence_action_key():return "Capture, selection, evidence, scene or clip changed. Select the finding again."
        if marker is None:return "Select a verified finding and capture role first."
        if context.binding_key!=controller.context.binding_key or marker!=controller.selected_marker:return "Evidence binding changed. Select the finding again."
        if context.payload is None or context.live or not context.ovd:return "Save and open a finalized OVD capture first."
        try:
            if _identity(context.payload.path.stat())!=context.payload.file_identity:return "The evidence payload changed. Reopen the capture."
        except OSError:return "The evidence payload is unavailable. Reopen the capture."
        if marker.scene_id!=self.scene_id:return "Select the finding's scene first."
        scene=next((s for s in self.scenes if s.id==marker.scene_id),None)
        if scene is None or not 0<=marker.frame<scene.frame_count:return "Evidence frame is outside the displayed scene."
        clip=self.playback.clip
        if clip is None or not clip.start<=marker.frame<clip.stop:return "Select a clip covering the evidence frame first."
        return ""

    def _cancel_evidence_go(self):
        if getattr(self,"pending_evidence",None) is not None:
            self.pending_evidence=None;self.pending_playback=None;self.pending_presentation=None;self.presentation_dirty=False

    def _go_evidence(self, key=None):
        marker=self.evidence_controller.selected_marker
        reason=self._evidence_go_reason(marker,key)
        if reason:self._report_error(reason);return False
        self._cancel_evidence_go()
        self.playback.pause()
        requested=deepcopy(self.playback)
        if requested.seek(marker.frame,snap=False)!=marker.frame:return False
        self.pending_evidence=(marker,self._evidence_action_key(),requested)
        return self._retry_evidence_go()

    def _retry_evidence_go(self):
        pending=getattr(self,"pending_evidence",None)
        if pending is None:return False
        marker,key,requested=pending
        reason=self._evidence_go_reason(marker,key)
        if reason:self._cancel_evidence_go();self._report_error(reason);return False
        old_scene,old_frame=self.scene_id,self.frame
        self.pending_playback=requested
        try:
            if not self._refresh(frame=marker.frame,scene_id=marker.scene_id,evidence_marker=marker):return False
        except (ValueError,IndexError,RuntimeError) as error:
            self._cancel_evidence_go()
            # Keep the private mutable reader aligned with the unchanged owned
            # display after a failed final identity/value check.
            try:self.session.presentation(old_scene,old_frame)
            except (ValueError,IndexError,RuntimeError):pass
            self._report_error(f"Evidence navigation rejected: {error}")
            return False
        self.pending_evidence=None
        self.find_inspection=None;self.selected_record=None;self.frame_summary=False
        self.selection.publish([f"/objects/{marker.instance_id}"],source="evidence")
        self._sync_evidence()
        return True

    def _select_evidence_marker(self, marker):
        controller=self.evidence_controller
        if marker not in controller.markers:return
        controller.select_set(marker.set_digest);controller.select(marker.finding_id,marker.role,marker.frame)
        self._sync_evidence()
        view=getattr(self,"evidence_view",None)
        if view is not None:view.show()

    def _evidence_color(self,marker):
        finding=next(f for f in self.evidence_controller.sets[marker.set_digest].evidence.findings if f.id==marker.finding_id)
        return ui.color(finding.data["evidence"]["color"] if finding.kind=="comparison" else "#D97706" if marker.status=="Warning" else "#1976D2")

    def _sync_evidence(self):
        controller=getattr(self,"evidence_controller",None)
        if controller is None:return
        context=self._evidence_context();controller.sync(context)
        track=getattr(self,"finding_track",None)
        if track is not None:
            scene=next((s for s in self.scenes if s.id==self.scene_id),None)
            clip=self.playback.clip
            track.update((0,max(0,scene.frame_count-1) if scene else 1),
                (clip.start,clip.last) if clip else (0,0),[m for m in controller.markers if m.scene_id==self.scene_id])
            visible=bool(track._markers)
            self.ruler_slot.height=ui.Pixel(56 if visible else 44)
            self.transport_controls.height=ui.Pixel(56 if visible else 44)
            self.transport_row.height=ui.Pixel(88 if visible else 76)
        view=getattr(self,"evidence_view",None)
        if view is not None:view.refresh(self._evidence_action_key(),self._evidence_go_reason(controller.selected_marker))

    def _sync_transport(self):
        self._sync_evidence()
        self._sync_analysis_availability()
        self._sync_connection_status()
        self._sync_contact_guidance()
        self.suppress_frame_callback=True
        self.frame_model.set_value(self.frame)
        self.mode_combo.model.set_value(int(self.playback.mode))
        self.frame_slider.min=self.playback.clip.start if self.playback.clip else 0
        self.frame_slider.max=self.playback.clip.last if self.playback.clip else 1
        scene=next((scene for scene in self.scenes if scene.id==self.scene_id),None)
        last=max(0,scene.frame_count-1) if scene else 1
        self.frame_slider.set_domain(0,last)
        self.frame_slider.enabled=bool(self.playback.clip)
        self.suppress_frame_callback=False
        self.play_button.image_url=icon_url("pause" if self.playback.direction==1 else "play")
        self.reverse_button.image_url=icon_url("pause" if self.playback.direction==-1 else "reverse")
        self.parity_button.text=self.playback.label
        # Reserve the capture's longest caption, including "Post" and padding.
        # Native Buttons otherwise expand their fixed parent to each new text
        # width, shifting the timeline and invalidating its retained ruler.
        self.parity_slot.width=ui.Pixel(max(72,48+8*len(str(last))))
        self.parity_button.enabled=self.playback.ovd
        self.mode_combo.enabled=self.playback.ovd

    def _sync_connection_status(self):
        label=getattr(self,"connection_label",None)
        if label is None:return
        connection=self.workflow.connection_status
        # Workflow promotion can precede a complete applied Presentation. The
        # background annotation must describe the view actually shown here.
        connection=replace(connection,background=self.session is not None and self.workflow.live_session is not None
                           and self.session is not self.workflow.live_session)
        if self.presentation is None:
            displayed="No capture displayed."
        else:
            info=self.presentation.info()
            mode="OVD" if info.ovd else "PVD"
            if self.session is self.workflow.live_session:
                mode+=" live capture"
            elif info.live:
                mode+=" recording (frozen)"
            else:
                mode+=" file"
            displayed=f"Displayed: {mode}."+(f"\n{self.capture.name}" if self.capture else "")
        tooltip=wrap_connection_tooltip(displayed+"\n"+connection.tooltip)
        # Shared muted typography and a separate label keep frame/capture/error
        # messages independent. Native observations are sampled only by workflow.
        if label.text!=connection.text:label.text=connection.text
        if label.tooltip!=tooltip:label.tooltip=tooltip
        self.accept_button.tooltip=tooltip+"\nOpen live capture controls."

    def seek(self, frame, *, snap=True):
        if not self.session or not self.scenes:
            return
        self._cancel_evidence_go()
        if self.session is self.workflow.live_session:
            self.workflow.follow_live=False
        requested=deepcopy(self.playback)
        frame=requested.seek(frame,snap=snap)
        self.pending_playback=requested
        return self._refresh(frame=frame)

    def _transport(self, action, *args):
        if not self.session:
            return
        self._cancel_evidence_go()
        requested=deepcopy(self.playback)
        frame=getattr(requested,action.__name__)(*args)
        self.pending_playback=requested
        if self.session is self.workflow.live_session:self.workflow.follow_live=False
        self._refresh(frame=frame)

    def _frame_jump_context(self):
        clip = self.playback.clip
        if not self.session or not self.scenes or clip is None:
            return None
        return (self.session.capture_key, self.scene_id, self.playback.index,
                clip.start, clip.stop, self.playback.ovd, int(self.playback.mode))

    def show_goto_frame(self):
        context = self._frame_jump_context()
        if context is None:
            self.status_label.text = "Open a capture with published frames first."
            return
        self._set_viewport_keyboard_focus(False)
        self.goto_frame_dialog.show(self.playback, context)

    def _goto_frame(self, frame, context):
        if context is None or context != self._frame_jump_context():
            raise ValueError("Capture, scene or clip changed. Reopen Go to frame.")
        # A frame jump stops playback; cancellation leaves playback untouched.
        self.playback.pause()
        self.seek(frame, snap=False)

    def _mode_changed(self, mode):
        if self.session and not self.suppress_frame_callback:
            self._transport(self.playback.set_mode,mode)

    def _contact_guidance(self):
        view=self.presentation
        show=bool(view is not None and view.ready and self.overlay_mask & 32
                  and view.info().ovd and view.frame>0 and view.frame%2==0)
        if not show:return False,None
        target=view.frame-1
        clip=self.playback.clip
        count=next((s.frame_count for s in self.scenes if s.id==view.scene_id),0)
        return True,target if clip and clip.start<=target<min(clip.stop,count) else None

    def _sync_contact_guidance(self):
        if getattr(self,"contact_hint",None) is None:return
        visible,target=self._contact_guidance()
        self.contact_hint.visible=visible
        self.contact_pre_button.enabled=target is not None
        self.contact_pre_button.tooltip=("Pause on the matching Pre-sim frame; keep All stepping, otherwise use Pre-sim."
            if target is not None else "The matching Pre-sim frame is outside the current clip.")

    def _go_contact_pre(self):
        _,target=self._contact_guidance()
        if target is None:return False
        self.playback.pause()
        requested=deepcopy(self.playback)
        if requested.mode==FrameMode.POST:requested.mode=FrameMode.PRE
        requested.seek(target,snap=False)
        self.pending_playback=requested
        if self.session is self.workflow.live_session:self.workflow.follow_live=False
        return self._refresh(frame=target,scene_id=self.presentation.scene_id)

    def _select_scene(self, model, _item):
        if not self.session:
            return
        index=model.get_item_value_model().as_int
        if index == self.scene_index:
            return
        self._cancel_evidence_go()
        scene=self.scenes[index]
        self.pending_playback=None
        if self.session is self.workflow.live_session:self.workflow.follow_live=False
        requested=deepcopy(self.playback)
        frame=requested.switch_scene(scene.frame_count,self.presentation.info().ovd)
        if not self._refresh(frame=frame,scene_id=scene.id):self.scene_frame.rebuild()

    def _tree_selection(self, items):
        if self.selection_guard:
            return
        items=[item for item in items if item is not None]
        self.row_actions.select(self.object_model.row_identities([item.key for item in items]))
        self.selected_record=next((i for i in reversed(items) if i and i.kind in ("array_element","unresolved")),None)
        ids=[f"/objects/{i.instance_id if i.instance_id is not None else i.query_id}"
             for i in items if i and (i.instance_id is not None or i.query_id is not None)]
        self.selection.publish(ids,source="objects")
        self._request_presentation()
        self.property_frame.rebuild()

    def _selection_changed(self,event):
        paths=event.snapshot.paths()
        if event.source!="frame-summary" and (paths or event.source!="interaction"):
            self.frame_summary=False
        if event.source not in ("interaction","find"):
            self.find_inspection=None
        if event.source!="objects":
            self.selected_record=None
        if not self.interaction_replay:
            self.interaction.select([int(path.rsplit("/",1)[-1]) for path in paths])
        preserve_rows=(event.source=="objects" or
                       event.source=="interaction" and self.replay_tree_rows)
        if hasattr(self,"tree"):
            self.selection_guard=True
            try:
                if preserve_rows:
                    items=self.object_model.restore_rows(self.row_actions.selected)
                    # Query/typed records are frame-owned. Restore their row
                    # metadata as well as selection after the next publication.
                    self.selected_record=next((item for item in reversed(items)
                                               if item.kind in ("array_element","unresolved")),None)
                else:
                    if hasattr(self, 'object_browser'):
                        self.object_browser.ensure_native_visible([int(p.rsplit('/',1)[-1]) for p in paths])
                    items=[self.object_model.canonical_item(int(p.rsplit('/',1)[-1])) for p in paths]
                    self.row_actions.select(self.object_model.row_identities(
                        [item.key for item in items if item is not None]))
                self.tree.selection=[item for item in items if item is not None]
                for item in (() if preserve_rows else items):
                    if item is not None and hasattr(self.object_model,"reveal_ancestors"):
                        self.object_model.reveal_ancestors(item,lambda parent:self.tree.set_expanded(parent,True,False))
                    else:
                        for root in self.object_model.roots:
                            if item in root.children:
                                self.tree.set_expanded(root,True,False)
            finally:
                self.selection_guard=False
        if self.renderer:
            self._sync_interaction()
            self._refresh_row_tags()
            self.property_frame.rebuild()
            self.dirty=True
        if not self.refreshing and event.source!="frame-summary":self._request_presentation()

    def show_frame_summary(self,capture_key=None):
        """Clear inspection intent without seeking or replacing the owned frame."""
        if self.adapter is None or self.presentation is None:return False
        if capture_key is not None and capture_key!=self.presentation.capture_key:return False
        self.frame_summary=True
        self.selection.publish([],source="frame-summary")
        return True

    def _sync_interaction(self):
        if not self.renderer:
            return
        state=self.interaction
        self.renderer.all_transparent=state.transparency_mode==TransparencyMode.ALL
        self.renderer.selected=set(state.selected)
        self.renderer.hidden=set(state.hidden)
        snapshot=self.renderer.snapshot
        if snapshot is not None:
            selected=set(state.selected)
            tagged=state.transparent
            mode=state.transparency_mode
            amount=min(.999,1-state.transparency_amount)
            alphas=[]
            for actor,shape,geometry in zip(snapshot.object_ids,snapshot.shape_ids,snapshot.geometry_ids):
                identities={int(actor),int(shape),int(geometry)}
                active=(mode==TransparencyMode.ALL
                    or mode==TransparencyMode.SELECTED and bool(identities & selected)
                    or mode==TransparencyMode.UNSELECTED and not identities & selected
                    or mode==TransparencyMode.TAGGED and bool(identities & tagged))
                alphas.append(amount if active else 1.)
                if not state.is_visible(int(actor)):
                    self.renderer.hidden.add(int(actor))
            new_alpha=np.asarray(alphas,dtype="f4")
            previous=self.renderer.instance_alpha
            # Identity participates in the renderer's appearance cache. Keep
            # unchanged owned values so the following draw can reuse uploads.
            if previous is None or not np.array_equal(previous,new_alpha):
                self.renderer.instance_alpha=new_alpha
        self.dirty=True

    def _row_tags(self,row):
        return (self.row_actions.is_hidden(row,self.interaction.hidden),
                self.row_actions.is_transparent(row,self.interaction.transparent))

    def _refresh_row_tags(self):
        self.object_browser.refresh_tags((frozenset(self.interaction.hidden),
            frozenset(self.interaction.transparent),self.row_actions.revision))

    def _object_action(self,action,scope=None):
        """Apply a frozen row action without changing selection or camera."""
        if action not in ("hide","show","hide_all","show_all","tag","untag","clear_tags"):
            raise ValueError("Unknown object action")
        transparency=action in ("tag","untag","clear_tags")
        if scope is None:
            scope=self.object_model.action_scope([row.key for row in self.row_actions.selected],
                action="transparency" if transparency else "visibility")
        if (not self.row_actions.accepts(scope) or
            action in ("hide_all","show_all") and scope.segment_epoch!=self.object_model.segment_epoch):
            self._report_error("Capture or recording segment changed. Reopen the object menu.")
            return False
        current=self.object_model.row_identities(scope.selected_rows)
        if (tuple(row.token for row in current)!=tuple(row.token for row in scope.selected_rows)
                or not set(scope.native_roots)<=self.interaction.alive):
            self._report_error("Object contents changed. Reopen the object menu.")
            return False
        if action=="hide_all":
            self.row_actions.hide_all(self.object_model.row_identities(self.object_model._specs))
            self.interaction.hide_all()
        elif action=="show_all":
            self.row_actions.show_all()
            self.interaction.show_all()
        elif action=="clear_tags":
            self.row_actions.clear_transparency()
            self.interaction.clear_transparency()
        else:
            expected="transparency" if transparency else "visibility"
            if scope.action!=expected:raise ValueError("Object action scope does not match the control")
            self.row_actions.apply(scope,action in ("show","tag"))
            if transparency:self.interaction.set_transparent(scope.native_roots,action=="tag")
            else:self.interaction.set_visible(scope.native_roots,action=="show")
        self._sync_interaction()
        self._refresh_row_tags()
        return True

    def _interaction_action(self, action, *args):
        if action==self.interaction.set_transparency:
            return self._set_transparency(*args)
        previous=self.interaction.sticky
        action(*args)
        self.interaction_replay=True
        self.replay_tree_rows=previous==self.interaction.sticky
        try:
            self.selection.publish([f"/objects/{identity}" for identity in self.interaction.selected],source="interaction")
        finally:
            self.interaction_replay=False
            self.replay_tree_rows=False
        # SelectionBus synchronously calls _selection_changed, which already
        # applies visibility and per-instance alpha for this same publication.

    def _refresh_overlays(self):
        self._request_presentation()

    def _viewport_settings(self):
        """Snapshot accepted choices; never serialize native IDs or calibration."""
        return ViewportDisplaySettings(
            **{name:self._overlay_mode(bit) for bit,name in MODE_FIELDS.items()},
            **{name:bool(self.overlay_mask & bit) for bit,name in TOGGLE_FIELDS.items()},
            **{name:getattr(self,"overlay_scales",{}).get(bit,self.gizmo_scale) for bit,name in SCALE_FIELDS.items()},
            transparency_mode=self.interaction.transparency_mode,
            master_scale=getattr(self,"master_scale",1.),
            transparency_amount=self.interaction.transparency_amount,gizmo_scale=self.gizmo_scale)

    def _sync_viewport_models(self, names):
        self._sync_contact_guidance()
        controls={**getattr(self,"transparency_controls",{}),**getattr(self,"view_option_controls",{})}
        if not controls:return
        settings=self._viewport_settings()
        mapping={**{name:bit for bit,name in (*MODE_FIELDS.items(),*TOGGLE_FIELDS.items())},
                 **{name:name for name in (*SCALE_FIELDS.values(),"master_scale")},
                 "transparency_mode":"transparency","transparency_amount":"amount","gizmo_scale":"gizmo_scale"}
        previous=getattr(self,"syncing_view_options",False)
        self.syncing_view_options=True
        try:
            for name in names:
                model=controls.get(mapping[name])
                if model is not None:
                    value=getattr(settings,name)
                    model.set_value(value*100 if name=="transparency_amount" else value)
        finally:self.syncing_view_options=previous

    def _set_viewport_settings(self, **changes):
        before=self._viewport_settings()
        try:
            settings=replace(before,**changes)
            if settings==before:
                self._sync_viewport_models(changes)
                return True
            validate_joint_scale(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT),master_scale=settings.master_scale)
            calibration=getattr(self,"joint_calibration",None)
            if calibration is not None:
                calibration.request(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT),ovd=self.presentation.info().ovd,
                                    master_scale=settings.master_scale)
            self.workflow.settings.set_many({VIEWPORT_PREFERENCE_KEY:settings.to_dict()})
        except (ValueError,OSError) as error:
            self._report_error(str(error))
            self._sync_viewport_models(changes)
            return False
        previous_mask=self.overlay_mask
        self.overlay_mask=(previous_mask & ~OWNED_MASK)|settings.mask
        for bit,mode in settings.modes.items():
            self.overlay_modes[bit]=mode
            if self.renderer:self.renderer.overlays.set_mode(bit,mode)
        self.gizmo_scale=settings.gizmo_scale
        self.master_scale=settings.master_scale
        self.overlay_scales=settings.scales
        self.interaction.set_transparency(settings.transparency_mode,settings.transparency_amount)
        if (before.transparency_mode,before.transparency_amount)!=(settings.transparency_mode,settings.transparency_amount):
            self._sync_interaction()
        if (previous_mask!=self.overlay_mask or self.overlay_mask & 64 and before.gizmo_scale!=settings.gizmo_scale
                or self.overlay_mask & (2|4|8|16|32|64|256) and before.master_scale!=settings.master_scale
                or any(self.overlay_mask & bit and before.scales[bit]!=value for bit,value in settings.scales.items())):
            self._refresh_overlays()
        self.dirty=True
        self._sync_viewport_models(changes)
        return True

    def _set_transparency(self, mode, amount=None):
        changes={"transparency_mode":mode}
        if amount is not None:changes["transparency_amount"]=amount
        return self._set_viewport_settings(**changes)

    def _set_overlay(self, bit, enabled):
        if bit in dict(SELECTION_OVERLAYS):
            return self._set_overlay_mode(bit,DisplayMode.ALL if enabled else DisplayMode.NONE)
        if bit in TOGGLE_FIELDS:
            return self._set_viewport_settings(**{TOGGLE_FIELDS[bit]:enabled})
        previous=self.overlay_mask
        self.overlay_mask=(self.overlay_mask|bit) if enabled else (self.overlay_mask & ~bit)
        if previous!=self.overlay_mask:self._refresh_overlays()

    def _overlay_mode(self,bit):
        return getattr(self,"overlay_modes",{}).get(bit,DisplayMode.ALL if self.overlay_mask & bit else DisplayMode.NONE)

    def _set_overlay_mode(self,bit,mode):
        if bit not in dict(SELECTION_OVERLAYS) or isinstance(mode,bool):
            raise ValueError("Unsupported overlay category or display mode")
        mode=DisplayMode(mode)
        if bit in MODE_FIELDS:
            return self._set_viewport_settings(**{MODE_FIELDS[bit]:mode})
        if self._overlay_mode(bit)==mode:return
        previous=self.overlay_mask
        self.overlay_modes[bit]=mode
        self.overlay_mask=self.overlay_mask & ~bit if mode==DisplayMode.NONE else self.overlay_mask | bit
        if self.renderer:self.renderer.overlays.set_mode(bit,mode)
        self.dirty=True
        # Native extraction is category-wide. Selection gates only filter the
        # already-owned rows; they never require another native publication.
        if previous!=self.overlay_mask:self._refresh_overlays()

    def _build_display_controls(self):
        self.syncing_view_options=False
        with ui.VStack(height=0,spacing=4):
            with ui.HStack(height=fl.pvd_control,spacing=8):
                ui.Label("Transparent",name="muted",width=80)
                combo=ui.ComboBox(int(self.interaction.transparency_mode),"None","All","Selected","Unselected","Tagged",
                    width=130,height=fl.pvd_control,explicit_hover=True,
                    tooltip="Which objects are transparent. Tagged uses the Trsp column; Vis independently hides subtrees.")
                amount=ui.SimpleFloatModel(self.interaction.transparency_amount*100)
                self.transparency_controls={"transparency":combo.model.get_item_value_model(),"amount":amount}
                combo.model.add_item_changed_fn(lambda m,_:self._view_option_changed(self._set_transparency,m.get_item_value_model().as_int))
                self.transparency_slider=SoftGlassFloatSlider(amount,min=0,max=100,
                    height=fl.pvd_control,suffix="%",identifier="transparency_amount",
                    tooltip="Transparency amount: 0% opaque · 100% invisible; applies to the chosen mode")
                amount.add_value_changed_fn(lambda m:self._view_option_changed(self._set_viewport_settings,transparency_amount=m.as_float/100))
                ui.Button("Display",name="ghost",width=68,clicked_fn=self.show_view_options,
                    explicit_hover=True,tooltip="Visibility actions, transparency tags, gizmos and scales")
            with ui.HStack(height=0,spacing=8,visible=False) as self.contact_hint:
                ui.Label("OVD contacts are shown on Pre-sim frames when recorded.",
                    name="muted",word_wrap=True,height=0)
                self.contact_pre_button=ui.Button("Go to Pre-sim",width=120,height=fl.pvd_control,
                    clicked_fn=self._go_contact_pre,explicit_hover=True)
        self._sync_contact_guidance()

    def show_view_options(self):
        if getattr(self,"view_panel",None) is None:return
        pending=self.pending_view_visible
        self.pending_view_visible=not (self.view_panel.visible if pending is None else pending)
        self._set_viewport_keyboard_focus(False)
        self.object_keyboard_focus=False

    def _build_display_panel(self):
        self.view_sliders=[]
        self.view_option_controls={}
        saved=self.workflow.settings.get("soft_glass_display_height",320)
        self.view_height=saved if type(saved) in (int,float) and 120<=saved<=600 else 320
        self.pending_view_visible=self.workflow.settings.get("soft_glass_display_open",False) is True
        self.view_panel=ui.VStack(height=self.view_height,visible=False,spacing=0,
            mouse_pressed_fn=self._workspace_pressed)
        with self.view_panel:
            self.view_grip=SplitGrip(self.view_panel,self._view_resized,axis="y",direction=-1,minimum=180,maximum=600)
            self.view_frame=ui.Frame()

    def _view_resized(self,height):
        self.view_height=height
        try:self.workflow.settings.set("soft_glass_display_height",height)
        except OSError as error:self._report_error(str(error))

    def _update_view_options(self):
        panel=getattr(self,"view_panel",None)
        if panel is None:return
        if self.pending_view_visible is not None:
            visible,self.pending_view_visible=self.pending_view_visible,None
            if visible and not self.view_option_controls:self._build_view_options()
            if visible!=panel.visible:
                panel.visible=visible
                try:
                    if self.workflow.settings.get("soft_glass_display_open",False)!=visible:
                        self.workflow.settings.set("soft_glass_display_open",visible)
                except OSError as error:self._report_error(str(error))
        if not panel.visible:return
        grip=self.view_grip
        # Reserve room for Properties; temporary window shrink must not replace
        # the preferred height or fight a divider drag awaiting native layout.
        grip.maximum=max(120,min(600,self.inspector_column.computed_height-180))
        grip.minimum=min(180,grip.maximum)
        if grip.pending_width is not None:
            grip.pending_width=max(grip.minimum,min(grip.maximum,grip.pending_width))
        grip.update()
        height=max(grip.minimum,min(grip.maximum,self.view_height))
        if grip.anchor is None and not grip.awaiting_layout and float(panel.height)!=height:
            panel.height=ui.Pixel(height)

    def _build_view_options(self):
        self.syncing_view_options=False
        with self.view_frame, island_surface(padding=10):
            with ui.HStack(height=fl.pvd_control):
                ui.Label("DISPLAY",name="eyebrow")
                ui.Button("Hide",name="quiet",width=48,clicked_fn=self.show_view_options)
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=fl.pvd_spacing):
                    ui.Label("Selection & visibility",name="section",height=28)
                    ui.Button("Scene palette",height=fl.pvd_control,clicked_fn=self.palette_view.show)
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        for label,action in (("Back",self.interaction.back),("Forward",self.interaction.forward)):
                            ui.Button(label,clicked_fn=lambda a=action:self._interaction_action(a))
                    ui.Button("Frame selected",height=fl.pvd_control,clicked_fn=self.frame_selected)
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Button("Hide selected",clicked_fn=lambda:self._object_action("hide"))
                        ui.Button("Show selected",clicked_fn=lambda:self._object_action("show"))
                        ui.Button("Isolate",clicked_fn=lambda:self._interaction_action(self.interaction.isolate))
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Button("Show all",clicked_fn=lambda:self._object_action("show_all"))
                        ui.Button("Hide all",clicked_fn=lambda:self._object_action("hide_all"))
                    ui.Label("Transparency",name="section",height=32)
                    ui.Label("Mode/amount: above the viewport. Trsp tags: Tagged mode only.",
                        name="muted",word_wrap=True)
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Button("Tag selected",clicked_fn=lambda:self._object_action("tag"))
                        ui.Button("Clear tags",clicked_fn=lambda:self._object_action("clear_tags"))
                    ui.Label("Debug overlays",name="section",height=32)
                    self._build_scale_control("master_scale","Master scale",
                        "Multiplier for all gizmo families, including joints. Objects, bounds and recorded geometry stay fixed.")
                    options=tuple((label,bit) for bit,label in SELECTION_OVERLAYS)+(
                             ("Scene queries",64),("Custom debug",128))
                    for label,bit in options:
                        with ui.HStack(height=28,spacing=fl.pvd_spacing):
                            if bit in dict(SELECTION_OVERLAYS):
                                ui.Label(label,width=ui.Fraction(1))
                                mode=ui.ComboBox(int(self._overlay_mode(bit)),"None","All","Selected","Unselected",
                                    width=ui.Fraction(1),height=fl.pvd_control,
                                    explicit_hover=True, tooltip=f"{label} for all, selected or unselected actors")
                                mode.model.add_item_changed_fn(lambda m,_,b=bit:
                                    self._view_option_changed(self._set_overlay_mode,b,m.get_item_value_model().as_int))
                                self.view_option_controls[bit]=mode.model.get_item_value_model()
                            else:
                                check=ui.CheckBox(width=22)
                                check.model.set_value(bool(self.overlay_mask & bit))
                                check.model.add_value_changed_fn(lambda m,b=bit:self._view_option_changed(self._set_overlay,b,m.as_bool))
                                self.view_option_controls[bit]=check.model
                                ui.Label(label)
                        if bit in SCALE_FIELDS:
                            self._build_scale_control(SCALE_FIELDS[bit],"Scale",
                                "OVD marker scale (minimum 0.2 before master); legacy PVD point pixels stay fixed."
                                if bit==32 else f"Size of {label.lower()} relative to the master; origins stay fixed.")
                    self._build_scale_control("gizmo_scale","Query / orbit",
                        "Query hit normals and orbit marker relative to master; recorded query distances stay fixed.")
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Label("Joint scale",width=100)
                        joint=SoftGlassFloatSlider(ui.SimpleFloatModel(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT)),
                            min=0,max=2,step=.02,precision=2,height=fl.pvd_control,identifier="joint_scale",
                            tooltip="Size of joint frames and limits. Use Joint frames to show or hide them.")
                        self.view_sliders.append(joint)
                        self.view_option_controls["joint_scale"]=joint.model
                        joint.model.add_value_changed_fn(lambda m:self._view_option_changed(self._set_joint_scale,m.as_float))

    def _build_scale_control(self, name, label, tooltip):
        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
            ui.Label(label,width=88)
            scale=LogScaleSlider(ui.SimpleFloatModel(getattr(self._viewport_settings(),name)),
                height=fl.pvd_control,identifier=name,tooltip=tooltip)
        self.view_sliders.append(scale)
        self.view_option_controls[name]=scale.model
        scale.model.add_value_changed_fn(lambda m:self._view_option_changed(self._set_gizmo_scale,m.as_float,name))

    def _destroy_view_options(self):
        for slider in getattr(self,"view_sliders",()):slider.destroy()
        self.view_sliders=[]
        for name in ("view_grip","view_panel"):
            value=getattr(self,name,None)
            if value is not None:value.destroy()
            setattr(self,name,None)
        self.view_frame=None
        self.pending_view_visible=None
        self.view_option_controls={}

    def _view_option_changed(self, callback, *args, **kwargs):
        if not self.syncing_view_options:callback(*args,**kwargs)

    def _set_gizmo_scale(self, value, name="gizmo_scale"):
        try:value=gizmo_value(value)
        except ValueError as error:
            self._report_error(str(error))
            self._sync_viewport_models((name,))
            return False
        return self._set_viewport_settings(**{name:value})

    def _set_joint_scale(self, value):
        try:
            value=validate_joint_scale(value,master_scale=getattr(self,"master_scale",1.))
            calibration=getattr(self,"joint_calibration",None)
            if calibration is not None:
                calibration.request(value,ovd=self.presentation.info().ovd,master_scale=getattr(self,"master_scale",1.))
            if value==getattr(self,"joint_scale",JOINT_SCALE_DEFAULT):return True
            self.workflow.settings.set_many({"joint_scale":value})
        except (ValueError,OSError) as error:
            self._report_error(str(error))
            model=getattr(self,"view_option_controls",{}).get("joint_scale")
            if model is not None:
                self.syncing_view_options=True
                try:model.set_value(getattr(self,"joint_scale",JOINT_SCALE_DEFAULT))
                finally:self.syncing_view_options=False
            return False
        self.joint_scale=value
        if self.overlay_mask & 16:self._request_presentation()
        self.dirty=True
        return True

    def _sync_visual_reset_controls(self):
        self.render_settings_view.set_settings(self.render_settings,fields=RENDER_FIELDS)
        if hasattr(self,"camera_view"):self.camera_view.sync_visual_reset()
        self._sync_viewport_models(("transparency_mode","transparency_amount"))
        if not getattr(self,"view_option_controls",None):return
        self.syncing_view_options=True
        try:
            values={"transparency":int(TransparencyMode.NONE),"amount":50.,"gizmo_scale":self.gizmo_scale,
                    "master_scale":self.master_scale,
                    **{name:self.overlay_scales[bit] for bit,name in SCALE_FIELDS.items()},
                    "joint_scale":self.joint_scale,
                    **{bit:int(DisplayMode.NONE) for bit in SELECTION_CATEGORIES},64:False}
            for key,value in values.items():
                model=self.view_option_controls.get(key)
                if model is not None:model.set_value(value)
        finally:self.syncing_view_options=False

    def reset_visual_settings(self):
        """Accept the measured visual reset without consuming unrelated drafts."""
        if self.navigation is None:return False
        try:
            candidate=VisualReset.prepare(self.render_settings,self.navigation.camera,self.overlay_mask)
            viewport=self._viewport_settings().visual_reset(GIZMO_SCALE)
            calibration=getattr(self,"joint_calibration",None)
            if calibration is not None:calibration.request(JOINT_SCALE_DEFAULT,ovd=self.presentation.info().ovd)
            self.workflow.settings.set_many({"rendering":candidate.rendering.to_preferences(),"joint_scale":JOINT_SCALE_DEFAULT,
                                            VIEWPORT_PREFERENCE_KEY:viewport.to_dict()})
        except (ValueError,OSError) as error:
            self._report_error(str(error))
            return False
        recapture=(self.overlay_mask!=candidate.overlay_mask or
                   bool(candidate.overlay_mask & (2|4|8|16|32|64|256)) and getattr(self,"master_scale",1.)!=1. or
                   bool(candidate.overlay_mask) and self.gizmo_scale!=GIZMO_SCALE or
                   any(candidate.overlay_mask & bit and value!=GIZMO_SCALE for bit,value in self._viewport_settings().scales.items()) or
                   bool(candidate.overlay_mask & 16) and getattr(self,"joint_scale",JOINT_SCALE_DEFAULT)!=JOINT_SCALE_DEFAULT)
        candidate.apply_camera(self.navigation)
        self.overlay_mask=candidate.overlay_mask
        self.gizmo_scale=GIZMO_SCALE
        self.master_scale=1.
        self.overlay_scales=viewport.scales
        self.joint_scale=JOINT_SCALE_DEFAULT
        for bit in SELECTION_CATEGORIES:
            self.overlay_modes[bit]=DisplayMode.NONE
            if self.renderer:self.renderer.overlays.set_mode(bit,DisplayMode.NONE)
        self.interaction.set_transparency(TransparencyMode.NONE,.5)
        self._render_settings_changed(candidate.rendering)
        self._sync_interaction()
        self._sync_visual_reset_controls()
        if recapture:self._request_presentation()
        self.dirty=True
        return True

    def _key_pressed(self,key,modifiers,pressed):
        self.last_camera_key=(key,modifiers,pressed)
        key=normalize_key(key)
        chord=modifiers & 15
        if pressed and key==256 and getattr(self,"mouse_anchor",None):
            self.mouse_anchor=None
            self.navigation.stop()
            return
        if (key in (257,335) and self.window.focused
                and self.object_browser.search_key(key,modifiers,pressed)):return
        if pressed and self.window.focused and chord==4 and key==70:
            self.show_find_profile()
            return
        if pressed and self.window.focused and chord==2 and key==70:
            self._set_viewport_keyboard_focus(False)
            self.object_browser.show_find()
            return
        if pressed and self.window.focused and chord==2 and key==71:
            self.show_goto_frame()
            return
        if pressed and self.window.focused and chord==3 and key==82:
            # Original QAction WindowShortcut also works from toolbar buttons
            # and main-window editors. OVUI's broad capture flag covers both.
            self.reset_visual_settings()
            return
        if pressed and self.window.focused and chord==3 and key==83:
            # Settings has the same original QAction WindowShortcut scope.
            self.render_settings_view.show()
            return
        if pressed and key==258:
            self._set_viewport_keyboard_focus(False)
            self.object_keyboard_focus=False
        if (pressed and self.window.focused and getattr(self,"object_keyboard_focus",False)
                and key==72 and chord in (0,1)):
            self._object_action("show_all" if chord==1 else "hide")
            return
        activity=getattr(self,"activity",None)
        if activity is not None and activity.key(key,modifiers,pressed):return
        workspace=getattr(self,"workspace",None)
        if workspace is not None and workspace.key(key,modifiers,pressed):return
        active=self.window.focused and self.viewport_keyboard_focus
        # ImGui also claims keyboard input for arrow navigation. A viewport
        # click grants ownership until another surface, Tab, or focus loss.
        analysis_navigation=workspace is not None and workspace.navigation_focused
        editing=bool(modifiers & ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD) and not (active or analysis_navigation)
        if self.navigation and self.navigation.key(key,modifiers,pressed,active=active,editing=editing):
            if pressed:self.dirty=True
            return
        if not pressed or not self.window.focused or editing:
            return
        # Match all public modifier bits, including Super. In particular the
        # original Trim start and Settings chords must never become Save as.
        if chord==2 and key==79:self.show_open()
        elif chord==2 and key==78:self.new_capture()
        elif chord==2 and key==83:self.show_save()
        elif chord==6 and key in (89,83,69):
            action={89:self.playback.split,83:self.playback.trim_start,69:self.playback.trim_end}[key]
            self._clip_action(action)
        elif not active:return
        elif chord==4 and key==82:self.reset_camera()
        elif chord==4 and key==84:self.track_selected()
        elif chord==4 and key==67:self.center_selected()
        elif chord==4 and key==90:self.frame_selected()
        elif chord==2 and key==90:self.frame_all()
        elif chord==0 and key==32:
            if self.playback.playing:self.playback.pause();self._sync_transport()
            else:self.toggle_play()
        elif chord==0 and key==70:self.frame_selected()
        elif chord in (0,1) and key==72:self._object_action("show_all" if chord==1 else "hide")
        elif chord==2 and key==262:self._transport(self.playback.step,1)
        elif chord==2 and key==263:self._transport(self.playback.step,-1)
        elif chord==0 and key==268:self._transport(self.playback.first)
        elif chord==0 and key==269:self._transport(self.playback.last)

    def _build_properties(self):
        identities=self._inspection_ids()
        record=self.selected_record
        if identities and identities[-1]<0:
            record=self.object_model.canonical_item(identities[-1])
        rows=self.object_model.restore_rows(self.row_actions.selected) if not identities else []
        self.property_inspector.build(self.adapter,identities,record,
            row_titles=tuple(item.values[0].as_string for item in rows),
            summary=getattr(self,"frame_summary",False),retain=True)

    def show_property_options(self):
        self.render_settings_view.show("properties")

    def _general_settings_changed(self,settings):
        """Adopt a complete durable settings transaction without moving the camera."""
        self._render_settings_changed(settings.rendering)
        arrays_changed=self.property_inspector.set_array_settings(settings.arrays)
        group_changed=self.object_model.page_size!=settings.arrays.group_size
        self.selection_guard=True
        try:
            items=self.object_model.set_array_settings(settings.arrays,selected_rows=self.row_actions.selected)
            if group_changed:
                self.object_browser.close_context_menu()
                self.row_actions.select(self.object_model.row_identities([item.key for item in items]))
                self.object_browser.restore_projection_selection(items)
                self.selected_record=next((item for item in reversed(items) if item.kind in ("array_element","unresolved")),None)
        finally:
            self.selection_guard=False
        self.property_frame.rebuild()
        if arrays_changed:self._request_presentation()
        self._palette_changed()

    def _filter(self,model):
        self.object_model.filter=model.as_string

    def toggle_theme(self):
        self.theme="light" if self.theme=="dark" else "dark"
        style.set_theme(self.theme)
        self.workflow.settings.set("theme",self.theme)
        self.theme_button.tooltip="Switch to dark mode" if self.theme=="light" else "Switch to light mode"
        self.dirty=True
        if getattr(self,"finding_track",None):self.finding_track.restyle()
        if getattr(self,"evidence_view",None):self.evidence_view.refresh(self._evidence_action_key(),self._evidence_go_reason(self.evidence_controller.selected_marker),force=True)

    def _palette_changed(self):
        self.dirty=True
        value=self.workflow.settings.get("ProfileFunctionColor",PROFILE_FUNCTION_DEFAULT)
        self.render_settings_view.set_profile_color(value)
        self.palette_view.sync_profile_color(value)
        for view in self.analysis_views:
            if isinstance(view,ProfileView):view.set_function_color(value)

    def _render_settings_changed(self, settings):
        self.render_settings=settings
        self.render_settings_view.settings=settings
        if self.renderer:
            self.renderer.wireframe=False
            self.renderer.set_settings(settings)
        if hasattr(self,"shading_control"):
            self.syncing_render_controls=True
            try:self.shading_control.model.set_value(int(settings.wireframe) if settings.wireframe in (0,1) else -1)
            finally:self.syncing_render_controls=False
        self.dirty=True

    def toggle_play(self, direction=1):
        self.playback.toggle(direction)
        if self.session is self.workflow.live_session:
            self.workflow.follow_live=False
        self._sync_transport()

    def toggle_wireframe(self):
        self.shading_control.model.set_value(int(self.render_settings.wireframe==0))

    def _set_wireframe(self, mode):
        if self.syncing_render_controls:return
        settings=replace(self.render_settings,wireframe=int(mode))
        try:self.workflow.settings.set("rendering",settings.to_preferences())
        except OSError as error:
            self._render_settings_changed(self.render_settings)
            self.error_message=f"Could not save rendering settings: {error}"
            self.status_label.text=self.error_message
            return
        self._render_settings_changed(settings)
        self.render_settings_view.set_settings(settings,fields=("wireframe",))

    def frame_all(self):
        if self.navigation:self.navigation.manual_change()
        self.renderer.frame_all()
        self.dirty=True

    def frame_selected(self):
        if self.navigation:self.navigation.manual_change()
        self.renderer.frame_all(selected=True)
        self.dirty=True

    def center_selected(self):
        if self.navigation:self.navigation.manual_change()
        self.renderer.frame_all(selected=True,center_only=True)
        self.dirty=True

    def reset_camera(self):
        self.navigation.manual_change()
        self.navigation.camera.reset()
        self.dirty=True

    def track_selected(self):
        if self.navigation.tracking_id is not None:
            self.navigation.manual_change()
            self._request_presentation()
            self.status_label.text="Camera tracking stopped."
            if hasattr(self,"camera_view"):self.camera_view.refresh_tracking()
            return
        identities=[int(path.rsplit("/",1)[-1]) for path in self.selection.get_snapshot().paths()]
        target=next((identity for identity in reversed(identities)
                     if self.adapter and (instance:=self.adapter.instances.get(identity)) is not None
                     and instance.alive and self.adapter.class_name(identity)=="PxRigidDynamic"),None)
        self.navigation.manual_change()
        self.navigation.captured=False
        self.navigation.tracking_id=target
        if target is None:
            self.status_label.text="Select a dynamic rigid body to track its recorded center of mass."
        else:
            self._request_presentation()
            self.status_label.text=f"Tracking body {target}; sleeping frames retain the camera position."
        if hasattr(self,"camera_view"):self.camera_view.refresh_tracking()

    def show_camera_options(self):
        from .widgets.camera_view import CameraView
        if not hasattr(self,"camera_view"):
            self.camera_view=CameraView(self.navigation,self._camera_changed,
                frame_all=self.frame_all,frame_selected=self.frame_selected,center_selected=self.center_selected,
                track_selected=self.track_selected,reset=self.reset_camera)
            self.dialogs.append(self.camera_view)
        self.camera_view.show()

    def _camera_changed(self):
        if self.presentation:
            self.navigation.apply_frame(self.presentation,self.adapter)
        self.dirty=True

    def _viewport_hovered(self,hovered):
        self.viewport_hovered=hovered


    def _set_viewport_keyboard_focus(self,focused):
        self.viewport_keyboard_focus=focused
        if focused:
            self.object_keyboard_focus=False
            if hasattr(self,"workspace"):self.workspace.blur()
        # ImGui otherwise also navigates buttons with the same arrow input.
        # This window flag only suppresses navigation while the camera owns it;
        # text fields/tree regain ordinary keyboard handling on their next click.
        no_nav_inputs=1<<16
        flags=getattr(self.window,"flags",0)
        self.window.flags=(flags|no_nav_inputs) if focused else (flags&~no_nav_inputs)
        if not focused and self.navigation:self.navigation.stop()

    def _window_focused(self,focused):
        if not focused:
            self.object_keyboard_focus=False
            if hasattr(self,"workspace"):self.workspace.blur()
        if not focused and self.navigation:
            self._set_viewport_keyboard_focus(False)
            self.mouse_anchor=None

    def _focus_objects(self):
        self._set_viewport_keyboard_focus(False)
        self.object_keyboard_focus=True
        if hasattr(self,"workspace"):self.workspace.blur()
        if hasattr(self,"activity"):self.activity.blur()

    def _workspace_pressed(self,x,y,button,modifiers):
        if not self.window.focused or getattr(self,"mouse_anchor",None):return
        viewport=getattr(self,"viewport_region",self.viewport)
        inside=(viewport.screen_position_x<=x<viewport.screen_position_x+viewport.computed_width
                and viewport.screen_position_y<=y<viewport.screen_position_y+viewport.computed_height)
        workspace=getattr(self,"workspace",None)
        if workspace is not None:
            inside=inside and workspace.active=="render"
            workspace.blur()
            host=workspace.pages[workspace.active].host
            frame=host.frame
            host.keyboard_focus=(workspace.active!="render" and frame.screen_position_x<=x<frame.screen_position_x+frame.computed_width
                and frame.screen_position_y<=y<frame.screen_position_y+frame.computed_height)
        activity=getattr(self,"activity",None)
        if activity is not None:
            activity.blur()
            host=activity.pages[activity.active].host
            frame=host.frame
            host.keyboard_focus=(host.visible and frame.screen_position_x<=x<frame.screen_position_x+frame.computed_width
                and frame.screen_position_y<=y<frame.screen_position_y+frame.computed_height)
            view=activity.pages[activity.active].view
            if view is not None and hasattr(view,"focus_at"):view.focus_at(x,y)
        self._set_viewport_keyboard_focus(inside)
        tree=self.tree
        self.object_keyboard_focus=(tree.screen_position_x<=x<tree.screen_position_x+tree.computed_width
            and tree.screen_position_y<=y<tree.screen_position_y+tree.computed_height)

    def _mouse_pressed(self,x,y,button,modifiers):
        if self.mouse_anchor and self.mouse_anchor[2]!=button:return
        self.mouse_double_clicked=False
        if not self._viewport_mouse_available():
            self.mouse_anchor=None
            return
        self._set_viewport_keyboard_focus(True)
        self.viewport_hovered=True
        self.mouse_anchor=(x,y,button)
        self.mouse_origin=(x,y)
        self.mouse_capture_context=self._viewport_context()
        self.mouse_modifiers=modifiers
        self.mouse_dragged=False

    def _mouse_double_clicked(self,x,y,button,modifiers):
        if self.mouse_anchor and self.mouse_anchor[2]==button==0 and not modifiers & 15:
            self.mouse_double_clicked=True

    def _mouse_released(self,x,y,button,*_):
        anchor=self.mouse_anchor
        if not anchor or anchor[2]!=button:return
        self.mouse_anchor=None
        if (not self._viewport_mouse_available()
                or self.mouse_capture_context!=self._viewport_context()):return
        dx,dy=x-self.mouse_origin[0],y-self.mouse_origin[1]
        if not self.mouse_dragged and button==1:
            # Original CLICK_MOUSE_TOLERANCE is2, with a strict radial test.
            # OVUI supplies logical coordinates consistently at press/release.
            if dx*dx+dy*dy<4:
                context=self._viewport_context()
                self._close_viewport_menu()
                generation=self.viewport_menu_generation
                asyncio.get_event_loop().call_soon(self._show_viewport_menu,x,y,context,generation)
        elif not self.mouse_dragged and button==0 and abs(dx)+abs(dy)<=3:
            local_x=x-self.viewport.screen_position_x
            local_y=y-self.viewport.screen_position_y
            hit=self.renderer.pick(local_x,local_y)
            paths=list(self.selection.get_snapshot().paths()) if self.mouse_modifiers & 3 else []
            if hit is not None:
                path=f"/objects/{hit}"
                if path in paths:paths.remove(path)
                else:paths.append(path)
            # OVUI also sets WantCaptureKeyboard in this mask; it is not a
            # held modifier key. Only Shift/Ctrl/Alt/Super alter a plain pick.
            if hit is None and not self.mouse_modifiers & 15:
                self.show_frame_summary()
            else:self.selection.publish(paths,source="viewport")
            if hit is not None and self.mouse_double_clicked and not self.mouse_modifiers & 15:
                self.center_selected()
            self.dirty=True

    def _workspace_mouse_moved(self,x,y,*_):
        anchor=self.mouse_anchor
        if anchor:
            # The parent receives movement before child release/key callbacks.
            # Apply only if that same gesture still owns input after dispatch.
            asyncio.get_event_loop().call_soon(
                lambda:self._mouse_moved(x,y) if self.mouse_anchor is anchor else None)

    def _mouse_moved(self,x,y,*_):
        if self.mouse_anchor:
            if (not self._viewport_mouse_available()
                    or self.mouse_capture_context!=self._viewport_context()):
                self.mouse_anchor=None
                return
            old_x,old_y,button=self.mouse_anchor
            dx,dy=x-self.mouse_origin[0],y-self.mouse_origin[1]
            if (dx*dx+dy*dy>=4 if button==1 else abs(dx)+abs(dy)>3):
                self.mouse_dragged=True
            if self.mouse_dragged:
                self.navigation.drag(x-old_x,y-old_y,*self.renderer.size,button,
                                     pivot=self.renderer.selection_pivot() if button==0 else None)
                self.mouse_anchor=(x,y,button)
                self.dirty=True

    def _viewport_mouse_available(self):
        # explicit_hover rejects overlapping windows. Native OVUI deliberately
        # allows hover through popups, so use its public current-menu identity.
        workspace=getattr(self,"workspace",None)
        return self.window.focused and ui.Menu.get_current() is None and (workspace is None or workspace.active=="render")

    def _viewport_context(self):
        if self.session is None or self.presentation is None:return None
        return (self.session.capture_key,self.presentation.segment_epoch,self.scene_id)

    def _close_viewport_menu(self):
        self.viewport_menu_generation=getattr(self,"viewport_menu_generation",0)+1
        menu=getattr(self,"viewport_menu",None)
        self.viewport_menu=None
        self.viewport_menu_context=None
        if menu is not None:menu.destroy()

    def _viewport_menu_shown(self,generation,shown):
        if not shown and generation==self.viewport_menu_generation:
            # Invalidate callbacks after Escape/outside-click dismissal too.
            self.viewport_menu_generation+=1
            self.viewport_menu_context=None

    def _show_viewport_menu(self,x,y,context,generation):
        if (context is None or context!=self._viewport_context()
                or generation!=self.viewport_menu_generation or not self._viewport_mouse_available()):return
        selected=self.object_model.action_scope([row.key for row in self.row_actions.selected])
        all_rows=self.object_model.action_scope(self.object_model._root_keys)
        selection=(tuple(self.selection.get_snapshot().paths()),tuple(row.token for row in self.row_actions.selected))
        self.viewport_menu_context=context
        self.viewport_menu=ui.Menu("Viewport",tearable=False)
        self.viewport_menu.set_shown_changed_fn(lambda shown:self._viewport_menu_shown(generation,shown))
        with self.viewport_menu:
            for label,action,target in (("Hide selected","hide",selected),("Show selected","show",selected),
                                       ("Hide all","hide_all",all_rows),("Show all","show_all",all_rows)):
                ui.MenuItem(label,width=180,enabled=bool(selected.selected_rows) or action.endswith("_all"),
                    triggered_fn=lambda a=action,s=target:self._invoke_viewport_menu(a,s,context,generation,selection))
            ui.Separator()
            for label,action in (("Center at selected","center"),("Zoom to selected","zoom")):
                ui.MenuItem(label,width=180,triggered_fn=lambda a=action:
                    self._invoke_viewport_menu(a,selected,context,generation,selection))
        self.viewport_menu.show_at(float(x),float(y))

    def _invoke_viewport_menu(self,action,scope,context,generation,selection):
        if generation!=self.viewport_menu_generation:return False
        if context!=self._viewport_context():
            self._report_error("Capture, recording segment or scene changed. Reopen the viewport menu.")
            return False
        if action in ("hide","show","hide_all","show_all"):
            return self._object_action(action,scope)
        current=(tuple(self.selection.get_snapshot().paths()),tuple(row.token for row in self.row_actions.selected))
        rows=self.object_model.row_identities(scope.selected_rows)
        if (current!=selection or tuple(row.token for row in rows)!=tuple(row.token for row in scope.selected_rows)
                or not set(scope.native_roots)<=self.interaction.alive):
            self._report_error("Selection or object contents changed. Reopen the viewport menu.")
            return False
        if action=="center":self.center_selected()
        elif action=="zoom":self.frame_selected()
        else:raise ValueError("Unknown viewport action")
        return True

    def _scroll(self,x,y,modifiers=0,*_):
        # viewport_input.scroll_only_window_hovered gates this callback using
        # current native hit-testing. A cached Python hover/focus flag can be
        # stale on the first wheel event and cannot provide this ownership.
        self.dirty=self.navigation.wheel(y,modifiers,
            pivot=self.renderer.selection_pivot() if not modifiers & 1 else None) or self.dirty

    def _build_viewport_input(self):
        self.viewport_input=ui.InvisibleButton(explicit_hover=True,scroll_only_window_hovered=True,
            mouse_hovered_fn=self._viewport_hovered,
            mouse_pressed_fn=self._mouse_pressed,mouse_released_fn=self._mouse_released,
            mouse_double_clicked_fn=self._mouse_double_clicked,
            mouse_wheel_fn=self._scroll)

    def show_open(self):
        directory=self.workflow.settings.get("open_directory") or Path.home()
        dialog=CaptureFileDialog(self.load,initial_path=self.capture or directory,
            sort_order=self.workflow.settings.get("capture_sort",0),
            on_closed=lambda:self._save_layout_preference("capture_sort",dialog.model.sort_order)
                if dialog.model.sort_order != self.workflow.settings.get("capture_sort",0) else None)
        self.dialogs.append(dialog)
        dialog.show()

    def show_save(self):
        suffix=".ovd" if self.session and self.session.info().ovd else Path(self.capture).suffix if self.capture else ".ovd"
        directory=self.workflow.settings.get("save_directory") or Path.home()
        source=self.workflow.display_session
        def save(path):
            if self.workflow.display_session is not source:
                raise RuntimeError("The displayed capture changed. Close this dialog and reopen Save as.")
            if self.workflow.save_capture(path,overwrite=path.exists()) is False:
                # The workflow reports errors as a False result. The picker
                # needs an exception to retain its destination and show it.
                raise RuntimeError(self.workflow.error)
        release=self.workflow.suspend_live_polling()
        try:
            dialog=CaptureFileDialog(save, on_closed=release,
                mode="save",initial_path=Path(directory)/("capture"+suffix),extensions=(suffix,),default_extension=suffix)
            self.dialogs.append(dialog)
            dialog.show()
        except BaseException:
            release()
            raise

    def show_export(self):
        if not self.session:
            self.status_label.text="Open a capture before exporting."
            return
        caps=self.session.export_capabilities()
        self.export_window=ui.Window("Export capture",width=660,height=340)
        with dialog_surface(self.export_window,"Export capture"):
            ui.Label("Export recorded physics",name="title",height=36)
            active=self.session.info().live and not self.session.info().ended
            ui.Label("Detach the live capture before exporting." if active else "Exports use the displayed frame and current clip range.",
                     name="muted",height=32)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                ui.Button("Save current clip",enabled=caps.clip and not active,
                          clicked_fn=lambda:self._choose_export("clip"))
                ui.Label(caps.clip_reason if not caps.clip else "Native PXD2",name="muted",elided_text=True,explicit_hover=True, tooltip=caps.clip_reason)
            ui.Label("RepX scene",name="section",height=32)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                ui.Button("All objects",enabled=caps.repx and not active,clicked_fn=lambda:self._choose_export("repx"))
                ui.Button("Selected",enabled=caps.repx and not active and bool(self.interaction.selected),
                          clicked_fn=lambda:self._choose_export("repx",tuple(i for i in self.interaction.selected if i>0)))
                ui.Button("Visible",enabled=caps.repx and not active,
                          clicked_fn=lambda:self._choose_export("repx",tuple(i for i in self.interaction.alive if i>0 and self.interaction.is_visible(i))))
            if not caps.repx:ui.Label(caps.repx_reason,name="muted",height=50,word_wrap=True)

    def _choose_export(self,mode,instance_ids=None,*,clip_index=None):
        if self.pending_export:
            self.status_label.text="An export is already running."
            return
        if not self.session or not self.capture or not self.capture.is_file():
            self.status_label.text="Save this recording before exporting."
            return
        if clip_index is not None and not 0<=clip_index<len(self.playback.clips):
            self.status_label.text="The selected clip no longer exists."
            return
        info=self.session.info()
        if info.live and not info.ended:
            self.status_label.text="Detach the live capture before exporting."
            return
        caps=self.session.export_capabilities()
        allowed,reason=(caps.clip,caps.clip_reason) if mode=="clip" else (caps.repx,caps.repx_reason)
        if not allowed:
            self.status_label.text=reason
            return
        original=self.capture
        source,scene_id,frame=self._capture_payload(),self.scene_id,self.frame
        clip=self.playback.clip if clip_index is None else self.playback.clips[clip_index]
        start,stop=(clip.start,clip.stop) if clip else (0,0)
        if mode=="clip" and clip:
            # Native crop frame zero is a snapshot. Preserve the visible range:
            # retain the preceding state so reopened frame one is clip.start,
            # rather than silently skipping the first requested visible frame.
            start=max(0,start-1)
        suffix=".pxd2" if mode=="clip" else ".repx"
        # Keep this capture's extracted payload alive while the picker is open.
        retained_session=[self.session]
        export_window=getattr(self,"export_window",None)
        def submit(path):
            request=ExportRequest(source,path,mode,scene_id,frame,start,stop,instance_ids,path.exists())
            self.pending_export=submit_capture(self.export_executor,source,export_capture,request)
            retained_session.clear()
            self.status_label.text=f"Exporting {path.name}…"
            if export_window is not None:export_window.visible=False
        directory=self.workflow.settings.get("export_directory") or original.parent
        dialog=CaptureFileDialog(submit,mode="save",initial_path=Path(directory)/(original.stem+"-export"+suffix),
                                 extensions=(suffix,),default_extension=suffix,title="Export capture",
                                 on_cancel=retained_session.clear)
        self.dialogs.append(dialog)
        dialog.show()

    def _finish_export(self):
        try:
            path,size=self.pending_export.result()
            self.workflow.settings.set("export_directory",str(path.parent))
            self.status_label.text=f"Exported {path.name} · {size:,} bytes"
        except Exception as error:
            self.status_label.text=f"Export failed: {error}"
        self.pending_export=None

    def show_live(self):
        self.live_dialog=ui.Window("Live capture",width=520,height=310,flags=ui.WINDOW_FLAGS_MODAL)
        with dialog_surface(self.live_dialog,"Live capture"):
            with ui.VStack(spacing=fl.pvd_spacing):
                ui.Label("Connect a physics producer",name="title",height=40)
                ui.Label("The viewer listens for an OVD stream on this port.",name="muted",height=30)
                port=ui.IntField(height=fl.pvd_control)
                port.model.set_value(self.workflow.port)
                with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                    keep=ui.CheckBox(width=22)
                    keep.model.set_value(self.workflow.keep_recordings)
                    ui.Label("Keep recordings on disk")
                self.recording_selection_button=ui.Button("Recording selection…",height=fl.pvd_control,
                    clicked_fn=self.show_recording_selection)
                def attach():
                    self.cancel_load(announce=False)
                    self.load_generation+=1
                    self.workflow.set_keep_recordings(keep.model.as_bool)
                    if self.workflow.attach(port.model.as_int):
                        self.live_dialog.visible=False
                with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                    ui.Spacer()
                    ui.Button("Cancel",width=fl.pvd_button_width,clicked_fn=lambda:setattr(self.live_dialog,"visible",False))
                    ui.Button("Listen",name="primary",width=fl.pvd_button_width,clicked_fn=attach)

    def show_recording_selection(self):
        if hasattr(self,"live_dialog"):self.live_dialog.visible=False
        self.capture_selection_view.show()

    def new_capture(self):
        self._close_viewport_menu()
        self.cancel_load(announce=False)
        self.load_generation+=1
        self.workflow.new()

    def show_catalogue(self):
        if not hasattr(self,"catalogue"):
            self.catalogue=WidgetCatalogue(on_toggle_theme=self.toggle_theme,on_open_capture=self.show_open,version=f"v{__version__} · migration preview")
        self.catalogue.show()

    def show_about(self):
        if not hasattr(self,"about_view"):
            self.about_view=AboutView(on_workspace_guide=self.show_catalogue,on_error=self._report_error)
        self.about_view.show()

    def show_history(self):
        self.activity.select("recordings")

    def _build_recordings(self,host):
        view=RecordingsView(self.workflow,self._open_recording,self._delete_recording,self.open_recordings_folder,host,
            on_save=self._save_session_recording,
            folder_status=getattr(self,"recordings_folder_status",""),
            folder_pending=getattr(self,"recordings_folder_request",None) is not None)
        self.recordings_view=view
        self.history_frame=view.recents
        self.recordings_folder_button=view.folder_button
        self.recordings_folder_label=view.folder_label
        return view

    def _save_session_recording(self,identity):
        directory=self.workflow.settings.get("save_directory") or Path.home()
        def save(path):
            if self.workflow.save_session_recording(identity,path,overwrite=path.exists()) is False:
                raise RuntimeError(self.workflow.error)
        dialog=CaptureFileDialog(save,mode="save",initial_path=Path(directory)/f"capture-{identity}.ovd",
                                extensions=(".ovd",),default_extension=".ovd")
        self.dialogs.append(dialog)
        dialog.show()

    def _set_recordings_folder_status(self, message):
        self.recordings_folder_status=message
        if hasattr(self,"recordings_folder_label"):
            self.recordings_folder_label.text=message
            self.recordings_folder_label.tooltip=message

    def open_recordings_folder(self):
        if getattr(self,"recordings_folder_request",None) is not None:
            return False
        try:
            path=self.workflow.recordings.ensure_directory()
            self.recordings_folder_request=OpenFolderRequest(path)
        except (DesktopError,OSError,ValueError) as error:
            message=(str(error) if isinstance(error,DesktopError) else
                     f"Could not open recordings folder {self.workflow.recordings.directory}: {error}")
            self._set_recordings_folder_status(message)
            self._report_error(message)
            return False
        if hasattr(self,"recordings_folder_button"):
            self.recordings_folder_button.enabled=False
        self._set_recordings_folder_status(f"Opening folder: {self.recordings_folder_request.path}")
        return True

    def _poll_recordings_folder(self):
        request=getattr(self,"recordings_folder_request",None)
        if request is None:
            return
        try:
            if not request.poll():
                return
            self._set_recordings_folder_status(f"Folder requested: {request.path}")
        except DesktopError as error:
            self._set_recordings_folder_status(str(error))
            self._report_error(str(error))
        request.close()
        self.recordings_folder_request=None
        if hasattr(self,"recordings_folder_button"):
            self.recordings_folder_button.enabled=True

    def _close_recordings_folder(self):
        request=getattr(self,"recordings_folder_request",None)
        if request is not None:
            request.close()
            self.recordings_folder_request=None

    def _open_recording(self, path):
        if self.workflow.live_session is not None and Path(path)==self.workflow.active_recording_path:
            self.workflow.follow_latest()
        else:
            self.load(path)

    def _delete_recording(self,path):
        from ovui_widgets.common.dialogs import confirm_dialog
        def remove():
            self.workflow.delete_recording(path)
            self.history_frame.rebuild()
        self.delete_confirmation=confirm_dialog("Delete recording",f"Permanently delete {path.name}?",
            on_confirm=remove,confirm_label="Delete",cancel_label="Keep recording")
        if self.delete_confirmation:
            style.style_dialog(self.delete_confirmation._window)

    def _analysis_frame(self, session, scene_id, frame):
        if self.session is not session:
            self.status_label.text="Reopen the source capture to inspect this graph's frame."
            return
        if self.scene_id!=scene_id:
            if self.session is self.workflow.live_session:self.workflow.follow_live=False
            self._refresh(frame=frame,scene_id=scene_id)
        else:self.seek(frame,snap=False)

    def _dispose_analysis(self, view):
        if view in self.analysis_views:self.analysis_views.remove(view)
        view.destroy()

    def _open_plot(self, job, title, on_frame, *, source_label=""):
        try:
            view=self.workspace.add(title,lambda host:PlotView(job,title,on_frame,page_surface,host=host,
                source_label=source_label),description=source_label)
        except BaseException:
            job.cancel()
            raise
        self.analysis_views.append(view)
        return view

    def _sync_workspace(self):
        if not hasattr(self,"workspace"):return
        context=self.analysis_availability.context
        if self.workspace.active!="render" and not self.playback.playing:
            self.analysis_availability.request()
        states={key:self.analysis_availability.state(key) for key in ("profile","memory","errors")}
        self.workspace.sync(context.key if context else None,states,paused=not self.playback.playing,
                            analysis_applicable=context is not None and not context.ovd)

    def _activate_workspace(self, page):
        self._set_viewport_keyboard_focus(False)
        self.object_keyboard_focus=False
        self.mouse_anchor=None
        self._close_viewport_menu()
        if hasattr(self,"activity"):self.activity.blur()
        if page.key=="render" or page.closable or page.view is not None:return
        page.host.closed=False
        if page.checkable and not self.playback.playing:
            self.analysis_availability.request()
        if not page.enabled or self.playback.playing:
            placeholder=(page.reason,self.playback.playing)
            if getattr(page,"placeholder",None)==placeholder:return
            page.placeholder=placeholder
            with page_surface(page.host,page.title):
                ui.Spacer()
                ui.Label("Pause playback to load this view." if self.playback.playing else
                         "Checking recorded analysis data…" if page.checkable else page.reason,
                         word_wrap=True,name="muted",height=70)
                ui.Spacer()
            return
        page.placeholder=None
        if page.key in ("profile","threads"):
            page.view=self._create_profile(page.host,page.key=="threads")
        else:page.view=self._create_diagnostic(page.host,page.key=="memory")
        page.source=self.workspace.source
        if page.view is not None:
            if page.controls:page.view.restore_workspace_state(page.controls)
            self.analysis_views.append(page.view)

    def plot_property(self, object_id, prop):
        try:
            clip=self.playback.clip
            request=HistoryRequest.from_display(self._capture_payload(),self.session,object_id,prop.id,
                clip.start,clip.stop,scene_id=self.scene_id)
            job=self.analysis_tasks.submit_history(request)
            session,scene_id=self.session,self.scene_id
            self._open_plot(job,f"{prop.name} · #{object_id}",lambda f:self._analysis_frame(session,scene_id,f),
                source_label=f"{request.path.name} · Scene {scene_id} · Object {object_id}\n"
                             f"{prop.name} · Frames {request.start}–{request.stop-1}")
        except (ValueError,RuntimeError) as error:
            self.status_label.text=str(error)

    def show_messages(self):
        if not self._guard_analysis("messages"):return
        self.activity.select("messages")

    def _find_profile_view(self):
        # Embedded results survive deactivation. Find can select that page;
        # a closed floating profiler must remain excluded.
        current=getattr(self.session,"capture_key",id(self.session)) if self.session else None
        views=[view for view in getattr(self,"analysis_views",())
               if isinstance(view,ProfileView) and view.find_ready and view.model.supported
               and (view.window.visible or getattr(view.window,"embedded",False))
               and view.capture_key==current and current is not None]
        return next((view for view in reversed(views) if view.window.focused),views[-1] if views else None)

    def show_find_profile(self,view=None,snapshot=None):
        if view is not None:
            current=getattr(self.session,"capture_key",id(self.session)) if self.session else None
            if (view not in self.analysis_views or not view.can_find or view.capture_key!=current
                    or view.model.snapshot is not snapshot):
                self.status_label.text="The profile result changed. Reopen Find profile events."
                return False
        else:view=self._find_profile_view()
        if view is None:
            self.status_label.text="Open Profiler and wait for its result before finding profile events."
            return False
        self._set_viewport_keyboard_focus(False)
        if getattr(view.window,"embedded",False):
            view.window.focus()
            self.workspace.poll()
        return view.show_find()

    def show_profile(self, threads=False):
        if not self._guard_analysis("profile"):return
        self.workspace.select("threads" if threads else "profile")

    def _create_profile(self, host, threads=False):
        try:
            request=DiagnosticsRequest.from_display(self._capture_payload(),self.presentation,self.scene_id)
            session=self.session
            def refresh(view):
                nonlocal request
                if self.session is not session:
                    view.progress.text="Reopen this capture to refresh its profile."
                    return
                try:
                    request=DiagnosticsRequest.from_display(self._capture_payload(),self.presentation,self.scene_id)
                    view.set_job(self.diagnostics_tasks.submit_profile(request))
                except (ValueError,RuntimeError) as error:view.progress.text=str(error)
            def plot(key,all_threads):
                clip=self.playback.clip
                if self.session is not session or clip is None:
                    self.status_label.text="Reopen this capture to plot the event."
                    return
                history=ProfileHistoryRequest(request.path,request.scene_id,key.zone_id,key.event_id,key.context_id,
                    None if all_threads else key.thread_id,clip.start,clip.stop)
                row=next(row for row in view.model.rows(include_hidden=True)
                    if (row.zone_id,row.event_id,row.context_id,row.thread_id)==
                       (key.zone_id,key.event_id,key.context_id,key.thread_id))
                scope="All threads" if all_threads else thread_name(key.thread_id)
                job=self.diagnostics_tasks.submit_history(history)
                scene_id=history.scene_id
                self._open_plot(job,f"{scope} · {row.name}",
                    lambda frame:self._analysis_frame(session,scene_id,frame),
                    source_label=f"{history.path.name} · Scene {scene_id} · Context {key.context_id}\n"
                                 f"Zone {key.zone_id} · Event {key.event_id} · Frames {history.start}–{history.stop-1}")
            view=ProfileView(self.diagnostics_tasks.submit_profile(request),page_surface,
                on_refresh=refresh,on_plot=plot,
                capture_key=getattr(session,"capture_key",id(session)),source_label=request.path.name,
                host=host,mode=int(threads),
                function_color=self.workflow.settings.get("ProfileFunctionColor",PROFILE_FUNCTION_DEFAULT))
            return view
        except (ValueError,RuntimeError) as error:self.status_label.text=str(error)

    def show_diagnostic(self,memory):
        if not self._guard_analysis("memory" if memory else "errors"):return
        self.workspace.select("memory" if memory else "errors")

    def _create_diagnostic(self, host, memory):
        session=self.session
        def submit():
            if self.session is not session:raise ValueError("Reopen this capture to refresh its diagnostics.")
            if memory:
                from dataclasses import replace
                request=DiagnosticsRequest.from_display(self._capture_payload(),self.presentation,self.scene_id)
                return self.diagnostics_tasks.submit_memory(replace(request,max_events=1_000_000))
            request=ErrorRequest.from_display(self._capture_payload(),self.presentation,self.scene_id)
            return self.error_tasks.submit_errors(request)
        try:
            view_type=MemoryView if memory else ErrorStreamView
            return view_type(submit(),on_refresh=submit,surface=page_surface,host=host)
        except (ValueError,RuntimeError) as error:self.status_label.text=str(error)

    def show_frame_graph(self):
        if not self._guard_analysis("frame_timing"):return
        if self.session and self.capture:
            clip=self.playback.clip
            session,scene_id=self.session,self.scene_id
            job=self.analysis_tasks.submit_frame_timing(self._capture_payload(),scene_id,clip.start,clip.stop)
            self._open_plot(job,"Frame timing",lambda f:self._analysis_frame(session,scene_id,f),
                source_label=f"{self._capture_payload().name} · Scene {scene_id}\n"
                             f"Frames {clip.start}–{clip.stop-1}")

    def show_clips(self):
        self.activity.select("clips")

    def _activate_activity(self,page):
        self._set_viewport_keyboard_focus(False)
        self.object_keyboard_focus=False
        self.workspace.blur()
        if page.view is not None:
            if page.key in ("messages","recordings"):page.view.refresh()
            elif page.key=="clips":self._refresh_clips()
            return
        if not page.enabled or not self.activity.expanded:return
        if page.key=="clips":
            self.clips_view=ClipsView(self._clip_row_action,page.host)
            page.view=self.clips_view
            self._refresh_clips()
        elif page.key=="recordings":page.view=self._build_recordings(page.host)
        else:
            session,scene_id=self.session,self.scene_id
            source=self._activity_source()
            page.view=MessagesView(session,lambda f:self._analysis_frame(session,scene_id,f)
                if self._activity_source()==source else None,page_surface,host=page.host)
            page.source=source
            self.analysis_views.append(page.view)

    def _activity_source(self):
        context=getattr(getattr(self,"analysis_availability",None),"context",None)
        return (context.capture_key,context.segment_epoch,context.scene_id) if context else None

    def _sync_activity(self):
        if not hasattr(self,"activity"):return
        source=self._activity_source()
        page=self.activity.pages["messages"]
        if self.activity.source!=source:
            if self.activity.pages['clips'].view is not None:self._refresh_clips()
            if page.view is not None:
                self.activity._dispose(page)
                page.host.closed=False
                page.host.frame.visible=self.activity.active=="messages"
            self.activity.source=source
            if self.activity.active=="messages":self.activity.pending="messages"
        state=self.analysis_availability.state("messages")
        self.activity.available("messages",state.enabled,state.reason)
        clips=self.activity.pages['clips'].view
        if clips is not None and clips.window.visible:
            # Live completion can change export availability without changing
            # the clip range. A retained view ignores unchanged snapshots.
            self._refresh_clips()

    def _clip_action(self,action,*args):
        if not self.session or not self.scenes or self.playback.clip is None:
            return
        requested=deepcopy(self.playback)
        getattr(requested,action.__name__)(*args)
        self.pending_playback=requested
        if self.session is self.workflow.live_session:self.workflow.follow_live=False
        return self._refresh(frame=requested.frame)

    def _clip_export_reason(self):
        if not self.session:return "Open a capture before exporting."
        info=self.session.info()
        if info.live and not info.ended:return "Detach the live capture before exporting."
        caps=self.session.export_capabilities()
        return caps.clip_reason if not caps.clip else ""

    def _refresh_clips(self):
        if hasattr(self,"clips_view"):
            self.clips_view.refresh(self.playback,self._activity_source(),self._clip_export_reason())

    def _clip_row_action(self,source,action,row):
        if source is None or source!=self._activity_source():return
        if not 0<=row.index<len(self.playback.clips):return
        clip=self.playback.clips[row.index]
        if row.signature!=(clip.name,clip.start,clip.stop):return
        if action=="save":
            reason=self._clip_export_reason()
            if reason:self.status_label.text=reason;return
            # Export the addressed row without changing the applied clip,
            # playhead or selection. The picker freezes its source/range.
            return self._choose_export("clip",clip_index=row.index)
        args=(row.index,) if action in ("select","delete") else ()
        if not args and row.index!=self.playback.index:return
        return self._activity_clip_action(source,action,*args)

    def _activity_clip_action(self,source,action,*args):
        if source is not None and source==self._activity_source():
            return self._clip_action(getattr(self.playback,action),*args)

    def _poll_workflow(self):
        errors=[]
        for event in self.workflow.poll():
            if event.kind=="display":
                self._display(event.path,event.session)
            elif event.kind=="frames" and self.session is event.session:
                candidates=list(event.scenes)
                if candidates and self.workflow.follow_live:
                    index=next((i for i,s in enumerate(candidates) if s.id==self.scene_id),0)
                    if event.segment_changed:
                        index=next((i for i,s in reversed(list(enumerate(candidates))) if s.alive),index)
                    scene=candidates[index]
                    # Event fields choose a request; only the returned native view
                    # may update the displayed scene, playhead or object metadata.
                    newest=max(0,min(scene.frame_count-1,scene.frame_count-1-max(0,event.max_frame-event.current_frame)))
                    requested=Playback()
                    requested.reset(newest+1,self.presentation.info().ovd)
                    requested.mode=self.playback.mode
                    self._refresh(frame=requested.seek(newest),scene_id=scene.id)
            elif event.kind=="error":
                errors.append(event.message)
            elif event.kind=="state":
                self.status=event.message
                self.status_label.text=self.status
            elif event.kind in ("recordings","recents") and hasattr(self,"history_frame"):
                self.history_frame.rebuild()
                if hasattr(self,"recordings_view"):self.recordings_view.refresh()
        error="; ".join(dict.fromkeys(errors))
        if error:
            self.error_message=error
            self.status=error
            self.status_label.text=error
        if self.session is self.workflow.display_session and self.capture!=self.workflow.display_path:
            self.capture=self.workflow.display_path
            self.capture_data_path=getattr(self.workflow,"display_data_path",None) or self.capture
            self.file_label.text=self.capture.name if self.capture else "Live capture"
        self.accept_button.image_url=icon_url("live" if self.workflow.accept_connections else "live_paused")
        self._sync_connection_status()

        panel=getattr(self,"capture_selection_view",None)
        if panel is not None:panel.refresh()

    def _build_scenes(self):
        names=[f"Scene {i+1}  ·  {s.name or s.id}" for i,s in enumerate(self.scenes)] or ["No scene loaded"]
        self.scene_combo=ui.ComboBox(self.scene_index,*names,height=20)
        self.scene_combo.model.add_item_changed_fn(self._select_scene)

    def show_workspace_menu(self):
        self._sync_analysis_availability(request=True)
        self.analysis_menu_items={}
        self.workspace_menu=ui.Menu("Workspace",tearable=False)
        with self.workspace_menu:
            for label,action in (("Viewport options",self.show_view_options),("General settings (Ctrl+Shift+S)",self.render_settings_view.show),
                    ("Camera",self.show_camera_options),
                    ("Restore default visual settings (Ctrl+Shift+R)",self.reset_visual_settings),
                    ("Go to frame (Ctrl+G)",self.show_goto_frame),
                    ("Evidence",self.show_evidence),("Import evidence JSON",self.show_import_evidence),
                    ("Scene palette",self.palette_view.show),("Profiler",self.show_profile),
                    ("Find profile events (Alt+F)",self.show_find_profile),
                    ("Memory",lambda:self.show_diagnostic(True)),("Errors",lambda:self.show_diagnostic(False)),
                    ("Export capture",self.show_export),("New capture",self.new_capture),
                    ("Accept connections" if not self.workflow.accept_connections else "Pause new connections",
                     lambda:self.workflow.set_accept_connections(not self.workflow.accept_connections)),
                    ("Workspace guide",self.show_catalogue),("About and help",self.show_about)):
                analysis={"Profiler":"profile","Memory":"memory","Errors":"errors"}.get(label)
                if label=="Find profile events (Alt+F)":
                    view=self._find_profile_view()
                    snapshot=view.model.snapshot if view else None
                    ui.MenuItem(label,enabled=view is not None,
                        explicit_hover=True, tooltip="Search a loaded profiler result; open Profiler first if none is available.",
                        triggered_fn=lambda v=view,s=snapshot:self.show_find_profile(v,s))
                elif analysis:
                    context=self.analysis_availability.context
                    key=context.key if context else None
                    state=self.analysis_availability.state(analysis)
                    item=ui.MenuItem(label,enabled=state.enabled,explicit_hover=True, tooltip=state.reason,
                        triggered_fn=lambda a=analysis,k=key,fn=action:self._invoke_analysis_menu(a,k,fn))
                    self.analysis_menu_items[analysis]=(item,key)
                else:ui.MenuItem(label,triggered_fn=action)
        self.workspace_menu.show_at(self.view_button.screen_position_x,
                                    self.view_button.screen_position_y+self.view_button.computed_height)

    def build(self):
        self._destroy_split_grips()
        self._destroy_view_options()
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR|ui.WINDOW_FLAGS_NO_RESIZE|ui.WINDOW_FLAGS_NO_MOVE|ui.WINDOW_FLAGS_NO_SCROLLBAR
        # Keep floating analysis views above the full-window workspace.
        flags|=1<<13
        self.window=ui.Window("PVD Workspace",width=self.width,height=self.height,flags=flags,fill_app_window=True)
        self.window.position_x=0;self.window.position_y=0
        self.window.padding_x=0;self.window.padding_y=0
        self.window.set_key_pressed_fn(self._key_pressed)
        with self.window.frame:
            with ui.ZStack():
                self.viewport=ui.ImageWithProvider(self.renderer.provider,
                    fill_policy=ui.IwpFillPolicy.IWP_STRETCH,opaque_for_mouse_events=False)
                # Continue an owned camera drag across panels; presses and wheel
                # remain viewport-only. OVUI delivers its release outside hover.
                with ui.VStack(spacing=0,opaque_for_mouse_events=False,
                               mouse_pressed_fn=self._workspace_pressed,mouse_moved_fn=self._workspace_mouse_moved):
                    ui.Spacer(height=14)
                    with ui.HStack(height=44,spacing=8):
                        ui.Spacer(width=16)
                        with centered_slot(48,30):
                            ui.Button("PVD",name="badge",clicked_fn=self.show_catalogue,
                                      explicit_hover=True, tooltip="Workspace guide and widget catalogue")
                        with centered_slot(210,36):
                            with ui.VStack(spacing=0):
                                self.file_label=ui.Label("Physics, in focus.",name="filename",height=20,elided_text=True)
                                self.summary_label=ui.Label("OVUI workspace",name="muted",height=16,elided_text=True)
                        ui.Spacer()
                        with centered_slot(fl.pvd_workspace_rail_width,fl.pvd_workspace_rail_height):
                            with workspace_tab_rail():
                                self.view_button=ui.Button("View",name="workspace_tab_active",width=51,clicked_fn=self.show_workspace_menu)
                                ui.Button("Captures",name="workspace_tab",width=74,clicked_fn=self.show_history)
                                self.accept_button=icon_button("live",text="Live",size=6,name="workspace_tab",width=58,clicked_fn=self.show_live)
                                self.messages_button=ui.Button("Messages",name="workspace_tab",width=80,clicked_fn=self.show_messages,enabled=False,explicit_hover=True)
                        ui.Spacer()
                        with centered_slot(68,30):
                            ui.Button("Save as",name="ghost",clicked_fn=self.show_save)
                        with centered_slot(64,30):
                            ui.Button("Detach",name="workspace_detach",clicked_fn=self.workflow.detach)
                        with centered_slot(112,30):
                            ui.Button("Open capture",name="primary",clicked_fn=self.show_open)
                        with centered_slot(32,30):
                            self.theme_button=icon_button("theme",name="outline",clicked_fn=self.toggle_theme,
                                tooltip="Switch to light mode" if self.theme=="dark" else "Switch to dark mode")
                        ui.Spacer(width=16)
                    ui.Spacer(height=12)
                    with ui.HStack(spacing=4):
                        ui.Spacer(width=12)
                        self.browser_column=ui.VStack(width=max(fl.pvd_panel_min_width,min(fl.pvd_panel_max_width,
                            self.workflow.settings.get("soft_glass_browser_width",292))))
                        with self.browser_column:
                            with island_surface():
                                with ui.HStack(height=18):
                                    ui.Label("OBJECTS",name="eyebrow")
                                    self.scene_frame=ui.Frame(width=132,height=20,build_fn=self._build_scenes)
                                self.object_browser=ObjectBrowser(self.object_model,self._tree_selection,
                                    on_activate=self.center_selected,on_find_object=self._find_object,
                                    on_action=self._object_action,row_tags=self._row_tags,on_focus=self._focus_objects,
                                    search_window_focused=lambda:self.window.focused)
                                self.tree=self.object_browser.tree
                        self.browser_grip=SplitGrip(self.browser_column,
                            lambda width:self._save_layout_preference("soft_glass_browser_width",width))
                        with ui.VStack(spacing=8):
                            self.workspace=WorkspaceTabs(self.window,self._activate_workspace,self._dispose_analysis)
                            with self.workspace.render_frame:
                                with ui.VStack(spacing=8):
                                    with ui.HStack(height=38,spacing=6):
                                        with centered_slot(146,36):
                                            self.shading_control=SegmentedControl(((0,"Shaded",62),(1,"Wireframe",76)),
                                                value=int(self.render_settings.wireframe) if self.render_settings.wireframe in (0,1) else -1,
                                                height=36,on_changed=self._set_wireframe)
                                        with centered_slot(114,30):
                                            ui.Button("Frame selection",name="workspace_frame",clicked_fn=self.frame_selected)
                                        with centered_slot(78,30):
                                            ui.Button("Frame all",name="workspace_frame",clicked_fn=self.frame_all)
                                        with centered_slot(66,30):
                                            ui.Button("Camera",name="ghost",clicked_fn=self.show_camera_options)
                                        ui.Spacer()
                                    self._build_display_controls()
                                    self.viewport_region=ui.Frame()
                                    with self.viewport_region:
                                        self._build_viewport_input()
                                    with ui.VStack(height=40,spacing=0):
                                        with ui.HStack(height=24):
                                            with ui.ZStack(width=300,height=24):
                                                ui.Rectangle(style={"background_color":ui.color.pvd_island_source,"border_radius":7},
                                                    opaque_for_mouse_events=False)
                                                with ui.HStack():
                                                    ui.Spacer(width=10)
                                                    self.viewport_info=ui.Label("Open a capture to render",name="muted",elided_text=True)
                                                    ui.Spacer(width=10)
                                            ui.Spacer()
                                        with ui.HStack(height=16,spacing=fl.pvd_spacing):
                                            self.status_label=ui.Label(self.status,name="muted",width=ui.Fraction(1),elided_text=True)
                                            self.connection_label=ui.Label("Live inactive",name="muted",width=ui.Fraction(1),explicit_hover=True,
                                                alignment=ui.Alignment.RIGHT_CENTER,elided_text=True)
                            self.activity=ActivityPanel(self.window,self._activate_activity,self._dispose_analysis,self.workflow.settings,
                                on_save=self._save_layout_preference)
                        inspector_grip=ui.Frame(width=fl.pvd_splitter_hit_width)
                        self.inspector_column=ui.VStack(width=max(fl.pvd_panel_min_width,min(fl.pvd_panel_max_width,
                            self.workflow.settings.get("soft_glass_inspector_width",320))))
                        with self.inspector_column:
                            with island_surface():
                                with ui.HStack(height=26):
                                    ui.Label("PROPERTY INSPECTOR",name="eyebrow")
                                    ui.Button("Format",width=56,height=24,clicked_fn=self.show_property_options)
                                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                        mouse_pressed_fn=self.property_inspector.background_pressed):
                                    self.property_frame=ui.Frame(build_fn=self._build_properties)
                            self._build_display_panel()
                        with inspector_grip:
                            self.inspector_grip=SplitGrip(self.inspector_column,
                                lambda width:self._save_layout_preference("soft_glass_inspector_width",width),direction=-1)
                        ui.Spacer(width=12)
                    ui.Spacer(height=10)
                    with ui.HStack(height=76) as self.transport_row:
                        ui.Spacer(width=16)
                        with island_surface(padding=14):
                            with ui.HStack(height=44,spacing=6) as self.transport_controls:
                                with centered_slot(28,30):
                                    icon_button("first",name="ghost",clicked_fn=lambda:self._transport(self.playback.first),tooltip="First frame")
                                with centered_slot(28,30):
                                    icon_button("previous",name="ghost",clicked_fn=lambda:self._transport(self.playback.step,-1),tooltip="Previous frame")
                                with centered_slot(28,30):
                                    self.reverse_button=icon_button("reverse",name="ghost",clicked_fn=lambda:self.toggle_play(-1),tooltip="Reverse playback")
                                with centered_slot(38,38):
                                    self.play_button=icon_button("play",name="primary",clicked_fn=self.toggle_play,tooltip="Play or pause")
                                with centered_slot(28,30):
                                    icon_button("next",name="ghost",clicked_fn=lambda:self._transport(self.playback.step,1),tooltip="Next frame")
                                with centered_slot(28,30):
                                    icon_button("last",name="ghost",clicked_fn=lambda:self._transport(self.playback.last),tooltip="Last frame")
                                with centered_slot(170,36):
                                    self.mode_combo=SegmentedControl(((2,"Post-sim",66),(1,"Pre-sim",62),(0,"All",32)),
                                        on_changed=self._mode_changed,height=36)
                                with centered_slot(72,30) as self.parity_slot:
                                    self.parity_button=ui.Button("Frame",name="ghost",clicked_fn=lambda:self._transport(self.playback.peek_parity),explicit_hover=True, tooltip="Toggle pre/post simulation")
                                self.frame_model=ui.SimpleIntModel(1)
                                with ui.VStack(height=44,spacing=0) as self.ruler_slot:
                                    self.frame_slider=SoftGlassTimeline(self.frame_model,min=0,max=1,height=44)
                                    self.finding_track=FindingTrack(self._select_evidence_marker,self._evidence_color)
                                self.frame_model.add_value_changed_fn(lambda m:self.seek(m.as_int) if not self.suppress_frame_callback else None)
                                with centered_slot(110,32):
                                    with ui.ZStack():
                                        ui.Rectangle(name="field")
                                        with ui.VStack(spacing=0):
                                            ui.Spacer(height=4)
                                            with ui.HStack(height=24,spacing=4):
                                                ui.Spacer(width=4)
                                                ui.Label("Frame",name="muted",width=34)
                                                ui.IntField(self.frame_model,name="mono",width=44,
                                                    style={"background_color":ui.color.pvd_surface})
                                                ui.Spacer(width=4)
                                            ui.Spacer(height=4)
                                with centered_slot(54,30):
                                    ui.Button("Clips",clicked_fn=self.show_clips)
                                with centered_slot(92,30):
                                    self.frame_graph_button=ui.Button("Frame graph",name="ghost",clicked_fn=self.show_frame_graph,enabled=False,explicit_hover=True)
                        ui.Spacer(width=16)
                    ui.Spacer(height=16)

    def _update_split_grips(self):
        for name in ("browser_grip","inspector_grip"):
            grip=getattr(self,name,None)
            if grip is not None:grip.update()
        self._update_view_options()

    def _destroy_split_grips(self):
        for name in ("browser_grip","inspector_grip"):
            grip=getattr(self,name,None)
            if grip is not None:grip.destroy()

    async def run_async(self):
        """Application work only; host through run_with_backend for teardown."""
        try:
            self.renderer=ViewportRenderer()
            for bit,mode in self.overlay_modes.items():self.renderer.overlays.set_mode(bit,mode)
            self.renderer.set_palette(self.palette)
            self.renderer.set_settings(self.render_settings)
            self.navigation=CameraController(self.renderer.camera)
            self.last_camera=time.monotonic()
            self.build()
            self.window.set_focused_changed_fn(self._window_focused)
            if hasattr(self.window,"set_drop_fn"):
                self.window.set_drop_fn(self._drop_capture)
            if self.live_endpoint is not None or not self.capture:
                self.workflow.startup(endpoint=self.live_endpoint,explicit_live=self.live_endpoint is not None)
            else:
                self.load(self.capture)
            while self.running:
                if self.inspector:
                    self.inspector.drain_pending(ui._ui,application=self)
                await ui.next_frame()
                self._update_split_grips()
                self.activity.poll()
                if hasattr(self,"recordings_view"):self.recordings_view.poll()
                if hasattr(self,"clips_view"):self.clips_view.poll()
                for slider in getattr(self,"view_sliders",()):slider.update()
                if hasattr(self,"transparency_slider"):self.transparency_slider.update()
                if self.pending and self.pending.done():
                    self._finish_load()
                if self.pending_export and self.pending_export.done():
                    self._finish_export()
                now=time.monotonic()
                elapsed=min(.1,now-self.last_camera)
                self.last_camera=now
                if not self.window.focused or not self.viewport_keyboard_focus:self.navigation.stop()
                elif self.navigation.update(elapsed):self.dirty=True
                if now-self.last_poll>1/30:
                    self.last_poll=now
                    self._poll_load_progress()
                    palette_revision=self.palette.revision
                    self.palette_view.refresh()
                    if hasattr(self,"about_view"):self.about_view.refresh()
                    self._poll_recordings_folder()
                    if self.palette.revision!=palette_revision:self.dirty=True
                    self._poll_workflow()
                    if self.evidence_controller.poll():self._sync_evidence()
                    if self.analysis_availability.poll():self._update_analysis_controls()
                    if self.pending_display:self._display(*self.pending_display)
                    if self.pending_evidence:self._retry_evidence_go()
                    elif self.presentation_dirty:self._refresh()
                    self._sync_workspace()
                    self._sync_activity()
                    self.workspace.poll()
                    for view in self.analysis_views:
                        if hasattr(view,"poll"):
                            view.poll()
                if self.playback.playing and now-self.last_play > 1/self.playback.fps:
                    self.last_play=now
                    self._transport(self.playback.tick)
                size=(int(self.viewport.computed_width),int(self.viewport.computed_height))
                if min(size)>0 and (self.dirty or size!=self.renderer.size):
                    self.renderer.render(*size,light=self.theme=="light",background=design_rgba("background",self.theme),
                        background_gradient=(design_rgba("stage_center",self.theme),design_rgba("stage_edge",self.theme)),
                        ground_color=design_rgba("elevated" if self.theme=="light" else "surface",self.theme),grid_contrast=.025,
                        grid_rgba=STAGE_GRID_RGBA[self.theme],
                        scene_length=self.navigation.scene_length,gizmo_scale=self.gizmo_scale*self.master_scale)
                    if self.presentation is not None:
                        # Visibility changes update the draw count without a
                        # new native publication. Report the completed draw.
                        self.viewport_info.text=f"{self.renderer.instance_count:,} draw objects   ·   Frame {self.frame}   ·   {self.playback.label}"
                    self.render_settings_view.refresh()
                    if hasattr(self,"camera_view"):self.camera_view.refresh_tracking()
                    self.dirty=False
        except Exception as error:
            self.run_error=error
            raise

    def _close_pending_export(self):
        """Retrieve an export's outcome after joining its executor."""
        future=self.pending_export
        try:
            # Queued cancellation is normal. A worker-raised CancelledError is
            # an actual failure (cancelled() is false), and must remain visible.
            if future is not None and not future.cancelled():future.result()
        finally:
            self.pending_export=None

    def close(self):
        """Attempt every resource close once, before backend teardown.

        A repeated failed close re-raises the recorded failure; it must never
        report successful cleanup or double-delete partially released GL data.
        This also handles setup failure before run_async has entered.
        """
        if self._close_started:
            if self._close_error is not None:raise self._close_error
            if not self.closed:raise RuntimeError("Application cleanup is already in progress")
            return
        self._close_started=True
        self.running=False
        errors=[]

        def attempt(name,callback):
            try:callback()
            except BaseException as error:
                error.add_note(f"Closing {name}")
                errors.append(error)

        def resource(name,method="destroy",**kwargs):
            value=getattr(self,name,None)
            if value is not None:
                attempt(name,lambda:getattr(value,method)(**kwargs))

        attempt("viewport menu",self._close_viewport_menu)
        attempt("viewport options",self._destroy_view_options)
        for name in ("browser_grip","inspector_grip"):resource(name)
        attempt("recordings folder request",self._close_recordings_folder)
        attempt("capture load",lambda:self.cancel_load(announce=False))
        resource("capture_opener","close")
        resource("analysis_availability","close")
        resource("capture_selection_view","close")
        resource("evidence_controller","close")
        resource("evidence_view")
        resource("finding_track")
        for name in ("frame_slider","transparency_slider","goto_frame_dialog","property_inspector","render_settings_view"):
            resource(name)
        resource("export_executor","shutdown",wait=True,cancel_futures=True)
        attempt("pending export",self._close_pending_export)
        resource("workspace")
        resource("activity")
        for index,view in enumerate(tuple(self.analysis_views)):
            attempt(f"analysis view {index}",lambda value=view:value.destroy())
        self.analysis_views.clear()
        resource("palette_view")
        for name in ("analysis_tasks","diagnostics_tasks","error_tasks"):resource(name,"close")
        for index,dialog in enumerate(tuple(self.dialogs)):
            attempt(f"dialog {index}",lambda value=dialog:value.destroy())
        self.dialogs.clear()
        for name in ("catalogue","about_view","object_browser"):resource(name)
        resource("workflow","shutdown")
        resource("renderer","close")
        self.session=self.adapter=None
        resource("temporary","cleanup")
        if errors:
            self._close_error=BaseExceptionGroup("Application resource cleanup failed",errors)
            raise self._close_error
        self.closed=True

    def run_with_backend(self, *, setup):
        """Run with public OVUI pumping and owned cleanup, including setup errors.

        Embedders/benchmarks supply their title, style and frame-rate setup.
        Direct ui.run(self.run_async()) does not own application cleanup.
        """
        run_standalone(self.run_async,ui.run_async,setup=setup,
                       close=self.close,shutdown=ui.shutdown)

    def run(self):
        def setup():
            ui.init("PVD on OVUI",width=self.width,height=self.height,max_fps=60)
            style.startup(self.theme)
            try:
                import ovuiinspect
                self.inspector=ovuiinspect
                self.inspector.attach_application(self)
            except ImportError:
                pass
        self.run_with_backend(setup=setup)
