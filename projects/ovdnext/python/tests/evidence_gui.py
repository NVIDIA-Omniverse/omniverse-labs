# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real application, renderer, file picker and native events; no desktop input."""
import argparse
import asyncio
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

# GLFW otherwise loads system OpenGL even when ModernGL has a CI override.
# Retain the explicit CI-only library for the process, outside the application.
_ci_graphics = None
if sys.platform == "win32" and (library := os.environ.get("GLCONTEXT_WIN_LIBGL")):
    _ci_graphics = ctypes.WinDLL(str(Path(library).resolve(strict=True)), winmode=8)

import omni.ui as ui
from gui_runtime import hide_owned_window, isolate_gui_output
from ovd_ui.application import Application
from ovd_ui.playback import FrameMode
from ovd_ui.style import layout as fl
from ovd_ui.widgets.timeline import FrameScale

ROOT = Path(__file__).resolve().parents[2]


class EvidenceRegression(Application):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.go_receipts=[]

    def _go_evidence(self,*args,**kwargs):
        applied=super()._go_evidence(*args,**kwargs)
        if applied:self.go_receipts.append(self.presentation)
        return applied

    async def frames(self, count=4):
        for _ in range(count):await ui.next_frame()

    async def wait(self, predicate, message):
        deadline = time.monotonic()+12
        while time.monotonic()<deadline:
            if predicate():return
            await ui.next_frame()
        raise AssertionError(message + ": " + self.error_message + "; " + self.evidence_controller.error)

    def pixel(self, widget, x=.5, y=.5):
        dpi=ui.Workspace.get_dpi_scale()
        return ((widget.screen_position_x+widget.computed_width*x)*dpi,
                (widget.screen_position_y+widget.computed_height*y)*dpi)

    async def click_at(self, x, y):
        ui._ui._inject_mouse_move(x,y);await self.frames(3)
        ui._ui._inject_mouse_button(0,True);await self.frames(1)
        ui._ui._inject_mouse_button(0,False);await self.frames(4)

    async def click(self, widget):
        assert widget.computed_width>0 and widget.computed_height>0
        await self.click_at(*self.pixel(widget))

    async def key(self, code):
        ui._ui._inject_key_event(code,True);await self.frames(1)
        ui._ui._inject_key_event(code,False);await self.frames(3)

    async def screenshot(self,path):
        assert ui._ui._schedule_screenshot(str(path))
        await self.wait(ui._ui._poll_screenshot_done,"Screenshot did not complete")

    async def choose_file(self, path, *, evidence=True):
        if evidence:await self.click(self.evidence_view.import_button)
        else:self.show_open();await self.frames(5)
        dialog=self.dialogs[-1]
        assert dialog.primary_button.text == ("Open Evidence JSON" if evidence else "Open capture")
        await self.click(dialog.filename_field)
        ui._ui._inject_key_event(341,True);await self.key(65)
        ui._ui._inject_key_event(341,False);await self.frames(1)
        ui._ui._inject_text_input(str(path));await self.frames(3)
        assert dialog.filename_field.model.as_string==str(path)
        await self.click(dialog.primary_button)
        await self.wait(lambda:not dialog.window.visible,"File picker did not accept typed path")

    async def choose_role(self, index):
        await self.click(self.evidence_view.role_combo)
        for _ in range(index):await self.key(264)
        await self.key(257)
        await self.frames(5)

    async def clip_row(self, index):
        tree=self.clips_view.tree;dpi=ui.Workspace.get_dpi_scale()
        await self.click_at((tree.screen_position_x+35)*dpi,
                            (tree.screen_position_y+fl.pvd_row*(index+.5))*dpi)
        assert self.playback.index==index, (self.playback.index,index,tree.screen_position_y)

    async def verify(self):
        await self.wait(lambda:self.presentation is not None,"Initial capture did not load")
        self.playback.set_mode(FrameMode.FULL);self.seek(1,snap=False);await self.frames(5)
        assert self.frame==1
        assert self.transport_row.computed_height==76 and self.frame_slider.widget.computed_height==44
        with tempfile.TemporaryDirectory(prefix="evidence-gesture-") as temporary:
            directory=Path(temporary)
            capture=ROOT/"test/data/resample_two_segment.ovd";payload=capture.read_bytes()
            sdk=json.loads((ROOT/"python/tests/fixtures/evidence/v1-two-segment-reused-handle.json").read_bytes())
            document=dict(format="ovdnext.timeline-evidence",version=1,
                captures={r:dict(sha256=hashlib.sha256(payload).hexdigest(),byteLength=str(len(payload))) for r in ("left","right")},
                findings=[dict(id="pair",kind="comparison",evidence=sdk)])
            path=directory/"comparison.json";path.write_text(json.dumps(document))
            bad=directory/"invalid.json";bad.write_text('{"version":1,"version":2}')
            self.show_evidence();await self.frames(5)
            self.evidence_view.window.position_x=40;self.evidence_view.window.position_y=64
            self.evidence_view.window.width=740;self.evidence_view.window.height=550
            await self.frames(5)
            print("Evidence GUI: native picker import",flush=True)
            await self.choose_file(path)
            await self.wait(lambda:not self.evidence_controller.pending and bool(self.evidence_controller.markers),"Binding did not produce real points")
            await self.frames(5)
            original=self.evidence_controller.selected_set
            assert len(self.evidence_controller.markers)==2 and self.frame==1
            assert self.transport_row.computed_height==88
            assert self.frame_slider.widget.computed_height==44
            assert self.finding_track.widget.computed_height==12
            assert self.frame_slider.widget.computed_width==self.finding_track.widget.computed_width
            await self.choose_file(bad)
            await self.wait(lambda:not self.evidence_controller.pending and bool(self.evidence_controller.error),"Malformed import error missing")
            assert self.evidence_controller.selected_set==original and self.frame==1
            await self.frames(5)
            left_marker=next(m for m in self.evidence_controller.markers if m.role=="left")
            right_marker=next(m for m in self.evidence_controller.markers if m.role=="right")
            await self.click(self.evidence_view.finding_buttons["pair"]);await self.choose_role(1)
            await self.click(self.evidence_view.scene_button)
            self.seek(1,snap=False);await self.frames(4)
            assert self.scene_id==left_marker.scene_id
            print("Evidence GUI: actual marker click and Go",flush=True)
            await self.key(256)
            assert not self.evidence_view.window.visible
            track=self.finding_track.widget;scale=FrameScale(0,self.scenes[self.scene_index].frame_count-1,track.computed_width)
            dpi=ui.Workspace.get_dpi_scale()
            await self.click_at((track.screen_position_x+scale.position(2))*dpi,(track.screen_position_y+6)*dpi)
            await self.frames(5)
            assert self.frame==1 and self.evidence_controller.selection==("pair","left",2)
            await self.click(self.evidence_view.go_button)
            assert self.frame==2 and self.selection.get_snapshot().paths()==[f"/objects/{left_marker.instance_id}"]
            await self.click(self.evidence_view.finding_buttons["pair"])
            assert not self.evidence_view.go_button.enabled
            await self.choose_role(2)
            assert self.evidence_controller.selection[1]=="right", self.evidence_controller.selection
            assert not self.evidence_view.go_button.enabled and "scene" in self.evidence_view.reason_label.text.lower()
            await self.click(self.evidence_view.scene_button)
            assert self.scene_id==right_marker.scene_id
            await self.click(self.evidence_view.go_button)
            assert self.frame==2 and self.selection.get_snapshot().paths()==[f"/objects/{right_marker.instance_id}"]
            assert self.go_receipts[-1].evidence_identity(right_marker.instance_id).recording_segment==1
            print("Evidence GUI: ruler drag, clips and no implicit covering clip",flush=True)
            await self.key(256)
            slider=self.frame_slider.widget;dpi=ui.Workspace.get_dpi_scale()
            scale=FrameScale(0,121,slider.computed_width)
            x0=(slider.screen_position_x+scale.position(2))*dpi;x1=(slider.screen_position_x+scale.position(3))*dpi
            y=(slider.screen_position_y+15)*dpi
            ui._ui._inject_mouse_move(x0,y);await self.frames(3)
            ui._ui._inject_mouse_button(0,True);await self.frames(2)
            ui._ui._inject_mouse_move(x1,y);await self.frames(2)
            ui._ui._inject_mouse_button(0,False);await self.frames(5)
            assert self.frame==3 and not self.frame_slider._dragging
            self.show_evidence();await self.frames(4)
            await self.click(self.evidence_view.clips_button);await self.key(256)
            await self.click(self.clips_view.buttons["split"])
            assert len(self.playback.clips)==2
            await self.clip_row(1)
            self.show_evidence();await self.frames(4)
            assert not self.evidence_view.go_button.enabled and "clip" in self.evidence_view.reason_label.text.lower()
            before=self.frame;await self.click(self.evidence_view.go_button)
            assert self.frame==before and len(self.playback.clips)==2
            await self.key(256);await self.clip_row(0)
            self.show_evidence();await self.frames(4)
            await self.click(self.evidence_view.go_button)
            assert self.frame==2 and len(self.playback.clips)==2
            await self.choose_file(path)
            await self.wait(lambda:not self.evidence_controller.pending,"Duplicate import did not settle")
            assert len(self.evidence_controller.sets)==1 and not self.evidence_controller.error
            print("Evidence GUI: themes, resize and marker rebuild stability",flush=True)
            for theme in ("dark","light"):
                if self.theme!=theme:await self.click(self.theme_button.button)
                await self.frames(5)
                self._sync_evidence();await self.frames(3)
                builds=self.finding_track.builds
                for n in (1,2,3):self.seek(n,snap=False);await self.frames(2)
                assert self.finding_track.builds==builds
                self.evidence_view.window.width=680 if theme=="light" else 740
                await self.frames(4)
                assert self.evidence_view.go_button.computed_width>0
                if self.artifacts:
                    target=self.artifacts/f"evidence-{theme}.png"
                    await self.screenshot(target)
            old_key=self.session.capture_key
            self.show_evidence();await self.frames(3)
            await self.choose_file(capture,evidence=False)
            await self.wait(lambda:self.session.capture_key!=old_key,"Explicit capture reopen did not replace session")
            await self.wait(lambda:not self.evidence_controller.pending,"Reopened capture did not rebind")
            assert len(self.evidence_controller.sets)==1 and self.scene_id==self.presentation.scene_id
            assert self.evidence_controller.context.capture_key==self.session.capture_key
            print("Evidence GUI gestures passed",flush=True)

    async def run_async(self):
        hide_owned_window()
        task=asyncio.create_task(super().run_async())
        try:await self.verify()
        except BaseException:
            if self.artifacts:
                try:await asyncio.wait_for(self.screenshot(self.artifacts/"evidence-failed.png"),2)
                except Exception:pass
            raise
        finally:self.request_exit();await task


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true");parser.add_argument("--artifacts",type=Path)
    args=parser.parse_args()
    if not args.verify:parser.error("Use --verify for this bounded runtime test")
    isolate_gui_output();faulthandler.enable();faulthandler.dump_traceback_later(60)
    app=EvidenceRegression(ROOT/"test/data/resample_two_segment.ovd",1280,800,"dark")
    app.artifacts=args.artifacts
    if args.artifacts:args.artifacts.mkdir(parents=True,exist_ok=True)
    try:app.run()
    finally:faulthandler.cancel_dump_traceback_later()
