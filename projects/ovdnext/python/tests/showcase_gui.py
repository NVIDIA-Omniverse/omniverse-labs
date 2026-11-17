# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded actual CPU-control walkthrough using finalized captures and evidence."""
import argparse
import faulthandler
from pathlib import Path

import omni.ui as ui
from evidence_gui import EvidenceRegression
from gui_runtime import isolate_gui_output
from ovd_ui.playback import FrameMode


class ShowcaseRegression(EvidenceRegression):
    async def finding(self,name):
        # Scroll the real finding list; no direct controller selection or seeking.
        first=next(iter(self.evidence_view.finding_buttons.values()))
        top=first.screen_position_y
        target=self.evidence_view.finding_buttons[name]
        dpi=ui.Workspace.get_dpi_scale()
        x=(first.screen_position_x+40)*dpi;y=(top+65)*dpi
        for _ in range(20):
            if top<=target.screen_position_y and target.screen_position_y+target.computed_height<=top+160:
                await self.click(target);await self.frames(5);return
            ui._ui._inject_mouse_move(x,y);await self.frames(2)
            ui._ui._inject_mouse_scroll(0,-3);await self.frames(5)
        raise AssertionError('Could not scroll to the actual finding '+name)

    async def verify(self):
        await self.wait(lambda:self.presentation is not None,'Control capture did not load')
        self.playback.set_mode(FrameMode.FULL);self.seek(1,snap=False);await self.frames(5)
        self.show_evidence();await self.frames(5)
        self.evidence_view.window.position_x=30;self.evidence_view.window.position_y=60
        self.evidence_view.window.width=650;self.evidence_view.window.height=690;await self.frames(5)
        await self.choose_file(self.evidence_file)
        await self.wait(lambda:not self.evidence_controller.pending and bool(self.evidence_controller.markers),'Control evidence did not bind')
        await self.frames(5)
        assert self.evidence_controller.sets[self.evidence_controller.selected_set].binding.state=='matching-left'
        before=self.frame
        await self.finding('left-handover-1');assert self.frame==before
        await self.choose_role(1)
        marker=self.evidence_controller.selected_marker;assert marker and marker.role=='left'
        assert self.evidence_view.go_button.enabled
        await self.click(self.evidence_view.go_button)
        assert self.frame==marker.frame and self.selection.get_snapshot().paths()==[f'/objects/{marker.instance_id}']
        assert self.go_receipts[-1].evidence_identity(marker.instance_id).object_uid==marker.identity['object_uid']
        first=min(self.evidence_view.point_buttons)
        await self.click(self.evidence_view.point_buttons[first])
        assert self.frame==marker.frame
        await self.click(self.evidence_view.go_button)
        assert self.frame==first and self.go_receipts[-1].request_applied
        if self.artifacts:await self.screenshot(self.artifacts/'control-energy-left.png')
        old_key=self.session.capture_key
        await self.choose_file(self.right_capture,evidence=False)
        await self.wait(lambda:self.session.capture_key!=old_key,'Second initial-condition capture did not open')
        await self.wait(lambda:not self.evidence_controller.pending,'Second role did not finish binding')
        self.show_evidence();await self.frames(5)
        assert self.evidence_controller.sets[self.evidence_controller.selected_set].binding.state=='matching-right'
        await self.finding('right-handover-1');await self.choose_role(2)
        marker=self.evidence_controller.selected_marker;assert marker and marker.role=='right'
        await self.click(self.evidence_view.go_button)
        assert self.frame==marker.frame and self.selection.get_snapshot().paths()==[f'/objects/{marker.instance_id}']
        assert self.go_receipts[-1].evidence_identity(marker.instance_id).api_handle==marker.identity['api_handle']
        stored=self.evidence_controller.sets[self.evidence_controller.selected_set]
        warning=next((f for f in stored.evidence.findings if f.kind=='scalar-window' and f.data['role']=='right' and f.data['measure']['status']=='Warning'),None)
        if warning is not None:
            await self.finding(warning.id);await self.choose_role(2)
            marker=self.evidence_controller.selected_marker;assert marker and marker.status=='Warning'
            await self.click(self.evidence_view.go_button)
            assert self.frame==marker.frame and self.go_receipts[-1].request_applied
        if self.artifacts:await self.screenshot(self.artifacts/'control-energy-right.png')
        assert self.evidence_controller.selection[1]=='right'
        await self.click(self.evidence_view.role_combo);await self.key(265);await self.key(257)
        assert self.evidence_controller.selection[1]=='left' and not self.evidence_view.go_button.enabled
        print('Qualified CPU control GUI gestures passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','evidence','artifacts'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    left,right,evidence,artifacts=(getattr(args,n).resolve() for n in ('left','right','evidence','artifacts'))
    artifacts.mkdir(exist_ok=False,parents=True)
    isolate_gui_output();faulthandler.enable();faulthandler.dump_traceback_later(60)
    app=ShowcaseRegression(left,1280,900,'dark');app.right_capture=right;app.evidence_file=evidence;app.artifacts=artifacts
    try:app.run()
    finally:faulthandler.cancel_dump_traceback_later()
