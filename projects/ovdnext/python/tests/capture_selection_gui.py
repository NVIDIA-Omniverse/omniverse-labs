# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual OvUI mouse actions against a native TCP Session; only this window is owned."""
from gui_runtime import hide_owned_window, isolate_gui_output
import asyncio
import faulthandler
import os
from pathlib import Path
import socket
import struct
import time

import omni.ui as ui

from ovd_ui.application import Application
from test_native_capture_control import (ACTOR, CONTEXT, SCENE, META, CATALOG, ack, catalog,
    create, field, metadata, no_reverse_bytes, receive, schema)


class SelectionRegression(Application):
    async def frames(self, count=4):
        for _ in range(count): await ui.next_frame()

    async def until(self, predicate, timeout=6):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, self.error_message
            await ui.next_frame()

    async def click(self, widget):
        await self.until(lambda: widget.computed_width > 0 and widget.computed_height > 0)
        dpi = ui.Workspace.get_dpi_scale()
        ui._ui._inject_mouse_move((widget.screen_position_x + widget.computed_width / 2) * dpi,
                                  (widget.screen_position_y + widget.computed_height / 2) * dpi)
        await self.frames(1)
        ui._ui._inject_mouse_button(0, True)
        await self.frames(1)
        ui._ui._inject_mouse_button(0, False)
        await self.frames()

    async def connect(self, capabilities=12):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0)); port = reserved.getsockname()[1]
        assert self.workflow.attach(port)
        for _ in range(100):
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=.2)
                break
            except (ConnectionRefusedError, TimeoutError): await self.frames(1)
        else: raise AssertionError("Listener failed")
        connection.settimeout(2)
        connection.sendall(struct.pack("<IHHIII", 0x4f56444c, 1, capabilities, 0, 4, 0)
            + schema(1, "PxOmniPvdMetaData", META) + schema(2, "PxOmniPvdCaptureEntry", CATALOG)
            + metadata() + (catalog() if capabilities & 8 else b""))
        await self.until(lambda: self.workflow.capture_selection.info.get("policy_known"))
        return connection

    async def verify(self):
        await self.until(lambda: self.presentation is not None)
        displayed = self.session
        assert hasattr(self, "show_recording_selection"), "Application panel entry is missing"
        peer = await self.connect()
        try:
            self.show_live()
            await self.frames()
            await self.click(self.recording_selection_button)
            panel = self.capture_selection_view
            assert panel.window.visible and not self.live_dialog.visible
            assert not (panel.window.flags & ui.WINDOW_FLAGS_MODAL)
            assert self.session is displayed and panel.model.session is self.workflow.live_session
            await self.until(lambda: ACTOR in panel._checks)
            assert not panel._checks[CONTEXT].enabled and not panel.apply_button.enabled
            no_reverse_bytes(peer)
            await self.click(panel._checks[ACTOR])
            assert panel.model.selected_count == 1 and panel.apply_button.enabled
            await self.click(panel.apply_button)
            assert receive(peer, 36) == struct.pack("<IIQIIIQ", 0x3343564f, 36, 1, 0, 0, 1, ACTOR)
            peer.sendall(ack(1, 3))
            await self.until(lambda: panel.model.info.get("state") == "busy")
            assert "busy" in panel.status.text
            await self.click(panel.apply_button)
            assert struct.unpack("<Q", receive(peer, 36)[8:16])[0] == 2
            peer.sendall(ack(2, 4))
            await self.until(lambda: panel.model.info.get("state") == "denied")
            # Fresh generation must clear checked rows even when labels/handles match.
            peer.sendall(b"".join(struct.pack("<BQQ", 10, 2, row) for row in (200, 201, 202)) + catalog(2))
            await self.until(lambda: panel.model.generation == 2)
            assert panel.model.selected_count == 0 and not panel.apply_button.enabled
            await self.click(panel.scope.buttons[1])
            await self.click(panel.policy.buttons[1])
            no_reverse_bytes(peer)
            await self.click(panel.apply_button)
            assert receive(peer, 28) == struct.pack("<IIQIII", 0x3343564f, 28, 3, 1, 0, 0)
            await self.until(lambda: panel.model.info.get("state") == "unknown", timeout=12)
            assert not panel.apply_button.enabled and not panel.refresh_button.enabled
            await self.click(panel.supersede_check)
            assert panel.apply_button.enabled
            await self.click(panel.apply_button)
            assert struct.unpack("<Q", receive(peer, 28)[8:16])[0] == 4
            peer.sendall(ack(3, 1, 8))
            await self.until(lambda: panel.model.info.get("ack_id") == 3)
            assert panel.model.info["request_id"] == 4 and panel.model.info["state"] == "awaiting_ack"
            peer.sendall(ack(4, 1, 9))
            await self.until(lambda: panel.model.info.get("state") == "applied")
            assert "waiting for the first complete frame" in panel.status.text
            assert self.session is displayed
            no_reverse_bytes(peer)
            await self.click(panel.refresh_button)
            assert receive(peer, 28) == struct.pack("<IIQIII", 0x3344564f, 28, 5, 0, 0, 0)
            peer.sendall(b"".join(struct.pack("<BQQ", 10, 2, row) for row in (200, 201, 202))
                + catalog(3) + ack(5, 1, 9))
            await self.until(lambda: panel.model.generation == 3 and panel.model.info.get("state") == "applied")
            assert panel.model.info["capture_epoch"] == 9 and panel.model.all and panel.model.secondary
            assert "Catalog refresh acknowledged" in panel.status.text
            assert not self.workflow.live_session.refresh_live().ready and self.session is displayed
            no_reverse_bytes(peer)
            if os.environ.get("OVDNEXT_CAPTURE_SCREENSHOT"):
                # The pinned synchronous testing helper stops a running asyncio
                # loop. Schedule on the existing pump instead of nesting a tick.
                assert ui._ui._schedule_screenshot(os.environ["OVDNEXT_CAPTURE_SCREENSHOT"])
                await self.until(ui._ui._poll_screenshot_done)
            peer.close()
            await self.until(lambda: len(self.workflow.session_recordings) == 1)
            assert self.workflow.session_recordings[0].status == "Empty"
            assert self.session is displayed and not self.workflow.recordings.entries
        finally:
            peer.close()
        peer = await self.connect(4)
        try:
            await self.frames()
            assert not panel.scope.buttons[0].enabled and not panel.refresh_button.enabled
            await self.click(panel.scope.buttons[1])
            await self.click(panel.apply_button)
            assert receive(peer, 28) == struct.pack("<IIQIII", 0x3343564f, 28, 1, 0, 0, 0)
            self.workflow.detach()
            await self.frames()
            assert panel.model.session is None and not panel.apply_button.enabled
            self.workflow.new()
            await self.frames()
            assert panel.model.selected_count == 0 and not panel.model.all
        finally:
            peer.close()
        peer = await self.connect(8)
        try:
            data = bytearray(b"".join(struct.pack("<BQQ", 10, 2, row) for row in (200, 201, 202)))
            for index in range(260):
                data += create(2, 2, 1000 + index)
                for attribute, value in zip(range(20, 26), [struct.pack("<Q", ACTOR + 100 + index),
                        b"PxRigidDynamic", f"actor-{index:04d}".encode(), struct.pack("<Q", 0),
                        struct.pack("<Q", SCENE), b"\1"]):
                    data += field(2, 1000 + index, attribute, value)
            peer.sendall(data + field(1, 100, 11, struct.pack("<Q", 2)))
            await self.until(lambda: panel.model.generation == 2)
            await self.frames()
            assert len(panel.rows.items) == 128 and 0 < len(panel._checks) < 128
            assert panel.model.page["total"] == 260
            await self.click(panel.next)
            assert panel.model.offset == 128 and len(panel.rows.items) == 128
            await self.click(panel.next)
            assert panel.model.offset == 256 and len(panel.rows.items) == 4 and not panel.next.enabled
            await self.click(panel.previous)
            assert panel.model.offset == 128 and len(panel.rows.items) == 128
            panel.prefix.model.set_value("actor-025")
            panel.page(0, SCENE)
            await self.frames()
            assert panel.model.page["total"] == 10 and len(panel.rows.items) == 10
            assert not panel.apply_button.enabled and panel.refresh_button.enabled
            no_reverse_bytes(peer)
            self.workflow.detach()
        finally:
            peer.close()
        # Both independently added features must release their real resources.
        self.show_evidence()
        await self.frames()
        assert self.evidence_view.window is not None and self.evidence_view.window.visible
        assert panel.window is not None and self.finding_track.widget is not None
        print("Recording selection actual-click checks passed", flush=True)

    async def run_async(self):
        hide_owned_window()
        task = asyncio.create_task(super().run_async())
        try: await self.verify()
        finally:
            self.request_exit()
            await task


if __name__ == "__main__":
    isolate_gui_output()
    root = Path(__file__).resolve().parents[2]
    application = SelectionRegression(str(root / "test/data/multi_geom.ovd"), 1280, 900, "dark")
    faulthandler.enable()
    faulthandler.dump_traceback_later(40)
    try:
        application.run()
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert application.closed and application.workflow.closed
    assert application.capture_selection_view._closed and application.capture_selection_view.window is None
    assert application.evidence_controller._closed and not application.evidence_controller.pending
    assert application.evidence_view.window is None and application.finding_track.widget is None
    assert not application.capture_selection_view.apply() and not application.capture_selection_view.refresh_catalog()
    print("Evidence and recording selection resources closed", flush=True)
