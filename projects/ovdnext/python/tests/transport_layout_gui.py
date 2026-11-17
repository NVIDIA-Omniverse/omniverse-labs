# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual application transport layout; not Inspector user-flow QA."""
from gui_runtime import isolate_gui_output, hide_owned_window, size_hidden_client
import asyncio
from pathlib import Path
from types import SimpleNamespace

import omni.ui as ui

from ovd_ui.application import Application
from ovd_ui.playback import FrameMode


class TransportRegression(Application):
    async def verify_transport(self):
        async def frames(count=4):
            for _ in range(count):await ui.next_frame()
        for _ in range(600):
            if self.presentation is not None:break
            await ui.next_frame()
        assert self.presentation is not None, self.error_message
        self.playback.set_mode(FrameMode.FULL)
        await frames()
        for theme in ("dark", "light"):
            if self.theme != theme:self.toggle_theme()
            await frames()
            ruler = self.frame_slider._ruler
            width = self.frame_slider.widget.computed_width
            for frame in (9, 10, 18, 19, 20, 21, 98, 99, 100, 198, 199, 200, 240):
                self._transport(self.playback.seek, frame)
                await frames(1)
                assert self.frame == self.presentation.frame == frame
                assert self.parity_button.text == self.playback.label
                assert self.frame_slider._badge_label.text == str(frame)
                assert self.frame_slider._ruler is ruler
                assert self.frame_slider.widget.computed_width == width
                assert self.parity_button.computed_width <= self.parity_slot.computed_width
                assert self.parity_button.screen_position_x + self.parity_button.computed_width < self.frame_slider.widget.screen_position_x
        # Isolate transport sizing for very large IDs without requesting a
        # nonexistent native publication. Keep the complete caption visible.
        self.scenes = [SimpleNamespace(id=self.scene_id, frame_count=2**63)]
        self.playback.reset(2**63, ovd=True)
        self.playback.set_mode(FrameMode.FULL)
        self.frame = self.playback.seek(2**63-4)
        self._sync_transport()
        await frames()
        ruler = self.frame_slider._ruler
        for frame in (2**63-3, 2**63-2):
            self.frame = self.playback.seek(frame)
            self._sync_transport()
            await frames()
            assert self.frame_slider._ruler is ruler
            assert self.parity_button.computed_width <= self.parity_slot.computed_width
            assert self.frame_slider.widget.computed_width > 200
        print("Application transport layout checks passed", flush=True)

    async def run_async(self):
        hide_owned_window()
        size_hidden_client(self.width, self.height)
        task = asyncio.create_task(super().run_async())
        try:await self.verify_transport()
        finally:
            self.request_exit()
            await task


if __name__ == "__main__":
    isolate_gui_output()
    root = Path(__file__).resolve().parents[2]
    TransportRegression(str(root / "test/data/multi_geom.ovd"), 1280, 800, "dark").run()
