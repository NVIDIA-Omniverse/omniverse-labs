# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render populated panels through the actual application, without desktop input."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import omni.ui as ui

from gui_runtime import hide_owned_window, isolate_gui_output, size_hidden_client
from ovd_ui.application import Application

ROOT = Path(__file__).resolve().parents[2]


def bounds(widget):
    dpi = ui.Workspace.get_dpi_scale()
    return [round(value * dpi) for value in (widget.screen_position_x, widget.screen_position_y,
                                            widget.computed_width, widget.computed_height)]


class PopulatedPanels(Application):
    async def frames(self, count=15):
        for _ in range(count):
            await ui.next_frame()

    async def until(self, predicate):
        deadline = time.monotonic() + 20
        while not predicate():
            assert time.monotonic() < deadline, self.error_message
            await ui.next_frame()

    async def verify(self):
        await self.until(lambda: self.presentation is not None)
        self.playback.pause()
        if args.panel == "graph":
            job = self.analysis_tasks.submit_frame_timing(self.capture, self.scene_id, 1, 250)
            graph = self._open_plot(job, "Recorded cloth frame timing", lambda frame: None)
            await self.until(lambda: graph.finished)
            assert graph.model and graph.model.has_values, graph.progress.text
            assert len(graph.series.frames) == 249
            await self.frames()
            report = {"canvas": bounds(graph.canvas), "frames": len(graph.series.frames)}
        else:
            library = self.workflow.recordings
            library.ensure_directory()
            source = (ROOT / "test/data/test_scene.ovd").read_bytes()
            for name, stamp in (("Baseline.ovd", 1793960400), ("Latest iteration.ovd", 1793960460)):
                path = library.directory / name
                path.write_bytes(source)
                library.upsert(path, recorded_at=stamp)
            self.activity.select("recordings")
            await self.frames()
            table = self.recordings_view
            report = {"outer": bounds(table.table.outer), "header": bounds(table.table.header),
                      "tree": bounds(table.tree), "panel": bounds(self.activity.container),
                      "rows": [[value.as_string for value in item.values] for item in table.model.items]}
        report["graphics"] = {name: self.renderer.context.info[name] for name in ("GL_RENDERER", "GL_VERSION")}
        args.output.mkdir(parents=True, exist_ok=True)
        assert ui._ui._schedule_screenshot(str(args.output / "panel.png"))
        await self.until(ui._ui._poll_screenshot_done)
        (args.output / "panel.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    async def run_async(self):
        hide_owned_window()
        size_hidden_client(self.width, self.height)
        task = asyncio.create_task(super().run_async())
        try:
            await self.verify()
        finally:
            self.request_exit()
            await task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("graph", "recordings"), required=True)
    parser.add_argument("--theme", choices=("dark", "light"), required=True)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    isolate_gui_output()
    name = "cloth_physx34.pxd2" if args.panel == "graph" else "test_scene.ovd"
    app = PopulatedPanels(ROOT / "test/data" / name, 1280, args.height, args.theme)
    app.run()
    assert app.closed and app.workflow.closed
