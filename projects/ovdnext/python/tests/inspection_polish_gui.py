# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise compact hierarchy and recorded contacts in the real OvUI application."""
import argparse
import asyncio
import os
from pathlib import Path
import tempfile
import time

import omni.ui as ui
from gui_runtime import hide_owned_window, isolate_gui_output, size_hidden_client
from ovd_ui.application import Application

ROOT = Path(__file__).resolve().parents[2]


class InspectionRegression(Application):
    async def frames(self, count=12):
        for _ in range(count):
            await ui.next_frame()

    async def capture_image(self, name):
        await self.frames()
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            assert ui._ui._schedule_screenshot(str(self.output / name))
            deadline = time.monotonic() + 10
            while not ui._ui._poll_screenshot_done():
                assert time.monotonic() < deadline, 'Application screenshot timed out'
                await ui.next_frame()

    async def verify(self):
        deadline = time.monotonic() + 30
        while self.presentation is None:
            assert time.monotonic() < deadline, self.error_message
            await ui.next_frame()
        self.playback.pause()
        self.overlay_mask = 32
        assert self._refresh(frame=111)
        await self.frames()
        model = self.object_model
        actor = next(i.id for i in self.adapter.instances.values()
                     if self.adapter.class_name(i.id) == 'PxRigidDynamic')
        shape, = self.adapter.references(actor, 'shapes')
        geometry, = self.adapter.references(shape, 'geom')
        children = model.get_item_children(model.canonical_item(actor))
        assert [child.instance_id for child in children] == [geometry]
        item = children[0]
        self.object_browser.select_item(item)
        self._tree_selection([item])
        self.tree.set_expanded(item, True, False)
        await self.capture_image('compact-shape.png')
        assert self.renderer.instance_count == 57
        assert len(self.presentation.overlay_snapshot().positions) > 0

        group = model._item(f'group:object:{shape}:materials')
        missing, = model.get_item_children(group)
        self.object_browser.select_item(missing)
        self._tree_selection([missing])
        assert self._refresh()
        await self.capture_image('unresolved-material.png')
        assert self.selected_record.kind == 'unresolved'

        group = next(n for n in model._specs.values() if n.name == 'pairsContactNormals')
        assert group.element_count == 584
        item = model._item(group.key)
        while item.kind != 'array_element':
            item = model.get_item_children(item)[0]
        self.object_browser.select_item(item)
        self._tree_selection([item])
        assert self._refresh()
        await self.capture_image('contact-normal.png')
        assert self.selected_record.element_size == 12
        assert len(self.adapter.array_record(item.source_instance_id, item.property_id, 0, 12)) == 12

        # Shape properties remain native data; explicit inspection reveals class view.
        self.selection.publish([f'/objects/{shape}'], source='property')
        await self.frames()
        assert model.grouping == 'class'
        assert [i.instance_id for i in self.tree.selection] == [shape]
        assert self._refresh()
        assert self.adapter.property(shape, 'materials') is not None
        assert self.adapter.session.references(shape, self.adapter.property(shape, 'materials').id) == [-1]
        print('Inspection GUI passed: 584 normals, compact geometry, unresolved material, native shape inspection', flush=True)
        print(self.renderer.context.info['GL_RENDERER'], flush=True)

    async def run_async(self):
        hide_owned_window()
        size_hidden_client(self.width, self.height)
        task = asyncio.create_task(super().run_async())
        try:
            await self.verify()
        finally:
            self.request_exit()
            await task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    isolate_gui_output()
    with tempfile.TemporaryDirectory(prefix='inspection-polish-settings-') as preferences:
        os.environ.update(LOCALAPPDATA=preferences, XDG_DATA_HOME=preferences, XDG_CONFIG_HOME=preferences)
        app = InspectionRegression(ROOT / 'test/data/2026_04_01_15_21_33_0000019f4952a142.ovd',
                                   1600, 1000, 'light')
        app.output = output
        app.run()
        assert app.closed and app.workflow.closed
