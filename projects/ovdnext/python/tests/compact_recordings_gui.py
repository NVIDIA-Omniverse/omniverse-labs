# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned native widgets: compact scrolling, selection and resize restoration."""
import argparse
from pathlib import Path
import tempfile

import omni.ui as ui

from gui_runtime import hide_owned_window, isolate_gui_output
from ovd_ui import style
from ovd_ui.recordings import RecordingLibrary
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow
from ovd_ui.widgets.activity_panel import ActivityPanel
from ovd_ui.widgets.recordings_view import RecordingsView

ROOT = Path(__file__).resolve().parents[2]


async def verify(theme):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = RecordingLibrary(root / 'recordings')
        library.ensure_directory()
        for name, stamp in (('Baseline.ovd', 100), ('Latest.ovd', 200)):
            path = library.directory / name
            path.write_bytes((ROOT / 'test/data/test_scene.ovd').read_bytes())
            library.upsert(path, recorded_at=stamp)
        workflow = CaptureWorkflow(Preferences(root / 'prefs.json'), library)
        opened = []
        window = ui.Window('Compact recordings regression', width=1000, height=600)

        def activate(page):
            if page.view is None and page.key == 'recordings':
                page.view = RecordingsView(workflow, opened.append, lambda _: None, lambda: None, page.host)
            elif page.view is None:
                with page.host.frame:
                    ui.Label(page.title)

        with window.frame:
            with ui.VStack():
                ui.Spacer()
                panel = ActivityPanel(window, activate, lambda view: view.destroy(), workflow.settings)
        panel.available('messages', True)
        panel.select('recordings')

        async def frames():
            for _ in range(12):
                panel.poll()
                view = panel.pages['recordings'].view
                if view:
                    view.poll()
                await ui.next_frame()

        try:
            await frames()
            view = panel.pages['recordings'].view
            assert [item.entry.path.name for item in view.model.items] == ['Latest.ovd', 'Baseline.ovd']
            scroll = view.window.scroll
            for height in (600, 800, 600):
                window.height = height
                await frames()
                assert view.table.outer.computed_height >= 60, 'Recording rows must retain a usable viewport'
                scroll.scroll_y = 0
                await frames()
                row_top = view.table.header.screen_position_y + view.table.header.computed_height
                clip_bottom = scroll.screen_position_y + scroll.computed_height
                assert min(28, clip_bottom - row_top) >= 20, 'First row is clipped out of the page'
                for index, item in enumerate(view.model.items):
                    view.tree.selection = [item]
                    await frames()
                    row_top = view.tree.screen_position_y + index * 28
                    scroll.scroll_y = min(scroll.scroll_y_max, max(0,
                        scroll.scroll_y + row_top - scroll.screen_position_y))
                    await frames()
                    assert view.tree.computed_height >= 28
                    row_top = view.tree.screen_position_y + index * 28
                    visible_top = max(row_top, scroll.screen_position_y)
                    visible_bottom = min(row_top + 28, scroll.screen_position_y + scroll.computed_height)
                    assert visible_bottom - visible_top >= 20, 'Selected recording row remains clipped'
                    assert view.selected_entry().path == item.entry.path
                    view.activate()
                    assert opened[-1] == item.entry.path
                scroll.scroll_y = scroll.scroll_y_max
                await frames()
                recent_bottom = view.recent_section.screen_position_y + view.recent_section.computed_height
                assert recent_bottom <= scroll.screen_position_y + scroll.computed_height + 1
                if height == 600:
                    assert scroll.scroll_y_max > 0 and scroll.scroll_y > 0
                selected = view.tree.selection[0]
                for key in ('clips', 'messages', 'recordings'):
                    assert panel.select(key)
                    await frames()
                    assert panel.active == key
                    assert panel.pages[key].host.frame.computed_height > 0
                assert view.tree.selection == [selected]
            print('Compact rows, scrolling, selection and resize checks passed', flush=True)
        finally:
            panel.destroy()
            window.destroy()
            workflow.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', required=True)
    args = parser.parse_args()
    errors = []

    async def run():
        try:
            await verify(args.theme)
        except Exception as error:
            errors.append(error)
        finally:
            ui.shutdown()

    isolate_gui_output()
    ui.init('Compact recordings regression', width=1280, height=900, max_fps=60)
    hide_owned_window()
    style.startup(args.theme)
    ui.run(run())
    if errors:
        raise errors[0]
