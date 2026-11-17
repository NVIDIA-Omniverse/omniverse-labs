# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise grouping reparenting in the real OVUI TreeView across drawn frames."""
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from gui_runtime import isolate_gui_output, hide_owned_window
from pathlib import Path
import tempfile

import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets.object_browser import ObjectBrowser

ROOT = Path(__file__).resolve().parents[2]


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        session = Session(str(ROOT / "test/data/multi_geom.ovd"), temporary)
        model = ObjectTreeModel()
        model.populate(PvdDataAdapter(session.presentation(frame=1)))
        ids=multi_geom_roles(session)
        events = []
        window = ui.Window("Grouping selection regression", width=640, height=700,
                           flags=ui.WINDOW_FLAGS_NO_TITLE_BAR)
        with window.frame:
            browser = ObjectBrowser(model,
                lambda items: events.append(tuple(item.key for item in items)),
                on_action=lambda *_: None)
        for _ in range(3):
            await ui.next_frame()
        # IDs 31 and 29 reproduce the application failure. ID 33 adds a deep
        # actor path when returning to scene ownership from class grouping.
        selected = [model.canonical_item(identity) for identity in (ids.physics, ids.gpu, ids.sphere)]
        for item in selected:
            browser.select_item(item)
        browser.tree.selection = selected
        for _ in range(3):
            await ui.next_frame()
        assert browser.tree.selection == selected
        expected = tuple(item.key for item in selected)
        for index in (1, 0, 1, 0):
            browser.collapse_all()
            events.clear()
            browser.grouping.model.get_item_value_model().set_value(index)
            for _ in range(8):
                await ui.next_frame()
                assert browser.tree.selection == selected, (index, events)
                assert events == [expected], (index, events)
            assert all(browser.tree.is_expanded(ancestor)
                       for item in selected for ancestor in model.ancestors(item))
        # A container absent from the alternate projection must disappear from
        # selection without substituting all of its native descendants.
        # The Physics root is paged at the current default group size. Scene 30
        # has direct class containers, which exercise this projection change.
        group = next(item for item in model.get_item_children(model.canonical_item(ids.scene))
                     if item.kind == "group")
        browser.select_item(group)
        events.clear()
        browser.grouping.model.get_item_value_model().set_value(1)
        for _ in range(8):
            await ui.next_frame()
            assert browser.tree.selection == []
            assert events == [()]
        # The callback guard must not swallow an ordinary later user selection.
        browser.select_item(model.canonical_item(ids.sphere))
        assert events[-1] == (f"object:{ids.sphere}",)
        print("Native tree grouping checks passed", expected, flush=True)
        browser.destroy()
        window.destroy()
        for _ in range(2):
            await ui.next_frame()


if __name__ == "__main__":
    errors = []

    async def run():
        try:
            await verify()
        except Exception as error:
            errors.append(error)
        finally:
            ui.shutdown()

    isolate_gui_output()
    ui.init("Grouping selection regression", width=640, height=700, max_fps=60)
    hide_owned_window()
    style.startup("dark")
    ui.run(run())
    if errors:
        raise errors[0]
