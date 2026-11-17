# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real input-event regression; separate from strict screenshot-first UI evidence."""
from pathlib import Path
import tempfile

import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.application import Application
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets.object_browser import ObjectBrowser
from gui_runtime import isolate_gui_output

ROOT=Path(__file__).resolve().parents[2]


async def verify():
    async def frames(count=4):
        for _ in range(count):await ui.next_frame()
    async def key(code,modifier=None):
        if modifier is not None:ui._ui._inject_key_event(modifier,True)
        ui._ui._inject_key_event(code,True)
        await frames(1)
        ui._ui._inject_key_event(code,False)
        if modifier is not None:ui._ui._inject_key_event(modifier,False)
        await frames()
    async def click(field,x=20,y=12):
        # Automated geometry-targeted input, not screenshot-first evidence.
        ui._ui._inject_mouse_move(field.screen_position_x+x,field.screen_position_y+y)
        await frames(1)
        ui._ui._inject_mouse_button(0,True)
        await frames(1)
        ui._ui._inject_mouse_button(0,False)
        await frames()
    with tempfile.TemporaryDirectory() as temporary:
        session=Session(str(ROOT/'test/data/test_scene.ovd'),temporary)
        model=ObjectTreeModel();model.populate(PvdDataAdapter(session.presentation(frame=1)))
        app=Application.__new__(Application)
        app.navigation=None
        app.viewport_keyboard_focus=app.object_keyboard_focus=False
        app.window=ui.Window('Hierarchy search keys',width=800,height=600,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR,fill_app_window=True)
        app.window.set_key_pressed_fn(app._key_pressed)
        events=[]
        with app.window.frame:
            with ui.HStack():
                with ui.Frame(width=400):
                    browser=ObjectBrowser(model,lambda rows:events.append(tuple(row.instance_id for row in rows)),
                        search_window_focused=lambda:app.window.focused)
                with ui.VStack():
                    unrelated=ui.StringField(ui.SimpleStringModel('Unrelated field'),height=30)
                    ui.Label('Moving the pointer here must not take editing focus')
        app.object_browser=browser
        for theme in ('dark','light'):
            style.set_theme(theme)
            await frames()
            # The circle and blank gutter are separate native click targets;
            # only clicking the input itself would miss their focus contract.
            for gutter,x,y in ((browser.search_prefix,18,16),(browser.search_suffix,3,16)):
                await click(unrelated)
                assert not browser._search_editing
                await click(gutter,x,y)
                assert browser._search_editing,'Search gutter did not focus native InputText'
                assert browser.search_background.selected
            await click(browser.search_field)
            browser.search_model.set_value('sphere')
            await frames()
            assert browser._search_editing
            ui._ui._inject_mouse_move(700,450)
            await frames()
            events.clear()
            for code,modifier,expected in ((257,None,3),(335,None,1),(257,528,3),(335,528,1)):
                await key(code,modifier)
                assert events[-1:]==[(expected,)],(theme,code,modifier,events)
                assert browser._search_editing,'Enter did not restore the editing field'
                assert browser.search_editor_model.as_string=='sphere'
            assert events==[(3,),(1,),(3,),(1,)],events
            # A different field in the same focused window must not navigate.
            await click(unrelated)
            assert not browser._search_editing
            events.clear();await key(257)
            assert not events
            # Filter options uses the authoritative shared model, not the
            # sidebar's editor model. Native begin-edit must remain distinct.
            other=ui.Window('Filter editor ownership',width=400,height=140)
            other.setPosition(410,100)
            with other.frame:
                filter_field=ui.StringField(browser.search_model,height=30)
            await frames();await click(filter_field)
            assert other.focused and not app.window.focused and not browser._search_editing
            events.clear();await key(335)
            assert not events
            other.destroy();await frames()
            app.window.focus();await frames()
            await key(257)
            assert not events,'Closing another editor granted sidebar ownership'
        assert session.scenes()[0].current_frame==1
        browser.destroy();app.window.destroy();await frames(2)
        print('Native hierarchy search keys passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Hierarchy search keys',width=800,height=600,max_fps=60)
    style.startup('dark')
    ui.run(run())
    if errors:raise errors[0]
