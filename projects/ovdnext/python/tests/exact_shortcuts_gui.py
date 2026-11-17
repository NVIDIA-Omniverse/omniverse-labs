# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual native key delivery; distinct from screenshot-first Inspector proof."""
from pathlib import Path
import tempfile
import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel
from ovd_ui.playback import FrameMode
from ovd_ui.widgets.object_browser import ObjectBrowser
from ovd_ui.widgets.goto_frame import GoToFrameDialog
from gui_runtime import isolate_gui_output
from test_exact_shortcuts import command_probe

ROOT=Path(__file__).resolve().parents[2]


async def verify():
    async def frames(count=4):
        for _ in range(count):await ui.next_frame()
    async def key(code,modifiers=()):
        for modifier in modifiers:ui._ui._inject_key_event(modifier,True)
        ui._ui._inject_key_event(code,True)
        await frames(1)
        ui._ui._inject_key_event(code,False)
        for modifier in reversed(modifiers):ui._ui._inject_key_event(modifier,False)
        await frames()
    with tempfile.TemporaryDirectory() as temporary:
        session=Session(str(ROOT/'test/data/test_scene.ovd'),temporary)
        model=ObjectTreeModel();model.populate(PvdDataAdapter(session.presentation(frame=1)))
        app,calls=command_probe()
        app.playback.reset(4,ovd=True);app.playback.set_mode(FrameMode.FULL)
        app.window=ui.Window('Exact shortcut delivery',width=800,height=600,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR,fill_app_window=True)
        app.window.set_key_pressed_fn(app._key_pressed)
        selected=[];jumps=[]
        goto=GoToFrameDialog(lambda frame,context:jumps.append((frame,context)))
        app.show_goto_frame=lambda:goto.show(app.playback,('owned capture',))
        with app.window.frame:
            app.object_browser=ObjectBrowser(model,lambda *_:None,
                on_find_object=lambda identity:(selected.append(identity),True)[1],
                search_window_focused=lambda:app.window.focused)
        for theme in ('dark','light'):
            style.set_theme(theme);app.window.focus();app._set_viewport_keyboard_focus(True)
            await frames()
            calls.clear()
            await key(82,(341,342))  # Ctrl+Alt+R cannot reset.
            await key(70,(341,342))  # Ctrl+Alt+F cannot fall through to F.
            await key(32,(341,))
            assert not calls,(theme,calls)
            await key(82,(342,));assert calls==['reset_camera']
            calls.clear();await key(70,(341,343))
            assert app.object_browser.find_dialog is None or not app.object_browser.find_dialog.window.visible
            assert not calls
            await key(70,(341,))
            find=app.object_browser.find_dialog
            assert find.window.visible and find.window.focused and not app.window.focused
            find.expression.set_value('sphere');await frames()
            before=len(selected)
            await key(335)
            assert find.focus=='results' and len(find.model.items)==1 and len(selected)==before
            await key(335)
            assert selected[-1:]==[1] and len(selected)==before+1 and not find.window.visible
            app.window.focus();await frames()
            await key(71,(341,343))
            assert goto.window is None or not goto.window.visible
            await key(71,(341,))
            assert goto.window.visible and goto.window.focused and not app.window.focused
            goto.model.set_value('3');await frames()
            before=len(jumps)
            await key(335,(343,))
            assert goto.window.visible and len(jumps)==before
            await key(335)
            assert not goto.window.visible and jumps[-1]==(3,('owned capture',)) and len(jumps)==before+1
        assert session.scenes()[0].current_frame==1
        goto.destroy();app.object_browser.destroy();app.window.destroy();await frames(2)
        print('Native exact modifier and keypad delivery passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Exact shortcut delivery',width=800,height=600,max_fps=60)
    style.startup('dark');ui.run(run())
    if errors:raise errors[0]
