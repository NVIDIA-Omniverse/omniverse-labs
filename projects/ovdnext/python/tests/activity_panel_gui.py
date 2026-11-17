# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI lifecycle checks; these are programmatic, not Inspector gestures."""
from gui_runtime import isolate_gui_output, hide_owned_window
from pathlib import Path
import tempfile
from types import SimpleNamespace

import omni.ui as ui
import _pvd_native as native
from ovd_ui import style
from ovd_ui.application import Application
from ovd_ui.playback import Playback
from ovd_ui.settings import Preferences
from ovd_ui.widgets.activity_panel import ActivityPanel
from ovd_ui.widgets.workspace_tabs import WorkspaceTabs

ROOT=Path(__file__).resolve().parents[2]


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        settings=Preferences(Path(temporary)/'preferences.json')
        session=native.Session(str(ROOT/'test/data/testRecordMessage.ovd'),temporary)
        app=Application.__new__(Application)
        app.session,app.scene_id=session,0
        app.analysis_views=[]
        app._set_viewport_keyboard_focus=lambda _:None
        app.playback=Playback();app.playback.reset(922,False)
        context=SimpleNamespace(capture_key=session.capture_key,segment_epoch=0,scene_id=0,frame=1)
        available=SimpleNamespace(enabled=True,reason='')
        app.analysis_availability=SimpleNamespace(context=context,state=lambda _:available)
        jumps=[]
        app._analysis_frame=lambda *args:jumps.append(args)
        window=ui.Window('Bottom workspace regression',width=1000,height=800)
        app.window=window
        with window.frame:
            with ui.VStack(spacing=0):
                app.workspace=WorkspaceTabs(window,lambda _:None,app._dispose_analysis)
                with app.workspace.render_frame:ui.Label('Render stays independent')
                app.activity=ActivityPanel(window,app._activate_activity,app._dispose_analysis,settings)
        panel=app.activity
        async def frames(count=6):
            for _ in range(count):
                app._sync_activity();panel.poll();app.workspace.poll()
                for view in app.analysis_views:view.poll()
                await ui.next_frame()
        await frames()
        assert not panel.expanded and panel.container.computed_height==40
        assert tuple(panel.pages)==('clips','messages','recordings')
        assert not any(panel.close(key) for key in panel.pages)
        panel.select('messages');await frames()
        page=panel.pages['messages'];first=page.view
        assert first.adapter.total==1 and len(first.model.items)==1
        assert not first.tree.root_visible
        first.tree.selection=[first.model.items[0]]
        first.text=first.model.items[0].record.text[:15];first._filter()
        first.tree.selection=[first.model.items[0]]
        selected=first.tree.selection[0]
        panel.select('clips');await frames()
        assert not first.window.visible and app.workspace.active=='render'
        panel.select('messages');await frames()
        assert page.view is first and first.tree.selection[0] is selected
        context.frame=2;await frames()
        assert page.view is first and first.text and first.tree.selection[0] is selected
        stale_jump=first.on_frame
        stale_jump(1);assert jumps==[(session,0,1)]
        # Capture/segment/scene replacements dispose the old native owner and
        # invalidate callbacks, while an ordinary frame change retains filters.
        context.segment_epoch=1;await frames()
        assert page.view is not first and first.window.closed is False  # host reused
        assert page.view.text=='' and len(app.analysis_views)==1
        stale_jump(1);assert len(jumps)==1
        available.enabled=False;available.reason='Legacy capture'
        await frames()
        assert panel.active=='clips' and not panel.select('messages')
        # A held horizontal grip follows Y outside its original hit rectangle.
        grip=panel.grip
        grip.press(200,400,0,0);grip.hover(False);grip.move(900,320)
        await frames()
        assert grip.anchor and grip.rule.visible and panel.container.computed_height==320
        assert settings.get('soft_glass_activity_height') is None
        grip.release(900,320,0)
        assert grip.anchor is None and grip.rule.visible==grip.hovered
        await frames()
        assert settings.get('soft_glass_activity_height')==320
        # Native frame polling can refresh hover from the host pointer even
        # while hidden. Check the explicit leave callback before another poll.
        grip.hover(False)
        assert not grip.rule.visible
        panel.set_expanded(False);await frames()
        assert panel.container.computed_height==40 and not first.window.visible
        panel.set_expanded(True);await frames()
        assert panel.container.computed_height==320
        window.height=600;await frames()
        assert panel.container.computed_height==180 and panel.saved_height==320
        window.height=800;await frames()
        assert panel.container.computed_height==320
        panel.destroy();app.workspace.destroy();window.destroy()
        assert not app.analysis_views
        # A fresh preferences reader and widget restore the accepted height/open state.
        restored=Preferences(settings.path)
        window=ui.Window('Restored bottom workspace',width=1000,height=800)
        with window.frame:
            with ui.VStack():
                ui.Spacer()
                panel=ActivityPanel(window,lambda _:None,lambda v:v.destroy(),restored)
        for _ in range(6):panel.poll();await ui.next_frame()
        assert panel.expanded and panel.container.computed_height==320 and panel.active=='clips'
        panel.destroy();window.destroy()
        for value,expected in ((True,240),('240',240),(float('nan'),240),(-100,160),(9999,420)):
            restored.values['soft_glass_activity_height']=value
            window=ui.Window('Invalid saved extent',width=1000,height=900)
            with window.frame:panel=ActivityPanel(window,lambda _:None,lambda v:None,restored)
            assert panel.saved_height==expected
            panel.destroy();window.destroy()
        print('Native bottom workspace checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Bottom workspace regression',width=1000,height=800,max_fps=60)
    hide_owned_window()
    style.startup('dark')
    ui.run(run())
    if errors:raise errors[0]
