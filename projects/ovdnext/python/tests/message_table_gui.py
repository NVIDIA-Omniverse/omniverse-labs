# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native widget lifecycle checks; Inspector captures separately prove user input."""
from gui_runtime import isolate_gui_output, hide_owned_window
import csv
from pathlib import Path
import tempfile
from unittest.mock import patch

import omni.ui as ui
import _pvd_native as native
from ovd_ui import style
from ovd_ui.analysis import MessageSeverity
from ovd_ui.settings import Preferences
from ovd_ui.widgets.activity_panel import ActivityPanel
from ovd_ui.widgets.analysis_views import MessagesView
from ovd_ui.widgets.workspace_tabs import page_surface
from test_message_table import make_messages_capture


async def verify():
    with tempfile.TemporaryDirectory() as directory:
        session=native.Session(str(make_messages_capture(directory)),directory)
        window=ui.Window('Recorded message regression',width=1000,height=800)
        jumps=[]
        def activate(page):
            if page.key=='messages' and page.view is None:
                page.view=MessagesView(session,jumps.append,page_surface,host=page.host)
        with window.frame:
            with ui.VStack():
                ui.Spacer()
                panel=ActivityPanel(window,activate,lambda v:v.destroy(),Preferences(Path(directory)/'prefs.json'))
        panel.available('messages',True);panel.select('messages')
        async def frames():
            for _ in range(5):
                panel.poll()
                view=panel.pages['messages'].view
                if view:view.poll()
                await ui.next_frame()
        await frames()
        view=panel.pages['messages'].view
        panel.saved_height=160;await frames()
        assert panel.container.computed_height==240
        panel.select('clips');await frames()
        assert panel.container.computed_height==160
        panel.select('messages');await frames()
        assert panel.container.computed_height==240
        assert [item.values[1].as_string for item in view.model.items]==[
            'eDEBUG_INFO','eINVALID_PARAMETER','eINVALID_PARAMETER','eDEBUG_WARNING','eDEBUG_INFO','(unknown)']
        selected=view.model.items[1];view.tree.selection=[selected]
        view.pending_sort=0;await frames()
        assert view.descending and [i.record.frame for i in view.model.items]==[11,10,3,2,2,0]
        assert view.tree.selection==[selected] and selected.selected
        view._jump();assert jumps==[2]
        view.pending_sort=3;await frames()
        assert [i.record.location for i in view.model.items][:2]==['A.cpp:10','A.cpp:2']
        assert view.tree.selection==[selected]
        view.search.model.set_value('radius');await frames()
        assert len(view.model.items)==1 and not view.tree.selection
        view.severity=MessageSeverity.INFO;view._filter();await frames()
        assert not view.model.items and view.adapter.total==6
        view.severity=None;view.search.model.set_value('');await frames()
        zero=next(i for i in view.model.items if i.record.frame==0)
        view.tree.selection=[zero];view._jump();assert jumps==[2]
        # Table width changes preserve the header/body contract and active sort.
        grip=view.table.grips[1]
        grip.press(200,300,0,0);grip.release(260,300,0);await frames()
        assert view.table.displayed[1]==220 and view.sort_key=='location'
        assert [float(w) for w in view.tree.column_widths]==list(view.table.displayed)
        assert view.table.buttons[0].tooltip=='Sort numerically by Frame'
        assert view.table.buttons[1].tooltip=='Sort by Type'
        style.startup('light');await frames()
        assert view.tree.selection==[zero] and view.table.displayed[1]==220
        # A save dialog owns the exact rows/order chosen when it opened, even
        # if filters change before its callback completes.
        class Dialog:
            def __init__(self,callback,**kwargs):self.callback=callback
            def show(self):pass
            def destroy(self):pass
        with patch('ovd_ui.widgets.analysis_views.CaptureFileDialog',Dialog):
            expected=[i.record.text for i in view.model.items]
            view.export();save=view.export_dialog.callback
            view.search.model.set_value('radius');await frames()
            output=Path(directory)/'export.csv';save(output)
            with output.open(newline='') as stream:rows=list(csv.reader(stream))
            assert [row[3] for row in rows[1:]]==expected and len(rows)==7
        panel.select('clips');await frames()
        panel.select('messages');await frames()
        assert view.search.model.as_string=='radius' and view.sort_key=='location'
        assert view.table.displayed[1]==220 and panel.pages['messages'].view is view
        panel.destroy();window.destroy()
        print('Native message table checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Recorded message regression',width=1000,height=800,max_fps=60)
    hide_owned_window()
    style.startup('dark');ui.run(run())
    if errors:raise errors[0]
