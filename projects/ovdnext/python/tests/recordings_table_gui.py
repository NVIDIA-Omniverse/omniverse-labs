# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native widget lifecycle regression; separate Inspector runs test actual input."""
from pathlib import Path
import socket
import struct
import tempfile
import time

import omni.ui as ui

from ovd_ui import style
from ovd_ui.recordings import RecordingLibrary
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow
from ovd_ui.widgets.activity_panel import ActivityPanel
from ovd_ui.widgets.recordings_view import RecordingsView
from gui_runtime import isolate_gui_output,hide_owned_window


ROOT=Path(__file__).resolve().parents[2]


async def verify():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);library=RecordingLibrary(root/'recordings');library.ensure_directory()
        source=(ROOT/'test/data/test_scene.ovd').read_bytes()
        older,newer,live=[library.directory/name for name in ['Older.ovd','Newer.ovd','Incoming.ovd']]
        older.write_bytes(source);newer.write_bytes(source)
        library.upsert(older,recorded_at=100);library.upsert(newer,recorded_at=200)
        workflow=CaptureWorkflow(Preferences(root/'prefs.json'),library)
        opened=[];deleted=[]
        window=ui.Window('Recording table regression',width=1000,height=800)
        def activate(page):
            if page.key=='recordings' and page.view is None:
                page.view=RecordingsView(workflow,opened.append,deleted.append,lambda:None,page.host)
        with window.frame:
            with ui.VStack():
                ui.Spacer()
                panel=ActivityPanel(window,activate,lambda v:v.destroy(),workflow.settings)
        panel.select('recordings')
        async def frames():
            for _ in range(5):
                panel.poll()
                view=panel.pages['recordings'].view
                if view:view.poll()
                await ui.next_frame()
        await frames();view=panel.pages['recordings'].view
        assert [i.entry.path for i in view.model.items]==[newer,older]
        assert all(button is None for button in view.table.buttons)
        selected=view.model.items[1];view.tree.selection=[selected]
        assert view.delete.enabled
        view.activate();assert opened==[older]
        view.window.focus();await frames()
        view.table_focus=True
        view._key(525,ui.Widget.FLAG_WANT_CAPTURE_KEYBOARD,True);view._key(525,0,False);view._key(525,1,True)
        assert opened==[older,older]
        view.focus_at(-1,-1);view._key(525,0,True)
        assert opened==[older,older]
        view.delete_selected();assert deleted==[older]
        workflow.display_path=older;view.refresh();await frames()
        assert view.tree.selection==[selected] and selected.displayed and not view.delete.enabled
        view.delete_selected();assert deleted==[older]
        library.upsert(live,status='Live',recorded_at=300,source=newer)
        view.refresh();await frames()
        assert view.model.items[-1] is selected and view.tree.selection==[selected]
        view.tree.selection=[view.model.items[0]]
        assert not view.delete.enabled and view.selected_entry().status=='Live'
        live.write_bytes(source);library.upsert(live,status='Complete')
        view.activate();assert opened[-1]==live  # Current path, independent of stale row status.
        view.refresh();await frames()
        assert view.delete.enabled  # Completed disk file is not the displayed older capture.
        # Library updates keep native item identity, selection, values and widths.
        selected=view.tree.selection[0]
        grip=view.table.grips[3];grip.press(200,300,0,0);grip.release(280,300,0);await frames()
        widths=view.table.displayed
        library.upsert(live,status='Complete',recorded_at=50)
        view.refresh();await frames()
        assert view.model.items[-1] is selected and view.tree.selection==[selected]
        assert selected.values[1].as_string=='Complete' and view.table.displayed==widths
        assert [float(v) for v in view.tree.column_widths]==list(widths)
        # The original table allows one row, including modifier-based changes.
        view.tree.selection=list(view.model.items);await frames()
        assert view.tree.selection==[view.model.items[-1]]
        library.delete(live);view.refresh();await frames()
        assert not view.tree.selection and not view.delete.enabled
        view.activate();assert opened[-1]==live
        workflow.follow_live=True;workflow.settings.set('keep_recordings',False)
        view.refresh();assert not view.follow.enabled and not view.keep.model.as_bool
        view.follow_latest();await frames()
        assert view.selected_entry().path==older and not view.delete.enabled
        panel.select('clips');await frames();panel.select('recordings');await frames()
        style.startup('light');await frames()
        assert view.table.displayed==widths
        view.recent_section.collapsed=False;await frames()
        assert panel.container.computed_height>=380
        assert view.table.outer.computed_height>=60
        assert view.recent_scroll.computed_height>=30
        view.recent_section.collapsed=True;await frames()
        assert view.window.minimum_height==280
        # Real native TCP sessions keep one row through growth and completion,
        # with and without auto-save, even while an older capture is displayed.
        with socket.socket() as candidate:
            candidate.bind(('127.0.0.1',0));port=candidate.getsockname()[1]
        frame_stop=struct.pack('<BQQ',12,0x138AB174E00,2)
        active_end=source.index(frame_stop)+len(frame_stop)
        saved=[];view.on_save=saved.append
        for count,keep in enumerate((False,True),1):
            workflow.set_keep_recordings(keep)
            assert workflow.attach(port)
            if count==1:assert workflow.open_file(older)
            else:assert workflow.select_session_recording(workflow.session_recordings[0].id)
            view.refresh();await frames()
            previous=view.tree.selection[0]
            deadline=time.monotonic()+5
            while True:
                try:
                    connection=socket.create_connection(('127.0.0.1',port),timeout=.1)
                    break
                except (ConnectionRefusedError,TimeoutError):
                    assert time.monotonic()<deadline
                    workflow.poll();await frames()
            with connection:
                connection.sendall(struct.pack('<IHH',0x4F56444C,1,0)+source[:12])
                while workflow.live_recording is None or workflow.live_recording.size_bytes!=12:
                    assert time.monotonic()<deadline
                    workflow.poll();await ui.next_frame()
                view.refresh();await frames()
                row=view.model.items[0]
                assert row.entry.status=='Live' and row.entry.size_bytes==12
                identity=row.entry.id
                assert row.entry.path is None and workflow.active_recording_path is None
                assert sum(getattr(item.entry,'id',None)==identity for item in view.model.items)==1
                assert view.tree.selection==[previous] and previous.displayed and not row.displayed
                connection.sendall(source[12:active_end])
                while workflow.live_recording.size_bytes!=active_end or not workflow._useful:
                    assert time.monotonic()<deadline
                    workflow.poll();await ui.next_frame()
                view.refresh();await frames()
                assert view.model.items[0] is row and row.entry.size_bytes==active_end
                assert view.tree.selection==[previous] and previous.displayed
                view.tree.selection=[row]
                assert not view.delete.enabled and not view.save.enabled
                view.save_selected();view.delete_selected()
                assert not saved and workflow.live_recording.id==identity
                view.activate();await frames()
                assert workflow.follow_live and row.displayed and view.tree.selection==[row]
                state=view.get_inspector_state();state_row=state['rows'][0]
                assert state['selected']==state_row['key']==str(identity)
                assert state_row['id']==identity
                assert state_row['path'] is None
                if keep:
                    assert workflow.select_session_recording(previous.entry.id)
                    view.refresh();await frames()
                    assert previous.displayed and not row.displayed
                    view.tree.selection=[row]
                connection.sendall(source[active_end:])
            while len(workflow.session_recordings)<count:
                assert time.monotonic()<deadline
                workflow.poll();await ui.next_frame()
            # Selection resolves the completed capture even before the UI poll.
            assert view.selected_entry().id==identity and view.selected_entry().status=='Complete'
            view.refresh();await frames()
            assert view.model.items[0] is row and view.tree.selection==[row]
            assert row.displayed==(not keep)
            if keep:assert previous.displayed
            assert view.save.enabled and view.delete.enabled and view.delete.text=='Remove'
            assert row.entry.size_bytes==len(source)
            assert (row.entry.path is not None)==keep
            if keep:assert row.entry.path.read_bytes()==source
            assert sum(getattr(item.entry,'id',None)==identity for item in view.model.items)==1
            if keep:assert sum(item.entry.path==row.entry.path for item in view.model.items)==1
            workflow.detach()
        view.refresh();await frames()
        first=workflow.session_recordings[-1]
        assert first.path is None
        row=next(item for item in view.model.items if getattr(item.entry,'id',None)==first.id)
        view.tree.selection=[row];view.activate();view.refresh();await frames()
        assert workflow.display_session is first.session and row.displayed
        assert view.delete.enabled and view.delete.text=='Remove'
        view.refresh(select_displayed=True)
        assert view.save.enabled
        view.save_selected();assert saved==[first.id]
        exported=root/'memory-export.ovd'
        assert workflow.save_session_recording(first.id,exported)
        assert exported.read_bytes()==source
        view.refresh();await frames()
        assert row in view.model.items and row.values[3].as_string==exported.name
        view.delete_selected();view.refresh();await frames()
        assert row not in view.model.items and exported.read_bytes()==source
        assert len(workflow.session_recordings)==1
        # Empty/rejected retained rows stay removable, but cannot replace the
        # useful displayed capture or reach the Save as callback.
        useful=workflow.session_recordings[0]
        assert workflow.select_session_recording(useful.id)
        workflow.set_keep_recordings(False);assert workflow.attach(port)
        saved.clear()
        for payload,status in ((source[:12],'Empty'),(b'\0','Rejected')):
            deadline=time.monotonic()+5
            while True:
                try:
                    connection=socket.create_connection(('127.0.0.1',port),timeout=.1)
                    break
                except (ConnectionRefusedError,TimeoutError):
                    assert time.monotonic()<deadline
                    workflow.poll();await frames()
            with connection:connection.sendall(struct.pack('<IHH',0x4F56444C,1,0)+payload)
            while len(workflow.session_recordings)<2:
                assert time.monotonic()<deadline
                workflow.poll();await ui.next_frame()
            view.refresh();await frames()
            row=view.model.items[0];view.tree.selection=[row]
            assert row.entry.status==status and not view.save.enabled and view.delete.enabled
            view.activate();view.save_selected();view.refresh();await frames()
            assert workflow.display_session is useful.session and not saved
            assert view.tree.selection==[row] and not row.displayed
            view.delete_selected();view.refresh();await frames()
            assert row not in view.model.items and len(workflow.session_recordings)==1
            assert useful.path.read_bytes()==source
        panel.destroy();window.destroy();workflow.shutdown()
        print('Native recordings table checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Recording table regression',width=1000,height=800,max_fps=60)
    hide_owned_window()
    style.startup('dark');ui.run(run())
    if errors:raise errors[0]
