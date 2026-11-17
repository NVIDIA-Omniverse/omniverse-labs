# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original recording metadata formatting and native widget ownership."""
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ovd_ui.widgets.recordings_view import recorded_size,recorded_time,RecordingModel,RecordingsView
from ovd_ui.application import Application
from ovd_ui.recordings import RecordingEntry
from ovd_ui.workflow import SessionRecording
from gui_runtime import GUI_AVAILABLE
from test_workflow import workflow


@pytest.mark.parametrize('size,text',[(0,'0.00 bytes'),(1023,'1023.00 bytes'),(1024,'1.00 kb'),
    (1048575,'1024.00 kb'),(1048576,'1.00 Mb'),(1073741824,'1.00 Gb'),(1099511627776,'1.00 Tb')])
def test_original_binary_size_units_and_precision(size,text):
    assert recorded_size(size)==text


@pytest.mark.skipif(not hasattr(time,'tzset'),reason='POSIX timezone fixture')
def test_recorded_timestamp_is_local_with_milliseconds_and_negative_epoch(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv('TZ','UTC0');time.tzset()
        assert recorded_time(1788671112.1239)=='2026-09-06 05:05:12.123'
        assert recorded_time(-.001)=='1969-12-31 23:59:59.999'
        patch.setenv('TZ','EST5EDT,M3.2.0,M11.1.0');time.tzset()
        assert recorded_time(1788671112.1239)=='2026-09-06 01:05:12.123'
    time.tzset()


@pytest.mark.skipif(not GUI_AVAILABLE,reason='Requires an OVUI graphics display')
def test_native_recordings_rows_follow_current_library_and_protection(tmp_path):
    root=Path(__file__).resolve().parents[2]
    caller_file=tmp_path/'caller-owned.txt';caller_file.write_text('unchanged')
    assert not (root/'imgui.ini').exists()
    result=subprocess.run([sys.executable,str(root/'python/tests/recordings_table_gui.py')],cwd=tmp_path,
        env=dict(os.environ,OVUIINSPECT_ENABLED='0'),capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'Native recordings table checks passed' in result.stdout
    assert caller_file.read_text()=='unchanged'
    assert not (tmp_path/'imgui.ini').exists() and not (root/'imgui.ini').exists()


@pytest.mark.parametrize('key,normalized',[(525,257),(257,257),(627,335)])
def test_application_routes_recordings_keys_through_the_focused_bottom_page(key,normalized):
    app=Application.__new__(Application);calls=[]
    app.window=SimpleNamespace(focused=True)
    app.object_browser=SimpleNamespace(search_key=lambda *_:False)
    app.activity=SimpleNamespace(key=lambda *args:(calls.append(args),True)[1])
    app._key_pressed(key,0,True)
    assert calls==[(normalized,0,True)]


def memory_entry(identity, *, path=None, status='Complete'):
    return SessionRecording(id=identity,session=object(),recorded_at=100+identity,
                            status=status,size_bytes=2048,path=path)


def test_refresh_broken_library_still_refreshes_rows_and_recents(workflow):
    workflow.recordings.directory.write_bytes(b"not a directory")
    refreshed = []
    view = RecordingsView.__new__(RecordingsView)
    view.workflow = workflow
    view.refresh = lambda: refreshed.append("rows")
    view.recents = SimpleNamespace(rebuild=lambda: refreshed.append("recents"))
    view.reload()
    assert refreshed == ["rows", "recents"]
    assert any(event.kind == "error" and "recordings" in event.message.lower()
               for event in workflow.poll())


def test_memory_recordings_have_distinct_stable_rows_without_files():
    first,second=memory_entry(1),memory_entry(2)
    model=RecordingModel()
    model.update((second,first),1)
    older=model.items[1]
    assert [item.displayed for item in model.items]==[False,True]
    assert len(model.cache)==2
    assert older.values[3].as_string=='Memory capture 1'
    saved=memory_entry(1,path=Path('saved.ovd'))
    model.update((second,saved),1)
    assert model.items[1] is older
    assert older.values[3].as_string=='saved.ovd'


def test_memory_row_actions_use_identity_not_a_missing_path():
    entry=memory_entry(7)
    selected=[];removed=[];saved=[];opened=[];deleted=[]
    workflow=SimpleNamespace(session_recordings=(entry,),live_recording=None,displayed_session_recording_id=7,
        display_path=None,active_recording_path=None,
        recordings=SimpleNamespace(entries=()),select_session_recording=selected.append,
        remove_session_recording=removed.append)
    view=RecordingsView.__new__(RecordingsView)
    view.workflow=workflow
    view.tree=SimpleNamespace(selection=[SimpleNamespace(entry=entry)])
    view.on_open,view.on_delete,view.on_save=opened.append,deleted.append,saved.append
    assert view.can_delete()  # Removing a RAM row never deletes a saved file.
    view.activate();view.save_selected();view.delete_selected()
    assert selected==[7] and saved==[7] and removed==[7]
    assert opened==[] and deleted==[]
    workflow.session_recordings=()
    assert view.selected_entry() is None
    view.activate();view.save_selected();view.delete_selected()
    assert selected==[7] and saved==[7] and removed==[7]


@pytest.mark.parametrize('path',[None,Path('incoming.ovd')])
def test_live_row_keeps_identity_and_selection_when_retained_before_refresh(path):
    live=memory_entry(8,path=path,status='Live')
    older=memory_entry(7)
    disk=() if path is None else (RecordingEntry(path,108,2048,'Live'),)
    workflow=SimpleNamespace(session_recordings=(older,),live_recording=live,
        recordings=SimpleNamespace(entries=disk),display_path=older.path,active_recording_path=path)
    view=RecordingsView.__new__(RecordingsView);view.workflow=workflow
    entries=view._entries()
    assert entries==(live,older)
    view.model=RecordingModel();view.model.update(entries,older.id)
    row=view.model.items[0];previous=view.model.items[1]
    view.tree=SimpleNamespace(selection=[row])
    assert view.selected_entry() is live and not view.can_delete()
    grown=replace(live,size_bytes=4096)
    workflow.live_recording=grown
    view.model.update(view._entries(),older.id)
    assert view.model.items==[row,previous]
    assert row.values[2].as_string=='4.00 kb'
    assert previous.displayed
    finished=replace(grown,status='Complete')
    workflow.live_recording=None;workflow.active_recording_path=None
    workflow.session_recordings=(finished,older)
    assert view.selected_entry() is finished and view.can_delete()
    view.model.update(view._entries(),older.id)
    assert view.tree.selection==[row] and view.model.items==[row,previous]
    assert row.values[1].as_string=='Complete'


def test_live_activation_follows_latest_without_saving_or_removing():
    live=memory_entry(8,status='Live');calls=[]
    workflow=SimpleNamespace(session_recordings=(),live_recording=live,
        recordings=SimpleNamespace(entries=()),display_path=None,active_recording_path=None,
        select_session_recording=lambda identity:calls.append(('select',identity)),
        remove_session_recording=lambda identity:calls.append(('remove',identity)))
    view=RecordingsView.__new__(RecordingsView);view.workflow=workflow
    view.tree=SimpleNamespace(selection=[SimpleNamespace(entry=live)])
    view.follow_latest=lambda:calls.append(('follow',))
    view.on_save=lambda identity:calls.append(('save',identity))
    view.activate();view.save_selected();view.delete_selected()
    assert calls==[('follow',)]


@pytest.mark.parametrize('status',['Empty','Rejected'])
def test_incomplete_memory_rows_cannot_be_saved_but_can_be_removed(status):
    entry=memory_entry(7,status=status);saved=[];removed=[]
    workflow=SimpleNamespace(session_recordings=(entry,),live_recording=None,
        recordings=SimpleNamespace(entries=()),display_path=None,active_recording_path=None,
        remove_session_recording=removed.append)
    view=RecordingsView.__new__(RecordingsView);view.workflow=workflow
    view.tree=SimpleNamespace(selection=[SimpleNamespace(entry=entry)])
    view.on_save=saved.append;view.save=SimpleNamespace(enabled=True);view.delete=SimpleNamespace(enabled=False)
    view._update_actions();view.save_selected();view.delete_selected()
    assert not view.save.enabled and not saved
    assert view.delete.enabled and view.delete.text=='Remove' and removed==[7]


def test_inspector_rows_expose_nullable_paths_and_keys_matching_selection():
    memory=memory_entry(7);saved=memory_entry(8,path=Path('saved.ovd'))
    library=RecordingEntry(Path('library.ovd'),109,1024)
    view=RecordingsView.__new__(RecordingsView)
    view.workflow=SimpleNamespace(session_recordings=(saved,memory),live_recording=None,
        recordings=SimpleNamespace(entries=(library,)))
    view.model=RecordingModel();view.model.update((library,saved,memory),7)
    view.tree=SimpleNamespace(selection=[view.model.items[2]])
    view.window=SimpleNamespace(visible=True);view.table=SimpleNamespace(displayed=(190,88,84,338))
    view.delete=SimpleNamespace(enabled=True);view.follow=SimpleNamespace(enabled=True)
    state=view.get_inspector_state();disk_row,saved_row,memory_row=state['rows']
    assert memory_row['path'] is None
    assert memory_row['id']==7 and memory_row['key']==state['selected']=='7'
    assert saved_row['path']=='saved.ovd' and saved_row['id']==8 and saved_row['key']=='8'
    assert disk_row['path']=='library.ovd' and disk_row.get('id') is None and disk_row['key']=='library.ovd'


def test_selected_memory_capture_save_dialog_keeps_identity_when_display_changes(monkeypatch,tmp_path):
    import ovd_ui.application as module
    requests=[];dialogs=[]
    class Picker:
        def __init__(self,callback,**kwargs):
            self.callback,self.options=callback,kwargs;dialogs.append(self)
        def show(self):pass
    monkeypatch.setattr(module,'CaptureFileDialog',Picker)
    app=Application.__new__(Application);app.dialogs=[]
    app.workflow=SimpleNamespace(settings={'save_directory':str(tmp_path)},error='',
        save_session_recording=lambda *args,**kwargs:requests.append((args,kwargs)))
    app._save_session_recording(7)
    app.workflow.display_session=object()
    destination=tmp_path/'chosen.ovd'
    dialogs[0].callback(destination)
    assert requests==[((7,destination),{'overwrite':False})]
    assert app.dialogs==dialogs
    app.workflow.error='The capture was removed'
    app.workflow.save_session_recording=lambda *args,**kwargs:False
    with pytest.raises(RuntimeError,match='capture was removed'):
        dialogs[0].callback(destination)
