# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The application crop range must reopen with every visible clip frame."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sys

import pytest
from _pvd_native import Session

from ovd_ui import application
from ovd_ui.playback import Playback
from ovd_ui.widgets.clips_view import ClipRow
from test_application_presentation import app, displayed, open_capture
from test_native_exports import compare_frame


class Picker:
    def __init__(self, submit, **kwargs):self.submit = submit;self.cancel = kwargs.get("on_cancel")
    def show(self):pass


@pytest.mark.parametrize("filename,scene_index,start,stop", [
    ("cloth_physx34.pxd2", 0, 1, 4),
    ("cloth_physx34.pxd2", 0, 5, 8),
    ("cloth_physx34.pxd2", 0, 5, 6),
    ("ue4_shooter.pxd2", 1, 30, 32),
])
def test_clip_picker_preserves_initial_snapshot_and_all_playable_frames(
        app, tmp_path, monkeypatch, filename, scene_index, start, stop):
    path, source = open_capture(tmp_path, filename)
    assert app._display(path, source)
    scene = source.scenes()[scene_index]
    assert app._refresh(scene_id=scene.id, frame=start)
    assert app.seek(start, snap=False)
    app._clip_action(app.playback.trim_start)
    assert app.seek(stop, snap=False)
    app._clip_action(app.playback.trim_end)
    assert (app.playback.clip.start, app.playback.clip.stop) == (start, stop)
    before = displayed(app)
    monkeypatch.setattr(application, "CaptureFileDialog", Picker)
    app.workflow.settings = {}
    app.pending_export = None
    app.dialogs = []
    app.export_window = SimpleNamespace(visible=True)
    destination = tmp_path / "application-clip.pxd2"
    with ThreadPoolExecutor(max_workers=1) as app.export_executor:
        app._choose_export("clip")
        app.dialogs[-1].submit(destination)
        result, size = app.pending_export.result(timeout=30)
    assert result == destination and size == destination.stat().st_size
    assert displayed(app) == before

    saved = Session(str(destination), str(tmp_path))
    target = next(value for value in saved.scenes() if value.id == scene.id)
    playback = Playback()
    playback.reset(target.frame_count, ovd=False)
    # Frame zero is the initial snapshot; normal playback begins at one.
    # A visible source clip [5,8) must reopen with frames 1,2,3 -> 5,6,7.
    assert playback.clip.stop - playback.clip.start == stop - start
    assert {value.id for value in saved.scenes()} == {value.id for value in source.scenes()}
    reference = Session(str(path), str(tmp_path))
    for local in range(target.frame_count):
        assert saved.seek(target.id, local)
        assert reference.seek(scene.id, start - 1 + local)
        compare_frame(reference, saved)


@pytest.mark.parametrize('replace_capture',[False,True])
def test_row_save_uses_noncurrent_clip_and_frozen_picker_source(app,tmp_path,monkeypatch,replace_capture):
    path,source=open_capture(tmp_path)
    assert app._display(path,source)
    assert app.seek(4,snap=False) and app._clip_action(app.playback.split)
    assert app._clip_action(app.playback.select,1)
    assert app.seek(8,snap=False) and app._clip_action(app.playback.trim_end)
    assert app._clip_action(app.playback.select,0)
    assert app.seek(2,snap=False)
    app._activity_source=lambda:(app.presentation.capture_key,app.presentation.segment_epoch,app.presentation.scene_id)
    clip=app.playback.clips[1];row=ClipRow(1,clip.name,clip.start,clip.stop)
    assert row.signature==('Capture_2',5,8)
    before=displayed(app)
    app.workflow.settings={};app.pending_export=None;app.dialogs=[]
    monkeypatch.setattr(application,'CaptureFileDialog',Picker)
    app._clip_row_action(app._activity_source(),'save',row)
    assert displayed(app)==before and app.playback.index==0
    if replace_capture:
        assert app._display(*open_capture(tmp_path,'multi_geom.ovd'))
        before=displayed(app)
    destination=tmp_path/'noncurrent-clip.pxd2'
    with ThreadPoolExecutor(max_workers=1) as app.export_executor:
        app.dialogs[-1].submit(destination)
        result,size=app.pending_export.result(timeout=30)
    assert result==destination and size==destination.stat().st_size
    assert displayed(app)==before
    saved=Session(str(destination),str(tmp_path));scene=saved.scenes()[0]
    assert scene.frame_count==4
    reference=Session(str(path),str(tmp_path))
    for local in range(4):
        assert saved.seek(scene.id,local) and reference.seek(scene.id,4+local)
        compare_frame(reference,saved)


def test_old_row_actions_reject_replaced_source_or_changed_range(app,tmp_path):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(4,snap=False) and app._clip_action(app.playback.split)
    context=[1,0,app.scene_id]
    app._activity_source=lambda:tuple(context)
    clip=app.playback.clips[1];row=ClipRow(1,clip.name,clip.start,clip.stop)
    source=app._activity_source();calls=[]
    app._choose_export=lambda *a,**kw:calls.append((a,kw))
    for field in range(3):
        context[field]+=1
        for action in ('save','delete','select'):app._clip_row_action(source,action,row)
        context[field]-=1
    assert not calls and app.playback.index==0 and len(app.playback.clips)==2
    app._clip_row_action(source,'save',row)
    assert calls==[(('clip',),{'clip_index':1})]
    assert app._clip_action(app.playback.select,1)
    assert app.seek(7,snap=False) and app._clip_action(app.playback.trim_start)
    for action in ('save','delete','select'):app._clip_row_action(source,action,row)
    assert len(calls)==1 and app.playback.clip.start==7 and len(app.playback.clips)==2


def test_row_delete_preserves_other_clip_position_and_protects_last_clip(app,tmp_path):
    assert app._display(*open_capture(tmp_path))
    assert app.seek(4,snap=False) and app._clip_action(app.playback.split)
    assert app._clip_action(app.playback.select,1)
    assert app.seek(8,snap=False)
    app._activity_source=lambda:(app.presentation.capture_key,app.presentation.segment_epoch,app.scene_id)
    clip=app.playback.clips[0];row=ClipRow(0,clip.name,clip.start,clip.stop)
    app._clip_row_action(app._activity_source(),'delete',row)
    assert len(app.playback.clips)==1 and app.playback.index==0 and app.frame==8
    assert app.playback.clip.name=='Capture_2'
    clip=app.playback.clip;last=ClipRow(0,clip.name,clip.start,clip.stop)
    app._clip_row_action(app._activity_source(),'delete',last)
    assert len(app.playback.clips)==1 and app.frame==8


@pytest.mark.parametrize("case", ["negative", "past_end", "missing", "no_session", "ovd", "live_ovd"])
def test_export_guards_do_not_open_picker(app, tmp_path, monkeypatch, case):
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd" if case in {"ovd", "live_ovd"} else "cloth_physx34.pxd2"))
    app.pending_export = None
    app.dialogs = []
    app.workflow.settings = {}
    monkeypatch.setattr(application, "CaptureFileDialog", Picker)
    if case == "missing": app.capture = tmp_path / "missing.pxd2"
    if case == "no_session": app.session = None
    if case == "live_ovd": app.workflow.live_session = app.session
    index = -1 if case == "negative" else len(app.playback.clips) if case == "past_end" else None
    app._choose_export("clip", clip_index=index)
    assert app.dialogs == [] and app.pending_export is None
    assert app.status_label.text


def test_default_range_export_reopens_final_actor_state(app, tmp_path, monkeypatch):
    path, source = open_capture(tmp_path, "apex_destruction.pxd2")
    assert app._display(path, source)
    scene = source.scenes()[0]
    before = displayed(app)
    app.workflow.settings = {};app.pending_export = None;app.dialogs = []
    monkeypatch.setattr(application, "CaptureFileDialog", Picker)
    destination = tmp_path / "full-range.pxd2"
    with ThreadPoolExecutor(max_workers=1) as app.export_executor:
        app._choose_export("clip")
        app.dialogs[-1].submit(destination)
        assert app.pending_export.result(timeout=30)[0] == destination
    assert displayed(app) == before
    saved = Session(str(destination), str(tmp_path))
    target = next(item for item in saved.scenes() if item.id == scene.id)
    assert target.frame_count == scene.frame_count
    reference = Session(str(path), str(tmp_path))
    assert saved.seek(scene.id, target.frame_count - 1)
    assert reference.seek(scene.id, scene.frame_count - 1)
    compare_frame(reference, saved)


def test_export_picker_cancel_releases_archive_lease(app, tmp_path, monkeypatch):
    import gc
    import zipfile
    from ovd_ui.ingestion import load_capture
    archive = tmp_path / "source.pvd"
    with zipfile.ZipFile(archive, "w") as package:
        package.write(open_capture(tmp_path)[0], "capture.pxd2")
    loaded = load_capture(archive)
    extracted = loaded.data_path
    assert app._display(archive, loaded.session)
    app.workflow.display_session = loaded.session
    app.workflow.display_data_path = extracted
    app.pending_export = None;app.dialogs = [];app.workflow.settings = {}
    monkeypatch.setattr(application, "CaptureFileDialog", Picker)
    app._choose_export("clip")
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd"))
    app.workflow.display_session = None
    app.workflow.display_data_path = None
    loaded.close();del loaded;gc.collect()
    assert extracted.exists()
    app.dialogs[-1].cancel();gc.collect()
    assert not extracted.exists() and app.pending_export is None


@pytest.mark.parametrize("failure", ["changed_parent", "private_root"] + (["sharing_denied"] if sys.platform == "win32" else []))
def test_export_filesystem_failure_preserves_source_and_directory(app, tmp_path, monkeypatch, failure):
    from ovd_ui import exports
    from ovd_ui.settings import Preferences
    path, source = open_capture(tmp_path)
    assert app._display(path, source)
    before, source_bytes = displayed(app), path.read_bytes()
    app.workflow.settings = Preferences(tmp_path / "preferences.json")
    app.workflow.settings.set("export_directory", str(tmp_path / "remembered"))
    app.pending_export = None;app.dialogs = []
    monkeypatch.setattr(application, "CaptureFileDialog", Picker)
    parent = tmp_path / "output";parent.mkdir()
    destination = parent / "sentinel.pxd2";destination.write_bytes(b"previous capture")
    sentinel = destination
    private = tmp_path / "private"
    if failure == "private_root": private.write_bytes(b"not a directory")
    else: private.mkdir()
    temporary_directory = exports.tempfile.TemporaryDirectory
    monkeypatch.setattr(exports.tempfile, "TemporaryDirectory", lambda **kwargs: temporary_directory(dir=private, **kwargs))
    handle = None
    try:
        with ThreadPoolExecutor(max_workers=1) as app.export_executor:
            app._choose_export("clip")
            if failure == "changed_parent":
                moved = tmp_path / "original-output";parent.rename(moved)
                sentinel = moved / destination.name;parent.write_bytes(b"changed parent")
            if failure == "sharing_denied":
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
                kernel.CreateFileW.restype = wintypes.HANDLE
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.CreateFileW(str(destination), 0x80000000, 1, None, 3, 0, None)
                assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
            app.dialogs[-1].submit(destination)
            with pytest.raises(ValueError if failure == "changed_parent" else OSError):
                app.pending_export.result(timeout=30)
            app._finish_export()
        assert app.pending_export is None and app.status_label.text.startswith("Export failed:")
        assert displayed(app) == before and path.read_bytes() == source_bytes
        assert sentinel.read_bytes() == b"previous capture"
        assert app.workflow.settings.get("export_directory") == str(tmp_path / "remembered")
        assert not list(tmp_path.rglob(".ovd-save-*"))
        if private.is_dir(): assert not list(private.iterdir())
        assert source.seek(source.scenes()[0].id, 1)
    finally:
        if handle not in (None, -1): kernel.CloseHandle(handle)
