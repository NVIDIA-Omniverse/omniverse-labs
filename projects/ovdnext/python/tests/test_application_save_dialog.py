# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-capture save failures stay recoverable in the actual picker controller."""
from pathlib import Path
import asyncio
import os
from types import SimpleNamespace

import pytest

from ovd_ui.widgets.file_browser import CaptureFileDialog
from ovd_ui.settings import Preferences
from test_application_ingestion import app, OVD


@pytest.mark.parametrize("saved", [None, 0, 1, 2])
@pytest.mark.parametrize("escape", [False, True])
def test_unchanged_capture_sort_does_not_rewrite_preferences(app, tmp_path, saved, escape):
    app.capture = None
    app.dialogs = []
    settings = app.workflow.settings
    settings.set("open_directory", str(tmp_path))
    if saved is not None:
        settings.set("capture_sort", saved)
    before = settings.path.read_bytes()
    os.utime(settings.path, (1000000, 1000000))
    modified = settings.path.stat().st_mtime_ns

    async def verify():
        app.show_open()
        picker = app.dialogs[-1]
        try:
            await picker._task
            if escape:
                picker._key(256, 0, True)
            else:
                picker.cancel()
            assert not picker.visible
            assert settings.path.read_bytes() == before
            assert settings.path.stat().st_mtime_ns == modified
        finally:
            picker.destroy()
    asyncio.run(verify())


def test_capture_sort_survives_picker_cancel_and_preferences_reload(app, tmp_path):
    for name, modified in (("a-old.ovd", 1000000), ("z-new.ovd", 2000000)):
        path = tmp_path / name
        path.write_bytes(b"capture")
        os.utime(path, (modified, modified))
    app.capture = None
    app.dialogs = []
    app.workflow.settings.set("open_directory", str(tmp_path))

    async def verify():
        for order, names in ((1, ["a-old.ovd", "z-new.ovd"]),
                             (2, ["z-new.ovd", "a-old.ovd"]),
                             (0, ["a-old.ovd", "z-new.ovd"])):
            app.show_open()
            picker = app.dialogs[-1]
            try:
                await picker._task
                files = [item.entry.path.name for item in picker.model.items if not item.entry.directory]
                assert files == names
                picker.sort_combo.model.get_item_value_model().set_value(order)
                picker.cancel()
                app.workflow.settings = Preferences(app.workflow.settings.path)
                assert app.workflow.settings.get("capture_sort") == order
            finally:
                picker.destroy()
    asyncio.run(verify())


@pytest.mark.parametrize('failure',['same_source','write_error'])
def test_failed_whole_save_keeps_picker_and_allows_corrected_destination(app,tmp_path,monkeypatch,failure):
    def show(dialog):
        dialog._dismissed=False
        dialog.filename_field=SimpleNamespace(model=SimpleNamespace(as_string=''))
    monkeypatch.setattr(CaptureFileDialog,'show',show)
    prompts=[]
    monkeypatch.setattr('ovd_ui.widgets.file_browser.confirm_dialog',lambda *args,**kwargs:prompts.append(kwargs))
    app.dialogs=[]
    app.show_save()
    picker=app.dialogs[-1]
    source_bytes=OVD.read_bytes()
    source_session=app.workflow.display_session
    if failure=='same_source':
        picker.filename_field.model.as_string=str(OVD)
        picker.choose()
        assert len(prompts)==1 and not picker._dismissed
        prompts[0]['on_confirm']()
        expected='Choose a destination different from the recording source'
    else:
        import ovd_ui.workflow as workflow_module
        copy=workflow_module.copy_capture
        def failed_copy(*args,**kwargs):raise OSError('No space left on device')
        monkeypatch.setattr(workflow_module,'copy_capture',failed_copy)
        picker.filename_field.model.as_string=str(tmp_path/'failed.ovd')
        picker.choose()
        assert not (tmp_path/'failed.ovd').exists()
        monkeypatch.setattr(workflow_module,'copy_capture',copy)
        expected='Could not save capture: No space left on device'
    assert not picker._dismissed
    assert picker.error==expected
    assert app.workflow.display_session is source_session and OVD.read_bytes()==source_bytes
    destination=tmp_path/'corrected.ovd'
    picker.filename_field.model.as_string=str(destination)
    picker.choose()
    assert picker._dismissed and not picker.error
    assert destination.read_bytes()==source_bytes
    assert app.workflow.display_session is source_session and app.workflow.display_path==OVD
    assert Path(app.workflow.settings.get('save_directory'))==tmp_path


def test_failed_picker_creation_does_not_leave_live_polling_suspended(app,monkeypatch):
    from test_workflow import producer, until, free_port, HANDSHAKE, ACTIVE_PAYLOAD
    def failed_show(_dialog):raise RuntimeError('Window creation failed')
    monkeypatch.setattr(CaptureFileDialog,'show',failed_show)
    app.dialogs=[]
    app.workflow.attach(free_port())
    with pytest.raises(RuntimeError,match='Window creation failed'):
        app.show_save()
    source=app.workflow.live_session
    with producer(app.workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(app.workflow,lambda:app.workflow.display_session is source)
        assert app.workflow.follow_live and not source.info().ended


def test_explicit_capture_replacement_cannot_retarget_save_picker(app,tmp_path,monkeypatch):
    def show(dialog):
        dialog._dismissed=False
        dialog.filename_field=SimpleNamespace(model=SimpleNamespace(as_string=str(tmp_path/'saved.ovd')))
    monkeypatch.setattr(CaptureFileDialog,'show',show)
    app.dialogs=[];app.show_save();picker=app.dialogs[-1]
    replacement=tmp_path/'replacement.ovd';replacement.write_bytes(OVD.read_bytes())
    assert app.workflow.open_file(replacement)
    current=app.workflow.display_session
    picker.choose()
    assert 'displayed capture changed' in picker.error
    assert not picker._dismissed and not (tmp_path/'saved.ovd').exists()
    assert app.workflow.display_session is current and app.workflow.display_path==replacement
    picker.cancel()
    assert not app.workflow.follow_live
