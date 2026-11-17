# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real desktop argv/lifetime and native capture preservation for Open folder."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from ovd_ui import desktop
from ovd_ui.recordings import RecordingLibrary
from test_application_presentation import app, displayed, open_capture
from desktop_test_handler import install_handler, missing_handler


@pytest.fixture(autouse=True)
def prevent_real_desktop_handler(monkeypatch):
    monkeypatch.setattr(desktop, '_launch', missing_handler)


def handler(tmp_path, monkeypatch, code="pass"):
    """Disposable executable records argv; it never invokes a file manager."""
    return install_handler(tmp_path, monkeypatch, code)


def until(predicate):
    deadline=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<deadline
        time.sleep(.005)


def test_exact_resolved_absolute_path_is_one_argument_including_spaces_and_shell_text(tmp_path,monkeypatch):
    real=tmp_path/'Real folder ; $(touch SHOULD_NOT_EXIST) {code}'
    real.mkdir()
    log=handler(tmp_path,monkeypatch)
    monkeypatch.chdir(tmp_path)
    request=desktop.OpenFolderRequest(Path(real.name))
    assert request.path==real.resolve()
    until(request.poll)
    request.close()
    assert json.loads(log.read_text())==[str(real)]
    assert not (tmp_path/'SHOULD_NOT_EXIST').exists()


def test_folder_alias_is_resolved_before_launch(tmp_path, monkeypatch):
    from filesystem_capabilities import symlink
    real = tmp_path / 'Real folder'; real.mkdir()
    alias = tmp_path / 'parent alias'
    symlink(alias, real, target_is_directory=True)
    log = handler(tmp_path, monkeypatch)
    request = desktop.OpenFolderRequest(alias)
    assert request.path == real.resolve()
    until(request.poll)
    request.close()
    assert json.loads(log.read_text()) == [str(real)]


def test_invalid_path_and_missing_handler_report_exact_folder_without_creation(tmp_path,monkeypatch):
    missing=tmp_path/'Not created'
    with pytest.raises(desktop.DesktopError,match='Not created'):
        desktop.OpenFolderRequest(missing)
    assert not missing.exists()
    existing=tmp_path/'Existing folder';existing.mkdir()
    monkeypatch.setenv('PATH',str(tmp_path))
    with pytest.raises(desktop.DesktopError) as caught:
        desktop.OpenFolderRequest(existing)
    assert 'No desktop file manager handler' in str(caught.value) and str(existing) in str(caught.value)
    file=tmp_path/'not-directory';file.write_bytes(b'capture bytes')
    with pytest.raises(desktop.DesktopError,match='Not a directory'):
        desktop.OpenFolderRequest(file)
    assert file.read_bytes()==b'capture bytes'


def test_rejected_and_unsupported_handlers_keep_literal_path_in_message(tmp_path,monkeypatch):
    folder=tmp_path/'literal {code} folder';folder.mkdir()
    handler(tmp_path,monkeypatch,'sys.exit(3)')
    request=desktop.OpenFolderRequest(folder)
    with pytest.raises(desktop.DesktopError) as caught:
        until(request.poll)
    request.close()
    assert 'exit 3' in str(caught.value) and str(folder) in str(caught.value)
    monkeypatch.setattr(desktop,'sys',SimpleNamespace(platform='unsupported'))
    with pytest.raises(desktop.DesktopError,match='unavailable on this platform'):
        desktop.OpenFolderRequest(folder)


def test_close_reaps_but_does_not_terminate_live_desktop_process(tmp_path,monkeypatch):
    release=tmp_path/'release';done=tmp_path/'done'
    code=f"while not Path({str(release)!r}).exists(): time.sleep(.005)\nPath({str(done)!r}).write_text('finished')"
    log=handler(tmp_path,monkeypatch,code)
    folder=tmp_path/'captures';folder.mkdir()
    request=desktop.OpenFolderRequest(folder)
    process=request._process
    try:
        until(log.exists)
        assert not request.poll()
        request.close()
        assert request._process is None and process.poll() is None
    finally:
        release.touch()
        until(done.exists)
        until(lambda:process.poll() is not None)
    assert process.returncode==0


def configure(app,directory):
    app.workflow.recordings=RecordingLibrary(directory)
    app.recordings_folder_button=SimpleNamespace(enabled=True)
    app.recordings_folder_label=SimpleNamespace(text='',tooltip='')
    app.recordings_folder_request=None
    errors=[];app._report_error=errors.append
    return errors


def test_application_creates_folder_and_preserves_native_display_selection_and_recordings(app,tmp_path,monkeypatch):
    path,session=open_capture(tmp_path,'test_scene.ovd')
    assert app._display(path,session)
    app.selection.publish(["/objects/1"],source="test")
    assert app._refresh()
    before=displayed(app)
    directory=tmp_path/'Saved recordings with spaces'
    errors=configure(app,directory)
    assert not directory.exists()
    log=handler(tmp_path,monkeypatch)
    assert app.open_recordings_folder() and directory.is_dir()
    assert not app.recordings_folder_button.enabled and not app.open_recordings_folder()
    until(lambda:(app._poll_recordings_folder(),app.recordings_folder_request is None)[1])
    assert app.recordings_folder_button.enabled and not errors
    assert json.loads(log.read_text())==[str(directory.resolve())]
    recording=directory/'unchanged.ovd';recording.write_bytes(path.read_bytes())
    app.workflow.recordings.reload();entries=app.workflow.recordings.entries
    assert app.open_recordings_folder()
    until(lambda:(app._poll_recordings_folder(),app.recordings_folder_request is None)[1])
    assert recording.read_bytes()==path.read_bytes() and app.workflow.recordings.entries==entries
    assert displayed(app)==before
    assert str(directory) in app.recordings_folder_label.text==app.recordings_folder_label.tooltip


def test_application_directory_and_launch_failures_leave_recordings_and_capture_untouched(app,tmp_path,monkeypatch):
    path,session=open_capture(tmp_path,'test_scene.ovd');assert app._display(path,session)
    before=displayed(app)
    directory=tmp_path/'Recordings';errors=configure(app,directory)
    directory.mkdir();recording=directory/'saved.ovd';recording.write_bytes(path.read_bytes())
    monkeypatch.setenv('PATH',str(tmp_path))
    assert not app.open_recordings_folder() and app.recordings_folder_request is None
    assert str(directory) in errors[-1] and 'No desktop file manager handler' in errors[-1]
    assert app.recordings_folder_button.enabled
    def fail():raise PermissionError('read-only library')
    monkeypatch.setattr(app.workflow.recordings,'ensure_directory',fail)
    assert not app.open_recordings_folder() and 'read-only library' in errors[-1]
    assert recording.read_bytes()==path.read_bytes() and displayed(app)==before


def test_application_poll_failure_allows_retry_and_cleanup_detaches_request(app,tmp_path,monkeypatch):
    configure(app,tmp_path/'Recordings')
    handler(tmp_path,monkeypatch,'sys.exit(4)')
    assert app.open_recordings_folder()
    until(lambda:(app._poll_recordings_folder(),app.recordings_folder_request is None)[1])
    assert 'exit 4' in app.recordings_folder_status and app.recordings_folder_button.enabled
    closed=[]
    app.recordings_folder_request=SimpleNamespace(close=lambda:closed.append(True))
    app._close_recordings_folder();app._close_recordings_folder()
    assert closed==[True] and app.recordings_folder_request is None
