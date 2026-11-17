# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening orchestration with real workers/native sessions and no GPU side effects."""
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import zipfile

import pytest

from ovd_ui.application import Application
from ovd_ui.ingestion import CaptureOpener
from ovd_ui.playback import Playback
from ovd_ui.recordings import RecordingLibrary
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow

ROOT=Path(__file__).resolve().parents[2]
OVD=ROOT/"test/data/test_scene.ovd"
LEGACY=ROOT/"test/data/cloth_physx34.pxd2"


@pytest.fixture
def app(tmp_path):
    value=Application.__new__(Application)
    value.workflow=CaptureWorkflow(Preferences(tmp_path/"settings.json"),RecordingLibrary(tmp_path/"library"))
    value.workflow.accept_connections=False
    assert value.workflow.open_file(OVD)
    value.workflow.poll()
    value.session=value.workflow.display_session
    value.capture=value.capture_data_path=OVD
    value.capture_opener=CaptureOpener(tmp_path)
    value.pending=value.load_dialog=None
    value.load_generation=value.pending_generation=0
    value.status_label=SimpleNamespace(text="")
    value.playback=Playback()
    value._show_load_progress=lambda source:None
    yield value
    value.cancel_load(announce=False)
    value.capture_opener.close()
    value.session=None
    value.workflow.shutdown()


def finish(app):
    app.pending.future.result(timeout=10)
    app._finish_load()


def test_archive_adoption_preserves_source_and_uses_native_payload(app,tmp_path):
    package=tmp_path/"cloth.pvd"
    with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.write(LEGACY,"cloth.pxd2")
    app.load(package)
    assert app.workflow.display_path==OVD
    finish(app)
    assert app.workflow.display_path==package
    assert app.workflow.display_data_path!=package
    assert app.workflow.recent_files[0]==str(package)
    # The useful old presentation persists until Application accepts the display event.
    assert app._capture_payload()==OVD
    app.capture=app.workflow.display_path
    app.session=app.workflow.display_session
    assert app._capture_payload()==app.workflow.display_data_path


def test_cancel_new_and_replacement_never_adopt_abandoned_capture(app,tmp_path):
    gate=Event()
    app.capture_opener.executor.submit(gate.wait)
    try:
        app.load(LEGACY)
        abandoned=app.pending
        app.load(OVD)
        assert abandoned.cancelled and abandoned.future.cancelled()
        latest=app.pending
        app.cancel_load()
        assert latest.cancelled and app.pending is None
        assert app.workflow.display_path==OVD and "cancelled" in app.status_label.text
        assert app.workflow.recent_files==(str(OVD),)
        app.load(LEGACY)
        pending=app.pending
        app.new_capture()
        assert pending.cancelled and app.workflow.display_session is None
        gate.set()
        with pytest.raises(CancelledError):pending.take_result()
        assert app.workflow.recent_files==(str(OVD),)
    finally:gate.set()


def test_failed_open_and_unsupported_drop_preserve_capture(app,tmp_path):
    app._drop_capture(SimpleNamespace(mime_data="https://host/capture.ovd\n/tmp/no.zip"))
    assert app.pending is None
    missing=tmp_path/"missing.ovd"
    app._drop_capture(SimpleNamespace(mime_data=missing.as_uri()+"\n"+OVD.as_uri()))
    job=app.pending
    with pytest.raises(FileNotFoundError):job.future.result(timeout=10)
    app._finish_load()
    assert "Unable to open" in app.status_label.text
    assert app.workflow.display_path==OVD and app.workflow.recent_files==(str(OVD),)


def test_superseded_completed_result_is_released_without_adoption(app):
    app.load(LEGACY)
    result=app.pending.future.result(timeout=10)
    directory=result._resources.path
    app.load_generation+=1
    app._finish_load()
    assert result.session is None and not directory.exists()
    assert app.workflow.display_path==OVD


def test_dialog_close_wins_when_native_worker_finishes_on_same_frame(app):
    app.load(LEGACY)
    loaded=app.pending.future.result(timeout=10)
    app.load_dialog=SimpleNamespace(visible=False,destroy=lambda:None)
    app._finish_load()
    assert loaded.session is None
    assert app.workflow.display_path==OVD and "cancelled" in app.status_label.text
