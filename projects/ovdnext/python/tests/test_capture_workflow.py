# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Genuine SDK Applies are segments of one connection, never separate clips."""
import json
import subprocess

import pytest
from _pvd_native import Session

from test_sdk_tcp_selection import sdk_sender, SdkCapture
from test_workflow import FIXTURE, until, workflow


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("finish", ["eof", "save", "detach", "new"])
def test_two_real_applies_keep_one_connection_clip_and_all_segments(workflow, tmp_path, sdk_sender, keep, follow, finish):
    workflow.set_keep_recordings(keep)
    assert workflow.startup() and workflow.open_file(FIXTURE)
    old, session, port = workflow.display_session, workflow.live_session, workflow.port
    workflow.follow_live = follow
    process = subprocess.Popen([str(sdk_sender), "--host", "127.0.0.1", "--port", str(port),
        "--scenario", "schema-lifetimes", "--report", str(tmp_path / "sender.json")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    driver = SdkCapture(session, process, sdk_sender, tmp_path / "unused.ovd", tmp_path / "sender.json")
    try:
        until(workflow, lambda: workflow.capture_selection.info.get("catalog_ready"))
        model = workflow.capture_selection
        assert workflow.display_session is old and not workflow.session_recordings
        assert not session.refresh_live().ready and model.info["issued_high_water"] == 0
        box = driver.rows()["box"]["target_handle"]
        model.select(box, True)
        assert model.apply() == 1
        driver.settle()
        workflow.poll()
        assert workflow.display_session is old and not session.refresh_live().ready
        for serial in (2, 5):
            if serial == 5:
                model.all = model.secondary = True
                assert model.apply() == 2
                driver.settle()
            driver.send("step")
            # Ingestion can advance after poll's snapshot; wait for publication.
            until(workflow, lambda: workflow._live_update is not None and workflow._live_update.serial == serial)
            assert session.info().completed_serial == serial
            assert workflow.live_session is session and workflow.port == port
            assert workflow.display_session is (session if follow else old)
            assert not workflow.session_recordings and not workflow.recordings.entries
            assert not workflow.recordings.directory.exists()
        assert session.capture_control_info()["segment_transition_serial"] == 2
        assert len(session.scenes()) == 2
        if finish == "eof":
            output, error = process.communicate("quit\n", timeout=10)
            assert process.returncode == 0, output + error
            until(workflow, lambda: workflow.live_session is not session)
        elif finish == "new":
            workflow.new()
            assert workflow.display_session is None
        elif finish == "detach":
            workflow.detach()
        else:
            saved = workflow.save_capture(tmp_path / "manual.ovd")
            assert saved and saved.is_file()
            if not follow:
                assert saved.read_bytes() == FIXTURE.read_bytes()
                assert workflow.live_session is session  # Save the old displayed file only.
                workflow.detach()
        assert len(workflow.session_recordings) == 1
        retained = workflow.session_recordings[0]
        assert retained.session is session and retained.status == "Complete"
        assert workflow.capture_selection.session is not session
        assert len(workflow.recordings.entries) == int(keep)
        if finish != "new":
            assert workflow.display_session is (session if follow else old)
        exact = tmp_path / "all-segments.ovd"
        session.save(str(exact))
        if retained.path is not None:
            assert retained.path.read_bytes() == exact.read_bytes()
        assert all(entry.path.read_bytes() == exact.read_bytes() for entry in workflow.recordings.entries)
        decoded = json.loads(subprocess.check_output([str(sdk_sender), "--inspect", str(exact)], text=True))
        assert decoded["reader_eof"] and len(decoded["segments"]) == 3
        assert all(str(box) in segment for segment in decoded["segments"][1:])
        reopened = Session(str(exact), str(tmp_path / "reopened"))
        assert len(reopened.scenes()) == 2
        assert {s.recording_segment for s in reopened.scenes()} == {1, 2}
        copied = workflow.save_session_recording(retained.id, tmp_path / "history-save.ovd")
        assert copied.read_bytes() == exact.read_bytes()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
