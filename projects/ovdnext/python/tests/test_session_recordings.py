# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native live sessions remain usable without recording files."""
from pathlib import Path
import socket
import struct
import time

import pytest

from ovd_ui.recordings import RecordingLibrary, copy_capture
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow


ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = (ROOT / "test/data/test_scene.ovd").read_bytes()
OTHER_PAYLOAD = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
HANDSHAKE = struct.pack("<IHH", 0x4F56444C, 1, 0)
FRAME2_STOP = struct.pack("<BQQ", 12, 0x138AB174E00, 2)
FRAME2_END = PAYLOAD.index(FRAME2_STOP) + len(FRAME2_STOP)
ACTIVE_PAYLOAD = PAYLOAD[:FRAME2_END]


def free_port():
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return candidate.getsockname()[1]


def producer(workflow):
    deadline = time.monotonic() + 5
    while True:
        try:
            return socket.create_connection(("127.0.0.1", workflow.port), timeout=1)
        except (ConnectionRefusedError, TimeoutError):
            workflow.poll()
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def until(workflow, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        events.extend(workflow.poll())
        if predicate():
            return events
        time.sleep(.005)
    pytest.fail(f"Timed out: {workflow.status}; events={[(e.kind, e.message) for e in events]}")


def send_capture(workflow, payload):
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + payload)
    return until(workflow, lambda: workflow.live_session is not active)


@pytest.fixture
def workflow(tmp_path):
    settings = Preferences(tmp_path / "preferences.json")
    settings.set("live_port", free_port())
    settings.set("keep_recordings", False)
    result = CaptureWorkflow(settings, RecordingLibrary(tmp_path / "recordings"))
    assert result.attach(result.port)
    yield result
    result.shutdown()


def test_repeated_memory_sessions_keep_stable_ids_and_survive_new(workflow):
    first_live = workflow.live_session
    first_events = send_capture(workflow, PAYLOAD)
    first = workflow.session_recordings[0]
    assert first.id > 0 and first.session is first_live and first.path is None
    assert first.status == "Complete" and first.size_bytes == len(PAYLOAD)
    assert any(event.kind == "recordings" for event in first_events)

    second_live = workflow.live_session
    send_capture(workflow, OTHER_PAYLOAD)
    second, retained_first = workflow.session_recordings
    assert second.id > first.id and retained_first.id == first.id
    assert second.session is second_live and retained_first.session is first_live
    assert not workflow.recordings.directory.exists()

    assert workflow.select_session_recording(first.id)
    assert workflow.display_session is first_live
    assert workflow.displayed_session_recording_id == first.id
    workflow.new()
    assert workflow.display_session is None
    assert [item.id for item in workflow.session_recordings] == [second.id, first.id]
    assert workflow.select_session_recording(first.id)
    assert workflow.display_session is first_live


def test_removing_displayed_memory_session_leaves_other_capture_usable(workflow):
    send_capture(workflow, PAYLOAD)
    first = workflow.session_recordings[0]
    send_capture(workflow, OTHER_PAYLOAD)
    second = workflow.session_recordings[0]
    assert workflow.select_session_recording(first.id)

    assert workflow.remove_session_recording(first.id)
    assert workflow.display_session is None
    assert [item.id for item in workflow.session_recordings] == [second.id]
    assert workflow.select_session_recording(second.id)
    assert workflow.display_session is second.session
    assert not workflow.remove_session_recording(first.id)


def test_empty_and_rejected_rows_preserve_useful_display(workflow, tmp_path):
    send_capture(workflow, OTHER_PAYLOAD)
    useful = workflow.session_recordings[0]
    displayed = workflow.display_session

    send_capture(workflow, PAYLOAD[:12])
    assert workflow.session_recordings[0].status == "Empty"
    assert workflow.display_session is displayed
    assert workflow.displayed_session_recording_id == useful.id
    empty = workflow.session_recordings[0]
    assert not workflow.select_session_recording(empty.id)
    assert not workflow.save_session_recording(empty.id, tmp_path / "empty.ovd")
    assert workflow.display_session is displayed and not (tmp_path / "empty.ovd").exists()

    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + b"\0")
    until(workflow, lambda: len(workflow.session_recordings) == 3)
    assert workflow.session_recordings[0].status == "Rejected"
    assert workflow.session_recordings[0].size_bytes == 1
    assert workflow.display_session is displayed
    rejected = workflow.session_recordings[0]
    assert not workflow.select_session_recording(rejected.id)
    assert not workflow.save_session_recording(rejected.id, tmp_path / "rejected.ovd")
    assert workflow.display_session is displayed and not (tmp_path / "rejected.ovd").exists()

    until(workflow, lambda: workflow.live_session is not None)
    rejected_live = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is rejected_live)
        connection.sendall(b"\xff")
    events = until(workflow, lambda: workflow.live_session is not rejected_live)
    assert workflow.session_recordings[0].status == "Rejected"
    assert workflow.display_session is displayed
    assert workflow.displayed_session_recording_id == useful.id
    assert any(event.kind == "display" and event.session is displayed for event in events)


def test_removing_rejected_preview_fallback_clears_display(workflow):
    send_capture(workflow, OTHER_PAYLOAD)
    useful = workflow.session_recordings[0]
    rejected_live = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is rejected_live)
        assert workflow.remove_session_recording(useful.id)
        assert workflow.display_session is rejected_live
        connection.sendall(b"\xff")
    until(workflow, lambda: workflow.live_session is not rejected_live)
    assert workflow.session_recordings[0].status == "Rejected"
    assert workflow.display_session is None


def test_rejected_preview_restores_file_backed_capture(workflow):
    path = ROOT / "test/data/resample_two_segment.ovd"
    assert workflow.open_file(path)
    previous = workflow.display_session
    assert workflow.follow_latest()
    rejected_live = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is rejected_live)
        connection.sendall(b"\xff")
    until(workflow, lambda: workflow.live_session is not rejected_live)
    assert workflow.session_recordings[0].status == "Rejected"
    assert workflow.display_session is previous
    assert workflow.display_path == path
    assert previous.scenes()


def test_memory_session_save_is_byte_exact_and_updates_optional_path(workflow, tmp_path):
    send_capture(workflow, PAYLOAD)
    recording = workflow.session_recordings[0]
    destination = tmp_path / "saved-memory"
    saved = workflow.save_session_recording(recording.id, destination)
    assert saved == destination.with_suffix(".ovd")
    assert saved.read_bytes() == PAYLOAD
    updated = workflow.session_recordings[0]
    assert updated.id == recording.id and updated.session is recording.session
    assert updated.path == saved and updated.size_bytes == len(PAYLOAD)

    saved.write_bytes(b"precious")
    assert not workflow.save_session_recording(recording.id, saved)
    assert saved.read_bytes() == b"precious"
    assert workflow.save_session_recording(recording.id, saved, overwrite=True) == saved
    assert saved.read_bytes() == PAYLOAD


def test_global_save_retries_displayed_memory_session_without_losing_it(workflow, tmp_path):
    send_capture(workflow, PAYLOAD)
    displayed = workflow.display_session
    destination = tmp_path / "existing.ovd"
    destination.write_bytes(b"precious")

    assert workflow.can_save
    assert not workflow.save_capture(destination)
    assert workflow.display_session is displayed
    assert destination.read_bytes() == b"precious"
    assert workflow.save_capture(destination, overwrite=True) == destination
    assert workflow.display_session is displayed
    assert destination.read_bytes() == PAYLOAD


def test_save_active_memory_capture_freezes_and_exports_exact_bytes(workflow, tmp_path):
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        assert workflow.can_save
        destination = tmp_path / "active"
        saved = workflow.save_capture(destination)
        assert saved == destination.with_suffix(".ovd")
        assert saved.read_bytes() == ACTIVE_PAYLOAD
        assert workflow.live_session is None and active.info().ended
        assert workflow.display_session is active
        assert workflow.session_recordings[0].session is active
        assert not workflow.recordings.directory.exists()


@pytest.mark.parametrize("kind", ["missing-directory", "existing", "directory", "symlink", "extension"])
def test_invalid_memory_save_destination_preserves_active_producer(workflow, tmp_path, kind):
    destination = tmp_path / "capture.ovd"
    if kind == "missing-directory":
        destination = tmp_path / "missing" / "capture.ovd"
    elif kind == "existing":
        destination.write_bytes(b"precious")
    elif kind == "directory":
        destination.mkdir()
    elif kind == "symlink":
        from filesystem_capabilities import symlink
        symlink(destination, tmp_path / "missing-target.ovd")
    else:
        destination = tmp_path / "capture.pxd2"
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        assert workflow.can_save
        assert not workflow.save_capture(destination, overwrite=kind != "existing")
        assert workflow.live_session is active and not active.info().ended
        assert workflow.display_session is active and not workflow.session_recordings
        if kind == "existing":
            assert destination.read_bytes() == b"precious"


@pytest.mark.parametrize("keep", [False, True])
def test_save_rejected_active_capture_does_not_export_restored_display(workflow, tmp_path, keep):
    assert workflow.set_keep_recordings(keep)
    send_capture(workflow, OTHER_PAYLOAD)
    previous = workflow.display_session
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        assert workflow.can_save
        connection.sendall(b"\xff")
        # Let native parsing reject the stream without polling away the preview.
        deadline = time.monotonic() + 5
        while not active.info().ended:
            assert time.monotonic() < deadline
            time.sleep(.005)
        destination = tmp_path / "rejected-preview.ovd"
        assert not workflow.save_capture(destination)
        assert not destination.exists()
        assert workflow.display_session is previous and workflow.live_session is None
        assert workflow.session_recordings[0].status == "Rejected"
        assert "rejected" in workflow.error.lower()


@pytest.mark.parametrize("keep", [False, True])
def test_live_recording_size_and_identity_survive_completion(workflow, keep):
    assert workflow.set_keep_recordings(keep)
    assert workflow.live_recording is None
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        # A useful frame can be displayed before the worker consumes all bytes
        # sent after it. Take the live snapshot only after that payload is read.
        until(workflow, lambda: workflow.display_session is active
              and workflow.live_recording is not None
              and workflow.live_recording.size_bytes == len(ACTIVE_PAYLOAD))
        live = workflow.live_recording
        assert live.session is active and live.id > 0 and live.status == "Live"
        assert live.size_bytes == len(ACTIVE_PAYLOAD)
        assert live.path is None
        assert workflow.displayed_session_recording_id is None
        connection.sendall(PAYLOAD[FRAME2_END:])
        until(workflow, lambda: workflow.live_recording.size_bytes == len(PAYLOAD))
        assert workflow.live_recording.id == live.id
    until(workflow, lambda: workflow.live_session is not active)
    assert workflow.live_recording is None
    retained = workflow.session_recordings[0]
    assert retained.id == live.id and retained.session is active and retained.status == "Complete"
    assert retained.size_bytes == len(PAYLOAD)
    assert (retained.path is not None) == keep
    if keep:
        assert retained.path.read_bytes() == PAYLOAD
    assert workflow.displayed_session_recording_id == live.id


def test_copy_capture_requires_a_file_source_without_creating_output(tmp_path):
    with pytest.raises(ValueError, match="source"):
        copy_capture(None, tmp_path / "copy.ovd")
    assert not list(tmp_path.iterdir())


def test_rejected_capture_cannot_replace_existing_destination(workflow, tmp_path):
    send_capture(workflow, OTHER_PAYLOAD)
    previous = workflow.display_session
    active = workflow.live_session
    destination = tmp_path / "precious.ovd"
    destination.write_bytes(b"previous recording")
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        connection.sendall(b"\xff")
        deadline = time.monotonic() + 5
        while not active.info().ended:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert not workflow.save_capture(destination, overwrite=True)
    assert destination.read_bytes() == b"previous recording"
    assert workflow.display_session is previous
    assert workflow.session_recordings[0].status == "Rejected"
    assert "rejected" in workflow.error.lower()
    assert not list(tmp_path.glob(".ovd-save-*"))


def test_never_connected_listener_stop_and_new_create_no_rows(workflow):
    assert workflow.live_session is not None
    assert not workflow.session_recordings
    workflow.detach()
    assert not workflow.session_recordings and workflow.display_session is None
    assert workflow.attach(workflow.port)
    workflow.new()
    assert not workflow.session_recordings and workflow.display_session is None
