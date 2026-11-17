# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native TCP producers exercise the frontend's independent lifecycle."""
from pathlib import Path
import socket
import struct
import time

import pytest

from ovd_ui.recordings import RecordingLibrary, copy_capture
from ovd_ui.settings import Preferences, parse_port
from ovd_ui.workflow import CaptureWorkflow


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test/data/test_scene.ovd"
PAYLOAD = FIXTURE.read_bytes()
HANDSHAKE = struct.pack("<IHH", 0x4F56444C, 1, 0)
# Fixture's second STOP_FRAME, before frame 3 and unpublished teardown.
# This is a known byte boundary of this checked-in fixture, not a second parser.
FRAME2_STOP = struct.pack("<BQQ", 12, 0x138AB174E00, 2)
assert PAYLOAD.count(FRAME2_STOP) == 1
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


def send_capture(workflow, payload=PAYLOAD):
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + payload)


@pytest.fixture
def workflow(tmp_path):
    settings = Preferences(tmp_path / "preferences.json")
    settings.set("live_port", free_port())
    settings.set("keep_recordings", True)
    workflow = CaptureWorkflow(settings, RecordingLibrary(tmp_path / "recordings"))
    yield workflow
    workflow.shutdown()


def test_verified_payload_follows_displayed_session_and_saves_displayed_bytes(workflow, tmp_path):
    source = tmp_path / "source.ovd";source.write_bytes(PAYLOAD)
    assert workflow.open_file(source)
    original_session = workflow.display_session
    proof = workflow.verified_payload
    assert proof and workflow.display_data_path == proof.path and workflow.display_path == source
    source.write_bytes(b"Different bytes after loading")
    destination = tmp_path / "saved.ovd"
    assert workflow.save_capture(destination) == destination
    assert destination.read_bytes() == PAYLOAD
    workflow.display_session = None
    assert workflow.verified_payload is None
    workflow.display_session = original_session
    assert workflow.verified_payload is proof
    workflow.new()
    assert workflow.verified_payload is None


def test_verified_payload_does_not_follow_a_live_preview(workflow):
    assert workflow.attach(workflow.port)
    active = workflow.live_session
    assert workflow.open_file(FIXTURE)
    proof = workflow.verified_payload
    assert proof is not None
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: active.info().completed_serial >= 2)
        assert workflow.verified_payload is proof
        assert workflow.follow_latest()
        until(workflow, lambda: workflow.display_session is active)
        assert workflow.verified_payload is None and workflow.display_data_path is None
        workflow.detach()
        assert workflow.verified_payload is None


def test_settings_defaults_validation_recents_and_persistence(tmp_path):
    path = tmp_path / "preferences.json"
    preferences = Preferences(path)
    assert preferences.get("live_port") == 5425
    assert preferences.get("keep_recordings") is False
    preferences.set("keep_recordings", True)
    preferences.set("live_port", "localhost:15425")
    for index in range(12):
        preferences.add_recent(tmp_path / f"capture-{index}.ovd")
    preferences.add_recent(tmp_path / "capture-7.ovd")
    reopened = Preferences(path)
    assert reopened.get("keep_recordings") is True
    assert reopened.get("live_port") == 15425
    assert len(reopened.recent_files) == 8
    assert Path(reopened.recent_files[0]).name == "capture-7.ovd"
    reopened.clear_recent()
    assert Preferences(path).recent_files == ()
    for invalid in (0, -1, 65536, "5425oops", "host:", "１２３", "1.0"):
        with pytest.raises(ValueError):
            parse_port(invalid)
    assert parse_port(" [::1]:5425 ") == 5425


def test_bad_preferences_recover_without_touching_legacy(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text("corrupt")
    preferences = Preferences(path)
    assert preferences.get("keep_recordings") is False
    assert preferences.load_error
    assert path.read_text() == "corrupt"


def test_default_startup_and_explicit_live_precedence(workflow, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "attach", lambda port: calls.append(port))
    workflow.port = 5425
    workflow.startup()
    workflow.startup(FIXTURE, "localhost:20000", explicit_live=True)
    assert calls == [5425, "localhost:20000"]


def test_library_reload_and_delete_safety(tmp_path):
    library = RecordingLibrary(tmp_path / "recordings")
    library.ensure_directory()
    final = library.candidate_path()
    final.write_bytes(PAYLOAD)
    entries = RecordingLibrary(library.directory).entries
    assert [entry.path for entry in entries] == [final]
    with pytest.raises(ValueError):
        library.delete(final, protected=[final])
    outside = tmp_path / "outside.ovd"
    outside.write_bytes(PAYLOAD)
    with pytest.raises(ValueError):
        library.delete(outside)
    assert len(library.reload()) == 1
    library.delete(final)
    assert outside.read_bytes() == PAYLOAD


def test_library_rejects_protected_and_outside_symbolic_links(tmp_path):
    from filesystem_capabilities import symlink
    library = RecordingLibrary(tmp_path / 'recordings')
    library.ensure_directory()
    final = library.directory / 'saved.ovd'; final.write_bytes(PAYLOAD)
    displayed_alias = tmp_path / 'displayed.ovd'
    symlink(displayed_alias, final)
    with pytest.raises(ValueError):
        library.delete(final, protected=[displayed_alias])
    outside = tmp_path / 'outside.ovd'; outside.write_bytes(PAYLOAD)
    alias = library.directory / 'alias.ovd'
    symlink(alias, outside)
    assert [entry.path for entry in library.reload()] == [final]
    with pytest.raises(ValueError):
        library.delete(alias)
    assert final.read_bytes() == outside.read_bytes() == PAYLOAD


def test_fixed_timestamp_candidates_are_unique_without_files(tmp_path):
    library = RecordingLibrary(tmp_path / "recordings")
    paths = [library.candidate_path(1784637296.789) for _ in range(100)]
    assert len(set(paths)) == 100
    assert all(path.parent == library.directory and path.suffix == ".ovd" for path in paths)
    assert len({path.name.rsplit("_", 1)[0] for path in paths}) == 1
    assert not library.directory.exists()


def test_recording_reload_filters_and_orders_real_files(tmp_path):
    import os
    library = RecordingLibrary(tmp_path / "recordings")
    directory = library.ensure_directory()
    for name, payload, stamp in [("old.ovd", b"old", 100), ("new.OVD", b"newer", 200),
                                  ("ignore.part", b"unfinished", 300), ("ignore.txt", b"text", 400)]:
        path = directory / name;path.write_bytes(payload);os.utime(path, (stamp, stamp))
    (directory / "folder.ovd").mkdir()
    entries = library.reload()
    assert [(entry.path.name, entry.size_bytes, entry.recorded_at, entry.status) for entry in entries] == [
        ("new.OVD", 5, 200, "Saved"), ("old.ovd", 3, 100, "Saved")]
    library.upsert(directory / "old.ovd", status="Complete", recorded_at=300)
    assert [(entry.path.name, entry.status) for entry in library.reload()] == [("old.ovd", "Complete"), ("new.OVD", "Saved")]


def test_positional_startup_does_not_listen(workflow):
    assert workflow.startup(FIXTURE)
    assert workflow.display_session is not None and workflow.display_path == FIXTURE
    assert workflow.live_session is None and not workflow.session_recordings


def test_restored_port_survives_startup_and_new(tmp_path):
    path = tmp_path / "preferences.json"
    preferences = Preferences(path)
    port = free_port()
    while port == 5425: port = free_port()
    preferences.set("live_port", port)
    workflow = CaptureWorkflow(Preferences(path), RecordingLibrary(tmp_path / "recordings"))
    try:
        assert workflow.startup() and workflow.port == port
        assert workflow.live_session is not None
        workflow.new()
        assert workflow.port == port and workflow.live_session is not None
        assert not workflow.session_recordings
    finally:
        workflow.shutdown()


def test_copy_is_byte_exact_without_truncation_or_link_overwrite(tmp_path):
    source, destination = tmp_path / "source.ovd", tmp_path / "destination.ovd"
    source.write_bytes(PAYLOAD)
    assert copy_capture(source, destination).read_bytes() == PAYLOAD
    with pytest.raises(FileExistsError):
        copy_capture(source, destination)
    with pytest.raises(ValueError):
        copy_capture(source, source, overwrite=True)
    destination.write_bytes(b"old")
    copy_capture(source, destination, overwrite=True)
    assert destination.read_bytes() == source.read_bytes() == PAYLOAD


def test_copy_rejects_symbolic_link_overwrite(tmp_path):
    from filesystem_capabilities import symlink
    source = tmp_path / 'source.ovd'; source.write_bytes(PAYLOAD)
    alias = tmp_path / 'alias.ovd'
    symlink(alias, source)
    with pytest.raises(ValueError):
        copy_capture(source, alias, overwrite=True)
    assert source.read_bytes() == PAYLOAD and alias.is_symlink()


def test_two_connections_rearm_and_keep_exact_bytes(workflow):
    assert workflow.startup()
    original = workflow.live_session
    send_capture(workflow)
    events = until(workflow, lambda: len(workflow.recordings.entries) == 1
                   and workflow.live_session is not original)
    displayed = workflow.display_session
    assert displayed is original
    assert any(event.kind == "display" and event.session is original for event in events)
    assert workflow.recordings.entries[0].status == "Complete"
    assert workflow.display_path.read_bytes() == PAYLOAD
    second = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: len(workflow.recordings.entries) == 2 and workflow.live_session is not second)
    assert workflow.display_session is second
    assert all(entry.path.read_bytes() == PAYLOAD for entry in workflow.recordings.entries)
    assert len({entry.path for entry in workflow.recordings.entries}) == 2


def test_autosave_captures_in_memory_without_disk_access_until_completion(workflow):
    assert workflow.startup()
    active = workflow.live_session
    assert active.recording_path() == ""
    assert not workflow.recordings.directory.exists()
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        assert workflow.live_recording.path is None
        assert not workflow.recordings.entries
        assert not workflow.recordings.directory.exists()
        connection.sendall(PAYLOAD[FRAME2_END:])
    until(workflow, lambda: workflow.live_session is not active)
    retained = workflow.session_recordings[0]
    assert retained.session is active and retained.status == "Complete"
    assert retained.path.read_bytes() == PAYLOAD
    assert workflow.recordings.entries[0].path == retained.path
    assert active.recording_path() == ""


@pytest.mark.parametrize("blocked_path", ["directory", "parent"])
def test_bad_recordings_directory_does_not_block_capture_or_rearm(workflow, tmp_path, blocked_path):
    blocked = workflow.recordings.directory
    blocked.write_bytes(b"not a directory")
    if blocked_path == "parent":
        workflow.recordings.directory = blocked / "recordings"
    assert workflow.startup()
    first = workflow.live_session
    send_capture(workflow)
    events = until(workflow, lambda: workflow.live_session is not first)
    retained = workflow.session_recordings[0]
    assert retained.session is first and retained.path is None
    assert retained.status == "Complete" and retained.size_bytes == len(PAYLOAD)
    assert workflow.display_session is first and workflow.can_save
    assert any(event.kind == "error" and "Automatic save failed" in event.message for event in events)
    assert not workflow.recordings.entries
    assert blocked.read_bytes() == b"not a directory"
    result = workflow.save_capture(tmp_path / "rescued.ovd")
    assert result.read_bytes() == PAYLOAD
    second = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not second)
    assert len(workflow.session_recordings) == 2
    assert workflow.display_session is second and workflow.accept_connections
    # Once the external folder problem is corrected, a later capture autosaves.
    blocked.unlink()
    third = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not third)
    assert len(workflow.session_recordings) == 3
    assert workflow.session_recordings[0].path.read_bytes() == PAYLOAD
    assert workflow.select_session_recording(retained.id)
    assert workflow.display_session is first


def test_existing_bad_library_warns_without_blocking_startup(tmp_path):
    directory = tmp_path / "recordings"
    directory.write_bytes(b"not a directory")
    settings = Preferences(tmp_path / "preferences.json")
    settings.set("live_port", free_port())
    settings.set("keep_recordings", True)
    with CaptureWorkflow(settings, RecordingLibrary(directory)) as capture:
        assert capture.startup()
        events = capture.poll()
        assert any(event.kind == "error" and "recordings" in event.message.lower() for event in events)
        active = capture.live_session
        send_capture(capture)
        until(capture, lambda: capture.live_session is not active)
        assert capture.session_recordings[0].session is active
        assert capture.session_recordings[0].status == "Complete"
        assert capture.display_session is active and capture.can_save
    assert directory.read_bytes() == b"not a directory"


def test_failed_library_reload_preserves_existing_entries(tmp_path, monkeypatch):
    library = RecordingLibrary(tmp_path / "recordings")
    library.ensure_directory()
    saved = library.directory / "saved.ovd"
    saved.write_bytes(PAYLOAD)
    entries = library.reload()
    original = Path.iterdir

    def denied(path):
        if path == library.directory:
            raise PermissionError("recordings directory is unreadable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    with pytest.raises(PermissionError, match="unreadable"):
        library.reload()
    assert library.entries == entries
    assert saved.read_bytes() == PAYLOAD


def test_empty_and_rejected_connections_never_blank_display(workflow):
    assert workflow.open_file(FIXTURE)
    displayed = workflow.display_session
    assert workflow.attach(workflow.port)
    send_capture(workflow, PAYLOAD[:12])  # Complete version header with no commands/scenes.
    until(workflow, lambda: any(entry.status == "Empty" for entry in workflow.session_recordings))
    assert workflow.display_session is displayed
    send_capture(workflow, struct.pack("<III", 999, 0, 0))
    events = until(workflow, lambda: any(entry.status == "Rejected" for entry in workflow.session_recordings))
    assert workflow.display_session is displayed
    assert any(event.kind == "error" and "rejected" in event.message for event in events)
    assert not workflow.recordings.entries
    assert not workflow.recordings.directory.exists()


def test_first_partial_schema_is_not_published(workflow):
    workflow.open_file(FIXTURE)
    displayed = workflow.display_session
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + PAYLOAD[:1024])
        until(workflow, lambda: workflow.live_session.info().connected)
        for _ in range(10):
            workflow.poll()
            assert workflow.display_session is displayed
            time.sleep(.005)
        assert workflow.live_session.info().unpublished_changes
        connection.sendall(PAYLOAD[1024:])
    until(workflow, lambda: workflow.display_session is not displayed)
    assert workflow.display_session.info().completed_serial > 0


def test_accept_off_lets_connected_producer_finish(workflow):
    workflow.attach(workflow.port)
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + PAYLOAD[:1024])
        until(workflow, lambda: active.info().connected)
        workflow.set_accept_connections(False)
        assert workflow.live_session is active and not active.info().ended
        connection.sendall(PAYLOAD[1024:])
    until(workflow, lambda: workflow.live_session is None)
    assert workflow.display_session is active
    assert workflow.display_path.read_bytes() == PAYLOAD
    assert not workflow.accept_connections
    workflow.set_accept_connections(True)
    assert workflow.live_session is not None and workflow.live_session is not active


def test_keep_setting_preserves_idle_listener_and_applies_next_connection(workflow):
    workflow.attach(workflow.port)
    idle = workflow.live_session
    assert workflow.set_keep_recordings(False)
    assert workflow.live_session is idle and workflow._auto_save_active is False
    assert workflow.set_keep_recordings(True)
    assert workflow._auto_save_active is True
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + PAYLOAD[:1024])
        until(workflow, lambda: active.info().connected)
        workflow.set_keep_recordings(False)
        assert workflow.live_session is active
        connection.sendall(PAYLOAD[1024:])
    until(workflow, lambda: workflow.live_session is not active)
    assert workflow.recordings.entries[0].path.read_bytes() == PAYLOAD
    assert workflow.live_session.recording_path() == ""
    assert Preferences(workflow.settings.path).get("keep_recordings") is False


def test_keep_toggle_preserves_tcp_producer_before_handshake(workflow):
    workflow.set_keep_recordings(False)
    assert workflow.attach(workflow.port)
    idle = workflow.live_session
    with producer(workflow) as connection:
        workflow.poll()
        assert not idle.info().connected
        assert workflow.set_keep_recordings(True)
        assert workflow.live_session is idle
        connection.sendall(HANDSHAKE + PAYLOAD)
    until(workflow, lambda: workflow.live_session is not idle)
    recording, = workflow.session_recordings
    assert recording.status == "Complete" and recording.session is idle
    assert recording.path.read_bytes() == PAYLOAD


@pytest.mark.parametrize("save_failure", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_next_listener_is_armed_before_autosave(workflow, monkeypatch, save_failure, background):
    assert workflow.attach(workflow.port)
    active = workflow.live_session
    if background:
        assert workflow.open_file(FIXTURE)
    previous_display = workflow.display_session
    ensure_directory = workflow.recordings.ensure_directory
    observed = []

    def check_rearmed():
        successor = workflow.live_session
        assert successor is not None and successor is not active
        observed.append(successor)
        if save_failure:
            raise PermissionError("recordings disk unavailable")
        return ensure_directory()

    monkeypatch.setattr(workflow.recordings, "ensure_directory", check_rearmed)
    send_capture(workflow)
    events = until(workflow, lambda: workflow.live_session is not active)
    assert observed == [workflow.live_session]
    recording, = workflow.session_recordings
    assert recording.session is active and recording.recorded_at > 0
    assert workflow.accept_connections and workflow.connection_status.state == "waiting"
    assert workflow.status.startswith("Listening on port")
    assert workflow.display_session is (previous_display if background else active)
    if save_failure:
        assert recording.path is None
        assert any(event.kind == "error" and "Automatic save failed" in event.message for event in events)
    else:
        assert recording.path.read_bytes() == PAYLOAD
    assert workflow.display_path == (FIXTURE if background else recording.path)


@pytest.mark.parametrize("stop", ["detach", "shutdown"])
def test_mid_command_stop_reports_save_failure_without_retention_promise(workflow, stop):
    assert workflow.attach(workflow.port)
    active = workflow.live_session
    partial = PAYLOAD[:FRAME2_END + 5]
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + partial)
        until(workflow, lambda: workflow.display_session is active
              and active.info().recording_size > FRAME2_END)
        getattr(workflow, stop)()
    errors = [event.message for event in workflow.poll() if event.kind == "error"]
    assert any("Automatic save failed" in message for message in errors)
    assert all("kept" not in message and "later" not in message for message in errors)
    assert not workflow.recordings.entries and workflow.live_session is None
    if stop == "shutdown":
        assert not workflow.session_recordings
    else:
        recording, = workflow.session_recordings
        assert recording.session is active and recording.path is None


def test_reload_broken_recordings_reports_error_and_recovers(workflow):
    directory = workflow.recordings.directory
    directory.write_bytes(b"not a directory")
    assert not workflow.reload_recordings()
    assert any(event.kind == "error" and "recordings" in event.message.lower()
               for event in workflow.poll())
    assert directory.read_bytes() == b"not a directory"
    directory.unlink()
    directory.mkdir()
    saved = directory / "saved.ovd"
    saved.write_bytes(PAYLOAD)
    assert workflow.reload_recordings()
    assert [entry.path for entry in workflow.recordings.entries] == [saved]


def test_reload_failure_preserves_library_clips_and_listener(workflow, monkeypatch):
    assert workflow.attach(workflow.port)
    active = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not active)
    entries, clips = workflow.recordings.entries, workflow.session_recordings
    listener, displayed = workflow.live_session, workflow.display_session
    original = Path.iterdir

    def denied(path):
        if path == workflow.recordings.directory:
            raise PermissionError("recordings directory is unreadable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    assert not workflow.reload_recordings()
    assert workflow.recordings.entries == entries and workflow.session_recordings == clips
    assert workflow.live_session is listener and workflow.display_session is displayed
    assert any(event.kind == "error" and "unreadable" in event.message for event in workflow.poll())


def test_history_browsing_and_follow_latest_leave_ingest_running(workflow):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is workflow.live_session)
        active = workflow.live_session
        assert workflow.open_file(FIXTURE)
        history = workflow.display_session
        assert history is not active and workflow.live_session is active
        assert not workflow.follow_live
        connection.sendall(PAYLOAD[FRAME2_END:])
        until(workflow, lambda: active.info().recording_size == len(PAYLOAD)
              and active.info().unpublished_changes)
        workflow.poll()
        assert workflow.display_session is history
        workflow.follow_latest()
        assert workflow.display_session is history and workflow.follow_live
        assert "complete live frame" in workflow.status
    until(workflow, lambda: workflow.live_session is not active)
    assert workflow.display_session is active


def test_open_disk_recording_preserves_waiting_and_receiving_listener(workflow):
    assert workflow.attach(workflow.port)
    first = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not first)
    saved = workflow.recordings.entries[0].path
    listener = workflow.live_session
    assert workflow.open_recording(saved)
    assert workflow.live_session is listener and workflow.display_session is not listener
    assert workflow.display_path == saved and not workflow.follow_live
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow._useful)
        assert workflow.open_recording(saved)
        displayed = workflow.display_session
        assert workflow.live_session is listener and not listener.info().ended
        assert workflow.display_path == saved and not workflow.follow_live
        connection.sendall(PAYLOAD[FRAME2_END:])
    until(workflow, lambda: workflow.live_session is not listener)
    assert workflow.display_session is displayed and workflow.display_path == saved
    assert len(workflow.session_recordings) == 2 and workflow.accept_connections


def test_detach_freezes_display_and_new_preserves_recordings(workflow):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is workflow.live_session)
        displayed = workflow.display_session
        workflow.detach()
        assert workflow.display_session is displayed and workflow.live_session is None
        assert not workflow.accept_connections and not workflow.follow_live
        scene = displayed.scenes()[0]
        displayed.seek(scene.id, 1)
        assert displayed.scenes()[0].current_frame == 1
    saved = workflow.display_path
    assert saved.read_bytes() == ACTIVE_PAYLOAD
    assert not workflow.delete_recording(saved)
    workflow.new()
    assert workflow.display_session is None and workflow.live_session is None
    assert saved.exists()
    assert workflow.delete_recording(saved)
    workflow.set_accept_connections(True)
    old = workflow.live_session
    workflow.new()
    assert workflow.live_session is not old
    assert not workflow.recordings.entries


def test_save_stops_displayed_live_and_preserves_exact_stream(workflow, tmp_path):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is workflow.live_session)
        result = workflow.save_capture(tmp_path / "export")
        assert result == tmp_path / "export.ovd"
        assert result.read_bytes() == ACTIVE_PAYLOAD
        assert workflow.live_session is None and not workflow.accept_connections
        assert workflow.display_path.exists()
        assert not workflow.save_capture(workflow.display_path)
    assert workflow.display_path.read_bytes() == ACTIVE_PAYLOAD


def test_legacy_save_preserves_format_and_rejects_mislabeled_destination(workflow, tmp_path):
    source = ROOT / "test/data/cloth_physx34.pxd2"
    assert workflow.open_file(source)
    assert not workflow.save_capture(tmp_path / "wrong.ovd")
    assert not (tmp_path / "wrong.ovd").exists()
    result = workflow.save_capture(tmp_path / "legacy-copy")
    assert result == tmp_path / "legacy-copy.pxd2"
    assert result.read_bytes() == source.read_bytes()


def test_save_old_capture_does_not_stop_next_listener(workflow, tmp_path):
    workflow.attach(workflow.port)
    active = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not active)
    next_listener = workflow.live_session
    assert workflow.save_capture(tmp_path / "old.ovd")
    assert workflow.live_session is next_listener and workflow.accept_connections


def test_failed_file_replacement_and_restart_restore_history(workflow):
    workflow.open_file(FIXTURE)
    old = workflow.display_session
    assert not workflow.open_file(FIXTURE.with_name("missing.ovd"))
    assert workflow.display_session is old
    workflow.attach(workflow.port)
    active = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not active)
    workflow.shutdown()
    workflow.shutdown()
    assert workflow.closed and workflow.live_session is None
    assert not list(workflow.recordings.directory.glob("*.part"))
    with CaptureWorkflow(Preferences(workflow.settings.path), RecordingLibrary(workflow.recordings.directory)) as reopened:
        assert reopened.accept_connections and reopened.keep_recordings
        assert len(reopened.recordings.entries) == 1
        assert str(FIXTURE) in reopened.recent_files


def test_occupied_port_reports_retry_and_recovers(workflow):
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        workflow.attach(occupied.getsockname()[1])
        events = until(workflow, lambda: bool(workflow.error))
        assert any(event.kind == "error" and "port" in event.message for event in events)
        assert not workflow.session_recordings
        assert workflow.live_session is None and workflow.accept_connections
    until(workflow, lambda: workflow.live_session is not None, timeout=3)
    send_capture(workflow)
    until(workflow, lambda: bool(workflow.display_session))


def test_resampling_publishes_segment_transition_and_reused_handles(workflow):
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    # Known STOP_FRAME offsets from the checked-in two-segment PhysX capture.
    first_end, second_end = 42117, 98924
    assert payload[first_end - 17] == payload[second_end - 17] == 12
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + payload[:first_end])
        first_events = until(workflow, lambda: workflow.display_session is workflow.live_session)
        first = next(event for event in reversed(first_events) if event.kind == "frames")
        session = workflow.display_session
        connection.sendall(payload[first_end:second_end])
        second_events = until(workflow, lambda: session.info().segment_epoch != first.segment_epoch
                              and workflow._epoch != first.segment_epoch)
        second = next(event for event in reversed(second_events) if event.kind == "frames")
        assert second.segment_changed
        assert second.serial > first.serial
        assert workflow.display_session is session
        assert len(second.scenes) == 2
        assert len({scene.id for scene in second.scenes}) == 2
        assert len({scene.stream_id for scene in second.scenes}) == 1
        scene = next(scene for scene in second.scenes if scene.alive)
        session.seek(scene.id, 1)
        instances = session.instances()
        assert instances and len({item.id for item in instances}) == len(instances)
        connection.sendall(payload[second_end:])
    until(workflow, lambda: workflow.live_session is not session)
    assert workflow.display_path.read_bytes() == payload


@pytest.mark.parametrize("frameless_restarts", [0, 1, 3])
def test_discovery_then_apply_publishes_first_and_second_completed_transition(workflow, frameless_restarts):
    from test_native_capture_control import schema, metadata, catalog, create, ack, wait_for, META, CATALOG, SCENE
    workflow.attach(workflow.port)
    session, port = workflow.live_session, workflow.port
    with producer(workflow) as connection:
        connection.sendall(struct.pack("<IHHIII", 0x4f56444c, 1, 12, 0, 4, 0)
            + schema(1, "PxOmniPvdMetaData", META) + schema(2, "PxOmniPvdCaptureEntry", CATALOG)
            + schema(3, "PxScene", []) + metadata() + catalog())
        wait_for(lambda: session.capture_control_info()["catalog_ready"])
        assert not any(e.kind == "frames" for e in workflow.poll())
        update = session.refresh_live()
        assert hasattr(update, "segment_transition_serial"), "Completed metadata-boundary serial is missing"
        assert not update.ready and update.segment_transition_serial == 0
        restart = struct.pack("<BQQ", 10, 1, 100) + metadata()
        connection.sendall(restart * frameless_restarts + ack(1, 1, 40))
        wait_for(lambda: session.capture_control_info()["capture_epoch"] == 40)
        assert not any(e.kind == "frames" for e in workflow.poll())
        assert session.refresh_live().segment_transition_serial == 0
        connection.sendall(create(1, 3, SCENE) + b"".join(
            struct.pack("<BQQBQQ", 11, SCENE, frame, 12, SCENE, frame) for frame in range(1, 6)))
        wait_for(lambda: session.info().completed_serial == 5)
        first_events = workflow.poll()
        first = next(e for e in first_events if e.kind == "frames")
        assert (first.serial, first.segment_epoch, first.current_frame, first.max_frame) == (5, frameless_restarts, 5, 5)
        assert first.segment_changed is bool(frameless_restarts)
        assert first.segment_transition_count == frameless_restarts
        assert first.segment_transition_serial == frameless_restarts
        # Refresh/rejection ACKs alone cannot promote a new frame or segment.
        connection.sendall(ack(2, 4, 40))
        wait_for(lambda: session.capture_control_info()["ack_id"] == 2)
        assert not any(e.kind == "frames" for e in workflow.poll())
        connection.sendall(struct.pack("<BQQ", 10, 1, SCENE) + restart + ack(3, 1, 41))
        wait_for(lambda: session.capture_control_info()["capture_epoch"] == 41)
        assert not any(e.kind == "frames" for e in workflow.poll())
        connection.sendall(create(1, 3, SCENE) + b"".join(
            struct.pack("<BQQBQQ", 11, SCENE, frame, 12, SCENE, frame) for frame in (1, 2)))
        wait_for(lambda: session.info().completed_serial == 7)
        second = next(e for e in workflow.poll() if e.kind == "frames")
        assert (second.serial, second.segment_epoch, second.current_frame, second.max_frame) == (7, frameless_restarts + 1, 2, 5)
        assert second.segment_changed and second.segment_transition_count == 1
        assert second.segment_transition_serial == frameless_restarts + 1
        assert workflow.live_session is session and workflow.port == port and workflow.accept_connections
        assert not any(e.kind == "frames" for e in workflow.poll())


@pytest.mark.parametrize("save_failure", ["existing-file", "missing-parent"])
def test_autosave_failure_retains_memory_clip_and_accepts_later_connection(workflow, tmp_path, monkeypatch, save_failure):
    workflow.recordings.ensure_directory()
    occupied = workflow.recordings.directory / "existing.ovd"
    occupied.write_bytes(b"precious")
    destination = occupied if save_failure == "existing-file" else workflow.recordings.directory / "missing" / "capture.ovd"
    monkeypatch.setattr(workflow.recordings, "candidate_path", lambda *_: destination)
    assert workflow.attach(workflow.port)
    active = workflow.live_session
    send_capture(workflow)
    events = until(workflow, lambda: workflow.live_session is not active)
    retained = workflow.session_recordings[0]
    assert retained.session is active and retained.status == "Complete" and retained.path is None
    assert retained.size_bytes == len(PAYLOAD)
    assert workflow.display_session is active and workflow.display_path is None and workflow.can_save
    assert any(event.kind == "error" and "Automatic save failed" in event.message for event in events)
    assert occupied.read_bytes() == b"precious"
    result = workflow.save_capture(tmp_path / "rescued.ovd")
    assert result.read_bytes() == PAYLOAD
    assert not list(workflow.recordings.directory.glob("*.part"))
    second = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not second)
    assert len(workflow.session_recordings) == 2 and workflow.display_session is second
    assert workflow.select_session_recording(retained.id)
    assert workflow.display_session is active and workflow.accept_connections


@pytest.mark.parametrize('kind', ['missing-directory', 'symlink'])
def test_invalid_save_destination_does_not_detach_active_producer(workflow, tmp_path, kind):
    workflow.attach(workflow.port)
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        if kind == 'symlink':
            from filesystem_capabilities import symlink
            destination = tmp_path / 'alias.ovd'
            symlink(destination, FIXTURE)
            assert not workflow.save_capture(destination, overwrite=True)
        else:
            assert not workflow.save_capture(tmp_path / 'missing-directory/export.ovd')
        assert workflow.live_session is active and not active.info().ended


def test_keep_off_receives_without_creating_recording_files(workflow):
    workflow.set_keep_recordings(False)
    workflow.attach(workflow.port)
    active = workflow.live_session
    send_capture(workflow)
    until(workflow, lambda: workflow.live_session is not active)
    assert workflow.display_session is active
    assert workflow.display_path is None and workflow.can_save
    assert workflow.session_recordings[0].status == "Complete"
    assert not workflow.recordings.entries
    assert not workflow.recordings.directory.exists()


def test_detach_during_unpublished_teardown_can_scrub_last_committed_index(workflow):
    workflow.attach(workflow.port)
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        scene = active.scenes()[0]
        active.seek(scene.id, 2)
        assert active.scenes()[0].current_frame == 2
        connection.sendall(PAYLOAD[FRAME2_END:])
        until(workflow, lambda: active.info().recording_size == len(PAYLOAD)
              and active.info().unpublished_changes)
        workflow.detach()
        assert workflow.display_session is active
        active.seek(scene.id, 1)
        assert active.scenes()[0].current_frame == 1


def test_follow_background_capture_during_partial_schema_preserves_viewport(workflow):
    workflow.attach(workflow.port)
    workflow.open_file(FIXTURE)
    history = workflow.display_session
    history.seek(history.scenes()[0].id, 1)
    assert len(history.geometry_snapshot().object_ids) == 1
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow._useful)
        assert workflow.display_session is history
        connection.sendall(PAYLOAD[FRAME2_END:])
        until(workflow, lambda: active.info().recording_size == len(PAYLOAD)
              and active.info().unpublished_changes)
        # Native background engine has no cached geometry; promoting its metadata
        # alone here would clear a useful viewport with an empty default snapshot.
        assert len(active.geometry_snapshot().object_ids) == 0
        workflow.follow_latest()
        assert workflow.display_session is history
        assert len(history.geometry_snapshot().object_ids) == 1
    until(workflow, lambda: workflow.live_session is not active)
    assert workflow.display_session is active
    active.seek(active.scenes()[0].id, 1)
    assert len(active.geometry_snapshot().object_ids) == 1


def test_frozen_live_analysis_preserves_native_object_identity(workflow):
    from ovd_ui.analysis import HistoryRequest, UnsupportedAnalysis, read_scalar_history
    workflow.attach(workflow.port)
    active = workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE + ACTIVE_PAYLOAD)
        until(workflow, lambda: workflow.display_session is active)
        scene = active.scenes()[0]
        active.seek(scene.id, 1)
        classes = {item.id: item.name for item in active.classes()}
        actor = next(item for item in active.instances() if item.alive and classes[item.class_id] == "PxRigidDynamic")
        mass = next(item for item in active.properties(actor.class_id) if item.name == "mass")
        with pytest.raises(UnsupportedAnalysis, match="Detach"):
            HistoryRequest.from_display(workflow.display_path, active, actor.id, mass.id, scene_id=scene.id)
        workflow.detach()
    request = HistoryRequest.from_display(workflow.display_path, active, actor.id, mass.id, stop=3, scene_id=scene.id)
    series = read_scalar_history(request, batch_size=1)
    assert series.values == (2., 2.)
    assert active.scenes()[0].current_frame == 1
