# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recent-menu refresh follows the original filesystem/MRU lifecycle."""
from ovd_ui.settings import Preferences
from test_workflow import FIXTURE, PAYLOAD, workflow


def recent_copies(workflow, tmp_path, count=10):
    paths = []
    for index in range(count):
        path = tmp_path / f"capture-{index}.ovd"
        path.write_bytes(PAYLOAD)
        paths.append(path)
        workflow.settings.add_recent(path)
    return paths


def test_refresh_prunes_only_missing_entries_and_preserves_display_and_recordings(workflow, tmp_path):
    assert workflow.open_file(FIXTURE)
    session = workflow.display_session
    scene = session.scenes()[0]
    session.seek(scene.id, 2)
    recording = workflow.recordings.ensure_directory() / "preserved.ovd"
    recording.write_bytes(PAYLOAD)
    workflow.recordings.reload()
    paths = recent_copies(workflow, tmp_path)
    before = workflow.recent_files
    assert len(before) == 8
    paths[8].unlink()
    paths[6].unlink()
    # A temporarily absent file restored before refresh retains its position.
    paths[6].write_bytes(PAYLOAD)
    assert workflow.recent_files == before
    expected = tuple(path for path in before if path != str(paths[8]))
    assert workflow.refresh_recent_files() == expected
    assert Preferences(workflow.settings.path).recent_files == expected
    assert workflow.display_session is session and workflow.display_path == FIXTURE
    assert session.scenes()[0].current_frame == 2
    assert recording.read_bytes() == PAYLOAD
    assert all(path.read_bytes() == PAYLOAD for path in paths if path.exists())


def test_restoring_pruned_path_requires_successful_reopen_then_retains_mru_cap(workflow, tmp_path):
    paths = recent_copies(workflow, tmp_path)
    removed = paths[5]
    removed.unlink()
    workflow.refresh_recent_files()
    removed.write_bytes(PAYLOAD)
    assert str(removed) not in workflow.refresh_recent_files()
    assert workflow.open_file(removed)
    assert workflow.recent_files[0] == str(removed)
    assert len(workflow.recent_files) == 8
    assert workflow.open_file(paths[9])
    assert workflow.recent_files[:2] == (str(paths[9]), str(removed))
    assert len(set(workflow.recent_files)) == len(workflow.recent_files) == 8
    events = workflow.poll()
    assert any(event.kind == "recents" for event in events)


def test_failed_refresh_persistence_preserves_history_and_useful_capture(workflow, tmp_path, monkeypatch):
    assert workflow.open_file(FIXTURE)
    session = workflow.display_session
    missing = tmp_path / "missing.ovd"
    workflow.settings.add_recent(missing)
    before = workflow.recent_files
    disk = workflow.settings.path.read_bytes()
    def fail_write(*_):
        raise PermissionError("read-only preferences")
    monkeypatch.setattr(workflow.settings, "set", fail_write)
    assert workflow.refresh_recent_files() == before
    assert workflow.settings.path.read_bytes() == disk
    assert workflow.display_session is session
    assert "read-only preferences" in workflow.error


def test_clear_history_keeps_current_capture_and_files(workflow, tmp_path):
    paths = recent_copies(workflow, tmp_path)
    assert workflow.open_file(paths[2])
    session = workflow.display_session
    workflow.poll()
    assert workflow.clear_recent()
    assert workflow.recent_files == Preferences(workflow.settings.path).recent_files == ()
    assert workflow.display_session is session and workflow.display_path == paths[2]
    assert all(path.read_bytes() == PAYLOAD for path in paths)
    assert any(event.kind == "recents" for event in workflow.poll())
