# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History, messages and CSV use actual native capture data without moving the UI."""
from concurrent.futures import CancelledError
import csv
from dataclasses import replace
from pathlib import Path
import struct
from threading import Event

import pytest
from _pvd_native import Session

from ovd_ui.analysis import (AnalysisTasks, CaptureMessages, HistoryRequest, MessageSeverity,
                             ScalarSeries, UnsupportedAnalysis, can_plot_property,
                             classify_message, export_series_csv, read_frame_timing, read_scalar_history)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test/data/test_scene.ovd"


def selected_mass(tmp_path, filename="test_scene.ovd"):
    path = ROOT / "test/data" / filename
    session = Session(str(path), str(tmp_path))
    scene = session.scenes()[0]
    session.seek(scene.id, 1)
    classes = {item.id: item.name for item in session.classes()}
    instance = next(item for item in session.instances() if item.alive and classes[item.class_id] == "PxRigidDynamic")
    prop = next(item for item in session.properties(instance.class_id) if item.name.lower() == "mass")
    request = HistoryRequest.from_display(path, session, instance.id, prop.id, scene_id=scene.id)
    return session, instance, prop, request


def test_scalar_history_values_missing_lifetimes_and_display_unchanged(tmp_path):
    displayed, instance, prop, request = selected_mass(tmp_path)
    assert can_plot_property(prop)
    displayed.seek(request.scene_id, 2)
    progress = []
    series = read_scalar_history(request, batch_size=1, progress=lambda *value: progress.append(value))
    assert series.name == "mass" and series.type == "PvdF32"
    assert series.frames == (1, 2, 3)
    assert series.values == (2., 2., None)
    assert series.missing == 1 and series.bounds == (2., 2.)
    assert displayed.scenes()[0].current_frame == 2
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert series.nearest_frame(1.6) == 2
    assert series.nearest_frame(-4) == 1
    assert series.nearest_frame(500) == 3
    assert series.nearest_frame(float("nan")) is None


def test_scalar_history_legacy_matches_typed_native_values(tmp_path):
    displayed, instance, prop, request = selected_mass(tmp_path, "cloth_physx34.pxd2")
    series = read_scalar_history(replace(request, start=5, stop=9), batch_size=2)
    assert series.frames == (5, 6, 7, 8)
    for frame, value in zip(series.frames, series.values):
        displayed.seek(request.scene_id, frame)
        expected, = struct.unpack("<f", displayed.read(instance.id, prop.id))
        assert value == expected


def test_history_rejects_nonscalar_bad_identity_and_ranges(tmp_path):
    displayed, instance, prop, request = selected_mass(tmp_path)
    pose = next(item for item in displayed.properties(instance.class_id) if item.name == "globalPose")
    assert not can_plot_property(pose)
    with pytest.raises(ValueError):
        read_scalar_history(replace(request, property_id=pose.id))
    for changed in (replace(request, instance_id=999999), replace(request, property_id=999999),
                    replace(request, scene_id=999999), replace(request, start=-1),
                    replace(request, start=3, stop=3), replace(request, stop=5)):
        with pytest.raises((ValueError, IndexError)):
            read_scalar_history(changed)
    with pytest.raises(ValueError):
        read_scalar_history(request, max_samples=2)
    with pytest.raises(ValueError):
        read_scalar_history(request, batch_size=5000)


def test_cancelled_history_never_returns_partial_series_or_moves_display(tmp_path):
    displayed, _, _, request = selected_mass(tmp_path)
    cancel = Event()
    def progress(done, total):
        if done == 1:
            cancel.set()
    with pytest.raises(CancelledError):
        read_scalar_history(request, batch_size=1, progress=progress, cancel=cancel)
    assert displayed.scenes()[0].current_frame == 1
    with pytest.raises(CancelledError):
        read_scalar_history(replace(request, path=Path("missing.ovd")), cancel=cancel)


def test_analysis_worker_returns_owned_history_and_closes(tmp_path):
    _, _, _, request = selected_mass(tmp_path)
    tasks = AnalysisTasks()
    try:
        job = tasks.submit_history(request, batch_size=1)
        series = job.future.result(timeout=5)
        assert series.values == (2., 2., None)
        assert job.progress == (3, 3)
    finally:
        tasks.close()


def test_series_csv_quotes_labels_and_missing_values_without_overwrite(tmp_path):
    request = HistoryRequest(FIXTURE, 1, 1, 1)
    series = ScalarSeries(request, 'mass, "kg"', "PvdF32", (1, 2, 3), (2., None, -3.5))
    path = export_series_csv(series, tmp_path / "history.csv")
    with path.open(newline="") as stream:
        assert list(csv.reader(stream)) == [["frame", 'mass, "kg"'], ["1", "2.0"], ["2", ""], ["3", "-3.5"]]
    with pytest.raises(FileExistsError):
        export_series_csv(series, path)
    with pytest.raises(ValueError):
        export_series_csv(series, tmp_path / "history.ovd")
    export_series_csv(replace(series, values=(0., 1., 2.)), path, overwrite=True)
    assert '0.0' in path.read_text()


def test_series_csv_rejects_symbolic_link_overwrite(tmp_path):
    from filesystem_capabilities import symlink
    series = ScalarSeries(HistoryRequest(FIXTURE, 1, 1, 1), "mass", "PvdF32", (1,), (2.,))
    path = export_series_csv(series, tmp_path / "history.csv")
    before = path.read_bytes()
    alias = tmp_path / "alias.csv"
    symlink(alias, path)
    with pytest.raises(ValueError):
        export_series_csv(series, alias, overwrite=True)
    assert path.read_bytes() == before and alias.is_symlink()


def test_csv_disk_failure_preserves_existing_export(tmp_path, monkeypatch):
    request = HistoryRequest(FIXTURE, 1, 1, 1)
    series = ScalarSeries(request, "mass", "PvdF32", (1,), (2.,))
    target = tmp_path / "previous.csv"
    target.write_text("previous export\n")
    def fail_flush(_):
        raise OSError("simulated full disk")
    monkeypatch.setattr("ovd_ui.analysis.os.fsync", fail_flush)
    with pytest.raises(OSError):
        export_series_csv(series, target, overwrite=True)
    assert target.read_text() == "previous export\n"
    assert list(tmp_path.glob(".pvd-csv-*")) == []


def test_message_severity_matches_legacy_classification():
    assert classify_message(None) == classify_message("") == MessageSeverity.INFO
    assert classify_message("eDEBUG_INFO") == MessageSeverity.INFO
    assert classify_message("eDEBUG_WARNING") == classify_message("ePERF_WARNING") == MessageSeverity.WARNING
    assert classify_message("eINTERNAL_ERROR") == classify_message("eINVALID_PARAMETER") == MessageSeverity.ERROR


def test_capture_messages_native_rows_filters_and_export(tmp_path):
    session = Session(str(ROOT / "test/data/testRecordMessage.ovd"), str(tmp_path))
    messages = CaptureMessages(session)
    assert messages.refresh()
    assert messages.total > 0 and messages.total == len(messages.records)
    assert sum(messages.counts.values()) == messages.total
    assert all(record.type_name for record in messages.records)
    assert any(record.text for record in messages.records)
    assert not messages.refresh()
    for severity in MessageSeverity:
        assert all(record.severity == severity for record in messages.filtered(severity))
    selected = messages.records[0]
    assert selected in messages.filtered(text=selected.text[:15])
    assert messages.filtered(sort="frame") == tuple(sorted(messages.records, key=lambda row: (row.frame, row.index)))
    assert selected.target_frame(1) is None
    path = messages.export_csv(tmp_path / "messages.csv")
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["frame", "severity", "type", "message", "file", "line"]
    assert len(rows) == messages.total + 1
    assert any(selected.text == row[3] for row in rows[1:])
    limited = CaptureMessages(session, limit=1)
    limited.refresh()
    assert len(limited.records) == 1 and limited.truncated == (messages.total > 1)
    del session
    assert selected.text  # Adapter/native messages are owned records, not borrowed char pointers.


def test_legacy_messages_empty_and_timing_measurements(tmp_path):
    session = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    messages = CaptureMessages(session)
    assert not messages.refresh() and messages.records == ()
    scene = session.scenes()[0]
    series = read_frame_timing(ROOT / "test/data/cloth_physx34.pxd2", scene.id, 1, 4)
    assert series.frames == (1, 2, 3)
    assert len(series.inner_ms) == len(series.total_ms) == 3
    assert all(value >= 0 for value in series.inner_ms + series.total_ms)
    assert any(value > 0 for value in series.total_ms)
    path = export_series_csv(series, tmp_path / "timing.csv")
    assert "PhysX frame (ms)" in path.read_text()
    ovd = Session(str(FIXTURE), str(tmp_path))
    with pytest.raises(UnsupportedAnalysis, match="do not contain"):
        read_frame_timing(FIXTURE, ovd.scenes()[0].id)
