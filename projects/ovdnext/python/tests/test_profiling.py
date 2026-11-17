# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real legacy parser parity, owned diagnostic data and UI-neutral interaction."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest
import _pvd_native as native
from ovd_ui.analysis import UnsupportedAnalysis
from ovd_ui.profiling import (DiagnosticsRequest, DiagnosticsTasks, EventKey, MemoryModel,
                              ProfileHistoryRequest, ProfileModel, export_profile_csv,
                              read_memory, read_profile, read_profile_history, thread_name)
from reference_data import load

ROOT = Path(__file__).resolve().parents[2]
CLOTH = ROOT / "test/data/cloth_physx34.pxd2"


@pytest.fixture
def display(tmp_path):
    session = native.Session(str(CLOTH), str(tmp_path))
    scene = session.scenes()[0]
    assert session.seek(scene.id, 2)
    return session, scene.id


@pytest.fixture(scope="session")
def reference():
    return {(r["capture"], r["frame"]): r for r in load("profiling")}


@pytest.mark.parametrize("filename,frame", [("cloth_physx34.pxd2", 2), ("ue4_shooter.pxd2", 2), ("apex_destruction.pxd2", 3)])
def test_native_profile_counts_and_ticks_equal_original_qt_handler(reference, tmp_path, filename, frame):
    path = ROOT / "test/data" / filename
    import hashlib
    record = reference[filename, frame]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["capture_sha256"]
    expected = record["summaries"]
    expected = {(row["zone"], row["event"], int(row["context"]), int(row["thread"])): (row["count"], int(row["ticks"])) for row in expected}
    s = native.Session(str(path), str(tmp_path))
    scene = s.scenes()[0]
    s.seek(scene.id, frame)
    before = s.geometry_snapshot().transforms.copy()
    data = native.profile_snapshot(s, scene.id)
    names={int(key):value for key,value in record["context_names"].items()}
    assert data.context_names==names and names
    timings_only=native.profile_snapshot(s,scene.id,include_context_names=False)
    assert timings_only.context_names=={}
    assert [(r.total_ticks,r.percent) for r in timings_only.summaries]==[(r.total_ticks,r.percent) for r in data.summaries]
    # Python receives an owned copy; changes cannot rewrite the native result.
    external=data.context_names;external.clear()
    assert data.context_names==names
    assert data.supported and data.current_frame == frame and data.scene_id == scene.id
    actual = {(row.zone_id, row.event_id, row.context_id, row.thread_id): (row.count, row.total_ticks) for row in data.summaries}
    assert actual == expected and len(actual) > 5
    assert sum(row.count for row in data.summaries) == len(data.scopes)
    assert all(scope.end_tick >= scope.start_tick and scope.start_ms >= 0 for scope in data.scopes)
    for row in data.summaries:
        assert row.total_ms == pytest.approx(row.total_ticks * data.tick_numerator / data.tick_denominator / 1e6)
        assert row.average_ms == pytest.approx(row.total_ms / row.count)
    assert s.scenes()[0].current_frame == frame
    assert (s.geometry_snapshot().transforms == before).all()
    s.seek(scene.id, 3 if frame != 3 else 2)
    del s
    assert data.context_names==names
    assert data.scopes and data.summaries[0].name  # owned after seek/destruction


def test_ovd_is_unsupported_and_empty_initial_frame_has_no_invented_scopes(tmp_path):
    s = native.Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    for reader in (native.profile_snapshot, native.memory_snapshot):
        data = reader(s)
        assert not data.supported and "OVD" in data.reason
    with pytest.raises(UnsupportedAnalysis, match="OVD"):
        DiagnosticsRequest.from_display(ROOT / "test/data/test_scene.ovd", s)
    s = native.Session(str(CLOTH), str(tmp_path))
    data = native.profile_snapshot(s)
    assert data.supported and data.scopes == [] and data.summaries == [] and data.current_frame == 0
    memory = native.memory_snapshot(s)
    assert memory.rows == []
    assert memory.outstanding and all(o.bytes is None and not o.valid for o in memory.outstanding)


def test_native_memory_preserves_original_frame_sums_and_separate_outstanding(display):
    session, scene = display
    data = native.memory_snapshot(session, scene)
    assert data.supported and data.current_frame == 2
    assert data.raw_event_count == 91
    assert data.total.allocated_bytes == 11236 and data.total.deallocated_bytes == 4576
    assert data.total.churn_bytes == 6660
    assert data.total.allocated_bytes == sum(r.allocated_bytes for r in data.rows)
    assert data.total.deallocated_bytes == sum(r.deallocated_bytes for r in data.rows)
    assert data.unmatched_deallocations == 1 and data.diagnostics
    assert len(data.outstanding) == 144
    assert all(o.valid for o in data.outstanding)
    assert sum(o.bytes for o in data.outstanding) == 1247073
    assert session.scenes()[0].current_frame == 2
    session.seek(scene, 3)
    assert sum(o.bytes for o in data.outstanding) == 1247073


def test_budget_errors_and_bad_ids_leave_native_display_unchanged(display):
    session, scene = display
    for reader in (native.profile_snapshot, native.memory_snapshot):
        with pytest.raises(ValueError):
            reader(session, scene, 0)
        with pytest.raises(ValueError):
            reader(session, scene, 1_000_001)
        with pytest.raises(IndexError):
            reader(session, 99999999)
    with pytest.raises(RuntimeError, match="budget"):
        native.profile_snapshot(session, scene, 200)
    session.seek(scene, 1)
    with pytest.raises(RuntimeError, match="budget"):
        native.memory_snapshot(session, scene, 200)
    assert session.scenes()[0].current_frame == 1
    assert native.memory_snapshot(session, scene).total.allocated_bytes == 4538780


def test_budget_callback_error_does_not_leak_original_parser_allocations(tmp_path):
    script = '''import _pvd_native as n,sys
s=n.Session(sys.argv[1],sys.argv[2]);scene=s.scenes()[0].id;s.seek(scene,1)
for reader in (n.profile_snapshot,n.memory_snapshot):
 try: reader(s,scene,200)
 except RuntimeError: pass
del s
'''
    result = subprocess.run([sys.executable, "-c", script, str(CLOTH), str(tmp_path)], check=True, capture_output=True, text=True)
    assert "Memory leak found" not in result.stdout + result.stderr


def test_private_read_and_concurrent_native_calls_never_move_display(display):
    session, scene = display
    req = DiagnosticsRequest.from_display(CLOTH, session, scene)
    profile, memory = read_profile(req), read_memory(req)
    assert profile.current_frame == memory.current_frame == 2
    assert session.scenes()[0].current_frame == 2
    with ThreadPoolExecutor(max_workers=3) as pool:
        snapshots = list(pool.map(lambda _: native.profile_snapshot(session, scene), range(6)))
    assert all(len(s.scopes) == 547 for s in snapshots)
    event = Event();event.set()
    with pytest.raises(CancelledError):
        read_profile(req, cancel=event)
    with pytest.raises(CancelledError):
        read_memory(req, cancel=event)


def test_profile_model_search_visibility_lanes_and_markers(display):
    session, scene = display
    data = native.profile_snapshot(session, scene)
    model = ProfileModel();model.load(data)
    row = next(row for row in model.rows() if row.total_ms > 0)
    key = EventKey.of(row)
    assert row in model.search(row.name)
    assert all(r.context_id == row.context_id for r in model.search(str(row.context_id), "context_id"))
    assert model.search("18446744073709551616", "context_id") == ()
    assert model.search("-1", "thread_id") == ()
    assert model.search("１２", "event_id") == ()
    assert thread_name(native.CROSS_THREAD_ID) == "Cross Thread"
    model.select([key]);model.set_selected_visibility(False)
    assert row not in model.rows() and row in model.rows(include_hidden=True)
    assert all(EventKey.of(s) != key for s in model.scopes())
    model.set_selected_visibility(True);assert row in model.rows()
    model.set_selected_visibility(False, inverse=True)
    assert model.rows() == (row,)
    model.set_selected_visibility(True, inverse=True)
    lane = next(l for l in model.lanes if any(s.duration_ms > 0 for s in model.scopes(l)))
    scope = next(s for s in model.scopes(lane) if s.duration_ms > 0)
    hit = model.hit_test(lane, (scope.start_ms + scope.end_ms) / 2)
    assert hit and EventKey.of(hit) == EventKey.of(scope)
    model.hide([EventKey.of(scope)]);assert model.hit_test(lane, (scope.start_ms + scope.end_ms) / 2) is None
    model.mark(data.duration_ms * .8, data.duration_ms * .2)
    assert model.zoom_to_mark() and model.view_range == pytest.approx((data.duration_ms * .2, data.duration_ms * .8))
    model.zoom(.5);width = model.view_range[1] - model.view_range[0]
    model.pan(1e8);assert model.view_range[1] == pytest.approx(data.duration_ms)
    assert model.view_range[1] - model.view_range[0] == pytest.approx(width)
    model.set_marker(1e9);assert model.marker_ms == data.duration_ms
    with pytest.raises(ValueError):model.mark(float("nan"), 1)
    with pytest.raises(ValueError):model.zoom(0)
    model.load(native.profile_snapshot(session, scene))
    assert key in model.selected  # per-frame identity persists
    model.begin_capture();assert not model.selected and not model.hidden and model.snapshot is None


def test_profile_history_uses_selected_key_gaps_and_native_duration(display, tmp_path):
    session, scene = display
    data = native.profile_snapshot(session, scene)
    incomplete = {EventKey.of(u) for u in data.unmatched}
    row = next(r for r in data.summaries if EventKey.of(r) not in incomplete and r.total_ms > 0)
    request = ProfileHistoryRequest(CLOTH, scene, row.zone_id, row.event_id, row.context_id, row.thread_id, 2, 3)
    series = read_profile_history(request)
    assert series.frames == (2,) and series.values == pytest.approx((row.total_ms,))
    assert series.nearest_frame(100) == 2 and session.scenes()[0].current_frame == 2
    target = export_profile_csv(series, tmp_path / "profile.csv")
    with target.open(newline="") as stream:rows = list(csv.reader(stream))
    assert rows[0][0] == "frame" and "ms" in rows[0][1]
    assert float(rows[1][1]) == pytest.approx(row.total_ms)
    with pytest.raises(FileExistsError):export_profile_csv(series, target)
    multi = ProfileHistoryRequest(CLOTH, scene, row.zone_id, row.event_id, row.context_id, None, 2, 3)
    multi_series = read_profile_history(multi)
    expected = sum(r.total_ms for r in data.summaries if (r.zone_id, r.event_id, r.context_id) == (row.zone_id, row.event_id, row.context_id))
    assert multi_series.values == pytest.approx((expected,))
    boundary = data.unmatched[0]
    missing = read_profile_history(ProfileHistoryRequest(CLOTH, scene, boundary.zone_id, boundary.event_id, boundary.context_id, boundary.thread_id, 2, 3))
    assert missing.values == (None,)
    with pytest.raises(ValueError, match="identity"):
        read_profile_history(ProfileHistoryRequest(CLOTH, scene, 999999, row.event_id, row.context_id, row.thread_id, 2, 3))
    with pytest.raises(ValueError, match="No timed scopes"):
        read_profile_history(ProfileHistoryRequest(CLOTH, scene, row.zone_id, row.event_id, 2**64 - 1, row.thread_id, 2, 3))


def test_memory_model_csv_and_background_service(display, tmp_path):
    session, scene = display
    request = DiagnosticsRequest.from_display(CLOTH, session, scene)
    tasks = DiagnosticsTasks()
    try:
        job = tasks.submit_memory(request)
        data = job.future.result(timeout=20)
        assert job.progress == (1, 1)
        profile = tasks.submit_profile(request).future.result(timeout=20)
        assert profile.scopes
    finally:tasks.close()
    model = MemoryModel(data)
    rows = model.rows();assert rows[0].allocated_bytes >= rows[-1].allocated_bytes
    assert model.rows("missing-substring") == ()
    assert len(model.outstanding()) == 144
    target = model.export_csv(tmp_path / "memory.csv")
    with target.open(newline="") as stream:rows = list(csv.reader(stream))
    assert rows[0] == ["file", "line", "type", "allocated_bytes", "allocations", "deallocated_bytes", "deallocations", "churn_bytes"]
    assert sum(int(r[3]) for r in rows[1:]) == 11236
    assert session.scenes()[0].current_frame == 2
