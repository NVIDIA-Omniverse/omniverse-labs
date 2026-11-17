# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy error records from the real shooter recording, distinct from OVD messages."""
from concurrent.futures import CancelledError
import csv
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import _pvd_native as native
from ovd_ui.analysis import UnsupportedAnalysis
from ovd_ui.error_stream import ErrorRequest, ErrorTasks, ErrorStreamModel, read_errors
from ovd_ui.profiling import DiagnosticsRequest, MemoryModel, read_memory

ROOT = Path(__file__).resolve().parents[2]
SHOOTER = ROOT / 'test/data/ue4_shooter.pxd2'
CLOTH = ROOT / 'test/data/cloth_physx34.pxd2'


def test_original_first_error_log_records_are_owned_and_frame_scoped(tmp_path):
    s = native.Session(str(SHOOTER), str(tmp_path))
    scene = s.scenes()[0]
    s.seek(scene.id, 921)
    data = native.error_snapshot(s, scene.id)
    assert data.supported and data.has_log and not data.truncated
    assert data.total == 1 and data.current_frame == 921
    record = data.records[0]
    assert record.index == 0 and record.code == 4 and record.line == 437
    assert record.message == 'RigidBody::setRigidBodyFlag: kinematic bodies with CCD enabled are not supported! CCD will be ignored.'
    assert record.file.endswith('NpRigidBodyTemplate.h')
    assert s.scenes()[0].current_frame == 921
    s.seek(scene.id, 0)
    empty = native.error_snapshot(s, scene.id)
    assert empty.supported and not empty.has_log and empty.records == [] and empty.total == 0
    del s
    assert record.file and record.message and data.records[0].line == 437


def test_error_limits_unknown_scene_and_ovd_are_explicit(tmp_path):
    s = native.Session(str(SHOOTER), str(tmp_path))
    for limit in (0, 100001):
        with pytest.raises(ValueError, match='limit'):
            native.error_snapshot(s, max_records=limit)
    with pytest.raises(IndexError, match='scene'):
        native.error_snapshot(s, scene_id=123)
    s = native.Session(str(ROOT/'test/data/testRecordMessage.ovd'), str(tmp_path))
    data = native.error_snapshot(s)
    assert not data.supported and 'Capture messages' in data.reason
    with pytest.raises(UnsupportedAnalysis, match='Capture messages'):
        ErrorRequest.from_display(ROOT/'test/data/testRecordMessage.ovd', s)


def test_error_private_worker_csv_cancel_and_original_display_unchanged(tmp_path):
    s = native.Session(str(SHOOTER), str(tmp_path))
    scene = s.scenes()[0]
    request = ErrorRequest(SHOOTER, scene.id, 921)
    progress = []
    data = read_errors(request, progress=lambda *p: progress.append(p))
    assert s.scenes()[0].current_frame == 0 and progress == [(0, 1), (1, 1)]
    model = ErrorStreamModel(data)
    assert len(model.rows('ccd')) == 1 and model.rows('nonexistent') == ()
    path = tmp_path/'errors.csv'
    model.export_csv(path, text='ccd')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['message'] == data.records[0].message and rows[0]['line'] == '437'
    with pytest.raises(FileExistsError):
        model.export_csv(path)
    cancellation = Event(); cancellation.set()
    with pytest.raises(CancelledError):
        read_errors(request, cancel=cancellation)
    tasks = ErrorTasks()
    try:
        job = tasks.submit_errors(request)
        assert job.future.result(timeout=10).total == 1 and job.progress == (1, 1)
    finally:
        tasks.close()


def test_memory_outstanding_and_filtered_csv_do_not_mix_frame_activity(tmp_path):
    s = native.Session(str(CLOTH), str(tmp_path))
    scene = s.scenes()[0]
    empty = MemoryModel(native.memory_snapshot(s, scene.id))
    assert empty.outstanding() and all(r.bytes is None for r in empty.outstanding())
    path = tmp_path/'outstanding-empty.csv'
    empty.export_csv(path, outstanding=True, sort='bytes')
    with path.open(newline='') as stream:
        assert all(r['recorded_outstanding_bytes'] == '' for r in csv.DictReader(stream))
    data = read_memory(DiagnosticsRequest(CLOTH, scene.id, 2, max_events=1_000_000))
    model = MemoryModel(data)
    for reverse in (False, True):
        amounts = [r.bytes for r in model.outstanding(sort='bytes', reverse=reverse)]
        assert amounts == sorted(amounts, reverse=reverse)
    term = model.rows()[0].type
    path = tmp_path/'outstanding.csv'
    model.export_csv(path, text=term, outstanding=True, sort='bytes')
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(model.outstanding(term))
    assert 'allocated_bytes' not in rows[0] and 'recorded_outstanding_bytes' in rows[0]


def test_error_numeric_sort_and_memory_unknown_values_always_last():
    records = [SimpleNamespace(index=i,code=code,message='quoted, message',file='x.cpp',line=100-i) for i,code in enumerate((100,2,20))]
    model = ErrorStreamModel(SimpleNamespace(records=records))
    assert [r.code for r in model.rows(sort='code')] == [2,20,100]
    with pytest.raises(ValueError):model.rows(sort='missing')
    rows = [SimpleNamespace(file='x',type='T',line=i,index=i,bytes=v) for i,v in enumerate((None,100,-5,2))]
    model = MemoryModel(SimpleNamespace(outstanding=rows))
    assert [r.bytes for r in model.outstanding(reverse=False)] == [-5,2,100,None]
    assert [r.bytes for r in model.outstanding(reverse=True)] == [100,2,-5,None]
