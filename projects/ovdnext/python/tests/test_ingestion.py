# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual native capture ingestion, archive confinement and ownership boundaries."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import gc
import hashlib
import os
from pathlib import Path
import stat
import struct
import subprocess
import time
from threading import Event
import zipfile

import pytest
from _pvd_native import LoadControl, Session

from ovd_ui.ingestion import (CaptureOpener, first_dropped_capture, load_capture,
                             retain_capture, safe_archive_member, submit_capture)
from ovd_ui.recordings import RecordingLibrary
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]
OVD = ROOT / "test/data/test_scene.ovd"
LEGACY = ROOT / "test/data/cloth_physx34.pxd2"


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-indexed")
    source = Session(str(LEGACY), str(directory))
    path = directory / "native-indexed.pxd3"
    source.save_clip(str(path), source.scenes()[0].id, 1, 3)
    # Native indexed output, not a raw PXD2 fixture with a renamed extension.
    assert struct.unpack("<I", path.read_bytes()[:4])[0] == 907025328
    return path


@pytest.fixture(scope="module")
def raw_streams(tmp_path_factory):
    binary=executable("pvd_ingestion_fixture","PVD_INGESTION_FIXTURE")
    directory=tmp_path_factory.mktemp("native-raw-streams")
    paths=[]
    for count in (3,20_000):
        path=directory/f"stream-{count}.pxd2"
        subprocess.run([str(binary),str(path),str(count)],check=True)
        assert struct.unpack("<I",path.read_bytes()[:4])[0]==837150850
        paths.append(path)
    return paths


@pytest.fixture
def archive(tmp_path, indexed):
    path = tmp_path / "recording.pvd"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(indexed, "nested/Recorded.PXD2")
        package.writestr("../unrelated.txt", "This member is never extracted")
    return path


def test_native_ovd_raw_stream_and_real_indexed_database(tmp_path, indexed,raw_streams):
    assert struct.unpack("<I",LEGACY.read_bytes()[:4])[0]==907025328
    for path in (OVD, LEGACY, indexed,raw_streams[0]):
        loaded = load_capture(path, temporary_directory=tmp_path)
        assert loaded.source_path == path
        if path.suffix == ".ovd":
            assert loaded.data_path != path and loaded.verified_payload.path == loaded.data_path
            assert not loaded.archived
        else:
            assert loaded.data_path == path and loaded.verified_payload is None
        assert any(scene.frame_count for scene in loaded.session.scenes())
        directory = loaded._resources.path
        loaded.close();gc.collect()
        assert not directory.exists()


def test_ovd_snapshot_survives_source_replacement_and_retains_lease(tmp_path):
    source = tmp_path / "original.ovd"
    original = OVD.read_bytes();source.write_bytes(original)
    loaded = load_capture(source, temporary_directory=tmp_path)
    proof = loaded.verified_payload
    assert proof.sha256 == hashlib.sha256(original).hexdigest()
    assert proof.byte_length == len(original) and proof.path.suffix == ".ovd"
    source.write_bytes(b"Replaced after successful load")
    native, payload = loaded.session, loaded.data_path
    with retain_capture(payload):
        loaded.close();gc.collect()
        assert payload.read_bytes() == original
        assert native.presentation(frame=2).request_applied
        del native;gc.collect()
        assert payload.exists()
    gc.collect()
    assert not payload.exists()


@pytest.mark.parametrize("cancel", [False, True])
def test_ovd_mutation_or_cancel_during_copy_discards_private_payload(tmp_path, monkeypatch, cancel):
    from ovd_ui.ingestion import OpenJob
    source = tmp_path / "source.ovd";source.write_bytes(OVD.read_bytes())
    job = OpenJob()
    original_update = job._update
    observed = []
    def during_copy(stage, processed=0, total=0):
        original_update(stage, processed, total)
        if stage == "Snapshotting OVD payload" and processed and not observed:
            observed.append(True)
            if cancel:
                job.cancel()
            else:
                source.write_bytes(b"Changed while copying")
    monkeypatch.setattr(job, "_update", during_copy)
    error = CancelledError if cancel else ValueError
    with pytest.raises(error, match="cancelled" if cancel else "changed"):
        job._run(source, tmp_path)
    assert observed
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source.ovd"]


def test_large_ovd_preserves_ordinary_loading_without_verified_identity(tmp_path, monkeypatch):
    from ovd_ui import ingestion
    monkeypatch.setattr(ingestion, "MAX_VERIFIED_OVD_BYTES", OVD.stat().st_size - 1)
    loaded = load_capture(OVD, temporary_directory=tmp_path)
    assert loaded.verified_payload is None and loaded.data_path == OVD and not loaded.archived
    assert "limit" in loaded.evidence_reason.lower()
    assert loaded.session.presentation(frame=2).request_applied
    loaded.close()


def test_archive_ownership_native_session_and_analysis_lease(archive):
    loaded = load_capture(archive)
    assert loaded.archived and loaded.source_path == archive
    assert loaded.data_path.name == "capture.pxd2"
    directory, payload = loaded._resources.path, loaded.data_path
    assert sorted(p.name for p in directory.iterdir()) == ["capture.pxd2"]
    native = loaded.session
    with retain_capture(payload):
        loaded.close();gc.collect()
        assert payload.exists() and any(scene.frame_count for scene in native.scenes())
        del native;gc.collect()
        assert payload.exists()
    gc.collect()
    assert not directory.exists()


@pytest.mark.parametrize("name", ["../bad.pxd2", "/bad.pxd2", "C:\\bad.pxd2", "nested/../bad.pxd2",
    "nested//bad.pxd2", "nested/./bad.pxd2", "a.pxd2:stream", "CON.pxd2", "aux.PXD2",
    "LPT¹.pxd2", "folder./bad.pxd2", "folder /bad.pxd2", "bad?.pxd2", "bad\x00.pxd2",
    "CLOCK$.pxd2", "CONIN$.pxd2", "CONOUT$.pxd2", "LPT9.pxd2", "COM².pxd2", "LPT³.pxd2",
    "capture.pxd2.", "capture.pxd2 ",
    "a"*256+"/bad.pxd2", "\U0001f916"*128+"/bad.pxd2"])
def test_archive_component_policy(name):
    assert not safe_archive_member(name)
    assert safe_archive_member("nested/valid capture.PXD2")
    # Windows component limits count UTF-16 code units, not Unicode characters.
    assert safe_archive_member("a"*255+"/valid.pxd2")
    assert safe_archive_member("\U0001f916"*127+"a/valid.pxd2")


@pytest.mark.parametrize("size,free,admitted", [(190, 200, True), (191, 200, False),
    (3*1024**3, 4*1024**3, True), (4*1024**3, 4*1024**3, False)])
def test_archive_reserve_boundary_and_large_member(tmp_path, monkeypatch, size, free, admitted):
    import io
    from types import SimpleNamespace
    from ovd_ui import ingestion
    original = zipfile.ZipFile
    data = io.BytesIO()
    with original(data, "w") as archive: archive.writestr("capture.pxd2", b"x")
    data.seek(0)
    opened = []
    class PayloadReached(Exception): pass
    class DeclaredSizeArchive(original):
        def infolist(self):
            members = super().infolist();members[0].file_size = size;return members
        def open(self, member, *args, **kwargs):
            opened.append(member.filename)
            raise PayloadReached("admitted without allocating the declared payload")
    monkeypatch.setattr(ingestion.zipfile, "ZipFile", DeclaredSizeArchive)
    monkeypatch.setattr(ingestion.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    with pytest.raises(PayloadReached if admitted else OSError, match="admitted|free space"):
        ingestion.OpenJob()._extract(data, SimpleNamespace(path=tmp_path))
    assert opened == (["capture.pxd2"] if admitted else [])
    assert (tmp_path / "capture.pxd2").exists() == admitted
    if admitted: assert (tmp_path / "capture.pxd2").stat().st_size == 0


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "symlink", "empty", "malformed", "bad_payload"])
def test_failed_archives_keep_useful_capture_and_recents(tmp_path, kind):
    workflow = CaptureWorkflow(Preferences(tmp_path / "settings.json"), RecordingLibrary(tmp_path / "library"))
    try:
        assert workflow.open_file(OVD)
        original = workflow.display_session
        workflow.poll()
        bad = tmp_path / (kind + ".pvd")
        if kind == "malformed":bad.write_bytes(b"not a ZIP archive")
        else:
            with zipfile.ZipFile(bad, "w") as package:
                if kind == "duplicate":
                    package.writestr("a.pxd2", b"bad");package.writestr("b.pxd2", b"bad")
                elif kind == "symlink":
                    member = zipfile.ZipInfo("capture.pxd2")
                    member.create_system = 3
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
                    package.writestr(member, "../source.pxd2")
                elif kind != "empty":package.writestr("../bad.pxd2" if kind == "traversal" else "capture.pxd2", b"bad")
        assert not workflow.open_file(bad)
        assert workflow.display_session is original and workflow.display_path == OVD
        assert workflow.recent_files == (str(OVD),)
        assert not any(event.kind == "display" for event in workflow.poll())
    finally:workflow.shutdown()


def test_archive_provenance_byte_exact_save_and_new_cleanup(tmp_path, archive):
    workflow = CaptureWorkflow(Preferences(tmp_path / "settings.json"), RecordingLibrary(tmp_path / "library"))
    try:
        assert workflow.open_file(archive)
        data = workflow.display_data_path
        assert data != archive and data.is_file()
        event = next(event for event in workflow.poll() if event.kind == "display")
        assert event.path == archive and event.data_path == data
        del event
        assert workflow.recent_files == (str(archive),)
        saved = tmp_path / "copied.pvd"
        assert workflow.save_capture(saved) == saved
        assert saved.read_bytes() == archive.read_bytes()
        assert not workflow.save_capture(tmp_path / "mislabelled.pxd2")
        workflow.accept_connections = False
        workflow.new();gc.collect()
        assert not data.exists()
        assert archive.exists() and saved.exists()
    finally:workflow.shutdown()


def test_drop_first_supported_local_uri_with_spaces(tmp_path):
    first = tmp_path / "capture with spaces.PVD"
    second = tmp_path / "second.pxd3"
    payload = "# URI list\r\nhttps://host/capture.ovd\r\n" + first.as_uri() + "\r\n" + second.as_uri()
    assert first_dropped_capture(payload) == first
    assert first_dropped_capture([str(tmp_path / "image.png"), str(second)]) == second
    assert first_dropped_capture("file://remote/capture.ovd") is None
    assert first_dropped_capture("file://[/capture.ovd") is None
    assert first_dropped_capture("file:///tmp/capture.ovd?download=1") is None
    assert first_dropped_capture("relative.ovd") is None
    assert first_dropped_capture("file:///tmp/capture%00.ovd") is None
    assert first_dropped_capture("file:///tmp/capture.zip") is None


def test_job_cancelled_before_adoption_and_opener_shutdown(tmp_path, archive):
    opener = CaptureOpener(tmp_path)
    try:
        first = opener.submit(archive)
        second = opener.submit(OVD)
        second.cancel()
        with pytest.raises(CancelledError):second.take_result()
        loaded = first.take_result()
        assert loaded.session.scenes()
        assert first.progress.fraction == 1
        assert not first.cancel()  # Transferred ownership belongs to the workflow.
        loaded.close()
        with pytest.raises(RuntimeError):first.take_result()
    finally:opener.close()
    with pytest.raises(RuntimeError):opener.submit(OVD)


def cancel_during_loading(tmp_path,path,interruptible):
    control = LoadControl()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(Session, str(path), str(tmp_path), control)
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            progress = control.progress()
            if progress.stage.startswith("Reading"):
                assert progress.interruptible==interruptible
                control.cancel()
                break
            time.sleep(.0001)
        assert control.cancelled, "Native load finished without exposing its parsing stage"
        with pytest.raises(RuntimeError, match="cancelled"):future.result(timeout=5)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("path,interruptible", [(LEGACY,False),
    (ROOT / "test/data/2026_04_01_15_21_33_0000019f4952a142.ovd",True)])
def test_native_cancellation_during_indexed_or_ovd_loading(tmp_path,path,interruptible):
    cancel_during_loading(tmp_path,path,interruptible)


def test_native_cancellation_during_raw_stream_loading(tmp_path,raw_streams):
    cancel_during_loading(tmp_path,raw_streams[1],True)


def test_native_pre_cancel_and_invalid_indexed_header(tmp_path):
    control = LoadControl();control.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):Session(str(OVD), str(tmp_path), control)
    for name, payload in (("short", struct.pack("<I", 907025328)),
                          ("version", struct.pack("<4I", 907025328, 0xFFFFFFFF, 0, 0)),
                          ("length", struct.pack("<4I", 907025328, 1, 0xFFFFFFFF, 0xFFFFFFFF))):
        path = tmp_path / (name + ".pxd3");path.write_bytes(payload)
        with pytest.raises(RuntimeError):load_capture(path)


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO capability required')
def test_non_regular_capture_is_rejected_without_blocking_on_fifo(tmp_path):
    fifo=tmp_path/"producer.ovd"
    os.mkfifo(fifo)
    with pytest.raises(ValueError,match="regular file"):
        load_capture(fifo)


def test_non_regular_directory_capture_is_rejected(tmp_path):
    directory = tmp_path / 'directory.ovd'
    directory.mkdir()
    with pytest.raises(ValueError, match='regular file'):
        load_capture(directory)
    assert directory.is_dir()


def test_archive_crc_and_space_failures_clean_partial_payloads(tmp_path,indexed,monkeypatch):
    package=tmp_path/"damaged.pvd"
    with zipfile.ZipFile(package,"w",zipfile.ZIP_STORED) as stream:
        stream.write(indexed,"capture.pxd2")
    data=bytearray(package.read_bytes())
    # Corrupt stored bytes, leaving ZIP metadata and CRC unchanged.
    name_length,extra_length=struct.unpack_from("<HH",data,26)
    data[30+name_length+extra_length+20]^=0x80
    package.write_bytes(data)
    with pytest.raises(ValueError,match="CRC"):
        load_capture(package,temporary_directory=tmp_path)
    assert not list(tmp_path.glob("pvd-open-*"))
    import ovd_ui.ingestion as ingestion
    from types import SimpleNamespace
    monkeypatch.setattr(ingestion.shutil,"disk_usage",lambda _:SimpleNamespace(free=1))
    with pytest.raises(OSError,match="free space"):
        load_capture(package,temporary_directory=tmp_path)
    assert not list(tmp_path.glob("pvd-open-*"))


@pytest.mark.parametrize("kind",["timing","profile","memory","errors","export"])
def test_queued_native_readers_retain_archives_after_display_replacement(archive,tmp_path,kind):
    from ovd_ui.analysis import AnalysisTasks
    from ovd_ui.profiling import DiagnosticsTasks,DiagnosticsRequest
    from ovd_ui.error_stream import ErrorTasks,ErrorRequest
    from ovd_ui.exports import ExportRequest,export_capture
    loaded=load_capture(archive)
    source=loaded.data_path
    scene=loaded.session.scenes()[0].id
    tasks=AnalysisTasks() if kind=="timing" else ErrorTasks() if kind=="errors" else DiagnosticsTasks()
    gate=Event()
    tasks.executor.submit(gate.wait)
    try:
        if kind=="timing":future=tasks.submit_frame_timing(source,scene,1,2).future
        elif kind=="profile":future=tasks.submit_profile(DiagnosticsRequest(source,scene,1)).future
        elif kind=="memory":future=tasks.submit_memory(DiagnosticsRequest(source,scene,1,1_000_000)).future
        elif kind=="errors":future=tasks.submit_errors(ErrorRequest(source,scene,1)).future
        else:
            request=ExportRequest(source,tmp_path/"exported.pxd2","clip",scene,1,1,2)
            future=submit_capture(tasks.executor,source,export_capture,request)
        loaded.close();gc.collect()
        assert source.exists(),"Queued native work lost its private extracted payload"
        gate.set()
        result=future.result(timeout=10)
        if kind=="timing":assert result.frames==(1,)
        elif kind=="export":assert result[0].is_file() and result[1]>0
        else:assert result.supported
    finally:
        gate.set();tasks.close()
    gc.collect()
    assert not source.exists()
