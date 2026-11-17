# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real capture publication must preserve source/winner bytes on I/O failures."""
import pytest
from ovd_ui import recordings


def test_competing_destination_survives_copy(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.ovd", tmp_path / "saved.ovd"
    source.write_bytes(b"source capture")
    sync = recordings.os.fsync
    def competing_writer(descriptor):
        sync(descriptor)
        destination.write_bytes(b"other writer wins")
    with monkeypatch.context() as patch:
        patch.setattr(recordings.os, "fsync", competing_writer)
        with pytest.raises(FileExistsError):
            recordings.copy_capture(source, destination)
    assert source.read_bytes() == b"source capture"
    assert destination.read_bytes() == b"other writer wins"
    assert not list(tmp_path.glob(".ovd-save-*"))
    retry = tmp_path / "retry.ovd"
    assert recordings.copy_capture(source, retry) == retry
    assert retry.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("failure", ["read", "write", "fsync"])
@pytest.mark.parametrize("existing", [False, True])
def test_read_write_fsync_failures_preserve_destinations(tmp_path, monkeypatch, failure, existing):
    source, destination = tmp_path / "source.ovd", tmp_path / "saved.ovd"
    payload = b"recorded bytes" * 1000
    source.write_bytes(payload)
    if existing:
        destination.write_bytes(b"previous recording")
    fdopen = recordings.os.fdopen
    class FailingStream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def __getattr__(self, name): return getattr(self.stream, name)
        def read(self, size): raise OSError("injected read failure")
        def write(self, block):
            self.stream.write(block[:3])
            raise OSError("injected write failure")
    def open_with_failure(descriptor, mode):
        stream = fdopen(descriptor, mode)
        return FailingStream(stream) if (failure, mode) in {("read", "rb"), ("write", "wb")} else stream
    def fail_sync(_descriptor): raise OSError("injected fsync failure")
    with monkeypatch.context() as patch:
        patch.setattr(recordings.os, "fdopen", open_with_failure)
        if failure == "fsync": patch.setattr(recordings.os, "fsync", fail_sync)
        with pytest.raises(OSError, match="injected"):
            recordings.copy_capture(source, destination, overwrite=existing)
    assert source.read_bytes() == payload
    assert destination.read_bytes() == b"previous recording" if existing else not destination.exists()
    assert not list(tmp_path.glob(".ovd-save-*"))
    recordings.copy_capture(source, destination, overwrite=existing)
    assert destination.read_bytes() == payload


@pytest.mark.parametrize("existing", [False, True])
def test_publish_failure_cleans_temporary(tmp_path, monkeypatch, existing):
    source, destination = tmp_path / "source.ovd", tmp_path / "saved.ovd"
    source.write_bytes(b"source")
    if existing: destination.write_bytes(b"previous")
    def fail_publish(*args, **kwargs): raise OSError("injected publication failure")
    with monkeypatch.context() as patch:
        patch.setattr(recordings.os, "replace" if existing else "link", fail_publish)
        with pytest.raises(OSError, match="publication"):
            recordings.copy_capture(source, destination, overwrite=existing)
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"previous" if existing else not destination.exists()
    assert not list(tmp_path.glob(".ovd-save-*"))
    recordings.copy_capture(source, destination, overwrite=existing)
    assert destination.read_bytes() == b"source"
