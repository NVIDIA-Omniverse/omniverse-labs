# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UTF-8 input, private backing and live recording paths share one contract."""
import gc
from pathlib import Path
import socket
import struct
import subprocess
import time

import pytest
from _pvd_native import Session
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["ovd", "indexed", "raw"])
@pytest.mark.parametrize("unicode_source", [False, True])
def test_unicode_input_and_independent_backing_roundtrip_without_leaks(tmp_path, kind, unicode_source):
    inputs = tmp_path / ("J\u00f6rg-\u4e2d-\U0001f30d" if unicode_source else "inputs")
    backing = tmp_path / "temp-\u00dc-\u65e5-\U0001f680"
    inputs.mkdir(); backing.mkdir()
    name = "capture-\u4e2d-\U0001f30d" if unicode_source else "capture"
    path = inputs / (name + (".ovd" if kind == "ovd" else ".pxd2"))
    if kind == "raw":
        generated = tmp_path / "generated.pxd2"
        subprocess.run([str(executable("pvd_ingestion_fixture", "PVD_INGESTION_FIXTURE")),
                        str(generated), "3"], check=True)
        path.write_bytes(generated.read_bytes())
    else:
        path.write_bytes((ROOT / "test/data" / ("test_scene.ovd" if kind == "ovd" else "cloth_physx34.pxd2")).read_bytes())
    before = path.read_bytes()
    source = Session(str(path), str(backing))
    scene = source.scenes()[0]
    assert scene.frame_count > 2
    output = inputs / ("saved-\u65e5\U0001f680" + path.suffix)
    if kind == "ovd":
        assert source.info().instance_count == 8
        source.save(str(output))
        assert output.read_bytes() == before
    else:
        baseline = set(backing.iterdir())
        with pytest.raises(RuntimeError):
            source.save_clip(str(inputs / "missing" / "clip.pxd2"), scene.id, 1, 3)
        assert set(backing.iterdir()) == baseline
        source.save_clip(str(output), scene.id, 1, 3)
        assert set(backing.iterdir()) == baseline
    reopened = Session(str(output), str(backing))
    assert reopened.scenes()[0].frame_count == (scene.frame_count if kind == "ovd" else 2)
    assert path.read_bytes() == before
    del reopened, source
    gc.collect()
    assert list(backing.iterdir()) == []
    assert not list(inputs.glob("*.ovui-*"))


def test_unicode_live_recording_is_exclusive_and_reopens_exact_bytes(tmp_path):
    directory = tmp_path / "live-\u00dc-\u4e2d-\U0001f30d"
    directory.mkdir()
    path = directory / "recording-\u65e5\U0001f680.ovd"
    payload = (ROOT / "test/data/test_scene.ovd").read_bytes()
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    session = Session.listen(port, str(path))
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                producer = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        with producer:
            producer.sendall(struct.pack("<IHH", 0x4F56444C, 1, 0) + payload)
        while not session.info().ended and time.monotonic() < deadline:
            time.sleep(.01)
        assert session.info().ended
    finally:
        session.stop_live()
    assert path.read_bytes() == payload
    assert Session(str(path), str(directory)).info().instance_count == 8
    with pytest.raises(RuntimeError):
        Session.listen(port, str(path))
    assert path.read_bytes() == payload
    assert list(directory.iterdir()) == [path]
