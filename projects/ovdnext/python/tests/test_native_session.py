# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual native engine boundary with repository captures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import struct
import time

import pytest
from _pvd_native import Session

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test/data/test_scene.ovd"


def open_session(tmp_path):
    session = Session(str(FIXTURE), str(tmp_path))
    session.seek(session.scenes()[0].id, 1)
    return session


def test_real_scene_typed_properties_and_references(tmp_path):
    session = open_session(tmp_path)
    assert session.info().ovd
    assert session.info().instance_count == 8
    classes = {c.id: c.name for c in session.classes()}
    instances = {classes[i.class_id]: i for i in session.instances()}
    actor = instances["PxRigidDynamic"]
    properties = {p.name: p for p in session.properties(actor.class_id)}
    assert session.string_value(actor.id, properties["name"].id) == "sphere"
    assert session.references(actor.id, properties["shapes"].id) == [instances["PxShape"].id]
    sphere = instances["PxSphereGeometry"]
    radius = next(p for p in session.properties(sphere.class_id) if p.name == "radius")
    assert struct.unpack("<f", session.read(sphere.id, radius.id)) == (10.0,)


def test_invalid_ids_limits_and_snapshot_lifetime(tmp_path):
    session = open_session(tmp_path)
    with pytest.raises(IndexError):
        session.read(999999, 1)
    with pytest.raises(ValueError):
        session.instances(0, 10001)
    with pytest.raises(IndexError):
        session.seek(999999, 0)
    with pytest.raises(IndexError):
        session.seek(session.scenes()[0].id, 999999)
    classes = {c.id: c.name for c in session.classes()}
    records = session.instances()
    actor = next(i for i in records if classes[i.class_id] == "PxRigidDynamic")
    session.seek(session.scenes()[0].id, 3)
    current = {i.id: i for i in session.instances()}
    assert actor.alive  # Value objects remain immutable after a seek.
    assert not current[actor.id].alive


def test_failed_load_and_parallel_session_lifetimes(tmp_path):
    with pytest.raises(RuntimeError):
        Session(str(tmp_path / "missing.ovd"), str(tmp_path))
    invalid = tmp_path / "bad.ovd"
    invalid.write_bytes(b"not an OVD stream")
    with pytest.raises(RuntimeError):
        Session(str(invalid), str(tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: open_session(tmp_path).info().instance_count, range(16)))
    assert results == [8] * 16


def test_legacy_pxd2_uses_same_engine(tmp_path):
    session = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    assert not session.info().ovd
    assert session.scenes()
    session.seek(session.scenes()[0].id, 1)
    assert session.info().instance_count > 0


def test_native_live_recording_and_exclusive_create(tmp_path):
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    recording = tmp_path / "live.ovd"
    session = Session.listen(port, str(recording))
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                producer = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        with producer:
            producer.sendall(struct.pack("<IHH", 0x4F56444C, 1, 0) + FIXTURE.read_bytes())
        while not session.info().ended and time.monotonic() < deadline:
            time.sleep(0.01)
        session.refresh()
        assert session.info().completed_serial > 0
    finally:
        session.stop_live()
    assert recording.read_bytes() == FIXTURE.read_bytes()
    with pytest.raises(RuntimeError):
        Session.listen(port, str(recording))
    assert recording.read_bytes() == FIXTURE.read_bytes()


def test_native_live_in_memory_recording_can_be_saved_after_disconnect(tmp_path):
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    output = tmp_path / "memory-copy.ovd"
    output.write_bytes(b"sentinel")
    session = Session.listen(port)
    deadline = time.monotonic() + 5
    try:
        while True:
            try:
                producer = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        with producer:
            assert not session.export_capabilities().save
            with pytest.raises(ValueError):
                session.save(str(output))
            assert output.read_bytes() == b"sentinel"
            producer.sendall(struct.pack("<IHH", 0x4F56444C, 1, 0) + FIXTURE.read_bytes())
        while not session.info().ended and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.info().ended, "Live recording did not complete after natural EOF"
        assert session.info().recording_size == len(FIXTURE.read_bytes())
        assert session.export_capabilities().save
        with pytest.raises(RuntimeError):
            session.save(str(output))
        assert output.read_bytes() == b"sentinel"
        output.unlink()
        assert session.save(str(output)) == len(FIXTURE.read_bytes())
        assert output.read_bytes() == FIXTURE.read_bytes()
        assert Session(str(output), str(tmp_path)).info().instance_count == 8
    finally:
        session.stop_live()
