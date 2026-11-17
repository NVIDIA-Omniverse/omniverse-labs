# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native Session controls; independently encoded producer/wire expectations."""
from contextlib import contextmanager
from pathlib import Path
import socket
import struct
import time

import pytest
from _pvd_native import Session


def wait_for(check, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.005)
    pytest.fail("Capture control publication timed out")


def field(context, row, attribute, value):
    return struct.pack("<BQQBII", 6, context, row, 1, attribute, len(value)) + value


def create(context, schema, row):
    return struct.pack("<BQIQH", 9, context, schema, row, 0)


def schema(handle, name, fields):
    name = name.encode()
    result = struct.pack("<BIIH", 1, handle, 0, len(name)) + name
    for attribute, name, kind in fields:
        name = name.encode()
        result += struct.pack("<BIIHIH", 3, handle, attribute, kind, 1, len(name)) + name
    return result


META = [(i + 1, name, kind) for i, (name, kind) in enumerate([
    ("physxVersionMajor", 6), ("physxVersionMinor", 6), ("physxVersionBugfix", 6),
    ("ovdIntegrationVersionMajor", 6), ("ovdIntegrationVersionMinor", 6),
    ("recordSecondaryData", 4), ("captureAllObjects", 4), ("captureRequestId", 7),
    ("captureRequestStatus", 6), ("captureEpoch", 7), ("captureCatalogGeneration", 7)])]
CATALOG = [(20 + i, name, kind) for i, (name, kind) in enumerate([
    ("targetHandle", 7), ("typeName", 10), ("name", 10),
    ("ownerHandle", 7), ("sceneHandle", 7), ("canSelect", 4)])]
SCENE, ACTOR, CONTEXT = (1 << 53) + 17, (1 << 63) + 19, (1 << 63) + 21


def metadata(row=100):
    return (create(1, 1, row) + field(1, row, 4, struct.pack("<I", 3))
            + field(1, row, 6, b"\0") + field(1, row, 7, b"\1"))


def catalog(generation=1):
    result = b""
    for row, target, kind, name, selectable in [
            (200, SCENE, "PxScene", "scene", 1),
            (201, ACTOR, "PxRigidDynamic", "actor", 1),
            (202, CONTEXT, "PxShape", "shape", 0)]:
        result += create(2, 2, row)
        for attr, value in zip(range(20, 26), [struct.pack("<Q", target), kind.encode(),
                name.encode(), struct.pack("<Q", 0), struct.pack("<Q", SCENE), bytes([selectable])]):
            result += field(2, row, attr, value)
    return result + field(1, 100, 11, struct.pack("<Q", generation))


def ack(request, status=1, epoch=1, row=100):
    return (field(1, row, 8, struct.pack("<Q", request))
            + field(1, row, 9, struct.pack("<I", status))
            + field(1, row, 10, struct.pack("<Q", epoch)))


def receive(connection, count):
    data = b""
    while len(data) < count:
        data += connection.recv(count - len(data))
    return data


def no_reverse_bytes(connection):
    connection.settimeout(.08)
    with pytest.raises(socket.timeout):
        connection.recv(1)
    connection.settimeout(2)


@contextmanager
def live(capabilities=12, with_catalog=True):
    assert hasattr(Session, "capture_control_info"), "native capture control API is missing"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    session = Session.listen(port)
    connection = None
    try:
        def connect():
            nonlocal connection
            connection = socket.socket()
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)
            connection.settimeout(2)
            try:
                connection.connect(("127.0.0.1", port))
                return True
            except ConnectionRefusedError:
                connection.close()
                connection = None
                return False
        wait_for(connect)
        connection.sendall(struct.pack("<IHHIII", 0x4f56444c, 1, capabilities, 0, 4, 0)
                           + schema(1, "PxOmniPvdMetaData", META)
                           + schema(2, "PxOmniPvdCaptureEntry", CATALOG) + metadata()
                           + (catalog() if with_catalog else b""))
        wait_for(lambda: session.capture_control_info()["policy_known"])
        if with_catalog:
            wait_for(lambda: session.capture_control_info()["catalog_ready"])
        yield session, connection
    finally:
        if connection is not None:
            connection.close()
        session.stop_live()


def test_no_frame_discovery_is_owned_paged_and_sends_nothing():
    with live() as (session, connection):
        info = session.capture_control_info()
        assert info["capabilities"] == 12 and info["generation"] == 1
        assert info["state"] == "idle" and info["issued_high_water"] == 0
        assert not session.refresh_live().ready
        page = session.capture_catalog(1, limit=1, name_prefix="act")
        assert page["state"] == "ready" and page["total"] == 1
        assert page["entries"][0]["target_handle"] == ACTOR
        assert page["entries"][0]["can_select"]
        assert session.capture_catalog(2)["state"] == "stale"
        with pytest.raises(ValueError):
            session.capture_catalog(1, limit=1001)
        no_reverse_bytes(connection)
        session.stop_live()
        assert page["entries"][0]["name"] == "actor"


def test_catalog_rows_do_not_become_reopened_evidence_objects(tmp_path):
    source = Path(__file__).resolve().parents[2] / "test/data/test_scene.ovd"
    recorded = source.read_bytes()
    assert recorded[:12] == struct.pack("<III", 0, 4, 0)
    row = (1 << 64) - 2
    catalog_schema = [(6001 + index, name, kind)
                      for index, (_, name, kind) in enumerate(CATALOG)]
    discovery = schema(6000, "PxOmniPvdCaptureEntry", catalog_schema) + create(2, 6000, row)
    values = [struct.pack("<Q", 1342902134704), b"PxRigidDynamic", b"actor",
              struct.pack("<Q", 0), struct.pack("<Q", 1342900227584), b"\1"]
    for index, value in enumerate(values):
        discovery += field(2, row, 6001 + index, value)
    capture = tmp_path / "catalog-and-evidence.ovd"
    capture.write_bytes(recorded[:12] + discovery + recorded[12:])
    session = Session(str(capture), str(tmp_path / "opened"))
    page = session.evidence_metadata(0, 512)
    assert page.next_offset is None
    assert all(entry.class_name != "PxOmniPvdCaptureEntry" for entry in page.rows)
    assert any(entry.api_handle == 1342902134704 for entry in page.rows)
    scope, = page.domain.scopes
    assert (scope.segment, scope.context, scope.first, scope.last, scope.count) == (
        0, 1342900227584, 1, 3, 3)
    assert scope.scene_mapped and page.domain.frame_scopes_complete
    assert session.presentation(session.scenes()[0].id, 2, include_evidence=True).ready


@pytest.mark.parametrize("status,state", [(1, "applied"), (2, "invalid"), (3, "busy"), (4, "denied")])
def test_selection_union_correlates_ack_and_retains_actual_policy(status, state):
    with live() as (session, connection):
        request = session.request_capture([SCENE], [ACTOR], True, False, 1)
        assert request == 1
        assert receive(connection, 44) == struct.pack("<IIQIIIQQ", 0x3343564f, 44, 1, 1, 1, 1, SCENE, ACTOR)
        # Receiving bytes does not synchronize the worker's later state store.
        wait_for(lambda: session.capture_control_info()["state"] == "awaiting_ack")
        with pytest.raises(RuntimeError):
            session.request_capture_catalog(supersede_unknown=True)
        connection.sendall(ack(1, status, 7))
        wait_for(lambda: session.capture_control_info()["state"] == state)
        info = session.capture_control_info()
        assert info["capture_epoch"] == info["ack_epoch"] == 7
        assert info["actual_all"] and not info["actual_secondary"]
        assert not session.refresh_live().ready
        assert session.request_capture_catalog() == 2
        assert receive(connection, 28) == struct.pack("<IIQIII", 0x3344564f, 28, 2, 0, 0, 0)


@pytest.mark.parametrize("capabilities", [0, 1, 2, 3, 4, 8, 0x8000])
def test_capabilities_are_independent_and_unsupported_actions_send_nothing(capabilities):
    with live(capabilities) as (session, connection):
        if capabilities & 4:
            with pytest.raises((ValueError, RuntimeError)):
                session.request_capture([SCENE], [], False, False, 1)
            assert session.request_capture([], [], False, True, 0) == 1
            assert receive(connection, 28) == struct.pack("<IIQIII", 0x3343564f, 28, 1, 0, 0, 0)
        else:
            with pytest.raises((ValueError, RuntimeError)):
                session.request_capture([], [], False, True, 0)
        if capabilities & 8:
            assert session.request_capture_catalog() == 1
            assert receive(connection, 28)[:4] == b"OVD3"
        else:
            with pytest.raises((ValueError, RuntimeError)):
                session.request_capture_catalog()
        no_reverse_bytes(connection)


@pytest.mark.parametrize("scenes,objects,generation", [([], [], 1), ([ACTOR], [], 1),
    ([], [SCENE], 1), ([], [CONTEXT], 1), ([], [ACTOR], 2), ([], [555], 1),
    ([], [-1], 1), ([], [1 << 64], 1), ([], [float(ACTOR)], 1), ([], [True], 1)])
def test_native_selection_guards_prevent_any_reverse_bytes(scenes, objects, generation):
    with live() as (session, connection):
        with pytest.raises((ValueError, RuntimeError, TypeError, OverflowError)):
            session.request_capture(scenes, objects, False, False, generation)
        assert session.capture_control_info()["issued_high_water"] == 0
        no_reverse_bytes(connection)


def test_file_sessions_cannot_control_and_new_connection_restarts_ids(tmp_path):
    source = Path(__file__).resolve().parents[2] / "test/data/test_scene.ovd"
    file = Session(str(source), str(tmp_path))
    assert hasattr(file, "capture_control_info"), "native capture control API is missing"
    assert file.capture_control_info()["capabilities"] == 0
    with pytest.raises(RuntimeError):
        file.request_capture_catalog()
    for _ in range(2):
        with live() as (session, connection):
            assert session.request_capture_catalog() == 1
            receive(connection, 28)


def test_unknown_explicit_supersession_and_late_ack_do_not_settle_new_request():
    with live() as (session, connection):
        assert session.request_capture([], [], True, True, 0) == 1
        receive(connection, 28)
        started = time.monotonic()
        wait_for(lambda: session.capture_control_info()["state"] == "unknown", seconds=12)
        assert time.monotonic() - started >= 9.9
        with pytest.raises(RuntimeError):
            session.request_capture_catalog()
        no_reverse_bytes(connection)
        assert session.request_capture_catalog(supersede_unknown=True) == 2
        assert receive(connection, 28) == struct.pack("<IIQIII", 0x3344564f, 28, 2, 0, 0, 0)
        connection.sendall(ack(1, 1, 9))
        wait_for(lambda: session.capture_control_info()["capture_epoch"] == 9)
        info = session.capture_control_info()
        assert info["state"] == "awaiting_ack" and info["request_id"] == 2
        assert info["ack_classification"] == "late" and info["request"]["catalog_only"]
        assert info["results"][0]["state"] == "superseded"
        connection.sendall(ack(2, 1, 9))
        wait_for(lambda: session.capture_control_info()["state"] == "applied")
        assert session.capture_control_info()["capture_epoch"] == 9
        assert not session.refresh_live().ready


def test_late_ack_without_supersession_settles_the_same_unknown_request():
    with live() as (session, connection):
        assert session.request_capture_catalog() == 1
        receive(connection, 28)
        wait_for(lambda: session.capture_control_info()["state"] == "unknown", seconds=12)
        no_reverse_bytes(connection)
        connection.sendall(ack(1, 1, 0))
        wait_for(lambda: session.capture_control_info()["state"] == "applied")
        info = session.capture_control_info()
        assert info["request_id"] == info["issued_high_water"] == 1
        assert info["ack_classification"] == "current" and len(info["results"]) == 1
        assert session.request_capture_catalog() == 2
        receive(connection, 28)


def test_catalog_mutation_revokes_selection_and_metadata_recreation_keeps_connection_ids():
    with live() as (session, connection):
        old_page = session.capture_catalog(1)
        connection.sendall(struct.pack("<BQQ", 10, 1, 100) + metadata())
        wait_for(lambda: session.capture_control_info()["metadata_boundary_serial"] == 1)
        assert session.capture_control_info()["generation"] == 1
        assert session.request_capture([SCENE], [], False, False, 1) == 1
        receive(connection, 36)
        connection.sendall(ack(1, 2, 0) + struct.pack("<BQQ", 10, 2, 200))
        wait_for(lambda: not session.capture_control_info()["catalog_ready"])
        assert session.capture_catalog(1)["state"] == "busy"
        with pytest.raises(RuntimeError):
            session.request_capture([], [ACTOR], False, False, 1)
        assert old_page["entries"][0]["target_handle"] == ACTOR
        connection.sendall(struct.pack("<BQQBQQ", 10, 2, 201, 10, 2, 202) + catalog(2))
        wait_for(lambda: session.capture_control_info()["generation"] == 2)
        with pytest.raises(RuntimeError):
            session.request_capture([], [ACTOR], False, False, 1)
        assert session.request_capture([], [ACTOR], False, False, 2) == 2
        receive(connection, 36)


def test_real_session_result_ring_is_bounded_and_evicted_ack_cannot_settle_new_request():
    with live() as (session, connection):
        for request in range(1, 67):
            assert session.request_capture_catalog() == request
            receive(connection, 28)
            connection.sendall(ack(request, 1, 0))
            wait_for(lambda: session.capture_control_info()["state"] == "applied")
        info = session.capture_control_info()
        assert len(info["results"]) == 64
        assert info["results"][0]["id"] == 3 and info["issued_high_water"] == 66
        assert session.request_capture_catalog() == 67
        receive(connection, 28)
        connection.sendall(ack(1, 1, 12))
        wait_for(lambda: session.capture_control_info()["capture_epoch"] == 12)
        info = session.capture_control_info()
        assert info["ack_classification"] == "late" and info["state"] == "awaiting_ack"


def test_partial_native_send_cannot_be_superseded_and_stop_does_not_wait_for_ack():
    with live() as (session, connection):
        assert session.request_capture([], [ACTOR] * 65536, False, False, 1) == 1
        wait_for(lambda: session.capture_control_info()["bytes_sent"] > 0)
        info = session.capture_control_info()
        assert info["state"] == "sending" and info["bytes_sent"] < 524316
        with pytest.raises(RuntimeError):
            session.request_capture_catalog(supersede_unknown=True)
        before = info["forward_bytes"]
        connection.sendall(field(1, 100, 6, b"\1"))
        wait_for(lambda: session.capture_control_info()["actual_secondary"])
        assert session.capture_control_info()["forward_bytes"] > before
        started = time.monotonic()
        session.stop_live()
        assert time.monotonic() - started < 2
        assert not session.capture_control_info()["can_apply"]
        with pytest.raises(RuntimeError):
            session.request_capture_catalog(supersede_unknown=True)
