# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recording selection model over the real native Session, without GUI widgets."""
import importlib
import struct

import pytest
from test_native_capture_control import (live, wait_for, no_reverse_bytes, receive,
    ack, catalog, metadata, field, create, SCENE, ACTOR, CONTEXT)


def model_for(session):
    assert importlib.util.find_spec("ovd_ui.capture_control"), "Capture selection model is missing"
    model = importlib.import_module("ovd_ui.capture_control").CaptureSelectionModel()
    model.bind(session)
    model.poll()
    return model


def test_model_never_sends_on_bind_poll_or_page_and_requires_explicit_nonempty_selection():
    with live() as (session, connection):
        model = model_for(session)
        assert not model.all and model.generation == 1
        assert len(model.page["entries"]) == 3
        with pytest.raises(ValueError):
            model.apply()
        with pytest.raises(ValueError):
            model.select(CONTEXT, True)
        with pytest.raises(ValueError):
            model.select(1234, True)
        model.load_page(name_prefix="act")
        assert model.page["total"] == 1 and model.page["entries"][0]["target_handle"] == ACTOR
        model.select(ACTOR, True)
        model.secondary = True
        no_reverse_bytes(connection)
        assert model.apply() == 1
        assert receive(connection, 36) == struct.pack("<IIQIIIQ", 0x3343564f, 36, 1, 1, 0, 1, ACTOR)
        connection.sendall(ack(1, 1, 1))
        wait_for(lambda: model.poll()["state"] == "applied")
        assert "waiting" in model.status.lower()


def test_model_preserves_checks_on_metadata_recreation_but_not_catalog_generation_change():
    with live() as (session, connection):
        model = model_for(session)
        model.select(SCENE, True)
        connection.sendall(struct.pack("<BQQ", 10, 1, 100) + metadata())
        wait_for(lambda: model.poll()["metadata_boundary_serial"] == 1)
        assert model.selected_count == 1
        connection.sendall(b"".join(struct.pack("<BQQ", 10, 2, row) for row in (200, 201, 202)) + catalog(2))
        wait_for(lambda: model.poll()["generation"] == 2)
        assert model.selected_count == 0 and model.generation == 2
        with pytest.raises(ValueError):
            model.apply()
        no_reverse_bytes(connection)


def test_explicit_all_and_refresh_are_independent_and_binding_clears_old_handles():
    with live(4) as (session, connection):
        model = model_for(session)
        with pytest.raises(RuntimeError):
            model.refresh_catalog()
        model.all = True
        assert model.apply() == 1
        assert receive(connection, 28)[:4] == b"OVC3"
        model.bind(None)
        assert model.selected_count == 0 and not model.page["entries"]
        with pytest.raises(RuntimeError):
            model.apply()
    with live(8) as (session, connection):
        model.bind(session)
        model.poll()
        assert not model.all and model.selected_count == 0
        with pytest.raises(RuntimeError):
            model.apply()
        assert model.refresh_catalog() == 1
        assert receive(connection, 28)[:4] == b"OVD3"


def test_model_keeps_one_page_and_rejects_selection_65537_without_sending():
    with live(with_catalog=False) as (session, connection):
        rows = bytearray()
        for index in range(65537):
            rows += create(2, 2, 1000 + index)
            for attribute, value in zip(range(20, 26), [struct.pack("<Q", ACTOR + index),
                    b"PxRigidDynamic", f"actor-{index:05d}".encode(), struct.pack("<Q", 0),
                    struct.pack("<Q", SCENE), b"\1"]):
                rows += field(2, 1000 + index, attribute, value)
        # This 14 MiB catalog uses the publication budget, not the small-control timeout.
        previous_timeout = connection.gettimeout()
        try:
            connection.settimeout(15)
            connection.sendall(rows + field(1, 100, 11, struct.pack("<Q", 1)))
        finally:
            connection.settimeout(previous_timeout)
        wait_for(lambda: session.capture_control_info()["catalog_ready"], seconds=15)
        model = model_for(session)
        for offset in range(0, 65536, 128):
            page = model.load_page(offset)
            assert len(model.page["entries"]) == 128 and page["total"] == 65537
            for row in page["entries"]:
                model.select(row["target_handle"], True)
        assert model.selected_count == 65536
        model.load_page(65536)
        with pytest.raises(ValueError, match="65536"):
            model.select(model.page["entries"][0]["target_handle"], True)
        assert len(model.page["entries"]) == 1 and model.selected_count == 65536
        no_reverse_bytes(connection)
