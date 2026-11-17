# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema 4.0 captures exercise the same native Session used by OvUI."""
import socket
import struct
import subprocess
import time

import numpy as np
import pytest
from _pvd_native import Session

from native_fixtures import executable
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel


def write_capture(tmp_path, mode):
    path = tmp_path / (mode + ".ovd")
    writer = executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")
    subprocess.run([str(writer), str(path), "--joint-inspector-" + mode], check=True)
    return path


def check_joint(session, *, core, second=False):
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    scene = next(s for s in session.scenes() if s.alive)
    joint, = [i for i in adapter.instances.values()
              if i.alive and adapter.class_name(i.id) == "PxD6Joint"
              and i.scene_id == scene.instance_id]
    assert model.canonical_item(joint.id) is not None
    assert joint.scene_joint
    common = adapter.references(joint.id, "constraint")[0] if core else joint.id
    assert adapter.instances[common].scene_id == scene.instance_id
    props = {p.name: p for p in adapter.properties(common)}
    assert adapter.references(common, "actor0") == [-1]
    actor, = adapter.references(common, "actor1")
    assert adapter.name(actor) == ("Second box" if second else "First box")
    assert adapter.value(common, props["breakForce"]) == ("250" if second else "125")
    assert adapter.value(common, props["breakTorque"]) == "75"
    assert struct.unpack("<H", session.read(common, props["constraintFlags"].id)) == (1,)
    overlay = session.overlay_snapshot(16)
    positions = overlay.positions[overlay.object_ids == joint.id]
    assert len(positions) == 6
    np.testing.assert_allclose(positions[:3, 0], [[1, 2, 3]] * 3)
    np.testing.assert_allclose(positions[3:, 0], [[44 if second else 14, 25, 36]] * 3)
    assert set(overlay.related_ids[overlay.object_ids == joint.id, 1]) == {actor}


@pytest.mark.parametrize("mode", ["v3", "v4", "v4-membership-first"])
def test_current_and_legacy_capture_properties_membership_and_joint_overlay(tmp_path, mode):
    session = Session(str(write_capture(tmp_path, mode)), str(tmp_path))
    session.seek(session.scenes()[0].id, 1)
    check_joint(session, core=mode.startswith("v4"))


def test_future_integration_version_is_still_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="unsupported OVD integration version"):
        Session(str(write_capture(tmp_path, "v5")), str(tmp_path))


@pytest.mark.parametrize("mode", ["v4-missing-endpoints", "v4-unknown-endpoint", "v3-unknown-endpoint"])
def test_unrecorded_core_endpoints_do_not_invent_world_joint_frames(tmp_path, mode):
    session = Session(str(write_capture(tmp_path, mode)), str(tmp_path))
    session.seek(session.scenes()[0].id, 1)
    assert len(session.overlay_snapshot(16).positions) == 0


def test_core_constraint_remains_inspectable_without_a_live_extension_joint(tmp_path):
    session = Session(str(write_capture(tmp_path, "v4-core-only")), str(tmp_path))
    scene = session.scenes()[0]
    session.seek(scene.id, 1)
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    constraint, = [i for i in adapter.instances.values()
                   if i.alive and adapter.class_name(i.id) == "PxConstraint"]
    assert constraint.scene_id == scene.instance_id
    assert model.canonical_item(constraint.id) is not None
    assert len(session.overlay_snapshot(16).positions) == 0


def test_core_constraint_lifetime_survives_rewind(tmp_path):
    session = Session(str(write_capture(tmp_path, "v4-lifetime")), str(tmp_path))
    scene = session.scenes()[0]
    classes = {c.id: c.name for c in session.classes()}
    for frame, expected in ((2, 0), (3, 1), (4, 1), (5, 0), (6, 0), (4, 1)):
        session.seek(scene.id, frame)
        for name in ("PxConstraint", "PxD6Joint"):
            objects = [i for i in session.instances() if i.alive and classes[i.class_id] == name]
            assert len(objects) == expected
            assert all(i.scene_id == scene.instance_id for i in objects)


def test_core_constraint_reused_handles_stay_in_their_segment(tmp_path):
    session = Session(str(write_capture(tmp_path, "v4-segments")), str(tmp_path))
    adapter = PvdDataAdapter(session)
    joints = [i for i in adapter.instances.values() if adapter.class_name(i.id) == "PxD6Joint"]
    assert len(joints) == 2
    for joint in joints:
        constraint, = adapter.references(joint.id, "constraint")
        actor, = adapter.references(constraint, "actor1")
        first = adapter.name(joint.id) == "First world joint"
        assert adapter.name(actor) == ("First box" if first else "Second box")
        assert adapter.instances[constraint].scene_id == joint.scene_id
        assert adapter.instances[actor].scene_id == joint.scene_id


def test_schema4_live_stream_saves_and_reopens_without_conversion(tmp_path):
    path = write_capture(tmp_path, "v4")
    payload = path.read_bytes()
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    session = Session.listen(port)
    deadline = time.monotonic() + 10
    try:
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
            session.refresh()
            time.sleep(.01)
        assert session.info().ended
        session.refresh()
        assert session.info().completed_serial == 2
        saved = tmp_path / "saved.ovd"
        assert session.save(str(saved)) == len(payload)
        assert saved.read_bytes() == payload
    finally:
        session.stop_live()
    reopened = Session(str(saved), str(tmp_path))
    reopened.seek(reopened.scenes()[0].id, 1)
    check_joint(reopened, core=True)
