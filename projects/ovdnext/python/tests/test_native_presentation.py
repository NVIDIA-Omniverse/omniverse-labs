# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned native views keep tree, selection and geometry on one publication."""
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path
import gc
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from _pvd_native import Session, SnapshotDataUnavailable
from ovd_ui.adapters import PvdDataAdapter

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = (ROOT / "test/data/test_scene.ovd").read_bytes()
HANDSHAKE = struct.pack("<IHH", 0x4F56444C, 1, 0)
FRAME2_STOP = struct.pack("<BQQ", 12, 0x138AB174E00, 2)
FRAME2_END = PAYLOAD.index(FRAME2_STOP) + len(FRAME2_STOP)
FRAME3_STOP = struct.pack("<BQQ", 12, 0x138AB174E00, 3)
FRAME3_END = PAYLOAD.index(FRAME3_STOP) + len(FRAME3_STOP)


def open_capture(tmp_path, name="test_scene.ovd"):
    return Session(str(ROOT / "test/data" / name), str(tmp_path))


def instance_signature(view):
    return [(i.id, i.class_id, i.scene_id, i.physics_id, i.alive)
            for i in view.instances()]


def owned_signature(view):
    adapter = PvdDataAdapter(view)
    names = [(i, adapter.name(i)) for i in adapter.instances]
    refs = [(i.id, p.id, tuple(view.references(i.id, p.id)))
            for i in view.instances() if i.alive
            for p in view.properties(i.class_id) if p.type == "ObjectRef"]
    geometry = view.geometry_snapshot()
    return (view.generation, view.serial, view.segment_epoch, view.scene_id,
            view.frame, instance_signature(view), names, refs,
            geometry.object_ids.tobytes(), geometry.scene_ids.tobytes(),
            geometry.transforms.tobytes())


@pytest.mark.parametrize("filename", ["test_scene.ovd", "cloth_physx34.pxd2",
                                      "ue4_shooter.pxd2"])
def test_owned_view_matches_native_tree_geometry_and_selected_fields(tmp_path, filename):
    source = open_capture(tmp_path, filename)
    scene = source.scenes()[0]
    first = source.presentation(scene.id, 1)
    actor = int(first.geometry_snapshot().object_ids[0])
    view = source.presentation(scene.id, 1, inspect_ids=[actor], overlay_mask=3)
    assert view.ready and view.request_applied and view.frame == 1
    assert view.capture_key == source.capture_key == view.owner_token.key
    assert view.inspected_ids == [actor] and view.unavailable_ids == []
    assert instance_signature(view) == instance_signature(source)
    direct, owned = PvdDataAdapter(source), PvdDataAdapter(view)
    for grouping in ("scene", "class"):
        assert owned.hierarchy(grouping) == direct.hierarchy(grouping)
    for instance in view.instances():
        assert owned.name(instance.id) == direct.name(instance.id)
    cls = next(i.class_id for i in view.instances() if i.id == actor)
    for prop in view.properties(cls):
        assert view.property_size(actor, prop.id) == source.property_size(actor, prop.id)
        assert view.read(actor, prop.id) == source.read(actor, prop.id)
        assert view.named_values(prop.id) == source.named_values(prop.id)
        if prop.type == "StringHandle":
            assert view.string_value(actor, prop.id) == source.string_value(actor, prop.id)
    np.testing.assert_array_equal(view.geometry_snapshot().transforms,
                                  source.geometry_snapshot().transforms)
    assert view.overlay_snapshot() is not None
    assert not view.geometry_snapshot().transforms.flags.writeable
    signature = owned_signature(view)
    later = source.presentation(scene.id, 2)
    assert later.capture_key == view.capture_key and later.generation > view.generation
    del source, first, later
    gc.collect()
    assert owned_signature(view) == signature


def test_uncaptured_pages_require_explicit_new_owned_request(tmp_path):
    source = open_capture(tmp_path, "cloth_physx34.pxd2")
    scene = source.scenes()[0]
    # Checked-in native cloth particle array: 654 records, 16 bytes each.
    instance, property_id = 15, 649
    first = source.presentation(scene.id, 1)
    with pytest.raises(SnapshotDataUnavailable):
        first.read(instance, property_id, 64, 16)
    assert not first.has_read(instance, property_id, 64, 16)
    view = source.presentation(scene.id, 1, read_requests=[(instance, property_id, 64, 16)])
    expected = source.read(instance, property_id, 64, 16)
    assert view.property_size(instance, property_id) == 10464
    assert view.read(instance, property_id, 64, 16) == expected
    assert view.has_read(instance, property_id, 64, 16)
    assert not view.has_read(instance, property_id, 64, 32)
    with pytest.raises(SnapshotDataUnavailable):
        view.read(instance, property_id, 64, 32)
    source.presentation(scene.id, 30)
    assert view.read(instance, property_id, 64, 16) == expected
    with pytest.raises(ValueError, match="8 MiB"):
        source.presentation(scene.id, 1, read_requests=[(instance, property_id, 0, 8*1024*1024+1)])
    assert source.scenes()[0].current_frame == 30


def test_query_pick_captures_authoritative_struct_fields_and_requested_record(tmp_path):
    source = open_capture(tmp_path, "cloth_physx34.pxd2")
    scene = source.scenes()[0]
    first = source.presentation(scene.id, 1, overlay_mask=64)
    query = first.query_records()[0]
    assert query.query_id in first.overlay_snapshot().object_ids
    request = (query.scene_instance_id, query.property_id, query.element_index)
    with pytest.raises(SnapshotDataUnavailable):
        first.array_record(*request)
    view = source.presentation(scene.id, 1, inspect_ids=[query.query_id], overlay_mask=64)
    assert view.inspected_ids == [query.query_id]
    record = view.array_record(*request)
    assert [(f.id, f.data, f.string_value) for f in record.fields] == [
        (f.id, f.data, f.string_value) for f in source.array_record(*request).fields]
    offset = query.element_index * query.element_size
    assert view.read(query.scene_instance_id, query.property_id, offset, query.element_size) == source.read(
        query.scene_instance_id, query.property_id, offset, query.element_size)
    explicit = source.presentation(scene.id, 1, record_requests=[request])
    assert explicit.array_record(*request).class_id == query.class_id


def test_capture_scope_and_error_restore_are_explicit(tmp_path):
    source = open_capture(tmp_path)
    view = source.presentation(frame=1, inspect_ids=[2147483647])
    assert view.unavailable_ids == [2147483647]
    assert view.inspected_ids == []
    other = open_capture(tmp_path)
    assert other.capture_key != source.capture_key
    with pytest.raises(IndexError):
        source.presentation(view.scene_id, 2, read_requests=[(2147483647, 0, 0, 16)])
    assert source.scenes()[0].current_frame == 1
    with pytest.raises(AttributeError):
        view.frame = 99
    with pytest.raises(ValueError):
        source.presentation(view.scene_id, 1, inspect_ids=list(range(65)))


def test_competing_native_seeks_cannot_split_owned_pose_metadata_and_geometry(tmp_path):
    source = open_capture(tmp_path)
    source.presentation(frame=1)
    actor = unique_instance_id(source, 'PxRigidDynamic')
    scene = source.scenes()[0]
    initial = source.presentation(scene.id, 1, inspect_ids=[actor])
    cls = next(i.class_id for i in initial.instances() if i.id == actor)
    pose = next(p.id for p in initial.properties(cls) if p.name == "globalPose")
    def seek_repeatedly():
        for iteration in range(100):
            source.seek(scene.id, 1 + iteration % 2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        competing = pool.submit(seek_repeatedly)
        for iteration in range(40):
            view = source.presentation(scene.id, 1 + iteration % 2, inspect_ids=[actor])
            expected_position = np.frombuffer(view.read(actor, pose), dtype="<f4")[4:]
            np.testing.assert_array_equal(view.geometry_snapshot().transforms[0, :3, 3], expected_position)
            assert view.scenes()[0].current_frame == view.frame == 1 + iteration % 2
        competing.result()


def test_obsolete_inspector_requests_do_not_prevent_playback_when_allowed(tmp_path):
    source = open_capture(tmp_path)
    source.presentation(frame=1)
    actor = unique_instance_id(source, 'PxRigidDynamic')
    first = source.presentation(frame=1, inspect_ids=[actor])
    cls = next(i.class_id for i in first.instances() if i.id == actor)
    pose = next(p.id for p in first.properties(cls) if p.name == "globalPose")
    request = (actor, pose, 0, 28)
    dead = source.presentation(first.scene_id, 3, inspect_ids=[actor],
                               read_requests=[request], allow_unavailable_requests=True)
    assert dead.ready and dead.request_applied and dead.frame == 3
    assert dead.unavailable_ids == [actor] and dead.unavailable_reads == [request]
    assert len(dead.geometry_snapshot().object_ids) == 0
    legacy = open_capture(tmp_path, "cloth_physx34.pxd2")
    scene = legacy.scenes()[0]
    # Five native ray records at frame 1 become one at frame 2.
    record, window = (6, 401, 4), (6, 401, 256, 64)
    first = legacy.presentation(scene.id, 1, record_requests=[record], read_requests=[window])
    assert first.array_record(*record).fields
    second = legacy.presentation(scene.id, 2, record_requests=[record], read_requests=[window],
                                  allow_unavailable_requests=True)
    assert second.request_applied and second.frame == 2
    assert second.unavailable_reads == [window] and second.unavailable_records == [record]
    assert second.property_size(6, 401) == 64 and second.read(*window) == b""
    with pytest.raises(SnapshotDataUnavailable):
        second.array_record(*record)
    with pytest.raises(IndexError):
        legacy.presentation(scene.id, 3, record_requests=[record])
    assert legacy.scenes()[0].current_frame == 2


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    pytest.fail("Timed out waiting for native producer publication")


@pytest.fixture
def live(tmp_path):
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    source = Session.listen(port, str(tmp_path / "capture.part"))
    def connect():
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=1)
        except ConnectionRefusedError:
            return None
    connection = wait_until(connect)
    try:
        yield source, connection
    finally:
        connection.close()
        source.stop_live()


def test_partial_live_publication_defers_the_entire_view_and_frozen_capture_scrubs(live):
    source, connection = live
    connection.sendall(HANDSHAKE + PAYLOAD[:1024])
    wait_until(lambda: source.info().unpublished_changes)
    empty = source.presentation()
    assert not empty.ready and not empty.request_applied
    assert empty.instances() == [] and empty.classes() == []
    connection.sendall(PAYLOAD[1024:FRAME2_END])
    wait_until(lambda: source.info().completed_serial == 2)
    first = source.presentation(frame=2)
    actor = int(first.geometry_snapshot().object_ids[0])
    first = source.presentation(first.scene_id, 2, inspect_ids=[actor], overlay_mask=3)
    signature = owned_signature(first)
    cls = next(i.class_id for i in first.instances() if i.id == actor)
    properties = [(p.id, first.read(actor, p.id)) for p in first.properties(cls)]
    # Leave STOP_FRAME incomplete so publication cannot race the observation
    # below; sending the entire tail briefly clears the gate at frame 3.
    connection.sendall(PAYLOAD[FRAME2_END:FRAME3_END-1])
    wait_until(lambda: source.info().unpublished_changes)
    deferred = source.presentation(first.scene_id, 1, inspect_ids=[6], overlay_mask=64)
    assert deferred.ready and not deferred.request_applied
    assert deferred.inspected_ids == first.inspected_ids
    assert owned_signature(deferred) == signature
    assert [(p, deferred.read(actor, p)) for p, _ in properties] == properties
    source.stop_live()
    frozen = source.presentation(first.scene_id, 1, inspect_ids=[actor], overlay_mask=3)
    assert frozen.ready and frozen.request_applied and frozen.frame == 1
    assert frozen.scenes()[0].current_frame == 1
    assert len(frozen.geometry_snapshot().object_ids) == 1
    assert owned_signature(first) == signature


def test_live_segment_reused_handles_have_coherent_distinct_scene_ownership(live):
    source, connection = live
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    first_end, second_end = 42117, 98924
    connection.sendall(HANDSHAKE + payload[:first_end])
    wait_until(lambda: source.info().completed_serial == 120)
    first = source.presentation(frame=1)
    signature = owned_signature(first)
    first_scene = next(s for s in first.scenes() if s.alive)
    # Stop within the second schema, before its first publication. This fixed
    # byte boundary avoids racing any intervening completed frame.
    connection.sendall(payload[first_end:first_end+1024])
    wait_until(lambda: source.info().unpublished_changes)
    deferred = source.presentation()
    assert not deferred.request_applied and owned_signature(deferred) == signature
    connection.sendall(payload[first_end+1024:second_end])
    wait_until(lambda: source.info().completed_serial == 241)
    second = source.presentation(frame=1)
    assert second.segment_epoch != first.segment_epoch
    assert second.capture_key == first.capture_key
    # The native shared timeline makes both segment objects alive at frame 1;
    # their creation IDs still disambiguate the reused stream handle exactly.
    active = next(s for s in second.scenes() if s.id != first_scene.id)
    assert active.alive and active.stream_id == first_scene.stream_id
    assert set(second.geometry_snapshot().scene_ids) == {
        scene.instance_id for scene in second.scenes() if scene.alive}
    selected = source.presentation(active.id, 1)
    assert selected.scene_id == active.id
    assert selected.geometry_snapshot().scene_ids.tolist() == second.geometry_snapshot().scene_ids.tolist()
    assert owned_signature(first) == signature
