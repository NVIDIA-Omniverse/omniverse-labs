# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-writer bytes over real TCP: reused handles, changed topology, held arrays."""
import socket
import struct
import time

import numpy as np
from _pvd_native import Session

from test_selected_deformables import selected_capture, recorded_ids, expected_starts


def until(check):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(.01)
    raise AssertionError("Synthetic selected-deformable stream did not reach its expected state")


def snapshot(session, scene):
    session.seek(scene.id, 2)
    return session.geometry_snapshot(), session.overlay_snapshot(4), recorded_ids(session), scene.instance_id


def assert_samples(state, second):
    geometry, overlay, handles, scene = state
    resources = {mesh.id: mesh for mesh in geometry.meshes}
    rows = {handles[int(shape)]: resources[int(resource)] for shape, resource, owner_scene in
            zip(geometry.shape_ids, geometry.resource_ids, geometry.scene_ids) if owner_scene == scene}
    assert rows.keys() == {201, 202, 204, 205}
    starts = expected_starts(2, 10 if second else 0)
    triangle = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]] if second else [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]] + ([[0, 0, -1]] if second else [])
    triangle_faces = {(0, 1, 2), (1, 2, 3)} if second else {(0, 1, 2)}
    tet_faces = {(0, 1, 3), (0, 2, 3), (1, 2, 3)} | (
        {(0, 1, 4), (0, 2, 4), (1, 2, 4)} if second else {(0, 1, 2)})
    for shape, mesh in rows.items():
        volume = shape in (201, 205)
        expected = np.array(tet if volume else triangle) + starts[shape]
        np.testing.assert_array_equal(mesh.vertices, expected)
        faces = {tuple(sorted(face)) for face in mesh.indices.reshape(-1, 3)}
        assert faces == (tet_faces if volume else triangle_faces)
        expected_indices = (18 if second else 12) if volume else (6 if second else 3)
        assert len(mesh.indices) == expected_indices
        assert handles[mesh.source_id] == (101 if volume else 103)
    for actor, shape, step in ((10, 205, .2), (20, 202, .4), (21, 204, -.3)):
        indices = [index for index, (owner, sid) in enumerate(zip(overlay.object_ids, overlay.scene_ids))
                   if sid == scene and handles[int(owner)] == actor]
        positions = overlay.positions[indices]
        expected = np.array(tet if actor == 10 else triangle) + starts[shape]
        np.testing.assert_array_equal(positions[:, 0], expected)
        np.testing.assert_allclose(positions[:, 1] - positions[:, 0], [[0, step, 0]] * len(expected), atol=1e-6)
    assert not geometry.diagnostics and not overlay.diagnostics
    return rows


def test_live_segments_and_reconnect_replace_topology_without_mutating_held_snapshots(selected_capture, tmp_path):
    payload = selected_capture("segments").read_bytes()
    boundary = struct.pack("<BQQ", 12, 1, 3)  # STOP_FRAME(context=1, timestamp=3)
    assert payload.count(boundary) == 2
    split = payload.index(boundary) + len(boundary)
    held = []
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    # Native Session owns one connection; reconnect is a fresh listener, as in
    # CaptureWorkflow. Keep the old session's snapshots throughout the next run.
    for connection in range(2):
        session = Session.listen(port)
        peer = None
        try:
            def connect():
                nonlocal peer
                try:
                    peer = socket.create_connection(("127.0.0.1", port), timeout=1)
                    return True
                except ConnectionRefusedError:
                    return False
            until(connect)
            peer.sendall(struct.pack("<IHH", 0x4F56444C, 1, 0) + payload[:split])
            until(lambda: session.info().completed_serial >= 3)
            session.refresh()
            first_scene, = session.scenes()
            first = snapshot(session, first_scene)
            first_rows = assert_samples(first, False)
            held.append((first, False))
            peer.sendall(payload[split:])
            until(lambda: session.info().completed_serial >= 6)
            session.refresh()
            scenes = session.scenes()
            assert len(scenes) == 2
            second_scene = next(scene for scene in scenes if scene.id != first_scene.id)
            second = snapshot(session, second_scene)
            second_rows = assert_samples(second, True)
            held.append((second, True))
            assert first[0].segment_epoch != second[0].segment_epoch
            for shape in first_rows:
                assert first_rows[shape].id != second_rows[shape].id
                assert first_rows[shape].topology_id != second_rows[shape].topology_id
                assert first_rows[shape].indices.ctypes.data != second_rows[shape].indices.ctypes.data
            # Earlier-segment seek is supported; it must resolve the old mesh,
            # not the new topology associated with the same writer handles.
            assert_samples(snapshot(session, first_scene), False)
            peer.close()
            peer = None
            until(lambda: session.info().ended)
            path = tmp_path / f"connection-{connection}.ovd"
            assert session.save(str(path)) == len(payload)
            assert path.read_bytes() == payload
            reopened = Session(str(path), str(tmp_path / f"reopened-{connection}"))
            for scene in reopened.scenes():
                assert_samples(snapshot(reopened, scene), scene.recording_segment == second_scene.recording_segment)
        finally:
            if peer is not None:
                peer.close()
            session.stop_live()
        for state, second in held:
            assert_samples(state, second)
