# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actor-owned samples must not alias shared mesh topology or legacy samples."""
import subprocess

import numpy as np
import pytest
from _pvd_native import Session
from native_fixtures import executable


@pytest.fixture
def selected_capture(tmp_path):
    def make(mode="selected"):
        path = tmp_path / f"{mode}.ovd"
        subprocess.run([str(executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")),
                        str(path), "--selected-deformable-" + mode], check=True)
        return path
    return make


def open_frame(path, cache, frame=1):
    session = Session(str(path), str(cache))
    session.seek(session.scenes()[0].id, frame)
    return session


def mesh_rows(snapshot):
    meshes = {mesh.id: mesh for mesh in snapshot.meshes}
    return [(int(owner), int(shape), meshes[int(resource)]) for owner, shape, resource in
            zip(snapshot.object_ids, snapshot.shape_ids, snapshot.resource_ids)]


def test_selected_geometry_channels_and_shared_topology_are_independent(selected_capture, tmp_path):
    session = open_frame(selected_capture(), tmp_path / "index")
    first = session.geometry_snapshot()
    rows = mesh_rows(first)
    assert len(rows) == 4  # two volume channels even when mesh handles are equal
    np.testing.assert_allclose(sorted(tuple(m.vertices[0]) for _, _, m in rows),
                               [(-6, -3, 0), (-6, 2, 0), (2, -3, 0), (2, 2, 0)])
    assert len({m.id for _, _, m in rows}) == 4
    assert len({shape for _, shape, _ in rows}) == 4
    assert len({m.source_id for _, _, m in rows}) == 2
    assert first.diagnostics == []
    assert np.all(first.scene_ids == session.scenes()[0].instance_id)
    held = [m.vertices.copy() for _, _, m in rows]
    again = mesh_rows(session.geometry_snapshot())
    assert [m.id for _, _, m in rows] == [m.id for _, _, m in again]
    session.seek(session.scenes()[0].id, 2)
    second = mesh_rows(session.geometry_snapshot())
    np.testing.assert_allclose(sorted(tuple(m.vertices[0]) for _, _, m in second),
                               [(-3, -3, 0), (-3, 2, 0), (-1, 2, 0), (5, -3, 0)])
    for (_, _, before), (_, _, after), values in zip(rows, second, held):
        assert before.id != after.id
        assert before.topology_id == after.topology_id
        assert before.indices.ctypes.data == after.indices.ctypes.data
        np.testing.assert_array_equal(before.vertices, values)
    session.seek(session.scenes()[0].id, 1)
    for (_, _, mesh), values in zip(mesh_rows(session.geometry_snapshot()), held):
        np.testing.assert_array_equal(mesh.vertices, values)


def test_selected_velocity_origins_use_simulation_not_collision_vertices(selected_capture, tmp_path):
    session = open_frame(selected_capture(), tmp_path / "index")
    overlay = session.overlay_snapshot(4)
    assert len(overlay.object_ids) == 10
    starts, ends = overlay.positions[:, 0], overlay.positions[:, 1]
    volume = starts[:, 1] < 0
    assert volume.sum() == 4
    np.testing.assert_allclose(starts[volume], [[2, -3, 0], [3, -3, 0], [2, -2, 0], [2, -3, 1]])
    np.testing.assert_allclose(ends[volume] - starts[volume], [[0, .2, 0]] * 4, atol=1e-6)
    left = (starts[:, 0] < 0) & ~volume
    right = ~volume & ~left
    np.testing.assert_allclose(ends[left] - starts[left], [[0, .4, 0]] * 3, atol=1e-6)
    np.testing.assert_allclose(ends[right] - starts[right], [[0, -.3, 0]] * 3, atol=1e-6)
    assert len(set(overlay.object_ids)) == 3
    assert not overlay.diagnostics
    scaled = session.overlay_snapshot(4, family_scales={4: 2.}, master_scale=3.)
    np.testing.assert_array_equal(scaled.positions[:, 0], starts)
    np.testing.assert_allclose(scaled.positions[:, 1] - starts, (ends - starts) * 6, atol=2e-6)
    np.testing.assert_array_equal(scaled.object_ids, overlay.object_ids)
    np.testing.assert_array_equal(scaled.related_ids, overlay.related_ids)


@pytest.mark.parametrize("mode", ["legacy", "empty"])
def test_registered_but_unsampled_actor_arrays_preserve_legacy_animation(selected_capture, tmp_path, mode):
    session = open_frame(selected_capture(mode), tmp_path / "index")
    for frame in (1, 2):
        session.seek(session.scenes()[0].id, frame)
        geometry = session.geometry_snapshot()
        assert len(geometry.object_ids) == 3  # legacy identical volume mesh deduplicated
        assert sorted(m.vertices[0, 0] for m in geometry.meshes) == [20 + frame, 30 + frame]
        overlay = session.overlay_snapshot(4)
        assert len(overlay.object_ids) == 10
        np.testing.assert_allclose(overlay.positions[:, 1] - overlay.positions[:, 0],
                                   [[.9, 0, 0]] * 10, atol=2e-6)


@pytest.mark.parametrize("mode", ["malformed", "nonfinite"])
def test_malformed_actor_positions_do_not_fall_back_to_legacy_mesh(selected_capture, tmp_path, mode):
    session = open_frame(selected_capture(mode), tmp_path / "index")
    geometry = session.geometry_snapshot()
    assert geometry.diagnostics
    assert len(geometry.object_ids) == 3
    assert max(float(m.vertices[:, 0].max()) for m in geometry.meshes) < 10


@pytest.mark.parametrize("mode", ["missing-velocity", "mismatch"])
def test_incomplete_actor_velocity_pair_never_borrows_mesh_samples(selected_capture, tmp_path, mode):
    session = open_frame(selected_capture(mode), tmp_path / "index")
    overlay = session.overlay_snapshot(4)
    assert len(overlay.object_ids) == 7
    assert overlay.diagnostics
    assert np.all(overlay.positions[:, 0, 0] >= 2)


def recorded_ids(session):
    page = session.evidence_metadata(0, 512)
    assert page.next_offset is None
    return {row.instance_id: row.api_handle for row in page.rows}


def expected_starts(frame, offset=0):
    dx = 3 * (frame - 1)
    return {201: (-6+dx+offset, -3, 0), 205: (2+dx+offset, -3, 0),
            202: (-6+dx+offset, 2, 0), 204: (2-dx+offset, 2, 0)}


@pytest.mark.parametrize("target,shape", [("surface", 202), ("collision", 201), ("simulation", 205)])
@pytest.mark.parametrize("fault", ["short", "stride", "nan", "inf"])
def test_invalid_selected_positions_reject_only_the_affected_channel(selected_capture, tmp_path, target, shape, fault):
    session = open_frame(selected_capture(target + "-" + fault), tmp_path / "index")
    handles = recorded_ids(session)
    for frame in (1, 2, 3, 1):
        session.seek(session.scenes()[0].id, frame)
        geometry = session.geometry_snapshot()
        rows = {handles[handle]: mesh for _, handle, mesh in mesh_rows(geometry)}
        expected = expected_starts(frame)
        if frame == 2:
            del expected[shape]
            assert geometry.diagnostics
        else:
            assert not geometry.diagnostics
        assert rows.keys() == expected.keys()
        for key, start in expected.items():
            np.testing.assert_array_equal(rows[key].vertices[0], start)
            assert np.isfinite(rows[key].vertices).all()
        if target == "collision":
            overlay = session.overlay_snapshot(4)
            assert len(overlay.object_ids) == 10 and not overlay.diagnostics


@pytest.mark.parametrize("target,shape,count", [("surface", 202, 3), ("collision", 201, 4), ("simulation", 205, 4)])
def test_valid_unused_selected_vertices_are_safe_without_exact_rest_count(selected_capture, tmp_path, target, shape, count):
    session = open_frame(selected_capture(target + "-extra"), tmp_path / "index", 2)
    handles = recorded_ids(session)
    geometry = session.geometry_snapshot()
    assert not geometry.diagnostics
    rows = {handles[handle]: mesh for _, handle, mesh in mesh_rows(geometry)}
    assert len(rows) == 4 and len(rows[shape].vertices) == count + 1
    np.testing.assert_array_equal(rows[shape].vertices[-1], [11, 12, 13])
    assert rows[shape].indices.max() == count - 1
    for key, start in expected_starts(2).items():
        np.testing.assert_array_equal(rows[key].vertices[0], start)


@pytest.mark.parametrize("target,actor,remaining", [("surface", 20, 7), ("simulation", 10, 6),
    ("surface-velocity", 20, 7), ("volume-velocity", 10, 6)])
@pytest.mark.parametrize("fault", ["short", "extra", "stride", "nan", "inf"])
def test_invalid_selected_velocity_pair_is_diagnosed_without_losing_other_actors(selected_capture, tmp_path, target, actor, remaining, fault):
    session = open_frame(selected_capture(target + "-" + fault), tmp_path / "index")
    handles = recorded_ids(session)
    for frame in (1, 2, 3, 1):
        session.seek(session.scenes()[0].id, frame)
        overlay = session.overlay_snapshot(4)
        assert len(overlay.object_ids) == (remaining if frame == 2 else 10)
        owners = [handles[int(identity)] for identity in overlay.object_ids]
        assert (actor not in owners) == (frame == 2)
        assert bool(overlay.diagnostics) == (frame == 2)
        assert np.isfinite(overlay.positions).all()
        if target.endswith("-velocity"):
            geometry = session.geometry_snapshot()
            assert len(geometry.object_ids) == 4 and not geometry.diagnostics
        for owner, shape, velocity in ((10, 205, .2), (20, 202, .4), (21, 204, -.3)):
            rows = overlay.positions[np.array(owners) == owner]
            if len(rows):
                np.testing.assert_array_equal(rows[0, 0], expected_starts(frame)[shape])
                np.testing.assert_allclose(rows[:, 1]-rows[:, 0], [[0, velocity, 0]]*len(rows), atol=1e-6)


@pytest.mark.parametrize("target", ["surface", "collision", "simulation", "surface-velocity", "volume-velocity"])
def test_fourth_float_is_not_a_rendered_coordinate(selected_capture, tmp_path, target):
    session = open_frame(selected_capture(target + "-w-nan"), tmp_path / "index", 2)
    geometry, overlay = session.geometry_snapshot(), session.overlay_snapshot(4)
    assert len(geometry.object_ids) == 4 and len(overlay.object_ids) == 10
    assert not geometry.diagnostics and not overlay.diagnostics
    assert all(np.isfinite(mesh.vertices).all() for mesh in geometry.meshes)
    assert np.isfinite(overlay.positions).all()


@pytest.mark.parametrize("target", ["surface", "collision", "simulation", "surface-velocity", "volume-velocity"])
def test_non_float_aligned_samples_fail_at_the_capture_schema_boundary(selected_capture, tmp_path, target):
    with pytest.raises(RuntimeError, match="attribute payload does not match its registered schema"):
        open_frame(selected_capture(target + "-byte"), tmp_path / "index")


@pytest.mark.parametrize("topology,remaining", [("triangle", {201, 205}), ("tet", {202, 204})])
@pytest.mark.parametrize("fault", ["index", "stride"])
def test_bad_topology_cannot_address_selected_vertices_or_hide_other_meshes(selected_capture, tmp_path, topology, remaining, fault):
    session = open_frame(selected_capture(topology + "-" + fault), tmp_path / "index")
    geometry = session.geometry_snapshot()
    handles = recorded_ids(session)
    rows = {handles[shape]: mesh for _, shape, mesh in mesh_rows(geometry)}
    assert rows.keys() == remaining and geometry.diagnostics
    for shape, mesh in rows.items():
        np.testing.assert_array_equal(mesh.vertices[0], expected_starts(1)[shape])
    assert all("index outside vertex buffer" in message if fault == "index" else "topology" in message
               for message in geometry.diagnostics)
