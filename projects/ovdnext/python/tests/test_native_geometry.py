# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native geometry parity on real captures and pinned-writer animation streams."""
from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import pytest
import _pvd_native
from _pvd_native import Session
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]


def capture(tmp_path, name, frame=1):
    s = Session(str(ROOT / "test/data" / name), str(tmp_path))
    s.seek(s.scenes()[0].id, frame)
    return s


@pytest.fixture(scope="session")
def animated_capture(tmp_path_factory):
    fixture = tmp_path_factory.mktemp("native_geometry") / "animated.ovd"
    binary = executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")
    subprocess.run([str(binary), str(fixture)], check=True)
    return fixture


def test_native_sphere_pose_dimensions_and_owned_arrays(tmp_path):
    s = capture(tmp_path, "test_scene.ovd")
    g = s.geometry_snapshot()
    assert g.kinds.tolist() == [1]
    np.testing.assert_allclose(g.dimensions, [[10, 10, 10]])
    np.testing.assert_allclose(g.transforms[0, :3, 3], [420, 69.69, 42], rtol=1e-6)
    assert g.object_ids.dtype == np.int32
    assert g.transforms.dtype == np.float32
    assert g.transforms.flags.c_contiguous
    assert g.object_ids[0] != g.shape_ids[0]
    assert g.scene_ids[0] == s.scenes()[0].instance_id
    arrays = [g.object_ids, g.transforms, g.bounds, g.dimensions]
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    old_pose = g.transforms.copy()
    s.seek(s.scenes()[0].id, 3)
    assert len(s.geometry_snapshot().object_ids) == 0
    del s, g
    np.testing.assert_array_equal(arrays[1], old_pose)


def test_all_rigid_geometries_and_static_mesh_reuse(tmp_path):
    s = capture(tmp_path, "multi_geom.ovd")
    first = s.geometry_snapshot()
    assert Counter(first.kinds.tolist()) == {1: 2, 2: 3, 3: 2, 4: 1, 5: 3}
    assert first.diagnostics == []
    assert sorted((len(m.vertices), len(m.indices)) for m in first.meshes) == [(18, 96), (64, 294), (81, 384)]
    for mesh in first.meshes:
        assert mesh.indices.dtype == np.uint32
        assert mesh.indices.max() < len(mesh.vertices)
        assert not mesh.vertices.flags.writeable
    s.seek(s.scenes()[0].id, 2)
    second = s.geometry_snapshot()
    # The fixture adds another geometry later; prior static sources remain shared.
    later = {m.source_id: m for m in second.meshes}
    for before in first.meshes:
        after = later[before.source_id]
        assert before.id == after.id
        assert before.vertices.ctypes.data == after.vertices.ctypes.data
        assert before.indices.ctypes.data == after.indices.ctypes.data


def test_articulation_pose_and_shape_identity(tmp_path):
    s = capture(tmp_path, "articulation.ovd")
    g = s.geometry_snapshot()
    names = {c.id: c.name for c in s.classes()}
    assert [names[c] for c in g.actor_classes] == ["PxArticulationLink"] * 3
    np.testing.assert_allclose(sorted(g.transforms[:, 0, 3]), [-6, -2, 2], atol=1e-5)
    np.testing.assert_allclose(g.transforms[:, 1, 3], [5, 5, 5])
    np.testing.assert_allclose(g.dimensions, [[0.5, 0.5, 2]] * 3)
    assert len(set(g.object_ids)) == len(set(g.shape_ids)) == 3
    assert len(set(g.scene_ids)) == 1


def test_legacy_convex_meshes_and_cloth_native_struct_decoding(tmp_path):
    s = capture(tmp_path, "cloth_physx34.pxd2")
    g = s.geometry_snapshot()
    assert Counter(g.kinds.tolist()) == {3: 5, 5: 1, 6: 5, 7: 5}
    assert g.diagnostics == []
    assert (g.scene_ids > 0).all()
    cloth = {int(rid): mesh for rid in g.resource_ids[g.kinds == 6] for mesh in g.meshes if mesh.id == rid}
    assert sorted(len(m.vertices) for m in cloth.values()) == [200, 200, 200, 200, 654]
    assert all(len(m.indices) % 2 == 0 for m in cloth.values())
    assert all(np.isfinite(m.vertices).all() for m in cloth.values())
    s = capture(tmp_path, "ue4_shooter.pxd2")
    g = s.geometry_snapshot()
    assert Counter(g.kinds.tolist()) == {2: 2, 3: 6, 5: 1072}
    assert len(g.meshes) < 1072  # shared source meshes stay instanced
    assert not g.diagnostics


def test_deformable_boundary_topology_and_particle_animation(tmp_path, animated_capture):
    s = Session(str(animated_capture), str(tmp_path))
    s.seek(s.scenes()[0].id, 1)
    first = s.geometry_snapshot()
    assert Counter(first.kinds.tolist()) == {5: 3, 7: 1}
    assert not first.diagnostics
    names = {c.id: c.name for c in s.classes()}
    assert Counter(names[c] for c in first.actor_classes) == {"PxDeformableVolume": 2, "PxDeformableSurface": 1, "PxPBDParticleSystem": 1}
    volume_rows = [j for j, c in enumerate(first.actor_classes) if names[c] == "PxDeformableVolume"]
    assert len(set(first.shape_ids[volume_rows])) == 2  # independent collision and simulation visibility handles
    tetrahedra = [m for m in first.meshes if len(m.vertices) == 5]
    for mesh in tetrahedra:
        faces = mesh.indices.reshape(-1, 3)
        assert len(faces) == 6  # two touching tetrahedra: shared face is interior
        assert not any(set(face) == {0, 1, 2} for face in faces)
    particles = next(m for m in first.meshes if len(m.vertices) == 2)
    np.testing.assert_allclose(particles.vertices, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(first.dimensions[first.kinds == 7], [[0.2] * 3])
    assert np.all(first.scene_ids > 0)
    s.seek(s.scenes()[0].id, 2)
    second = s.geometry_snapshot()
    for before, after in zip(first.meshes, second.meshes):
        assert before.id != after.id
        assert before.topology_id == after.topology_id
        if len(before.indices):
            assert before.indices.ctypes.data == after.indices.ctypes.data
        assert before.vertices.ctypes.data != after.vertices.ctypes.data
    np.testing.assert_allclose(particles.vertices, [[1, 2, 3], [4, 5, 6]])
    animated_particles = next(m for m in second.meshes if len(m.vertices) == 2)
    np.testing.assert_allclose(animated_particles.vertices, [[2, 2, 3], [4, 5, 6]])


def test_no_placeholder_particles_when_capture_has_no_buffers(tmp_path):
    s = capture(tmp_path, "particle_system.ovd")
    g = s.geometry_snapshot()
    assert len(g.object_ids) == 0
    assert any("no particle buffers" in d for d in g.diagnostics)


@pytest.mark.parametrize("name", ["multi_geom.ovd", "multiscene_rigid.ovd", "cloth_physx34.pxd2"])
def test_scene_selection_uses_explicit_resolved_identity(tmp_path, name):
    s = capture(tmp_path, name)
    scene_ids = {scene.instance_id for scene in s.scenes()}
    assert all(identity > 0 for identity in scene_ids)
    g = s.geometry_snapshot()
    assert set(g.scene_ids).issubset(scene_ids)


def test_recreated_scenes_do_not_alias_their_reused_writer_handle(tmp_path):
    s = capture(tmp_path, "resample_two_segment.ovd")
    scenes = s.scenes()
    assert len(scenes) == 2
    assert scenes[0].id != scenes[1].id
    assert scenes[0].stream_id == scenes[1].stream_id
    alive = [scene for scene in scenes if scene.alive]
    assert len(alive) == 2
    assert set(s.geometry_snapshot().scene_ids) == {scene.instance_id for scene in alive}
    assert s.seek(alive[0].id, 2) is True
    assert all(scene.current_frame == 2 for scene in s.scenes())
