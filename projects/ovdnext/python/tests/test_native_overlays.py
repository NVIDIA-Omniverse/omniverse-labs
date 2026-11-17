# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured overlays reuse native PVD parsing and preserve frame/identity semantics."""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from _pvd_native import Session
from test_native_geometry import animated_capture

ROOT = Path(__file__).resolve().parents[2]


def capture(tmp_path, name):
    s = Session(str(ROOT / "test/data" / name), str(tmp_path))
    s.seek(s.scenes()[0].id, 1)
    return s


def test_captured_actor_bounds_axes_velocity_and_owned_snapshots(tmp_path):
    s = capture(tmp_path, "test_scene.ovd")
    classes = {c.id: c.name for c in s.classes()}
    actor = next(i for i in s.instances() if classes[i.class_id] == "PxRigidDynamic")
    props = {p.name: p for p in s.properties(actor.class_id)}
    world_bounds = np.frombuffer(s.read(actor.id, props["worldBounds"].id), dtype="<f4").reshape(2, 3)
    o = s.overlay_snapshot(1 | 2 | 4 | 256, gizmo_scale=2, tolerance_speed=10)
    assert o.ready and o.current_frame == 1
    assert Counter(o.categories.tolist()) == {1: 12, 2: 3, 4: 1, 256: 3}
    points = o.positions[o.categories == 1, :2].reshape(-1, 3)
    np.testing.assert_allclose(points.min(axis=0), world_bounds[0])
    np.testing.assert_allclose(points.max(axis=0), world_bounds[1])
    axes = o.positions[o.categories == 2]
    np.testing.assert_allclose(np.linalg.norm(axes[:, 1] - axes[:, 0], axis=1), [0.6] * 3, rtol=1e-4)
    velocity = np.frombuffer(s.read(actor.id, props["linearVelocity"].id), dtype="<f4")
    arrow = o.positions[o.categories == 4][0]
    np.testing.assert_allclose(arrow[1], arrow[0] + velocity * 0.2, atol=1e-6)
    assert (o.object_ids == actor.id).all()
    assert (o.scene_ids == s.scenes()[0].instance_id).all()
    old = o.positions.copy()
    s.seek(s.scenes()[0].id, 3)
    assert len(s.overlay_snapshot().positions) == 0
    del s
    assert not o.positions.flags.writeable
    np.testing.assert_array_equal(o.positions, old)


def test_ovd_contact_cross_uses_next_recorded_step_and_does_not_leak(tmp_path, animated_capture):
    s = Session(str(animated_capture), str(tmp_path))
    scene = s.scenes()[0]
    s.seek(scene.id, 1)
    o = s.overlay_snapshot(32)
    assert o.kinds.tolist() == [2, 2, 2]
    assert o.categories.tolist() == [32, 32, 32]
    np.testing.assert_allclose(o.positions[:2, :2].mean(axis=1), [[5, 6, 7]] * 2)
    np.testing.assert_allclose(o.positions[2, 0], [5, 6, 7])
    np.testing.assert_allclose(o.positions[2, 1], [5, 6.005, 7])
    assert all(first > 0 and second > 0 and first != second for first, second in o.related_ids)
    assert set(o.scene_ids) == {scene.instance_id}
    s.seek(scene.id, 2)
    assert len(s.overlay_snapshot(32).kinds) == 0  # post-simulation body positions
    s.seek(scene.id, 3)
    assert len(s.overlay_snapshot(32).kinds) == 0  # no contact sample at step 4


def test_real_contact_fixture_and_articulation_joint_frames(tmp_path):
    s = capture(tmp_path, "2026_04_28_15_24_52_000000a751246bf6.ovd")
    contacts = s.overlay_snapshot(32)
    assert len(contacts.kinds) == 36
    assert contacts.kinds.tolist() == [2] * 36
    classes = {c.id: c.name for c in s.classes()}
    alive = {i.id: i for i in s.instances() if i.alive}
    dynamic = next(i.id for i in alive.values() if classes[i.class_id] == "PxRigidDynamic")
    statics = {i.id for i in alive.values() if classes[i.class_id] == "PxRigidStatic"}
    assert Counter(map(tuple, contacts.related_ids.tolist())) == Counter(
        {(dynamic, static): 18 for static in statics}
    )
    assert all(endpoint in alive for pair in contacts.related_ids for endpoint in pair)
    assert all(classes[alive[endpoint].class_id] != "PxScene" for pair in contacts.related_ids for endpoint in pair)
    s = capture(tmp_path, "articulation.ovd")
    joints = s.overlay_snapshot(16)
    assert len(joints.kinds) == 12 and (joints.kinds == 4).all()
    assert len(set(joints.object_ids)) == 2
    assert (joints.related_ids > 0).all()
    # Two joint frames per articulation joint: parent RGB at 1.5x, child RGB at 1x.
    lengths = np.linalg.norm(joints.positions[:, 1] - joints.positions[:, 0], axis=1)
    np.testing.assert_allclose(sorted(lengths), [1] * 6 + [1.5] * 6, rtol=1e-5)


def test_legacy_queries_and_exact_sweep_endpoint_shapes(tmp_path):
    s = capture(tmp_path, "cloth_physx34.pxd2")
    overlay = s.overlay_snapshot(64)
    assert Counter(overlay.kinds.tolist()) == {1: 5, 4: 10}
    assert overlay.object_ids[:5].tolist() == [-6, -22, -38, -54, -70]
    assert overlay.object_ids[5:].tolist() == [-5, -5, -21, -21, -37, -37, -53, -53, -69, -69]
    classes = {c.name: c for c in s.classes()}
    assert classes["PvdRaycast"].byte_size > 0
    s = capture(tmp_path, "ue4_shooter.pxd2")
    overlay = s.overlay_snapshot(64)
    g = overlay.query_geometry
    assert g.kinds.tolist() == [3, 3]
    assert g.object_ids.tolist() == [-7, -7]
    np.testing.assert_allclose(g.dimensions, [[42, 48.6, 0]] * 2)
    np.testing.assert_array_equal(g.override_colors, [[1, 0, 0, 1]] * 2)
    arrow = overlay.positions[(overlay.kinds == 4) & (overlay.object_ids == -7)][0]
    np.testing.assert_allclose(g.transforms[:, :3, 3], arrow[:2])
    np.testing.assert_array_equal(g.transforms[0, :3, :3], g.transforms[1, :3, :3])
    assert not overlay.diagnostics and not g.diagnostics


def test_native_debug_event_parser_retains_points_lines_limit_arcs_and_ids(tmp_path, animated_capture):
    s = Session(str(animated_capture), str(tmp_path))
    s.seek(s.scenes()[0].id, 1)
    classes = {c.id: c.name for c in s.classes()}
    sample = next(i for i in s.instances() if classes[i.class_id] == "AnalysisSample")
    custom = s.overlay_snapshot(128)
    assert Counter(custom.kinds.tolist()) == {1: 1, 2: 4}
    np.testing.assert_allclose(custom.positions[0, 0], [1, 2, 3])
    np.testing.assert_allclose(custom.colors[0, 0], [1, 0, 0, 1])
    assert (custom.object_ids == sample.id).all()
    joints = s.overlay_snapshot(16)
    assert Counter(joints.kinds.tolist()) == {2: 236, 4: 6}
    assert not joints.diagnostics
    arrows = joints.positions[joints.kinds == 4]
    np.testing.assert_allclose(arrows[:3, 0], [[2, 0, 0]] * 3)
    np.testing.assert_allclose(arrows[3:, 0], [[3, 0, 0]] * 3)
    velocity = s.overlay_snapshot(4)
    assert velocity.kinds.tolist() == [4] * 5
    np.testing.assert_allclose(velocity.positions[:, 1] - velocity.positions[:, 0], [[0.1, 0, 0]] * 5, atol=1e-6)


def test_overlay_mask_limits_and_scales(tmp_path):
    s = capture(tmp_path, "test_scene.ovd")
    assert len(s.overlay_snapshot(0).kinds) == 0
    for kwargs in [{"mask": 1024}, {"gizmo_scale": 0}, {"gizmo_scale": float("nan")}, {"tolerance_speed": 0}]:
        with pytest.raises(ValueError):
            s.overlay_snapshot(**kwargs)
