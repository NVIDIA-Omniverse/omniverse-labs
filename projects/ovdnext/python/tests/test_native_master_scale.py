# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Master size changes generated local extents without moving capture data."""
import numpy as np
import pytest
from _pvd_native import Session
from test_joint_scale import joint_captures
from test_native_family_scales import (CONTACT_CAPTURE, ROOT, assert_metadata_equal,
                                       open_capture, overlay)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("families", [{}, {2: 3, 4: 4, 8: 5, 256: 6}])
def test_master_doubles_actor_extents_and_preserves_bounds(tmp_path, api, families):
    source = open_capture(tmp_path)
    before = overlay(source, api, family_scales=families)
    after = overlay(source, api, family_scales=families, master_scale=2)
    assert_metadata_equal(before, after)
    for category in (2, 4, 8, 256):
        rows = before.categories == category
        assert rows.any()
        old, new = before.positions[rows, :2], after.positions[rows, :2]
        assert np.linalg.norm(old[:, 1] - old[:, 0], axis=1).max() > .01
        if category == 256:
            np.testing.assert_allclose(new.mean(axis=1), old.mean(axis=1), atol=2e-6)
        else:
            np.testing.assert_array_equal(new[:, 0], old[:, 0])
        np.testing.assert_allclose(new[:, 1] - new[:, 0],
                                   2 * (old[:, 1] - old[:, 0]), atol=4e-6)
    bounds = before.categories == 1
    assert bounds.any()
    np.testing.assert_array_equal(after.positions[bounds], before.positions[bounds])
    assert after.family_scales == families


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("contact_scale,normal_length", [(.01, .002), (8, .08)])
def test_master_applies_after_contact_minimum_without_moving_contacts(
        tmp_path, api, contact_scale, normal_length):
    source = open_capture(tmp_path, CONTACT_CAPTURE, frame=1)
    before = overlay(source, api, family_scales={32: contact_scale})
    after = overlay(source, api, family_scales={32: contact_scale}, master_scale=2)
    assert_metadata_equal(before, after)
    for role in (9, 10):
        rows = (before.categories == 32) & (before.color_roles == role)
        assert rows.any()
        old, new = before.positions[rows, :2], after.positions[rows, :2]
        if role == 9:
            np.testing.assert_allclose(new.mean(axis=1), old.mean(axis=1), atol=1e-6)
        else:
            np.testing.assert_array_equal(new[:, 0], old[:, 0])
            np.testing.assert_allclose(np.linalg.norm(new[:, 1] - new[:, 0], axis=1),
                                       normal_length, atol=1e-6)
        np.testing.assert_allclose(new[:, 1] - new[:, 0],
                                   2 * (old[:, 1] - old[:, 0]), atol=2e-6)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("joint_scale,child_length", [(None, 4), (0, .4), (.75, .6)])
def test_master_scales_structured_joint_arms_after_the_floor(
        tmp_path, api, joint_scale, child_length):
    source = open_capture(tmp_path, "articulation.ovd", frame=1)
    before = overlay(source, api, joint_scale=joint_scale)
    after = overlay(source, api, joint_scale=joint_scale, master_scale=2)
    rows = before.categories == 16
    assert rows.sum() == 12
    assert_metadata_equal(before, after)
    np.testing.assert_array_equal(after.positions[rows, 0], before.positions[rows, 0])
    lengths = np.linalg.norm(after.positions[rows, 1] - after.positions[rows, 0], axis=1)
    np.testing.assert_allclose(sorted(lengths), [child_length] * 6 + [child_length * 1.5] * 6,
                               rtol=1e-5, atol=1e-6)
    assert after.joint_scale == joint_scale
    assert after.joint_base_length == pytest.approx(.4)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("filename", ["sdk.ovd", "sdk.pxd2"])
@pytest.mark.parametrize("joint_scale", [None, 0, .75])
def test_master_scales_sdk_visuals_around_fixed_frames_and_physical_limit_offset(
        tmp_path, joint_captures, api, filename, joint_scale):
    source = Session(str(joint_captures[filename]), str(tmp_path))
    source.presentation(frame=1)
    before = overlay(source, api, joint_scale=joint_scale)
    after = overlay(source, api, joint_scale=joint_scale, master_scale=2)
    assert_metadata_equal(before, after)
    frames = (before.categories == 16) & (before.kinds == 4)
    assert frames.sum() == 6
    np.testing.assert_array_equal(after.positions[frames, 0], before.positions[frames, 0])
    np.testing.assert_allclose(after.positions[frames, 1] - after.positions[frames, 0],
                               2 * (before.positions[frames, 1] - before.positions[frames, 0]),
                               atol=1e-6)
    limits = np.flatnonzero((before.categories == 16) & (before.kinds == 2))
    assert len(limits) == 236
    # The native fixture records a linear limit at x=2, independently of its radius.
    anchor = np.array([2, 0, 0])
    np.testing.assert_allclose(after.positions[limits[:20], :2] - anchor,
                               2 * (before.positions[limits[:20], :2] - anchor), atol=1e-6)
    np.testing.assert_allclose(after.positions[limits[20:], :2],
                               2 * before.positions[limits[20:], :2], atol=1e-6)
    custom = before.categories == 128
    assert custom.any()
    np.testing.assert_array_equal(after.positions[custom], before.positions[custom])


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
def test_master_doubles_deformable_velocity_without_moving_particles(
        tmp_path, joint_captures, api):
    source = Session(str(joint_captures["sdk.ovd"]), str(tmp_path))
    publication = source.presentation(frame=1)
    classes = {c.id: c.name for c in publication.classes()}
    actors = {i.id for i in publication.instances()
              if classes[i.class_id] in ("PxDeformableVolume", "PxDeformableSurface")}
    before = overlay(source, api, family_scales={4: 8})
    after = overlay(source, api, family_scales={4: 8}, master_scale=2)
    rows = (before.categories == 4) & np.isin(before.object_ids, list(actors))
    assert rows.sum() == 5
    np.testing.assert_array_equal(after.positions[rows, 0], before.positions[rows, 0])
    np.testing.assert_allclose(after.positions[rows, 1] - after.positions[rows, 0],
                               [[1.6, 0, 0]] * 5, atol=1e-6)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("filename", ["cloth_physx34.pxd2", "ue4_shooter.pxd2"])
def test_master_changes_query_hit_normals_without_scaling_queries(tmp_path, api, filename):
    source = open_capture(tmp_path, filename, frame=1)
    before = overlay(source, api)
    after = overlay(source, api, master_scale=2)
    assert_metadata_equal(before, after)
    queries = before.categories == 64
    assert queries.any()
    normals = queries & (before.kinds == 4) & np.isin(-before.object_ids % 16, [2, 5])
    if filename == "cloth_physx34.pxd2":
        assert normals.sum() == 5
    np.testing.assert_array_equal(after.positions[queries & ~normals],
                                  before.positions[queries & ~normals])
    np.testing.assert_array_equal(after.positions[normals, 0], before.positions[normals, 0])
    tolerance = max(1e-5, 2 * np.spacing(np.abs(before.positions[queries, :2]).max()))
    np.testing.assert_allclose(after.positions[normals, 1] - after.positions[normals, 0],
                               2 * (before.positions[normals, 1] - before.positions[normals, 0]),
                               atol=tolerance)
    if before.query_geometry is not None:
        for field in ("transforms", "dimensions", "bounds", "object_ids", "geometry_ids"):
            np.testing.assert_array_equal(getattr(after.query_geometry, field),
                                          getattr(before.query_geometry, field))


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
def test_legacy_contact_markers_stay_fixed_while_normals_double(tmp_path, api):
    source = open_capture(tmp_path, "ue4_shooter.pxd2", frame=241)
    before = overlay(source, api, family_scales={32: 8})
    after = overlay(source, api, family_scales={32: 8}, master_scale=2)
    assert_metadata_equal(before, after)
    points = (before.categories == 32) & (before.kinds == 1)
    normals = (before.categories == 32) & (before.kinds == 4)
    assert points.sum() == 148 and normals.sum() == 89
    np.testing.assert_array_equal(after.positions[points], before.positions[points])
    np.testing.assert_array_equal(after.positions[normals, 0], before.positions[normals, 0])
    assert np.any(after.positions[normals, 1] != before.positions[normals, 1])
    # These world coordinates are large; allow endpoint subtraction's float32 ULPs.
    tolerance = 2 * np.spacing(np.abs(before.positions[normals, :2]).max())
    np.testing.assert_allclose(after.positions[normals, 1] - after.positions[normals, 0],
                               2 * (before.positions[normals, 1] - before.positions[normals, 0]),
                               atol=tolerance)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("settings", [
    {"master_scale": 0}, {"master_scale": -1}, {"master_scale": float("nan")},
    {"master_scale": float("inf")}, {"master_scale": -float("inf")},
    {"master_scale": 2, "gizmo_scale": 3e38},
    {"master_scale": 2, "family_scales": {4: 3e38}},
    {"master_scale": 2, "joint_scale": 2, "joint_base_length": 1e38},
])
def test_invalid_master_or_products_rejected_before_frame_change(tmp_path, api, settings):
    source = open_capture(tmp_path, "articulation.ovd", frame=1)
    scene = source.scenes()[0]
    with pytest.raises(ValueError):
        if api == "presentation":
            source.presentation(scene.id, 2, overlay_mask=0, **settings)
        else:
            source.overlay_snapshot(0, **settings)
    assert source.scenes()[0].current_frame == 1


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
def test_default_master_preserves_exact_buffers_and_owned_request(tmp_path, api):
    source = open_capture(tmp_path, CONTACT_CAPTURE, frame=1)
    before = overlay(source, api)
    explicit = overlay(source, api, master_scale=1)
    assert_metadata_equal(before, explicit)
    np.testing.assert_array_equal(explicit.positions, before.positions)
    assert before.master_scale == explicit.master_scale == 1
    owned = overlay(source, api, master_scale=2)
    positions = owned.positions.copy()
    source.presentation(frame=2, overlay_mask=511, master_scale=.5)
    del source
    assert owned.master_scale == 2
    with pytest.raises(AttributeError):
        owned.master_scale = 1
    assert not owned.positions.flags.writeable
    np.testing.assert_array_equal(owned.positions, positions)


def test_master_is_keyword_only_and_old_positional_calls_remain(tmp_path):
    source = open_capture(tmp_path, "articulation.ovd", frame=1)
    old = source.presentation(0, 1, [], [], [], 16, 2, 10, False).overlay_snapshot()
    np.testing.assert_array_equal(source.overlay_snapshot(16, 2, 10).positions, old.positions)
    assert source.overlay_snapshot(16, 2, 10, master_scale=2).master_scale == 2
    with pytest.raises(TypeError):
        source.presentation(0, 1, [], [], [], 16, 2, 10, False, 2)
    with pytest.raises(TypeError):
        source.overlay_snapshot(16, 2, 10, 2)
