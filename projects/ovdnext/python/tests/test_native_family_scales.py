# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family sizes change native local extents without moving captured anchors."""
from pathlib import Path

import numpy as np
import pytest
from _pvd_native import Session
from test_joint_scale import joint_captures

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (2, 4, 8, 32, 256)
CONTACT_CAPTURE = "2026_04_28_15_24_52_000000a751246bf6.ovd"


def open_capture(tmp_path, filename="multi_geom.ovd", frame=150):
    source = Session(str(ROOT / "test/data" / filename), str(tmp_path))
    source.presentation(frame=frame)
    return source


def overlay(source, api, **kwargs):
    settings = dict(gizmo_scale=2, tolerance_speed=10,
                    joint_scale=.75, joint_base_length=.4)
    settings.update(kwargs)
    if api == "presentation":
        return source.presentation(overlay_mask=511, **settings).overlay_snapshot()
    return source.overlay_snapshot(511, **settings)


def assert_metadata_equal(before, after):
    for field in ("kinds", "categories", "object_ids", "related_ids", "scene_ids",
                  "color_roles", "colors"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("category", [2, 4, 8, 256])
def test_actor_family_scale_changes_only_its_local_extent(tmp_path, api, category):
    source = open_capture(tmp_path)
    publication = source.presentation()
    geometry = publication.geometry_snapshot()
    before = overlay(source, api)
    after = overlay(source, api, family_scales={category: 8})
    assert_metadata_equal(before, after)
    chosen = before.categories == category
    assert chosen.any()
    assert set(before.object_ids[chosen]) <= {i.id for i in publication.instances()}
    assert set(before.scene_ids[chosen]) <= {s.instance_id for s in publication.scenes()}
    np.testing.assert_array_equal(after.positions[~chosen], before.positions[~chosen])
    old, new = before.positions[chosen, :2], after.positions[chosen, :2]
    assert np.linalg.norm(old[:, 1] - old[:, 0], axis=1).max() > .01
    if category == 256:
        np.testing.assert_allclose(new.mean(axis=1), old.mean(axis=1), atol=2e-6)
    else:
        np.testing.assert_array_equal(new[:, 0], old[:, 0])
    np.testing.assert_allclose(new[:, 1] - new[:, 0],
                               4 * (old[:, 1] - old[:, 0]), atol=4e-6)
    np.testing.assert_array_equal(source.geometry_snapshot().transforms, geometry.transforms)
    np.testing.assert_array_equal(source.geometry_snapshot().bounds, geometry.bounds)
    assert after.joint_scale == .75 and after.joint_base_length == pytest.approx(.4)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
def test_contact_scale_preserves_cross_midpoints_and_normal_origins(tmp_path, api):
    source = open_capture(tmp_path, CONTACT_CAPTURE, frame=1)
    publication = source.presentation()
    before = overlay(source, api)
    after = overlay(source, api, family_scales={32: 8})
    assert_metadata_equal(before, after)
    chosen = before.categories == 32
    assert chosen.sum() == 36
    assert set(before.related_ids[chosen].ravel()) <= {i.id for i in publication.instances()}
    np.testing.assert_array_equal(after.positions[~chosen], before.positions[~chosen])
    for role in (9, 10):  # ContactPoint crosses and ContactNormal lines.
        selected = chosen & (before.color_roles == role)
        assert selected.any()
        old, new = before.positions[selected, :2], after.positions[selected, :2]
        if role == 9:
            np.testing.assert_allclose(new.mean(axis=1), old.mean(axis=1), atol=1e-6)
        else:
            np.testing.assert_array_equal(new[:, 0], old[:, 0])
            np.testing.assert_allclose(np.linalg.norm(new[:, 1] - new[:, 0], axis=1),
                                       .04, atol=1e-6)
        np.testing.assert_allclose(new[:, 1] - new[:, 0],
                                   4 * (old[:, 1] - old[:, 0]), atol=2e-6)


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("filename,frame", [("multi_geom.ovd", 150), (CONTACT_CAPTURE, 1)])
def test_omitted_empty_and_explicit_legacy_family_scales_match(tmp_path, api, filename, frame):
    source = open_capture(tmp_path, filename, frame)
    before = overlay(source, api)
    for scales in ({}, dict.fromkeys(FAMILIES, 2)):
        after = overlay(source, api, family_scales=scales)
        assert_metadata_equal(before, after)
        np.testing.assert_array_equal(after.positions, before.positions)
        assert after.family_scales == scales


@pytest.mark.parametrize("api", ["presentation", "overlay_snapshot"])
@pytest.mark.parametrize("scales", [
    {0: 1}, {1: 1}, {6: 1}, {16: 1}, {64: 1}, {128: 1}, {511: 1}, {1024: 1},
    {2: 0}, {4: -1}, {8: float("nan")}, {32: float("inf")}, {256: -float("inf")},
])
def test_invalid_family_requests_are_rejected(tmp_path, api, scales):
    source = open_capture(tmp_path)
    with pytest.raises(ValueError):
        overlay(source, api, family_scales=scales)
    # Validate even when no overlay extraction is requested.
    with pytest.raises(ValueError):
        source.presentation(overlay_mask=0, family_scales=scales)


def test_owned_family_request_survives_later_requests_and_session_destruction(tmp_path):
    source = open_capture(tmp_path)
    requested = {2: 8, 4: 3, 8: 5, 32: 6, 256: 7}
    old = overlay(source, "presentation", family_scales=requested)
    positions = old.positions.copy()
    requested[2] = 100
    source.presentation(frame=1, overlay_mask=511, family_scales={2: 1})
    del source
    assert old.family_scales == {2: 8, 4: 3, 8: 5, 32: 6, 256: 7}
    copied = old.family_scales
    copied[2] = 200
    assert old.family_scales[2] == 8
    with pytest.raises(AttributeError):
        old.family_scales = {}
    assert not old.positions.flags.writeable
    np.testing.assert_array_equal(old.positions, positions)


def test_deformable_velocity_scale_keeps_recorded_particle_positions(tmp_path, joint_captures):
    source = Session(str(joint_captures["sdk.ovd"]), str(tmp_path))
    publication = source.presentation(frame=1)
    classes = {c.id: c.name for c in publication.classes()}
    actors = {i.id for i in publication.instances()
              if classes[i.class_id] in ("PxDeformableVolume", "PxDeformableSurface")}
    before = overlay(source, "presentation")
    after = overlay(source, "presentation", family_scales={4: 8})
    chosen = (before.categories == 4) & np.isin(before.object_ids, list(actors))
    assert chosen.sum() == 5
    assert_metadata_equal(before, after)
    np.testing.assert_array_equal(after.positions[~chosen], before.positions[~chosen])
    np.testing.assert_array_equal(after.positions[chosen, 0], before.positions[chosen, 0])
    np.testing.assert_allclose(after.positions[chosen, 1] - after.positions[chosen, 0],
                               [[.8, 0, 0]] * 5, atol=1e-6)


@pytest.mark.parametrize("filename", ["sdk.ovd", "sdk.pxd2"])
@pytest.mark.parametrize("joint_scale", [None, .75])
def test_family_scales_preserve_sdk_joint_and_custom_primitives(tmp_path, joint_captures,
                                                              filename, joint_scale):
    source = Session(str(joint_captures[filename]), str(tmp_path))
    source.presentation(frame=1)
    before = overlay(source, "presentation", joint_scale=joint_scale)
    after = overlay(source, "presentation", joint_scale=joint_scale,
                    family_scales=dict.fromkeys(FAMILIES, 8))
    assert_metadata_equal(before, after)
    for category in (16, 128):
        chosen = before.categories == category
        assert chosen.any()
        np.testing.assert_array_equal(after.positions[chosen], before.positions[chosen])


@pytest.mark.parametrize("filename", ["cloth_physx34.pxd2", "ue4_shooter.pxd2"])
def test_family_scales_preserve_query_endpoints_normals_and_geometry(tmp_path, filename):
    source = open_capture(tmp_path, filename, frame=1)
    before = overlay(source, "overlay_snapshot")
    after = overlay(source, "overlay_snapshot", family_scales=dict.fromkeys(FAMILIES, 8))
    chosen = before.categories == 64
    assert chosen.any()
    np.testing.assert_array_equal(after.positions[chosen], before.positions[chosen])
    if before.query_geometry is not None:
        for field in ("transforms", "dimensions", "bounds", "object_ids", "geometry_ids"):
            np.testing.assert_array_equal(getattr(after.query_geometry, field),
                                          getattr(before.query_geometry, field))
