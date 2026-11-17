# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GL attachment, winding, visibility and physical-particle regressions."""
from dataclasses import replace
import tempfile

import numpy as np
import pytest
from _pvd_native import Session

from ovd_ui.render_settings import FaceCulling, RenderingSettings
from ovd_ui.renderer import unit_box, unit_sphere
from test_native_geometry import animated_capture
from test_renderer import ROOT, duplicate_geometry, renderer, snapshot


@pytest.mark.parametrize("values", [{"geometry_scale": 0}, {"geometry_scale": float("nan")},
    {"geometry_scale": 10 ** 1000},
    {"particle_size": 11}, {"face_culling": 7}, {"face_culling": True}, {"face_culling": 1.0}, {"planes": True},
    {"anti_aliasing": "yes"}, {"background_color": "not a color"}])
@pytest.mark.parametrize("decode", [RenderingSettings.from_dict, RenderingSettings.from_preferences])
def test_invalid_preferences_rejected(values, decode):
    with pytest.raises(ValueError):
        decode(values)


def test_options_round_trip_and_background_scope():
    settings = RenderingSettings(anti_aliasing=True, geometry_scale=.01, background_mode="custom", background_color="#123456")
    assert RenderingSettings.from_dict(settings.to_dict()) == settings
    assert settings.background((0, 0, 0, 1)) == (0x12 / 255, 0x34 / 255, 0x56 / 255, 1.)
    assert replace(settings, background_mode="reference").background() == (.8, .8, .8, 1.)
    assert RenderingSettings().background((.1, .2, .3, 1)) == (.1, .2, .3, 1)


@pytest.mark.parametrize("culling", [FaceCulling.NONE, FaceCulling.NORMAL, FaceCulling.INVERSE])
def test_in_memory_round_trip_preserves_explicit_face_culling(culling):
    settings = RenderingSettings(face_culling=culling, anti_aliasing=True, wireframe=2)
    payload = settings.to_dict()
    assert "schema" not in payload
    restored = RenderingSettings.from_dict(payload)
    assert restored.face_culling == culling
    assert restored == settings


@pytest.mark.parametrize("saved,expected", [(1, 0), (2, 2), (0, 0)])
def test_rendering_preference_migration_preserves_other_fields_and_input(saved, expected):
    payload = {"face_culling": saved, "anti_aliasing": True, "wireframe": 2,
               "background_color": "#123456", "future_setting": "keep"}
    before = payload.copy()
    restored = RenderingSettings.from_preferences(payload)
    assert restored.face_culling == expected
    assert restored.anti_aliasing and restored.wireframe == 2
    assert restored.background_color == "#123456"
    assert payload == before


@pytest.mark.parametrize("schema", [2, 3])
def test_marked_preferences_preserve_explicit_normal_culling(schema):
    payload = {"face_culling": 1, "schema": schema}
    assert RenderingSettings.from_preferences(payload).face_culling == FaceCulling.NORMAL


def test_primitive_winding_agrees_with_outward_normals():
    for data in (unit_box(), unit_sphere()):
        faces = data.reshape(-1, 3, 6)
        cross = np.cross(faces[:, 1, :3] - faces[:, 0, :3], faces[:, 2, :3] - faces[:, 0, :3])
        assert (np.einsum("ij,ij->i", cross, faces[:, 0, 3:]) >= -1e-8).all()


def test_actual_msaa_edges_resize_toggle_and_single_sample_picking(renderer):
    renderer.set_snapshot(snapshot("test_scene.ovd"), True)
    renderer.set_settings(RenderingSettings(face_culling=0))
    renderer.render(401, 301)
    plain = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").copy()
    identity = renderer.pick(200, 150)
    renderer.set_settings(RenderingSettings(face_culling=0, anti_aliasing=True))
    renderer.render(401, 301)
    assert renderer.msaa_color.samples == renderer.samples >= 2
    assert renderer.pick_texture.samples == 0
    sampled = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1")
    assert np.count_nonzero(plain != sampled) > 100
    assert renderer.pick(200, 150) == identity
    renderer.render(603, 401)
    assert renderer.msaa_color.size == (603, 401)
    renderer.set_settings(RenderingSettings(anti_aliasing=False))
    renderer.render(603, 401)
    assert renderer.msaa_buffer is None
    assert renderer.context.error == "GL_NO_ERROR"


def test_msaa_transparency_resolves_opaque_depth(renderer):
    geometry = duplicate_geometry(snapshot("test_scene.ovd"))
    geometry.dimensions[1] = 5.
    renderer.set_settings(RenderingSettings(anti_aliasing=True))
    renderer.set_snapshot(geometry, True)
    renderer.hidden = {1001}
    renderer.render(400, 300)
    front = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
    renderer.hidden = set()
    renderer.selected = {1001}
    renderer.alpha_by_id = {1001: .8}
    renderer.render(400, 300)
    assert renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3) == front
    assert renderer.context.error == "GL_NO_ERROR"


def test_native_plane_culling_chirality_and_modes(renderer):
    geometry = snapshot("multi_geom.ovd")
    row = np.flatnonzero(geometry.kinds == 4)[0]
    identity = int(geometry.object_ids[row])
    center = geometry.transforms[row, :3, 3]
    normal = geometry.transforms[row, :3, 0]
    up = geometry.transforms[row, :3, 1]
    renderer.set_snapshot(geometry)
    renderer.hidden = set(map(int, geometry.object_ids)) - {identity}
    for left_handed in (False, True):
        renderer.camera.left_handed = left_handed
        renderer.camera.look_at(center + normal * 50, center, up)
        renderer.set_settings(RenderingSettings(face_culling=1))
        renderer.render(320, 240)
        assert renderer.pick(160, 120) == identity
        renderer.set_settings(RenderingSettings(face_culling=2))
        renderer.render(320, 240)
        assert renderer.pick(160, 120) is None
    for mode in (1, 2):
        renderer.set_settings(RenderingSettings(planes=mode))
        renderer.render(320, 240)
        assert renderer.instance_count == 0
        assert renderer.pick(160, 120) is None


def test_subdued_theme_ground_preserves_geometry_and_pick_buffer(renderer):
    geometry = snapshot("multi_geom.ovd", 2)
    renderer.set_snapshot(geometry, True)
    renderer.render(400, 300)
    before = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3).copy()
    renderer.pick(200, 150)
    picks = np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4").copy()
    renderer.render(400, 300, ground_color=(.1, .1, .11), grid_contrast=.025)
    after = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    renderer.pick(200, 150)
    assert np.array_equal(np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4"), picks)
    plane_ids = geometry.object_ids[geometry.kinds == 4].astype("u4") + 1
    plane_pixels = np.isin(picks, plane_ids)
    assert plane_pixels.any()
    assert np.any(before[plane_pixels] != after[plane_pixels])
    assert np.array_equal(before[~plane_pixels], after[~plane_pixels])


def test_selected_wireframe_keeps_filled_picking_and_msaa_overlays(renderer):
    geometry = snapshot("test_scene.ovd")
    identity = int(geometry.object_ids[0])
    renderer.set_snapshot(geometry, True)
    renderer.selected = {identity}
    renderer.set_settings(RenderingSettings(wireframe=2, anti_aliasing=True))
    renderer.render(400, 300)
    assert all(mesh.wireframe for _, mesh in renderer._groups)
    assert renderer.pick(200, 150) == identity
    with tempfile.TemporaryDirectory() as directory:
        session = Session(str(ROOT / "test/data/test_scene.ovd"), directory)
        session.seek(session.scenes()[0].id, 1)
        renderer.set_overlays(session.overlay_snapshot(1 | 2 | 4 | 256))
    renderer.render(400, 300)
    assert renderer.overlays.primitive_count > 0
    assert renderer.context.error == "GL_NO_ERROR"


def test_native_heightfield_is_two_sided_and_world_scale_frames_correctly(renderer):
    geometry = snapshot("multi_geom.ovd")
    assert geometry.double_sided.sum() == 1
    assert not geometry.double_sided.flags.writeable
    identity = int(geometry.object_ids[geometry.double_sided.astype(bool)][0])
    renderer.set_snapshot(geometry)
    renderer.hidden = set(map(int, geometry.object_ids)) - {identity}
    renderer.set_settings(RenderingSettings(face_culling=1, geometry_scale=10))
    renderer.frame_all()
    renderer.render(400, 300)
    normal = renderer.framebuffer.read(components=3)
    assert renderer.pick(200, 150) == identity
    renderer.set_settings(RenderingSettings(face_culling=2, geometry_scale=10))
    renderer.render(400, 300)
    assert renderer.framebuffer.read(components=3) == normal


def test_sleep_and_source_trigger_gates_apply_to_picking(renderer):
    geometry = duplicate_geometry(snapshot("test_scene.ovd"), 2)
    geometry.sleeping[:] = 1
    geometry.shape_flags = np.array([4, 0], dtype="u4")
    renderer.set_snapshot(geometry, True)
    renderer.selected = {1000}
    renderer.set_settings(RenderingSettings(sleeping_objects=2))
    renderer.render(400, 300)
    assert renderer.instance_count == 1
    assert renderer.pick(200, 150) == 1000
    renderer.set_settings(RenderingSettings(sleeping_objects=2, trigger_shapes=2))
    renderer.render(400, 300)
    assert renderer.instance_count == 0  # Original stored Selected excludes selected triggers.
    assert renderer.pick(200, 150) is None
    renderer.set_settings(RenderingSettings(sleeping_objects=2, trigger_shapes=3))
    renderer.render(400, 300)
    assert renderer.pick(200, 150) == 1000


def test_legacy_cloth_billboards_and_visibility(renderer):
    geometry = snapshot("cloth_physx34.pxd2")
    assert set(geometry.particle_radius_modes[geometry.kinds == 7]) == {1}
    renderer.set_settings(RenderingSettings(particle_size=2))
    renderer.set_snapshot(geometry, True)
    renderer.render(500, 400)
    small = renderer.framebuffer.read(components=3)
    assert all(mesh.indices is None for kind, mesh in renderer._groups if kind == 7)
    renderer.set_settings(RenderingSettings(particle_size=8))
    renderer.render(500, 400)
    assert renderer.framebuffer.read(components=3) != small
    renderer.set_settings(RenderingSettings(show_particles=False))
    renderer.render(500, 400)
    assert 7 not in {kind for kind, _ in renderer._groups}
    assert renderer.context.error == "GL_NO_ERROR"


def test_ovd_particle_radius_is_physical_above_legacy_threshold(renderer, animated_capture):
    with tempfile.TemporaryDirectory() as directory:
        session = Session(str(animated_capture), directory)
        session.seek(session.scenes()[0].id, 1)
        geometry = session.geometry_snapshot()
    assert set(geometry.particle_radius_modes[geometry.kinds == 7]) == {2}
    particles = set(map(int, geometry.object_ids[geometry.kinds == 7]))
    renderer.set_settings(RenderingSettings(particle_size=1))
    renderer.set_snapshot(geometry)
    renderer.hidden = set(map(int, geometry.object_ids)) - particles
    renderer.frame_all()
    renderer.render(400, 300)
    physical = renderer.framebuffer.read(components=3)
    assert any(kind == 7 for kind, _ in renderer._groups)
    renderer.set_settings(RenderingSettings(particle_size=9))
    renderer.render(400, 300)
    assert renderer.framebuffer.read(components=3) == physical
    renderer.set_settings(RenderingSettings(particle_size=0))
    renderer.render(400, 300)
    assert renderer.instance_count == 0
