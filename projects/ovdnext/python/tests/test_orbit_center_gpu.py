# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual GL camera-marker coverage, depth, cache writes and native picking."""
from dataclasses import replace

import numpy as np
import pytest

from ovd_ui.render_settings import RenderingSettings
from ovd_ui.camera import CameraController
from ovd_ui.renderer import INSTANCE_DTYPE
from test_application_alpha import watch_uploads
from test_renderer import renderer, snapshot


def pixels(renderer):
    return renderer.framebuffer.read(components=3)


def picks(renderer):
    renderer.pick(renderer.size[0] // 2, renderer.size[1] // 2)
    return renderer.pick_buffer.read(components=4)


def test_selected_off_axis_marker_matches_actual_orbit_and_zoom(renderer):
    geometry = snapshot("test_scene.ovd")
    renderer.set_settings(RenderingSettings(draw_orbit_center=True, geometry_scale=2., face_culling=0))
    renderer.set_snapshot(geometry)
    renderer.selected = {int(geometry.object_ids[0])}
    bounds = geometry.bounds[geometry.object_ids == geometry.object_ids[0]]
    center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) * .5
    pivot = center.astype("f8") * [2., 2., -2.]
    renderer.camera.left_handed = True
    renderer.camera.position = pivot + [7., 3., 30.]
    control = CameraController(renderer.camera)
    for gesture in (None, "orbit", "zoom"):
        if gesture == "orbit": control.drag(4, 0, 400, 300, 0, pivot=renderer.selection_pivot())
        elif gesture == "zoom": control.wheel(1, pivot=renderer.selection_pivot())
        renderer.render(400, 300)
        record = np.frombuffer(renderer._orbit_mesh.instances.read(), dtype=INSTANCE_DTYPE)
        np.testing.assert_allclose(record["matrix"][0].T[:3, 3], center)
        assert renderer.context.error == "GL_NO_ERROR"


def test_marker_camera_updates_preserve_native_buffers_and_exact_pick_pixels(renderer, monkeypatch):
    geometry = snapshot("test_scene.ovd")
    renderer.set_settings(RenderingSettings(face_culling=0))
    renderer.set_snapshot(geometry, True)
    renderer.camera.distance *= .5  # Pivot in front of the native sphere; eye stays fixed.
    size = float(np.max(geometry.bounds[0, 1] - geometry.bounds[0, 0]))
    renderer.render(400, 300)
    before, ids = pixels(renderer), picks(renderer)
    native_groups = tuple((kind, id(mesh)) for kind, mesh in renderer._groups)
    writes = watch_uploads(renderer, monkeypatch)
    renderer.set_settings(replace(renderer.settings, draw_orbit_center=True))
    renderer.render(400, 300, scene_length=size, gizmo_scale=.5)
    assert pixels(renderer) != before
    assert picks(renderer) == ids and writes == []
    assert native_groups == tuple((kind, id(mesh)) for kind, mesh in renderer._groups)
    marker = renderer._orbit_mesh
    marker_topology = (marker.vertices.glo, marker.indices.glo, marker.vao.glo)
    marker_data = marker.instances.read()
    renderer.camera.turn(13., -8., orbit=True)
    renderer.render(400, 300, scene_length=size, gizmo_scale=.5)
    assert writes == [] and renderer._orbit_mesh is marker
    assert (marker.vertices.glo, marker.indices.glo, marker.vao.glo) == marker_topology
    assert marker.instances.read() == marker_data  # Orbit retains its pivot.
    renderer.camera.pan(35., 13., 300)
    renderer.render(400, 300, scene_length=size, gizmo_scale=.5)
    assert marker.instances.read() != marker_data and writes == []
    renderer.set_settings(replace(renderer.settings, geometry_scale=1.1))
    renderer.render(400, 300, scene_length=size, gizmo_scale=.5)
    assert writes  # Excluding the aid's flag must retain other invalidation.
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("msaa", [False, True])
def test_marker_obeys_opaque_depth_but_is_visible_through_transparency(renderer, msaa):
    geometry = snapshot("test_scene.ovd")
    radius = float(geometry.dimensions[0, 0])
    renderer.set_settings(RenderingSettings(face_culling=0, anti_aliasing=msaa))
    renderer.set_snapshot(geometry, True)
    renderer.render(400, 300)
    opaque = pixels(renderer)
    renderer.set_settings(replace(renderer.settings, draw_orbit_center=True))
    renderer.render(400, 300, scene_length=radius, gizmo_scale=.5)
    assert pixels(renderer) == opaque  # Marker lies entirely inside opaque sphere.
    renderer.opacity = .4
    renderer.render(400, 300, scene_length=radius, gizmo_scale=.5)
    through = pixels(renderer)
    ids = picks(renderer)
    renderer.set_settings(replace(renderer.settings, draw_orbit_center=False))
    renderer.render(400, 300, scene_length=radius, gizmo_scale=.5)
    assert pixels(renderer) != through
    assert picks(renderer) == ids
    assert renderer.context.error == "GL_NO_ERROR"


def test_marker_has_no_native_identity_or_selection_tint_and_closes_its_buffers(renderer):
    geometry = snapshot("test_scene.ovd")
    renderer.set_snapshot(geometry, True)
    renderer.set_settings(RenderingSettings(draw_orbit_center=True))
    renderer.hidden = set(map(int, geometry.object_ids))
    renderer.render(400, 300, scene_length=5.)
    wire, ids = pixels(renderer), picks(renderer)
    assert not any(ids) and renderer.instance_count == 0 and renderer.draw_count == 1
    marker = renderer._orbit_mesh
    renderer.selected = set(renderer.hidden)
    renderer.all_transparent = True
    renderer.render(400, 300, scene_length=5.)
    assert pixels(renderer) == wire and picks(renderer) == ids
    renderer.clear()
    renderer.render(400, 300, scene_length=5.)
    assert pixels(renderer) != wire and renderer.pick(200, 150) is None
    assert renderer._orbit_mesh is marker  # Context cache can survive a capture.
    renderer.close()
    assert renderer._orbit_mesh is None
