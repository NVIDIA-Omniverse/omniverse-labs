# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GL surface shading, comparing interpolated and actual face normals."""
from types import SimpleNamespace

import numpy as np
import pytest
import moderngl

import ovd_ui.renderer as rendering
from ovd_ui.render_settings import RenderingSettings
from test_renderer import TextureConsumer, renderer, snapshot


def smooth_reference(context, monkeypatch):
    """Keep the same light/material rule while removing the face override."""
    fragment = rendering.FRAGMENT
    start = fragment.index("        if (kind == 5) {")
    stop = fragment.index("        float light =", start)
    with monkeypatch.context() as patch:
        patch.setattr(rendering, "FRAGMENT", fragment[:start] + fragment[stop:])
        return rendering.ViewportRenderer(context, TextureConsumer())


def folded_surface():
    # Two planar patches share a crease. Averaging their indexed normals makes
    # each patch look curved even though every point on it is coplanar.
    vertices = np.array([[-1, -1, 0], [0, -1, 0], [0, 1, 0], [-1, 1, 0],
                         [1, -1, 1], [1, 1, 1]], dtype="f4")
    resource = SimpleNamespace(id=10, topology_id=10, vertices=vertices,
        indices=np.array([0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2], dtype="u4"))
    return SimpleNamespace(object_ids=np.array([1001], dtype="i4"),
        shape_ids=np.array([1002], dtype="i4"), geometry_ids=np.array([1003], dtype="i4"),
        scene_ids=np.array([1], dtype="i4"), kinds=np.array([5], dtype="u1"),
        resource_ids=np.array([10], dtype="u8"), transforms=np.eye(4, dtype="f4")[None],
        dimensions=np.ones((1, 3), dtype="f4"), sleeping=np.zeros(1, dtype="u1"),
        bounds=np.array([[vertices.min(axis=0), vertices.max(axis=0)]]),
        meshes=[resource], diagnostics=[])


def surface_pixels(view, points):
    rgb = np.frombuffer(view.framebuffer.read(components=3), dtype="u1").reshape(300, 400, 3)
    result = []
    for point in points:
        clip = view._view_projection() @ np.array([*point, 1], dtype="f4")
        ndc = clip[:2] / clip[3]
        x, y = int((ndc[0] + 1) * 200), int((1 - ndc[1]) * 150)
        assert view.pick(x, y) == 1001
        result.append(rgb[y, x].astype(int))
    return np.array(result)


def picking_pixels(view):
    with view.context.scope(framebuffer=view.pick_buffer, enable_only=moderngl.DEPTH_TEST):
        view.context.viewport = (0, 0, *view.size)
        view.pick_buffer.clear(0, 0, 0, 0)
        view._draw(picking=True)
        return view.pick_buffer.read(components=4)


@pytest.mark.parametrize("alpha", [1., .65])
def test_shared_crease_stays_planar_in_opaque_and_transparent_passes(renderer, monkeypatch, alpha):
    old = smooth_reference(renderer.context, monkeypatch)
    try:
        geometry = folded_surface()
        for view in (old, renderer):
            view.set_snapshot(geometry)
            view.camera.look_at((0, 0, 5), (0, 0, 0))
            view.opacity = alpha
            view.render(400, 300)
        points = [(-.8, 0, 0), (-.5, 0, 0), (-.2, 0, 0)]
        flat = surface_pixels(renderer, points)
        smooth = surface_pixels(old, points)
        assert np.ptp(flat, axis=0).max() <= 1
        assert np.ptp(smooth, axis=0).max() >= 8
        other = surface_pixels(renderer, [(.2, 0, .2), (.5, 0, .5), (.8, 0, .8)])
        assert np.ptp(other, axis=0).max() <= 1
        assert np.abs(other.mean(axis=0) - flat.mean(axis=0)).max() >= 8
        assert picking_pixels(renderer) == picking_pixels(old)
        assert renderer.context.error == "GL_NO_ERROR"
    finally:
        old.close()


@pytest.mark.parametrize("capture", ["multi_geom.ovd", "cloth_physx34.pxd2"])
def test_real_nonmesh_primitives_lines_and_particles_are_pixel_identical(renderer, monkeypatch, capture):
    native = snapshot(capture)
    mask = native.kinds != 5
    assert mask.any()
    fields = {name: getattr(native, name)[mask] for name in
        ("object_ids", "shape_ids", "geometry_ids", "scene_ids", "kinds", "resource_ids",
         "transforms", "dimensions", "bounds", "sleeping", "shape_flags", "actor_flags",
         "double_sided", "particle_radius_modes")}
    geometry = SimpleNamespace(**fields, meshes=native.meshes, diagnostics=[])
    old = smooth_reference(renderer.context, monkeypatch)
    try:
        captures = []
        for view in (old, renderer):
            view.set_settings(RenderingSettings(face_culling=0, planes=2))
            view.set_snapshot(geometry, True)
            view.render(400, 300)
            color = view.framebuffer.read(components=4)
            captures.append((color, picking_pixels(view)))
        assert captures[0] == captures[1]
        assert renderer.context.error == "GL_NO_ERROR"
    finally:
        old.close()


def test_pose_and_vertex_revisions_keep_existing_mesh_buffers(renderer):
    geometry = folded_surface()
    renderer.set_snapshot(geometry)
    renderer.camera.look_at((0, 0, 5), (0, 0, 0))
    renderer.render(400, 300)
    mesh = renderer.meshes[(5, 10, 0)]
    vertices, indices, instances = mesh.vertices.read(), mesh.indices.read(), mesh.instances.read()
    geometry.transforms[0, 0, 3] = .1
    renderer.set_snapshot(geometry)
    renderer.render(400, 300)
    assert renderer.meshes[(5, 10, 0)] is mesh
    assert mesh.vertices.read() == vertices and mesh.indices.read() == indices
    assert mesh.instances.read() != instances
    moved = surface_pixels(renderer, [(-.7, 0, 0), (-.4, 0, 0), (-.1, 0, 0)])
    assert np.ptp(moved, axis=0).max() <= 1

    geometry.meshes[0].vertices[4:, 2] = .5
    geometry.meshes[0].id = 11
    geometry.resource_ids[:] = 11
    renderer.set_snapshot(geometry)
    renderer.render(400, 300)
    assert renderer.meshes[(5, 10, 0)] is mesh
    assert mesh.vertices.read() != vertices and mesh.indices.read() == indices
    assert mesh.revision == 11
    assert renderer.context.error == "GL_NO_ERROR"


def test_face_normal_uses_actual_world_surface_after_nonuniform_transform(renderer):
    geometry = folded_surface()
    matrix = geometry.transforms[0]
    matrix[:3, :3] = [[1.5, .1, 0], [0, .8, .2], [.3, 0, .5]]
    renderer.set_snapshot(geometry)
    renderer.camera.look_at((0, 0, 5), (0, 0, 0))
    renderer.render(400, 300)
    points = [(matrix @ [x, 0, 0, 1])[:3] for x in (-.8, -.5, -.2)]
    colors = surface_pixels(renderer, points)
    normal = np.cross(matrix[:3, 0], matrix[:3, 1])
    normal /= np.linalg.norm(normal)
    light = np.array([1., 2., 3.])
    light /= np.linalg.norm(light)
    strength = np.sqrt(abs(np.dot(normal, light)))
    expected = np.array([.39, .68, .83]) * strength * 255
    assert np.ptp(colors, axis=0).max() <= 1
    assert np.max(np.abs(colors - expected)) <= 1
