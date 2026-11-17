# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare mesh face pixels to the unchanged original Qt GLSL programs."""
from pathlib import Path
from types import SimpleNamespace

import moderngl
import numpy as np
import pytest

from ovd_ui.render_settings import FaceCulling, RenderingSettings
from test_mesh_face_normals import picking_pixels
from test_renderer import renderer

ROOT = Path(__file__).resolve().parents[2]
SIZE = (400, 300)
TINT = np.array([.39, .68, .83, 1.], dtype="f4")


def patches(heightfields):
    count = len(heightfields)
    vertices = np.array([[-.8, -.8, 0], [.8, -.8, 0], [.8, .8, 0], [-.8, .8, 0]], dtype="f4")
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="u4")
    transforms = np.repeat(np.eye(4, dtype="f4")[None], count, axis=0)
    transforms[:, :3, :3] = [[1.1, .1, 0], [0, .85, .15], [.25, .1, .7]]
    transforms[:, 0, 3] = (np.arange(count) - (count - 1) / 2) * 2.2
    bounds = []
    for matrix in transforms:
        points = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        bounds.append([points.min(axis=0), points.max(axis=0)])
    return SimpleNamespace(object_ids=np.arange(1001, 1001 + count, dtype="i4"),
        shape_ids=np.arange(2001, 2001 + count, dtype="i4"),
        geometry_ids=np.arange(3001, 3001 + count, dtype="i4"),
        scene_ids=np.ones(count, dtype="i4"), kinds=np.full(count, 5, dtype="u1"),
        resource_ids=np.arange(10, 10 + count, dtype="u8"), transforms=transforms,
        dimensions=np.ones((count, 3), dtype="f4"), sleeping=np.zeros(count, dtype="u1"),
        double_sided=np.asarray(heightfields, dtype="u1"), bounds=np.asarray(bounds, dtype="f4"),
        meshes=[SimpleNamespace(id=10 + i, topology_id=10 + i, vertices=vertices.copy(),
                                indices=indices.copy()) for i in range(count)], diagnostics=[])


def qt_pixels(context, camera, geometry):
    """Compile unmodified Qt vertex/fragment files, including the HF view path.

    The oracle does not reuse the candidate lighting expression or transformed
    light. It supplies View/Projection/InstanceMatrix exactly as the Qt draw
    paths do; the final array is flipped to the OVUI texture convention.
    """
    texture = context.texture(SIZE, 4)
    depth = context.depth_renderbuffer(SIZE)
    framebuffer = context.framebuffer([texture], depth)
    resources = [framebuffer, depth, texture]
    try:
        with context.scope(framebuffer=framebuffer, enable_only=moderngl.DEPTH_TEST):
            context.viewport = (0, 0, *SIZE)
            framebuffer.clear(.2, .2, .2, 1.)
            for row, mesh in enumerate(geometry.meshes):
                hf = bool(geometry.double_sided[row])
                name = "hf" if hf else "simple"
                folder = Path(__file__).parent / "reference_data/shaders"
                program = context.program(vertex_shader=(folder / (name + ".vert")).read_text(),
                                          fragment_shader=(folder / (name + ".frag")).read_text())
                resources.append(program)
                program["View"].write(camera.view_matrix().T.tobytes())
                program["Projection"].write(camera.projection(*SIZE).T.tobytes())
                vb, ib = context.buffer(mesh.vertices.tobytes()), context.buffer(mesh.indices.tobytes())
                resources.extend([vb, ib])
                attributes = [(vb, "3f", "Position")]
                if hf:
                    program["InstanceMatrix"].write(geometry.transforms[row].T.tobytes())
                    program["Color"].value = tuple(TINT)
                    light = program["LightDir"]
                    light.write(np.tile(np.array([1., 2., 3.], dtype="f4"), (light.array_length, 1)).tobytes())
                else:
                    instance = context.buffer(np.concatenate((TINT, geometry.transforms[row].T.ravel())).tobytes())
                    resources.append(instance)
                    attributes.append((instance, "4f 16f /i", "InstanceColor", "InstanceTransformMatrix"))
                vao = context.vertex_array(program, attributes, ib, index_element_size=4)
                resources.append(vao)
                vao.render(moderngl.TRIANGLES, instances=1)
            result = np.frombuffer(framebuffer.read(components=3), dtype="u1").reshape(SIZE[1], SIZE[0], 3)
            return result[::-1].copy()
    finally:
        for resource in reversed(resources):
            resource.release()


def centers(view, geometry):
    result = []
    for row, matrix in enumerate(geometry.transforms):
        clip = view._view_projection() @ matrix[:, 3]
        ndc = clip[:2] / clip[3]
        x = int((ndc[0] + 1) * SIZE[0] / 2)
        y = int((1 - ndc[1]) * SIZE[1] / 2)
        assert view.pick(x, y) == int(geometry.object_ids[row])
        result.append((x, y))
    return result


@pytest.mark.parametrize("alpha", [1., .65])
@pytest.mark.parametrize("saved", [None, {}, {"anti_aliasing": True}, {"face_culling": 1, "wireframe": 0}])
def test_default_settings_keep_thin_mesh_visible_from_both_sides(renderer, alpha, saved):
    geometry = patches([False])
    renderer.set_settings(RenderingSettings() if saved is None else RenderingSettings.from_preferences(saved))
    renderer.set_snapshot(geometry)
    renderer.opacity = alpha
    for side in (-1, 1):
        renderer.camera.look_at((0, 0, side * 7), (0, 0, 0), (0, 1, 0))
        renderer.render(*SIZE, background=(.2, .2, .2, 1.))
        pixel = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
        assert pixel != bytes([51, 51, 51]), "Default settings must render either side of a thin mesh"
        assert renderer.pick(200, 150) == 1001
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("culling,visible_sides", [
    (FaceCulling.NONE, [True, True]),
    (FaceCulling.NORMAL, [False, True]),
    (FaceCulling.INVERSE, [True, False]),
])
def test_saved_culling_choice_controls_visible_mesh_faces(renderer, culling, visible_sides):
    renderer.set_settings(RenderingSettings.from_preferences({"face_culling": int(culling), "schema": 2}))
    renderer.set_snapshot(patches([False]))
    for side, visible in zip((-1, 1), visible_sides):
        renderer.camera.look_at((0, 0, side * 7), (0, 0, 0), (0, 1, 0))
        renderer.render(*SIZE, background=(.2, .2, .2, 1.))
        pixel = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
        assert (pixel != bytes([51, 51, 51])) == visible
        assert renderer.pick(200, 150) == (1001 if visible else None)
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("heightfield", [False, True])
@pytest.mark.parametrize("with_settings", [False, True])
@pytest.mark.parametrize("left_handed", [False, True])
def test_actual_qt_shaders_match_faces_under_camera_rotation_and_handedness(
        renderer, heightfield, with_settings, left_handed):
    geometry = patches([heightfield])
    if with_settings:
        renderer.set_settings(RenderingSettings(face_culling=0))
    renderer.set_snapshot(geometry)
    mesh = next(iter(renderer.meshes.values()))
    vertices, indices = mesh.vertices.read(), mesh.indices.read()
    renderer.camera.left_handed = left_handed
    colors = []
    for eye in ((0, 0, 7), (4, 2, 7)):
        renderer.camera.look_at(eye, (0, 0, 0), (0, 1, 0))
        renderer.render(*SIZE, background=(.2, .2, .2, 1.))
        point = centers(renderer, geometry)[0]
        actual = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(SIZE[1], SIZE[0], 3)
        reference = qt_pixels(renderer.context, renderer.camera, geometry)
        x, y = point
        np.testing.assert_allclose(actual[y-2:y+3, x-2:x+3], reference[y-2:y+3, x-2:x+3], atol=1, rtol=0)
        colors.append(actual[y, x].astype(int))
        assert next(iter(renderer.meshes.values())) is mesh
        assert mesh.vertices.read() == vertices and mesh.indices.read() == indices
        assert renderer.context.error == "GL_NO_ERROR"
    if heightfield:
        assert np.max(np.abs(colors[0] - colors[1])) > 5
    else:
        np.testing.assert_allclose(colors[0], colors[1], atol=1, rtol=0)


@pytest.mark.parametrize("heightfields", [(False, True, False), (True, False, True)])
def test_consecutive_mesh_and_heightfield_groups_do_not_inherit_light_uniform(renderer, heightfields):
    geometry = patches(heightfields)
    renderer.set_settings(RenderingSettings(face_culling=0))
    renderer.set_snapshot(geometry)
    assert [mesh.double_sided for kind, mesh in renderer._groups] == list(heightfields)
    for eye in ((0, 0, 9), (4, 3, 9)):
        renderer.camera.look_at(eye, (0, 0, 0), (0, 1, 0))
        renderer.render(*SIZE, background=(.2, .2, .2, 1.))
        actual = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(SIZE[1], SIZE[0], 3)
        reference = qt_pixels(renderer.context, renderer.camera, geometry)
        for x, y in centers(renderer, geometry):
            np.testing.assert_allclose(actual[y-2:y+3, x-2:x+3], reference[y-2:y+3, x-2:x+3], atol=1, rtol=0)
        assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("heightfield", [False, True])
def test_mesh_transparency_preserves_alpha_and_picking_with_qt_lit_color(renderer, heightfield):
    geometry = patches([heightfield])
    renderer.set_settings(RenderingSettings(face_culling=0))
    renderer.set_snapshot(geometry)
    renderer.camera.look_at((3, 2, 7), (0, 0, 0), (0, 1, 0))
    renderer.render(*SIZE, background=(.2, .2, .2, 1.))
    x, y = centers(renderer, geometry)[0]
    before_pick = picking_pixels(renderer)
    lit = qt_pixels(renderer.context, renderer.camera, geometry)[y, x].astype(float)
    renderer.opacity = .65
    renderer.render(*SIZE, background=(.2, .2, .2, 1.))
    actual = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(SIZE[1], SIZE[0], 3)[y, x]
    np.testing.assert_allclose(actual, lit * .65 + 51 * .35, atol=2, rtol=0)
    assert picking_pixels(renderer) == before_pick
    assert renderer.context.error == "GL_NO_ERROR"
