# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real OpenGL integration: native geometry, instancing, and object-ID picking."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import moderngl
import glcontext
import numpy as np
import pytest
from _pvd_native import Session
from ovd_ui.design_tokens import STAGE_GRID_RGBA, rgba as design_rgba
from ovd_ui.renderer import INSTANCE_DTYPE, ViewportRenderer, mesh_data, mesh_data_batch
from ovd_ui.palette import ColorPalette, DEFAULT_PALETTE_PATH
from ovd_ui.render_settings import RenderingSettings

ROOT = Path(__file__).resolve().parents[2]
NEEDS_X = pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="GLX requires an X display")


class TextureConsumer:
    def destroy(self):
        pass

    def set_image_data(self, *_):
        pass


@pytest.fixture
def renderer():
    # These are integration tests; no fallback to a mock GL implementation.
    options = {} if sys.platform == "win32" else {"backend": "egl"}
    context = moderngl.create_standalone_context(require=330, **options)
    result = ViewportRenderer(context, TextureConsumer())
    yield result
    result.close()
    context.release()


def snapshot(path, frame=1):
    with tempfile.TemporaryDirectory() as temporary:
        session = Session(str(ROOT / "test/data" / path), temporary)
        session.seek(session.scenes()[0].id, frame)
        return session.geometry_snapshot()  # Owned arrays outlive native Session.


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime GL/EGL libraries")
@pytest.mark.parametrize("backend", ["egl", pytest.param("glx", marks=NEEDS_X),
                                     pytest.param("glx_without_egl", marks=NEEDS_X)])
def test_renderer_attaches_without_opengl_development_libraries(monkeypatch, backend):
    factory = glcontext.get_backend_by_name("egl") if backend == "egl" else glcontext.default_backend()
    options = {"libegl": "libEGL.so.1"} if backend == "egl" else {"libx11": "libX11.so.6"}
    native = factory(glversion=330, mode="standalone", libgl="libGL.so.1", **options)
    original_cdll = ctypes.CDLL

    def runtime_only(name, *args, **kwargs):
        if name in ("libEGL.so", "libGL.so"):
            raise OSError(f"Development-only library unavailable: {name}")
        if backend == "glx_without_egl" and name == "libEGL.so.1":
            raise OSError("EGL runtime unavailable")
        return original_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", runtime_only)
    monkeypatch.setattr(moderngl._store, "default_context", None)
    try:
        with native:
            result = ViewportRenderer(provider=TextureConsumer())
            try:
                geometry = snapshot("test_scene.ovd")
                result.set_snapshot(geometry, True)
                result.render(400, 300)
                assert result.pick(200, 150) == int(geometry.object_ids[0])
                assert any(result.framebuffer.read(components=3))
                assert result.context.error == "GL_NO_ERROR"
            finally:
                result.close()
                result.context.release()
    finally:
        native.release()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime GL/EGL libraries")
def test_renderer_reports_missing_runtime_libraries(monkeypatch):
    def unavailable(name, *args, **kwargs):
        raise OSError(f"Runtime unavailable: {name}")

    monkeypatch.setattr(ctypes, "CDLL", unavailable)
    with pytest.raises(RuntimeError, match=r"current OpenGL context: libEGL.so.1:.*libGL.so.1:"):
        ViewportRenderer(provider=TextureConsumer())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime GL/EGL libraries")
def test_renderer_reports_no_current_context(monkeypatch):
    def no_context():
        return 0

    library = SimpleNamespace(eglGetCurrentContext=no_context, glXGetCurrentContext=no_context)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: library)
    with pytest.raises(RuntimeError, match=r"libEGL\.so\.1: no current egl context; libGL\.so\.1: no current glX context"):
        ViewportRenderer(provider=TextureConsumer())


def test_native_sphere_pick_selection_and_visibility(renderer):
    geometry = snapshot("test_scene.ovd")
    object_id = int(geometry.object_ids[0])
    renderer.set_snapshot(geometry, True)
    renderer.render(400, 300)
    assert renderer.pick(200, 150) == object_id
    assert renderer.pick(0, 0) is None
    normal = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
    renderer.selected = {object_id}
    renderer.render(400, 300)
    selected = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
    assert selected != normal
    renderer.hidden = {object_id}
    renderer.render(400, 300)
    assert renderer.instance_count == 0
    assert renderer.pick(200, 150) is None


def test_instances_batch_one_thousand_spheres_in_one_draw(renderer):
    native = snapshot("test_scene.ovd")
    count = 1000
    fields = {name: np.repeat(getattr(native, name), count, axis=0)
              for name in ("object_ids", "shape_ids", "geometry_ids", "scene_ids", "kinds",
                           "resource_ids", "transforms", "dimensions", "bounds", "sleeping")}
    fields["object_ids"] = np.arange(count, dtype="i4")
    # Place a 1000-sphere grid far from its recorded fixture origin.
    centers = np.array([(i % 10, i // 10 % 10, i // 100) for i in range(count)], dtype="f4") * 30
    fields["transforms"][:, :3, 3] = centers
    fields["bounds"][:, 0] = centers - 10
    fields["bounds"][:, 1] = centers + 10
    geometry = SimpleNamespace(**fields, meshes=[], diagnostics=[])
    renderer.set_snapshot(geometry, True)
    renderer.render(640, 480)
    assert renderer.instance_count == 1000
    assert renderer.draw_count == 1
    assert len(renderer.meshes) == 1


def test_cached_membership_updates_gpu_poses_and_invalidates_visibility(renderer):
    geometry = duplicate_geometry(snapshot("test_scene.ovd"))
    renderer.set_snapshot(geometry, True)
    original = renderer._group_rows
    geometry.transforms[1, 0, 3] += 35
    renderer.set_snapshot(geometry)
    assert renderer._group_rows is original
    mesh = next(iter(renderer.meshes.values()))
    uploaded = np.frombuffer(mesh.instances.read(), dtype=INSTANCE_DTYPE)
    assert np.array_equal(uploaded["matrix"], geometry.transforms.transpose(0, 2, 1))
    renderer.hidden = {1000}
    renderer.render(400, 300)
    assert renderer._group_rows is not original
    assert renderer.instance_count == 1
    renderer.hidden.clear()
    renderer.alpha_by_id = {1001: .5}
    renderer.render(400, 300)
    assert len(renderer._group_rows) == 2
    assert {mesh.transparent for _, mesh in renderer._groups} == {True, False}
    renderer.clear()
    assert renderer._group_signature is None


def test_real_ue4_membership_reused_across_frame_advance(renderer):
    with tempfile.TemporaryDirectory() as directory:
        session = Session(str(ROOT / "test/data/ue4_shooter.pxd2"), directory)
        session.seek(686661696, 30)
        renderer.scene_id = 1143
        renderer.set_snapshot(session.geometry_snapshot(), True)
        original = renderer._group_rows
        session.seek(686661696, 31)
        geometry = session.geometry_snapshot()
        renderer.set_snapshot(geometry)
        assert renderer._group_rows is original
        renderer.render(400, 300)
        assert renderer.instance_count == 1080
        assert renderer.draw_count == 1026
        assert renderer.context.error == "GL_NO_ERROR"


def test_batched_real_ue4_normals_are_bit_exact():
    with tempfile.TemporaryDirectory() as directory:
        session = Session(str(ROOT / "test/data/ue4_shooter.pxd2"), directory)
        session.seek(686661696, 30)
        geometry = session.geometry_snapshot()
        resources = {int(value) for value in geometry.resource_ids[geometry.kinds == 5]}
        entries = [(int(mesh.id), mesh.vertices, mesh.indices)
                   for mesh in geometry.meshes if int(mesh.id) in resources]
        assert len(entries) > 1000
        actual = mesh_data_batch(entries)
        for key, vertices, indices in entries:
            reference_data, reference_indices = mesh_data(vertices, indices, 5)
            assert np.array_equal(actual[key][0], reference_data)
            assert np.array_equal(actual[key][1], reference_indices)


def test_mesh_batch_is_bounded_and_large_mesh_keeps_existing_path(renderer):
    small = SimpleNamespace(id=10, topology_id=10, vertices=np.zeros((3, 3), dtype="f4"),
                            indices=np.array([0, 1, 2], dtype="u4"))
    large = SimpleNamespace(id=11, topology_id=11, vertices=np.zeros((65537, 3), dtype="f4"),
                            indices=np.array([0, 1, 2], dtype="u4"))
    renderer._group_rows = [(5, 10, False, np.array([0])), (5, 11, False, np.array([1]))]
    # A single small mesh cannot batch past the next oversized mesh.
    assert renderer._prepare_mesh_batch(0, {10: small, 11: large}) == {}
    assert renderer._prepare_mesh_batch(1, {10: small, 11: large}) == {}
    second = SimpleNamespace(id=12, topology_id=12, vertices=small.vertices, indices=small.indices)
    renderer._group_rows.insert(1, (5, 12, False, np.array([2])))
    batch = renderer._prepare_mesh_batch(0, {10: small, 11: large, 12: second})
    assert set(batch) == {(5, 10, False), (5, 12, False)}


def test_mixed_native_meshes_and_animated_legacy_cloth(renderer):
    mixed = snapshot("multi_geom.ovd", 2)
    renderer.set_snapshot(mixed, True)
    renderer.render(640, 480)
    assert renderer.instance_count >= 11
    assert {key[0] for key in renderer.meshes} >= {1, 2, 3, 4, 5}
    assert renderer.context.error == "GL_NO_ERROR"
    cloth = snapshot("cloth_physx34.pxd2")
    renderer.clear()  # Resource keys are session-local.
    renderer.set_snapshot(cloth, True)
    renderer.render(640, 480)
    assert 6 in {key[0] for key in renderer.meshes}
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("light", [False, True])
@pytest.mark.parametrize("identity_field", ["object_ids", "shape_ids", "geometry_ids"])
def test_selected_plane_highlights_without_changing_other_pixels_or_picking(renderer, light, identity_field, tmp_path):
    session = Session(str(ROOT / "test/data/multi_geom.ovd"), str(tmp_path / "plane"))
    session.seek(session.scenes()[0].id, 2)
    geometry = session.geometry_snapshot()
    class_names = {cls.id: cls.name for cls in session.classes()}
    row = np.flatnonzero(geometry.kinds == 4).item()
    actor = int(geometry.object_ids[row])
    assert class_names[int(geometry.actor_classes[row])] == "PxRigidStatic"
    # Translucent particles correctly change color when the ground behind them
    # changes. Isolate opaque geometry for the unaffected-pixel assertion.
    renderer.hidden = set(map(int, geometry.object_ids[geometry.kinds == 7]))
    renderer.set_settings(RenderingSettings())
    renderer.set_palette(ColorPalette(DEFAULT_PALETTE_PATH), class_names)
    renderer.set_snapshot(geometry, True)
    theme = "light" if light else "dark"
    options = dict(light=light, ground_color=design_rgba("elevated" if light else "surface", theme),
                   grid_contrast=.025, grid_rgba=STAGE_GRID_RGBA[theme])
    renderer.render(400, 300, **options)
    before = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3).copy()
    renderer.pick(0, 0)
    ids = np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4").copy()
    plane = ids == actor + 1
    assert np.count_nonzero(plane) > 1000
    renderer.selected = {int(getattr(geometry, identity_field)[row])}
    renderer.render(400, 300, **options)
    after = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3)
    assert np.abs(after[plane].astype("i2") - before[plane].astype("i2")).mean() > 20
    assert np.array_equal(after[~plane], before[~plane])
    renderer.pick(0, 0)
    assert np.array_equal(np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4"), ids)
    renderer.selected.clear()
    renderer.render(400, 300, **options)
    assert np.array_equal(np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(-1, 3), before)
    assert renderer.context.error == "GL_NO_ERROR"


def duplicate_geometry(native, count=2):
    fields={name:np.repeat(getattr(native,name),count,axis=0)
            for name in ("object_ids","shape_ids","geometry_ids","scene_ids","kinds",
                         "resource_ids","transforms","dimensions","bounds","sleeping")}
    fields["object_ids"]=np.arange(1000,1000+count,dtype="i4")
    fields["shape_ids"]=np.arange(2000,2000+count,dtype="i4")
    fields["geometry_ids"]=np.arange(3000,3000+count,dtype="i4")
    return SimpleNamespace(**fields,meshes=[],diagnostics=[])


def test_transparency_endpoints_and_order_independent_composite(renderer):
    geometry=duplicate_geometry(snapshot("test_scene.ovd"))
    renderer.set_snapshot(geometry,True)
    renderer.selected={1001}
    renderer.alpha_by_id={1000:.35,1001:.35}
    renderer.render(400,300)
    first=np.frombuffer(renderer.framebuffer.read(components=3),dtype="u1").copy()
    fields={name:getattr(geometry,name)[::-1].copy() for name in vars(geometry)
            if isinstance(getattr(geometry,name),np.ndarray)}
    reverse=SimpleNamespace(**fields,meshes=[],diagnostics=[])
    renderer.set_snapshot(reverse)
    renderer.render(400,300)
    second=np.frombuffer(renderer.framebuffer.read(components=3),dtype="u1")
    assert np.max(np.abs(first.astype(int)-second.astype(int)))<=2
    renderer.alpha_by_id={1000:0.,1001:0.}
    renderer.render(400,300)
    assert renderer.pick(200,150) is None
    center=renderer.framebuffer.read(viewport=(200,150,1,1),components=3)
    corner=renderer.framebuffer.read(viewport=(0,0,1,1),components=3)
    assert center==corner
    assert renderer.context.error=="GL_NO_ERROR"


def test_transparent_pass_preserves_opaque_depth(renderer):
    geometry=duplicate_geometry(snapshot("test_scene.ovd"))
    geometry.dimensions[1]=5. # Inside the opaque sphere, fully occluded.
    renderer.set_snapshot(geometry,True)
    renderer.hidden={1001}
    renderer.render(400,300)
    opaque=renderer.framebuffer.read(viewport=(200,150,1,1),components=3)
    renderer.hidden=set()
    renderer.selected={1001}
    renderer.alpha_by_id={1001:.8}
    renderer.render(400,300)
    assert renderer.framebuffer.read(viewport=(200,150,1,1),components=3)==opaque
    assert renderer.context.error=="GL_NO_ERROR"


def test_captured_debug_overlays_change_pixels_without_gl_errors(renderer):
    with tempfile.TemporaryDirectory() as directory:
        session=Session(str(ROOT/"test/data/test_scene.ovd"),directory)
        scene=session.scenes()[0]
        session.seek(scene.id,1)
        renderer.set_snapshot(session.geometry_snapshot(),True)
        renderer.render(400,300)
        before=renderer.framebuffer.read(components=3)
        renderer.set_overlays(session.overlay_snapshot(1|2|4|256))
        renderer.render(400,300)
        after=renderer.framebuffer.read(components=3)
        assert before!=after
        assert renderer.overlays.primitive_count>0
        assert renderer.overlays.draw_count<=3
        assert renderer.context.error=="GL_NO_ERROR"


def test_exact_captured_sweep_geometry_uses_wireframe_query_pass(renderer):
    with tempfile.TemporaryDirectory() as directory:
        session=Session(str(ROOT/"test/data/ue4_shooter.pxd2"),directory)
        session.seek(session.scenes()[0].id,1)
        renderer.set_snapshot(session.geometry_snapshot(),True)
        renderer.set_overlays(session.overlay_snapshot(64))
        renderer.render(400,300)
        query=renderer.query_renderer
        assert query is not None
        assert query.instance_count==2
        assert query.snapshot.object_ids.tolist()==[-7,-7]
        assert query.wireframe
        assert query.draw_count>0
        assert renderer.context.error=="GL_NO_ERROR"
