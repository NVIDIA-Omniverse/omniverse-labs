# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application appearance updates retain exact GPU results without duplicate writes."""
from capture_semantics import test_scene_roles, unique_instance_id
from types import SimpleNamespace

import numpy as np
from _pvd_native import Session

from ovd_ui.application import Application
from ovd_ui.interaction import InteractionState, TransparencyMode
from ovd_ui.render_settings import RenderingSettings
from ovd_ui.renderer import INSTANCE_DTYPE
from test_renderer import ROOT, renderer, snapshot


class CountedBuffer:
    """Observe real ModernGL writes; every operation reaches the original buffer."""
    def __init__(self, buffer, writes):
        self.buffer, self.writes = buffer, writes

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def write(self, data, *args, **kwargs):
        self.writes.append(bytes(data))
        return self.buffer.write(data, *args, **kwargs)


def watch_uploads(renderer, monkeypatch):
    writes = []

    def watch(mesh):
        if not isinstance(mesh.instances, CountedBuffer):
            mesh.instances = CountedBuffer(mesh.instances, writes)
        return mesh

    for mesh in renderer.meshes.values():
        watch(mesh)
    original = renderer._mesh
    monkeypatch.setattr(renderer, "_mesh", lambda *args, **kwargs: watch(original(*args, **kwargs)))
    return writes


def update_lifetimes(app, geometry):
    ids = set(int(value) for name in ("object_ids", "shape_ids", "geometry_ids", "scene_ids")
              for value in getattr(geometry, name) if value > 0)
    owners = {}
    for actor, scene in zip(geometry.object_ids, geometry.scene_ids):
        if scene > 0:
            owners.setdefault(int(actor), set()).add(int(scene))
    app.interaction.update_frame(ids, ids, owners=owners,
                                scene_ids=set(map(int, geometry.scene_ids)) - {0})


def application(renderer, geometry):
    app = Application.__new__(Application)
    app.renderer, app.interaction = renderer, InteractionState()
    renderer.set_settings(RenderingSettings(face_culling=0))
    update_lifetimes(app, geometry)
    renderer.set_snapshot(geometry, True)
    app._sync_interaction()
    renderer.render(400, 300)
    return app


def pixels(renderer):
    return renderer.framebuffer.read(components=3)


def uploaded(renderer):
    return np.concatenate([np.frombuffer(mesh.instances.read(), dtype=INSTANCE_DTYPE)[:mesh.count]
                           for _, mesh in renderer._groups])


def test_owned_alpha_reuse_skips_second_upload_but_new_native_pose_uploads(renderer, monkeypatch, tmp_path):
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    scene = source.scenes()[0].id
    source.seek(scene, 1)
    first = source.geometry_snapshot()
    actor=unique_instance_id(source,'PxRigidDynamic')
    app = application(renderer, first)
    original_alpha = renderer.instance_alpha
    assert original_alpha.dtype == np.dtype("f4") and original_alpha.flags.owndata
    assert renderer.pick(200, 150) == actor
    first_pixels = pixels(renderer)
    writes = watch_uploads(renderer, monkeypatch)

    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is original_alpha
    assert writes == [] and pixels(renderer) == first_pixels

    source.seek(scene, 2)
    second = source.geometry_snapshot()
    assert not np.array_equal(second.transforms, first.transforms)
    renderer.set_snapshot(second)
    assert len(writes) == 1  # The new native pose must reach the actual GL buffer.
    update_lifetimes(app, second)
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is original_alpha
    assert len(writes) == 1  # Drawing after synchronization adds no second upload.
    records = uploaded(renderer)
    assert np.array_equal(records["matrix"], second.transforms.transpose(0, 2, 1))
    assert pixels(renderer) != first_pixels
    assert renderer.pick(200, 150) is None
    assert len(writes) == 1  # Picking also reuses the already prepared pose.

    source.seek(scene, 3)
    empty = source.geometry_snapshot()
    renderer.set_snapshot(empty)
    update_lifetimes(app, empty)
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is not original_alpha
    assert renderer.instance_alpha.size == 0 and renderer.instance_count == 0
    assert renderer.pick(200, 150) is None
    assert renderer.context.error == "GL_NO_ERROR"


def test_retained_alpha_still_invalidates_selection_visibility_and_transparency(renderer, monkeypatch):
    geometry=snapshot("test_scene.ovd")
    actors=set(map(int,geometry.object_ids))
    assert len(actors)==1
    actor=actors.pop()
    app = application(renderer, geometry)
    original_alpha = renderer.instance_alpha
    normal = pixels(renderer)
    writes = watch_uploads(renderer, monkeypatch)

    app.interaction.select([actor])
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is original_alpha
    assert len(writes) == 1 and pixels(renderer) != normal
    selected = pixels(renderer)
    assert renderer.pick(200, 150) == actor

    app.interaction.set_visible([actor], False)
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is original_alpha
    assert renderer.instance_count == 0 and renderer.pick(200, 150) is None
    app.interaction.show_all()
    app._sync_interaction()
    renderer.render(400, 300)
    assert pixels(renderer) == selected and renderer.pick(200, 150) == actor

    app.interaction.set_transparency(TransparencyMode.SELECTED, .7)
    app._sync_interaction()
    translucent_alpha = renderer.instance_alpha
    assert translucent_alpha is not original_alpha
    assert np.array_equal(translucent_alpha, np.array([.3], dtype="f4"))
    renderer.render(400, 300)
    assert pixels(renderer) != selected
    assert uploaded(renderer)["color"][0, 3] == np.float32(.3)
    assert renderer.pick(200, 150) == actor

    writes.clear()
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is translucent_alpha and writes == []
    app.interaction.set_transparency(TransparencyMode.SELECTED, 1.)
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.pick(200, 150) is None
    assert renderer.context.error == "GL_NO_ERROR"


def test_same_size_reordered_native_rows_recompute_tagged_alpha(renderer,tmp_path):
    from capture_semantics import dynamic_geometry_owner
    source=Session(str(ROOT/'test/data/multi_geom.ovd'),str(tmp_path))
    actor=dynamic_geometry_owner(source,2,30)
    native=source.presentation(frame=30).geometry_snapshot()
    app = application(renderer, native)
    app.interaction.set_transparent([actor], cascade=False)
    app.interaction.set_transparency(TransparencyMode.TAGGED, .65)
    app._sync_interaction()
    renderer.render(400, 300)
    previous = renderer.instance_alpha
    # Owned native rows are permuted together to exercise equal-size identity
    # changes without inventing geometry or mutating the native snapshot.
    fields = {name: getattr(native, name)[::-1].copy() for name in
              ("object_ids", "shape_ids", "geometry_ids", "scene_ids", "kinds", "resource_ids",
               "transforms", "dimensions", "bounds", "sleeping", "shape_flags", "double_sided",
               "particle_radius_modes")}
    reordered = SimpleNamespace(**fields, meshes=native.meshes, diagnostics=[])
    renderer.set_snapshot(reordered)
    app._sync_interaction()
    assert renderer.instance_alpha is not previous
    assert np.array_equal(renderer.instance_alpha, previous[::-1])
    renderer.render(400, 300)
    records = uploaded(renderer)
    assert np.all(records["color"][records["id"] == actor, 3] == np.float32(.35))
    assert np.all(records["color"][records["id"] != actor, 3] == np.float32(1.))
    assert renderer.context.error == "GL_NO_ERROR"


def test_unchanged_alpha_does_not_hide_camera_dependent_plane_upload(renderer, monkeypatch):
    native = snapshot("multi_geom.ovd", 30)
    app = application(renderer, native)
    row = np.flatnonzero(native.kinds == 4)[0]
    plane = int(native.object_ids[row])
    others = set(map(int, native.object_ids)) - {plane}
    app.interaction.set_visible(others, False, cascade=False)
    app._sync_interaction()
    center, normal, up = (native.transforms[row, :3, i] for i in (3, 0, 1))
    renderer.camera.look_at(center + normal * 80, center, up)
    renderer.distance = 80
    renderer.render(400, 300)
    original_alpha = renderer.instance_alpha
    dimensions = uploaded(renderer)["dimensions"].copy()
    writes = watch_uploads(renderer, monkeypatch)
    assert renderer.pick(200, 150) == plane
    renderer.distance = 160
    app._sync_interaction()
    renderer.render(400, 300)
    assert renderer.instance_alpha is original_alpha
    assert len(writes) == 1
    assert np.array_equal(uploaded(renderer)["dimensions"], dimensions * 2)
    assert renderer.pick(200, 150) == plane
    assert renderer.context.error == "GL_NO_ERROR"
