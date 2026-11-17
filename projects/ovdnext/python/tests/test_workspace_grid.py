# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored grid alpha and real native-plane material/depth/picking checks."""
import re

import numpy as np
import pytest

from ovd_ui.design_tokens import STAGE_GRID_RGBA, rgba
from ovd_ui.renderer import ViewportRenderer
from ovd_ui.render_settings import RenderingSettings
from test_renderer import ROOT, renderer, snapshot


@pytest.mark.parametrize("theme,section", (("dark", "1b"), ("light", "1c")))
def test_grid_tokens_retain_exact_authored_css_alpha(theme, section):
    source = (ROOT / "docs/modernization/design-reference/PVD Soft Glass - src.dc.html").read_text(encoding='utf-8')
    canvas = source.split(f'id="{section}"', 1)[1]
    style = re.search(r'background-image:([^"\n]+)', canvas).group(1)
    authored = re.findall(r'rgba\((\d+),(\d+),(\d+),([.\d]+)\)', style)
    assert len(authored) == 2
    assert authored[0] == authored[1]
    expected = tuple(int(channel) / 255 for channel in authored[0][:3]) + (float(authored[0][3]),)
    assert STAGE_GRID_RGBA[theme] == expected


@pytest.mark.parametrize("mode,contrast,expected", (
    (None, .025, True), ("workspace", .025, True),
    ("reference", .025, False), ("custom", .025, False), ("workspace", 0., False),
))
def test_grid_request_scope_before_gpu_work(mode, contrast, expected):
    # Dispatch sink only: exercise real render parameter setup, stop before GL.
    item = object.__new__(ViewportRenderer)
    item.settings = None if mode is None else RenderingSettings(background_mode=mode)
    class BeforeResize(Exception):
        pass
    def stop(_size):
        raise BeforeResize
    item._resize = stop
    with pytest.raises(BeforeResize):
        item.render(16, 16, grid_rgba=STAGE_GRID_RGBA["light"], grid_contrast=contrast)
    assert item.grid_rgba == (STAGE_GRID_RGBA["light"] if expected else (0., 0., 0., 0.))
    with pytest.raises(BeforeResize):
        item.render(16, 16, grid_contrast=contrast)
    assert item.grid_rgba == (0., 0., 0., 0.)  # A following omitted request clears the material.


def _plane_view(item, left_handed=False):
    geometry = snapshot("multi_geom.ovd", 2)
    row = np.flatnonzero(geometry.kinds == 4)[0]
    identity = int(geometry.object_ids[row])
    item.set_snapshot(geometry)
    item.hidden = set(map(int, geometry.object_ids)) - {identity}
    item.camera.left_handed = left_handed
    center = geometry.transforms[row, :3, 3]
    normal = geometry.transforms[row, :3, 0]
    up = geometry.transforms[row, :3, 1]
    item.camera.look_at(center + normal * 50, center, up)
    return geometry, identity


def _pixels(item):
    return np.frombuffer(item.framebuffer.read(components=3), dtype="u1").reshape(item.size[1], item.size[0], 3).copy()


@pytest.mark.parametrize("theme", ("dark", "light"))
def test_native_plane_grid_matches_independent_source_over_arithmetic(renderer, theme):
    _, identity = _plane_view(renderer)
    renderer.set_settings(RenderingSettings(face_culling=0))
    # The retained additive material independently exposes the actual unchanged
    # antialiased union coverage. No candidate mix expression supplies this mask.
    renderer.render(320, 240, ground_color=(0., 0., 0.), grid_contrast=1.)
    coverage = _pixels(renderer).astype(float) / 255.
    renderer.pick(160, 120)
    assert np.all(np.frombuffer(renderer.pick_buffer.read(components=4), dtype="<u4") == identity + 1)
    assert coverage.max() > .5 and np.count_nonzero((coverage > .05) & (coverage < .9)) > 100
    ground = rgba("elevated" if theme == "light" else "surface", theme)
    rgb, alpha = np.array(STAGE_GRID_RGBA[theme][:3]), STAGE_GRID_RGBA[theme][3]
    renderer.render(320, 240, ground_color=ground, grid_contrast=.025, grid_rgba=STAGE_GRID_RGBA[theme])
    expected = np.array(ground[:3]) * (1. - coverage * alpha) + rgb * coverage * alpha
    assert np.max(np.abs(_pixels(renderer).astype(float) - expected * 255.)) <= 1.1
    # This intentionally changes dark too: white alpha .05 replaces +.025.
    renderer.render(320, 240, ground_color=ground, grid_contrast=.025)
    assert np.max(np.abs(_pixels(renderer).astype(float) - expected * 255.)) > 3


@pytest.mark.parametrize("theme,msaa,transparent,left_handed", (
    ("dark", False, False, False), ("light", True, False, True),
    ("light", False, True, False), ("dark", True, True, True),
))
def test_native_grid_preserves_full_depth_pick_and_nonplane_pixels(renderer, theme, msaa, transparent, left_handed):
    geometry = snapshot("multi_geom.ovd", 2)
    renderer.set_snapshot(geometry, True)
    renderer.camera.left_handed = left_handed
    renderer.set_settings(RenderingSettings(face_culling=0, anti_aliasing=msaa))
    plane_ids = geometry.object_ids[geometry.kinds == 4].astype("u4") + 1
    if transparent:
        renderer.alpha_by_id = {int(identity) - 1: .45 for identity in plane_ids}
    options = dict(ground_color=rgba("elevated" if theme == "light" else "surface", theme), grid_contrast=.025)
    renderer.render(400, 300, **options)
    before = _pixels(renderer)
    depth = renderer.framebuffer.read(attachment=-1, components=1, dtype="f4")
    renderer.pick(200, 150)
    picks = renderer.pick_buffer.read(components=4)
    pick_ids = np.frombuffer(picks, dtype="<u4").reshape(300, 400)
    plane_pixels = np.isin(pick_ids, plane_ids)
    assert plane_pixels.any() and (~plane_pixels).any()
    groups = tuple((id(mesh), mesh.instances.glo) for _, mesh in renderer._groups)
    renderer.render(400, 300, grid_rgba=STAGE_GRID_RGBA[theme], **options)
    after = _pixels(renderer)
    # Picking shares the renderer's depth attachment and clears/redraws it.
    # Compare the two completed shaded renders before that separate pass.
    assert renderer.framebuffer.read(attachment=-1, components=1, dtype="f4") == depth
    renderer.pick(200, 150)
    assert renderer.pick_buffer.read(components=4) == picks
    assert tuple((id(mesh), mesh.instances.glo) for _, mesh in renderer._groups) == groups
    assert np.any(before[plane_pixels] != after[plane_pixels])
    # A nearest pick ID does not describe every MSAA sample or transparent
    # particle layer. Their final pixels legitimately contain the changed floor.
    opaque_ids = geometry.object_ids[~np.isin(geometry.kinds, (4, 7))].astype("u4") + 1
    interior = np.isin(pick_ids, opaque_ids)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= pick_ids == np.roll(pick_ids, (dy, dx), axis=(0, 1))
    assert np.count_nonzero(interior) > 100
    assert np.array_equal(before[interior], after[interior])
    # Independently retain every non-plane native object, including OIT points:
    # with planes hidden the complete composed image must be identical.
    renderer.hidden = set(map(int, plane_ids - 1))
    renderer.render(400, 300, **options)
    nonplane = renderer.framebuffer.read(components=4)
    assert renderer.instance_count > 0
    renderer.render(400, 300, grid_rgba=STAGE_GRID_RGBA[theme], **options)
    assert renderer.framebuffer.read(components=4) == nonplane
    assert renderer.context.error == "GL_NO_ERROR"


@pytest.mark.parametrize("mode,contrast", (("reference", .025), ("custom", .025), ("workspace", 0.)))
def test_explicit_background_and_grid_off_bypass_with_native_plane(renderer, mode, contrast):
    _, identity = _plane_view(renderer)
    renderer.set_settings(RenderingSettings(face_culling=0, background_mode=mode, background_color="#123456"))
    options = dict(ground_color=rgba("elevated", "light"), grid_contrast=contrast)
    renderer.render(320, 240, **options)
    before = renderer.framebuffer.read(components=4)
    assert renderer.pick(160, 120) == identity  # The test actually contains a plane.
    for theme in ("dark", "light"):
        renderer.render(320, 240, grid_rgba=STAGE_GRID_RGBA[theme], **options)
        assert renderer.framebuffer.read(components=4) == before
        assert renderer.pick(160, 120) == identity
