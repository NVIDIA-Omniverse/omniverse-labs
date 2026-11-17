# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual designer CSS/browser pixels against real GPU background composition."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_renderer import renderer, snapshot
from ovd_ui.design_tokens import rgba
from ovd_ui.render_settings import RenderingSettings


def colors(theme):
    return rgba("stage_center", theme), rgba("stage_edge", theme)


@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize("msaa", (False, True))
def test_workspace_gradient_matches_actual_chromium_css_pixels(renderer, theme, msaa):
    oracle = json.loads(Path(__file__).with_name("workspace_gradient_oracle.json").read_text())
    expected = next(row for row in oracle["captures"] if row["theme"] == theme)
    renderer.set_settings(RenderingSettings(anti_aliasing=msaa))
    renderer.render(*oracle["size"], background_gradient=colors(theme))
    pixels = np.frombuffer(renderer.framebuffer.read(components=3), dtype="u1").reshape(800, 1280, 3)
    for sample in expected["samples"]:
        actual = pixels[sample["y"], sample["x"]].astype(int)
        # Browser dithering and framebuffer quantization can differ by one unit.
        assert np.max(np.abs(actual - sample["rgb"])) <= 1, (theme, msaa, sample, actual)
    assert renderer.pick(640, 440) is None


def test_explicit_flat_backgrounds_override_workspace_gradient(renderer):
    renderer.set_settings(RenderingSettings(background_mode="reference"))
    renderer.render(128, 80, background_gradient=colors("light"))
    assert set(renderer.framebuffer.read(components=3)) == {204}
    renderer.set_settings(RenderingSettings(background_mode="custom", background_color="#123456"))
    renderer.render(128, 80, background_gradient=colors("dark"))
    assert renderer.framebuffer.read(components=3) == bytes((18, 52, 86)) * (128 * 80)


def test_background_preserves_native_geometry_depth_and_object_picking(renderer):
    geometry = snapshot("test_scene.ovd")
    renderer.set_snapshot(geometry, True)
    renderer.set_settings(RenderingSettings())
    renderer.render(400, 300, background=rgba("background", "dark"))
    before = renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3)
    depth = renderer.framebuffer.read(attachment=-1, components=1, dtype="f4")
    identity = renderer.pick(200, 150)
    assert identity == int(geometry.object_ids[0])
    renderer.render(400, 300, background_gradient=colors("dark"))
    assert renderer.framebuffer.read(viewport=(200, 150, 1, 1), components=3) == before
    assert renderer.framebuffer.read(attachment=-1, components=1, dtype="f4") == depth
    assert renderer.pick(200, 150) == identity and renderer.pick(0, 0) is None
