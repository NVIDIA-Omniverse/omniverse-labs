# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge anti-aliasing by resolution supersampling (cycles-parity n-05-03).

The OVRTX RTPT worker leaves silhouette/shadow edges aliased and none of its AA,
DLSS, jitter or accumulation settings change that, so F12 renders above the
output size and averages down. These tests cover the factor policy, the box
downsample (HDR linear and LDR in linear light), the request plumbing per
render intent, and the F12 hook / UI rows at source level.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import supersample  # noqa: E402
from ovrtx_blender_example.blender_signal_translation import RenderRequestTranslator  # noqa: E402
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


def _result(width, height, *, rgba8=b"", hdr=b"", **extra):
    return RenderResult(
        width=width, height=height, rgba8=rgba8, completed_samples=1, session_completed_samples=1,
        simulation_time_ns=0, linear_rgba16f=hdr, **extra,
    )


def test_parse_and_effective_factor() -> None:
    assert supersample.parse_factor("2") == 2
    assert supersample.parse_factor(9) == supersample.SUPERSAMPLE_MAX_FACTOR
    assert supersample.parse_factor("bad") == 1
    assert supersample.parse_factor(0) == 1
    assert supersample.effective_factor(960, 540, "3") == 3
    # 4K * 3 would be 11520x6480 = 74.6M pixels, over the budget: reduced to 2.
    assert supersample.effective_factor(3840, 2160, "3") == 2
    # 8K * 2 would be 132M pixels: no factor fits, supersampling turns off.
    assert supersample.effective_factor(7680, 4320, "2") == 1
    assert supersample.effective_factor(9000, 100, "3") == 1


def test_hdr_downsample_is_a_linear_box_average() -> None:
    source = np.zeros((4, 4, 4), np.float16)
    source[..., 3] = 1
    source[0:2, 0:2, :3] = 8.0       # one fully lit 2x2 block (HDR value above 1)
    source[0, 2, :3] = 4.0           # one lit pixel in the next block
    result = supersample.downsample_result(_result(4, 4, hdr=source.tobytes()), 2)
    assert (result.width, result.height) == (2, 2)
    out = np.frombuffer(result.linear_rgba16f, np.float16).reshape(2, 2, 4).astype(np.float32)
    assert out[0, 0, 0] == pytest.approx(8.0)
    assert out[0, 1, 0] == pytest.approx(1.0)   # 4 / 4
    assert out[1, 1, 0] == 0.0
    assert np.all(out[..., 3] == 1.0)
    assert result.rgba8 == b""


def test_ldr_downsample_blends_in_linear_light_and_keeps_alpha() -> None:
    source = np.zeros((2, 2, 4), np.uint8)
    source[..., 3] = 255
    source[0, :, :3] = 255            # top row white, bottom row black
    result = supersample.downsample_result(_result(2, 2, rgba8=source.tobytes()), 2)
    out = np.frombuffer(result.rgba8, np.uint8).reshape(1, 1, 4)
    # 50% coverage of white over black is 0.5 linear = 188 encoded, not 128.
    assert abs(int(out[0, 0, 0]) - 188) <= 1
    assert out[0, 0, 3] == 255


def test_downsample_edge_profile_gets_intermediate_values() -> None:
    # A hard diagonal edge rendered at 2x averages into a mix of coverage levels.
    n = 16
    yy, xx = np.mgrid[0:n, 0:n]
    lit = (xx > yy * 0.9 + 1.3)
    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[lit, :3] = 255
    out = np.frombuffer(
        supersample.downsample_result(_result(n, n, rgba8=rgba.tobytes()), 2).rgba8, np.uint8
    ).reshape(n // 2, n // 2, 4)[..., 0]
    assert ((out > 20) & (out < 235)).sum() >= 6
    assert len(np.unique(out)) > 3


def test_downsample_is_identity_for_factor_one_and_indivisible_sizes() -> None:
    result = _result(3, 3, rgba8=bytes(36))
    assert supersample.downsample_result(result, 1) is result
    assert supersample.downsample_result(result, 2) is result


def _scene(**ovrtx: object) -> SimpleNamespace:
    return SimpleNamespace(
        render=SimpleNamespace(resolution_x=640, resolution_y=360, resolution_percentage=50),
        ovrtx_example=SimpleNamespace(
            render_product_path="/Render/Test/Product", min_samples=1, max_samples=128,
            camera_prim_path="/World/Camera", sync_viewport_camera=False, simulation_reset_token=0, **ovrtx,
        ),
        frame_current=1, frame_start=1, frame_end=1,
    )


def _translate(scene, intent):
    return RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE, intent, scene, "",
            camera_prim_path="/Generated/Camera", render_product_path="/Generated/Product",
            context=SimpleNamespace(), current_scene_generation=True,
        )
    )


def test_request_default_is_unsupersampled() -> None:
    assert RenderRequest(input_usd_path="x").supersample == 1


def test_final_render_defaults_to_2x_and_scales_the_rendered_size() -> None:
    request = _translate(_scene(), BlenderRenderIntent.FINAL_RENDER)
    assert request.supersample == 2
    assert (request.width, request.height) == (640, 360)   # 320x180 output * 2


def test_final_render_honors_the_setting_and_viewport_defaults_off() -> None:
    off = _translate(_scene(final_supersample="1"), BlenderRenderIntent.FINAL_RENDER)
    assert (off.supersample, off.width, off.height) == (1, 320, 180)
    three = _translate(_scene(final_supersample="3"), BlenderRenderIntent.FINAL_RENDER)
    assert (three.supersample, three.width, three.height) == (3, 960, 540)
    viewport = _translate(_scene(), BlenderRenderIntent.VIEWPORT)
    assert (viewport.supersample, viewport.width, viewport.height) == (1, 320, 180)
    viewport2 = _translate(_scene(viewport_supersample="2"), BlenderRenderIntent.VIEWPORT)
    assert (viewport2.supersample, viewport2.width, viewport2.height) == (2, 640, 360)
    # The final setting never leaks into the viewport and vice versa.
    assert _translate(_scene(final_supersample="3"), BlenderRenderIntent.VIEWPORT).supersample == 1
    assert _translate(_scene(viewport_supersample="3"), BlenderRenderIntent.FINAL_RENDER).supersample == 2


def test_engine_downsamples_final_results_and_panel_exposes_both_rows() -> None:
    engine = (ROOT / "addon/ovrtx_blender_example/engine.py").read_text(encoding="utf-8")
    assert "supersample_module.downsample_result(render_result, request.supersample)" in engine
    assert engine.index("downsample_result") < engine.index("self._write_blender_result(render_result)\n            self._write_result_artifact(")
    ui = (ROOT / "addon/ovrtx_blender_example/ui.py").read_text(encoding="utf-8")
    assert '"final_supersample"' in ui and '"viewport_supersample"' in ui
    properties = (ROOT / "addon/ovrtx_blender_example/properties.py").read_text(encoding="utf-8")
    assert 'final_supersample: EnumProperty' in properties and 'default="2"' in properties
    assert 'viewport_supersample: EnumProperty' in properties
