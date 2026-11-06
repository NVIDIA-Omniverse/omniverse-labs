# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase09 ownership, fallback, and orientation boundaries without Blender."""
from dataclasses import replace
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

from ovrtx_blender_example import engine, ovrtx_runtime_client as client  # noqa: E402


TOP = bytes((255, 0, 0, 255, 0, 255, 0, 255))
BOTTOM = bytes((0, 0, 255, 255, 255, 255, 0, 255))
PIXELS = TOP + BOTTOM




def _owned_result():
    payload = memoryview(PIXELS)
    return client.RenderResult(2, 2, payload, 1, 1, 0, rgba8_origin="top_left"), payload


def test_owned_result_retains_view_and_materializes_each_orientation():
    result, payload = _owned_result()
    assert result.rgba8 is payload
    assert result.rgba8.readonly
    assert result.rgba8_origin == "top_left"
    assert result.rgba8_bytes() == BOTTOM + TOP
    assert result.rgba8_bytes(origin="top_left") == PIXELS
    del payload
    gc.collect()
    assert result.rgba8.tobytes() == PIXELS
    with pytest.raises(TypeError):
        result.rgba8[0] = 7














class _Buffer:
    def __init__(self, kind, length, data):
        self.kind, self.length, self.data = kind, length, data


class _Texture:
    def __init__(self, size, *, format, data):
        self.size, self.format, self.data = size, format, data
    def filter_mode(self, _enabled):
        pass
    def update(self, buffer, *, format):
        self.data = buffer


def _upload(result, texture=None, *, accepts_rgba8=True):
    gpu = SimpleNamespace(types=SimpleNamespace(Buffer=_Buffer, GPUTexture=_Texture))
    return engine._upload_viewport_texture(
        gpu, result, cached_texture=texture,
        cached_texture_size=(2, 2, result.frame_format) if texture else None,
        cached_texture_snapshot_index=0, snapshot_index=1,
        accepts_rgba8=accepts_rgba8,
    )


@pytest.mark.parametrize("update", [False, True])
def test_gpu_staging_copy_cannot_mutate_native_pixels(update):
    result, _ = _owned_result()
    texture = _Texture((2, 2), format="RGBA8", data=None) if update else None
    uploaded = _upload(result, texture)
    assert uploaded.diagnostics["texture_buffer_copy_bytes"] == 16
    assert uploaded.diagnostics["texture_buffer_storage"] == "owned_mutable_staging"
    staging = uploaded.texture.data.data
    assert isinstance(staging, bytearray)
    assert staging == PIXELS
    staging[0] = 0
    assert result.rgba8.tobytes() == PIXELS


def test_gpu_float_fallback_preserves_native_row_order_for_draw_flip():
    result, _ = _owned_result()
    uploaded = _upload(result, accepts_rgba8=False)
    assert uploaded.diagnostics["texture_path"] == "fallback_float"
    assert list(uploaded.texture.data.data) == [value / 255.0 for value in PIXELS]
    assert uploaded.diagnostics["texture_buffer_copy_bytes"] == 0  # float conversion reported separately


def test_viewport_draw_flip_and_final_output_have_legacy_orientation():
    candidate, _ = _owned_result()
    legacy = replace(candidate, rgba8=BOTTOM + TOP, rgba8_origin="bottom_left")
    assert engine._viewport_texture_draw_rect(candidate, 10, 20, 100, 200) == ((10, 220), 100, -200)
    assert engine._viewport_texture_draw_rect(legacy, 10, 20, 100, 200) == ((10, 20), 100, 200)
    assert list(engine._rgba8_to_float_array(candidate.rgba8_bytes())) == list(engine._rgba8_to_float_array(legacy.rgba8_bytes()))


def test_capture_bytes_identical_for_legacy_and_owned_rows(monkeypatch):
    captured = []
    monkeypatch.setattr(engine, "write_rgba_png", lambda path, width, height, rgba: captured.append(rgba) or {})
    candidate, _ = _owned_result()
    legacy = replace(candidate, rgba8=BOTTOM + TOP, rgba8_origin="bottom_left")
    engine._write_image("candidate.png", candidate)
    engine._write_image("legacy.png", legacy)
    assert captured == [PIXELS, PIXELS]


