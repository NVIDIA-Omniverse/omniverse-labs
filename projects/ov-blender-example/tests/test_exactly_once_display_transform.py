# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exactly-once display transform (task02-05).

Contract step 4 of the scene-linear ownership contract: Blender's View
Transform, Look, Exposure, and Gamma are applied exactly once. Scene-linear
``HdrColor`` RGBA16F frames are drawn through Blender's display-space shader
(the one application point); LDR ``LdrColor`` RGBA8 frames are already
display-encoded by OVRTX and pass through raw with no Blender transform on
top. These tests pin the classifier, the viewport upload/draw handoff, and
the F12 render-result insertion so a regression that double-transforms (or
inserts LDR pixels as linear) fails loudly.
"""

from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example.engine as engine_module  # noqa: E402
from ovrtx_blender_example import color_presentation, viewport_handoff  # noqa: E402
from ovrtx_blender_example.engine import _upload_viewport_texture  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult  # noqa: E402


# Half-float 1.0 is 0x3C00 (little-endian bytes 0x00, 0x3C).
_HALF_ONE = struct.pack("<e", 1.0)


def _ldr_result(width: int = 2, height: int = 2, fill: int = 255) -> RenderResult:
    return RenderResult(
        width=width,
        height=height,
        rgba8=bytes([fill]) * (width * height * 4),
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=10,
        frame_format=color_presentation.FRAME_FORMAT_RGBA8,
        frame_color_mode=color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )


def _scene_linear_result(width: int = 2, height: int = 2) -> RenderResult:
    return RenderResult(
        width=width,
        height=height,
        # rgba8 is deliberately all-zero so an insertion/upload that wrongly
        # used it (instead of the linear payload) is detectable.
        rgba8=bytes(width * height * 4),
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=10,
        frame_format=color_presentation.FRAME_FORMAT_RGBA16F,
        frame_color_mode=color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
        render_var=color_presentation.RENDER_VAR_HDR_COLOR,
        linear_rgba16f=_HALF_ONE * (width * height * 4),
    )


# --------------------------------------------------------------------------
# Classifier (viewport_handoff)
# --------------------------------------------------------------------------


def test_scene_linear_frame_marked_for_blender_transform() -> None:
    result = _scene_linear_result()
    assert (
        viewport_handoff.frame_display_transform(result)
        == viewport_handoff.FRAME_DISPLAY_BLENDER_TRANSFORM
    )
    assert viewport_handoff.frame_applies_blender_display_transform(result) is True


def test_ldr_frame_marked_passthrough() -> None:
    result = _ldr_result()
    assert (
        viewport_handoff.frame_display_transform(result)
        == viewport_handoff.FRAME_DISPLAY_PASSTHROUGH
    )
    assert viewport_handoff.frame_applies_blender_display_transform(result) is False


def test_scene_linear_frame_without_linear_payload_is_passthrough() -> None:
    # A frame that claims RGBA16F/scene-linear but carries no linear payload
    # must not be routed to Blender's transform (there is nothing to draw
    # through the display-space shader) — fail closed to passthrough.
    result = RenderResult(
        width=2,
        height=2,
        rgba8=bytes(16),
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=10,
        frame_format=color_presentation.FRAME_FORMAT_RGBA16F,
        frame_color_mode=color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
        linear_rgba16f=b"",
    )
    assert viewport_handoff.frame_applies_blender_display_transform(result) is False


# --------------------------------------------------------------------------
# Request classification (contract steps 1 & 4)
# --------------------------------------------------------------------------


def test_scene_linear_requests_carry_hdrcolor_and_ldr_carry_ldrcolor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, "scene_linear")
    hdr = color_presentation.presentation_from_scene(None, hdr_readback_available=True)
    assert hdr["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
    assert hdr["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F
    assert hdr["frame_color_mode"] == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    # The display transform is owned by the consumer (Blender), not OVRTX's
    # LDR encoding, so the two are never combined on the same frame.
    assert hdr["display_transform_owner"] == "consumer"

    monkeypatch.setenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, "ldr")
    ldr = color_presentation.presentation_from_scene(None)
    assert ldr["render_var"] == color_presentation.RENDER_VAR_LDR_COLOR
    assert ldr["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
    assert ldr["frame_color_mode"] == color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR


# --------------------------------------------------------------------------
# Viewport upload handoff
# --------------------------------------------------------------------------


class _FakeBuffer:
    def __init__(self, kind: str, length: int, data: object) -> None:
        self.kind = kind
        self.length = length
        self.data = data


class _FakeTexture:
    created: list["_FakeTexture"] = []

    def __init__(self, size: tuple[int, int], *, format: str, data: _FakeBuffer) -> None:
        self.size = size
        self.format = format
        self.data = data
        self.filtered = False
        self.updates: list[tuple[_FakeBuffer, str]] = []
        _FakeTexture.created.append(self)

    def filter_mode(self, enabled: bool) -> None:
        self.filtered = enabled

    def update(self, buffer: _FakeBuffer, *, format: str) -> None:
        self.updates.append((buffer, format))


class _FakeGpu:
    class types:
        Buffer = _FakeBuffer
        GPUTexture = _FakeTexture


def test_scene_linear_upload_builds_rgba16f_float_texture() -> None:
    _FakeTexture.created = []
    upload = _upload_viewport_texture(
        _FakeGpu(),
        _scene_linear_result(),
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    assert upload.color_mode == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    assert upload.diagnostics["display_transform"] is True
    assert upload.diagnostics["texture_path"] == "scene_linear_float"
    assert upload.texture.format == "RGBA16F"
    assert upload.texture.data.kind == "FLOAT"
    assert len(_FakeTexture.created) == 1


def test_mode_flip_ldr_to_scene_linear_rebuilds_texture() -> None:
    _FakeTexture.created = []
    ldr = _upload_viewport_texture(
        _FakeGpu(),
        _ldr_result(),
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    flipped = _upload_viewport_texture(
        _FakeGpu(),
        _scene_linear_result(),
        cached_texture=ldr.texture,
        cached_texture_size=ldr.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=2,
        accepts_rgba8=ldr.accepts_rgba8,
        cached_texture_color_mode=ldr.color_mode,
    )

    # No in-place UBYTE update on the RGBA8 texture; a fresh RGBA16F texture.
    assert flipped.texture is not ldr.texture
    assert flipped.diagnostics["texture_path"] == "scene_linear_float"
    assert flipped.texture.format == "RGBA16F"
    assert ldr.texture.updates == []


def test_mode_flip_scene_linear_to_ldr_rebuilds_texture() -> None:
    _FakeTexture.created = []
    linear = _upload_viewport_texture(
        _FakeGpu(),
        _scene_linear_result(),
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    flipped = _upload_viewport_texture(
        _FakeGpu(),
        _ldr_result(),
        cached_texture=linear.texture,
        cached_texture_size=linear.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=2,
        accepts_rgba8=linear.accepts_rgba8,
        cached_texture_color_mode=linear.color_mode,
    )

    # No UBYTE update applied to the RGBA16F texture: build a new RGBA8 one.
    assert flipped.texture is not linear.texture
    assert flipped.diagnostics["texture_path"] == "new_texture"
    assert flipped.texture.format == "RGBA8"
    assert linear.texture.updates == []


# --------------------------------------------------------------------------
# Engine-instance tests (viewport draw + F12), require the fake-bpy engine
# --------------------------------------------------------------------------


@contextmanager
def _fake_bpy_engine():
    fake_bpy = ModuleType("bpy")
    fake_bpy.types = SimpleNamespace(RenderEngine=object)
    had_bpy = "bpy" in sys.modules
    original_bpy = sys.modules.get("bpy")
    sys.modules["bpy"] = fake_bpy
    try:
        module = importlib.reload(engine_module)
        yield module
    finally:
        if had_bpy:
            sys.modules["bpy"] = original_bpy
        else:
            sys.modules.pop("bpy", None)
        importlib.reload(engine_module)


class _FakeFramebuffer:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self, *, color: tuple[float, ...]) -> None:
        self.cleared = True


class _PushPop:
    def __enter__(self) -> "_PushPop":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _DrawFakeGpu(ModuleType):
    def __init__(self) -> None:
        super().__init__("gpu")
        self.blend_calls: list[str] = []
        gpu_self = self

        class _State:
            @staticmethod
            def active_framebuffer_get() -> _FakeFramebuffer:
                return _FakeFramebuffer()

            @staticmethod
            def blend_set(mode: str) -> None:
                gpu_self.blend_calls.append(mode)

        class _Matrix:
            @staticmethod
            def push_pop() -> _PushPop:
                return _PushPop()

            @staticmethod
            def push_pop_projection() -> _PushPop:
                return _PushPop()

            @staticmethod
            def load_matrix(_matrix: object) -> None:
                return None

            @staticmethod
            def load_projection_matrix(_matrix: object) -> None:
                return None

        self.state = _State()
        self.matrix = _Matrix()


class _FakeMatrix:
    def __init__(self, rows: object = ()) -> None:
        self.rows = rows

    @staticmethod
    def Identity(_size: int) -> "_FakeMatrix":
        return _FakeMatrix()


@contextmanager
def _fake_draw_modules():
    draw_calls: list[object] = []
    gpu_mod = _DrawFakeGpu()
    presets = ModuleType("gpu_extras.presets")
    presets.draw_texture_2d = lambda *args: draw_calls.append(args)  # type: ignore[attr-defined]
    gpu_extras = ModuleType("gpu_extras")
    gpu_extras.presets = presets  # type: ignore[attr-defined]
    mathutils = ModuleType("mathutils")
    mathutils.Matrix = _FakeMatrix  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in ("gpu", "gpu_extras", "gpu_extras.presets", "mathutils")}
    sys.modules["gpu"] = gpu_mod
    sys.modules["gpu_extras"] = gpu_extras
    sys.modules["gpu_extras.presets"] = presets
    sys.modules["mathutils"] = mathutils
    try:
        yield gpu_mod, draw_calls
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _draw_context() -> SimpleNamespace:
    return SimpleNamespace(
        region=SimpleNamespace(width=64, height=64),
        region_data=SimpleNamespace(view_perspective="PERSP", view_matrix=None),
        scene=SimpleNamespace(camera=None),
    )


def test_scene_linear_draw_binds_display_space_shader_once() -> None:
    with _fake_bpy_engine() as module:
        engine = module.OvrtxExampleRenderEngine()
        bind_calls: list[object] = []
        unbind_calls: list[int] = []
        engine.bind_display_space_shader = lambda scene: bind_calls.append(scene)
        engine.unbind_display_space_shader = lambda: unbind_calls.append(1)

        with _fake_draw_modules() as (gpu_mod, draw_calls):
            context = _draw_context()
            engine._draw_viewport_texture(context, object(), _scene_linear_result())

        assert bind_calls == [context.scene]
        assert unbind_calls == [1]
        assert len(draw_calls) == 1
        # Premultiplied alpha under the display shader, restored to NONE after.
        assert gpu_mod.blend_calls == ["ALPHA_PREMULT", "NONE"]
        operator_view = engine._viewport_last_operator_view
        assert operator_view["display_transform_applied_by_blender"] is True


def test_ldr_draw_does_not_bind_display_space_shader() -> None:
    with _fake_bpy_engine() as module:
        engine = module.OvrtxExampleRenderEngine()
        bind_calls: list[object] = []
        unbind_calls: list[int] = []
        engine.bind_display_space_shader = lambda scene: bind_calls.append(scene)
        engine.unbind_display_space_shader = lambda: unbind_calls.append(1)

        with _fake_draw_modules() as (gpu_mod, draw_calls):
            engine._draw_viewport_texture(_draw_context(), object(), _ldr_result())

        assert bind_calls == []
        assert unbind_calls == []
        assert len(draw_calls) == 1
        assert gpu_mod.blend_calls == ["NONE"]
        assert (
            engine._viewport_last_operator_view["display_transform_applied_by_blender"]
            is False
        )


# --------------------------------------------------------------------------
# F12 render-result insertion (contract step 3)
# --------------------------------------------------------------------------


class _CapturingRect:
    def __init__(self) -> None:
        self.values: list[float] | None = None

    def foreach_set(self, values: object) -> None:
        self.values = list(values)


class _CapturingResult:
    def __init__(self) -> None:
        self.rect = _CapturingRect()
        pass_map = {"Combined": SimpleNamespace(rect=self.rect)}
        self.layers = [SimpleNamespace(passes=pass_map)]


def _run_write_blender_result(module: object, render_result: RenderResult) -> _CapturingResult:
    engine = module.OvrtxExampleRenderEngine()
    captured = _CapturingResult()
    engine.begin_result = lambda *_args, **_kwargs: captured
    engine.end_result = lambda *_args, **_kwargs: None
    engine._write_blender_result(render_result)
    return captured


def test_final_render_inserts_linear_pixels_for_scene_linear() -> None:
    with _fake_bpy_engine() as module:
        captured = _run_write_blender_result(module, _scene_linear_result())
    # All-1.0 linear payload -> all 1.0 inserted; the all-zero rgba8 would
    # have produced 0.0, so this proves the linear branch fired.
    assert captured.rect.values == [1.0] * 16


def test_final_render_inserts_display_encoded_pixels_for_ldr() -> None:
    with _fake_bpy_engine() as module:
        captured = _run_write_blender_result(module, _ldr_result(fill=128))
    # LDR path divides the display-encoded bytes by 255; never inserts the
    # (empty) linear payload as if linear (a linear branch would raise on the
    # empty payload). The distinct 128/255 value proves the rgba8 branch.
    assert captured.rect.values is not None
    assert captured.rect.values == pytest.approx([128 / 255] * 16)
