# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orthographic user views with camera sync stay on the OVRTX pipeline.

Updated for task02-04 (non-blocking callbacks): ``view_update`` hands the
session to the async seam (``_begin_async_viewport_session``) instead of
performing a synchronous ensure, and ``view_draw`` presents the newest
frame published to the latest-frame slot instead of rendering inline. The
assertions keep their original intent — an orthographic viewport context
that does not trigger the native fallback starts/keeps an OVRTX session
and draws the OVRTX texture.
"""

from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

# Cache the package (bpy unavailable) so loading engine.py under a fake bpy
# below does not execute the add-on __init__ against the fake module.
import ovrtx_blender_example  # noqa: E402,F401


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []
        self.redraw_requested = False
        self.reports: list[tuple[set[str], str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        self.redraw_requested = True

    def report(self, levels: set[str], message: str) -> None:
        self.reports.append((levels, message))


def _load_engine_with_fake_bpy(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_native_fallback_lifecycle_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=SimpleNamespace()),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _orthographic_context() -> SimpleNamespace:
    return SimpleNamespace(
        region_data=SimpleNamespace(view_perspective="ORTHO"),
        scene=SimpleNamespace(),
        space_data=None,
    )


def test_view_update_orthographic_view_starts_ovrtx_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    request = module.RenderRequest(max_samples=8)
    adapter_calls: list[tuple[object, ...]] = []
    begin_calls: list[object] = []

    class _Adapter:
        def view_update(self, context: object, depsgraph: object) -> object:
            adapter_calls.append(("view_update", context, depsgraph))
            return request

    def render_adapter(engine_id: str = "") -> object:
        adapter_calls.append(("factory", engine_id))
        return _Adapter()

    def begin_session(
        seen_request: object,
        _scene: object = None,
        _depsgraph: object = None,
    ) -> None:
        begin_calls.append(seen_request)
        engine._viewport_request = seen_request
        engine._viewport_simulation_id = "sim"

    monkeypatch.setattr(module, "_render_callback_adapter", render_adapter)
    engine._begin_async_viewport_session = begin_session

    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    context = _orthographic_context()
    engine.view_update(context, depsgraph)

    assert adapter_calls == [
        ("factory", f"{module.ENGINE_ID}:{id(engine):x}"),
        ("view_update", context, depsgraph),
    ]
    assert begin_calls == [request]
    assert engine._viewport_request is request
    assert engine._viewport_simulation_id is not None
    # The translated request was handed off latest-wins to the render loop.
    written = engine._camera_mailbox.peek()
    assert written is not None
    assert written.max_samples == request.max_samples
    assert engine.redraw_requested
    assert engine._viewport_presentation["presentation_mode"] == "ovrtx_rendered"


def test_view_draw_orthographic_view_draws_ovrtx_texture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    request = module.RenderRequest(max_samples=8)
    result = SimpleNamespace(completed_samples=8)
    adapter_calls: list[tuple[object, ...]] = []
    begin_calls: list[object] = []
    draw_calls: list[tuple[object, object, object]] = []
    profile_calls: list[dict[str, object]] = []

    class _Adapter:
        def view_draw(self, context: object, depsgraph: object) -> object:
            adapter_calls.append(("view_draw", context, depsgraph))
            return request

        def _translation_timings_snapshot(self) -> dict[str, float]:
            return {}

    def render_adapter(engine_id: str = "") -> object:
        adapter_calls.append(("factory", engine_id))
        return _Adapter()

    def begin_session(
        seen_request: object,
        _scene: object = None,
        _depsgraph: object = None,
    ) -> None:
        begin_calls.append(seen_request)
        engine._viewport_request = seen_request
        engine._viewport_simulation_id = "sim"

    monkeypatch.setattr(module, "_render_callback_adapter", render_adapter)
    engine._begin_async_viewport_session = begin_session
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda context, texture, seen_result, _scene: draw_calls.append(
        (context, texture, seen_result)
    )
    engine._record_profile = lambda *_args, **kwargs: profile_calls.append(kwargs)
    engine._write_viewport_artifact = lambda *_args, **_kwargs: None
    # The render thread published a frame; the draw callback presents it.
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            status=module.viewport_handoff.FRAME_STATUS_FRAME,
            render_result=result,
            snapshot_key=("published",),
            completed_samples=8,
        )
    )

    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    context = _orthographic_context()
    engine.view_draw(context, depsgraph)

    assert adapter_calls == [
        ("factory", f"{module.ENGINE_ID}:{id(engine):x}"),
        ("view_draw", context, depsgraph),
    ]
    assert begin_calls == [request]
    assert draw_calls == [(context, "texture", result)]
    assert engine._viewport_request is request
    assert engine._viewport_simulation_id is not None
    assert engine._camera_mailbox.peek() is not None
    assert engine._viewport_presentation["presentation_mode"] == "ovrtx_rendered"
    assert profile_calls[0]["rgba_available_monotonic_ns"] <= profile_calls[0]["ended_monotonic_ns"]


def test_orthographic_view_does_not_end_active_ovrtx_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    shutdown_calls: list[str] = []
    engine._viewport_client = SimpleNamespace(shutdown=lambda: shutdown_calls.append("shutdown"))
    engine._viewport_simulation_id = "sim"
    engine._ovrtx_scene_composition = SimpleNamespace(
        source_scene_path="/fixtures/stage.usda",
        composed_scene_path="/tmp/composed.usda",
        presentation_layers=(),
        digest="digest",
        pass_through=False,
    )
    request = module.RenderRequest(max_samples=8)
    begin_calls: list[object] = []
    adapter_calls: list[tuple[object, ...]] = []

    class _Adapter:
        def view_update(self, context: object, depsgraph: object) -> object:
            adapter_calls.append(("view_update", context, depsgraph))
            return request

    def render_adapter(engine_id: str = "") -> object:
        adapter_calls.append(("factory", engine_id))
        return _Adapter()

    monkeypatch.setattr(module, "_render_callback_adapter", render_adapter)
    engine._begin_async_viewport_session = (
        lambda seen_request, _scene=None, _depsgraph=None: begin_calls.append(seen_request)
    )

    context = _orthographic_context()
    depsgraph = SimpleNamespace(scene=SimpleNamespace())
    engine.view_update(context, depsgraph)

    assert shutdown_calls == []
    assert adapter_calls == [
        ("factory", f"{module.ENGINE_ID}:{id(engine):x}"),
        ("view_update", context, depsgraph),
    ]
    assert begin_calls == [request]
    assert engine._viewport_client is not None
    assert engine._viewport_simulation_id is not None
    assert engine._ovrtx_scene_composition is not None
    assert engine._viewport_presentation["presentation_mode"] == "ovrtx_rendered"
