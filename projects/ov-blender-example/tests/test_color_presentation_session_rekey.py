# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presentation-mode change re-keys the live session (task02-02).

A resolved presentation-mode change flips the render var (``LdrColor`` RGBA8
<-> ``HdrColor`` RGBA16F). The render var joins session identity so the change
flows through the ordinary ``reuse_decision`` -> background-resync path (no
ad-hoc teardown), the replacement reads back the new render var, and the
artist's RTPT quality values (task01-03) survive the re-key.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (
    color_presentation,
    ovrtx_scene_composition,
    ovrtx_session,
    ovrtx_session_controller as controller_module,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult
from ovrtx_blender_example.render_requests import RenderRequest


class _EchoingClient:
    """Fake native client that echoes the requested render var into results."""

    def __init__(self, simulation_id: str) -> None:
        self.simulation_id = simulation_id
        self.starts = 0
        self.deletes = 0
        self.closed = 0
        self.render_vars: list[str] = []
        self.call_thread_idents: list[int] = []
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings = {"native_render_ms": 1.0}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.call_thread_idents.append(threading.get_ident())
        self.starts += 1
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.call_thread_idents.append(threading.get_ident())
        render_var = str(kwargs["render_var"])
        self.render_vars.append(render_var)
        if render_var == color_presentation.RENDER_VAR_HDR_COLOR:
            frame_format = color_presentation.FRAME_FORMAT_RGBA16F
            frame_color_mode = color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
        else:
            frame_format = color_presentation.FRAME_FORMAT_RGBA8
            frame_color_mode = color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=len(self.render_vars),
            simulation_time_ns=42,
            frame_format=frame_format,
            frame_color_mode=frame_color_mode,
            render_var=render_var,
        )

    def shutdown(self) -> None:
        self.closed += 1

    def delete_simulation(self, simulation_id: str) -> str:
        self.call_thread_idents.append(threading.get_ident())
        assert simulation_id == self.simulation_id
        self.deletes += 1
        return "stopped"


def _request(tmp_path: Path, mode: str, **changes: object) -> RenderRequest:
    """Build a live-viewport request whose presentation resolves to ``mode``."""

    scene = _Scene(mode)
    return replace(
        RenderRequest(
            input_usd_path=str(tmp_path / "scene.usda"),
            current_scene_generation=True,
            sensor_paths=("/Render/Product",),
            selected_sensor_paths=("/Render/Product",),
            width=1,
            height=1,
            min_samples=1,
            max_samples=4,
            camera_prim_path="/World/OVRTXCamera",
            worker_command="worker",
            native_client_module="client",
            color_presentation=color_presentation.presentation_from_scene(scene),
        ),
        **changes,
    )


class _SceneSettings:
    def __init__(self, mode: str) -> None:
        self.color_presentation_mode = mode


class _Scene:
    def __init__(self, mode: str) -> None:
        self.ovrtx_example = _SceneSettings(mode)


def _factory(monkeypatch: pytest.MonkeyPatch, clients: list[_EchoingClient]) -> None:
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: clients.pop(0),
    )


def test_mode_flip_resolves_to_a_new_render_var() -> None:
    """The UI enum resolves LdrColor / HdrColor into the request diagnostics."""

    ldr = color_presentation.presentation_from_scene(
        _Scene(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    )
    hdr = color_presentation.presentation_from_scene(
        _Scene(color_presentation.MODE_SCENE_LINEAR_HDR)
    )

    assert ldr["render_var"] == color_presentation.RENDER_VAR_LDR_COLOR
    assert ldr["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
    assert hdr["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
    assert hdr["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F


def test_mode_flip_rekeys_session_as_one_background_resync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live mode change re-keys through would_replace/ensure, once.

    The controller reuses on an identical request, reports ``render_var_changed``
    when the mode flips (the render loop's would_replace probe), performs exactly
    one replacement, and the replacement's frames report the new render var and
    frame format. The re-key reuses the running worker client (same runtime
    binding, no worker restart), so the replacement session starts on the same
    client and the spare factory client stays unused.
    """

    first = _EchoingClient("ldr-sim")
    second = _EchoingClient("hdr-sim")
    _factory(monkeypatch, [first, second])
    controller = controller_module.OvrtxSessionController()

    ldr_request = _request(tmp_path, color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    hdr_request = _request(tmp_path, color_presentation.MODE_SCENE_LINEAR_HDR)

    assert controller.ensure(ldr_request).session_started is True
    ldr_result = controller.render(ldr_request, additional_samples=1)
    assert ldr_result.render_var == color_presentation.RENDER_VAR_LDR_COLOR

    # Same mode reuses; the mode flip is the only replacement trigger.
    assert controller.would_replace(ldr_request) == ""
    assert controller.would_replace(hdr_request) == "render_var_changed"

    replacement = controller.ensure(hdr_request)
    assert replacement.session_started is True
    hdr_result = controller.render(hdr_request, additional_samples=1)

    # New session reads back the new render var and frame format. The
    # replacement rides the reused worker client (merged controller design:
    # a re-key preserves the worker when the runtime binding is unchanged).
    assert hdr_result.render_var == color_presentation.RENDER_VAR_HDR_COLOR
    assert hdr_result.frame_format == color_presentation.FRAME_FORMAT_RGBA16F
    assert first.render_vars == [
        color_presentation.RENDER_VAR_LDR_COLOR,
        color_presentation.RENDER_VAR_HDR_COLOR,
    ]

    # Exactly one background resync: a single replacement, no teardown churn,
    # no worker restart (the client is never shut down, the spare unused).
    events = [event["event"] for event in controller.diagnostics()["lifecycle_events"]]
    assert events == ["created", "stopped", "replaced"]
    assert first.deletes == 1
    assert first.starts == 2
    assert first.closed == 0
    assert second.starts == 0


def test_quality_values_survive_the_rekey(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTPT quality (task01-03) persists across the mode-change re-key.

    Composing with non-default quality values then flipping the mode yields a
    replace decision (render_var_changed), and the replacement session's
    composition authors the same quality attribute values.
    """

    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    # Both values are non-default (UI defaults: maxBounces 1, fireflyFilter True)
    # so their presence in the re-keyed composition proves the values travel.
    # rtpt_quality carries artist-facing UI values; Max Bounces UI 7 authors as
    # wire 9 (+2 camera-ray offset).
    rtpt_quality = {
        "rtpt_max_bounces": 7,
        "rtpt_firefly_filter_enabled": False,
    }

    ldr_request = _request(
        tmp_path,
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        rtpt_quality=rtpt_quality,
    )
    hdr_request = _request(
        tmp_path,
        color_presentation.MODE_SCENE_LINEAR_HDR,
        rtpt_quality=rtpt_quality,
    )

    ldr_spec = ovrtx_session.build_spec(ldr_request)
    hdr_spec = ovrtx_session.build_spec(hdr_request)

    # The mode flip re-keys the session (render var is identity).
    assert ovrtx_session.reuse_decision(ldr_spec, hdr_spec).reason == "render_var_changed"

    # The re-keyed composition authors the current (non-default) quality values.
    record = next(
        item
        for item in hdr_spec.ovrtx_scene_composition.presentation_layers
        if item["source"] == "viewport_camera_projection"
    )
    text = Path(str(record["path"])).read_text(encoding="utf-8")
    assert "int omni:rtx:rtpt:maxBounces = 9" in text
    assert "bool omni:rtx:rtpt:fireflyFilter:enabled = false" in text

    ldr_record = next(
        item
        for item in ldr_spec.ovrtx_scene_composition.presentation_layers
        if item["source"] == "viewport_camera_projection"
    )
    ldr_text = Path(str(ldr_record["path"])).read_text(encoding="utf-8")
    assert "int omni:rtx:rtpt:maxBounces = 9" in ldr_text
