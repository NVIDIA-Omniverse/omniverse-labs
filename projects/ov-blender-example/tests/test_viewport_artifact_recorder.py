# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import color_presentation, viewport_artifact_recorder  # noqa: E402


def test_recorder_status_tracks_running_and_completion() -> None:
    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": True},
        enabled=lambda: True,
    )
    request = SimpleNamespace(
        width=1280,
        height=720,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        timeline_controls_enabled=False,
        timeline_playing=False,
        timeline_frame=1,
        timeline_start=1,
        timeline_end=1,
        simulation_reset_token=0,
        render_product_path="/Render/Product",
        color_presentation=color_presentation.presentation_from_scene(
            None,
            requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        ),
    )

    def render_result(completed_samples: int) -> SimpleNamespace:
        return SimpleNamespace(
            width=1280,
            height=720,
            completed_samples=completed_samples,
            session_completed_samples=completed_samples,
            simulation_time_ns=0,
        )

    partial = render_result(2)
    complete = render_result(4)

    def artifact_status(result: SimpleNamespace, *, running: bool) -> str:
        return recorder.artifact(
            viewport_artifact_recorder.State(
                    simulation_id=None,
                request=request,
                result=result,
                snapshot_index=0,
                render_count=0,
                draw_count=0,
                snapshot_count=0,
                camera_update_count=0,
                camera_controls_mode="usd_camera",
                running=running,
            )
        )["status"]

    assert artifact_status(partial, running=True) == "running"
    assert artifact_status(partial, running=False) == "stopped"
    assert artifact_status(complete, running=False) == "complete"
    request.max_samples = 0
    assert artifact_status(complete, running=False) == "stopped"

def test_viewport_artifact_marks_native_fallback_status() -> None:
    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": False},
        enabled=lambda: False,
    )
    request = SimpleNamespace(
        width=1280,
        height=720,
        min_samples=1,
        max_samples=8,
        camera_prim_path="/World/Camera",
        timeline_controls_enabled=False,
        timeline_playing=False,
        timeline_frame=1,
        timeline_start=1,
        timeline_end=24,
        simulation_reset_token=0,
        render_product_path="/Render/Product",
    )
    result = SimpleNamespace(
        width=1280,
        height=720,
        completed_samples=8,
        session_completed_samples=8,
        simulation_time_ns=10,
    )

    artifact = recorder.artifact(
        viewport_artifact_recorder.State(
            simulation_id=None,
            request=request,
            result=result,
            snapshot_index=1,
            render_count=1,
            draw_count=1,
            snapshot_count=1,
            camera_update_count=0,
            camera_controls_mode="native_viewport_fallback",
            viewport_presentation={
                "presentation_mode": "native_viewport_fallback",
                "fallback_reason": "orthographic_user_view",
                "fallback_owned_by_addon": True,
            },
        )
    )

    assert artifact["status"] == "native_fallback"
    assert artifact["viewport_presentation"]["fallback_reason"] == "orthographic_user_view"


def _provenance_state(**overrides: object) -> viewport_artifact_recorder.State:
    fields: dict[str, object] = dict(
        simulation_id=None,
        request=None,
        result=None,
        snapshot_index=0,
        render_count=0,
        draw_count=0,
        snapshot_count=0,
        camera_update_count=0,
        camera_controls_mode="usd_camera",
    )
    fields.update(overrides)
    return viewport_artifact_recorder.State(**fields)


def test_artifact_emits_authored_scene_provenance_keys() -> None:
    """task05-04: the provenance mapping lands as additive top-level keys.

    The same keys the render-result artifact records, so viewport/F12
    same-generation parity is an artifact-level equality check.
    """

    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": False},
        enabled=lambda: False,
    )

    artifact = recorder.artifact(
        _provenance_state(
            authored_scene_provenance={
                "input_source": "active_scene",
                "input_usd_path": "/work/generation-2/scene.usdc",
                "authored_generation_digest": "d" * 64,
                "authored_generation": 2,
            }
        )
    )

    assert artifact["input_source"] == "active_scene"
    assert artifact["input_usd_path"] == "/work/generation-2/scene.usdc"
    assert artifact["authored_generation_digest"] == "d" * 64
    assert artifact["authored_generation"] == 2


def test_artifact_provenance_defaults_stay_empty() -> None:
    """States built without provenance (older callers) emit empty keys."""

    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": False},
        enabled=lambda: False,
    )

    artifact = recorder.artifact(_provenance_state())

    assert artifact["input_source"] == ""
    assert artifact["input_usd_path"] == ""
    assert artifact["authored_generation_digest"] == ""
    assert artifact["authored_generation"] is None
