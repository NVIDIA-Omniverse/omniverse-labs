# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport artifact/profile recorder facade."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping

from . import color_presentation
from . import render_requests


@dataclass(frozen=True)
class State:
    simulation_id: str | None
    request: Any | None
    result: Any | None
    snapshot_index: int
    render_count: int
    draw_count: int
    snapshot_count: int
    camera_update_count: int
    camera_controls_mode: str
    viewport_presentation: Mapping[str, Any] = field(default_factory=dict)
    operator_view: Mapping[str, Any] = field(default_factory=dict)
    pose_mirror: Mapping[str, Any] = field(default_factory=dict)
    playback_lock: Mapping[str, Any] = field(default_factory=dict)
    image_artifact: Mapping[str, Any] = field(default_factory=dict)
    edit_bridge: Mapping[str, Any] = field(default_factory=dict)
    edit_workflow: Mapping[str, Any] = field(default_factory=dict)
    usd_prim_resolution: Mapping[str, Any] = field(default_factory=dict)
    texture_upload: Mapping[str, Any] = field(default_factory=dict)
    # Scene-input provenance (task05-04): input_source
    # (active_scene/env_override), the authored generation's ADR 0014
    # content digest and number, and the composed input path. Mirrors the
    # render-result artifact's provenance keys so viewport/F12
    # same-generation parity is an artifact-level equality check.
    authored_scene_provenance: Mapping[str, Any] = field(default_factory=dict)
    ovrtx_scene_composition: Mapping[str, Any] = field(default_factory=dict)
    ovrtx_session_reuse: Mapping[str, Any] = field(default_factory=dict)
    ovrtx_lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    shared_stage_composition: Mapping[str, Any] = field(default_factory=dict)
    startup: Mapping[str, Any] = field(default_factory=dict)
    session_lifecycle: Mapping[str, Any] = field(default_factory=dict)
    session_started_at_ns: int = 0
    end_reason: str = ""
    write_ms: float = 0.0
    running: bool = False
    # Thread-aware diagnostics (task02-09, schema_version 2): identity and
    # counters for the per-session render thread, its latest-view loop,
    # the publication→redraw signaler, and the tick-absorb handoff, plus
    # the render thread's per-loop-iteration timing records (aggregated
    # here, on the main thread, at artifact-write time).
    render_thread: Mapping[str, Any] = field(default_factory=dict)
    render_loop: Mapping[str, Any] = field(default_factory=dict)
    redraw_signaling: Mapping[str, Any] = field(default_factory=dict)
    tick_absorb: Mapping[str, Any] = field(default_factory=dict)
    render_loop_records: list[dict[str, Any]] = field(default_factory=list)


class Recorder:
    """Owns the active viewport profile lifecycle."""

    def __init__(
        self,
        *,
        profile_factory: Callable[[], dict[str, Any]],
        record_profile: Callable[[dict[str, Any], dict[str, Any]], None],
        profile_summary: Callable[[dict[str, Any], float], dict[str, Any]],
        enabled: Callable[[], bool],
        render_records_summary: (
            Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
            | None
        ) = None,
    ) -> None:
        self._profile_factory = profile_factory
        self._record_profile = record_profile
        self._profile_summary = profile_summary
        self._enabled = enabled
        self._render_records_summary = render_records_summary
        self._profile = self._profile_factory()

    def reset(self) -> None:
        self._profile = self._profile_factory()

    def record(self, record: Mapping[str, Any]) -> None:
        if not self._enabled():
            return
        self._record_profile(self._profile, dict(record))

    def summary(self, write_latency_ms: float = 0.0) -> dict[str, Any]:
        if not self._enabled():
            return {"enabled": False}
        return self._profile_summary(self._profile, write_latency_ms)

    def artifact(self, state: State) -> dict[str, Any]:
        request = state.request
        result = state.result
        status = _status(result, request, state.running)
        if state.viewport_presentation.get("presentation_mode") == "native_viewport_fallback":
            status = "native_fallback"
        profile = self.summary(state.write_ms)
        recent_draws = profile.get("recent_draws")
        if not isinstance(recent_draws, list):
            recent_draws = []
        if self._render_records_summary is not None:
            render_thread_profile = self._render_records_summary(
                list(state.render_loop_records), recent_draws
            )
        else:
            render_thread_profile = {
                "thread": "render",
                "record_count": len(state.render_loop_records),
                "records": [dict(record) for record in state.render_loop_records],
            }
        return {
            "schema_version": 3,
            "artifact_id": "ovrtx-viewport-preview",
            "status": status,
            "snapshot_index": int(state.snapshot_index),
            "render_count": int(state.render_count),
            "draw_count": int(state.draw_count),
            "snapshot_count": int(state.snapshot_count),
            "camera_update_count": int(state.camera_update_count),
            "width": result.width if result is not None else (request.width if request is not None else 0),
            "height": result.height if result is not None else (request.height if request is not None else 0),
            "min_samples": request.min_samples if request is not None else 0,
            "max_samples": request.max_samples if request is not None else 0,
            "completed_samples": result.completed_samples if result is not None else 0,
            "session_completed_samples": result.session_completed_samples if result is not None else 0,
            "simulation_time_ns": result.simulation_time_ns if result is not None else 0,
            "render_product_path": request.render_product_path if request is not None else "",
            "simulation_id": state.simulation_id or "",
            "camera_controls": _camera_controls(request, state.camera_controls_mode),
            "color_presentation": color_presentation.diagnostics_from_request_result(request, result),
            "viewport_presentation": dict(state.viewport_presentation),
            "operator_viewport": dict(state.operator_view),
            "runtime_pose_mirror": _pose_mirror(state.pose_mirror),
            "physics_playback_lock": dict(state.playback_lock),
            "image_artifact": dict(state.image_artifact),
            "interactive_edit_bridge": dict(state.edit_bridge),
            "interactive_edit_workflow": dict(state.edit_workflow),
            "usd_prim_resolution": dict(state.usd_prim_resolution),
            "texture_upload": dict(state.texture_upload),
            # Additive provenance keys (task05-04), shared with the
            # render-result artifact for the same-generation parity check.
            "input_source": str(state.authored_scene_provenance.get("input_source", "")),
            "input_usd_path": str(state.authored_scene_provenance.get("input_usd_path", "")),
            "authored_generation_digest": str(
                state.authored_scene_provenance.get("authored_generation_digest", "")
            ),
            "authored_generation": state.authored_scene_provenance.get(
                "authored_generation"
            ),
            "timeline_controls": _timeline_controls(request),
            "profile": profile,
            "thread_model": {
                "render_thread": dict(state.render_thread),
                "render_loop": dict(state.render_loop),
                "redraw_signaling": dict(state.redraw_signaling),
                "tick_absorb": dict(state.tick_absorb),
            },
            "render_thread_profile": render_thread_profile,
            "latest_view": _latest_view_evidence(state.render_loop),
            "ovrtx_scene_composition": dict(state.ovrtx_scene_composition),
            "ovrtx_session_reuse": dict(state.ovrtx_session_reuse),
            "ovrtx_lifecycle_events": list(state.ovrtx_lifecycle_events),
            "shared_stage_composition": dict(state.shared_stage_composition),
            "runtime_startup": dict(state.startup),
            "session_lifecycle": dict(state.session_lifecycle),
            "session_started_at_ns": int(state.session_started_at_ns),
            "session_end_reason": str(state.end_reason),
            "session_ended_at_ns": time.time_ns(),
            "written_at_ns": time.time_ns(),
        }


def _latest_view_evidence(render_loop: Mapping[str, Any]) -> dict[str, Any]:
    """Latest-view (ADR 0013) contract evidence from loop diagnostics.

    - ``abandoned_refinement_count``: adopted views whose refinement was
      abandoned because a newer view arrived (loop-side supersession).
    - ``superseded_snapshot_count``: those plus distinct pending views the
      mailbox replaced before the render thread ever adopted them.
    - ``final_view_refined``: the last adopted view before idle reached
      ``max_samples``.
    """

    mailbox = render_loop.get("mailbox")
    mailbox_superseded = 0
    if isinstance(mailbox, Mapping):
        mailbox_superseded = _optional_count(mailbox.get("superseded_snapshots"))
    abandoned = _optional_count(render_loop.get("snapshots_superseded"))
    return {
        "superseded_snapshot_count": mailbox_superseded + abandoned,
        "abandoned_refinement_count": abandoned,
        "final_view_refined": bool(render_loop.get("final_view_refined", False)),
    }


def _optional_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _status(result: Any | None, request: Any | None, running: bool) -> str:
    if running:
        return "running"
    if (
        result is not None
        and request is not None
        and not render_requests.viewport_sampling_due(
            result.completed_samples, request.max_samples
        )
    ):
        return "complete"
    return "stopped"


def _camera_controls(request: Any | None, mode: str) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "camera_prim_path": request.camera_prim_path if request is not None else "",
        "projection": render_requests.camera_projection_diagnostics(
            getattr(request, "camera_projection", None)
            if request is not None
            else None
        ),
    }


def _pose_mirror(pose_mirror: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(pose_mirror)
    if artifact.get("status") == "scheduled" and isinstance(artifact.get("last_applied"), Mapping):
        pending_count = artifact.get("scheduled_mirror_count", 0)
        artifact = dict(artifact["last_applied"])
        artifact["pending_scheduled_mirror_count"] = pending_count
    return artifact


def _timeline_controls(request: Any | None) -> dict[str, Any]:
    if request is None:
        return {"enabled": False}
    return {
        "enabled": request.timeline_controls_enabled,
        "playing": request.timeline_playing,
        "frame": request.timeline_frame,
        "start": request.timeline_start,
        "end": request.timeline_end,
        "simulation_reset_token": request.simulation_reset_token,
    }
