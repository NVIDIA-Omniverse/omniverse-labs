# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport cadence and timing profile aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import render_requests


RECENT_DRAW_LIMIT = 120
#: Cross-thread span phases (task02-09): each measures a boundary crossing
#: between the render thread and the Blender main thread, so it belongs to
#: neither thread's work decomposition. Existing phase names are unchanged;
#: these are additive.
CROSS_THREAD_PHASES = (
    "snapshot_to_render_start_ms",
    "publish_to_redraw_ms",
    "publish_to_draw_ms",
)
TIMING_PHASES = (
    "presentation_ms",
    "request_ms",
    "request_translation_ms",
    "request_scene_inputs_ms",
    "request_camera_ms",
    "request_runtime_inputs_ms",
    "request_runtime_state_ms",
    "request_runtime_defaults_ms",
    "request_native_client_preflight_ms",
    "request_material_ms",
    "request_build_ms",
    "request_adapter_overhead_ms",
    "ensure_session_ms",
    "ensure_controller_ms",
    "ensure_build_spec_ms",
    "ensure_reuse_decision_ms",
    "ensure_controller_other_ms",
    "ensure_outer_overhead_ms",
    "ensure_crash_marker_write_ms",
    "ensure_crash_marker_clear_ms",
    "ensure_diagnostics_ms",
    "acquire_result_ms",
    "acquire_overhead_ms",
    "runtime_update_ms",
    "camera_update_ms",
    "controller_render_ms",
    "render_diagnostics_ms",
    "render_ms",
    "result_convert_ms",
    "render_accounting_ms",
    "composition_update_ms",
    "texture_upload_ms",
    "artifact_write_ms",
    "viewport_texture_draw_ms",
    "viewport_callback_ms",
    "callback_overhead_ms",
    "profile_diagnostics_ms",
) + CROSS_THREAD_PHASES
#: Thread attribution per timing phase (task02-09, spec success criterion 2):
#: request/texture/draw phases run in the Blender draw callback (``main``);
#: session/value-apply/composition/render/readback phases run on the
#: per-session render thread (``render``); the cross-thread spans above are
#: ``cross``. Fine-grained sub-phases inherit the thread of the top-level
#: phase they decompose (e.g. ``request_translation_ms`` is part of
#: ``request_ms``); only the top-level phases join the per-thread work
#: decompositions below so nothing is double counted.
PHASE_THREADS = {
    "presentation_ms": "main",
    "request_ms": "main",
    "request_translation_ms": "main",
    "request_scene_inputs_ms": "main",
    "request_camera_ms": "main",
    "request_runtime_inputs_ms": "main",
    "request_runtime_state_ms": "main",
    "request_runtime_defaults_ms": "main",
    "request_native_client_preflight_ms": "main",
    "request_material_ms": "main",
    "request_build_ms": "main",
    "request_adapter_overhead_ms": "main",
    "ensure_session_ms": "render",
    "ensure_controller_ms": "render",
    "ensure_build_spec_ms": "render",
    "ensure_reuse_decision_ms": "render",
    "ensure_controller_other_ms": "render",
    "ensure_outer_overhead_ms": "render",
    "ensure_crash_marker_write_ms": "render",
    "ensure_crash_marker_clear_ms": "render",
    "ensure_diagnostics_ms": "render",
    "acquire_result_ms": "main",
    "acquire_overhead_ms": "main",
    "runtime_update_ms": "render",
    "camera_update_ms": "render",
    "controller_render_ms": "render",
    "render_diagnostics_ms": "render",
    "render_ms": "render",
    "result_convert_ms": "render",
    "render_accounting_ms": "render",
    "composition_update_ms": "render",
    "texture_upload_ms": "main",
    "artifact_write_ms": "main",
    "viewport_texture_draw_ms": "main",
    "viewport_callback_ms": "main",
    "callback_overhead_ms": "main",
    "profile_diagnostics_ms": "main",
    "snapshot_to_render_start_ms": "cross",
    "publish_to_redraw_ms": "cross",
    "publish_to_draw_ms": "cross",
}
#: Top-level phases only (their sub-phases are contained within them, so
#: including both would double count a thread's work).
MAIN_THREAD_PHASES = (
    "request_ms",
    "texture_upload_ms",
    "artifact_write_ms",
    "viewport_texture_draw_ms",
    "viewport_callback_ms",
)
RENDER_THREAD_PHASES = (
    "ensure_session_ms",
    "camera_update_ms",
    "render_ms",
    "result_convert_ms",
    "composition_update_ms",
)
#: Phases aggregated from the render thread's per-loop-iteration records.
RENDER_LOOP_TIMING_PHASES = (
    "ensure_session_ms",
    "camera_update_ms",
    "composition_update_ms",
    "render_ms",
    "snapshot_to_render_start_ms",
)
#: Non-overlapping phases summed as "measured work" for a render interval.
#: ``acquire_result_ms`` already contains the render/readback phases and the
#: request/ensure sub-phases are contained in their parents, so only the
#: top-level, mutually exclusive spans join the sum; latency (cross-thread)
#: spans overlap real work and are excluded.
RENDER_INTERVAL_WORK_PHASES = (
    "presentation_ms",
    "request_ms",
    "ensure_session_ms",
    "acquire_result_ms",
    "texture_upload_ms",
    "viewport_texture_draw_ms",
    "callback_overhead_ms",
)
NATIVE_TIMING_SCOPES = (
    "render_result",
    "value_update",
)
TEXTURE_TIMING_FIELDS = (
    "texture_convert_ms",
    "texture_buffer_ms",
    "texture_create_ms",
    "texture_filter_ms",
)


def snapshot_key_token(key: Any) -> str:
    """Stable, JSON-safe token for a snapshot key (record correlation)."""

    return "" if key is None else repr(key)


def new() -> dict[str, Any]:
    return {
        "draw_count": 0,
        "render_count": 0,
        "started_at_ns": None,
        "ended_at_ns": None,
        "started_monotonic_ns": None,
        "ended_monotonic_ns": None,
        "camera_changed_draw_count": 0,
        "snapshot_changed_draw_count": 0,
        "render_reason_counts": {},
        "reuse_reason_counts": {},
        "texture_path_counts": {},
        "draw_phase_counts": {},
        "texture_cache_hit_count": 0,
        "gpu_texture_update_available_count": 0,
        "texture_upload_bytes": 0,
        "steady_started": False,
        "last_draw_monotonic_ns": None,
        "last_draw_started_monotonic_ns": None,
        "last_render_monotonic_ns": None,
        "render_interval_callback_wait_ms": 0.0,
        "render_interval_work_ms_by_phase": {
            phase: 0.0 for phase in RENDER_INTERVAL_WORK_PHASES
        },
        "phase_stats": {phase: _empty_timing_stat() for phase in TIMING_PHASES},
        "native_timing_stats": {scope: {} for scope in NATIVE_TIMING_SCOPES},
        "texture_timing_stats": {
            field: _empty_timing_stat() for field in TEXTURE_TIMING_FIELDS
        },
        "recent_draws": [],
    }


def record(profile: dict[str, Any], record: dict[str, Any]) -> None:
    # Per-record thread attribution (task02-09): the per-draw stream is
    # produced by the Blender draw callback; render-thread iteration
    # records flow through render_thread_summary instead.
    record.setdefault("thread", "main")
    profile["draw_count"] += 1
    if record["rendered"]:
        profile["render_count"] += 1
    started_at_ns = _optional_int(record.get("started_at_ns"))
    if started_at_ns is not None:
        current_started_at_ns = _optional_int(profile.get("started_at_ns"))
        profile["started_at_ns"] = (
            started_at_ns
            if current_started_at_ns is None
            else min(current_started_at_ns, started_at_ns)
        )
    ended_at_ns = _optional_int(record.get("ended_at_ns"))
    if ended_at_ns is not None:
        current_ended_at_ns = _optional_int(profile.get("ended_at_ns"))
        profile["ended_at_ns"] = (
            ended_at_ns
            if current_ended_at_ns is None
            else max(current_ended_at_ns, ended_at_ns)
        )
    started_monotonic_ns = _optional_int(record.get("started_monotonic_ns"))
    if started_monotonic_ns is not None:
        current_started_monotonic_ns = _optional_int(profile.get("started_monotonic_ns"))
        profile["started_monotonic_ns"] = (
            started_monotonic_ns
            if current_started_monotonic_ns is None
            else min(current_started_monotonic_ns, started_monotonic_ns)
        )
    ended_monotonic_ns = _optional_int(record.get("ended_monotonic_ns"))
    if ended_monotonic_ns is not None:
        current_ended_monotonic_ns = _optional_int(profile.get("ended_monotonic_ns"))
        profile["ended_monotonic_ns"] = (
            ended_monotonic_ns
            if current_ended_monotonic_ns is None
            else max(current_ended_monotonic_ns, ended_monotonic_ns)
        )
    if record["camera_changed"]:
        profile["camera_changed_draw_count"] += 1
    if record["snapshot_changed"]:
        profile["snapshot_changed_draw_count"] += 1
    for phase, value_ms in record["timings_ms"].items():
        _record_timing_stat(profile["phase_stats"][phase], float(value_ms))
    for scope, timings in record.get("native_timings", {}).items():
        if not isinstance(timings, Mapping):
            continue
        scope = _native_timing_scope(scope)
        scope_stats = profile["native_timing_stats"].setdefault(scope, {})
        for key, value_ms in timings.items():
            if not str(key).endswith("_ms"):
                continue
            try:
                numeric_value = float(value_ms)
            except (TypeError, ValueError):
                continue
            _record_timing_stat(scope_stats.setdefault(key, _empty_timing_stat()), numeric_value)
    _annotate_draw(profile, record)
    record["draw_phase"] = _draw_phase(profile, record)
    _increment_counter(profile.setdefault("draw_phase_counts", {}), record.get("draw_phase"))
    if record["draw_phase"] == "timeline_reset":
        profile["steady_started"] = False
    elif record.get("rendered"):
        profile["steady_started"] = True
    _increment_counter(profile.setdefault("render_reason_counts", {}), record.get("render_reason"))
    _increment_counter(profile.setdefault("reuse_reason_counts", {}), record.get("reuse_reason"))
    _increment_counter(profile.setdefault("texture_path_counts", {}), record.get("texture_path"))
    if record.get("texture_cache_hit"):
        profile["texture_cache_hit_count"] = int(profile.get("texture_cache_hit_count", 0)) + 1
    if record.get("gpu_texture_update_available"):
        profile["gpu_texture_update_available_count"] = int(
            profile.get("gpu_texture_update_available_count", 0)
        ) + 1
    upload_bytes = _optional_int(record.get("texture_upload_bytes")) or 0
    profile["texture_upload_bytes"] = int(profile.get("texture_upload_bytes", 0)) + upload_bytes
    for field in TEXTURE_TIMING_FIELDS:
        _record_timing_stat(
            profile["texture_timing_stats"][field],
            float(record.get(field, 0.0)),
        )
    profile["recent_draws"].append(record)
    if len(profile["recent_draws"]) > RECENT_DRAW_LIMIT:
        del profile["recent_draws"][: len(profile["recent_draws"]) - RECENT_DRAW_LIMIT]


def summary(profile: dict[str, Any], artifact_write_ms: float = 0.0) -> dict[str, Any]:
    recent_draws = profile["recent_draws"]
    window_started_at_ns = _optional_int(profile.get("started_at_ns"))
    window_ended_at_ns = _optional_int(profile.get("ended_at_ns"))
    window_started_monotonic_ns = _optional_int(profile.get("started_monotonic_ns"))
    window_ended_monotonic_ns = _optional_int(profile.get("ended_monotonic_ns"))
    window_ms = _duration_ms(window_started_monotonic_ns, window_ended_monotonic_ns)
    draw_count = int(profile["draw_count"])
    render_count = int(profile["render_count"])
    reuse_count = max(0, draw_count - render_count)
    summary = {
        "enabled": True,
        "draw_count": draw_count,
        "render_count": render_count,
        "reuse_count": reuse_count,
        "profile_window_started_at_ns": window_started_at_ns,
        "profile_window_ended_at_ns": window_ended_at_ns,
        "profile_window_started_monotonic_ns": window_started_monotonic_ns,
        "profile_window_ended_monotonic_ns": window_ended_monotonic_ns,
        "profile_window_ms": window_ms,
        "render_fps": (
            render_count / (window_ms / 1_000.0)
            if window_ms is not None and window_ms > 0.0
            else None
        ),
        "render_cadence_ms": (
            window_ms / render_count
            if window_ms is not None and render_count > 0
            else None
        ),
        "camera_changed_draw_count": profile["camera_changed_draw_count"],
        "snapshot_changed_draw_count": profile["snapshot_changed_draw_count"],
        "draw_outcome_counts": {
            "rendered": render_count,
            "reused": reuse_count,
        },
        "draw_phase_counts": dict(profile.get("draw_phase_counts", {})),
        "render_reason_counts": dict(profile.get("render_reason_counts", {})),
        "reuse_reason_counts": dict(profile.get("reuse_reason_counts", {})),
        "texture_path_counts": dict(profile.get("texture_path_counts", {})),
        "texture_cache_hit_count": int(profile.get("texture_cache_hit_count", 0)),
        "gpu_texture_update_available_count": int(profile.get("gpu_texture_update_available_count", 0)),
        "texture_upload_bytes": int(profile.get("texture_upload_bytes", 0)),
        "phase_stats": profile["phase_stats"],
        "native_timing_stats": profile["native_timing_stats"],
        "texture_timing_stats": profile["texture_timing_stats"],
        "recent_draw_limit": RECENT_DRAW_LIMIT,
        "recent_draws": recent_draws,
        "session_artifact_write_ms": artifact_write_ms,
        "render_interval_measured_work_phases": list(RENDER_INTERVAL_WORK_PHASES),
        "thread_attribution": _thread_attribution(recent_draws),
    }
    steady_recent_draws = _recent_draws_by_phase(recent_draws, "steady")
    timeline_reset_recent_draws = _recent_draws_by_phase(recent_draws, "timeline_reset")
    summary["recent_phase_stats"] = _recent_phase_stats(recent_draws)
    summary["recent_draw_phase_counts"] = _recent_draw_phase_counts(recent_draws)
    summary["steady_recent_summary"] = _recent_window_summary(steady_recent_draws)
    summary["timeline_reset_recent_summary"] = _recent_window_summary(timeline_reset_recent_draws)
    summary["recent_render_interval_stats_ms"] = _recent_render_interval_stats(recent_draws)
    summary["recent_time_since_previous_draw_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "time_since_previous_draw_ms",
    )
    summary["recent_time_since_previous_draw_start_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "time_since_previous_draw_start_ms",
    )
    summary["recent_callback_wait_since_previous_draw_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "callback_wait_since_previous_draw_ms",
    )
    summary["recent_time_since_previous_render_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "time_since_previous_render_ms",
    )
    summary["recent_render_interval_measured_work_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "render_interval_measured_work_ms",
    )
    summary["recent_render_interval_measured_work_phase_stats_ms"] = (
        _recent_render_interval_work_phase_stats(recent_draws)
    )
    summary["recent_render_interval_unaccounted_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "render_interval_unaccounted_ms",
    )
    summary["recent_render_interval_callback_wait_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "render_interval_callback_wait_ms",
    )
    summary["recent_render_interval_unaccounted_after_callback_wait_stats_ms"] = (
        _recent_field_stats_ms(
            recent_draws,
            "render_interval_unaccounted_after_callback_wait_ms",
        )
    )
    summary["recent_texture_update_stats_ms"] = _recent_field_stats_ms(
        recent_draws,
        "texture_update_ms",
    )
    return summary


def _thread_attribution(recent_draws: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-thread cost decomposition (task02-09, spec success criterion 2).

    ``recent_main_thread_callback_stats_ms`` is the whole draw-callback
    cost (``viewport_callback_ms`` — it already contains the other main
    phases, so they are not summed on top);
    ``recent_render_thread_attributed_stats_ms`` is the per-draw sum of
    the phases attributed to the render thread. Their separation is the
    direct answer: main-thread callback cost excludes RPC time.
    """

    main_values: list[float] = []
    render_values: list[float] = []
    for record in recent_draws:
        timings = record.get("timings_ms", {})
        if not isinstance(timings, Mapping):
            continue
        try:
            main_values.append(float(timings.get("viewport_callback_ms", 0.0)))
        except (TypeError, ValueError):
            pass
        render_total = 0.0
        for phase in RENDER_THREAD_PHASES:
            try:
                render_total += max(0.0, float(timings.get(phase, 0.0)))
            except (TypeError, ValueError):
                continue
        render_values.append(render_total)
    return {
        "record_thread_field": "thread",
        "phase_threads": dict(PHASE_THREADS),
        "main_thread_phases": list(MAIN_THREAD_PHASES),
        "render_thread_phases": list(RENDER_THREAD_PHASES),
        "cross_thread_phases": list(CROSS_THREAD_PHASES),
        "recent_main_thread_callback_stats_ms": _timing_distribution(main_values),
        "recent_render_thread_attributed_stats_ms": _timing_distribution(render_values),
    }


def render_thread_summary(
    records: list[dict[str, Any]],
    recent_draws: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate the render thread's per-loop-iteration records (task02-09).

    The render thread appends records to its own list; this aggregation
    runs on the main thread at artifact-write time (single-writer). Loop
    records correlate to per-draw records by publication index plus
    snapshot key: a presented publication appears in both streams.
    """

    records = [record for record in records if isinstance(record, Mapping)]
    stats = {phase: _empty_timing_stat() for phase in RENDER_LOOP_TIMING_PHASES}
    status_counts: dict[str, int] = {}
    publications: dict[int, str] = {}
    for record in records:
        _increment_counter(status_counts, record.get("status"))
        timings = record.get("timings_ms", {})
        if isinstance(timings, Mapping):
            for phase in RENDER_LOOP_TIMING_PHASES:
                value = timings.get(phase)
                if value is None:
                    continue
                try:
                    _record_timing_stat(stats[phase], float(value))
                except (TypeError, ValueError):
                    continue
        publication_index = _optional_int(record.get("publication_index"))
        if publication_index is not None and publication_index > 0:
            publications[publication_index] = str(record.get("snapshot_key", ""))
    correlated_draws = 0
    key_matched_draws = 0
    for draw in recent_draws or []:
        if not isinstance(draw, Mapping):
            continue
        publication_index = _optional_int(draw.get("publication_index"))
        if publication_index is None or publication_index not in publications:
            continue
        correlated_draws += 1
        if str(draw.get("snapshot_key", "")) == publications[publication_index]:
            key_matched_draws += 1
    return {
        "thread": "render",
        "record_count": len(records),
        "status_counts": status_counts,
        "phase_stats": stats,
        "published_record_count": len(publications),
        "correlated_draw_count": correlated_draws,
        "snapshot_key_matched_draw_count": key_matched_draws,
        "records": [dict(record) for record in records],
    }


def _empty_timing_stat() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "total_ms": 0.0,
        "mean_ms": 0.0,
        "min_ms": None,
        "max_ms": None,
    }


def _record_timing_stat(stat: dict[str, float | int | None], value_ms: float) -> None:
    value = float(value_ms)
    count = int(stat["count"]) + 1
    total = float(stat["total_ms"]) + value
    stat["count"] = count
    stat["total_ms"] = total
    stat["mean_ms"] = total / count
    stat["min_ms"] = value if stat["min_ms"] is None else min(float(stat["min_ms"]), value)
    stat["max_ms"] = value if stat["max_ms"] is None else max(float(stat["max_ms"]), value)


def _native_timing_scope(scope: Any) -> str:
    if scope == "render_session_frame":
        return "render_result"
    return str(scope)


def _increment_counter(counters: dict[str, int], key: Any) -> None:
    if key is None:
        return
    name = str(key)
    if not name:
        return
    counters[name] = int(counters.get(name, 0)) + 1


def _elapsed_ms(previous_ns: int | None, current_ns: int | None) -> float | None:
    if previous_ns is None or current_ns is None or current_ns < previous_ns:
        return None
    return (current_ns - previous_ns) / 1_000_000.0


def _viewport_draw_outcome(record: Mapping[str, Any]) -> str:
    return "rendered" if record.get("rendered") else "reused"


def _viewport_render_reason(record: Mapping[str, Any]) -> str | None:
    if not record.get("rendered"):
        return None
    if record.get("composition_changed"):
        return "composition_changed"
    if record.get("camera_changed"):
        return "camera_changed"
    if record.get("snapshot_changed"):
        return "snapshot_changed"
    requested_samples = _optional_int(record.get("requested_additional_samples")) or 0
    if requested_samples > 0:
        return "refinement_samples"
    return "rendered"


def _viewport_reuse_reason(record: Mapping[str, Any]) -> str | None:
    if record.get("rendered"):
        return None
    completed_samples = _optional_int(record.get("completed_samples"))
    max_samples = _optional_int(record.get("max_samples"))
    if (
        completed_samples is not None
        and max_samples is not None
        and not render_requests.viewport_sampling_due(completed_samples, max_samples)
    ):
        return "reached_max_samples"
    requested_samples = _optional_int(record.get("requested_additional_samples"))
    if requested_samples == 0:
        return "no_additional_samples"
    return "reused"


def _draw_phase(profile: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    if record.get("timeline_reset"):
        return "timeline_reset"
    if not bool(profile.get("steady_started")):
        return "startup_warmup"
    return "steady"


def _render_interval_work_ms_by_phase(record: Mapping[str, Any]) -> dict[str, float]:
    timings = record.get("timings_ms", {})
    work_ms_by_phase = {phase: 0.0 for phase in RENDER_INTERVAL_WORK_PHASES}
    if not isinstance(timings, Mapping):
        return work_ms_by_phase
    for phase in RENDER_INTERVAL_WORK_PHASES:
        try:
            value_ms = float(timings.get(phase, 0.0))
        except (TypeError, ValueError):
            continue
        work_ms_by_phase[phase] = max(0.0, value_ms)
    return work_ms_by_phase


def _annotate_draw(profile: dict[str, Any], record: dict[str, Any]) -> None:
    started_monotonic_ns = _optional_int(record.get("started_monotonic_ns"))
    completed_monotonic_ns = _record_completed_monotonic_ns(record)
    previous_draw_started_ns = _optional_int(profile.get("last_draw_started_monotonic_ns"))
    previous_draw_completed_ns = _optional_int(profile.get("last_draw_monotonic_ns"))
    previous_render_ns = _optional_int(profile.get("last_render_monotonic_ns"))
    render_interval_ms = _elapsed_ms(previous_render_ns, completed_monotonic_ns)
    callback_wait_ms = _elapsed_ms(previous_draw_completed_ns, started_monotonic_ns)
    current_interval_callback_wait_ms = (
        max(0.0, callback_wait_ms)
        if callback_wait_ms is not None and previous_render_ns is not None
        else 0.0
    )
    current_work_ms_by_phase = _render_interval_work_ms_by_phase(record)
    accumulated_callback_wait_ms = float(profile.get("render_interval_callback_wait_ms") or 0.0)
    accumulated_work_ms_by_phase = profile.setdefault(
        "render_interval_work_ms_by_phase",
        {phase: 0.0 for phase in RENDER_INTERVAL_WORK_PHASES},
    )

    record["draw_outcome"] = _viewport_draw_outcome(record)
    record["render_reason"] = _viewport_render_reason(record)
    record["reuse_reason"] = _viewport_reuse_reason(record)
    record["time_since_previous_draw_ms"] = _elapsed_ms(previous_draw_completed_ns, completed_monotonic_ns)
    record["time_since_previous_draw_start_ms"] = _elapsed_ms(previous_draw_started_ns, started_monotonic_ns)
    record["callback_wait_since_previous_draw_ms"] = callback_wait_ms
    record["time_since_previous_render_ms"] = render_interval_ms
    if record.get("rendered") and render_interval_ms is not None:
        interval_callback_wait_ms = accumulated_callback_wait_ms + current_interval_callback_wait_ms
        interval_work_ms_by_phase = {
            phase: (
                float(accumulated_work_ms_by_phase.get(phase, 0.0))
                + current_work_ms_by_phase[phase]
            )
            for phase in RENDER_INTERVAL_WORK_PHASES
        }
        measured_work_ms = sum(interval_work_ms_by_phase.values())
        unaccounted_ms = max(0.0, render_interval_ms - measured_work_ms)
        unaccounted_after_callback_wait_ms = max(0.0, unaccounted_ms - interval_callback_wait_ms)
        record["render_interval_callback_wait_ms"] = interval_callback_wait_ms
        record["render_interval_measured_work_ms_by_phase"] = interval_work_ms_by_phase
        record["render_interval_measured_work_ms"] = measured_work_ms
        record["render_interval_unaccounted_ms"] = unaccounted_ms
        record["render_interval_unaccounted_after_callback_wait_ms"] = unaccounted_after_callback_wait_ms
    else:
        record["render_interval_callback_wait_ms"] = None
        record["render_interval_measured_work_ms_by_phase"] = None
        record["render_interval_measured_work_ms"] = None
        record["render_interval_unaccounted_ms"] = None
        record["render_interval_unaccounted_after_callback_wait_ms"] = None

    if completed_monotonic_ns is not None:
        if started_monotonic_ns is not None:
            profile["last_draw_started_monotonic_ns"] = started_monotonic_ns
        profile["last_draw_monotonic_ns"] = completed_monotonic_ns
        if record.get("rendered"):
            profile["last_render_monotonic_ns"] = completed_monotonic_ns
            profile["render_interval_callback_wait_ms"] = 0.0
            profile["render_interval_work_ms_by_phase"] = {
                phase: 0.0 for phase in RENDER_INTERVAL_WORK_PHASES
            }
    if not record.get("rendered") and previous_render_ns is not None:
        profile["render_interval_callback_wait_ms"] = (
            accumulated_callback_wait_ms + current_interval_callback_wait_ms
        )
        for phase in RENDER_INTERVAL_WORK_PHASES:
            accumulated_work_ms_by_phase[phase] = (
                float(accumulated_work_ms_by_phase.get(phase, 0.0))
                + current_work_ms_by_phase[phase]
            )


def _recent_draws_by_phase(recent_draws: list[dict[str, Any]], draw_phase: str) -> list[dict[str, Any]]:
    return [record for record in recent_draws if record.get("draw_phase") == draw_phase]


def _recent_draw_phase_counts(recent_draws: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in recent_draws:
        _increment_counter(counts, record.get("draw_phase"))
    return counts


def _recent_window_summary(recent_draws: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "draw_count": len(recent_draws),
        "render_count": sum(1 for record in recent_draws if record.get("rendered")),
        "phase_stats": _recent_phase_stats(recent_draws),
        "render_interval_stats_ms": _recent_render_interval_stats(recent_draws),
        "render_interval_measured_work_stats_ms": _recent_field_stats_ms(
            recent_draws,
            "render_interval_measured_work_ms",
        ),
        "render_interval_callback_wait_stats_ms": _recent_field_stats_ms(
            recent_draws,
            "render_interval_callback_wait_ms",
        ),
        "render_interval_unaccounted_after_callback_wait_stats_ms": _recent_field_stats_ms(
            recent_draws,
            "render_interval_unaccounted_after_callback_wait_ms",
        ),
    }


def _recent_phase_stats(recent_draws: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    stats = {phase: _empty_timing_stat() for phase in TIMING_PHASES}
    for record in recent_draws:
        timings = record.get("timings_ms", {})
        for phase in TIMING_PHASES:
            _record_timing_stat(stats[phase], float(timings.get(phase, 0.0)))
    return stats


def _recent_render_interval_stats(recent_draws: list[dict[str, Any]]) -> dict[str, float | int | None]:
    render_times_ns: list[int] = []
    for record in recent_draws:
        if not record.get("rendered"):
            continue
        timestamp_ns = _record_completed_monotonic_ns(record)
        if timestamp_ns is not None:
            render_times_ns.append(timestamp_ns)
    intervals_ms = [
        (current_ns - previous_ns) / 1_000_000.0
        for previous_ns, current_ns in zip(render_times_ns, render_times_ns[1:])
        if current_ns >= previous_ns
    ]
    return _timing_distribution(intervals_ms)


def _recent_field_stats_ms(recent_draws: list[dict[str, Any]], field_name: str) -> dict[str, float | int | None]:
    values_ms: list[float] = []
    for record in recent_draws:
        value = record.get(field_name)
        if value is None:
            continue
        try:
            values_ms.append(float(value))
        except (TypeError, ValueError):
            continue
    return _timing_distribution(values_ms)


def _recent_render_interval_work_phase_stats(
    recent_draws: list[dict[str, Any]]
) -> dict[str, dict[str, float | int | None]]:
    stats = {phase: _empty_timing_stat() for phase in RENDER_INTERVAL_WORK_PHASES}
    for record in recent_draws:
        values = record.get("render_interval_measured_work_ms_by_phase")
        if not isinstance(values, Mapping):
            continue
        for phase in RENDER_INTERVAL_WORK_PHASES:
            try:
                value_ms = float(values.get(phase, 0.0))
            except (TypeError, ValueError):
                continue
            _record_timing_stat(stats[phase], value_ms)
    return stats


def _timing_distribution(values_ms: list[float]) -> dict[str, float | int | None]:
    if not values_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    sorted_values = sorted(float(value) for value in values_ms)
    total_ms = sum(sorted_values)
    return {
        "count": len(sorted_values),
        "total_ms": total_ms,
        "mean_ms": total_ms / len(sorted_values),
        "min_ms": sorted_values[0],
        "p50_ms": _median(sorted_values),
        "p95_ms": _nearest_rank(sorted_values, 0.95),
        "max_ms": sorted_values[-1],
    }


def _median(sorted_values: list[float]) -> float:
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _nearest_rank(sorted_values: list[float], percentile: float) -> float:
    position = len(sorted_values) * percentile
    rank = int(position)
    if rank < position:
        rank += 1
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


def _record_completed_monotonic_ns(record: Mapping[str, Any]) -> int | None:
    return _optional_int(record.get("ended_monotonic_ns")) or _optional_int(
        record.get("started_monotonic_ns")
    )


def _duration_ms(started_at_ns: int | None, ended_at_ns: int | None) -> float | None:
    if started_at_ns is None or ended_at_ns is None or ended_at_ns <= started_at_ns:
        return None
    return (ended_at_ns - started_at_ns) / 1_000_000.0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CROSS_THREAD_PHASES",
    "MAIN_THREAD_PHASES",
    "NATIVE_TIMING_SCOPES",
    "PHASE_THREADS",
    "RECENT_DRAW_LIMIT",
    "RENDER_INTERVAL_WORK_PHASES",
    "RENDER_LOOP_TIMING_PHASES",
    "RENDER_THREAD_PHASES",
    "TEXTURE_TIMING_FIELDS",
    "TIMING_PHASES",
    "new",
    "record",
    "render_thread_summary",
    "snapshot_key_token",
    "summary",
]
