# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import viewport_profile  # noqa: E402


def _timings(**values: float) -> dict[str, float]:
    timings = {phase: 0.0 for phase in viewport_profile.TIMING_PHASES}
    timings.update(values)
    return timings


def test_viewport_profile_records_cadence_and_timing_summary() -> None:
    profile = viewport_profile.new()
    first = {
        "rendered": True,
        "camera_changed": False,
        "snapshot_changed": True,
        "timeline_reset": False,
        "requested_additional_samples": 1,
        "completed_samples": 1,
        "max_samples": 4,
        "started_at_ns": 1_000,
        "ended_at_ns": 2_000,
        "started_monotonic_ns": 10_000_000,
        "ended_monotonic_ns": 20_000_000,
        "texture_path": "new_texture",
        "texture_cache_hit": False,
        "gpu_texture_update_available": True,
        "texture_upload_bytes": 16,
        "timings_ms": _timings(render_ms=2.5),
    }
    second = {
        "rendered": False,
        "camera_changed": False,
        "snapshot_changed": False,
        "timeline_reset": False,
        "requested_additional_samples": 0,
        "completed_samples": 4,
        "max_samples": 4,
        "started_at_ns": 2_000,
        "ended_at_ns": 3_000,
        "started_monotonic_ns": 21_000_000,
        "ended_monotonic_ns": 22_000_000,
        "texture_path": "reuse",
        "texture_cache_hit": True,
        "gpu_texture_update_available": True,
        "texture_upload_bytes": 0,
        "timings_ms": _timings(),
    }

    viewport_profile.record(profile, first)
    viewport_profile.record(profile, second)
    summary = viewport_profile.summary(profile, artifact_write_ms=0.5)

    assert summary["draw_count"] == 2
    assert summary["render_count"] == 1
    assert summary["reuse_reason_counts"] == {"reached_max_samples": 1}
    assert summary["texture_path_counts"] == {"new_texture": 1, "reuse": 1}
    assert summary["texture_upload_bytes"] == 16
    assert summary["session_artifact_write_ms"] == 0.5
    assert summary["phase_stats"]["render_ms"]["mean_ms"] == 1.25


def test_unbounded_view_is_not_classified_as_reaching_max_samples() -> None:
    profile = viewport_profile.new()
    viewport_profile.record(
        profile,
        {
            "rendered": False,
            "camera_changed": False,
            "snapshot_changed": False,
            "timeline_reset": False,
            "requested_additional_samples": 0,
            "completed_samples": 8,
            "max_samples": 0,
            "texture_path": "reuse",
            "texture_cache_hit": True,
            "gpu_texture_update_available": True,
            "texture_upload_bytes": 0,
            "timings_ms": _timings(),
        },
    )

    assert viewport_profile.summary(profile, 0.0)["reuse_reason_counts"] == {
        "no_additional_samples": 1
    }


def test_viewport_profile_accepts_legacy_native_render_frame_scope() -> None:
    profile = viewport_profile.new()
    record = {
        "rendered": True,
        "camera_changed": False,
        "snapshot_changed": False,
        "timings_ms": _timings(),
        "native_timings": {
            "render_session_frame": {
                "total_native_ms": 7.0,
            },
        },
    }

    viewport_profile.record(profile, record)
    summary = viewport_profile.summary(profile)

    assert "render_session_frame" not in summary["native_timing_stats"]
    assert summary["native_timing_stats"]["render_result"]["total_native_ms"]["max_ms"] == 7.0
