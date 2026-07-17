#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared workload and record logic for Blender navigation measurements."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPETITIONS = 1
SCHEMA_VERSION = 12
WARMUP_NS = 2_000_000_000
MEASUREMENT_NS = 10_000_000_000
NAVIGATION_DRIVER = "blender-viewport-closed-loop-turntable-v1"
NAVIGATION_STEP_DEGREES = 1.0
MAX_SAFE_INTEGER = 2**53 - 1


class ContractError(ValueError):
    """The input is not strict JSON or violates the navigation contract."""


def _parse_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def _parse_integer(value: str) -> int:
    parsed = int(value)
    if abs(parsed) > MAX_SAFE_INTEGER:
        raise ContractError(f"JSON integer is outside the safe range: {value}")
    return parsed


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"non-finite JSON number: {value}")
    return parsed


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load JSON while rejecting extensions that cannot round-trip reliably."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_parse_constant,
            parse_int=_parse_integer,
            parse_float=_parse_float,
            object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(str(error)) from error


def navigation_workload(
    fixture: Mapping[str, Any],
    color_presentation: str,
    *,
    repetition_count: int = DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Return one selected navigation workload and its scene identity."""

    pixel_formats = {
        "scene_linear_hdr": "rgba16f_scene_linear",
        "ldr_rgba8_display_passthrough": "rgba8_display",
    }
    if color_presentation not in pixel_formats:
        raise ValueError(
            f"unsupported navigation color presentation: {color_presentation}"
        )
    return {
        "fixture_id": fixture["id"],
        "blend_file_sha256": fixture["sha256"],
        "navigation_driver": NAVIGATION_DRIVER,
        "navigation_step_degrees": NAVIGATION_STEP_DEGREES,
        "width_px": 1280,
        "height_px": 720,
        "color_presentation": color_presentation,
        "pixel_format": pixel_formats[color_presentation],
        "min_samples": 1,
        "max_samples": 128,
        "warmup_duration_ns": WARMUP_NS,
        "measurement_duration_ns": MEASUREMENT_NS,
        "repetition_count": repetition_count,
        "fresh_session_per_repetition": True,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


FRAME_EVENT_KEYS = {
    "publication_index",
    "render_started_monotonic_ns",
    "post_pixel_monotonic_ns",
}


def _monotonic_ns(value: Any) -> int:
    if not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value:
        raise ValueError
    return int(value)


def finish_frame_latency_repetition(
    *,
    repetition_index: int,
    warmup_start_ns: int,
    measurement_start_ns: int,
    measurement_end_ns: int,
    frame_events: Sequence[tuple[int, int, int]],
    stopped_view_complete: bool,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain only frames observed during the fixed measurement window."""

    measured = [
        {
            "publication_index": publication,
            "render_started_monotonic_ns": str(render_started_ns),
            "post_pixel_monotonic_ns": str(post_pixel_ns),
        }
        for publication, render_started_ns, post_pixel_ns in frame_events
        if measurement_start_ns <= post_pixel_ns < measurement_end_ns
    ]
    return {
        "measurement_complete": bool(stopped_view_complete and measured),
        "repetition_index": repetition_index,
        "warmup_start_monotonic_ns": str(warmup_start_ns),
        "measurement_start_monotonic_ns": str(measurement_start_ns),
        "measurement_end_monotonic_ns": str(measurement_end_ns),
        "frame_events": measured,
        "materialization": dict(materialization),
    }


def _valid_materialization(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"generation", "render_request"}:
        return False
    generation = value["generation"]
    request = value["render_request"]
    if (
        not isinstance(generation, Mapping)
        or set(generation) != {"digest", "usd_sha256", "runtime_files"}
        or not isinstance(generation["digest"], str)
        or not generation["digest"]
        or not _is_sha256(generation["usd_sha256"])
        or not isinstance(generation["runtime_files"], list)
        or not generation["runtime_files"]
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or not _is_sha256(item["sha256"])
            for item in generation["runtime_files"]
        )
    ):
        return False
    return (
        isinstance(request, Mapping)
        and set(request) == {"camera_prim_path", "render_product_path"}
        and all(isinstance(value, str) and value for value in request.values())
    )


def validate_frame_latency_repetition(
    index: int,
    repetition: Mapping[str, Any],
) -> list[str]:
    expected_keys = {
        "measurement_complete",
        "repetition_index",
        "warmup_start_monotonic_ns",
        "measurement_start_monotonic_ns",
        "measurement_end_monotonic_ns",
        "frame_events",
        "materialization",
    }
    if set(repetition) != expected_keys or repetition.get("measurement_complete") is not True:
        return ["runs[].fields"]
    if repetition.get("repetition_index") != index:
        return ["runs[].identity"]
    try:
        warmup_start = _monotonic_ns(repetition["warmup_start_monotonic_ns"])
        start = _monotonic_ns(repetition["measurement_start_monotonic_ns"])
        end = _monotonic_ns(repetition["measurement_end_monotonic_ns"])
        events = repetition["frame_events"]
        if (
            start - warmup_start != WARMUP_NS
            or end - start != MEASUREMENT_NS
            or not isinstance(events, list)
            or not events
            or not _valid_materialization(repetition["materialization"])
        ):
            raise ValueError
        previous_publication = 0
        previous_start = -1
        previous_completion = -1
        for event in events:
            if not isinstance(event, Mapping) or set(event) != FRAME_EVENT_KEYS:
                raise ValueError
            publication = event["publication_index"]
            render_started = _monotonic_ns(event["render_started_monotonic_ns"])
            post_pixel = _monotonic_ns(event["post_pixel_monotonic_ns"])
            if (
                type(publication) is not int
                or publication <= previous_publication
                or render_started > post_pixel
                or render_started <= previous_start
                or not start <= post_pixel < end
                or post_pixel <= previous_completion
            ):
                raise ValueError
            previous_publication = publication
            previous_start = render_started
            previous_completion = post_pixel
    except (KeyError, TypeError, ValueError):
        return ["runs[].frame_events"]
    return []


def run_frame_latency_measurements(
    repetition_count: int,
    run_repetition: Callable[[int], dict[str, Any]],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index in range(repetition_count):
        try:
            repetition = run_repetition(index)
        except Exception as error:
            repetition = {
                "measurement_complete": False,
                "repetition_index": index,
                "error": f"{type(error).__name__}: {error}",
            }
        runs.append(repetition)
    return runs


def validate_frame_latency_record(record: Mapping[str, Any]) -> list[str]:
    """Validate one Blender moving-view render throughput record."""

    blockers: list[str] = []
    expected_keys = {
        "schema_version",
        "artifact_id",
        "case_kind",
        "workload",
        "runs",
    }
    if set(record) != expected_keys:
        blockers.append("record.fields")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("artifact_id") != "ovrtx-navigation-render-throughput"
        or record.get("case_kind") != "blender"
    ):
        blockers.append("record.identity")
    workload = record.get("workload")
    if not isinstance(workload, Mapping):
        return sorted(set(blockers + ["workload"]))
    expected_workload_keys = {
        "fixture_id",
        "blend_file_sha256",
        "navigation_driver",
        "navigation_step_degrees",
        "width_px",
        "height_px",
        "color_presentation",
        "pixel_format",
        "min_samples",
        "max_samples",
        "warmup_duration_ns",
        "measurement_duration_ns",
        "repetition_count",
        "fresh_session_per_repetition",
    }
    repetitions = workload.get("repetition_count")
    if (
        set(workload) != expected_workload_keys
        or not isinstance(workload.get("fixture_id"), str)
        or not workload["fixture_id"]
        or not _is_sha256(workload.get("blend_file_sha256"))
        or type(repetitions) is not int
        or repetitions <= 0
        or workload.get("navigation_driver") != NAVIGATION_DRIVER
        or workload.get("navigation_step_degrees") != NAVIGATION_STEP_DEGREES
        or workload.get("width_px") != 1280
        or workload.get("height_px") != 720
        or workload.get("min_samples") != 1
        or workload.get("max_samples") != 128
        or workload.get("warmup_duration_ns") != WARMUP_NS
        or workload.get("measurement_duration_ns") != MEASUREMENT_NS
        or workload.get("fresh_session_per_repetition") is not True
        or (
            workload.get("color_presentation"),
            workload.get("pixel_format"),
        )
        not in {
            ("scene_linear_hdr", "rgba16f_scene_linear"),
            ("ldr_rgba8_display_passthrough", "rgba8_display"),
        }
    ):
        blockers.append("workload")
        return sorted(set(blockers))
    runs = record.get("runs")
    expected_identities = list(range(repetitions))
    if not isinstance(runs, list) or len(runs) != len(expected_identities):
        return sorted(set(blockers + ["runs"]))
    for run, index in zip(runs, expected_identities, strict=True):
        if not isinstance(run, Mapping):
            blockers.append("runs[].fields")
            continue
        if run.get("measurement_complete") is False and "error" in run:
            if (
                set(run)
                != {
                    "measurement_complete",
                    "repetition_index",
                    "error",
                }
                or run.get("repetition_index") != index
                or not isinstance(run.get("error"), str)
                or not run["error"]
            ):
                blockers.append("runs[].identity")
            continue
        blockers.extend(validate_frame_latency_repetition(index, run))
    return sorted(set(blockers))
