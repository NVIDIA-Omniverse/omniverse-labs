# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import navigation  # noqa: E402


def _event(publication: int, started: int, completed: int) -> tuple[int, int, int]:
    return publication, started, completed


def _materialization(digest: str = "generation-1") -> dict[str, object]:
    return {
        "generation": {
            "digest": digest,
            "usd_sha256": "b" * 64,
            "runtime_files": [{"path": "scene.usda", "sha256": "c" * 64}],
        },
        "render_request": {
            "camera_prim_path": "/OvrtxCamera",
            "render_product_path": "/Render/Product",
        },
    }


def _repetition(index: int = 0) -> dict[str, object]:
    warmup_start = 1_000_000_000
    start = warmup_start + navigation.WARMUP_NS
    end = start + navigation.MEASUREMENT_NS
    return navigation.finish_frame_latency_repetition(
        repetition_index=index,
        warmup_start_ns=warmup_start,
        measurement_start_ns=start,
        measurement_end_ns=end,
        frame_events=[
            _event(1, start - 2, start - 1),
            _event(2, start, start + 10),
            _event(3, start + 20, start + 30),
            _event(4, end, end + 1),
        ],
        stopped_view_complete=True,
        materialization=_materialization(),
    )


def _record(*, repetitions: int = 1) -> dict[str, object]:
    runs = navigation.run_frame_latency_measurements(
        repetitions, _repetition
    )
    return {
        "schema_version": navigation.SCHEMA_VERSION,
        "artifact_id": "ovrtx-navigation-render-throughput",
        "case_kind": "blender",
        "workload": navigation.navigation_workload(
            {"id": "perf_junk_shop_1280x720", "sha256": "a" * 64},
            "scene_linear_hdr",
            repetition_count=repetitions,
        ),
        "runs": runs,
    }


def test_raw_measurement_schema_version() -> None:
    assert navigation.SCHEMA_VERSION == 12


def test_load_json_accepts_safe_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    expected = {"integer": 9007199254740991, "values": [0.0, 2, "three"]}
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert navigation.load_json(path) == expected


def test_frame_latency_repetition_keeps_only_fixed_window_completions() -> None:
    repetition = _repetition()

    assert repetition["measurement_complete"] is True
    assert [event["publication_index"] for event in repetition["frame_events"]] == [
        2,
        3,
    ]
    assert repetition["materialization"] == _materialization()
    assert navigation.validate_frame_latency_repetition(0, repetition) == []


def test_frame_latency_repetition_requires_stopped_view_completion() -> None:
    repetition = _repetition()
    repetition["measurement_complete"] = False

    assert navigation.validate_frame_latency_repetition(0, repetition) == [
        "runs[].fields"
    ]


@pytest.mark.parametrize(
    "field,value",
    (
        ("render_started_monotonic_ns", "3000000011"),
        ("post_pixel_monotonic_ns", "2999999999"),
    ),
)
def test_frame_latency_repetition_rejects_invalid_boundaries(
    field: str, value: str
) -> None:
    repetition = _repetition()
    repetition["frame_events"][0][field] = value

    assert navigation.validate_frame_latency_repetition(0, repetition) == [
        "runs[].frame_events"
    ]


def test_frame_latency_repetition_rejects_duplicate_publication() -> None:
    repetition = _repetition()
    repetition["frame_events"][1]["publication_index"] = 2

    assert navigation.validate_frame_latency_repetition(0, repetition) == [
        "runs[].frame_events"
    ]


def test_record_rejects_non_hex_sha256_identities() -> None:
    record = _record()
    record["workload"]["blend_file_sha256"] = "z" * 64
    assert navigation.validate_frame_latency_record(record) == ["workload"]

    repetition = _repetition()
    repetition["materialization"]["generation"]["usd_sha256"] = "z" * 64
    assert navigation.validate_frame_latency_repetition(0, repetition) == [
        "runs[].frame_events"
    ]

    repetition = _repetition()
    repetition["materialization"]["generation"]["runtime_files"][0]["sha256"] = (
        "z" * 64
    )
    assert navigation.validate_frame_latency_repetition(0, repetition) == [
        "runs[].frame_events"
    ]


def test_measurement_composition_is_caller_selected() -> None:
    calls: list[int] = []

    def run(index: int) -> dict[str, object]:
        calls.append(index)
        return _repetition(index)

    runs = navigation.run_frame_latency_measurements(1, run)

    assert calls == [0]
    assert len(runs) == 1


def test_frame_latency_record_validates_selected_composition() -> None:
    record = _record()

    assert navigation.validate_frame_latency_record(record) == []


def test_frame_latency_record_rejects_old_proxy_contract() -> None:
    record = _record()
    record["schema_version"] = 8
    record["artifact_id"] = "ovrtx-navigation-measurement"
    record["runs"][0]["legacy_proxy_events"] = []

    assert navigation.validate_frame_latency_record(record) == [
        "record.identity",
        "runs[].fields",
    ]


def test_frame_latency_record_rejects_duplicate_failure_summary() -> None:
    record = _record()
    record["failures"] = []

    assert navigation.validate_frame_latency_record(record) == ["record.fields"]


def test_navigation_workload_binds_presentation_to_pixel_contract() -> None:
    fixture = {"id": "perf_junk_shop_1280x720", "sha256": "a" * 64}

    hdr = navigation.navigation_workload(fixture, "scene_linear_hdr")
    ldr = navigation.navigation_workload(fixture, "ldr_rgba8_display_passthrough")

    assert hdr["pixel_format"] == "rgba16f_scene_linear"
    assert ldr["pixel_format"] == "rgba8_display"


def test_navigation_workload_binds_blend_and_closed_loop_driver() -> None:
    fixture = {"id": "perf_junk_shop_1280x720", "sha256": "a" * 64}
    workload = navigation.navigation_workload(fixture, "scene_linear_hdr")

    assert workload["blend_file_sha256"] == "a" * 64
    assert workload["navigation_driver"] == navigation.NAVIGATION_DRIVER
    assert (
        workload["navigation_step_degrees"]
        == navigation.NAVIGATION_STEP_DEGREES
    )
