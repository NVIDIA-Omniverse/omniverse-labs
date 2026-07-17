# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_blender_navigation  # noqa: E402
import run_blender_light_edit_responsiveness  # noqa: E402
import validation  # noqa: E402
from ovrtx_blender_example.viewport_handoff import ViewSnapshot  # noqa: E402


def _materialization() -> dict[str, object]:
    return {
        "generation": {
            "digest": "generation-1",
            "usd_sha256": "b" * 64,
            "runtime_files": [{"path": "scene.usda", "sha256": "c" * 64}],
        },
        "render_request": {
            "camera_prim_path": "/Camera",
            "render_product_path": "/Render/Product",
        },
    }


def _measurement() -> run_blender_navigation._ThroughputMeasurement:
    return run_blender_navigation._ThroughputMeasurement(
        {
            "repetition_index": 0,
            "max_samples": 128,
        },
        _materialization(),
    )


@pytest.mark.parametrize(
    "command",
    ("blender", "/prepared/blender", r"C:\prepared\blender.exe"),
)
def test_prepared_blender_keeps_interactive_timeout(command: str) -> None:
    assert validation._command_timeout([command, "--window-geometry"]) == 180


def test_navigation_allows_cold_shader_compilation() -> None:
    assert validation._command_timeout(
        ["/prepared/blender", "--python", "/tests/run_blender_navigation.py"]
    ) == 600


def test_light_edit_allows_cold_shader_compilation_before_measurement() -> None:
    script = "/tests/run_blender_light_edit_responsiveness.py"
    assert validation._command_timeout(["/prepared/blender", "--python", script]) == 975
    assert validation._command_timeout(
        ["/prepared/blender", "--python", script, "--", "--inside-blender"]
    ) == 960


@pytest.mark.parametrize(
    ("comparison", "expected"),
    (
        ({"outcome": "regression", "metrics": {"alpha_mismatches": 0}}, True),
        ({"outcome": "pass", "metrics": {"alpha_mismatches": 0}}, False),
        ({"outcome": "unavailable", "metrics": None}, False),
        ({"outcome": "regression", "metrics": {"alpha_mismatches": 1}}, False),
    ),
)
def test_light_edit_requires_an_opaque_visible_change(
    comparison: dict[str, object], expected: bool
) -> None:
    assert run_blender_light_edit_responsiveness._visible_change(comparison) is expected


def test_prepared_blender_keeps_background_render_timeout() -> None:
    assert validation._command_timeout(["/prepared/blender", "--background"]) == 7200


def test_frame_latency_event_uses_the_presented_frame_boundaries() -> None:
    frame = SimpleNamespace(
        publication_index=7,
        timing_marks={"render_call_started_monotonic_ns": 100},
    )

    assert run_blender_navigation._frame_latency_event(frame, 150) == (7, 100, 150)


def test_frame_latency_event_fails_closed_without_a_valid_start() -> None:
    frame = SimpleNamespace(publication_index=7, timing_marks={})

    with pytest.raises(RuntimeError, match="no valid render start"):
        run_blender_navigation._frame_latency_event(frame, 150)


def test_render_throughput_advances_once_per_presented_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement = _measurement()
    monkeypatch.setattr(
        run_blender_navigation.time, "perf_counter_ns", lambda: 100
    )

    class Quaternion:
        def __init__(self, axis, angle):
            self.axis = axis
            self.angle = angle

        def __matmul__(self, other):
            return self.angle, other

    monkeypatch.setitem(
        sys.modules, "mathutils", SimpleNamespace(Quaternion=Quaternion)
    )
    initial_rotation = SimpleNamespace(copy=lambda: "initial")
    region_data = SimpleNamespace(view_rotation=initial_rotation)

    measurement.start(region_data)
    assert measurement.advance_render_throughput(region_data) is True
    assert measurement.advance_render_throughput(region_data) is False

    measurement.last_presented_frame = object()
    assert measurement.advance_render_throughput(region_data) is True
    assert measurement.advance_render_throughput(region_data) is False

    measurement.last_presented_frame = object()
    assert measurement.advance_render_throughput(region_data) is True
    assert measurement.view_index == 3


def test_post_pixel_records_each_new_render_result_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovrtx_blender_example import engine

    measurement = _measurement()
    measurement.started = True
    measurement.phase = "settle"
    measurement.final_view_matrix = ((2.0,),)
    first_result = object()
    engine_instance = SimpleNamespace(_presented_frame=None)
    monkeypatch.setattr(engine, "_ACTIVE_VIEWPORT_ENGINES", [engine_instance])

    def frame(
        publication: int,
        result: object,
        matrix: object,
        samples: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            publication_index=publication,
            render_result=result,
            snapshot_key=ViewSnapshot(camera_matrix=matrix).key,
            completed_samples=samples,
            timing_marks={"render_call_started_monotonic_ns": publication},
        )

    first = frame(1, first_result, ((1.0,),), 1)
    engine_instance._presented_frame = first
    measurement.post_pixel()
    measurement.post_pixel()  # cached redraw of the same publication

    engine_instance._presented_frame = frame(2, first_result, ((1.0,),), 1)
    measurement.post_pixel()  # re-publication of the same rendered result

    engine_instance._presented_frame = frame(3, object(), ((1.0,),), 2)
    measurement.post_pixel()  # same-view refinement from a new render invocation

    unpresented_failure = frame(98, object(), ((8.0,),), 0)
    unpresented_lifecycle = frame(99, object(), ((9.0,),), 128)
    assert unpresented_failure is not engine_instance._presented_frame
    assert unpresented_lifecycle is not engine_instance._presented_frame
    measurement.post_pixel()

    engine_instance._presented_frame = frame(4, object(), ((2.0,),), 128)
    measurement.post_pixel()

    assert [event[0] for event in measurement.frame_events] == [1, 3, 4]
    assert measurement.stopped_view_complete is True


def test_production_materialization_records_generated_closure_and_intent(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "generation" / "scene.usda"
    texture_path = usd_path.parent / "textures" / "albedo.png"
    texture_path.parent.mkdir(parents=True)
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    texture_path.write_bytes(b"texture")
    generation = SimpleNamespace(
        digest="generation-1", materialize_usd=lambda: str(usd_path)
    )
    request = SimpleNamespace(
        camera_prim_path="/GeneratedCamera",
        render_product_path="/GeneratedProduct",
    )

    artifact = run_blender_navigation._production_materialization(
        generation, request
    )

    assert [item["path"] for item in artifact["generation"]["runtime_files"]] == [
        "scene.usda",
        "textures/albedo.png",
    ]
    assert artifact["render_request"] == {
        "camera_prim_path": "/GeneratedCamera",
        "render_product_path": "/GeneratedProduct",
    }


def test_view3d_context_requires_window_area_and_region() -> None:
    bpy = SimpleNamespace(context=SimpleNamespace(window=None))
    assert run_blender_navigation._view3d_context(bpy) is None

    window = SimpleNamespace(screen=SimpleNamespace(areas=[]))
    bpy.context.window = window
    assert run_blender_navigation._view3d_context(bpy) is None

    region = SimpleNamespace(type="WINDOW", width=1280, height=720)
    space = SimpleNamespace(region_3d=object())
    area = SimpleNamespace(
        type="VIEW_3D",
        regions=[],
        spaces=SimpleNamespace(active=space),
    )
    window.screen.areas = [area]
    assert run_blender_navigation._view3d_context(bpy) is None

    area.regions = [region]

    assert run_blender_navigation._view3d_context(bpy) == (
        window,
        area,
        region,
        space,
    )
