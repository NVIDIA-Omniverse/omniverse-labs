#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure fixed latest-view navigation through Blender's production viewport."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

HARNESS_SCRIPTS_DIR = Path(__file__).resolve().parent


def _bootstrap_source_root(argv: Sequence[str]) -> Path:
    if "--source-root" not in argv:
        return HARNESS_SCRIPTS_DIR.parent
    index = list(argv).index("--source-root")
    if index + 1 >= len(argv):
        return HARNESS_SCRIPTS_DIR.parent
    return Path(argv[index + 1]).expanduser().resolve()


ROOT = _bootstrap_source_root(sys.argv)
SCRIPTS_DIR = HARNESS_SCRIPTS_DIR
ADDON_DIR = ROOT / "addon"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
for _path in (SCRIPTS_DIR, ADDON_DIR, FIXTURES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import navigation as shared_navigation  # noqa: E402
import validation  # noqa: E402
from fixture_manifest import fixture_input, load_manifest  # noqa: E402
from ovrtx_blender_example import color_presentation, render_requests  # noqa: E402

DEFAULT_FIXTURE_MANIFEST = FIXTURES_DIR


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "performance" / "blender.json",
    )
    parser.add_argument("--blender-command", default="blender")
    parser.add_argument("--native-client-path", type=Path)
    parser.add_argument("--native-client-module", default="ovrtx_bridge_client")
    parser.add_argument("--worker-command")
    parser.add_argument(
        "--repetitions", type=int, default=shared_navigation.DEFAULT_REPETITIONS
    )
    parser.add_argument(
        "--color-presentation",
        choices=(
            color_presentation.MODE_SCENE_LINEAR_HDR,
            color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        ),
        default=(
            os.environ.get(color_presentation.ENV_COLOR_PRESENTATION_MODE)
            or color_presentation.DEFAULT_MODE
        ),
    )
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv))
    args.source_root = args.source_root.expanduser().resolve()
    args.manifest = (
        args.manifest or args.source_root / "tests" / "fixtures"
    )
    if args.inside_blender and args.config is None:
        parser.error("--inside-blender requires --config")
    if not args.inside_blender and (
        args.native_client_path is None
        or not args.worker_command
    ):
        parser.error("--native-client-path and --worker-command are required")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    return args


def _argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def _frame_latency_event(frame: Any, post_pixel_ns: int) -> tuple[int, int, int]:
    """Join the perf-only callback to the exact frame Blender just drew."""

    started_ns = frame.timing_marks.get("render_call_started_monotonic_ns")
    if not started_ns or int(started_ns) > post_pixel_ns:
        raise RuntimeError("presented frame has no valid render start boundary")
    return int(frame.publication_index), int(started_ns), post_pixel_ns


class _ThroughputMeasurement:
    def __init__(
        self, config: Mapping[str, Any], materialization: Mapping[str, Any]
    ) -> None:
        self.config = config
        self.index = int(config["repetition_index"])
        self.materialization = dict(materialization)
        self.frame_events: list[tuple[int, int, int]] = []
        self.window: dict[str, int] = {}
        self.last_presented_frame: Any = None
        self.last_presented_result: Any = None
        self.final_view_matrix: Any = None
        self.stopped_view_complete = False
        self.handler: Any = None
        self.started = False
        self.phase = "idle"
        self.last_advanced_frame: Any = object()
        self.initial_view_rotation: Any = None
        self.view_index = 0

    def start(self, region_data: Any) -> None:
        now_ns = time.perf_counter_ns()
        self.window.update(
            warmup_start=now_ns,
            start=now_ns + shared_navigation.WARMUP_NS,
            end=(
                now_ns
                + shared_navigation.WARMUP_NS
                + shared_navigation.MEASUREMENT_NS
            ),
        )
        self.started = True
        self.initial_view_rotation = region_data.view_rotation.copy()
        self.phase = "drive"

    def advance_render_throughput(self, region_data: Any) -> bool:
        from ovrtx_blender_example import render_requests

        if self.phase == "drive":
            if time.perf_counter_ns() >= self.window["end"]:
                self.phase = "release"
                return False
            if self.last_presented_frame is self.last_advanced_frame:
                return False
            self.last_advanced_frame = self.last_presented_frame
            self.view_index += 1
            from mathutils import Quaternion

            turn = Quaternion(
                (0.0, 0.0, 1.0),
                math.radians(
                    shared_navigation.NAVIGATION_STEP_DEGREES * self.view_index
                ),
            )
            region_data.view_rotation = turn @ self.initial_view_rotation
            return True
        if self.phase == "release":
            self.phase = "settle"
            return False
        if self.phase == "settle" and self.final_view_matrix is None:
            actual = render_requests.matrix_to_usd_rows(
                region_data.view_matrix.inverted()
            )
            self.final_view_matrix = render_requests.stable_camera_matrix(actual)
            if self.final_view_matrix is None:
                raise RuntimeError("Blender final camera matrix normalization failed")
            self._note_stopped_view()
        return False

    def _note_stopped_view(self) -> None:
        frame = self.last_presented_frame
        if frame is None or self.final_view_matrix is None:
            return
        snapshot_key = frame.snapshot_key
        camera_matrix = (
            snapshot_key[-2]
            if isinstance(snapshot_key, tuple) and len(snapshot_key) >= 2
            else None
        )
        if (
            camera_matrix == self.final_view_matrix
            and int(frame.completed_samples) >= int(self.config["max_samples"])
        ):
            self.stopped_view_complete = True

    def post_pixel(self) -> None:
        from ovrtx_blender_example import engine

        engines = list(engine._ACTIVE_VIEWPORT_ENGINES)
        if not engines:
            return
        instance = engines[0]
        frame = instance._presented_frame
        if frame is None or frame is self.last_presented_frame:
            return
        self.last_presented_frame = frame
        if frame.render_result is self.last_presented_result:
            return
        self.last_presented_result = frame.render_result
        completed_ns = time.perf_counter_ns()
        self.frame_events.append(_frame_latency_event(frame, completed_ns))
        self._note_stopped_view()

    def finished(self) -> bool:
        self._note_stopped_view()
        return (
            self.started
            and self.phase == "settle"
            and self.stopped_view_complete
            and bool(self.window)
            and time.perf_counter_ns() >= self.window["end"]
        )

    def complete(self) -> dict[str, Any]:
        from ovrtx_blender_example import engine

        instance = list(engine._ACTIVE_VIEWPORT_ENGINES)[0]
        if getattr(instance, "_ovrtx_session_controller", None) is None:
            raise RuntimeError(
                "Blender throughput measurement has no active OVRTX session"
            )
        repetition = shared_navigation.finish_frame_latency_repetition(
            repetition_index=self.index,
            warmup_start_ns=self.window["warmup_start"],
            measurement_start_ns=self.window["start"],
            measurement_end_ns=self.window["end"],
            frame_events=self.frame_events,
            stopped_view_complete=self.stopped_view_complete,
            materialization=self.materialization,
        )
        blockers = shared_navigation.validate_frame_latency_repetition(
            self.index, repetition
        )
        if blockers:
            raise RuntimeError(
                "throughput repetition validation failed: " + ", ".join(blockers)
            )
        return {
            "repetition": repetition,
        }


def _inside_blender_throughput(args: argparse.Namespace) -> int:
    import bpy

    bpy.context.preferences.view.show_splash = False
    config = json.loads(args.config.read_text())
    for value in (config["addon_path"], config["native_client_path"]):
        if value and value not in sys.path:
            sys.path.insert(0, value)
    bpy.ops.wm.open_mainfile(filepath=config["blend_path"])
    os.environ["OV_BLENDER_EXAMPLE_WORKER_COMMAND"] = str(config["worker_command"])
    os.environ["OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE"] = str(
        config["native_client_module"]
    )
    os.environ[color_presentation.ENV_COLOR_PRESENTATION_MODE] = config[
        "color_presentation"
    ]

    import ovrtx_blender_example
    from ovrtx_blender_example import engine, scene_generation_sessions
    from ovrtx_blender_example.blender_callback_adapters import (
        BlenderRenderCallbackAdapter,
    )
    from ovrtx_blender_example.interactive_setup import (
        configure_scene,
        configure_viewports,
    )

    ovrtx_blender_example.register()
    print("[navigation] Blender ready", flush=True)
    state = SimpleNamespace(
        measurement=None, workload=None, completed_samples=-1, full_area=False
    )

    def quit_blender() -> None:
        ovrtx_blender_example.unregister()
        bpy.ops.wm.quit_blender()

    def fail(exc: Exception) -> None:
        Path(config["error_result"]).write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        quit_blender()

    def post_pixel() -> None:
        try:
            state.measurement.post_pixel()
        except Exception as exc:
            fail(exc)

    def pump() -> float | None:
        try:
            view3d_context = _view3d_context(bpy)
            if view3d_context is None:
                return 0.0
            _window, area, _region, space = view3d_context
            if state.measurement.advance_render_throughput(space.region_3d):
                area.tag_redraw()
            if not state.measurement.finished():
                return 0.0
            Path(config["raw_result"]).write_text(
                json.dumps(
                    {
                        **state.measurement.complete(),
                        "workload": state.workload,
                    },
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print("[navigation] measurement complete", flush=True)
            quit_blender()
            return None
        except Exception as exc:
            fail(exc)
            return None

    def tick() -> float | None:
        try:
            view3d_context = _view3d_context(bpy)
            if view3d_context is None:
                return 0.1
            window, area, region, space = view3d_context
            if not state.full_area:
                # Settle Blender's layout once before rendered
                # shading creates an engine; client size varies by windowing OS.
                space.show_region_toolbar = False
                space.show_region_ui = False
                with bpy.context.temp_override(window=window, area=area):
                    bpy.ops.screen.screen_full_area(use_hide_panels=True)
                state.full_area = True
                return 0.25
            if state.measurement is None:
                with bpy.context.temp_override(
                    window=window,
                    area=area,
                    region=region,
                    region_data=space.region_3d,
                ):
                    scene = bpy.context.scene
                    sync_viewport_camera = bool(config["sync_viewport_camera"])
                    config["sync_viewport_camera"] = False
                    configure_scene(scene, config)
                    generation = scene_generation_sessions.generation_for_scene(scene)
                    request = BlenderRenderCallbackAdapter(
                        generation_for_scene=lambda _scene: generation
                    ).final_render_from_scene(scene)
                    config.update(
                        camera_prim_path=request.camera_prim_path,
                        render_product_path=request.render_product_path,
                        sync_viewport_camera=sync_viewport_camera,
                    )
                    # Reconcile camera sync after generated paths exist;
                    # starting the viewport from the discovery pass stalls reads.
                    configure_scene(scene, config)
                    generation = scene_generation_sessions.generation_for_scene(scene)
                    request = BlenderRenderCallbackAdapter(
                        generation_for_scene=lambda _scene: generation
                    ).final_render_from_scene(scene)
                    config.update(
                        usd_path=generation.materialize_usd(),
                        camera_prim_path=request.camera_prim_path,
                        render_product_path=request.render_product_path,
                    )
                    configure_viewports(bpy, scene, config, screen=window.screen)
                    materialization = _production_materialization(
                        generation, request
                    )
                    state.measurement = _ThroughputMeasurement(
                        config, materialization
                    )
                    state.workload = shared_navigation.navigation_workload(
                        {
                            "id": config["fixture_id"],
                            "sha256": config["blend_sha256"],
                        },
                        config["color_presentation"],
                        repetition_count=config["repetitions"],
                    )
                    print("[navigation] scene ready", flush=True)
            if state.measurement.handler is None:
                state.measurement.handler = space.draw_handler_add(
                    post_pixel, (), "WINDOW", "POST_PIXEL"
                )
            engines = list(engine._ACTIVE_VIEWPORT_ENGINES)
            if not engines:
                area.tag_redraw()
                return 0.01
            if not state.measurement.started:
                area.tag_redraw()
                result = engines[0]._current_result
                completed_samples = (
                    int(result.completed_samples) if result is not None else 0
                )
                if completed_samples != state.completed_samples:
                    state.completed_samples = completed_samples
                    print(
                        f"[navigation] convergence samples={completed_samples}",
                        flush=True,
                    )
                if result is not None and result.completed_samples >= 128:
                    print("[navigation] measurement started", flush=True)
                    state.measurement.start(space.region_3d)
                    bpy.app.timers.register(pump, first_interval=0.0)
                    return None
            return 0.01
        except Exception as exc:
            fail(exc)
            return None

    bpy.app.timers.register(tick, first_interval=0.5)
    return 0


def _view3d_context(bpy: Any) -> tuple[Any, Any, Any, Any] | None:
    window = bpy.context.window
    if window is None:
        return None
    area = next(
        (area for area in window.screen.areas if area.type == "VIEW_3D"), None
    )
    if area is None:
        return None
    region = next(
        (region for region in area.regions if region.type == "WINDOW"), None
    )
    if region is None:
        return None
    return window, area, region, area.spaces.active


def _production_materialization(generation: Any, request: Any) -> dict[str, Any]:
    generation_root = Path(generation.materialize_usd()).parent
    runtime_files = [
        {
            "path": path.relative_to(generation_root).as_posix(),
            "sha256": shared_navigation.file_sha256(path),
        }
        for path in sorted(generation_root.rglob("*"))
        if path.is_file()
    ]
    return {
        "generation": {
            "digest": generation.digest,
            "usd_sha256": shared_navigation.file_sha256(
                Path(generation.materialize_usd())
            ),
            "runtime_files": runtime_files,
        },
        "render_request": {
            "camera_prim_path": request.camera_prim_path,
            "render_product_path": request.render_product_path,
        },
    }


def _run_blender_throughput_subrun(
    args: argparse.Namespace,
    index: int,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    work_dir = (
        args.output.resolve().parent
        / ".blender-navigation-throughput"
        / f"repetition-{index}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_result = work_dir / "raw-result.json"
    error_result = work_dir / "error.json"
    for path in (raw_result, error_result):
        path.unlink(missing_ok=True)
    config = {
        "addon_path": str(ADDON_DIR),
        "source_root": str(args.source_root),
        "fixture_id": fixture["id"],
        "blend_path": str(Path(fixture["path"]).resolve()),
        "blend_sha256": fixture["sha256"],
        "camera_prim_path": "???",
        "width": 1280,
        "height": 720,
        "min_samples": 1,
        "max_samples": 128,
        "color_presentation": args.color_presentation,
        "native_client_module": args.native_client_module,
        "native_client_path": str(args.native_client_path.resolve()),
        "worker_command": args.worker_command,
        "sync_viewport_camera": True,
        "selectable_imported_objects": False,
        "tag_dynamic_body_transforms": False,
        "shared_stage_composition": False,
        "body_root": "/World/PhysicsIsland/DynamicBodies",
        "composition_update_fps": 30.0,
        "viewport_redraw_pressure_mode": "scheduled",
        "forced_redraw_timer_type": "DRAW_WIN_SWAP",
        "interactive_duration_s": 0.0,
        "timeline_end_frame": 120,
        "viewport_orbit_distance": None,
        "viewport_pitch_offset_degrees": None,
        "viewport_clip_start": None,
        "viewport_clip_end": None,
        "repetition_index": index,
        "repetitions": args.repetitions,
        "raw_result": str(raw_result),
        "error_result": str(error_result),
    }
    config_path = work_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.pop("OV_BLENDER_EXAMPLE_VIEWPORT_ARTIFACT", None)
    env.pop("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", None)
    env.update(
        {
            "OVRTX_ACTIVE_CUDA_GPUS": "0",
            render_requests.ENV_FIXED_VIEWPORT_RESOLUTION: "1",
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR": str(work_dir / "viewport-work"),
            "OV_BLENDER_EXAMPLE_WORKER_LOG": str(work_dir / "worker.log"),
        }
    )
    command = [
        args.blender_command,
        "--factory-startup",
        "--window-geometry",
        "20",
        "20",
        "1280",
        "720",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--inside-blender",
        "--source-root",
        str(args.source_root),
        "--config",
        str(config_path),
    ]
    returncode = validation.SubprocessExecutor(verbose=True).run(
        command,
        cwd=ROOT,
        output_dir=work_dir,
        label="blender",
        env=env,
    ).exit_status
    if returncode or error_result.exists() or not raw_result.exists():
        for name in ("blender.log", "worker.log"):
            source = work_dir / name
            if source.is_file():
                retained = args.output.parent / f"{args.output.stem}-{name}"
                retained.write_bytes(source.read_bytes())
                print(f"[navigation] retained {retained}", flush=True)
                if name == "worker.log":
                    print(source.read_text(errors="replace")[-20_000:], flush=True)
        detail = (
            error_result.read_text()
            if error_result.exists()
            else f"Blender exited {returncode}"
        )
        raise RuntimeError(detail.strip())
    try:
        result = json.loads(raw_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Blender throughput sub-run produced invalid JSON") from exc
    if not isinstance(result, dict) or set(result) != {"repetition", "workload"}:
        raise RuntimeError("Blender throughput sub-run produced an invalid result")
    return result


def _outer_throughput(args: argparse.Namespace) -> int:
    fixture = fixture_input(
        load_manifest(args.manifest.resolve()), "perf_junk_shop_1280x720"
    )
    workloads: list[Mapping[str, Any]] = []

    def run(index: int) -> dict[str, Any]:
        result = _run_blender_throughput_subrun(args, index, fixture)
        workloads.append(result["workload"])
        return result["repetition"]

    runs = shared_navigation.run_frame_latency_measurements(
        args.repetitions, run
    )
    if not workloads:
        error = runs[0].get("error") if runs else None
        raise RuntimeError(
            str(error or "Blender throughput run produced no workload identity")
        )
    if any(workload != workloads[0] for workload in workloads[1:]):
        raise RuntimeError("Blender throughput sub-runs used different workloads")
    record = {
        "schema_version": shared_navigation.SCHEMA_VERSION,
        "artifact_id": "ovrtx-navigation-render-throughput",
        "case_kind": "blender",
        "workload": workloads[0],
        "runs": runs,
    }
    blockers = shared_navigation.validate_frame_latency_record(record)
    if blockers:
        raise RuntimeError("record validation failed: " + ", ".join(blockers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = [run for run in runs if not run["measurement_complete"]]
    print(
        json.dumps(
            {
                "result": str(args.output),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = _argv() if argv is None else list(argv)
    args = _parse_args(raw_argv)
    if args.inside_blender:
        return _inside_blender_throughput(args)
    try:
        return _outer_throughput(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    if "--inside-blender" in _argv():
        main()
    else:
        raise SystemExit(main())
