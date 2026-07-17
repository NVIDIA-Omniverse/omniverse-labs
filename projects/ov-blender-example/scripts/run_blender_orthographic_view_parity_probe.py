#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe Blender orthographic request parity without starting OVRTX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ovrtx_probe_support import BLENDER_COMMAND  # noqa: E402


ARTIFACT_ID = "blender-orthographic-view-parity-probe"
DEFAULT_OUTPUT_DIR = REPO / "out" / "artifacts" / ARTIFACT_ID


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": args.output_dir / "orthographic-view-parity.json",
        "setup": args.output_dir / "blender_orthographic_view_parity_setup.py",
    }
    paths["setup"].write_text(_setup_script(), encoding="utf-8")

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": "running",
        "started_at_ns": time.time_ns(),
        "runs": [],
        "paths": {name: str(path) for name, path in paths.items()},
    }
    run_results: list[dict[str, Any]] = []
    for config in _run_configs(args.output_dir):
        completed = _run_blender(args, paths["setup"], config)
        log_path = Path(config["log_path"])
        metrics_path = Path(config["metrics_path"])
        log_path.write_text(completed.stdout, encoding="utf-8")
        metrics = _read_json(metrics_path)
        run_result = {
            "id": config["run_id"],
            "window_geometry": list(config["window_geometry"]),
            "blender_exit_status": completed.returncode,
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "metrics": metrics,
        }
        if completed.returncode != 0 and not metrics:
            run_result["metrics"] = {
                "status": "failed",
                "error": "Blender exited before writing metrics.",
            }
        run_results.append(run_result)

    result["runs"] = run_results
    result["completed_at_ns"] = time.time_ns()
    result["status"] = "pass" if _all_runs_passed(run_results) else "failed"
    result["classification"] = (
        "orthographic-request-parity-proven"
        if result["status"] == "pass"
        else "orthographic-request-parity-failed"
    )
    _write_json(paths["result"], result)
    print(json.dumps(_summary(result, paths), indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blender-command", default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND))
    return parser.parse_args(list(argv))


def _run_configs(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "run_id": "landscape",
            "window_geometry": [0, 0, 1600, 900],
            "metrics_path": str(output_dir / "landscape-metrics.json"),
            "log_path": str(output_dir / "landscape-blender.log"),
        },
        {
            "run_id": "portrait",
            "window_geometry": [0, 0, 900, 1600],
            "metrics_path": str(output_dir / "portrait-metrics.json"),
            "log_path": str(output_dir / "portrait-blender.log"),
        },
    ]


def _run_blender(
    args: argparse.Namespace,
    setup_path: Path,
    config: Mapping[str, Any],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ORTHO_VIEW_PARITY_CONFIG"] = json.dumps(
        {
            **dict(config),
            "repo": str(REPO),
        }
    )
    geometry = [str(value) for value in config["window_geometry"]]
    return subprocess.run(
        [args.blender_command, "--window-geometry", *geometry, "--python", str(setup_path)],
        cwd=str(REPO),
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _all_runs_passed(runs: Sequence[Mapping[str, Any]]) -> bool:
    return bool(runs) and all(
        int(run.get("blender_exit_status", 1)) == 0
        and isinstance(run.get("metrics"), Mapping)
        and run["metrics"].get("status") == "pass"
        for run in runs
    )


def _summary(result: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "classification": result.get("classification"),
        "result": str(paths["result"]),
        "runs": [
            {
                "id": run.get("id"),
                "status": (run.get("metrics") or {}).get("status"),
                "failures": (run.get("metrics") or {}).get("failures", []),
                "metrics_path": run.get("metrics_path"),
                "log_path": run.get("log_path"),
            }
            for run in result.get("runs", [])
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _setup_script() -> str:
    return textwrap.dedent(
        r'''
        import json
        import math
        import os
        from pathlib import Path
        import sys
        import traceback
        from types import SimpleNamespace

        import bpy
        from mathutils import Quaternion, Vector

        CONFIG = json.loads(os.environ["ORTHO_VIEW_PARITY_CONFIG"])
        REPO = Path(CONFIG["repo"])
        sys.path.insert(0, str(REPO / "addon"))

        from ovrtx_blender_example import properties
        from ovrtx_blender_example import render_requests
        from ovrtx_blender_example.engine import build_request_from_scene
        from ovrtx_blender_example.blender_signals import (
            BlenderRenderIntent,
            BlenderRenderSignalSource,
        )


        SCENE_CAMERA_CASES = (
            {
                "id": "scene-landscape-auto-sync-true",
                "width": 1280,
                "height": 720,
                "sensor_fit": "AUTO",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-landscape-horizontal-sync-true",
                "width": 1280,
                "height": 720,
                "sensor_fit": "HORIZONTAL",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-landscape-vertical-sync-true",
                "width": 1280,
                "height": 720,
                "sensor_fit": "VERTICAL",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-portrait-auto-sync-true",
                "width": 720,
                "height": 1280,
                "sensor_fit": "AUTO",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-portrait-horizontal-sync-true",
                "width": 720,
                "height": 1280,
                "sensor_fit": "HORIZONTAL",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-portrait-vertical-sync-true",
                "width": 720,
                "height": 1280,
                "sensor_fit": "VERTICAL",
                "sync_viewport_camera": True,
            },
            {
                "id": "scene-landscape-auto-sync-false",
                "width": 1280,
                "height": 720,
                "sensor_fit": "AUTO",
                "sync_viewport_camera": False,
            },
            {
                "id": "scene-landscape-shifted-auto-sync-false",
                "width": 1280,
                "height": 720,
                "sensor_fit": "AUTO",
                "sync_viewport_camera": False,
                "shift_x": 0.1,
                "shift_y": -0.2,
            },
        )

        USER_VIEW_CASES = (
            {"id": "ortho-distance-2-origin", "distance": 2.0, "location": (0.0, 0.0, 0.0)},
            {"id": "ortho-distance-5-origin", "distance": 5.0, "location": (0.0, 0.0, 0.0)},
            {"id": "ortho-distance-10-origin", "distance": 10.0, "location": (0.0, 0.0, 0.0)},
            {"id": "ortho-distance-5-pan-x", "distance": 5.0, "location": (1.0, 0.0, 0.0)},
            {"id": "ortho-distance-5-pan-y", "distance": 5.0, "location": (0.0, 2.0, 0.0)},
            {"id": "ortho-distance-5-pan-xy", "distance": 5.0, "location": (1.0, 2.0, 0.0)},
        )


        def _run_probe():
            result = {
                "schema_version": 1,
                "run_id": CONFIG["run_id"],
                "window_geometry": list(CONFIG["window_geometry"]),
                "status": "running",
                "scene_camera_cases": [],
                "active_camera_cases": [],
                "user_view_cases": [],
                "relationship_checks": {},
                "failures": [],
            }
            try:
                properties.register()
                _reset_scene()
                scene = bpy.context.scene
                _configure_scene_settings(scene, width=1280, height=720, sync_viewport_camera=True)
                view_context = _active_view3d_context()
                if view_context is None:
                    raise RuntimeError("no active VIEW_3D context")

                for case in SCENE_CAMERA_CASES:
                    result["scene_camera_cases"].append(_scene_camera_case(scene, case, result["failures"]))

                result["active_camera_cases"].append(
                    _active_camera_case(scene, view_context, result["failures"])
                )

                for case in USER_VIEW_CASES:
                    result["user_view_cases"].append(
                        _orthographic_user_view_case(scene, view_context, case, result["failures"])
                    )
                result["relationship_checks"] = _relationship_checks(
                    result["user_view_cases"],
                    result["failures"],
                )
                result["status"] = "pass" if not result["failures"] else "failed"
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["traceback"] = traceback.format_exc()
                result["failures"].append("probe_exception")
            Path(CONFIG["metrics_path"]).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            bpy.ops.wm.quit_blender()
            return None


        def _reset_scene():
            bpy.ops.object.select_all(action="SELECT")
            bpy.ops.object.delete()


        def _configure_scene_settings(scene, *, width, height, sync_viewport_camera):
            scene.render.resolution_x = int(width)
            scene.render.resolution_y = int(height)
            scene.render.resolution_percentage = 100
            settings = scene.ovrtx_example
            settings.camera_prim_path = "/World/Camera"
            settings.render_product_path = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
            settings.min_samples = 1
            settings.max_samples = 1
            settings.sync_viewport_camera = bool(sync_viewport_camera)


        def _create_orthographic_camera(
            scene,
            *,
            name,
            sensor_fit,
            ortho_scale=6.0,
            shift_x=0.0,
            shift_y=0.0,
        ):
            data = bpy.data.cameras.new(name + "Data")
            data.type = "ORTHO"
            data.ortho_scale = float(ortho_scale)
            data.sensor_fit = str(sensor_fit)
            data.shift_x = float(shift_x)
            data.shift_y = float(shift_y)
            data.lens = 45.0
            data.clip_start = 0.25
            data.clip_end = 400.0
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            scene.camera = obj
            return obj


        def _scene_camera_case(scene, case, failures):
            _configure_scene_settings(
                scene,
                width=case["width"],
                height=case["height"],
                sync_viewport_camera=case["sync_viewport_camera"],
            )
            camera = _create_orthographic_camera(
                scene,
                name="Camera_" + case["id"].replace("-", "_"),
                sensor_fit=case["sensor_fit"],
                shift_x=case.get("shift_x", 0.0),
                shift_y=case.get("shift_y", 0.0),
            )
            bpy.context.view_layer.update()
            request = build_request_from_scene(
                scene,
                context=None,
                source=BlenderRenderSignalSource.FINAL_RENDER,
                intent=BlenderRenderIntent.FINAL_RENDER,
            )
            projection = request.camera_projection
            attrs = projection.usd_attributes() if projection is not None else {}
            frame = _camera_view_frame(camera.data, scene)
            expected_horizontal_aperture = round(frame["width"] * 10.0, 9)
            expected_vertical_aperture = round(frame["height"] * 10.0, 9)
            expected_horizontal_offset = round(frame["center"][0] * 10.0, 9)
            expected_vertical_offset = round(frame["center"][1] * 10.0, 9)
            record = {
                "id": case["id"],
                "sensor_fit": case["sensor_fit"],
                "shift": [float(case.get("shift_x", 0.0)), float(case.get("shift_y", 0.0))],
                "sync_viewport_camera": bool(case["sync_viewport_camera"]),
                "render_size": [request.width, request.height],
                "camera_matrix_available": request.camera_matrix is not None,
                "view_frame": frame,
                "expected": {
                    "projection": "orthographic",
                    "horizontalAperture": expected_horizontal_aperture,
                    "verticalAperture": expected_vertical_aperture,
                    "horizontalApertureOffset": expected_horizontal_offset,
                    "verticalApertureOffset": expected_vertical_offset,
                },
                "actual": attrs,
                "source": getattr(projection, "source", ""),
            }
            _require(case["id"], "scene projection exists", projection is not None, failures)
            _require(case["id"], "scene source is active camera", getattr(projection, "source", "") == render_requests.ACTIVE_CAMERA_VIEW, failures)
            _require(case["id"], "scene projection token", attrs.get("projection") == "orthographic", failures)
            _require_close(case["id"], "horizontalAperture", attrs.get("horizontalAperture"), expected_horizontal_aperture, failures)
            _require_close(case["id"], "verticalAperture", attrs.get("verticalAperture"), expected_vertical_aperture, failures)
            _require_close(case["id"], "horizontalApertureOffset", attrs.get("horizontalApertureOffset"), expected_horizontal_offset, failures)
            _require_close(case["id"], "verticalApertureOffset", attrs.get("verticalApertureOffset"), expected_vertical_offset, failures)
            _require(case["id"], "final render keeps usd transform", request.camera_matrix is None, failures)
            _require(case["id"], "final render size preserved", (request.width, request.height) == (case["width"], case["height"]), failures)
            return record


        def _active_camera_case(scene, view_context, failures):
            case_id = "active-camera-orthographic-view"
            _configure_scene_settings(scene, width=1280, height=720, sync_viewport_camera=True)
            camera = _create_orthographic_camera(scene, name="Camera_active_camera_view", sensor_fit="AUTO")
            rd = view_context["region_data"]
            rd.view_perspective = "CAMERA"
            rd.update()
            bpy.context.view_layer.update()
            context = _request_context(scene, view_context)
            request = build_request_from_scene(
                scene,
                context=context,
                source=BlenderRenderSignalSource.VIEW_UPDATE,
                intent=BlenderRenderIntent.VIEWPORT,
            )
            projection = request.camera_projection
            attrs = projection.usd_attributes() if projection is not None else {}
            frame = _camera_view_frame(camera.data, scene)
            expected_horizontal_aperture = round(frame["width"] * 10.0, 9)
            expected_vertical_aperture = round(frame["height"] * 10.0, 9)
            record = {
                "id": case_id,
                "render_size": [request.width, request.height],
                "camera_matrix_available": request.camera_matrix is not None,
                "view_frame": frame,
                "expected": {
                    "projection": "orthographic",
                    "horizontalAperture": expected_horizontal_aperture,
                    "verticalAperture": expected_vertical_aperture,
                },
                "actual": attrs,
                "source": getattr(projection, "source", ""),
            }
            _require(case_id, "active camera matrix exists", request.camera_matrix is not None, failures)
            _require(case_id, "active camera source", getattr(projection, "source", "") == render_requests.ACTIVE_CAMERA_VIEW, failures)
            _require(case_id, "active projection token", attrs.get("projection") == "orthographic", failures)
            _require_close(case_id, "horizontalAperture", attrs.get("horizontalAperture"), expected_horizontal_aperture, failures)
            _require_close(case_id, "verticalAperture", attrs.get("verticalAperture"), expected_vertical_aperture, failures)
            return record


        def _orthographic_user_view_case(scene, view_context, case, failures):
            _configure_scene_settings(scene, width=1280, height=720, sync_viewport_camera=True)
            if scene.camera is None:
                _create_orthographic_camera(scene, name="Camera_user_view_anchor", sensor_fit="AUTO")
            rd = view_context["region_data"]
            rd.view_perspective = "ORTHO"
            rd.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
            rd.view_location = Vector(case["location"])
            rd.view_distance = float(case["distance"])
            rd.update()
            bpy.context.view_layer.update()
            context = _request_context(scene, view_context)
            request = build_request_from_scene(
                scene,
                context=context,
                source=BlenderRenderSignalSource.VIEW_UPDATE,
                intent=BlenderRenderIntent.VIEWPORT,
            )
            projection = request.camera_projection
            attrs = projection.usd_attributes() if projection is not None else {}
            window_matrix = rd.window_matrix.copy()
            expected = _orthographic_expected_from_window_matrix(window_matrix)
            record = {
                "id": case["id"],
                "distance": float(case["distance"]),
                "location": list(case["location"]),
                "viewport_region": [int(context.region.width), int(context.region.height)],
                "render_size": [request.width, request.height],
                "raw_window_matrix": _matrix_rows(window_matrix),
                "raw_view_matrix": _matrix_rows(rd.view_matrix),
                "camera_matrix": request.camera_matrix,
                "camera_matrix_available": request.camera_matrix is not None,
                "source": getattr(projection, "source", ""),
                "expected": expected,
                "actual": attrs,
                "clipping_status": "omitted-unproven",
            }
            _require(case["id"], "user view projection exists", projection is not None, failures)
            _require(case["id"], "user view source", getattr(projection, "source", "") == render_requests.ORTHOGRAPHIC_USER_VIEW, failures)
            _require(case["id"], "user view matrix exists", request.camera_matrix is not None, failures)
            _require(case["id"], "user view projection token", attrs.get("projection") == "orthographic", failures)
            _require_close(case["id"], "horizontalAperture", attrs.get("horizontalAperture"), expected["horizontalAperture"], failures)
            _require_close(case["id"], "verticalAperture", attrs.get("verticalAperture"), expected["verticalAperture"], failures)
            _require_close(case["id"], "horizontalApertureOffset", attrs.get("horizontalApertureOffset"), expected["horizontalApertureOffset"], failures)
            _require_close(case["id"], "verticalApertureOffset", attrs.get("verticalApertureOffset"), expected["verticalApertureOffset"], failures)
            _require(case["id"], "orthographic user view omits clippingRange", "clippingRange" not in attrs, failures)
            return record


        def _orthographic_expected_from_window_matrix(window_matrix):
            window_x_scale = float(window_matrix[0][0])
            window_y_scale = float(window_matrix[1][1])
            horizontal_offset_term = float(window_matrix[0][2])
            vertical_offset_term = float(window_matrix[1][2])
            return {
                "projection": "orthographic",
                "horizontalAperture": round(20.0 / abs(window_x_scale), 9),
                "verticalAperture": round(20.0 / abs(window_y_scale), 9),
                "horizontalApertureOffset": round(-10.0 * horizontal_offset_term / window_x_scale, 9),
                "verticalApertureOffset": round(-10.0 * vertical_offset_term / window_y_scale, 9),
                "window_z_scale": round(float(window_matrix[2][2]), 9),
                "window_z_offset": round(float(window_matrix[2][3]), 9),
            }


        def _relationship_checks(user_view_cases, failures):
            by_id = {case["id"]: case for case in user_view_cases}
            checks = {}
            zoom_ids = ("ortho-distance-2-origin", "ortho-distance-5-origin", "ortho-distance-10-origin")
            zoom_values = [by_id[item]["actual"].get("horizontalAperture", 0.0) for item in zoom_ids]
            checks["zoom_changes_aperture"] = {
                "status": "pass" if zoom_values[0] < zoom_values[1] < zoom_values[2] else "failed",
                "horizontal_apertures": zoom_values,
            }
            if checks["zoom_changes_aperture"]["status"] != "pass":
                failures.append("user-view-zoom:aperture-not-monotonic")

            origin = by_id["ortho-distance-5-origin"]
            panned = by_id["ortho-distance-5-pan-xy"]
            same_window = _matrix_close(origin["raw_window_matrix"], panned["raw_window_matrix"])
            same_apertures = (
                _close(origin["actual"].get("horizontalAperture"), panned["actual"].get("horizontalAperture"))
                and _close(origin["actual"].get("verticalAperture"), panned["actual"].get("verticalAperture"))
                and _close(origin["actual"].get("horizontalApertureOffset"), panned["actual"].get("horizontalApertureOffset"))
                and _close(origin["actual"].get("verticalApertureOffset"), panned["actual"].get("verticalApertureOffset"))
            )
            matrix_changed = not _matrix_close(origin["camera_matrix"], panned["camera_matrix"])
            checks["pan_owned_by_view_matrix"] = {
                "status": "pass" if same_window and same_apertures and matrix_changed else "failed",
                "window_matrix_unchanged": same_window,
                "aperture_attrs_unchanged": same_apertures,
                "camera_matrix_changed": matrix_changed,
            }
            if checks["pan_owned_by_view_matrix"]["status"] != "pass":
                failures.append("user-view-pan:not-owned-by-camera-matrix")
            return checks


        def _active_view3d_context():
            best = None
            for window in bpy.context.window_manager.windows:
                screen = getattr(window, "screen", None)
                for area in getattr(screen, "areas", ()):
                    if area.type != "VIEW_3D":
                        continue
                    region = next((region for region in area.regions if region.type == "WINDOW"), None)
                    space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
                    region_data = getattr(space, "region_3d", None) if space else None
                    if region is None or space is None or region_data is None:
                        continue
                    area_pixels = int(region.width) * int(region.height)
                    if best is None or area_pixels > best[0]:
                        best = (area_pixels, window, screen, area, region, space, region_data)
            if best is None:
                return None
            return {
                "window": best[1],
                "screen": best[2],
                "area": best[3],
                "region": best[4],
                "space_data": best[5],
                "region_data": best[6],
            }


        def _request_context(scene, view_context):
            return SimpleNamespace(
                window=view_context["window"],
                screen=view_context["screen"],
                area=view_context["area"],
                region=view_context["region"],
                space_data=view_context["space_data"],
                region_data=view_context["region_data"],
                scene=scene,
            )


        def _camera_view_frame(data, scene):
            frame = tuple(data.view_frame(scene=scene))
            xs = [float(item.x) for item in frame]
            ys = [float(item.y) for item in frame]
            return {
                "width": round(max(xs) - min(xs), 9),
                "height": round(max(ys) - min(ys), 9),
                "center": [round((max(xs) + min(xs)) * 0.5, 9), round((max(ys) + min(ys)) * 0.5, 9)],
                "points": [[round(float(item.x), 9), round(float(item.y), 9), round(float(item.z), 9)] for item in frame],
            }


        def _matrix_rows(matrix):
            return [
                [round(float(matrix[row][column]), 9) for column in range(4)]
                for row in range(4)
            ]


        def _require(case_id, label, condition, failures):
            if not condition:
                failures.append(f"{case_id}:{label}")


        def _require_close(case_id, label, actual, expected, failures):
            if not _close(actual, expected):
                failures.append(f"{case_id}:{label}:expected={expected}:actual={actual}")


        def _close(actual, expected, tolerance=1.0e-6):
            try:
                return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance
            except Exception:
                return False


        def _matrix_close(a, b, tolerance=1.0e-6):
            if a is None or b is None:
                return False
            try:
                if len(a) != len(b):
                    return False
                for row_a, row_b in zip(a, b):
                    if len(row_a) != len(row_b):
                        return False
                    for value_a, value_b in zip(row_a, row_b):
                        if abs(float(value_a) - float(value_b)) > tolerance:
                            return False
                return True
            except Exception:
                return False


        bpy.app.timers.register(_run_probe, first_interval=0.1)
        '''
    ).lstrip()


if __name__ == "__main__":
    raise SystemExit(main())
