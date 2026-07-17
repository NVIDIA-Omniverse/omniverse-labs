#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded GUI probe for the interactive operator seam.

Launch Blender with the OVRTX viewport renderer, move one tagged imported
interaction object, then record whether the production depsgraph bridge reaches
update_transforms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import zlib


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
ADDON = REPO / "addon"
if str(ADDON) not in sys.path:
    sys.path.insert(0, str(ADDON))
DEFAULT_TARGET_PRIM = "/World/PhysicsIsland/DynamicBodies/Cube_00"

from ovrtx_probe_support import (  # noqa: E402
    BLENDER_COMMAND,
    DEFAULT_FIXTURE_MANIFEST,
    default_native_client_path,
    default_worker_command,
)
from fixture_manifest import (  # noqa: E402
    load_manifest,
    render_fixture,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = (
        _generated_red_cube_fixture(args.output_dir / "generated-red-cube.usda", args.width, args.height)
        if args.generated_red_cube_fixture
        else render_fixture(load_manifest(args.manifest), args.fixture_id)
    )
    if args.generated_red_cube_fixture and args.target_prim == DEFAULT_TARGET_PRIM:
        args.target_prim = "/World/Cube_00"
    setup_path = args.output_dir / "operator_seam_setup.py"
    metrics_path = args.output_dir / "operator-seam-metrics.json"
    log_path = args.output_dir / "blender.log"
    viewport_artifact_path = args.output_dir / "viewport-preview.json"
    image_path = args.output_dir / "image.png"
    worker_log_path = args.output_dir / "worker.log"

    setup_path.write_text(
        _setup_script(
            {
                "repo": str(REPO),
                "input_usd_path": fixture["fixture_usd_path"],
                "render_product_path": fixture["render_product_path"],
                "camera_prim_path": fixture["camera_prim_path"],
                "native_client_module": args.native_client_module,
                "native_client_path": args.native_client_path,
                "worker_command": args.worker_command,
                "target_object": args.target_object,
                "target_prim": args.target_prim,
                "width": args.width,
                "height": args.height,
                "min_samples": args.min_samples,
                "max_samples": args.max_samples,
                "move_x": args.move_x,
                "move_y": args.move_y,
                "move_z": args.move_z,
                "selection_mode": args.selection_mode,
                "selection_only": args.selection_only,
                "selection_repetitions": args.selection_repetitions,
                "selection_settle_seconds": args.selection_settle_seconds,
                "current_scene_generation": args.current_scene_generation,
                "world_assignment_only": args.world_assignment_only,
                "default_blender_scene": args.default_blender_scene,
                "include_unmapped_selection": args.include_unmapped_selection,
                "expect_selection_group_rejected": args.expect_selection_group_rejected,
                "view_perspective": args.view_perspective,
                "metrics_path": str(metrics_path),
                "viewport_artifact_path": str(viewport_artifact_path),
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["OV_BLENDER_EXAMPLE_VIEWPORT_ARTIFACT"] = str(viewport_artifact_path)
    env["OV_BLENDER_EXAMPLE_IMAGE_ARTIFACT"] = str(image_path)
    env["OV_BLENDER_EXAMPLE_WORKER_LOG"] = str(worker_log_path)
    if args.active_cuda_gpus:
        env["OVRTX_ACTIVE_CUDA_GPUS"] = args.active_cuda_gpus

    completed = subprocess.run(
        [BLENDER_COMMAND, "--factory-startup", "--python", str(setup_path)],
        cwd=str(REPO),
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path.write_text(completed.stdout, encoding="utf-8")
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    if metrics_payload:
        if not (args.selection_only or args.world_assignment_only):
            _attach_alignment_analysis(metrics_payload, image_path)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probe_passed = completed.returncode == 0 and metrics_payload.get("status") == "pass"
    result = {
        "status": "pass" if probe_passed else "failed",
        "blender_exit_status": completed.returncode,
        "metrics": str(metrics_path),
        "blender_log": str(log_path),
        "viewport_artifact": str(viewport_artifact_path),
        "worker_log": str(worker_log_path),
    }
    if metrics_payload:
        result["metrics_payload"] = metrics_payload
    if not probe_passed:
        result["error"] = metrics_payload.get("error", "operator seam probe failed inside Blender")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if probe_passed else 1


def _attach_alignment_analysis(metrics: dict[str, Any], image_path: Path) -> None:
    backend_status = str(metrics.get("status", ""))
    metrics["backend_status"] = backend_status
    if bool(metrics.get("expect_selection_group_rejected", False)):
        metrics["alignment"] = {"status": "skipped", "reason": "selection_group_rejection_expected"}
        if backend_status == "pass":
            metrics["status"] = "pass"
            return
        metrics["status"] = "failed"
        metrics.setdefault("error", "expected selection group rejection diagnostics were not recorded")
        return
    alignment = _alignment_analysis(metrics, image_path)
    metrics["alignment"] = alignment
    if backend_status == "pass" and alignment.get("status") == "pass":
        metrics["status"] = "pass"
        return
    metrics["status"] = "failed"
    if backend_status != "pass":
        metrics.setdefault("error", "production depsgraph bridge did not submit a target edit")
    else:
        metrics["error"] = str(alignment.get("error", "viewport interaction-object alignment diagnostics failed"))


def _alignment_analysis(metrics: Mapping[str, Any], image_path: Path) -> dict[str, Any]:
    if not image_path.is_file():
        return {"status": "failed", "error": "image artifact was not written"}
    try:
        width, height, rgba = _read_rgba_png(image_path)
    except Exception as exc:
        return {"status": "failed", "error": f"could not read image: {type(exc).__name__}: {exc}"}

    red_bbox = _red_pixel_bbox(width, height, rgba)
    if red_bbox is None:
        return {
            "status": "failed",
            "error": "red target pixels were not found in the rendered image",
            "image_artifact": str(image_path),
            "image": {"width": width, "height": height},
        }
    projection = (
        metrics.get("target_projection", {}).get("after_move")
        if isinstance(metrics.get("target_projection"), Mapping)
        else None
    )
    projected_bbox = _projected_bbox_pixels(projection, width, height, metrics)
    if projected_bbox is None:
        return {
            "status": "failed",
            "error": "target interaction-object projection was not recorded",
            "image_artifact": str(image_path),
            "image": {"width": width, "height": height},
            "red_pixel_bbox": red_bbox,
        }

    overlap = _rect_intersection_area(red_bbox, projected_bbox)
    red_in_projection = _red_pixel_stats_in_rect(width, height, rgba, projected_bbox)
    red_center = _rect_center(red_bbox)
    projected_center = _rect_center(projected_bbox)
    center_distance_px = math.dist(red_center, projected_center)
    projected_diag = math.hypot(
        max(0.0, projected_bbox["x_max"] - projected_bbox["x_min"]),
        max(0.0, projected_bbox["y_max"] - projected_bbox["y_min"]),
    )
    passed = red_in_projection["red_pixel_count"] >= max(4.0, red_in_projection["sampled_pixel_count"] * 0.005)
    return {
        "status": "pass" if passed else "failed",
        **(
            {}
            if passed
            else {"error": "target interaction-object projection did not contain OVRTX-rendered red target pixels"}
        ),
        "image_artifact": str(image_path),
        "image": {"width": width, "height": height},
        "red_pixel_bbox": red_bbox,
        "projected_bbox": projected_bbox,
        "red_in_projected_bbox": red_in_projection,
        "overlap_area_px": overlap,
        "center_distance_px": center_distance_px,
    }


def _read_rgba_png(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    offset = 8
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    compressed = bytearray()
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        payload = data[payload_start:payload_end]
        if len(payload) != length:
            raise ValueError("truncated PNG payload")
        if kind == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
        offset = payload_end + 4
    if width <= 0 or height <= 0:
        raise ValueError("missing PNG dimensions")
    if bit_depth != 8 or color_type != 6:
        raise ValueError(f"expected 8-bit RGBA PNG, got bit_depth={bit_depth} color_type={color_type}")
    raw = zlib.decompress(bytes(compressed))
    return width, height, _unfilter_rgba_png_rows(raw, width, height)


def _unfilter_rgba_png_rows(raw: bytes, width: int, height: int) -> bytes:
    stride = width * 4
    rows: list[bytes] = []
    cursor = 0
    previous = bytearray(stride)
    for _row in range(height):
        filter_type = raw[cursor]
        cursor += 1
        current = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        if filter_type == 1:
            for index in range(stride):
                current[index] = (current[index] + (current[index - 4] if index >= 4 else 0)) & 0xFF
        elif filter_type == 2:
            for index in range(stride):
                current[index] = (current[index] + previous[index]) & 0xFF
        elif filter_type == 3:
            for index in range(stride):
                left = current[index - 4] if index >= 4 else 0
                up = previous[index]
                current[index] = (current[index] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            for index in range(stride):
                left = current[index - 4] if index >= 4 else 0
                up = previous[index]
                up_left = previous[index - 4] if index >= 4 else 0
                current[index] = (current[index] + _paeth(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        rows.append(bytes(current))
        previous = current
    return b"".join(rows)


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    up_left_distance = abs(estimate - up_left)
    if left_distance <= up_distance and left_distance <= up_left_distance:
        return left
    if up_distance <= up_left_distance:
        return up
    return up_left


def _red_pixel_bbox(width: int, height: int, rgba: bytes) -> dict[str, float] | None:
    x_min = width
    y_min = height
    x_max = -1
    y_max = -1
    count = 0
    for y in range(height):
        row = y * width * 4
        for x in range(width):
            offset = row + x * 4
            red, green, blue, alpha = rgba[offset : offset + 4]
            if _is_target_red_pixel(red, green, blue, alpha):
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                count += 1
    if count == 0:
        return None
    return {"x_min": float(x_min), "y_min": float(y_min), "x_max": float(x_max), "y_max": float(y_max), "pixel_count": float(count)}


def _red_pixel_stats_in_rect(width: int, height: int, rgba: bytes, rect: Mapping[str, float]) -> dict[str, float]:
    x_min = max(0, min(width, int(math.floor(float(rect["x_min"])))))
    x_max = max(0, min(width, int(math.ceil(float(rect["x_max"])))))
    y_min = max(0, min(height, int(math.floor(float(rect["y_min"])))))
    y_max = max(0, min(height, int(math.ceil(float(rect["y_max"])))))
    sampled = 0
    red_count = 0
    for y in range(y_min, y_max):
        row = y * width * 4
        for x in range(x_min, x_max):
            offset = row + x * 4
            red, green, blue, alpha = rgba[offset : offset + 4]
            sampled += 1
            if _is_target_red_pixel(red, green, blue, alpha):
                red_count += 1
    ratio = (red_count / sampled) if sampled else 0.0
    return {
        "sampled_pixel_count": float(sampled),
        "red_pixel_count": float(red_count),
        "red_pixel_ratio": ratio,
    }


def _projected_bbox_pixels(
    projection: Any,
    width: int,
    height: int,
    metrics: Mapping[str, Any],
) -> dict[str, float] | None:
    if not isinstance(projection, Mapping):
        return None
    region_rect = projection.get("region_rect")
    draw_rect = metrics.get("actual_draw_rect") or metrics.get("requested_draw_rect")
    if not isinstance(region_rect, Mapping) or not isinstance(draw_rect, Mapping):
        return None
    try:
        draw_x = float(draw_rect["x"])
        draw_y = float(draw_rect["y"])
        draw_width = float(draw_rect["width"])
        draw_height = float(draw_rect["height"])
        if draw_width <= 0.0 or draw_height <= 0.0:
            return None
        x_min = (float(region_rect["x_min"]) - draw_x) / draw_width * width
        x_max = (float(region_rect["x_max"]) - draw_x) / draw_width * width
        y_min = (float(region_rect["y_min"]) - draw_y) / draw_height * height
        y_max = (float(region_rect["y_max"]) - draw_y) / draw_height * height
    except (KeyError, TypeError, ValueError):
        return None
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _is_target_red_pixel(red: int, green: int, blue: int, alpha: int) -> bool:
    return (
        alpha > 16
        and red > 20
        and red > green * 1.12
        and red > blue * 1.12
        and red - max(green, blue) > 6
    )


def _rect_center(rect: Mapping[str, float]) -> tuple[float, float]:
    return (
        (float(rect["x_min"]) + float(rect["x_max"])) * 0.5,
        (float(rect["y_min"]) + float(rect["y_max"])) * 0.5,
    )


def _rect_intersection_area(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    x_min = max(float(first["x_min"]), float(second["x_min"]))
    y_min = max(float(first["y_min"]), float(second["y_min"]))
    x_max = min(float(first["x_max"]), float(second["x_max"]))
    y_max = min(float(first["y_max"]), float(second["y_max"]))
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--fixture-id", default="demo_stair_drop_1280x720")
    parser.add_argument(
        "--generated-red-cube-fixture",
        action="store_true",
        help="Use a generated pure OVRTX single-red-cube fixture instead of a manifest fixture.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-object", default="Cube_00")
    parser.add_argument("--target-prim", default=DEFAULT_TARGET_PRIM)
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument("--native-client-module", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE", "ovrtx_bridge_client"))
    parser.add_argument("--native-client-path", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path()))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--move-x", type=float, default=0.0)
    parser.add_argument("--move-y", type=float, default=-1.0)
    parser.add_argument("--move-z", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode",
        choices=("view3d", "direct"),
        default="view3d",
        help="Use Blender's View3D selection operator before moving, or keep the old direct-object path.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Repeat View3D selection without moving the object and report scene-generation dirtiness/exports.",
    )
    parser.add_argument("--selection-repetitions", type=int, default=5)
    parser.add_argument("--selection-settle-seconds", type=float, default=8.0)
    parser.add_argument(
        "--current-scene-generation",
        action="store_true",
        help="Render the imported Blender scene through scene generation instead of the exact-stage adapter.",
    )
    parser.add_argument(
        "--world-assignment-only",
        action="store_true",
        help="Remove and restore the assigned World and report scoped generation/export activity.",
    )
    parser.add_argument(
        "--default-blender-scene",
        action="store_true",
        help="Keep Blender's factory scene instead of importing the selected USD fixture.",
    )
    parser.add_argument(
        "--include-unmapped-selection",
        action="store_true",
        help="In direct mode, also select one intentionally unmapped imported object to validate group rejection.",
    )
    parser.add_argument(
        "--expect-selection-group-rejected",
        action="store_true",
        help="Treat unsupported selection-group diagnostics as the expected backend result.",
    )
    parser.add_argument(
        "--view-perspective",
        choices=("CAMERA", "PERSP"),
        default="CAMERA",
        help="View3D perspective used for the operator seam probe.",
    )
    args = parser.parse_args(list(argv))
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.min_samples <= 0 or args.max_samples <= 0:
        parser.error("--min-samples and --max-samples must be positive")
    if args.max_samples < args.min_samples:
        parser.error("--max-samples must be greater than or equal to --min-samples")
    if args.selection_repetitions <= 0:
        parser.error("--selection-repetitions must be positive")
    if args.selection_settle_seconds <= 0:
        parser.error("--selection-settle-seconds must be positive")
    if args.selection_only and args.world_assignment_only:
        parser.error("--selection-only and --world-assignment-only are mutually exclusive")
    if args.world_assignment_only and not args.current_scene_generation:
        parser.error("--world-assignment-only requires --current-scene-generation")
    if args.selection_only and not args.current_scene_generation:
        parser.error("--selection-only requires --current-scene-generation")
    if args.default_blender_scene and not args.current_scene_generation:
        parser.error("--default-blender-scene requires --current-scene-generation")
    if not args.target_prim.startswith("/"):
        parser.error("--target-prim must be an absolute USD prim path")
    return args


def _generated_red_cube_fixture(path: Path, width: int, height: int) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Scope "Looks"
    {{
        def Material "TargetRed"
        {{
            token outputs:surface.connect = </World/Looks/TargetRed/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1, 0.02, 0.01)
                float inputs:roughness = 0.35
                float inputs:metallic = 0
                float inputs:ior = 1.45
                color3f inputs:emissiveColor = (0, 0, 0)
                float inputs:opacity = 1
                token outputs:surface
            }}
        }}
    }}

    def Cube "Cube_00" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = </World/Looks/TargetRed>
        double size = 1
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        double3 xformOp:translate = (0, 0, 0.6)
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def Camera "Camera"
    {{
        double3 xformOp:translate = (0, -5, 1.2)
        quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        float focalLength = 35
        float horizontalAperture = 36
        float verticalAperture = 20.25
        float2 clippingRange = (0.05, 100)
    }}

    def DomeLight "AmbientDome"
    {{
        float inputs:intensity = 600
        color3f inputs:color = (1, 1, 1)
    }}

    def RectLight "KeyLight"
    {{
        float inputs:intensity = 5000
        color3f inputs:color = (1, 0.95, 0.9)
        float inputs:width = 4
        float inputs:height = 3
        bool inputs:normalize = false
        double3 xformOp:translate = (-1.5, -3.0, 4.0)
        quatd xformOp:orient = (0.9238795, 0.3826834, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
    }}
}}

def "Render"
{{
    def "OmniverseKit"
    {{
        def "HydraTextures"
        {{
            def RenderProduct "ViewportTexture0"
            {{
                rel camera = </World/Camera>
                token omni:rtx:rendermode = "RealTimePathTracing"
                token omni:rtx:background:source:type = "sky"
                rel orderedVars = </Render/OmniverseKit/HydraTextures/ViewportTexture0/LdrColor>
                uniform int2 resolution = ({int(width)}, {int(height)})
                bool omni:rtx:autoExposure:enabled = false
                bool omni:rtx:rt:ecoMode:enabled = false
                def RenderVar "LdrColor"
                {{
                    uniform string sourceName = "LdrColor"
                }}
            }}
        }}
    }}
}}
''',
        encoding="utf-8",
    )
    return {
        "fixture_usd_path": str(path),
        "render_product_path": "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        "camera_prim_path": "/World/Camera",
    }


def _setup_script(config: Mapping[str, Any]) -> str:
    return r'''
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy
from mathutils import Vector

CONFIG = json.loads(__CONFIG_JSON__)
repo = Path(CONFIG["repo"])
for value in (str(repo / "addon"), CONFIG["native_client_path"]):
    if value and value not in sys.path:
        sys.path.insert(0, value)

metrics = {
    "schema_version": 1,
    "artifact_id": "operator-seam-probe",
    "status": "running",
    "started_at_ns": time.time_ns(),
    "events": [],
    "handler_edit_count": 0,
    "submitted_results": [],
    "bridge_diagnostic_samples": [],
    "selection_resolution": {},
    "selection_attempts": [],
    "stock_export_events": [],
    "dirty_events": [],
    "depsgraph_submissions": [],
    "target_projection": {},
    "move_executed": False,
    "expect_selection_group_rejected": bool(CONFIG["expect_selection_group_rejected"]),
}
move_started_at = time.monotonic()
submitted_once = False
selection_baseline = None
selection_completed_at = None
world_assignment_phase = ""
original_world = None
world_generation_numbers = []


def _jsonable(value):
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("ovrtx_blender_example"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _active_viewport_info():
    best = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
            if region is None:
                continue
            if best is None or region.width * region.height > best["region_width"] * best["region_height"]:
                best = {
                    "region_width": int(region.width),
                    "region_height": int(region.height),
                    "area_width": int(area.width),
                    "area_height": int(area.height),
                    "view_perspective": str(getattr(getattr(space, "region_3d", None), "view_perspective", "")),
                }
    return best or {}


def _active_camera_frame_info(scene):
    camera = scene.camera
    if camera is None:
        return {}
    try:
        from bpy_extras.view3d_utils import location_3d_to_region_2d
    except Exception:
        return {"error": "view3d_utils_unavailable"}

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
            region_data = getattr(space, "region_3d", None) if space else None
            if region is None or region_data is None:
                continue
            points = []
            for corner in camera.data.view_frame(scene=scene):
                world = camera.matrix_world @ corner
                projected = location_3d_to_region_2d(region, region_data, world)
                if projected is None:
                    return {"error": "camera_frame_projection_failed"}
                points.append((float(projected.x), float(projected.y)))
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            return {
                "points": points,
                "rect": {
                    "x": min(xs),
                    "y": min(ys),
                    "width": max(xs) - min(xs),
                    "height": max(ys) - min(ys),
                },
            }
    return {}


def _object_descendants(obj):
    descendants = []
    stack = list(getattr(obj, "children", ()))
    while stack:
        child = stack.pop()
        descendants.append(child)
        stack.extend(getattr(child, "children", ()))
    return descendants


def _active_view3d_region():
    best = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((region for region in area.regions if region.type == "WINDOW"), None)
            space = next((space for space in area.spaces if space.type == "VIEW_3D"), None)
            region_data = getattr(space, "region_3d", None) if space else None
            if region is None or region_data is None:
                continue
            area_pixels = int(region.width) * int(region.height)
            if best is None or area_pixels > best[0]:
                best = (area_pixels, region, region_data)
    if best is None:
        return None, None
    return best[1], best[2]


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


def _object_world_points(obj):
    points = []

    def _add_bound_box_points(item):
        bound_box = getattr(item, "bound_box", None)
        if not bound_box:
            return
        for corner in bound_box:
            points.append(item.matrix_world @ Vector(corner))

    if getattr(obj, "type", "") != "EMPTY":
        _add_bound_box_points(obj)
    for child in _object_descendants(obj):
        _add_bound_box_points(child)
    if points:
        return points

    center = obj.matrix_world.to_translation()
    radius = max(0.05, float(getattr(obj, "empty_display_size", 0.25) or 0.25))
    for dx in (-radius, radius):
        for dy in (-radius, radius):
            for dz in (-radius, radius):
                points.append(center + Vector((dx, dy, dz)))
    return points


def _target_projection(obj):
    try:
        from bpy_extras.view3d_utils import location_3d_to_region_2d
    except Exception as exc:
        return {"status": "failed", "error": f"view3d_utils_unavailable:{type(exc).__name__}"}
    region, region_data = _active_view3d_region()
    if region is None or region_data is None:
        return {"status": "failed", "error": "no_active_view3d_region"}
    projected = []
    for point in _object_world_points(obj):
        value = location_3d_to_region_2d(region, region_data, point)
        if value is not None:
            projected.append((float(value.x), float(value.y)))
    if not projected:
        return {"status": "failed", "error": "target_projection_empty"}
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    width = max(1.0, float(region.width))
    height = max(1.0, float(region.height))
    return {
        "status": "pass",
        "point_count": len(projected),
        "region": {"width": int(region.width), "height": int(region.height)},
        "region_rect": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "normalized_rect": {
            "x_min": x_min / width,
            "y_min": y_min / height,
            "x_max": x_max / width,
            "y_max": y_max / height,
        },
    }


def _projection_center(projection):
    rect = projection.get("region_rect") if isinstance(projection, dict) else None
    if not isinstance(rect, dict):
        return None
    return (
        (float(rect["x_min"]) + float(rect["x_max"])) * 0.5,
        (float(rect["y_min"]) + float(rect["y_max"])) * 0.5,
    )


def _raycast_at_region_coord(view_context, coord):
    try:
        from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
    except Exception as exc:
        return {"status": "failed", "error": f"view3d_utils_unavailable:{type(exc).__name__}"}
    try:
        region = view_context["region"]
        region_data = view_context["region_data"]
        origin = region_2d_to_origin_3d(region, region_data, coord)
        direction = region_2d_to_vector_3d(region, region_data, coord)
        hit, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
            bpy.context.evaluated_depsgraph_get(),
            origin,
            direction,
            distance=1000.0,
        )
        return {
            "status": "hit" if hit else "miss",
            "object": getattr(obj, "name", "") if obj is not None else "",
            "location": [float(value) for value in location] if hit else [],
            "normal": [float(value) for value in normal] if hit else [],
            "index": int(index) if hit else -1,
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _select_target_via_view3d(target, projection):
    view_context = _active_view3d_context()
    if view_context is None:
        return {"status": "failed", "error": "no_active_view3d_context"}
    center = _projection_center(projection)
    if center is None:
        return {"status": "failed", "error": "target_projection_center_unavailable"}
    click = (int(round(center[0])), int(round(center[1])))
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None
    raycast = _raycast_at_region_coord(view_context, click)
    try:
        with bpy.context.temp_override(**view_context):
            result = bpy.ops.view3d.select(location=click, extend=False, deselect_all=True)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"view3d_select_failed:{type(exc).__name__}: {exc}",
            "click_region": [click[0], click[1]],
            "raycast": raycast,
        }
    active = getattr(bpy.context.view_layer.objects, "active", None)
    selected = [obj.name for obj in bpy.context.selected_objects]
    return {
        "status": "pass" if target.name in selected else "failed",
        "error": "" if target.name in selected else "view3d_select_did_not_select_target",
        "operator_result": sorted(str(item) for item in result),
        "click_region": [click[0], click[1]],
        "expected_object": target.name,
        "active_object": getattr(active, "name", "") if active is not None else "",
        "selected_objects": selected,
        "raycast": raycast,
    }


def _fit_size(source_width, source_height, target_width, target_height):
    source_aspect = source_width / max(1, source_height)
    target_aspect = target_width / max(1, target_height)
    if target_aspect > source_aspect:
        height = float(target_height)
        return height * source_aspect, height
    width = float(target_width)
    return width, width / source_aspect


try:
    if not CONFIG["default_blender_scene"]:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        bpy.ops.wm.usd_import(
            filepath=str(Path(CONFIG["input_usd_path"]).expanduser().resolve()),
            property_import_mode="ALL",
            merge_parent_xform=False,
        )

    os.environ["OV_BLENDER_EXAMPLE_WORKER_COMMAND"] = CONFIG["worker_command"]
    os.environ["OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE"] = CONFIG["native_client_module"]

    import ovrtx_blender_example
    from ovrtx_blender_example.engine import (
        _ACTIVE_VIEWPORT_ENGINES,
        interactive_edit_bridge_diagnostics,
    )

    ovrtx_blender_example.register()
    metrics["runtime_service_requested"] = bool(
        ovrtx_blender_example.start_runtime_services_async()
    ) if CONFIG["world_assignment_only"] else False
    from ovrtx_blender_example import engine as ovrtx_engine
    from ovrtx_blender_example import runtime_services, scene_generation, scene_generation_sessions
    def _world_dome_present(usd_path):
        from pxr import Usd

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError("scene generation could not be opened: " + str(usd_path))
        return any(str(prim.GetTypeName()) == "DomeLight" for prim in stage.Traverse())

    original_stock_export = scene_generation._stock_export
    def _recording_stock_export(export_scene, path, **kwargs):
        metrics["stock_export_events"].append({
            "path": str(path),
            "selected_objects_only": bool(kwargs.get("selected_objects_only", False)),
            "time_s": time.monotonic() - move_started_at,
        })
        return original_stock_export(export_scene, path, **kwargs)
    scene_generation._stock_export = _recording_stock_export

    original_mark_scene_dirty = scene_generation_sessions.mark_scene_dirty
    def _recording_mark_scene_dirty(dirty_scene, affected_ids=(), **kwargs):
        event = {
            "affected_ids": [
                {"kind": identity.kind, "session_uid": identity.session_uid}
                for identity in sorted(affected_ids)
            ],
            "time_s": time.monotonic() - move_started_at,
        }
        accepted = original_mark_scene_dirty(dirty_scene, affected_ids, **kwargs)
        event["accepted"] = bool(accepted)
        metrics["dirty_events"].append(event)
        return accepted
    scene_generation_sessions.mark_scene_dirty = _recording_mark_scene_dirty

    original_submit_depsgraph = ovrtx_engine.submit_depsgraph_interactive_edits_to_active_viewports
    def _recording_submit_depsgraph(depsgraph, *, context=None, scene=None):
        results = original_submit_depsgraph(depsgraph, context=context, scene=scene)
        metrics["depsgraph_submissions"].append({
            "ids": [
                {
                    "identifier": str(getattr(getattr(getattr(update, "id", update), "bl_rna", None), "identifier", "")),
                    "name": str(getattr(getattr(update, "id", update), "name", "")),
                }
                for update in getattr(depsgraph, "updates", ())
            ],
            "results": [_jsonable(result) for result in results],
            "time_s": time.monotonic() - move_started_at,
        })
        return results
    ovrtx_engine.submit_depsgraph_interactive_edits_to_active_viewports = _recording_submit_depsgraph

    if not CONFIG["current_scene_generation"]:
        ovrtx_engine.configure_exact_stage(
            input_usd_path=CONFIG["input_usd_path"],
            camera_prim_path=CONFIG["camera_prim_path"],
            render_product_path=CONFIG["render_product_path"],
        )
    metrics["production_bridge_registered"] = bool(interactive_edit_bridge_diagnostics().get("registered"))

    scene = bpy.context.scene
    if scene.camera is None:
        imported_cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
        if len(imported_cameras) == 1:
            scene.camera = imported_cameras[0]
    scene.render.engine = "OVRTX_EXAMPLE"
    scene.render.resolution_x = int(CONFIG["width"])
    scene.render.resolution_y = int(CONFIG["height"])
    scene.render.resolution_percentage = 100
    scene.ovrtx_example.min_samples = int(CONFIG["min_samples"])
    scene.ovrtx_example.max_samples = int(CONFIG["max_samples"])
    scene.ovrtx_example.sync_viewport_camera = True

    target_name = "Cube" if CONFIG["default_blender_scene"] else CONFIG["target_object"]
    target = bpy.data.objects.get(target_name)
    if target is None:
        raise RuntimeError("target object not found: " + target_name)
    target.hide_select = False
    target["ovrtx.usd_layer_id"] = "/layers/operator-seam-probe.usda"
    target["ovrtx.usd_prim_path"] = CONFIG["target_prim"]
    target["ovrtx.blender_property_path"] = "matrix_world"
    target["ovrtx.data_authority"] = "view"
    if CONFIG["selection_mode"] == "direct":
        for obj in bpy.data.objects:
            obj.select_set(False)
        target.select_set(True)
        if CONFIG["include_unmapped_selection"]:
            unmapped = bpy.data.objects.new("OVRTX_Unmapped_Selection_Source", None)
            bpy.context.collection.objects.link(unmapped)
            unmapped.select_set(True)
            metrics["unmapped_selection_object"] = unmapped.name
        bpy.context.view_layer.objects.active = target

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "RENDERED"
                        space.show_gizmo = True
                        space.overlay.show_outline_selected = True
                        if getattr(space, "region_3d", None) is not None and scene.camera is not None:
                            space.region_3d.view_perspective = CONFIG["view_perspective"]
                            if CONFIG["view_perspective"] == "CAMERA":
                                space.region_3d.view_camera_offset[0] = 0.0
                                space.region_3d.view_camera_offset[1] = 0.0
                                space.region_3d.view_camera_zoom = 0.0
                            space.region_3d.update()
                area.tag_redraw()

    viewport = _active_viewport_info()
    metrics["viewport"] = viewport
    metrics["camera"] = {
        "scene_camera": scene.camera.name if scene.camera else None,
        "lens": float(scene.camera.data.lens) if scene.camera else None,
        "sensor_width": float(scene.camera.data.sensor_width) if scene.camera else None,
        "sensor_height": float(scene.camera.data.sensor_height) if scene.camera else None,
    }
    metrics["camera_frame"] = _active_camera_frame_info(scene)
    if viewport:
        draw_width, draw_height = _fit_size(
            int(CONFIG["width"]),
            int(CONFIG["height"]),
            viewport["region_width"],
            viewport["region_height"],
        )
        metrics["requested_draw_rect"] = {
            "x": (viewport["region_width"] - draw_width) * 0.5,
            "y": (viewport["region_height"] - draw_height) * 0.5,
            "width": draw_width,
            "height": draw_height,
        }
    metrics["target_projection"]["before_move"] = _target_projection(target)
    metrics["selection_mode"] = CONFIG["selection_mode"]

    def _record_bridge_state(event):
        global submitted_once
        bridge = interactive_edit_bridge_diagnostics()
        sample = {
            "event": event,
            "bridge": _jsonable(bridge),
            "active_engine_count": len(list(_ACTIVE_VIEWPORT_ENGINES)),
            "time_s": time.monotonic() - move_started_at,
        }
        metrics["bridge_diagnostic_samples"].append(sample)
        del metrics["bridge_diagnostic_samples"][:-12]
        selection_resolution = bridge.get("selection_resolution", {})
        if isinstance(selection_resolution, dict):
            metrics["selection_resolution"] = _jsonable(selection_resolution)
        submitted_count = int(bridge.get("last_submitted_edit_count", 0) or 0)
        result_count = int(bridge.get("last_result_count", 0) or 0)
        if not metrics.get("move_executed") or submitted_count <= 0 or result_count <= 0:
            return
        if submitted_once:
            return
        metrics["handler_edit_count"] = max(int(metrics["handler_edit_count"]), submitted_count)
        metrics["submitted_results"].append({"source": "production_bridge", "bridge": _jsonable(bridge)})
        metrics["active_engine_count_at_submit"] = len(list(_ACTIVE_VIEWPORT_ENGINES))
        submitted_once = True

    def _watch_bridge():
        if CONFIG["selection_only"] or CONFIG["world_assignment_only"]:
            if selection_completed_at is None:
                return 0.25
            if time.monotonic() - selection_completed_at >= float(CONFIG["selection_settle_seconds"]):
                return _finish()
            return 0.25
        if not metrics.get("move_executed"):
            return 0.25
        _record_bridge_state("bridge_poll")
        if submitted_once:
            return _finish()
        return 0.25

    def _redraw_viewports():
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
        try:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        except Exception:
            pass

    def _redraw_tick():
        _redraw_viewports()
        return 0.25

    def _move_when_ready():
        global selection_baseline, selection_completed_at
        global world_assignment_phase, original_world
        if not list(_ACTIVE_VIEWPORT_ENGINES):
            return 0.25
        if CONFIG["selection_only"]:
            if CONFIG["current_scene_generation"]:
                generation = scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)
                if generation.get("status") not in {"current", "pending"}:
                    return 0.25
            if selection_baseline is None:
                selection_baseline = {
                    "stock_export_count": len(metrics["stock_export_events"]),
                    "dirty_event_count": len(metrics["dirty_events"]),
                    "generation_number": int(generation["number"]),
                    "generation_digest": str(generation["digest"]),
                    "generation_usd_path": str(generation["usd_path"]),
                }
                metrics["selection_baseline"] = dict(selection_baseline)
            selection = _select_target_via_view3d(
                target,
                metrics["target_projection"]["before_move"],
            )
            metrics["selection_attempts"].append(selection)
            metrics["events"].append({
                "event": "view3d_select",
                "index": len(metrics["selection_attempts"]),
                "status": selection.get("status"),
            })
            if selection.get("status") != "pass":
                metrics["selection_failed"] = True
                selection_completed_at = time.monotonic()
                return None
            bpy.context.view_layer.update()
            _redraw_viewports()
            if len(metrics["selection_attempts"]) >= int(CONFIG["selection_repetitions"]):
                selection_completed_at = time.monotonic()
                return None
            return 0.5
        if CONFIG["world_assignment_only"]:
            generation = scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)
            runtime_service = runtime_services.owner.diagnostics()
            metrics["world_probe_snapshot"] = {
                "phase": world_assignment_phase,
                "generation": _jsonable(generation),
                "runtime_service": _jsonable(runtime_service),
                "scene_generation_sessions": _jsonable(
                    scene_generation_sessions.diagnostics()
                ),
                "stock_export_count": len(metrics["stock_export_events"]),
                "dirty_event_count": len(metrics["dirty_events"]),
            }
            Path(CONFIG["metrics_path"]).write_text(
                json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if runtime_service.get("status") == "failed":
                metrics["world_assignment_failed"] = str(runtime_service.get("error", ""))
                selection_completed_at = time.monotonic()
                return None
            if metrics["runtime_service_requested"] and runtime_service.get("status") != "ready":
                return 0.25
            if generation.get("status") != "current":
                return 0.25
            if selection_baseline is None:
                selection_baseline = {
                    "stock_export_count": len(metrics["stock_export_events"]),
                    "dirty_event_count": len(metrics["dirty_events"]),
                    "generation_number": int(generation["number"]),
                }
                metrics["selection_baseline"] = dict(selection_baseline)
            if not world_assignment_phase:
                original_world = scene.world
                if original_world is None:
                    metrics["world_assignment_failed"] = "scene has no World to remove"
                    selection_completed_at = time.monotonic()
                    return None
                scene.world = None
                _redraw_viewports()
                world_assignment_phase = "waiting_for_removal"
                return 0.25
            if world_assignment_phase == "waiting_for_removal":
                if int(generation["number"]) <= selection_baseline["generation_number"]:
                    return 0.25
                metrics["world_removal_dome_present"] = _world_dome_present(
                    generation["usd_path"]
                )
                if metrics["world_removal_dome_present"]:
                    metrics["world_assignment_failed"] = (
                        "removed World generation still contains a DomeLight"
                    )
                    selection_completed_at = time.monotonic()
                    return None
                world_generation_numbers.append(int(generation["number"]))
                scene.world = original_world
                _redraw_viewports()
                world_assignment_phase = "waiting_for_restore"
                return 0.25
            if int(generation["number"]) <= world_generation_numbers[-1]:
                return 0.25
            metrics["world_restore_dome_present"] = _world_dome_present(
                generation["usd_path"]
            )
            if not metrics["world_restore_dome_present"]:
                metrics["world_assignment_failed"] = (
                    "restored World generation has no DomeLight"
                )
                selection_completed_at = time.monotonic()
                return None
            world_generation_numbers.append(int(generation["number"]))
            metrics["world_generation_numbers"] = list(world_generation_numbers)
            selection_completed_at = time.monotonic()
            return None
        if CONFIG["selection_mode"] == "view3d" and not metrics.get("selection_attempt"):
            selection = _select_target_via_view3d(target, metrics["target_projection"]["before_move"])
            metrics["selection_attempt"] = selection
            metrics["events"].append({
                "event": "view3d_select",
                "status": selection.get("status"),
                "active_engine_count": len(list(_ACTIVE_VIEWPORT_ENGINES)),
            })
            if selection.get("status") != "pass":
                metrics["selection_failed"] = True
                return None
        target.location.x += float(CONFIG["move_x"])
        target.location.y += float(CONFIG["move_y"])
        target.location.z += float(CONFIG["move_z"])
        metrics["move_executed"] = True
        metrics["events"].append({
            "event": "moved_target",
            "active_engine_count": len(list(_ACTIVE_VIEWPORT_ENGINES)),
            "location": [float(target.location.x), float(target.location.y), float(target.location.z)],
        })
        bpy.context.view_layer.update()
        metrics["target_projection"]["after_move"] = _target_projection(target)
        _record_bridge_state("after_move_update")
        _redraw_viewports()
        return None

    def _finish():
        try:
            _record_bridge_state("finish")
            written = ovrtx_blender_example.write_viewport_session_outputs()
            metrics["viewport_output_write_count"] = written
            artifact_path = Path(CONFIG["viewport_artifact_path"])
            if artifact_path.is_file():
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                metrics["viewport_artifact"] = {
                    "status": artifact.get("status"),
                    "snapshot_index": artifact.get("snapshot_index"),
                    "render_count": artifact.get("render_count"),
                    "draw_count": artifact.get("draw_count"),
                    "width": artifact.get("width"),
                    "height": artifact.get("height"),
                    "last_update": artifact.get("shared_stage_composition", {}).get("last_update", {}),
                    "interactive_edit_bridge": artifact.get("interactive_edit_bridge", {}),
                    "interactive_edit_workflow": artifact.get("interactive_edit_workflow", {}),
                    "operator_viewport": artifact.get("operator_viewport", {}),
                }
                operator_viewport = artifact.get("operator_viewport", {})
                texture_draw_rect = (
                    operator_viewport.get("texture_draw_rect", {})
                    if isinstance(operator_viewport, dict)
                    else {}
                )
                if isinstance(texture_draw_rect, dict) and texture_draw_rect:
                    metrics["actual_draw_rect"] = texture_draw_rect
                elif metrics.get("viewport") and artifact.get("width") and artifact.get("height"):
                    draw_width, draw_height = _fit_size(
                        int(artifact["width"]),
                        int(artifact["height"]),
                        int(metrics["viewport"]["region_width"]),
                        int(metrics["viewport"]["region_height"]),
                    )
                    metrics["actual_draw_rect"] = {
                        "x": (int(metrics["viewport"]["region_width"]) - draw_width) * 0.5,
                        "y": (int(metrics["viewport"]["region_height"]) - draw_height) * 0.5,
                        "width": draw_width,
                        "height": draw_height,
                    }
            if metrics.get("selection_failed"):
                metrics["status"] = "failed"
                metrics["error"] = "Blender View3D selection did not select the target object"
            elif CONFIG["selection_only"]:
                baseline = selection_baseline or {"stock_export_count": 0, "dirty_event_count": 0}
                metrics["selection_stock_export_count"] = (
                    len(metrics["stock_export_events"]) - baseline["stock_export_count"]
                )
                metrics["selection_dirty_event_count"] = (
                    len(metrics["dirty_events"]) - baseline["dirty_event_count"]
                )
                metrics["scene_generation"] = _jsonable(
                    scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)
                )
                generation_unchanged = all(
                    metrics["scene_generation"].get(key) == baseline.get(baseline_key)
                    for key, baseline_key in (
                        ("number", "generation_number"),
                        ("digest", "generation_digest"),
                        ("usd_path", "generation_usd_path"),
                    )
                )
                metrics["status"] = (
                    "pass"
                    if metrics["selection_stock_export_count"] == 0
                    and metrics["selection_dirty_event_count"] == 0
                    and generation_unchanged
                    else "failed"
                )
                if metrics["status"] != "pass":
                    metrics["error"] = "selection caused scene-generation dirtiness or USD export"
            elif CONFIG["world_assignment_only"]:
                baseline = selection_baseline or {"stock_export_count": 0, "dirty_event_count": 0}
                post_baseline_dirty = metrics["dirty_events"][baseline["dirty_event_count"]:]
                metrics["world_stock_export_count"] = (
                    len(metrics["stock_export_events"]) - baseline["stock_export_count"]
                )
                metrics["world_accepted_dirty_count"] = sum(
                    bool(event.get("accepted")) for event in post_baseline_dirty
                )
                metrics["world_rejected_dirty_count"] = sum(
                    not bool(event.get("accepted")) for event in post_baseline_dirty
                )
                metrics["scene_generation"] = _jsonable(
                    scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)
                )
                metrics["status"] = (
                    "pass"
                    if not metrics.get("world_assignment_failed")
                    and len(world_generation_numbers) == 2
                    and metrics["world_stock_export_count"] == 2
                    and metrics["world_accepted_dirty_count"] == 2
                    else "failed"
                )
                if metrics["status"] != "pass":
                    metrics["error"] = "World remove/restore did not reconcile exactly once per transition"
            elif CONFIG["expect_selection_group_rejected"]:
                selection_resolution = metrics.get("selection_resolution", {})
                group_rejected = bool(selection_resolution.get("group_rejected", False))
                unresolved_reasons = selection_resolution.get("unresolved_reasons", [])
                unmapped_reason = "unmapped_selection_source" in unresolved_reasons
                metrics["status"] = "pass" if group_rejected and unmapped_reason else "failed"
                if not group_rejected:
                    metrics["error"] = "expected selection group rejection diagnostics were not recorded"
                elif not unmapped_reason:
                    metrics["error"] = "expected unmapped selection source diagnostics were not recorded"
            else:
                metrics["status"] = "pass" if submitted_once else "incomplete"
            if (
                metrics["status"] != "pass"
                and not submitted_once
                and not CONFIG["selection_only"]
                and not CONFIG["world_assignment_only"]
            ):
                metrics["error"] = metrics.get("error") or "production depsgraph bridge did not submit a target edit"
        except Exception as exc:
            metrics["status"] = "failed"
            metrics["error"] = str(exc)
            metrics["traceback"] = traceback.format_exc()
        finally:
            metrics["completed_at_ns"] = time.time_ns()
            Path(CONFIG["metrics_path"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(_move_when_ready, first_interval=1.0)
    bpy.app.timers.register(_watch_bridge, first_interval=1.25)
    bpy.app.timers.register(_redraw_tick, first_interval=0.1)
except Exception as exc:
    metrics["status"] = "failed"
    metrics["error"] = str(exc)
    metrics["traceback"] = traceback.format_exc()
    Path(CONFIG["metrics_path"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise
'''.replace("__CONFIG_JSON__", repr(json.dumps(dict(config), sort_keys=True)))


if __name__ == "__main__":
    raise SystemExit(main())
