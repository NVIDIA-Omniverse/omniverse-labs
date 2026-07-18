#!/usr/bin/env python3
"""Report registered reference-landmark projection error from inside Blender."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


DEFAULTS = {
    "max_error_px": 4.0,
    "rmse_error_px": 3.0,
    "max_normalized_error": 0.01,
}
DEPTH_STATUSES = {"measured", "constrained_by_other_views", "inferred"}


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-error-px", type=float)
    parser.add_argument("--rmse-error-px", type=float)
    parser.add_argument("--max-normalized-error", type=float)
    return parser.parse_args(raw)


def _finite_vector(values, label: str) -> Vector:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label} must contain three numbers")
    point = Vector(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label} contains a non-finite value")
    return point


def _world_point(item: dict) -> Vector:
    if "world" in item:
        return _finite_vector(item["world"], f"landmark {item.get('id')} world")
    name = item.get("object")
    obj = bpy.data.objects.get(name) if name else None
    if obj is None:
        raise ValueError(f"landmark {item.get('id')} has missing object: {name}")
    local = _finite_vector(item.get("local", [0.0, 0.0, 0.0]), f"landmark {item.get('id')} local")
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return evaluated.matrix_world @ local


def _thresholds(manifest: dict, view: dict, args: argparse.Namespace) -> dict:
    result = dict(DEFAULTS)
    result.update(manifest.get("thresholds", {}))
    result.update(view.get("thresholds", {}))
    overrides = {
        "max_error_px": args.max_error_px,
        "rmse_error_px": args.rmse_error_px,
        "max_normalized_error": args.max_normalized_error,
    }
    result.update({key: value for key, value in overrides.items() if value is not None})
    for key, value in result.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid threshold {key}: {value}")
        result[key] = value
    return result


def _calibration(view: dict, by_id: dict) -> dict | None:
    item = view.get("known_distance")
    if not item:
        return None
    a = by_id.get(item.get("landmark_a"))
    b = by_id.get(item.get("landmark_b"))
    if a is None or b is None:
        raise ValueError("known_distance landmark IDs must exist in the same view")
    pa, pb = a["expected_px"], b["expected_px"]
    pixel_distance = math.hypot(float(pb[0]) - float(pa[0]), float(pb[1]) - float(pa[1]))
    distance_world = float(item.get("distance_world", 0.0))
    if pixel_distance <= 0 or distance_world <= 0:
        raise ValueError("known_distance requires positive pixel and world distances")
    actual_world = math.dist(a["world"], b["world"])
    tolerance_world = float(item.get("tolerance_world", max(1.0e-6, distance_world * 1.0e-4)))
    if not math.isfinite(tolerance_world) or tolerance_world < 0:
        raise ValueError("known_distance tolerance_world must be finite and non-negative")
    error_world = abs(actual_world - distance_world)
    return {
        "landmark_a": item["landmark_a"],
        "landmark_b": item["landmark_b"],
        "pixel_distance": pixel_distance,
        "distance_world": distance_world,
        "actual_distance_world": actual_world,
        "distance_error_world": error_world,
        "tolerance_world": tolerance_world,
        "consistent": error_world <= tolerance_world,
        "unit": item.get("unit", "scene_unit"),
        "world_per_pixel": distance_world / pixel_distance,
    }


def _view_report(scene, manifest: dict, view: dict, args: argparse.Namespace) -> dict:
    view_id = str(view.get("id", ""))
    width, height = [int(value) for value in view.get("image_size_px", [])]
    if width <= 0 or height <= 0:
        raise ValueError(f"view {view_id} has invalid image_size_px")
    camera = bpy.data.objects.get(view.get("camera"))
    if camera is None or camera.type != "CAMERA":
        raise ValueError(f"view {view_id} camera is missing or not a camera")
    expected_projection = view.get("projection", "ORTHO")
    depth_status = view.get("depth_status")
    if depth_status not in DEPTH_STATUSES:
        raise ValueError(f"view {view_id} must declare a valid depth_status")
    if not str(view.get("depth_notes", "")).strip():
        raise ValueError(f"view {view_id} must explain depth_notes")

    old = (scene.render.resolution_x, scene.render.resolution_y,
           scene.render.resolution_percentage, scene.render.pixel_aspect_x,
           scene.render.pixel_aspect_y)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    bpy.context.view_layer.update()
    try:
        records = []
        seen = set()
        landmarks = view.get("landmarks", [])
        for item in landmarks:
            landmark_id = str(item.get("id", ""))
            if not landmark_id or landmark_id in seen:
                raise ValueError(f"view {view_id} has missing or duplicate landmark ID")
            seen.add(landmark_id)
            expected = item.get("expected_px")
            if not isinstance(expected, list) or len(expected) != 2:
                raise ValueError(f"landmark {landmark_id} expected_px must have two values")
            expected_x, expected_y = float(expected[0]), float(expected[1])
            world = _world_point(item)
            projected = world_to_camera_view(scene, camera, world)
            actual_x = float(projected.x) * width
            actual_y = (1.0 - float(projected.y)) * height
            error_px = math.hypot(actual_x - expected_x, actual_y - expected_y)
            records.append({
                "id": landmark_id,
                "expected_px": [expected_x, expected_y],
                "actual_px": [actual_x, actual_y],
                "world": [float(value) for value in world],
                "camera_depth": float(projected.z),
                "in_front": float(projected.z) > 0.0,
                "inside_image": 0.0 <= actual_x <= width and 0.0 <= actual_y <= height,
                "error_px": error_px,
                "normalized_error": error_px / math.hypot(width, height),
            })
        if not records:
            raise ValueError(f"view {view_id} has no landmarks")
        by_id = {record["id"]: record for record in records}
        limits = _thresholds(manifest, view, args)
        rmse = math.sqrt(sum(record["error_px"] ** 2 for record in records) / len(records))
        max_error = max(record["error_px"] for record in records)
        max_normalized = max(record["normalized_error"] for record in records)
        projection_ok = camera.data.type == expected_projection
        calibration = _calibration(view, by_id)
        passed = (
            projection_ok
            and all(record["in_front"] and record["inside_image"] for record in records)
            and max_error <= limits["max_error_px"]
            and rmse <= limits["rmse_error_px"]
            and max_normalized <= limits["max_normalized_error"]
            and (calibration is None or calibration["consistent"])
        )
        if calibration:
            calibration["image_width_world"] = width * calibration["world_per_pixel"]
            calibration["image_height_world"] = height * calibration["world_per_pixel"]
            calibration["camera_ortho_scale"] = float(camera.data.ortho_scale)
        return {
            "id": view_id,
            "ok": passed,
            "camera": camera.name,
            "projection_expected": expected_projection,
            "projection_actual": camera.data.type,
            "image_size_px": [width, height],
            "screen_axes": view.get("screen_axes", {}),
            "depth_status": depth_status,
            "depth_notes": view["depth_notes"],
            "thresholds": limits,
            "metrics": {"max_error_px": max_error, "rmse_error_px": rmse,
                        "max_normalized_error": max_normalized},
            "calibration": calibration,
            "landmarks": records,
        }
    finally:
        (scene.render.resolution_x, scene.render.resolution_y,
         scene.render.resolution_percentage, scene.render.pixel_aspect_x,
         scene.render.pixel_aspect_y) = old


def main() -> int:
    args = _args()
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refuse to overwrite existing output: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    scene = bpy.context.scene
    views = manifest.get("views", [])
    if not views:
        raise ValueError("manifest must contain at least one view")
    reports = []
    for index, view in enumerate(views):
        try:
            reports.append(_view_report(scene, manifest, view, args))
        except Exception as exc:
            reports.append({
                "id": str(view.get("id", f"view-{index}")),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "camera": view.get("camera"),
                "depth_status": view.get("depth_status"),
                "depth_notes": view.get("depth_notes"),
            })
    result = {
        "schema_version": 1,
        "ok": all(report["ok"] for report in reports),
        "blend_file": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "manifest": str(args.manifest.resolve()),
        "views": reports,
        "limitations": [
            "A passing report proves declared 2D landmark projection only.",
            "Inferred depth, occluded surfaces, silhouette continuity, and lens provenance require separate evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "operation": "reference_projection_report",
                      "output": str(args.output.resolve()), "views": len(reports)}, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
