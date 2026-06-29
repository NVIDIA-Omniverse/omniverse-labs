#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render all authored Moana cameras with the Vulkan RT backend.

The Moana island scene does not require pxr.USD for this report: the root
USDA file authors the camera matrices directly, so this script parses those
camera blocks, renders each view through the local nusd renderer binding, and
emits individual PNGs, a contact sheet, and timing metadata.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = Path("$HOME/moana-island-scene-usd/island/usd/island.usda")
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "reports" / "moana_vulkan_rt_materials_2026-05-31"
MOANA_ENV_DEFAULTS = {
    "NUSD_ENABLE_MATERIALS": "1",
    "NUSD_ENABLE_PTEX_MATERIALS": "1",
    "NUSD_CLAY_VIZ": "0",
    "NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL": "1",
    "NUSD_RENDER_PI_BATCHES": "1",
    "NUSD_NATIVE_ARC_CHASE_AFTER_DIRECT": "1",
    "NUSD_NATIVE_CURVES": "all",
    "NUSD_CURVE_SUBSEGS": "1",
}
MOANA_CULL_ENV = (
    "NUSD_RT_CULL",
    "NUSD_NO_CULL_ALL_GEOMETRY",
    "NUSD_ALL_GEOMETRY_NO_CULL",
    "NUSD_RT_CAMERA_RESIDENCY",
)


@dataclass
class CameraSpec:
    name: str
    path: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float
    near_clip: float
    far_clip: float
    focus_distance: float
    projection_shift: tuple[float, float]
    image: str | None = None
    render_seconds: float | None = None
    fetch_seconds: float | None = None
    phase_timings_ms: dict[str, float] | None = None


def _parse_float(body: str, attr: str, default: float) -> float:
    m = re.search(rf"\bfloat\s+{re.escape(attr)}\s*=\s*([-+0-9.eE]+)", body)
    return float(m.group(1)) if m else default


def _parse_float2(body: str, attr: str, default: tuple[float, float]) -> tuple[float, float]:
    m = re.search(
        rf"\bfloat2\s+{re.escape(attr)}\s*=\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        body,
    )
    return (float(m.group(1)), float(m.group(2))) if m else default


def _parse_matrix(body: str, op_name: str) -> np.ndarray | None:
    m = re.search(rf"\bmatrix4d\s+{re.escape(op_name)}\s*=\s*(\([^\n]+\))", body)
    if not m:
        return None
    rows = ast.literal_eval(m.group(1))
    mat = np.asarray(rows, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{op_name} parsed as {mat.shape}, expected (4, 4)")
    return mat


def _normalize3(v: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _compose_camera_matrix(body: str) -> np.ndarray:
    order_match = re.search(r"\buniform\s+token\[\]\s+xformOpOrder\s*=\s*\[([^\]]*)\]", body)
    order = re.findall(r'"([^"]+)"', order_match.group(1)) if order_match else []
    ops: dict[str, np.ndarray] = {}
    for op_name in order:
        mat = _parse_matrix(body, op_name)
        if mat is not None:
            ops[op_name] = mat

    if not ops:
        transform = _parse_matrix(body, "xformOp:transform:transform")
        return transform if transform is not None else np.eye(4, dtype=np.float64)

    # USD stores camera transforms in row-vector form. Reversing op order here
    # preserves the authored translation row for stacks like [translate, lookat].
    composed = np.eye(4, dtype=np.float64)
    for op_name in reversed(order):
        mat = ops.get(op_name)
        if mat is not None:
            composed = composed @ mat
    return composed


def parse_moana_cameras(usd_path: Path) -> list[CameraSpec]:
    text = usd_path.read_text(encoding="utf-8")
    cameras: list[CameraSpec] = []
    for match in re.finditer(r'def\s+Camera\s+"([^"]+)"\s*\{(.*?)\n\s*\}', text, re.S):
        name = match.group(1)
        body = match.group(2)
        matrix = _compose_camera_matrix(body)

        eye_v = matrix[3, :3].astype(np.float64)
        up_v = _normalize3(matrix[1, :3], (0.0, 1.0, 0.0))
        forward_v = _normalize3(-matrix[2, :3], (0.0, 0.0, -1.0))
        focus = _parse_float(body, "focusDistance", 100.0)
        if not np.isfinite(focus) or focus <= 1e-6:
            focus = 100.0
        target_v = eye_v + forward_v * focus

        focal = _parse_float(body, "focalLength", 35.0)
        vertical_aperture = _parse_float(body, "verticalAperture", 20.955)
        fov = math.degrees(2.0 * math.atan(vertical_aperture / max(2.0 * focal, 1e-6)))
        horizontal_aperture = _parse_float(body, "horizontalAperture", 36.0)
        horizontal_offset = _parse_float(body, "horizontalApertureOffset", 0.0)
        vertical_offset = _parse_float(body, "verticalApertureOffset", 0.0)
        clip = _parse_float2(body, "clippingRange", (0.1, 10000.0))
        near_clip = float(clip[0])
        if focus > 0.0:
            near_clip = min(near_clip, max(0.1, focus * 0.02))
        shift_x = (2.0 * horizontal_offset / horizontal_aperture) if horizontal_aperture > 0.0 else 0.0
        shift_y = (2.0 * vertical_offset / vertical_aperture) if vertical_aperture > 0.0 else 0.0

        cameras.append(
            CameraSpec(
                name=name,
                path=f"/island/cam/{name}",
                eye=tuple(float(x) for x in eye_v),
                target=tuple(float(x) for x in target_v),
                up=tuple(float(x) for x in up_v),
                fov_degrees=float(fov),
                near_clip=near_clip,
                far_clip=float(clip[1]),
                focus_distance=float(focus),
                projection_shift=(float(shift_x), float(shift_y)),
            )
        )
    if not cameras:
        raise RuntimeError(f"No cameras found in {usd_path}")
    return cameras


def _tuple_arg(text: str, count: int, label: str) -> tuple[float, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != count:
        raise argparse.ArgumentTypeError(f"{label} must have {count} comma-separated numbers")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be numeric") from exc


def _mean_target(cameras: list[CameraSpec]) -> np.ndarray:
    if not cameras:
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    return np.asarray([c.target for c in cameras], dtype=np.float64).mean(axis=0)


def _median_camera_distance(cameras: list[CameraSpec]) -> float:
    distances = [
        float(np.linalg.norm(np.asarray(c.eye, dtype=np.float64) -
                             np.asarray(c.target, dtype=np.float64)))
        for c in cameras
    ]
    distances = [d for d in distances if np.isfinite(d) and d > 1.0]
    if not distances:
        return 1200.0
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def make_orbit_cameras(authored: list[CameraSpec], args: argparse.Namespace) -> list[CameraSpec]:
    count = max(0, int(args.orbit_count))
    if count == 0:
        return []

    center = (np.asarray(args.orbit_center, dtype=np.float64)
              if args.orbit_center is not None
              else _mean_target(authored))
    radius = float(args.orbit_radius) if args.orbit_radius is not None else (
        _median_camera_distance(authored) * float(args.orbit_radius_scale)
    )
    radius = max(radius, 1.0)
    start = math.radians(float(args.orbit_start_degrees))

    cameras: list[CameraSpec] = []
    for i in range(count):
        angle = start + 2.0 * math.pi * (float(i) / float(count))
        eye = np.asarray((
            center[0] + math.cos(angle) * radius,
            center[1] + float(args.orbit_height),
            center[2] + math.sin(angle) * radius,
        ), dtype=np.float64)
        cameras.append(
            CameraSpec(
                name=f"orbit_{i:02d}",
                path=f"/orbit/{i:02d}",
                eye=tuple(float(x) for x in eye),
                target=tuple(float(x) for x in center),
                up=(0.0, 1.0, 0.0),
                fov_degrees=float(args.orbit_fov),
                near_clip=float(args.orbit_near_clip),
                far_clip=float(args.orbit_far_clip),
                focus_distance=radius,
                projection_shift=(0.0, 0.0),
            )
        )
    return cameras


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _camera_label(camera: CameraSpec) -> str:
    phase = camera.phase_timings_ms or {}
    rt_ms = phase.get("rt_dispatch_ms")
    if rt_ms is not None and rt_ms > 0.0:
        return f"{camera.name}  {rt_ms:.1f} ms GPU"
    if camera.render_seconds is not None and camera.fetch_seconds is not None:
        return f"{camera.name}  {(camera.render_seconds + camera.fetch_seconds) * 1000.0:.1f} ms"
    return camera.name


def save_contact_sheet(
    cameras: list[CameraSpec],
    image_paths: list[Path],
    out_path: Path,
    columns: int,
    tile_w: int,
    tile_h: int,
) -> None:
    label_h = 30
    gap = 10
    columns = max(1, columns)
    rows = math.ceil(len(cameras) / columns)
    sheet_w = columns * tile_w + (columns + 1) * gap
    sheet_h = rows * (tile_h + label_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 21, 23))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)

    for i, (camera, image_path) in enumerate(zip(cameras, image_paths)):
        col = i % columns
        row = i // columns
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + label_h + gap)
        draw.rounded_rectangle(
            (x - 1, y - 1, x + tile_w + 1, y + tile_h + label_h + 1),
            radius=4,
            fill=(30, 34, 37),
            outline=(76, 84, 91),
        )
        draw.text((x + 8, y + 6), _camera_label(camera), fill=(235, 238, 240), font=font)
        frame = Image.open(image_path).convert("RGB")
        if frame.size != (tile_w, tile_h):
            frame = frame.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(frame, (x, y + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def configure_render_environment(args: argparse.Namespace) -> dict[str, str]:
    for key, value in MOANA_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ["NUSD_ENABLE_PTEX_MATERIALS"] = "1"
    os.environ["NUSD_CLAY_VIZ"] = "0"
    if args.rt_cull:
        os.environ["NUSD_RT_CULL"] = "1"
    else:
        # The report/contact-sheet path wants a reusable compact TLAS. The
        # camera-culled proxy path intentionally rebuilds RT state per camera.
        for key in MOANA_CULL_ENV:
            os.environ[key] = "0"
    keys = tuple(MOANA_ENV_DEFAULTS.keys()) + MOANA_CULL_ENV
    return {key: os.environ.get(key, "") for key in keys}


def render_cameras(args: argparse.Namespace, cameras: list[CameraSpec]) -> dict[str, Any]:
    env_metadata = configure_render_environment(args)

    sys.path.insert(0, str(REPO_ROOT / "python"))
    from nusd_renderer._bindings import NU_RENDER_RT, NuRenderer

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = NuRenderer(width=args.width, height=args.height, enable_rt=True, enable_materials=True)
    image_paths: list[Path] = []
    try:
        if not renderer.rt_available:
            raise RuntimeError("Vulkan RT is not available on this renderer")

        t0 = time.perf_counter()
        mesh_count = renderer.load_usd(str(args.usd))
        load_seconds = time.perf_counter() - t0
        phase_after_load = renderer.get_phase_timings_ms()
        scene_bounds = renderer.get_scene_bounds()

        authored_cameras = list(cameras)
        orbit_cameras = make_orbit_cameras(authored_cameras, args)
        cameras = orbit_cameras if args.only_orbit else (authored_cameras + orbit_cameras)
        if not cameras:
            raise RuntimeError("No cameras selected for rendering")

        for camera in cameras:
            renderer.set_camera_explicit_window(
                camera.eye,
                camera.target,
                camera.up,
                camera.fov_degrees,
                camera.near_clip,
                camera.far_clip,
                camera.projection_shift,
            )
            rt0 = time.perf_counter()
            renderer.render(NU_RENDER_RT)
            camera.render_seconds = time.perf_counter() - rt0

            fetch0 = time.perf_counter()
            pixels = renderer.fetch_pixels()
            camera.fetch_seconds = time.perf_counter() - fetch0
            camera.phase_timings_ms = renderer.get_phase_timings_ms()

            image_path = frames_dir / f"{camera.name}.png"
            Image.fromarray(pixels[:, :, :3], mode="RGB").save(image_path)
            camera.image = str(image_path.relative_to(out_dir))
            image_paths.append(image_path)

        contact_sheet = out_dir / "moana_camera_contact_sheet.png"
        save_contact_sheet(cameras, image_paths, contact_sheet, args.columns, args.width, args.height)
        final_phase_timings = renderer.get_phase_timings_ms()

        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "usd": str(args.usd),
            "tile_size": [args.width, args.height],
            "contact_sheet": str(contact_sheet.relative_to(out_dir)),
            "environment": env_metadata,
            "renderer": {
                "mesh_count": int(mesh_count),
                "load_seconds": load_seconds,
                "gpu_memory_bytes": int(renderer.gpu_memory_used),
                "gpu_memory_gib": float(renderer.gpu_memory_used) / (1024.0 ** 3),
                "scene_bounds": scene_bounds,
                "phase_timings_after_load_ms": phase_after_load,
                "phase_timings_final_ms": final_phase_timings,
                "cmd_cache_stats": renderer.cmd_cache_stats(),
            },
            "orbit": {
                "count": int(args.orbit_count),
                "only_orbit": bool(args.only_orbit),
                "center": (list(args.orbit_center) if args.orbit_center is not None else None),
                "radius": args.orbit_radius,
                "radius_scale": args.orbit_radius_scale,
                "height": args.orbit_height,
                "fov": args.orbit_fov,
                "start_degrees": args.orbit_start_degrees,
            },
            "cameras": [asdict(camera) for camera in cameras],
        }
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=201)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--rt-cull",
        action="store_true",
        help="Use the camera-dependent culled RT proxy path. This rebuilds RT state per camera.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse cameras and print metadata only.")
    parser.add_argument("--orbit-count", type=int, default=0, help="Append this many generated orbit cameras.")
    parser.add_argument("--only-orbit", action="store_true", help="Render only generated orbit cameras.")
    parser.add_argument(
        "--orbit-center",
        type=lambda s: _tuple_arg(s, 3, "orbit center"),
        default=None,
        help="Orbit center as x,y,z. Defaults to the mean authored camera target.",
    )
    parser.add_argument("--orbit-radius", type=float, default=None, help="Orbit radius in scene units.")
    parser.add_argument(
        "--orbit-radius-scale",
        type=float,
        default=2.25,
        help="Radius multiplier for the median authored camera distance when --orbit-radius is omitted.",
    )
    parser.add_argument("--orbit-height", type=float, default=620.0, help="Eye height above orbit center in Y-up scene units.")
    parser.add_argument("--orbit-fov", type=float, default=35.0, help="Orbit camera vertical FOV in degrees.")
    parser.add_argument("--orbit-start-degrees", type=float, default=20.0, help="First orbit camera azimuth in degrees.")
    parser.add_argument("--orbit-near-clip", type=float, default=10.0)
    parser.add_argument("--orbit-far-clip", type=float, default=1000000.0)
    args = parser.parse_args()

    cameras = parse_moana_cameras(args.usd)
    if args.dry_run:
        orbit_cameras = make_orbit_cameras(cameras, args)
        cameras = orbit_cameras if args.only_orbit else (cameras + orbit_cameras)
        print(json.dumps([asdict(camera) for camera in cameras], indent=2))
        return 0

    metadata = render_cameras(args, cameras)
    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "moana_camera_contact_sheet_metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Rendered {len(metadata['cameras'])} cameras to {out_dir / metadata['contact_sheet']} "
        f"in {metadata['renderer']['load_seconds']:.2f}s load, "
        f"{metadata['renderer']['gpu_memory_gib']:.2f} GiB GPU memory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
