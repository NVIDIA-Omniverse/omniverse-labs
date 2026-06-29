#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile Moana orbit RT frame time across secondary-visibility modes.

Loads the scene once, then renders the same generated orbit cameras with
different RT visibility flags. This isolates the runtime cost of AO/direct
shadow ray queries without conflating it with scene load or AS build.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

import render_moana_camera_contact_sheet as moana


VIS_FLAGS = {
    "NUSD_RT_SKIP_SECONDARY_VISIBILITY",
    "NUSD_RT_SKIP_AO_VISIBILITY",
    "NUSD_RT_SKIP_DIRECT_SHADOWS",
    "NUSD_RT_RECT_SHARED_SHADOWS",
}


MODES: dict[str, dict[str, str]] = {
    "quality": {},
    "rect_shared": {"NUSD_RT_RECT_SHARED_SHADOWS": "1"},
    "skip_ao": {"NUSD_RT_SKIP_AO_VISIBILITY": "1"},
    "skip_direct": {"NUSD_RT_SKIP_DIRECT_SHADOWS": "1"},
    "skip_secondary": {"NUSD_RT_SKIP_SECONDARY_VISIBILITY": "1"},
}


def _font(size: int) -> ImageFont.ImageFont:
    return moana._font(size)  # reuse the contact-sheet script's font fallback


def set_visibility_mode(mode: str) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in VIS_FLAGS}
    for key in VIS_FLAGS:
        os.environ.pop(key, None)
    for key, value in MODES[mode].items():
        os.environ[key] = value
    return previous


def restore_visibility_mode(previous: dict[str, str | None]) -> None:
    for key in VIS_FLAGS:
        value = previous.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def image_diff_metrics(a: Image.Image, b: Image.Image) -> dict[str, float]:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    stat = ImageStat.Stat(diff)
    mae = float(sum(stat.mean) / 3.0)
    rms = float(math.sqrt(sum(v * v for v in stat.rms) / 3.0))
    max_abs = float(max(channel[1] for channel in diff.getextrema()))
    return {"mae": mae, "rms": rms, "max_abs": max_abs}


def save_mode_contact_sheet(
    records: list[dict],
    out_path: Path,
    width: int,
    height: int,
    scale: float,
    columns: int,
) -> None:
    if not records:
        return
    tile_w = max(1, int(width * scale))
    tile_h = max(1, int(height * scale))
    label_h = 26
    gap = 6
    rows = math.ceil(len(records) / columns)
    sheet_w = columns * tile_w + (columns + 1) * gap
    sheet_h = rows * (tile_h + label_h) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 20, 22))
    draw = ImageDraw.Draw(sheet)
    font = _font(12)

    for idx, rec in enumerate(records):
        col = idx % columns
        row = idx // columns
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + label_h + gap)
        label = (
            f"{rec['mode']} {rec['camera']} "
            f"{rec['phase_timings_ms']['rt_dispatch_ms']:.1f} ms"
        )
        draw.rectangle((x - 1, y - 1, x + tile_w + 1, y + tile_h + label_h + 1),
                       fill=(30, 34, 37), outline=(76, 84, 91))
        draw.text((x + 6, y + 6), label, fill=(235, 238, 240), font=font)
        im = Image.open(rec["absolute_image"]).convert("RGB")
        if im.size != (tile_w, tile_h):
            im = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, y + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for mode in MODES:
        vals = [
            rec["phase_timings_ms"]["rt_dispatch_ms"] +
            rec["phase_timings_ms"]["pixel_readback_ms"]
            for rec in records
            if rec["mode"] == mode
        ]
        rt_vals = [
            rec["phase_timings_ms"]["rt_dispatch_ms"]
            for rec in records
            if rec["mode"] == mode
        ]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rt_arr = np.asarray(rt_vals, dtype=np.float64)
        summary[mode] = {
            "frame_min_ms": float(arr.min()),
            "frame_median_ms": float(np.median(arr)),
            "frame_mean_ms": float(arr.mean()),
            "frame_max_ms": float(arr.max()),
            "rt_min_ms": float(rt_arr.min()),
            "rt_median_ms": float(np.median(rt_arr)),
            "rt_mean_ms": float(rt_arr.mean()),
            "rt_max_ms": float(rt_arr.max()),
            "fps_equiv_from_mean": float(1000.0 / arr.mean()),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path, default=moana.DEFAULT_USD)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=804)
    parser.add_argument("--orbit-count", type=int, default=8)
    parser.add_argument("--orbit-radius", type=float, default=1800.0)
    parser.add_argument("--orbit-height", type=float, default=700.0)
    parser.add_argument("--orbit-fov", type=float, default=32.0)
    parser.add_argument("--orbit-start-degrees", type=float, default=25.0)
    parser.add_argument("--sheet-scale", type=float, default=0.375)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    authored = moana.parse_moana_cameras(args.usd)
    orbit_args = argparse.Namespace(
        orbit_count=args.orbit_count,
        orbit_center=None,
        orbit_radius=args.orbit_radius,
        orbit_radius_scale=2.25,
        orbit_height=args.orbit_height,
        orbit_fov=args.orbit_fov,
        orbit_start_degrees=args.orbit_start_degrees,
        orbit_near_clip=10.0,
        orbit_far_clip=1000000.0,
    )
    cameras = moana.make_orbit_cameras(authored, orbit_args)

    env_args = argparse.Namespace(rt_cull=False)
    env_metadata = moana.configure_render_environment(env_args)

    sys.path.insert(0, str(moana.REPO_ROOT / "python"))
    from nusd_renderer._bindings import NU_RENDER_RT, NuRenderer

    renderer = NuRenderer(
        width=args.width,
        height=args.height,
        enable_rt=True,
        enable_materials=True,
    )
    records: list[dict] = []
    quality_images: dict[str, Image.Image] = {}
    try:
        t0 = time.perf_counter()
        mesh_count = renderer.load_usd(str(args.usd))
        load_seconds = time.perf_counter() - t0
        scene_bounds = renderer.get_scene_bounds()

        for mode in MODES:
            previous = set_visibility_mode(mode)
            try:
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
                    render_seconds = time.perf_counter() - rt0
                    fetch0 = time.perf_counter()
                    pixels = renderer.fetch_pixels()
                    fetch_seconds = time.perf_counter() - fetch0
                    phase = renderer.get_phase_timings_ms()

                    image_path = frames_dir / f"{mode}_{camera.name}.png"
                    im = Image.fromarray(pixels[:, :, :3], mode="RGB")
                    im.save(image_path)
                    if mode == "quality":
                        quality_images[camera.name] = im.copy()
                        diff = {"mae": 0.0, "rms": 0.0, "max_abs": 0.0}
                    else:
                        diff = image_diff_metrics(quality_images[camera.name], im)

                    records.append({
                        "mode": mode,
                        "camera": camera.name,
                        "camera_spec": asdict(camera),
                        "image": str(image_path.relative_to(args.out_dir)),
                        "absolute_image": str(image_path),
                        "render_seconds": render_seconds,
                        "fetch_seconds": fetch_seconds,
                        "phase_timings_ms": phase,
                        "diff_vs_quality": diff,
                    })
            finally:
                restore_visibility_mode(previous)

        contact_sheet = args.out_dir / "visibility_mode_contact_sheet.png"
        save_mode_contact_sheet(records, contact_sheet, args.width, args.height,
                                args.sheet_scale, args.columns)

        metrics = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "usd": str(args.usd),
            "tile_size": [args.width, args.height],
            "environment": env_metadata,
            "modes": MODES,
            "renderer": {
                "mesh_count": int(mesh_count),
                "load_seconds": load_seconds,
                "gpu_memory_bytes": int(renderer.gpu_memory_used),
                "gpu_memory_gib": float(renderer.gpu_memory_used) / (1024.0 ** 3),
                "scene_bounds": scene_bounds,
                "cmd_cache_stats": renderer.cmd_cache_stats(),
            },
            "summary": summarize(records),
            "records": [
                {k: v for k, v in rec.items() if k != "absolute_image"}
                for rec in records
            ],
            "contact_sheet": str(contact_sheet.relative_to(args.out_dir)),
        }
        (args.out_dir / "visibility_mode_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

        for mode, summary in metrics["summary"].items():
            print(
                f"{mode}: mean {summary['frame_mean_ms']:.3f} ms "
                f"({summary['fps_equiv_from_mean']:.1f} FPS), "
                f"RT mean {summary['rt_mean_ms']:.3f} ms"
            )
        print(f"wrote {args.out_dir}")
    finally:
        renderer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
