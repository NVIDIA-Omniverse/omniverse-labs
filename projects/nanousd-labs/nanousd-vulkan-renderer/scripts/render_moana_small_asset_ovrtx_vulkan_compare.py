#!/usr/bin/env python3
"""Render small Moana assets with nanousdview OVRTX and Vulkan RT."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from render_moana_small_asset_storm_vulkan_compare import (
    DEFAULT_ASSETS,
    REPO_ROOT,
    AssetSpec,
    CameraRig,
    _fmt_tuple,
    _font,
    image_metrics,
    render_vulkan,
    write_wrapper,
)


WORKSPACE = REPO_ROOT.parent
NANOUSDVIEW_PYTHONPATH = WORKSPACE / "nanousdview" / "python"
OVRTX_PYTHON = Path(
    os.environ.get(
        "OVRTX_PYTHON",
        str(Path.home() / "workspace" / ".venv" / "bin" / "python"),
    )
)
HELPER = REPO_ROOT / "scripts" / "_nanousdview_ovrtx_render.py"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "docs" / "reports" / "moana_small_asset_ovrtx_vulkan_compare_2026-06-02"
)


@dataclass
class RenderRecord:
    label: str
    element: str
    wrapper: str
    ovrtx_png: str
    vulkan_png: str
    compare_png: str
    diff_png: str
    metrics: dict[str, Any]
    camera: CameraRig
    ovrtx_seconds: float
    vulkan_load_seconds: float
    vulkan_render_seconds: float
    vulkan_fetch_seconds: float
    vulkan_meshes: int
    vulkan_curve_segments: int
    vulkan_gpu_memory_gib: float
    vulkan_phase_timings_ms: dict[str, float] | None
    note: str


def _ovrtx_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(NANOUSDVIEW_PYTHONPATH)
    env.pop("PXR_PLUGINPATH_NAME", None)
    env.pop("USD_PLUGIN_PATH", None)
    env["DISPLAY"] = env.get("DISPLAY", ":1")
    env["XAUTHORITY"] = env.get("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env.setdefault("OVRTX_SKIP_USD_CHECK", "1")
    env.setdefault("NANOUSD_VIEW_STEP_TIMEOUT_NS", "60000000000")
    env.setdefault("NUVIEW_OVRTX_RENDER_MODE", "rt2")
    env.setdefault("NUVIEW_OVRTX_SPP", "1")
    env.setdefault("NUVIEW_OVRTX_DENOISE", "0")
    env.setdefault("NUVIEW_OVRTX_DEFAULT_LIGHTING", "1")

    venv_root = OVRTX_PYTHON.parent.parent
    ovrtx_bin = (
        venv_root
        / "lib"
        / "python3.12"
        / "site-packages"
        / "ovrtx"
        / "bin"
    )
    ovrtx_plugins = ovrtx_bin / "plugins"
    plugin_dir = (
        venv_root
        / "lib"
        / "python3.12"
        / "site-packages"
        / "ovrtx"
        / "bin"
        / "plugins"
        / "gpu.foundation"
    )
    glib = plugin_dir / "libglib-2.0.so.0"
    gobject = plugin_dir / "libgobject-2.0.so.0"
    if glib.exists() and gobject.exists():
        preload = f"{glib}:{gobject}"
        if env.get("LD_PRELOAD"):
            preload += ":" + env["LD_PRELOAD"]
        env["LD_PRELOAD"] = preload
    ld_paths = [str(ovrtx_plugins), str(ovrtx_bin)]
    inherited_ld = [
        entry
        for entry in env.get("LD_LIBRARY_PATH", "").split(":")
        if entry and "OpenUSD" not in entry and "openusd" not in entry.lower()
    ]
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths + inherited_ld)
    return env


def render_nanousdview_ovrtx(
    wrapper: Path,
    out_png: Path,
    camera: CameraRig,
    width: int,
    height: int,
    frames: int,
    warmup: int,
) -> float:
    out_ppm = out_png.with_suffix(".ppm")
    camera_payload = json.dumps(asdict(camera), separators=(",", ":"))
    cmd = [
        str(OVRTX_PYTHON),
        str(HELPER),
        "--usd",
        str(wrapper),
        "--out",
        str(out_ppm),
        "--width",
        str(width),
        "--height",
        str(height),
        "--camera-json",
        camera_payload,
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        env=_ovrtx_env(),
        text=True,
        capture_output=True,
        timeout=1800,
    )
    seconds = time.perf_counter() - t0
    log = proc.stdout + proc.stderr
    out_png.parent.mkdir(parents=True, exist_ok=True)
    (out_png.parent.parent / "logs" / f"{out_png.stem}.ovrtx.log").write_text(
        log,
        encoding="utf-8",
    )
    if proc.returncode != 0 or not out_ppm.exists():
        raise RuntimeError(
            f"nanousdview OVRTX render failed for {wrapper} rc={proc.returncode}\n"
            f"{log[-4000:]}"
        )
    Image.open(out_ppm).convert("RGB").save(out_png)
    out_ppm.unlink(missing_ok=True)
    return seconds


def make_compare(
    ovrtx_png: Path,
    vulkan_png: Path,
    diff_png: Path,
    out_png: Path,
    label: str,
    metrics: dict[str, Any],
) -> None:
    ovrtx = Image.open(ovrtx_png).convert("RGB")
    vulkan = Image.open(vulkan_png).convert("RGB")
    if vulkan.size != ovrtx.size:
        vulkan = vulkan.resize(ovrtx.size, Image.Resampling.LANCZOS)
    diff = Image.open(diff_png).convert("RGB")
    w, h = ovrtx.size
    bar_h = 48
    out = Image.new("RGB", (w * 3, h + bar_h), (18, 21, 23))
    out.paste(ovrtx, (0, bar_h))
    out.paste(vulkan, (w, bar_h))
    out.paste(diff, (w * 2, bar_h))
    draw = ImageDraw.Draw(out)
    font = _font(16)
    small = _font(13)
    draw.text((10, 7), f"{label}  OVRTX | Vulkan | abs diff x4", fill=(245, 247, 249), font=font)
    draw.text(
        (10, 27),
        f"RMS {metrics['rms']:.1f}  MAE {metrics['mae']:.1f}  silhouette IoU {metrics['silhouette_iou_luma_gt_8']:.3f}",
        fill=(190, 198, 205),
        font=small,
    )
    out.save(out_png)


def make_contact_sheet(records: list[RenderRecord], out_png: Path) -> None:
    compares = [Image.open(REPO_ROOT / r.compare_png).convert("RGB") for r in records]
    tile_w = 720
    resized = []
    for img in compares:
        scale = tile_w / img.size[0]
        resized.append(img.resize((tile_w, int(img.size[1] * scale)), Image.Resampling.LANCZOS))
    gap = 14
    sheet_w = tile_w + gap * 2
    sheet_h = gap + sum(img.size[1] + gap for img in resized)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 17, 20))
    y = gap
    for img in resized:
        sheet.paste(img, (gap, y))
        y += img.size[1] + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def render_asset(
    asset: AssetSpec,
    out_dir: Path,
    width: int,
    height: int,
    ovrtx_frames: int,
    ovrtx_warmup: int,
) -> RenderRecord:
    frames = out_dir / "frames"
    logs = out_dir / "logs"
    frames.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    wrapper, camera = write_wrapper(asset, out_dir)
    ovrtx_png = frames / f"{asset.label}_ovrtx.png"
    vulkan_png = frames / f"{asset.label}_vulkan.png"
    diff_png = frames / f"{asset.label}_diff_x4.png"
    compare_png = frames / f"{asset.label}_compare.png"

    ovrtx_seconds = render_nanousdview_ovrtx(
        wrapper, ovrtx_png, camera, width, height, ovrtx_frames, ovrtx_warmup
    )
    vulkan_info = render_vulkan(wrapper, vulkan_png, camera, width, height)
    metrics = image_metrics(ovrtx_png, vulkan_png, diff_png)
    if "mean_storm_rgb" in metrics:
        metrics["mean_ovrtx_rgb"] = list(metrics["mean_storm_rgb"])
    if "silhouette_iou_luma_gt_8" in metrics:
        metrics["silhouette_iou_background_delta"] = metrics[
            "silhouette_iou_luma_gt_8"
        ]
    make_compare(ovrtx_png, vulkan_png, diff_png, compare_png, asset.label, metrics)

    return RenderRecord(
        label=asset.label,
        element=asset.element,
        wrapper=str(wrapper.relative_to(REPO_ROOT)),
        ovrtx_png=str(ovrtx_png.relative_to(REPO_ROOT)),
        vulkan_png=str(vulkan_png.relative_to(REPO_ROOT)),
        compare_png=str(compare_png.relative_to(REPO_ROOT)),
        diff_png=str(diff_png.relative_to(REPO_ROOT)),
        metrics=metrics,
        camera=camera,
        ovrtx_seconds=ovrtx_seconds,
        vulkan_load_seconds=float(vulkan_info["load_seconds"]),
        vulkan_render_seconds=float(vulkan_info["render_seconds"]),
        vulkan_fetch_seconds=float(vulkan_info["fetch_seconds"]),
        vulkan_meshes=int(vulkan_info["meshes"]),
        vulkan_curve_segments=int(vulkan_info["curve_segments"]),
        vulkan_gpu_memory_gib=float(vulkan_info["gpu_memory_gib"]),
        vulkan_phase_timings_ms=vulkan_info["phase_timings_ms"],
        note=asset.note,
    )


def write_report(out_dir: Path, records: list[RenderRecord], contact_sheet: Path, args: argparse.Namespace) -> None:
    def report_rel(repo_relative: str) -> str:
        return os.path.relpath(REPO_ROOT / repo_relative, out_dir)

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference": "nanousdview OVRTX",
        "ovrtx_python": str(OVRTX_PYTHON),
        "assets": [asdict(r) for r in records],
        "contact_sheet": str(contact_sheet.relative_to(REPO_ROOT)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    curve_subsegs = os.environ.get("NUSD_CURVE_SUBSEGS", "8")
    lines = [
        "# Moana Small Asset OVRTX vs Vulkan Visual Compare - 2026-06-02",
        "",
        "## Summary",
        "",
        "Rendered small Moana element wrappers with the NVIDIA OVRTX backend selected through `nanousdview._backend`, then rendered the same wrappers with local nanousd Vulkan RT.",
        "",
        f"![OVRTX/Vulkan contact sheet]({contact_sheet.name})",
        "",
        "## Notes",
        "",
        "- Reference path: `nanousdview` OVRTX backend, using the official OVRTX Python environment.",
        "- Vulkan path: local `NuRenderer`, RT mode, real materials enabled, native curves enabled.",
        "- OVRTX fallback camera+dome lighting is enabled by default (`NUVIEW_OVRTX_DEFAULT_LIGHTING=1`) so small wrapped assets render visibly even when OVRTX rejects Moana multi-node material graphs.",
        "- The wrapper camera/light rig is shared across both renders; the OVRTX Qt `--screenshot` loop was too slow on Pandanus, so this report uses nanousdview's renderer helper directly.",
        "- Silhouette IoU uses a background-delta mask because both paths render a gray default background in this report.",
        f"- Vulkan cubic curve tessellation: `NUSD_CURVE_SUBSEGS={curve_subsegs}`.",
        f"- OVRTX frames: `{args.ovrtx_frames}`, warmup: `{args.ovrtx_warmup}`.",
        "",
        "## Metrics",
        "",
        "| Asset | Note | Vulkan meshes | Vulkan curve segments | OVRTX s | Vulkan load s | Vulkan render s | GPU GiB | RMS | MAE | Silhouette IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| {r.label} | {r.note} | {r.vulkan_meshes} | {r.vulkan_curve_segments:,} | "
            f"{r.ovrtx_seconds:.2f} | {r.vulkan_load_seconds:.2f} | {r.vulkan_render_seconds:.2f} | "
            f"{r.vulkan_gpu_memory_gib:.2f} | {r.metrics['rms']:.1f} | {r.metrics['mae']:.1f} | "
            f"{r.metrics['silhouette_iou_luma_gt_8']:.3f} |"
        )
    lines += ["", "## Per-Asset Comparisons", ""]
    for r in records:
        lines += [
            f"### {r.label}",
            "",
            f"![{r.label} compare]({report_rel(r.compare_png)})",
            "",
            f"- Wrapper: [{Path(r.wrapper).name}]({report_rel(r.wrapper)})",
            f"- OVRTX: [{Path(r.ovrtx_png).name}]({report_rel(r.ovrtx_png)})",
            f"- Vulkan: [{Path(r.vulkan_png).name}]({report_rel(r.vulkan_png)})",
            f"- Diff x4: [{Path(r.diff_png).name}]({report_rel(r.diff_png)})",
            f"- Camera eye {_fmt_tuple(r.camera.eye)}, target {_fmt_tuple(r.camera.target)}, FOV {r.camera.fov_degrees:.1f} deg",
            "",
        ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--asset", action="append", help="Optional element name, e.g. isPandanusA. Can be repeated.")
    parser.add_argument("--ovrtx-frames", type=int, default=1)
    parser.add_argument("--ovrtx-warmup", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.asset:
        specs = tuple(AssetSpec(a.replace("is", "", 1) if a.startswith("is") else a, a, "requested asset") for a in args.asset)
    else:
        specs = DEFAULT_ASSETS

    records: list[RenderRecord] = []
    for spec in specs:
        print(f"=== {spec.label} ({spec.usd_path}) ===", flush=True)
        records.append(render_asset(spec, out_dir, args.width, args.height, args.ovrtx_frames, args.ovrtx_warmup))

    contact = out_dir / "ovrtx_vulkan_small_asset_contact_sheet.png"
    make_contact_sheet(records, contact)
    write_report(out_dir, records, contact, args)
    print(f"Wrote {out_dir / 'README.md'}")
    print(f"Wrote {contact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
