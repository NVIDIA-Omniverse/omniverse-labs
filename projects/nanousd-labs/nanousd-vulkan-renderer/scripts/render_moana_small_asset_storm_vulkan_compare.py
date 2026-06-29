#!/usr/bin/env python3
"""Render small Moana assets with Storm and nanousd Vulkan, then report.

The script generates one wrapper USD per asset with a shared camera/light rig,
renders the wrapper with OpenUSD Storm via usdrecord, renders the same wrapper
through the local Vulkan RT bindings, and writes side-by-side comparison images,
metrics JSON, and a markdown report.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parents[1]
MOANA_ROOT = Path("$HOME/moana-island-scene-usd/island/usd")
OPENUSD_ROOT = Path("$HOME/OpenUSD_install")
USDRECORD = OPENUSD_ROOT / "bin" / "usdrecord"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "reports" / "moana_small_asset_storm_vulkan_compare_2026-06-01"


@dataclass(frozen=True)
class AssetSpec:
    label: str
    element: str
    note: str

    @property
    def prim_name(self) -> str:
        return self.element

    @property
    def usd_path(self) -> Path:
        return MOANA_ROOT / "elements" / self.element / "element.usda"


@dataclass
class CameraRig:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_degrees: float
    near_clip: float
    far_clip: float
    horizontal_aperture: float
    vertical_aperture: float
    focal_length: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


@dataclass
class RenderRecord:
    label: str
    element: str
    wrapper: str
    storm_png: str
    vulkan_png: str
    compare_png: str
    diff_png: str
    metrics: dict[str, Any]
    camera: CameraRig
    storm_seconds: float
    vulkan_load_seconds: float
    vulkan_render_seconds: float
    vulkan_fetch_seconds: float
    vulkan_meshes: int
    vulkan_curve_segments: int
    vulkan_gpu_memory_gib: float
    vulkan_phase_timings_ms: dict[str, float] | None
    note: str


DEFAULT_ASSETS = (
    AssetSpec("IronwoodA1", "isIronwoodA1", "tree with native instance proxy B-spline branch/needle curves"),
    AssetSpec("IronwoodB", "isIronwoodB", "alternate ironwood tree variant with dense instancing"),
    AssetSpec("PandanusA", "isPandanusA", "pandanus bush with repeated curve-bearing instances"),
    AssetSpec("PalmRig", "isPalmRig", "palm rig asset with mesh fronds and cubic curve detail"),
)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _normalize(v: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1.0e-8:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _look_at_matrix_rows(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
) -> list[list[float]]:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_hint = _normalize(np.asarray(up, dtype=np.float64), (0.0, 1.0, 0.0))
    forward = _normalize(target_v - eye_v, (0.0, 0.0, -1.0))
    back = -forward
    right = _normalize(np.cross(up_hint, back), (1.0, 0.0, 0.0))
    true_up = _normalize(np.cross(back, right), (0.0, 1.0, 0.0))
    return [
        [float(right[0]), float(right[1]), float(right[2]), 0.0],
        [float(true_up[0]), float(true_up[1]), float(true_up[2]), 0.0],
        [float(back[0]), float(back[1]), float(back[2]), 0.0],
        [float(eye_v[0]), float(eye_v[1]), float(eye_v[2]), 1.0],
    ]


def _compute_bounds(stage: Usd.Stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Missing prim {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if hasattr(bbox, "GetRange"):
        bbox = bbox.GetRange()
    if hasattr(bbox, "GetBox"):
        bbox = bbox.GetBox()
    mn = np.asarray(bbox.GetMin(), dtype=np.float64)
    mx = np.asarray(bbox.GetMax(), dtype=np.float64)
    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)) or np.any(mx <= mn):
        raise RuntimeError(f"Invalid bounds for {prim_path}: {mn} - {mx}")
    return mn, mx


def _make_camera(bounds_min: np.ndarray, bounds_max: np.ndarray) -> CameraRig:
    center = (bounds_min + bounds_max) * 0.5
    extent = bounds_max - bounds_min
    radius = float(np.linalg.norm(extent) * 0.5)
    radius = max(radius, 1.0)
    fov = 38.0
    dist = radius / max(math.sin(math.radians(fov) * 0.5), 1.0e-4) * 1.22
    direction = _normalize(np.asarray((0.72, 0.34, 1.0), dtype=np.float64), (0.0, 0.0, 1.0))
    eye = center + direction * dist
    # Aim slightly high for trees and palms so the crown does not sit on the
    # top edge, but keep the target near center for bushes.
    target = center.copy()
    target[1] += extent[1] * 0.08
    near = max(0.1, dist - radius * 2.0)
    far = max(near + 10.0, dist + radius * 2.8)
    focal = 50.0
    vap = 2.0 * focal * math.tan(math.radians(fov) * 0.5)
    return CameraRig(
        eye=tuple(float(x) for x in eye),
        target=tuple(float(x) for x in target),
        up=(0.0, 1.0, 0.0),
        fov_degrees=fov,
        near_clip=float(near),
        far_clip=float(far),
        horizontal_aperture=float(vap),
        vertical_aperture=float(vap),
        focal_length=focal,
        bounds_min=tuple(float(x) for x in bounds_min),
        bounds_max=tuple(float(x) for x in bounds_max),
    )


def _fmt_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{v:.12g}" for v in values) + ")"


def write_wrapper(asset: AssetSpec, out_dir: Path) -> tuple[Path, CameraRig]:
    wrappers_dir = out_dir / "wrappers"
    wrappers_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrappers_dir / f"{asset.label}.usda"
    root_path = f"/World/Asset"

    prelude = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.0254
    upAxis = "Y"
)

def Xform "World"
{{
    def Xform "Asset" (
        kind = "group"
        prepend references = @{asset.usd_path}@</{asset.prim_name}>
    )
    {{
    }}
}}
"""
    wrapper.write_text(prelude, encoding="utf-8")
    stage = Usd.Stage.Open(str(wrapper))
    if not stage:
        raise RuntimeError(f"Could not open generated wrapper {wrapper}")
    bounds_min, bounds_max = _compute_bounds(stage, root_path)
    camera = _make_camera(bounds_min, bounds_max)
    rows = _look_at_matrix_rows(camera.eye, camera.target, camera.up)
    matrix_text = "( " + ", ".join(
        "(" + ", ".join(f"{v:.12g}" for v in row) + ")" for row in rows
    ) + " )"

    lighting = f"""
def DistantLight "KeyLight"
{{
    color3f inputs:color = (1, 0.94, 0.82)
    float inputs:intensity = 2.0
    float inputs:angle = 1.0
    float3 xformOp:rotateXYZ = (-42, 32, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}}

def Camera "Camera"
{{
    token projection = "perspective"
    float focalLength = {camera.focal_length:.12g}
    float horizontalAperture = {camera.horizontal_aperture:.12g}
    float verticalAperture = {camera.vertical_aperture:.12g}
    float2 clippingRange = ({camera.near_clip:.12g}, {camera.far_clip:.12g})
    matrix4d xformOp:transform = {matrix_text}
    uniform token[] xformOpOrder = ["xformOp:transform"]
}}
"""
    wrapper.write_text(prelude + lighting, encoding="utf-8")
    return wrapper, camera


def _openusd_env() -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [
        str(OPENUSD_ROOT / "lib" / "python"),
        "$HOME/.venvs/ovrtx/lib/python3.12/site-packages",
        "$HOME/.venv/lib/python3.12/site-packages",
    ]
    lib_path = str(OPENUSD_ROOT / "lib")
    env["PYTHONPATH"] = ":".join(python_paths) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LD_LIBRARY_PATH"] = lib_path + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    return env


def render_storm(wrapper: Path, out_png: Path, width: int) -> tuple[float, str]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(USDRECORD),
        "--renderer",
        "Storm",
        "--camera",
        "Camera",
        "--complexity",
        "high",
        "--colorCorrectionMode",
        "sRGB",
        "--imageWidth",
        str(width),
        str(wrapper),
        str(out_png),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=_openusd_env(), text=True, capture_output=True, timeout=600)
    seconds = time.perf_counter() - t0
    log = proc.stdout + proc.stderr
    if proc.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"Storm render failed for {wrapper} rc={proc.returncode}\n{log[-4000:]}")
    return seconds, log


def _configure_vulkan_env() -> dict[str, str]:
    requested_curve_subsegs = os.environ.get("NUSD_CURVE_SUBSEGS", "8")
    defaults = {
        "NUSD_ENABLE_MATERIALS": "1",
        "NUSD_ENABLE_PTEX_MATERIALS": "1",
        "NUSD_CLAY_VIZ": "0",
        "NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL": "1",
        "NUSD_RENDER_PI_BATCHES": "1",
        "NUSD_NATIVE_ARC_CHASE_AFTER_DIRECT": "1",
        "NUSD_NATIVE_CURVES": "all",
        "NUSD_CURVE_SUBSEGS": requested_curve_subsegs,
        "NUSD_RT_CULL": "0",
        "NUSD_NO_CULL_ALL_GEOMETRY": "0",
        "NUSD_ALL_GEOMETRY_NO_CULL": "0",
        "DISPLAY": ":1",
        "XAUTHORITY": "/run/user/1000/gdm/Xauthority",
        "__NV_PRIME_RENDER_OFFLOAD": "1",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    }
    prior = {}
    for key, value in defaults.items():
        prior[key] = os.environ.get(key)
        os.environ[key] = value
    return prior


def _restore_env(prior: dict[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def render_vulkan(wrapper: Path, out_png: Path, camera: CameraRig, width: int, height: int) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from nusd_renderer._bindings import NU_RENDER_RT, NuRenderer

    prior = _configure_vulkan_env()
    renderer = NuRenderer(width=width, height=height, enable_rt=True, enable_materials=True)
    try:
        t0 = time.perf_counter()
        meshes = int(renderer.load_usd(str(wrapper)))
        load_seconds = time.perf_counter() - t0
        renderer.set_camera_explicit(
            camera.eye,
            camera.target,
            camera.up,
            camera.fov_degrees,
            camera.near_clip,
            camera.far_clip,
        )
        t1 = time.perf_counter()
        renderer.build_accel()
        renderer.render(NU_RENDER_RT)
        renderer.render(NU_RENDER_RT)
        render_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        rgba = renderer.fetch_pixels()
        fetch_seconds = time.perf_counter() - t2
        Image.fromarray(rgba[:, :, :3], mode="RGB").save(out_png)
        curve_segments = 0
        fn = getattr(renderer._lib, "nu_get_curve_segment_count", None)
        if fn is not None:
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            curve_segments = int(fn(renderer._handle))
        return {
            "meshes": meshes,
            "curve_segments": curve_segments,
            "load_seconds": load_seconds,
            "render_seconds": render_seconds,
            "fetch_seconds": fetch_seconds,
            "gpu_memory_gib": float(renderer.gpu_memory_used) / (1024.0 ** 3),
            "phase_timings_ms": renderer.get_phase_timings_ms(),
        }
    finally:
        renderer.close()
        _restore_env(prior)


def image_metrics(storm_png: Path, vulkan_png: Path, diff_png: Path) -> dict[str, Any]:
    a_img = Image.open(storm_png).convert("RGB")
    b_img = Image.open(vulkan_png).convert("RGB")
    if b_img.size != a_img.size:
        b_img = b_img.resize(a_img.size, Image.Resampling.LANCZOS)
    a = np.asarray(a_img).astype(np.float32)
    b = np.asarray(b_img).astype(np.float32)
    d = a - b
    absd = np.abs(d)
    rms = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(absd))
    max_abs = float(np.max(absd))
    def foreground_mask(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        n = max(6, min(h, w) // 32)
        corners = np.concatenate(
            [
                img[:n, :n].reshape(-1, 3),
                img[:n, w - n :].reshape(-1, 3),
                img[h - n :, :n].reshape(-1, 3),
                img[h - n :, w - n :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(corners, axis=0)
        return np.linalg.norm(img - bg.reshape(1, 1, 3), axis=2) > 12.0

    silhouette_a = foreground_mask(a)
    silhouette_b = foreground_mask(b)
    inter = int(np.count_nonzero(silhouette_a & silhouette_b))
    union = int(np.count_nonzero(silhouette_a | silhouette_b))
    silhouette_iou = float(inter / union) if union else 1.0
    diff = np.clip(absd * 4.0, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(diff_png)
    return {
        "rms": rms,
        "mae": mae,
        "max_abs": max_abs,
        "mean_storm_rgb": [float(a[..., c].mean()) for c in range(3)],
        "mean_vulkan_rgb": [float(b[..., c].mean()) for c in range(3)],
        "silhouette_iou_luma_gt_8": silhouette_iou,
    }


def make_compare(storm_png: Path, vulkan_png: Path, diff_png: Path, out_png: Path, label: str, metrics: dict[str, Any]) -> None:
    storm = Image.open(storm_png).convert("RGB")
    vulkan = Image.open(vulkan_png).convert("RGB")
    if vulkan.size != storm.size:
        vulkan = vulkan.resize(storm.size, Image.Resampling.LANCZOS)
    diff = Image.open(diff_png).convert("RGB")
    w, h = storm.size
    bar_h = 48
    out = Image.new("RGB", (w * 3, h + bar_h), (18, 21, 23))
    out.paste(storm, (0, bar_h))
    out.paste(vulkan, (w, bar_h))
    out.paste(diff, (w * 2, bar_h))
    draw = ImageDraw.Draw(out)
    font = _font(16)
    small = _font(13)
    draw.text((10, 7), f"{label}  Storm | Vulkan | abs diff x4", fill=(245, 247, 249), font=font)
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
    cols = 1
    sheet_w = tile_w + gap * 2
    sheet_h = gap + sum(img.size[1] + gap for img in resized)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 17, 20))
    y = gap
    for img in resized:
        sheet.paste(img, (gap, y))
        y += img.size[1] + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def render_asset(asset: AssetSpec, out_dir: Path, width: int, height: int) -> RenderRecord:
    frames = out_dir / "frames"
    logs = out_dir / "logs"
    frames.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    wrapper, camera = write_wrapper(asset, out_dir)
    storm_png = frames / f"{asset.label}_storm.png"
    vulkan_png = frames / f"{asset.label}_vulkan.png"
    diff_png = frames / f"{asset.label}_diff_x4.png"
    compare_png = frames / f"{asset.label}_compare.png"

    storm_seconds, storm_log = render_storm(wrapper, storm_png, width)
    (logs / f"{asset.label}_storm.log").write_text(storm_log, encoding="utf-8")
    vulkan_info = render_vulkan(wrapper, vulkan_png, camera, width, height)
    metrics = image_metrics(storm_png, vulkan_png, diff_png)
    make_compare(storm_png, vulkan_png, diff_png, compare_png, asset.label, metrics)

    return RenderRecord(
        label=asset.label,
        element=asset.element,
        wrapper=str(wrapper.relative_to(REPO_ROOT)),
        storm_png=str(storm_png.relative_to(REPO_ROOT)),
        vulkan_png=str(vulkan_png.relative_to(REPO_ROOT)),
        compare_png=str(compare_png.relative_to(REPO_ROOT)),
        diff_png=str(diff_png.relative_to(REPO_ROOT)),
        metrics=metrics,
        camera=camera,
        storm_seconds=storm_seconds,
        vulkan_load_seconds=float(vulkan_info["load_seconds"]),
        vulkan_render_seconds=float(vulkan_info["render_seconds"]),
        vulkan_fetch_seconds=float(vulkan_info["fetch_seconds"]),
        vulkan_meshes=int(vulkan_info["meshes"]),
        vulkan_curve_segments=int(vulkan_info["curve_segments"]),
        vulkan_gpu_memory_gib=float(vulkan_info["gpu_memory_gib"]),
        vulkan_phase_timings_ms=vulkan_info["phase_timings_ms"],
        note=asset.note,
    )


def write_report(out_dir: Path, records: list[RenderRecord], contact_sheet: Path) -> None:
    def report_rel(repo_relative: str) -> str:
        return os.path.relpath(REPO_ROOT / repo_relative, out_dir)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "assets": [asdict(r) for r in records],
        "contact_sheet": str(contact_sheet.relative_to(REPO_ROOT)),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    curve_subsegs = os.environ.get("NUSD_CURVE_SUBSEGS", "8")
    lines = [
        "# Moana Small Asset Storm vs Vulkan Visual Compare - 2026-06-01",
        "",
        "## Summary",
        "",
        f"Rendered {len(records)} small Moana element wrapper(s) with OpenUSD Storm and nanousd Vulkan RT using generated wrapper USDs with identical cameras and lights.",
        "",
        f"![Storm/Vulkan contact sheet]({contact_sheet.name})",
        "",
        "## Verdict",
        "",
        "Short answer: no, Vulkan does not visually match Storm yet.",
        "",
        "The focused renders show that Vulkan now carries the Storm-visible tree and plant geometry into the frame: Ironwood, Pandanus, and PalmRig curve detail appears instead of being dropped. The remaining mismatch is visual rather than gross geometry absence. The largest differences are:",
        "",
        "- Storm renders the small asset frames against a black background with its default camera light; Vulkan's RT miss/background is grey in this path.",
        "- Plant curve colors and lighting still differ; Storm's black background and default-light response make RMS/MAE mostly diagnostic.",
        "- PalmRig remains one of the weaker silhouette matches when included, so it still needs follow-up curve/shading debugging.",
        f"- Vulkan tessellates cubic curves into `{curve_subsegs}` fixed subsegments per patch for this run; Storm uses screen-space adaptive tessellation.",
        "",
        "## Metrics",
        "",
        "| Asset | Note | Vulkan meshes | Vulkan curve segments | Storm s | Vulkan load s | Vulkan render s | GPU GiB | RMS | MAE | Silhouette IoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| {r.label} | {r.note} | {r.vulkan_meshes} | {r.vulkan_curve_segments:,} | "
            f"{r.storm_seconds:.2f} | {r.vulkan_load_seconds:.2f} | {r.vulkan_render_seconds:.2f} | "
            f"{r.vulkan_gpu_memory_gib:.2f} | {r.metrics['rms']:.1f} | {r.metrics['mae']:.1f} | "
            f"{r.metrics['silhouette_iou_luma_gt_8']:.3f} |"
        )
    lines += [
        "",
        "## Per-Asset Comparisons",
        "",
    ]
    for r in records:
        lines += [
            f"### {r.label}",
            "",
            f"![{r.label} compare]({report_rel(r.compare_png)})",
            "",
            f"- Wrapper: [{Path(r.wrapper).name}]({report_rel(r.wrapper)})",
            f"- Storm: [{Path(r.storm_png).name}]({report_rel(r.storm_png)})",
            f"- Vulkan: [{Path(r.vulkan_png).name}]({report_rel(r.vulkan_png)})",
            f"- Diff x4: [{Path(r.diff_png).name}]({report_rel(r.diff_png)})",
            f"- Camera eye {_fmt_tuple(r.camera.eye)}, target {_fmt_tuple(r.camera.target)}, FOV {r.camera.fov_degrees:.1f} deg",
            "",
        ]
    lines += [
        "## Render Setup",
        "",
        "- Storm command path: OpenUSD `usdrecord --renderer Storm --complexity high` with Storm's default camera light enabled.",
        f"- Vulkan path: local `NuRenderer`, RT mode, real materials enabled, native curves enabled, `NUSD_CURVE_SUBSEGS={curve_subsegs}`.",
        "- The generated wrappers include one referenced element, one `DistantLight`, and one square-aspect camera.",
        "- Metrics are image-space diagnostics, not strict acceptance thresholds; the key geometry check is that the Storm-visible tree/plant curves are visible in Vulkan.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--asset", action="append", help="Optional element name, e.g. isIronwoodA1. Can be repeated.")
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
        records.append(render_asset(spec, out_dir, args.width, args.height))

    contact = out_dir / "storm_vulkan_small_asset_contact_sheet.png"
    make_contact_sheet(records, contact)
    write_report(out_dir, records, contact)
    print(f"Wrote {out_dir / 'README.md'}")
    print(f"Wrote {contact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
