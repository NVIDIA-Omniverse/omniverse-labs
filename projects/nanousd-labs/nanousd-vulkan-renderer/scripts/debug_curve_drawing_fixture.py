#!/usr/bin/env python3
"""Render a controlled BasisCurves fixture with Storm and nanousd Vulkan.

The fixture isolates curve drawing differences that are hard to see in Moana:
width-only tubes, authored-normal oriented ribbons, edge-on ribbons, and cubic
B-spline ribbons with varying width/displayColor.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
for _path in (
    "$HOME/OpenUSD_install/lib/python",
    "$HOME/OpenUSD/lib/python",
    "$HOME/.venvs/ovrtx/lib/python3.12/site-packages",
    "$HOME/.venv/lib/python3.12/site-packages",
):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ["LD_LIBRARY_PATH"] = (
    "$HOME/OpenUSD_install/lib:$HOME/OpenUSD/lib"
    + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
)

from render_moana_small_asset_storm_vulkan_compare import (  # noqa: E402
    CameraRig,
    image_metrics,
    make_compare,
    render_storm,
    render_vulkan,
)


DEFAULT_OUT_DIR = (
    REPO_ROOT / "docs" / "reports" / "curve_drawing_fixture_debug_2026-06-02"
)


@dataclass
class RegionMetric:
    name: str
    bbox: tuple[int, int, int, int]
    storm_mask_pixels: int
    vulkan_mask_pixels: int
    silhouette_iou: float
    storm_mean_rgb: tuple[float, float, float]
    vulkan_mean_rgb: tuple[float, float, float]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _look_at_matrix_rows(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
) -> list[list[float]]:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)
    up_v /= max(float(np.linalg.norm(up_v)), 1.0e-8)
    forward = target_v - eye_v
    forward /= max(float(np.linalg.norm(forward)), 1.0e-8)
    back = -forward
    right = np.cross(up_v, back)
    right /= max(float(np.linalg.norm(right)), 1.0e-8)
    true_up = np.cross(back, right)
    return [
        [float(right[0]), float(right[1]), float(right[2]), 0.0],
        [float(true_up[0]), float(true_up[1]), float(true_up[2]), 0.0],
        [float(back[0]), float(back[1]), float(back[2]), 0.0],
        [float(eye_v[0]), float(eye_v[1]), float(eye_v[2]), 1.0],
    ]


def _matrix_text(rows: list[list[float]]) -> str:
    return "( " + ", ".join(
        "(" + ", ".join(f"{v:.12g}" for v in row) + ")" for row in rows
    ) + " )"


def write_fixture(out_dir: Path) -> tuple[Path, CameraRig]:
    out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = out_dir / "curve_drawing_fixture.usda"

    camera = CameraRig(
        eye=(0.0, 0.0, 12.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_degrees=38.0,
        near_clip=0.1,
        far_clip=60.0,
        horizontal_aperture=34.0,
        vertical_aperture=34.0,
        focal_length=50.0,
        bounds_min=(-4.6, -3.2, -0.4),
        bounds_max=(4.6, 3.2, 0.4),
    )
    matrix = _matrix_text(_look_at_matrix_rows(camera.eye, camera.target, camera.up))

    usd_path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "World"
{{
    def DistantLight "KeyLight"
    {{
        color3f inputs:color = (1, 1, 1)
        float inputs:intensity = 2.5
        float inputs:angle = 0.6
        float3 xformOp:rotateXYZ = (-25, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def Camera "Camera"
    {{
        token projection = "perspective"
        float focalLength = {camera.focal_length:.12g}
        float horizontalAperture = {camera.horizontal_aperture:.12g}
        float verticalAperture = {camera.vertical_aperture:.12g}
        float2 clippingRange = ({camera.near_clip:.12g}, {camera.far_clip:.12g})
        matrix4d xformOp:transform = {matrix}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def Scope "Curves"
    {{
        def BasisCurves "TubeWidthOnly"
        {{
            uniform token type = "linear"
            int[] curveVertexCounts = [2]
            point3f[] points = [(-4.0, 2.1, 0), (4.0, 2.1, 0)]
            float[] widths = [0.42, 0.42] (interpolation = "vertex")
            color3f[] primvars:displayColor = [(0.95, 0.12, 0.08)] (interpolation = "constant")
        }}

        def BasisCurves "RibbonFacing"
        {{
            uniform token type = "linear"
            int[] curveVertexCounts = [2]
            point3f[] points = [(-4.0, 0.7, 0), (4.0, 0.7, 0)]
            float[] widths = [0.42, 0.42] (interpolation = "vertex")
            normal3f[] normals = [(0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
            color3f[] primvars:displayColor = [(0.1, 0.75, 0.18)] (interpolation = "constant")
        }}

        def BasisCurves "RibbonEdgeOn"
        {{
            uniform token type = "linear"
            int[] curveVertexCounts = [2]
            point3f[] points = [(-4.0, -0.7, 0), (4.0, -0.7, 0)]
            float[] widths = [0.42, 0.42] (interpolation = "vertex")
            normal3f[] normals = [(0, 1, 0), (0, 1, 0)] (interpolation = "vertex")
            color3f[] primvars:displayColor = [(0.1, 0.25, 0.95)] (interpolation = "constant")
        }}

        def BasisCurves "CubicBsplineRibbon"
        {{
            uniform token type = "cubic"
            uniform token basis = "bspline"
            int[] curveVertexCounts = [7]
            point3f[] points = [
                (-4.0, -2.45, 0), (-2.8, -1.8, 0), (-1.4, -2.8, 0),
                (0.0, -1.65, 0), (1.4, -2.8, 0), (2.8, -1.8, 0),
                (4.0, -2.45, 0)
            ]
            float[] widths = [0.15, 0.35, 0.65, 0.25, 0.65, 0.35, 0.15] (interpolation = "vertex")
            normal3f[] normals = [
                (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
                (0, 0, 1), (0, 0, 1), (0, 0, 1)
            ] (interpolation = "vertex")
            color3f[] primvars:displayColor = [
                (1.0, 0.95, 0.1), (0.7, 0.9, 0.1), (0.2, 0.8, 0.15),
                (0.1, 0.7, 0.45), (0.1, 0.45, 0.9), (0.35, 0.2, 0.9),
                (0.9, 0.1, 0.7)
            ] (interpolation = "vertex")
        }}
    }}
}}
""",
        encoding="utf-8",
    )
    return usd_path, camera


def _mask(img: np.ndarray) -> np.ndarray:
    corners = np.concatenate(
        [
            img[:10, :10].reshape(-1, 3),
            img[:10, -10:].reshape(-1, 3),
            img[-10:, :10].reshape(-1, 3),
            img[-10:, -10:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    return np.linalg.norm(img - bg.reshape(1, 1, 3), axis=2) > 10.0


def _region_metrics(storm_png: Path, vulkan_png: Path) -> list[RegionMetric]:
    a = np.asarray(Image.open(storm_png).convert("RGB")).astype(np.float32)
    b = np.asarray(Image.open(vulkan_png).convert("RGB")).astype(np.float32)
    h, w = a.shape[:2]
    bands = [
        ("TubeWidthOnly", (0, int(h * 0.17), w, int(h * 0.32))),
        ("RibbonFacing", (0, int(h * 0.34), w, int(h * 0.49))),
        ("RibbonEdgeOn", (0, int(h * 0.53), w, int(h * 0.64))),
        ("CubicBsplineRibbon", (0, int(h * 0.68), w, int(h * 0.91))),
    ]
    out: list[RegionMetric] = []
    ma = _mask(a)
    mb = _mask(b)
    for name, bbox in bands:
        x0, y0, x1, y1 = bbox
        ra = ma[y0:y1, x0:x1]
        rb = mb[y0:y1, x0:x1]
        inter = int(np.count_nonzero(ra & rb))
        union = int(np.count_nonzero(ra | rb))
        iou = float(inter / union) if union else 1.0
        pa = a[y0:y1, x0:x1][ra]
        pb = b[y0:y1, x0:x1][rb]
        mean_a = tuple(float(x) for x in (pa.mean(axis=0) if pa.size else np.zeros(3)))
        mean_b = tuple(float(x) for x in (pb.mean(axis=0) if pb.size else np.zeros(3)))
        out.append(
            RegionMetric(
                name=name,
                bbox=bbox,
                storm_mask_pixels=int(np.count_nonzero(ra)),
                vulkan_mask_pixels=int(np.count_nonzero(rb)),
                silhouette_iou=iou,
                storm_mean_rgb=mean_a,
                vulkan_mean_rgb=mean_b,
            )
        )
    return out


def _annotated_compare(compare_png: Path, region_metrics: list[RegionMetric]) -> None:
    img = Image.open(compare_png).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(13)
    x = 10
    y = img.height - 86
    draw.rectangle((0, y - 8, img.width, img.height), fill=(15, 18, 21))
    for metric in region_metrics:
        draw.text(
            (x, y),
            f"{metric.name}: IoU {metric.silhouette_iou:.3f} "
            f"px {metric.storm_mask_pixels}/{metric.vulkan_mask_pixels}",
            fill=(225, 229, 234),
            font=font,
        )
        y += 18
    img.save(compare_png)


def write_report(out_dir: Path, metrics: dict[str, Any], regions: list[RegionMetric]) -> None:
    payload = {
        "overall": metrics,
        "regions": [asdict(r) for r in regions],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Curve Drawing Fixture Debug - 2026-06-02",
        "",
        "Controlled Storm vs Vulkan render for BasisCurves drawing.",
        "",
        "![compare](frames/curve_fixture_compare.png)",
        "",
        "## Overall",
        "",
        f"- RMS: {metrics['rms']:.3f}",
        f"- MAE: {metrics['mae']:.3f}",
        f"- silhouette IoU: {metrics['silhouette_iou_luma_gt_8']:.3f}",
        "",
        "## Regions",
        "",
    ]
    for r in regions:
        lines.append(
            f"- {r.name}: IoU {r.silhouette_iou:.3f}, "
            f"mask pixels Storm/Vulkan {r.storm_mask_pixels}/{r.vulkan_mask_pixels}, "
            f"mean RGB Storm {tuple(round(x, 1) for x in r.storm_mean_rgb)}, "
            f"Vulkan {tuple(round(x, 1) for x in r.vulkan_mean_rgb)}"
        )
    lines.extend(
        [
            "",
            "## Fixture",
            "",
            "- `TubeWidthOnly`: linear curve with widths and no authored normals.",
            "- `RibbonFacing`: linear curve with normals `(0,0,1)`; should render as a flat oriented ribbon.",
            "- `RibbonEdgeOn`: linear curve with normals `(0,1,0)`; should be nearly edge-on to the camera.",
            "- `CubicBsplineRibbon`: cubic B-spline oriented ribbon with varying width and displayColor.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)
    out_dir = out_dir.resolve()
    frames = out_dir / "frames"
    logs = out_dir / "logs"
    frames.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    prior_subsegs = os.environ.get("NUSD_CURVE_SUBSEGS")
    os.environ.setdefault("NUSD_CURVE_SUBSEGS", "16")
    try:
        usd_path, camera = write_fixture(out_dir)
        storm_png = frames / "curve_fixture_storm.png"
        vulkan_png = frames / "curve_fixture_vulkan.png"
        diff_png = frames / "curve_fixture_diff_x4.png"
        compare_png = frames / "curve_fixture_compare.png"

        _, storm_log = render_storm(usd_path, storm_png, args.width)
        (logs / "curve_fixture_storm.log").write_text(storm_log, encoding="utf-8")
        render_vulkan(usd_path, vulkan_png, camera, args.width, args.height)
        metrics = image_metrics(storm_png, vulkan_png, diff_png)
        make_compare(storm_png, vulkan_png, diff_png, compare_png, "Curve fixture", metrics)
        regions = _region_metrics(storm_png, vulkan_png)
        _annotated_compare(compare_png, regions)
        write_report(out_dir, metrics, regions)
    finally:
        if prior_subsegs is None:
            os.environ.pop("NUSD_CURVE_SUBSEGS", None)
        else:
            os.environ["NUSD_CURVE_SUBSEGS"] = prior_subsegs

    print(f"Wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
