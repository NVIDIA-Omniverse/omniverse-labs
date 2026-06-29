#!/usr/bin/env python3
"""Build no-diff renderer comparison gallery images from generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import flip_evaluator as _flip
    _FLIP_OK = True
except ModuleNotFoundError:  # fall back to RMS labels if NVIDIA FLIP isn't installed
    _FLIP_OK = False


ROOT = Path(__file__).resolve().parents[1]
VULKAN = ROOT / "nanousd-vulkan-renderer" / "comparisons"
OPENGL = ROOT / "nanousd-opengl-renderer" / "comparisons"
METAL = ROOT / "nanousd-metal-renderer" / "comparisons"
OUT = ROOT / "comparisons" / "all_renderers_contact_sheet.png"
GALLERY = ROOT / "comparisons" / "gallery"
SETS = ("chess", "apple", "warehouse")
SCENE_TITLES = {
    "chess": "Chess Set",
    "apple": "Asset Variety Set",
    "warehouse": "Warehouse",
}
HIGHLIGHTS = (
    ("warehouse", "warehouse", "camA"),
    ("warehouse", "warehouse", "camB"),
    ("chess", "chess_set", "camA"),
    ("apple", "fender_stratocaster", "camA"),
    ("apple", "robot", "camA"),
    ("apple", "pancakes", "camB"),
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _luma(rgb: list[float]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _thumb(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (22, 24, 26))
    x = (size - img.size[0]) // 2
    y = (size - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def _missing_thumb(size: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    tile = Image.new("RGB", (size, size), (42, 27, 30))
    draw = ImageDraw.Draw(tile)
    draw.text((18, size // 2 - 8), "missing", fill=(245, 190, 190), font=font)
    return tile


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("/", "_")


_FLIP_STUDIO_BG_FRACTION = 0.40
_FLIP_FG_THRESHOLD = 12.0


def _flip_corner_bg(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    n = max(6, min(h, w) // 32)
    cs = np.concatenate(
        [img[:n, :n].reshape(-1, 3), img[:n, w - n :].reshape(-1, 3),
         img[h - n :, :n].reshape(-1, 3), img[h - n :, w - n :].reshape(-1, 3)], axis=0)
    return np.median(cs, axis=0)


def _flip_fg(img: np.ndarray) -> np.ndarray:
    return np.linalg.norm(img - _flip_corner_bg(img).reshape(1, 1, 3), axis=2) > _FLIP_FG_THRESHOLD


def flip_score(golden_path: Path, test_path: Path) -> float | None:
    """Calibrated NVIDIA-FLIP of a backend tile vs the OVRTX golden (lower == closer).

    Studio scenes (large uniform backdrop) composite both subjects onto the golden's
    background and average FLIP over the foreground — so a backend's studio-backdrop
    colour and subject size drop out; full scenes use plain FLIP. Mirrors
    nanousd-metal-renderer/comparisons/flip_compare.py. Returns None if FLIP is
    unavailable, so callers fall back to RMS labels.
    """
    if not _FLIP_OK:
        return None
    g = np.asarray(Image.open(golden_path).convert("RGB")).astype(np.float32)
    t = np.asarray(Image.open(test_path).convert("RGB")).astype(np.float32)
    if t.shape != g.shape:
        t = np.asarray(Image.open(test_path).convert("RGB").resize(
            (g.shape[1], g.shape[0]))).astype(np.float32)
    mg, mt = _flip_fg(g), _flip_fg(t)
    union = mg | mt
    if 1.0 - float(union.mean()) >= _FLIP_STUDIO_BG_FRACTION:
        bg = _flip_corner_bg(g).reshape(1, 1, 3)
        cg = (np.where(mg[..., None], g, bg) / 255.0).astype(np.float32)
        ct = (np.where(mt[..., None], t, bg) / 255.0).astype(np.float32)
        err, _, _ = _flip.evaluate(cg, ct, "LDR", applyMagma=False)
        err = np.asarray(err, np.float32)
        if err.ndim == 3:
            err = err[..., 0]
        return float(err[union].mean()) if union.any() else float(err.mean())
    # Full scene: 95th percentile of the FLIP map (not the mean), matching the
    # gate in flip_compare.py — the mean is gameable on full scenes (a
    # tone-matched-but-murky render scores low across the large background).
    err, _, _ = _flip.evaluate(g / 255.0, t / 255.0, "LDR", applyMagma=False)
    err = np.asarray(err, np.float32)
    if err.ndim == 3:
        err = err[..., 0]
    return float(np.percentile(err, 95.0))


def _metric_text(metrics: dict[str, Any] | None, mean_rgb: list[float] | None,
                 flip: float | None = None) -> str:
    if flip is not None:
        iou = f"  IoU {metrics['silhouette_iou']:.3f}" if metrics else ""
        return f"FLIP {flip:.3f}{iou}"
    if metrics:
        return f"RMS {metrics['rms']:.1f}  IoU {metrics['silhouette_iou']:.3f}"
    if mean_rgb:
        return f"Luma {_luma(mean_rgb):.1f}"
    return ""


def _collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for set_name in SETS:
        vk_payload = _load_json(VULKAN / set_name / "metrics.json")
        gl_payload = _load_json(OPENGL / set_name / "metrics.json")
        mt_payload = _load_json(METAL / set_name / "metrics.json")
        gl_cams: dict[tuple[str, str], dict[str, Any]] = {}
        for asset in gl_payload["assets"]:
            for cam in asset["cams"]:
                gl_cams[(asset["label"], cam["cam"])] = cam
        mt_cams: dict[tuple[str, str], dict[str, Any]] = {}
        for asset in mt_payload["assets"]:
            for cam in asset["cams"]:
                mt_cams[(asset["label"], cam["cam"])] = cam

        for asset in vk_payload["assets"]:
            label = asset["label"]
            for cam in asset["cams"]:
                gl_cam = gl_cams.get((label, cam["cam"]))
                mt_cam = mt_cams.get((label, cam["cam"]))
                frame_dir = VULKAN / set_name / "frames"
                gl_frame_dir = OPENGL / set_name / "frames"
                mt_frame_dir = METAL / set_name / "frames"
                rows.append({
                    "set": set_name,
                    "asset": label,
                    "cam": cam["cam"],
                    "paths": [
                        frame_dir / cam["ovrtx_png"].split("/", 1)[-1],
                        frame_dir / cam["rt_png"].split("/", 1)[-1],
                        frame_dir / cam["raster_png"].split("/", 1)[-1],
                        gl_frame_dir / gl_cam["opengl_png"].split("/", 1)[-1] if gl_cam else None,
                        mt_frame_dir / mt_cam["rt_png"].split("/", 1)[-1] if mt_cam else None,
                        mt_frame_dir / mt_cam["raster_png"].split("/", 1)[-1] if mt_cam else None,
                    ],
                    "metrics": [
                        None,
                        cam["metrics_rt"],
                        cam["metrics_raster"],
                        gl_cam["metrics"] if gl_cam else None,
                        mt_cam["metrics_rt"] if mt_cam else None,
                        mt_cam["metrics_raster"] if mt_cam else None,
                    ],
                    "means": [
                        cam["mean_rgb"]["ovrtx"],
                        cam["mean_rgb"]["vk_rt"],
                        cam["mean_rgb"]["vk_raster"],
                        gl_cam["mean_rgb"]["opengl"] if gl_cam else None,
                        mt_cam["mean_rgb"]["metal_rt"] if mt_cam else None,
                        mt_cam["mean_rgb"]["metal_raster"] if mt_cam else None,
                    ],
                    "flip": [None, None, None, None, None, None],
                })
    # Perceptual FLIP of every backend tile vs the row's OVRTX golden (column 0).
    for row in rows:
        paths = row["paths"]
        golden = paths[0]
        if not (golden and golden.exists()):
            continue
        for j in range(1, len(paths)):
            p = paths[j]
            if p and p.exists():
                row["flip"][j] = flip_score(golden, p)
    return rows


def _draw_sheet(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    subtitle: str,
    *,
    thumb: int = 248,
    label_w: int = 210,
) -> Path:
    gap = 18
    col_w = thumb
    title_h = 108
    header_h = 44
    row_h = thumb + 58
    ncol = 6
    width = label_w + gap * (ncol + 2) + col_w * ncol
    height = title_h + header_h + gap + len(rows) * (row_h + gap)

    img = Image.new("RGB", (width, height), (13, 16, 19))
    draw = ImageDraw.Draw(img)
    font_title = _font(30, True)
    font_head = _font(15, True)
    font = _font(13)
    font_small = _font(11)

    draw.rectangle((0, 0, width, title_h), fill=(17, 22, 27))
    draw.rectangle((0, title_h - 4, width, title_h), fill=(54, 117, 136))
    draw.text((gap, 18), title, fill=(247, 250, 252), font=font_title)
    draw.text((gap, 60), subtitle, fill=(181, 192, 201), font=font)

    headers = ["Scene", "OVRTX", "Vulkan RT", "Vulkan Raster", "OpenGL GLES",
               "Metal RT", "Metal Raster"]
    x = gap
    y = title_h
    draw.text((x, y + 11), headers[0], fill=(230, 235, 240), font=font_head)
    x += label_w + gap
    for header in headers[1:]:
        draw.text((x, y + 11), header, fill=(230, 235, 240), font=font_head)
        x += col_w + gap

    y = title_h + header_h + gap
    for idx, row in enumerate(rows):
        fill = (18, 22, 26) if idx % 2 == 0 else (22, 26, 30)
        draw.rectangle((0, y - 8, width, y + row_h + 8), fill=fill)
        label = f"{SCENE_TITLES.get(row['set'], row['set'])}\n{row['asset']}\n{row['cam']}"
        draw.multiline_text((gap, y + 8), label, fill=(226, 232, 236), font=font, spacing=5)
        x = gap + label_w + gap
        for path, metrics, mean_rgb, flipv in zip(
                row["paths"], row["metrics"], row["means"], row["flip"]):
            if path and path.exists():
                tile = _thumb(path, thumb)
            else:
                tile = _missing_thumb(thumb, font)
            draw.rectangle((x - 1, y - 1, x + thumb, y + thumb), outline=(48, 56, 62))
            img.paste(tile, (x, y))
            metric = _metric_text(metrics, mean_rgb, flipv)
            draw.text((x, y + thumb + 9), metric, fill=(177, 186, 194), font=font_small)
            x += col_w + gap
        y += row_h + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _write_scene_sheets(rows: list[dict[str, Any]]) -> list[Path]:
    outputs: list[Path] = []
    for set_name in SETS:
        scene_rows = [row for row in rows if row["set"] == set_name]
        title = f"{SCENE_TITLES.get(set_name, set_name)} Renderer Gallery"
        subtitle = "Six outputs per camera, arranged for direct visual scanning. No subtract/diff panels."
        outputs.append(
            _draw_sheet(
                scene_rows,
                GALLERY / f"{_slug(set_name)}_all_renderers.png",
                title,
                subtitle,
                thumb=300 if set_name != "apple" else 274,
            )
        )
    return outputs


def _write_highlights(rows: list[dict[str, Any]]) -> Path:
    row_map = {(row["set"], row["asset"], row["cam"]): row for row in rows}
    selected = [row_map[key] for key in HIGHLIGHTS if key in row_map]
    return _draw_sheet(
        selected,
        GALLERY / "visual_highlights.png",
        "Visual Highlights",
        "Representative cameras from the warehouse, chess, and asset-variety scenes.",
        thumb=318,
    )


def build(out_path: Path = OUT) -> list[Path]:
    rows = _collect_rows()
    outputs = [
        _draw_sheet(
            rows,
            out_path,
            "Renderer Comparison",
            "OVRTX reference and renderer outputs. No subtract/diff panels.",
        )
    ]
    outputs.extend(_write_scene_sheets(rows))
    outputs.append(_write_highlights(rows))
    return outputs


if __name__ == "__main__":
    for output in build():
        print(output)
