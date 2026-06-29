# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrated NVIDIA-ϜLIP comparison of a renderer output against the OVRTX golden.

Naive full-image LDR-FLIP is misleading for this comparison set for two reasons:

  1. Background mismatch. The Metal backend renders the studio sets (apple, chess)
     on a near-white background while the OVRTX golden uses neutral grey, so a
     whole-image FLIP is dominated by the backdrop, not the subject — a textured,
     correct subject can score *worse* than a broken one.
  2. Subject size. A small studio subject is diluted by the matched background, so
     a near-black (clearly wrong) robot can score *lower* than a good one.

This module fixes both so the score tracks the perceived difference from the golden:

  * Studio scenes (a large, uniform backdrop is detected): composite both the
    golden and the test subject onto the *golden's own* background colour, then
    average FLIP error over the foreground only — background colour and subject
    size both drop out.
  * Full scenes (warehouse — the backdrop *is* the scene and already matches the
    golden): plain full-image FLIP.

The reference is always the OVRTX golden image. Lower FLIP == closer to the golden.
Note that single-pass backends sit ~0.5 FLIP from the path-traced golden even when
they look good — the useful signal is the *relative* ordering across backends/scenes
and the heatmap localisation, not an absolute "good == 0".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage.color import deltaE_ciede2000, rgb2lab

import flip_evaluator as flip

# Fraction of foreground (subject) pixels below which a frame is treated as a
# studio shot with a clear backdrop (apple/chess ~0.6-0.85 background; warehouse
# fills the frame, ~0.05-0.20 background).
_STUDIO_BG_FRACTION = 0.40
# Per-pixel colour distance from the corner-median background that counts as subject.
# Matches the silhouette-IoU mask in render_backend_comparison.py for consistency.
_FG_THRESHOLD = 12.0


def _load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def _corner_bg(img: np.ndarray) -> np.ndarray:
    """Median colour of the four corner patches — the assumed background colour."""
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
    return np.median(corners, axis=0)


def _foreground_mask(img: np.ndarray) -> np.ndarray:
    bg = _corner_bg(img)
    return np.linalg.norm(img - bg.reshape(1, 1, 3), axis=2) > _FG_THRESHOLD


def _flip_raw(ref01: np.ndarray, test01: np.ndarray) -> np.ndarray:
    """Raw (uncoloured) per-pixel LDR-FLIP error in [0,1], HxW, for inputs in [0,1]."""
    errmap, _, _ = flip.evaluate(ref01, test01, "LDR", inputsRGB=True, applyMagma=False)
    err = np.asarray(errmap, dtype=np.float32)
    if err.ndim == 3:
        err = err[..., 0]
    return err


def _flip_magma(ref01: np.ndarray, test01: np.ndarray) -> np.ndarray:
    """Magma-coloured FLIP error map, HxWx3 uint8."""
    heat, _, _ = flip.evaluate(ref01, test01, "LDR", inputsRGB=True, applyMagma=True)
    return (np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


# ---- Orthogonal structure / degeneracy legs ----------------------------------
# FLIP mean is gameable: a render that merely matches the golden's overall tone
# scores low everywhere even when salient structure is destroyed (the warehouse
# "murky racks beat the good render" failure). These cheap, deterministic legs
# are brightness-invariant / degeneracy-specific so a render cannot pass the gate
# by exploiting any one axis. See comparisons/README and flip_pipeline.gate().

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _gmsd(golden: np.ndarray, test: np.ndarray) -> float:
    """Gradient Magnitude Similarity Deviation on luma (Xue et al. 2014).

    Brightness-INVARIANT structure metric: a uniform exposure/tone offset leaves
    gradients unchanged, but crushing racks to black or over-smoothing destroys
    them. Lower == structurally closer. Prewitt gradients after a 2x average
    pool; c=170 matches the paper's 0..255 stability constant.
    """
    def luma_pool(img: np.ndarray) -> np.ndarray:
        l = img @ _LUMA
        h, w = l.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        return l[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))

    g, t = luma_pool(golden), luma_pool(test)
    px = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float32) / 3.0
    py = px.T
    gm = np.sqrt(convolve(g, px, mode="nearest") ** 2 + convolve(g, py, mode="nearest") ** 2)
    tm = np.sqrt(convolve(t, px, mode="nearest") ** 2 + convolve(t, py, mode="nearest") ** 2)
    c = 170.0
    gms = (2.0 * gm * tm + c) / (gm ** 2 + tm ** 2 + c)
    return float(gms.std())


def _black_frac(img: np.ndarray) -> float:
    """Fraction of pixels crushed to near-black (luma < 0.05). The anti-degenerate
    guard: a render that blacks out salient geometry (e.g. SSAO over-darkening the
    racks) reads far above the golden here even when its FLIP mean looks great."""
    return float(((img @ _LUMA) < 0.05 * 255.0).mean())


_DELTAE_KL = 2.0  # CIEDE2000 lightness weight >1 down-weights exposure, isolating hue/chroma error


def _deltae(golden: np.ndarray, test: np.ndarray, region: np.ndarray | None) -> float:
    """Mean exposure-suppressed CIEDE2000 colour difference over `region` (or full image).

    Catches 'wrong colour while exposure-matched' — e.g. a floor that reads grey where the
    golden is blue — which FLIP's lightness-heavy HyAB term and GMSD's brightness-invariant
    structure both miss. kL>1 down-weights the CIEDE2000 lightness term so a pure exposure
    offset does not dominate the colour signal."""
    lg, lt = rgb2lab(golden / 255.0), rgb2lab(test / 255.0)
    de = deltaE_ciede2000(lg, lt, kL=_DELTAE_KL)
    if region is not None and region.any():
        return float(de[region].mean())
    return float(de.mean())


def _dist_stats(err: np.ndarray) -> dict[str, float]:
    """Distribution of the per-pixel FLIP error, so a single gated number always
    ships inside a visible distribution (the FLIP authors recommend the map/
    histogram over any scalar)."""
    q = np.percentile(err, [25, 50, 75, 90, 95, 99])
    return {"q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]),
            "p90": float(q[3]), "p95": float(q[4]), "p99": float(q[5])}


_REGION_LABELS = (
    ("upper-left", "upper-center", "upper-right"),
    ("mid-left", "center", "mid-right"),
    ("lower-left", "lower-center", "lower-right"),
)


def _worst_region(err: np.ndarray, weight: np.ndarray | None) -> str:
    """3x3 grid cell with the highest mean FLIP error (optionally weighted by a mask)."""
    h, w = err.shape
    best, best_label = -1.0, "center"
    for gy in range(3):
        for gx in range(3):
            ys, ye = gy * h // 3, (gy + 1) * h // 3
            xs, xe = gx * w // 3, (gx + 1) * w // 3
            cell = err[ys:ye, xs:xe]
            if weight is not None:
                m = weight[ys:ye, xs:xe]
                if m.sum() < 16:
                    continue
                val = float(cell[m].mean())
            else:
                val = float(cell.mean())
            if val > best:
                best, best_label = val, _REGION_LABELS[gy][gx]
    return best_label


def _tonal(golden: np.ndarray, test: np.ndarray, region: np.ndarray | None) -> dict[str, float]:
    """Brightness/colour-temperature delta of test vs golden over `region` (or full)."""
    if region is not None and region.any():
        g = golden[region]
        t = test[region]
    else:
        g = golden.reshape(-1, 3)
        t = test.reshape(-1, 3)
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    g_l = float((g @ lum).mean())
    t_l = float((t @ lum).mean())
    luma_delta_pct = ((t_l - g_l) / g_l * 100.0) if g_l > 1e-3 else 0.0
    # R-B as a crude warm(+)/cool(-) axis.
    g_rb = float((g[:, 0] - g[:, 2]).mean())
    t_rb = float((t[:, 0] - t[:, 2]).mean())
    return {"luma_delta_pct": luma_delta_pct, "rb_delta": t_rb - g_rb}


@dataclass
class FlipResult:
    score: float                 # studio: mean FLIP over foreground; fullscene: p95 FLIP (lower == closer)
    mode: str                    # "studio" | "fullscene"
    bg_fraction: float
    raw_err: np.ndarray = field(repr=False)   # HxW raw FLIP error used for the score
    heatmap: np.ndarray = field(repr=False)   # HxWx3 uint8 magma error map
    tonal: dict[str, float] = field(default_factory=dict)
    worst_region: str = "center"
    gmsd: float = 0.0                 # brightness-invariant structure leg (lower == closer)
    black_frac: float = 0.0          # fraction of test crushed to near-black
    black_frac_golden: float = 0.0   # same for the golden (crush guard reference)
    deltae: float = 0.0              # exposure-suppressed CIEDE2000 colour difference (lower == closer)
    stats: dict[str, float] = field(default_factory=dict)  # FLIP error distribution

    def diagnosis(self) -> str:
        """One-line human-readable diagnosis of how the test diverges from the golden."""
        d = self.tonal.get("luma_delta_pct", 0.0)
        rb = self.tonal.get("rb_delta", 0.0)
        bright = "brighter" if d >= 0 else "darker"
        warm = "warmer" if rb >= 0 else "cooler"
        parts = [f"{abs(d):.0f}% {bright}"]
        if abs(rb) >= 2.0:
            parts.append(warm)
        parts.append(f"worst error {self.worst_region}")
        return ", ".join(parts)


def flip_vs_golden(golden_path: str | Path, test_path: str | Path) -> FlipResult:
    """Compare a renderer output to the OVRTX golden with the calibrated recipe."""
    golden = _load_rgb(golden_path)
    test = _load_rgb(test_path)
    if test.shape != golden.shape:
        test = np.asarray(
            Image.open(test_path).convert("RGB").resize(
                (golden.shape[1], golden.shape[0]), Image.Resampling.LANCZOS
            )
        ).astype(np.float32)

    mg, mt = _foreground_mask(golden), _foreground_mask(test)
    union = mg | mt
    bg_fraction = 1.0 - float(union.mean())

    if bg_fraction >= _STUDIO_BG_FRACTION:
        # Studio: composite both subjects onto the golden's own background colour so
        # the backdrop contributes zero error, then score over the foreground only.
        # The masked mean is well-behaved here (the subject is already isolated and
        # there is no demonstrated failure), so keep it for ranking stability.
        bg = _corner_bg(golden).reshape(1, 1, 3)
        comp_g = np.where(mg[..., None], golden, bg)
        comp_t = np.where(mt[..., None], test, bg)
        raw = _flip_raw(comp_g / 255.0, comp_t / 255.0)
        score_err = raw[union] if union.any() else raw.reshape(-1)
        score = float(score_err.mean())
        heatmap = _flip_magma(comp_g / 255.0, comp_t / 255.0)
        tonal = _tonal(golden, test, union)
        worst = _worst_region(raw, union)
        mode = "studio"
    else:
        # Full scene: the mean is gameable — a render that merely matches the
        # golden's overall (dark) tone scores low across the large background even
        # when salient geometry is destroyed (the "murky racks beat the good
        # render" failure). Gate on the 95th percentile of FLIP's own map: it reads
        # the worst regions (crushed/blown structure) and ignores the forgivable
        # global tone offset that lives in the bulk of the distribution.
        raw = _flip_raw(golden / 255.0, test / 255.0)
        score_err = raw.reshape(-1)
        score = float(np.percentile(raw, 95.0))
        heatmap = _flip_magma(golden / 255.0, test / 255.0)
        tonal = _tonal(golden, test, None)
        worst = _worst_region(raw, None)
        mode = "fullscene"

    return FlipResult(
        score=score,
        mode=mode,
        bg_fraction=bg_fraction,
        raw_err=raw,
        heatmap=heatmap,
        tonal=tonal,
        worst_region=worst,
        gmsd=_gmsd(golden, test),
        black_frac=_black_frac(test),
        black_frac_golden=_black_frac(golden),
        deltae=_deltae(golden, test, union if mode == "studio" else None),
        stats=_dist_stats(score_err),
    )


def compare_strip(golden_path: str | Path, test_path: str | Path,
                  result: FlipResult, out_path: str | Path) -> None:
    """Write a `golden | test | FLIP heatmap` strip PNG for visual inspection."""
    g = np.asarray(Image.open(golden_path).convert("RGB"))
    t = np.asarray(Image.open(test_path).convert("RGB"))
    h = result.heatmap
    if t.shape != g.shape:
        t = np.asarray(Image.open(test_path).convert("RGB").resize((g.shape[1], g.shape[0])))
    strip = np.concatenate([g, t, h], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)
