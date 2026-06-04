# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth-from-ovrtx example: USD stage → render step → CPU map → optional hole fill → false-color PNG.

This script uses the Kit viewport render product with ``DepthSensorDistance`` from
``OmniSensorDepthSensorSingleViewAPI`` (see ``robot_with_depth.usda``). That output is often **sparse**.

**Hole-fill tradeoff:** ``dilate_depth_holes`` spreads *average neighbor depth* into invalid pixels.
A few passes (roughly 8–32) can make sparse output easier to see without destroying edges. Very large
values (hundreds/thousands) eventually fill most of the frame but the map becomes **blobby**: repeated
averaging smears depth across occlusion boundaries, so you stop seeing a crisp robot / warehouse—
you only see smooth color regions. That is expected; hole-fill is not a substitute for **dense**
depth from the renderer. Prefer tuning ``robot_with_depth.usda`` (confidence, disparity, noise) for
more real samples first, then add modest hole-fill for display.

For **renderer geometric depth** on ``/Render/Camera`` (UsdRenderVar ``sourceName = "depth"``), use the
layer ``robot_camera_renderer_depth.usda`` and step **only** ``/Render/Camera`` in a separate small
script—do **not** combine camera + viewport in one ``step()`` on current ovrtx builds (can error or
crash). The ``depth`` buffer may still be empty depending on RTX/ovrtx version; see
``skills/reading-render-output``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ovrtx.Renderer: owns the RTX session, stage, and step/map lifecycle.
# ovrtx.Device: map() requires Device.CPU or Device.CUDA (enum). Strings like "cpu" fail on current bindings.
from ovrtx import Device, Renderer

# Pillow: write RGB PNG without adding heavy visualization dependencies (e.g. matplotlib).
from PIL import Image

# ---------------------------------------------------------------------------
# Paths and stage wiring
# ---------------------------------------------------------------------------

# robot_with_depth.usda is a *root layer*: it subLayers the public Robot-OVRTX URL and adds
# OmniSensorDepthSensorSingleViewAPI on the Kit viewport RenderProduct so RTX emits DepthSensorDistance.
STAGE_PATH = Path(__file__).resolve().parent / "robot_with_depth.usda"

# Single render product (viewport). Do not add /Render/Camera in the same step() here—unstable on some builds.
RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"

# RTX AOV from the depth sensor (see omni.sensors.nv.camera).
DEPTH_RENDER_VAR = "DepthSensorDistance"

# ---------------------------------------------------------------------------
# Visualization tuning (PNG appearance, not sensor physics)
# ---------------------------------------------------------------------------

VISUAL_GAMMA = 0.48

# More steps can let the depth path settle; only the last step is followed by map().
DEPTH_WARMUP_STEPS = 4

# Hole-fill for sparse sensor output (display only): ~one pixel radius per pass from valid seeds.
# Low (0–8): sparse, sharper structure. Medium (12–32): more filled, still some shape. Very high
# (100+): near-full frame but **blobby**—depth is smeared by repeated averaging, not real detail.
DEPTH_HOLE_FILL_PASSES = 20

# -----------------------------------------------------------------------------
# Viewer guide (demo / livestream): story of this file
# -----------------------------------------------------------------------------
# 1. Constants above: which USD layer, which RenderProduct, which AOV name, and display tuning.
# 2. Helpers below: decode GPU buffer → meters, optional hole-fill, jet false-color for PNG.
# 3. main(): Renderer() → open_usd → step the viewport a few times → map DepthSensorDistance on CPU
#    → hole-fill → save RGB PNG next to the stage (or paths from --usd / -o).


def decode_depth_distance_m(raw: np.ndarray) -> np.ndarray:
    """Decode the CPU-mapped buffer to float32 distances in **meters**."""
    raw = np.asarray(raw)
    if raw.dtype == np.uint32:
        return raw.view(np.float32).reshape(raw.shape)
    return raw.astype(np.float32, copy=False)


def _valid_depth_mask(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float32)
    return np.isfinite(d) & (d > 0.0) & (d < 1.0e5)


def _as_2d_depth(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        return d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected H×W or H×W×1 depth, got shape {d.shape}")
    return d


def dilate_depth_holes(depth_m: np.ndarray, passes: int) -> np.ndarray:
    """Fill invalid pixels with the mean of valid 8-neighbors (display aid only).

    This is **not** edge-aware: high ``passes`` propagates flat depth regions and blurs object
    boundaries, which is why very large values yield a full but blobby image.
    """
    d = _as_2d_depth(depth_m)
    valid = _valid_depth_mask(d)
    h, w = d.shape
    work = np.where(valid, d, 0.0).astype(np.float64)
    v = valid
    for _ in range(passes):
        acc = np.zeros((h, w), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.int32)
        wp = np.pad(work, ((1, 1), (1, 1)), mode="edge")
        vp = np.pad(v, ((1, 1), (1, 1)), mode="edge")
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                sl = wp[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
                sv = vp[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
                use = sv & (sl > 0.0)
                acc += np.where(use, sl, 0.0)
                cnt += use.astype(np.int32)
        fill = ~v & (cnt > 0)
        work[fill] = acc[fill] / np.maximum(cnt[fill], 1).astype(np.float64)
        v = v | fill
    return np.where(v, work, 0.0).astype(np.float32)


def jet_colormap(t: np.ndarray) -> np.ndarray:
    t = np.clip(t.astype(np.float64), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def depth_to_color_rgb_u8(depth_m: np.ndarray, *, visual_gamma: float = VISUAL_GAMMA) -> np.ndarray:
    d = _as_2d_depth(depth_m)
    valid = _valid_depth_mask(d)
    if not np.any(valid):
        raise ValueError("No valid depth samples (check depth sensor parameters / stage).")
    h, w = d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    dv = d[valid]
    lo, hi = np.percentile(dv, [0.0, 99.5])
    if hi <= lo:
        lo, hi = float(dv.min()), float(dv.max())
    t = np.zeros_like(d, dtype=np.float64)
    t[valid] = (np.clip(d[valid], lo, hi) - lo) / (hi - lo + 1e-8)
    t[valid] = np.power(t[valid], visual_gamma)
    rgb = jet_colormap(t[valid])
    out[valid] = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render DepthSensorDistance to a false-color PNG (see robot_with_depth.usda).",
    )
    p.add_argument(
        "--usd",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Root layer .usda (default: robot_with_depth.usda next to this script). "
            "Use the copy under usd-viewer-example/ when driving the viewer from that folder."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output PNG path (default: depth_map.png next to the root layer).",
    )
    return p.parse_args()


def main() -> None:
    # --- CLI: default stage is robot_with_depth.usda beside this script; default PNG is depth_map.png ---
    args = parse_args()
    stage_path = args.usd.resolve() if args.usd is not None else STAGE_PATH
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (stage_path.parent / "depth_map.png")
    )

    if not stage_path.is_file():
        print(f"Missing stage file:\n  {stage_path}", file=sys.stderr)
        sys.exit(1)

    # Viewer: RTX session + stage load (same pattern as other ovrtx Python examples).
    print("Creating renderer (first run may take a while)...", file=sys.stderr)
    renderer = Renderer()
    print(f"Loading {stage_path.name}...", file=sys.stderr)
    renderer.open_usd(str(stage_path))

    # Viewer: warmup steps let the depth pipeline settle; only the last step’s frame is read below.
    print(
        f"Stepping {RENDER_PRODUCT} ({DEPTH_WARMUP_STEPS}x, then map once)...",
        file=sys.stderr,
    )
    products = None
    for _i in range(DEPTH_WARMUP_STEPS):
        products = renderer.step(
            render_products={RENDER_PRODUCT},
            delta_time=1.0 / 60.0,
        )
    assert products is not None

    # Viewer: one product → frames → map the DepthSensorDistance tensor on CPU (DLPack → NumPy).
    for _name, product in products.items():
        for frame in product.frames:
            if DEPTH_RENDER_VAR not in frame.render_vars:
                print(
                    f"No {DEPTH_RENDER_VAR!r} render var. Available: "
                    f"{list(frame.render_vars.keys())}",
                    file=sys.stderr,
                )
                sys.exit(1)
            with frame.render_vars[DEPTH_RENDER_VAR].map(device=Device.CPU) as var:
                raw = np.from_dlpack(var.tensor)
                depth_m = decode_depth_distance_m(raw)
                depth_m = _as_2d_depth(depth_m)
                n_raw = int(_valid_depth_mask(depth_m).sum())
                # Viewer: optional display-only dilate—see DEPTH_HOLE_FILL_PASSES note at top of file.
                if DEPTH_HOLE_FILL_PASSES > 0:
                    depth_m = dilate_depth_holes(depth_m, DEPTH_HOLE_FILL_PASSES)
                n_fill = int(_valid_depth_mask(depth_m).sum())
                print(
                    f"Depth pixels (sensor / after {DEPTH_HOLE_FILL_PASSES} fill passes): "
                    f"{n_raw} / {n_fill}",
                    file=sys.stderr,
                )
                # Viewer: normalize depth → jet colormap → 8-bit RGB; Pillow writes a single PNG.
                color = depth_to_color_rgb_u8(depth_m)
                Image.fromarray(color, mode="RGB").save(output_path)
                print(f"Saved:\n  {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
