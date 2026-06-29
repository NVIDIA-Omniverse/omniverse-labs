#!/usr/bin/env python3
"""Render one USD through nanousdview's NVIDIA OVRTX backend.

This helper intentionally imports ``nanousdview._backend`` rather than the
local nanousd Vulkan facade. The caller must run it with an official OVRTX
Python environment and a PYTHONPATH containing only the nanousdview package.
Output is a raw PPM so the helper does not depend on Pillow in the OVRTX venv.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _as_tuple(values, n: int) -> tuple[float, ...]:
    if len(values) != n:
        raise ValueError(f"expected {n} values, got {len(values)}")
    return tuple(float(v) for v in values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    os.environ["NANOUSD_VIEW_BACKEND"] = "ovrtx"

    from nanousdview._backend import (  # noqa: PLC0415
        OvrtxViewportRenderer,
        VIEW_RENDER_RT,
        configure_backend,
    )

    configure_backend("ovrtx")
    cam = json.loads(args.camera_json)
    renderer = OvrtxViewportRenderer(width=args.width, height=args.height)
    try:
        default_lighting = os.environ.get("NUVIEW_OVRTX_DEFAULT_LIGHTING", "1")
        if default_lighting.strip().lower() not in ("0", "false", "off", "no"):
            renderer.set_default_lighting(camera_light=True, dome_light=True)
        renderer.load_stage(str(Path(args.usd).resolve()))
        renderer.set_render_mode(VIEW_RENDER_RT)
        renderer.set_camera(
            _as_tuple(cam["eye"], 3),
            _as_tuple(cam["target"], 3),
            _as_tuple(cam["up"], 3),
            float(cam["fov_degrees"]),
            float(cam["near_clip"]),
            float(cam["far_clip"]),
            focal_length=float(cam["focal_length"]),
            horizontal_aperture=float(cam["horizontal_aperture"]),
            vertical_aperture=float(cam["vertical_aperture"]),
        )

        pixels = None
        for _ in range(max(args.warmup, 0) + max(args.frames, 1)):
            pixels = renderer.render_ldr(render_mode=VIEW_RENDER_RT)
        if pixels is None:
            raise RuntimeError("nanousdview OVRTX produced no pixels")
        arr = np.asarray(pixels)
        if arr.dtype != np.uint8:
            rgb = np.clip(arr[..., :3].astype(np.float32), 0.0, None)
            rgb = rgb / (1.0 + rgb)
            arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            arr = arr[..., :3]

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        h, w = arr.shape[:2]
        with out.open("wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(np.ascontiguousarray(arr[:, :, :3]).tobytes())
        print(
            json.dumps(
                {
                    "ok": True,
                    "shape": [int(h), int(w), 3],
                    "mean_rgb": [float(arr[..., c].mean()) for c in range(3)],
                }
            ),
            flush=True,
        )
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
