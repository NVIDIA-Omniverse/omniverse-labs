# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side comparison harness: stock ovgear (pxr + ovrtx) vs
this stack (nanousd + OpenGL ES). For each USD path, launch both
backends, capture the inner GLFW window of each, save individual
images and a labelled composite to ``docs/compare/<name>``.

Usage::

    python scripts/compare.py NAME PATH [SELECT_PATH]

The third arg is an optional prim path the launcher should auto-select
so the Property Inspector populates. Defaults to ``/World``.

Both backends are run in turn (sequentially — they each grab the GLFW
context). The script kills any existing ovgear before launching, then
sleeps 12 s to let panels render, then xwd captures the inner window
by ID and PNG-decodes via PIL. Side-by-side composite is saved to
``docs/compare/<name>.png``.

Environment requirements (must be set in the launching shell):
* venv at ``$NUSD_OVGEAR_VENV`` (default: ``<repo>/../.venv-ovgear``)
* ``DISPLAY`` (default ``:1``) and ``XAUTHORITY`` set
* For ovrtx: a non-monolithic OpenUSD install at
  ``$NUSD_USD_INSTALL`` plus the ovrtx bundled ``usd_plugins/*``
  discovered relative to the venv (override with
  ``$NUSD_OVRTX_PLUGINS``).

Exit codes: 0 on success, 1 if a capture or composite was skipped, 2
on bad invocation.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
VENV = Path(os.environ.get("NUSD_OVGEAR_VENV") or (REPO.parent / ".venv-ovgear"))
DEFAULT_DOCS = REPO / "docs" / "compare"
DOCS = Path(os.environ.get("NUSD_COMPARE_OUT_DIR") or DEFAULT_DOCS)
USD_INSTALL = os.environ.get("NUSD_USD_INSTALL", "")
LOG_DIR = Path(os.environ.get("NUSD_COMPARE_LOG_DIR", "/tmp")).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _discover_ovrtx_plugins() -> str:
    override = os.environ.get("NUSD_OVRTX_PLUGINS")
    if override:
        return override
    # Look for ovrtx/bin/usd_plugins under the venv's site-packages.
    site = VENV / "lib"
    if site.is_dir():
        for entry in sorted(site.iterdir()):
            cand = entry / "site-packages" / "ovrtx" / "bin" / "usd_plugins"
            if cand.is_dir():
                return str(cand)
    return ""


OVRTX_PLUGINS = _discover_ovrtx_plugins()
PXR_PLUGINPATH = (
    ":".join(
        f"{OVRTX_PLUGINS}/{d}"
        for d in ("rtx_settings", "omni_lens_distortion", "omni_nurec_types",
                  "omni_playback", "omni_sensors", "usd_particle_field")
    )
    if OVRTX_PLUGINS
    else ""
)


def _kill_running() -> None:
    subprocess.run(["pkill", "-f", "python -m ovgear"], check=False)
    subprocess.run(["pkill", "-f", "nusd_gles.ovgear_app"], check=False)
    time.sleep(2)


def _launch(args: list[str], env_extra: dict, cwd: Optional[Path] = None) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.Popen(
        args, env=env, cwd=str(cwd or REPO),
        stdout=open(LOG_DIR / "compare.out", "ab"),
        stderr=open(LOG_DIR / "compare.err", "ab"),
    )


def _capture_inner_window(name: str) -> Optional[bytes]:
    out = subprocess.run(
        ["xwininfo", "-display", os.environ.get("DISPLAY", ":1"),
         "-name", name, "-tree"],
        capture_output=True, text=True,
    )
    inner_id = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("0x"):
            continue
        if f'"{name}":' in line:
            inner_id = line.split()[0]
            break
    if inner_id is None:
        return None
    res = subprocess.run(
        ["xwd", "-display", os.environ.get("DISPLAY", ":1"),
         "-id", inner_id, "-silent"],
        capture_output=True,
    )
    if res.returncode != 0:
        return None
    return res.stdout


def _decode_xwd(blob: bytes) -> Optional[np.ndarray]:
    if len(blob) < 100:
        return None
    hdr = struct.unpack(">25I", blob[:100])
    hsz, ncolors, bpl, W, H = hdr[0], hdr[18], hdr[12], hdr[4], hdr[5]
    off = hsz + ncolors * 12
    bpp = bpl // W
    img = np.frombuffer(blob, dtype=np.uint8, count=bpl * H, offset=off).reshape(H, W, bpp)
    if bpp >= 3:
        return np.stack([img[..., 2], img[..., 1], img[..., 0]], axis=-1)
    return img


def capture(name: str, usd_path: str, select: str = "/World") -> dict:
    """Run both backends in turn, return ``{'nanousd': arr, 'ovrtx': arr}``."""
    DOCS.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # 1) Our nanousd + GLES launcher
    _kill_running()
    proc = _launch(
        [str(VENV / "bin" / "python"), "-m", "nusd_gles.ovgear_app", str(usd_path)],
        env_extra=dict(
            PYTHONPATH=str(REPO / "python"),
            NUSD_GLES_AUTOSELECT=select,
            NUSD_GLES_EXPAND_ALL="1",
        ),
    )
    time.sleep(12)
    blob = _capture_inner_window("OvGear")
    proc.terminate()
    proc.wait(timeout=10)
    if blob is not None:
        arr = _decode_xwd(blob)
        if arr is not None:
            out["nanousd"] = arr
            Image.fromarray(arr).save(DOCS / f"{name}_nanousd.png")

    # 2) Stock ovgear + pxr + ovrtx
    if not USD_INSTALL:
        print("  SKIPPED ovrtx leg (set NUSD_USD_INSTALL=<path-to-OpenUSD>)",
              flush=True)
        return out
    _kill_running()
    proc = _launch(
        [str(VENV / "bin" / "python"), "-m", "ovgear", str(usd_path)],
        env_extra=dict(
            OVRTX_SKIP_USD_CHECK="1",
            PYTHONPATH=f"{USD_INSTALL}/lib/python",
            LD_LIBRARY_PATH=f"{USD_INSTALL}/lib",
            PXR_PLUGINPATH_NAME=PXR_PLUGINPATH,
        ),
    )
    time.sleep(20)
    blob = _capture_inner_window("OvGear")
    proc.terminate()
    proc.wait(timeout=10)
    if blob is not None:
        arr = _decode_xwd(blob)
        if arr is not None:
            out["ovrtx"] = arr
            Image.fromarray(arr).save(DOCS / f"{name}_ovrtx.png")

    return out


def composite(name: str, label: str, sub: str, captures: dict) -> Path:
    if "nanousd" not in captures or "ovrtx" not in captures:
        raise RuntimeError(f"missing capture for {name}: have {list(captures)}")
    a = captures["nanousd"]
    b = captures["ovrtx"]
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    pad_h = 60
    gap = 12
    canvas = np.full((H + pad_h, W * 2 + gap, 3), (24, 24, 28), dtype=np.uint8)
    canvas[pad_h:pad_h + a.shape[0], :a.shape[1]] = a
    canvas[pad_h:pad_h + b.shape[0], W + gap:W + gap + b.shape[1]] = b
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    draw.text((20, 8), f"nanousd + GLES (this repo)  ·  {label}", font=font_h, fill=(220, 240, 220))
    draw.text((20, 36), sub, font=font_s, fill=(150, 180, 150))
    draw.text((W + gap + 20, 8), f"pxr + ovrtx (stock ovgear)  ·  {label}", font=font_h, fill=(240, 220, 220))
    draw.text((W + gap + 20, 36), sub, font=font_s, fill=(180, 150, 150))
    out_path = DOCS / f"{name}_compare.png"
    img.save(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="short name for the asset (file basename, no ext)")
    ap.add_argument("usd_path", help="absolute USD path")
    ap.add_argument("--select", default="/World", help="auto-selected prim path")
    ap.add_argument("--label", default=None, help="header label for composite")
    ap.add_argument("--sub", default="", help="sub-label below header")
    args = ap.parse_args(argv)
    print(f"capturing {args.name}: {args.usd_path}", flush=True)
    caps = capture(args.name, args.usd_path, select=args.select)
    print(f"  captured: {sorted(caps)}", flush=True)
    if "nanousd" in caps and "ovrtx" in caps:
        out = composite(args.name, args.label or args.name, args.sub, caps)
        print(f"  composite: {out}", flush=True)
        return 0
    print(f"  SKIPPED composite (missing capture)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
