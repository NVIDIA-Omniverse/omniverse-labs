#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render any USD file through nanousdview's OpenGL OVRTX backend.

Usage:
  scripts/render_usd.py <usd-path> [out.png] [W H]
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent


def _python_exe() -> str:
    venv_py = WORKSPACE / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":1")
    env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
    paths = [str(WORKSPACE / "nanousdview" / "python")]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(paths) if not existing else f"{':'.join(paths)}:{existing}"
    return env


def _read_ppm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"P6\n"):
        raise ValueError(f"{path}: expected binary PPM")
    pos = 3
    while data[pos:pos + 1] == b"#":
        pos = data.index(b"\n", pos) + 1
    width_s, height_s = data[pos:data.index(b"\n", pos)].split()
    pos = data.index(b"\n", pos) + 1
    max_s = data[pos:data.index(b"\n", pos)].strip()
    pos = data.index(b"\n", pos) + 1
    if max_s != b"255":
        raise ValueError(f"{path}: unsupported max value {max_s!r}")
    width, height = int(width_s), int(height_s)
    rgb = np.frombuffer(data[pos:], dtype=np.uint8)
    return np.ascontiguousarray(rgb.reshape(height, width, 3))


def _write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def _capture(usd_path: Path, ppm_path: Path, width: int, height: int) -> None:
    cmd = [
        _python_exe(),
        "-m",
        "nanousdview",
        "--backend",
        "opengl",
        str(usd_path),
        "--screenshot",
        str(ppm_path),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    subprocess.run(cmd, cwd=WORKSPACE, env=_child_env(), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd_path")
    ap.add_argument("out_png", nargs="?", default="/tmp/render.png")
    ap.add_argument("width", nargs="?", type=int, default=1024)
    ap.add_argument("height", nargs="?", type=int, default=768)
    # Retained for old invocations; nanousdview frames through the OVRTX path.
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--preserve-upaxis", action="store_true")
    args = ap.parse_args()

    usd_path = Path(args.usd_path).resolve()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ppm = out_png.with_suffix(".ppm")
    _capture(usd_path, ppm, args.width, args.height)
    rgb = _read_ppm(ppm)
    if out_png.suffix.lower() == ".ppm":
        print(f"[render_usd] wrote {ppm} through nanousdview --backend opengl")
    else:
        _write_png(out_png, rgb)
        print(f"[render_usd] wrote {out_png} through nanousdview --backend opengl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
