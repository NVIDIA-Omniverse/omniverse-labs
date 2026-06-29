#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the renderer against a sample of Agibot/GenieSim assets.

Pulls one asset per category from the HuggingFace catalog at
agibot-world/GenieSimAssets, renders each via test_headless_render,
records pass/fail + render time + a thumbnail. Builds a contact-sheet
grid of every successful thumbnail at the end.

Why a sampler vs. all 5,140: full catalog is gigabytes of textures and
hours of download time. One asset per category gives material-diversity
coverage without the storage hit. The first-N-categories selection is
deterministic so re-runs hit the same fixtures.

Usage:
    python3 tools/agibot_smoke/run.py [--limit 50] [--category-prefix building]

Outputs (relative to tools/agibot_smoke/):
    out/<category>_<asset>.png   per-asset thumbnail (1920x1080 → 480x270)
    out/_grid.png                contact sheet of all successes
    out/results.json             per-asset summary (status, ms, errors)

Cache lives at ~/assets/agibot/cache/ — re-running skips already-downloaded
assets. Delete that dir to force a fresh pull.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
RENDERER = REPO_ROOT / "build" / "test_headless_render"
CACHE = Path.home() / "assets" / "agibot" / "cache"
OUT = HERE / "out"

HF_BASE = "https://huggingface.co/datasets/agibot-world/GenieSimAssets/resolve/main"
HF_API = "https://huggingface.co/api/datasets/agibot-world/GenieSimAssets/tree/main"

# Categories to sweep, in order. Picking under objects/benchmark/ gets
# the densest variety; objects/genie/ has additional retail/industry/
# catering/home/office assets we can layer in later.
CATEGORY_ROOT = "objects/benchmark"


@dataclass
class Result:
    category: str
    asset: str
    status: str          # "ok" | "no_asset" | "download_fail" | "render_fail"
    ms: float = 0.0
    n_meshes: int = 0
    error: str = ""
    thumbnail: str = ""  # relative to OUT


def http_get_json(url: str) -> list:
    """One-shot HF API call; returns the parsed JSON list."""
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def http_download(url: str, dst: Path) -> bool:
    """Download to dst; returns True on success."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as out:
            shutil.copyfileobj(r, out)
        return True
    except Exception as e:
        print(f"    download fail: {url} → {e}", file=sys.stderr)
        if dst.exists():
            dst.unlink()
        return False


def list_categories() -> list[str]:
    data = http_get_json(f"{HF_API}/{CATEGORY_ROOT}")
    return sorted(d["path"].rsplit("/", 1)[-1]
                  for d in data
                  if d["type"] == "directory")


def pick_asset(category: str) -> str | None:
    """First asset name under a category."""
    try:
        data = http_get_json(f"{HF_API}/{CATEGORY_ROOT}/{category}")
    except Exception as e:
        print(f"  category list fail: {e}", file=sys.stderr)
        return None
    dirs = [d for d in data if d["type"] == "directory"]
    return dirs[0]["path"].rsplit("/", 1)[-1] if dirs else None


def download_asset(category: str, asset: str) -> Path | None:
    """Pull Aligned.usda + Aligned.usd + textures into the cache.
    Returns the path to Aligned.usda on success."""
    asset_dir = CACHE / category / asset
    base = f"{HF_BASE}/{CATEGORY_ROOT}/{category}/{asset}"

    # Aligned.usda must exist; Aligned.usd carries the binary geometry crate.
    usda = asset_dir / "Aligned.usda"
    if not http_download(f"{base}/Aligned.usda", usda):
        return None
    if not http_download(f"{base}/Aligned.usd", asset_dir / "Aligned.usd"):
        return None

    # Texture list — best-effort. Each material may live under
    # textures/* or textures/<material>/* — we sweep up to one level deep.
    try:
        tex_list = http_get_json(f"{HF_API}/{CATEGORY_ROOT}/{category}/{asset}/textures")
    except Exception:
        return usda  # textures dir may not exist; ok
    for entry in tex_list:
        name = entry["path"].rsplit("/", 1)[-1]
        if entry["type"] == "file" and (name.lower().endswith((".jpg", ".png", ".jpeg", ".tga"))):
            http_download(
                f"{base}/textures/{name}",
                asset_dir / "textures" / name,
            )
        elif entry["type"] == "directory":
            try:
                sub_list = http_get_json(
                    f"{HF_API}/{CATEGORY_ROOT}/{category}/{asset}/textures/{name}"
                )
            except Exception:
                continue
            for s in sub_list:
                sname = s["path"].rsplit("/", 1)[-1]
                if s["type"] == "file" and sname.lower().endswith((".jpg", ".png", ".jpeg", ".tga")):
                    http_download(
                        f"{base}/textures/{name}/{sname}",
                        asset_dir / "textures" / name / sname,
                    )
    return usda


def render(usda: Path, png: Path) -> tuple[bool, float, int, str]:
    """Run test_headless_render. Returns (ok, ms, n_meshes, err)."""
    ppm = png.with_suffix(".ppm")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [str(RENDERER), str(usda), str(ppm)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        return False, 0.0, 0, f"timeout: {e}"
    ms = (time.perf_counter() - t0) * 1000.0
    if result.returncode != 0 or not ppm.exists():
        tail = (result.stderr or result.stdout or "")[-200:].strip()
        return False, ms, 0, f"rc={result.returncode}: {tail}"

    # Parse mesh count from stderr ("loaded N meshes from ...").
    n_meshes = 0
    for line in result.stderr.splitlines():
        if "loaded " in line and " meshes from " in line:
            try:
                n_meshes = int(line.split("loaded ")[1].split()[0])
            except (ValueError, IndexError):
                pass
            break

    try:
        img = Image.open(ppm).convert("RGB")
        img.thumbnail((480, 270), Image.Resampling.LANCZOS)
        img.save(png, optimize=True)
        ppm.unlink()
    except Exception as e:
        return False, ms, n_meshes, f"PPM→PNG: {e}"

    return True, ms, n_meshes, ""


def make_grid(thumbs: list[Path], out: Path, cols: int = 5) -> None:
    """Lay out thumbnails into a single grid PNG."""
    if not thumbs:
        return
    images = [Image.open(p) for p in thumbs]
    tw, th = max(im.width for im in images), max(im.height for im in images)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (tw * cols, th * rows), (240, 240, 240))
    for i, im in enumerate(images):
        r, c = i // cols, i % cols
        grid.paste(im, (c * tw, r * th))
    grid.save(out, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=50,
                    help="Max categories to sweep (default 50)")
    ap.add_argument("--category-prefix", default="",
                    help="Only include categories whose name starts with this string.")
    args = ap.parse_args()

    if not RENDERER.exists():
        print(f"renderer binary missing at {RENDERER} — build it first.", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print("listing categories…")
    cats = list_categories()
    if args.category_prefix:
        cats = [c for c in cats if c.startswith(args.category_prefix)]
    cats = cats[: args.limit]
    print(f"  {len(cats)} categories will be tested")

    results: list[Result] = []
    thumbs: list[Path] = []
    for i, cat in enumerate(cats, 1):
        print(f"[{i:3d}/{len(cats)}] {cat}")
        asset = pick_asset(cat)
        if not asset:
            results.append(Result(cat, "?", "no_asset",
                                  error="no asset directory"))
            continue
        usda = download_asset(cat, asset)
        if not usda:
            results.append(Result(cat, asset, "download_fail",
                                  error="download failed"))
            continue
        png = OUT / f"{cat}__{asset}.png"
        ok, ms, nm, err = render(usda, png)
        if ok:
            results.append(Result(cat, asset, "ok", ms=ms, n_meshes=nm,
                                  thumbnail=png.name))
            thumbs.append(png)
            print(f"    ok  {ms:6.0f} ms, {nm} meshes")
        else:
            results.append(Result(cat, asset, "render_fail", ms=ms, error=err))
            print(f"    FAIL {err[:80]}")

    # Write summary + grid
    (OUT / "results.json").write_text(json.dumps(
        [asdict(r) for r in results], indent=2))
    make_grid(thumbs, OUT / "_grid.png")

    ok = sum(1 for r in results if r.status == "ok")
    print(f"\nsummary: {ok}/{len(results)} OK")
    print(f"  thumbnails: {OUT}")
    print(f"  grid:       {OUT / '_grid.png'}")
    print(f"  json:       {OUT / 'results.json'}")
    return 0 if ok == len(results) else (0 if ok > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
