#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive scripts/compare.py over the warehouse asset suite, then diff
each freshly captured PNG against the committed baseline in
docs/compare/. Anything with a non-trivial RMSE delta is a candidate
regression to investigate.

Usage:
    python scripts/run_extensive_compare.py [name1 name2 ...]

If no asset names are passed, all entries in ASSETS are run. Output
captures land under <NUSD_COMPARE_OUT_DIR or /tmp/compare_run/> and
the committed baselines in docs/compare/ are NEVER touched.

Required env: DISPLAY (default :1), XAUTHORITY, NUSD_USD_INSTALL
(only if you want the ovrtx leg). NUSD_OVGEAR_VENV defaults to
<repo>/../.venv-ovgear.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
WAREHOUSE = Path(os.environ.get("NUSD_WAREHOUSE", str(Path.home() / "warehouse")))


# (short name, USD path under WAREHOUSE/Props/general/<dir>/<file>.usd)
ASSETS: list[tuple[str, str]] = [
    ("barrier",     "SM_PlasticWallBarrier_A03_Orange_01/SM_PlasticWallBarrier_A03_Orange_01.usd"),
    ("greencrate",  "SM_Crate_A20_Green_01/SM_Crate_A20_Green_01.usd"),
    ("guardrail",   "SM_GuardRail_A09_01/SM_GuardRail_A09_01.usd"),
    ("ladder",      "SM_AluminiumMultiPurposeLadder_A03_01/SM_AluminiumMultiPurposeLadder_A03_01.usd"),
    ("redcrate",    "SM_Crate_A19_Red_01/SM_Crate_A19_Red_01.usd"),
    ("redtrucker",  "SM_TiltTruck_A04_Red_01/SM_TiltTruck_A04_Red_01.usd"),
    ("screwpail",   "SM_ScrewTopPail_A02_GlossyWhite_01/SM_ScrewTopPail_A02_GlossyWhite_01.usd"),
    ("wetsign",     "SM_WetFloorSign_A02_01/SM_WetFloorSign_A02_01.usd"),
    ("yellowpail",  "SM_PlasticPail_A02_Yellow_01/SM_PlasticPail_A02_Yellow_01.usd"),
    # The earlier 9-asset gallery — exercise both sets when present.
    ("forklift",    "SM_Forklift_A01_Orange_01/SM_Forklift_A01_Orange_01.usd"),
    ("crate",       "SM_Crate_A20_Orange_01/SM_Crate_A20_Orange_01.usd"),
    ("crate2",      "SM_Crate_A20_Gray_01/SM_Crate_A20_Gray_01.usd"),
    ("box",         "SM_Box_A01_Blue_01/SM_Box_A01_Blue_01.usd"),
    ("bucket",      "SM_UtilityBucket_A02_Gray_01/SM_UtilityBucket_A02_Gray_01.usd"),
    ("tilttruck",   "SM_TiltTruck_A05_Gray_01/SM_TiltTruck_A05_Gray_01.usd"),
    ("packingtable","SM_DeluxePackingTable_A01_01/SM_DeluxePackingTable_A01_01.usd"),
]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    da = a.astype(np.float32) / 255.0
    db = b.astype(np.float32) / 255.0
    return math.sqrt(float(np.mean((da - db) ** 2)))


def diff_against_baseline(name: str, out_dir: Path) -> dict[str, str]:
    """Return per-leg status string for the captured PNGs."""
    results: dict[str, str] = {}
    for leg in ("nanousd", "ovrtx", "compare"):
        captured = out_dir / f"{name}_{leg}.png"
        baseline = REPO / "docs" / "compare" / f"{name}_{leg}.png"
        if not captured.exists():
            results[leg] = "no_capture"
            continue
        if not baseline.exists():
            results[leg] = f"no_baseline (captured ok, {captured.stat().st_size}B)"
            continue
        try:
            a = np.array(Image.open(captured).convert("RGB"))
            b = np.array(Image.open(baseline).convert("RGB"))
        except Exception as e:
            results[leg] = f"image_load_error: {e}"
            continue
        if a.shape != b.shape:
            results[leg] = (f"DIM_MISMATCH "
                            f"new={a.shape[1]}x{a.shape[0]} "
                            f"baseline={b.shape[1]}x{b.shape[0]}")
            continue
        r = rmse(a, b)
        # Threshold loose enough for natural FPS-counter / cursor jitter
        # in the captured window (the title-bar contains a live FPS
        # counter); but tight enough that a real rendering regression
        # will pop. The captured PNGs are full-window screenshots with
        # imgui dock state and stage browser hierarchy — those are
        # deterministic across runs, so 0.01 RMSE is a fair gate.
        gate = 0.01
        results[leg] = f"RMSE={r:.4f} {'ok' if r < gate else 'CHANGED'}"
    return results


def run_one(name: str, usd_path: Path, out_dir: Path) -> int:
    print(f"\n=== {name} ===  {usd_path}", flush=True)
    if not usd_path.exists():
        print(f"  SKIP: USD not found at {usd_path}")
        return 2
    env = os.environ.copy()
    env["NUSD_COMPARE_OUT_DIR"] = str(out_dir)
    cmd = [sys.executable, str(REPO / "scripts" / "compare.py"),
           name, str(usd_path), "--label", name]
    try:
        rc = subprocess.run(cmd, env=env, timeout=90).returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 90s")
        return 1
    diffs = diff_against_baseline(name, out_dir)
    for leg, status in diffs.items():
        marker = "    "
        if "CHANGED" in status or "DIM_MISMATCH" in status or "no_capture" in status:
            marker = " >> "
        print(f"  {marker}{leg:<8} {status}")
    return rc


def main(argv: Iterable[str]) -> int:
    requested = list(argv)
    if requested:
        selected = [(n, p) for (n, p) in ASSETS if n in requested]
        missing = [n for n in requested if n not in [a[0] for a in ASSETS]]
        if missing:
            print(f"unknown asset names: {missing}")
            print(f"available: {[a[0] for a in ASSETS]}")
            return 2
    else:
        selected = ASSETS

    out_dir = Path(os.environ.get(
        "NUSD_COMPARE_OUT_DIR", "/tmp/compare_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"capturing to: {out_dir}")
    print(f"baselines:    {REPO / 'docs' / 'compare'}")
    print(f"assets:       {len(selected)}")

    fails = 0
    summary: list[tuple[str, dict[str, str]]] = []
    for name, rel in selected:
        usd = WAREHOUSE / "Props" / "general" / rel
        rc = run_one(name, usd, out_dir)
        if rc != 0:
            fails += 1
        summary.append((name, diff_against_baseline(name, out_dir)))

    # Final summary table
    print("\n=== SUMMARY ===")
    print(f"{'Asset':<14} {'nanousd':<26} {'ovrtx':<26} {'compare':<26}")
    print("-" * 100)
    regressions = []
    for name, diffs in summary:
        n = diffs.get("nanousd", "n/a")
        o = diffs.get("ovrtx",   "n/a")
        c = diffs.get("compare", "n/a")
        print(f"{name:<14} {n:<26} {o:<26} {c:<26}")
        if "CHANGED" in n or "CHANGED" in c or "DIM_MISMATCH" in n:
            regressions.append(name)
    if regressions:
        print(f"\nREGRESSIONS to investigate: {regressions}")
    return 1 if fails or regressions else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
