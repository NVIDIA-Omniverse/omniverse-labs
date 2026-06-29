#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit a running sketch-stage session for OBB overlaps and slot vs asset
size mismatches.

The session's collision gate (engine/spatial_index.py) rejects placements
whose slot OBBs overlap any existing placement's slot OBB. If a user reports
"I see overlapping objects" the cause is usually one of:

  1. Two slot OBBs do overlap — gate bug or gate-bypass at author time.
  2. Slot OBBs don't overlap, but the rendered asset bbox is larger than
     its declared slot, so adjacent placements (slot edges touching) have
     rendered geometry that visibly overlaps.
  3. The asset's pivot doesn't match bbox-center-floor, so it renders
     drifted from the slot center; if drift > (neighbor-slot-edge - asset-
     edge), the rendered geometry visibly overlaps a neighbor's footprint.

This script checks all three by:
  - Pulling every placement from /tool/query_stage_graph.
  - Brute-force-or-grid-checking pairwise OBB overlap (case 1).
  - For each unique (archetype, assetPath) pair, opening the asset USD with
    usd-core and measuring the world-axis bbox after the engine's Y→Z
    rotation if upAxis is Y. Compares to the slot dims (case 2).

Run after a realize when something looks wrong. Reports the first 20
offending pairs and per-archetype slot/asset-bbox table.

Usage:
    python check_overlaps.py [--session http://127.0.0.1:8766] \\
        [--pack /path/to/pack] [--limit-pairs 20] [--full]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.request
from collections import defaultdict


def _http(url: str, payload: dict | None = None, timeout: float = 600.0):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def _obb_overlap(a_pos, a_slot, a_yaw, b_pos, b_slot, b_yaw) -> bool:
    """Mirror of engine/spatial_index.py::_obb_overlap (yaw-only SAT)."""
    a_hh = a_slot[2] / 2
    b_hh = b_slot[2] / 2
    if abs(a_pos[2] - b_pos[2]) > a_hh + b_hh:
        return False
    a_r = math.radians(a_yaw)
    b_r = math.radians(b_yaw)
    a_x = (math.cos(a_r), math.sin(a_r))
    a_y = (-math.sin(a_r), math.cos(a_r))
    b_x = (math.cos(b_r), math.sin(b_r))
    b_y = (-math.sin(b_r), math.cos(b_r))
    a_hw, a_hd = a_slot[0] / 2, a_slot[1] / 2
    b_hw, b_hd = b_slot[0] / 2, b_slot[1] / 2
    dx = b_pos[0] - a_pos[0]
    dy = b_pos[1] - a_pos[1]
    for ax, ay in (a_x, a_y, b_x, b_y):
        t = abs(dx * ax + dy * ay)
        ra = (a_hw * abs(ax * a_x[0] + ay * a_x[1])
              + a_hd * abs(ax * a_y[0] + ay * a_y[1]))
        rb = (b_hw * abs(ax * b_x[0] + ay * b_x[1])
              + b_hd * abs(ax * b_y[0] + ay * b_y[1]))
        if t > ra + rb:
            return False
    return True


def _aabb_xy(pos, slot, yaw_deg):
    cx, cy = pos[0], pos[1]
    w, d = slot[0], slot[1]
    hw, hd = w / 2, d / 2
    c, s = abs(math.cos(math.radians(yaw_deg))), abs(math.sin(math.radians(yaw_deg)))
    rw = hw * c + hd * s
    rd = hw * s + hd * c
    return (cx - rw, cy - rd, cx + rw, cy + rd)


def _grid_pairwise(placements, cell=10.0):
    """Yield (i, j) candidate pairs whose AABBs share at least one grid cell."""
    grid = defaultdict(list)
    aabbs = [_aabb_xy(p["posM"], p["slotM"], p.get("yawDeg", 0)) for p in placements]
    for idx, (x0, y0, x1, y1) in enumerate(aabbs):
        i0, j0 = math.floor(x0 / cell), math.floor(y0 / cell)
        i1, j1 = math.floor(x1 / cell), math.floor(y1 / cell)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                grid[(i, j)].append(idx)
    seen: set[tuple[int, int]] = set()
    for ids in grid.values():
        ids.sort()
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (ids[a], ids[b])
                if key in seen:
                    continue
                seen.add(key)
                yield key


def _measure_asset_world_bbox(asset_path: str, up_axis: str, mpu: float):
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None
    if not os.path.exists(asset_path):
        return None
    stage = Usd.Stage.Open(asset_path)
    if stage is None:
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = cache.ComputeWorldBound(stage.GetPseudoRoot())
    r = bbox.ComputeAlignedRange()
    if r.IsEmpty():
        return None
    mn = r.GetMin()
    mx = r.GetMax()
    # Asset-frame extents in meters (apply mpu).
    ex = (mx[0] - mn[0]) * mpu
    ey = (mx[1] - mn[1]) * mpu
    ez = (mx[2] - mn[2]) * mpu
    if up_axis.upper() == "Y":
        # Engine rotates +90° about X: asset Y→world Z, asset Z→world Y.
        return (ex, ez, ey)  # (worldW, worldD, worldH)
    return (ex, ey, ez)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", default="http://127.0.0.1:8766")
    ap.add_argument("--pack", default=None,
                    help="Pack root for asset bbox check. If omitted, the "
                         "asset/slot mismatch check is skipped.")
    ap.add_argument("--limit-pairs", type=int, default=20,
                    help="Max OBB-overlap pairs to print (default 20)")
    ap.add_argument("--full", action="store_true",
                    help="Print every overlapping pair (overrides --limit-pairs)")
    args = ap.parse_args(argv)

    base = args.session.rstrip("/")
    placements = _http(base + "/tool/query_stage_graph", {})
    if not isinstance(placements, list):
        raise SystemExit(f"query_stage_graph returned: {placements!r}")
    # Normalize slot dict→list (the session emits dict for some shapes).
    for p in placements:
        s = p.get("slotM")
        if isinstance(s, dict):
            p["slotM"] = [float(s["widthM"]), float(s["depthM"]), float(s["heightM"])]
        else:
            p["slotM"] = [float(x) for x in s]
        p["posM"] = [float(x) for x in p.get("posM", [0, 0, 0])]
        p["yawDeg"] = float(p.get("yawDeg", 0))
    print(f"placements: {len(placements)}", file=sys.stderr)

    # --- Case 1: slot OBBs that overlap each other --- #
    overlapping: list[tuple[int, int, float]] = []
    for i, j in _grid_pairwise(placements):
        a, b = placements[i], placements[j]
        if _obb_overlap(a["posM"], a["slotM"], a["yawDeg"],
                        b["posM"], b["slotM"], b["yawDeg"]):
            # XY center distance for context
            dx = a["posM"][0] - b["posM"][0]
            dy = a["posM"][1] - b["posM"][1]
            overlapping.append((i, j, math.hypot(dx, dy)))
    print(f"slot-OBB overlapping pairs: {len(overlapping)}")
    if overlapping:
        show = overlapping if args.full else overlapping[: args.limit_pairs]
        for i, j, dist in show:
            a, b = placements[i], placements[j]
            print(f"  {a['archetype']:>14}@{tuple(round(x,2) for x in a['posM'])}"
                  f" slot {a['slotM']} yaw {a['yawDeg']}  vs  "
                  f"{b['archetype']:>14}@{tuple(round(x,2) for x in b['posM'])}"
                  f" slot {b['slotM']} yaw {b['yawDeg']}  dist={dist:.2f}m")

    # --- Case 2: per-(archetype, asset) slot vs asset world bbox --- #
    if args.pack:
        try:
            pack = json.load(open(os.path.join(args.pack, "pack.json")))
        except OSError:
            print(f"  pack.json not readable under {args.pack}", file=sys.stderr)
            return 0
        up_axis = pack.get("upAxis", "Z")
        mpu = float(pack.get("metersPerUnit", 1.0))
        per_pair: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
        for p in placements:
            ap_ = p.get("assetPath") or ""
            if not ap_:
                continue
            per_pair[(p["archetype"], ap_)].append(p["slotM"])
        print("\nper-(archetype, asset) slot vs world-axis asset bbox "
              f"(upAxis={up_axis}, mpu={mpu}):")
        print(f"  {'archetype':>14} {'slot W×D×H':>20} "
              f"{'asset W×D×H':>20}  {'slot/asset':>12}  asset")
        for (arch, ap_), slots in sorted(per_pair.items()):
            full = ap_ if os.path.isabs(ap_) else os.path.join(args.pack, ap_)
            bbox = _measure_asset_world_bbox(full, up_axis, mpu)
            wd = (statistics.median(s[0] for s in slots),
                  statistics.median(s[1] for s in slots),
                  statistics.median(s[2] for s in slots))
            if bbox is None:
                bbox_str = "?"
                ratio_str = "?"
            else:
                bbox_str = f"{bbox[0]:.2f}×{bbox[1]:.2f}×{bbox[2]:.2f}"
                ratios = [wd[k] / bbox[k] if bbox[k] > 0 else 0 for k in range(3)]
                ratio_str = f"{ratios[0]:.1f}×/{ratios[1]:.1f}×/{ratios[2]:.1f}×"
            print(f"  {arch:>14} {wd[0]:>6.2f}×{wd[1]:>5.2f}×{wd[2]:>5.2f}  "
                  f"{bbox_str:>20}  {ratio_str:>12}  "
                  f"{os.path.basename(ap_)}  n={len(slots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
