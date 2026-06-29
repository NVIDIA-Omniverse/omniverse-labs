#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile an anchor template across a grid to hit a target composed prim count.

Use case: after you've placed an initial sketch (via `densify_zones.py`,
`random_fill.py`, or any other path), you want to replicate the whole
thing on a non-overlapping grid until the realized USD has roughly N
composed prims. Common for benchmarks against synth-only packs (pack.json
with empty `archetypes` → every placement becomes a Cube+Xform at realize
time, 2 prims/placement) and for "give me N copies of this layout" runs
against real-asset packs.

Algorithm:
  1. placements_per_tile = anchor placements currently in the session
  2. tiles_needed        = ceil(target_prim_count / prims_per_placement
                                / placements_per_tile)
  3. roughly square grid = ceil(sqrt(tiles)) cols × ceil(tiles/cols) rows
  4. place_many of (rows*cols - 1) translated copies of the anchor
  5. caller realize()s — onMiss=synth for synth packs, default for
     real-asset packs whose placements are already bound

The anchor itself stays in the (0,0) slot; tiles fill the remaining
slots column-by-column then row-by-row. Same anchor + same args →
identical output (no RNG).

Usage:
    python tile_template.py \
        --session http://127.0.0.1:8766 \
        --target-prims 100000 \
        [--prims-per-placement 2]   (synth default = Cube+Xform = 2;
                                     for real-asset packs pass the
                                     average composed-prim count per
                                     placement so the math accounts for
                                     the asset's internal hierarchy)
        [--gap 2.0]                 (meters between tiles; default 2)
        [--onCollision skip|reject] (default skip — the grid layout
                                     shouldn't overlap, but this guards
                                     against bad gap values)

Prints the chosen grid + the place_many response. Realize separately
once the grid is laid; the script doesn't realize on your behalf.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


def _post(session: str, tool: str, body: dict) -> dict:
    url = f"{session.rstrip('/')}/tool/{tool}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{tool}: HTTP {e.code} — {e.read().decode(errors='replace')}")


def _anchor_footprint(anchor: list[dict]) -> tuple[float, float]:
    xs = [p["posM"][0] for p in anchor]
    ys = [p["posM"][1] for p in anchor]
    ws = [p["slotM"][0] for p in anchor]
    ds = [p["slotM"][1] for p in anchor]
    n = len(anchor)
    fp_w = max(xs[i] + ws[i] / 2 for i in range(n)) - min(xs[i] - ws[i] / 2 for i in range(n))
    fp_d = max(ys[i] + ds[i] / 2 for i in range(n)) - min(ys[i] - ds[i] / 2 for i in range(n))
    return fp_w, fp_d


def tile_template(session: str, target_prims: int, prims_per_placement: float,
                  gap_m: float, on_collision: str) -> dict:
    anchor = _post(session, "query_stage_graph", {})
    if not anchor:
        raise SystemExit(
            "[tile_template] no anchor placements in session — place the "
            "tile-able template first (e.g. via densify_zones.py).")
    placements_per_tile = len(anchor)
    tiles_needed = max(1, math.ceil(
        target_prims / max(prims_per_placement, 1e-9) / placements_per_tile))
    cols = max(1, math.ceil(math.sqrt(tiles_needed)))
    rows = max(1, math.ceil(tiles_needed / cols))
    fp_w, fp_d = _anchor_footprint(anchor)

    batch: list[dict] = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue  # anchor already occupies slot (0, 0)
            dx = c * (fp_w + gap_m)
            dy = r * (fp_d + gap_m)
            for p in anchor:
                batch.append({
                    "archetype": p["archetype"],
                    "posM": [p["posM"][0] + dx, p["posM"][1] + dy, p["posM"][2]],
                    "slotM": p["slotM"],
                    "yawDeg": p["yawDeg"],
                    "id": f"{p['id']}__tile_{r}_{c}",
                })

    summary = {
        "anchorPlacements": placements_per_tile,
        "anchorFootprintM": [fp_w, fp_d],
        "targetPrims": target_prims,
        "tilesNeeded": tiles_needed,
        "gridRows": rows,
        "gridCols": cols,
        "tileCopiesAuthored": len(batch) // placements_per_tile,
        "placementsEmitted": len(batch),
    }
    if not batch:
        summary["placeMany"] = {"placed": 0, "rejected": 0, "note": "anchor alone fits the target"}
        return summary

    resp = _post(session, "place_many",
                 {"placements": batch, "onCollision": on_collision})
    summary["placeMany"] = resp
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="http://127.0.0.1:8766",
                    help="Session HTTP URL (default %(default)s)")
    ap.add_argument("--target-prims", type=int, required=True,
                    help="Target composed prim count after realize")
    ap.add_argument("--prims-per-placement", type=float, default=2.0,
                    help="Expected composed prims per placement at realize time "
                         "(default %(default)s = Cube+Xform for synth packs; "
                         "pass higher values when binding to real assets)")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="Gap (m) between tile copies on the grid (default %(default)s)")
    ap.add_argument("--onCollision", choices=["skip", "reject"], default="skip",
                    help="Behavior when a tile copy collides with existing content "
                         "(default %(default)s)")
    args = ap.parse_args()
    out = tile_template(args.session, args.target_prims,
                        args.prims_per_placement, args.gap, args.onCollision)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
