#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Densify rectangular regions of a running sketch-stage session with extra
placements of named archetypes, on a deterministic row-major grid.

Domain-agnostic: no archetype names are hard-coded. The slot for each
archetype is inferred from the session's existing placements (median of
their slotM), so this works for any scenario. If the archetype has no
existing placements in the session, pass `--slot <arch>=w,d,h` to provide
canonical dims.

Region sources: each name in `--zones` resolves to (in order)
  1. a `--region NAME=ox,oy,w,d` literal (takes precedence), OR
  2. a zone returned by the live session's `query_zones`.

This lets the script work for templates with no zones (empty-canvas mode,
custom anchors with no zone prims): partition the canvas into theme
rectangles via `--region` and pass their names in `--zones`.

Usage:
    python densify_zones.py \
        --session http://127.0.0.1:8766 \
        --zones '{"<region_or_zone>":{"<archetype>":<count>, ...}, ...}'
        [--region <name>=<ox>,<oy>,<w>,<d>]   (repeatable)
        [--slot <arch>=<w>,<d>,<h>]            (repeatable)
        [--yaws 0,90,180,270]

For each (region, archetype, count):
  1. Resolve slot dims (inferred from session, or via --slot).
  2. Walk a regular grid over the region with cell pitch = max(slotW, slotD)
     and an edge buffer of half that pitch so the slot's OBB stays inside
     the region regardless of yaw.
  3. Emit the first `count` cells in row-major order (y outer, x inner).
  4. POST /tool/place_many. The session's collision gate culls overlaps;
     `rejected` in the response counts placements that hit existing content.

Same session state + same args → identical output (no RNG).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request


def _http(url: str, payload: dict | None = None, timeout: float = 600.0):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def _parse_slot_overrides(items: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--slot expects arch=w,d,h (got {item!r})")
        arch, vals = item.split("=", 1)
        parts = [float(x) for x in vals.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--slot {arch!r} expects 3 floats, got {parts}")
        out[arch.strip()] = parts
    return out


def _parse_region_overrides(items: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--region expects NAME=ox,oy,w,d (got {item!r})")
        name, vals = item.split("=", 1)
        parts = [float(x) for x in vals.split(",")]
        if len(parts) != 4:
            raise SystemExit(
                f"--region {name!r} expects 4 floats ox,oy,w,d, got {parts}"
            )
        ox, oy, w, d = parts
        if w <= 0 or d <= 0:
            raise SystemExit(f"--region {name!r}: width/depth must be > 0")
        out[name.strip()] = {
            "id": name.strip(),
            "originWorldM": [ox, oy, 0.0],
            "boundsM": {"widthM": w, "depthM": d, "heightM": 0.0},
            "_source": "region",
        }
    return out


def _slot_as_list(s) -> tuple[float, float, float] | None:
    if isinstance(s, dict):
        if {"widthM", "depthM", "heightM"} <= s.keys():
            return (float(s["widthM"]), float(s["depthM"]), float(s["heightM"]))
        return None
    if isinstance(s, (list, tuple)) and len(s) == 3:
        return (float(s[0]), float(s[1]), float(s[2]))
    return None


def _infer_slot(base: str, arch: str) -> list[float]:
    """Median slotM across existing placements of this archetype."""
    placements = _http(base + "/tool/query_stage_graph", {"archetype": arch})
    if not isinstance(placements, list) or not placements:
        raise SystemExit(
            f"archetype {arch!r} has no existing placements in the session; "
            f"pass --slot {arch}=w,d,h to provide canonical dims"
        )
    triples = [t for p in placements for t in [_slot_as_list(p.get("slotM"))] if t]
    if not triples:
        raise SystemExit(
            f"archetype {arch!r} placements expose no parseable slotM; "
            f"pass --slot {arch}=w,d,h"
        )
    w = statistics.median(t[0] for t in triples)
    d = statistics.median(t[1] for t in triples)
    h = statistics.median(t[2] for t in triples)
    print(f"  inferred slot for {arch!r}: [{w:.3f},{d:.3f},{h:.3f}] "
          f"from {len(triples)} existing placements", file=sys.stderr)
    return [w, d, h]


def _grid_cells(origin_x: float, origin_y: float,
                width: float, depth: float,
                slot_w: float, slot_d: float) -> list[tuple[float, float]]:
    """Row-major cell centers, edge-buffered by max(slot_w, slot_d) / 2 so the
    OBB fits inside the zone for any yaw."""
    # Engine SAT uses strict `>` for separation, so adjacent cells at pitch
    # exactly equal to the slot dim are *touching* → flagged as overlapping →
    # rejected by the collision gate. Add a tiny epsilon so neighboring cells
    # are guaranteed strictly separated.
    pitch = max(slot_w, slot_d) * 1.001
    buf = pitch / 2.0
    xs: list[float] = []
    x = origin_x + buf
    stop_x = origin_x + width - buf
    while x <= stop_x:
        xs.append(x)
        x += pitch
    ys: list[float] = []
    y = origin_y + buf
    stop_y = origin_y + depth - buf
    while y <= stop_y:
        ys.append(y)
        y += pitch
    return [(cx, cy) for cy in ys for cx in xs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", default="http://127.0.0.1:8766",
                    help="HTTP base URL of the sketch session")
    ap.add_argument("--zones", required=True,
                    help='JSON: {"<zone_id>":{"<archetype>":<count>}, ...}')
    ap.add_argument("--slot", action="append", default=[],
                    help="Override inferred slot: arch=w,d,h (repeatable)")
    ap.add_argument("--region", action="append", default=[],
                    help="Declare an ad-hoc rectangle when the template has "
                         "no zones (or to override one): NAME=ox,oy,w,d "
                         "(repeatable). Names in --zones resolve to "
                         "regions first, then to session zones.")
    ap.add_argument("--yaws", default="0",
                    help="Comma-separated yaw choices in degrees, cycled "
                         "deterministically across cells (default: 0)")
    args = ap.parse_args(argv)

    base = args.session.rstrip("/")
    slot_overrides = _parse_slot_overrides(args.slot)
    region_overrides = _parse_region_overrides(args.region)
    yaws = [float(y.strip()) for y in args.yaws.split(",") if y.strip()]
    if not yaws:
        yaws = [0.0]
    zone_attempts: dict[str, dict[str, int]] = json.loads(args.zones)

    zlist = _http(base + "/tool/query_zones", {})
    if not isinstance(zlist, list):
        raise SystemExit(f"query_zones returned unexpected payload: {zlist!r}")
    # --region takes precedence over a session zone of the same name.
    regions_by_id = {z["id"]: z for z in zlist}
    regions_by_id.update(region_overrides)
    if not regions_by_id:
        raise SystemExit(
            "no zones in session AND no --region declared; pass "
            "--region NAME=ox,oy,w,d for each rectangle you want to fill"
        )

    slot_cache: dict[str, list[float]] = {}

    def slot_for(arch: str) -> list[float]:
        if arch in slot_cache:
            return slot_cache[arch]
        if arch in slot_overrides:
            slot_cache[arch] = slot_overrides[arch]
        else:
            slot_cache[arch] = _infer_slot(base, arch)
        return slot_cache[arch]

    candidates: list[dict] = []
    for region_id, mix in zone_attempts.items():
        if region_id not in regions_by_id:
            raise SystemExit(
                f"unknown region/zone {region_id!r}; have "
                f"{sorted(regions_by_id)} (declare more with --region)"
            )
        z = regions_by_id[region_id]
        ox, oy, _ = z["originWorldM"]
        zw = z["boundsM"]["widthM"]
        zd = z["boundsM"]["depthM"]
        for arch, count in mix.items():
            slot = slot_for(arch)
            cells = _grid_cells(ox, oy, zw, zd, slot[0], slot[1])
            if len(cells) < count:
                print(f"  note: region {region_id!r}+{arch!r} grid yields "
                      f"{len(cells)} cells; capping requested count from "
                      f"{count} to {len(cells)}", file=sys.stderr)
                count = len(cells)
            for i, (cx, cy) in enumerate(cells[:count]):
                yaw = yaws[i % len(yaws)]
                candidates.append({
                    "archetype": arch,
                    "posM": [round(cx, 3), round(cy, 3), 0.0],
                    "slotM": slot,
                    "yawDeg": yaw,
                })

    print(f"attempting {len(candidates)} candidate placements "
          f"({len(zone_attempts)} zones)", file=sys.stderr)
    resp = _http(base + "/tool/place_many", {"placements": candidates})
    print(json.dumps({
        "placed": resp.get("placed"),
        "rejected": resp.get("rejected"),
        "attempted": len(candidates),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
