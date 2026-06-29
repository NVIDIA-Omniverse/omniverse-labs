#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bulk on-surface placement for a running sketch-stage session.

Symmetric counterpart to ``random_fill.py``. Where ``random_fill`` walks a
floor-cell grid and POSTs ``place_many`` (flat placements at z=0), this
script walks the runtime **surface registry** (top faces of existing
placements that declare ``semantics.surfaces`` in pack.json) and POSTs
``place_on`` per surface.

Use this when the user wants things ON other things — *"fill the rack
shelves with boxes"*, *"plates on the dining table"*, *"monitors on the
desks"*. Floor placement is the wrong tool for that.

Workflow:

  1. ``GET /tool/query_surfaces`` filtered by ``--surface-label``,
     optional ``--region``, optional ``--min-free-area-m2``.
  2. Optionally sort/shuffle surfaces deterministically by seed.
  3. For each surface, decide how many on-surface placements it should
     hold based on its free area and the chosen archetype's slot
     footprint. Default is to fill up to ``--per-surface-max`` (or fit-
     to-area, whichever is smaller).
  4. Per slot on the surface, pick an archetype from the weighted mix,
     pick a localXY (uniform random inside the surface's footprint with
     margins), pick a yaw, POST ``/tool/place_on``. The session's
     place_on already does sibling-collision + region.allowedArchetypes
     filtering.

Usage:
    python surface_fill.py \\
        --session http://127.0.0.1:8766 \\
        --surface-label rack_top \\
        --archetypes '{"Boxes": 6, "Containers": 2}' \\
        --slot 'Boxes=1.0,1.2,1.0' \\
        --slot 'Containers=3.0,2.1,2.1' \\
        --per-surface-max 4 \\
        --seed 42

Common patterns:
    # boxes on rack shelves
    --surface-label shelf --archetypes '{"Boxes": 1}'

    # cups/bowls on a counter
    --surface-label counter --archetypes '{"cup": 4, "fruit_bowl": 1}' \\
                  --per-surface-max 8

    # only certain surfaces (filter by region from query_zones)
    --region storage_aisles
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import urllib.request


def _http(url: str, payload: dict | None = None, timeout: float = 120.0):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def _parse_slot_overrides(items: list[str]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--slot expects arch=w,d,h (got {item!r})")
        arch, vals = item.split("=", 1)
        parts = [float(x) for x in vals.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"--slot {arch!r} expects 3 floats, got {parts}")
        out[arch.strip()] = (parts[0], parts[1], parts[2])
    return out


def _slot_capacity(surface_w: float, surface_d: float,
                   slot_w: float, slot_d: float) -> int:
    """How many slot-sized items theoretically fit on a single surface."""
    if slot_w <= 0 or slot_d <= 0:
        return 0
    return max(0, int(surface_w / slot_w) * int(surface_d / slot_d))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", default="http://127.0.0.1:8766",
                    help="HTTP base URL of the sketch session")
    ap.add_argument("--surface-label", required=True,
                    help="Surface label to target (e.g. shelf, counter, "
                         "rack_top, top). Comes from pack.semantics.surfaces.")
    ap.add_argument("--archetypes", required=True,
                    help='JSON: {"<arch>": <weight>, ...} — archetypes that '
                         'will be placed on each matching surface. Weights '
                         'are per-pick, not absolute counts.')
    ap.add_argument("--region", default=None,
                    help="Restrict to surfaces whose owner is inside this "
                         "region id (see query_zones)")
    ap.add_argument("--min-free-area-m2", type=float, default=0.0,
                    help="Skip surfaces with less than this much free area")
    ap.add_argument("--per-surface-max", type=int, default=4,
                    help="Cap of placements per surface (default 4). "
                         "Actual count is min(this, area-derived capacity).")
    ap.add_argument("--slot", action="append", default=[],
                    help="Override slot dims: arch=w,d,h (repeatable). "
                         "REQUIRED for archetypes not yet in the session.")
    ap.add_argument("--yaws", default="0,90,180,270",
                    help="Comma-separated yaw choices in degrees (default "
                         "0,90,180,270)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for surface order + archetype + yaw choice")
    ap.add_argument("--margin-m", type=float, default=0.02,
                    help="Edge margin around surface footprint when sampling "
                         "localXY (default 0.02 m)")
    ap.add_argument("--layout", choices=["grid", "random"], default="grid",
                    help="grid (default): deterministic row-major lattice "
                         "sized to slot dims; random: legacy uniform sampling "
                         "(many siblings collide after the 2nd-3rd placement). "
                         "Grid mode reliably hits per-surface-max when the "
                         "footprint fits; random mode is useful for organic "
                         "scatter (plates on a table).")
    args = ap.parse_args(argv)

    base = args.session.rstrip("/")
    archetype_mix = json.loads(args.archetypes)
    if not isinstance(archetype_mix, dict) or not archetype_mix:
        raise SystemExit("--archetypes must be a non-empty {arch: weight} object")
    archs = list(archetype_mix.keys())
    weights = [float(archetype_mix[a]) for a in archs]
    if sum(weights) <= 0:
        raise SystemExit("--archetypes weights must sum > 0")

    slot_overrides = _parse_slot_overrides(args.slot)
    for a in archs:
        if a not in slot_overrides:
            raise SystemExit(
                f"--slot {a}=w,d,h is required for surface_fill (the script "
                f"needs to know the size of each placement to compute per-"
                f"surface capacity and to validate against surface footprint)")
    yaws = [float(y.strip()) for y in args.yaws.split(",") if y.strip()] or [0.0]

    # Step 1: fetch matching surfaces.
    body: dict = {"label": args.surface_label,
                  "minFreeAreaM2": float(args.min_free_area_m2)}
    if args.region:
        body["regionId"] = args.region
    surfaces = _http(base + "/tool/query_surfaces", body)
    if not isinstance(surfaces, list):
        raise SystemExit(f"query_surfaces returned unexpected: {surfaces!r}")
    if not surfaces:
        print(json.dumps({"placed": 0, "rejected": 0, "skipped": 0,
                          "surfacesMatched": 0, "note": "no matching surfaces"},
                         indent=2))
        return 0

    print(f"  matched {len(surfaces)} surface(s) with label={args.surface_label!r}"
          + (f" in region {args.region!r}" if args.region else ""),
          file=sys.stderr)

    rng = random.Random(f"{args.seed}:{args.surface_label}")
    rng.shuffle(surfaces)

    total_placed = 0
    total_rejected = 0
    by_owner_archetype: dict[str, int] = {}
    rejection_reasons: dict[str, int] = {}

    def _grid_slots(fw: float, fd: float, slot_w: float, slot_d: float,
                    margin: float, n_target: int) -> list[tuple[float, float]]:
        """Return up to n_target (lx, ly) positions that fit on the surface
        without overlapping each other, laid out row-major with the long
        axis filled first. Returns fewer positions when the footprint
        can't accommodate n_target.
        """
        half_w = fw / 2 - slot_w / 2 - margin
        half_d = fd / 2 - slot_d / 2 - margin
        if half_w < 0 or half_d < 0:
            return []
        long_axis_is_x = fw >= fd
        # Conservative pitch: slot + a hair so collision-rejection at the
        # engine (which uses (sw+sw')/2 strict inequality) won't catch us.
        pitch_long = slot_w if long_axis_is_x else slot_d
        pitch_short = slot_d if long_axis_is_x else slot_w
        # Step a touch larger than the slot so the engine's strict <
        # collision check (abs(d) < (sw+sw')/2) never fires.
        step_long = pitch_long * 1.01
        step_short = pitch_short * 1.01
        span_long = 2 * (half_w if long_axis_is_x else half_d)
        span_short = 2 * (half_d if long_axis_is_x else half_w)
        n_long = max(1, int(span_long // step_long) + 1)
        n_short = max(1, int(span_short // step_short) + 1)
        # Centre the lattice.
        def _positions(n: int, half: float, step: float) -> list[float]:
            if n == 1:
                return [0.0]
            used = (n - 1) * step
            start = -used / 2
            return [start + i * step for i in range(n) if abs(start + i * step) <= half + 1e-6]
        longs = _positions(n_long, half_w if long_axis_is_x else half_d, step_long)
        shorts = _positions(n_short, half_d if long_axis_is_x else half_w, step_short)
        out: list[tuple[float, float]] = []
        for sh in shorts:
            for lo in longs:
                if long_axis_is_x:
                    out.append((lo, sh))
                else:
                    out.append((sh, lo))
                if len(out) >= n_target:
                    return out
        return out

    # Step 2: per surface, distribute placements.
    for s in surfaces:
        fw, fd = s["footprintM"]
        cap_per_arch = max(_slot_capacity(fw, fd, slot_overrides[a][0],
                                            slot_overrides[a][1])
                            for a in archs)
        target = min(args.per_surface_max, cap_per_arch)
        if target <= 0:
            continue

        # Sub-stream per surface so two runs with the same seed give the
        # same result regardless of surface-list order changes.
        local_rng = random.Random(f"{args.seed}:{s['owner']}:{s['label']}")
        # Pick archetype upfront (uniform mix has one canonical slot per
        # placement; grid uses per-slot dims). For multi-archetype mixes
        # we pick before computing positions so the grid is sized to the
        # chosen archetype.
        positions: list[tuple[str, tuple[float, float]]] = []
        if args.layout == "grid":
            # Generate lattice for the largest-slot archetype; we then
            # downscale per pick if a smaller archetype gets chosen at
            # that slot (the collision math is conservative either way).
            biggest = max(archs, key=lambda a: slot_overrides[a][0] * slot_overrides[a][1])
            slot = slot_overrides[biggest]
            grid = _grid_slots(fw, fd, slot[0], slot[1], args.margin_m, target)
            for pos in grid:
                arch = local_rng.choices(archs, weights=weights, k=1)[0]
                positions.append((arch, pos))
        else:
            for _ in range(target):
                arch = local_rng.choices(archs, weights=weights, k=1)[0]
                slot = slot_overrides[arch]
                half_w = fw / 2 - slot[0] / 2 - args.margin_m
                half_d = fd / 2 - slot[1] / 2 - args.margin_m
                if half_w <= 0 or half_d <= 0:
                    continue
                positions.append((arch, (local_rng.uniform(-half_w, half_w),
                                          local_rng.uniform(-half_d, half_d))))
        for arch, (lx, ly) in positions:
            slot = slot_overrides[arch]
            yaw = local_rng.choice(yaws)
            # Pass topWorldZ to disambiguate multi-shelf owners (e.g. racks
            # with surfaces at z=1/2/3). Surfaces with a single label per
            # owner ignore this; multi-label owners need it to target the
            # right shelf.
            resp = _http(base + "/tool/place_on", {
                "parentPlacementId": s["owner"],
                "surfaceLabel": s["label"],
                "localXY": [round(lx, 4), round(ly, 4)],
                "yawDeg": yaw,
                "archetype": arch,
                "archetypeSizeM": list(slot),
                "regionId": args.region,
                "topWorldZ": s.get("topWorldZ"),
            })
            if resp.get("ok"):
                total_placed += 1
                key = f"{s['ownerArchetype']}.{s['label']} <- {arch}"
                by_owner_archetype[key] = by_owner_archetype.get(key, 0) + 1
            else:
                total_rejected += 1
                reason = (resp.get("rejected") or {}).get("stage", "unknown")
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    print(json.dumps({
        "surfacesMatched": len(surfaces),
        "placed": total_placed,
        "rejected": total_rejected,
        "byOwnerArchetype": by_owner_archetype,
        "rejectionReasons": rejection_reasons,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
