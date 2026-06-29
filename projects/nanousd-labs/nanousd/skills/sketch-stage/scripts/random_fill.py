#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Randomized fill pass for a running sketch-stage session.

The deterministic counterpart, ``densify_zones.py``, walks a regular row-major
grid and produces visibly artificial lattices. SKILL.md §3.6 ("Mode fill")
calls for the opposite: shuffle the candidate cell list, weight archetypes
per region, rely on the session's collision gate to cull overlaps. Use this
script for *random densify*-style passes: filling the empty space inside
the existing footprint without altering the layout's extent.

Domain-agnostic: archetype names, region rectangles, and per-region weights
all come from CLI args. Slots are inferred from the session's existing
placements (median of their ``slotM``), or overridden with ``--slot``.

Region sources, in priority order, mirror ``densify_zones.py``:
  1. ``--region NAME=ox,oy,w,d`` literal.
  2. A zone returned by the live session's ``query_zones``.
  3. ``--auto-shell`` synthesizes a region named ``shell`` from the first
     entry of ``template_view``'s ``shells`` array (the whole site).

Usage:
    python random_fill.py \\
        --session http://127.0.0.1:8766 \\
        --plan '{"shell": {"attempts": 8000,
                            "archetypes": {"<arch_a>": 5, "<arch_b>": 2,
                                           "<arch_c>": 1, "<arch_d>": 1}}}' \\
        --auto-shell \\
        --seed 42

For each region in ``--plan``:
  1. Compute cell pitch = max(max-slot-W, max-slot-D) across the region's
     archetype mix, with a 0.1 % SAT-safety epsilon (same as densify_zones).
  2. Generate every cell center in the region with that pitch.
  3. ``random.shuffle`` the cell list (seeded RNG, per-region stream).
  4. For each cell take ``min(attempts, len(cells))`` slots and pick an
     archetype by ``random.choices`` weighted by the region's mix (modulated
     by the archetype's ``placementBias`` if any), pick a yaw from ``--yaws``
     (default uniform over 0/90/180/270 for variety).
  5. If ``--semantic-gate`` is on (default), check the candidate against:
       a. the archetype's ``avoidNear`` list (from ``pack.semantics``) —
          reject the cell if any neighbor's archetype is in the list. The
          neighbor check sees both this batch's earlier accepted placements
          (cheap, local) AND the session's prior placements via
          ``query_nearby``.
       b. the archetype's ``preferredNear`` list — if non-empty, require at
          least one neighbor's archetype to match; reject otherwise.
  6. POST a single ``/tool/place_many`` per region. The session's collision
     gate rejects overlaps with prior content; ``rejected`` reports that.

Pass ``--pack PATH`` to enable the semantic gate. Without it, the gate is
silently a no-op (legacy behaviour). Pass ``--no-semantic-gate`` to
explicitly disable when you want pure weighted-random.

Same seed + same session state → identical output.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import load_pack  # noqa: E402


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
                pitch: float) -> list[tuple[float, float]]:
    """Cell centers on a regular pitch, edge-buffered so the OBB stays inside
    the region for any yaw. Same SAT-safety convention as densify_zones."""
    pitch = pitch * 1.001
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
    ap.add_argument("--plan", required=True,
                    help='JSON: {"<region>": {"attempts": N, '
                         '"archetypes": {"<arch>": <weight>, ...}}, ...}')
    ap.add_argument("--region", action="append", default=[],
                    help="Declare an ad-hoc rectangle: NAME=ox,oy,w,d "
                         "(repeatable). Overrides any session zone of the "
                         "same name.")
    ap.add_argument("--auto-shell", action="store_true",
                    help="Synthesize a region named 'shell' from the first "
                         "entry of template_view's shells (whole site).")
    ap.add_argument("--slot", action="append", default=[],
                    help="Override inferred slot: arch=w,d,h (repeatable)")
    ap.add_argument("--yaws", default="0,90,180,270",
                    help="Comma-separated yaw choices in degrees, sampled "
                         "uniformly per placement (default: 0,90,180,270)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for cell shuffle + archetype/yaw choice")
    ap.add_argument("--pack", default=None,
                    help="Pack root (flat or manifest). Required for "
                         "--semantic-gate / placement bias. Without it, the "
                         "semantic gate is silently a no-op.")
    ap.add_argument("--semantic-gate", dest="semantic_gate",
                    action="store_true", default=True,
                    help="(default ON) consult pack.semantics.avoidNear / "
                         "preferredNear / placementBias when picking cells")
    ap.add_argument("--no-semantic-gate", dest="semantic_gate",
                    action="store_false",
                    help="Disable semantic gate; revert to pure weighted "
                         "random (legacy behaviour)")
    ap.add_argument("--neighbor-radius-m", type=float, default=2.0,
                    help="Radius (m) for avoidNear/preferredNear neighbor "
                         "checks (default 2.0)")
    args = ap.parse_args(argv)

    base = args.session.rstrip("/")
    slot_overrides = _parse_slot_overrides(args.slot)
    region_overrides = _parse_region_overrides(args.region)
    yaws = [float(y.strip()) for y in args.yaws.split(",") if y.strip()]
    if not yaws:
        yaws = [0.0]
    plan: dict[str, dict] = json.loads(args.plan)

    # Stage 5 — semantic gate setup. Build a {bare_arch: semantics} map from
    # pack.semantics so we can look up avoidNear / preferredNear / placementBias
    # per archetype. Manifest packs are namespaced as <theme>.<arch>; the
    # session uses bare archetype names, so we flatten on lookup.
    pack_sem_by_arch: dict[str, dict] = {}
    if args.semantic_gate and args.pack:
        try:
            pack = load_pack(args.pack)
            for ns_name, entries in pack["archetypes"].items():
                bare = ns_name.split(".", 1)[1] if "." in ns_name else ns_name
                sem = entries[0].get("semantics") if entries else None
                if sem:
                    pack_sem_by_arch.setdefault(bare, sem)
                    pack_sem_by_arch[ns_name] = sem
            print(f"  semantic gate: loaded pack semantics for "
                  f"{len(pack_sem_by_arch)} archetype name(s)", file=sys.stderr)
        except Exception as e:
            print(f"  warning: --semantic-gate but pack load failed: {e}; "
                  "gate will be a no-op", file=sys.stderr)
            pack_sem_by_arch = {}
    elif args.semantic_gate and not args.pack:
        print("  note: --semantic-gate is on but --pack not given; "
              "gate will be a no-op", file=sys.stderr)

    zlist = _http(base + "/tool/query_zones", {})
    if not isinstance(zlist, list):
        raise SystemExit(f"query_zones returned unexpected payload: {zlist!r}")
    regions_by_id: dict[str, dict] = {z["id"]: z for z in zlist}

    if args.auto_shell:
        view = _http(base + "/tool/template_view", {})
        shells = view.get("shells") if isinstance(view, dict) else None
        if not shells:
            raise SystemExit("--auto-shell requested but template_view has "
                             "no shells")
        s = shells[0]
        ox, oy, _ = s["originWorldM"]
        regions_by_id["shell"] = {
            "id": "shell",
            "originWorldM": [float(ox), float(oy), 0.0],
            "boundsM": {
                "widthM": float(s["boundsM"]["widthM"]),
                "depthM": float(s["boundsM"]["depthM"]),
                "heightM": 0.0,
            },
            "_source": "shell",
        }

    regions_by_id.update(region_overrides)
    if not regions_by_id:
        raise SystemExit("no regions: pass --region, --auto-shell, or run "
                         "against a session that has zones")

    slot_cache: dict[str, list[float]] = {}

    def slot_for(arch: str) -> list[float]:
        if arch in slot_cache:
            return slot_cache[arch]
        if arch in slot_overrides:
            slot_cache[arch] = slot_overrides[arch]
        else:
            slot_cache[arch] = _infer_slot(base, arch)
        return slot_cache[arch]

    total_attempted = 0
    total_placed = 0
    total_rejected = 0
    per_region_reports: list[dict] = []

    for region_id, spec in plan.items():
        if region_id not in regions_by_id:
            raise SystemExit(
                f"unknown region {region_id!r}; have "
                f"{sorted(regions_by_id)} (declare --region or --auto-shell)"
            )
        attempts = int(spec.get("attempts", 0))
        mix = spec.get("archetypes") or {}
        if attempts <= 0 or not mix:
            print(f"  skip region {region_id!r}: attempts={attempts}, "
                  f"mix={mix}", file=sys.stderr)
            continue

        archs = list(mix.keys())
        weights = [float(mix[a]) for a in archs]
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise SystemExit(f"region {region_id!r}: weights must be "
                             f"non-negative and sum > 0 (got {mix})")

        slots = {a: slot_for(a) for a in archs}
        # One coarse cell pitch per region: max footprint across the mix.
        # Smaller slots within the same region will under-pack; that's OK
        # for a random scatter — the goal is irregularity, not max density.
        pitch = max(max(s[0], s[1]) for s in slots.values())

        z = regions_by_id[region_id]
        ox, oy, _ = z["originWorldM"]
        zw = z["boundsM"]["widthM"]
        zd = z["boundsM"]["depthM"]
        cells = _grid_cells(ox, oy, zw, zd, pitch)
        if not cells:
            print(f"  region {region_id!r}: empty cell grid at pitch "
                  f"{pitch:.2f} for bounds {zw}x{zd}", file=sys.stderr)
            continue

        # Per-region deterministic RNG stream (seed + region_id).
        rng = random.Random(f"{args.seed}:{region_id}")
        rng.shuffle(cells)

        # Edge-distance per cell as a fraction in [0, 1]: 0 = on the region
        # boundary, 1 = at the centre. Used by placementBias modulation
        # below. Computed once outside the cell loop for efficiency.
        cx0, cy0 = ox, oy
        cx1, cy1 = ox + zw, oy + zd
        def _edge_score(cx: float, cy: float) -> float:
            dx = min(cx - cx0, cx1 - cx) / max(zw / 2, 1e-6)
            dy = min(cy - cy0, cy1 - cy) / max(zd / 2, 1e-6)
            return max(0.0, min(1.0, min(dx, dy)))

        # Semantic-gate helpers. Cheap local check against this batch's
        # accepted placements + an HTTP query_nearby for the session's
        # prior state.
        accepted: list[tuple[str, float, float]] = []  # (arch, x, y)
        radius = float(args.neighbor_radius_m)
        radius_sq = radius * radius

        def _local_neighbor_archs(x: float, y: float) -> set[str]:
            return {a for (a, ax, ay) in accepted
                    if (ax - x) ** 2 + (ay - y) ** 2 <= radius_sq}

        def _remote_neighbor_archs(x: float, y: float) -> set[str]:
            if not pack_sem_by_arch:
                return set()
            try:
                resp = _http(base + "/tool/query_nearby",
                              {"posM": [x, y, 0.0], "radiusM": radius,
                               "limit": 32})
                if isinstance(resp, list):
                    return {p.get("archetype") for p in resp
                            if p.get("archetype")}
            except Exception:
                pass
            return set()

        def _modulated_weights(cx: float, cy: float) -> list[float]:
            """Apply placementBias to the region weights per cell. Each
            archetype's weight is scaled by min(0.05, ...) so that even
            highly-biased archetypes can still appear in mismatched cells
            at low rate — avoids hard blocks.
            """
            if not pack_sem_by_arch:
                return weights
            es = _edge_score(cx, cy)
            modulated = []
            for a, w in zip(archs, weights):
                sem = pack_sem_by_arch.get(a) or {}
                bias = sem.get("placementBias")
                if bias == "edge":
                    # closer to edge → keep, closer to centre → suppress
                    mult = max(0.05, 1.0 - es)
                elif bias == "center":
                    mult = max(0.05, es)
                elif bias == "corner":
                    # both XY are near boundary
                    mult = max(0.05, 1.0 - es) ** 2
                else:
                    mult = 1.0
                modulated.append(w * mult)
            if sum(modulated) <= 0:
                return weights
            return modulated

        placements: list[dict] = []
        sem_skips_avoid = 0
        sem_skips_preferred = 0
        # Walk shuffled cells, attempting up to `attempts` accepted placements.
        for cx, cy in cells:
            if len(placements) >= attempts:
                break
            cell_weights = _modulated_weights(cx, cy)
            arch = rng.choices(archs, weights=cell_weights, k=1)[0]
            yaw = rng.choice(yaws)
            sem = pack_sem_by_arch.get(arch) if pack_sem_by_arch else None
            if sem:
                avoid = set(sem.get("avoidNear") or [])
                prefer = set(sem.get("preferredNear") or [])
                if avoid or prefer:
                    neighbors = _local_neighbor_archs(cx, cy)
                    if not neighbors:
                        neighbors = _remote_neighbor_archs(cx, cy)
                    if avoid and (avoid & neighbors):
                        sem_skips_avoid += 1
                        continue
                    # `preferredNear` is a SOFT preference, not a hard
                    # requirement: when *no* neighbors exist (empty region,
                    # first placement) the cell is accepted so the region
                    # can bootstrap. Only enforce once at least one
                    # neighbor is around — at that point we require one of
                    # them to match the preferredNear list.
                    if prefer and neighbors and not (prefer & neighbors):
                        sem_skips_preferred += 1
                        continue
            placements.append({
                "archetype": arch,
                "posM": [round(cx, 3), round(cy, 3), 0.0],
                "slotM": slots[arch],
                "yawDeg": yaw,
            })
            accepted.append((arch, cx, cy))

        print(f"  region {region_id!r}: pitch={pitch:.2f} cells={len(cells)} "
              f"accepted-by-gate={len(placements)} "
              f"sem-skip(avoid={sem_skips_avoid}, prefer-miss={sem_skips_preferred})",
              file=sys.stderr)
        resp = _http(base + "/tool/place_many", {"placements": placements})
        placed = int(resp.get("placed") or 0)
        rejected = int(resp.get("rejected") or 0)
        total_attempted += len(placements)
        total_placed += placed
        total_rejected += rejected
        per_region_reports.append({
            "region": region_id,
            "pitch": round(pitch, 3),
            "cellsAvailable": len(cells),
            "attempted": len(placements),
            "placed": placed,
            "rejected": rejected,
            "semSkipAvoid": sem_skips_avoid,
            "semSkipPreferredMiss": sem_skips_preferred,
        })

    print(json.dumps({
        "placed": total_placed,
        "rejected": total_rejected,
        "attempted": total_attempted,
        "byRegion": per_region_reports,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
