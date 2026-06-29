#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process an absorbed sketch by detecting functional regions and
writing them back into the sketch as `templateZones[]`.

Inputs:
  - absorbed sketch JSON (`absorb_pack.py` output), with `tree` of placements
  - pack root (flat or manifest) with per-archetype `semantics` fields from
    `harvest_pack_semantics.py` + LLM gap-fill + `validate_pack_semantics.py`

The script applies four passes:

  1. ANCHOR pass — every placement whose archetype declares
     `semantics.anchors=<region_type>` seeds a named region. Other placements
     that fall inside that region's footprint are absorbed into it.
  2. CLUSTER pass — remaining placements get spatial-grid clustered (1m grid
     + adjacency union); each cluster becomes a region named by its dominant
     archetype.
  3. GAP pass — large negative-space rectangles within the shell become
     "circulation" regions.
  4. (Separate LLM step) — the LLM reads the resulting `templateZones[]`
     and edits each region in place to add `name`, `purpose`,
     `allowedArchetypes`, and `note`. A companion `validate_absorbed_regions.py`
     verifies the labels.

The resulting `templateZones[]` is consumed by the runtime session
(MCP `query_zones` / `template_view`) and by the on-surface placement
gate (Stage 4).

Usage:
  extract_regions.py --sketch /path/to/absorbed.sketch.json \\
                     --pack /path/to/pack
                     [--write]    # default: dry-run (prints regions)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import load_pack


# --- placement extraction --------------------------------------------------

def _walk_placements(node, out: list) -> None:
    """Walk the absorbed-sketch tree, accumulating placement objects.

    Schema: a placement node has `type=="placement"` and carries
    `archetype`, `slotM: {widthM, depthM, heightM}`, `transform: {translateM:
    [x, y, z], yawDeg}`. The tree mixes container nodes (with `children`)
    and placement leaves.
    """
    if isinstance(node, dict):
        if node.get("type") == "placement":
            out.append(node)
            return
        children = node.get("children")
        if isinstance(children, list):
            for c in children:
                _walk_placements(c, out)
    elif isinstance(node, list):
        for c in node:
            _walk_placements(c, out)


def _placement_summary(pl: dict) -> dict:
    """Distill a placement node down to {id, archetype, centerM, sizeM, yawDeg}.

    Up-axis convention: absorbed sketches we've seen use Z-up with the floor
    plane at z=0, so the 2D position is `(translateM[0], translateM[1])`.
    """
    t = pl.get("transform", {}) or {}
    tr = t.get("translateM", [0.0, 0.0, 0.0])
    s = pl.get("slotM") or pl.get("sizeM") or {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0}
    return {
        "id": pl.get("id"),
        "archetype": pl.get("archetype"),
        "centerM": [float(tr[0]), float(tr[1]), float(tr[2])],
        "sizeM": [float(s.get("widthM", 1.0)), float(s.get("depthM", 1.0)),
                  float(s.get("heightM", 1.0))],
        "yawDeg": float(t.get("yawDeg", 0.0)),
    }


# --- pack semantics lookup -------------------------------------------------

def _archetype_semantics_for_sketch(pack: dict) -> dict:
    """Build a mapping from un-namespaced archetype name to the first-found
    semantics block. Absorbed sketches name placements by un-namespaced
    archetype (e.g. "<arch>", not "<theme>.<arch>"), so we flatten the
    pack's namespaces.
    """
    out: dict = {}
    for ns_name, entries in pack["archetypes"].items():
        # ns_name may be "<theme>.<arch>" or just "<arch>" depending on pack shape
        bare = ns_name.split(".", 1)[1] if "." in ns_name else ns_name
        if not entries:
            continue
        sem = entries[0].get("semantics") if entries[0] else None
        if sem:
            # First write wins for plain `bare` lookup; namespace-aware later
            out.setdefault(bare, sem)
            out[ns_name] = sem
    return out


# --- pass 1: anchors -------------------------------------------------------

def anchor_pass(placements: list, sem_by_arch: dict,
                growth: float = 1.5, pad: float = 0.5
                ) -> tuple[list[dict], set[str]]:
    regions: list[dict] = []
    assigned: set[str] = set()
    for i, p in enumerate(placements):
        sem = sem_by_arch.get(p["archetype"]) or {}
        anchor_kind = sem.get("anchors")
        if not anchor_kind:
            continue
        cx, cy, _ = p["centerM"]
        w, d, _ = p["sizeM"]
        gw = w * growth + pad
        gd = d * growth + pad
        regions.append({
            "id": f"z_anchor_{i}",
            "footprintM": [[cx - gw, cy - gd], [cx + gw, cy + gd]],
            "anchorPlacementId": p["id"],
            "anchorArchetype": p["archetype"],
            "anchorSemantic": anchor_kind,
            "containedPlacements": [p["id"]],
            "archetypeMix": {p["archetype"]: 1},
            "source": "anchor_pass",
        })
        assigned.add(p["id"])
    # absorb other placements that fall inside an anchor's footprint
    for r in regions:
        (x0, y0), (x1, y1) = r["footprintM"]
        for p in placements:
            if p["id"] in assigned:
                continue
            cx, cy, _ = p["centerM"]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                r["containedPlacements"].append(p["id"])
                r["archetypeMix"][p["archetype"]] = (
                    r["archetypeMix"].get(p["archetype"], 0) + 1)
                assigned.add(p["id"])
    return regions, assigned


# --- pass 2: clusters ------------------------------------------------------

def cluster_pass(placements: list, assigned: set[str],
                 grid_m: float = 1.0, pad: float = 0.3
                 ) -> tuple[list[dict], set[str]]:
    remaining = [p for p in placements if p["id"] not in assigned]
    buckets: dict[tuple[int, int], list[dict]] = {}
    for p in remaining:
        cx, cy, _ = p["centerM"]
        buckets.setdefault((int(cx // grid_m), int(cy // grid_m)), []).append(p)

    parent = {k: k for k in buckets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for k in list(buckets):
        for dx, dy in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
            nk = (k[0] + dx, k[1] + dy)
            if nk in buckets:
                union(k, nk)

    components: dict[tuple[int, int], list[dict]] = {}
    for k in buckets:
        components.setdefault(find(k), []).extend(buckets[k])

    regions: list[dict] = []
    newly: set[str] = set()
    for i, members in enumerate(components.values()):
        xs = [m["centerM"][0] for m in members]
        ys = [m["centerM"][1] for m in members]
        ws = [m["sizeM"][0] for m in members]
        ds = [m["sizeM"][1] for m in members]
        fp = [
            [min(xs) - max(ws) / 2 - pad, min(ys) - max(ds) / 2 - pad],
            [max(xs) + max(ws) / 2 + pad, max(ys) + max(ds) / 2 + pad],
        ]
        mix: Counter = Counter()
        for m in members:
            mix[m["archetype"]] += 1
        dominant = mix.most_common(1)[0][0]
        regions.append({
            "id": f"z_cluster_{i}",
            "footprintM": fp,
            "dominantArchetype": dominant,
            "containedPlacements": [m["id"] for m in members],
            "archetypeMix": dict(mix),
            "source": "cluster_pass",
        })
        for m in members:
            newly.add(m["id"])
    return regions, newly


# --- pass 3: gaps ----------------------------------------------------------

def gap_pass(bounds_m: tuple[float, float],
             occupied_regions: list[dict],
             step: float = 1.0, min_cells: int = 16) -> list[dict]:
    w, d = bounds_m

    def occupied(x, y):
        for r in occupied_regions:
            (x0, y0), (x1, y1) = r["footprintM"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    nx = max(1, int(w / step))
    ny = max(1, int(d / step))
    free = set()
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            if not occupied(x, y):
                free.add((i, j))

    seen: set[tuple[int, int]] = set()
    regions: list[dict] = []
    for cell in free:
        if cell in seen:
            continue
        comp: list[tuple[int, int]] = []
        stack = [cell]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.append(c)
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (c[0] + di, c[1] + dj)
                if nb in free:
                    stack.append(nb)
        if len(comp) < min_cells:
            continue
        xs = [c[0] * step for c in comp]
        ys = [c[1] * step for c in comp]
        fp = [[min(xs), min(ys)], [max(xs) + step, max(ys) + step]]
        cell_w = (max(xs) - min(xs)) + step
        cell_d = (max(ys) - min(ys)) + step
        narrow = "depth" if cell_d < cell_w else "width"
        regions.append({
            "id": f"z_gap_{len(regions)}",
            "footprintM": fp,
            "narrowSide": narrow,
            "approxAreaM2": round(len(comp) * step * step, 2),
            "containedPlacements": [],
            "archetypeMix": {},
            "source": "gap_pass",
        })
    return regions


# --- driver ----------------------------------------------------------------

def extract(sketch: dict, pack: dict, *,
            anchor_growth: float = 1.5, anchor_pad: float = 0.5,
            cluster_grid_m: float = 1.0, cluster_pad: float = 0.3,
            gap_step: float = 1.0, gap_min_cells: int = 16) -> list[dict]:
    placements_raw: list = []
    _walk_placements(sketch.get("tree"), placements_raw)
    placements = [_placement_summary(p) for p in placements_raw]

    sem_by_arch = _archetype_semantics_for_sketch(pack)
    anchor_regs, assigned = anchor_pass(placements, sem_by_arch,
                                          growth=anchor_growth, pad=anchor_pad)
    cluster_regs, cluster_assigned = cluster_pass(placements, assigned,
                                                   grid_m=cluster_grid_m,
                                                   pad=cluster_pad)
    assigned |= cluster_assigned

    shell = (sketch.get("tree") or {}).get("shell") or {}
    bounds = shell.get("boundsM") or {}
    w = float(bounds.get("widthM", 0))
    d = float(bounds.get("depthM", 0))
    gap_regs = (gap_pass((w, d), anchor_regs + cluster_regs,
                          step=gap_step, min_cells=gap_min_cells)
                if w > 0 and d > 0 else [])

    return anchor_regs + cluster_regs + gap_regs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sketch", required=True, help="Absorbed sketch JSON path")
    ap.add_argument("--pack", required=True, help="Pack root (flat or manifest)")
    ap.add_argument("--write", action="store_true",
                    help="Persist the regions back into the sketch's "
                         "`templateZones[]`. Default: dry-run, print only.")
    args = ap.parse_args()

    sketch_path = Path(args.sketch).expanduser().resolve()
    sketch = json.loads(sketch_path.read_text())
    try:
        pack = load_pack(args.pack)
    except FileNotFoundError as e:
        sys.exit(str(e))

    regions = extract(sketch, pack)
    summary = [
        {"id": r["id"], "source": r["source"],
         "footprintM": r["footprintM"],
         "containedCount": len(r.get("containedPlacements", [])),
         "archetypeMix": r.get("archetypeMix", {}),
         "anchorSemantic": r.get("anchorSemantic"),
         "dominantArchetype": r.get("dominantArchetype")}
        for r in regions
    ]
    print(json.dumps(summary, indent=2))
    by_pass = Counter(r["source"] for r in regions)
    print(f"\nextracted {len(regions)} region(s) "
          f"({', '.join(f'{k}={v}' for k, v in by_pass.items())})",
          file=sys.stderr)
    print("\nNext LLM step: read each region and edit it in place to add "
          "`name`, `purpose`, `allowedArchetypes`, and `note` using the "
          "vocabulary in references/pack_semantics_vocabulary.json. Then "
          "run validate_absorbed_regions.py.", file=sys.stderr)

    if args.write:
        sketch["templateZones"] = regions
        sketch_path.write_text(json.dumps(sketch, indent=2))
        print(f"wrote {len(regions)} region(s) to {sketch_path}", file=sys.stderr)
    else:
        print("(dry-run; pass --write to persist into the sketch)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
