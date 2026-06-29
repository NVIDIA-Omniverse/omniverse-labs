#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact summary of an asset pack for the LLM that will produce an
archetype plan.

Reading raw `pack.json` to inventory archetypes is a deterministic file
op that costs the LLM ~tens of lines of JSON context per archetype.
A large pack (200+ archetypes) makes this expensive.

This script does the deterministic plumbing:
  - Reads `pack.json` (flat) or the manifest + each sub-pack's pack.json
    (manifest mode — see scripts/pack_loader.py)
  - For manifest packs, archetype names are namespaced `<theme>.<archetype>`
    and the sub-pack roster is printed as a header. Use `--subpack <theme>`
    (repeatable) to restrict to a chosen theme set.
  - Drops archetypes whose pack entries have sentinel-invalid bboxes
    (gen_pack_json.py leftovers from assets whose payloads couldn't be
    measured at scan time)
  - Prints one line per valid archetype: asset count + a representative
    bbox in metres
  - Optionally: given `--regions <json>` and `--target-placements <int>`,
    also prints the per-region attempt budget allocated by weight, so
    the LLM doesn't have to multiply `target × weight / total × 1.3` in
    its head

The LLM then handles the irreducible reasoning steps:
  1. Sub-pack (theme) selection — pick which themes match the user's intent
     (manifest packs only; the script lists themes + their archetype counts).
  2. Region definition (what regions exist, what footprint each occupies)
  3. Archetype filtering for the theme
  4. Archetype → region mapping
  5. Per-region archetype weights
  6. Missing-archetype flagging (typical-for-theme but absent from pack)
  7. Free-form per-region notes

Discovery modes (compact output for LLM token-thrift):
    --themes                      → list sub-packs (theme + counts), nothing else
    --list-archetypes             → archetype names + asset counts, no bboxes
    --list-assets <archetype>     → asset paths under one archetype, one per line

Usage:
    pack_summary.py --pack /path/to/asset_pack
    pack_summary.py --pack /path/to/asset_pack --themes
    pack_summary.py --pack /path/to/asset_pack --subpack <theme> --list-archetypes
    pack_summary.py --pack /path/to/asset_pack --list-assets <theme>.<archetype>
    pack_summary.py --pack /path/to/asset_pack \\
        --regions '{"<region_a>": {"weight": 0.80}, "<region_b>": {"weight": 0.15}}' \\
        --target-placements 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import load_pack, select_subpacks


def _bbox_ok(entry: dict) -> bool:
    size = entry.get("size", {})
    w = size.get("widthM", 0)
    return w > 0 and w < 1e30  # gen_pack_json sentinel is ~-6.8e38


def _fmt_bbox(entry: dict) -> str:
    s = entry.get("size", {})
    return f"{s.get('widthM', 0):.2g}×{s.get('depthM', 0):.2g}×{s.get('heightM', 0):.2g} m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True,
                    help="Asset pack root (must contain pack.json; flat or manifest)")
    ap.add_argument("--subpack", action="append", default=[],
                    help="Restrict to these theme(s) for manifest packs "
                         "(repeatable). Default: include every sub-pack.")
    ap.add_argument("--themes", action="store_true",
                    help="Compact discovery: list sub-packs (theme + counts) "
                         "and exit. For manifest packs only.")
    ap.add_argument("--list-archetypes", action="store_true",
                    help="Compact discovery: list archetype names + asset counts, "
                         "no bbox details. Honors --subpack.")
    ap.add_argument("--list-assets", default=None, metavar="ARCH",
                    help="Compact discovery: list the asset paths under the "
                         "given archetype (use the namespaced name "
                         "`<theme>.<archetype>` for manifest packs).")
    ap.add_argument("--regions", default=None,
                    help="JSON: {<region>: {weight: <float>, ...}, ...}. "
                         "When given with --target-placements, prints the "
                         "per-region attempt budget.")
    ap.add_argument("--target-placements", type=int, default=None,
                    help="Total placement target. With --regions, used to "
                         "compute per-region attempts.")
    ap.add_argument("--cushion", type=float, default=1.3,
                    help="Multiplier applied to per-region attempts to "
                         "compensate for collision-gate rejects "
                         "(default 1.3, i.e. 30%% headroom)")
    args = ap.parse_args()

    try:
        pack = load_pack(args.pack)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if args.subpack:
        if not pack["isManifest"]:
            print("warning: --subpack ignored for flat pack", file=sys.stderr)
        else:
            pack = select_subpacks(pack, args.subpack)

    archs = pack["archetypes"]
    if not archs:
        sys.exit("pack has no archetypes (after sub-pack filter, if any)")

    # --- compact discovery modes (short-circuit before the full inventory) ---

    if args.themes:
        subs = pack["subpacks"]
        if not subs:
            print(f"flat pack {pack['name']!r} ({len(archs)} archetype(s)) "
                  "— no themes, single library")
        else:
            for s in subs:
                print(f"{s['theme']}\t{s['archetypeCount']} archetype(s)\t"
                      f"{s['assetCount']} asset(s)\tpath={s['path']}")
        return 0

    if args.list_assets is not None:
        target = args.list_assets
        entries = archs.get(target)
        if entries is None:
            # Helpful fuzzy fallback for missing namespace prefix
            for name in archs:
                if name.endswith("." + target) or name == target:
                    entries = archs[name]
                    target = name
                    break
        if entries is None:
            print(f"archetype {args.list_assets!r} not found "
                  f"(available examples: {sorted(archs)[:5]}{'…' if len(archs) > 5 else ''})",
                  file=sys.stderr)
            return 1
        for e in entries:
            if _bbox_ok(e):
                print(e["path"])
        return 0

    if args.list_archetypes:
        for name in sorted(archs):
            n_ok = sum(1 for e in archs[name] if _bbox_ok(e))
            print(f"{name}\t{n_ok} asset(s)")
        return 0

    valid: dict[str, list[dict]] = {}
    invalid: dict[str, int] = {}
    for name, entries in archs.items():
        ok = [e for e in entries if _bbox_ok(e)]
        if ok:
            valid[name] = ok
        else:
            invalid[name] = len(entries)

    if pack["isManifest"]:
        subs = pack["subpacks"]
        print(f"manifest pack {pack['name']!r}: {len(subs)} sub-pack(s)")
        sub_col = max(len(s["theme"]) for s in subs) if subs else 0
        for s in subs:
            print(f"  {s['theme']:<{sub_col}}  ({s['archetypeCount']} archetype(s), "
                  f"{s['assetCount']} asset(s), upAxis={s['upAxis']}, "
                  f"mpu={s['metersPerUnit']})")
        print()
        print("Archetypes below are namespaced as `<theme>.<archetype>`. "
              "Pick the sub-pack(s) you need with --subpack to narrow down, "
              "or pass the sub-pack root as --pack to operate on one theme.")
        print()

    up = pack.get("upAxis", "?")
    mpu = pack.get("metersPerUnit", "?")
    total_valid = sum(len(v) for v in valid.values())
    print(f"{len(valid)} archetype(s), {total_valid} valid asset(s), "
          f"upAxis={up}, mpu={mpu}")
    if invalid:
        print(f"(skipped {sum(invalid.values())} asset(s) across "
              f"{len(invalid)} archetype(s) with invalid bboxes: "
              f"{sorted(invalid)})")
    name_col = max(len(n) for n in valid) if valid else 0
    for name in sorted(valid):
        entries = valid[name]
        first = _fmt_bbox(entries[0])
        if len(entries) == 1:
            print(f"  {name:<{name_col}} : 1 asset  · {first}")
        else:
            last = _fmt_bbox(entries[-1])
            print(f"  {name:<{name_col}} : {len(entries)} assets · "
                  f"{first} – {last}")

    if args.regions and args.target_placements:
        regions = json.loads(args.regions)
        total_w = sum(r.get("weight", 1.0) for r in regions.values())
        print()
        print(f"per-region attempt budget "
              f"(target={args.target_placements}, ×{args.cushion} cushion):")
        region_col = max(len(n) for n in regions)
        for name, info in regions.items():
            w = info.get("weight", 1.0)
            attempts = max(1, int(args.target_placements * (w / total_w) * args.cushion))
            print(f"  {name:<{region_col}} (weight {w}): {attempts} attempts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
