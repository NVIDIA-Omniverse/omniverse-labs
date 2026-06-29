# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a pack.json for a directory of USD assets.

Walks a directory, opens each *.usd file, reads the asset stage's own
upAxis + metersPerUnit, measures bbox via UsdGeom.BBoxCache, groups assets
by parent-folder name, and writes a pack.json compatible with the
sketch-stage skill.

Two output shapes:

  Flat pack (today's behavior, used when the root holds a single themed
  library): one `pack.json` at the root with the full `archetypes` map.

  Manifest pack (NEW, auto-detected when the root holds multiple themed
  sub-libraries): one top-level `pack.json` that lists `subpacks: [{name,
  theme, path, ...}]`, plus a flat `pack.json` written into each sub-pack's
  own root. Each sub-pack has its own theme. The LLM can later refine the
  themes (phase 2 semantic enrichment, separate script).

Detection rule for sub-packs: the root's *immediate child directories* each
hold at least `--subpack-min-assets` (default 5) entry-point USDs. If at
least two such children exist the root is treated as a manifest; otherwise
the root is scanned flat. `--flat` forces the flat path. `--subpacks DIR
[DIR…]` overrides detection with an explicit list. The default of 5 is
deliberately conservative — real single-theme packs often split their
assets across manufacturer-named subdirs (e.g. `GB300/`, `Vertiv/`), and a
3-asset threshold would misclassify those as two themes. Lower the
threshold explicitly if your pack genuinely has tiny themed sub-libraries.

Skips obvious non-asset paths (payloads/, looks/, materials/) so a pack
contains entry-point USDs only.

The asset stage's own upAxis/mpu is authoritative — pack-level metadata
that disagrees with the assets is a source of bugs (see SKILL.md).

Usage:
  gen_pack_json.py /path/to/assets [--name pack_name] [--scenario s]
                                   [--flat] [--subpacks dir1 [dir2 ...]]
                                   [--subpack-min-assets N]
                                   [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from pxr import Usd, UsdGeom


SKIP_DIRS = {"payloads", "looks", "materials", ".thumbs"}


def is_asset_usd(path: Path) -> bool:
    if path.suffix.lower() != ".usd":
        return False
    parts = {p.lower() for p in path.parts}
    if SKIP_DIRS & parts:
        return False
    name = path.name.lower()
    if name.startswith("_") or name in {"main.usd", "stage.usd"}:
        return False
    return True


def measure(path: Path):
    s = Usd.Stage.Open(str(path))
    if not s:
        return None
    dp = s.GetDefaultPrim()
    if not dp:
        return None
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = bb.ComputeWorldBound(dp).ComputeAlignedRange()
    sz = rng.GetSize()
    if any(v == 0 or v != v for v in sz):
        return None
    up = str(UsdGeom.GetStageUpAxis(s) or "Z").upper()
    mpu = float(UsdGeom.GetStageMetersPerUnit(s) or 1.0)
    return {
        "size_units": (float(sz[0]), float(sz[1]), float(sz[2])),
        "size_m": (float(sz[0]) * mpu, float(sz[1]) * mpu, float(sz[2]) * mpu),
        "upAxis": up,
        "mpu": mpu,
    }


def _count_assets_under(d: Path) -> int:
    """Count entry-point USDs anywhere under directory `d`."""
    return sum(1 for p in d.rglob("*.usd") if is_asset_usd(p))


def detect_subpacks(root: Path, min_assets_per_subpack: int) -> list[Path]:
    """Return the root's immediate child directories that look like sub-packs.

    A child counts as a sub-pack candidate when it contains at least
    `min_assets_per_subpack` entry-point USDs anywhere under it. Returned in
    sorted order for deterministic output.

    Returns [] when no children qualify. Returns a single-element list iff
    one child qualifies — the caller should still scan flat in that case
    (nested-pack handling only matters at 2+).
    """
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.lower() in SKIP_DIRS:
            continue
        if child.name.startswith("."):
            continue
        if _count_assets_under(child) >= min_assets_per_subpack:
            candidates.append(child)
    return candidates


def scan_pack(root: Path, pack_name: str) -> dict:
    """Scan a single themed sub-library (or a flat root) and return a dict
    with archetypes + voted upAxis/mpu + scan stats. Pure data; caller writes
    pack.json.
    """
    archetypes: dict[str, list[dict]] = {}
    up_votes: Counter = Counter()
    mpu_votes: Counter = Counter()
    n_scanned = 0
    n_kept = 0

    for path in root.rglob("*.usd"):
        if not is_asset_usd(path):
            continue
        n_scanned += 1
        m = measure(path)
        if not m:
            continue
        n_kept += 1
        up_votes[m["upAxis"]] += 1
        mpu_votes[m["mpu"]] += 1
        rel_parts = path.relative_to(root).parts[:-1]
        cat = path.parent.name
        if cat.lower() == pack_name.lower():
            cat = path.parent.parent.name
        else:
            parent = path.parent
            sibling_assets = [q for q in parent.glob("*.usd") if is_asset_usd(q)]
            if len(sibling_assets) <= 1 and len(rel_parts) >= 2:
                cat = rel_parts[-2]
        rel = path.relative_to(root).as_posix()
        archetypes.setdefault(cat, []).append({
            "path": rel,
            "size": {
                "widthM": round(m["size_m"][0], 4),
                "depthM": round(m["size_m"][1], 4),
                "heightM": round(m["size_m"][2], 4),
            },
        })

    up = up_votes.most_common(1)[0][0] if up_votes else "Z"
    mpu = mpu_votes.most_common(1)[0][0] if mpu_votes else 1.0

    return {
        "archetypes": archetypes,
        "upAxis": up,
        "metersPerUnit": float(mpu),
        "_scanStats": {
            "scanned": n_scanned, "kept": n_kept,
            "upAxisVotes": dict(up_votes),
            "mpuVotes": {str(k): v for k, v in mpu_votes.items()},
        },
    }


def write_flat_pack(root: Path, pack_name: str, scenarios: list[str],
                     theme: Optional[str] = None) -> tuple[Path, dict]:
    """Scan `root` and write a flat pack.json. Returns (path, brief_stats)."""
    scan = scan_pack(root, pack_name)
    pack = {
        "name": pack_name,
        "metersPerUnit": scan["metersPerUnit"],
        "upAxis": scan["upAxis"],
        "scenarios": scenarios,
    }
    if theme:
        pack["theme"] = theme
    pack["archetypes"] = scan["archetypes"]
    pack["_generatedBy"] = "sketch-stage/gen_pack_json.py"
    pack["_scanStats"] = scan["_scanStats"]
    pack_path = root / "pack.json"
    pack_path.write_text(json.dumps(pack, indent=2))
    brief = {
        "name": pack_name,
        "theme": theme or pack_name,
        "archetypeCount": len(scan["archetypes"]),
        "assetCount": scan["_scanStats"]["kept"],
        "upAxis": scan["upAxis"],
        "metersPerUnit": scan["metersPerUnit"],
    }
    return pack_path, brief


def write_manifest(root: Path, root_name: str, scenarios: list[str],
                    subpack_briefs: list[dict]) -> Path:
    """Write a top-level pack.json manifest pointing at sub-packs."""
    manifest = {
        "name": root_name,
        "scenarios": scenarios,
        "_isManifest": True,
        "subpacks": subpack_briefs,
        "_generatedBy": "sketch-stage/gen_pack_json.py",
        "_note": ("This pack.json is a manifest. Each entry under `subpacks` "
                  "lives in its own subdirectory with its own pack.json. The "
                  "sketch-stage LLM picks which sub-pack(s) match the user's "
                  "intent before doing per-region archetype selection. Themes "
                  "default to subdir names; a separate phase-2 semantic "
                  "enrichment can refine them."),
    }
    pack_path = root / "pack.json"
    pack_path.write_text(json.dumps(manifest, indent=2))
    return pack_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Asset pack root directory")
    ap.add_argument("--name", default=None, help="Pack name (default: dirname)")
    ap.add_argument("--scenario", action="append", default=[],
                    help="Add a scenario tag; repeat for multiple")
    ap.add_argument("--flat", action="store_true",
                    help="Force flat-pack output even when sub-packs would be "
                         "auto-detected")
    ap.add_argument("--subpacks", nargs="+", default=None,
                    help="Override sub-pack detection with an explicit list of "
                         "subdirectory names (relative to root)")
    ap.add_argument("--subpack-min-assets", type=int, default=5,
                    help="Minimum entry-point USDs a child dir must contain to "
                         "qualify as a sub-pack (default 5). Conservative on "
                         "purpose: single-theme packs that split assets across "
                         "manufacturer-named subdirs (GB300/, Vertiv/, ...) "
                         "would mis-detect as multiple themes at a lower "
                         "threshold. Lower it if your pack has genuinely "
                         "small themed sub-libraries.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing pack.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    pack_path = root / "pack.json"
    if pack_path.exists() and not args.force:
        print(f"pack.json already exists at {pack_path}; use --force to overwrite",
              file=sys.stderr)
        return 0

    root_name = args.name or root.name.lower().replace(" ", "_")
    scenarios = args.scenario or []

    # Resolve sub-packs: explicit list, auto-detect, or none.
    if args.flat:
        subpack_dirs: list[Path] = []
    elif args.subpacks is not None:
        subpack_dirs = []
        for d in args.subpacks:
            sd = (root / d).resolve()
            if not sd.is_dir():
                print(f"error: --subpacks entry {d!r} is not a directory under {root}",
                      file=sys.stderr)
                return 2
            subpack_dirs.append(sd)
    else:
        subpack_dirs = detect_subpacks(root, args.subpack_min_assets)
        if len(subpack_dirs) < 2:
            subpack_dirs = []  # one or zero sub-packs → scan flat

    if not subpack_dirs:
        pack_path, brief = write_flat_pack(root, root_name, scenarios)
        if brief["assetCount"] == 0:
            print(f"no entry-point USDs found under {root}", file=sys.stderr)
            return 2
        print(f"wrote {pack_path}: {brief['assetCount']} assets, "
              f"{brief['archetypeCount']} categories, "
              f"upAxis={brief['upAxis']}, mpu={brief['metersPerUnit']}",
              file=sys.stderr)
        return 0

    # Manifest path: write a flat pack.json into each sub-pack root, then
    # write the top-level manifest.
    briefs = []
    for sd in subpack_dirs:
        sub_name = sd.name.lower().replace(" ", "_")
        sub_pack_path = sd / "pack.json"
        if sub_pack_path.exists() and not args.force:
            print(f"  skipping {sub_pack_path} (exists; --force to overwrite)",
                  file=sys.stderr)
            try:
                existing = json.loads(sub_pack_path.read_text())
            except Exception as e:
                print(f"error: existing {sub_pack_path} unreadable: {e}",
                      file=sys.stderr)
                return 2
            brief = {
                "name": existing.get("name", sub_name),
                "theme": existing.get("theme", sub_name),
                "archetypeCount": len(existing.get("archetypes") or {}),
                "assetCount": int((existing.get("_scanStats") or {}).get("kept", 0)),
                "upAxis": existing.get("upAxis", "Z"),
                "metersPerUnit": existing.get("metersPerUnit", 1.0),
                "path": sd.relative_to(root).as_posix(),
            }
            briefs.append(brief)
            continue
        try:
            written_path, brief = write_flat_pack(sd, sub_name, scenarios,
                                                   theme=sub_name)
        except Exception as e:
            print(f"error scanning sub-pack {sd}: {e}", file=sys.stderr)
            return 2
        brief["path"] = sd.relative_to(root).as_posix()
        briefs.append(brief)
        print(f"  wrote {written_path}: {brief['assetCount']} assets, "
              f"{brief['archetypeCount']} categories, theme={brief['theme']!r}",
              file=sys.stderr)

    manifest_path = write_manifest(root, root_name, scenarios, briefs)
    print(f"wrote manifest {manifest_path}: {len(briefs)} sub-pack(s) — "
          f"{', '.join(b['theme'] for b in briefs)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
