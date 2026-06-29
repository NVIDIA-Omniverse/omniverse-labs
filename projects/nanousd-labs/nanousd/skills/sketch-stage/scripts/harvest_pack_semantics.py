#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2a: deterministic harvest of per-archetype semantic info from each
asset's USD metadata. Writes into pack.json's archetype entries.

Sources scanned per asset (first asset of each archetype is sampled by
default; pass --all-assets to scan every asset):

  - `customData["simReady"]` on the default prim — NVIDIA SimReady packs
    expose structured `category` / `subcategory` / `function` fields that
    map cleanly to anchors/affordances.
  - `Kind` metadata (`assembly` / `component` / `group` / `subcomponent`) —
    coarser, but useful as a fallback signal.
  - `UsdSemantics` schema on the default prim — `class:semanticType` and
    `class:semanticData` strings. Most useful for off-the-shelf USD assets.
  - Prim displayName / asset name — last-ditch fallback for `affordances`
    if everything else is empty.

What it writes back into pack.json:

  archetypes:
    <name>:
      - path: ...
        size: ...
        semantics:
          anchors: <string|null>
          affordances: [<string>, ...]    # may be partial
          surfaces: null                   # geometry-derived in a later pass;
                                           # LLM fills label semantics
          _harvestSource: "<which-source-fired>"

Plus a top-level `_pendingSemanticGapFill: [{archetype, missing: [...]}, ...]`
listing archetypes where any field is still unset. The LLM workflow reads
this list, edits pack.json directly to fill in the gaps using the vocabulary
in `references/pack_semantics_vocabulary.json`, then runs
`validate_pack_semantics.py` to confirm.

For manifest packs, the script recurses into each sub-pack and updates the
sub-pack's own pack.json.

Usage:
    harvest_pack_semantics.py --pack /path/to/pack [--all-assets] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import is_manifest, load_pack

from pxr import Sdf, Usd


_VOCAB_PATH = Path(__file__).resolve().parent.parent / "references" / "pack_semantics_vocabulary.json"


def _load_vocab() -> dict:
    return json.loads(_VOCAB_PATH.read_text())


def _harvest_from_asset(asset_path: Path) -> dict:
    """Best-effort metadata extraction from one USD."""
    out: dict = {"_sources": []}
    s = Usd.Stage.Open(str(asset_path))
    if not s:
        return out
    dp = s.GetDefaultPrim()
    if not dp:
        return out

    # 1. customData["simReady"]
    cd = dp.GetCustomData() or {}
    sr = cd.get("simReady") if isinstance(cd, dict) else None
    if sr:
        out["_sources"].append("customData.simReady")
        if isinstance(sr, dict):
            cat = sr.get("category") or sr.get("Category")
            sub = sr.get("subcategory") or sr.get("Subcategory")
            func = sr.get("function") or sr.get("Function")
            if cat:
                out["_simReadyCategory"] = str(cat)
            if sub:
                out["_simReadySubcategory"] = str(sub)
            if func:
                out["_simReadyFunction"] = str(func)

    # 2. Kind
    from pxr import Kind
    model = Usd.ModelAPI(dp)
    kind_str = model.GetKind()
    if kind_str:
        out["_sources"].append("Kind")
        out["_kind"] = kind_str

    # 3. UsdSemantics — semanticType / semanticData attributes on the default
    #    prim, if the asset uses the schema.
    sem_type_attr = dp.GetAttribute("class:semanticType")
    sem_data_attr = dp.GetAttribute("class:semanticData")
    sem_type = sem_type_attr.Get() if sem_type_attr and sem_type_attr.IsValid() else None
    sem_data = sem_data_attr.Get() if sem_data_attr and sem_data_attr.IsValid() else None
    if sem_type or sem_data:
        out["_sources"].append("UsdSemantics")
        if sem_type:
            out["_semanticType"] = str(sem_type)
        if sem_data:
            out["_semanticData"] = str(sem_data)

    # 4. Prim displayName
    dn = dp.GetMetadata("displayName")
    if dn:
        out["_displayName"] = str(dn)

    return out


def _map_to_vocabulary(raw: dict, vocab: dict, archetype_name: str) -> dict:
    """Translate raw harvested signals into anchors/affordances/surfaces
    using the controlled vocabulary. Returns {anchors, affordances, surfaces}
    where each may be None (= gap, LLM to fill).

    This is the only place where harvested raw strings get normalised. The
    mapping is intentionally conservative — only emit a vocabulary token
    when the signal is unambiguous. Gaps go to the LLM.
    """
    anchors_v = set(vocab["anchors"])
    affordances_v = set(vocab["affordances"])

    anchors: Optional[str] = None
    affordances: list[str] = []
    surfaces = None  # always None at harvest time; geometry pass + LLM fills

    # SimReady category often names a room/floor type directly.
    cat = (raw.get("_simReadyCategory") or "").lower().replace("-", "_").replace(" ", "_")
    if cat in anchors_v:
        anchors = cat
    # subcategory is more specific; affordance mapping
    sub = (raw.get("_simReadySubcategory") or "").lower().replace("-", "_").replace(" ", "_")
    if sub in affordances_v:
        affordances.append(sub)
    func = (raw.get("_simReadyFunction") or "").lower().replace("-", "_").replace(" ", "_")
    if func in affordances_v:
        affordances.append(func)

    # UsdSemantics tokens
    sem_type = (raw.get("_semanticType") or "").lower().replace("-", "_").replace(" ", "_")
    if sem_type in anchors_v and not anchors:
        anchors = sem_type
    if sem_type in affordances_v:
        affordances.append(sem_type)

    sem_data = (raw.get("_semanticData") or "").lower()
    for token in [t.strip().replace("-", "_").replace(" ", "_") for t in sem_data.split(",") if t.strip()]:
        if token in affordances_v and token not in affordances:
            affordances.append(token)
        if token in anchors_v and not anchors:
            anchors = token

    # Archetype name itself is sometimes a vocabulary token (e.g.
    # "server_rack" → anchors=rack_row via simple naming). We do this
    # last as a conservative fallback.
    arch_lower = archetype_name.lower().replace(".", "_")
    # split on namespace separator
    arch_basename = arch_lower.split("_", 1)[-1] if "." in archetype_name else arch_lower
    if arch_basename in affordances_v and arch_basename not in affordances:
        # tiny nudge — if the archetype shares a name with an affordance
        affordances.append(arch_basename)

    # De-dup, preserve order
    seen = set()
    affordances_dedup = []
    for a in affordances:
        if a not in seen:
            seen.add(a)
            affordances_dedup.append(a)

    return {
        "anchors": anchors,
        "affordances": affordances_dedup or None,
        "surfaces": surfaces,
        # New placement-side fields. Harvest leaves them null; the LLM
        # gap-fill stage populates from theme knowledge.
        "preferredNear": None,
        "avoidNear": None,
        "placementBias": None,
        "_harvestSource": ",".join(raw.get("_sources", [])) or "none",
    }


def harvest_flat_pack(pack_root: Path, vocab: dict, all_assets: bool = False) -> dict:
    """Modify pack.json at `pack_root` in place: enrich each archetype entry
    with `semantics`, and add a top-level `_pendingSemanticGapFill` listing
    archetypes that still need LLM input. Returns a summary dict.
    """
    pack_path = pack_root / "pack.json"
    pack = json.loads(pack_path.read_text())
    archetypes = pack.get("archetypes") or {}
    if not archetypes:
        return {"updated": 0, "gaps": []}

    gaps: list[dict] = []
    updated_archs = 0
    for arch_name, entries in archetypes.items():
        # Sample one or all assets under this archetype
        sample = entries if all_assets else entries[:1]
        raw_agg: dict = {"_sources": []}
        for entry in sample:
            asset_path = (pack_root / entry["path"]).resolve()
            if not asset_path.exists():
                continue
            try:
                raw = _harvest_from_asset(asset_path)
            except Exception as e:
                print(f"  warning: harvest of {asset_path} failed: {e}",
                      file=sys.stderr)
                continue
            for k, v in raw.items():
                if k == "_sources":
                    raw_agg["_sources"].extend(v)
                elif k not in raw_agg or not raw_agg[k]:
                    raw_agg[k] = v
        mapped = _map_to_vocabulary(raw_agg, vocab, arch_name)
        # Attach semantics to each entry (some packs have many assets per archetype)
        for entry in entries:
            entry["semantics"] = dict(mapped)
        updated_archs += 1
        missing = [k for k in ("anchors", "surfaces", "affordances")
                   if mapped.get(k) in (None, [])]
        if missing:
            gaps.append({"archetype": arch_name, "missing": missing,
                         "harvestedHints": {k: raw_agg.get(k) for k in (
                             "_simReadyCategory", "_simReadySubcategory",
                             "_simReadyFunction", "_kind",
                             "_semanticType", "_semanticData",
                             "_displayName"
                         ) if raw_agg.get(k)}})

    pack["_pendingSemanticGapFill"] = gaps
    pack_path.write_text(json.dumps(pack, indent=2))
    return {"updated": updated_archs, "gaps": gaps, "pack": str(pack_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True,
                    help="Pack root (flat or manifest)")
    ap.add_argument("--all-assets", action="store_true",
                    help="Scan every asset under each archetype (default: "
                         "sample one). Slower but more thorough for archetypes "
                         "with inconsistently-tagged assets.")
    ap.add_argument("--force", action="store_true",
                    help="Re-harvest even if archetypes already have a "
                         "semantics block.")
    args = ap.parse_args()

    pack_root = Path(args.pack).expanduser().resolve()
    pack_path = pack_root / "pack.json"
    if not pack_path.exists():
        sys.exit(f"pack.json not found at {pack_path}")
    raw = json.loads(pack_path.read_text())

    vocab = _load_vocab()

    if is_manifest(raw):
        # Recurse into each sub-pack
        totals = {"updated": 0, "gaps": [], "subpacks": []}
        for entry in raw.get("subpacks", []):
            sub_root = pack_root / entry["path"]
            print(f"harvest sub-pack {entry['theme']!r} @ {sub_root}",
                  file=sys.stderr)
            summary = harvest_flat_pack(sub_root, vocab,
                                         all_assets=args.all_assets)
            totals["updated"] += summary["updated"]
            for g in summary["gaps"]:
                g["_subpack"] = entry["theme"]
            totals["gaps"].extend(summary["gaps"])
            totals["subpacks"].append({
                "theme": entry["theme"],
                "updatedArchetypes": summary["updated"],
                "gapsCount": len(summary["gaps"]),
            })
        print(f"\nharvested {totals['updated']} archetype(s) across "
              f"{len(totals['subpacks'])} sub-pack(s); "
              f"{len(totals['gaps'])} archetype(s) still need LLM gap-fill",
              file=sys.stderr)
        if totals["gaps"]:
            print("\nGaps:", file=sys.stderr)
            for g in totals["gaps"][:20]:
                hints = ", ".join(f"{k}={v}" for k, v in (g.get("harvestedHints") or {}).items())
                print(f"  [{g['_subpack']}] {g['archetype']}: missing {g['missing']}"
                      f"{' (hints: ' + hints + ')' if hints else ''}",
                      file=sys.stderr)
            if len(totals["gaps"]) > 20:
                print(f"  ... and {len(totals['gaps']) - 20} more", file=sys.stderr)
    else:
        summary = harvest_flat_pack(pack_root, vocab,
                                     all_assets=args.all_assets)
        print(f"\nharvested {summary['updated']} archetype(s); "
              f"{len(summary['gaps'])} still need LLM gap-fill",
              file=sys.stderr)
        if summary["gaps"]:
            print("\nGaps:", file=sys.stderr)
            for g in summary["gaps"][:20]:
                hints = ", ".join(f"{k}={v}" for k, v in (g.get("harvestedHints") or {}).items())
                print(f"  {g['archetype']}: missing {g['missing']}"
                      f"{' (hints: ' + hints + ')' if hints else ''}",
                      file=sys.stderr)

    print("\nLLM workflow: read each sub-pack's pack.json, look at the "
          "`_pendingSemanticGapFill` list, edit the `semantics` block of "
          "each archetype directly with values from "
          "references/pack_semantics_vocabulary.json, then run "
          "validate_pack_semantics.py to confirm.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
