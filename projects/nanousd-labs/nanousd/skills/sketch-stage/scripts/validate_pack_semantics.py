#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that an LLM-completed pack.json's semantics blocks use values
from references/pack_semantics_vocabulary.json. Run after
harvest_pack_semantics.py + LLM gap-fill, before launching a session that
consumes the pack.

Checks per archetype's `semantics`:
  - `anchors` is null or a known anchor token
  - `surfaces` is null or a list of {label, localTopZ, footprintM}; each
    label is a known surface label; localTopZ is a non-negative float;
    footprintM is [w, d] floats > 0
  - `affordances` is null or a list of known affordance tokens

Plus:
  - If `_pendingSemanticGapFill` is non-empty, those archetypes are flagged
    as STILL pending (warning, not error — the LLM may have skipped some
    on purpose).
  - For manifest packs, recurses into each sub-pack's pack.json.

Exit codes:
  0  all semantics valid; no pending gaps
  1  hard errors (unknown tokens, malformed surface entries, bad types)
  2  warnings only (pending gaps)

Usage:
  validate_pack_semantics.py --pack /path/to/pack [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import is_manifest


_VOCAB_PATH = Path(__file__).resolve().parent.parent / "references" / "pack_semantics_vocabulary.json"


def _vocab() -> dict:
    return json.loads(_VOCAB_PATH.read_text())


def _validate_one_archetype(arch: str, entries: list, vocab: dict,
                             where: str = "") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    anchors_v = set(vocab["anchors"])
    surfaces_v = set(vocab["surfaceLabels"])
    affordances_v = set(vocab["affordances"])
    prefix = f"{where}/" if where else ""
    for i, entry in enumerate(entries):
        sem = entry.get("semantics")
        if sem is None:
            warnings.append(
                f"{prefix}{arch}[{i}]: no `semantics` block (harvest may "
                f"not have run, or this asset predates the change)")
            continue
        a = sem.get("anchors")
        if a is not None and a not in anchors_v:
            errors.append(f"{prefix}{arch}[{i}].semantics.anchors={a!r} is not in vocabulary")
        affs = sem.get("affordances")
        if affs is not None:
            if not isinstance(affs, list):
                errors.append(f"{prefix}{arch}[{i}].semantics.affordances must be a list")
            else:
                for af in affs:
                    if af not in affordances_v:
                        errors.append(f"{prefix}{arch}[{i}].semantics.affordances has unknown {af!r}")
        # preferredNear / avoidNear: free-form lists of archetype names
        # (no vocab check — they reference other archetypes in this or any
        # other pack, not a controlled vocabulary).
        for fld in ("preferredNear", "avoidNear"):
            v = sem.get(fld)
            if v is None:
                continue
            if not isinstance(v, list):
                errors.append(f"{prefix}{arch}[{i}].semantics.{fld} must be a list or null")
            else:
                for item in v:
                    if not isinstance(item, str) or not item:
                        errors.append(f"{prefix}{arch}[{i}].semantics.{fld} entries "
                                       f"must be non-empty strings (got {item!r})")
        # placementBias: optional, must be in {edge, center, corner, null}
        pb = sem.get("placementBias")
        if pb is not None and pb not in vocab.get("placementBiases", ["edge", "center", "corner", None]):
            errors.append(f"{prefix}{arch}[{i}].semantics.placementBias={pb!r} "
                          "is not in placementBiases vocabulary")
        surfs = sem.get("surfaces")
        if surfs is not None:
            if not isinstance(surfs, list):
                errors.append(f"{prefix}{arch}[{i}].semantics.surfaces must be a list (or null)")
            else:
                for j, sf in enumerate(surfs):
                    if not isinstance(sf, dict):
                        errors.append(f"{prefix}{arch}[{i}].semantics.surfaces[{j}] must be an object")
                        continue
                    if sf.get("label") not in surfaces_v:
                        errors.append(
                            f"{prefix}{arch}[{i}].semantics.surfaces[{j}].label="
                            f"{sf.get('label')!r} is not in vocabulary")
                    tz = sf.get("localTopZ")
                    if not isinstance(tz, (int, float)) or tz < 0:
                        errors.append(
                            f"{prefix}{arch}[{i}].semantics.surfaces[{j}].localTopZ "
                            f"must be a non-negative number (got {tz!r})")
                    fp = sf.get("footprintM")
                    if (not isinstance(fp, list) or len(fp) != 2
                            or not all(isinstance(v, (int, float)) and v > 0 for v in fp)):
                        errors.append(
                            f"{prefix}{arch}[{i}].semantics.surfaces[{j}].footprintM "
                            f"must be [w>0, d>0] (got {fp!r})")
    return errors, warnings


def _validate_flat_pack(pack_path: Path, vocab: dict,
                         where: str = "") -> tuple[list[str], list[str], int]:
    raw = json.loads(pack_path.read_text())
    archetypes = raw.get("archetypes") or {}
    errors: list[str] = []
    warnings: list[str] = []
    for arch, entries in archetypes.items():
        e, w = _validate_one_archetype(arch, entries, vocab, where=where)
        errors.extend(e)
        warnings.extend(w)
    # A pending gap is only "real" if the entry's semantics still has the
    # missing field set to None / empty.
    real_pending = []
    for g in raw.get("_pendingSemanticGapFill") or []:
        arch = g["archetype"]
        entries = archetypes.get(arch) or []
        if not entries:
            continue
        sem = (entries[0].get("semantics") or {})
        still_missing = [k for k in g.get("missing", [])
                         if sem.get(k) in (None, [])]
        if still_missing:
            real_pending.append({"archetype": arch, "missing": still_missing,
                                  "where": where or "(flat)"})
    for g in real_pending:
        warnings.append(f"{g['where']}: {g['archetype']} still missing {g['missing']}")
    return errors, warnings, len(real_pending)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True,
                    help="Pack root (flat or manifest)")
    ap.add_argument("--strict", action="store_true",
                    help="Treat pending gaps as hard errors (exit 1)")
    args = ap.parse_args()

    pack_root = Path(args.pack).expanduser().resolve()
    pack_path = pack_root / "pack.json"
    if not pack_path.exists():
        sys.exit(f"pack.json not found at {pack_path}")
    raw = json.loads(pack_path.read_text())
    vocab = _vocab()

    all_errors: list[str] = []
    all_warnings: list[str] = []
    pending_count = 0

    if is_manifest(raw):
        for entry in raw.get("subpacks", []):
            sub_pack_path = pack_root / entry["path"] / "pack.json"
            if not sub_pack_path.exists():
                all_errors.append(f"sub-pack {entry['theme']!r}: pack.json not found")
                continue
            e, w, p = _validate_flat_pack(sub_pack_path, vocab, where=entry["theme"])
            all_errors.extend(e)
            all_warnings.extend(w)
            pending_count += p
    else:
        e, w, p = _validate_flat_pack(pack_path, vocab)
        all_errors.extend(e)
        all_warnings.extend(w)
        pending_count += p

    for w in all_warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in all_errors:
        print(f"error:   {e}", file=sys.stderr)

    if all_errors:
        return 1
    if pending_count and args.strict:
        return 1
    if pending_count:
        print(f"ok-with-warnings: {pending_count} archetype(s) still need "
              "LLM gap-fill", file=sys.stderr)
        return 2
    print("ok: all archetype semantics validated against vocabulary",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
