#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-check an LLM-produced archetype plan against an asset pack.

The LLM that runs sketch-stage produces a plan JSON directly when
expanding a user's free-form intent into a concrete archetype mix
(see SKILL.md §3.1). This script catches the failure modes the LLM
genuinely can't see by reading the JSON alone:

  - archetype names the LLM hallucinated that don't exist in pack.json
  - archetypes in the plan whose pack entries have sentinel-invalid
    bboxes (gen_pack_json.py leftovers from assets whose payloads
    couldn't be measured — these silently fail at realize time)
  - obvious omissions: archetypes that exist in the pack but the plan
    never references (sometimes intentional, often an LLM oversight)
  - basic sanity: positive integer weights, non-zero attempt budgets,
    each region has a non-empty archetypes map.

Exit codes:
  0  plan is well-formed and consistent with the pack
  1  hard errors (hallucinated archetypes, bad bboxes, malformed JSON)
  2  warnings only (omissions, suspicious-looking weights) — caller
     decides whether to proceed

Usage:
    validate_archetype_plan.py --pack /path/to/asset_pack \\
                               --plan /path/to/expanded_plan.json

Expected plan schema (the LLM produces this directly):

    {
      "intent": "<the user's free-form theme>",
      "targetPlacements": <int>,
      "perRegion": {
        "<region_name>": {
          "attempts": <int>,
          "archetypes": {"<arch_name>": <int_weight>, ...},
          "note": "<LLM's reasoning, optional>"
        },
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_loader import load_pack, select_subpacks


_INVALID_BBOX_SENTINEL = -1e30  # gen_pack_json writes ~-6.8e38 on parse failure


def _load_json(path: Path, label: str) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"{label} not found at {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"{label} at {path} is not valid JSON: {e}")


def _archetype_bbox_ok(entries: list[dict]) -> bool:
    """At least one of the entries under an archetype must have a
    positive bbox. gen_pack_json sometimes leaves sentinel-negative
    sizes when an asset's payloads couldn't be loaded; those archetypes
    will silently fail at realize time."""
    for e in entries:
        size = e.get("size", {})
        if size.get("widthM", 0) > 0 and size.get("widthM", 0) > _INVALID_BBOX_SENTINEL:
            return True
    return False


def validate(pack: dict, plan: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    pack_archs: dict = pack.get("archetypes", {})
    if not pack_archs:
        errors.append("pack has no archetypes")
        return errors, warnings

    valid_pack_archs = {n for n, entries in pack_archs.items()
                       if _archetype_bbox_ok(entries)}
    invalid_pack_archs = set(pack_archs) - valid_pack_archs

    regions = plan.get("perRegion") or {}
    if not regions:
        errors.append("plan has no `perRegion` block, or it's empty")
        return errors, warnings

    used_archs: set[str] = set()
    for region_name, region in regions.items():
        if "attempts" not in region or not isinstance(region["attempts"], int):
            errors.append(f"region {region_name!r}: missing or non-int `attempts`")
        elif region["attempts"] <= 0:
            errors.append(f"region {region_name!r}: `attempts` must be > 0 "
                          f"(got {region['attempts']})")
        archs = region.get("archetypes") or {}
        if not archs:
            errors.append(f"region {region_name!r}: empty `archetypes` map")
            continue
        for arch, weight in archs.items():
            used_archs.add(arch)
            if arch not in pack_archs:
                errors.append(
                    f"region {region_name!r}: archetype {arch!r} is not in "
                    f"the pack (typo or LLM hallucination?)")
                continue
            if arch in invalid_pack_archs:
                errors.append(
                    f"region {region_name!r}: archetype {arch!r} exists in "
                    f"the pack but has an invalid bbox (gen_pack_json "
                    f"sentinel); will silently fail at realize")
            if not isinstance(weight, int) or weight <= 0:
                errors.append(
                    f"region {region_name!r}: archetype {arch!r} has "
                    f"non-positive-int weight ({weight!r})")

    # Warnings — non-blocking.
    unused = sorted(valid_pack_archs - used_archs)
    if unused:
        warnings.append(
            f"the pack has these valid archetypes that no region uses: "
            f"{unused}. If this is intentional (off-theme for the intent), "
            f"ignore. Otherwise the LLM may have missed them.")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True,
                    help="Asset pack root (flat or manifest)")
    ap.add_argument("--subpack", action="append", default=[],
                    help="Restrict validation to these theme(s) for manifest "
                         "packs (repeatable). Default: all sub-packs included.")
    ap.add_argument("--plan", required=True,
                    help="LLM-produced archetype plan JSON")
    ap.add_argument("--warnings-as-errors", action="store_true",
                    help="Treat warnings as hard errors (exit 1)")
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
    plan = _load_json(Path(args.plan).expanduser(), "plan")

    errors, warnings = validate(pack, plan)

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in errors:
        print(f"error:   {e}", file=sys.stderr)

    if errors:
        return 1
    if warnings and args.warnings_as_errors:
        return 1
    if warnings:
        return 2
    print(f"ok: plan validated against pack ({len(plan.get('perRegion', {}))} "
          f"region(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
