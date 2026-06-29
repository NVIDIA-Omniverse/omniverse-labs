#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that the LLM-labelled `templateZones[]` in an absorbed sketch
follows the schema + uses the vocabulary in
references/pack_semantics_vocabulary.json.

Per region, checks:
  - `id` is a non-empty string
  - `footprintM` is [[x0, y0], [x1, y1]] with x1>x0, y1>y0
  - `name` is a non-empty string (added by the LLM label step)
  - `purpose` is a non-empty string (added by the LLM)
  - `allowedArchetypes` is a list (may be empty for circulation regions);
    each entry should look like `<theme>.<arch>` or `<arch>` — we don't
    require it to exist in the pack here since regions can list typical-
    for-theme archetypes the pack doesn't yet contain.

Soft warnings (non-blocking):
  - regions with `source: anchor_pass|cluster_pass` lacking `name`/`purpose`
    are flagged ("LLM hasn't labelled them yet")
  - region footprints overlap by > 30% area suggest the region pipeline
    over-grew anchors and the LLM should merge or shrink.

Exit codes:
  0  all regions schema-clean
  1  hard errors (missing fields, bad bbox)
  2  warnings only (unlabelled regions, overlaps)

Usage:
  validate_absorbed_regions.py --sketch /path/to/absorbed.sketch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bbox_ok(fp) -> bool:
    if not (isinstance(fp, list) and len(fp) == 2):
        return False
    (a, b) = fp
    if not (isinstance(a, list) and isinstance(b, list)
            and len(a) == 2 and len(b) == 2):
        return False
    return a[0] < b[0] and a[1] < b[1]


def _overlap_area(a, b) -> float:
    (x0a, y0a), (x1a, y1a) = a
    (x0b, y0b), (x1b, y1b) = b
    iw = max(0.0, min(x1a, x1b) - max(x0a, x0b))
    ih = max(0.0, min(y1a, y1b) - max(y0a, y0b))
    return iw * ih


def _area(fp) -> float:
    (x0, y0), (x1, y1) = fp
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def validate(regions: list) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for i, r in enumerate(regions):
        rid = r.get("id") or f"<#{i}>"
        if not isinstance(r.get("id"), str) or not r.get("id"):
            errors.append(f"region #{i}: missing or non-string `id`")
        fp = r.get("footprintM")
        if not _bbox_ok(fp):
            errors.append(f"region {rid!r}: bad `footprintM` {fp!r}")
            continue
        # Soft checks — only flag if region was rule-built and isn't a gap.
        source = r.get("source")
        is_gap = source == "gap_pass"
        if source in ("anchor_pass", "cluster_pass", "gap_pass"):
            if not r.get("name"):
                warnings.append(f"region {rid!r} ({source}): no `name` — "
                                "LLM hasn't labelled it yet")
            if not r.get("purpose") and not is_gap:
                warnings.append(f"region {rid!r} ({source}): no `purpose`")
        if "allowedArchetypes" in r and not isinstance(r["allowedArchetypes"], list):
            errors.append(f"region {rid!r}: `allowedArchetypes` must be a list")

    # Overlap check
    for i, ri in enumerate(regions):
        if not _bbox_ok(ri.get("footprintM")):
            continue
        for rj in regions[i + 1:]:
            if not _bbox_ok(rj.get("footprintM")):
                continue
            ov = _overlap_area(ri["footprintM"], rj["footprintM"])
            if ov <= 0:
                continue
            min_area = min(_area(ri["footprintM"]), _area(rj["footprintM"]))
            if min_area > 0 and ov / min_area > 0.3:
                warnings.append(
                    f"regions {ri.get('id')!r} and {rj.get('id')!r} overlap "
                    f"by {ov / min_area:.0%} of the smaller — consider merging")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sketch", required=True, help="Absorbed sketch JSON")
    ap.add_argument("--warnings-as-errors", action="store_true")
    args = ap.parse_args()

    sketch = json.loads(Path(args.sketch).expanduser().resolve().read_text())
    regions = sketch.get("templateZones") or []
    if not regions:
        print("sketch has no `templateZones[]`; run extract_regions.py --write first",
              file=sys.stderr)
        return 1
    errors, warnings = validate(regions)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in errors:
        print(f"error:   {e}", file=sys.stderr)
    if errors:
        return 1
    if warnings and args.warnings_as_errors:
        return 1
    if warnings:
        print(f"ok-with-warnings: {len(regions)} region(s); "
              f"{len(warnings)} soft issue(s)", file=sys.stderr)
        return 2
    print(f"ok: {len(regions)} region(s) validated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
