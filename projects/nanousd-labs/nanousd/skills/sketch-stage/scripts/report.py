#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concise report on a sketch-stage run output dir.

Usage:
    python3 scripts/report.py <out_dir> [--session-url URL]

Reads <out_dir>/realized/manifest.json (if present) and optionally pings the
live session HTTP for in-memory counts. Prints paths, placement counts,
composition strategy, per-archetype real-asset breakdown, and unfilled
warnings. Intended for the LLM's end-of-run summary; replaces ad-hoc
inline-python heredoc snippets that previously bloated tool calls.
"""
import argparse
import json
import pathlib
import urllib.request
from collections import Counter


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def report(out_dir: pathlib.Path, session_url: str) -> None:
    manifest_path = out_dir / "realized" / "manifest.json"
    snapshot_path = out_dir / "_snapshot.sketch.json"

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        s = m.get("summary", {})
        cs = m.get("compositionStrategy", {})
        tb = m.get("templateBreakdown", {})
        layers = m.get("layers", {})

        section("paths")
        print(f"root.usd : {layers.get('rootPath', out_dir / 'realized' / 'root.usd')}")
        print(f"manifest : {manifest_path}")
        print(f"snapshot : {snapshot_path}")

        section("placement counts")
        print(f"sketch total      : {s.get('placements')}")
        print(f"realized (filled) : {s.get('filled')}")
        print(f"unfilled          : {len(s.get('unfilled') or [])}")
        print(f"composed prims    : {m.get('composedPrimCount_usdcore')}")
        print(f"anchor placements : {tb.get('anchorPlacementCount')}")
        print(f"added placements  : {tb.get('addedPlacementCount')}")

        section("composition strategy")
        for k in ("composition", "hierarchy", "structure",
                  "instancing", "upAxis", "metersPerUnit"):
            print(f"{k:10s}: {cs.get(k)}")

        section("by archetype (real assets used)")
        for a, v in (s.get("byArchetype") or {}).items():
            picks = ", ".join(f"{p}={n}" for p, n in (v.get("picks") or {}).items())
            print(f"  {a:14s} filled={v.get('filled', 0):6d}  picks: {picks}")

        section("unfilled / warnings")
        unfilled = s.get("unfilled") or []
        c = Counter((u.get("archetype"), u.get("reason")) for u in unfilled)
        if not c:
            print("(none)")
        for (a, r), n in c.most_common():
            print(f"  {a:12s} reason='{r}': {n}")
    else:
        print(f"no manifest at {manifest_path}")

    section(f"session status ({session_url})")
    try:
        st = json.loads(urllib.request.urlopen(f"{session_url}/status", timeout=2).read())
        for k in ("placementCount", "anchorPlacementCount", "addedPlacementCount",
                  "zoneCount", "archetypes", "footprintM"):
            print(f"  {k}: {st.get(k)}")
    except Exception as e:
        print(f"  session not reachable: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=pathlib.Path, help="sketch-stage run output dir")
    p.add_argument("--session-url", default="http://127.0.0.1:8766",
                   help="live session HTTP base URL (default: %(default)s)")
    args = p.parse_args()
    report(args.out_dir, args.session_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
