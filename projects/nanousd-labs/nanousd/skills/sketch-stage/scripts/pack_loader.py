# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared loader for both flat and manifest pack.json shapes.

A pack root can be:

  - **Flat** — pack.json has `archetypes: {name: [...]}`. This is the
    historical shape and what gen_pack_json produces with `--flat`.
  - **Manifest** — pack.json has `_isManifest: true` and `subpacks: [{name,
    theme, path, ...}]`. Each sub-pack lives in `<root>/<path>/pack.json`
    and is itself flat.

`load_pack(root)` returns one merged view of either shape:

    {
      "name": "...",
      "scenarios": [...],
      "upAxis": "Z",                  # voted across sub-packs (most common)
      "metersPerUnit": 1.0,            # voted across sub-packs (most common)
      "archetypes": {
          "<theme>.<name>": [...],     # manifest: namespaced by theme
          "<name>": [...],             # flat: original archetype name
      },
      "subpacks": [                    # empty for flat
          {"theme": "<theme>", "path": "<theme>", "archetypeCount": 3, ...},
          ...
      ],
      "isManifest": True/False,
      "raw": {<the as-loaded pack.json>},
    }

Why namespacing matters: two themes can legitimately contain overlapping
archetype names (e.g. a "table" in two different sub-packs). Forcing a
`<theme>.<name>` namespace at merge time keeps them distinguishable.
Callers that want only one theme should pass the sub-pack root directly
(which is a flat pack); the namespace prefix is for "all themes at once"
operations.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def _read_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}")


def is_manifest(pack: dict) -> bool:
    return bool(pack.get("_isManifest")) or (
        "subpacks" in pack and "archetypes" not in pack
    )


def load_pack(root) -> dict:
    """Load either a flat or a manifest pack.json. See module docstring."""
    root = Path(root).expanduser().resolve()
    pack_path = root / "pack.json"
    raw = _read_json(pack_path)

    if not is_manifest(raw):
        return {
            "name": raw.get("name", root.name),
            "scenarios": raw.get("scenarios", []),
            "upAxis": raw.get("upAxis", "Z"),
            "metersPerUnit": raw.get("metersPerUnit", 1.0),
            "archetypes": raw.get("archetypes", {}),
            "subpacks": [],
            "isManifest": False,
            "raw": raw,
        }

    # Manifest: load each sub-pack and namespace by theme.
    merged_archetypes: dict[str, list] = {}
    subpack_briefs: list[dict] = []
    up_votes: Counter = Counter()
    mpu_votes: Counter = Counter()

    for entry in raw.get("subpacks", []):
        sub_path = root / entry["path"]
        try:
            sub_raw = _read_json(sub_path / "pack.json")
        except (FileNotFoundError, ValueError) as e:
            print(f"warning: sub-pack {entry['path']!r} unreadable: {e}",
                  file=sys.stderr)
            continue
        theme = entry.get("theme", sub_raw.get("theme", entry.get("name")))
        for arch_name, items in (sub_raw.get("archetypes") or {}).items():
            ns_name = f"{theme}.{arch_name}"
            # Rewrite each item's `path` so it remains valid relative to the
            # MANIFEST root rather than the sub-pack root.
            rewritten = []
            for it in items:
                ni = dict(it)
                if "path" in ni:
                    ni["path"] = f"{entry['path']}/{ni['path']}"
                rewritten.append(ni)
            merged_archetypes[ns_name] = rewritten
        up_votes[sub_raw.get("upAxis", "Z")] += 1
        mpu_votes[sub_raw.get("metersPerUnit", 1.0)] += 1
        subpack_briefs.append({
            "name": entry.get("name", theme),
            "theme": theme,
            "path": entry["path"],
            "archetypeCount": len(sub_raw.get("archetypes") or {}),
            "assetCount": int((sub_raw.get("_scanStats") or {}).get("kept", 0)),
            "upAxis": sub_raw.get("upAxis", "Z"),
            "metersPerUnit": sub_raw.get("metersPerUnit", 1.0),
        })

    up = up_votes.most_common(1)[0][0] if up_votes else "Z"
    mpu = mpu_votes.most_common(1)[0][0] if mpu_votes else 1.0

    return {
        "name": raw.get("name", root.name),
        "scenarios": raw.get("scenarios", []),
        "upAxis": up,
        "metersPerUnit": mpu,
        "archetypes": merged_archetypes,
        "subpacks": subpack_briefs,
        "isManifest": True,
        "raw": raw,
    }


def select_subpacks(pack: dict, themes: list[str]) -> dict:
    """Filter a loaded manifest pack down to the named themes.

    Returns a dict in the same shape as `load_pack`, but with `archetypes`
    restricted to entries whose theme prefix is in `themes`. For a flat pack
    this is a no-op (returns the input unchanged).
    """
    if not pack.get("isManifest"):
        return pack
    keep = set(themes)
    filtered = {k: v for k, v in pack["archetypes"].items()
                if k.split(".", 1)[0] in keep}
    return {
        **pack,
        "archetypes": filtered,
        "subpacks": [b for b in pack["subpacks"] if b["theme"] in keep],
    }


def main():  # CLI debug entrypoint: pretty-print what load_pack sees.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True,
                    help="Pack root (manifest or flat)")
    ap.add_argument("--subpack", action="append", default=[],
                    help="Restrict to these themes (repeatable; only meaningful "
                         "for manifests). Default: all themes.")
    args = ap.parse_args()

    pack = load_pack(args.pack)
    if args.subpack:
        pack = select_subpacks(pack, args.subpack)
    print(json.dumps({
        "name": pack["name"],
        "isManifest": pack["isManifest"],
        "upAxis": pack["upAxis"],
        "metersPerUnit": pack["metersPerUnit"],
        "subpacks": pack["subpacks"],
        "archetypes": {k: f"{len(v)} asset(s)" for k, v in pack["archetypes"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
