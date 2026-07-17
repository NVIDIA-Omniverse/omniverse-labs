#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare every discovered test fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from fixture_manifest import fixture_input, load_catalog, render_fixture


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
from semantic_validation import GOLDENS  # noqa: E402


def fixture_ids(suite: str) -> tuple[str, ...]:
    if suite == "blender-integration":
        return ("demo_stair_drop_1280x720",)
    if suite in {"performance-large", "performance-small"}:
        return ("perf_junk_shop_1280x720",)
    if suite not in GOLDENS:
        raise ValueError(f"suite has no fixture preparation contract: {suite}")
    return tuple(fixture for fixture, _golden in GOLDENS[suite])


def _verify(catalog: dict, fixture: dict) -> None:
    fixture_id = str(fixture["id"])
    if "blend_file" in fixture:
        fixture_input(catalog, fixture_id)
    else:
        render_fixture(catalog, fixture_id)


def _valid(catalog: dict, fixture_id: str) -> bool:
    fixture = next(item for item in catalog["fixtures"] if item["id"] == fixture_id)
    try:
        _verify(catalog, fixture)
        return True
    except (FileNotFoundError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--suite")
    args = parser.parse_args()
    catalog = dict(load_catalog(ROOT))
    ids = {str(item["id"]) for item in catalog["fixtures"]}
    selected = set(fixture_ids(args.suite)) if args.suite else ids
    for script in sorted(ROOT.glob("*/prepare.py")):
        spec = next(item for item in catalog["fixtures"] if ROOT / item["id"] == script.parent)
        if spec["id"] not in selected:
            continue
        outputs = spec.get("preparation", {}).get("outputs", [spec["id"]])
        if not isinstance(outputs, list) or not outputs or not set(outputs) <= ids:
            raise ValueError(f"invalid preparation outputs: {script}")
        if not args.force and all(_valid(catalog, str(item)) for item in outputs):
            print(f"valid: {spec['id']}")
            continue
        command = [sys.executable, str(script)]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)
        print(f"prepared: {spec['id']}")
    for fixture in catalog["fixtures"]:
        if fixture["id"] in selected:
            _verify(catalog, fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
