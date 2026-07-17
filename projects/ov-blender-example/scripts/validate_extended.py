#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Extended Validation's authoritative suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SUITES = ("golden-large", "performance-large")
RESULT_FILE = "composition.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-root", type=Path, default=ROOT / "addon")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runtime_root = args.runtime_root.resolve()
    output_dir = args.output_dir.resolve()

    results = []
    for suite in SUITES:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "validate_suite.py"),
            suite,
            "--addon-root",
            str(args.addon_root.resolve()),
            "--runtime-root",
            str(runtime_root),
            "--output-dir",
            str(output_dir / suite),
        ]
        results.append(
            {
                "suite": suite,
                "exit_code": subprocess.run(command, cwd=ROOT, check=False).returncode,
            }
        )
    outcome = _outcome(results)
    record = {
        "schema_version": 1,
        "kind": "extended-validation",
        "suites": results,
        "outcome": outcome,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RESULT_FILE).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"pass": 0, "fail": 1, "unavailable": 2}[outcome]


def _outcome(results: Sequence[Mapping[str, Any]]) -> str:
    codes = {item.get("exit_code") for item in results}
    return "unavailable" if codes - {0, 1} else "fail" if 1 in codes else "pass"


if __name__ == "__main__":
    raise SystemExit(main())
