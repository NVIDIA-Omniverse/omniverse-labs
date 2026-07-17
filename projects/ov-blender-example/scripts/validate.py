#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Task Validation's authoritative suites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "unit",
    "golden-small",
    "ov-integration",
    "blender-integration",
    "performance-small",
)
RESULT_FILE = "composition.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-root", type=Path, default=ROOT / "addon")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    return run(
        ROOT,
        args.addon_root.resolve(),
        args.runtime_root.resolve(),
        args.output_dir.resolve(),
    )


def run(root: Path, addon_root: Path, runtime_root: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for suite in SUITES:
        command = [
            sys.executable,
            str(root / "scripts" / "validate_suite.py"),
            suite,
            "--addon-root",
            str(addon_root),
            "--runtime-root",
            str(runtime_root),
            "--output-dir",
            str(output_dir / suite),
        ]
        completed = subprocess.run(
            command, cwd=root, text=True, capture_output=True, check=False
        )
        stdout = output_dir / f"{suite}.stdout.log"
        stderr = output_dir / f"{suite}.stderr.log"
        stdout.write_text(completed.stdout, encoding="utf-8")
        stderr.write_text(completed.stderr, encoding="utf-8")
        results.append(
            {
                "suite": suite,
                "exit_code": completed.returncode,
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
            }
        )
    outcome = _outcome(results)
    record = {
        "schema_version": 1,
        "kind": "task-validation",
        "suites": results,
        "outcome": outcome,
    }
    (output_dir / RESULT_FILE).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"pass": 0, "fail": 1, "unavailable": 2}[outcome]


def _outcome(results: Sequence[Mapping[str, Any]]) -> str:
    codes = {item.get("exit_code") for item in results}
    return "unavailable" if codes - {0, 1} else "fail" if 1 in codes else "pass"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path) -> Mapping[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    suites = record.get("suites") if isinstance(record, Mapping) else None
    if (
        set(record) != {"schema_version", "kind", "suites", "outcome"}
        or record.get("schema_version") != 1
        or record.get("kind") != "task-validation"
        or not isinstance(suites, list)
        or [item.get("suite") for item in suites if isinstance(item, Mapping)]
        != list(SUITES)
        or any(
            set(item)
            != {"suite", "exit_code", "stdout_sha256", "stderr_sha256"}
            or type(item.get("exit_code")) is not int
            or item["exit_code"] < 0
            or any(
                item.get(f"{stream}_sha256")
                != _sha256(path.parent / f"{item.get('suite')}.{stream}.log")
                for stream in ("stdout", "stderr")
            )
            for item in suites
        )
        or record.get("outcome") != _outcome(suites)
    ):
        raise ValueError("invalid Task Validation composition evidence")
    performance_measurements(path.parent / "performance-small.stdout.log")
    return record


def performance_measurements(path: Path) -> list[Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("suite") != "performance-small"
        or not isinstance(payload.get("measurements"), list)
    ):
        raise ValueError("invalid small-performance evidence")
    return payload["measurements"]


if __name__ == "__main__":
    raise SystemExit(main())
