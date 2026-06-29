#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run handle performance adapters and enforce operation budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from command_utils import ensure_executable, split_command


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_adapter(command: str, targets: Path) -> Any:
    argv = split_command(command)
    argv.append(str(targets))
    ensure_executable(argv)
    result = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"adapter command failed with exit {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--adapter-cmd", required=True)
    args = parser.parse_args()

    spec = load_json(args.targets)
    actual = run_adapter(args.adapter_cmd, args.targets)

    measurement_model = spec.get("measurement_model", {})
    if measurement_model and not actual.get("measurement_model"):
        actual["measurement_model"] = measurement_model
    if measurement_model:
        print(f"Measurement: {measurement_model.get('id', 'unspecified')}")
        if measurement_model.get("description"):
            print(measurement_model["description"])

    failed = False
    results_by_id: dict[str, dict[str, Any]] = {
        result["id"]: result for result in actual.get("results", [])
    }

    for target in spec.get("targets", []):
        target_id = target["id"]
        result = results_by_id.get(target_id)
        if not result:
            print(f"FAIL {target_id}: missing adapter result")
            failed = True
            continue
        if not result.get("ok", False):
            print(f"FAIL {target_id}: adapter reported failure {result}")
            failed = True
            continue
        ns_per_op = float(result["ns_per_op"])
        max_ns = float(target.get("max_ns_per_op", "inf"))
        status = "PASS" if ns_per_op <= max_ns else "SLOW"
        if status == "SLOW":
            failed = True
        print(f"{status} {target_id}: {ns_per_op:.3f} ns/op target<={max_ns:.3f} ns/op")

    for ratio in spec.get("ratios", []):
        ratio_id = ratio["id"]
        num = results_by_id.get(ratio["numerator"])
        den = results_by_id.get(ratio["denominator"])
        if not num or not den:
            print(f"FAIL {ratio_id}: missing ratio input")
            failed = True
            continue
        den_value = float(den["ns_per_op"])
        value = float("inf") if den_value == 0.0 else float(num["ns_per_op"]) / den_value
        max_ratio = float(ratio["max_ratio"])
        status = "PASS" if value <= max_ratio else "SLOW"
        if status == "SLOW":
            failed = True
        print(f"{status} {ratio_id}: ratio={value:.3f} target<={max_ratio:.3f}")

    print(json.dumps(actual, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
