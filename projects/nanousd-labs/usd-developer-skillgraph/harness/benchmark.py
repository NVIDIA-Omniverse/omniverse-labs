#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal benchmark runner for generated dump commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from command_utils import ensure_executable, split_command


def fixture_exists(fixture: str) -> bool:
    if "[" in fixture and fixture.endswith("]"):
        return Path(fixture.split("[", 1)[0]).exists()
    return Path(fixture).exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="benchmarks/targets.json", type=Path)
    parser.add_argument("--dump-cmd", required=True)
    args = parser.parse_args()

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    argv_base = split_command(args.dump_cmd)
    ensure_executable(argv_base)
    measurement_model = targets.get("measurement_model", {})
    if measurement_model:
        print(f"Measurement: {measurement_model.get('id', 'unspecified')}")
        if measurement_model.get("description"):
            print(measurement_model["description"])

    results = []
    failed = False
    for target in targets["targets"]:
        fixture = str(target["fixture"])
        if not fixture_exists(fixture):
            print(f"SKIP {target['id']}: missing fixture {fixture}")
            continue
        runs = int(target.get("runs", 5))
        elapsed = []
        for _ in range(runs):
            start = time.perf_counter()
            result = subprocess.run(
                argv_base + [fixture],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            elapsed.append(time.perf_counter() - start)
            if result.returncode != 0:
                print(f"FAIL {target['id']}: command exited {result.returncode}")
                failed = True
                break
        if not elapsed:
            continue
        best_ms = min(elapsed) * 1000.0
        max_ms = float(target.get("max_ms", "inf"))
        status = "PASS" if best_ms <= max_ms else "SLOW"
        if status == "SLOW":
            failed = True
        print(f"{status} {target['id']}: best={best_ms:.3f} ms target<={max_ms:.3f} ms")
        results.append({"id": target["id"], "best_ms": best_ms, "max_ms": max_ms, "status": status})

    print(json.dumps({"measurement_model": measurement_model, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
