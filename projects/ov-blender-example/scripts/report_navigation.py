#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print presented FPS from one navigation throughput record."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys

import navigation


def analyze(path: Path) -> float:
    record = navigation.load_json(path)
    if not isinstance(record, Mapping):
        raise navigation.ContractError("navigation record must be a JSON object")
    blockers = navigation.validate_frame_latency_record(record)
    if blockers:
        raise navigation.ContractError(
            f"invalid navigation record: {', '.join(blockers)}"
        )

    runs = record["runs"]
    if len(runs) != 1 or runs[0].get("measurement_complete") is not True:
        raise navigation.ContractError("navigation record must contain one complete run")

    run = runs[0]
    duration_ns = int(run["measurement_end_monotonic_ns"]) - int(
        run["measurement_start_monotonic_ns"]
    )
    return len(run["frame_events"]) * 1_000_000_000 / duration_ns


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path)
    args = parser.parse_args(argv)
    try:
        fps = analyze(args.filename)
    except navigation.ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"FPS: {fps:.6f} frames/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
