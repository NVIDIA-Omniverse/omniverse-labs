#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run nanousd stage-load benchmarks across backend implementations."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from command_utils import split_command


NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_bytes(text: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB)\s*", text)
    if not match:
        raise ValueError(f"unrecognized byte quantity: {text!r}")
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
    return int(value * scale)


def find_float(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing benchmark field {name}")
    return float(match.group(1))


def find_int(pattern: str, text: str, name: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing benchmark field {name}")
    return int(match.group(1))


def find_bytes(pattern: str, text: str, name: str) -> int:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing benchmark field {name}")
    return parse_bytes(match.group(1))


def parse_stage_load_output(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "open_ms": find_float(r"Stage open \(parse \+ compose\):\s*" + NUMBER + r"\s*ms", stdout, "open_ms"),
        "traverse_ms": find_float(r"Traverse all prims \+ attribs:\s*" + NUMBER + r"\s*ms", stdout, "traverse_ms"),
        "traverse_build_ms": find_float(r"traverse \(build prim list\):\s*" + NUMBER + r"\s*ms", stdout, "traverse_build_ms"),
        "diagnostics_ms": find_float(r"Collect diagnostics:\s*" + NUMBER + r"\s*ms", stdout, "diagnostics_ms"),
        "total_ms": find_float(r"Total:\s*" + NUMBER + r"\s*ms", stdout, "total_ms"),
        "rss_before_bytes": find_bytes(r"Before open:\s*([0-9.]+\s*(?:B|KB|MB|GB))", stdout, "rss_before_bytes"),
        "rss_after_bytes": find_bytes(r"After open:\s*([0-9.]+\s*(?:B|KB|MB|GB))", stdout, "rss_after_bytes"),
        "rss_delta_bytes": find_bytes(r"Delta:\s*([0-9.]+\s*(?:B|KB|MB|GB))", stdout, "rss_delta_bytes"),
        "rss_peak_bytes": find_bytes(r"Peak:\s*([0-9.]+\s*(?:B|KB|MB|GB))", stdout, "rss_peak_bytes"),
        "total_prims": find_int(r"Total prims:\s*([0-9]+)", stdout, "total_prims"),
        "total_attributes": find_int(r"Total attributes:\s*([0-9]+)", stdout, "total_attributes"),
        "diagnostic_count": find_int(r"--- Diagnostics:\s*([0-9]+)\s+total\s*---", stdout, "diagnostic_count"),
    }

    loop = re.search(
        r"loop \(nprims iterations\):\s*" + NUMBER + r"\s*ms\s*"
        r"\(prim_handle=([0-9.]+)\s+nattribs=([0-9.]+)\s+free=([0-9.]+)\)",
        stdout,
        re.MULTILINE,
    )
    if not loop:
        raise ValueError("missing loop timing breakdown")
    metrics["loop_ms"] = float(loop.group(1))
    metrics["prim_handle_ms"] = float(loop.group(2))
    metrics["nattribs_ms"] = float(loop.group(3))
    metrics["freeprim_ms"] = float(loop.group(4))

    prims = metrics["total_prims"]
    if prims > 0:
        metrics["traverse_ns_per_prim"] = metrics["traverse_ms"] * 1_000_000.0 / prims
        metrics["prim_handle_ns_per_prim"] = metrics["prim_handle_ms"] * 1_000_000.0 / prims
        metrics["nattribs_ns_per_prim"] = metrics["nattribs_ms"] * 1_000_000.0 / prims
        metrics["freeprim_ns_per_prim"] = metrics["freeprim_ms"] * 1_000_000.0 / prims
        metrics["rss_delta_bytes_per_prim"] = metrics["rss_delta_bytes"] / prims
        metrics["attributes_per_prim"] = metrics["total_attributes"] / prims
    return metrics


def parse_backend(value: str) -> dict[str, str | None]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("backend must use id=path or id=DEFAULT")
    backend_id, backend_path = value.split("=", 1)
    backend_id = backend_id.strip()
    backend_path = backend_path.strip()
    if not backend_id:
        raise argparse.ArgumentTypeError("backend id cannot be empty")
    if not backend_path or backend_path.upper() == "DEFAULT":
        return {"id": backend_id, "path": None}
    return {"id": backend_id, "path": backend_path}


def run_benchmark_once(
    benchmark_argv: list[str],
    fixture: Path,
    backend: dict[str, str | None],
) -> dict[str, Any]:
    env = os.environ.copy()
    backend_path = backend["path"]
    if backend_path is None:
        env.pop("NANOUSD_BACKEND", None)
    else:
        env["NANOUSD_BACKEND"] = backend_path

    result = subprocess.run(
        benchmark_argv + [str(fixture)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    try:
        metrics = parse_stage_load_output(result.stdout)
    except ValueError as exc:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parse_error": str(exc),
        }
    return {"ok": True, "returncode": result.returncode, "metrics": metrics}


def summarize_runs(target: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [run for run in runs if not run.get("ok")]
    if failures:
        return {"ok": False, "runs": runs, "failure": failures[0]}

    selection_metric = target.get("selection_metric", "total_ms")
    best = min(runs, key=lambda run: float(run["metrics"][selection_metric]))
    return {
        "ok": True,
        "runs": runs,
        "selection_metric": selection_metric,
        "best": best["metrics"],
    }


def compare_results(results: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (result["target"], result["backend"]): result
        for result in results
    }
    output = []
    for comparison in comparisons:
        target = comparison["target"]
        metric = comparison["metric"]
        numerator_key = (target, comparison["numerator_backend"])
        denominator_key = (target, comparison["denominator_backend"])
        required = bool(comparison.get("required", True))
        numerator = by_key.get(numerator_key)
        denominator = by_key.get(denominator_key)
        if not numerator or not denominator or not numerator.get("ok") or not denominator.get("ok"):
            output.append({
                "id": comparison["id"],
                "ok": not required,
                "status": "SKIP" if not required else "FAIL",
                "reason": "missing_or_failed_backend_result",
            })
            continue
        denominator_value = float(denominator["best"][metric])
        ratio = float("inf") if denominator_value == 0.0 else float(numerator["best"][metric]) / denominator_value
        max_ratio = float(comparison["max_ratio"])
        status = "PASS" if ratio <= max_ratio else "SLOW"
        output.append({
            "id": comparison["id"],
            "ok": status == "PASS" or not required,
            "status": status,
            "target": target,
            "metric": metric,
            "ratio": ratio,
            "max_ratio": max_ratio,
            "required": required,
        })
    return output


def format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--benchmark-cmd", required=True, help="Command for nanousd benchmark_stage_load")
    parser.add_argument(
        "--backend",
        action="append",
        type=parse_backend,
        default=[],
        help="Backend as id=path. Use id=DEFAULT to leave NANOUSD_BACKEND unset.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    spec = load_json(args.targets)
    benchmark_argv = split_command(args.benchmark_cmd)
    backends = args.backend or [{"id": "nanousd_default", "path": None}]
    measurement_model = spec.get("measurement_model", {})

    print(f"Measurement: {measurement_model.get('id', 'nanousd_backend_stage_load')}")
    if measurement_model.get("description"):
        print(measurement_model["description"])

    results = []
    failed = False
    for target in spec.get("targets", []):
        fixture = Path(target["fixture"])
        if not fixture.exists():
            print(f"SKIP {target['id']}: missing fixture {fixture}")
            continue
        runs = int(target.get("runs", spec.get("runs", 3)))
        for backend in backends:
            run_results = [
                run_benchmark_once(benchmark_argv, fixture, backend)
                for _ in range(runs)
            ]
            summary = summarize_runs(target, run_results)
            record = {
                "target": target["id"],
                "fixture": str(fixture),
                "backend": backend["id"],
                **summary,
            }
            results.append(record)
            if summary["ok"]:
                best = summary["best"]
                print(
                    f"OK {target['id']} {backend['id']}: "
                    f"total={format_ms(best['total_ms'])} "
                    f"open={format_ms(best['open_ms'])} "
                    f"traverse={format_ms(best['traverse_ms'])} "
                    f"rss_delta={int(best['rss_delta_bytes'])} B"
                )
            else:
                failed = True
                failure = summary["failure"]
                reason = failure.get("parse_error") or failure.get("stderr") or failure.get("stdout")
                print(f"FAIL {target['id']} {backend['id']}: {str(reason).strip()}")

    comparisons = compare_results(results, spec.get("comparisons", []))
    for comparison in comparisons:
        status = comparison["status"]
        if status == "SKIP":
            print(f"SKIP {comparison['id']}: {comparison['reason']}")
        else:
            print(
                f"{status} {comparison['id']}: "
                f"ratio={comparison['ratio']:.3f} target<={comparison['max_ratio']:.3f}"
            )
        if not comparison["ok"]:
            failed = True

    output = {
        "suite": spec.get("suite", "nanousd-backend-stage-load"),
        "measurement_model": measurement_model,
        "benchmark_cmd": args.benchmark_cmd,
        "backends": backends,
        "results": results,
        "comparisons": comparisons,
    }
    print(json.dumps(output, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
