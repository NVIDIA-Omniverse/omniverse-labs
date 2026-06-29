#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score generated single-layer dump output against golden cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from command_utils import ensure_executable, split_command


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def diff(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: type mismatch expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        ek = set(expected)
        ak = set(actual)
        if ek != ak:
            missing = sorted(ek - ak)
            extra = sorted(ak - ek)
            return f"{path}: key mismatch missing={missing} extra={extra}"
        for key in sorted(ek):
            d = diff(expected[key], actual[key], f"{path}.{key}")
            if d:
                return d
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length mismatch expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            d = diff(e, a, f"{path}[{i}]")
            if d:
                return d
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def run_dump_path(command: str, input_path: str | Path) -> Any:
    argv = split_command(command)
    # Pass the location through verbatim. Routing it through Path here would
    # rewrite forward slashes to backslashes on Windows, including inside a
    # package-entry suffix like foo.usdz[scenes/root.usda], corrupting the
    # authored package path. benchmark.py passes the raw fixture string too.
    argv.append(str(input_path))
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
            f"dump command failed for {input_path} with exit {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def fixture_exists(fixture: str) -> bool:
    if "[" in fixture and fixture.endswith("]"):
        outer = fixture.split("[", 1)[0]
        return Path(outer).exists()
    return Path(fixture).exists()


def case_text_input(case_input: dict[str, Any]) -> tuple[str, str] | None:
    if "text" in case_input:
        extension = case_input.get("extension")
        if not isinstance(extension, str) or not extension:
            raise ValueError("input.text cases must include a non-empty input.extension")
        return str(case_input["text"]), extension
    if "usda" in case_input:
        return str(case_input["usda"]), "usda"
    return None


def run_dump_cmd(command: str, case_input: dict[str, Any], case_id: str) -> Any:
    if "fixture" in case_input:
        fixture = str(case_input["fixture"])
        if not fixture_exists(fixture):
            raise FileNotFoundError(f"fixture does not exist: {fixture}")
        return run_dump_path(command, fixture)

    text_input = case_text_input(case_input)
    if text_input is None:
        raise ValueError("case input must contain input.fixture, input.text, or input.usda")
    text, extension = text_input
    suffix = extension if extension.startswith(".") else "." + extension
    with tempfile.TemporaryDirectory(prefix="usd-skillgraph-") as td:
        input_path = Path(td) / f"{case_id}{suffix}"
        input_path.write_text(text, encoding="utf-8")
        return run_dump_path(command, input_path)


def load_actual_file(actual_dir: Path, case_id: str) -> Any:
    return load_json(actual_dir / f"{case_id}.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldens", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dump-cmd")
    group.add_argument("--actual-dir", type=Path)
    parser.add_argument("--dump-failures", type=Path)
    args = parser.parse_args()

    suite = load_json(args.goldens)
    cases = suite.get("cases", [])
    failures: list[dict[str, Any]] = []

    print(f"Scoring suite: {suite.get('suite', args.goldens.name)}")
    print(f"Cases: {len(cases)}")

    for case in cases:
        case_id = case["id"]
        expected = case["expected"]
        try:
            if args.dump_cmd:
                actual = run_dump_cmd(args.dump_cmd, case["input"], case_id)
            else:
                actual = load_actual_file(args.actual_dir, case_id)
            d = diff(canonical(expected), canonical(actual))
        except Exception as exc:
            d = str(exc)
            actual = None
        if d:
            print(f"FAIL {case_id}: {d}")
            failures.append({"id": case_id, "error": d, "actual": actual})
        else:
            print(f"PASS {case_id}")

    passed = len(cases) - len(failures)
    print(f"OVERALL: {passed}/{len(cases)} ({(passed / len(cases) * 100.0) if cases else 0:.1f}%)")

    if args.dump_failures:
        args.dump_failures.parent.mkdir(parents=True, exist_ok=True)
        args.dump_failures.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
