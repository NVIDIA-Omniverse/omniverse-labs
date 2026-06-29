#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score a handle adapter against language-neutral golden programs."""

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


def _resolve_local_fixture(fixture_path: str) -> dict[str, str]:
    """Mirror usd_resource_protocol::read_resource(classify_resolved(...)) for a LocalFile.

    Adapters must never load fixtures via raw std::ifstream / read_binary_file; the
    contract requires bytes to arrive through the resource protocol layer. The
    harness performs that read once here and feeds the adapter pre-opened bytes
    via bytes_hex + file_backed_path. Adapters that still hit the filesystem are
    flagged by the no_ifstream_in_adapters / no_read_binary_file_in_adapters
    lint rules in contracts/lint/usdc-single-layer.lint.json.
    """
    raw = Path(fixture_path).read_bytes()
    return {
        "bytes_hex": raw.hex(),
        "file_backed_path": str(Path(fixture_path)),
    }


def _resolve_opened_resources(node: Any) -> Any:
    """Recursively walk the case input and pre-resolve any opened_resource.fixture key.

    Every dict whose `opened_resource` value contains a `fixture` key has the
    fixture path replaced with bytes_hex + file_backed_path (matching the shape
    a ResourceReadResult would deliver). Leaves other fields untouched.
    """
    if isinstance(node, dict):
        opened = node.get("opened_resource")
        if isinstance(opened, dict) and "fixture" in opened and "bytes_hex" not in opened:
            fixture = opened.pop("fixture")
            opened.update(_resolve_local_fixture(fixture))
        return {k: _resolve_opened_resources(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_opened_resources(item) for item in node]
    return node


def run_adapter(command: str, case_input: dict[str, Any], case_id: str) -> Any:
    resolved_input = _resolve_opened_resources(case_input)
    with tempfile.TemporaryDirectory(prefix="usd-path-handle-") as td:
        input_path = Path(td) / f"{case_id}.json"
        input_path.write_text(json.dumps(resolved_input, indent=2, ensure_ascii=False), encoding="utf-8")
        argv = split_command(command)
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
                f"adapter command failed for {case_id} with exit {result.returncode}\n"
                f"stderr:\n{result.stderr}"
            )
        return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldens", required=True, type=Path)
    parser.add_argument("--adapter-cmd", required=True)
    parser.add_argument("--dump-failures", type=Path)
    args = parser.parse_args()

    suite = load_json(args.goldens)
    cases = suite.get("cases", [])
    failures: list[dict[str, Any]] = []

    print(f"Scoring handle suite: {suite.get('suite', args.goldens.name)}")
    print(f"Contract: {suite.get('contract', '<unspecified>')}")
    print(f"Cases: {len(cases)}")

    for case in cases:
        case_id = case["id"]
        expected = case["expected"]
        try:
            actual = run_adapter(args.adapter_cmd, case["input"], case_id)
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
