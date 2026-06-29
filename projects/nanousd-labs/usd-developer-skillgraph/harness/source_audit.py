#!/usr/bin/env python3
"""Source-level audit that mechanically enforces compliance laws the adapters
used to assert via self-attestation ops.

A `source_audit` block on a scope in graph/skillgraph.json names artifacts
(production C++ files and adapter C++ files) whose source must NOT contain any
of a fixed set of forbidden symbols. The script reports the count per artifact
and exits non-zero if any count is nonzero.

This replaces the deleted `inspect_resource_protocol_calls` and
`inspect_dump_construction` golden cases, which let an adapter self-attest
compliance ("calls_to_resource_io": 0) by hard-coding the answer string. The
harness now performs the scan itself.

Forbidden patterns:
- Raw std::ifstream construction in any audited file.
- Calls to read_binary_file (the raw-ifstream wrapper from adapter_runtime.h).
- Direct calls to read_resource / classify_resolved from a *handler* (the
  handler must accept ResourceReadResult from usd-layer-open, not call
  resource protocol itself); this is keyed off the artifact name --
  generated/cpp/usdc_layer_open.{h,cpp} must NOT reference read_resource at
  all. Adapter files (usdc_*_adapter.cpp) ARE expected to call read_resource
  (they're substituting for usd-layer-open in unit tests).
- Direct dump_result calls from USDC parser or layer-open domain files. Dump
  serialization belongs in adapters and dump commands, not in format handlers
  or parsers.
- Canned layer-dump raw-string literals.

Invocation (from regen_graph.py validate_scope or directly):
    python3 harness/source_audit.py --graph graph/skillgraph.json --scope <scope-name>
or for ad-hoc auditing of a file list:
    python3 harness/source_audit.py --artifacts <path1> [<path2> ...]

Exits non-zero on any violation. Prints a JSON report on stdout for
machine consumption and a human-readable summary on stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Forbidden-pattern set. Each entry is (id, regex, applies_to_filter).
# applies_to_filter is None to apply universally, or a callable(path:str)->bool
# to restrict to a subset of artifacts (e.g. handlers but not adapters).
FORBIDDEN = [
    (
        "raw_ifstream",
        re.compile(r"\bstd::ifstream\b"),
        None,
        "Raw std::ifstream construction bypasses usd_resource_protocol::read_resource. "
        "Route every file open through usdsg::read_resource(usdsg::classify_resolved(path)).",
    ),
    (
        "read_binary_file_call",
        re.compile(r"\bread_binary_file\b"),
        None,
        "read_binary_file is the raw-ifstream wrapper from adapter_runtime.h. "
        "It must be deleted and replaced with usdsg::read_resource(usdsg::classify_resolved(path)).",
    ),
    (
        "read_resource_in_handler",
        re.compile(r"\bread_resource\b"),
        lambda p: p.endswith("usdc_layer_open.h") or p.endswith("usdc_layer_open.cpp"),
        "usdc-layer-open accepts a ResourceReadResult from usd-layer-open; calling "
        "read_resource inside the handler is a contract violation "
        "(contracts/handles/usdc-layer-open.handle.json#laws). "
        "Adapters legitimately call read_resource; handlers do not.",
    ),
    (
        "classify_resolved_in_handler",
        re.compile(r"\bclassify_resolved\b"),
        lambda p: p.endswith("usdc_layer_open.h") or p.endswith("usdc_layer_open.cpp"),
        "usdc-layer-open does not resolve identifiers; usd-layer-open performs "
        "classify_resolved + read_resource before invoking this handler.",
    ),
    (
        "dump_result_in_usdc_domain_file",
        re.compile(r"\bdump_result\s*\("),
        lambda p: (
            p.endswith("usdc_layer_open.h")
            or p.endswith("usdc_layer_open.cpp")
            or p.endswith("usdc_parser.h")
            or p.endswith("usdc_parser.cpp")
        ),
        "USDC parser and layer-open domain files must return target-native "
        "LayerResult values. Canonical dump serialization is allowed only in "
        "adapters, diagnostics serialization, and dump commands.",
    ),
    (
        "canned_layer_dump_literal",
        re.compile(r'R"[A-Z0-9_]*\(\s*\{"ok":true,"diagnostics":\[\],"layer":\{', re.DOTALL),
        None,
        "Canned layer-dump raw-string literal. The dump must be assembled by emitting "
        "fields from usd_document_model into a growable output buffer, not returned as a "
        "pre-authored JSON literal selected by fixture content or section bytes.",
    ),
]


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scope_artifacts(graph: dict[str, Any], scope_name: str, target: str | None = None) -> list[str]:
    scope = graph.get("scopes", {}).get(scope_name)
    if scope is None:
        raise KeyError(f"scope {scope_name!r} not found in graph")
    audit = None
    if target:
        audit = scope.get("targets", {}).get(target, {}).get("source_audit")
    if audit is None:
        audit = scope.get("source_audit")
    if not audit:
        return []
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{scope_name}: source_audit.artifacts must be a list")
    return list(artifacts)


def audit_file(repo: Path, rel: str) -> list[dict[str, Any]]:
    path = repo / rel
    if not path.exists():
        # Missing artifacts are treated as zero-violations; the regen step is what
        # produces them. A separate harness check verifies presence.
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[dict[str, Any]] = []
    for fid, pattern, applies, hint in FORBIDDEN:
        if applies is not None and not applies(rel):
            continue
        for m in pattern.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            snippet = _line_at(text, m.start())
            findings.append(
                {
                    "file": rel,
                    "id": fid,
                    "line": line_no,
                    "snippet": snippet,
                    "hint": hint,
                }
            )
    return findings


def _line_at(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    line = text[start:end].strip()
    if len(line) > 200:
        line = line[:200] + "..."
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="graph manifest")
    parser.add_argument("--scope", help="scope name whose source_audit.artifacts to audit")
    parser.add_argument(
        "--target",
        help="target name (e.g. cpp). When set, prefers scope.targets.<target>.source_audit "
        "over scope.source_audit so language-specific audit lists can live per-target.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=[],
        help="explicit artifact list (alternative to --scope)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to harness/..)",
    )
    args = parser.parse_args()

    if args.scope and not args.graph:
        print("source_audit: --scope requires --graph", file=sys.stderr)
        return 2

    artifacts: list[str]
    if args.scope:
        graph = load_graph(args.graph.resolve())
        artifacts = scope_artifacts(graph, args.scope, args.target)
    else:
        artifacts = list(args.artifacts)

    if not artifacts:
        print(json.dumps({"ok": True, "artifacts": [], "violations": []}))
        print("source_audit: no artifacts to audit", file=sys.stderr)
        return 0

    all_findings: list[dict[str, Any]] = []
    for rel in artifacts:
        all_findings.extend(audit_file(args.repo.resolve(), rel))

    report = {
        "ok": not all_findings,
        "artifacts": artifacts,
        "violations": all_findings,
    }
    print(json.dumps(report, indent=2))

    if all_findings:
        print(
            f"source_audit: {len(all_findings)} violation(s) across {len(artifacts)} artifact(s)",
            file=sys.stderr,
        )
        for v in all_findings:
            print(f"  {v['file']}:{v['line']}  [{v['id']}]  {v['snippet']}", file=sys.stderr)
            print(f"      hint: {v['hint']}", file=sys.stderr)
        return 1

    print(
        f"source_audit: 0 forbidden references across {len(artifacts)} artifact(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
