#!/usr/bin/env python3
"""Static-pattern lint for contract laws that forbid specific code shapes.

Reads a rules JSON file (e.g. contracts/lint/usdc-single-layer.lint.json) and
checks the listed generated artifacts for forbidden symbols and regex patterns.
Exits non-zero with one diagnostic per violation found.

The lint is invoked from harness/regen_graph.py validate_scope when a scope
declares a `contract_lints` field. It translates prose contract laws like
"the binary-format unit must not expose a helper that searches the full opened
payload as text" into a mechanical gate the harness runs against the
generated code.

Supported `pattern_kind` values:
  - symbol_present:  fires once per occurrence of \\b<symbol>\\b in an artifact
                     (catches forbidden code shapes)
  - regex_match:     fires once per re.search(regex, content, re.DOTALL) match
                     (catches forbidden code shapes)
  - symbol_required: fires once per artifact in which \\b<symbol>\\b is absent
                     (catches missing required code patterns)
  - regex_required:  fires once per artifact in which re.search(regex, content,
                     re.DOTALL) returns no match (catches missing required
                     code patterns)

`symbol_required` / `regex_required` are for document-model invariants that the
generated parser must positively materialize -- e.g. the relationship-variability
guard or the prim-spec default field set. Missing artifacts (e.g. before regen)
are skipped; the cross-reference checker is responsible for artifact presence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def load_rules(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"{path}: 'rules' must be a list")
    return rules


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def node_artifacts(graph: dict[str, Any], node_id: str, target: str) -> set[str]:
    node = graph.get("nodes", {}).get(node_id)
    if not node:
        return set()
    target_info = node.get("targets", {}).get(target, {})
    return set(target_info.get("artifacts", []))


def filter_rules_by_nodes(
    rules: list[dict[str, Any]],
    graph: dict[str, Any],
    target: str,
    nodes: list[str],
) -> list[dict[str, Any]]:
    if not nodes:
        return rules
    in_scope: set[str] = set()
    for n in nodes:
        in_scope |= node_artifacts(graph, n, target)
    selected: list[dict[str, Any]] = []
    for rule in rules:
        artifacts = set(rule.get("applies_to_artifacts", []))
        if artifacts & in_scope:
            selected.append(rule)
    return selected


def check_symbol_present(
    repo: Path, rule: dict[str, Any]
) -> list[tuple[str, int, str]]:
    symbol = rule["symbol"]
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    return _scan(repo, rule, pattern)


def check_regex_match(
    repo: Path, rule: dict[str, Any]
) -> list[tuple[str, int, str]]:
    pattern = re.compile(rule["regex"], re.DOTALL)
    return _scan(repo, rule, pattern)


def check_symbol_required(
    repo: Path, rule: dict[str, Any]
) -> list[tuple[str, int, str]]:
    symbol = rule["symbol"]
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    return _scan_required(repo, rule, pattern, f"required symbol {symbol!r} not found")


def check_regex_required(
    repo: Path, rule: dict[str, Any]
) -> list[tuple[str, int, str]]:
    pattern = re.compile(rule["regex"], re.DOTALL)
    return _scan_required(
        repo, rule, pattern, f"required pattern not found: {rule['regex']!r}"
    )


def _scan(
    repo: Path, rule: dict[str, Any], pattern: re.Pattern[str]
) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for rel in rule.get("applies_to_artifacts", []):
        path = repo / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in pattern.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            snippet = _line_at(text, m.start())
            hits.append((rel, line_no, snippet))
    return hits


def _scan_required(
    repo: Path,
    rule: dict[str, Any],
    pattern: re.Pattern[str],
    not_found_message: str,
) -> list[tuple[str, int, str]]:
    """One hit per artifact in which the required pattern is absent.

    Missing artifacts are skipped (the artifact may not yet exist before regen);
    the contract-cross-reference checker is responsible for catching missing
    artifacts. This function only reports artifacts that exist but do not
    contain the pattern.
    """
    hits: list[tuple[str, int, str]] = []
    for rel in rule.get("applies_to_artifacts", []):
        path = repo / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not pattern.search(text):
            hits.append((rel, 1, not_found_message))
    return hits


def _line_at(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    line = text[start:end].strip()
    if len(line) > 200:
        line = line[:200] + "..."
    return line


def run_rule(
    repo: Path, rule: dict[str, Any]
) -> list[tuple[str, int, str]]:
    kind = rule.get("pattern_kind")
    if kind == "symbol_present":
        return check_symbol_present(repo, rule)
    if kind == "regex_match":
        return check_regex_match(repo, rule)
    if kind == "symbol_required":
        return check_symbol_required(repo, rule)
    if kind == "regex_required":
        return check_regex_required(repo, rule)
    raise ValueError(f"unknown pattern_kind: {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--target", default="cpp")
    parser.add_argument(
        "--graph", type=Path, help="graph manifest (required if --node is given)"
    )
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="restrict rules to those that apply to artifacts owned by these nodes; "
        "pass repeatedly to enumerate multiple nodes",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to harness/.. )",
    )
    args = parser.parse_args()

    rules = load_rules(args.rules.resolve())
    if args.node:
        if not args.graph:
            print(
                "contract_lint: --node requires --graph",
                file=sys.stderr,
            )
            return 2
        graph = load_graph(args.graph.resolve())
        rules = filter_rules_by_nodes(rules, graph, args.target, args.node)

    total_violations = 0
    for rule in rules:
        hits = run_rule(args.repo.resolve(), rule)
        for rel, line_no, snippet in hits:
            total_violations += 1
            print(f"contract_lint: VIOLATION rule={rule['id']}")
            print(f"  file: {rel}:{line_no}")
            print(f"  line: {snippet}")
            if rule.get("source_law"):
                print(f"  law:  {rule['source_law']}")
            if rule.get("message"):
                print(f"  hint: {rule['message']}")
            print()

    if total_violations:
        print(
            f"contract_lint: {total_violations} violation(s) found across "
            f"{len(rules)} rule(s)",
            file=sys.stderr,
        )
        return 1

    print(f"contract_lint: OK ({len(rules)} rule(s) checked, no violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
