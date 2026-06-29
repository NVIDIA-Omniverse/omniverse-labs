#!/usr/bin/env python3
"""Contract cross-reference checker.

Walks every JSON contract under contracts/{document-model-productions,
usda-productions, usdc-productions, capabilities, handles} and verifies that
every `derives_from`, `dependency_contracts`, `capability_contracts`, and
`performance_contracts` reference points at a file that exists.

For `derives_from` references that include a JSON Pointer-style anchor
(path#fragment), the file part is checked for existence and the anchor's
top-level key is checked for presence in the target JSON when the anchor
looks like dotted field navigation. Anchors that look descriptive (free
text with spaces or non-JSON tokens) are not walked -- they are treated as
human-readable markers.

Exits non-zero on any broken reference. Designed to be invoked from
harness/regen_graph.py validate_scope alongside contract_lint and source_audit.

Invocation:
    python3 harness/contract_xref.py                # scan repo root
    python3 harness/contract_xref.py --strict       # treat anchor warnings as errors
    python3 harness/contract_xref.py --repo <path>  # scan a different repo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONTRACT_DIRS = (
    "contracts/capabilities",
    "contracts/document-model-productions",
    "contracts/handles",
    "contracts/usda-productions",
    "contracts/usdc-productions",
    "contracts/spec-coverage",
    "contracts/performance",
)

# Fields that hold path references the checker walks.
REF_FIELDS = (
    "derives_from",
    "dependency_contracts",
    "capability_contracts",
    "performance_contracts",
)


def discover_contracts(repo: Path) -> list[Path]:
    out: list[Path] = []
    for d in CONTRACT_DIRS:
        root = repo / d
        if root.exists():
            out.extend(sorted(p for p in root.glob("*.json") if p.is_file()))
    return out


def parse_ref(ref: str) -> tuple[str, str]:
    file_part, _, fragment = ref.partition("#")
    return file_part, fragment


def looks_like_dotted_anchor(fragment: str) -> bool:
    """True if the fragment looks like a JSON Pointer-style dotted path
    (e.g. spec_forms.prim, fields.timeSamples, laws). False for descriptive
    free-text markers that contain spaces, punctuation, or operator-style
    suffixes."""
    if not fragment:
        return False
    if any(c in fragment for c in (" ", ":", ";", "/", "(", ")", "[", "]")):
        # Brackets like laws[3] would be valid; but our existing contracts use
        # only dotted form, so the presence of any of these implies a marker.
        # Adjust if/when the contracts adopt JSON Pointer brackets.
        return False
    return all(part.replace("_", "").replace("-", "").isalnum() for part in fragment.split("."))


def walk_anchor(doc: Any, fragment: str) -> tuple[bool, str]:
    """Walk a dotted anchor into the target document. Returns (found, where).
    'where' is the deepest token that resolved (or '' if the first failed)."""
    cur = doc
    walked = []
    for token in fragment.split("."):
        walked.append(token)
        try:
            cur = cur[token]
        except (KeyError, IndexError, TypeError):
            return False, ".".join(walked[:-1])
    return True, ".".join(walked)


def check_one_ref(
    repo: Path,
    source: Path,
    field: str,
    ref: str,
    strict_anchors: bool,
) -> list[str]:
    issues: list[str] = []
    file_part, fragment = parse_ref(ref)
    target_path = repo / file_part
    src_label = str(source.relative_to(repo))

    if not target_path.exists():
        issues.append(f"{src_label}: {field} -> {file_part!r} does not exist")
        return issues

    # Load target for further checks. Tolerate non-JSON (some refs point at *.md).
    target_doc: dict | None = None
    if target_path.suffix == ".json":
        try:
            target_doc = json.load(target_path.open())
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{src_label}: {field} -> {file_part!r} not valid JSON: {exc}")
            return issues

    if fragment and target_doc is not None and looks_like_dotted_anchor(fragment):
        found, where = walk_anchor(target_doc, fragment)
        if not found:
            msg = (
                f"{src_label}: {field} -> {file_part}#{fragment} anchor not "
                f"resolvable (stopped at {where or '<root>'})"
            )
            if strict_anchors:
                issues.append(msg)
            else:
                issues.append(f"WARN  {msg}")

    return issues


def check_contract(
    repo: Path, source: Path, doc: dict, strict_anchors: bool
) -> list[str]:
    issues: list[str] = []
    for field in REF_FIELDS:
        val = doc.get(field)
        if val is None:
            continue
        refs = [val] if isinstance(val, str) else val
        if not isinstance(refs, list):
            issues.append(
                f"{source.relative_to(repo)}: {field} is neither string nor list"
            )
            continue
        for ref in refs:
            if not isinstance(ref, str):
                issues.append(
                    f"{source.relative_to(repo)}: {field} contains non-string entry {ref!r}"
                )
                continue
            issues.extend(check_one_ref(repo, source, field, ref, strict_anchors))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to harness/..)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat unresolvable anchors as errors (default: warn)",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    contracts = discover_contracts(repo)
    if not contracts:
        print("contract_xref: no contracts found", file=sys.stderr)
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    for path in contracts:
        try:
            doc = json.load(path.open())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(repo)}: not valid JSON: {exc}")
            continue
        if not isinstance(doc, dict):
            continue
        for msg in check_contract(repo, path, doc, args.strict):
            if msg.startswith("WARN  "):
                warnings.append(msg[len("WARN  "):])
            else:
                errors.append(msg)

    for w in warnings:
        print(f"contract_xref: warn: {w}", file=sys.stderr)
    for e in errors:
        print(f"contract_xref: ERROR {e}", file=sys.stderr)

    if errors:
        print(
            f"contract_xref: {len(errors)} broken reference(s) across "
            f"{len(contracts)} contract(s)",
            file=sys.stderr,
        )
        return 1

    print(
        f"contract_xref: OK ({len(contracts)} contract(s) checked, "
        f"{len(warnings)} anchor warning(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
