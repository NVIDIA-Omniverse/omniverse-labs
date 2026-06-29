#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document-model field-vocabulary audit.

Enforces strict AOUSD §7.6 conformance for the document-model field set: a field
token the spec marks **Deprecated** (§7.6.1.8 / §7.6.2.7 / §7.6.3.4 / §7.6.4.4 /
§7.6.5.2) or **Out of Scope** (§7.6.1.7 / §7.6.4.3), or a token with no spec
basis at all, must NOT appear in a *live* field position of any field-defining
contract. Live positions are:

  - top-level ``fields`` keys                       (document-model-productions/*)
  - ``spec_kinds.*.fields`` keys                    (variant-spec)
  - ``field_categories.*`` array elements           (common-metadata master vocab)
  - ``storage_mapping.field_resolution`` keys       (usdc *-spec-mapping)

Legitimate mentions inside ``excluded_fields`` / ``rejected_fields`` /
``forbidden_field_enforcement`` / laws / prose are NOT scanned -- the check looks
only at the structural positions above, by exact token match, so it never trips
on a diagnostic example or a rejection list.

This is the mechanical guard for the decision recorded in
docs/spec-driven-contracts.md and the ``excluded_fields`` blocks: the codebase
references the pinned spec's live field set, not OpenUSD's SdfFieldKeys /
SdfChildrenKeys.

Exit non-zero on any violation. Prints a JSON report on stdout and a
human-readable summary on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Tokens that must never appear in a live field position. Each maps to the spec
# clause that retires it (or notes it has no spec basis).
REJECTED_FIELD_TOKENS: dict[str, str] = {
    # Layer deprecated (§7.6.1.8)
    "framePrecision": "AOUSD §7.6.1.8 deprecated layer field",
    "hasOwnedSublayers": "AOUSD §7.6.1.8 deprecated layer field",
    "owner": "AOUSD §7.6.1.8 deprecated layer field",
    "sessionOwner": "AOUSD §7.6.1.8 deprecated layer field",
    "startFrame": "AOUSD §7.6.1.8 deprecated layer field",
    "endFrame": "AOUSD §7.6.1.8 deprecated layer field",
    # Layer out of scope / future (§7.6.1.7)
    "expressionVariables": "AOUSD §7.6.1.7 out-of-scope (future) layer field",
    "colorManagementSystem": "AOUSD §7.6.1.7 out-of-scope (future) layer field",
    "colorConfiguration": "AOUSD §7.6.1.7 out-of-scope (future) layer field",
    # Attribute out of scope / future (§7.6.4.3)
    "colorSpace": "AOUSD §7.6.4.3 out-of-scope (future) attribute field",
    # Prim / property / relationship deprecated (§7.6.2.7 / §7.6.3.4 / §7.6.4.4 / §7.6.5.2)
    "permission": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "prefix": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "prefixSubstitutions": "AOUSD §7.6.2.7 deprecated prim field",
    "suffix": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "suffixSubstitutions": "AOUSD §7.6.2.7 deprecated prim field",
    "symmetricPeer": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "symmetryArguments": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "symmetryFunction": "AOUSD deprecated field (§7.6.2.7 / §7.6.3.4)",
    "connectionChildren": "AOUSD §7.6.4.4 deprecated attribute field",
    "relationshipTargetChildren": "AOUSD §7.6.5.2 deprecated relationship field",
    "noLoadHint": "AOUSD §7.6.5.2 deprecated relationship field; never an AOUSD prim field (OpenUSD-only)",
    # No spec basis and not an OpenUSD field key
    "arraySizeConstraint": "no AOUSD or OpenUSD basis (fabricated)",
    "limits": "no AOUSD or OpenUSD basis (fabricated)",
    # OpenUSD SdfChildrenKeys spelling for the property-children list; AOUSD
    # §7.6.2.2.2 canonical is 'propertyChildren'. The full rename is complete, so
    # a live 'properties' field token is now a violation (the on-disk Crate FIELDS
    # token is still 'properties', but that lives in prose, not a field position).
    "properties": "OpenUSD SdfChildrenKeys token; AOUSD §7.6.2.2.2 canonical is 'propertyChildren'",
}

WARN_FIELD_TOKENS: dict[str, str] = {}

SCAN_DIRS = (
    "contracts/document-model-productions",
    "contracts/usdc-productions",
)


def live_field_tokens(contract: dict[str, Any]) -> list[str]:
    """Extract field tokens from the live (admitted) positions only."""
    tokens: list[str] = []
    fields = contract.get("fields")
    if isinstance(fields, dict):
        tokens.extend(fields.keys())
    spec_kinds = contract.get("spec_kinds")
    if isinstance(spec_kinds, dict):
        for kind in spec_kinds.values():
            kfields = kind.get("fields") if isinstance(kind, dict) else None
            if isinstance(kfields, dict):
                tokens.extend(kfields.keys())
    cats = contract.get("field_categories")
    if isinstance(cats, dict):
        for arr in cats.values():
            if isinstance(arr, list):
                tokens.extend(t for t in arr if isinstance(t, str))
    sm = contract.get("storage_mapping")
    if isinstance(sm, dict):
        fr = sm.get("field_resolution")
        if isinstance(fr, dict):
            tokens.extend(fr.keys())
    return tokens


def audit_contract(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ([{"file": str(path), "token": None, "reason": f"unreadable: {exc}"}], [])
    if not isinstance(contract, dict):
        return ([], [])
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for token in live_field_tokens(contract):
        if token in REJECTED_FIELD_TOKENS:
            violations.append({"file": str(path), "token": token, "reason": REJECTED_FIELD_TOKENS[token]})
        elif token in WARN_FIELD_TOKENS:
            warnings.append({"file": str(path), "token": token, "reason": WARN_FIELD_TOKENS[token]})
    return (violations, warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args()

    all_violations: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    scanned = 0
    for d in SCAN_DIRS:
        root = args.repo / d
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.json")):
            scanned += 1
            v, w = audit_contract(path)
            all_violations.extend(v)
            all_warnings.extend(w)

    report = {"ok": not all_violations, "scanned": scanned, "violations": all_violations, "warnings": all_warnings}
    print(json.dumps(report, indent=2))

    for w in all_warnings:
        print(f"spec_field_audit: WARN  {w['file']}: live token '{w['token']}' -- {w['reason']}", file=sys.stderr)
    for v in all_violations:
        print(f"spec_field_audit: VIOLATION  {v['file']}: live token '{v['token']}' -- {v['reason']}", file=sys.stderr)

    if all_violations or (args.warnings_as_errors and all_warnings):
        print(
            f"spec_field_audit: FAIL ({len(all_violations)} violation(s), {len(all_warnings)} warning(s)) across {scanned} contract(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"spec_field_audit: OK (0 violations, {len(all_warnings)} warning(s)) across {scanned} contract(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
