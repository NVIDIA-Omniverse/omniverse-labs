# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Conformance regression gate: compare current divergences to a frozen baseline.

The conformance suites flag many known divergences from OpenUSD. To turn them
from a report into a regression gate without requiring every divergence be fixed
first, we freeze the *current* set of divergence signatures in
``tests/conformance_baseline.json`` and fail only on **new** ones:

* list signatures (divergence locations) — current must be a SUBSET of baseline
  (so fixes that shrink the set pass; new divergences fail).
* numeric signatures (failure/error counts) — current must be ``<=`` baseline.

Regenerate the baseline after an intentional change with::

    NANOUSD_CONFORMANCE_UPDATE_BASELINE=1 python -m pytest tests/test_*conformance*.py

Completion criterion for divergence-fixing work: the baseline shrinks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_BASELINE = Path(__file__).resolve().parent.parent / "conformance_baseline.json"


def update_enabled() -> bool:
    return os.environ.get("NANOUSD_CONFORMANCE_UPDATE_BASELINE") == "1"


def _load() -> dict:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


def _save(data: dict) -> None:
    _BASELINE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def gate(suite: str, current: dict) -> tuple[bool, dict]:
    """Compare ``current`` signatures for ``suite`` to the baseline.

    ``current`` maps key -> {category: [locations] | count}. Returns
    ``(ok, regressions)``. In update mode the baseline is rewritten and ``ok`` is
    True. ``regressions`` maps key -> {category: new-items-or-count-message}.
    """
    data = _load()
    if update_enabled():
        data[suite] = current
        _save(data)
        return True, {}
    base = data.get(suite, {})
    regressions: dict = {}
    for key, cats in current.items():
        bkey = base.get(key, {})
        for cat, val in cats.items():
            bval = bkey.get(cat)
            if isinstance(val, list):
                extra = sorted(set(val) - set(bval or []))
                if extra:
                    regressions.setdefault(key, {})[cat] = extra
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                if bval is None or val > bval:
                    regressions.setdefault(key, {})[cat] = f"{val} > baseline {bval}"
    return (not regressions), regressions


def baseline_exists() -> bool:
    return _BASELINE.exists()


def signature_from_divs(divs) -> dict:
    """Build a {asset: {category: [sorted locations]}} signature from Divergences."""
    sig: dict = {}
    for d in divs:
        sig[d.asset] = {
            "value_mismatch": sorted(loc for loc, _s, _o in d.value_mismatch),
            "shim_only": sorted(loc for loc, _v in d.shim_only),
            "openusd_only": sorted(loc for loc, _v in d.openusd_only),
        }
    return sig


def format_regressions(regressions: dict) -> str:
    lines = []
    for key, cats in sorted(regressions.items()):
        for cat, items in cats.items():
            if isinstance(items, list):
                lines.append(f"  {key} [{cat}]: {len(items)} NEW")
                for it in items[:20]:
                    lines.append(f"      + {it}")
            else:
                lines.append(f"  {key} [{cat}]: {items}")
    return "\n".join(lines)
