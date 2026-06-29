# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Three-category differential comparison of two UsdPhysics dumps.

Given the JSON produced by ``dump_usdphysics.py`` for the shim and for real
OpenUSD, classify every divergence into:

1. ``shim_only``      — field present in the nanousd shim, absent in OpenUSD.
2. ``openusd_only``   — field present in OpenUSD, absent in the shim (a coverage
                        gap in the shim's parser).
3. ``value_mismatch`` — field present in *both* but with a different value.
                        This is the bug list: where nanousd parses physics
                        differently from OpenUSD.

Plus structural divergences: per-ObjectType object-count and prim-path-set
mismatches, which are reported first because they invalidate field comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_MISSING = "∅"  # ∅ sentinel for "field absent in this backend"
_RTOL = 1e-3
_ATOL = 1e-4


@dataclass
class Divergence:
    asset: str
    object_count: list = field(default_factory=list)   # (objtype, shim_n, openusd_n)
    path_set: list = field(default_factory=list)        # (objtype, only_shim, only_openusd)
    shim_only: list = field(default_factory=list)       # (loc, value)
    openusd_only: list = field(default_factory=list)    # (loc, value)
    value_mismatch: list = field(default_factory=list)  # (loc, shim_val, openusd_val)

    @property
    def total(self) -> int:
        return (
            len(self.object_count)
            + len(self.path_set)
            + len(self.shim_only)
            + len(self.openusd_only)
            + len(self.value_mismatch)
        )


def _near(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= _ATOL + _RTOL * abs(float(b))
    except (TypeError, ValueError):
        return False


def _walk(shim, openusd, loc: str, d: Divergence) -> None:
    """Recurse two normalized subtrees, appending categorized leaf divergences."""
    # An "<unsupported:...>" sentinel means the shim does not implement that API —
    # a coverage gap, not a value bug, so route it to the coverage buckets.
    if isinstance(shim, str) and shim.startswith("<unsupported"):
        d.openusd_only.append((loc, openusd))
        return
    if isinstance(openusd, str) and openusd.startswith("<unsupported"):
        d.shim_only.append((loc, shim))
        return
    if isinstance(shim, bool) or isinstance(openusd, bool):
        if shim != openusd:
            d.value_mismatch.append((loc, shim, openusd))
        return
    if isinstance(shim, (int, float)) and isinstance(openusd, (int, float)):
        if not _near(shim, openusd):
            d.value_mismatch.append((loc, shim, openusd))
        return
    if isinstance(shim, dict) and isinstance(openusd, dict):
        for k in sorted(set(shim) | set(openusd)):
            sub = f"{loc}.{k}"
            if k not in shim:
                d.openusd_only.append((sub, openusd[k]))
            elif k not in openusd:
                d.shim_only.append((sub, shim[k]))
            else:
                _walk(shim[k], openusd[k], sub, d)
        return
    if isinstance(shim, list) and isinstance(openusd, list):
        if len(shim) != len(openusd):
            d.value_mismatch.append((f"{loc}.len", len(shim), len(openusd)))
            return
        for i, (x, y) in enumerate(zip(shim, openusd)):
            _walk(x, y, f"{loc}[{i}]", d)
        return
    if type(shim).__name__ != type(openusd).__name__:
        d.value_mismatch.append((loc, f"<{type(shim).__name__}>{shim}", f"<{type(openusd).__name__}>{openusd}"))
        return
    if shim != openusd:
        d.value_mismatch.append((loc, shim, openusd))


def compare(shim: dict, openusd: dict, asset: str) -> Divergence:
    d = Divergence(asset=asset)
    for objtype in sorted(set(shim) | set(openusd)):
        s_items = shim.get(objtype, [])
        o_items = openusd.get(objtype, [])
        if len(s_items) != len(o_items):
            d.object_count.append((objtype, len(s_items), len(o_items)))
        s_by_path = {it["path"]: it["desc"] for it in s_items}
        o_by_path = {it["path"]: it["desc"] for it in o_items}
        only_shim = sorted(set(s_by_path) - set(o_by_path))
        only_openusd = sorted(set(o_by_path) - set(s_by_path))
        if only_shim or only_openusd:
            d.path_set.append((objtype, only_shim, only_openusd))
        for path in sorted(set(s_by_path) & set(o_by_path)):
            _walk(s_by_path[path], o_by_path[path], f"{objtype}@{path}", d)
    return d


def compare_generic(shim: dict, openusd: dict, asset: str) -> Divergence:
    """Three-category diff of two prim-path-keyed trees (full-stage dumps)."""
    d = Divergence(asset=asset)
    only_shim = sorted(set(shim) - set(openusd))
    only_openusd = sorted(set(openusd) - set(shim))
    if only_shim or only_openusd:
        d.path_set.append(("Prim", only_shim, only_openusd))
    for path in sorted(set(shim) & set(openusd)):
        _walk(shim[path], openusd[path], f"@{path}", d)
    return d


def field_leaf(loc: str) -> str:
    """The descriptor field name in a location, for aggregation.

    Locations look like ``ObjectType@/prim/path.field.subfield[i]``. The
    ``ObjectType@/prim/path`` prefix uses ``/`` (never ``.``), so the first ``.``
    separates the prim path from the descriptor field; we return that field.
    """
    after = loc.split(".", 1)
    if len(after) < 2:
        return loc
    return after[1].split("[", 1)[0].split(".", 1)[0]


def render_markdown(divs: list[Divergence], backends: dict) -> str:
    from collections import Counter

    lines = ["# UsdPhysics parsing conformance: nanousd vs OpenUSD", ""]
    lines.append(f"- nanousd shim: `{backends.get('shim', '?')}`")
    lines.append(f"- OpenUSD (usd-core): `{backends.get('openusd', '?')}`")
    lines.append("")
    lines.append("Each robot is parsed via `UsdPhysics.LoadUsdPhysicsFromRange` through both")
    lines.append("backends; divergences are grouped into structural, shim-only, OpenUSD-only,")
    lines.append("and **value-mismatch** (the parser-bug list).")
    lines.append("")
    lines.append("| robot | obj-count | path-set | shim-only | openusd-only | **value-mismatch** |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for d in divs:
        lines.append(
            f"| `{d.asset}` | {len(d.object_count)} | {len(d.path_set)} | "
            f"{len(d.shim_only)} | {len(d.openusd_only)} | **{len(d.value_mismatch)}** |"
        )
    lines.append("")

    # Aggregate value-mismatch + openusd-only by field name across all robots.
    vm_fields = Counter()
    oo_fields = Counter()
    for d in divs:
        for loc, _s, _o in d.value_mismatch:
            vm_fields[field_leaf(loc)] += 1
        for loc, _v in d.openusd_only:
            oo_fields[field_leaf(loc)] += 1
    if vm_fields:
        lines.append("## Value mismatches by field (parser bugs to investigate)")
        lines.append("")
        for f, c in vm_fields.most_common():
            lines.append(f"- `{f}` — {c}")
        lines.append("")
    if oo_fields:
        lines.append("## OpenUSD-only fields by name (shim coverage gaps)")
        lines.append("")
        for f, c in oo_fields.most_common():
            lines.append(f"- `{f}` — {c}")
        lines.append("")

    for d in divs:
        if d.total == 0:
            continue
        lines.append(f"## `{d.asset}` — details")
        lines.append("")
        for objtype, sn, on in d.object_count:
            lines.append(f"- **object-count** `{objtype}`: shim={sn} openusd={on}")
        for objtype, os_, oo in d.path_set:
            if os_:
                lines.append(f"- **path-set** `{objtype}` only-in-shim: {os_}")
            if oo:
                lines.append(f"- **path-set** `{objtype}` only-in-openusd: {oo}")
        if d.value_mismatch:
            lines.append("")
            lines.append("<details><summary>value mismatches</summary>")
            lines.append("")
            for loc, s, o in d.value_mismatch[:200]:
                lines.append(f"- `{loc}`: shim=`{s}` openusd=`{o}`")
            if len(d.value_mismatch) > 200:
                lines.append(f"- … {len(d.value_mismatch) - 200} more")
            lines.append("")
            lines.append("</details>")
        lines.append("")
    return "\n".join(lines)
