# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Full-stage differential conformance: nanousd ``pxr_compat`` vs real OpenUSD.

Where ``test_usdphysics_conformance.py`` validates one bulk-parse API, this walks
the *entire* stage of every vendored robot through both backends and diffs the
broad read/query surface — traversal, ``GetTypeName``, ``GetAppliedSchemas``,
every authored ``attr.Get()`` (across all USD value types present), relationships
and metadata — answering "validate every (read) API call, against every robot."

Coverage stats (prims / authored attributes / distinct value types / relationships
compared per robot) are written alongside the divergences to
``tests/stage_conformance_report.md``. Report-only; set
``NANOUSD_CONFORMANCE_USDCORE_PYTHON`` to enable the OpenUSD comparison.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_CONFORMANCE = _HERE / "conformance"
_ROBOTS = sorted((_HERE / "assets" / "robots").glob("*.usda"))
_DUMP = _CONFORMANCE / "dump_stage.py"
_REPORT = _HERE / "stage_conformance_report.md"
_ENV_VAR = "NANOUSD_CONFORMANCE_USDCORE_PYTHON"

sys.path.insert(0, str(_CONFORMANCE))
import compare  # noqa: E402
import baseline  # noqa: E402


def _run_dump(python: str, backend: str, asset: Path) -> dict:
    proc = subprocess.run(
        [python, str(_DUMP), "--backend", backend, "--asset", str(asset)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} stage dump of {asset.name} failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _usdcore_python() -> str | None:
    python = os.environ.get(_ENV_VAR)
    if not python or not Path(python).exists():
        return None
    probe = subprocess.run([python, "-c", "import pxr.Usd"], capture_output=True, text=True)
    return python if probe.returncode == 0 else None


def _coverage(dump: dict) -> dict:
    prims = len(dump)
    attrs = types = rels = 0
    typeset = set()
    for entry in dump.values():
        a = entry.get("attrs", {})
        attrs += len(a)
        for meta in a.values():
            t = meta.get("type")
            if t:
                typeset.add(t)
        rels += len(entry.get("rels", {}))
    types = len(typeset)
    return {"prims": prims, "attrs": attrs, "value_types": types, "rels": rels}


@pytest.mark.parametrize("asset", _ROBOTS, ids=lambda p: p.stem)
def test_shim_stage_walk_without_crashing(asset):
    dump = _run_dump(sys.executable, "shim", asset)
    assert isinstance(dump, dict) and dump, f"{asset.name}: empty stage walk"


def test_stage_conformance_report():
    python = _usdcore_python()
    if python is None:
        pytest.skip(f"set {_ENV_VAR} to a Python with usd-core to run the comparison")
    divs = []
    cov = {}
    for asset in _ROBOTS:
        shim = _run_dump(sys.executable, "shim", asset)
        openusd = _run_dump(python, "openusd", asset)
        divs.append(compare.compare_generic(shim, openusd, asset.stem))
        cov[asset.stem] = _coverage(shim)

    lines = ["# Full-stage read-API conformance: nanousd vs OpenUSD", ""]
    lines.append("Every prim of every robot walked through both backends; the broad read")
    lines.append("surface (type, applied schemas, every authored `attr.Get()`, relationships,")
    lines.append("metadata) is diffed. Coverage shows how much API surface each robot exercises.")
    lines.append("")
    lines.append("| robot | prims | authored attrs | value types | rels | **value-mismatch** | shim-only | openusd-only |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    tot = {"prims": 0, "attrs": 0, "rels": 0, "vm": 0}
    for d in divs:
        c = cov[d.asset]
        tot["prims"] += c["prims"]; tot["attrs"] += c["attrs"]; tot["rels"] += c["rels"]
        tot["vm"] += len(d.value_mismatch)
        lines.append(
            f"| `{d.asset}` | {c['prims']} | {c['attrs']} | {c['value_types']} | {c['rels']} "
            f"| **{len(d.value_mismatch)}** | {len(d.shim_only)} | {len(d.openusd_only)} |"
        )
    lines.append("")
    lines.append(
        f"**Totals across {len(divs)} robots:** {tot['prims']} prims, {tot['attrs']} authored "
        f"attributes, {tot['rels']} relationships compared; {tot['vm']} value-mismatches."
    )
    lines.append("")
    for d in divs:
        if not d.value_mismatch and not d.path_set:
            continue
        lines.append(f"## `{d.asset}` — details")
        for objtype, os_, oo in d.path_set:
            if os_:
                lines.append(f"- **prims only in shim**: {os_[:20]}")
            if oo:
                lines.append(f"- **prims only in openusd**: {oo[:20]}")
        if d.value_mismatch:
            lines.append("")
            lines.append("<details><summary>value mismatches</summary>")
            lines.append("")
            for loc, s, o in d.value_mismatch[:150]:
                lines.append(f"- `{loc}`: shim=`{s}` openusd=`{o}`")
            if len(d.value_mismatch) > 150:
                lines.append(f"- … {len(d.value_mismatch) - 150} more")
            lines.append("")
            lines.append("</details>")
        lines.append("")
    _REPORT.write_text("\n".join(lines))
    assert _REPORT.exists()
    print(f"\nStage conformance report -> {_REPORT}")
    print(f"compared {tot['prims']} prims / {tot['attrs']} attrs; value-mismatches={tot['vm']}")
    ok, regressions = baseline.gate("stage", baseline.signature_from_divs(divs))
    assert ok, (
        "NEW full-stage conformance divergences vs baseline "
        "(set NANOUSD_CONFORMANCE_UPDATE_BASELINE=1 if intentional):\n"
        + baseline.format_regressions(regressions)
    )


if __name__ == "__main__":
    python = _usdcore_python()
    if python is None:
        raise SystemExit(f"Set {_ENV_VAR} to a Python with usd-core installed.")
    test_stage_conformance_report()
