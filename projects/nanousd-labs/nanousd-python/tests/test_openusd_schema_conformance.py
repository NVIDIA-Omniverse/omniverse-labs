# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Run OpenUSD's own schema python tests against the nanousd ``pxr_compat`` shim.

This complements ``test_usdphysics_conformance.py`` (a *live* differential parse
of robot assets vs usd-core) by reusing OpenUSD's **own** unit tests for the
``UsdPhysics``, ``UsdGeom`` and ``UsdSkel`` schemas, copied verbatim into
``tests/openusd_compat/upstream/`` (with their original license headers).

Each upstream module is executed in an isolated subprocess against the shim; the
pass/fail/error tally is written to ``tests/openusd_schema_conformance_report.md``.

This is **report-only** (a living conformance matrix), not a pass/fail gate — the
shim does not yet implement every schema API, and the report is precisely the
record of where it diverges from OpenUSD. The only hard assertions are that the
copied corpus is present and carries upstream license headers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).parent
_UPSTREAM = _HERE / "openusd_compat" / "upstream" / "pxr" / "usd"
_DRIVER = _HERE / "conformance" / "run_schema_test.py"
_REPORT = _HERE / "openusd_schema_conformance_report.md"
_SCHEMAS = ("usdPhysics", "usdGeom", "usdSkel")

sys.path.insert(0, str(_HERE / "conformance"))
import baseline  # noqa: E402  (regression-gate baseline)


def _schema_test_modules() -> list[Path]:
    mods: list[Path] = []
    for schema in _SCHEMAS:
        testenv = _UPSTREAM / schema / "testenv"
        if testenv.is_dir():
            mods += sorted(testenv.glob("test*.py"))
    return mods


_MODULES = _schema_test_modules()


def test_copied_schema_tests_present_with_license_headers():
    assert _MODULES, "no copied schema tests found under openusd_compat/upstream"
    for mod in _MODULES:
        text = mod.read_text(encoding="utf-8")
        assert "Copyright" in text, mod
        assert "openusd.org/license" in text, mod


def _run_module(mod: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_DRIVER), str(mod)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except (json.JSONDecodeError, IndexError):
        return {"load_error": f"driver rc={proc.returncode}: {proc.stderr[-300:]}"}


def _generate_report() -> list[tuple[Path, dict]]:
    rows = [(mod, _run_module(mod)) for mod in _MODULES]
    lines = ["# OpenUSD schema tests vs nanousd `pxr_compat`", ""]
    lines.append("Upstream OpenUSD unit tests run against the shim. Report-only — the")
    lines.append("tally is the conformance matrix, not a pass/fail gate.")
    lines.append("")
    lines.append("| schema | test module | ran | pass | fail | error | status |")
    lines.append("|---|---|--:|--:|--:|--:|---|")
    by_schema: dict[str, list[int]] = {}
    for mod, r in rows:
        schema = mod.parts[mod.parts.index("usd") + 1]
        agg = by_schema.setdefault(schema, [0, 0, 0, 0])
        if "load_error" in r:
            lines.append(f"| {schema} | `{mod.stem}` | – | – | – | – | ❌ load: {r['load_error'][:80]} |")
            continue
        ran, fail, err = r.get("ran", 0), r.get("failures", 0), r.get("errors", 0)
        passed = max(0, ran - fail - err)
        status = "✅ all pass" if (fail == 0 and err == 0 and ran > 0) else (
            "⚠️ partial" if passed > 0 else "❌ none pass")
        agg[0] += ran; agg[1] += passed; agg[2] += fail; agg[3] += err
        lines.append(f"| {schema} | `{mod.stem}` | {ran} | {passed} | {fail} | {err} | {status} |")
    lines.append("")
    lines.append("## Per-schema totals")
    lines.append("")
    lines.append("| schema | ran | pass | fail | error |")
    lines.append("|---|--:|--:|--:|--:|")
    for schema, (ran, passed, fail, err) in by_schema.items():
        lines.append(f"| {schema} | {ran} | {passed} | {fail} | {err} |")
    lines.append("")
    lines.append("## Representative failures (first few per module)")
    lines.append("")
    for mod, r in rows:
        samples = r.get("samples") or ([] if "load_error" not in r else [r["load_error"]])
        if samples:
            lines.append(f"### `{mod.stem}`")
            for s in samples:
                lines.append(f"- {s}")
            lines.append("")
    _REPORT.write_text("\n".join(lines))
    return rows


def test_openusd_schema_conformance_report():
    rows = _generate_report()
    assert _REPORT.exists()
    total_ran = sum(r.get("ran", 0) for _m, r in rows)
    total_pass = sum(r.get("ran", 0) - r.get("failures", 0) - r.get("errors", 0)
                     for _m, r in rows if "load_error" not in r)
    print(f"\nSchema conformance report -> {_REPORT}")
    print(f"modules={len(rows)} tests_ran={total_ran} passing={total_pass}")
    sig = {}
    for mod, r in rows:
        if "load_error" in r:
            sig[mod.stem] = {"load_error": 1}
        else:
            sig[mod.stem] = {"failures": r.get("failures", 0), "errors": r.get("errors", 0)}
    ok, regressions = baseline.gate("schema", sig)
    assert ok, (
        "NEW OpenUSD schema-test failures/errors vs baseline "
        "(set NANOUSD_CONFORMANCE_UPDATE_BASELINE=1 if intentional):\n"
        + baseline.format_regressions(regressions)
    )


if __name__ == "__main__":
    rows = _generate_report()
    for mod, r in rows:
        if "load_error" in r:
            print(f"  {mod.stem:42} LOAD-ERROR {r['load_error'][:70]}")
        else:
            ran, fail, err = r.get("ran", 0), r.get("failures", 0), r.get("errors", 0)
            print(f"  {mod.stem:42} ran={ran:3} pass={ran-fail-err:3} fail={fail:3} err={err:3}")
    print(f"\nReport -> {_REPORT}")
