# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Differential INSTANCING conformance: nanousd ``pxr_compat`` vs real OpenUSD.

Companion to ``test_stage_conformance.py`` focused on the instancing surface —
the area that produced real renderer bugs (a PointInstancer prototype whose
nested sub-meshes were dropped; ``IsA(PointInstancer)`` disagreeing with
``GetTypeName``; geometry going hollow under nested instances). Each fixture in
``tests/assets/instancing/`` is walked through both backends by
``conformance/dump_instancing.py`` (native/scene-graph instancing, point
instancers, nested instancing, instance proxies, per-prototype mesh
enumeration, composed world transforms) and the result is diffed.

The fixtures are deliberately TINY, hand-readable repros — each is a simple
reproduction of an instancing bug class so a renderer/consumer regression is
obvious. Report-only unless ``NANOUSD_CONFORMANCE_USDCORE_PYTHON`` points at a
Python with usd-core; then the full comparison runs and gates against the frozen
``conformance_baseline.json`` (regenerate with
``NANOUSD_CONFORMANCE_UPDATE_BASELINE=1``).
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
_ASSETS = _HERE / "assets" / "instancing"
_DUMP = _CONFORMANCE / "dump_instancing.py"
_REPORT = _HERE / "instancing_conformance_report.md"
_ENV_VAR = "NANOUSD_CONFORMANCE_USDCORE_PYTHON"

# Companion layers (referenced/payloaded by an "entry" fixture) are not opened
# directly as test cases — they carry no instancing of their own.
_COMPANION_SUFFIXES = ("_ext_proto.usda", "_payload_asset.usda", "_payload_geom.usda")
_FIXTURES = sorted(
    p for p in _ASSETS.glob("*.usda") if not p.name.endswith(_COMPANION_SUFFIXES)
)

sys.path.insert(0, str(_CONFORMANCE))
import baseline  # noqa: E402
import compare  # noqa: E402


def _run_dump(python: str, backend: str, asset: Path) -> dict:
    proc = subprocess.run(
        [python, str(_DUMP), "--backend", backend, "--asset", str(asset)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} instancing dump of {asset.name} failed:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


def _usdcore_python() -> str | None:
    python = os.environ.get(_ENV_VAR)
    if not python or not Path(python).exists():
        return None
    probe = subprocess.run([python, "-c", "import pxr.Usd"], capture_output=True, text=True)
    return python if probe.returncode == 0 else None


def test_have_fixtures():
    assert _FIXTURES, f"no instancing fixtures found under {_ASSETS}"


@pytest.mark.parametrize("asset", _FIXTURES, ids=lambda p: p.stem)
def test_shim_instancing_walk_without_crashing(asset):
    """The shim must walk every instancing surface of every fixture without
    crashing — an exception in instancing introspection is itself a failure."""
    dump = _run_dump(sys.executable, "shim", asset)
    assert isinstance(dump, dict) and dump, f"{asset.name}: empty instancing walk"
    # the synthetic stage-level keys must always be present
    assert "__prototypes__" in dump and "__instance_proxies__" in dump


# Fixtures where nanousd currently drops (or invents) instanced geometry vs
# OpenUSD — REAL instancing bugs of the "pawns" class. xfail(strict) keeps CI
# green while guarding them: each flips to a loud XPASS the moment nanousd is
# fixed (forcing the marker's removal), and a NEW geometry divergence in ANY
# other (unmarked) fixture hard-fails immediately. Flip these to plain failures
# if you prefer red CI surfacing the open bugs.
_GEOM_KNOWN_BUGS = {
    "native_class": "nanousd traverses the abstract `class` prim and yields a phantom "
    "instanced Mesh (OpenUSD excludes class prims from the default predicate).",
    "nested_instance_in_instance": "nanousd drops geometry of an instance nested inside "
    "another instance (7 instanced meshes via proxies vs OpenUSD's 11).",
    "nested_pi_in_proto": "nanousd's inner PointInstancer prototype goes HOLLOW under "
    "native-instance copies (the pawns class): 4 instanced meshes vs OpenUSD's 8.",
}


def _mesh_proxy_count(dump: dict) -> int:
    """Number of Mesh-typed prims reachable via instance-proxy traversal = the
    total renderable instanced geometry. A backend-neutral integer; if it is
    lower than OpenUSD, geometry was dropped under instancing (pawns)."""
    ip = dump.get("__instance_proxies__", {})
    det = ip.get("detail", {}) if isinstance(ip, dict) else {}
    return sum(1 for v in det.values() if isinstance(v, dict) and v.get("type") == "Mesh")


def _geom_params():
    out = []
    for p in _FIXTURES:
        marks = []
        if p.stem in _GEOM_KNOWN_BUGS:
            marks = [pytest.mark.xfail(reason=_GEOM_KNOWN_BUGS[p.stem], strict=True)]
        out.append(pytest.param(p, marks=marks, id=p.stem))
    return out


@pytest.mark.parametrize("asset", _geom_params())
def test_no_geometry_dropped_under_instancing(asset):
    """HARD geometry-correctness gate (the "catch pawns" invariant): the count of
    instanced Mesh geometry reachable through instance proxies must EQUAL OpenUSD.
    Unlike the broad baseline-gated report (which tolerates documented API gaps),
    a divergence here is geometry silently dropped/invented under instancing and
    fails outright. Oracle-gated."""
    python = _usdcore_python()
    if python is None:
        pytest.skip(f"set {_ENV_VAR} to a Python with usd-core to run the comparison")
    shim = _run_dump(sys.executable, "shim", asset)
    openusd = _run_dump(python, "openusd", asset)
    sm, om = _mesh_proxy_count(shim), _mesh_proxy_count(openusd)
    assert sm == om, (
        f"{asset.stem}: instanced Mesh geometry diverges from OpenUSD "
        f"(nanousd={sm}, OpenUSD={om}) — geometry dropped or invented under "
        f"instancing (the pawns bug class)."
    )


def test_instancing_conformance_report():
    python = _usdcore_python()
    if python is None:
        pytest.skip(f"set {_ENV_VAR} to a Python with usd-core to run the comparison")

    divs = []
    for asset in _FIXTURES:
        shim = _run_dump(sys.executable, "shim", asset)
        openusd = _run_dump(python, "openusd", asset)
        divs.append(compare.compare_generic(shim, openusd, asset.stem))

    total_vm = sum(len(d.value_mismatch) for d in divs)
    total_so = sum(len(d.shim_only) for d in divs)
    total_oo = sum(len(d.openusd_only) for d in divs)

    lines = ["# Instancing conformance: nanousd vs OpenUSD", ""]
    lines.append("Each fixture is a tiny instancing repro walked through both backends "
                 "(`dump_instancing.py`): native/scene-graph instancing, point instancers, "
                 "nested instancing, instance proxies, per-prototype mesh enumeration, and "
                 "composed world transforms. A divergence here is a subtle instancing bug.")
    lines.append("")
    lines.append("| fixture | value-mismatch | shim-missing (vs OpenUSD) | shim-extra |")
    lines.append("|---|--:|--:|--:|")
    for d in divs:
        lines.append(
            f"| `{d.asset}` | **{len(d.value_mismatch)}** | {len(d.openusd_only)} | {len(d.shim_only)} |"
        )
    lines.append("")
    lines.append(
        f"**Totals across {len(divs)} fixtures:** {total_vm} value-mismatches, "
        f"{total_oo} surfaces present in OpenUSD but missing in the shim, "
        f"{total_so} shim-only."
    )
    lines.append("")
    for d in divs:
        if not (d.value_mismatch or d.shim_only or d.openusd_only):
            continue
        lines.append(f"## `{d.asset}`")
        for loc, s, o in d.value_mismatch[:80]:
            lines.append(f"- **mismatch** `{loc}`: shim=`{s}` openusd=`{o}`")
        for loc, o in d.openusd_only[:40]:
            lines.append(f"- **shim-missing** `{loc}`: openusd=`{o}`")
        for loc, s in d.shim_only[:40]:
            lines.append(f"- **shim-extra** `{loc}`: shim=`{s}`")
        lines.append("")
    _REPORT.write_text("\n".join(lines))

    print(f"\nInstancing conformance report -> {_REPORT}")
    print(f"value-mismatches={total_vm} shim-missing={total_oo} shim-only={total_so}")

    ok, regressions = baseline.gate("instancing", baseline.signature_from_divs(divs))
    assert ok, (
        "NEW instancing conformance divergences vs baseline "
        "(a subtle instancing regression; set NANOUSD_CONFORMANCE_UPDATE_BASELINE=1 "
        "if intentional):\n" + baseline.format_regressions(regressions)
    )


if __name__ == "__main__":
    python = _usdcore_python()
    if python is None:
        raise SystemExit(f"Set {_ENV_VAR} to a Python with usd-core installed.")
    test_instancing_conformance_report()
