# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Differential conformance test: nanousd ``pxr_compat`` vs real OpenUSD.

Parses every vendored Newton robot (``tests/assets/robots/*.usda``) through both
backends via ``UsdPhysics.LoadUsdPhysicsFromRange`` and flags divergences:

* **structural** — object-count / prim-path-set mismatches per ObjectType,
* **shim-only** / **openusd-only** fields (coverage gaps),
* **value-mismatch** — same field, different value (the parser-bug list).

Real OpenUSD runs in a separate interpreter (both backends register a top-level
``pxr``, so they cannot coexist in one process). Point the test at a Python that
has ``usd-core`` installed::

    NANOUSD_CONFORMANCE_USDCORE_PYTHON=/path/to/usdcore-venv/bin/python \
        python -m pytest tests/test_usdphysics_conformance.py

or run it directly to (re)generate ``tests/conformance_report.md``::

    NANOUSD_CONFORMANCE_USDCORE_PYTHON=... python tests/test_usdphysics_conformance.py

If the env var is unset (or the interpreter lacks ``usd-core``), the
OpenUSD-comparison tests skip; the shim-only "does it parse without crashing"
checks still run.
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
_DUMP = _CONFORMANCE / "dump_usdphysics.py"
_REPORT = _HERE / "conformance_report.md"
_ENV_VAR = "NANOUSD_CONFORMANCE_USDCORE_PYTHON"

sys.path.insert(0, str(_CONFORMANCE))
import compare  # noqa: E402  (local conformance helper)
import baseline  # noqa: E402  (regression-gate baseline)


def _run_dump(python: str, backend: str, asset: Path) -> dict:
    """Run the dump script in ``python`` and return the parsed JSON (or raise)."""
    proc = subprocess.run(
        [python, str(_DUMP), "--backend", backend, "--asset", str(asset)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} dump of {asset.name} failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


def _usdcore_python() -> str | None:
    """Resolve a Python interpreter that can ``import pxr`` (real usd-core)."""
    python = os.environ.get(_ENV_VAR)
    if not python or not Path(python).exists():
        return None
    probe = subprocess.run(
        [python, "-c", "import pxr.UsdPhysics"], capture_output=True, text=True
    )
    return python if probe.returncode == 0 else None


@pytest.mark.parametrize("asset", _ROBOTS, ids=lambda p: p.stem)
def test_shim_parses_without_crashing(asset):
    """The shim's UsdPhysics parser must not crash on any vendored robot."""
    result = _run_dump(sys.executable, "shim", asset)
    assert isinstance(result, dict)
    assert sum(len(v) for v in result.values()) > 0, f"{asset.name}: no physics objects parsed"


def _build_divergences():
    python = _usdcore_python()
    if python is None:
        pytest.skip(
            f"set {_ENV_VAR} to a Python with `usd-core` installed to run the "
            f"OpenUSD differential comparison"
        )
    divs = []
    for asset in _ROBOTS:
        shim = _run_dump(sys.executable, "shim", asset)
        openusd = _run_dump(python, "openusd", asset)
        divs.append(compare.compare(shim, openusd, asset.stem))
    report = compare.render_markdown(
        divs, {"shim": sys.executable, "openusd": python}
    )
    _REPORT.write_text(report)
    return divs


def test_usdphysics_conformance_report():
    """Generate the report AND gate against the frozen baseline (no NEW divergences)."""
    divs = _build_divergences()
    assert divs, "no robot assets found"
    assert _REPORT.exists()
    total_vm = sum(len(d.value_mismatch) for d in divs)
    print(f"\nConformance report -> {_REPORT}")
    print(f"value-mismatches across {len(divs)} robots: {total_vm}")
    ok, regressions = baseline.gate("usdphysics", baseline.signature_from_divs(divs))
    assert ok, (
        "NEW UsdPhysics conformance divergences vs baseline "
        "(set NANOUSD_CONFORMANCE_UPDATE_BASELINE=1 if intentional):\n"
        + baseline.format_regressions(regressions)
    )


if __name__ == "__main__":
    if _usdcore_python() is None:
        raise SystemExit(
            f"Set {_ENV_VAR} to a Python interpreter with `usd-core` installed."
        )
    divs = _build_divergences()
    for d in divs:
        print(
            f"{d.asset:34} struct={len(d.object_count) + len(d.path_set):3} "
            f"shim_only={len(d.shim_only):3} openusd_only={len(d.openusd_only):3} "
            f"value_mismatch={len(d.value_mismatch):3}"
        )
    print(f"\nReport written to {_REPORT}")
