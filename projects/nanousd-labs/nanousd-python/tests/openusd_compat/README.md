# OpenUSD Compatibility Corpus

This directory contains a curated subset of Python tests copied from the local
`~/OpenUSD` checkout. The upstream files keep their original copyright and
license headers.

The copied files are compatibility corpus material, not a claim of full OpenUSD
compliance. They are intentionally ignored by pytest's default discovery because
many upstream tests require full OpenUSD behavior that `nanousd.pxr_compat` does
not implement. `tests/test_openusd_copied_subset.py` imports selected copied
tests and runs the parts that should currently pass.

## Schema conformance corpus (UsdPhysics / UsdGeom / UsdSkel)

`upstream/pxr/usd/{usdPhysics,usdGeom,usdSkel}/testenv/` holds OpenUSD's own
schema unit tests (copied verbatim from the `~/OpenUSD` checkout, v24.11-2166,
keeping their license headers). `tests/test_openusd_schema_conformance.py` runs
each module against the shim in an isolated subprocess and writes a living
conformance matrix to `tests/openusd_schema_conformance_report.md`. It is
**report-only** — the matrix records exactly where the shim still diverges from
OpenUSD per schema; it is not a pass/fail gate.

This is complemented by `tests/test_usdphysics_conformance.py`, which does a
*live* differential parse of the vendored Newton robots
(`tests/assets/robots/`) through both the shim and real `usd-core` via
`UsdPhysics.LoadUsdPhysicsFromRange`, flagging value-level divergences. Set
`NANOUSD_CONFORMANCE_USDCORE_PYTHON` to a Python with `usd-core` installed to
enable the live comparison (it skips cleanly otherwise).
