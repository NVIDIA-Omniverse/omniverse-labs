#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Run one copied OpenUSD schema test module against the nanousd ``pxr_compat`` shim.

Isolated in a subprocess (one per module) so a crash, hang, or ``sys.exit`` in an
upstream test cannot take down the whole conformance sweep. Emits a JSON line:

    {"ran": 20, "failures": 3, "errors": 17, "skipped": 0,
     "samples": ["test_x :: AttributeError: ... 'Gf' has no attribute 'Transform'"]}

or ``{"load_error": "..."}`` if the module cannot even be imported under the shim.

Upstream tests open data files by bare name (``Stage.Open("Sphere.usda")``) with
the data sitting in a ``<testname>/`` sibling directory, so we ``chdir`` there.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path


def main() -> None:
    mod_path = Path(sys.argv[1]).resolve()
    data_dir = mod_path.parent / mod_path.stem  # testenv/<testname>/ holds the data
    os.chdir(str(data_dir if data_dir.is_dir() else mod_path.parent))

    import nanousd  # noqa: F401 — registers the `pxr` shim in sys.modules

    out: dict = {}
    try:
        spec = importlib.util.spec_from_file_location("copied_" + mod_path.stem, mod_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            result = unittest.TestResult()
            suite.run(result)
        samples = []
        for who, tb in (result.errors + result.failures)[:6]:
            name = str(who).split()[0]
            last = tb.strip().splitlines()[-1][:160] if tb.strip() else ""
            samples.append(f"{name} :: {last}")
        out = {
            "ran": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "samples": samples,
        }
    except BaseException as exc:  # noqa: BLE001 — upstream modules may sys.exit/raise anything
        import traceback

        out = {"load_error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-600:]}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
