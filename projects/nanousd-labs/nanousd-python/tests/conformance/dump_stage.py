#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic full-stage dump for the OpenUSD ⇄ nanousd differential test.

Where ``dump_usdphysics.py`` exercises a single bulk-parse API, this walks the
*entire* stage and records the broad read/query surface for every prim, so a
diff validates that surface call-for-call across each robot:

* ``Usd.Stage.Open`` / ``Traverse``
* ``prim.GetTypeName`` / ``IsActive`` / ``IsInstanceable`` / ``GetSpecifier``
* ``prim.GetAppliedSchemas``  (UsdGeom/UsdPhysics/UsdShade/... API schemas)
* ``prim.GetMetadata('kind')``
* every **authored** ``attr.Get()`` (every USD value type present in the asset)
  with ``attr.GetTypeName`` / ``attr.HasAuthoredValue`` / ``GetVariability``
* every ``rel.GetTargets()``

Emitted as JSON keyed by prim path; see ``compare.compare_generic``.
"""
from __future__ import annotations

import argparse
import json
import warnings

from normalize import normalize

warnings.filterwarnings("ignore")


def _canon_token(value) -> str:
    """Canonicalize enum-ish tokens that differ only in spelling between backends.

    OpenUSD stringifies enums as ``Sdf.VariabilityVarying`` / ``Sdf.SpecifierDef``;
    the shim returns bare ``varying`` / ``def``. Reduce both to the bare lowercase
    token so only *semantic* differences (e.g. varying-vs-uniform) are flagged.
    """
    last = str(value).split(".")[-1]
    for prefix in ("Variability", "Specifier", "Visibility", "Purpose"):
        if last.startswith(prefix):
            last = last[len(prefix):]
    return last.lower()


def _get_mods(backend: str):
    if backend == "shim":
        from nanousd.pxr_compat import Sdf, Usd  # noqa: F401
    elif backend == "openusd":
        from pxr import Sdf, Usd  # noqa: F401
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend {backend!r}")
    return Usd, Sdf


def _attr_entry(attr) -> dict:
    entry = {}
    try:
        entry["type"] = str(attr.GetTypeName())
    except Exception:
        entry["type"] = "<err>"
    try:
        entry["variability"] = _canon_token(attr.GetVariability())
    except Exception:
        pass
    try:
        entry["value"] = normalize(attr.Get())
    except Exception as exc:
        entry["value"] = f"<err:{type(exc).__name__}>"
    return entry


def _prim_entry(prim) -> dict:
    entry: dict = {}
    # All getters are lambdas so attribute access is deferred into the try: a
    # method the shim does not implement is *recorded* as "<unsupported>" (and
    # thus flagged against OpenUSD), never allowed to crash the walk.
    getters = {
        "type": lambda: str(prim.GetTypeName()),
        "active": lambda: prim.IsActive(),
        "instanceable": lambda: prim.IsInstanceable(),
        "specifier": lambda: _canon_token(prim.GetSpecifier()),
    }
    for label, getter in getters.items():
        try:
            entry[label] = getter()
        except Exception as exc:
            entry[label] = f"<unsupported:{type(exc).__name__}>"
    try:
        entry["schemas"] = sorted(str(s) for s in prim.GetAppliedSchemas())
    except Exception:
        entry["schemas"] = "<err>"
    try:
        kind = prim.GetMetadata("kind")
        if kind:
            entry["kind"] = str(kind)
    except Exception:
        pass
    attrs = {}
    try:
        for attr in prim.GetAttributes():
            try:
                if attr.HasAuthoredValue():
                    attrs[attr.GetName()] = _attr_entry(attr)
            except Exception:
                attrs[getattr(attr, "GetName", lambda: "?")()] = {"value": "<err>"}
    except Exception:
        attrs = {"<err>": {}}
    entry["attrs"] = attrs
    rels = {}
    try:
        for rel in prim.GetRelationships():
            try:
                rels[rel.GetName()] = sorted(str(t) for t in rel.GetTargets())
            except Exception:
                rels[rel.GetName()] = "<err>"
    except Exception:
        rels = {"<err>": []}
    entry["rels"] = rels
    return entry


def dump(backend: str, asset: str) -> dict:
    Usd, Sdf = _get_mods(backend)
    stage = Usd.Stage.Open(asset)
    if stage is None:
        raise SystemExit(f"{backend}: failed to open {asset}")
    result: dict[str, dict] = {}
    for prim in stage.Traverse():
        result[str(prim.GetPath())] = _prim_entry(prim)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("shim", "openusd"))
    ap.add_argument("--asset", required=True)
    args = ap.parse_args()
    print(json.dumps(dump(args.backend, args.asset), sort_keys=True))


if __name__ == "__main__":
    main()
