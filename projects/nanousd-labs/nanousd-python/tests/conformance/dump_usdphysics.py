#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic dump of ``UsdPhysics.LoadUsdPhysicsFromRange`` output.

Runs under *either* USD backend and emits a normalized JSON description of every
physics object the parser discovers in a stage, so two backends can be diffed:

    # real OpenUSD (a usd-core interpreter)
    python dump_usdphysics.py --backend openusd --asset robot.usda > openusd.json
    # nanousd's pxr_compat shim
    python dump_usdphysics.py --backend shim    --asset robot.usda > shim.json

The two backends disagree on representation in ways that are *not* bugs (e.g.
``axis`` is an int enum in OpenUSD but a token string in the shim; descriptors
carry a redundant ``type`` field). Those are canonicalized / dropped here so the
downstream diff surfaces real semantic divergences rather than encoding noise.
"""
from __future__ import annotations

import argparse
import json
import warnings

from normalize import canon_field as _canon_field
from normalize import normalize as _normalize

warnings.filterwarnings("ignore")

# Dropped: ``type`` duplicates the ObjectType key we group by; OpenUSD exposes it
# as an int enum, the shim usually omits it — comparing it is pure noise.
_DROP_FIELDS = {"type"}


def _get_mods(backend: str):
    if backend == "shim":
        from nanousd.pxr_compat import Gf, Sdf, Usd, UsdPhysics  # noqa: F401
    elif backend == "openusd":
        from pxr import Gf, Sdf, Usd, UsdPhysics  # noqa: F401
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend {backend!r}")
    return Usd, UsdPhysics, Sdf


def _desc_to_dict(desc) -> dict:
    out = {}
    for name in dir(desc):
        if name.startswith("_") or name in _DROP_FIELDS:
            continue
        try:
            val = getattr(desc, name)
        except Exception:
            out[name] = "<err>"
            continue
        if callable(val):
            continue
        out[name] = _canon_field(name, _normalize(val))
    return out


def _objtype_name_map(UsdPhysics) -> dict:
    """Map each ObjectType member *value* (int enum or token str) to its name."""
    m = {}
    ot = UsdPhysics.ObjectType
    for n in dir(ot):
        if n.startswith("_"):
            continue
        try:
            val = getattr(ot, n)
            if not callable(val):
                m[val] = n  # may raise if a member value is unhashable; skip those
        except Exception:
            continue
    return m


def dump(backend: str, asset: str) -> dict:
    Usd, UsdPhysics, Sdf = _get_mods(backend)
    stage = Usd.Stage.Open(asset)
    if stage is None:
        raise SystemExit(f"{backend}: failed to open {asset}")
    default_prim = stage.GetDefaultPrim()
    root = default_prim.GetPath() if default_prim else Sdf.Path("/")
    ret = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root])
    namemap = _objtype_name_map(UsdPhysics)
    result: dict[str, list] = {}
    for key, value in ret.items():
        kname = namemap.get(key, str(key))
        paths, descs = value
        items = [
            {"path": str(p), "desc": _desc_to_dict(d)}
            for p, d in zip(list(paths), list(descs))
        ]
        items.sort(key=lambda x: x["path"])
        result[kname] = items
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=("shim", "openusd"))
    ap.add_argument("--asset", required=True)
    args = ap.parse_args()
    print(json.dumps(dump(args.backend, args.asset), sort_keys=True))


if __name__ == "__main__":
    main()
