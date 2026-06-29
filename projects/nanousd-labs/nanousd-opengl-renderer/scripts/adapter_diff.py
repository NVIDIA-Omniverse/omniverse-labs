# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic comparison: walk a USD stage with both pxr and our
NanousdStage, report per-prim differences in type-name, attribute
names, and attribute values.

Run with the venv active and the custom USD on PYTHONPATH (set
``$NUSD_USD_INSTALL`` to point at the OpenUSD install root):

    PYTHONPATH=$NUSD_USD_INSTALL/lib/python:python \
    LD_LIBRARY_PATH=$NUSD_USD_INSTALL/lib \
    python scripts/adapter_diff.py PATH

Output is a text report to stdout (also written to
``docs/compare/<basename>.diff.txt``).

The ``--max-prims`` flag (default 30) bounds the per-attribute diff
walk; pass ``0`` for full coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _summary(value: object) -> str:
    s = repr(value)
    return s if len(s) <= 120 else s[:117] + "..."


def diff(path: str, out_path: Optional[Path] = None, max_prims: int = 30) -> int:
    try:
        from pxr import Usd, Sdf  # custom OpenUSD build  # noqa: F401
    except Exception as exc:
        import traceback
        print(f"[pxr] import failed: {exc}")
        print("Hint: set NUSD_USD_INSTALL and add $NUSD_USD_INSTALL/lib/python "
              "to PYTHONPATH and $NUSD_USD_INSTALL/lib to LD_LIBRARY_PATH.")
        traceback.print_exc()
        return 2
    try:
        from nusd_gles._nanousd import NanousdStage
    except Exception as exc:
        import traceback
        print(f"[nanousd] import failed: {exc}")
        traceback.print_exc()
        return 2

    try:
        pxr_stage = Usd.Stage.Open(path)
    except Exception as exc:
        print(f"[pxr] open failed: {exc}")
        return 1
    try:
        nu_stage = NanousdStage(path)
    except Exception as exc:
        print(f"[nanousd] open failed: {exc}")
        return 2

    pxr_paths: list[str] = [str(p.GetPath()) for p in pxr_stage.TraverseAll()]
    nu_paths: list[str] = []
    for i in range(nu_stage.n_prims()):
        p = nu_stage.prim_at_index(i)
        if p is not None:
            nu_paths.append(p.path)

    pxr_set = set(pxr_paths)
    nu_set = set(nu_paths)
    missing = sorted(pxr_set - nu_set)
    extra = sorted(nu_set - pxr_set)

    lines: list[str] = []
    out = lines.append
    out(f"=== adapter diff for {path} ===")
    out(f"pxr prims: {len(pxr_paths)}")
    out(f"nanousd prims: {len(nu_paths)}")
    if missing:
        out(f"\nMISSING in nanousd ({len(missing)}):")
        for p in missing:
            out(f"  - {p}")
    if extra:
        out(f"\nEXTRA in nanousd ({len(extra)}):")
        for p in extra:
            out(f"  + {p}")

    # Per-prim attribute diff for shared paths.
    shared = sorted(pxr_set & nu_set)
    sample = shared if max_prims <= 0 else shared[:max_prims]
    label = "all" if max_prims <= 0 else f"first {max_prims}"
    out(f"\n--- attribute diffs ({label} of {len(shared)} shared prims) ---")
    diffs_total = 0
    for path in sample:
        pxr_prim = pxr_stage.GetPrimAtPath(path)
        nu_prim = nu_stage.prim_at_path(path)
        if pxr_prim is None or nu_prim is None:
            continue
        ptype = pxr_prim.GetTypeName()
        ntype = nu_prim.type_name
        if str(ptype) != ntype:
            out(f"\n{path}: TYPE MISMATCH pxr={ptype!r} nanousd={ntype!r}")
            diffs_total += 1
        # attribute name set
        pattrs = {a.GetName() for a in pxr_prim.GetAuthoredAttributes()}
        nattrs = set(nu_prim.attrib_names())
        only_pxr = pattrs - nattrs
        only_nu = nattrs - pattrs
        if only_pxr or only_nu:
            out(f"\n{path}:")
            if only_pxr:
                out(f"   attrs pxr-only: {sorted(only_pxr)[:10]}")
                diffs_total += 1
            if only_nu:
                out(f"   attrs nanousd-only: {sorted(only_nu)[:10]}")
                diffs_total += 1
        # value mismatches for shared scalar attrs
        for name in sorted(pattrs & nattrs):
            try:
                pv = pxr_prim.GetAttribute(name).Get()
            except Exception:
                continue
            ptype_str = str(pxr_prim.GetAttribute(name).GetTypeName()).lower().replace("[]", "")
            try:
                if "color3" in ptype_str or "vec3" in ptype_str or ptype_str in ("normal3f", "point3f", "vector3f", "double3", "float3"):
                    nv = nu_prim.get_attrib_vec3(name)
                elif "bool" in ptype_str:
                    nv = nu_prim.get_attrib_bool(name)
                elif "int" in ptype_str:
                    nv = nu_prim.get_attrib_int(name)
                elif "string" in ptype_str or "token" in ptype_str:
                    nv = nu_prim.get_attrib_str(name)
                elif "float" in ptype_str or "double" in ptype_str:
                    nv = nu_prim.get_attrib_float(name)
                else:
                    continue
            except Exception:
                continue

            # Normalise pv for comparison
            if pv is None and nv in (None, 0.0, "", False, 0):
                continue
            if hasattr(pv, "__len__") and not isinstance(pv, str):
                try:
                    pv_t = tuple(float(x) for x in pv)
                except Exception:
                    pv_t = pv
                if isinstance(nv, tuple) and len(nv) == len(pv_t):
                    if all(abs(a - b) < 1e-5 for a, b in zip(pv_t, nv)):
                        continue
                out(f"  {path}.{name}: pxr={_summary(pv)} nanousd={_summary(nv)}")
                diffs_total += 1
            else:
                try:
                    if isinstance(pv, str) and isinstance(nv, str) and pv == nv:
                        continue
                    if isinstance(pv, (int, float)) and isinstance(nv, (int, float)) and abs(float(pv) - float(nv)) < 1e-5:
                        continue
                    if pv == nv:
                        continue
                except Exception:
                    pass
                out(f"  {path}.{name}: pxr={_summary(pv)} nanousd={_summary(nv)}")
                diffs_total += 1

    out(f"\n=== total diffs: {diffs_total} ===")

    text = "\n".join(lines)
    print(text)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return 0 if diffs_total == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-prims", type=int, default=30,
                    help="Max shared prims to per-attribute diff (0 = all)")
    args = ap.parse_args(argv)
    out_path = Path(args.out) if args.out else None
    return diff(args.usd_path, out_path, max_prims=args.max_prims)


if __name__ == "__main__":
    sys.exit(main())
