# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dump the first N Material/Shader prims from a USD via nanousdapi.

Bypasses GLES, scene loader, and any rendering. We just want to know
WHAT the material loader sees when it walks the warehouse — does
nanousd actually surface the `inputs:AlbedoTexture` etc. asset attributes
on Shader prims that live inside referenced prop USDs?

Usage:
    python tests/textures_debug/dump_warehouse_materials.py [usd_path] [max_mats]
"""

from __future__ import annotations
import ctypes
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))

# Force the dylib to load via the existing wrapper.
from nusd_gles import _nanousd  # noqa: F401

lib = ctypes.CDLL(str(REPO / "build/libnusd_gles.dylib"))

lib.nanousd_open.restype = ctypes.c_void_p
lib.nanousd_open.argtypes = [ctypes.c_char_p]
lib.nanousd_close.restype = None
lib.nanousd_close.argtypes = [ctypes.c_void_p]
lib.nanousd_nprims.restype = ctypes.c_int
lib.nanousd_nprims.argtypes = [ctypes.c_void_p]
lib.nanousd_prim.restype = ctypes.c_void_p
lib.nanousd_prim.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.nanousd_freeprim.restype = None
lib.nanousd_freeprim.argtypes = [ctypes.c_void_p]
lib.nanousd_path.restype = ctypes.c_char_p
lib.nanousd_path.argtypes = [ctypes.c_void_p]
lib.nanousd_typename.restype = ctypes.c_char_p
lib.nanousd_typename.argtypes = [ctypes.c_void_p]
lib.nanousd_isa.restype = ctypes.c_int
lib.nanousd_isa.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nanousd_nchildren.restype = ctypes.c_int
lib.nanousd_nchildren.argtypes = [ctypes.c_void_p]
lib.nanousd_child.restype = ctypes.c_void_p
lib.nanousd_child.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.nanousd_nattribs.restype = ctypes.c_int
lib.nanousd_nattribs.argtypes = [ctypes.c_void_p]
lib.nanousd_attribname.restype = ctypes.c_char_p
lib.nanousd_attribname.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.nanousd_attribtype.restype = ctypes.c_char_p
lib.nanousd_attribtype.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.nanousd_attribs.restype = ctypes.c_char_p
lib.nanousd_attribs.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
]
lib.nanousd_attribasset.restype = ctypes.c_char_p
lib.nanousd_attribasset.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
]


def s(p):
    return None if not p else p.decode("utf-8", errors="replace")


def dump_shader(shader, indent="    "):
    n = lib.nanousd_nattribs(shader)
    print(f"{indent}attrs: {n}")
    for i in range(n):
        name = s(lib.nanousd_attribname(shader, i))
        if not name:
            continue
        typ = s(lib.nanousd_attribtype(shader, name.encode()))
        ok = ctypes.c_int(0)
        sval = s(lib.nanousd_attribs(shader, name.encode(), ctypes.byref(ok)))
        sok = ok.value
        ok = ctypes.c_int(0)
        aval = s(lib.nanousd_attribasset(shader, name.encode(), ctypes.byref(ok)))
        aok = ok.value
        print(f"{indent}  {name} type={typ}  s_ok={sok} s={sval!r}  a_ok={aok} a={aval!r}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.environ.get(
            "NUSD_ISAAC_WAREHOUSE",
            str(Path.home() / "assets/isaac/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"),
        )
    max_mats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Opening {path}")
    h = lib.nanousd_open(path.encode())
    if not h:
        print("  FAILED")
        return 1

    n = lib.nanousd_nprims(h)
    print(f"  prims: {n}")

    found = 0
    for i in range(n):
        prim = lib.nanousd_prim(h, i)
        if not prim:
            continue
        try:
            if lib.nanousd_isa(prim, b"Material"):
                pp = s(lib.nanousd_path(prim))
                tn = s(lib.nanousd_typename(prim))
                print(f"\n=== MATERIAL [{found}] {pp}  type={tn}")
                nch = lib.nanousd_nchildren(prim)
                print(f"  children: {nch}")
                for c in range(nch):
                    child = lib.nanousd_child(prim, c)
                    if not child:
                        continue
                    try:
                        cp = s(lib.nanousd_path(child))
                        ctn = s(lib.nanousd_typename(child))
                        print(f"  child {c}: {cp}  type={ctn}")
                        if lib.nanousd_isa(child, b"Shader"):
                            dump_shader(child)
                    finally:
                        lib.nanousd_freeprim(child)
                found += 1
                if found >= max_mats:
                    break
        finally:
            lib.nanousd_freeprim(prim)

    lib.nanousd_close(h)
    print(f"\ndumped {found} materials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
