#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the nusd_gles Python ctypes layer.

Loads each test USDA via NanousdStage and verifies:
  - n_prims() returns a positive count
  - prim_at_index(i) returns objects with non-empty path/type_name
  - prim_at_path() round-trips the paths discovered by index walk
  - get_attrib_* signatures don't trample memory on common types

Skipped (exit 77) if libnanousdapi.so isn't on the loader path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    try:
        from nusd_gles._nanousd import NanousdStage
    except OSError as e:
        print(f"SKIP: cannot load nanousd ctypes: {e}")
        return 77
    except ImportError as e:
        print(f"SKIP: nusd_gles import failed: {e}")
        return 77

    stages = [
        REPO / "test_cube.usda",
        REPO / "test_materials.usda",
        REPO / "test_pbr_materials.usda",
    ]

    fails = 0
    for path in stages:
        print(f"\n--- {path.name} ---")
        s = NanousdStage(str(path))
        n = s.n_prims()
        if n <= 0:
            print(f"FAIL: {path.name}: n_prims() = {n}")
            fails += 1
            continue
        print(f"ok:   n_prims = {n}")

        # Walk all prims by index and collect paths.
        paths: list[str] = []
        types: dict[str, int] = {}
        for i in range(n):
            p = s.prim_at_index(i)
            if p is None:
                print(f"FAIL: prim_at_index({i}) returned None")
                fails += 1
                break
            if not p.path:
                print(f"FAIL: prim {i} has empty path")
                fails += 1
                break
            if not p.type_name:
                print(f"FAIL: prim {i} ({p.path}) has empty type_name")
                fails += 1
                break
            paths.append(p.path)
            types[p.type_name] = types.get(p.type_name, 0) + 1

        if len(paths) == n:
            print(f"ok:   walked all {n} prims")
            print(f"      types: {types}")

        # Round-trip a few paths.
        for sample in paths[:3]:
            p2 = s.prim_at_path(sample)
            if p2 is None:
                print(f"FAIL: prim_at_path({sample!r}) returned None")
                fails += 1
            elif p2.path != sample:
                print(f"FAIL: prim_at_path({sample!r}) returned path {p2.path!r}")
                fails += 1
        else:
            print(f"ok:   prim_at_path round-trip on {min(3, len(paths))} samples")

        # Exercise some attribute readers — must not segfault on prims
        # that don't carry the attr.
        first_mesh = next((p for p in (s.prim_at_index(i) for i in range(n))
                           if p and p.type_name == "Mesh"), None)
        if first_mesh is not None:
            try:
                _ = first_mesh.attrib_names()
                _ = first_mesh.get_attrib_float("nonexistent_attr", -1.0)
                _ = first_mesh.get_attrib_str("nonexistent_attr", "")
                _ = first_mesh.get_attrib_vec3("nonexistent_attr")
                print("ok:   attribute getters tolerate missing attrs")
            except Exception as e:
                print(f"FAIL: attribute getter raised on missing attr: {e}")
                fails += 1

    if fails:
        print(f"\nFAIL: {fails} failures")
        return 1
    print("\nPASS: 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
