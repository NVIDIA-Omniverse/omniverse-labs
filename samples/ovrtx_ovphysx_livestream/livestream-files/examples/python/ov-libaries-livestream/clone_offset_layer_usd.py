# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the clone root transform with OpenUSD (pxr); print a 4x4 row-major matrix for ovrtx.

ovrtx cannot import ``pxr`` in-process. This helper runs under a separate interpreter with **usd-core**::

    uv run --with usd-core python clone_offset_layer_usd.py <usd_url> <source_prim_path> <offset_x>
    uv run --with usd-core python clone_offset_layer_usd.py <usd_url> <source_prim_path> --local-matrix

With ``<offset_x>``, it writes **one line** of 16 whitespace-separated ``float64`` values (USD
``GfMatrix4d`` / row-major) for the **local** transform of ``<source_prim_path>`` composed with a
parent-space +X translation of ``offset_x``.

With ``--local-matrix``, it prints only ``UsdGeom.Xformable.GetLocalTransformation`` (no extra
translation) so the parent can compose grid offsets in NumPy.
"""

from __future__ import annotations

import sys
import urllib.request

from pxr import Gf, Sdf, Usd, UsdGeom


def _open_stage_from_url(usd_url: str) -> Usd.Stage:
    if usd_url.startswith("http://") or usd_url.startswith("https://"):
        with urllib.request.urlopen(usd_url, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(text)
        return Usd.Stage.Open(layer)
    return Usd.Stage.Open(usd_url)


def get_local_matrix(usd_url: str, source_prim_path: str) -> Gf.Matrix4d:
    """Local-to-parent matrix for ``source_prim`` (no extra translation)."""
    stage = _open_stage_from_url(usd_url)
    src = stage.GetPrimAtPath(Sdf.Path(source_prim_path))
    if not src.IsValid():
        raise SystemExit(f"Source prim not found: {source_prim_path!r}")

    xf = UsdGeom.Xformable(src)
    t = Usd.TimeCode.Default()
    return xf.GetLocalTransformation(t)


def compute_offset_local_matrix(usd_url: str, source_prim_path: str, offset_parent_x: float) -> Gf.Matrix4d:
    """Local-to-parent matrix for ``source_prim`` with +offset_parent_x applied in parent +X."""
    local = get_local_matrix(usd_url, source_prim_path)
    translate = Gf.Matrix4d()
    translate.SetTranslate(Gf.Vec3d(float(offset_parent_x), 0.0, 0.0))
    return local * translate


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "usage: clone_offset_layer_usd.py <usd_url_or_path> <source_prim_path> <offset_x>",
            file=sys.stderr,
        )
        print(
            "   or: clone_offset_layer_usd.py <usd_url_or_path> <source_prim_path> --local-matrix",
            file=sys.stderr,
        )
        raise SystemExit(2)
    url, source_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) == 4 and sys.argv[3] == "--local-matrix":
        m = get_local_matrix(url, source_path)
    else:
        ox_s = sys.argv[3]
        try:
            ox = float(ox_s)
        except ValueError:
            print(f"invalid offset_x: {ox_s!r}", file=sys.stderr)
            raise SystemExit(2) from None

        m = compute_offset_local_matrix(url, source_path, ox)
    row = []
    for i in range(4):
        for j in range(4):
            row.append(str(m[i][j]))
    sys.stdout.write(" ".join(row))


if __name__ == "__main__":
    main()
