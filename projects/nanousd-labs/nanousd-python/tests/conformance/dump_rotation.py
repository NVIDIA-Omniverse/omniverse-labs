# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dump Gf.Matrix4d rotation-extraction results for differential conformance.

Runs under either the nanousd ``pxr_compat`` shim (``sys.executable``) or a real
``usd-core`` interpreter; both build the SAME deterministic battery of row-vector
rotation matrices (identical input floats, independent of any RNG or of either
``SetRotate``), then run each interpreter's own ``ExtractRotation`` / ``Decompose``
/ ``Transform`` and emit the results as JSON on stdout. The caller diffs the two.

Usage:
    python dump_rotation.py        # emits {label: {quat, yxz, xyz, tp}} JSON
"""

from __future__ import annotations

import json
import math
import sys


def battery():
    """Deterministic ``(label, axis, angle_deg)`` cases — identical everywhere."""
    axes = {
        "X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1),
        "XY": (1, 1, 0), "XZ": (1, 0, 1), "YZ": (0, 1, 1), "XYZ": (1, 1, 1),
        "obliqA": (0.3, -0.7, 0.65), "obliqB": (-0.5, 0.2, 0.84),
        "obliqC": (0.91, 0.13, -0.39),
    }
    angles = (-179, -135, -90, -89.999, -45, -1, 0, 1, 30, 45, 89.999, 90, 135, 179, 180)
    return [(f"{nm}@{ang}", ax, ang) for nm, ax in axes.items() for ang in angles]


def rowvec_matrix(axis, ang_deg):
    """Row-vector (p' = p*M) rotation as a flat 16-float list (OpenUSD layout)."""
    ax = [float(c) for c in axis]
    n = math.sqrt(sum(c * c for c in ax)) or 1.0
    ax = [c / n for c in ax]
    h = math.radians(ang_deg) * 0.5
    s = math.sin(h)
    w = math.cos(h)
    x, y, z = ax[0] * s, ax[1] * s, ax[2] * s
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
        2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
        2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
        0, 0, 0, 1,
    ]


def main():
    from pxr import Gf

    out = {}
    for lab, ax, ang in battery():
        M = Gf.Matrix4d(*rowvec_matrix(ax, ang))
        r = M.ExtractRotation()
        q = r.GetQuat()
        out[lab] = {
            "quat": [float(q.GetReal())] + [float(v) for v in q.GetImaginary()],
            "yxz": [float(v) for v in r.Decompose(
                Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis(), Gf.Vec3d.ZAxis())],
            "xyz": [float(v) for v in r.Decompose(
                Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())],
            "tp": [float(v) for v in M.Transform(Gf.Vec3d(1.0, 2.0, 3.0))],
        }
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
