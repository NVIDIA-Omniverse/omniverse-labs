# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance for Gf.Matrix4d rotation/compose helpers vs real OpenUSD.

These cover the pieces usdview's free camera depends on (freeCamera.py:151-157):
``Matrix4d.__mul__`` (matrix product), ``ExtractRotation``, ``Rotation.Decompose``
and ``Matrix4d.Transform`` — all in OpenUSD's row-vector (p' = p*M) convention.

The self-contained tests (reconstruction + round-trip) need no oracle and run in
CI: they would have caught the earlier wrong-handedness bug, where ``SetRotate``
built the column-vector transpose and ``__mul__`` was numpy element-wise. The
``test_matches_usdcore`` battery additionally diffs against a real ``usd-core``
interpreter when ``NANOUSD_CONFORMANCE_USDCORE_PYTHON`` is set.

    NANOUSD_CONFORMANCE_USDCORE_PYTHON=/path/to/usdcore-venv/bin/python pytest ...
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pxr import Gf

_HERE = Path(__file__).resolve().parent
_DUMP = _HERE / "conformance" / "dump_rotation.py"
_ENV_VAR = "NANOUSD_CONFORMANCE_USDCORE_PYTHON"

sys.path.insert(0, str(_HERE / "conformance"))
import dump_rotation  # noqa: E402  (battery + row-vector reference builder)

_AXIS_VEC = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}


def _rmat3(axis_index: int, ang_deg: float) -> np.ndarray:
    """Reference row-vector 3x3 rotation about a cardinal axis."""
    flat = dump_rotation.rowvec_matrix(_AXIS_VEC[axis_index], ang_deg)
    return np.array(flat, dtype=np.float64).reshape(4, 4)[:3, :3]


# --------------------------------------------------------------------------- #
# Self-contained (no oracle) — the durable CI gate.
# --------------------------------------------------------------------------- #

def test_setrotate_is_row_vector():
    """SetRotate must build OpenUSD's row-vector matrix (not its transpose)."""
    m = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d.XAxis(), 30.0))
    # Row-vector Rx(30): row1 = (0, cos, +sin); the column-vector transpose would
    # put -sin here (the prior silent bug).
    assert m[1][1] == pytest.approx(math.cos(math.radians(30)), abs=1e-12)
    assert m[1][2] == pytest.approx(math.sin(math.radians(30)), abs=1e-12)
    assert m[2][1] == pytest.approx(-math.sin(math.radians(30)), abs=1e-12)


def test_mul_is_matrix_product_not_elementwise():
    """Matrix4d * Matrix4d is the matrix product; * scalar scales components."""
    R = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d.XAxis(), 30.0))
    T = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(10, 20, 30))
    prod = np.asarray(R * T, dtype=np.float64)
    assert np.allclose(prod, np.asarray(R) @ np.asarray(T), atol=1e-12)
    # Element-wise (the bug) would have dropped both the off-diagonal rotation
    # and the translation row.
    assert prod[3][0] == pytest.approx(10.0) and prod[1][2] == pytest.approx(0.5)
    assert np.allclose(np.asarray(R * 2.0), np.asarray(R) * 2.0, atol=1e-12)
    assert np.allclose(np.asarray(2.0 * R), np.asarray(R) * 2.0, atol=1e-12)


def test_transform_row_vector_with_homogeneous_divide():
    m = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(10, 20, 30))
    assert tuple(m.Transform(Gf.Vec3d(1, 2, 3))) == pytest.approx((11, 22, 33))


@pytest.mark.parametrize("order", ["yxz", "xyz"])
def test_extract_decompose_reconstructs(order):
    """ExtractRotation().Decompose() must reconstruct the original rotation.

    OpenUSD's Decompose(a0,a1,a2) returns (t0,t1,t2) with the REVERSE-order
    product M == R(a2,t2) @ R(a1,t1) @ R(a0,t0). This round-trip catches any
    handedness/convention error without needing an oracle.
    """
    axis_order = {"yxz": (1, 0, 2), "xyz": (0, 1, 2)}[order]
    decompose_axes = [_AXIS_VEC[i] for i in axis_order]
    worst = 0.0
    for lab, ax, ang in dump_rotation.battery():
        M = Gf.Matrix4d(*dump_rotation.rowvec_matrix(ax, ang))
        m3 = np.asarray(M, dtype=np.float64)[:3, :3]
        r = M.ExtractRotation()
        # (a) ExtractRotation must recover the same rotation (quat round-trip).
        q = r.GetQuat()
        w, (x, y, z) = q.GetReal(), q.GetImaginary()
        rebuilt = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(r.GetAxis(), r.GetAngle()))
        assert np.allclose(np.asarray(rebuilt)[:3, :3], m3, atol=1e-9), lab
        # (b) Decompose must reconstruct via the reverse-order product.
        t0, t1, t2 = (float(v) for v in r.Decompose(*[Gf.Vec3d(*a) for a in decompose_axes]))
        recon = _rmat3(axis_order[2], t2) @ _rmat3(axis_order[1], t1) @ _rmat3(axis_order[0], t0)
        worst = max(worst, np.abs(recon - m3).max())
    assert worst < 1e-9, f"{order}: worst reconstruction error {worst:.2e}"


# --------------------------------------------------------------------------- #
# Differential vs real usd-core (oracle-gated).
# --------------------------------------------------------------------------- #

def _usdcore_python():
    python = os.environ.get(_ENV_VAR)
    if not python or not Path(python).exists():
        return None
    probe = subprocess.run([python, "-c", "import pxr.Gf"], capture_output=True, text=True)
    return python if probe.returncode == 0 else None


def _run_dump(python: str) -> dict:
    proc = subprocess.run([python, str(_DUMP)], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"dump_rotation failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _angdiff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def test_matches_usdcore():
    python = _usdcore_python()
    if python is None:
        pytest.skip(f"set {_ENV_VAR} to a usd-core interpreter for the OpenUSD diff")
    shim = _run_dump(sys.executable)
    oracle = _run_dump(python)
    assert shim.keys() == oracle.keys()
    for lab in oracle:
        s, o = shim[lab], oracle[lab]
        qs, qo = np.array(s["quat"]), np.array(o["quat"])
        dq = min(np.abs(qs - qo).max(), np.abs(qs + qo).max())  # quat sign ambiguity
        assert dq < 1e-6, f"{lab}: quat diff {dq:.2e}"
        for order in ("yxz", "xyz"):
            de = max(_angdiff(a, b) for a, b in zip(s[order], o[order]))
            assert de < 1e-4, f"{lab}: {order} euler diff {de:.2e} deg"
        dt = np.abs(np.array(s["tp"]) - np.array(o["tp"])).max()
        assert dt < 1e-6, f"{lab}: Transform diff {dt:.2e}"
