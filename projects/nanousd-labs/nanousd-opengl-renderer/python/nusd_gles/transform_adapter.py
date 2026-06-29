# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanousdTransformAdapter — read-only TransformAdapter for nanousd prims.

The C API exposes ``nanousd_get_local_transform`` for local matrices but
not a world-space accumulator. We compose world matrices ourselves by
walking parent pointers. Edits are stubbed for now.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional

from ovgear.adapters import TransformAdapter

from nusd_gles._nanousd import NanousdStage, _Prim, _lib  # noqa: WPS450 — internal lib reuse


# Bind nanousd_get_local_transform. C signature (nanousd/nanousdapi.h):
#   int nanousd_get_local_transform(NanousdPrim prim, double time,
#                                   double out[16], int* resetXformStack)
# Returns 1 on success and writes a 4x4 row-major matrix to ``out``.
# ``resetXformStack`` is set to 1 if the prim has !resetXformStack!
# in its xformOpOrder (we do not use this signal but the pointer is
# required — the backend dereferences it before writing).
try:
    _lib.nanousd_get_local_transform.restype = ctypes.c_int
    _lib.nanousd_get_local_transform.argtypes = [
        ctypes.c_void_p,                       # prim
        ctypes.c_double,                       # time
        ctypes.POINTER(ctypes.c_double * 16),  # out[16]
        ctypes.POINTER(ctypes.c_int),          # resetXformStack
    ]
    _HAS_LOCAL_XFORM = True
except AttributeError:
    _HAS_LOCAL_XFORM = False


def _identity4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = _identity4()
    for r in range(4):
        for c in range(4):
            out[r][c] = sum(a[r][k] * b[k][c] for k in range(4))
    return out


def _read_local(prim: _Prim) -> List[List[float]]:
    if not _HAS_LOCAL_XFORM or prim.handle is None:
        return _identity4()
    buf = (ctypes.c_double * 16)()
    reset = ctypes.c_int(0)
    ok = _lib.nanousd_get_local_transform(
        prim.handle, ctypes.c_double(0.0), ctypes.byref(buf), ctypes.byref(reset)
    )
    if not ok:
        return _identity4()
    # USD's matrix layout is row-major; expose as 4×4 row-major nested list.
    flat = [float(buf[i]) for i in range(16)]
    return [flat[i * 4:(i + 1) * 4] for i in range(4)]


class NanousdTransformAdapter(TransformAdapter):
    """Read-only transform adapter — local from the C API, world by composition."""

    def __init__(self, stage: NanousdStage) -> None:
        self._stage = stage

    def _prim(self, path: str) -> Optional[_Prim]:
        return self._stage.prim_at_path(path)

    def get_local_transform(self, path: str) -> List[List[float]]:
        prim = self._prim(path)
        if prim is None:
            return _identity4()
        return _read_local(prim)

    def get_world_transform(self, path: str) -> List[List[float]]:
        # Walk from root → leaf accumulating local matrices. Path
        # ancestry is derivable from string slashes on a USD path.
        parts = [s for s in path.split("/") if s]
        accum = _identity4()
        cur = ""
        for part in parts:
            cur = f"{cur}/{part}"
            local = self.get_local_transform(cur)
            accum = _matmul(accum, local)
        return accum

    def set_local_transform(self, path: str, matrix: List[List[float]]) -> None:
        prim = self._prim(path)
        if prim is None:
            return
        # Mirror UsdGeom.Xformable.MakeMatrixXform: declare a single
        # ``xformOp:transform`` matrix4d xformOp, author the matrix,
        # and rewrite xformOpOrder to that single op. ``create_attrib``
        # is a no-op when the attribute already exists, so this is
        # safe to call repeatedly during a drag.
        try:
            if not prim.has_attrib("xformOp:transform"):
                prim.create_attrib("xformOp:transform", "matrix4d")
            if not prim.has_attrib("xformOpOrder"):
                prim.create_attrib("xformOpOrder", "token[]")
            prim.set_attrib_matrix4d("xformOp:transform", matrix)
            prim.set_attrib_array_tokens("xformOpOrder", ["xformOp:transform"])
        except Exception:
            return

    def can_transform(self, path: str) -> bool:
        return self._prim(path) is not None
