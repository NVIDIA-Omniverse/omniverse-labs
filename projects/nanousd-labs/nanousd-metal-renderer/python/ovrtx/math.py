# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Math helpers exposed by the OVRTX Python API."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np


class Matrix4d(ctypes.Structure):
    """4x4 double-precision matrix compatible with OVRTX examples."""

    _fields_ = [("v", (ctypes.c_double * 4) * 4)]
    _pack_ = 1

    def __init__(self):
        super().__init__()
        self.SetIdentity()

    def SetIdentity(self) -> None:
        for row in range(4):
            for col in range(4):
                self.v[row][col] = 1.0 if row == col else 0.0

    def SetTranslate(self, x: float, y: float, z: float) -> None:
        self.SetIdentity()
        self.v[3][0] = float(x)
        self.v[3][1] = float(y)
        self.v[3][2] = float(z)

    def SetScale(self, scale: float) -> None:
        self.SetIdentity()
        for i in range(3):
            self.v[i][i] = float(scale)

    def __getitem__(self, row: int):
        return self.v[row]

    def __setitem__(self, row: int, value: Any) -> None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"Row value must be list/tuple of 4 floats, got {type(value)}")
        for col in range(4):
            self.v[row][col] = float(value[col])

    def __array__(self, dtype=None):
        arr = np.array([[self.v[row][col] for col in range(4)] for row in range(4)], dtype=np.float64)
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def to_dltensor(self):
        from . import ManagedDLTensor

        return ManagedDLTensor(np.ascontiguousarray(self.__array__(dtype=np.float64)))

    def __repr__(self) -> str:
        rows = []
        for row in range(4):
            rows.append("(" + ", ".join(f"{self.v[row][col]:.6g}" for col in range(4)) + ")")
        return f"Matrix4d({', '.join(rows)})"


__all__ = ["Matrix4d"]
