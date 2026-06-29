# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level ``pxr`` import shim backed by ``nanousd.pxr_compat``."""

from __future__ import annotations

import importlib
import sys

importlib.import_module("nanousd.pxr_compat")
_pxr = sys.modules["pxr"]

for _name, _value in _pxr.__dict__.items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

__path__ = getattr(_pxr, "__path__", [])
__all__ = [name for name in globals() if not name.startswith("_")]
