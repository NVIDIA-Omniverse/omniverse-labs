# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for the shared nanousd pxr compatibility shim."""

from __future__ import annotations

import importlib
import sys

try:
    _impl = importlib.import_module("nanousd.pxr_compat._ctypes_compat")
except ModuleNotFoundError:
    _impl = importlib.import_module("nanousd.pxr_compat._native_compat")

for _name, _value in _impl.__dict__.items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

sys.modules[__name__] = _impl
