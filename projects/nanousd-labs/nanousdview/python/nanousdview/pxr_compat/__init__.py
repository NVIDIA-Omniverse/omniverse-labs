# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nanousdview compatibility entry point for shared nanousd Python bindings.

nanousdview historically carried a local ``pxr`` shim here. The
authoritative shim now lives in ``nanousd-python`` as ``nanousd.pxr_compat``;
this module preserves existing imports such as::

    from nanousdview.pxr_compat import Usd, UsdGeom

while ensuring the implementation is provided by the shared package.
"""

from __future__ import annotations

import importlib
import sys

try:
    from nanousd import pxr_compat as _pxr_compat
except ImportError as exc:  # pragma: no cover - exercised by install failures
    raise ImportError(
        "nanousdview requires the nanousd-python package. Build the workspace "
        "with ./build.sh or install nanousd-python from the sibling "
        "nanousd-python repo."
    ) from exc

try:
    _compat_impl = importlib.import_module("nanousd.pxr_compat._ctypes_compat")
except ModuleNotFoundError:
    _compat_impl = importlib.import_module("nanousd.pxr_compat._native_compat")

sys.modules.setdefault(__name__ + "._ctypes_compat", _compat_impl)
sys.modules.setdefault(__name__ + "._native_compat", _compat_impl)

for _name, _value in _pxr_compat.__dict__.items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

__all__ = getattr(_pxr_compat, "__all__", [])
