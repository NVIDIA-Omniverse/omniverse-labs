# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python bindings for nanousd.

The package has two layers:

* ``nanousd.Stage`` / ``nanousd.Prim`` from the small nanobind extension.
* ``nanousd.pxr_compat`` for the extracted ``from pxr import Usd, Sdf, ...``
  compatibility shim used by nanousdview.
"""

from __future__ import annotations


# Single canonical libnanousd backend (single-inode ODR fix). In a workspace that
# also builds the renderers, nanousdapi would otherwise dlopen the bundled core —
# a 2nd inode of the same soname as the renderers' nanousd/build core — and glibc
# maps BOTH, so their global state collides and silently breaks composition (the
# Vulkan viewer renders 0 meshes). Default NANOUSD_BACKEND to the canonical renderer
# core recorded at build time so every consumer loads ONE inode. Respects an explicit
# override, and is a no-op for standalone wheels (no recorded path -> bundled core).
def _nanousd_set_canonical_backend() -> None:
    import os

    if os.environ.get("NANOUSD_BACKEND"):
        return
    try:
        from ._backend_path import CANONICAL_BACKEND as _cb
    except Exception:
        return
    if _cb and os.path.exists(_cb):
        os.environ["NANOUSD_BACKEND"] = _cb


_nanousd_set_canonical_backend()

from ._nanousd import (  # noqa: F401
    Diagnostic,
    ListOp,
    Path,
    Prim,
    Stage,
    cross3d,
    cross3f,
    dot3d,
    dot3f,
    length3d,
    length3f,
    mul_matrix4d,
    normalize3d,
    normalize3f,
    quat_slerp,
    quat_to_matrix,
    register_schemas_json,
    transform_point3d,
)

# Importing pxr_compat also registers the synthetic ``pxr`` package in
# sys.modules, so downstream code can keep using ``from pxr import Usd``.
from .pxr_compat import (  # noqa: F401
    Ar,
    CameraUtil,
    Gf,
    Kind,
    Sdf,
    Tf,
    Trace,
    Ts,
    Usd,
    UsdAppUtils,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdRender,
    UsdSemantics,
    UsdShade,
    UsdUtils,
    Vt,
)

__all__ = [
    "Stage",
    "Prim",
    "Path",
    "ListOp",
    "Diagnostic",
    "register_schemas_json",
    "dot3f",
    "dot3d",
    "length3f",
    "length3d",
    "normalize3f",
    "normalize3d",
    "cross3f",
    "cross3d",
    "mul_matrix4d",
    "transform_point3d",
    "quat_slerp",
    "quat_to_matrix",
    "Usd",
    "Sdf",
    "UsdGeom",
    "UsdShade",
    "UsdPhysics",
    "UsdLux",
    "UsdRender",
    "Gf",
    "Vt",
    "Tf",
    "Kind",
    "Ar",
    "Trace",
    "Ts",
    "UsdSemantics",
    "CameraUtil",
    "UsdAppUtils",
    "UsdUtils",
]

__version__ = "0.1.0"
