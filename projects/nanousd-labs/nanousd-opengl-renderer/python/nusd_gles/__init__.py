# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nusd_gles — Python bindings for the OpenGL ES nanoUSD viewer + ovgear adapters.

Submodules load their backing dylibs at import time, so we deliberately
do NOT eagerly re-export :class:`GlesViewer` / :class:`NanousdStage` at
the package level. On macOS, ``libnusd_gles.dylib`` statically links
GLFW; loading it before ``omni.ui`` does causes our embedded GLFW to
register its Objective-C classes (``GLFWContentView``, etc.) first, and
ovui's NSWindow then picks up our class for its content view —
silently dropping every mouse event in the viewport. Letting clients
opt into the submodule import keeps that ordering in their hands.

Import from the submodules directly::

    from nusd_gles._bindings import GlesViewer
    from nusd_gles._nanousd import NanousdStage
"""
