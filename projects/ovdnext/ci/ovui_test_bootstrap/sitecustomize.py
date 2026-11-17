# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local Windows DLL registration inherited by pytest child Python."""
import os
import sys


DLL_HANDLES = []
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(3)
    for directory in os.environ.get("OVUI_TEST_DLL_DIRECTORIES", "").split(os.pathsep):
        if directory:
            DLL_HANDLES.append(os.add_dll_directory(directory))
    if os.environ.get("OVD_UI_HIDDEN_GUI_TESTS") == "1" and (library := os.environ.get("GLCONTEXT_WIN_LIBGL")):
        # GLFW must bind the same CI-only implementation as ModernGL. A PATH
        # entry alone still lets Windows choose its system OpenGL library.
        try:
            DLL_HANDLES.append(ctypes.WinDLL(os.path.abspath(library), winmode=8))
        except OSError as error:
            print(f"Cannot load configured CI OpenGL library: {error}", file=sys.stderr)
            # Ordinary sitecustomize exceptions are printed but ignored by Python.
            raise SystemExit(1) from error
