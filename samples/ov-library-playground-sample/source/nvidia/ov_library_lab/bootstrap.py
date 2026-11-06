# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate native dependencies supplied by the installed libraries, without Kit."""
import importlib.util
import os
from pathlib import Path

_dll_directories=[]
_native_libraries=[]


def configure():
    os.environ.setdefault('OVRTX_SKIP_USD_CHECK','1')
    # In the pinned ovui build Vulkan renders offscreen; it has no desktop
    # swapchain. OpenGL presents desktop windows and also supports CUDA upload.
    # Keep Vulkan for headless recording and respect diagnostic overrides.
    headless = os.environ.get('OMNIUI_HEADLESS','').lower() in ('1','true','yes','on')
    os.environ.setdefault('OMNIUI_BACKEND', 'vulkan' if headless else 'opengl')
    if os.name=='nt' and not _dll_directories:
        # ovui's Windows wheel needs cudart64_12.dll. The pinned ovstream wheel
        # already supplies that runtime; no system-wide CUDA toolkit is needed.
        spec=importlib.util.find_spec('ovstream')
        if spec is None:
            raise RuntimeError('Run tools/setup_libraries.ps1 to install the required libraries.')
        folder=Path(spec.origin).parent
        _dll_directories.append(os.add_dll_directory(str(folder)))
        # NVRTC uses a legacy LoadLibrary lookup for its builtins. Python's
        # add_dll_directory alone does not satisfy it on Windows. Preload the
        # matching library shipped alongside OVRTX's compiler, by absolute path;
        # never search a system CUDA toolkit or alter the user's global PATH.
        rtx=importlib.util.find_spec('ovrtx')
        if rtx is not None:
            import ctypes
            compiler_dir=Path(rtx.origin).parent/'bin/plugins/rtx'
            _dll_directories.append(os.add_dll_directory(str(compiler_dir)))
            for builtins in compiler_dir.glob('nvrtc-builtins64_*.dll'):
                _native_libraries.append(ctypes.WinDLL(str(builtins)))
