# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes wrapper around libnusd_gles.so.

Drives the headless OpenGL renderer from Python: load a USD file, render
a frame at an explicit camera, get RGBA bytes back. Used by
``GlesRendererAdapter`` to satisfy ovgear's ``RendererAdapter`` ABC.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LIB_DIRS = [
    _REPO_ROOT / "build",
    Path.cwd() / "build",
]
_DYLIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"


def _locate(stem: str) -> Path:
    """Find a shared library by stem (e.g. 'libnusd_gles'), trying the
    platform's native extension."""
    env = os.environ.get("NUSD_GLES_LIB_DIR")
    candidates: list[Path] = []
    libname = f"{stem}{_DYLIB_EXT}"
    if env:
        candidates.append(Path(env) / libname)
    for d in _DEFAULT_LIB_DIRS:
        candidates.append(d / libname)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {libname}. Build the project (cmake --build build) "
        f"or set NUSD_GLES_LIB_DIR. Searched: {candidates}"
    )


_lib_path = _locate("libnusd_gles")
# Preload nanousdapi too so the viewer's symbol references resolve when
# the ctypes loader walks RUNPATHs that don't include the build dir.
_nanousdapi_dir = (_lib_path.parent / "Release")
_nanousd_dylib = _nanousdapi_dir / f"libnanousd{_DYLIB_EXT}"
_nanousdapi_dylib = _nanousdapi_dir / f"libnanousdapi{_DYLIB_EXT}"
if _nanousdapi_dylib.exists():
    ctypes.CDLL(str(_nanousd_dylib), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(_nanousdapi_dylib), mode=ctypes.RTLD_GLOBAL)
_lib = ctypes.CDLL(str(_lib_path))


# ── viewer.h C signatures (threaded variants from gl_thread.h) ──
# All viewer GL work runs on a dedicated worker thread inside the
# library so we never disturb the embedder's GL context.

_lib.nusd_gl_warmup.restype = ctypes.c_int
_lib.nusd_gl_warmup.argtypes = [ctypes.c_int, ctypes.c_int]


def warmup(width: int = 1280, height: int = 720) -> bool:
    """Pre-initialize the worker thread's EGL context.

    Must be called BEFORE the embedder loads Vulkan (e.g. before
    ``omni.ui.init()``); NVIDIA's driver refuses fresh EGL setup once
    Vulkan has touched the same GPU. Returns True on success.
    """
    return bool(_lib.nusd_gl_warmup(int(width), int(height)))


_lib.nusd_gl_viewer_create.restype = ctypes.c_void_p
_lib.nusd_gl_viewer_create.argtypes = [
    ctypes.c_char_p,  # path
    ctypes.c_int,     # width
    ctypes.c_int,     # height
    ctypes.c_int,     # max_tex_size (0 = default)
    ctypes.c_char_p,  # envmap path (NULL = auto-discover)
    ctypes.c_int,     # headless
]

_lib.nusd_gl_viewer_destroy.restype = None
_lib.nusd_gl_viewer_destroy.argtypes = [ctypes.c_void_p]

_lib.nusd_gl_viewer_resize.restype = None
_lib.nusd_gl_viewer_resize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

_lib.nusd_gl_viewer_set_render_mode.restype = None
_lib.nusd_gl_viewer_set_render_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.nusd_gl_viewer_set_overlay_enabled.restype = None
_lib.nusd_gl_viewer_set_overlay_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.viewer_get_scene_bounds.restype = ctypes.c_int
_lib.viewer_get_scene_bounds.argtypes = [
    ctypes.c_void_p,                # viewer*
    ctypes.POINTER(ctypes.c_float * 3),  # out_min
    ctypes.POINTER(ctypes.c_float * 3),  # out_max
]

_lib.nusd_gl_viewer_render_to_rgba.restype = ctypes.c_int
_lib.nusd_gl_viewer_render_to_rgba.argtypes = [
    ctypes.c_void_p,                    # viewer*
    ctypes.c_int, ctypes.c_int,         # width, height
    ctypes.POINTER(ctypes.c_float),     # view16 (column-major)
    ctypes.POINTER(ctypes.c_float),     # proj16 (column-major)
    ctypes.POINTER(ctypes.c_float),     # eye3
    ctypes.POINTER(ctypes.c_ubyte),     # out_rgba
]


class GlesViewer:
    """Headless OpenGL ES nanoUSD renderer.

    Constructed once a USD file path is known. Subsequent calls to
    :meth:`render` re-render with the supplied camera matrices and write
    pixels into the caller-managed numpy buffer.
    """

    MODE_RASTER = 0
    MODE_MATERIAL = 1

    def __init__(
        self,
        usd_path: str,
        width: int = 1280,
        height: int = 720,
        max_tex_size: int = 0,
        envmap_path: Optional[str] = None,
    ) -> None:
        path_b = str(usd_path).encode("utf-8")
        env_b = envmap_path.encode("utf-8") if envmap_path else None
        handle = _lib.nusd_gl_viewer_create(
            path_b, int(width), int(height), int(max_tex_size), env_b, 1
        )
        if not handle:
            raise RuntimeError(f"viewer_create failed for {usd_path!r}")
        self._handle = ctypes.c_void_p(handle)
        self.width = int(width)
        self.height = int(height)
        # On-render HUD disabled — the text/rect overlay style doesn't
        # blend well against ovgear's image-bridge composition. Code is
        # preserved in viewer.c for a later revisit.
        _lib.nusd_gl_viewer_set_overlay_enabled(self._handle, 0)
        # Default to material mode so PBR is on by default — ovgear users
        # expect a lit, textured viewport.
        _lib.nusd_gl_viewer_set_render_mode(self._handle, self.MODE_MATERIAL)

    def set_render_mode(self, mode: int) -> None:
        _lib.nusd_gl_viewer_set_render_mode(self._handle, int(mode))

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if width == self.width and height == self.height:
            return
        _lib.nusd_gl_viewer_resize(self._handle, int(width), int(height))
        self.width, self.height = int(width), int(height)

    def render(
        self,
        width: int,
        height: int,
        view: np.ndarray,
        proj: np.ndarray,
        eye: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render one frame, return ``(H, W, 4)`` uint8 RGBA top-down.

        ``view``/``proj`` are 4×4 column-major (GL convention). ``eye`` is
        the camera origin in world space. If ``out`` is provided it must
        be a contiguous ``(height, width, 4)`` uint8 buffer; otherwise a
        fresh array is allocated.
        """
        w, h = int(width), int(height)
        view_f = np.ascontiguousarray(view, dtype=np.float32).reshape(-1)
        proj_f = np.ascontiguousarray(proj, dtype=np.float32).reshape(-1)
        eye_f = np.ascontiguousarray(eye, dtype=np.float32).reshape(-1)
        if view_f.size != 16 or proj_f.size != 16 or eye_f.size != 3:
            raise ValueError("view/proj must be 4×4 and eye must be 3-vector")

        if out is None:
            out = np.empty((h, w, 4), dtype=np.uint8)
        elif out.shape != (h, w, 4) or out.dtype != np.uint8:
            raise ValueError(f"out must be ({h},{w},4) uint8, got {out.shape} {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            out = np.ascontiguousarray(out)

        ok = _lib.nusd_gl_viewer_render_to_rgba(
            self._handle, w, h,
            view_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            proj_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            eye_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )
        if not ok:
            out.fill(0)
        # Track the current size for free resizes.
        self.width, self.height = w, h
        return out

    def get_scene_bounds(self) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Return the loaded scene's world-space AABB as ``(min, max)``.

        Returns ``None`` if no scene is loaded. Used by embedders to
        compute camera-framing distances.
        """
        if not getattr(self, "_handle", None):
            return None
        mn = (ctypes.c_float * 3)()
        mx = (ctypes.c_float * 3)()
        if not _lib.viewer_get_scene_bounds(self._handle, mn, mx):
            return None
        return ((float(mn[0]), float(mn[1]), float(mn[2])),
                (float(mx[0]), float(mx[1]), float(mx[2])))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.nusd_gl_viewer_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
