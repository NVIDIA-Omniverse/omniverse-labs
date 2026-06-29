# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes wrapper around libnanousdapi.so — minimum surface for the read-only
StageAdapter implementation.

The C API uses opaque handles. `NanousdPrim` handles must be freed with
``nanousd_freeprim``. Stage handles are freed with ``nanousd_close``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DYLIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"


def _locate_lib(stem: str) -> Path:
    """Find a shared library by stem (e.g. 'libnanousdapi'), using the
    platform's native extension."""
    env = os.environ.get("NUSD_GLES_LIB_DIR")
    candidates: list[Path] = []
    libname = f"{stem}{_DYLIB_EXT}"
    if env:
        candidates.append(Path(env) / libname)
    for d in (
        _REPO_ROOT / "build" / "Release",
        _REPO_ROOT / "build",
        Path.cwd() / "build" / "Release",
        Path.cwd() / "build",
    ):
        candidates.append(d / libname)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find {libname}. Build the project first; searched: {candidates}"
    )


_lib_path = _locate_lib("libnanousdapi")
# RTLD_GLOBAL so the implementation lib (libnanousd) finds its own
# symbols once libnanousdapi dlopens it, and so any sibling libraries
# (our libnusd_gles) re-resolve to the same instance.
_core_path = _lib_path.parent / f"libnanousd{_DYLIB_EXT}"
if _core_path.exists():
    ctypes.CDLL(str(_core_path), mode=ctypes.RTLD_GLOBAL)
_lib = ctypes.CDLL(str(_lib_path), mode=ctypes.RTLD_GLOBAL)

# Stage handles
_lib.nanousd_open.restype = ctypes.c_void_p
_lib.nanousd_open.argtypes = [ctypes.c_char_p]

_lib.nanousd_close.restype = None
_lib.nanousd_close.argtypes = [ctypes.c_void_p]

_lib.nanousd_isvalid.restype = ctypes.c_int
_lib.nanousd_isvalid.argtypes = [ctypes.c_void_p]

_lib.nanousd_error.restype = ctypes.c_char_p
_lib.nanousd_error.argtypes = [ctypes.c_void_p]

_lib.nanousd_stage_get_root_layer_path.restype = ctypes.c_char_p
_lib.nanousd_stage_get_root_layer_path.argtypes = [ctypes.c_void_p]

# Prim traversal
_lib.nanousd_nprims.restype = ctypes.c_int
_lib.nanousd_nprims.argtypes = [ctypes.c_void_p]

_lib.nanousd_prim.restype = ctypes.c_void_p
_lib.nanousd_prim.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.nanousd_primpath.restype = ctypes.c_void_p
_lib.nanousd_primpath.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.nanousd_defaultprim.restype = ctypes.c_void_p
_lib.nanousd_defaultprim.argtypes = [ctypes.c_void_p]

_lib.nanousd_nchildren.restype = ctypes.c_int
_lib.nanousd_nchildren.argtypes = [ctypes.c_void_p]

_lib.nanousd_child.restype = ctypes.c_void_p
_lib.nanousd_child.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.nanousd_parent.restype = ctypes.c_void_p
_lib.nanousd_parent.argtypes = [ctypes.c_void_p]

_lib.nanousd_freeprim.restype = None
_lib.nanousd_freeprim.argtypes = [ctypes.c_void_p]

# Prim queries
_lib.nanousd_path.restype = ctypes.c_char_p
_lib.nanousd_path.argtypes = [ctypes.c_void_p]

_lib.nanousd_name.restype = ctypes.c_char_p
_lib.nanousd_name.argtypes = [ctypes.c_void_p]

_lib.nanousd_typename.restype = ctypes.c_char_p
_lib.nanousd_typename.argtypes = [ctypes.c_void_p]

_lib.nanousd_kind.restype = ctypes.c_char_p
_lib.nanousd_kind.argtypes = [ctypes.c_void_p]

_lib.nanousd_isactive.restype = ctypes.c_int
_lib.nanousd_isactive.argtypes = [ctypes.c_void_p]

_lib.nanousd_isdefined.restype = ctypes.c_int
_lib.nanousd_isdefined.argtypes = [ctypes.c_void_p]

_lib.nanousd_isabstract.restype = ctypes.c_int
_lib.nanousd_isabstract.argtypes = [ctypes.c_void_p]

_lib.nanousd_isinstanceable.restype = ctypes.c_int
_lib.nanousd_isinstanceable.argtypes = [ctypes.c_void_p]

_lib.nanousd_isa.restype = ctypes.c_int
_lib.nanousd_isa.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# Attribute basics (used by property adapter, get_value-only)
_lib.nanousd_nattribs.restype = ctypes.c_int
_lib.nanousd_nattribs.argtypes = [ctypes.c_void_p]

_lib.nanousd_attribname.restype = ctypes.c_char_p
_lib.nanousd_attribname.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.nanousd_attribtype.restype = ctypes.c_char_p
_lib.nanousd_attribtype.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.nanousd_hasattrib.restype = ctypes.c_int
_lib.nanousd_hasattrib.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.nanousd_attribf.restype = ctypes.c_float
_lib.nanousd_attribf.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

_lib.nanousd_attribi.restype = ctypes.c_int
_lib.nanousd_attribi.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

_lib.nanousd_attribb.restype = ctypes.c_int
_lib.nanousd_attribb.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

_lib.nanousd_attribs.restype = ctypes.c_char_p
_lib.nanousd_attribs.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]

_lib.nanousd_attribv2f.restype = ctypes.c_int
_lib.nanousd_attribv2f.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float * 2),
]
_lib.nanousd_attribv3f.restype = ctypes.c_int
_lib.nanousd_attribv3f.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float * 3),
]
_lib.nanousd_attribv4f.restype = ctypes.c_int
_lib.nanousd_attribv4f.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float * 4),
]
_lib.nanousd_attribv2d.restype = ctypes.c_int
_lib.nanousd_attribv2d.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double * 2),
]
_lib.nanousd_attribv3d.restype = ctypes.c_int
_lib.nanousd_attribv3d.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double * 3),
]
_lib.nanousd_attribv4d.restype = ctypes.c_int
_lib.nanousd_attribv4d.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double * 4),
]
_lib.nanousd_attribv2i.restype = ctypes.c_int
_lib.nanousd_attribv2i.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int * 2),
]
_lib.nanousd_attribv3i.restype = ctypes.c_int
_lib.nanousd_attribv3i.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int * 3),
]
_lib.nanousd_attribv4i.restype = ctypes.c_int
_lib.nanousd_attribv4i.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int * 4),
]
_lib.nanousd_attribm4d.restype = ctypes.c_int
_lib.nanousd_attribm4d.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double * 16),
]
_lib.nanousd_attribarraylen.restype = ctypes.c_int
_lib.nanousd_attribarraylen.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.nanousd_attribarrayv3f.restype = ctypes.c_int
_lib.nanousd_attribarrayv3f.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
_lib.nanousd_attribarraytokens_len.restype = ctypes.c_int
_lib.nanousd_attribarraytokens_len.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.nanousd_attribarraytokens.restype = ctypes.c_char_p
_lib.nanousd_attribarraytokens.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

# Attribute writes (used by NanousdPropertyAdapter for edits).
_lib.nanousd_set_attribf.restype = ctypes.c_int
_lib.nanousd_set_attribf.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
_lib.nanousd_set_attribd.restype = ctypes.c_int
_lib.nanousd_set_attribd.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
_lib.nanousd_set_attribi.restype = ctypes.c_int
_lib.nanousd_set_attribi.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.nanousd_set_attribb.restype = ctypes.c_int
_lib.nanousd_set_attribb.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.nanousd_set_attribs.restype = ctypes.c_int
_lib.nanousd_set_attribs.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
_lib.nanousd_set_attrib_token.restype = ctypes.c_int
_lib.nanousd_set_attrib_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
for _dim, _ct in ((2, ctypes.c_float), (3, ctypes.c_float), (4, ctypes.c_float),
                  (2, ctypes.c_double), (3, ctypes.c_double), (4, ctypes.c_double),
                  (2, ctypes.c_int), (3, ctypes.c_int), (4, ctypes.c_int)):
    _suf = {ctypes.c_float: "f", ctypes.c_double: "d", ctypes.c_int: "i"}[_ct]
    _fn = getattr(_lib, f"nanousd_set_attribv{_dim}{_suf}")
    _fn.restype = ctypes.c_int
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_ct * _dim)]
del _dim, _ct, _suf, _fn
_lib.nanousd_set_attribm4d.restype = ctypes.c_int
_lib.nanousd_set_attribm4d.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double * 16),
]

_lib.nanousd_create_attrib.restype = ctypes.c_int
_lib.nanousd_create_attrib.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

_lib.nanousd_set_attribarraytokens.restype = ctypes.c_int
_lib.nanousd_set_attribarraytokens.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
]

# Stage write.
_lib.nanousd_write_usda.restype = ctypes.c_int
_lib.nanousd_write_usda.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.nanousd_write_usdc.restype = ctypes.c_int
_lib.nanousd_write_usdc.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# ListOp introspection — used for composition badges (references,
# payloads, inherits, specializes).
_lib.nanousd_prim_listop.restype = ctypes.c_void_p
_lib.nanousd_prim_listop.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.nanousd_listop_nitems.restype = ctypes.c_int
_lib.nanousd_listop_nitems.argtypes = [ctypes.c_void_p]
_lib.nanousd_listop_nprepended.restype = ctypes.c_int
_lib.nanousd_listop_nprepended.argtypes = [ctypes.c_void_p]
_lib.nanousd_listop_nappended.restype = ctypes.c_int
_lib.nanousd_listop_nappended.argtypes = [ctypes.c_void_p]
_lib.nanousd_listop_free.restype = None
_lib.nanousd_listop_free.argtypes = [ctypes.c_void_p]


def _decode(b: Optional[bytes]) -> str:
    if b is None:
        return ""
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return str(b)


class _Prim:
    """Owns a `NanousdPrim` handle; frees it on destruction.

    Equality / hash use the (immutable, owned-by-stage) prim path so the
    Stage Browser can keep the same item across reads and reparenting
    operations even though the underlying handle may have been freed.
    """

    __slots__ = ("_h", "_path")

    def __init__(self, handle: int, path: Optional[str] = None) -> None:
        self._h = ctypes.c_void_p(handle) if handle else None
        if path is None and self._h is not None:
            path = _decode(_lib.nanousd_path(self._h))
        self._path = path or ""

    @property
    def handle(self) -> Optional[ctypes.c_void_p]:
        return self._h

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        if not self._h:
            return ""
        return _decode(_lib.nanousd_name(self._h))

    @property
    def type_name(self) -> str:
        if not self._h:
            return ""
        return _decode(_lib.nanousd_typename(self._h))

    @property
    def kind(self) -> str:
        if not self._h:
            return ""
        return _decode(_lib.nanousd_kind(self._h))

    @property
    def is_active(self) -> bool:
        return bool(self._h and _lib.nanousd_isactive(self._h))

    @property
    def is_abstract(self) -> bool:
        return bool(self._h and _lib.nanousd_isabstract(self._h))

    @property
    def is_instanceable(self) -> bool:
        return bool(self._h and _lib.nanousd_isinstanceable(self._h))

    def is_a(self, schema: str) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_isa(self._h, schema.encode("utf-8")))

    def get_attrib_float(self, name: str, fallback: float = 0.0) -> float:
        if not self._h:
            return fallback
        ok = ctypes.c_int(0)
        v = _lib.nanousd_attribf(self._h, name.encode("utf-8"), ctypes.byref(ok))
        return float(v) if ok.value else fallback

    def get_attrib_int(self, name: str, fallback: int = 0) -> int:
        if not self._h:
            return fallback
        ok = ctypes.c_int(0)
        v = _lib.nanousd_attribi(self._h, name.encode("utf-8"), ctypes.byref(ok))
        return int(v) if ok.value else fallback

    def get_attrib_bool(self, name: str, fallback: bool = False) -> bool:
        if not self._h:
            return fallback
        ok = ctypes.c_int(0)
        v = _lib.nanousd_attribb(self._h, name.encode("utf-8"), ctypes.byref(ok))
        return bool(v) if ok.value else fallback

    def get_attrib_str(self, name: str, fallback: str = "") -> str:
        if not self._h:
            return fallback
        ok = ctypes.c_int(0)
        out = _lib.nanousd_attribs(self._h, name.encode("utf-8"), ctypes.byref(ok))
        return _decode(out) if ok.value else fallback

    def get_attrib_vec3(self, name: str) -> Optional[tuple[float, float, float]]:
        """Read a 3-component vector. Auto-dispatches by attribute type:
        ``float3`` / ``half3`` / ``normal3f`` / ``point3f`` / ``vector3f``
        / ``color3f`` use the float reader; ``double3`` / ``point3d`` /
        ``normal3d`` / ``vector3d`` / ``color3d`` use the double reader.
        Returns ``None`` if no value is authored.
        """
        if not self._h:
            return None
        type_name = self.attrib_type(name).lower()
        # double-typed vec3s — must use v3d, v3f returns zero & ok=0
        if (type_name in ("double3", "point3d", "normal3d", "vector3d", "color3d")
                or "double3" in type_name):
            buf_d = (ctypes.c_double * 3)()
            if not _lib.nanousd_attribv3d(self._h, name.encode("utf-8"), buf_d):
                return None
            return (float(buf_d[0]), float(buf_d[1]), float(buf_d[2]))
        # default: float
        buf = (ctypes.c_float * 3)()
        if not _lib.nanousd_attribv3f(self._h, name.encode("utf-8"), buf):
            return None
        return (float(buf[0]), float(buf[1]), float(buf[2]))

    def get_attrib_vec(self, name: str, dim: int, dtype: str) -> Optional[tuple]:
        """Generic vector accessor.

        ``dim`` is 2/3/4; ``dtype`` is ``f`` (float32), ``d`` (float64),
        or ``i`` (int32). Returns a tuple or None on miss.
        """
        if not self._h:
            return None
        if dtype == "f":
            ct = ctypes.c_float
            fn = {2: _lib.nanousd_attribv2f, 3: _lib.nanousd_attribv3f, 4: _lib.nanousd_attribv4f}[dim]
        elif dtype == "d":
            ct = ctypes.c_double
            fn = {2: _lib.nanousd_attribv2d, 3: _lib.nanousd_attribv3d, 4: _lib.nanousd_attribv4d}[dim]
        elif dtype == "i":
            ct = ctypes.c_int
            fn = {2: _lib.nanousd_attribv2i, 3: _lib.nanousd_attribv3i, 4: _lib.nanousd_attribv4i}[dim]
        else:
            return None
        buf = (ct * dim)()
        if not fn(self._h, name.encode("utf-8"), buf):
            return None
        py = float if dtype != "i" else int
        return tuple(py(buf[i]) for i in range(dim))

    def get_attrib_matrix4d(self, name: str) -> Optional[tuple]:
        if not self._h:
            return None
        buf = (ctypes.c_double * 16)()
        if not _lib.nanousd_attribm4d(self._h, name.encode("utf-8"), buf):
            return None
        return tuple(float(buf[i]) for i in range(16))

    def get_attrib_array_len(self, name: str) -> int:
        if not self._h:
            return 0
        return int(_lib.nanousd_attribarraylen(self._h, name.encode("utf-8")))

    def get_attrib_array_tokens(self, name: str) -> List[str]:
        """Read a ``token[]`` attribute (e.g. ``xformOpOrder``)."""
        if not self._h:
            return []
        n = int(_lib.nanousd_attribarraytokens_len(self._h, name.encode("utf-8")))
        if n <= 0:
            return []
        out: List[str] = []
        for i in range(n):
            ptr = _lib.nanousd_attribarraytokens(self._h, name.encode("utf-8"), i)
            out.append(_decode(ptr))
        return out

    def get_attrib_arrayv3f(self, name: str, maxcount: Optional[int] = None) -> List[tuple]:
        """Read an array-of-float3 attribute, e.g. ``extent`` (length 2)
        or ``points`` (length N). Returns a list of (x, y, z) tuples.
        """
        if not self._h:
            return []
        n = self.get_attrib_array_len(name)
        if n <= 0:
            return []
        if maxcount is not None:
            n = min(n, maxcount)
        buf = (ctypes.c_float * (n * 3))()
        got = int(_lib.nanousd_attribarrayv3f(
            self._h, name.encode("utf-8"), buf, n))
        if got <= 0:
            return []
        return [
            (float(buf[i * 3]), float(buf[i * 3 + 1]), float(buf[i * 3 + 2]))
            for i in range(got)
        ]

    # ── Writes ─────────────────────────────────────────────────────

    def set_attrib_float(self, name: str, value: float) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attribf(self._h, name.encode("utf-8"), float(value)))

    def set_attrib_double(self, name: str, value: float) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attribd(self._h, name.encode("utf-8"), float(value)))

    def set_attrib_int(self, name: str, value: int) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attribi(self._h, name.encode("utf-8"), int(value)))

    def set_attrib_bool(self, name: str, value: bool) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attribb(self._h, name.encode("utf-8"), 1 if value else 0))

    def set_attrib_str(self, name: str, value: str) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attribs(
            self._h, name.encode("utf-8"), str(value).encode("utf-8")))

    def set_attrib_token(self, name: str, value: str) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_set_attrib_token(
            self._h, name.encode("utf-8"), str(value).encode("utf-8")))

    def set_attrib_vec(self, name: str, dim: int, dtype: str, values) -> bool:
        if not self._h:
            return False
        if dtype == "f":
            ct = ctypes.c_float
            fn = {2: _lib.nanousd_set_attribv2f, 3: _lib.nanousd_set_attribv3f,
                  4: _lib.nanousd_set_attribv4f}[dim]
        elif dtype == "d":
            ct = ctypes.c_double
            fn = {2: _lib.nanousd_set_attribv2d, 3: _lib.nanousd_set_attribv3d,
                  4: _lib.nanousd_set_attribv4d}[dim]
        elif dtype == "i":
            ct = ctypes.c_int
            fn = {2: _lib.nanousd_set_attribv2i, 3: _lib.nanousd_set_attribv3i,
                  4: _lib.nanousd_set_attribv4i}[dim]
        else:
            return False
        buf = (ct * dim)(*[ct(v) for v in list(values)[:dim]])
        return bool(fn(self._h, name.encode("utf-8"), buf))

    def create_attrib(self, name: str, type_name: str) -> bool:
        """Define a new attribute on this prim. No-op if it already
        exists. Returns 1 on success.
        """
        if not self._h:
            return False
        return bool(_lib.nanousd_create_attrib(
            self._h, name.encode("utf-8"), type_name.encode("utf-8")))

    def set_attrib_array_tokens(self, name: str, tokens: List[str]) -> bool:
        if not self._h:
            return False
        encoded = [t.encode("utf-8") for t in tokens]
        arr = (ctypes.c_char_p * len(encoded))(*encoded)
        return bool(_lib.nanousd_set_attribarraytokens(
            self._h, name.encode("utf-8"), arr, len(encoded)))

    def set_attrib_matrix4d(self, name: str, values) -> bool:
        if not self._h:
            return False
        flat = []
        for row in values:
            try:
                flat.extend(row)
            except TypeError:
                flat.append(row)
        if len(flat) != 16:
            return False
        buf = (ctypes.c_double * 16)(*[ctypes.c_double(v) for v in flat])
        return bool(_lib.nanousd_set_attribm4d(self._h, name.encode("utf-8"), buf))

    def has_listop(self, field: str) -> bool:
        """True if the named composition list-op has any items.

        ``field`` is one of ``"references"``, ``"payloads"``,
        ``"inheritPaths"``, ``"specializes"``, etc.
        """
        if not self._h:
            return False
        op = _lib.nanousd_prim_listop(self._h, field.encode("utf-8"))
        if not op:
            return False
        try:
            n = int(_lib.nanousd_listop_nitems(op))
            n += int(_lib.nanousd_listop_nprepended(op))
            n += int(_lib.nanousd_listop_nappended(op))
            return n > 0
        finally:
            _lib.nanousd_listop_free(op)

    def attrib_names(self) -> List[str]:
        if not self._h:
            return []
        n = int(_lib.nanousd_nattribs(self._h))
        return [_decode(_lib.nanousd_attribname(self._h, i)) for i in range(n)]

    def has_attrib(self, name: str) -> bool:
        if not self._h:
            return False
        return bool(_lib.nanousd_hasattrib(self._h, name.encode("utf-8")))

    def attrib_type(self, name: str) -> str:
        if not self._h:
            return ""
        return _decode(_lib.nanousd_attribtype(self._h, name.encode("utf-8")))

    def children(self) -> Iterator["_Prim"]:
        if not self._h:
            return
        n = int(_lib.nanousd_nchildren(self._h))
        for i in range(n):
            ch = _lib.nanousd_child(self._h, i)
            if ch:
                yield _Prim(int(ch))

    def n_children(self) -> int:
        if not self._h:
            return 0
        return int(_lib.nanousd_nchildren(self._h))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Prim) and self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    def __repr__(self) -> str:
        return f"<NanousdPrim {self._path!r} ({self.type_name})>"

    def __del__(self) -> None:  # pragma: no cover
        try:
            if self._h:
                _lib.nanousd_freeprim(self._h)
                self._h = None
        except Exception:
            pass


class NanousdStage:
    """A nanousd-loaded USD stage.

    Owns the C handle; closes on context exit / destruction. The stage
    itself owns prim ownership, but each call that returns a NanousdPrim
    transfers ownership to the caller (free with nanousd_freeprim — the
    `_Prim` wrapper handles that automatically).
    """

    def __init__(self, path: str) -> None:
        h = _lib.nanousd_open(path.encode("utf-8"))
        if not h:
            raise RuntimeError(f"nanousd_open failed for {path!r}")
        self._h: Optional[ctypes.c_void_p] = ctypes.c_void_p(h)
        if not _lib.nanousd_isvalid(self._h):
            err = _decode(_lib.nanousd_error(self._h))
            _lib.nanousd_close(self._h)
            self._h = None
            raise RuntimeError(f"nanousd stage invalid: {err}")
        self._path = path

    @property
    def filepath(self) -> str:
        return self._path

    @property
    def root_layer_path(self) -> str:
        if not self._h:
            return ""
        return _decode(_lib.nanousd_stage_get_root_layer_path(self._h))

    def n_prims(self) -> int:
        if not self._h:
            return 0
        return int(_lib.nanousd_nprims(self._h))

    def prim_at_index(self, i: int) -> Optional[_Prim]:
        if not self._h:
            return None
        h = _lib.nanousd_prim(self._h, int(i))
        return _Prim(int(h)) if h else None

    def prim_at_path(self, path: str) -> Optional[_Prim]:
        if not self._h:
            return None
        h = _lib.nanousd_primpath(self._h, path.encode("utf-8"))
        return _Prim(int(h)) if h else None

    def default_prim(self) -> Optional[_Prim]:
        if not self._h:
            return None
        h = _lib.nanousd_defaultprim(self._h)
        return _Prim(int(h)) if h else None

    def root_children(self) -> List[_Prim]:
        """Top-level prims in the stage.

        nanousd_nprims/nanousd_prim enumerate ALL prims (depth-first), so
        we walk and collect those whose path has exactly one slash —
        i.e. ``/Foo`` but not ``/Foo/Bar``.
        """
        if not self._h:
            return []
        out: List[_Prim] = []
        for i in range(self.n_prims()):
            p = self.prim_at_index(i)
            if p is None:
                continue
            path = p.path
            if path.count("/") == 1 and path != "/":
                out.append(p)
        return out

    def write(self, path: str) -> bool:
        """Write the stage to ``path``. Format chosen from extension —
        ``.usdc`` writes binary, anything else writes USDA. Returns
        True on success.
        """
        if not self._h:
            return False
        path_b = str(path).encode("utf-8")
        if str(path).lower().endswith(".usdc"):
            return bool(_lib.nanousd_write_usdc(self._h, path_b))
        return bool(_lib.nanousd_write_usda(self._h, path_b))

    def close(self) -> None:
        if self._h:
            _lib.nanousd_close(self._h)
            self._h = None

    def __enter__(self) -> "NanousdStage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
