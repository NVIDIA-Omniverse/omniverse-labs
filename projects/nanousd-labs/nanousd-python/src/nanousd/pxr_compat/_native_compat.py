# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Thin pxr compatibility layer backed by the nanousd nanobind API.

Provides ``Usd``, ``UsdGeom``, ``UsdPhysics``, ``UsdShade``, ``Gf``, ``Sdf``,
and ``Vt`` namespaces that mirror the subset of pxr APIs used by nanousdview,
implemented through the ``nanousd._nanousd`` extension.

Usage::

    from nanousd.pxr_compat import Usd, UsdGeom, UsdPhysics, Gf, Sdf

Set ``NANOUSD_LIB`` to the full path of the shared library if it cannot be
found automatically. ``NANOUSD_LIB_DIR``, ``NANOUSD_DIR``, and
``NANOUSD_WORKSPACE`` are also honored. ``AIUSD_LIB_PATH`` is accepted as a
legacy alias.
"""

from __future__ import annotations

import math
import os
import sys
import weakref
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from .. import _nanousd as _native


# ===========================================================================
# String encoding helpers
# ===========================================================================


def _enc(s: str | bytes | None) -> bytes:
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8") if s else b""


def _dec(b: bytes | None) -> str:
    if b is None:
        return ""
    if isinstance(b, bytes):
        return b.decode("utf-8")
    return str(b)


def _find_usda_prim_block(usda: str, prim_path: str) -> str | None:
    """Return the USDA text block for a prim path using a lightweight brace scan."""
    import re

    target = str(prim_path)
    if not target or target == "/":
        return usda

    stack: list[str] = []
    pending_name: str | None = None
    collecting = False
    collect_depth = 0
    collected: list[str] = []
    prim_decl = re.compile(r'^\s*(?:def|over|class)\s+(?:[A-Za-z_][\w:]*\s+)?"([^"]+)"')

    for line in usda.splitlines():
        match = prim_decl.match(line)
        if match:
            pending_name = match.group(1)

        opens = line.count("{")
        closes = line.count("}")
        pushed = False
        if opens and pending_name is not None:
            stack.append(pending_name)
            pushed = True
            pending_name = None
            current_path = "/" + "/".join(stack)
            if current_path == target:
                collecting = True
                collect_depth = 0

        if collecting:
            collected.append(line)
            collect_depth += opens - closes
            if collect_depth <= 0:
                return "\n".join(collected)

        for _ in range(closes):
            if stack:
                stack.pop()
        if opens and not pushed:
            pending_name = None

    return None


def _find_usda_prim_body_by_name(usda: str, prim_path: str) -> str | None:
    """Return a prim body by scanning past declaration metadata.

    This is a fallback for editor metadata. It intentionally keys off the final
    prim name rather than trying to implement a full USDA parser.
    """
    import re

    target = str(prim_path)
    if not target or target == "/":
        return usda
    name = target.rstrip("/").rsplit("/", 1)[-1]
    if not name:
        return None
    decl = re.compile(
        r'(?ms)^\s*(?:def|over|class)\s+'
        r'(?:[A-Za-z_][\w:]*\s+)?"' + re.escape(name) + r'"'
    )
    match = decl.search(usda)
    if not match:
        return None

    paren_depth = 0
    body_start = -1
    for idx in range(match.end(), len(usda)):
        ch = usda[idx]
        if ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "{" and paren_depth == 0:
            body_start = idx + 1
            break
    if body_start < 0:
        return None

    depth = 1
    for idx in range(body_start, len(usda)):
        ch = usda[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return usda[body_start:idx]
    return None


def _find_usda_layer_metadata(usda: str, key: str) -> str | None:
    """Return a root-layer metadata string/token from a USDA header block."""
    import re

    header = re.search(r"#usda\s+1\.0\s*\((.*?)\)", usda, re.S)
    if not header:
        return None
    match = re.search(
        rf"^\s*{re.escape(str(key))}\s*=\s*\"([^\"]*)\"",
        header.group(1),
        re.M,
    )
    return match.group(1) if match else None


def _parse_usda_custom_data_dict(text: str) -> dict[str, Any]:
    import re

    out: dict[str, Any] = {}
    scalar_pattern = re.compile(
        r"^\s*(string|token|int|bool|float|double)\s+([A-Za-z_]\w*)\s*=\s*(\"[^\"]*\"|[^\s}]+)"
    )
    dictionary_pattern = re.compile(
        r"^\s*dictionary\s+([A-Za-z_]\w*)\s*=\s*\{"
    )

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        dict_match = dictionary_pattern.match(line)
        if dict_match:
            key = dict_match.group(1)
            depth = line.count("{") - line.count("}")
            nested_lines: list[str] = []
            i += 1
            while i < len(lines) and depth > 0:
                nested = lines[i]
                depth += nested.count("{") - nested.count("}")
                if depth >= 0 and nested.strip() != "}":
                    nested_lines.append(nested)
                i += 1
            out[key] = _parse_usda_custom_data_dict("\n".join(nested_lines))
            continue

        scalar_match = scalar_pattern.match(line)
        if not scalar_match:
            i += 1
            continue
        type_name, key, raw_value = scalar_match.groups()
        if raw_value.startswith('"') and raw_value.endswith('"'):
            value: Any = raw_value[1:-1]
        elif type_name == "bool":
            value = raw_value.lower() in {"1", "true"}
        elif type_name == "int":
            value = int(raw_value)
        elif type_name in {"float", "double"}:
            value = float(raw_value)
        else:
            value = raw_value
        out[key] = value
        i += 1
    return out


def _find_usda_attribute_custom_data(usda: str, prim_path: str, attr_name: str) -> dict[str, Any] | None:
    import re

    block = _find_usda_prim_block(usda, prim_path)
    if not block:
        return None

    attr_re = re.compile(
        rf"^\s*(?:custom\s+)?[A-Za-z_][\w:<>,\[\]]*\s+{re.escape(attr_name)}\b"
    )
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if not attr_re.match(line):
            continue
        window = "\n".join(lines[index : index + 80])
        match = re.search(r"customData\s*=\s*\{(?P<body>.*?)\}", window, re.S)
        if match:
            return _parse_usda_custom_data_dict(match.group("body"))
        return None
    return None


def _stage_source_usda(stage: _MuStage) -> str | None:
    cached = getattr(stage, "_usda_source_cache", None)
    if cached is not None:
        return cached

    text: str | None = None
    root_path = str(getattr(getattr(stage, "_h", None), "root_layer_path", "") or "")
    if root_path:
        try:
            root = Path(root_path)
            with root.open("rb") as fh:
                prefix = fh.read(16)
            if prefix.startswith(b"#usda"):
                text = root.read_text(encoding="utf-8")
        except OSError:
            text = None

    if text is None:
        try:
            text = stage.write_usda_string()
        except Exception:
            text = None

    setattr(stage, "_usda_source_cache", text)
    return text


def _stage_composed_usda(stage: _MuStage | None) -> str | None:
    if stage is None:
        return None
    try:
        return stage.write_usda_string()
    except Exception:
        return None


def _get_usda_attribute_custom_data(stage: _MuStage | None, prim_path: str, attr_name: str) -> dict[str, Any] | None:
    if stage is None:
        return None
    cache = getattr(stage, "_attr_custom_data_cache", None)
    if cache is None:
        cache = {}
        setattr(stage, "_attr_custom_data_cache", cache)
    key = (str(prim_path), str(attr_name))
    if key in cache:
        return cache[key]
    usda = _stage_source_usda(stage)
    value = _find_usda_attribute_custom_data(usda, key[0], key[1]) if usda else None
    cache[key] = value
    return value


_PY_ATTR_VALUES: dict[tuple[int, str], Any] = {}
_NUMERIC_STAGE_METADATA_KEYS = frozenset(
    {
        "metersPerUnit",
        "kilogramsPerUnit",
        "timeCodesPerSecond",
        "framesPerSecond",
        "startTimeCode",
        "endTimeCode",
    }
)
_STRING_STAGE_METADATA_FALLBACK_KEYS = frozenset(
    {"upAxis", "defaultPrim", "documentation", "comment"}
)


def _prim_value_key(prim: _MuPrim, name: str) -> tuple[int, str] | None:
    handle = getattr(prim, "_h", None)
    if not handle:
        return None
    return (id(handle), name)


def _coerce_python_attr_value(value: Any, type_str: str) -> Any:
    if type_str.endswith("[]"):
        base = type_str[:-2]
        if base in ("int2", "int3", "int4"):
            comps = int(base[-1])
            return np.asarray(value, dtype=np.int32).reshape(-1, comps)
        if base in _VEC3F_ARRAY_BASES:
            return np.asarray(value, dtype=np.float32).reshape(-1, 3)
        if base in _VEC3D_ARRAY_BASES:
            return np.asarray(value, dtype=np.float64).reshape(-1, 3)
        if base in ("int",):
            return np.asarray(value, dtype=np.int32)
        if base in ("double",):
            return np.asarray(value, dtype=np.float64)
        if base in ("float", "half"):
            return np.asarray(value, dtype=np.float32)
    if type_str in _VEC3F_TYPES:
        return np.asarray(value, dtype=np.float32).reshape(3)
    if type_str in _VEC3D_TYPES:
        return np.asarray(value, dtype=np.float64).reshape(3)
    return value


# ===========================================================================
# _MuPrim — compatibility wrapper around the nanobind Prim object
# ===========================================================================

# Type string sets for dispatch
_VEC2F_TYPES = frozenset(("float2", "texCoord2f"))
_VEC3F_TYPES = frozenset(("float3", "point3f", "normal3f", "color3f", "vector3f"))
_VEC4F_TYPES = frozenset(("float4", "color4f"))
_VEC2D_TYPES = frozenset(("double2", "texCoord2d"))
_VEC3D_TYPES = frozenset(("double3", "point3d", "normal3d", "color3d", "vector3d"))
_VEC4D_TYPES = frozenset(("double4", "color4d"))
_VEC3F_ARRAY_BASES = frozenset(("point3f", "normal3f", "color3f", "vector3f", "float3"))
_VEC3D_ARRAY_BASES = frozenset(("point3d", "normal3d", "color3d", "vector3d", "double3"))


def _build_attribute_namespace_map(names: list[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        start = 0
        while True:
            pos = name.find(":", start)
            if pos < 0:
                break
            grouped.setdefault(name[:pos + 1], []).append(name)
            start = pos + 1
    return {prefix: tuple(values) for prefix, values in grouped.items()}


def _install_newton_runtime_accelerators() -> None:
    """Patch hot Newton USD helpers to use nanousd native kernels when present."""
    module = sys.modules.get("newton._src.usd.utils")
    if module is None:
        return
    original_fan = getattr(module, "fan_triangulate_faces", None)
    native_fan = getattr(_native, "fan_triangulate_indices", None)

    if (
        original_fan is not None
        and native_fan is not None
        and not getattr(original_fan, "_nanousd_accelerated", False)
    ):
        def _fan_triangulate_faces(counts: Any, indices: Any) -> np.ndarray:
            counts_arr = np.asarray(counts, dtype=np.int32)
            indices_arr = np.asarray(indices, dtype=np.int32)
            try:
                return native_fan(
                    np.ascontiguousarray(counts_arr),
                    np.ascontiguousarray(indices_arr),
                )
            except (RuntimeError, TypeError, ValueError):
                return original_fan(counts_arr, indices_arr)

        _fan_triangulate_faces.__name__ = getattr(original_fan, "__name__", "fan_triangulate_faces")
        _fan_triangulate_faces.__doc__ = getattr(original_fan, "__doc__", None)
        _fan_triangulate_faces._nanousd_accelerated = True
        _fan_triangulate_faces._nanousd_original = original_fan
        module.fan_triangulate_faces = _fan_triangulate_faces

    original_attrs = getattr(module, "get_attributes_in_namespace", None)
    if original_attrs is not None and not getattr(original_attrs, "_nanousd_accelerated", False):
        def _get_attributes_in_namespace(prim: Any, namespace: str) -> dict[str, Any]:
            getter = getattr(prim, "GetAuthoredAttributeValuesInNamespace", None)
            if getter is not None:
                try:
                    return getter(namespace)
                except (RuntimeError, TypeError, ValueError):
                    pass
            return original_attrs(prim, namespace)

        _get_attributes_in_namespace.__name__ = getattr(
            original_attrs, "__name__", "get_attributes_in_namespace"
        )
        _get_attributes_in_namespace.__doc__ = getattr(original_attrs, "__doc__", None)
        _get_attributes_in_namespace._nanousd_accelerated = True
        _get_attributes_in_namespace._nanousd_original = original_attrs
        module.get_attributes_in_namespace = _get_attributes_in_namespace

    schema_module = sys.modules.get("newton._src.usd.schema_resolver")
    if schema_module is None:
        return

    original_by_name = getattr(schema_module, "_collect_attrs_by_name", None)
    if original_by_name is not None and not getattr(original_by_name, "_nanousd_accelerated", False):
        def _collect_attrs_by_name(prim: Any, names: Sequence[str]) -> dict[str, Any]:
            getter = getattr(prim, "GetAuthoredAttributeValues", None)
            if getter is not None:
                try:
                    return getter(names)
                except (RuntimeError, TypeError, ValueError):
                    pass
            return original_by_name(prim, names)

        _collect_attrs_by_name.__name__ = getattr(original_by_name, "__name__", "_collect_attrs_by_name")
        _collect_attrs_by_name.__doc__ = getattr(original_by_name, "__doc__", None)
        _collect_attrs_by_name._nanousd_accelerated = True
        _collect_attrs_by_name._nanousd_original = original_by_name
        schema_module._collect_attrs_by_name = _collect_attrs_by_name

    original_by_namespace = getattr(schema_module, "_collect_attrs_by_namespace", None)
    if original_by_namespace is None or getattr(original_by_namespace, "_nanousd_accelerated", False):
        return

    def _collect_attrs_by_namespace(prim: Any, namespaces: Sequence[str]) -> dict[str, Any]:
        getter = getattr(prim, "GetAuthoredAttributeValuesInNamespaces", None)
        if getter is not None:
            try:
                return getter(namespaces)
            except (RuntimeError, TypeError, ValueError):
                pass
        return original_by_namespace(prim, namespaces)

    _collect_attrs_by_namespace.__name__ = getattr(
        original_by_namespace, "__name__", "_collect_attrs_by_namespace"
    )
    _collect_attrs_by_namespace.__doc__ = getattr(original_by_namespace, "__doc__", None)
    _collect_attrs_by_namespace._nanousd_accelerated = True
    _collect_attrs_by_namespace._nanousd_original = original_by_namespace
    schema_module._collect_attrs_by_namespace = _collect_attrs_by_namespace


class _MuPrim:
    """Wrapper around nanousd._nanousd.Prim providing the nanobind interface."""

    __slots__ = (
        "_h",
        "_owned",
        "_stage_h",
        "_attr_names_cache",
        "_authored_attr_names_cache",
        "_attr_namespace_cache",
        "_attr_namespace_map_cache",
        "_path_cache",
    )

    def __init__(self, handle: Any, owned: bool = False, stage_h: Any = None):
        self._h = handle
        self._owned = owned
        self._stage_h = stage_h
        self._attr_names_cache: list[str] | None = None
        self._authored_attr_names_cache: list[str] | None = None
        self._attr_namespace_cache: dict[str, list[str]] = {}
        self._attr_namespace_map_cache: dict[str, tuple[str, ...]] | None = None
        self._path_cache: str | None = None

    def __del__(self) -> None:
        self._h = None

    @property
    def is_valid(self) -> bool:
        return bool(self._h) and bool(getattr(self._h, "is_valid", False))

    @property
    def path(self) -> str:
        if not self._h:
            return ""
        if self._path_cache is None:
            self._path_cache = str(self._h.path)
        return self._path_cache

    @property
    def name(self) -> str:
        if not self._h:
            return ""
        return str(self._h.name)

    @property
    def type_name(self) -> str:
        if not self._h:
            return ""
        return str(self._h.type_name)

    @property
    def is_active(self) -> bool:
        return bool(self._h and self._h.is_active)

    @property
    def is_instance(self) -> bool:
        return bool(self._h and getattr(self._h, "is_instance", False))

    @property
    def is_in_prototype(self) -> bool:
        return bool(self._h and getattr(self._h, "is_in_prototype", False))

    def is_a(self, type_name: str) -> bool:
        return bool(self._h and self._h.is_a(type_name))

    def has_api(self, api_name: str) -> bool:
        return bool(self._h and self._h.has_api(api_name))

    def _invalidate_attr_caches(self) -> None:
        self._attr_names_cache = None
        self._authored_attr_names_cache = None
        self._attr_namespace_cache.clear()
        self._attr_namespace_map_cache = None
        if self._stage_h is not None:
            self._stage_h.invalidate_prim_attribute_cache(self.path)

    def _mark_attribute_authored(self, name: str, type_name: str | None = None) -> None:
        if self._stage_h is None:
            return
        key = (self.path, str(name))
        self._stage_h._attr_has_cache[key] = True
        self._stage_h._attr_authored_cache[key] = True
        if type_name:
            self._stage_h._attr_type_cache[key] = str(type_name)

    def has_attribute(self, name: str) -> bool:
        if not self._h:
            return False
        stage = self._stage_h
        if stage is not None:
            key = (self.path, str(name))
            cached = stage._attr_has_cache.get(key)
            if cached is not None:
                return cached
            value = bool(self._h.has_attribute(name))
            stage._attr_has_cache[key] = value
            return value
        return bool(self._h.has_attribute(name))

    def get_attribute_type(self, name: str) -> str:
        if not self._h:
            return ""
        stage = self._stage_h
        if stage is not None:
            key = (self.path, str(name))
            cached = stage._attr_type_cache.get(key)
            if cached is not None:
                return cached
            value = str(self._h.attribute_type(name))
            stage._attr_type_cache[key] = value
            return value
        return str(self._h.attribute_type(name))

    def get_attribute(self, name: str, type_str: str | None = None) -> Any:
        type_str = type_str or self.get_attribute_type(name)
        if not type_str:
            return None
        return self._read_typed_value(name, type_str)

    def _read_typed_value(self, name: str, type_str: str) -> Any:
        if type_str in ("float", "half"):
            v = self._h.read_float(name)
            return float(v) if v is not None else None
        if type_str == "double":
            v = self._h.read_double(name)
            return float(v) if v is not None else None
        if type_str == "int":
            v = self._h.read_int(name)
            return int(v) if v is not None else None
        if type_str in ("int64", "uint64"):
            v = self._h.read_int64(name)
            return int(v) if v is not None else None
        if type_str == "bool":
            v = self._h.read_bool(name)
            return bool(v) if v is not None else None
        if type_str == "string":
            return self._h.read_string(name)
        if type_str == "token":
            return self._h.read_token(name) or self._h.read_string(name)
        if type_str == "asset":
            return self._h.read_asset(name)
        if type_str in _VEC2F_TYPES:
            v = self._h.read_vec2f(name)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str in _VEC3F_TYPES:
            v = self._h.read_vec3f(name)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str in _VEC4F_TYPES:
            v = self._h.read_vec4f(name)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str in _VEC2D_TYPES:
            v = self._h.read_vec2d(name)
            return np.array(v, dtype=np.float64) if v is not None else None
        if type_str in _VEC3D_TYPES:
            v = self._h.read_vec3d(name)
            return np.array(v, dtype=np.float64) if v is not None else None
        if type_str in _VEC4D_TYPES:
            v = self._h.read_vec4d(name)
            return np.array(v, dtype=np.float64) if v is not None else None
        if type_str == "matrix4d":
            return self.read_matrix4d(name)
        if type_str == "quatf":
            v = self._h.read_quatf(name)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str == "quatd":
            v = self._h.read_quatd(name)
            return np.array(v, dtype=np.float64) if v is not None else None
        if type_str.endswith("[]"):
            return self._read_array(name, type_str)
        return self._h.get(name)

    def _array_len(self, name: str) -> int:
        if not self._h:
            return -1
        if hasattr(self._h, "array_len"):
            return int(self._h.array_len(name))
        return len(self._h.read_float_array(name) or [])

    def _read_array(self, name: str, type_str: str) -> Any:
        n = self._array_len(name)
        if n <= 0:
            strings = self._h.read_string_array(name)
            if strings:
                return list(strings)
            tokens = self._h.read_token_array(name)
            if tokens:
                return list(tokens)
            return [] if n == 0 else None
        base = type_str.rstrip("[]")
        if base in ("int",):
            if hasattr(self._h, "read_int_array_numpy"):
                return self._h.read_int_array_numpy(name)
            values = self._h.read_int_array(name)
            return np.array(values, dtype=np.int32) if values else None
        if base in ("int64", "uint64"):
            if hasattr(self._h, "read_int64_array_numpy"):
                return self._h.read_int64_array_numpy(name)
            values = self._h.read_int64_array(name)
            return np.array(values, dtype=np.int64) if values else None
        if base in ("int2", "int3", "int4"):
            comps = int(base[-1])
            if hasattr(self._h, "read_int_array_numpy"):
                values = self._h.read_int_array_numpy(name, comps)
                if values is not None:
                    return values
            values = self._h.read_int_array_flat(name, comps) if hasattr(self._h, "read_int_array_flat") else []
            if values and len(values) % comps == 0:
                return np.array(values, dtype=np.int32).reshape(-1, comps)
            return self._read_intN_from_usda(name, comps)
        if base in ("double",):
            if hasattr(self._h, "read_double_array_numpy"):
                return self._h.read_double_array_numpy(name)
            values = self._h.read_double_array(name)
            return np.array(values, dtype=np.float64) if values else None
        if base in _VEC3D_ARRAY_BASES:
            if hasattr(self._h, "read_double_array_numpy"):
                return self._h.read_double_array_numpy(name, 3)
            values = self._h.read_double_array_flat(name, 3)
            return np.array(values, dtype=np.float64).reshape(-1, 3) if values else None
        if base in ("quatf", "quath"):
            if hasattr(self._h, "read_float_array_numpy"):
                return self._h.read_float_array_numpy(name, 4)
            values = self._h.read_float_array_flat(name, 4)
            return np.array(values, dtype=np.float32).reshape(-1, 4) if values else None
        if base == "quatd":
            if hasattr(self._h, "read_double_array_numpy"):
                return self._h.read_double_array_numpy(name, 4)
            values = self._h.read_double_array_flat(name, 4)
            return np.array(values, dtype=np.float64).reshape(-1, 4) if values else None
        if base in ("texCoord2d", "double2"):
            if hasattr(self._h, "read_double_array_numpy"):
                return self._h.read_double_array_numpy(name, 2)
            values = self._h.read_double_array_flat(name, 2)
            return np.array(values, dtype=np.float64).reshape(-1, 2) if values else None
        if base in ("double4", "color4d"):
            if hasattr(self._h, "read_double_array_numpy"):
                return self._h.read_double_array_numpy(name, 4)
            values = self._h.read_double_array_flat(name, 4)
            return np.array(values, dtype=np.float64).reshape(-1, 4) if values else None
        if base in ("token", "string"):
            return list(self._h.read_token_array(name) or self._h.read_string_array(name) or [])
        components = (
            3
            if base in _VEC3F_ARRAY_BASES
            else 2
            if base in ("texCoord2f", "float2")
            else 4
            if base in ("float4", "color4f")
            else 1
        )
        if hasattr(self._h, "read_float_array_numpy"):
            values = self._h.read_float_array_numpy(name, components)
            if values is not None:
                return values
        if base in _VEC3F_ARRAY_BASES:
            values = self._h.read_float_array_flat(name, 3)
        elif components == 1:
            values = self._h.read_float_array(name)
        else:
            values = self._h.read_float_array_flat(name, components)
        if not values or len(values) % components != 0:
            return None
        arr = np.array(values, dtype=np.float32)
        if base in _VEC3F_ARRAY_BASES:
            return arr.reshape(-1, 3)
        if base in ("texCoord2f", "float2"):
            return arr.reshape(-1, 2)
        if base in ("float4", "color4f"):
            return arr.reshape(-1, 4)
        return arr

    def get_time_sample(self, name: str, time: float, type_str: str | None = None) -> Any:
        type_str = type_str or self.get_attribute_type(name)
        if not type_str:
            return None
        return self._read_time_sample(name, type_str, float(time))

    def _read_time_sample(self, name: str, type_str: str, time: float) -> Any:
        if type_str == "double":
            v = self._h.read_sample_double(name, time)
            return float(v) if v is not None else None
        if type_str in ("float", "half", "int", "bool"):
            v = self._h.read_sample_float(name, time)
            return float(v) if v is not None else None
        if type_str in _VEC2F_TYPES:
            v = self._h.read_sample_vec2f(name, time)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str in _VEC3F_TYPES:
            v = self._h.read_sample_vec3f(name, time)
            return np.array(v, dtype=np.float32) if v is not None else None
        if type_str in _VEC3D_TYPES:
            v = self._h.read_sample_vec3d(name, time)
            return np.array(v, dtype=np.float64) if v is not None else None
        if type_str in _VEC2D_TYPES:
            values = self._h.read_sample_double_array_flat(name, time, 2)
            return np.array(values[:2], dtype=np.float64) if len(values) >= 2 else None
        if type_str in _VEC4F_TYPES:
            values = self._h.read_sample_float_array_flat(name, time, 4)
            return np.array(values[:4], dtype=np.float32) if len(values) >= 4 else None
        if type_str in _VEC4D_TYPES:
            values = self._h.read_sample_double_array_flat(name, time, 4)
            return np.array(values[:4], dtype=np.float64) if len(values) >= 4 else None
        if type_str == "matrix4d":
            values = self._h.read_sample_double_array_flat(name, time, 16)
            return np.array(values[:16], dtype=np.float64).reshape(4, 4) if len(values) >= 16 else None
        if type_str in ("quatf", "quath"):
            values = self._h.read_sample_float_array_flat(name, time, 4)
            if len(values) >= 4:
                arr = np.array(values[:4], dtype=np.float32)
                return _GfModule.Quatf(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            return None
        if type_str == "quatd":
            values = self._h.read_sample_double_array_flat(name, time, 4)
            if len(values) >= 4:
                arr = np.array(values[:4], dtype=np.float64)
                return _GfModule.Quatd(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            return None
        if type_str.endswith("[]"):
            return self._read_time_sample_array(name, type_str, time)
        v = self._h.read_sample_float(name, time)
        return float(v) if v is not None else None

    def _read_time_sample_array(self, name: str, type_str: str, time: float) -> Any:
        base = type_str.rstrip("[]")
        if base in ("int",):
            values = self._h.read_sample_int_array_flat(name, time, 1)
            return np.array(values, dtype=np.int32) if values else None
        if base in ("int64", "uint64"):
            values = self._h.read_sample_double_array_flat(name, time, 1)
            return np.array(values, dtype=np.float64).astype(np.int64) if values else None
        if base in ("double",):
            values = self._h.read_sample_double_array_flat(name, time, 1)
            return np.array(values, dtype=np.float64) if values else None
        if base in _VEC3D_ARRAY_BASES:
            values = self._h.read_sample_double_array_flat(name, time, 3)
            return np.array(values, dtype=np.float64).reshape(-1, 3) if values else None
        if base in ("texCoord2d", "double2"):
            values = self._h.read_sample_double_array_flat(name, time, 2)
            return np.array(values, dtype=np.float64).reshape(-1, 2) if values else None
        if base in ("double4", "color4d"):
            values = self._h.read_sample_double_array_flat(name, time, 4)
            return np.array(values, dtype=np.float64).reshape(-1, 4) if values else None
        if base in ("quatf", "quath"):
            values = self._h.read_sample_float_array_flat(name, time, 4)
            return np.array(values, dtype=np.float32).reshape(-1, 4) if values else None
        if base == "quatd":
            values = self._h.read_sample_double_array_flat(name, time, 4)
            return np.array(values, dtype=np.float64).reshape(-1, 4) if values else None
        components = (
            3
            if base in _VEC3F_ARRAY_BASES
            else 2
            if base in ("texCoord2f", "float2")
            else 4
            if base in ("float4", "color4f")
            else 1
        )
        values = self._h.read_sample_float_array_flat(name, time, components)
        if not values:
            return None
        arr = np.array(values, dtype=np.float32)
        if base in _VEC3F_ARRAY_BASES:
            return arr.reshape(-1, 3)
        if base in ("texCoord2f", "float2"):
            return arr.reshape(-1, 2)
        if base in ("float4", "color4f"):
            return arr.reshape(-1, 4)
        return arr

    def _read_intN_from_usda(self, name: str, comps: int) -> np.ndarray | None:
        """Fallback: parse intN[] from USDA export when native flat reads are unavailable."""
        import re
        if not self._h:
            return None
        stage = self._stage_h
        if not stage:
            return None
        usda = stage.write_usda_string() if hasattr(stage, "write_usda_string") else stage.to_usda_string()
        block = _find_usda_prim_block(usda, self.path)
        text = block if block is not None else usda
        pat = re.compile(rf'int{comps}\[\]\s+{re.escape(name)}\s*=\s*\[([^\]]*)\]')
        m = pat.search(text)
        if not m:
            return None
        tuples = re.findall(r'\(([^)]+)\)', m.group(1))
        if not tuples:
            return None
        values = []
        for t in tuples:
            values.extend(int(x.strip()) for x in t.split(','))
        return np.array(values, dtype=np.int32).reshape(-1, comps)

    def has_time_samples(self, name: str) -> bool:
        return bool(self._h and self._h.has_samples(name))

    def get_time_sample_keys(self, name: str) -> list[float]:
        return sorted(float(v) for v in (self._h.sample_keys(name) if self._h else []))

    def has_authored_attribute(self, name: str) -> bool:
        if not self._h:
            return False
        stage = self._stage_h
        if stage is not None:
            key = (self.path, str(name))
            cached = stage._attr_authored_cache.get(key)
            if cached is not None:
                return cached
            value = bool(self._h.is_attribute_authored(name))
            stage._attr_authored_cache[key] = value
            return value
        return bool(self._h.is_attribute_authored(name))

    def get_attribute_metadata(self, name: str, key: str) -> Any:
        if key == "interpolation":
            value = self._h.attribute_interpolation(name) if self._h else None
            return str(value) if value else None
        if key == "customData":
            return _get_usda_attribute_custom_data(self._stage_h, self.path, name)
        return None

    def get_attribute_names(self) -> list[str]:
        if not self._h:
            return []
        stage = self._stage_h
        if stage is not None:
            path = self.path
            cached = stage._attr_names_cache.get(path)
            if cached is None:
                cached = tuple(self._h.attribute_names())
                stage._attr_names_cache[path] = cached
            return list(cached)
        if self._attr_names_cache is None:
            self._attr_names_cache = list(self._h.attribute_names())
        return list(self._attr_names_cache)

    def get_authored_attribute_names(self) -> list[str]:
        if not self._h:
            return []
        stage = self._stage_h
        if stage is not None:
            path = self.path
            cached = stage._authored_attr_names_cache.get(path)
            if cached is None:
                if hasattr(self._h, "authored_attribute_names"):
                    cached = tuple(self._h.authored_attribute_names())
                else:
                    cached = tuple(n for n in self.get_attribute_names() if self.has_authored_attribute(n))
                stage._authored_attr_names_cache[path] = cached
            return list(cached)
        if self._authored_attr_names_cache is None:
            if hasattr(self._h, "authored_attribute_names"):
                self._authored_attr_names_cache = list(self._h.authored_attribute_names())
            else:
                self._authored_attr_names_cache = [
                    n for n in self.get_attribute_names() if self.has_authored_attribute(n)
                ]
        return list(self._authored_attr_names_cache)

    def get_attribute_names_in_namespace(self, namespace: str) -> list[str]:
        if not self._h:
            return []
        prefix = namespace if namespace.endswith(":") else namespace + ":"
        stage = self._stage_h
        if stage is not None:
            path = self.path
            ns_map = stage._attr_namespace_map_cache.get(path)
            if ns_map is None:
                ns_map = _build_attribute_namespace_map(self.get_attribute_names())
                stage._attr_namespace_map_cache[path] = ns_map
            return list(ns_map.get(prefix, ()))
        if self._attr_namespace_map_cache is None:
            self._attr_namespace_map_cache = _build_attribute_namespace_map(self.get_attribute_names())
        return list(self._attr_namespace_map_cache.get(prefix, ()))

    def get_authored_attribute_names_in_namespace(self, namespace: str) -> list[str]:
        if not self._h:
            return []
        if hasattr(self._h, "authored_attribute_names_in_namespace"):
            return list(self._h.authored_attribute_names_in_namespace(namespace))
        return [
            n for n in self.get_attribute_names_in_namespace(namespace)
            if self.has_authored_attribute(n)
        ]

    def get_children(self) -> list[_MuPrim]:
        if not self._h:
            return []
        children = [_MuPrim(c, stage_h=self._stage_h) for c in self._h.children()]
        if self._stage_h is None:
            return children
        # Child handles produced by traversal/children() can be read safely but
        # may not survive later write calls through nanobind. Re-resolve them
        # by path so callers can mutate returned prims.
        return [self._stage_h.get_prim_at_path(c.path) for c in children]

    def has_relationship(self, name: str) -> bool:
        return bool(self._h and self._h.has_relationship(name))

    def get_relationship_names(self) -> list[str]:
        if not self._h:
            return []
        rels: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                rels.append(name)

        getter = getattr(self._h, "relationship_names", None)
        if getter is not None:
            for name in getter():
                add(str(name))

        rel_prefixes = (
            "material:binding",
            "physics:body",
            "physics:collisionGroup",
            "prototypes",
            "collection:",
            "proxyPrim",
            "skel:",
        )
        for name in self.get_attribute_names():
            if any(name.startswith(p) or name == p for p in rel_prefixes) or self.has_relationship(name):
                add(name)
        for name in (
            "material:binding",
            "material:binding:preview",
            "material:binding:full",
            "material:binding:physics",
            "outputs:surface.connect",
            "outputs:mdl:surface.connect",
        ):
            if name not in seen and self.has_relationship(name):
                add(name)

        usda = _stage_source_usda(self._stage_h)
        block = _find_usda_prim_body_by_name(usda, self.path) if usda else None
        if block:
            import re
            pattern = re.compile(
                r"^\s*(?:custom\s+)?(?:prepend\s+|append\s+|delete\s+|add\s+)?"
                r"rel\s+([A-Za-z_][\w:]*)\b",
                re.M,
            )
            for name in pattern.findall(block):
                add(name)
        if self.type_name in {
            "BasisCurves",
            "Capsule",
            "Cone",
            "Cube",
            "Cylinder",
            "Mesh",
            "Plane",
            "Points",
            "Scope",
            "Sphere",
            "Xform",
        }:
            add("proxyPrim")
        try:
            if "PhysicsCollisionAPI" in self.get_applied_schemas():
                add("physics:simulationOwner")
        except Exception:
            pass
        return rels

    def read_rel_targets(self, name: str) -> list[str]:
        return list(self._h.relationship_targets(name)) if self._h else []

    def get_applied_schemas(self) -> list[str]:
        listop = self.get_listop("apiSchemas")
        return list(listop.items) if listop is not None else []

    def get_listop(self, field: str) -> Any:
        return self._h.listop(field) if self._h else None

    def get_prototype(self) -> _MuPrim | None:
        if not self._h:
            return None
        proto = self._h.prototype()
        return _MuPrim(proto, stage_h=self._stage_h) if proto else None

    def has_connections(self, attr_name: str) -> bool:
        return bool(self._h and self._h.has_connections(attr_name))

    def read_connections(self, attr_name: str) -> list[str]:
        return list(self._h.connections(attr_name)) if self._h else []

    def variant_set_names(self) -> list[str]:
        return list(self._h.variant_set_names()) if self._h else []

    def variant_names(self, set_name: str) -> list[str]:
        return list(self._h.variant_names(set_name)) if self._h else []

    def variant_selection(self, set_name: str) -> str:
        return str(self._h.variant_selection(set_name)) if self._h else ""

    def apply_api(self, schema_name: str) -> bool:
        ok = bool(self._h and self._h.apply_api(schema_name))
        if ok:
            self._invalidate_attr_caches()
        return ok

    def clear_attribute(self, name: str) -> bool:
        ok = bool(self._h and self._h.clear_default(name))
        if ok and self._stage_h is not None:
            self._stage_h._attr_authored_cache.pop((self.path, str(name)), None)
        return ok

    def create_attribute(self, name: str, type_name: str = "") -> bool:
        ok = bool(self._h and self._h.create_attribute(str(name), str(type_name or "")))
        if ok:
            self._invalidate_attr_caches()
            if self._stage_h is not None:
                key = (self.path, str(name))
                self._stage_h._attr_has_cache[key] = True
                if type_name:
                    self._stage_h._attr_type_cache[key] = str(type_name)
        return ok

    def read_float(self, name: str, fallback: float = 0.0) -> float:
        v = self._h.read_float(name) if self._h else None
        return float(v) if v is not None else fallback

    def read_int(self, name: str, fallback: int = 0) -> int:
        v = self._h.read_int(name) if self._h else None
        return int(v) if v is not None else fallback

    def read_string(self, name: str, fallback: str = "") -> str:
        if not self._h:
            return fallback
        v = self._h.read_string(name)
        if v is not None:
            return str(v)
        v = self._h.read_token(name)
        return str(v) if v is not None else fallback

    def read_matrix4d(self, name: str) -> np.ndarray | None:
        v = self._h.read_matrix4d(name) if self._h else None
        return np.array(v, dtype=np.float64).reshape(4, 4) if v is not None else None

    def get_prim_metadata(self, key: str) -> Any:
        if key == "customData":
            return None  # customData dict not supported via C API
        if not self._h:
            return None
        v = self._h.metadata_string(key)
        if v is not None:
            return str(v)
        v = self._h.metadata_double(key)
        return float(v) if v is not None else None

    def set_prim_metadata_d(self, key: str, value: float) -> bool:
        return bool(self._h and self._h.set_metadata_double(key, float(value)))

    def set_prim_metadata_s(self, key: str, value: str) -> bool:
        return bool(self._h and self._h.set_metadata_string(key, str(value)))

    def set_prim_metadata_token(self, key: str, value: str) -> bool:
        return bool(self._h and self._h.set_metadata_token(key, str(value)))

    # --- Write operations ---

    def set_attribute_s(self, name: str, value: str) -> bool:
        return bool(self._h and self._h.set_string(name, str(value)))

    def set_attribute_token(self, name: str, value: str) -> bool:
        return bool(self._h and self._h.set_token(name, str(value)))

    def set_attribute_b(self, name: str, value: bool) -> bool:
        return bool(self._h and self._h.set_bool(name, bool(value)))

    def set_attribute_f(self, name: str, value: float) -> bool:
        return bool(self._h and self._h.set_float(name, float(value)))

    def set_attribute_d(self, name: str, value: float) -> bool:
        return bool(self._h and self._h.set_double(name, float(value)))

    def set_attribute_i(self, name: str, value: int) -> bool:
        return bool(self._h and self._h.set_int(name, int(value)))

    def set_attribute_i64(self, name: str, value: int) -> bool:
        return bool(self._h and self._h.set_int64(name, int(value)))

    def set_attribute_v2f(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_vec2f(name, [float(a[0]), float(a[1])]))

    def set_attribute_v2d(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_vec2d(name, [float(a[0]), float(a[1])]))

    def set_attribute_v3f(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_vec3f(name, [float(a[0]), float(a[1]), float(a[2])]))

    def set_attribute_v3d(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_vec3d(name, [float(a[0]), float(a[1]), float(a[2])]))

    def set_attribute_v4f(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_vec4f(name, [float(v) for v in a[:4]]))

    def set_attribute_v4d(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_vec4d(name, [float(v) for v in a[:4]]))

    def set_attribute_m4d(self, name: str, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_matrix4d(name, [float(v) for v in a[:16]]))

    def set_attribute_quatf(self, name: str, w: float, x: float, y: float, z: float) -> bool:
        return bool(self._h and self._h.set_quatf(name, [float(w), float(x), float(y), float(z)]))

    def set_attribute_array_f(self, name: str, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_float_array(name, [float(v) for v in a]))

    def set_attribute_array_d(self, name: str, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_double_array(name, [float(v) for v in a]))

    def set_attribute_array_i(self, name: str, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.int32).ravel()
        return bool(self._h and self._h.set_int_array(name, [int(v) for v in a]))

    def set_attribute_array_v3f(self, name: str, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_vec3f_array(name, [float(v) for v in a]))

    def set_attribute_array_v3d(self, name: str, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_vec3d_array(name, [float(v) for v in a]))

    def set_attribute_token_array(self, name: str, tokens: list[str]) -> bool:
        return bool(self._h and self._h.set_token_array(name, [str(t) for t in tokens]))

    def set_attribute_string_array(self, name: str, strings: list[str]) -> bool:
        return bool(self._h and self._h.set_string_array(name, [str(s) for s in strings]))

    # --- Time sample setters ---

    def set_time_sample_f(self, name: str, time: float, value: float) -> bool:
        return bool(self._h and self._h.set_sample_float(name, float(time), float(value)))

    def set_time_sample_d(self, name: str, time: float, value: float) -> bool:
        return bool(self._h and self._h.set_sample_double(name, float(time), float(value)))

    def set_time_sample_v3f(self, name: str, time: float, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_sample_vec3f(name, float(time), [float(v) for v in a[:3]]))

    def set_time_sample_v3d(self, name: str, time: float, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_vec3d(name, float(time), [float(v) for v in a[:3]]))

    def set_time_sample_v4f(self, name: str, time: float, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_sample_vec4f(name, float(time), [float(v) for v in a[:4]]))

    def set_time_sample_token(self, name: str, time: float, value: str) -> bool:
        return bool(self._h and self._h.set_sample_token(name, float(time), str(value)))

    def set_time_sample_quatf(self, name: str, time: float, w: float, x: float, y: float, z: float) -> bool:
        return bool(self._h and self._h.set_sample_quatf(name, float(time), [float(w), float(x), float(y), float(z)]))

    def set_time_sample_array_f(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_sample_float_array(name, float(time), [float(v) for v in a]))

    def set_time_sample_array_i(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.int32).ravel()
        return bool(self._h and self._h.set_sample_int_array(name, float(time), [int(v) for v in a]))

    def set_time_sample_array_v3f(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        return bool(self._h and self._h.set_sample_vec3f_array(name, float(time), [float(v) for v in a]))

    def set_time_sample_v2d(self, name: str, time: float, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_vec2d(name, float(time), [float(v) for v in a[:2]]))

    def set_time_sample_v4d(self, name: str, time: float, arr: Any) -> bool:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_vec4d(name, float(time), [float(v) for v in a[:4]]))

    def set_time_sample_m4d(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_matrix4d(name, float(time), [float(v) for v in a[:16]]))

    def set_time_sample_array_d(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_double_array(name, float(time), [float(v) for v in a]))

    def set_time_sample_array_v3d(self, name: str, time: float, arr: Any) -> bool:
        a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
        return bool(self._h and self._h.set_sample_vec3d_array(name, float(time), [float(v) for v in a]))

    # --- Relationships ---

    def create_relationship(self, name: str, targets: list[str]) -> bool:
        if not self._h:
            return False
        self._h.create_relationship(name)
        ok = bool(self._h.set_relationship_targets(name, [str(t) for t in targets]))
        if ok:
            self._invalidate_attr_caches()
        return ok

    def add_relationship_target(self, name: str, target: str) -> bool:
        if not self._h:
            return False
        self._h.create_relationship(name)
        ok = bool(self._h.add_relationship_target(name, str(target)))
        if ok:
            self._invalidate_attr_caches()
        return ok

    # ---- Composition-arc authoring (panel-c-api) ----

    def add_reference(self, asset_path: str, prim_path: str) -> bool:
        """Add a reference. asset_path may be empty for an internal ref."""
        if not self._h:
            return False
        return bool(self._h.add_reference(asset_path or "", prim_path or ""))

    def add_payload(self, asset_path: str, prim_path: str) -> bool:
        if not self._h:
            return False
        return bool(self._h.add_payload(asset_path or "", prim_path or ""))

    def add_inherit(self, prim_path: str) -> bool:
        if not self._h:
            return False
        return bool(self._h.add_inherit(prim_path))

    def add_specialize(self, prim_path: str) -> bool:
        if not self._h:
            return False
        return bool(self._h.add_specialize(prim_path))

    def remove_reference(self, index: int) -> bool:
        if not self._h:
            return False
        return bool(self._h.remove_reference(int(index)))

    def remove_payload(self, index: int) -> bool:
        if not self._h:
            return False
        return bool(self._h.remove_payload(int(index)))

    def remove_inherit(self, index: int) -> bool:
        if not self._h:
            return False
        return bool(self._h.remove_inherit(int(index)))

    def remove_specialize(self, index: int) -> bool:
        if not self._h:
            return False
        return bool(self._h.remove_specialize(int(index)))

    def remove_listop_item(self, field: str, kind: int, index: int) -> bool:
        """Remove the index-th item from the named listop sublist.
        kind: 0=explicit, 1=prepended, 2=appended, 3=deleted."""
        if not self._h:
            return False
        return bool(self._h.remove_listop_item(field, int(kind), int(index)))

    # ---- Prim-state writers (panel-c-api) ----

    def set_active(self, active: bool) -> bool:
        return bool(self._h and self._h.set_active(bool(active)))

    def set_instanceable(self, instanceable: bool) -> bool:
        return bool(self._h and self._h.set_instanceable(bool(instanceable)))

    def remove_api(self, schema_name: str) -> bool:
        ok = bool(self._h and self._h.remove_api(schema_name))
        if ok:
            self._invalidate_attr_caches()
        return ok

    def remove_prim(self) -> bool:
        """Delete this prim and all of its descendants from the root layer.
        Invalidates self and any descendant handles."""
        if not self._h:
            return False
        ok = bool(self._h.remove())
        if ok:
            self._h = None  # handle is now stale
        return ok

    # ---- Variant set authoring (panel-c-api) ----

    def create_variantset(self, set_name: str) -> bool:
        return bool(self._h and self._h.create_variant_set(set_name))

    def create_variant(self, set_name: str, variant_name: str) -> bool:
        return bool(self._h and self._h.create_variant(set_name, variant_name))

    def set_variant_selection(self, set_name: str, variant_name: str | None,
                              layer_index: int = 0) -> bool:
        """Author a variant selection. variant_name=None or '' clears it."""
        if not self._h:
            return False
        return bool(self._h.set_variant_selection(set_name, variant_name or "", int(layer_index)))


# ===========================================================================
# _MuStage — compatibility wrapper around the nanobind Stage object
# ===========================================================================


class _MuStage:
    """Wrapper around nanousd._nanousd.Stage providing the nanobind interface."""

    __slots__ = (
        "_h",
        "_usda_source_cache",
        "_attr_custom_data_cache",
        "_attr_names_cache",
        "_authored_attr_names_cache",
        "_attr_namespace_cache",
        "_attr_namespace_map_cache",
        "_attr_has_cache",
        "_attr_authored_cache",
        "_attr_type_cache",
    )

    def __init__(self, handle: Any):
        self._h = handle
        self._usda_source_cache = None
        self._attr_custom_data_cache = {}
        self._attr_names_cache: dict[str, tuple[str, ...]] = {}
        self._authored_attr_names_cache: dict[str, tuple[str, ...]] = {}
        self._attr_namespace_cache: dict[tuple[str, str], tuple[str, ...]] = {}
        self._attr_namespace_map_cache: dict[str, dict[str, tuple[str, ...]]] = {}
        self._attr_has_cache: dict[tuple[str, str], bool] = {}
        self._attr_authored_cache: dict[tuple[str, str], bool] = {}
        self._attr_type_cache: dict[tuple[str, str], str] = {}

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def open(filepath: str) -> _MuStage:
        try:
            return _MuStage(_native.Stage.open(str(filepath)))
        except Exception as exc:
            raise RuntimeError(f"Failed to open USD file: {filepath}: {exc}") from exc

    @staticmethod
    def create() -> _MuStage:
        return _MuStage(_native.Stage.create())

    def invalidate_prim_attribute_cache(self, path: str) -> None:
        self._attr_names_cache.pop(path, None)
        self._authored_attr_names_cache.pop(path, None)
        self._attr_namespace_map_cache.pop(path, None)
        for cache in (self._attr_namespace_cache, self._attr_has_cache, self._attr_authored_cache, self._attr_type_cache):
            for key in tuple(cache):
                if key[0] == path:
                    cache.pop(key, None)

    def get_prim_at_path(self, path: str) -> _MuPrim:
        return _MuPrim(self._h.get_prim_at_path(str(path)) if self._h else None, stage_h=self)

    def traverse(self) -> list[_MuPrim]:
        return [_MuPrim(p, stage_h=self) for p in (self._h.traverse() if self._h else [])]

    def get_root_prims(self) -> list[_MuPrim]:
        root = self._h.get_prim_at_path("/") if self._h else None
        if root and getattr(root, "is_valid", False):
            return [
                self.get_prim_at_path(_MuPrim(c, stage_h=self).path)
                for c in root.children()
            ]
        # Fallback: nanousd may not expose "/" as a real prim.
        # Filter traverse() results for top-level prims (path = "/<name>").
        return [
            self.get_prim_at_path(p.path)
            for p in self.traverse()
            if p.path.count("/") == 1
        ]

    def get_default_prim(self) -> _MuPrim:
        return _MuPrim(self._h.default_prim() if self._h else None, stage_h=self)

    def set_default_prim(self, name: str) -> None:
        if self._h:
            self._h.set_metadata_token("defaultPrim", str(name))

    def get_metadata(self, key: str) -> float | str | None:
        if not self._h:
            return None
        v = self._h.metadata_double(key)
        if v is not None:
            return float(v)
        if key in _NUMERIC_STAGE_METADATA_KEYS:
            return None
        v = self._h.metadata_string(key)
        if v is not None:
            return str(v)
        if key in _STRING_STAGE_METADATA_FALLBACK_KEYS:
            return _find_usda_layer_metadata(self.write_usda_string(), key)
        return None

    @property
    def frames_per_second(self) -> float:
        return float(self._h.frames_per_second) if self._h else 24.0

    @property
    def timecodes_per_second(self) -> float:
        return float(self._h.time_codes_per_second) if self._h else 24.0

    @property
    def start_time(self) -> float:
        return float(self._h.start_time) if self._h else 0.0

    @property
    def end_time(self) -> float:
        return float(self._h.end_time) if self._h else 0.0

    def set_timecodes_per_second(self, tps: float) -> None:
        if self._h:
            self._h.set_metadata_double("timeCodesPerSecond", float(tps))

    def set_frames_per_second(self, fps: float) -> None:
        if self._h:
            self._h.set_metadata_double("framesPerSecond", float(fps))

    def set_start_time(self, time: float) -> None:
        if self._h:
            self._h.set_metadata_double("startTimeCode", float(time))

    def set_end_time(self, time: float) -> None:
        if self._h:
            self._h.set_metadata_double("endTimeCode", float(time))

    def set_up_axis(self, axis: str) -> None:
        if self._h:
            self._h.set_metadata_token("upAxis", str(axis))

    def set_meters_per_unit(self, value: float) -> None:
        if self._h:
            self._h.set_metadata_double("metersPerUnit", float(value))

    def set_metadata_d(self, key: str, value: float) -> None:
        if self._h:
            self._h.set_metadata_double(key, float(value))

    def set_metadata_s(self, key: str, value: str) -> None:
        if self._h:
            self._h.set_metadata_string(key, str(value))

    def define_prim(self, path: str, type_name: str = "") -> None:
        if self._h:
            self._h.define_prim(str(path), str(type_name or ""))

    def write_usda(self, filepath: str) -> bool:
        return bool(self._h and self._h.write_usda(str(filepath)))

    def write_usda_string(self) -> str:
        return str(self._h.to_usda_string()) if self._h else ""

    def copy_prim_subtree(self, src: str, dst: str) -> bool:
        return False  # Not directly supported via C API

    def close(self) -> None:
        self._h = None

    @property
    def error(self) -> str:
        return str(self._h.error) if self._h else ""

    @property
    def used_layers(self) -> list[str]:
        return list(self._h.used_layers) if self._h else []

    def layer_sublayer_paths(self, layer_index: int) -> list[str]:
        return list(self._h.layer_sublayer_paths(int(layer_index))) if self._h else []

    def layer_offset(self, layer_index: int) -> tuple[float, float]:
        value = self._h.layer_offset(int(layer_index)) if self._h else None
        return (float(value[0]), float(value[1])) if value else (0.0, 1.0)

    def layer_has_prim_spec(self, layer_index: int, prim_path: str) -> bool:
        return bool(self._h and self._h.layer_has_prim_spec(int(layer_index), str(prim_path)))

    def layer_has_attr_opinion(self, layer_index: int, prim_path: str, attr_name: str) -> bool:
        return bool(self._h and self._h.layer_has_attr_opinion(int(layer_index), str(prim_path), str(attr_name)))


def _is_instance_proxy_mu_prim(prim: _MuPrim) -> bool:
    stage = getattr(prim, "_stage_h", None)
    path = prim.path
    if stage is None or not path or path == "/":
        return False
    parts = path.strip("/").split("/")
    for i in range(len(parts) - 1, 0, -1):
        ancestor_path = "/" + "/".join(parts[:i])
        ancestor = stage.get_prim_at_path(ancestor_path)
        if ancestor and ancestor.is_valid and ancestor.is_instance:
            return True
    return False


# ===========================================================================
# Attribute proxy — wraps a prim + attr name to provide pxr-style Get() API
# ===========================================================================


class _Attribute:
    """Wraps a nanousd prim + attribute name to provide pxr Usd.Attribute interface."""

    def __init__(
        self,
        prim: _MuPrim,
        name: str,
        *,
        valid: bool | None = None,
        type_name: str | None = None,
        authored: bool | None = None,
    ):
        self._prim = prim
        self._name = name
        self._valid = prim.has_attribute(name) if valid is None else bool(valid)
        self._type: str | None = type_name  # cached type name
        self._authored: bool | None = authored

    def __bool__(self) -> bool:
        return self._valid

    def IsValid(self) -> bool:
        return self._valid

    def GetName(self) -> str:
        return self._name

    def GetTypeName(self) -> str:
        if self._valid:
            if self._type is None:
                self._type = self._prim.get_attribute_type(self._name)
            return self._type
        return ""

    def _get_type(self) -> str:
        if self._type is None:
            self._type = self._prim.get_attribute_type(self._name)
        return self._type

    def Get(self, time: float | None = None) -> Any:
        if not self._valid:
            return None
        atype = self._get_type()
        if time is not None:
            val = self._prim.get_time_sample(self._name, time, atype)
        else:
            val = self._prim.get_attribute(self._name, atype)
        if val is None:
            key = _prim_value_key(self._prim, self._name)
            if key in _PY_ATTR_VALUES:
                return _PY_ATTR_VALUES[key]
            return None
        # Auto-wrap quaternion types to match pxr behavior
        if atype in ("quatf", "quatd", "quath"):
            arr = np.asarray(val, dtype=np.float64).ravel()
            if len(arr) >= 4:
                # nanousd C API returns (w, i, j, k) = (real, imaginary...)
                if atype == "quatd":
                    return _GfModule.Quatd(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                elif atype == "quath":
                    return _GfModule.Quath(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                return _GfModule.Quatf(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        return val

    def HasTimeSamples(self) -> bool:
        return self._prim.has_time_samples(self._name)

    def GetTimeSamples(self) -> list[float]:
        return self._prim.get_time_sample_keys(self._name)

    def GetTimeSamplesInRange(self, start: float, end: float) -> list[float]:
        return [t for t in self.GetTimeSamples() if start <= t <= end]

    def HasAuthoredValue(self) -> bool:
        key = _prim_value_key(self._prim, self._name)
        if key in _PY_ATTR_VALUES:
            return True
        if not self._valid:
            return False
        if self._authored is not None:
            return self._authored
        if hasattr(self._prim, "has_authored_attribute"):
            if self._prim.has_authored_attribute(self._name):
                return True
        if self._prim.is_in_prototype or _is_instance_proxy_mu_prim(self._prim):
            return self._prim.get_attribute(self._name) is not None
        return False

    def HasValue(self) -> bool:
        return self.HasAuthoredValue()

    def IsDefined(self) -> bool:
        return self._valid

    def GetCustomDataByKey(self, key: str) -> Any:
        """Read custom metadata from attribute's customData dict."""
        cd = self._prim.get_attribute_metadata(self._name, "customData")
        if isinstance(cd, dict):
            return cd.get(key)
        return None

    # --- Write operations ---

    def Set(self, value: Any, time: float | None = None) -> bool:
        """Set attribute value, optionally at a specific time sample."""
        if not self._valid and value is not None and self._prim._h:
            # Auto-create attribute with inferred type for schema-like Set()
            type_hint = _infer_usd_type(value)
            if type_hint:
                self._prim.create_attribute(self._name, type_hint)
                self._valid = self._prim.has_attribute(self._name)
                self._type = type_hint
        ok = _set_attribute_value(self._prim, self._name, value, time)
        type_str = self.GetTypeName()
        if ok:
            self._authored = True
            self._prim._mark_attribute_authored(self._name, type_str)
        if ok and type_str.rstrip("[]") in {"int2", "int3", "int4"} and time is None:
            key = _prim_value_key(self._prim, self._name)
            if key is not None:
                _PY_ATTR_VALUES[key] = _coerce_python_attr_value(value, type_str)
        if ok or time is not None:
            return ok
        key = _prim_value_key(self._prim, self._name)
        if key is not None:
            _PY_ATTR_VALUES[key] = _coerce_python_attr_value(value, type_str)
            return True
        return False

    def SetInterpolation(self, interp: str) -> bool:
        """Set interpolation metadata (for primvar attributes)."""
        meta_name = f"{self._name}:interpolation"
        if not self._prim.has_attribute(meta_name):
            self._prim.create_attribute(meta_name, "token")
        return self._prim.set_attribute_token(meta_name, str(interp))

    def SetIndices(self, indices: Any, time: float | None = None) -> bool:
        """Set primvar indices."""
        return _set_attribute_value(self._prim, f"{self._name}:indices", indices, time)


class _QuatAttribute(_Attribute):
    """Attribute wrapper that returns Quatf from Get() for quaternion-typed attributes."""

    def Get(self, time: float | None = None) -> Any:
        val = super().Get(time)
        if val is None:
            return None
        # Already a quaternion object — return as-is
        if hasattr(val, "GetReal") and hasattr(val, "GetImaginary"):
            return val
        arr = np.asarray(val, dtype=np.float64).ravel()
        if len(arr) >= 4:
            # nanousd C API returns (w, i, j, k) = (real, imaginary...)
            return _GfModule.Quatf(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        return val


def _infer_usd_type(value: Any) -> str:
    """Infer a USD type name from a Python value for attribute auto-creation."""
    if isinstance(value, str):
        return "token"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if hasattr(value, "GetReal"):  # Quatf/Quatd
        return "quatf"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return "double" if value.dtype == np.float64 else "float"
        if value.ndim == 1:
            n = value.size
            is_double = value.dtype in (np.float64,)
            is_int = value.dtype in (np.int32, np.int64)
            if is_int:
                return "int[]" if n > 4 else "int"
            if n == 2:
                return "double2" if is_double else "float2"
            if n == 3:
                return "point3d" if is_double else "point3f"
            if n == 4:
                return "double4" if is_double else "float4"
            if n == 16:
                return "matrix4d"
            return "double[]" if is_double else "float[]"
        if value.ndim == 2:
            _, cols = value.shape
            is_double = value.dtype in (np.float64,)
            if cols == 3:
                return "point3d[]" if is_double else "point3f[]"
            if value.shape == (4, 4):
                return "matrix4d"
            return "double[]" if is_double else "float[]"
    if isinstance(value, (list, tuple)):
        if len(value) > 0:
            if isinstance(value[0], (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.ndim == 2:
                    _, cols = arr.shape
                    if arr.dtype.kind in ("i", "u"):
                        if cols in (2, 3, 4):
                            return f"int{cols}[]"
                        return "int[]"
                    is_double = arr.dtype in (np.float64,)
                    if cols == 2:
                        return "double2[]" if is_double else "float2[]"
                    if cols == 3:
                        return "point3d[]" if is_double else "point3f[]"
                    if cols == 4:
                        return "double4[]" if is_double else "float4[]"
            if isinstance(value[0], str):
                return "token[]"
            if isinstance(value[0], int):
                return "int[]"
            if isinstance(value[0], float):
                if len(value) == 2:
                    return "double2"
                if len(value) == 3:
                    return "double3"
                if len(value) == 4:
                    return "double4"
                return "double[]"
    return ""


def _set_attribute_value(prim: _MuPrim, name: str, value: Any, time: float | None = None) -> bool:
    """Route a Python value to the correct typed setter on the nanousd Prim."""
    if value is None:
        return False
    type_str = prim.get_attribute_type(name) if prim.has_attribute(name) else ""
    is_array_attr = type_str.endswith("[]")
    base_type = type_str[:-2] if is_array_attr else type_str

    # String / token values
    if isinstance(value, str):
        if time is not None:
            return prim.set_time_sample_token(name, time, value)
        if base_type == "string":
            return prim.set_attribute_s(name, value)
        return prim.set_attribute_token(name, value)

    # Boolean (check before int since bool is subclass of int)
    if isinstance(value, bool):
        if time is not None:
            return prim.set_time_sample_f(name, time, 1.0 if value else 0.0)
        return prim.set_attribute_b(name, value)

    # Quaternion wrappers (Gf.Quatf / Gf.Quatd)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        r = float(value.GetReal())
        imag = value.GetImaginary()
        i, j, k = float(imag[0]), float(imag[1]), float(imag[2])
        if time is not None:
            return prim.set_time_sample_quatf(name, time, r, i, j, k)
        return prim.set_attribute_quatf(name, r, i, j, k)

    # List of quaternion objects (must check before np.asarray which can't handle them)
    if isinstance(value, (list, tuple)) and len(value) > 0 and hasattr(value[0], "GetReal"):
        quats = np.array(
            [[v.GetImaginary()[0], v.GetImaginary()[1], v.GetImaginary()[2], v.GetReal()] for v in value],
            dtype=np.float32,
        )
        flat = quats.ravel()
        if time is not None:
            return prim.set_time_sample_array_f(name, time, flat)
        return prim.set_attribute_array_f(name, flat)

    # Scalar int
    if isinstance(value, int):
        if time is not None:
            return prim.set_time_sample_f(name, time, float(value))
        return prim.set_attribute_i(name, value)

    # Scalar float — use double setter (Python float is 64-bit);
    # the C backend downcasts for "float"-typed attributes automatically
    if isinstance(value, float):
        if time is not None:
            return prim.set_time_sample_d(name, time, value) or prim.set_time_sample_f(name, time, value)
        return prim.set_attribute_d(name, value) or prim.set_attribute_f(name, value)

    # Lists of ints (e.g. [0]*N, list(range(N)))
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], int):
        arr = np.array(value, dtype=np.int32)
        if is_array_attr and base_type in _VEC3F_ARRAY_BASES:
            farr = arr.astype(np.float32).reshape(-1, 3)
            if time is not None:
                return prim.set_time_sample_array_v3f(name, time, farr)
            return prim.set_attribute_array_v3f(name, farr)
        if is_array_attr and base_type in _VEC3D_ARRAY_BASES:
            darr = arr.astype(np.float64).reshape(-1, 3)
            if time is not None:
                return prim.set_time_sample_array_v3d(name, time, darr)
            return prim.set_attribute_array_v3d(name, darr)
        if time is not None:
            return prim.set_time_sample_array_i(name, time, arr)
        return prim.set_attribute_array_i(name, arr)

    # Lists of strings (token[] or string[])
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str):
        return prim.set_attribute_token_array(name, list(value))

    # numpy arrays and lists
    arr = np.asarray(value)
    if arr.ndim == 0:
        # Scalar wrapped in array
        return _set_attribute_value(prim, name, arr.item(), time)

    if arr.ndim == 1:
        n = len(arr)
        if is_array_attr and base_type in _VEC3F_ARRAY_BASES:
            farr = arr.astype(np.float32).reshape(-1, 3)
            if time is not None:
                return prim.set_time_sample_array_v3f(name, time, farr)
            return prim.set_attribute_array_v3f(name, farr)
        if is_array_attr and base_type in _VEC3D_ARRAY_BASES:
            darr = arr.astype(np.float64).reshape(-1, 3)
            if time is not None:
                return prim.set_time_sample_array_v3d(name, time, darr)
            return prim.set_attribute_array_v3d(name, darr)
        # Integer arrays always go as int[] (never as vec types)
        if arr.dtype in (np.int32, np.int64):
            iarr = arr.astype(np.int32)
            if time is not None:
                return prim.set_time_sample_array_i(name, time, iarr)
            return prim.set_attribute_array_i(name, iarr)
        if is_array_attr:
            if arr.dtype == np.float64 or base_type == "double":
                darr = arr.astype(np.float64)
                if time is not None:
                    return prim.set_time_sample_array_d(name, time, darr)
                return prim.set_attribute_array_d(name, darr)
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_array_f(name, time, farr)
            return prim.set_attribute_array_f(name, farr)
        # Small float vectors: vec2/3/4
        if n == 2 and arr.dtype in (np.float32, np.float64):
            if base_type in _VEC2D_TYPES:
                if time is not None:
                    return prim.set_time_sample_v2d(name, time, arr.astype(np.float64))
                return prim.set_attribute_v2d(name, arr.astype(np.float64))
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_array_f(name, time, farr)
            return prim.set_attribute_v2f(name, farr)
        if n == 3:
            if base_type in _VEC3D_TYPES:
                if time is not None:
                    return prim.set_time_sample_v3d(name, time, arr.astype(np.float64))
                return prim.set_attribute_v3d(name, arr.astype(np.float64))
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_v3f(name, time, farr)
            return prim.set_attribute_v3f(name, farr)
        if n == 4:
            if base_type in _VEC4D_TYPES:
                if time is not None:
                    return prim.set_time_sample_v4d(name, time, arr.astype(np.float64))
                return prim.set_attribute_v4d(name, arr.astype(np.float64))
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_v4f(name, time, farr)
            return prim.set_attribute_v4f(name, farr)
        # 1D float array — preserve float64 when possible
        if arr.dtype == np.float64:
            if time is not None:
                return prim.set_time_sample_array_d(name, time, arr)
            return prim.set_attribute_array_d(name, arr)
        farr = arr.astype(np.float32)
        if time is not None:
            return prim.set_time_sample_array_f(name, time, farr)
        return prim.set_attribute_array_f(name, farr)

    if arr.ndim == 2:
        if is_array_attr and base_type in _VEC3F_ARRAY_BASES:
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_array_v3f(name, time, farr)
            return prim.set_attribute_array_v3f(name, farr)
        if is_array_attr and base_type in _VEC3D_ARRAY_BASES:
            darr = arr.astype(np.float64)
            if time is not None:
                return prim.set_time_sample_array_v3d(name, time, darr)
            return prim.set_attribute_array_v3d(name, darr)
        if arr.dtype in (np.int32, np.int64) or arr.dtype.kind in ("i", "u"):
            iarr = arr.astype(np.int32).ravel()
            if time is not None:
                return prim.set_time_sample_array_i(name, time, iarr)
            return prim.set_attribute_array_i(name, iarr)
        # 4x4 matrix → Matrix4d
        if arr.shape == (4, 4):
            darr = arr.astype(np.float64)
            if time is not None:
                return prim.set_time_sample_m4d(name, time, darr)
            return prim.set_attribute_m4d(name, darr)
        if arr.shape[1] == 4:
            # Nx4 float array (e.g. quaternion arrays) — flatten to typed 1D array
            if arr.dtype == np.float64:
                darr = arr.astype(np.float64).ravel()
                if time is not None:
                    return prim.set_time_sample_array_d(name, time, darr)
                return prim.set_attribute_array_d(name, darr)
            farr = arr.astype(np.float32).ravel()
            if time is not None:
                return prim.set_time_sample_array_f(name, time, farr)
            return prim.set_attribute_array_f(name, farr)
        if arr.shape[1] == 3:
            if arr.dtype == np.float64 and base_type in _VEC3D_ARRAY_BASES:
                if time is not None:
                    return prim.set_time_sample_array_v3d(name, time, arr)
                return prim.set_attribute_array_v3d(name, arr)
            farr = arr.astype(np.float32)
            if time is not None:
                return prim.set_time_sample_array_v3f(name, time, farr)
            return prim.set_attribute_array_v3f(name, farr)
        # Flatten for generic 2D
        if arr.dtype == np.float64:
            flat = arr.ravel().astype(np.float64)
            if time is not None:
                return prim.set_time_sample_array_d(name, time, flat)
            return prim.set_attribute_array_d(name, flat)
        flat = arr.ravel().astype(np.float32)
        if time is not None:
            return prim.set_time_sample_array_f(name, time, flat)
        return prim.set_attribute_array_f(name, flat)

    return False


class _Relationship:
    """Proxy for Usd.Relationship backed by nanousd read_rel_targets."""

    def __init__(self, prim: _MuPrim, name: str, *, valid: bool | None = None):
        self._prim = prim
        self._name = name
        self._valid = valid

    def __bool__(self) -> bool:
        return self.IsValid()

    def IsValid(self) -> bool:
        return self._prim.has_relationship(self._name) or bool(self._valid)

    def GetName(self) -> str:
        return self._name

    def GetTargets(self) -> list:
        """Return targets as Sdf.Path objects (matching pxr behavior)."""
        return [_SdfModule.Path(t) for t in self._prim.read_rel_targets(self._name)]

    def HasAuthoredTargets(self) -> bool:
        """Return True if this relationship has any authored targets."""
        return len(self.GetTargets()) > 0

    def SetTargets(self, targets: list) -> bool:
        """Set relationship targets (replaces existing)."""
        str_targets = [str(t) for t in targets]
        return self._prim.create_relationship(self._name, str_targets)

    def AddTarget(self, target: Any) -> bool:
        """Add a single target to this relationship."""
        return self._prim.add_relationship_target(self._name, str(target))


class _ReferencesProxy:
    """Proxy for prim.GetReferences().AddInternalReference(path)."""

    def __init__(self, prim: _MuPrim, stage: Any = None):
        self._prim = prim
        self._stage = stage

    def AddInternalReference(self, path: Any) -> bool:
        """Add an internal reference (spec §6.3.5 composition arc).

        An internal reference has an empty asset path and a prim-path
        target inside the same layer stack. nanousd composes it as a
        true reference arc, not a copy or a relationship. The underlying
        nanousd_add_reference recomposes internally and refreshes the
        passed-in prim handle; do not recompose again here — recompose
        invalidates every other outstanding NanousdPrim handle the shim
        has cached, which would cause use-after-free on the next access.
        """
        src = str(path)
        if self._prim.add_reference("", src):
            return True
        # Fallback only if the new C entry isn't present (very old backend).
        if hasattr(self._prim, "add_relationship_target"):
            return self._prim.add_relationship_target("references", src)
        return False

    def AddReference(self, asset_path: str, prim_path: str = "") -> bool:
        """Add an external reference (spec §6.3.5).
        nanousd_add_reference handles its own recompose internally; we
        do not call recompose here for the same handle-invalidation
        reason documented on AddInternalReference."""
        return self._prim.add_reference(asset_path, prim_path)


# ===========================================================================
# Sdf module
# ===========================================================================


class _SdfModule:
    """Compatibility shim for pxr.Sdf."""

    class Path:
        """Thin path wrapper matching Sdf.Path interface."""

        emptyPath = None  # Set after class definition

        def __init__(self, path_str: str = ""):
            self._str = str(path_str)

        def __str__(self) -> str:
            return self._str

        def __repr__(self) -> str:
            return f"Sdf.Path({self._str!r})"

        def __eq__(self, other: object) -> bool:
            if isinstance(other, _SdfModule.Path):
                return self._str == other._str
            if isinstance(other, str):
                return self._str == other
            return NotImplemented

        def __hash__(self) -> int:
            return hash(self._str)

        def __bool__(self) -> bool:
            return bool(self._str)

        def GetText(self) -> str:
            return self._str

        def GetString(self) -> str:
            return self._str

        @property
        def pathString(self) -> str:
            return self._str

        def IsAbsolutePath(self) -> bool:
            return self._str.startswith("/")

        def GetParentPath(self) -> _SdfModule.Path:
            parent = self._str.rsplit("/", 1)[0]
            return _SdfModule.Path(parent if parent else "/")

        def AppendPath(self, suffix: str | _SdfModule.Path) -> _SdfModule.Path:
            s = str(suffix).lstrip("/")
            if self._str == "/":
                return _SdfModule.Path("/" + s)
            return _SdfModule.Path(self._str + "/" + s)

        def AppendChild(self, name: str) -> _SdfModule.Path:
            return self.AppendPath(name)

        def ReplacePrefix(
            self, old_prefix: str | _SdfModule.Path, new_prefix: str | _SdfModule.Path
        ) -> _SdfModule.Path:
            old_str = str(old_prefix)
            new_str = str(new_prefix)
            if self._str == old_str:
                return _SdfModule.Path(new_str)
            if self._str.startswith(old_str + "/"):
                return _SdfModule.Path(new_str + self._str[len(old_str) :])
            return _SdfModule.Path(self._str)

        def GetPrimPath(self) -> _SdfModule.Path:
            # Strip property part (after '.')
            prim = self._str.split(".")[0] if "." in self._str else self._str
            return _SdfModule.Path(prim)

        @property
        def name(self) -> str:
            return self._str.rsplit("/", 1)[-1]

        def GetName(self) -> str:
            return self._str.rsplit("/", 1)[-1]

        def __lt__(self, other: object) -> bool:
            if isinstance(other, _SdfModule.Path):
                return self._str < other._str
            return NotImplemented

        def GetPrefixes(self) -> list[_SdfModule.Path]:
            """Return all ancestor paths including self."""
            parts = self._str.split("/")
            result = []
            for i in range(1, len(parts)):
                result.append(_SdfModule.Path("/".join(parts[: i + 1]) or "/"))
            return result

    class AssetPath:
        """Minimal Sdf.AssetPath shim."""

        def __init__(self, path: str = "", resolved: str = ""):
            self.path = path
            self.resolvedPath = resolved

    class ValueTypeNames:
        """Sdf.ValueTypeNames token stubs for type descriptors."""

        Color3f = "color3f"
        Color3fArray = "color3f[]"
        Float3 = "float3"
        Float3Array = "float3[]"
        Float4 = "float4"
        Float4Array = "float4[]"
        Float2 = "float2"
        Float2Array = "float2[]"
        FloatArray = "float[]"
        Double2 = "double2"
        Double2Array = "double2[]"
        Double3 = "double3"
        Double3Array = "double3[]"
        Double4 = "double4"
        Double4Array = "double4[]"
        DoubleArray = "double[]"
        IntArray = "int[]"
        Int2Array = "int2[]"
        Int3Array = "int3[]"
        Int4Array = "int4[]"
        Token = "token"
        TokenArray = "token[]"
        Float = "float"
        Double = "double"
        Half = "half"
        Int = "int"
        Bool = "bool"
        String = "string"
        StringArray = "string[]"
        Asset = "asset"
        Matrix4d = "matrix4d"
        Quatf = "quatf"
        QuatfArray = "quatf[]"
        Quath = "quath"
        QuathArray = "quath[]"
        Point3f = "point3f"
        Point3fArray = "point3f[]"
        Normal3f = "normal3f"
        Normal3fArray = "normal3f[]"
        Vector3f = "vector3f"
        Vector3fArray = "vector3f[]"
        TexCoord2fArray = "texCoord2f[]"

    class Layer:
        """Sdf.Layer proxy for stage persistence."""

        _registry: dict[str, _LayerProxy] = {}

        @staticmethod
        def Find(path: str) -> _LayerProxy | None:
            """Find an already-opened layer by path."""
            abspath = os.path.abspath(path)
            return _SdfModule.Layer._registry.get(abspath)

        @staticmethod
        def _register(path: str, proxy: _LayerProxy) -> None:
            if path:
                _SdfModule.Layer._registry[os.path.abspath(path)] = proxy

    @staticmethod
    def ComputeAssetPathRelativeToLayer(layer: Any, asset_path: str) -> str | None:
        if layer is not None and hasattr(layer, "realPath") and layer.realPath:
            base = os.path.dirname(layer.realPath)
            return os.path.abspath(os.path.join(base, asset_path))
        return asset_path


class _TokenListOp:
    """Sdf.TokenListOp shim for creating token list operations."""

    def __init__(
        self, prepended: list[str] | None = None, appended: list[str] | None = None, explicit: list[str] | None = None
    ):
        self._prepended = list(prepended) if prepended else []
        self._appended = list(appended) if appended else []
        self._explicit = list(explicit) if explicit else []

    @staticmethod
    def Create(
        prependedItems: list[str] | None = None,
        appendedItems: list[str] | None = None,
        explicitItems: list[str] | None = None,
    ) -> _TokenListOp:
        return _TokenListOp(prependedItems, appendedItems, explicitItems)

    @property
    def prependedItems(self) -> list[str]:
        return self._prepended

    @property
    def appendedItems(self) -> list[str]:
        return self._appended

    @property
    def explicitItems(self) -> list[str]:
        return self._explicit

    def GetAppliedItems(self) -> list[str]:
        return self._prepended + self._appended + self._explicit

    def GetAddedOrExplicitItems(self) -> list[str]:
        return self._prepended + self._appended + self._explicit

    GetComposedItems = GetAppliedItems


_SdfModule.TokenListOp = _TokenListOp

Sdf = _SdfModule()
_SdfModule.Path.emptyPath = _SdfModule.Path("")
_SdfModule.Path.absoluteRootPath = _SdfModule.Path("/")
Sdf.absoluteRootPath = _SdfModule.Path("/")


# ===========================================================================
# ListOp wrapper — used for apiSchemas metadata
# ===========================================================================


class _ListOp:
    """Sdf.ListOp shim backed by the nanousd C API listop functions."""

    def __init__(
        self,
        *,
        prepended: list[str] | None = None,
        appended: list[str] | None = None,
        explicit: list[str] | None = None,
        deleted: list[str] | None = None,
        _composed: list[str] | None = None,
    ):
        self._prepended = list(prepended) if prepended else []
        self._appended = list(appended) if appended else []
        self._explicit = list(explicit) if explicit else []
        self._deleted = list(deleted) if deleted else []
        self._composed = _composed

    @classmethod
    def Create(cls, prependedItems: list[str] | None = None, **kwargs: Any) -> _ListOp:
        return cls(
            prepended=prependedItems,
            appended=kwargs.get("appendedItems"),
            explicit=kwargs.get("explicitItems"),
            deleted=kwargs.get("deletedItems"),
        )

    @classmethod
    def _from_c_handle(cls, handle: Any) -> _ListOp:
        """Read all listop data from a native nanobind ListOp object."""
        if not handle:
            return cls()
        composed = [str(item) for item in getattr(handle, "items", [])]
        if bool(getattr(handle, "is_explicit", False)):
            return cls(explicit=composed, _composed=composed)
        prepended = [str(item) for item in getattr(handle, "prepended_items", [])]
        appended = [str(item) for item in getattr(handle, "appended_items", [])]
        deleted = [str(item) for item in getattr(handle, "deleted_items", [])]
        return cls(prepended=prepended, appended=appended, deleted=deleted, _composed=composed)

    @property
    def prependedItems(self) -> list[str]:
        return self._prepended

    @property
    def appendedItems(self) -> list[str]:
        return self._appended

    @property
    def explicitItems(self) -> list[str]:
        return self._explicit

    @property
    def deletedItems(self) -> list[str]:
        return self._deleted

    def GetComposedItems(self) -> list[str]:
        """Return the fully composed list (prepend + append, minus deletes).

        When constructed from the C API, this is the authoritative composed
        result from ``nanousd_listop_nitems``. For Python-constructed listops
        it falls back to concatenating prepend + append + explicit.
        """
        if self._composed is not None:
            return self._composed
        deleted = set(self._deleted)
        result = self._prepended + self._appended + self._explicit
        return [s for s in result if s not in deleted] if deleted else result

    GetAppliedItems = GetComposedItems
    GetAddedOrExplicitItems = GetComposedItems

    def __bool__(self) -> bool:
        return bool(self.GetComposedItems())

    def __iter__(self):
        return iter(self.GetComposedItems())

    def __len__(self) -> int:
        return len(self.GetComposedItems())

    def __repr__(self) -> str:
        field = getattr(self, "_listop_field", "")
        type_name = {
            "apiSchemas": "SdfTokenListOp",
            "inheritPaths": "SdfPathListOp",
            "inherits": "SdfPathListOp",
            "payload": "SdfPayloadListOp",
            "payloads": "SdfPayloadListOp",
            "references": "SdfReferenceListOp",
            "specializes": "SdfPathListOp",
        }.get(field, "SdfListOp")

        def _fmt(item: str) -> str:
            text = str(item)
            if field in {"inheritPaths", "inherits", "specializes"}:
                return text[1:-1] if text.startswith("<") and text.endswith(">") else text
            if field in {"references", "payload", "payloads"} and text.startswith("@"):
                end = text.find("@", 1)
                if end > 0:
                    asset = text[1:end]
                    tail = text[end + 1:]
                    prim_path = "/"
                    if tail.startswith("<") and tail.endswith(">"):
                        prim_path = tail[1:-1]
                    if field == "references":
                        return f"SdfReference({asset}, {prim_path}, SdfLayerOffset(0, 1), {{}})"
                    return f"SdfPayload({asset}, {prim_path}, SdfLayerOffset(0, 1))"
            return text

        parts: list[str] = []
        if self._explicit:
            parts.append("Explicit Items: [{}]".format(", ".join(_fmt(v) for v in self._explicit)))
        if self._prepended:
            parts.append("Prepended Items: [{}]".format(", ".join(_fmt(v) for v in self._prepended)))
        if self._appended:
            parts.append("Appended Items: [{}]".format(", ".join(_fmt(v) for v in self._appended)))
        if self._deleted:
            parts.append("Deleted Items: [{}]".format(", ".join(_fmt(v) for v in self._deleted)))
        return "{}({})".format(type_name, "; ".join(parts)) if parts else f"{type_name}()"

    __str__ = __repr__


# ===========================================================================
# Gf module — numpy-backed math types
# ===========================================================================


class _GfVec(np.ndarray):
    """numpy array subclass with pxr Gf.Vec methods (GetLength, GetNormalized)."""

    def __eq__(self, other: Any) -> bool:
        try:
            other_arr = np.asarray(other, dtype=self.dtype)
        except Exception:
            return False
        return bool(
            other_arr.shape == self.shape
            and np.array_equal(np.asarray(self), other_arr)
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def GetLength(self) -> float:
        return float(np.linalg.norm(self))

    def GetNormalized(self) -> _GfVec:
        n = np.linalg.norm(self)
        if n > 0:
            return (self / n).view(_GfVec)
        return self.copy().view(_GfVec)

    def GetComplement(self, other: np.ndarray) -> _GfVec:
        """Return component of self orthogonal to other."""
        other = np.asarray(other, dtype=self.dtype)
        d = float(np.dot(self, other))
        n2 = float(np.dot(other, other))
        if n2 > 0:
            return (self - (d / n2) * other).view(_GfVec)
        return self.copy().view(_GfVec)


def _make_gf_vec(arr: np.ndarray) -> _GfVec:
    """Convert a plain ndarray to a _GfVec view."""
    return arr.view(_GfVec)


class _GfModule:
    """Compatibility shim for pxr.Gf."""

    @staticmethod
    def Vec2f(*args: float) -> _GfVec:
        return _make_gf_vec(np.array(args, dtype=np.float32))

    @staticmethod
    def Vec3f(*args: float) -> _GfVec:
        if len(args) == 1:
            return _make_gf_vec(np.full(3, args[0], dtype=np.float32))
        return _make_gf_vec(np.array(args, dtype=np.float32))

    @staticmethod
    def Vec4f(*args: float) -> _GfVec:
        if len(args) == 1:
            return _make_gf_vec(np.full(4, args[0], dtype=np.float32))
        return _make_gf_vec(np.array(args, dtype=np.float32))

    @staticmethod
    def Vec2d(*args: float) -> _GfVec:
        if len(args) == 1:
            return _make_gf_vec(np.full(2, args[0], dtype=np.float64))
        return _make_gf_vec(np.array(args, dtype=np.float64))

    @staticmethod
    def Vec3d(*args: float) -> _GfVec:
        if len(args) == 1:
            return _make_gf_vec(np.full(3, args[0], dtype=np.float64))
        return _make_gf_vec(np.array(args, dtype=np.float64))

    @staticmethod
    def Vec4d(*args: float) -> _GfVec:
        if len(args) == 1:
            return _make_gf_vec(np.full(4, args[0], dtype=np.float64))
        return _make_gf_vec(np.array(args, dtype=np.float64))

    @staticmethod
    def Matrix3f(*args: Any) -> np.ndarray:
        if len(args) == 0:
            return np.eye(3, dtype=np.float32)
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                return args[0].astype(np.float32).reshape(3, 3)
            # Single scalar: s on the diagonal (Gf.Matrix3f(s) == s*I), NOT a
            # full matrix of s. (Gf.Matrix3f(0.0) -> zero matrix; (1.0) -> identity.)
            return np.eye(3, dtype=np.float32) * float(args[0])
        if len(args) == 9:
            return np.array(args, dtype=np.float32).reshape(3, 3)
        return np.eye(3, dtype=np.float32)

    @staticmethod
    def Matrix4d(*args: Any) -> np.ndarray:
        # NOTE: this underlying factory is replaced at import time by the richer
        # `_Matrix4dArr` subclass in __init__.py (`_Gf.Matrix4d = _Matrix4dArr`);
        # that wrapper is the one consumers actually get. Kept minimal here.
        if len(args) == 0:
            return np.eye(4, dtype=np.float64)
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            return args[0].astype(np.float64).reshape(4, 4)
        if len(args) == 16:
            return np.array(args, dtype=np.float64).reshape(4, 4)
        return np.eye(4, dtype=np.float64)

    class Quatf:
        """Minimal quaternion wrapper (real, imaginary)."""

        def __init__(self, real: Any = 1.0, i: float = 0.0, j: float = 0.0, k: float = 0.0):
            if hasattr(real, "GetReal") and hasattr(real, "GetImaginary"):
                self.real = float(real.GetReal())
                self.imaginary = np.asarray(real.GetImaginary(), dtype=np.float32).ravel()[:3].copy()
                return
            self.real = float(real)
            self.imaginary = np.array([float(i), float(j), float(k)], dtype=np.float32)

        def GetReal(self) -> float:
            return self.real

        def GetImaginary(self) -> np.ndarray:
            return self.imaginary

        def Normalize(self) -> _GfModule.Quatf:
            n = math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))
            if n > 0:
                self.real /= n
                self.imaginary /= n
            return self

        def GetNormalized(self) -> _GfModule.Quatf:
            q = _GfModule.Quatf(self.real, float(self.imaginary[0]), float(self.imaginary[1]), float(self.imaginary[2]))
            q.Normalize()
            return q

        def GetLength(self) -> float:
            return math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))

        def __iter__(self):
            yield self.imaginary[0]
            yield self.imaginary[1]
            yield self.imaginary[2]
            yield self.real

    class Quatd:
        """Minimal quaternion wrapper (real, imaginary) for doubles."""

        def __init__(self, real: Any = 1.0, i: float = 0.0, j: float = 0.0, k: float = 0.0):
            if hasattr(real, "GetReal") and hasattr(real, "GetImaginary"):
                self.real = float(real.GetReal())
                self.imaginary = np.asarray(real.GetImaginary(), dtype=np.float64).ravel()[:3].copy()
                return
            self.real = float(real)
            self.imaginary = np.array([float(i), float(j), float(k)], dtype=np.float64)

        def GetReal(self) -> float:
            return self.real

        def GetImaginary(self) -> np.ndarray:
            return self.imaginary

        def Normalize(self) -> _GfModule.Quatd:
            n = math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))
            if n > 0:
                self.real /= n
                self.imaginary /= n
            return self

        def GetNormalized(self) -> _GfModule.Quatd:
            q = _GfModule.Quatd(self.real, float(self.imaginary[0]), float(self.imaginary[1]), float(self.imaginary[2]))
            q.Normalize()
            return q

        def GetLength(self) -> float:
            return math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))

        def __iter__(self):
            yield self.imaginary[0]
            yield self.imaginary[1]
            yield self.imaginary[2]
            yield self.real

    class Quath:
        """Half-precision quaternion wrapper (real, imaginary)."""

        def __init__(self, real: Any = 1.0, i_or_imag: Any = 0.0, j: float = 0.0, k: float = 0.0):
            # Accept Quath(other_quat) — copy constructor
            if hasattr(real, "GetReal") and hasattr(real, "GetImaginary"):
                self.real = float(real.GetReal())
                self.imaginary = np.asarray(real.GetImaginary(), dtype=np.float32).ravel()[:3].copy()
                return
            self.real = float(real)
            if isinstance(i_or_imag, np.ndarray) or isinstance(i_or_imag, (list, tuple)):
                # Quath(w, Vec3h(x,y,z))
                imag = np.asarray(i_or_imag, dtype=np.float32).ravel()
                self.imaginary = imag[:3] if len(imag) >= 3 else np.zeros(3, dtype=np.float32)
            else:
                self.imaginary = np.array([float(i_or_imag), float(j), float(k)], dtype=np.float32)

        def GetReal(self) -> float:
            return self.real

        def GetImaginary(self) -> np.ndarray:
            return self.imaginary

        def Normalize(self) -> _GfModule.Quath:
            n = math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))
            if n > 0:
                self.real /= n
                self.imaginary /= n
            return self

        def GetNormalized(self) -> _GfModule.Quath:
            q = _GfModule.Quath(self.real, float(self.imaginary[0]), float(self.imaginary[1]), float(self.imaginary[2]))
            q.Normalize()
            return q

        def GetLength(self) -> float:
            return math.sqrt(self.real**2 + float(np.dot(self.imaginary, self.imaginary)))

        def __iter__(self):
            yield self.imaginary[0]
            yield self.imaginary[1]
            yield self.imaginary[2]
            yield self.real

    @staticmethod
    def Vec3h(*args: float) -> _GfVec:
        """Half-precision vec3 (stored as float32)."""
        if len(args) == 1:
            return _make_gf_vec(np.full(3, args[0], dtype=np.float32))
        return _make_gf_vec(np.array(args, dtype=np.float32))

    class Rotation:
        """Gf.Rotation shim — represents a rotation in 3D space."""

        def __init__(self, axis: Any = None, angle: float = 0.0):
            if axis is not None:
                ax = np.asarray(axis, dtype=np.float64).ravel()[:3]
                n = np.linalg.norm(ax)
                if n > 0:
                    ax = ax / n
                self._axis = ax
                self._angle = float(angle)
            else:
                self._axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self._angle = 0.0

        def SetRotateInto(self, from_vec: Any, to_vec: Any) -> _GfModule.Rotation:
            """Set rotation that rotates from_vec into to_vec."""
            f = np.asarray(from_vec, dtype=np.float64).ravel()[:3]
            t = np.asarray(to_vec, dtype=np.float64).ravel()[:3]
            fn = np.linalg.norm(f)
            tn = np.linalg.norm(t)
            if fn > 0:
                f = f / fn
            if tn > 0:
                t = t / tn
            d = float(np.dot(f, t))
            if d >= 1.0 - 1e-10:
                self._axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self._angle = 0.0
            elif d <= -1.0 + 1e-10:
                # 180-degree rotation: pick an orthogonal axis
                if abs(f[0]) < 0.9:
                    perp = np.cross(f, np.array([1.0, 0.0, 0.0]))
                else:
                    perp = np.cross(f, np.array([0.0, 1.0, 0.0]))
                self._axis = perp / np.linalg.norm(perp)
                self._angle = 180.0
            else:
                cross = np.cross(f, t)
                self._axis = cross / np.linalg.norm(cross)
                self._angle = math.degrees(math.acos(max(-1.0, min(1.0, d))))
            return self

        def GetQuat(self) -> _GfModule.Quatd:
            """Return the rotation as a quaternion."""
            half = math.radians(self._angle) * 0.5
            s = math.sin(half)
            c = math.cos(half)
            return _GfModule.Quatd(c, self._axis[0] * s, self._axis[1] * s, self._axis[2] * s)

        def GetAxis(self) -> np.ndarray:
            return self._axis.copy()

        def GetAngle(self) -> float:
            return self._angle

        def _row_vector_matrix3(self) -> np.ndarray:
            """This rotation as a row-vector (p' = p*M) 3x3, matching OpenUSD."""
            q = self.GetQuat()
            w = q.GetReal()
            x, y, z = q.GetImaginary()
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            return np.array([
                [1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy)],
                [2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx)],
                [2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy)],
            ], dtype=np.float64)

        def Decompose(self, axis0: Any, axis1: Any, axis2: Any) -> _GfModule.Vec3d:
            """Euler angles (degrees) about three axes, matching GfRotation::Decompose.

            Returns (t0, t1, t2) such that, in OpenUSD's row-vector convention,
            ``self == Rotation(axis0,t0) * Rotation(axis1,t1) * Rotation(axis2,t2)``.
            Supports distinct cardinal (positive ±X/±Y/±Z) Tait-Bryan sequences —
            covering usdview's free camera (Y,X,Z) and UsdPhysics' (X,Y,Z). A
            repeated or non-cardinal axis raises, rather than silently returning a
            wrong angle. Validated against the usd-core oracle.
            """
            def axis_index(ax) -> int:
                v = np.asarray(ax, dtype=np.float64).ravel()[:3]
                nz = np.flatnonzero(np.abs(v) > 1e-6)
                k = int(np.argmax(np.abs(v)))
                if nz.size != 1 or v[k] < 1.0 - 1e-6:
                    raise NotImplementedError(
                        "Gf.Rotation.Decompose: only positive cardinal axes "
                        "(X/Y/Z) are supported")
                return k

            i, j, k = axis_index(axis0), axis_index(axis1), axis_index(axis2)
            if len({i, j, k}) != 3:
                raise NotImplementedError(
                    "Gf.Rotation.Decompose: repeated-axis (proper-Euler) "
                    "sequences are not supported")

            # Sign of the permutation (i,j,k) of (0,1,2): +1 even, -1 odd.
            perm = [i, j, k]
            sigma = 1.0
            for p in range(3):
                for r in range(p + 1, 3):
                    if perm[p] > perm[r]:
                        sigma = -sigma

            # GfRotation::Decompose returns (t0,t1,t2) with the matrix as the
            # REVERSE-order product M = R(a2,t2)*R(a1,t1)*R(a0,t0). The closed
            # form below extracts the FORWARD product N = R(a0,b0)*R(a1,b1)*R(a2,b2);
            # since M^T = R(a0,-t0)*R(a1,-t1)*R(a2,-t2), feed M^T and negate.
            M = self._row_vector_matrix3().T
            b1 = math.asin(max(-1.0, min(1.0, -sigma * M[i, k])))   # = -t1
            if abs(math.cos(b1)) > 1e-6:
                b0 = math.atan2(sigma * M[j, k], M[k, k])            # = -t0
                b2 = math.atan2(sigma * M[i, j], M[i, i])            # = -t2
            else:
                # Gimbal lock (|t1| = 90 deg): only t0+/-t2 is determined; pin
                # t2 = 0 and fold the combined angle into t0 (OpenUSD convention).
                b2 = 0.0
                b0 = math.atan2(-sigma * M[j, i], M[j, j])
            return _GfModule.Vec3d(math.degrees(-b0), math.degrees(-b1), math.degrees(-b2))

    class Transform:
        """Gf.Transform shim — holds/decomposes an affine transform (T, R, S).

        Constructed from a USD row-major 4x4 (row-vector convention: v' = v·M,
        translation in the last row). Exposes ``GetTranslation``/``GetRotation``/
        ``GetScale`` used by the UsdPhysics mass tests.
        """

        def __init__(self, matrix: Any = None):
            self._translation = np.zeros(3, dtype=np.float64)
            self._scale = np.ones(3, dtype=np.float64)
            self._rotation = _GfModule.Rotation(np.array([0.0, 0.0, 1.0]), 0.0)
            if matrix is not None:
                self.SetMatrix(matrix)

        @staticmethod
        def _matrix3_to_rotation(r_row: np.ndarray) -> "_GfModule.Rotation":
            # r_row is a row-vector rotation; the standard quat-from-matrix formula
            # is column-vector, so operate on the transpose.
            m = np.asarray(r_row, dtype=np.float64).T
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0.0:
                s = math.sqrt(tr + 1.0) * 2.0
                w = 0.25 * s
                x = (m[2, 1] - m[1, 2]) / s
                y = (m[0, 2] - m[2, 0]) / s
                z = (m[1, 0] - m[0, 1]) / s
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
            axis = np.array([x, y, z], dtype=np.float64)
            n = np.linalg.norm(axis)
            if n < 1e-12:
                return _GfModule.Rotation(np.array([0.0, 0.0, 1.0]), 0.0)
            angle = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, w))))
            return _GfModule.Rotation(axis / n, angle)

        def SetMatrix(self, matrix: Any) -> "_GfModule.Transform":
            m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            self._translation = m[3, :3].copy()
            a = m[:3, :3].copy()
            scale = np.linalg.norm(a, axis=1)  # row norms (row-vector convention)
            scale[scale == 0.0] = 1.0
            r = a / scale[:, None]
            if np.linalg.det(r) < 0.0:  # negative (reflection): fold into first axis
                scale[0] = -scale[0]
                r = a / scale[:, None]
            self._scale = scale
            self._rotation = self._matrix3_to_rotation(r)
            return self

        def GetTranslation(self):
            return _GfModule.Vec3d(*self._translation)

        def GetScale(self):
            return _GfModule.Vec3d(*self._scale)

        def GetRotation(self):
            return self._rotation

        def SetTranslation(self, t: Any) -> "_GfModule.Transform":
            self._translation = np.asarray(t, dtype=np.float64).ravel()[:3]
            return self

        def SetScale(self, s: Any) -> "_GfModule.Transform":
            self._scale = np.asarray(s, dtype=np.float64).ravel()[:3]
            return self

        def SetRotation(self, r: Any) -> "_GfModule.Transform":
            self._rotation = r
            return self


Gf = _GfModule()


# ===========================================================================
# Usd module
# ===========================================================================


class _TimeCode:
    """Usd.TimeCode shim."""

    def __init__(self, time: float = 0.0):
        self.time = time

    def __float__(self) -> float:
        return self.time

    @staticmethod
    def Default() -> _TimeCode:
        return _TimeCode(0.0)


class _UsdPrim:
    """Prim wrapper providing pxr.Usd.Prim interface backed by nanousd.Prim."""

    def __init__(self, mu_prim: _MuPrim, stage: _UsdStage | None = None):
        self._prim = mu_prim
        self._stage = stage

    def IsValid(self) -> bool:
        return self._prim.is_valid

    def __bool__(self) -> bool:
        return self._prim.is_valid

    def GetPath(self) -> _SdfModule.Path:
        return _SdfModule.Path(self._prim.path)

    def GetName(self) -> str:
        return self._prim.name

    def GetTypeName(self) -> str:
        return self._prim.type_name

    def IsA(self, type_name: str | type) -> bool:
        if isinstance(type_name, type):
            # Use _usd_type_name if the class defines one (e.g. _Scene -> "PhysicsScene")
            name = getattr(type_name, "_usd_type_name", None)
            if name is None:
                name = type_name.__name__.lstrip("_")
            return self._prim.is_a(name)
        return self._prim.is_a(str(type_name))

    def HasAPI(self, api_name: str | type) -> bool:
        # Accept class objects from UsdPhysics (e.g. UsdPhysics.RigidBodyAPI)
        if isinstance(api_name, type):
            api_name = getattr(api_name, "_api_schema_name", api_name.__name__)
        return self._prim.has_api(str(api_name))

    def HasRelationship(self, name: str) -> bool:
        return self._prim.has_relationship(name)

    def GetChildren(self) -> list[_UsdPrim]:
        if self.IsInstance() and self._stage is not None:
            proto = self.GetPrototype()
            if proto and proto.IsValid():
                children = []
                for child in proto.GetChildren():
                    inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), self.GetPath())
                    inst_child = self._stage.GetPrimAtPath(inst_path)
                    if inst_child and inst_child.IsValid():
                        children.append(inst_child)
                return children
        children = [_UsdPrim(c, self._stage) for c in self._prim.get_children()]
        order = getattr(self._stage, "_prim_order", None)
        if order:
            children.sort(key=lambda c: order.get(c._prim.path, float("inf")))
        return children

    def GetAttribute(self, name: str) -> _Attribute:
        return _Attribute(self._prim, name)

    def HasAttribute(self, name: str) -> bool:
        return self._prim.has_attribute(name)

    def GetAttributes(self) -> list[_Attribute]:
        return [_Attribute(self._prim, n, valid=True) for n in self._prim.get_attribute_names()]

    def GetParent(self) -> _UsdPrim | None:
        path = self._prim.path
        parent_path = path.rsplit("/", 1)[0]
        if not parent_path:
            return None
        if self._stage:
            return self._stage.GetPrimAtPath(parent_path)
        return None

    def GetStage(self) -> _UsdStage | None:
        return self._stage

    def IsActive(self) -> bool:
        return self._prim.is_active

    def SetActive(self, active: bool) -> bool:
        """Activate or deactivate this prim (spec §6.3.6).
        nanousd_set_active recomposes internally and refreshes the
        passed-in prim handle; we do not recompose again here because
        that would invalidate every other NanousdPrim handle the shim
        has cached."""
        return self._prim.set_active(bool(active))

    def SetInstanceable(self, instanceable: bool) -> bool:
        """Mark this prim as instanceable (spec §6.4 — prim metadata)."""
        return self._prim.set_instanceable(bool(instanceable))

    def IsInstanceable(self) -> bool:
        """Whether the ``instanceable`` metadata is authored true (spec §6.4)."""
        val = self.GetMetadata("instanceable")
        return bool(val) if val is not None else False

    def RemoveAPI(self, api_class: type | str) -> bool:
        """Remove an applied API schema (counterpart to ApplyAPI)."""
        name = api_class if isinstance(api_class, str) else getattr(
            api_class, "__name__", str(api_class))
        return self._prim.remove_api(name)

    def Remove(self) -> bool:
        """Delete this prim and its descendants from the root layer."""
        return self._prim.remove_prim()

    def IsInstanceProxy(self) -> bool:
        return self._nearest_instance_ancestor() is not None

    def IsInstance(self) -> bool:
        return bool(getattr(self._prim, "is_instance", False))

    def IsInPrototype(self) -> bool:
        return bool(getattr(self._prim, "is_in_prototype", False))

    def GetPrototype(self) -> _UsdPrim | None:
        proto = self._prim.get_prototype()
        if not proto:
            return None
        return _UsdPrim(proto, self._stage)

    def GetPrimInPrototype(self) -> _UsdPrim | None:
        ancestor = self._nearest_instance_ancestor()
        if ancestor is None:
            return self if self.IsInPrototype() else None
        instance_path, instance_prim = ancestor
        proto = instance_prim.GetPrototype()
        if not proto or not proto.IsValid() or self._stage is None:
            return None
        suffix = self._prim.path[len(instance_path):]
        return self._stage.GetPrimAtPath(str(proto.GetPath()) + suffix)

    def _nearest_instance_ancestor(self) -> tuple[str, _UsdPrim] | None:
        if self._stage is None:
            return None
        path = self._prim.path
        if not path or path == "/":
            return None
        parts = path.strip("/").split("/")
        for i in range(len(parts) - 1, 0, -1):
            ancestor_path = "/" + "/".join(parts[:i])
            ancestor = self._stage.GetPrimAtPath(ancestor_path)
            if ancestor and ancestor.IsValid() and ancestor.IsInstance():
                return ancestor_path, ancestor
        return None

    def CreateAttribute(self, name: str, type_name: Any = None, custom: bool = True) -> _Attribute:
        """Create (or return existing) attribute on this prim."""
        if type_name is not None and self._prim._h:
            tn = str(type_name) if type_name else ""
            self._prim.create_attribute(name, tn)
            return _Attribute(self._prim, name, valid=True, type_name=tn or None)
        return _Attribute(self._prim, name)

    # API schemas that imply a base schema (apply base when derived is applied)
    _API_SCHEMA_DEPS: dict[str, list[str]] = {
        "NewtonCollisionAPI": ["PhysicsCollisionAPI"],
        "NewtonMeshCollisionAPI": ["NewtonCollisionAPI", "PhysicsCollisionAPI"],
        "NewtonRigidBodyAPI": ["PhysicsRigidBodyAPI"],
        "NewtonArticulationAPI": ["PhysicsArticulationRootAPI"],
        "NewtonMaterialAPI": ["PhysicsMaterialAPI"],
    }

    def ApplyAPI(self, api_class: type | str, *args: Any) -> Any:
        """Apply an API schema class to this prim."""
        if isinstance(api_class, str):
            self._prim.apply_api(api_class)
            for dep in self._API_SCHEMA_DEPS.get(api_class, ()):
                if not self._prim.has_api(dep):
                    self._prim.apply_api(dep)
            return None
        schema_name = getattr(api_class, "_api_schema_name", api_class.__name__)
        self._prim.apply_api(schema_name)
        for dep in self._API_SCHEMA_DEPS.get(schema_name, ()):
            if not self._prim.has_api(dep):
                self._prim.apply_api(dep)
        if args:
            return api_class(self, *args)
        return api_class(self)

    def GetAppliedSchemas(self) -> list[str]:
        """Return applied API schema names (e.g. ['PhysicsRigidBodyAPI', 'PhysicsMassAPI'])."""
        # Prefer the native binding method that reads from spec fields
        if hasattr(self._prim, "get_applied_schemas"):
            schemas = self._prim.get_applied_schemas()
            if schemas:
                return [str(s) for s in schemas]
        # Fallback: try reading from attribute (works for file-loaded stages)
        val = self._prim.get_attribute("apiSchemas")
        if val is not None:
            if isinstance(val, (list, tuple)):
                return [str(s) for s in val]
            if isinstance(val, str):
                return [val]
        return []

    def GetAuthoredPropertiesInNamespace(self, namespace: str) -> list[_Attribute]:
        """Return attributes whose names start with the given namespace prefix."""
        return [
            _Attribute(self._prim, n, valid=True, authored=self._prim.has_authored_attribute(n))
            for n in self._prim.get_attribute_names_in_namespace(namespace)
        ]

    def GetAuthoredAttributeValuesInNamespace(self, namespace: str) -> dict[str, Any]:
        """Return authored attribute values for Newton's namespace scan fast path."""
        getter = getattr(self._prim._h, "authored_attribute_values_in_namespace", None)
        if getter is not None:
            return dict(getter(namespace))
        out: dict[str, Any] = {}
        for name in self._prim.get_attribute_names_in_namespace(namespace):
            if not self._prim.has_authored_attribute(name):
                continue
            value = self._prim.get_attribute(name)
            if value is not None:
                out[name] = value
        return out

    def GetAuthoredAttributeValues(self, names: Sequence[str]) -> dict[str, Any]:
        """Return authored values for the requested attribute names."""
        getter = getattr(self._prim._h, "authored_attribute_values", None)
        if getter is not None:
            return dict(getter([str(name) for name in names]))
        out: dict[str, Any] = {}
        for name in names:
            name = str(name)
            attr = self.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                value = attr.Get()
                if value is not None:
                    out[name] = value
        return out

    def GetAuthoredAttributeValuesInNamespaces(self, namespaces: Sequence[str]) -> dict[str, Any]:
        """Return authored attribute values across namespaces with one prim-level cache fill."""
        getter = getattr(self._prim._h, "authored_attribute_values_in_namespaces", None)
        if getter is not None:
            return dict(getter([str(ns) for ns in namespaces]))
        prefixes = tuple(str(ns) if str(ns).endswith(":") else str(ns) + ":" for ns in namespaces)
        out: dict[str, Any] = {}
        for name in self._prim.get_authored_attribute_names():
            if not name.startswith(prefixes):
                continue
            value = self._prim.get_attribute(name)
            if value is not None:
                out[name] = value
        return out

    def GetRelationships(self) -> list[_Relationship]:
        """Return all relationships on this prim.

        Scans attribute names for known relationship patterns (e.g. material:binding,
        physics:body0, prototypes) since nanousd doesn't distinguish them from attributes.
        """
        getter = getattr(self._prim, "get_relationship_names", None)
        if getter is not None:
            return [_Relationship(self._prim, name, valid=True) for name in getter()]
        rels = []
        seen = set()
        _rel_prefixes = (
            "material:binding",
            "physics:body",
            "physics:collisionGroup",
            "prototypes",
            "collection:",
            "proxyPrim",
            "skel:",
        )
        for name in self._prim.get_attribute_names():
            if any(name.startswith(p) or name == p for p in _rel_prefixes):
                seen.add(name)
                rels.append(_Relationship(self._prim, name, valid=True))
            elif self._prim.has_relationship(name):
                seen.add(name)
                rels.append(_Relationship(self._prim, name, valid=True))
        for name in (
            "material:binding",
            "material:binding:preview",
            "material:binding:full",
            "material:binding:physics",
            "outputs:surface.connect",
            "outputs:mdl:surface.connect",
        ):
            if name not in seen and self._prim.has_relationship(name):
                rels.append(_Relationship(self._prim, name, valid=True))
        return rels

    def GetRelationship(self, name: str) -> _Relationship:
        """Return a relationship proxy for the given name."""
        valid = False
        try:
            valid = str(name) in self._prim.get_relationship_names()
        except Exception:
            pass
        return _Relationship(self._prim, name, valid=valid)

    def GetReferences(self) -> _ReferencesProxy:
        """Return a references proxy for adding internal references."""
        return _ReferencesProxy(self._prim, self._stage)

    def GetCustomData(self) -> dict:
        """Read all custom data from prim metadata as a dict."""
        if hasattr(self._prim, "get_prim_metadata"):
            cd = self._prim.get_prim_metadata("customData")
            if isinstance(cd, dict):
                return cd
        return {}

    def GetCustomDataByKey(self, key: str) -> Any:
        """Read custom data from prim metadata."""
        cd = self.GetCustomData()
        return cd.get(key) if cd else None

    def GetMetadata(self, key: str) -> Any:
        """Read prim metadata via the native metadata getters."""
        if key == "apiSchemas":
            handle = self._prim.get_listop("apiSchemas")
            if not handle:
                return None
            return _ListOp._from_c_handle(handle)
        return self._prim.get_prim_metadata(key)

    def SetMetadata(self, key: str, value: Any) -> bool:
        """Write prim metadata via the native metadata setters."""
        if key == "apiSchemas":
            if hasattr(value, "GetComposedItems"):
                items = value.GetComposedItems()
            elif isinstance(value, (list, tuple)):
                items = list(value)
            else:
                items = []
            for schema in items:
                self._prim.apply_api(str(schema))
            return True
        if isinstance(value, bool):
            return self._prim.set_prim_metadata_d(key, 1.0 if value else 0.0)
        if isinstance(value, (int, float)):
            return self._prim.set_prim_metadata_d(key, float(value))
        if isinstance(value, str):
            # Token-typed prim metadata fields per the USD spec
            _TOKEN_METADATA = {"kind", "purpose", "typeName", "specifier", "defaultPrim"}
            if key in _TOKEN_METADATA:
                return self._prim.set_prim_metadata_token(key, value)
            return self._prim.set_prim_metadata_s(key, value)
        return False

    def CreateRelationship(self, name: str, custom: bool = True) -> _Relationship:
        """Create (or return existing) relationship on this prim."""
        return _Relationship(self._prim, name)

    def RemoveProperty(self, name: str) -> bool:
        """Remove a property (attribute or relationship) from this prim."""
        self._prim.clear_attribute(name)
        return True

    # Convenience methods used by pxr_compat wrappers (UsdGeom, UsdShade, etc.)
    def _read_float(self, name: str, fallback: float = 0.0) -> float:
        return self._prim.read_float(name, fallback)

    def _read_int(self, name: str, fallback: int = 0) -> int:
        return self._prim.read_int(name, fallback)

    def _read_string(self, name: str, fallback: str = "") -> str:
        return self._prim.read_string(name, fallback)

    def _read_rel_targets(self, name: str) -> list[str]:
        return self._prim.read_rel_targets(name)

    def _has_rel(self, name: str) -> bool:
        return self._prim.has_relationship(name)

    def _has_connections(self, attr_name: str) -> bool:
        return self._prim.has_connections(attr_name)

    def _read_connections(self, attr_name: str) -> list[str]:
        return self._prim.read_connections(attr_name)


def _create_schema_attr(prim: _UsdPrim, name: str, type_name: Any = "", default: Any = None) -> _Attribute:
    """Create a schema attribute and optionally author a default value."""
    attr = prim.CreateAttribute(name, type_name)
    if default is not None:
        attr.Set(default)
    return attr


def _quat_to_matrix(q: Any) -> np.ndarray:
    if hasattr(q, "GetReal") and hasattr(q, "GetImaginary"):
        w = float(q.GetReal())
        x, y, z = (float(v) for v in q.GetImaginary())
    else:
        arr = np.asarray(q if q is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64).ravel()
        if len(arr) >= 4:
            w, x, y, z = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 0.0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class _PrimDefinition:
    """Small Usd.PrimDefinition shim exposing authored schema property names."""

    def __init__(self, property_names: list[str]):
        self._property_names = list(property_names)

    def GetPropertyNames(self) -> list[str]:
        return list(self._property_names)


_APPLIED_API_PROPERTY_NAMES = {
    "NewtonPDControlAPI": ["newton:kp", "newton:kd"],
    "NewtonPIDControlAPI": ["newton:kp", "newton:ki", "newton:kd"],
    "NewtonNeuralControlAPI": ["newton:modelPath"],
    "NewtonMaxEffortClampingAPI": ["newton:maxEffort"],
    "NewtonDCMotorClampingAPI": [
        "newton:maxEffort",
        "newton:velocityConstant",
        "newton:armature",
    ],
    "NewtonPositionBasedClampingAPI": ["newton:maxEffort"],
    "NewtonActuatorDelayAPI": ["newton:delaySteps"],
}


_CONCRETE_PRIM_PROPERTY_NAMES = {
    "PhysicsScene": [
        "physics:gravityDirection",
        "physics:gravityMagnitude",
        "newton:maxSolverIterations",
        "newton:timeStepsPerSecond",
        "newton:gravityEnabled",
    ],
}


class _SchemaRegistry:
    """Stub for Usd.SchemaRegistry."""

    def FindConcretePrimDefinition(self, type_name: str) -> _PrimDefinition | None:
        names = _CONCRETE_PRIM_PROPERTY_NAMES.get(str(type_name))
        return _PrimDefinition(names) if names is not None else None

    def FindAppliedAPIPrimDefinition(self, schema_name: str) -> _PrimDefinition | None:
        names = _APPLIED_API_PROPERTY_NAMES.get(str(schema_name))
        return _PrimDefinition(names) if names is not None else None


class _PseudoRootPrim:
    """Synthetic pseudo-root when nanousd's get_prim_at_path('/') is invalid."""

    def __init__(self, stage: _UsdStage):
        self._stage_ref = stage

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> _SdfModule.Path:
        return _SdfModule.Path("/")

    def GetChildren(self) -> list[_UsdPrim]:
        children = [_UsdPrim(p, self._stage_ref) for p in self._stage_ref._stage.get_root_prims()]
        order = getattr(self._stage_ref, "_prim_order", None)
        if order:
            children.sort(key=lambda c: order.get(c._prim.path, float("inf")))
        return children

    def GetTypeName(self) -> str:
        return ""

    def GetName(self) -> str:
        return ""


class _UsdStage:
    """Stage wrapper providing pxr.Usd.Stage interface backed by nanousd.Stage."""

    LoadAll = "LoadAll"
    LoadNone = "LoadNone"

    def __init__(self, mu_stage: _MuStage, filepath: str = ""):
        self._stage = mu_stage
        self._filepath = filepath
        self._root_layer: _LayerProxy | None = None
        # Track prim definition order so GetChildren/Traverse return
        # declaration order (matching pxr behaviour) instead of alphabetical.
        self._prim_order: dict[str, int] = {}
        self._composition_errors: list[str] = []

    @staticmethod
    def Open(filepath: str | _LayerProxy, load_policy: Any = None) -> _UsdStage:
        _install_newton_runtime_accelerators()
        if isinstance(filepath, _LayerProxy):
            # Use the layer's live stage (may have been Clear()'d) instead of reopening from disk
            return _UsdStage(filepath._stage, filepath.realPath)
        stage = _UsdStage(_MuStage.open(filepath), filepath)
        # Capture any non-fatal composition warnings from the C layer
        raw_err = stage._stage.error
        if raw_err:
            stage._composition_errors.append(str(raw_err))
        return stage

    def GetPseudoRoot(self) -> _UsdPrim:
        """Return the pseudo-root prim (path '/')."""
        raw = self._stage.get_prim_at_path("/")
        if raw is not None and raw.is_valid:
            return _UsdPrim(raw, self)
        return _PseudoRootPrim(self)

    def GetCompositionErrors(self) -> list:
        """Return composition errors recorded during import/open."""
        return self._composition_errors

    def Traverse(self, predicate: Any = None) -> list[_UsdPrim]:
        result = self._stage.traverse()
        if result:
            prims = [_UsdPrim(p, self) for p in result]
            prims = [p for p in prims if not p.IsInPrototype()]
            if self._prim_order:
                prims.sort(key=lambda p: self._prim_order.get(p._prim.path, float("inf")))
            seen = {p._prim.path for p in prims}
            expanded: list[_UsdPrim] = []
            for p in prims:
                expanded.append(p)
                if p.IsInstance():
                    self._expand_instance_children(p, expanded, seen)
            if len(expanded) > len(prims):
                prims = expanded
            return prims
        # Fallback DFS for in-memory stages where traverse() returns empty
        prims: list[_UsdPrim] = []
        root_prims = list(self._stage.get_root_prims())
        if self._prim_order:
            root_prims.sort(key=lambda p: self._prim_order.get(p.path, float("inf")))
        stack = list(reversed(root_prims))
        while stack:
            raw = stack.pop()
            prims.append(_UsdPrim(raw, self))
            children = list(raw.get_children())
            if self._prim_order:
                children.sort(key=lambda c: self._prim_order.get(c.path, float("inf")))
            stack.extend(reversed(children))
        return prims

    def _expand_instance_children(
        self, prim: _UsdPrim, out: list[_UsdPrim], seen: set[str]
    ) -> None:
        """Recursively add children of an instanceable prim to flat traversal."""
        for child in prim.GetChildren():
            child_path = child._prim.path
            if child_path not in seen:
                seen.add(child_path)
                out.append(child)
                self._expand_instance_children(child, out, seen)

    def GetPrimAtPath(self, path: str | _SdfModule.Path) -> _UsdPrim:
        p = str(path)
        if p == "/":
            return _PseudoRootPrim(self)
        return _UsdPrim(self._stage.get_prim_at_path(p), self)

    def HasDefaultPrim(self) -> bool:
        try:
            dp = self._stage.get_default_prim()
            return dp is not None and dp.is_valid
        except Exception:
            return False

    def GetDefaultPrim(self) -> _UsdPrim:
        return _UsdPrim(self._stage.get_default_prim(), self)

    def GetRootLayer(self) -> _LayerProxy:
        if self._root_layer is None:
            self._root_layer = _LayerProxy(self._filepath, self._stage, owner=self)
        return self._root_layer

    def HasAuthoredMetersPerUnit(self) -> bool:
        return self._stage.get_metadata("metersPerUnit") is not None

    def GetMetersPerUnit(self) -> float:
        val = self._stage.get_metadata("metersPerUnit")
        return float(val) if val is not None else 0.01

    def GetUpAxis(self) -> str:
        val = self._stage.get_metadata("upAxis")
        return str(val) if val is not None else "Y"

    def HasAuthoredKilogramsPerUnit(self) -> bool:
        return self._stage.get_metadata("kilogramsPerUnit") is not None

    def GetKilogramsPerUnit(self) -> float:
        val = self._stage.get_metadata("kilogramsPerUnit")
        return float(val) if val is not None else 1.0

    def GetFramesPerSecond(self) -> float:
        return self._stage.frames_per_second

    def GetTimeCodesPerSecond(self) -> float:
        return self._stage.timecodes_per_second

    def GetStartTimeCode(self) -> float:
        return self._stage.start_time

    def GetEndTimeCode(self) -> float:
        return self._stage.end_time

    # --- Write operations ---

    @staticmethod
    def CreateNew(filepath: str) -> _UsdStage:
        """Create a new empty stage that saves to the given filepath."""
        stage = _UsdStage(_MuStage.create(), filepath)
        _SdfModule.Layer._register(filepath, stage.GetRootLayer())
        return stage

    @staticmethod
    def CreateInMemory() -> _UsdStage:
        """Create an in-memory stage."""
        _install_newton_runtime_accelerators()
        return _UsdStage(_MuStage.create(), "")

    def SetTimeCodesPerSecond(self, tps: float) -> None:
        self._stage.set_timecodes_per_second(tps)

    def SetFramesPerSecond(self, fps: float) -> None:
        self._stage.set_frames_per_second(fps)

    def SetStartTimeCode(self, time: float) -> None:
        self._stage.set_start_time(time)

    def SetEndTimeCode(self, time: float) -> None:
        self._stage.set_end_time(time)

    def SetDefaultPrim(self, prim: _UsdPrim) -> None:
        name = prim.GetName() if hasattr(prim, "GetName") else str(prim)
        self._stage.set_default_prim(name)

    def DefinePrim(self, path: str | _SdfModule.Path, typeName: str = "", type_name: str | None = None) -> _UsdPrim:
        """Define a prim at the given path.

        Accepts OpenUSD's ``typeName`` keyword (the legacy ``type_name`` is
        still honored for existing callers; it wins when both are given).
        """
        resolved_type = type_name if type_name is not None else typeName
        path_str = str(path)
        self._stage.define_prim(path_str, resolved_type)
        if path_str not in self._prim_order:
            self._prim_order[path_str] = len(self._prim_order)
        return _UsdPrim(self._stage.get_prim_at_path(path_str), self)

    def OverridePrim(self, path: str | _SdfModule.Path) -> _UsdPrim:
        """Override a prim at the given path (same as DefinePrim for nanousd)."""
        return self.DefinePrim(path)


class _LayerProxy:
    """Minimal layer proxy for stage.GetRootLayer().realPath etc."""

    # Weak cache: same file path → same _LayerProxy while referenced (allows GC)
    _cache: weakref.WeakValueDictionary[str, _LayerProxy] = weakref.WeakValueDictionary()

    def __new__(cls, filepath: str = "", stage: _MuStage | None = None, owner: _UsdStage | None = None):
        key = os.path.abspath(filepath) if filepath else ""
        if key and key in cls._cache:
            inst = cls._cache[key]
            # Update backing stage and owner
            inst._stage = stage
            if owner is not None:
                inst._owner = owner
            return inst
        inst = super().__new__(cls)
        if key:
            cls._cache[key] = inst
        return inst

    def __init__(self, filepath: str = "", stage: _MuStage | None = None, owner: _UsdStage | None = None):
        self.realPath = os.path.abspath(filepath) if filepath else ""
        # Pixar`s Sdf.Layer exposes .identifier - same as realPath for
        # file-backed layers (anonymous layers use `anon:0xNNNN`).
        self.identifier = self.realPath
        self._stage = stage
        self._owner = owner  # back-reference to update stage pointer

    def __bool__(self) -> bool:
        return bool(self.realPath)

    def Save(self) -> None:
        """Write the stage to disk as USDA."""
        if self._stage and self.realPath:
            path = self.realPath
            # nanousd only writes USDA; ensure .usda extension
            if path.endswith(".usd") and not path.endswith(".usda"):
                path = path[:-4] + ".usda"
                self.realPath = path
            self._stage.write_usda(path)

    def Clear(self) -> None:
        """Clear the layer by replacing the backing stage with a fresh empty one."""
        old = self._stage
        self._stage = _MuStage.create()
        if old:
            old.close()
        if self._owner is not None:
            self._owner._prim_order.clear()

    def ExportToString(self) -> str:
        """Serialize the stage to a USDA string."""
        if self._stage:
            return self._stage.write_usda_string()
        return ""

    def ImportFromString(self, usda_text: str) -> bool:
        """Import USDA text into this layer via a temporary file."""
        import re
        import tempfile

        missing_refs: list[str] = []
        # Greedy capture with [^@\n] cannot overlap the trailing @, so this is
        # provably linear-time (no backtracking) — addresses python:S5852.
        # The .strip() below handles any leading/trailing whitespace inside the
        # captured token, so the regex doesn't need to model it.
        for ref in re.findall(r'@([^@\n]+)@', usda_text):
            ref = ref.strip()
            if not ref or ref.startswith(("./", "../", "/")) is False and "://" in ref:
                continue
            if ref.startswith("/"):
                candidate = ref
            else:
                base = os.path.dirname(self.realPath) if self.realPath else os.getcwd()
                candidate = os.path.abspath(os.path.join(base, ref))
            if not os.path.exists(candidate):
                missing_refs.append(ref)

        try:
            with tempfile.NamedTemporaryFile(suffix=".usda", mode="w", delete=False) as f:
                f.write(usda_text)
                tmp_path = f.name
            new_stage = _MuStage.open(tmp_path)
            os.unlink(tmp_path)
            old = self._stage
            self._stage = new_stage
            if self._owner is not None:
                self._owner._stage = new_stage
                self._owner._prim_order.clear()
                for ref in missing_refs:
                    self._owner._composition_errors.append(f"Unresolved reference: {ref}")
            if old:
                old.close()
            return True
        except Exception as e:
            # Record the error as a composition error on the owning stage
            if self._owner is not None:
                self._owner._composition_errors.append(str(e))
            return False


class _PrimRange:
    """Usd.PrimRange shim — iterates prim and all descendants with pruning support."""

    def __init__(self, root_prim: _UsdPrim, predicate: Any = None):
        self._prims: list[_UsdPrim] = []
        if root_prim and root_prim.IsValid():
            if isinstance(root_prim, _PseudoRootPrim):
                for child in root_prim.GetChildren():
                    self._collect(child)
            else:
                self._collect(root_prim)
        self._index = 0
        self._prune_children = False

    def _collect(self, prim: _UsdPrim) -> None:
        self._prims.append(prim)
        for child in prim.GetChildren():
            self._collect(child)

    def __iter__(self):
        self._index = 0
        self._skip_prefix: str | None = None
        return self

    def __next__(self) -> _UsdPrim:
        while self._index < len(self._prims):
            prim = self._prims[self._index]
            self._index += 1
            # If pruning, skip descendants of the pruned prim
            if self._skip_prefix is not None:
                if str(prim.GetPath()).startswith(self._skip_prefix):
                    continue
                self._skip_prefix = None
            return prim
        raise StopIteration

    def PruneChildren(self) -> None:
        """Skip children of the most recently yielded prim."""
        if self._index > 0:
            self._skip_prefix = str(self._prims[self._index - 1].GetPath()) + "/"


def _TraverseInstanceProxies():
    """Stub predicate for Usd.TraverseInstanceProxies()."""
    return None


class _UsdModule:
    """Compatibility shim for pxr.Usd."""

    Stage = _UsdStage
    Prim = _UsdPrim
    TimeCode = _TimeCode
    SchemaRegistry = _SchemaRegistry
    PrimRange = _PrimRange

    @staticmethod
    def TraverseInstanceProxies():
        return _TraverseInstanceProxies()


Usd = _UsdModule()


# ===========================================================================
# UsdGeom module
# ===========================================================================


class _Tokens:
    """UsdGeom.Tokens shim."""

    faceVarying = "faceVarying"
    vertex = "vertex"
    varying = "varying"
    uniform = "uniform"
    constant = "constant"
    x = "X"
    y = "Y"
    z = "Z"
    purpose = "purpose"
    face = "face"
    leftHanded = "leftHanded"


def _quat_to_matrix4d(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) or (i,j,k,r) to 4x4 rotation matrix.

    Uses row-vector convention (v_out = v_in * M) to match pxr/USD.
    """
    arr = np.asarray(q, dtype=np.float64).ravel()
    if len(arr) < 4:
        return np.eye(4, dtype=np.float64)
    x, y, z, w = arr[0], arr[1], arr[2], arr[3]
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y + z * w)
    m[0, 2] = 2 * (x * z - y * w)
    m[1, 0] = 2 * (x * y - z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z + x * w)
    m[2, 0] = 2 * (x * z + y * w)
    m[2, 1] = 2 * (y * z - x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _euler_to_matrix4d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (degrees) in XYZ order to 4x4 rotation matrix (row-major)."""
    return _single_axis_rotation(0, rx) @ _single_axis_rotation(1, ry) @ _single_axis_rotation(2, rz)


def _single_axis_rotation(axis: int, degrees: float) -> np.ndarray:
    """Build a 4x4 row-major rotation matrix for a single axis (0=X, 1=Y, 2=Z)."""
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float64)
    if axis == 0:  # X
        m[1, 1] = c
        m[1, 2] = s
        m[2, 1] = -s
        m[2, 2] = c
    elif axis == 1:  # Y
        m[0, 0] = c
        m[0, 2] = -s
        m[2, 0] = s
        m[2, 2] = c
    else:  # Z
        m[0, 0] = c
        m[0, 1] = s
        m[1, 0] = -s
        m[1, 1] = c
    return m


def _euler_yxz_to_matrix4d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (degrees) to 4x4 rotation matrix with YXZ application order.

    Args are always (X_angle, Y_angle, Z_angle) — USD float3 is always (X,Y,Z)
    regardless of the rotation order name. Application order: Y first, then X, then Z.
    """
    return _single_axis_rotation(1, ry) @ _single_axis_rotation(0, rx) @ _single_axis_rotation(2, rz)


def _euler_zyx_to_matrix4d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (degrees) to 4x4 rotation matrix with ZYX application order.

    Args are always (X_angle, Y_angle, Z_angle). Application order: Z first, then Y, then X.
    """
    return _single_axis_rotation(2, rz) @ _single_axis_rotation(1, ry) @ _single_axis_rotation(0, rx)


def _get_local_transform_matrix(prim: _UsdPrim) -> tuple[np.ndarray, bool]:
    """Compose local transform from xformOps following xformOpOrder.

    Supports: xformOp:transform, xformOp:translate, xformOp:orient,
    xformOp:rotateXYZ, xformOp:scale, and the full xformOpOrder pipeline.

    Returns:
        (matrix, resetXformStack) where resetXformStack is True when the
        xformOpOrder contains ``!resetXformStack!``, indicating that parent
        transforms should be ignored.
    """
    reset_xform_stack = False

    # Check for single xformOp:transform first (common fast path)
    mat = prim._prim.read_matrix4d("xformOp:transform")
    if mat is not None:
        return mat, False

    # Read xformOpOrder to determine which ops to compose
    order_attr = prim.GetAttribute("xformOpOrder")
    if order_attr and order_attr.IsValid():
        order = order_attr.Get()
        if order is not None:
            if isinstance(order, str):
                ops = [order]
            elif hasattr(order, "__iter__"):
                ops = [str(o) for o in order]
            else:
                ops = None
        else:
            ops = None
    else:
        ops = None

    # Filter out !resetXformStack! sentinel
    if ops is not None:
        if "!resetXformStack!" in ops:
            reset_xform_stack = True
            ops = [o for o in ops if o != "!resetXformStack!"]

    # If no xformOpOrder, try individual ops in standard order
    if ops is None:
        ops = []
        for candidate in (
            "xformOp:transform",
            "xformOp:translate",
            "xformOp:orient",
            "xformOp:rotateXYZ",
            "xformOp:scale",
        ):
            if prim.HasAttribute(candidate):
                ops.append(candidate)
        if not ops:
            return np.eye(4, dtype=np.float64), reset_xform_stack

    # Compose ops: USD row-major convention — each op pre-multiplies into result.
    # Op names may have a suffix (e.g. xformOp:scale:unitsResolve); strip suffix
    # to determine the op type, but read the attribute using the full op name.
    result = np.eye(4, dtype=np.float64)
    for op in ops:
        # Determine op type by stripping optional :suffix after the base type
        base = op
        parts = op.split(":")
        if len(parts) >= 3:
            base = parts[0] + ":" + parts[1]  # e.g. "xformOp:scale"
        if base == "xformOp:transform":
            m = prim._prim.read_matrix4d(op)
            if m is not None:
                result = m @ result
        elif base == "xformOp:translate":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 3:
                    t = np.eye(4, dtype=np.float64)
                    t[3, 0] = arr[0]
                    t[3, 1] = arr[1]
                    t[3, 2] = arr[2]
                    result = t @ result
        elif base == "xformOp:orient":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                if hasattr(val, "GetReal") and hasattr(val, "GetImaginary"):
                    imag = val.GetImaginary()
                    arr = np.array([imag[0], imag[1], imag[2], val.GetReal()], dtype=np.float64)
                else:
                    arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 4:
                    result = _quat_to_matrix4d(arr) @ result
        elif base == "xformOp:rotateXYZ":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 3:
                    result = _euler_to_matrix4d(arr[0], arr[1], arr[2]) @ result
        elif base == "xformOp:scale":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 3:
                    s = np.eye(4, dtype=np.float64)
                    s[0, 0] = arr[0]
                    s[1, 1] = arr[1]
                    s[2, 2] = arr[2]
                    result = s @ result
        elif base == "xformOp:rotateYXZ":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 3:
                    result = _euler_yxz_to_matrix4d(arr[0], arr[1], arr[2]) @ result
        elif base == "xformOp:rotateZYX":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 3:
                    result = _euler_zyx_to_matrix4d(arr[0], arr[1], arr[2]) @ result
        elif base == "xformOp:rotateX":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                result = _euler_to_matrix4d(float(val), 0.0, 0.0) @ result
        elif base == "xformOp:rotateY":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                result = _euler_to_matrix4d(0.0, float(val), 0.0) @ result
        elif base == "xformOp:rotateZ":
            val = prim.GetAttribute(op).Get()
            if val is not None:
                result = _euler_to_matrix4d(0.0, 0.0, float(val)) @ result
    return result, reset_xform_stack


def _extract_scale_from_xform_ops(prim: _UsdPrim) -> np.ndarray:
    """Extract the product of all scale xformOps from a prim, ignoring rotation.

    Reads the xformOpOrder and multiplies together all scale ops (including
    suffixed variants like xformOp:scale:unitsResolve).  When xformOp:transform
    is the only op, falls back to column-norm extraction (correct for uniform
    scaling or pure scale matrices).
    """
    order_attr = prim.GetAttribute("xformOpOrder")
    ops: list[str] | None = None
    if order_attr and order_attr.IsValid():
        order = order_attr.Get()
        if order is not None:
            if isinstance(order, str):
                ops = [order]
            elif hasattr(order, "__iter__"):
                ops = [str(o) for o in order]

    if ops is not None:
        scale = np.ones(3, dtype=np.float64)
        for op in ops:
            parts = op.split(":")
            base = parts[0] + ":" + parts[1] if len(parts) >= 2 else op
            if base == "xformOp:scale":
                val = prim.GetAttribute(op).Get()
                if val is not None:
                    s = np.asarray(val, dtype=np.float64).ravel()[:3]
                    scale *= np.abs(s)
        return scale

    # Fallback for xformOp:transform or no xformOpOrder: column norms.
    local_mat, _ = _get_local_transform_matrix(prim)
    sx = np.linalg.norm(local_mat[:3, 0])
    sy = np.linalg.norm(local_mat[:3, 1])
    sz = np.linalg.norm(local_mat[:3, 2])
    if sx == 0.0 or sy == 0.0 or sz == 0.0:
        return np.ones(3, dtype=np.float64)
    return np.array([sx, sy, sz], dtype=np.float64)


def _c_local_transform(prim_h: Any, time: float = 0.0) -> tuple[np.ndarray, bool]:
    """Read local transform from the native binding. Returns (matrix4x4, resetXformStack)."""
    native_prim = prim_h
    while hasattr(native_prim, "_h") and not (
        hasattr(native_prim, "local_transform_info_at")
        or hasattr(native_prim, "local_transform_at")
    ):
        native_prim = native_prim._h
    if native_prim:
        if hasattr(native_prim, "local_transform_info_at"):
            matrix, reset = native_prim.local_transform_info_at(float(time))
            if matrix is not None:
                return np.array(matrix, dtype=np.float64).reshape(4, 4), bool(reset)
        matrix = native_prim.local_transform_at(float(time))
        if matrix is not None:
            return np.array(matrix, dtype=np.float64).reshape(4, 4), False
    return np.eye(4, dtype=np.float64), False


def _get_world_transform_matrix(
    stage: _UsdStage,
    prim: _UsdPrim,
    time: float = 0.0,
) -> np.ndarray:
    """Walk up hierarchy composing transforms: world = local @ parent @ ... @ root.

    Uses row-vector convention (v_out = v_in * M) to match pxr/USD.
    """
    result, reset = _c_local_transform(prim._prim, time)
    if reset:
        return result
    path = prim._prim.path
    while True:
        parent_path = path.rsplit("/", 1)[0]
        if not parent_path:
            break
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            break
        parent_mat, parent_reset = _c_local_transform(parent._prim, time)
        result = result @ parent_mat  # row-vector: local @ parent
        if parent_reset:
            break
        path = parent_path
    return result


class _XformCache:
    """UsdGeom.XformCache shim."""

    def __init__(self, time_code: _TimeCode | None = None):
        self._cache: dict[str, np.ndarray] = {}
        if time_code is None:
            self._time = 0.0
        elif isinstance(time_code, _TimeCode):
            self._time = time_code.time
        else:
            self._time = float(time_code)

    def Clear(self) -> None:
        self._cache.clear()

    def SetTime(self, time_code: _TimeCode | float) -> None:
        if isinstance(time_code, _TimeCode):
            self._time = time_code.time
        else:
            self._time = float(time_code)
        self._cache.clear()

    def GetLocalToWorldTransform(self, prim: _UsdPrim) -> np.ndarray:
        path_str = str(prim.GetPath())
        if path_str not in self._cache:
            stage = prim.GetStage()
            if stage:
                self._cache[path_str] = _get_world_transform_matrix(
                    stage,
                    prim,
                    self._time,
                )
            else:
                mat, _ = _c_local_transform(prim._prim, self._time)
                self._cache[path_str] = mat
        return self._cache[path_str]


class _XformOp:
    """UsdGeom.XformOp shim — wraps a single xform op for read/write access.

    Op-type and precision are represented as strings (a consistent enum space):
    ``GetOpType()``/``GetPrecision()`` return values that compare equal to the
    ``Type*``/``Precision*`` class constants below, matching OpenUSD usage like
    ``op.GetOpType() == UsdGeom.XformOp.TypeTranslate``.
    """

    # Op-type constants (string-valued; same space as the internal op_type)
    TypeInvalid = "invalid"
    TypeTranslate = "translate"
    TypeScale = "scale"
    TypeRotateX = "rotateX"
    TypeRotateY = "rotateY"
    TypeRotateZ = "rotateZ"
    TypeRotateXYZ = "rotateXYZ"
    TypeRotateXZY = "rotateXZY"
    TypeRotateYXZ = "rotateYXZ"
    TypeRotateYZX = "rotateYZX"
    TypeRotateZXY = "rotateZXY"
    TypeRotateZYX = "rotateZYX"
    TypeOrient = "orient"
    TypeTransform = "transform"
    # Single-axis translate/scale op-type constants (no Add*Op constructors are
    # exposed yet — see the AddRotate{X,Y,Z}Op note below — but the constants
    # let callers compare GetOpType() of externally-authored single-axis ops).
    TypeTranslateX = "translateX"
    TypeTranslateY = "translateY"
    TypeTranslateZ = "translateZ"
    TypeScaleX = "scaleX"
    TypeScaleY = "scaleY"
    TypeScaleZ = "scaleZ"

    # Precision constants
    PrecisionDouble = "double"
    PrecisionFloat = "float"
    PrecisionHalf = "half"

    def __init__(self, prim: _UsdPrim, attr_name: str, op_type: str,
                 isInverseOp: bool = False, precision: str = "float"):
        self._prim = prim
        self._attr_name = attr_name
        self._op_type = op_type
        self._isInverseOp = bool(isInverseOp)
        self._precision = precision

    def __bool__(self) -> bool:
        return self._op_type != _XformOp.TypeInvalid and self._prim.IsValid()

    def Set(self, value: Any, time: float | None = None) -> bool:
        ok = _set_attribute_value(self._prim._prim, self._attr_name, value, time)
        if ok:
            # Keep HasAuthoredValue()/attr caches consistent after the write
            # (these direct setters previously bypassed cache invalidation).
            self._prim._prim._mark_attribute_authored(self._attr_name)
        return ok

    def Get(self, time: float | None = None) -> Any:
        attr = self._prim.GetAttribute(self._attr_name)
        if attr and attr.IsValid():
            return attr.Get(time)
        return None

    def GetOpType(self) -> str:
        return self._op_type

    def GetOpName(self) -> str:
        """Full op token, e.g. ``xformOp:translate`` or ``!invert!xformOp:...``."""
        return ("!invert!" + self._attr_name) if self._isInverseOp else self._attr_name

    def GetName(self) -> str:
        return self._attr_name

    def GetAttr(self) -> _Attribute:
        return self._prim.GetAttribute(self._attr_name)

    def IsInverseOp(self) -> bool:
        return self._isInverseOp

    def GetPrecision(self) -> str:
        return self._precision

    def GetTimeSamples(self) -> list[float]:
        attr = self.GetAttr()
        return attr.GetTimeSamples() if attr and attr.IsValid() else []

    def MightBeTimeVarying(self) -> bool:
        return len(self.GetTimeSamples()) > 1

    def HasSuffix(self) -> bool:
        return self._attr_name.count(":") > 1


class _Xformable:
    """UsdGeom.Xformable shim."""

    def __init__(self, prim: _UsdPrim | None = None):
        # OpenUSD allows a default-constructed (invalid) schema, e.g.
        # ``UsdGeom.Xformable()``; keep that from raising at construction.
        self._prim = prim

    def __bool__(self) -> bool:
        return self._prim is not None and self._prim.IsValid()

    def GetLocalTransformation(self, time: float = 0.0) -> np.ndarray:
        mat, ok_reset = _c_local_transform(self._prim._prim, float(time))
        if mat is not None:
            return mat
        # Fallback to Python-side composition
        mat, _ = _get_local_transform_matrix(self._prim)
        return mat

    def ComputeLocalToWorldTransform(self, time: Any = None) -> np.ndarray:
        t = float(time) if time is not None else 0.0
        stage = self._prim.GetStage()
        if stage is not None:
            return _get_world_transform_matrix(stage, self._prim, t)
        return self.GetLocalTransformation(t)

    def GetWorldTransformation(self, time: Any = None) -> np.ndarray:
        """Alias for ComputeLocalToWorldTransform (Kamino compat)."""
        return self.ComputeLocalToWorldTransform(time)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def HasAttribute(self, name: str) -> bool:
        return self._prim is not None and self._prim.HasAttribute(name)

    def GetAttribute(self, name: str) -> _Attribute:
        return self._prim.GetAttribute(name)

    # ---- xform op construction ---------------------------------------------
    @staticmethod
    def _op_attr_type(base: str, precision: str) -> str:
        """USD attribute type name for an op of (base shape, precision)."""
        if base == "matrix":
            return "matrix4d"
        if base == "quat":
            return {"double": "quatd", "float": "quatf", "half": "quath"}[precision]
        return precision + ("3" if base == "3" else "")

    def _add_op(self, op_type: str, base: str, token: str, default_prec: str,
                precision: str | None = None, opSuffix: str = "",
                isInverseOp: bool = False) -> _XformOp:
        prec = precision or default_prec
        name = "xformOp:" + token
        if opSuffix:
            name = name + ":" + opSuffix
        self._prim._prim.create_attribute(name, self._op_attr_type(base, prec))
        _append_xform_op(self._prim, name, isInverseOp)
        return _XformOp(self._prim, name, op_type, isInverseOp=isInverseOp, precision=prec)

    def AddTranslateOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeTranslate, "3", "translate", "double", precision, opSuffix, isInverseOp)

    def AddScaleOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeScale, "3", "scale", "float", precision, opSuffix, isInverseOp)

    def AddOrientOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeOrient, "quat", "orient", "float", precision, opSuffix, isInverseOp)

    def AddTransformOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeTransform, "matrix", "transform", "double", precision, opSuffix, isInverseOp)

    # Euler-order rotate ops (float3, default precision Float).
    def AddRotateXYZOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateXYZ, "3", "rotateXYZ", "float", precision, opSuffix, isInverseOp)

    def AddRotateXZYOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateXZY, "3", "rotateXZY", "float", precision, opSuffix, isInverseOp)

    def AddRotateYXZOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateYXZ, "3", "rotateYXZ", "float", precision, opSuffix, isInverseOp)

    def AddRotateYZXOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateYZX, "3", "rotateYZX", "float", precision, opSuffix, isInverseOp)

    def AddRotateZXYOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateZXY, "3", "rotateZXY", "float", precision, opSuffix, isInverseOp)

    def AddRotateZYXOp(self, precision: str | None = None, opSuffix: str = "", isInverseOp: bool = False) -> _XformOp:
        return self._add_op(_XformOp.TypeRotateZYX, "3", "rotateZYX", "float", precision, opSuffix, isInverseOp)

    # NOTE: single-axis AddRotate{X,Y,Z}Op are intentionally omitted for now.
    # They unblock testUsdGeomPointInstancer.setUp, which then authors a
    # `prototypes` relationship whose targets carry reference arcs — a code
    # path that segfaults inside nanousd core (both nanousd_set_reltargets and
    # nanousd_add_reltarget). See [[reference_nanousd_core_reltarget_ref_segfault]].
    # Re-add once that core bug is fixed (the Type{RotateX,Y,Z} constants and
    # the _add_op plumbing are already in place).

    def ClearXformOpOrder(self) -> None:
        """Clear existing xform ops by removing the xformOpOrder attribute."""
        self._prim._prim.clear_attribute("xformOpOrder")

    def GetXformOpOrderAttr(self) -> _Attribute:
        """Return the xformOpOrder attribute (creating the accessor handle)."""
        return self._prim.GetAttribute("xformOpOrder")

    def SetXformOpOrder(self, ordered_ops: list, resetXformStack: bool = False) -> bool:
        """Author xformOpOrder from a list of XformOp (or op-name strings)."""
        tokens: list[str] = []
        if resetXformStack:
            tokens.append("!resetXformStack!")
        for op in ordered_ops:
            tokens.append(op.GetOpName() if isinstance(op, _XformOp) else str(op))
        self._prim._prim.set_attribute_token_array("xformOpOrder", tokens)
        self._prim._prim._mark_attribute_authored("xformOpOrder")
        return True

    def MakeMatrixXform(self) -> _XformOp:
        """Clear the op stack and replace it with a single matrix (transform) op."""
        self.ClearXformOpOrder()
        return self.AddTransformOp()

    def GetResetXformStack(self) -> bool:
        """Return True if !resetXformStack! is present in xformOpOrder."""
        order_attr = self._prim.GetAttribute("xformOpOrder")
        if not order_attr or not order_attr.IsValid():
            return False
        order = order_attr.Get()
        if order is None:
            return False
        return "!resetXformStack!" in order

    def GetOrderedXformOps(self) -> list[_XformOp]:
        """Return existing xform ops in authored order (inverse-op aware)."""
        order_attr = self._prim.GetAttribute("xformOpOrder")
        if not order_attr or not order_attr.IsValid():
            return []
        order = order_attr.Get()
        if order is None:
            return []
        if isinstance(order, np.ndarray):
            raw_names = [str(o) for o in order.ravel()]
        elif isinstance(order, (list, tuple)):
            raw_names = [str(o) for o in order]
        elif isinstance(order, str):
            raw_names = [order]
        else:
            try:
                raw_names = [str(o) for o in order]
            except TypeError:
                return []
        ops = []
        for raw in raw_names:
            if raw == "!resetXformStack!":
                continue
            inverse = raw.startswith("!invert!")
            name = raw[len("!invert!"):] if inverse else raw
            # op type is the token directly under "xformOp:" (ignores any suffix)
            parts = name.split(":")
            op_type = parts[1] if len(parts) > 1 and parts[0] == "xformOp" else name
            attr = self._prim.GetAttribute(name)
            prec = _prec_from_type_name(attr.GetTypeName()) if (attr and attr.IsValid()) else "float"
            ops.append(_XformOp(self._prim, name, op_type, isInverseOp=inverse, precision=prec))
        return ops

    @classmethod
    def Get(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Xformable:
        """Return the schema wrapping the prim already at ``path``."""
        return cls(stage.GetPrimAtPath(str(path)))

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Xformable:
        """Define an Xform prim at the given path."""
        prim = stage.DefinePrim(str(path), "Xform")
        return cls(prim)


def _prec_from_type_name(type_name: str) -> str:
    """Map a USD attribute type name to an XformOp precision token."""
    tn = (type_name or "").lower()
    if tn.startswith("double") or tn == "quatd" or tn.startswith("matrix"):
        return "double"
    if tn.startswith("half") or tn == "quath":
        return "half"
    return "float"


def _append_xform_op(prim: _UsdPrim, op_name: str, is_inverse: bool = False) -> None:
    """Add an op token to the xformOpOrder token array (dedup, inverse-aware)."""
    token = ("!invert!" + op_name) if is_inverse else op_name
    order_attr = prim.GetAttribute("xformOpOrder")
    current = []
    if order_attr and order_attr.IsValid():
        val = order_attr.Get()
        if isinstance(val, np.ndarray):
            current = [str(v) for v in val.ravel()]
        elif isinstance(val, (list, tuple)):
            current = [str(v) for v in val]
        elif isinstance(val, str) and val:
            current = [val]
    if token not in current:
        current.append(token)
    prim._prim.set_attribute_token_array("xformOpOrder", current)


class _Mesh(_Xformable):
    """UsdGeom.Mesh shim."""

    # Marker value for infinitely-sharp creases/corners (matches OpenUSD).
    SHARPNESS_INFINITE = 10.0

    def __init__(self, prim: _UsdPrim):
        _install_newton_runtime_accelerators()
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Mesh:
        prim = stage.DefinePrim(str(path), "Mesh")
        return cls(prim)

    def GetPointsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("points")

    def CreatePointsAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "points", "point3f[]", default)

    def GetFaceVertexIndicesAttr(self) -> _Attribute:
        return self._prim.GetAttribute("faceVertexIndices")

    def CreateFaceVertexIndicesAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "faceVertexIndices", "int[]", default)

    def GetFaceVertexCountsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("faceVertexCounts")

    def CreateFaceVertexCountsAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "faceVertexCounts", "int[]", default)

    def GetNormalsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("normals")

    def CreateNormalsAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "normals", "normal3f[]", default)

    def CreateDoubleSidedAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "doubleSided", "bool", default)

    def GetNormalsInterpolation(self) -> str:
        meta = self._prim._prim.get_attribute_metadata("normals", "interpolation")
        if meta is not None:
            return str(meta)
        return self._prim._read_string("normals:interpolation", "vertex")

    def SetNormalsInterpolation(self, interp: str) -> None:
        if not self._prim._prim.has_attribute("normals:interpolation"):
            self._prim._prim.create_attribute("normals:interpolation", "token")
        self._prim._prim.set_attribute_token("normals:interpolation", str(interp))

    def GetOrientationAttr(self) -> _Attribute:
        return self._prim.GetAttribute("orientation")

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetVisibilityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("visibility")


class _TetMesh(_Xformable):
    """UsdGeom.TetMesh shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    def GetPointsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("points")

    def GetTetVertexIndicesAttr(self) -> _Attribute:
        return self._prim.GetAttribute("tetVertexIndices")

    def GetOrientationAttr(self) -> _Attribute:
        return self._prim.GetAttribute("orientation")

    def CreatePointsAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "points", "point3f[]", default)

    def CreateTetVertexIndicesAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "tetVertexIndices", "int4[]", default)


class _Subset:
    """UsdGeom.Subset stub."""

    pass


class _PointInstancer(_Xformable):
    """UsdGeom.PointInstancer shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _PointInstancer:
        prim = stage.DefinePrim(str(path), "PointInstancer")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetPositionsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("positions")

    def GetOrientationsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("orientations")

    def GetScalesAttr(self) -> _Attribute:
        return self._prim.GetAttribute("scales")

    def GetProtoIndicesAttr(self) -> _Attribute:
        return self._prim.GetAttribute("protoIndices")

    def GetVisibilityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("visibility")

    def CreateIdsAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "ids", "int64[]", default)

    def CreateProtoIndicesAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "protoIndices", "int[]", default)

    def GetPrototypesRel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "prototypes")

    def CreatePrototypesRel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "prototypes")


class _Points(_Xformable):
    """UsdGeom.Points shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Points:
        prim = stage.DefinePrim(str(path), "Points")
        return cls(prim)

    @classmethod
    def Get(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Points:
        prim = stage.GetPrimAtPath(str(path))
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetPointsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("points")

    def GetWidthsAttr(self) -> _Attribute:
        return self._prim.GetAttribute("widths")

    def GetDisplayColorAttr(self) -> _Attribute:
        return self._prim.GetAttribute("primvars:displayColor")

    def GetVisibilityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("visibility")


class _Capsule(_Xformable):
    """UsdGeom.Capsule shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Capsule:
        prim = stage.DefinePrim(str(path), "Capsule")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radius")

    def CreateRadiusAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "radius", "double", default)

    def GetHeightAttr(self) -> _Attribute:
        return self._prim.GetAttribute("height")

    def CreateHeightAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "height", "double", default)

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")

    def CreateAxisAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "axis", "token", default)


class _Capsule_1(_Xformable):
    """UsdGeom.Capsule_1 shim (USD 24.08+ capsule with separate top/bottom radii)."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Capsule_1:
        prim = stage.DefinePrim(str(path), "Capsule_1")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusTopAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radiusTop")

    def GetRadiusBottomAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radiusBottom")

    def GetHeightAttr(self) -> _Attribute:
        return self._prim.GetAttribute("height")

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")


class _Cylinder_1(_Xformable):
    """UsdGeom.Cylinder_1 shim (USD 24.08+ cylinder with separate top/bottom radii)."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Cylinder_1:
        prim = stage.DefinePrim(str(path), "Cylinder_1")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusTopAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radiusTop")

    def GetRadiusBottomAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radiusBottom")

    def GetHeightAttr(self) -> _Attribute:
        return self._prim.GetAttribute("height")

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")


class _BasisCurves(_Xformable):
    """UsdGeom.BasisCurves shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _BasisCurves:
        prim = stage.DefinePrim(str(path), "BasisCurves")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()


class _Cone(_Xformable):
    """UsdGeom.Cone shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Cone:
        prim = stage.DefinePrim(str(path), "Cone")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radius")

    def CreateRadiusAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "radius", "double", default)

    def GetHeightAttr(self) -> _Attribute:
        return self._prim.GetAttribute("height")

    def CreateHeightAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "height", "double", default)

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")

    def CreateAxisAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "axis", "token", default)


class _Cube(_Xformable):
    """UsdGeom.Cube shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Cube:
        prim = stage.DefinePrim(str(path), "Cube")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetSizeAttr(self) -> _Attribute:
        return self._prim.GetAttribute("size")

    def CreateSizeAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "size", "double", default)


class _Cylinder(_Xformable):
    """UsdGeom.Cylinder shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Cylinder:
        prim = stage.DefinePrim(str(path), "Cylinder")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radius")

    def CreateRadiusAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "radius", "double", default)

    def GetHeightAttr(self) -> _Attribute:
        return self._prim.GetAttribute("height")

    def CreateHeightAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "height", "double", default)

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")

    def CreateAxisAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "axis", "token", default)


class _Sphere(_Xformable):
    """UsdGeom.Sphere shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Sphere:
        prim = stage.DefinePrim(str(path), "Sphere")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetRadiusAttr(self) -> _Attribute:
        return self._prim.GetAttribute("radius")

    def CreateRadiusAttr(self, default: Any = None, writeSparsely: bool = False) -> _Attribute:
        return _create_schema_attr(self._prim, "radius", "double", default)


class _Plane(_Xformable):
    """UsdGeom.Plane shim."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Plane:
        prim = stage.DefinePrim(str(path), "Plane")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetPath(self) -> _SdfModule.Path:
        return self._prim.GetPath()

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("axis")


class _Gprim(_Xformable):
    """UsdGeom.Gprim shim — base for geometric prims."""

    def __init__(self, prim: _UsdPrim):
        super().__init__(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetDisplayColorAttr(self) -> _Attribute:
        return self._prim.GetAttribute("primvars:displayColor")

    def GetDisplayOpacityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("primvars:displayOpacity")


class _Scope:
    """UsdGeom.Scope shim."""

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Scope:
        prim = stage.DefinePrim(str(path), "Scope")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim


class _Imageable:
    """UsdGeom.Imageable shim."""

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    def __bool__(self) -> bool:
        if self._prim is None or not self._prim.IsValid():
            return False
        type_name = self._prim.GetTypeName()
        return bool(type_name) and type_name not in {
            "PhysicsScene",
            "PhysicsFixedJoint",
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
            "PhysicsSphericalJoint",
            "PhysicsDistanceJoint",
            "PhysicsJoint",
        }

    def GetVisibilityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("visibility")

    def ComputeEffectiveVisibility(self, purpose: str = "default") -> str:
        prim = self._prim
        while prim is not None and prim.IsValid():
            attr = prim.GetAttribute("visibility")
            if attr is not None and attr.IsValid():
                v = attr.Get()
                if v == "invisible":
                    return "invisible"
                if v == "inherited":
                    pass
            prim = prim.GetParent()
        return "inherited"

    def ComputeVisibility(self, time: Any = None) -> str:
        return self.ComputeEffectiveVisibility()


class _PrimvarProxy:
    """Wraps a single primvar attribute for UsdGeom.PrimvarsAPI."""

    def __init__(self, prim_or_attr: _UsdPrim | _Attribute, name: str = ""):
        if isinstance(prim_or_attr, _Attribute):
            # UsdGeom.Primvar(attr) pattern
            self._prim = _UsdPrim(prim_or_attr._prim)
            attr_name = prim_or_attr.GetName()
            if attr_name.startswith("primvars:"):
                self._name = attr_name[len("primvars:") :]
            else:
                self._name = attr_name
            self._attr_name = attr_name
        else:
            self._prim = prim_or_attr
            self._name = name
            self._attr_name = f"primvars:{name}"

    def __bool__(self) -> bool:
        return self._prim.HasAttribute(self._attr_name)

    def Get(self) -> Any:
        attr = self._prim.GetAttribute(self._attr_name)
        if attr and attr.IsValid():
            return attr.Get()
        return None

    def GetInterpolation(self) -> str:
        # Try spec metadata first (e.g., attribute metadata from USDA parenthetical)
        meta = self._prim._prim.get_attribute_metadata(self._attr_name, "interpolation")
        if meta is not None:
            return str(meta)
        # Fallback to child attribute (legacy approach)
        return self._prim._read_string(f"{self._attr_name}:interpolation", "vertex")

    def IsIndexed(self) -> bool:
        return self._prim.HasAttribute(f"{self._attr_name}:indices")

    def GetIndices(self) -> np.ndarray | None:
        attr = self._prim.GetAttribute(f"{self._attr_name}:indices")
        if attr and attr.IsValid():
            return attr.Get()
        return None

    def GetPrimvarName(self) -> str:
        return self._name

    def Set(self, value: Any, time: float | None = None) -> bool:
        """Set the primvar value."""
        ok = _set_attribute_value(self._prim._prim, self._attr_name, value, time)
        if ok:
            self._prim._prim._mark_attribute_authored(self._attr_name)
        return ok

    def SetInterpolation(self, interp: str) -> None:
        meta_name = f"{self._attr_name}:interpolation"
        if not self._prim._prim.has_attribute(meta_name):
            self._prim._prim.create_attribute(meta_name, "token")
        self._prim._prim.set_attribute_token(meta_name, str(interp))

    def SetIndices(self, indices: Any, time: float | None = None) -> bool:
        ok = _set_attribute_value(self._prim._prim, f"{self._attr_name}:indices", indices, time)
        if ok:
            self._prim._prim._mark_attribute_authored(f"{self._attr_name}:indices")
        return ok

    def GetAttr(self) -> _Attribute:
        return self._prim.GetAttribute(self._attr_name)


class _PrimvarsAPI:
    """UsdGeom.PrimvarsAPI shim."""

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    def GetPrimvar(self, name: str) -> _PrimvarProxy:
        return _PrimvarProxy(self._prim, name)

    def CreatePrimvar(
        self, name: str, type_name: Any = None, interpolation: str = "", element_size: int = -1
    ) -> _PrimvarProxy:
        """Create a primvar with optional interpolation."""
        proxy = _PrimvarProxy(self._prim, name)
        if interpolation:
            proxy.SetInterpolation(str(interpolation))
        return proxy

    def GetPrimvarsWithValues(self) -> list[_PrimvarProxy]:
        result = []
        for attr in self._prim.GetAttributes():
            attr_name = attr.GetName()
            if attr_name.startswith("primvars:") and ":indices" not in attr_name and ":interpolation" not in attr_name:
                pv_name = attr_name[len("primvars:") :]
                proxy = _PrimvarProxy(self._prim, pv_name)
                if proxy.Get() is not None:
                    result.append(proxy)
        return result


class _AxisValue:
    """Axis enum value that supports int() for pxr compat and str comparison."""

    def __init__(self, name: str, value: int):
        self._name = name
        self._value = value

    def __int__(self) -> int:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _AxisValue):
            return self._value == other._value
        if isinstance(other, str):
            return self._name == other
        if isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Axis.{self._name}"


class _Axis:
    """UsdPhysics.Axis enum shim — values support int() and str comparison."""

    X = _AxisValue("X", 0)
    Y = _AxisValue("Y", 1)
    Z = _AxisValue("Z", 2)


def _str_to_axis(s: str) -> _AxisValue:
    """Convert axis string to _AxisValue, defaulting to X."""
    return {"X": _Axis.X, "Y": _Axis.Y, "Z": _Axis.Z}.get(s.upper().strip(), _Axis.X)


class _UsdGeomModule:
    """Compatibility shim for pxr.UsdGeom."""

    Xformable = _Xformable
    Xform = _Xformable  # Xform.Define is on _Xformable
    XformCache = _XformCache
    Mesh = _Mesh
    TetMesh = _TetMesh
    Subset = _Subset
    PointInstancer = _PointInstancer
    Points = _Points
    Capsule = _Capsule
    Capsule_1 = _Capsule_1
    BasisCurves = _BasisCurves
    Cone = _Cone
    Cylinder_1 = _Cylinder_1
    Cube = _Cube
    Cylinder = _Cylinder
    Sphere = _Sphere
    Plane = _Plane
    Gprim = _Gprim
    Scope = _Scope
    Imageable = _Imageable
    PrimvarsAPI = _PrimvarsAPI
    Primvar = _PrimvarProxy  # UsdGeom.Primvar(attr) → PrimvarProxy
    Tokens = _Tokens
    XformOp = _XformOp

    @staticmethod
    def StageHasAuthoredMetersPerUnit(stage: _UsdStage) -> bool:
        return stage.HasAuthoredMetersPerUnit()

    @staticmethod
    def GetStageMetersPerUnit(stage: _UsdStage) -> float:
        return stage.GetMetersPerUnit()

    @staticmethod
    def GetStageUpAxis(stage: _UsdStage) -> str:
        return stage.GetUpAxis()

    @staticmethod
    def SetStageUpAxis(stage: _UsdStage, axis: str) -> None:
        stage._stage.set_up_axis(str(axis))

    @staticmethod
    def SetStageMetersPerUnit(stage: _UsdStage, value: float) -> None:
        stage._stage.set_meters_per_unit(value)


UsdGeom = _UsdGeomModule()


# ===========================================================================
# UsdPhysics module
# ===========================================================================


class _MassInformation:
    """Container for collider mass information used by ComputeMassProperties."""

    def __init__(self) -> None:
        self.volume: float = 0.0
        self.centerOfMass: np.ndarray = np.zeros(3, dtype=np.float32)
        self.localPos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.localRot: Any = None  # Gf.Quatf
        self.inertia: np.ndarray = np.zeros((3, 3), dtype=np.float32)


def _read_positive_physics_density(prim: _UsdPrim) -> float | None:
    attr = prim.GetAttribute("physics:density")
    if attr and attr.HasAuthoredValue():
        value = attr.Get()
        if value is not None and float(value) > 0.0:
            return float(value)
    return None


def _collider_density(collider: _UsdPrim, body: _UsdPrim) -> float:
    mass_attr = collider.GetAttribute("physics:mass")
    if mass_attr and mass_attr.HasAuthoredValue():
        # ComputeMassProperties receives unit-density volume in MassInformation.
        # The caller handles the actual volume, so authored mass is converted
        # below after the callback returns.
        pass

    density = _read_positive_physics_density(collider)
    if density is not None:
        return density

    stage = collider.GetStage()
    if stage is not None:
        targets = collider._read_rel_targets("material:binding:physics") or collider._read_rel_targets("material:binding")
        for target in targets:
            mat = stage.GetPrimAtPath(target)
            if mat and mat.IsValid() and (mat.GetTypeName() == "PhysicsMaterial" or mat.HasAPI("PhysicsMaterialAPI")):
                density = _read_positive_physics_density(mat)
                if density is not None:
                    return density

    density = _read_positive_physics_density(body)
    return density if density is not None else 1.0


class _RigidBodyAPI:
    """UsdPhysics.RigidBodyAPI shim — wraps a prim for rigid body operations."""

    _api_schema_name = "PhysicsRigidBodyAPI"

    MassInformation = _MassInformation

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Apply(cls, prim: _UsdPrim) -> _RigidBodyAPI:
        """Apply RigidBodyAPI to a prim (adds to apiSchemas metadata)."""
        prim._prim.apply_api("PhysicsRigidBodyAPI")
        return cls(prim)

    def __bool__(self) -> bool:
        return self._prim.IsValid() and self._prim.HasAPI("PhysicsRigidBodyAPI")

    def CreateKinematicEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:kinematicEnabled", "bool")

    def ComputeMassProperties(self, mass_info_fn: Any = None) -> tuple[float, np.ndarray, np.ndarray, Any]:
        """Compute aggregated mass properties from child colliders.

        Args:
            mass_info_fn: Callback(prim) -> MassInformation for each collider child.

        Returns:
            (mass, inertia_diagonal, center_of_mass, principal_axes)
        """
        total_mass = 0.0
        total_com = np.zeros(3, dtype=np.float64)
        total_inertia = np.zeros((3, 3), dtype=np.float64)

        stage = self._prim.GetStage()
        if stage is None or mass_info_fn is None:
            return -1.0, np.zeros(3), np.zeros(3), _GfModule.Quatf(1.0, 0.0, 0.0, 0.0)

        # Gather mass info from child colliders
        collider_infos = []
        for child in _PrimRange(self._prim):
            if child.GetPath() == self._prim.GetPath():
                continue
            if child.HasAPI("PhysicsCollisionAPI"):
                info = mass_info_fn(child)
                if info is not None and info.volume > 0.0:
                    collider_infos.append((child, info))

        if not collider_infos:
            return -1.0, np.zeros(3), np.zeros(3), _GfModule.Quatf(1.0, 0.0, 0.0, 0.0)

        # Aggregate
        weighted_infos = []
        for collider, info in collider_infos:
            density = _collider_density(collider, self._prim)
            if info.volume > 0.0:
                mass_attr = collider.GetAttribute("physics:mass")
                if mass_attr and mass_attr.HasAuthoredValue():
                    authored_mass = float(mass_attr.Get())
                    if authored_mass > 0.0:
                        density = authored_mass / float(info.volume)
            m = float(info.volume) * density
            rot = _quat_to_matrix(info.localRot)
            com = np.asarray(info.centerOfMass, dtype=np.float64).ravel()[:3]
            local_pos = np.asarray(info.localPos, dtype=np.float64).ravel()[:3]
            com_world = local_pos + rot @ com
            inertia = np.asarray(info.inertia, dtype=np.float64).reshape(3, 3) * density
            inertia = rot @ inertia @ rot.T
            weighted_infos.append((m, com_world, inertia))
            total_mass += m
            total_com += m * com_world

        if total_mass > 0.0:
            total_com /= total_mass

        # Parallel axis theorem for inertia
        for m, com_world, I in weighted_infos:
            d = com_world - total_com
            total_inertia += I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        inertia_diag = np.diag(total_inertia)
        return total_mass, inertia_diag, total_com, _GfModule.Quatf(1.0, 0.0, 0.0, 0.0)


class _ArticulationRootAPI:
    """UsdPhysics.ArticulationRootAPI sentinel class for HasAPI() calls."""

    _api_schema_name = "PhysicsArticulationRootAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Apply(cls, prim: _UsdPrim) -> _ArticulationRootAPI:
        """Apply ArticulationRootAPI to a prim (adds to apiSchemas metadata)."""
        prim._prim.apply_api("PhysicsArticulationRootAPI")
        return cls(prim)

    def __bool__(self) -> bool:
        return self._prim.IsValid() and self._prim.HasAPI("PhysicsArticulationRootAPI")


class _CollisionAPI:
    """UsdPhysics.CollisionAPI shim."""

    _api_schema_name = "PhysicsCollisionAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim
        self._valid = prim.HasAPI("PhysicsCollisionAPI") if prim.IsValid() else False

    @classmethod
    def Apply(cls, prim: _UsdPrim) -> _CollisionAPI:
        """Apply CollisionAPI to a prim (adds to apiSchemas metadata)."""
        prim._prim.apply_api("PhysicsCollisionAPI")
        return cls(prim)

    def __bool__(self) -> bool:
        return self._valid

    def GetCollisionEnabledAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:collisionEnabled")


# ---------------------------------------------------------------------------
# ObjectType / JointDOF enums
# ---------------------------------------------------------------------------


class _ObjectType:
    """UsdPhysics.ObjectType enum — classifies prims by physics schema."""

    Scene = "Scene"
    RigidBody = "RigidBody"
    Articulation = "Articulation"
    RigidBodyMaterial = "RigidBodyMaterial"
    FixedJoint = "FixedJoint"
    RevoluteJoint = "RevoluteJoint"
    PrismaticJoint = "PrismaticJoint"
    SphericalJoint = "SphericalJoint"
    D6Joint = "D6Joint"
    DistanceJoint = "DistanceJoint"
    CubeShape = "CubeShape"
    SphereShape = "SphereShape"
    CapsuleShape = "CapsuleShape"
    CylinderShape = "CylinderShape"
    ConeShape = "ConeShape"
    MeshShape = "MeshShape"
    PlaneShape = "PlaneShape"
    # Aliases used by collision group backfill and some pxr code
    Box = "CubeShape"
    Sphere = "SphereShape"
    Capsule = "CapsuleShape"
    Cylinder = "CylinderShape"
    Cone = "ConeShape"
    ConvexMesh = "ConvexMesh"
    TriangleMesh = "TriangleMesh"
    Plane = "PlaneShape"
    Capsule1Shape = "Capsule1Shape"
    Cylinder1Shape = "Cylinder1Shape"
    CustomJoint = "CustomJoint"
    CollisionGroup = "CollisionGroup"
    EllipsoidShape = "EllipsoidShape"


class _JointDOF:
    """UsdPhysics.JointDOF enum — identifies degree-of-freedom axes."""

    TransX = "transX"
    TransY = "transY"
    TransZ = "transZ"
    RotX = "rotX"
    RotY = "rotY"
    RotZ = "rotZ"


# ---------------------------------------------------------------------------
# MassAPI
# ---------------------------------------------------------------------------


class _MassAPI:
    """UsdPhysics.MassAPI shim — wraps a prim to read mass properties."""

    _api_schema_name = "PhysicsMassAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Apply(cls, prim: _UsdPrim) -> _MassAPI:
        """Apply MassAPI to a prim (adds to apiSchemas metadata)."""
        prim._prim.apply_api("PhysicsMassAPI")
        return cls(prim)

    def CreateMassAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:mass", "float", default)

    def CreateDensityAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:density", "float", default)

    def CreateDiagonalInertiaAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:diagonalInertia", "float3", default)

    def CreateCenterOfMassAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:centerOfMass", "point3f", default)

    def CreatePrincipalAxesAttr(self, default: Any = None) -> _QuatAttribute:
        attr = _create_schema_attr(self._prim, "physics:principalAxes", "quatf", default)
        return _QuatAttribute(attr._prim, attr.GetName())

    def GetMassAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:mass")

    def GetDiagonalInertiaAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:diagonalInertia")

    def GetCenterOfMassAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:centerOfMass")

    def GetDensityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:density")

    def GetPrincipalAxesAttr(self) -> _QuatAttribute:
        return _QuatAttribute(self._prim._prim, "physics:principalAxes")


# ---------------------------------------------------------------------------
# Descriptor classes returned by LoadUsdPhysicsFromRange
# ---------------------------------------------------------------------------


class _JointLimit:
    """Limit parameters for a single joint axis."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.lower: float = -math.inf
        self.upper: float = math.inf


class _JointDrive:
    """Drive parameters for a single joint axis."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.acceleration: bool = False
        self.targetPosition: float = 0.0
        self.targetVelocity: float = 0.0
        self.stiffness: float = 0.0
        self.damping: float = 0.0
        self.forceLimit: float = math.inf


class _JointLimitPair:
    """A (JointDOF, JointLimit) pair for D6 joints."""

    def __init__(self, dof: str, limit: _JointLimit) -> None:
        self.first = dof
        self.second = limit


class _JointDrivePair:
    """A (JointDOF, JointDrive) pair for D6 joints."""

    def __init__(self, dof: str, drive: _JointDrive) -> None:
        self.first = dof
        self.second = drive


class _CollisionGroupDesc:
    """Descriptor for PhysicsCollisionGroup (placeholder)."""

    def __init__(self) -> None:
        self.isValid: bool = True


class _SceneDesc:
    """Descriptor for PhysicsScene."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.gravityDirection: np.ndarray = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self.gravityMagnitude: float = 9.81


class _RigidBodyDesc:
    """Descriptor for RigidBodyAPI-bearing prim."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rotation = _GfModule.Quatd(1.0, 0.0, 0.0, 0.0)
        self.linearVelocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self.angularVelocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self.rigidBodyEnabled: bool = True
        self.kinematicBody: bool = False
        self.collisions: list = []


class _ArticulationDesc:
    """Descriptor for ArticulationRootAPI-bearing prim."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.articulatedBodies: list[_SdfModule.Path] = []
        self.articulatedJoints: list[_SdfModule.Path] = []


class _RigidBodyMaterialDesc:
    """Descriptor for UsdPhysics material prim."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.staticFriction: float = 0.0
        self.dynamicFriction: float = 0.0
        self.restitution: float = 0.0
        self.density: float = 0.0


class _JointDesc:
    """Descriptor for a physics joint."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.type: str = ""
        self.primPath: _SdfModule.Path = _SdfModule.Path("")
        self.body0: _SdfModule.Path = _SdfModule.Path("")
        self.body1: _SdfModule.Path = _SdfModule.Path("")
        self.localPose0Position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.localPose0Orientation = _GfModule.Quatd(1.0, 0.0, 0.0, 0.0)
        self.localPose1Position: np.ndarray = np.zeros(3, dtype=np.float64)
        self.localPose1Orientation = _GfModule.Quatd(1.0, 0.0, 0.0, 0.0)
        self.jointEnabled: bool = True
        self.excludeFromArticulation: bool = False
        self.axis: _AxisValue = _Axis.X
        self.limit: _JointLimit = _JointLimit()
        self.drive: _JointDrive = _JointDrive()
        self.jointLimits: list[_JointLimitPair] = []
        self.jointDrives: list[_JointDrivePair] = []
        self.minEnabled: bool = False
        self.maxEnabled: bool = False


class _ShapeDesc:
    """Descriptor for a collision shape."""

    def __init__(self) -> None:
        self.isValid: bool = True
        self.rigidBody: _SdfModule.Path = _SdfModule.Path("")
        self.collisionGroups: list[_SdfModule.Path] = []
        self.filteredCollisions: list[_SdfModule.Path] = []
        self.materials: list[_SdfModule.Path] = []
        self.localPos: np.ndarray = np.zeros(3, dtype=np.float64)
        self.localRot = _GfModule.Quatd(1.0, 0.0, 0.0, 0.0)
        self.localScale: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.collisionEnabled: bool = True
        self.halfExtents: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        self.radius: float = 0.5
        self.halfHeight: float = 0.5
        self.axis: _AxisValue = _Axis.Y
        self.meshScale: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# LoadUsdPhysicsFromRange implementation
# ---------------------------------------------------------------------------


def _read_vec3(prim: _UsdPrim, name: str, fallback: np.ndarray | None = None) -> np.ndarray:
    """Read a vec3 attribute, return fallback if missing."""
    val = prim._prim.get_attribute(name)
    if val is not None:
        if isinstance(val, np.ndarray):
            return val.astype(np.float64).ravel()[:3]
        if isinstance(val, (list, tuple)):
            return np.array(val[:3], dtype=np.float64)
    return fallback if fallback is not None else np.zeros(3, dtype=np.float64)


def _read_quat(prim: _UsdPrim, name: str) -> _GfModule.Quatf:
    """Read a quaternion attribute, return identity if missing.

    Returns a Quatf so that value_to_warp() recognises it via GetReal/GetImaginary.
    _Attribute.Get() already auto-wraps quaternion types as Gf.Quatf/Quatd.
    """
    val = prim._prim.get_attribute(name)
    if val is not None:
        if hasattr(val, "GetReal") and hasattr(val, "GetImaginary"):
            imag = val.GetImaginary()
            return _GfModule.Quatf(float(val.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
        # nanousd native reads return (w, i, j, k)
        arr = np.asarray(val, dtype=np.float64).ravel()
        if len(arr) >= 4:
            return _GfModule.Quatf(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    return _GfModule.Quatf(1.0, 0.0, 0.0, 0.0)  # identity


def _read_float(prim: _UsdPrim, name: str, fallback: float = 0.0) -> float:
    """Read a float attribute.  If the value is a vector, return the first element."""
    val = prim._prim.get_attribute(name)
    if val is not None:
        if isinstance(val, np.ndarray) and val.ndim > 0:
            return float(val.flat[0])
        return float(val)
    return fallback


def _read_bool(prim: _UsdPrim, name: str, fallback: bool = True) -> bool:
    """Read a bool attribute."""
    val = prim._prim.get_attribute(name)
    if val is not None:
        return bool(val)
    return fallback


def _read_token(prim: _UsdPrim, name: str, fallback: str = "") -> str:
    """Read a token/string attribute."""
    val = prim._prim.get_attribute(name)
    if val is not None:
        return str(val)
    return fallback


def _find_rigid_body_ancestor(stage: _UsdStage, prim: _UsdPrim) -> _SdfModule.Path:
    """Walk up prim hierarchy to find nearest prim (including self) with RigidBodyAPI."""
    # Check the prim itself first (a prim can be both a body and a collider)
    if prim.IsValid() and prim.HasAPI("PhysicsRigidBodyAPI"):
        return prim.GetPath()
    path_str = str(prim.GetPath())
    while True:
        parent_str = path_str.rsplit("/", 1)[0]
        if not parent_str:
            break
        parent = stage.GetPrimAtPath(parent_str)
        if parent.IsValid() and parent.HasAPI("PhysicsRigidBodyAPI"):
            return _SdfModule.Path(parent_str)
        path_str = parent_str
    return _SdfModule.Path("")


def _joint_target_frame_matrix(
    stage: _UsdStage,
    target_path: _SdfModule.Path,
) -> tuple[np.ndarray, _UsdPrim | None] | None:
    path = str(target_path)
    if not path or path == "/":
        return None
    target = stage.GetPrimAtPath(path)
    if not target or not target.IsValid():
        return None
    body = _find_rigid_body_ancestor_prim(target)
    if body is None:
        return _get_world_transform_matrix(stage, target), None

    target_to_body = np.eye(4, dtype=np.float64)
    cur = target
    while cur and cur.IsValid() and cur.GetPath() != body.GetPath():
        local, _ = _get_local_transform_matrix(cur)
        target_to_body = target_to_body @ local
        cur = cur.GetParent()
    return target_to_body, body


def _resolve_joint_body_target(stage: _UsdStage, target: str) -> _SdfModule.Path:
    """Resolve a joint relationship target to the body path reported by UsdPhysics."""
    if not target or target == "/":
        return _SdfModule.Path("")
    target_prim = stage.GetPrimAtPath(target)
    if not target_prim.IsValid():
        return _SdfModule.Path("")
    return _find_rigid_body_ancestor(stage, target_prim)


def _find_rigid_body_ancestor_prim(prim: _UsdPrim | None) -> _UsdPrim | None:
    cur = prim
    while cur and cur.IsValid():
        if cur.HasAPI("PhysicsRigidBodyAPI"):
            return cur
        cur = cur.GetParent()
    return None


def _scale_matrix(scale: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = float(scale[0])
    mat[1, 1] = float(scale[1])
    mat[2, 2] = float(scale[2])
    return mat


def _adjust_joint_local_position(
    stage: _UsdStage,
    target_path: _SdfModule.Path,
    position: np.ndarray,
) -> np.ndarray:
    """Match UsdPhysics local-pose resolution for scaled bodies and child targets."""
    frame = _joint_target_frame_matrix(stage, target_path)
    if frame is None:
        return position

    point = np.array([position[0], position[1], position[2], 1.0], dtype=np.float64)
    target_to_frame, body = frame
    if body is not None:
        target_to_frame = target_to_frame @ _scale_matrix(_extract_scale_from_xform_ops(body))
    return (point @ target_to_frame)[:3]


def _quat_to_matrix4d_from_gf(quat: Any) -> np.ndarray:
    imag = quat.GetImaginary()
    return _quat_to_matrix4d(np.array([imag[0], imag[1], imag[2], quat.GetReal()], dtype=np.float64))


def _row_mat3_to_quat(mat: np.ndarray) -> Any:
    return _mat3_to_quat(np.asarray(mat, dtype=np.float64).T)


def _adjust_joint_local_orientation(
    stage: _UsdStage,
    target_path: _SdfModule.Path,
    orientation: Any,
) -> Any:
    """Match UsdPhysics local-pose orientation resolution for target prim frames."""
    frame = _joint_target_frame_matrix(stage, target_path)
    if frame is None:
        return orientation
    target_to_frame, _body = frame
    local = _quat_to_matrix4d_from_gf(orientation)
    return _row_mat3_to_quat((local @ target_to_frame)[:3, :3])


_JOINT_TYPE_MAP = {
    "PhysicsFixedJoint": _ObjectType.FixedJoint,
    "PhysicsRevoluteJoint": _ObjectType.RevoluteJoint,
    "PhysicsPrismaticJoint": _ObjectType.PrismaticJoint,
    "PhysicsSphericalJoint": _ObjectType.SphericalJoint,
    "PhysicsDistanceJoint": _ObjectType.DistanceJoint,
    "PhysicsJoint": _ObjectType.D6Joint,
}

_SHAPE_TYPE_MAP = {
    "Cube": _ObjectType.CubeShape,
    "Sphere": _ObjectType.SphereShape,
    "Capsule": _ObjectType.CapsuleShape,
    "Cylinder": _ObjectType.CylinderShape,
    "Cone": _ObjectType.ConeShape,
    "Mesh": _ObjectType.MeshShape,
    "Plane": _ObjectType.PlaneShape,
}


def _classify_joint_type(prim: _UsdPrim) -> str | None:
    """Classify a joint prim into an ObjectType value."""
    return _JOINT_TYPE_MAP.get(prim.GetTypeName())


def _classify_shape_type(prim: _UsdPrim) -> str | None:
    """Classify a collision shape prim into an ObjectType value."""
    if not prim.HasAPI("PhysicsCollisionAPI"):
        return None
    return _SHAPE_TYPE_MAP.get(prim.GetTypeName())


def _build_joint_desc(prim: _UsdPrim, joint_type: str, stage: _UsdStage) -> _JointDesc:
    """Populate a _JointDesc from prim attributes."""
    desc = _JointDesc()
    desc.type = joint_type
    desc.primPath = prim.GetPath()

    # Body relationships
    body0_targets = prim._read_rel_targets("physics:body0")
    body1_targets = prim._read_rel_targets("physics:body1")
    body0_target = body0_targets[0] if body0_targets else ""
    body1_target = body1_targets[0] if body1_targets else ""
    desc.body0 = _resolve_joint_body_target(stage, body0_target)
    desc.body1 = _resolve_joint_body_target(stage, body1_target)

    # Local poses
    desc.localPose0Position = _adjust_joint_local_position(
        stage,
        _SdfModule.Path(body0_target),
        _read_vec3(prim, "physics:localPos0"),
    )
    desc.localPose0Orientation = _adjust_joint_local_orientation(
        stage,
        _SdfModule.Path(body0_target),
        _read_quat(prim, "physics:localRot0"),
    )
    desc.localPose1Position = _adjust_joint_local_position(
        stage,
        _SdfModule.Path(body1_target),
        _read_vec3(prim, "physics:localPos1"),
    )
    desc.localPose1Orientation = _adjust_joint_local_orientation(
        stage,
        _SdfModule.Path(body1_target),
        _read_quat(prim, "physics:localRot1"),
    )

    desc.jointEnabled = _read_bool(prim, "physics:jointEnabled", True)
    desc.excludeFromArticulation = _read_bool(prim, "physics:excludeFromArticulation", False)

    # Axis (for revolute/prismatic) — return _AxisValue to match pxr behavior
    axis_str = _read_token(prim, "physics:axis", "X")
    desc.axis = _str_to_axis(axis_str)

    # Single-axis limit (revolute/prismatic/distance)
    lower = _read_float(prim, "physics:lowerLimit", -math.inf)
    upper = _read_float(prim, "physics:upperLimit", math.inf)
    has_lower = prim.HasAttribute("physics:lowerLimit")
    has_upper = prim.HasAttribute("physics:upperLimit")
    desc.limit.enabled = has_lower or has_upper
    desc.limit.lower = lower
    desc.limit.upper = upper
    desc.minEnabled = has_lower
    desc.maxEnabled = has_upper

    # Single-axis drive
    _populate_drive_from_prefix(
        prim, "drive:angular" if joint_type == _ObjectType.RevoluteJoint else "drive:linear", desc.drive
    )

    # D6 multi-axis limits and drives
    if joint_type == _ObjectType.D6Joint:
        for dof_name, dof_enum in [
            ("transX", _JointDOF.TransX),
            ("transY", _JointDOF.TransY),
            ("transZ", _JointDOF.TransZ),
            ("rotX", _JointDOF.RotX),
            ("rotY", _JointDOF.RotY),
            ("rotZ", _JointDOF.RotZ),
        ]:
            # D6 limits: limit:transX:physics:low, limit:transX:physics:high
            lim = _JointLimit()
            low_name = f"limit:{dof_name}:physics:low"
            high_name = f"limit:{dof_name}:physics:high"
            has_low = prim.HasAttribute(low_name)
            has_high = prim.HasAttribute(high_name)
            if has_low or has_high:
                lim.enabled = True
                lim.lower = _read_float(prim, low_name, -math.inf)
                lim.upper = _read_float(prim, high_name, math.inf)
                desc.jointLimits.append(_JointLimitPair(dof_enum, lim))

            # D6 drives: drive:transX:physics:damping, etc.
            drv = _JointDrive()
            _populate_drive_from_prefix(prim, f"drive:{dof_name}", drv)
            if drv.enabled:
                desc.jointDrives.append(_JointDrivePair(dof_enum, drv))

    return desc


def _populate_drive_from_prefix(prim: _UsdPrim, prefix: str, drive: _JointDrive) -> None:
    """Read drive attributes from a namespaced prefix like 'drive:angular'."""
    type_attr = f"{prefix}:physics:type"
    has_type = prim.HasAttribute(type_attr)
    stiffness = _read_float(prim, f"{prefix}:physics:stiffness", 0.0)
    damping = _read_float(prim, f"{prefix}:physics:damping", 0.0)
    if has_type or stiffness != 0.0 or damping != 0.0:
        drive.enabled = True
        drive.stiffness = stiffness
        drive.damping = damping
        drive.targetPosition = _read_float(prim, f"{prefix}:physics:targetPosition", 0.0)
        drive.targetVelocity = _read_float(prim, f"{prefix}:physics:targetVelocity", 0.0)
        drive.forceLimit = _read_float(prim, f"{prefix}:physics:maxForce", math.inf)
        drive_type = _read_token(prim, type_attr, "force")
        drive.acceleration = drive_type == "acceleration"


def _build_shape_desc(prim: _UsdPrim, stage: _UsdStage) -> _ShapeDesc:
    """Populate a _ShapeDesc from prim attributes."""
    desc = _ShapeDesc()

    # Find owning rigid body
    desc.rigidBody = _find_rigid_body_ancestor(stage, prim)

    # Collision groups
    n_groups = 0
    try:
        targets = prim._read_rel_targets("physics:collisionGroup")
        for t in targets:
            desc.collisionGroups.append(_SdfModule.Path(t))
            n_groups += 1
    except Exception:
        pass

    # Materials
    mat_targets = prim._read_rel_targets("material:binding:physics")
    if not mat_targets:
        mat_targets = prim._read_rel_targets("material:binding")
    for t in mat_targets:
        mat_prim = stage.GetPrimAtPath(t)
        if mat_prim and mat_prim.IsValid() and (
            mat_prim.GetTypeName() == "PhysicsMaterial" or mat_prim.HasAPI("PhysicsMaterialAPI")
        ):
            desc.materials.append(_SdfModule.Path(t))

    # Filtered collisions
    try:
        filt_targets = prim._read_rel_targets("physics:filteredPairs")
        for t in filt_targets:
            desc.filteredCollisions.append(_SdfModule.Path(t))
    except Exception:
        pass

    # Collision enabled
    desc.collisionEnabled = _read_bool(prim, "physics:collisionEnabled", True)

    # Local transform relative to rigid body
    desc.localPos = _read_vec3(prim, "physics:localPos", np.zeros(3, dtype=np.float64))
    desc.localRot = _read_quat(prim, "physics:localRot")

    # If no explicit localPos/localRot, compute from the composed local transform.
    has_physics_local_pos = (
        prim._prim.has_authored_attribute("physics:localPos")
        if hasattr(prim._prim, "has_authored_attribute")
        else prim.HasAttribute("physics:localPos")
    )
    has_physics_local_rot = (
        prim._prim.has_authored_attribute("physics:localRot")
        if hasattr(prim._prim, "has_authored_attribute")
        else prim.HasAttribute("physics:localRot")
    )
    if not has_physics_local_pos or not has_physics_local_rot:
        local_mat, _ = _get_local_transform_matrix(prim)
        if not has_physics_local_pos:
            desc.localPos = local_mat[3, :3].astype(np.float64)
        if not has_physics_local_rot:
            # Extract rotation via polar decomposition to handle non-uniform scale.
            upper = local_mat[:3, :3]
            try:
                u, _s, vt = np.linalg.svd(upper)
                rot = u @ vt
                if np.linalg.det(rot) < 0:
                    rot = -rot
                desc.localRot = _mat3_to_quat(rot.T)
            except np.linalg.LinAlgError:
                pass

    # Compute accumulated scale from collision prim to body (exclusive).
    # This accounts for intermediate xform scales and the prim's own scale.
    # Multiply all xformOp:scale (and suffixed variants) directly from the
    # xformOpOrder to avoid contaminating scale with rotation (column norms
    # of R*S give wrong per-axis values when non-uniform scaling is combined
    # with rotation).
    body_path = str(desc.rigidBody) if desc.rigidBody else ""
    acc_scale = np.ones(3, dtype=np.float64)
    cur = prim
    while True:
        prim_scale = _extract_scale_from_xform_ops(cur)
        acc_scale *= prim_scale
        cur_path = str(cur.GetPath())
        if cur_path == body_path or cur_path == "/":
            break
        parent_path = cur_path.rsplit("/", 1)[0]
        if not parent_path:
            break
        cur = stage.GetPrimAtPath(parent_path)
        if not cur.IsValid() or str(cur.GetPath()) == body_path:
            break

    desc.localScale = acc_scale.copy()

    # Shape-specific geometry (dimensions include accumulated scale to body)
    type_name = prim.GetTypeName()
    if type_name == "Cube":
        size = _read_float(prim, "size", 2.0)
        desc.halfExtents = np.array([size / 2.0] * 3, dtype=np.float64) * acc_scale
    elif type_name == "Sphere":
        desc.radius = _read_float(prim, "radius", 0.5) * float(np.max(acc_scale))
    elif type_name in ("Capsule", "Cylinder", "Cone"):
        axis_str = _read_token(prim, "axis", "Y")
        desc.axis = _str_to_axis(axis_str)
        # Scale radius/height using the correct axis components
        if axis_str == "X":
            height_scale = float(acc_scale[0])
            radius_scale = float(max(acc_scale[1], acc_scale[2]))
        elif axis_str == "Z":
            height_scale = float(acc_scale[2])
            radius_scale = float(max(acc_scale[0], acc_scale[1]))
        else:  # Y (default)
            height_scale = float(acc_scale[1])
            radius_scale = float(max(acc_scale[0], acc_scale[2]))
        desc.radius = _read_float(prim, "radius", 0.5) * radius_scale
        desc.halfHeight = _read_float(prim, "height", 1.0) / 2.0 * height_scale
    elif type_name == "Mesh":
        # Mesh scale from xformOp:scale or accumulated scale
        val = prim._prim.get_attribute("xformOp:scale")
        if val is not None:
            if isinstance(val, np.ndarray):
                desc.meshScale = val.ravel()[:3].astype(np.float64)
            elif isinstance(val, (list, tuple)):
                desc.meshScale = np.array(val[:3], dtype=np.float64)
        else:
            desc.meshScale = acc_scale
    elif type_name == "Plane":
        desc.axis = _str_to_axis(_read_token(prim, "axis", "Y"))

    return desc


def _build_scene_desc(prim: _UsdPrim) -> _SceneDesc:
    """Populate a _SceneDesc from PhysicsScene attributes."""
    desc = _SceneDesc()
    desc.gravityDirection = _read_vec3(prim, "physics:gravityDirection", np.array([0.0, -1.0, 0.0]))
    desc.gravityMagnitude = _read_float(prim, "physics:gravityMagnitude", 9.81)
    # USD physics schema sentinels: (0,0,0) direction means "use negative stage upAxis",
    # -inf magnitude means "use Earth gravity (9.81 m/s²)".
    if np.allclose(desc.gravityDirection, 0.0):
        stage = prim.GetStage()
        up = "Y"
        if stage is not None:
            up = stage.GetUpAxis() if hasattr(stage, "GetUpAxis") else "Y"
        if up == "Z":
            desc.gravityDirection = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        elif up == "X":
            desc.gravityDirection = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        else:
            desc.gravityDirection = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if not math.isfinite(desc.gravityMagnitude):
        desc.gravityMagnitude = 9.81
    return desc


def _build_rigid_body_desc(prim: _UsdPrim) -> _RigidBodyDesc:
    """Populate a _RigidBodyDesc from prim transform + API attributes."""
    desc = _RigidBodyDesc()
    mat = prim._prim.read_matrix4d("xformOp:transform")
    if mat is not None:
        desc.position = mat[3, :3].astype(np.float64)
        # Extract rotation as quaternion from 3x3 rotation submatrix
        desc.rotation = _mat3_to_quat(mat[:3, :3])
    else:
        desc.position = _read_vec3(prim, "xformOp:translate")
        val = prim._prim.get_attribute("xformOp:orient")
        if val is not None:
            if hasattr(val, "GetImaginary"):
                im = val.GetImaginary()
                desc.rotation = _GfModule.Quatd(float(val.GetReal()), float(im[0]), float(im[1]), float(im[2]))
            elif isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val, dtype=np.float64).ravel()
                if len(arr) >= 4:
                    desc.rotation = _GfModule.Quatd(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    desc.linearVelocity = _read_vec3(prim, "physics:velocity", np.zeros(3, dtype=np.float64))
    desc.angularVelocity = _read_vec3(prim, "physics:angularVelocity", np.zeros(3, dtype=np.float64))
    desc.rigidBodyEnabled = _read_bool(prim, "physics:rigidBodyEnabled", True)
    desc.kinematicBody = prim.HasAPI("PhysicsKinematicBodyAPI") or _read_bool(prim, "physics:kinematicEnabled", False)
    # Collect collision child paths
    for child in _PrimRange(prim):
        if child.GetPath() == prim.GetPath():
            continue
        if child.HasAPI("PhysicsCollisionAPI"):
            desc.collisions.append(child.GetPath())
    return desc


def _build_material_desc(prim: _UsdPrim) -> _RigidBodyMaterialDesc:
    """Populate a _RigidBodyMaterialDesc from material attributes."""
    desc = _RigidBodyMaterialDesc()
    desc.staticFriction = _read_float(prim, "physics:staticFriction", 0.0)
    desc.dynamicFriction = _read_float(prim, "physics:dynamicFriction", 0.0)
    desc.restitution = _read_float(prim, "physics:restitution", 0.0)
    desc.density = _read_float(prim, "physics:density", 0.0)
    return desc


def _mat3_to_quat(m: np.ndarray) -> _GfModule.Quatf:
    """Convert 3x3 rotation matrix to Gf.Quatf quaternion.

    Returns Gf.Quatf so value_to_warp() can detect it via .real/.imaginary.
    """
    # Shepperd's method
    t = np.trace(m)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _GfModule.Quatf(w, x, y, z)


def _load_usd_physics_from_range(
    stage: _UsdStage,
    paths: list[str | _SdfModule.Path],
    excludePaths: list[str | _SdfModule.Path] | None = None,
) -> dict:
    """Traverse stage and classify prims into physics ObjectType buckets.

    Returns dict mapping ObjectType -> (list[Sdf.Path], list[descriptor]).
    """
    exclude_set = {str(p) for p in excludePaths} if excludePaths else set()
    result: dict[str, tuple[list[_SdfModule.Path], list]] = {}

    def _add(obj_type: str, path: _SdfModule.Path, desc: Any) -> None:
        if obj_type not in result:
            result[obj_type] = ([], [])
        result[obj_type][0].append(path)
        result[obj_type][1].append(desc)

    # Traverse all prims under given root paths
    for root_path in paths:
        root_path_str = str(root_path)

        # "/" is pseudo-root — traverse entire stage
        if root_path_str == "/":
            prim_list = stage.Traverse()
        else:
            root_prim = stage.GetPrimAtPath(root_path_str)
            if not root_prim.IsValid():
                continue
            prim_list = list(_PrimRange(root_prim))

        for prim in prim_list:
            prim_path_str = str(prim.GetPath())

            # Check exclusions
            if any(prim_path_str.startswith(ex) for ex in exclude_set):
                continue

            type_name = prim.GetTypeName()

            # PhysicsScene
            if type_name == "PhysicsScene" or prim.HasAPI("PhysicsSceneAPI"):
                _add(_ObjectType.Scene, prim.GetPath(), _build_scene_desc(prim))

            # RigidBodyAPI
            if prim.HasAPI("PhysicsRigidBodyAPI"):
                _add(_ObjectType.RigidBody, prim.GetPath(), _build_rigid_body_desc(prim))

            # ArticulationRootAPI
            if prim.HasAPI("PhysicsArticulationRootAPI"):
                _add(_ObjectType.Articulation, prim.GetPath(), _ArticulationDesc())

            # Physics materials
            if type_name == "PhysicsMaterial" or (type_name == "Material" and prim.HasAPI("PhysicsMaterialAPI")):
                _add(_ObjectType.RigidBodyMaterial, prim.GetPath(), _build_material_desc(prim))

            # Joints
            joint_type = _classify_joint_type(prim)
            if joint_type is not None:
                _add(joint_type, prim.GetPath(), _build_joint_desc(prim, joint_type, stage))

            # Collision shapes
            shape_type = _classify_shape_type(prim)
            if shape_type is not None:
                _add(shape_type, prim.GetPath(), _build_shape_desc(prim, stage))

            # Collision groups
            if type_name == "PhysicsCollisionGroup":
                _add(_ObjectType.CollisionGroup, prim.GetPath(), _CollisionGroupDesc())

    # Collision group membership: scan for collection:colliders:includes on group prims
    # and backfill shape_desc.collisionGroups for shapes that didn't have direct relationships
    if _ObjectType.CollisionGroup in result:
        for root_path in paths:
            root_path_str = str(root_path)
            if root_path_str == "/":
                prim_list = stage.Traverse()
            else:
                root_prim = stage.GetPrimAtPath(root_path_str)
                if not root_prim.IsValid():
                    continue
                prim_list = list(_PrimRange(root_prim))

            for prim in prim_list:
                # Check if this prim has a collection:colliders:includes relationship
                collider_targets = prim._read_rel_targets("collection:colliders:includes")
                if not collider_targets:
                    continue
                group_path = prim.GetPath()
                target_set = set(collider_targets)
                # For each shape type, check if any shape paths match the collection targets
                for shape_type_key in (
                    _ObjectType.SphereShape,
                    _ObjectType.CapsuleShape,
                    _ObjectType.CubeShape,
                    _ObjectType.PlaneShape,
                    _ObjectType.CylinderShape,
                    _ObjectType.ConeShape,
                    _ObjectType.MeshShape,
                    _ObjectType.ConvexMesh,
                    _ObjectType.TriangleMesh,
                ):
                    if shape_type_key not in result:
                        continue
                    shape_paths, shape_descs = result[shape_type_key]
                    for sp, sd in zip(shape_paths, shape_descs):
                        sp_str = str(sp)
                        # Check exact match OR if shape is a descendant of a target
                        matched = sp_str in target_set
                        if not matched:
                            for t in target_set:
                                if sp_str.startswith(t + "/"):
                                    matched = True
                                    break
                        if matched and group_path not in sd.collisionGroups:
                            sd.collisionGroups.append(group_path)

    # Second pass: populate articulatedBodies and articulatedJoints on ArticulationDescs
    if _ObjectType.Articulation in result:
        art_paths, art_descs = result[_ObjectType.Articulation]

        # Collect all body paths for quick lookup
        body_path_set: set[str] = set()
        if _ObjectType.RigidBody in result:
            for p in result[_ObjectType.RigidBody][0]:
                body_path_set.add(str(p))

        # Collect all joints across all joint types
        all_joints: list[tuple[_SdfModule.Path, _JointDesc]] = []
        for jt in (
            _ObjectType.FixedJoint,
            _ObjectType.RevoluteJoint,
            _ObjectType.PrismaticJoint,
            _ObjectType.SphericalJoint,
            _ObjectType.D6Joint,
            _ObjectType.DistanceJoint,
        ):
            if jt in result:
                j_paths, j_descs = result[jt]
                all_joints.extend(zip(j_paths, j_descs))

        for art_idx, (art_path, art_desc) in enumerate(zip(art_paths, art_descs)):
            art_path_str = str(art_path)

            # Find bodies that are descendants of (or equal to) the articulation root
            art_body_set: set[str] = set()
            if _ObjectType.RigidBody in result:
                for bp in result[_ObjectType.RigidBody][0]:
                    bp_str = str(bp)
                    if bp_str == art_path_str or bp_str.startswith(art_path_str + "/"):
                        art_body_set.add(bp_str)

            # Expand body set by following joints: any body connected via a joint
            # to an articulation body is also part of the articulation.
            changed = True
            while changed:
                changed = False
                for _j_path, j_desc in all_joints:
                    if j_desc.excludeFromArticulation:
                        continue
                    b0 = str(j_desc.body0)
                    b1 = str(j_desc.body1)
                    if b0 in art_body_set and b1 and b1 not in art_body_set:
                        art_body_set.add(b1)
                        changed = True
                    elif b1 in art_body_set and b0 and b0 not in art_body_set:
                        art_body_set.add(b0)
                        changed = True

            # Build ordered body path list
            art_body_paths: list[_SdfModule.Path] = []
            if _ObjectType.RigidBody in result:
                for bp in result[_ObjectType.RigidBody][0]:
                    if str(bp) in art_body_set:
                        art_body_paths.append(bp)
            art_body_paths.sort(key=str)

            # Find joints connecting bodies in this articulation
            art_joint_paths: list[_SdfModule.Path] = []
            for j_path, j_desc in all_joints:
                if j_desc.excludeFromArticulation:
                    continue
                b0 = str(j_desc.body0)
                b1 = str(j_desc.body1)
                if b0 in art_body_set or b1 in art_body_set:
                    art_joint_paths.append(j_path)
            art_joint_paths.sort(key=str)

            art_desc.articulatedBodies = art_body_paths
            art_desc.articulatedJoints = art_joint_paths

    return result


class _Scene:
    """UsdPhysics.Scene shim."""

    _api_schema_name = "PhysicsSceneAPI"
    _usd_type_name = "PhysicsScene"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Scene:
        prim = stage.DefinePrim(str(path), "PhysicsScene")
        return cls(prim)

    @classmethod
    def Get(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Scene:
        prim = stage.GetPrimAtPath(str(path))
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetGravityDirectionAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:gravityDirection")

    def GetGravityMagnitudeAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:gravityMagnitude")


class _DriveAPI:
    """UsdPhysics.DriveAPI shim — wraps a prim for joint drive attributes."""

    _api_schema_name = "PhysicsDriveAPI"

    def __init__(self, prim: _UsdPrim, name: str = ""):
        self._prim = prim
        self._name = name

    @staticmethod
    def Apply(prim: _UsdPrim, name: str) -> _DriveAPI:
        return _DriveAPI(prim, name)

    def GetStiffnessAttr(self) -> _Attribute:
        return self._prim.GetAttribute(f"drive:{self._name}:physics:stiffness")

    def GetDampingAttr(self) -> _Attribute:
        return self._prim.GetAttribute(f"drive:{self._name}:physics:damping")

    def GetMaxForceAttr(self) -> _Attribute:
        return self._prim.GetAttribute(f"drive:{self._name}:physics:maxForce")

    def CreateStiffnessAttr(self) -> _Attribute:
        return self._prim.CreateAttribute(f"drive:{self._name}:physics:stiffness", "float")

    def CreateDampingAttr(self) -> _Attribute:
        return self._prim.CreateAttribute(f"drive:{self._name}:physics:damping", "float")

    def CreateMaxForceAttr(self) -> _Attribute:
        return self._prim.CreateAttribute(f"drive:{self._name}:physics:maxForce", "float")


class _LimitAPI:
    """UsdPhysics.LimitAPI shim — wraps a prim for joint limit attributes."""

    _api_schema_name = "PhysicsLimitAPI"

    def __init__(self, prim: _UsdPrim, name: str = ""):
        self._prim = prim
        self._name = name

    @staticmethod
    def Apply(prim: _UsdPrim, name: str) -> _LimitAPI:
        return _LimitAPI(prim, name)

    def GetLowAttr(self) -> _Attribute:
        return self._prim.GetAttribute(f"limit:{self._name}:physics:low")

    def GetHighAttr(self) -> _Attribute:
        return self._prim.GetAttribute(f"limit:{self._name}:physics:high")

    def CreateLowAttr(self) -> _Attribute:
        return self._prim.CreateAttribute(f"limit:{self._name}:physics:low", "float")

    def CreateHighAttr(self) -> _Attribute:
        return self._prim.CreateAttribute(f"limit:{self._name}:physics:high", "float")


class _FixedJoint:
    """UsdPhysics.FixedJoint shim."""

    _usd_type_name = "PhysicsFixedJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _FixedJoint:
        prim = stage.DefinePrim(str(path), "PhysicsFixedJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def CreateBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateLocalPos0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos0")

    def CreateLocalPos1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos1")

    def CreateLocalRot0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot0")

    def CreateLocalRot1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot1")

    def CreateJointEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:jointEnabled", "bool")

    def CreateExcludeFromArticulationAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:excludeFromArticulation", "bool", default)


class _RevoluteJoint:
    """UsdPhysics.RevoluteJoint shim."""

    _usd_type_name = "PhysicsRevoluteJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _RevoluteJoint:
        prim = stage.DefinePrim(str(path), "PhysicsRevoluteJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:axis")

    def CreateAxisAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:axis", "token", default)

    def CreateBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def CreateBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateLocalPos0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos0")

    def CreateLocalPos1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos1")

    def CreateLocalRot0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot0")

    def CreateLocalRot1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot1")

    def CreateJointEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:jointEnabled", "bool")

    def GetLowerLimitAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:lowerLimit")

    def GetUpperLimitAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:upperLimit")

    def CreateLowerLimitAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:lowerLimit", "float")

    def CreateUpperLimitAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:upperLimit", "float")

    def CreateExcludeFromArticulationAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:excludeFromArticulation", "bool", default)


class _PrismaticJoint:
    """UsdPhysics.PrismaticJoint shim."""

    _usd_type_name = "PhysicsPrismaticJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _PrismaticJoint:
        prim = stage.DefinePrim(str(path), "PhysicsPrismaticJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def GetAxisAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:axis")

    def CreateAxisAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:axis", "token", default)

    def CreateBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def CreateBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateLocalPos0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos0")

    def CreateLocalPos1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos1")

    def CreateLocalRot0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot0")

    def CreateLocalRot1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot1")

    def CreateJointEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:jointEnabled", "bool")

    def CreateExcludeFromArticulationAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:excludeFromArticulation", "bool", default)


class _Joint:
    """UsdPhysics.Joint shim (generic D6 joint)."""

    _usd_type_name = "PhysicsJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Joint:
        prim = stage.DefinePrim(str(path), "PhysicsJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def CreateBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateLocalPos0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos0")

    def CreateLocalPos1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos1")

    def CreateLocalRot0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot0")

    def CreateLocalRot1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot1")

    def CreateJointEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:jointEnabled", "bool")

    def CreateExcludeFromArticulationAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:excludeFromArticulation", "bool", default)


class _SphericalJoint:
    """UsdPhysics.SphericalJoint shim."""

    _usd_type_name = "PhysicsSphericalJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _SphericalJoint:
        prim = stage.DefinePrim(str(path), "PhysicsSphericalJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def CreateBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")

    def CreateLocalPos0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos0")

    def CreateLocalPos1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localPos1")

    def CreateLocalRot0Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot0")

    def CreateLocalRot1Attr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:localRot1")

    def CreateJointEnabledAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:jointEnabled", "bool")

    def CreateExcludeFromArticulationAttr(self, default: Any = None) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:excludeFromArticulation", "bool", default)


class _DistanceJoint:
    """UsdPhysics.DistanceJoint shim."""

    _usd_type_name = "PhysicsDistanceJoint"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _DistanceJoint:
        prim = stage.DefinePrim(str(path), "PhysicsDistanceJoint")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetBody0Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body0")

    def GetBody1Rel(self) -> _Relationship:
        return _Relationship(self._prim._prim, "physics:body1")


class _MeshCollisionAPI:
    """UsdPhysics.MeshCollisionAPI shim."""

    _api_schema_name = "PhysicsMeshCollisionAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @staticmethod
    def Apply(prim: _UsdPrim) -> _MeshCollisionAPI:
        prim._prim.apply_api("PhysicsMeshCollisionAPI")
        return _MeshCollisionAPI(prim)

    def GetApproximationAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:approximation")


class _MaterialAPI:
    """UsdPhysics.MaterialAPI shim."""

    _api_schema_name = "PhysicsMaterialAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @staticmethod
    def Apply(prim: _UsdPrim) -> _MaterialAPI:
        prim._prim.apply_api("PhysicsMaterialAPI")
        return _MaterialAPI(prim)

    def GetStaticFrictionAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:staticFriction")

    def GetDynamicFrictionAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:dynamicFriction")

    def GetRestitutionAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:restitution")

    def GetDensityAttr(self) -> _Attribute:
        return self._prim.GetAttribute("physics:density")

    def CreateDensityAttr(self) -> _Attribute:
        return _create_schema_attr(self._prim, "physics:density", "float")


class _PhysicsTokens:
    """UsdPhysics.Tokens shim."""

    none = "none"
    convexHull = "convexHull"
    convexDecomposition = "convexDecomposition"
    boundingSphere = "boundingSphere"
    boundingCube = "boundingCube"
    meshSimplification = "meshSimplification"


class _UsdPhysicsModule:
    """Compatibility shim for pxr.UsdPhysics."""

    Axis = _Axis
    CollisionAPI = _CollisionAPI
    RigidBodyAPI = _RigidBodyAPI
    ArticulationRootAPI = _ArticulationRootAPI
    MassAPI = _MassAPI
    ObjectType = _ObjectType
    JointDOF = _JointDOF
    JointDesc = _JointDesc
    Scene = _Scene
    DriveAPI = _DriveAPI
    LimitAPI = _LimitAPI
    FixedJoint = _FixedJoint
    RevoluteJoint = _RevoluteJoint
    PrismaticJoint = _PrismaticJoint
    SphericalJoint = _SphericalJoint
    DistanceJoint = _DistanceJoint
    Joint = _Joint
    MeshCollisionAPI = _MeshCollisionAPI
    MaterialAPI = _MaterialAPI
    Tokens = _PhysicsTokens
    RigidBodyDesc = _RigidBodyDesc

    @staticmethod
    def StageHasAuthoredKilogramsPerUnit(stage: _UsdStage) -> bool:
        return stage.HasAuthoredKilogramsPerUnit()

    @staticmethod
    def GetStageKilogramsPerUnit(stage: _UsdStage) -> float:
        return stage.GetKilogramsPerUnit()

    @staticmethod
    def SetStageKilogramsPerUnit(stage: _UsdStage, value: float) -> None:
        stage._stage.set_metadata_d("kilogramsPerUnit", value)

    @staticmethod
    def LoadUsdPhysicsFromRange(
        stage: _UsdStage,
        paths: list[str | _SdfModule.Path],
        excludePaths: list[str | _SdfModule.Path] | None = None,
    ) -> dict:
        return _load_usd_physics_from_range(stage, paths, excludePaths)


UsdPhysics = _UsdPhysicsModule()


# ===========================================================================
# UsdShade module
# ===========================================================================


class _ShaderInput:
    """Minimal UsdShade.Shader input shim."""

    def __init__(self, prim: _UsdPrim, input_name: str):
        self._prim = prim
        self._name = input_name
        self._attr_name = f"inputs:{input_name}"

    def __bool__(self) -> bool:
        return self._prim.HasAttribute(self._attr_name)

    def Get(self) -> Any:
        attr = self._prim.GetAttribute(self._attr_name)
        if attr and attr.IsValid():
            return attr.Get()
        return None

    def Set(self, value: Any, time: float | None = None) -> bool:
        """Set the input value."""
        attr = self._prim.GetAttribute(self._attr_name)
        return attr.Set(value, time)

    def GetAttr(self) -> _Attribute:
        return self._prim.GetAttribute(self._attr_name)

    def GetBaseName(self) -> str:
        return self._name

    def HasConnectedSource(self) -> bool:
        return self._prim._has_connections(self._attr_name) or self._prim._has_rel(f"{self._attr_name}.connect")

    def GetConnectedSource(self) -> tuple | None:
        targets = self._prim._read_connections(self._attr_name)
        if not targets:
            targets = self._prim._read_rel_targets(f"{self._attr_name}.connect")
        if targets:
            stage = self._prim.GetStage()
            if stage:
                source_path = targets[0].rsplit(".", 1)[0] if "." in targets[0] else targets[0]
                source_prim = stage.GetPrimAtPath(source_path)
                return (_Shader(source_prim),)
        return None

    def GetInputs(self) -> list[_ShaderInput]:
        return []


class _ShaderOutput:
    """Minimal UsdShade.Shader output shim."""

    def __init__(self, prim: _UsdPrim, output_name: str):
        self._prim = prim
        self._name = output_name
        self._attr_name = f"outputs:{output_name}"

    def __bool__(self) -> bool:
        return (
            self._prim.HasAttribute(self._attr_name)
            or self._prim._has_connections(self._attr_name)
            or self._prim._has_rel(f"{self._attr_name}.connect")
        )

    def GetConnectedSource(self) -> tuple | None:
        targets = self._prim._read_connections(self._attr_name)
        if not targets:
            targets = self._prim._read_rel_targets(f"{self._attr_name}.connect")
        if targets:
            stage = self._prim.GetStage()
            if stage:
                source_path = targets[0].rsplit(".", 1)[0] if "." in targets[0] else targets[0]
                source_prim = stage.GetPrimAtPath(source_path)
                return (_Shader(source_prim),)
        return None

    def ConnectToSource(self, connectable: Any, output_name: str = "surface") -> bool:
        """Connect this output to a source shader's output."""
        if hasattr(connectable, "_prim"):
            source_path = str(connectable._prim.GetPath())
        elif hasattr(connectable, "GetPrim"):
            source_path = str(connectable.GetPrim().GetPath())
        else:
            source_path = str(connectable)
        target = f"{source_path}.outputs:{output_name}"
        return self._prim._prim.create_relationship(f"{self._attr_name}.connect", [target])


class _Shader:
    """UsdShade.Shader shim."""

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Shader:
        prim = stage.DefinePrim(str(path), "Shader")
        return cls(prim)

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def GetIdAttr(self) -> _Attribute:
        return self._prim.GetAttribute("info:id")

    def CreateIdAttr(self, value: str = "") -> _Attribute:
        return _create_schema_attr(self._prim, "info:id", "token", value if value else None)

    def CreateInput(self, name: str, type_name: Any = None) -> _ShaderInput:
        if type_name is not None:
            self._prim.CreateAttribute(f"inputs:{name}", type_name)
        return _ShaderInput(self._prim, name)

    def CreateOutput(self, name: str, type_name: Any = None) -> _ShaderOutput:
        if type_name is not None:
            self._prim.CreateAttribute(f"outputs:{name}", type_name)
        return _ShaderOutput(self._prim, name)

    def ConnectableAPI(self) -> _Shader:
        """Return self as the connectable interface."""
        return self

    def GetInput(self, name: str) -> _ShaderInput:
        return _ShaderInput(self._prim, name)

    def GetInputs(self) -> list[_ShaderInput]:
        inputs = []
        for attr in self._prim.GetAttributes():
            attr_name = attr.GetName()
            if attr_name.startswith("inputs:"):
                input_name = attr_name[len("inputs:") :]
                inputs.append(_ShaderInput(self._prim, input_name))
        return inputs

    def GetOutput(self, name: str) -> _ShaderOutput:
        return _ShaderOutput(self._prim, name)


class _Material:
    """UsdShade.Material shim."""

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @classmethod
    def Define(cls, stage: _UsdStage, path: str | _SdfModule.Path) -> _Material:
        prim = stage.DefinePrim(str(path), "Material")
        return cls(prim)

    def __bool__(self) -> bool:
        return self._prim.IsValid()

    def GetPrim(self) -> _UsdPrim:
        return self._prim

    def CreateSurfaceOutput(self) -> _ShaderOutput:
        return _ShaderOutput(self._prim, "surface")

    def GetSurfaceOutput(self) -> _ShaderOutput:
        return _ShaderOutput(self._prim, "surface")

    def GetOutput(self, name: str) -> _ShaderOutput:
        return _ShaderOutput(self._prim, name)

    def GetInputs(self) -> list[_ShaderInput]:
        inputs = []
        for attr in self._prim.GetAttributes():
            attr_name = attr.GetName()
            if attr_name.startswith("inputs:"):
                input_name = attr_name[len("inputs:") :]
                inputs.append(_ShaderInput(self._prim, input_name))
        return inputs


class _DirectBinding:
    """UsdShade.MaterialBindingAPI.DirectBinding shim."""

    def __init__(self, prim: _UsdPrim, rel_name: str = "material:binding"):
        self._prim = prim
        self._rel_name = rel_name

    def GetMaterialPath(self) -> _SdfModule.Path:
        targets = self._prim._read_rel_targets(self._rel_name)
        if targets:
            return _SdfModule.Path(targets[0])
        return _SdfModule.Path("")

    def GetMaterial(self) -> _Material | None:
        targets = self._prim._read_rel_targets(self._rel_name)
        if targets:
            stage = self._prim.GetStage()
            if stage:
                mat_prim = stage.GetPrimAtPath(targets[0])
                if mat_prim and mat_prim.IsValid():
                    return _Material(mat_prim)
        return None

    def __bool__(self) -> bool:
        return bool(self._prim._read_rel_targets(self._rel_name))


class _MaterialBindingAPI:
    """UsdShade.MaterialBindingAPI shim."""

    _api_schema_name = "MaterialBindingAPI"

    def __init__(self, prim: _UsdPrim):
        self._prim = prim

    @staticmethod
    def Apply(prim: _UsdPrim) -> _MaterialBindingAPI:
        prim._prim.apply_api("MaterialBindingAPI")
        return _MaterialBindingAPI(prim)

    def Bind(self, material: _Material, purpose: str = "") -> bool:
        """Bind a material to this prim."""
        mat_path = str(material.GetPrim().GetPath())
        if purpose:
            rel_name = f"material:binding:{purpose}"
        else:
            rel_name = "material:binding"
        return self._prim._prim.create_relationship(rel_name, [mat_path])

    def ComputeBoundMaterial(self, materialPurpose: str | None = None) -> tuple[_Material | None, Any]:
        rel_names = []
        if materialPurpose:
            rel_names.append(f"material:binding:{materialPurpose}")
        rel_names.extend(["material:binding", "material:binding:preview", "material:binding:full", "material:binding:physics"])
        seen = set()
        stage = self._prim.GetStage()
        for rel_name in rel_names:
            if rel_name in seen:
                continue
            seen.add(rel_name)
            targets = self._prim._read_rel_targets(rel_name)
            if targets and stage:
                mat_prim = stage.GetPrimAtPath(targets[0])
                if mat_prim and mat_prim.IsValid():
                    return _Material(mat_prim), None
        return None, None

    def GetDirectBinding(self) -> _DirectBinding:
        return _DirectBinding(self._prim)

    def GetDirectBindingRel(self, purpose: str = "") -> _BindingRel:
        if purpose:
            rel_name = f"material:binding:{purpose}"
        else:
            rel_name = "material:binding"
        return _BindingRel(self._prim, rel_name)


class _BindingRel:
    """Relationship proxy for material binding."""

    def __init__(self, prim: _UsdPrim, rel_name: str):
        self._prim = prim
        self._rel_name = rel_name

    def __bool__(self) -> bool:
        return self._prim._has_rel(self._rel_name)

    def GetTargets(self) -> list:
        return [_SdfModule.Path(t) for t in self._prim._read_rel_targets(self._rel_name)]


class _ShadeUtils:
    """UsdShade.Utils shim."""

    @staticmethod
    def GetValueProducingAttributes(shader_input: _ShaderInput) -> list[_Attribute]:
        source = shader_input.GetConnectedSource()
        if source:
            shader = source[0]
            for out_name in ("outputs:out", "outputs:result", "outputs:rgb", "outputs:r"):
                attr = shader.GetPrim().GetAttribute(out_name)
                if attr and attr.IsValid():
                    return [attr]
            attr = shader_input.GetAttr()
            if attr and attr.IsValid():
                return [attr]
        attr = shader_input.GetAttr()
        if attr and attr.IsValid():
            return [attr]
        return []


class _UsdShadeModule:
    """Compatibility shim for pxr.UsdShade."""

    Shader = _Shader
    Material = _Material
    MaterialBindingAPI = _MaterialBindingAPI
    Utils = _ShadeUtils


UsdShade = _UsdShadeModule()


# ===========================================================================
# Vt module (minimal)
# ===========================================================================


def _vt_numeric_array(args: tuple, dtype: Any, comps: int) -> np.ndarray:
    """Build a numpy-backed VtArray matching pxr.Vt constructor semantics.

    Supports the four OpenUSD forms: ``()`` empty, ``(n)`` n zero-filled
    elements, ``(n, fill)`` n copies of ``fill``, and ``(sequence)`` from data.
    A leading ``int`` is a length (as in OpenUSD); a sequence is data.
    """
    empty_shape: Any = (0, comps) if comps > 1 else 0
    if len(args) == 0 or args[0] is None:
        return np.zeros(empty_shape, dtype=dtype)
    a0 = args[0]
    if isinstance(a0, (int, np.integer)) and not isinstance(a0, bool):
        n = int(a0)
        shape: Any = (n, comps) if comps > 1 else (n,)
        arr = np.zeros(shape, dtype=dtype)
        if len(args) >= 2 and args[1] is not None:
            arr[:] = np.asarray(args[1], dtype=dtype)
        return arr
    arr = np.asarray(a0, dtype=dtype)
    return arr.reshape(-1, comps) if comps > 1 else arr.reshape(-1)


def _vt_object_array(args: tuple, fill_default: Any) -> list:
    """List-backed VtArray for non-numeric element types (tokens/strings/quats)."""
    if len(args) == 0 or args[0] is None:
        return []
    a0 = args[0]
    if isinstance(a0, (int, np.integer)) and not isinstance(a0, bool):
        n = int(a0)
        fill = args[1] if len(args) >= 2 else fill_default
        return [fill] * n
    return list(a0)


class _VtModule:
    """Compatibility shim for pxr.Vt — numpy-backed numeric arrays, list-backed
    token/string/quat arrays. All constructors accept (), (n), (n, fill), (seq)."""

    # --- floating-point vector/scalar arrays ---
    @staticmethod
    def FloatArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float32, 1)

    @staticmethod
    def DoubleArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float64, 1)

    @staticmethod
    def HalfArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float16, 1)

    @staticmethod
    def Vec2fArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float32, 2)

    @staticmethod
    def Vec3fArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float32, 3)

    @staticmethod
    def Vec4fArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float32, 4)

    @staticmethod
    def Vec2dArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float64, 2)

    @staticmethod
    def Vec3dArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float64, 3)

    @staticmethod
    def Vec4dArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.float64, 4)

    # point/normal/color/texcoord are float3/float2 semantically
    Point3fArray = Vec3fArray
    Point3dArray = Vec3dArray
    Normal3fArray = Vec3fArray
    Color3fArray = Vec3fArray
    Color4fArray = Vec4fArray
    TexCoord2fArray = Vec2fArray

    # --- integer / bool arrays ---
    @staticmethod
    def IntArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.int32, 1)

    @staticmethod
    def UIntArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.uint32, 1)

    @staticmethod
    def Int64Array(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.int64, 1)

    @staticmethod
    def Vec2iArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.int32, 2)

    @staticmethod
    def Vec3iArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.int32, 3)

    @staticmethod
    def Vec4iArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, np.int32, 4)

    @staticmethod
    def BoolArray(*args: Any) -> np.ndarray:
        return _vt_numeric_array(args, bool, 1)

    # --- token / string / quaternion arrays (object/list-backed) ---
    @staticmethod
    def TokenArray(*args: Any) -> list:
        return _vt_object_array(args, "")

    @staticmethod
    def StringArray(*args: Any) -> list:
        return _vt_object_array(args, "")

    @staticmethod
    def QuatfArray(*args: Any) -> list:
        return _vt_object_array(args, _GfModule.Quatf(1.0, 0.0, 0.0, 0.0))

    @staticmethod
    def QuatdArray(*args: Any) -> list:
        return _vt_object_array(args, _GfModule.Quatd(1.0, 0.0, 0.0, 0.0))

    @staticmethod
    def QuathArray(*args: Any) -> list:
        return _vt_object_array(args, _GfModule.Quath(1.0, 0.0, 0.0, 0.0) if hasattr(_GfModule, "Quath") else None)


Vt = _VtModule()
