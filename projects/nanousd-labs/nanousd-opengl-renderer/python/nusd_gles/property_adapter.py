# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanousdPropertyAdapter — read-only PropertyAdapter for nanousd prims.

Multi-selection collapses to ambiguous values when the per-prim values
differ. Edits are silently ignored — the property panel still renders,
but spinner drags don't write back to the stage.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, List, Optional

from ovgear.adapters import AttributeMetadata, PropertyAdapter
from ovgear.settings import Subscription

from nusd_gles._nanousd import NanousdStage, _Prim


_AMBIGUOUS = object()


# Common USD schema attributes per prim type. Used to surface
# unauthored properties in the panel (Imageable + Xformable for
# everything; Mesh-specific, Sphere-specific, etc. for the rest).
# Keys are (attr_name, type_name, group).
_IMAGEABLE_DEFAULTS = (
    ("visibility", "token", "Display"),
    ("purpose", "token", "Display"),
)
_XFORMABLE_DEFAULTS = (
    ("xformOpOrder", "token[]", "Transform"),
)
_BOUNDABLE_DEFAULTS = (
    ("extent", "float3[]", "Geometry"),
)
_GPRIM_DEFAULTS = (
    ("doubleSided", "bool", "Geometry"),
    ("orientation", "token", "Geometry"),
    ("primvars:displayColor", "color3f[]", "Display"),
    ("primvars:displayOpacity", "float[]", "Display"),
)
_MESH_DEFAULTS = (
    ("subdivisionScheme", "token", "Geometry"),
    ("interpolateBoundary", "token", "Geometry"),
    ("faceVaryingLinearInterpolation", "token", "Geometry"),
    ("triangleSubdivisionRule", "token", "Geometry"),
    ("faceVertexCounts", "int[]", "Geometry"),
    ("faceVertexIndices", "int[]", "Geometry"),
    ("points", "point3f[]", "Geometry"),
    ("normals", "normal3f[]", "Geometry"),
)
_SPHERE_DEFAULTS = (
    ("radius", "double", "Geometry"),
)
_CUBE_DEFAULTS = (
    ("size", "double", "Geometry"),
)
_CYLINDER_DEFAULTS = (
    ("radius", "double", "Geometry"),
    ("height", "double", "Geometry"),
    ("axis", "token", "Geometry"),
)
_CONE_DEFAULTS = _CYLINDER_DEFAULTS
_CAPSULE_DEFAULTS = _CYLINDER_DEFAULTS
_CAMERA_DEFAULTS = (
    ("focalLength", "float", "Camera"),
    ("focusDistance", "float", "Camera"),
    ("fStop", "float", "Camera"),
    ("horizontalAperture", "float", "Camera"),
    ("verticalAperture", "float", "Camera"),
    ("clippingRange", "float2", "Camera"),
    ("projection", "token", "Camera"),
)
_LIGHT_DEFAULTS = (
    ("inputs:intensity", "float", "Light"),
    ("inputs:color", "color3f", "Light"),
    ("inputs:exposure", "float", "Light"),
    ("inputs:colorTemperature", "float", "Light"),
    ("inputs:enableColorTemperature", "bool", "Light"),
)


def _schema_defaults_for_type(type_name: str) -> tuple:
    """Return the set of likely-relevant default attributes for a prim
    of ``type_name`` — Imageable + Xformable + per-schema additions."""
    out: list = list(_IMAGEABLE_DEFAULTS) + list(_XFORMABLE_DEFAULTS)
    if type_name in (
        "Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Plane",
        "Points", "BasisCurves", "NurbsCurves", "NurbsPatch",
    ):
        out += list(_BOUNDABLE_DEFAULTS) + list(_GPRIM_DEFAULTS)
    if type_name == "Mesh":
        out += list(_MESH_DEFAULTS)
    elif type_name == "Sphere":
        out += list(_SPHERE_DEFAULTS)
    elif type_name == "Cube":
        out += list(_CUBE_DEFAULTS)
    elif type_name in ("Cylinder", "Cone", "Capsule"):
        out += list(_CYLINDER_DEFAULTS)
    elif type_name == "Camera":
        out += list(_CAMERA_DEFAULTS)
    elif type_name.endswith("Light"):
        out += list(_LIGHT_DEFAULTS)
    return tuple(out)


def _write_value(prim: _Prim, attr_name: str, base: str, value: Any) -> None:
    """Write ``value`` to ``attr_name`` on ``prim`` using the typed setter
    matching ``base`` (the USD type name with ``[]`` stripped). No-op on
    type mismatch or unsupported types.
    """
    if base in _VEC3_F_TYPES:
        prim.set_attrib_vec(attr_name, 3, "f", value)
    elif base in _VEC3_D_TYPES:
        prim.set_attrib_vec(attr_name, 3, "d", value)
    elif base in _VEC2_F_TYPES:
        prim.set_attrib_vec(attr_name, 2, "f", value)
    elif base in _VEC2_D_TYPES:
        prim.set_attrib_vec(attr_name, 2, "d", value)
    elif base in _VEC4_F_TYPES:
        prim.set_attrib_vec(attr_name, 4, "f", value)
    elif base in _VEC4_D_TYPES:
        prim.set_attrib_vec(attr_name, 4, "d", value)
    elif base in _VEC2_I_TYPES:
        prim.set_attrib_vec(attr_name, 2, "i", value)
    elif base in _VEC3_I_TYPES:
        prim.set_attrib_vec(attr_name, 3, "i", value)
    elif base in _VEC4_I_TYPES:
        prim.set_attrib_vec(attr_name, 4, "i", value)
    elif base == "matrix4d":
        prim.set_attrib_matrix4d(attr_name, value)
    elif base in _BOOL_TYPES:
        prim.set_attrib_bool(attr_name, bool(value))
    elif base in _INT_TYPES:
        prim.set_attrib_int(attr_name, int(value))
    elif base == "token":
        prim.set_attrib_token(attr_name, str(value))
    elif base in _STRING_TYPES:
        prim.set_attrib_str(attr_name, str(value))
    elif base in ("double", "timecode"):
        prim.set_attrib_double(attr_name, float(value))
    elif base in _FLOAT_TYPES:
        prim.set_attrib_float(attr_name, float(value))


# USD type-name → reader. Mirrors the registrations in
# ovuiproperty/builders/{vec,color,scalar,ivec,matrix,asset,token}.py so
# our values flow through to the right Vec*/Float/Int/etc. row widgets.
_VEC2_F_TYPES = ("half2", "float2", "texCoord2f", "texCoord2h")
_VEC2_D_TYPES = ("double2", "texCoord2d")
_VEC3_F_TYPES = ("half3", "float3", "normal3f", "point3f", "vector3f",
                 "color3f", "texCoord3f", "texCoord3h")
_VEC3_D_TYPES = ("double3", "normal3d", "point3d", "vector3d",
                 "color3d", "texCoord3d")
_VEC4_F_TYPES = ("half4", "float4", "color4f", "quatf", "quath")
_VEC4_D_TYPES = ("double4", "color4d", "quatd")
_VEC2_I_TYPES = ("int2",)
_VEC3_I_TYPES = ("int3",)
_VEC4_I_TYPES = ("int4",)
_FLOAT_TYPES = ("float", "double", "half", "timecode")
_INT_TYPES = ("int", "uint", "int64", "uint64", "uchar")
_BOOL_TYPES = ("bool",)
_STRING_TYPES = ("string", "token", "asset")
_MATRIX_TYPES = ("matrix2d", "matrix3d", "matrix4d")


def _read_value(prim: _Prim, attr_name: str) -> Any:
    """Read an attribute as a Python value, using the correct typed
    accessor for each USD type. Returns ``None`` if the attribute is
    not authored or the type is unknown.
    """
    if not prim.has_attrib(attr_name):
        return None
    type_name = prim.attrib_type(attr_name)
    base = type_name.replace("[]", "").strip()
    is_array = type_name.endswith("[]")
    try:
        if is_array:
            # token[] reads as a tuple of strings (small; e.g. xformOpOrder
            # has 3-5 tokens). Other array types fall back to a length
            # summary — the property panel doesn't have a generic array
            # editor anyway.
            if base == "token":
                tokens = prim.get_attrib_array_tokens(attr_name)
                return tuple(tokens) if tokens else None
            n = prim.get_attrib_array_len(attr_name)
            return n if n > 0 else None

        if base in _VEC3_F_TYPES:
            return prim.get_attrib_vec(attr_name, 3, "f")
        if base in _VEC3_D_TYPES:
            return prim.get_attrib_vec(attr_name, 3, "d")
        if base in _VEC2_F_TYPES:
            return prim.get_attrib_vec(attr_name, 2, "f")
        if base in _VEC2_D_TYPES:
            return prim.get_attrib_vec(attr_name, 2, "d")
        if base in _VEC4_F_TYPES:
            return prim.get_attrib_vec(attr_name, 4, "f")
        if base in _VEC4_D_TYPES:
            return prim.get_attrib_vec(attr_name, 4, "d")
        if base in _VEC2_I_TYPES:
            return prim.get_attrib_vec(attr_name, 2, "i")
        if base in _VEC3_I_TYPES:
            return prim.get_attrib_vec(attr_name, 3, "i")
        if base in _VEC4_I_TYPES:
            return prim.get_attrib_vec(attr_name, 4, "i")
        if base == "matrix4d":
            return prim.get_attrib_matrix4d(attr_name)
        if base in _BOOL_TYPES:
            return prim.get_attrib_bool(attr_name)
        if base in _INT_TYPES:
            return prim.get_attrib_int(attr_name)
        if base in _STRING_TYPES:
            return prim.get_attrib_str(attr_name)
        if base in _FLOAT_TYPES:
            return prim.get_attrib_float(attr_name)
    except Exception:
        return None
    # Unknown type: fall back to string read so the row at least shows
    # something instead of crashing or going blank.
    return prim.get_attrib_str(attr_name) or None


class NanousdPropertyAdapter(PropertyAdapter):
    """Read-only multi-selection property adapter."""

    def __init__(
        self,
        stage: NanousdStage,
        paths: Optional[List[str]] = None,
    ) -> None:
        self._stage = stage
        self._paths: List[str] = list(paths or [])
        self._subscribers: List[Callable[[], None]] = []
        # Cache of attribute-name unions across the current selection so
        # repeated queries don't re-walk every prim.
        self._attr_cache: Optional[List[str]] = None

    # ── Selection management ──────────────────────────────────────────────

    def set_paths(self, paths: List[str]) -> None:
        new = list(paths or [])
        if new != self._paths:
            self._paths = new
            self._attr_cache = None
            self._notify()

    # ── ABC: identity ─────────────────────────────────────────────────────

    def get_paths(self) -> List[str]:
        return list(self._paths)

    def is_valid(self) -> bool:
        return bool(self._paths)

    def get_scheme(self) -> str:
        return "nanousd"

    # ── ABC: attribute discovery ──────────────────────────────────────────

    def _prims(self) -> List[_Prim]:
        out: List[_Prim] = []
        for p in self._paths:
            prim = self._stage.prim_at_path(p)
            if prim is not None:
                out.append(prim)
        return out

    def get_attribute_names(self) -> List[str]:
        if self._attr_cache is not None:
            return list(self._attr_cache)
        seen: dict[str, None] = {}
        for prim in self._prims():
            for name in prim.attrib_names():
                seen.setdefault(name, None)
            # Surface schema-default attributes too so the panel shows
            # a Sphere's `radius` / a Mesh's `xformOpOrder` / a Camera's
            # `focalLength` even when the prim has no authored opinion.
            for n, _t, _g in _schema_defaults_for_type(prim.type_name):
                seen.setdefault(n, None)
        names = list(seen.keys())
        self._attr_cache = names
        return list(names)

    def _schema_default_for(self, attr_name: str) -> Optional[tuple]:
        for prim in self._prims():
            for name, t, group in _schema_defaults_for_type(prim.type_name):
                if name == attr_name:
                    return (name, t, group)
        return None

    def get_attribute_metadata(self, attr_name: str) -> AttributeMetadata:
        type_name = ""
        is_authored = False
        group = ""
        for prim in self._prims():
            if prim.has_attrib(attr_name):
                type_name = prim.attrib_type(attr_name)
                is_authored = True
                break
        if not type_name:
            sd = self._schema_default_for(attr_name)
            if sd is not None:
                _, type_name, group = sd
        base = (type_name or "string").replace("[]", "").strip()
        if base in _BOOL_TYPES:
            value_type: type = bool
        elif base in _INT_TYPES or base in _VEC2_I_TYPES or base in _VEC3_I_TYPES or base in _VEC4_I_TYPES:
            value_type = int
        elif (
            base in _FLOAT_TYPES
            or base in _VEC2_F_TYPES or base in _VEC3_F_TYPES or base in _VEC4_F_TYPES
            or base in _VEC2_D_TYPES or base in _VEC3_D_TYPES or base in _VEC4_D_TYPES
            or base in _MATRIX_TYPES
        ):
            value_type = float
        else:
            value_type = str
        return AttributeMetadata(
            name=attr_name,
            display_name=attr_name,
            type_name=type_name or "string",
            value_type=value_type,
            group=group,
            is_authored=is_authored,
            # is_locked stays False — the row's edit handler is the place
            # to enforce read-only; flagging the metadata as locked draws
            # a misleading padlock and disables the spinner outright.
            is_locked=False,
        )

    # ── ABC: value access ─────────────────────────────────────────────────

    def get_value(self, attr_name: str) -> Any:
        # Multi-selection: return value if all match, else first non-None.
        first: Any = _AMBIGUOUS
        ambiguous = False
        for prim in self._prims():
            v = _read_value(prim, attr_name)
            if first is _AMBIGUOUS:
                first = v
            elif v != first:
                ambiguous = True
                break
        if first is _AMBIGUOUS:
            return None
        return first if not ambiguous else first

    def is_ambiguous(self, attr_name: str) -> bool:
        first: Any = _AMBIGUOUS
        for prim in self._prims():
            v = _read_value(prim, attr_name)
            if first is _AMBIGUOUS:
                first = v
            elif v != first:
                return True
        return False

    def get_per_component_ambiguity(self, attr_name: str) -> Optional[List[bool]]:
        # Adopt a coarse "all components ambiguous together" signal —
        # finer-grained per-channel comparison would require type
        # introspection we'd rather centralise once edits land.
        return None

    # ── ABC: edits ────────────────────────────────────────────────────────

    def begin_edit(self, attr_name: str) -> None:
        # nanousd has no transactional edit API; the begin/end pair is
        # purely a UI hint. We don't snapshot for undo because ovgear's
        # UndoManager.command() expects a Sdf-style writeback we'd
        # have to invent. End-of-drag still goes through end_edit so
        # listeners that watch for value changes get a final notify.
        return

    def set_value(self, attr_name: str, value: Any) -> None:
        prims = self._prims()
        if not prims:
            return
        type_name = ""
        for prim in prims:
            if prim.has_attrib(attr_name):
                type_name = prim.attrib_type(attr_name)
                break
        base = type_name.replace("[]", "").strip()
        for prim in prims:
            try:
                _write_value(prim, attr_name, base, value)
            except Exception:
                continue
        self._notify()

    def end_edit(self, attr_name: str) -> None:
        self._notify()

    # ── ABC: change subscriptions ─────────────────────────────────────────

    def subscribe_changes(self, callback: Callable[[], None]) -> Subscription:
        self._subscribers.append(callback)
        return Subscription(weakref.ref(self), "property_changes", callback)

    def _remove_subscriber(self, key: str, callback: Callable) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def _notify(self) -> None:
        for cb in list(self._subscribers):
            try:
                cb()
            except Exception:
                pass
