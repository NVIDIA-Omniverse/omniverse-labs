# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD path and interactive object path index helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


USD_LAYER_ID_PROP = "ovrtx.usd_layer_id"
USD_PRIM_PATH_PROP = "ovrtx.usd_prim_path"
USD_PROPERTY_PATH_PROP = "ovrtx.usd_property_path"
SOURCE_USD_PATH_PROP = "ovrtx:sourceUsdPath"
BLENDER_PROPERTY_PATH_PROP = "ovrtx.blender_property_path"
DATA_AUTHORITY_PROP = "ovrtx.data_authority"
SELECTION_OWNER_OBJECT_PROP = "ovrtx.selection_owner_object"
SELECTION_OWNER_USD_PRIM_PATH_PROP = "ovrtx.selection_owner_usd_prim_path"
CAMERA_SOURCE_USD_PATH_MATCH = "sourceUsdPath"
CAMERA_HIERARCHY_PATH_MATCH = "hierarchy_path"
CAMERA_ROOT_OBJECT_PATH_MATCH = "root_object_path"
CAMERA_MATCH_SOURCE_ORDER = (
    CAMERA_SOURCE_USD_PATH_MATCH,
    CAMERA_HIERARCHY_PATH_MATCH,
    CAMERA_ROOT_OBJECT_PATH_MATCH,
)


def valid_usd_identifier(value: Any) -> str:
    """Return a valid USD identifier derived from a Blender display name."""

    identifier = re.sub(r"[^A-Za-z0-9_]", "_", str(value or "").strip())
    identifier = re.sub(r"_+", "_", identifier).strip("_") or "Prim"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier


def reserve_unique_child_path(
    parent_path: str,
    initial_name: Any,
    occupied_paths: set[str],
) -> str:
    """Validate a parent and reserve one unique sanitized child path."""

    parent = clean_usd_path(parent_path).rstrip("/")
    if not parent:
        raise ValueError("USD topology parent must be an absolute prim path")
    leaf = valid_usd_identifier(initial_name)
    candidate = parent + "/" + leaf
    suffix = 2
    while candidate in occupied_paths:
        candidate = f"{parent}/{leaf}_{suffix}"
        suffix += 1
    occupied_paths.add(candidate)
    return candidate


def load_usd_path_index(usd_path: Path | str) -> dict[str, Any]:
    try:
        from pxr import Usd
    except Exception as exc:
        return {
            "available": False,
            "reason": "pxr_unavailable:" + type(exc).__name__,
            "valid_paths": set(),
            "rigid_body_paths": set(),
        }
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return {
            "available": False,
            "reason": "stage_open_failed",
            "valid_paths": set(),
            "rigid_body_paths": set(),
        }
    valid_paths = set()
    rigid_body_paths = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        valid_paths.add(path)
        schemas = _usd_api_schema_names(prim.GetMetadata("apiSchemas"))
        if "PhysicsRigidBodyAPI" in schemas:
            rigid_body_paths.add(path)
    return {
        "available": True,
        "reason": "",
        "valid_paths": valid_paths,
        "rigid_body_paths": rigid_body_paths,
    }


def authoring_prim_path(id_data: Any) -> str:
    """Return the ID's ``ov.usd.prim_path`` authoring identity, if any.

    The topology orchestrator assigns each converted object's and visual
    material's root prim path to this authoring property; the converters
    author the root at exactly that path (blender-live-render tasks
    04-01/04-02).
    """

    path = _direct_authoring_prim_path(id_data)
    if path:
        return path
    # Evaluated depsgraph copies of some ID types (materials in Blender
    # 5.1, verified headless) do not carry add-on PointerProperty data.
    # The authoring identity is evaluation-independent, so read it from
    # the original datablock when the evaluated copy has none.
    original = getattr(id_data, "original", None)
    if original is not None:
        return _direct_authoring_prim_path(original)
    return ""


def _direct_authoring_prim_path(id_data: Any) -> str:
    authoring = getattr(getattr(id_data, "ov", None), "usd", None)
    return clean_usd_path(getattr(authoring, "prim_path", ""))


def selection_owner_object_name(id_data: Any) -> str:
    return _string_value(id_property(id_data, SELECTION_OWNER_OBJECT_PROP, ""))


def source_usd_path_from_blender_id(id_data: Any) -> str:
    direct = clean_usd_path(id_property(id_data, SOURCE_USD_PATH_PROP, ""))
    if direct:
        return direct
    return clean_usd_path(id_property(getattr(id_data, "data", None), SOURCE_USD_PATH_PROP, ""))


def resolved_usd_path(obj: Any, path_index: Mapping[str, Any]) -> str:
    valid_paths = path_index.get("valid_paths", set())
    imported_path = source_usd_path_from_blender_id(obj)
    if imported_path and (not valid_paths or imported_path in valid_paths):
        return imported_path
    hierarchy_path = hierarchy_usd_path(obj)
    if hierarchy_path and hierarchy_path in valid_paths:
        return hierarchy_path
    return ""


def hierarchy_usd_path(obj: Any) -> str:
    names = []
    current = obj
    while current is not None:
        name = normalized_blender_object_name(getattr(current, "name", ""))
        if name:
            names.append(name)
        current = getattr(current, "parent", None)
    return "/" + "/".join(reversed(names)) if names else ""


def nested_hierarchy_usd_path(obj: Any) -> str:
    if getattr(obj, "parent", None) is None:
        return ""
    return hierarchy_usd_path(obj)


def root_object_usd_path(obj: Any) -> str:
    if getattr(obj, "parent", None) is not None:
        return ""
    name = normalized_blender_object_name(getattr(obj, "name", ""))
    return "/" + name if name else ""


def camera_match_sources(camera_prim_path: str) -> tuple[str, ...]:
    if len(path_parts(clean_usd_path(camera_prim_path))) == 1:
        return CAMERA_MATCH_SOURCE_ORDER
    return CAMERA_MATCH_SOURCE_ORDER[:2]


def camera_usd_path_for_source(obj: Any, source: str) -> str:
    if source == CAMERA_SOURCE_USD_PATH_MATCH:
        return source_usd_path_from_blender_id(obj)
    if source == CAMERA_HIERARCHY_PATH_MATCH:
        return nested_hierarchy_usd_path(obj)
    if source == CAMERA_ROOT_OBJECT_PATH_MATCH:
        return root_object_usd_path(obj)
    raise ValueError(f"Unsupported camera path match source: {source!r}")


def nearest_dynamic_body_path(
    path: str,
    path_index: Mapping[str, Any],
    dynamic_body_root: str,
) -> str:
    current = str(path or "")
    rigid_body_paths = path_index.get("rigid_body_paths", set())
    while current:
        if current in rigid_body_paths and is_under_root(current, dynamic_body_root):
            return current
        current = parent_usd_path(current)
    return ""


def tag_body_edit_owner(
    obj: Any,
    body_path: str,
    *,
    usd_layer_id: str = "",
) -> None:
    if usd_layer_id:
        obj[USD_LAYER_ID_PROP] = usd_layer_id
    obj[USD_PRIM_PATH_PROP] = body_path
    obj[USD_PROPERTY_PATH_PROP] = body_path + ".xformOp:transform"
    obj[BLENDER_PROPERTY_PATH_PROP] = "matrix_world"
    obj[DATA_AUTHORITY_PROP] = "sim"
    obj["ovrtx.physics_backed"] = True
    obj["ovrtx.physics_affecting_transform"] = True


def tag_body_selection_source(obj: Any, body_path: str, owner_obj: Any) -> None:
    obj[SELECTION_OWNER_OBJECT_PROP] = owner_obj.name
    obj[SELECTION_OWNER_USD_PRIM_PATH_PROP] = body_path
    obj["ovrtx.physics_backed"] = True


def parent_usd_path(path: str) -> str:
    path = str(path).rstrip("/")
    if not path or path == "/":
        return ""
    parent = path.rsplit("/", 1)[0]
    return parent if parent else "/"


def is_under_root(path: str, root: str) -> bool:
    root = str(root or "").rstrip("/")
    path = str(path or "").rstrip("/")
    return bool(path and root and (path == root or path.startswith(root + "/")))


def normalized_blender_object_name(name: Any) -> str:
    return str(name).split(".", 1)[0]


def clean_usd_path(path: Any) -> str:
    value = str(path or "").strip()
    return value if value.startswith("/") and value != "???" else ""


def known_usd_path(path: Any) -> bool:
    value = str(path or "")
    return bool(value) and value != "???" and value.startswith("/")


def usd_prim_path_from_prim(prim: Any) -> str:
    try:
        return clean_usd_path(str(prim.GetPath()))
    except Exception:
        return clean_usd_path(getattr(prim, "path", ""))


def usd_prim_type_name_from_prim(prim: Any) -> str:
    getter = getattr(prim, "GetTypeName", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return ""
    return str(getattr(prim, "type_name", ""))


def path_parts(path: str) -> list[str]:
    return [part for part in str(path).split("/") if part]


def normalized_usd_leaf_name(path: Any) -> str:
    parts = path_parts(str(path))
    return normalized_blender_object_name(parts[-1]) if parts else ""


def _string_value(value: Any) -> str:
    return str(value or "").strip()


def id_property(id_data: Any, key: str, default: Any) -> Any:
    if id_data is None:
        return default
    getter = getattr(id_data, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return id_data[key]
    except (KeyError, TypeError, AttributeError):
        return default


def _usd_api_schema_names(value: Any) -> set[str]:
    if value is None:
        return set()
    getter = getattr(value, "GetAppliedItems", None)
    if callable(getter):
        try:
            return {str(item) for item in getter()}
        except Exception:
            pass
    names = set()
    for attr in ("explicitItems", "prependedItems", "appendedItems", "addedItems"):
        try:
            names.update(str(item) for item in getattr(value, attr, ()))
        except Exception:
            pass
    if names:
        return names
    try:
        return {str(item) for item in value}
    except Exception:
        return set()
