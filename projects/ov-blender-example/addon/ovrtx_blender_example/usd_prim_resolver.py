# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached USD prim resolution for interactive edits."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from . import light_usd_prim
from . import material_usd_prim
from . import material_value_conversion
from . import uv_usd_prim
from . import world_dome_usd_prim
from . import usd_paths
from .usd_prim_resolution import UsdPrimResolution, UsdPrimResolutionStatus


class UsdPrimResolver:
    """Own one scene scan, domain indexes, and UV validation lifetime."""

    def __init__(
        self,
        *,
        object_paths_by_session_uid: Mapping[int, str] | None = None,
        light_paths_by_object_session_uid: Mapping[int, str] | None = None,
        mesh_topology_change_resolver: Callable[[Any], Mapping[str, Any] | None]
        | None = None,
    ) -> None:
        self._object_paths_by_session_uid = dict(object_paths_by_session_uid or {})
        self._light_paths_by_object_session_uid = dict(
            light_paths_by_object_session_uid or {}
        )
        self._mesh_topology_change_resolver = mesh_topology_change_resolver
        self.reset()

    def reset(self) -> None:
        self._scene_path: str | None = None
        self._material_indexes: dict[str, Mapping[str, Any]] = {}
        self._light_index: Mapping[str, Any] = _unavailable_index("not_loaded")
        self._world_dome_index: Mapping[str, Any] = _unavailable_index("not_loaded")
        self._uv_index: Mapping[str, Any] = _unavailable_index("not_loaded")
        self._blender_object_index: Mapping[str, Any] = _unavailable_index("not_loaded")
        self._uv_loop_order_validations: dict[str, Mapping[str, Any]] = {}
        self._scan_diagnostics: dict[str, Any] = {"available": False, "reason": "not_loaded"}

    def scan(self, request: Any | None) -> None:
        scene_path = str(getattr(request, "input_usd_path", "") or "")
        if scene_path == self._scene_path:
            return
        self._scene_path = scene_path
        self._uv_loop_order_validations = {}
        if not scene_path:
            self._set_unavailable("missing_input_usd_path")
            return
        try:
            stage, prims, material_type = _open_stage_prims(scene_path)
            material_indexes = {
                usd_attribute: material_usd_prim._material_prim_index_from_prims(
                    prims,
                    usd_attribute=usd_attribute,
                    material_type=material_type,
                )
                for usd_attribute in material_value_conversion.SUPPORTED_USD_ATTRIBUTES
            }
            light_index = light_usd_prim._light_prim_index_from_prims(prims)
            world_dome_index = world_dome_usd_prim._world_dome_prim_index_from_prims(prims)
            uv_index = uv_usd_prim._uv_prim_index_from_prims(prims)
            blender_object_index = _blender_object_index_from_prims(prims)
            indexed_paths = dict(blender_object_index["by_session_uid"])
            indexed_paths.update(
                {
                    int(session_uid): (str(path),)
                    for session_uid, path in self._object_paths_by_session_uid.items()
                }
            )
            blender_object_index["by_session_uid"] = indexed_paths
        except Exception as exc:
            self._set_unavailable(f"{type(exc).__name__}: {exc}")
            return
        self._material_indexes = material_indexes
        self._light_index = light_index
        self._world_dome_index = world_dome_index
        self._uv_index = uv_index
        self._blender_object_index = blender_object_index
        self._scan_diagnostics = {
            "available": True,
            "reason": "",
            "input_usd_path": scene_path,
            "prim_count": len(prims),
        }
        del stage

    def resolve_material(
        self,
        material: Any,
        *,
        usd_attribute: str,
        property_name: str,
    ) -> UsdPrimResolution[material_usd_prim.MaterialUsdPrim]:
        index = self._material_indexes.get(usd_attribute, _unavailable_index("attribute_not_scanned"))
        return material_usd_prim.resolve_material_usd_prim(
            material,
            index,
            usd_attribute=usd_attribute,
            property_name=property_name,
        )

    def resolve_light(self, light_object: Any) -> UsdPrimResolution[light_usd_prim.LightUsdPrim]:
        session_uid = int(getattr(light_object, "session_uid", 0) or 0)
        return light_usd_prim.resolve_light_usd_prim(
            light_object,
            self._light_index,
            known_prim_path=self._light_paths_by_object_session_uid.get(
                session_uid, ""
            ),
        )

    def resolve_world_dome(self) -> UsdPrimResolution[world_dome_usd_prim.WorldDomeUsdPrim]:
        return world_dome_usd_prim.resolve_world_dome_usd_prim(self._world_dome_index)

    def resolve_uv(self, mesh: Any) -> UsdPrimResolution[uv_usd_prim.UvUsdPrim]:
        return uv_usd_prim.resolve_uv_usd_prim(mesh, self._uv_index)

    def mesh_topology_change(self, mesh: Any) -> Mapping[str, Any] | None:
        if self._mesh_topology_change_resolver is None:
            return None
        change = self._mesh_topology_change_resolver(mesh)
        return None if change is None else dict(change)

    def resolve_blender_object(self, obj: Any) -> UsdPrimResolution[str]:
        name = str(getattr(obj, "name", "") or "")
        session_uid = int(getattr(obj, "session_uid", 0) or 0)
        uid_candidates = tuple(
            self._blender_object_index.get("by_session_uid", {}).get(session_uid, ())
        )
        authoring_path = usd_paths.authoring_prim_path(obj)
        match_source = "blender_session_uid" if uid_candidates else "blender_object_name"
        diagnostics = {
            "match_source": match_source,
            "blender_object_name": name,
            "blender_session_uid": session_uid,
            "authoring_prim_path": authoring_path,
            "candidate_paths": (
                uid_candidates
                if uid_candidates
                else tuple(self._blender_object_index.get("by_name", {}).get(name, ()))
            ),
        }
        if not bool(self._blender_object_index.get("available", False)):
            return UsdPrimResolution(
                UsdPrimResolutionStatus.ERROR,
                error_reason="usd_stage_unavailable",
                diagnostics={
                    **diagnostics,
                    "stage_reason": str(self._blender_object_index.get("reason", "not_loaded")),
                },
            )
        # Authored scene composition (blender-live-render task04-01): the
        # topology orchestrator assigns each converted object's root prim
        # path to the ``ov.usd.prim_path`` authoring property, and the
        # converters author the object root at exactly that path (an Xform
        # for meshes, the UsdLux prim itself for lights). Authored
        # generations carry no exported-name attributes, so the authoring
        # identity — verified against the scanned stage — is the primary
        # resolution source; the exported-name index remains the fallback
        # for direct-USD stages produced by Blender's stock exporter.
        if authoring_path and authoring_path in self._blender_object_index.get("stage_paths", frozenset()):
            return UsdPrimResolution(
                UsdPrimResolutionStatus.OK,
                value=authoring_path,
                diagnostics={**diagnostics, "match_source": "authoring_prim_path"},
            )
        candidate_paths = diagnostics["candidate_paths"]
        if not candidate_paths:
            return UsdPrimResolution(
                UsdPrimResolutionStatus.ERROR,
                error_reason=(
                    # The object claims an authored identity the scanned
                    # composition does not contain (stale reconcile, or an
                    # object the converters do not emit): fail closed with
                    # the precise reason instead of a generic name miss.
                    "authoring_prim_path_not_in_scene"
                    if authoring_path
                    else "blender_object_name_not_found"
                ),
                diagnostics=diagnostics,
            )
        if len(candidate_paths) != 1:
            return UsdPrimResolution(
                UsdPrimResolutionStatus.ERROR,
                error_reason="ambiguous_blender_object_name",
                diagnostics=diagnostics,
            )
        return UsdPrimResolution(
            UsdPrimResolutionStatus.OK,
            value=str(candidate_paths[0]),
            diagnostics=diagnostics,
        )

    def uv_loop_order_validation(self, prim_path: str) -> Mapping[str, Any] | None:
        validation = self._uv_loop_order_validations.get(prim_path)
        return dict(validation) if validation is not None else None

    def record_uv_loop_order_validation(self, edit: Any) -> None:
        if getattr(edit, "usd_attribute", "") != uv_usd_prim.TARGET_USD_ATTRIBUTE:
            return
        metadata = getattr(edit, "metadata", {})
        validation = metadata.get("loop_order_validation") if isinstance(metadata, Mapping) else None
        if not isinstance(validation, Mapping):
            return
        prim_path = str(validation.get("mesh_prim_path", ""))
        if prim_path:
            self._uv_loop_order_validations[prim_path] = dict(validation)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "scan": dict(self._scan_diagnostics),
            "material": {
                "attributes": {
                    attribute: _index_diagnostics(index)
                    for attribute, index in self._material_indexes.items()
                }
            },
            "light": _index_diagnostics(self._light_index),
            "world_dome": _index_diagnostics(self._world_dome_index),
            "uv": {
                **_index_diagnostics(self._uv_index),
                "loop_order_validation_count": len(self._uv_loop_order_validations),
                "loop_order_validation_paths": sorted(self._uv_loop_order_validations),
            },
            "blender_object": _index_diagnostics(self._blender_object_index),
        }

    def _set_unavailable(self, reason: str) -> None:
        attributes = material_value_conversion.SUPPORTED_USD_ATTRIBUTES
        self._material_indexes = {attribute: _unavailable_index(reason) for attribute in attributes}
        self._light_index = _unavailable_index(reason)
        self._world_dome_index = _unavailable_index(reason)
        self._uv_index = _unavailable_index(reason)
        self._blender_object_index = _unavailable_index(reason)
        self._scan_diagnostics = {
            "available": False,
            "reason": reason,
            "input_usd_path": self._scene_path or "",
            "prim_count": 0,
        }


def _open_stage_prims(scene_path: str) -> tuple[Any, tuple[Any, ...], Any]:
    from pxr import Usd, UsdShade  # type: ignore

    stage = Usd.Stage.Open(scene_path)
    if stage is None:
        raise RuntimeError("stage_open_failed")
    return stage, tuple(stage.Traverse()), UsdShade.Material


def _blender_object_index_from_prims(prims: tuple[Any, ...]) -> dict[str, Any]:
    by_name: dict[str, list[str]] = {}
    candidates: list[dict[str, str]] = []
    stage_paths: set[str] = set()
    for prim in prims:
        prim_path = usd_paths.usd_prim_path_from_prim(prim)
        if prim_path:
            stage_paths.add(prim_path)
        get_type_name = getattr(prim, "GetTypeName", None)
        type_name = str(get_type_name() if callable(get_type_name) else getattr(prim, "type_name", ""))
        if type_name != "Xform":
            continue
        get_attribute = getattr(prim, "GetAttribute", None)
        attribute = get_attribute("userProperties:blender:object_name") if callable(get_attribute) else None
        if attribute is None:
            continue
        is_valid = getattr(attribute, "IsValid", None)
        if callable(is_valid) and not is_valid():
            continue
        get_value = getattr(attribute, "Get", None)
        name = str(get_value() if callable(get_value) else "")
        path = usd_paths.usd_prim_path_from_prim(prim)
        if not name or not path:
            continue
        by_name.setdefault(name, []).append(path)
        candidates.append({"blender_object_name": name, "prim_path": path})
    return {
        "available": True,
        "reason": "",
        "by_name": {name: tuple(paths) for name, paths in by_name.items()},
        "by_session_uid": {},
        "candidates": tuple(candidates),
        "prim_paths": tuple(candidate["prim_path"] for candidate in candidates),
        # Every scanned prim path (any type): authoring-property resolution
        # verifies the object's claimed root prim against the composition.
        "stage_paths": frozenset(stage_paths),
    }


def _unavailable_index(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "stage_reason": reason, "candidates": ()}


def _index_diagnostics(index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(index.get("available", False)),
        "reason": str(index.get("reason", index.get("stage_reason", ""))),
        "candidate_count": len(index.get("candidates", ())),
        "prim_count": len(index.get("prim_paths", index.get("material_prim_paths", ()))),
    }


__all__ = ["UsdPrimResolver"]
