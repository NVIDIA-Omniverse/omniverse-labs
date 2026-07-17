# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-state rules for stock Blender interaction."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Mapping, Sequence

from .blender_interactive_edit_builders import object_transform_edit
from .interactive_edit_planner import DataAuthority, EditStatus, EditShape, InteractiveEdit
from .interactive_edit_workflow import WorkflowAction, EditWorkflowResult
from .shared_stage_composition import BodyPose
from . import usd_paths as usd_paths


_INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH = 0
_INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS: dict[str, Any] = {
    "registered": False,
    "handler": "depsgraph_update_post",
    "suppressed": False,
    "last_active_viewport_engine_count": 0,
    "last_submitted_edit_count": 0,
    "last_result_count": 0,
    "last_error": "",
    "updated_at_ns": 0,
}
_SELECTION_RESOLUTION_DIAGNOSTICS: dict[str, Any] = {
    "status": "no_selection",
    "changed": False,
    "selected_object_count": 0,
    "resolved_owner_count": 0,
    "missing_owner_count": 0,
    "group_supported": True,
    "group_rejected": False,
    "last_source_name": "",
    "last_owner_name": "",
    "owner_names": [],
    "owner_categories": [],
    "unresolved_reasons": [],
    "sources": [],
    "updated_at_ns": 0,
}
@contextmanager
def suppress_interactive_edit_bridge() -> Any:
    """Temporarily suppress live edit extraction while code mirrors runtime state."""

    global _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH
    _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH += 1
    try:
        yield
    finally:
        _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH = max(0, _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH - 1)


def interactive_edit_bridge_suppressed() -> bool:
    return _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH > 0


def interactive_edit_bridge_diagnostics() -> dict[str, Any]:
    diagnostics = dict(_INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS)
    diagnostics["suppress_depth"] = _INTERACTIVE_EDIT_BRIDGE_SUPPRESS_DEPTH
    diagnostics["selection_resolution"] = dict(_SELECTION_RESOLUTION_DIAGNOSTICS)
    return diagnostics


def record_interactive_edit_bridge_diagnostics(
    *,
    registered: bool | None = None,
    suppressed: bool | None = None,
    active_engine_count: int | None = None,
    submitted_edit_count: int | None = None,
    result_count: int | None = None,
    last_error: str = "",
) -> None:
    if registered is not None:
        _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["registered"] = registered
    if suppressed is not None:
        _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["suppressed"] = suppressed
    if active_engine_count is not None:
        _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["last_active_viewport_engine_count"] = active_engine_count
    if submitted_edit_count is not None:
        _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["last_submitted_edit_count"] = submitted_edit_count
    if result_count is not None:
        _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["last_result_count"] = result_count
    _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["last_error"] = last_error
    _INTERACTIVE_EDIT_BRIDGE_DIAGNOSTICS["updated_at_ns"] = time.time_ns()


def register_interactive_edit_bridge(bpy_module: Any, handler: Any) -> bool:
    """Register the stock-Blender depsgraph bridge used for live viewport edits."""

    if bpy_module is None:
        return False
    handlers = bpy_module.app.handlers.depsgraph_update_post
    if handler in handlers:
        record_interactive_edit_bridge_diagnostics(registered=True)
        return False
    handlers.append(handler)
    record_interactive_edit_bridge_diagnostics(registered=True)
    return True


def unregister_interactive_edit_bridge(bpy_module: Any, handler: Any) -> bool:
    """Unregister the stock-Blender depsgraph bridge."""

    if bpy_module is None:
        return False
    handlers = bpy_module.app.handlers.depsgraph_update_post
    removed = False
    while handler in handlers:
        handlers.remove(handler)
        removed = True
    record_interactive_edit_bridge_diagnostics(registered=False)
    return removed


def resolve_blender_selection_to_edit_owners(
    context: Any | None = None,
    *,
    bpy_module: Any | None = None,
) -> dict[str, Any]:
    """Resolve selected native selection sources to their Blender edit owners."""

    resolved_context = context
    if resolved_context is None:
        if bpy_module is None:
            return _record_selection_resolution_diagnostics()
        resolved_context = getattr(bpy_module, "context", None)
    if resolved_context is None:
        return _record_selection_resolution_diagnostics()

    selected_objects = list(getattr(resolved_context, "selected_objects", ()) or ())
    if not selected_objects:
        return _record_selection_resolution_diagnostics(selected_object_count=0)

    active_source = _active_object_from_context(resolved_context)
    resolved_owners: list[Any] = []
    missing_owner_count = 0
    owner_selection_required = False
    last_source_name = ""
    last_owner_name = ""
    source_records: list[dict[str, Any]] = []

    for obj in selected_objects:
        record, owner = _resolve_selection_source(resolved_context, obj, bpy_module=bpy_module)
        source_records.append(record)
        if record["unresolved_reason"] == "selection_owner_object_missing":
            missing_owner_count += 1
            continue
        if owner is None:
            continue
        last_source_name = str(record["source_name"])
        resolved_owners.append(owner)
        last_owner_name = str(record["owner_name"])
        if owner is not obj:
            owner_selection_required = True

    unique_owners = _dedupe_objects(resolved_owners)
    group_supported = bool(selected_objects) and all(record["status"] == "resolved" for record in source_records)
    group_rejected = bool(selected_objects) and not group_supported
    changed = owner_selection_required and not group_rejected
    if changed:
        view_layer = getattr(resolved_context, "view_layer", None)
        with suppress_interactive_edit_bridge():
            for obj in selected_objects:
                if not _contains_object_identity(unique_owners, obj):
                    _select_object(obj, False, view_layer=view_layer)
            for owner in unique_owners:
                _select_object(owner, True, view_layer=view_layer)
            _set_active_object(
                resolved_context,
                _resolved_active_owner(
                    resolved_context,
                    active_source,
                    unique_owners,
                    bpy_module=bpy_module,
                ),
            )

    return _record_selection_resolution_diagnostics(
        changed=changed,
        selected_object_count=len(selected_objects),
        resolved_owner_count=len(unique_owners),
        missing_owner_count=missing_owner_count,
        last_source_name=last_source_name,
        last_owner_name=last_owner_name,
        group_supported=group_supported,
        group_rejected=group_rejected,
        sources=source_records,
    )


def request_at_initial_condition(request: Any) -> bool:
    timeline_frame = int(request.timeline_frame)
    return (
        bool(request.timeline_controls_enabled)
        and not bool(request.timeline_playing)
        and (timeline_frame == 0 or timeline_frame == int(request.timeline_start))
    )


def should_mirror_runtime_poses(*, at_initial_condition: bool, lock_was_active: bool) -> bool:
    return not at_initial_condition or lock_was_active


def prepare_runtime_pose_mirror(
    body_poses: tuple[BodyPose, ...],
    bpy_module: Any,
    context: Any | None = None,
    *,
    lock_runtime_owned: bool = True,
    owning_generation: int | None = None,
    last_applied: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive pending runtime pose mirror work and diagnostics from scene state."""

    if not body_poses:
        return {}, {
            "enabled": True,
            "source_authority": "OVPhysX",
            "available_body_pose_count": 0,
            "mirrored_count": 0,
            "mirrored_paths": [],
        }
    if bpy_module is None:
        return {}, {"enabled": False, "reason": "bpy_unavailable"}

    scene = getattr(context, "scene", None) if context is not None else None
    scene_source = "viewport_context" if scene is not None else "global_context"
    if scene is None:
        scene = getattr(getattr(bpy_module, "context", None), "scene", None)
    if scene is None:
        return {}, {
            "enabled": False,
            "reason": "scene_unavailable",
            "source_authority": "OVPhysX",
            "available_body_pose_count": len(body_poses),
            "scene_source": scene_source,
        }

    poses_by_path = {pose.prim_path: pose for pose in body_poses}
    matched_paths: set[str] = set()
    matched_pose_count = 0
    interaction_object_paths: list[str] = []
    pending_poses: dict[str, dict[str, tuple[float, ...]]] = {}
    inspected_object_count = 0
    interaction_object_count = 0
    for obj in getattr(scene, "objects", ()):
        inspected_object_count += 1
        try:
            prim_path = str(obj.get("ovrtx.usd_prim_path", ""))
        except Exception:
            continue
        if prim_path:
            interaction_object_count += 1
            if len(interaction_object_paths) < 12:
                interaction_object_paths.append(prim_path)
        pose = poses_by_path.get(prim_path)
        if pose is None:
            continue
        matched_pose_count += 1
        matched_paths.add(prim_path)
        pending_poses[prim_path] = {
            "translate": tuple(float(value) for value in pose.translate),
            "orient": tuple(float(value) for value in pose.orient),
        }

    pending = {}
    if pending_poses:
        pending = {
            "scene_name": str(getattr(scene, "name", "")),
            "poses_by_path": pending_poses,
            "source_authority": "OVPhysX",
            "scene_source": scene_source,
            "lock_runtime_owned": bool(lock_runtime_owned),
            "owning_generation": owning_generation,
        }

    missing_object_paths = sorted(set(poses_by_path) - matched_paths)
    return pending, {
        "enabled": True,
        "source_authority": "OVPhysX",
        "scene_source": scene_source,
        "status": "scheduled" if pending_poses else "no_matching_object",
        "available_body_pose_count": len(poses_by_path),
        "pose_path_samples": sorted(poses_by_path)[:12],
        "inspected_object_count": inspected_object_count,
        "interaction_object_count": interaction_object_count,
        "interaction_object_paths": interaction_object_paths,
        "matched_pose_count": matched_pose_count,
        "scheduled_mirror_count": len(pending_poses),
        "last_applied": dict(last_applied or {}),
        "mirrored_count": 0,
        "mirrored_paths": [],
        "missing_object_paths": missing_object_paths[:12],
    }


def apply_pending_runtime_pose_mirror(
    bpy_module: Any,
    pending: Mapping[str, Any],
    physics_playback_lock: Any,
) -> dict[str, Any] | None:
    """Apply scheduled runtime poses to Blender interaction objects."""

    poses_by_path = pending.get("poses_by_path", {}) if isinstance(pending, Mapping) else {}
    if not isinstance(poses_by_path, Mapping) or not poses_by_path:
        return None
    if bpy_module is None:
        return {
            "enabled": False,
            "status": "failed",
            "reason": "bpy_unavailable",
            "available_body_pose_count": len(poses_by_path),
        }
    try:
        from mathutils import Matrix, Quaternion, Vector  # type: ignore
    except Exception as exc:
        return {
            "enabled": False,
            "status": "failed",
            "reason": f"mathutils_unavailable:{type(exc).__name__}",
        }

    scene_name = str(pending.get("scene_name", ""))
    scene = bpy_module.data.scenes.get(scene_name) if scene_name else None
    if scene is None:
        scene = getattr(getattr(bpy_module, "context", None), "scene", None)
    if scene is None:
        return {
            "enabled": False,
            "status": "failed",
            "reason": "scene_unavailable",
            "available_body_pose_count": len(poses_by_path),
        }

    mirrored_paths: list[str] = []
    matched_paths: set[str] = set()
    matched_objects: dict[str, Any] = {}
    mirrored_objects: dict[str, Any] = {}
    mirror_error_count = 0
    first_mirror_error = ""
    with suppress_interactive_edit_bridge():
        for obj in getattr(scene, "objects", ()):
            try:
                prim_path = str(obj.get("ovrtx.usd_prim_path", ""))
            except Exception:
                continue
            pose = poses_by_path.get(prim_path)
            if not isinstance(pose, Mapping):
                continue
            matched_paths.add(prim_path)
            matched_objects[prim_path] = obj
            try:
                translate = tuple(float(value) for value in pose["translate"])
                orient = tuple(float(value) for value in pose["orient"])
                scale = obj.matrix_world.decompose()[2]
                quat = Quaternion((orient[3], orient[0], orient[1], orient[2]))
                matrix_world = Matrix.LocRotScale(Vector(translate), quat, scale)
                offset = _matrix_from_blender_id_property(
                    obj.get("ovrtx.body_visual_offset_matrix", None),
                    Matrix,
                )
                if offset is not None:
                    matrix_world = matrix_world @ offset
                obj.matrix_world = matrix_world
            except Exception as exc:
                mirror_error_count += 1
                if not first_mirror_error:
                    first_mirror_error = f"{type(exc).__name__}: {exc}"
                continue
            mirrored_paths.append(prim_path)
            mirrored_objects[prim_path] = obj
        try:
            view_layer = getattr(getattr(bpy_module, "context", None), "view_layer", None)
            if view_layer is not None:
                view_layer.update()
        except Exception:
            pass

    lock_enabled = bool(pending.get("lock_runtime_owned", True))
    owning_generation = pending.get("owning_generation", None)
    if lock_enabled:
        for prim_path, obj in matched_objects.items():
            physics_playback_lock.lock_object(
                prim_path,
                obj,
                generation=_optional_int(owning_generation),
                reason="active_physics_generation",
            )
        physics_playback_lock.unlock_missing_paths(set(matched_objects))

    missing_object_paths = sorted(set(str(path) for path in poses_by_path) - set(mirrored_paths))
    return {
        "enabled": True,
        "source_authority": str(pending.get("source_authority", "OVPhysX")),
        "scene_source": str(pending.get("scene_source", "")),
        "status": "applied" if mirrored_paths else "failed",
        "available_body_pose_count": len(poses_by_path),
        "matched_pose_count": len(matched_paths),
        "mirror_error_count": mirror_error_count,
        "first_mirror_error": first_mirror_error,
        "mirrored_count": len(mirrored_paths),
        "mirrored_paths": mirrored_paths,
        "missing_object_paths": missing_object_paths[:12],
    }


class PhysicsPlaybackLock:
    """Tracks stock-Blender locks while OVPhysX owns physics playback state."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._rejected_edit_count = 0
        self._last_rejected_data_authority = ""
        self._last_rejected_edit_path = ""
        self._ignored_internal_update_count = 0
        self._last_reason = ""
        self._owning_generation: int | None = None
        self._frame1_cleared = False
        self._updated_at_ns = 0

    def lock_object(
        self,
        prim_path: str,
        obj: Any,
        *,
        generation: int | None,
        reason: str = "active_physics_generation",
    ) -> None:
        if not prim_path:
            return
        record = self._records.get(prim_path)
        if record is None:
            record = {
                "object": obj,
                "object_name": str(getattr(obj, "name", "")),
                "previous_locks": _object_transform_lock_values(obj),
            }
            self._records[prim_path] = record
        else:
            record["object"] = obj
            record["object_name"] = str(getattr(obj, "name", ""))
        record["matrix_world"] = _copy_matrix_world(getattr(obj, "matrix_world", None))
        try:
            record["edit_value"] = object_transform_edit(obj, data_authority=DataAuthority.VIEW).value
        except Exception:
            record["edit_value"] = None
        with suppress_interactive_edit_bridge():
            _set_object_transform_locks(obj, True)
        self._owning_generation = generation
        self._last_reason = reason
        self._frame1_cleared = False
        self._updated_at_ns = time.time_ns()

    def unlock_missing_paths(self, active_paths: set[str]) -> None:
        for prim_path in sorted(set(self._records) - set(active_paths)):
            self._unlock_path(prim_path)

    def clear(self, *, reason: str, frame1_cleared: bool = False) -> None:
        for prim_path in sorted(self._records):
            self._unlock_path(prim_path)
        self._last_reason = reason
        self._owning_generation = None
        if frame1_cleared:
            self._frame1_cleared = True
        self._updated_at_ns = time.time_ns()

    def is_active(self) -> bool:
        return bool(self._records)

    def reject_edit(self, edit: InteractiveEdit) -> EditWorkflowResult | None:
        prim_path = str(getattr(edit, "usd_prim_path", ""))
        if not _physics_playback_locks_edit(edit) or prim_path not in self._records:
            return None
        record = self._records[prim_path]
        if edit.shape == EditShape.VALUE and edit.blender_property_path == "matrix_world" and edit.value == record.get("edit_value"):
            self._ignored_internal_update_count += 1
            self._updated_at_ns = time.time_ns()
            return EditWorkflowResult(
                action=WorkflowAction.UNSUPPORTED,
                status=EditStatus.UNSUPPORTED,
                reason="physics_playback_lock_internal_update",
                diagnostics={
                    "physics_playback_lock": True,
                    "usd_prim_path": prim_path,
                    "shape": edit.shape.value,
                    "data_authority": edit.data_authority.value,
                    "internal_update": True,
                },
            )
        self._rejected_edit_count += 1
        self._last_rejected_data_authority = edit.data_authority.value
        self._last_rejected_edit_path = prim_path
        self._last_reason = "physics_playback_lock"
        self._restore_locked_object(prim_path)
        self._updated_at_ns = time.time_ns()
        return EditWorkflowResult(
            action=WorkflowAction.UNSUPPORTED,
            status=EditStatus.UNSUPPORTED,
            reason="physics_playback_locked",
            diagnostics={
                "physics_playback_lock": True,
                "usd_prim_path": prim_path,
                "shape": edit.shape.value,
                "data_authority": edit.data_authority.value,
                "discarded_attempted_value": True,
            },
        )

    def diagnostics(self) -> dict[str, Any]:
        locked_paths = sorted(self._records)
        return {
            "active": bool(self._records),
            "reason": self._last_reason,
            "owning_physics_generation": self._owning_generation,
            "locked_object_count": len(locked_paths),
            "locked_object_paths": locked_paths[:12],
            "rejected_edit_count": self._rejected_edit_count,
            "ignored_internal_update_count": self._ignored_internal_update_count,
            "last_rejected_data_authority": self._last_rejected_data_authority,
            "last_rejected_edit_path": self._last_rejected_edit_path,
            "frame1_cleared": self._frame1_cleared,
            "updated_at_ns": self._updated_at_ns,
        }

    def _unlock_path(self, prim_path: str) -> None:
        record = self._records.pop(prim_path, None)
        if record is None:
            return
        obj = record.get("object")
        if obj is not None:
            with suppress_interactive_edit_bridge():
                _restore_object_transform_locks(obj, record.get("previous_locks", {}))

    def _restore_locked_object(self, prim_path: str) -> None:
        record = self._records.get(prim_path)
        if record is None:
            return
        obj = record.get("object")
        matrix_world = record.get("matrix_world")
        if obj is None or matrix_world is None:
            return
        try:
            with suppress_interactive_edit_bridge():
                obj.matrix_world = _copy_matrix_world(matrix_world)
        except Exception:
            return


def _record_selection_resolution_diagnostics(
    *,
    changed: bool = False,
    selected_object_count: int = 0,
    resolved_owner_count: int = 0,
    missing_owner_count: int = 0,
    group_supported: bool = True,
    group_rejected: bool = False,
    last_source_name: str = "",
    last_owner_name: str = "",
    sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    source_records = [dict(source) for source in sources]
    unresolved_reasons = sorted(
        {
            str(source.get("unresolved_reason", ""))
            for source in source_records
            if source.get("unresolved_reason", "")
        }
    )
    owner_names = [
        str(source.get("owner_name", ""))
        for source in source_records
        if source.get("owner_name", "")
    ]
    owner_categories = sorted(
        {
            str(source.get("owner_category", ""))
            for source in source_records
            if source.get("owner_category", "")
        }
    )
    if selected_object_count <= 0:
        status = "no_selection"
    elif group_rejected:
        status = "unsupported_selection_group"
    else:
        status = "resolved"
    _SELECTION_RESOLUTION_DIAGNOSTICS.update(
        {
            "status": status,
            "changed": bool(changed),
            "selected_object_count": int(selected_object_count),
            "resolved_owner_count": int(resolved_owner_count),
            "missing_owner_count": int(missing_owner_count),
            "group_supported": bool(group_supported),
            "group_rejected": bool(group_rejected),
            "last_source_name": str(last_source_name),
            "last_owner_name": str(last_owner_name),
            "owner_names": owner_names,
            "owner_categories": owner_categories,
            "unresolved_reasons": unresolved_reasons,
            "sources": source_records,
            "updated_at_ns": time.time_ns(),
        }
    )
    return dict(_SELECTION_RESOLUTION_DIAGNOSTICS)


def _resolve_selection_source(context: Any, obj: Any, *, bpy_module: Any | None) -> tuple[dict[str, Any], Any | None]:
    owner_name = usd_paths.selection_owner_object_name(obj)
    if owner_name:
        owner = _find_object_by_name(context, owner_name, bpy_module=bpy_module)
        if owner is None:
            return (
                _selection_resolution_record(
                    obj,
                    owner=None,
                    owner_name=owner_name,
                    status="unresolved",
                    ownership_source="usd_selection_owner",
                    unresolved_reason="selection_owner_object_missing",
                ),
                None,
            )
        return (
            _selection_resolution_record(
                obj,
                owner=owner,
                owner_name=owner_name,
                status="resolved",
                ownership_source="usd_selection_owner",
            ),
            owner,
        )

    owner_path = _object_usd_path(obj)
    data_authority = _object_data_authority(obj)
    if data_authority:
        return (
            _selection_resolution_record(
                obj,
                owner=obj,
                owner_name=str(getattr(obj, "name", "")),
                status="resolved",
                ownership_source="usd_edit_owner",
            ),
            obj,
        )
    if owner_path:
        return (
            _selection_resolution_record(
                obj,
                owner=None,
                owner_name="",
                status="preview_only",
                ownership_source="usd_source_identity",
                unresolved_reason="preview_only_selection_source",
            ),
            None,
        )
    return (
        _selection_resolution_record(
            obj,
            owner=None,
            owner_name="",
            status="unresolved",
            ownership_source="none",
            unresolved_reason="unmapped_selection_source",
        ),
        None,
    )


def _selection_resolution_record(
    source: Any,
    *,
    owner: Any | None,
    owner_name: str,
    status: str,
    ownership_source: str,
    unresolved_reason: str = "",
) -> dict[str, Any]:
    resolved = status == "resolved"
    owner_obj = owner if owner is not None else (source if resolved else None)
    source_usd_path = usd_paths.source_usd_path_from_blender_id(source)
    owner_usd_path = _selection_owner_usd_path(source) or (_object_usd_path(owner_obj) if resolved else "")
    data_authority = _object_data_authority(owner_obj)
    owner_category = _owner_category(owner_obj, owner_usd_path=owner_usd_path) if resolved else "inspection_only"
    mapping_basis = _mapping_basis(ownership_source)
    preview_only = status != "resolved"
    usd_layer_id = _object_string_property(
        owner_obj,
        usd_paths.USD_LAYER_ID_PROP,
    )
    write_target_available = (
        status == "resolved"
        and not preview_only
        and bool(owner_usd_path)
        and bool(usd_layer_id)
    )
    source_data = getattr(source, "data", None)
    owner_data = getattr(owner_obj, "data", None)
    return {
        "status": status,
        "source_name": str(getattr(source, "name", "")),
        "source_type": str(getattr(source, "type", "")),
        "source_session_uid": int(getattr(source, "session_uid", 0) or 0),
        "source_data_session_uid": int(
            getattr(source_data, "session_uid", 0) or 0
        ),
        "source_usd_path": source_usd_path,
        "owner_name": str(owner_name),
        "owner_session_uid": int(getattr(owner_obj, "session_uid", 0) or 0),
        "owner_data_session_uid": int(
            getattr(owner_data, "session_uid", 0) or 0
        ),
        "owner_category": owner_category,
        "owner_usd_path": owner_usd_path,
        "ownership_source": ownership_source,
        "mapping_basis": mapping_basis,
        "inferred_mapping": mapping_basis == "inferred",
        "preview_only": preview_only,
        "persistence": "write" if write_target_available else "none",
        "write_target_available": write_target_available,
        "unresolved_reason": unresolved_reason,
        "edit_target_identity": {
            "usd_layer_id": usd_layer_id,
            "usd_prim_path": owner_usd_path,
            "usd_attribute": _object_string_property(owner_obj, "ovrtx.usd_attribute"),
            "usd_property_path": _object_string_property(owner_obj, usd_paths.USD_PROPERTY_PATH_PROP),
            "blender_property_path": _object_string_property(owner_obj, usd_paths.BLENDER_PROPERTY_PATH_PROP),
            "data_authority": data_authority,
        },
    }


def _mapping_basis(ownership_source: str) -> str:
    if ownership_source in {"usd_edit_owner", "usd_selection_owner"}:
        return "usd_path_owner"
    if ownership_source == "usd_source_identity":
        return "source_identity"
    if ownership_source.startswith("inferred_"):
        return "inferred"
    return "none"


def _owner_category(obj: Any, *, owner_usd_path: str) -> str:
    data_authority = _object_data_authority(obj)
    blender_property_path = _object_string_property(obj, usd_paths.BLENDER_PROPERTY_PATH_PROP)
    if not owner_usd_path and not data_authority:
        return "inspection_only"
    if bool(_object_property(obj, "ovrtx.physics_backed", False)) or data_authority == DataAuthority.SIM.value:
        return "physics_body"
    if blender_property_path in {"diffuse_color", "roughness", "metallic"}:
        return "material_owner"
    if blender_property_path in {"energy", "data.type", "data.shape"} or str(getattr(obj, "type", "")) == "LIGHT":
        return "light_owner"
    if blender_property_path == "world_dome":
        return "world_owner"
    if blender_property_path == "matrix_world":
        return "view_value_owner"
    if owner_usd_path:
        return "mapped_render_prim"
    return "inspection_only"


def _object_usd_path(obj: Any) -> str:
    return (
        usd_paths.clean_usd_path(_object_property(obj, usd_paths.USD_PRIM_PATH_PROP, ""))
        or usd_paths.source_usd_path_from_blender_id(obj)
    )


def _selection_owner_usd_path(obj: Any) -> str:
    return usd_paths.clean_usd_path(
        _object_property(obj, usd_paths.SELECTION_OWNER_USD_PRIM_PATH_PROP, "")
    )


def _object_data_authority(obj: Any) -> str:
    return str(_object_property(obj, usd_paths.DATA_AUTHORITY_PROP, "") or "")


def _physics_playback_locks_edit(edit: InteractiveEdit) -> bool:
    if edit.data_authority == DataAuthority.SIM:
        return True
    if edit.shape == EditShape.TOPOLOGY:
        return True
    return edit.shape == EditShape.VALUE and edit.blender_property_path == "matrix_world"


def _object_string_property(obj: Any, key: str) -> str:
    return str(_object_property(obj, key, "") or "")


def _object_property(obj: Any, key: str, default: Any) -> Any:
    return usd_paths.id_property(obj, key, default)


def _active_object_from_context(context: Any) -> Any | None:
    view_layer = getattr(context, "view_layer", None)
    objects = getattr(view_layer, "objects", None)
    if objects is not None and hasattr(objects, "active"):
        return getattr(objects, "active", None)
    return getattr(context, "object", None)


def _resolved_active_owner(
    context: Any,
    active_source: Any | None,
    owners: list[Any],
    *,
    bpy_module: Any | None,
) -> Any | None:
    if active_source is not None:
        owner_name = usd_paths.selection_owner_object_name(active_source)
        if owner_name:
            owner = _find_object_by_name(context, owner_name, bpy_module=bpy_module)
            if owner is not None:
                return owner
        if _contains_object_identity(owners, active_source):
            return active_source
    if owners:
        return owners[-1]
    return None


def _set_active_object(context: Any, obj: Any | None) -> None:
    view_layer = getattr(context, "view_layer", None)
    objects = getattr(view_layer, "objects", None)
    if objects is not None and hasattr(objects, "active"):
        try:
            objects.active = obj
        except Exception:
            return


def _find_object_by_name(context: Any, name: str, *, bpy_module: Any | None) -> Any | None:
    scene = getattr(context, "scene", None)
    scene_objects = getattr(scene, "objects", None)
    obj = _get_object_by_name(scene_objects, name)
    if obj is not None:
        return obj
    if bpy_module is None:
        return None
    return _get_object_by_name(getattr(getattr(bpy_module, "data", None), "objects", None), name)


def _get_object_by_name(objects: Any, name: str) -> Any | None:
    if objects is None or not name:
        return None
    getter = getattr(objects, "get", None)
    if callable(getter):
        return getter(name)
    try:
        for obj in objects:
            if getattr(obj, "name", "") == name:
                return obj
    except TypeError:
        return None
    return None


def _dedupe_objects(objects: list[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[int] = set()
    for obj in objects:
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        unique.append(obj)
    return unique


def _contains_object_identity(objects: list[Any], target: Any) -> bool:
    target_id = id(target)
    return any(id(obj) == target_id for obj in objects)


def _select_object(
    obj: Any,
    selected: bool,
    *,
    view_layer: Any | None = None,
) -> None:
    select_set = getattr(obj, "select_set", None)
    if callable(select_set):
        try:
            if view_layer is None:
                select_set(bool(selected))
            else:
                select_set(bool(selected), view_layer=view_layer)
        except TypeError:
            try:
                select_set(bool(selected))
            except Exception:
                return
        except Exception:
            return


def _object_transform_lock_values(obj: Any) -> dict[str, tuple[bool, ...]]:
    locks: dict[str, tuple[bool, ...]] = {}
    for attr in ("lock_location", "lock_rotation", "lock_scale"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        try:
            locks[attr] = tuple(bool(item) for item in value)
        except TypeError:
            continue
    return locks


def _set_object_transform_locks(obj: Any, locked: bool) -> None:
    for attr in ("lock_location", "lock_rotation", "lock_scale"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        try:
            length = len(value)
        except TypeError:
            continue
        _assign_bool_sequence(obj, attr, tuple(bool(locked) for _ in range(length)))


def _restore_object_transform_locks(obj: Any, locks: Mapping[str, Sequence[bool]]) -> None:
    for attr, values in locks.items():
        _assign_bool_sequence(obj, attr, tuple(bool(value) for value in values))


def _assign_bool_sequence(obj: Any, attr: str, values: tuple[bool, ...]) -> None:
    try:
        setattr(obj, attr, values)
        return
    except Exception:
        pass
    current = getattr(obj, attr, None)
    try:
        for index, value in enumerate(values):
            current[index] = bool(value)
    except Exception:
        pass


def _copy_matrix_world(matrix_world: Any) -> Any:
    copier = getattr(matrix_world, "copy", None)
    if callable(copier):
        return copier()
    return matrix_world


def _matrix_from_blender_id_property(value: Any, matrix_type: Any) -> Any | None:
    if value in (None, ""):
        return None
    try:
        values = _plain_matrix_property_values(value)
        if len(values) == 4 and all(isinstance(row, Sequence) and len(row) == 4 for row in values):
            return matrix_type(values)
        if len(values) == 16:
            return matrix_type([values[index : index + 4] for index in range(0, 16, 4)])
    except Exception:
        return None
    return None


def _plain_matrix_property_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_matrix_property_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_matrix_property_values(item) for item in value)
    try:
        return tuple(value)
    except TypeError:
        return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
