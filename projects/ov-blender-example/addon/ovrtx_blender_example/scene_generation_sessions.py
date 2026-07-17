# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender-scene generation owners and file-lifetime handlers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping

from .generation_runtime_adapters import (
    OvphysxGenerationAdapter,
    OvrtxGenerationAdapter,
    generation_requires_physics,
)
from .render_requests import RenderRequest
from .runtime_scheduler import RuntimeScheduler, RuntimeTickResult, RuntimeTickStatus
from .interactive_edit_planner import (
    EditMechanism,
    EditStatus,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from .interactive_edit_workflow import (
    EditWorkflowResult,
    InteractiveEditWorkflow,
    WorkflowAction,
)
from . import interactive_operator_state
from . import scene_generation
from .scene_generation import BlenderId, SceneGeneration, SceneGenerationOwner, blender_id
from .usd_prim_resolver import UsdPrimResolver

try:
    import bpy  # type: ignore
    from bpy.app.handlers import persistent  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]

    def persistent(function: Any) -> Any:
        return function


_owners: dict[int, SceneGenerationOwner] = {}
_runtimes: dict[int, "AuthoringGenerationRuntime"] = {}
_dirty: dict[int, set[BlenderId]] = {}
_reconciling: set[int] = set()
_blocked_reconciliations: dict[int, tuple[str, set[BlenderId]]] = {}
_pending_affected: dict[int, set[BlenderId]] = {}
_constructing: set[int] = set()
_load_pre_completed = False
_load_post_count = 0
_initial_generation_revision = 0
_initial_generation_diagnostics: dict[str, Any] = {
    "status": "unavailable",
    "scene_uid": 0,
    "error": "",
    "last_error": "",
}
_preparation_condition = threading.Condition()
_preparation_pending: dict[int, int] = {}
_preparation_revisions: dict[int, int] = {}
_preparation_thread: threading.Thread | None = None
_preparation_stopping = False
_preparation_paused = False
_playback_errors: dict[int, str] = {}
_rejected_unscoped_dirty_requests: dict[int, int] = {}
_accepted_world_dirty_requests: dict[int, int] = {}
_ignored_scene_updates: dict[int, int] = {}
_reconciliation_timers: set[int] = set()
PREPARATION_RETRY_BACKOFF_SECONDS = 0.5


def fail_closed_runtime_reuse(runtime: "AuthoringGenerationRuntime") -> None:
    """Retain and disable one runtime after unconfirmed teardown."""

    runtime.reuse_blocked = True
    with _preparation_condition:
        for uid, candidate in tuple(_runtimes.items()):
            if candidate is runtime:
                _preparation_pending.pop(uid, None)
        _preparation_condition.notify_all()


def runtime_reuse_blocked(scene: Any) -> bool:
    runtime = _runtimes.get(_scene_uid(scene))
    return bool(runtime is not None and getattr(runtime, "reuse_blocked", False))


def _require_runtime_reuse(runtime: "AuthoringGenerationRuntime" | None) -> None:
    if runtime is not None and getattr(runtime, "reuse_blocked", False):
        raise RuntimeError(
            "viewport runtime reuse is disabled after a teardown deadline was exceeded"
        )


class AuthoringGenerationRuntime:
    """Own one Blender scene's scheduler and prepared OVRTX/OVPhysX state."""

    def __init__(
        self,
        controller: Any,
        owner: SceneGenerationOwner,
    ) -> None:
        self.viewport_ids: set[str] = set()
        self.owner = owner
        self.ovrtx = OvrtxGenerationAdapter(controller)
        self.ovphysx = OvphysxGenerationAdapter()
        self.scheduler = RuntimeScheduler(
            controller_provider=lambda: self.ovphysx.controller,
            controller_reset=self.ovphysx.reset,
            ovrtx_transform_sink=owner.retain_transform_values,
            ovrtx_attribute_sink=owner.retain_attribute_values,
            ovphysx_initial_condition_sink=owner.retain_initial_conditions,
        )
        self.workflow = InteractiveEditWorkflow(runtime_scheduler=self.scheduler)
        self.playback_lock = interactive_operator_state.PhysicsPlaybackLock()
        self._admission_lock = threading.Lock()
        self.lifecycle_status = "open"
        self._presentation_lock = threading.Lock()
        self._presentation_wakes: dict[str, Callable[[], None]] = {}
        self.scheduler.set_edit_wake_hook(self._wake_presentations)
        self._generation_number: int | None = None
        self._request: RenderRequest | None = None
        self.last_activation_update = RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
        )
        self.preparation_status = "unavailable"
        self.preparation_error = ""
        self.reuse_blocked = False
        self._preparation_lock = threading.RLock()

    def attach(
        self,
        viewport_id: str,
        wake_hook: Callable[[], None] | None = None,
    ) -> None:
        if self.lifecycle_status != "open":
            raise RuntimeError("authoring runtime is closing")
        with self._presentation_lock:
            self.viewport_ids.add(viewport_id)
            if wake_hook is not None:
                self._presentation_wakes[viewport_id] = wake_hook

    def detach(self, viewport_id: str) -> None:
        with self._presentation_lock:
            self.viewport_ids.discard(viewport_id)
            self._presentation_wakes.pop(viewport_id, None)

    def _wake_presentations(self) -> None:
        with self._presentation_lock:
            wake_hooks = tuple(self._presentation_wakes.values())
        for wake_hook in wake_hooks:
            wake_hook()

    def submit_edit_group(
        self,
        edits: tuple[InteractiveEdit, ...],
    ) -> tuple[EditWorkflowResult, ...]:
        """Apply playback admission, then commit one complete group."""

        with self._admission_lock:
            if self.lifecycle_status != "open":
                return tuple(
                    EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.FAILED,
                        reason="authoring_runtime_closing",
                    )
                    for _edit in edits
                )
            locked = tuple(self.playback_lock.reject_edit(edit) for edit in edits)
            rejection = next((result for result in locked if result is not None), None)
            if rejection is not None:
                return tuple(
                    result
                    if result is not None
                    else EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.UNSUPPORTED,
                        reason="edit_group_rejected",
                    )
                    for result in locked
                )
            return self.workflow.preview_edit_group(edits)

    def replay_retained_values(self, generation: SceneGeneration) -> None:
        if self._request is None:
            return
        with self._preparation_lock, self.ovrtx.controller._serialized_transport():
            self.preparation_status = "preparing"
            self.last_activation_update = RuntimeTickResult(
                status=RuntimeTickStatus.NOOP,
                enabled=True,
            )
            try:
                transforms, attributes, _initial_conditions = self.owner.retained_values_for(
                    generation
                )
                self.ovrtx.update_request(self._request)
                if not self._activate_ovrtx(generation, transforms, attributes):
                    raise RuntimeError(self.ovrtx.last_error or "retained value replay failed")
                if not self._finish_activation(generation):
                    raise RuntimeError(self.ovrtx.last_error or "pending value replay failed")
                note_applied = getattr(self.scheduler, "note_applied_content", None)
                if callable(note_applied):
                    note_applied()
            except Exception as exc:
                self.preparation_status = "failed"
                self.preparation_error = f"{type(exc).__name__}: {exc}"
                raise

    def activate_blocking(
        self,
        generation: SceneGeneration,
        request: RenderRequest,
        *,
        predecessor: SceneGeneration | None = None,
    ) -> None:
        with self._preparation_lock, self.ovrtx.controller._serialized_transport():
            if self.lifecycle_status != "open":
                raise RuntimeError("authoring runtime is closing")
            self.activate(generation, request, predecessor=predecessor)

    def activate(
        self,
        generation: SceneGeneration,
        request: RenderRequest,
        *,
        predecessor: SceneGeneration | None = None,
    ) -> None:
        previous_request = getattr(self.ovrtx, "request", None)
        self._request = request
        self.preparation_status = "preparing"
        self.last_activation_update = RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
        )
        transforms, attributes, initial_conditions = self.owner.retained_values_for(
            generation
        )
        sim_update: RuntimeTickResult | None = None
        try:
            self.ovrtx.update_request(request)
            replacing = self._generation_number != generation.number
            ovrtx_replacing = (
                replacing or self.ovrtx.active_generation != generation.number
            )
            if replacing:
                if (
                    generation_requires_physics(generation)
                    and self.ovphysx.active_generation != generation.number
                ):
                    if not self.ovphysx.activate(
                        generation,
                        initial_conditions=initial_conditions,
                    ):
                        raise RuntimeError(
                            self.ovphysx.last_error or "OVPhysX generation activation failed"
                        )
                elif self.ovphysx.active_generation is not None:
                    status = self.ovphysx.deactivate()
                    if status not in {"stopped", "not_found"}:
                        raise RuntimeError("OVPhysX generation deactivation failed")
            if getattr(self.scheduler, "has_pending_sim_updates", False):
                if self.ovphysx.controller is None:
                    raise RuntimeError("SIM edit target has no active physics runtime")
                sim_update = self.scheduler.apply_pending_sim_values(
                    self.ovphysx.controller
                )
                self.last_activation_update = sim_update
                if sim_update.status in {
                    RuntimeTickStatus.BUSY,
                    RuntimeTickStatus.FAILED,
                }:
                    raise RuntimeError(
                        sim_update.skipped_reason or "retained SIM value replay failed"
                    )
            if ovrtx_replacing:
                ready = self._activate_ovrtx(generation, transforms, attributes)
            else:
                ready = self.ovrtx.ensure_request()
            if not ready:
                raise RuntimeError(
                    self.ovrtx.last_error or "OVRTX generation activation failed"
                )
            if not self._finish_activation(
                generation,
                (() if sim_update is None else (sim_update,)),
            ):
                raise RuntimeError(
                    self.ovrtx.last_error or "pending value replay failed"
                )
            if (
                ovrtx_replacing
                or self.last_activation_update.should_reset_refinement
                or bool(
                    getattr(self.ovrtx.last_ensure_result, "session_started", False)
                )
            ):
                note_applied = getattr(self.scheduler, "note_applied_content", None)
                if callable(note_applied):
                    note_applied()
        except Exception as exc:
            self.preparation_status = "failed"
            self.preparation_error = f"{type(exc).__name__}: {exc}"
            if predecessor is None:
                raise
            restored = self._restore_predecessor(predecessor, previous_request)
            if not restored:
                self._generation_number = None
                raise RuntimeError(
                    f"{exc}; predecessor_restore=failed"
                ) from exc
            raise RuntimeError(
                f"{exc}; predecessor_restore=succeeded"
            ) from exc

    def _restore_predecessor(
        self,
        predecessor: SceneGeneration,
        request: RenderRequest | None,
    ) -> bool:
        try:
            transforms, attributes, initial_conditions = self.owner.retained_values_for(
                predecessor
            )
            if generation_requires_physics(predecessor):
                if not self.ovphysx.activate(
                    predecessor,
                    initial_conditions=initial_conditions,
                ):
                    return False
            elif self.ovphysx.active_generation is not None:
                status = self.ovphysx.deactivate()
                if status not in {"stopped", "not_found"}:
                    return False
            if request is None:
                request = RenderRequest(input_usd_path=predecessor.usd_path)
            self.ovrtx.update_request(request)
            if not self._activate_ovrtx(predecessor, transforms, attributes):
                return False
        except Exception:
            return False
        self._generation_number = predecessor.number
        return True

    def _activate_ovrtx(
        self,
        generation: SceneGeneration,
        transforms: tuple[Any, ...],
        attributes: tuple[Any, ...],
    ) -> bool:
        pending_transforms, pending_attributes = self.scheduler.pending_view_targets()
        retained_transforms = tuple(
            value for value in transforms if value.prim_path not in pending_transforms
        )
        retained_attributes = tuple(
            value
            for value in attributes
            if (value.prim_path, value.attribute) not in pending_attributes
        )
        if not self.ovrtx.activate(generation):
            return False
        results: list[RuntimeTickResult] = []
        if retained_transforms or retained_attributes:
            controller = self.ovrtx.controller
            results.append(
                controller.apply_runtime_updates(
                    lambda port, _project: self.scheduler.replay_retained_values(
                        port,
                        retained_transforms,
                        retained_attributes,
                    )
                )
            )
        if self.scheduler.has_pending_view_updates:
            result = self._apply_pending_view_values()
            results.append(result)
        failed = next(
            (result for result in results if result.status == RuntimeTickStatus.FAILED),
            None,
        )
        if failed is not None:
            self.last_activation_update = failed
            self.ovrtx.last_error = failed.skipped_reason or "retained_value_replay_failed"
            return False
        self.last_activation_update = RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
            values_written=any(result.values_written for result in results),
            should_reset_refinement=any(
                result.should_reset_refinement for result in results
            ),
        )
        return True

    def _apply_pending_view_values(self) -> RuntimeTickResult:
        return self.ovrtx.controller.apply_runtime_updates(
            lambda port, _project: self.scheduler.apply_pending_view_values(port)
        )

    def _finish_activation(
        self,
        generation: SceneGeneration,
        prior: tuple[RuntimeTickResult, ...] = (),
    ) -> bool:
        """Drain state committed before the update-ready linearization point."""

        results = [self.last_activation_update, *prior]
        while True:
            if getattr(self.scheduler, "has_pending_sim_updates", False):
                controller = self.ovphysx.controller
                if controller is None:
                    self.ovrtx.last_error = "SIM edit target has no active physics runtime"
                    return False
                result = self.scheduler.apply_pending_sim_values(controller)
                results.append(result)
                if result.status in {
                    RuntimeTickStatus.BUSY,
                    RuntimeTickStatus.FAILED,
                }:
                    self.last_activation_update = result
                    self.ovrtx.last_error = (
                        result.skipped_reason or "retained SIM value replay failed"
                    )
                    return False
            if self.scheduler.has_pending_view_updates:
                result = self._apply_pending_view_values()
                results.append(result)
                if result.status == RuntimeTickStatus.FAILED:
                    self.last_activation_update = result
                    self.ovrtx.last_error = (
                        result.skipped_reason or "retained value replay failed"
                    )
                    return False
            with self._admission_lock:
                if getattr(self.scheduler, "has_pending_sim_updates", False) or getattr(
                    self.scheduler, "has_pending_view_updates", False
                ):
                    continue
                # The admission lock is only held for this linearization point;
                # Blender callbacks never wait on native activation work.
                self._generation_number = generation.number
                self.preparation_status = "ready"
                self.preparation_error = ""
                self.last_activation_update = RuntimeTickResult(
                    status=RuntimeTickStatus.NOOP,
                    enabled=True,
                    values_written=any(result.values_written for result in results),
                    should_reset_refinement=any(
                        result.should_reset_refinement for result in results
                    ),
                    update={
                        "updates": [
                            dict(result.update) for result in results if result.update
                        ]
                    },
                )
                return True

    def close(self) -> bool:
        with self._admission_lock:
            if self.lifecycle_status == "closed":
                return True
            if self.lifecycle_status != "open" or self.reuse_blocked:
                return False
            self.lifecycle_status = "closing"
            adopt = getattr(self.ovrtx.controller, "adopt_owning_thread", None)
            if callable(adopt):
                adopt()
            scheduler_stopped = True
            try:
                self.scheduler.set_edit_wake_hook(None)
                with self._presentation_lock:
                    self._presentation_wakes.clear()
                self.scheduler.shutdown()
            except Exception:
                scheduler_stopped = False
            statuses = []
            for adapter in (self.ovrtx, self.ovphysx):
                try:
                    statuses.append(adapter.deactivate())
                except Exception:
                    statuses.append("failed")
            if not scheduler_stopped or any(
                status not in {"stopped", "not_found"} for status in statuses
            ):
                self.reuse_blocked = True
                return False
            self.lifecycle_status = "closed"
            return True

class PreparedFinalRenderLease:
    """Temporarily retarget scene-owned preparation for final presentation."""

    def __init__(
        self,
        runtime: AuthoringGenerationRuntime,
        generation: SceneGeneration,
        previous_request: RenderRequest | None,
    ) -> None:
        self._runtime = runtime
        self._generation = generation
        self._previous_request = previous_request

    @property
    def controller(self) -> Any:
        return self._runtime.ovrtx.controller

    @property
    def last_ensure_result(self) -> Any:
        return self._runtime.ovrtx.last_ensure_result

    def deactivate(self) -> str:
        if self._previous_request is None:
            try:
                status = self._runtime.ovrtx.deactivate()
            except Exception:
                status = "failed"
            if status in {"stopped", "not_found"}:
                return "stopped"
            fail_closed_runtime_reuse(self._runtime)
            return "failed"
        try:
            activate = getattr(
                self._runtime, "activate_blocking", self._runtime.activate
            )
            activate(self._generation, self._previous_request)
        except Exception:
            fail_closed_runtime_reuse(self._runtime)
            return "failed"
        return "stopped"

def mark_scene_dirty(
    scene: Any,
    affected_ids: set[BlenderId] | frozenset[BlenderId] | tuple[BlenderId, ...] = (),
    *,
    defer_world_reconciliation: bool = False,
) -> bool:
    """Queue precise Blender IDs for sparse reconciliation."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid <= 0:
        return False
    affected = set(affected_ids)
    if not affected:
        _rejected_unscoped_dirty_requests[uid] = (
            _rejected_unscoped_dirty_requests.get(uid, 0) + 1
        )
        return False
    if any(identity.kind == "WORLD" for identity in affected):
        _accepted_world_dirty_requests[uid] = (
            _accepted_world_dirty_requests.get(uid, 0) + 1
        )
    _dirty.setdefault(uid, set()).update(affected)
    _preparation_revisions[uid] = _preparation_revisions.get(uid, 0) + 1
    if defer_world_reconciliation:
        _schedule_world_reconciliation(scene, uid, affected)
    return True


def record_ignored_scene_update(scene: Any) -> None:
    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid > 0:
        _ignored_scene_updates[uid] = _ignored_scene_updates.get(uid, 0) + 1


def _schedule_world_reconciliation(
    scene: Any,
    uid: int,
    newly_affected: set[BlenderId],
) -> None:
    owner = _owners.get(uid)
    if (
        bpy is None
        or owner is None
        or owner.current_generation is None
        or uid in _reconciliation_timers
        or not any(identity.kind == "WORLD" for identity in _dirty[uid])
    ):
        return
    blocked = _blocked_reconciliations.get(uid)
    if blocked is not None:
        blocked_world = {
            identity for identity in blocked[1] if identity.kind == "WORLD"
        }
        current_world = {
            identity
            for identity in topology_identity_changes(scene)
            if identity.kind == "WORLD"
        }
        if (
            current_world == blocked_world
            and all(identity.kind == "WORLD" for identity in newly_affected)
        ):
            return
    _reconciliation_timers.add(uid)

    def reconcile() -> None:
        _reconciliation_timers.discard(uid)
        if _owners.get(uid) is owner:
            had_pending = getattr(owner, "pending_generation", None) is not None
            generation_for_scene(scene)
            if (
                not had_pending
                and getattr(owner, "pending_generation", None) is not None
            ):
                scene.update_tag()
        return None

    try:
        bpy.app.timers.register(reconcile, first_interval=0.0)
    except Exception:
        _reconciliation_timers.discard(uid)
        raise


def topology_identity_changed(scene: Any) -> bool:
    """Detect supported Blender IDs added to or removed from the generation."""

    return bool(topology_identity_changes(scene))


def topology_identity_changes(scene: Any) -> set[BlenderId]:
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generation = (
        None
        if owner is None
        else getattr(owner, "pending_generation", None) or owner.current_generation
    )
    if generation is None:
        return set()
    mapped = {
        identity
        for identity in generation.blender_prim_paths
        if identity.kind in {"OBJECT", "MATERIAL"}
    }
    world_uid = int(getattr(generation, "world_session_uid", 0) or 0)
    if world_uid <= 0:
        world_uid = next(
            (
                identity.session_uid
                for identity in generation.blender_prim_paths
                if identity.kind == "WORLD"
            ),
            0,
        )
    if world_uid > 0:
        mapped.add(BlenderId("WORLD", world_uid))
    current = _export_reachable_current_identities(scene, mapped)
    return current.symmetric_difference(mapped)


def _export_reachable_current_identities(
    scene: Any,
    mapped: set[BlenderId],
) -> set[BlenderId]:
    """Return identities that stock USD export can reach, or already mapped.

    Raw Blender references are not necessarily authored topology: exporters may
    omit dormant dependencies such as unused material slots. Keep each kind's
    reachability rule here so omitted data cannot permanently dirty the scene.
    New transitive dependencies arrive through their owning datablock's
    depsgraph update; identity comparison must never inspect per-element data.
    """
    result = set()
    world = getattr(scene, "world", None)
    if world is not None:
        result.add(blender_id(world, "WORLD"))
    for obj in getattr(scene, "objects", ()):
        if str(getattr(obj, "type", "")) not in {"CAMERA", "LIGHT", "MESH"}:
            continue
        identity = blender_id(obj, "OBJECT")
        if _is_editable_id(obj) or identity in mapped:
            result.add(identity)
        if str(getattr(obj, "type", "")) != "MESH":
            continue
        data = getattr(obj, "data", None)
        for material in getattr(data, "materials", ()):
            if material is None:
                continue
            identity = blender_id(material, "MATERIAL")
            if identity in mapped:
                result.add(identity)
    return result


def _is_editable_id(value: Any) -> bool:
    return (
        getattr(value, "library", None) is None
        or getattr(value, "override_library", None) is not None
    )


def affected_blender_ids(depsgraph: Any) -> set[BlenderId]:
    result = set()
    for update in getattr(depsgraph, "updates", ()):
        value = getattr(update, "id", None)
        value = getattr(value, "original", value)
        if not _is_editable_id(value):
            continue
        identifier = _canonical_blender_id_kind(
            str(getattr(getattr(value, "bl_rna", None), "identifier", ""))
        )
        if identifier == "OBJECT":
            kind = (
                "OBJECT"
                if str(getattr(value, "type", "")) in {"LIGHT", "MESH"}
                else None
            )
        elif identifier == "WORLD":
            kind = "WORLD"
        elif identifier == "LIGHT":
            kind = identifier
        else:
            kind = {
                "MESH": "MESH",
                "MATERIAL": "MATERIAL",
            }.get(identifier)
        if kind is None:
            continue
        uid = int(getattr(value, "session_uid", 0) or 0)
        if uid > 0:
            result.add(BlenderId(kind, uid))
    return result


def _canonical_blender_id_kind(identifier: str) -> str:
    identifier = str(identifier or "").upper()
    return "LIGHT" if identifier.endswith("LIGHT") else identifier


def generation_for_scene(
    scene: Any,
    *,
    work_root: str | Path | None = None,
) -> SceneGeneration:
    """Return the current private generation for one Blender render callback."""

    uid = _scene_uid(scene)
    owner = _owners.get(uid)
    if owner is None:
        scene_work_directory = _scene_work_directory(scene, work_root)
        scene_work_directory.mkdir(parents=True, exist_ok=True)
        owner = SceneGenerationOwner(
            Path(
                tempfile.mkdtemp(
                    prefix="scene-generations-",
                    dir=scene_work_directory,
                )
            ),
            scene_work_directory.parent / "reusable-bases",
        )
        _owners[uid] = owner
        _dirty.setdefault(uid, set())
    if owner.current_generation is None:
        if uid in _constructing:
            raise RuntimeError("scene generation construction is already in progress")
        _dirty.pop(uid, None)
        _constructing.add(uid)
        try:
            with interactive_operator_state.suppress_interactive_edit_bridge():
                generation = owner.replace(scene)
        finally:
            _constructing.discard(uid)
        if generation is not None:
            _mark_initial_generation_ready(uid, generation)
            return generation
    if getattr(owner, "pending_generation", None) is not None:
        return owner.reuse()
    if uid in _dirty:
        affected = _dirty.pop(uid)
        if uid in _blocked_reconciliations:
            identity_changes = topology_identity_changes(scene)
            affected = {
                identity for identity in affected if identity.kind != "WORLD"
            }
            affected.update(identity_changes)
            mapped_objects = {
                identity
                for identity in owner.current_generation.blender_prim_paths
                if identity.kind == "OBJECT"
            }
            current_objects = {
                blender_id(obj, "OBJECT")
                for obj in getattr(scene, "objects", ())
            }
            unreachable_additions = {
                identity
                for identity in affected
                if identity.kind == "OBJECT"
                and identity not in mapped_objects
                and identity not in current_objects
            }
            affected.difference_update(unreachable_additions)
            if not affected:
                _blocked_reconciliations.pop(uid, None)
                return owner.reuse()
        if not affected:
            raise RuntimeError("scene topology changed without affected Blender IDs")
        _reconciling.add(uid)
        try:
            with interactive_operator_state.suppress_interactive_edit_bridge():
                generation = owner.reconcile(scene, affected)
        except Exception as exc:
            retained = _dirty.setdefault(uid, set())
            retained.update(affected)
            _blocked_reconciliations[uid] = (
                f"{type(exc).__name__}: {exc}",
                set(retained),
            )
            raise
        finally:
            _reconciling.discard(uid)
        if generation is not None:
            _pending_affected[uid] = set(affected)
            blocked = _blocked_reconciliations.get(uid)
            if blocked is not None:
                _blocked_reconciliations[uid] = (blocked[0], set(affected))
            return generation
        _blocked_reconciliations.pop(uid, None)
    return owner.reuse()


def generation_for_viewport(scene: Any) -> SceneGeneration:
    """Return accepted/pending World state without exporting in a render callback."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid <= 0:
        return generation_for_scene(scene)
    owner = _owners.get(uid)
    if (
        owner is None
        or owner.current_generation is None
        or uid not in _dirty
    ):
        return generation_for_scene(scene)
    if not any(identity.kind == "WORLD" for identity in _dirty[uid]):
        return generation_for_scene(scene)
    _schedule_world_reconciliation(scene, uid, set(_dirty[uid]))
    return owner.reuse()


def retain_interactive_edit(
    scene: Any,
    edit: InteractiveEdit,
) -> EditWorkflowResult | None:
    """Observe a current-scene edit without requiring a viewport."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    if owner is None or owner.current_generation is None:
        return None
    plan = InteractiveEditPlanner().plan(edit)
    if plan.mechanism == EditMechanism.UPDATE:
        runtime = _runtime_for_owner(uid, owner)
        submission = runtime.scheduler.submit_edit(plan.to_intent())
        if not runtime.viewport_ids:
            _wake_preparation(uid)
        return EditWorkflowResult(
            action=WorkflowAction.UPDATE,
            status=submission.status,
            reason=submission.reason,
            plan=plan,
            submission_result=submission,
            diagnostics={"scheduler": dict(submission.diagnostics)},
        )
    if not plan.impact.scene_generation_replacement_requested:
        return None
    return EditWorkflowResult(
        action=WorkflowAction.COMPOSE,
        status=EditStatus.QUEUED,
        reason="scene_generation_dirty",
        plan=plan,
    )


def current_generation_edit_context(
    scene: Any,
) -> tuple[UsdPrimResolver | None, tuple[Any, ...]]:
    """Resolve untagged Blender values against the current scene generation."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generation = None if owner is None else owner.current_generation
    if generation is None:
        return None, ()
    resolver = UsdPrimResolver(
        object_paths_by_session_uid={
            identity.session_uid: mapping.object_path
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT"
        },
        light_paths_by_object_session_uid={
            identity.session_uid: mapping.schema_path
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT" and mapping.blender_id_type == "LIGHT"
        },
        mesh_topology_change_resolver=lambda mesh: _mesh_topology_change(
            scene, generation, mesh
        ),
    )
    resolver.scan(RenderRequest(input_usd_path=generation.usd_path))
    return resolver, tuple(getattr(scene, "objects", ()) or ())


def _mesh_topology_change(
    scene: Any, generation: SceneGeneration, mesh: Any
) -> Mapping[str, Any] | None:
    fingerprints = getattr(generation, "topology_fingerprints", {})
    data = getattr(mesh, "data", mesh)
    objects = (
        (mesh,)
        if getattr(mesh, "data", None) is not None
        and str(getattr(mesh, "type", "")) == "MESH"
        else tuple(
            obj
            for obj in getattr(scene, "objects", ()) or ()
            if getattr(obj, "data", None) is data
        )
    )
    for obj in objects:
        identity = blender_id(obj, "OBJECT")
        previous = fingerprints.get(identity)
        mapping = generation.blender_prim_paths.get(identity)
        if not previous or mapping is None:
            continue
        current = scene_generation._topology_fingerprints(scene, {identity}).get(identity)
        if not current or current == previous:
            continue
        return {
            "usd_prim_path": mapping.schema_path or mapping.object_path,
            "previous_fingerprint": previous,
            "current_fingerprint": current,
        }
    return None


def resolve_current_scene_edit_group(
    scene: Any,
    edits: Iterable[InteractiveEdit],
    selection_resolution: Mapping[str, Any],
) -> tuple[InteractiveEdit, ...]:
    """Map one callback's edits and enforce selection-driven completeness."""

    edits = tuple(edits)
    if not edits:
        return ()

    owner = _owners.get(_scene_uid(scene))
    generation = None if owner is None else owner.current_generation
    if generation is None:
        return ()

    selection_edits: list[tuple[InteractiveEdit, Mapping[str, Any]]] = []
    for edit in edits:
        mapping = _edit_source_mapping(generation, edit)
        if mapping is None or not _mapping_owns_target(mapping, edit.usd_prim_path):
            return ()
        selection = edit.provenance.get("selection_resolution")
        if isinstance(selection, Mapping) and _selection_matches_edit(edit, selection):
            selection_edits.append((edit, selection))
    if not selection_edits:
        return edits

    selected_count = int(selection_resolution.get("selected_object_count", 0))
    source_records = tuple(selection_resolution.get("sources", ()))
    if (
        selected_count <= 0
        or len(source_records) != selected_count
        or any(not isinstance(record, Mapping) for record in source_records)
    ):
        return ()
    source_uids = tuple(
        int(record.get("source_session_uid", 0) or 0)
        for record in source_records
    )
    if (
        any(uid <= 0 for uid in source_uids)
        or len(set(source_uids)) != selected_count
        or any(
            generation.blender_prim_paths.get(BlenderId("OBJECT", uid)) is None
            for uid in source_uids
        )
    ):
        return ()

    operations_by_source: dict[int, set[tuple[Any, ...]]] = {
        uid: set() for uid in source_uids
    }
    for edit, selection in selection_edits:
        uid = int(selection.get("source_session_uid", 0) or 0)
        if uid not in operations_by_source:
            return ()
        operations_by_source[uid].add(
            (edit.shape, edit.data_authority, edit.blender_property_path)
        )
    operations = tuple(frozenset(value) for value in operations_by_source.values())
    if any(not value for value in operations) or len(set(operations)) != 1:
        return ()
    return edits


def submit_current_scene_edit_group(
    scene: Any,
    edits: Iterable[InteractiveEdit],
    selection_resolution: Mapping[str, Any],
) -> tuple[EditWorkflowResult, ...]:
    """Resolve and commit one current-scene callback group exactly once."""

    admitted = resolve_current_scene_edit_group(scene, edits, selection_resolution)
    if not admitted:
        return ()
    uid = _scene_uid(scene)
    owner = _owners.get(uid)
    if owner is None:
        return ()
    runtime = _runtime_for_owner(uid, owner)
    results = runtime.submit_edit_group(admitted)
    if not runtime.viewport_ids and any(
        result.accepted for result in results
    ):
        _wake_preparation(uid)
    return results


def _edit_source_mapping(
    generation: SceneGeneration,
    edit: InteractiveEdit,
) -> Any | None:
    provenance = edit.provenance
    uid = int(provenance.get("blender_session_uid", 0) or 0)
    kind = _canonical_blender_id_kind(
        str(provenance.get("blender_id_kind", "") or "")
    )
    if uid <= 0 or not kind:
        return None
    mapping = generation.blender_prim_paths.get(BlenderId(kind, uid))
    if mapping is not None:
        return mapping
    candidates = tuple(
        candidate
        for identity, candidate in generation.blender_prim_paths.items()
        if identity.kind == "OBJECT"
        and candidate.blender_id_type == kind
        and candidate.data_session_uid == uid
    )
    if len(candidates) == 1:
        return candidates[0]
    matching_path = tuple(
        candidate
        for candidate in candidates
        if edit.usd_prim_path in {candidate.object_path, candidate.schema_path}
    )
    return matching_path[0] if len(matching_path) == 1 else None


def _selection_matches_edit(
    edit: InteractiveEdit,
    selection: Mapping[str, Any],
) -> bool:
    uid = int(edit.provenance.get("blender_session_uid", 0) or 0)
    return uid > 0 and uid in {
        int(selection.get(key, 0) or 0)
        for key in (
            "source_session_uid",
            "source_data_session_uid",
            "owner_session_uid",
            "owner_data_session_uid",
        )
    }


def _mapping_owns_target(mapping: Any, prim_path: str) -> bool:
    if prim_path in {mapping.object_path, mapping.schema_path}:
        return True
    return bool(
        mapping.blender_id_type in {"MATERIAL", "WORLD"}
        and prim_path.startswith(mapping.schema_path + "/")
    )


def _runtime_for_owner(
    uid: int,
    owner: SceneGenerationOwner,
    *,
    expected_runtime: AuthoringGenerationRuntime | None = None,
) -> AuthoringGenerationRuntime:
    """Atomically return the one runtime owned by a Blender scene."""

    with _preparation_condition:
        runtime = _runtimes.get(uid)
        _require_runtime_reuse(runtime)
        if runtime is None:
            if expected_runtime is not None:
                raise RuntimeError(
                    "viewport attachment attempted to replace the authoring runtime"
                )
            from .ovrtx_session_controller import OvrtxSessionController

            runtime = AuthoringGenerationRuntime(OvrtxSessionController(), owner)
            _runtimes[uid] = runtime
        elif expected_runtime is not None and runtime is not expected_runtime:
            raise RuntimeError(
                "viewport attachment attempted to replace the authoring runtime"
            )
        return runtime


def runtime_for_viewport(
    scene: Any,
    *,
    viewport_id: str,
) -> AuthoringGenerationRuntime:
    """Return the scene-owned runtime before an authored pane starts."""

    uid = _scene_uid(scene)
    owner = _owners.get(uid)
    if owner is None or (
        owner.current_generation is None
        and getattr(owner, "pending_generation", None) is None
    ):
        raise RuntimeError("current scene generation is unavailable for viewport startup")
    runtime = _runtime_for_owner(uid, owner)
    runtime.attach(viewport_id)
    return runtime


def activate_for_viewport(
    scene: Any,
    request: RenderRequest,
    *,
    viewport_id: str,
    wake_hook: Callable[[], None] | None = None,
    on_generation_settled: Callable[[], None] | None = None,
    expected_runtime: AuthoringGenerationRuntime | None = None,
) -> AuthoringGenerationRuntime:
    uid = _scene_uid(scene)
    owner = _owners.get(uid)
    current = None if owner is None else owner.current_generation
    pending = None if owner is None else getattr(owner, "pending_generation", None)
    generation = next(
        (
            item
            for item in (pending, current)
            if item is not None and item.usd_path == request.input_usd_path
        ),
        None,
    )
    if generation is None or generation.usd_path != request.input_usd_path:
        raise RuntimeError("current scene generation is unavailable for viewport activation")
    runtime = _runtime_for_owner(
        uid,
        owner,
        expected_runtime=expected_runtime,
    )
    runtime.attach(viewport_id, wake_hook)
    runtime_controller = runtime.ovrtx.controller
    if len(runtime.viewport_ids) > 1:
        allow_shared = getattr(runtime_controller, "_allow_serialized_threads", None)
        if callable(allow_shared):
            allow_shared()
    else:
        adopt = getattr(runtime_controller, "adopt_owning_thread", None)
        if callable(adopt):
            adopt()
    try:
        runtime.activate_blocking(
            generation,
            request,
            predecessor=current if pending is generation else None,
        )
    except Exception as exc:
        if pending is generation:
            affected = _pending_affected.pop(uid, set())
            blocked = _blocked_reconciliations.get(uid)
            if blocked is not None:
                affected.update(blocked[1])
            affected.update(_dirty.get(uid, ()))
            if affected:
                _dirty.setdefault(uid, set()).update(affected)
                _blocked_reconciliations[uid] = (
                    f"{type(exc).__name__}: {exc}",
                    set(affected),
                )
            owner.reject(generation)
            if on_generation_settled is not None and any(
                identity.kind == "WORLD" for identity in _dirty.get(uid, ())
            ):
                on_generation_settled()
        raise
    if pending is generation:
        owner.accept(generation)
        _pending_affected.pop(uid, None)
        _blocked_reconciliations.pop(uid, None)
        if on_generation_settled is not None and any(
            identity.kind == "WORLD" for identity in _dirty.get(uid, ())
        ):
            on_generation_settled()
    return runtime


def activate_for_final_render(
    scene: Any,
    request: RenderRequest,
    *,
    controller: Any,
) -> OvrtxGenerationAdapter | PreparedFinalRenderLease:
    uid = _scene_uid(scene)
    owner = _owners.get(uid)
    if owner is None:
        raise RuntimeError("current scene generation owner is unavailable")
    generation, predecessor = _generation_for_request(owner, request)
    prepared = _runtimes.get(uid)
    _require_runtime_reuse(prepared)
    if prepared is not None:
        previous_request = prepared._request
        try:
            activate = getattr(prepared, "activate_blocking", prepared.activate)
            activate(
                generation,
                request,
                predecessor=predecessor,
            )
        except Exception:
            if predecessor is not None:
                owner.reject(generation)
                _pending_affected.pop(uid, None)
                _blocked_reconciliations.pop(uid, None)
            raise
        if predecessor is not None:
            owner.accept(generation)
            _pending_affected.pop(uid, None)
            _blocked_reconciliations.pop(uid, None)
        return PreparedFinalRenderLease(prepared, generation, previous_request)
    adapter = OvrtxGenerationAdapter(controller)
    adapter.update_request(request)
    try:
        transforms, attributes, _initial_conditions = owner.retained_values_for(generation)
        if not adapter.activate(
            generation,
            transform_values=transforms,
            attribute_values=attributes,
        ):
            raise RuntimeError(
                adapter.last_error or "final scene generation activation failed"
            )
    except Exception:
        if predecessor is not None:
            owner.reject(generation)
            _pending_affected.pop(uid, None)
            _blocked_reconciliations.pop(uid, None)
        adapter.deactivate()
        raise
    if predecessor is not None:
        owner.accept(generation)
        _pending_affected.pop(uid, None)
        _blocked_reconciliations.pop(uid, None)
    return adapter


def _generation_for_request(
    owner: SceneGenerationOwner,
    request: RenderRequest,
) -> tuple[SceneGeneration, SceneGeneration | None]:
    current = owner.current_generation
    pending = owner.pending_generation
    generation = next(
        (
            item
            for item in (pending, current)
            if item is not None and item.usd_path == request.input_usd_path
        ),
        None,
    )
    if generation is None:
        raise RuntimeError("current scene generation is unavailable for activation")
    return generation, current if pending is generation else None


def owns_request(scene: Any, request: RenderRequest) -> bool:
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generations = tuple(
        generation
        for generation in (
            None if owner is None else getattr(owner, "pending_generation", None),
            None if owner is None else owner.current_generation,
        )
        if generation is not None
    )
    return any(
        generation is not None and generation.usd_path == request.input_usd_path
        for generation in generations
    )


def detach_viewport(
    viewport_id: str,
    *,
    runtime: AuthoringGenerationRuntime | None = None,
) -> bool:
    candidates = tuple(_runtimes.items())
    if runtime is not None:
        candidates = tuple(
            (uid, candidate)
            for uid, candidate in candidates
            if candidate is runtime
        )
    for uid, candidate in candidates:
        candidate.detach(viewport_id)
        if not candidate.viewport_ids:
            _wake_preparation(uid)
    return True


def diagnostics_for_scene(
    scene: Any,
    *,
    input_usd_path: str = "",
) -> dict[str, Any]:
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generation = None if owner is None else owner.current_generation
    pending = None if owner is None else getattr(owner, "pending_generation", None)
    if input_usd_path:
        generation = next(
            (
                candidate
                for candidate in (pending, generation)
                if candidate is not None
                and candidate.usd_path == input_usd_path
            ),
            None,
        )
    if generation is None:
        if input_usd_path or pending is None:
            return {
                "status": "unavailable",
                "scene_uid": uid,
                "initial_generation": dict(_initial_generation_diagnostics),
            }
        generation = pending
    result = {
        "status": "pending" if pending is generation else "current",
        "scene_uid": uid,
        "number": generation.number,
        "digest": generation.digest,
        "predecessor_number": generation.predecessor_number,
        "usd_path": generation.usd_path,
        "generation": dict(getattr(generation, "diagnostics", {})),
        "failed_generation_numbers": [
            item.number for item in getattr(owner, "failed_generations", ())
        ],
        "initial_generation": dict(_initial_generation_diagnostics),
    }
    blocked = _blocked_reconciliation_diagnostics(uid)
    if blocked is not None:
        result["blocked_reconciliation"] = blocked
    return result


def active_runtime_for_scene(scene: Any) -> AuthoringGenerationRuntime | None:
    """Return the scene-owned runtime for internal Blender integration harnesses."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    return _runtimes.get(uid)


def deactivate_all_ovrtx() -> bool:
    """Stop every retained OVRTX session before replacing the shared worker."""

    confirmed = True
    for runtime in tuple(_runtimes.values()):
        controller = runtime.ovrtx.controller
        adopt = getattr(controller, "adopt_owning_thread", None)
        try:
            if callable(adopt):
                adopt()
            status = runtime.ovrtx.deactivate()
        except Exception:
            status = "failed"
        if status not in {"stopped", "not_found"}:
            fail_closed_runtime_reuse(runtime)
            confirmed = False
            continue
    return confirmed


def pause_preparation() -> bool:
    """Drain and suppress hidden preparation during shared worker replacement."""

    global _preparation_paused
    with _preparation_condition:
        _preparation_paused = True
        _preparation_condition.notify_all()
    deadline = time.monotonic() + 5.0
    for runtime in tuple(_runtimes.values()):
        if not runtime._preparation_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            fail_closed_runtime_reuse(runtime)
            return False
        runtime._preparation_lock.release()
    return True


def resume_preparation() -> None:
    global _preparation_paused
    with _preparation_condition:
        _preparation_paused = False
        if _preparation_pending:
            _start_preparation_worker_locked()
            _preparation_condition.notify_all()


def demand_physics_playback(scene: Any) -> None:
    """Reconcile a barrier on Blender's thread, then prepare physics off-thread."""

    uid = _scene_uid(scene)
    _require_runtime_reuse(_runtimes.get(uid))
    generation_for_scene(scene)
    _playback_errors.pop(uid, None)
    owner = _owners[uid]
    if uid not in _runtimes:
        _runtime_for_owner(uid, owner)
    _wake_preparation(uid)


def diagnostics() -> dict[str, Any]:
    return {
        "scene_uids": sorted(_owners),
        "dirty_scene_uids": sorted(_dirty),
        "dirty_blender_ids": {
            str(uid): [
                {"kind": identity.kind, "session_uid": identity.session_uid}
                for identity in sorted(identities)
            ]
            for uid, identities in sorted(_dirty.items())
        },
        "runtime_scene_uids": sorted(_runtimes),
        "preparation": {
            str(uid): {
                "status": getattr(runtime, "preparation_status", "unavailable"),
                "error": getattr(runtime, "preparation_error", ""),
                "reuse_blocked": getattr(runtime, "reuse_blocked", False),
                "presentations": sorted(getattr(runtime, "viewport_ids", ())),
                "revision": _preparation_revisions.get(uid, 0),
            }
            for uid, runtime in sorted(_runtimes.items())
        },
        "physics_playback_errors": dict(sorted(_playback_errors.items())),
        "reconciling_scene_uids": sorted(_reconciling),
        "blocked_reconciliations": {
            str(uid): _blocked_reconciliation_diagnostics(uid)
            for uid in sorted(_blocked_reconciliations)
        },
        "rejected_unscoped_dirty_requests": {
            str(uid): count
            for uid, count in sorted(_rejected_unscoped_dirty_requests.items())
        },
        "accepted_world_dirty_requests": {
            str(uid): count
            for uid, count in sorted(_accepted_world_dirty_requests.items())
        },
        "ignored_scene_updates": {
            str(uid): count for uid, count in sorted(_ignored_scene_updates.items())
        },
        "reconciliation_timer_scene_uids": sorted(_reconciliation_timers),
        "constructing_scene_uids": sorted(_constructing),
        "initial_generation": dict(_initial_generation_diagnostics),
        "load_pre_completed": _load_pre_completed,
        "load_post_count": _load_post_count,
    }


def _blocked_reconciliation_diagnostics(uid: int) -> dict[str, Any] | None:
    blocked = _blocked_reconciliations.get(uid)
    if blocked is None:
        return None
    error, affected = blocked
    return {
        "error": error,
        "affected_ids": [
            {"kind": identity.kind, "session_uid": identity.session_uid}
            for identity in sorted(affected)
        ],
    }


def close() -> None:
    _invalidate_initial_generation()
    for runtime in tuple(_runtimes.values()):
        try:
            runtime.playback_lock.clear(reason="authoring_runtime_shutdown")
        except Exception:
            pass
    closed_uids = _stop_preparation_worker(close_runtimes=True)
    for uid in closed_uids:
        del _runtimes[uid]
    for uid, owner in tuple(_owners.items()):
        if uid not in _runtimes:
            owner.close()
            del _owners[uid]
    _dirty.clear()
    _reconciling.clear()
    _blocked_reconciliations.clear()
    _pending_affected.clear()
    _constructing.clear()
    _preparation_pending.clear()
    _preparation_revisions.clear()
    _playback_errors.clear()
    _rejected_unscoped_dirty_requests.clear()
    _accepted_world_dirty_requests.clear()
    _ignored_scene_updates.clear()
    _reconciliation_timers.clear()


def is_reconciling(scene: Any) -> bool:
    return int(getattr(scene, "session_uid", 0) or 0) in _reconciling


def is_authoring(scene: Any) -> bool:
    uid = int(getattr(scene, "session_uid", 0) or 0)
    return uid in _constructing or uid in _reconciling


@persistent
def load_pre(_unused: Any = None) -> None:
    global _load_pre_completed
    if _load_pre_completed:
        return
    from . import engine

    engine.stop_viewport_render_threads_for_file_load()
    close()
    _load_pre_completed = True


@persistent
def load_post(_unused: Any = None) -> None:
    global _load_post_count, _load_pre_completed
    _load_post_count += 1
    _load_pre_completed = False
    from . import start_runtime_services_async

    schedule_initial_generation()
    start_runtime_services_async()


def schedule_initial_generation(blender: Any | None = None) -> bool:
    """Schedule current-scene generation on Blender's next main-thread turn."""

    global _initial_generation_revision, _initial_generation_diagnostics
    module = blender or bpy
    scene = getattr(getattr(module, "context", None), "scene", None)
    try:
        uid = _scene_uid(scene)
    except (TypeError, ValueError) as exc:
        _initial_generation_diagnostics = {
            "status": "failed",
            "scene_uid": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "last_error": f"{type(exc).__name__}: {exc}",
        }
        return False
    if (
        _initial_generation_diagnostics.get("scene_uid") == uid
        and _initial_generation_diagnostics.get("status")
        in {"scheduled", "constructing", "ready"}
    ):
        return False
    timers = getattr(getattr(module, "app", None), "timers", None)
    if timers is None:
        _initial_generation_diagnostics = {
            "status": "failed",
            "scene_uid": uid,
            "error": "Blender main-thread timers are unavailable",
            "last_error": "Blender main-thread timers are unavailable",
        }
        return False
    _initial_generation_revision += 1
    revision = _initial_generation_revision
    _initial_generation_diagnostics = {
        "status": "scheduled",
        "scene_uid": uid,
        "error": "",
        "last_error": (
            str(_initial_generation_diagnostics.get("last_error", ""))
            if _initial_generation_diagnostics.get("scene_uid") == uid
            else ""
        ),
    }

    def generate() -> None:
        global _initial_generation_diagnostics
        if revision != _initial_generation_revision:
            return None
        context = getattr(module, "context", None)
        current_scene = getattr(context, "scene", None)
        try:
            current_uid = _scene_uid(current_scene)
        except (TypeError, ValueError) as exc:
            _initial_generation_diagnostics = {
                "status": "failed",
                "scene_uid": uid,
                "error": f"{type(exc).__name__}: {exc}",
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            return None
        if current_uid != uid:
            return None
        _initial_generation_diagnostics = {
            "status": "constructing",
            "scene_uid": uid,
            "error": "",
            "last_error": str(_initial_generation_diagnostics.get("last_error", "")),
        }
        cursor_set = getattr(getattr(context, "window", None), "cursor_set", None)
        try:
            if callable(cursor_set):
                cursor_set("WAIT")
            generation = generation_for_scene(current_scene)
        except Exception as exc:
            _initial_generation_diagnostics = {
                "status": "failed",
                "scene_uid": uid,
                "error": f"{type(exc).__name__}: {exc}",
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            traceback.print_exc()
            return None
        finally:
            if callable(cursor_set):
                cursor_set("DEFAULT")
        _mark_initial_generation_ready(uid, generation)
        return None

    try:
        timers.register(generate, first_interval=0.0)
    except Exception as exc:
        _initial_generation_diagnostics = {
            "status": "failed",
            "scene_uid": uid,
            "error": f"{type(exc).__name__}: {exc}",
            "last_error": f"{type(exc).__name__}: {exc}",
        }
        return False
    return True


def _mark_initial_generation_ready(uid: int, generation: Any) -> None:
    global _initial_generation_diagnostics
    if (
        _initial_generation_diagnostics.get("scene_uid") != uid
        or _initial_generation_diagnostics.get("status")
        not in {"constructing", "failed"}
    ):
        return
    _initial_generation_diagnostics = {
        "status": "ready",
        "scene_uid": uid,
        "error": "",
        "last_error": str(
            _initial_generation_diagnostics.get("last_error")
            or _initial_generation_diagnostics.get("error", "")
        ),
        "number": int(getattr(generation, "number", 0)),
        "digest": str(getattr(generation, "digest", "")),
    }


def _invalidate_initial_generation() -> None:
    global _initial_generation_revision, _initial_generation_diagnostics
    _initial_generation_revision += 1
    _initial_generation_diagnostics = {
        "status": "unavailable",
        "scene_uid": 0,
        "error": "",
        "last_error": "",
    }


@persistent
def animation_playback_pre(scene: Any, *_unused: Any) -> None:
    try:
        demand_physics_playback(scene)
    except Exception as exc:
        uid = int(getattr(scene, "session_uid", 0) or 0)
        if uid > 0:
            _playback_errors[uid] = f"{type(exc).__name__}: {exc}"


def register_handlers(blender: Any | None = None) -> None:
    module = blender or bpy
    if module is None:
        raise RuntimeError("scene generation handlers require Blender")
    if load_pre not in module.app.handlers.load_pre:
        module.app.handlers.load_pre.append(load_pre)
    if load_post not in module.app.handlers.load_post:
        module.app.handlers.load_post.append(load_post)
    playback_handlers = getattr(module.app.handlers, "animation_playback_pre", None)
    if playback_handlers is not None and animation_playback_pre not in playback_handlers:
        playback_handlers.append(animation_playback_pre)
    _start_preparation_worker()


def unregister_handlers(blender: Any | None = None) -> None:
    module = blender or bpy
    if module is None:
        raise RuntimeError("scene generation handlers require Blender")
    for handlers, function in (
        (module.app.handlers.load_pre, load_pre),
        (module.app.handlers.load_post, load_post),
    ):
        if function in handlers:
            handlers.remove(function)
    playback_handlers = getattr(module.app.handlers, "animation_playback_pre", None)
    if playback_handlers is not None and animation_playback_pre in playback_handlers:
        playback_handlers.remove(animation_playback_pre)
    close()


def _wake_preparation(uid: int) -> None:
    global _preparation_thread
    with _preparation_condition:
        runtime = _runtimes.get(uid)
        if runtime is not None and getattr(runtime, "reuse_blocked", False):
            return
        revision = _preparation_revisions.get(uid, 0) + 1
        _preparation_revisions[uid] = revision
        _preparation_pending[uid] = revision
        _start_preparation_worker_locked()
        _preparation_condition.notify()


def _start_preparation_worker() -> None:
    with _preparation_condition:
        _start_preparation_worker_locked()


def _start_preparation_worker_locked() -> None:
    global _preparation_thread, _preparation_stopping
    if _preparation_paused:
        return
    if _preparation_thread is not None and _preparation_thread.is_alive():
        return
    _preparation_stopping = False
    _preparation_thread = threading.Thread(
        target=_preparation_loop,
        name="ovrtx-authoring-preparation",
        daemon=True,
    )
    _preparation_thread.start()


def _stop_preparation_worker(*, close_runtimes: bool = False) -> set[int]:
    global _preparation_thread, _preparation_stopping
    with _preparation_condition:
        if close_runtimes:
            _start_preparation_worker_locked()
        thread = _preparation_thread
        _preparation_stopping = True
        _preparation_condition.notify_all()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5.0)
    if thread is not None and thread.is_alive():
        for runtime in _runtimes.values():
            runtime.reuse_blocked = True
        return set()
    closed = (
        {
            uid
            for uid, runtime in _runtimes.items()
            if runtime.lifecycle_status == "closed"
        }
        if close_runtimes
        else set()
    )
    with _preparation_condition:
        _preparation_thread = None
        _preparation_stopping = False
    return closed


def _preparation_loop() -> None:
    while True:
        with _preparation_condition:
            _preparation_condition.wait_for(
                lambda: _preparation_stopping
                or (not _preparation_paused and bool(_preparation_pending))
            )
            if _preparation_stopping:
                runtimes = tuple(_runtimes.items())
                break
            uid, revision = next(iter(_preparation_pending.items()))
            del _preparation_pending[uid]
        runtime = _runtimes.get(uid)
        owner = _owners.get(uid)
        generation = None if owner is None else owner.current_generation
        if (
            runtime is None
            or generation is None
            or runtime.viewport_ids
            or uid in _dirty
            or getattr(owner, "pending_generation", None) is not None
        ):
            continue
        controller = runtime.ovrtx.controller
        adopt = getattr(controller, "adopt_owning_thread", None)
        if callable(adopt):
            adopt()
        runtime.preparation_status = "preparing"
        runtime.last_activation_update = RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
        )
        try:
            if (
                generation_requires_physics(generation)
                and runtime.ovphysx.active_generation != generation.number
                and not runtime.ovphysx.activate(
                    generation,
                    initial_conditions=owner.retained_values_for(generation)[2],
                )
            ):
                raise RuntimeError(
                    runtime.ovphysx.last_error or "OVPhysX preparation failed"
                )
            with runtime._preparation_lock:
                with _preparation_condition:
                    if _preparation_paused:
                        _preparation_pending[uid] = max(
                            revision, _preparation_pending.get(uid, 0)
                        )
                        continue
                runtime.replay_retained_values(generation)
        except Exception as exc:
            runtime.preparation_status = "failed"
            runtime.preparation_error = f"{type(exc).__name__}: {exc}"
            if (
                runtime.last_activation_update.status == RuntimeTickStatus.BUSY
                and getattr(runtime.scheduler, "has_pending_sim_updates", False)
            ):
                # BUSY application restores only the latest desired values.
                # Retry them after a bounded wait; no callback event is replayed.
                with _preparation_condition:
                    stopping = _preparation_condition.wait_for(
                        lambda: _preparation_stopping,
                        timeout=PREPARATION_RETRY_BACKOFF_SECONDS,
                    )
                    if not stopping:
                        _preparation_pending[uid] = _preparation_revisions.get(
                            uid, revision
                        )
                        _preparation_condition.notify()
            continue
        runtime.preparation_status = "ready"
        runtime.preparation_error = ""
        with _preparation_condition:
            latest = _preparation_revisions.get(uid, revision)
            if latest != revision:
                _preparation_pending[uid] = latest
                _preparation_condition.notify()
    for _uid, runtime in runtimes:
        if getattr(runtime, "reuse_blocked", False):
            continue
        try:
            if not runtime.close():
                runtime.reuse_blocked = True
        except Exception:
            runtime.reuse_blocked = True


def _scene_uid(scene: Any) -> int:
    value = int(getattr(scene, "session_uid", 0) or 0)
    if value <= 0:
        raise ValueError("Blender scene session_uid is unavailable")
    return value


def _scene_work_directory(scene: Any, work_root: str | Path | None) -> Path:
    root = Path(
        work_root
        or os.environ.get("OV_BLENDER_EXAMPLE_SCENE_GENERATION_DIR")
        or Path(tempfile.gettempdir()) / "ov-blender-example" / "scene-generations"
    ).expanduser().resolve()
    return root / f"scene-{_scene_uid(scene)}"


__all__ = [
    "close",
    "affected_blender_ids",
    "activate_for_final_render",
    "activate_for_viewport",
    "active_runtime_for_scene",
    "deactivate_all_ovrtx",
    "detach_viewport",
    "diagnostics",
    "diagnostics_for_scene",
    "generation_for_scene",
    "is_authoring",
    "is_reconciling",
    "load_post",
    "load_pre",
    "mark_scene_dirty",
    "owns_request",
    "topology_identity_changed",
    "topology_identity_changes",
    "register_handlers",
    "runtime_for_viewport",
    "pause_preparation",
    "resume_preparation",
    "submit_current_scene_edit_group",
    "schedule_initial_generation",
    "unregister_handlers",
]
