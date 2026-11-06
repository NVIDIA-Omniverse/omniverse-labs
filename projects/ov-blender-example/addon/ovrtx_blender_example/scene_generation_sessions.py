# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender-scene generation owners and file-lifetime handlers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from .generation_runtime_adapters import (
    OvphysxGenerationAdapter,
    OvrtxGenerationAdapter,
    generation_requires_physics,
)
from .render_requests import RenderRequest
from .runtime_scheduler import RuntimeScheduler, RuntimeTickResult, RuntimeTickStatus
from .interactive_edit_planner import (
    DataAuthority,
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
from . import performance_profile
from . import playback_hold
from . import scene_generation
from .scene_generation import BlenderId, SceneGeneration, SceneGenerationOwner, SceneUpdateProvenance, blender_id
from .usd_prim_resolver import UsdPrimResolver
from .ovrtx_runtime_client import RuntimeServicesPreparingError

try:
    import bpy  # type: ignore
    from bpy.app.handlers import persistent  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]

    def persistent(function: Any) -> Any:
        return function


_owners: dict[int, SceneGenerationOwner] = {}
_runtimes: dict[int, "AuthoringGenerationRuntime"] = {}
_camera_capture_owners: dict[int, Any] = {}
_dirty: dict[int, set[BlenderId]] = {}
_dirty_provenance: dict[int, dict[BlenderId, SceneUpdateProvenance]] = {}
_reconciling: set[int] = set()
_blocked_reconciliations: dict[int, tuple[str, set[BlenderId]]] = {}
_pending_affected: dict[int, set[BlenderId]] = {}
_constructing: set[int] = set()
_visibility_observers: dict[int, Any] = {}
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


def _record_playback_error(uid: int, error: str) -> None:
    """Keep the accepted simulation, but tell the user this edit did not apply."""
    _playback_errors[uid] = error
    from . import user_messages
    user_messages.report_error(
        f"Physics not updated: {error}; the previous simulation is still playing",
        context=f"physics-playback:{uid}",
    )


def _clear_playback_error(uid: int) -> None:
    if _playback_errors.pop(uid, None) is None:
        return
    from . import user_messages
    # Same context, so a later recurrence of the old error is reported again.
    user_messages.report_info("Physics updated; the previous physics error is resolved",
                              context=f"physics-playback:{uid}")


def physics_playback_error(scene: Any) -> str:
    """The unresolved playback error for a scene's viewport status, if any."""
    return _playback_errors.get(int(getattr(scene, "session_uid", 0) or 0), "")
_rejected_unscoped_dirty_requests: dict[int, int] = {}
_accepted_world_dirty_requests: dict[int, int] = {}
_ignored_scene_updates: dict[int, int] = {}
_reconciliation_timers: set[int] = set()
PREPARATION_RETRY_BACKOFF_SECONDS = 0.5
_bake_capture_owner: Any = None
_baked_playback_paused: set[int] = set()
_baked_drain_requests: dict[int, Any] = {}


def _baked_active(scene: Any) -> bool:
    return bool(getattr(getattr(scene, "ov_physics_bake", None), "active", False))


def set_baked_playback(scene: Any, active: bool):
    """Gate immediately; return a future for owner-thread producer/native drain.

    The caller sets its Scene active property, polls this future from its modal
    loop, then calls finish_baked_playback_pause before assigning any Actions.
    Deactivation deliberately leaves physics paused until explicit demand.
    """
    from concurrent.futures import Future
    global _presentation_epoch
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Baked playback admission requires the main thread")
    if _bake_capture_owner is not None:
        raise RuntimeError("Finish or cancel capture before activating baked playback")
    if _preparation_paused:
        raise RuntimeError("Runtime services are paused; baked ownership cannot be drained yet")
    uid = _scene_uid(scene)
    _baked_playback_paused.add(uid)
    _presentation_stopped.add(uid)
    _presentation_epoch += 1
    runtime = _runtimes.get(uid)
    future = Future()
    future._ov_baked_scene_uid = uid
    if runtime is None:
        future.set_result(None)
        return future
    runtime.scheduler.baked_playback_paused = True
    with _preparation_condition:
        previous = _baked_drain_requests.get(uid)
        if previous is not None and not previous.done():
            return previous
        _baked_drain_requests[uid] = future
        _start_preparation_worker_locked()
        _preparation_condition.notify_all()
    runtime._wake_presentations()
    return future


def finish_baked_playback_pause(scene: Any, future: Any) -> None:
    """Main-thread final admission; never wait while blocking Blender evaluation."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Baked playback finalization requires the main thread")
    if not future.done():
        raise RuntimeError("Physics ownership is still draining")
    if getattr(future, "_ov_baked_scene_uid", None) != _scene_uid(scene):
        raise RuntimeError("Baked-playback drain belongs to another scene")
    future.result()
    restore_physics_presentation(scene)
    # The successful owner-thread drain precedes this writable handoff and all
    # baked Action assignment. Old GPU/timer packets no longer own this scene.
    from . import engine
    engine.retire_physics_presentation(scene)


def acquire_bake_capture(owner: Any) -> tuple:
    """Main-thread admission gate; the capture worker drains existing locks."""
    global _bake_capture_owner, _presentation_epoch
    if _bake_capture_owner is not None:
        raise RuntimeError("a physics bake capture is already active")
    _bake_capture_owner = owner
    _presentation_epoch += 1
    runtimes = tuple(_runtimes.values())
    for runtime in runtimes:
        runtime.scheduler.capture_active = True
    return runtimes


def release_bake_capture(owner: Any) -> None:
    global _bake_capture_owner, _presentation_epoch
    if _bake_capture_owner is owner:
        _bake_capture_owner = None
        _presentation_epoch += 1
        for runtime in _runtimes.values():
            runtime.scheduler.capture_active = False


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
    _require_session_open()
    if runtime is not None and getattr(runtime, "reuse_blocked", False):
        raise RuntimeError(
            "viewport runtime reuse is disabled after a teardown deadline was exceeded"
        )


@dataclass(frozen=True)
class _MaterialEditIdentity:
    session_uid: int


class AuthoringGenerationRuntime:
    """Own one Blender scene's scheduler and prepared OVRTX/OVPhysX state."""

    def __init__(
        self,
        controller: Any,
        owner: SceneGenerationOwner,
    ) -> None:
        self.viewport_ids: set[str] = set()
        self.owner = owner
        self.scene_uid = 0
        self.ovrtx = OvrtxGenerationAdapter(controller)
        self.ovphysx = OvphysxGenerationAdapter()
        self._native_view_generation: SceneGeneration | None = None
        self.scheduler = RuntimeScheduler(
            controller_provider=lambda: self.ovphysx.controller,
            controller_reset=self.ovphysx.reset,
            kinematic_source_provider=self._kinematic_source,
            ovrtx_transform_sink=owner.retain_transform_values,
            ovrtx_attribute_sink=owner.retain_attribute_values,
            ovrtx_view_target_resolver=self._resolve_native_view_target,
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

    def _kinematic_source(self):
        from .physics_kinematic_inputs import kinematic_targets
        generation = getattr(self.ovphysx, "_active_scene_generation", None)
        if generation is None or not kinematic_targets(generation):
            return None
        from .physics_live_kinematics import LiveKinematicSource
        return LiveKinematicSource(self.scene_uid, generation)

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
            if _bake_capture_owner is not None:
                return tuple(EditWorkflowResult(action=WorkflowAction.UNSUPPORTED,
                    status=EditStatus.FAILED, reason="physics_bake_capture_active") for _ in edits)
            if self.lifecycle_status != "open":
                return tuple(
                    EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.FAILED,
                        reason="authoring_runtime_closing",
                    )
                    for _edit in edits
                )
            if getattr(self.scheduler, "baked_playback_paused", False):
                from dataclasses import replace
                edits = tuple(replace(edit, data_authority=DataAuthority.VIEW)
                    if edit.data_authority == DataAuthority.SIM and edit.blender_property_path == "matrix_world"
                    else edit for edit in edits)
            # Kinematic poses are sampled from RNA before each native step.
            # A legacy SIM tag on a render transform must not enqueue a dynamic
            # initial-condition reset through the generic edit transport.
            generation = getattr(self.ovphysx, "_active_scene_generation", None)
            if generation is not None and any(edit.data_authority == DataAuthority.SIM for edit in edits):
                from dataclasses import replace
                from .physics_kinematic_inputs import kinematic_targets
                targets = kinematic_targets(generation)
                paths = {target.body_path for target in targets}
                object_uids = {target.object_session_uid for target in targets if not target.bone_identity}
                paths.update(mapping.object_path for identity, mapping in generation.blender_prim_paths.items()
                             if identity.kind == "OBJECT" and identity.session_uid in object_uids)
                edits = tuple(replace(edit, data_authority=DataAuthority.VIEW)
                    if edit.data_authority == DataAuthority.SIM and edit.usd_prim_path in paths
                    and edit.blender_property_path == "matrix_world" else edit for edit in edits)
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
        # Viewport startup already owns the controller transport when it
        # enters activate_blocking. Keep the same order here so preparation
        # cannot hold this lock while waiting for that transport.
        with self.ovrtx.controller._serialized_transport(), self._preparation_lock:
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
        if _bake_capture_owner is not None:
            raise RuntimeError("physics bake capture owns the runtime")
        with self.ovrtx.controller._serialized_transport(), self._preparation_lock:
            if _bake_capture_owner is not None:
                raise RuntimeError("physics bake capture owns the runtime")
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
            if replacing and not getattr(self.scheduler, "baked_playback_paused", False):
                if generation_requires_physics(generation):
                    if self.ovphysx.active_generation != generation.number:
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
            if getattr(self.scheduler, "has_pending_sim_updates", False) and not getattr(self.scheduler, "baked_playback_paused", False):
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
                previous_composition = getattr(self.ovrtx, 'composition', None)
                ready = self.ovrtx.ensure_request()
                composition = getattr(self.ovrtx, 'composition', None)
                if ready and (
                    getattr(previous_composition, 'composed_scene_path', None)
                    != getattr(composition, 'composed_scene_path', None)
                    or getattr(self.ovrtx.last_ensure_result, 'session_started', False)
                ):
                    # Camera composition changes can repopulate the same native
                    # stage without changing the authoring generation. Restore
                    # latest view values before it can publish that stock pose.
                    ready = self._activate_ovrtx(generation, transforms, attributes)
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
            if generation_requires_physics(predecessor) and not getattr(self.scheduler, "baked_playback_paused", False):
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
        from .ovphysx_to_ovrtx import read_rigid_body_view_paths, rigid_body_view_matrix
        if getattr(self, '_rigid_body_view_generation', None) is not generation:
            object_paths = tuple(mapping.object_path for identity, mapping
                in getattr(generation, 'blender_prim_paths', {}).items()
                if identity.kind == "OBJECT")
            self._rigid_body_view_paths = read_rigid_body_view_paths(
                generation.materialize_usd() if object_paths else '', object_paths)
            self._rigid_body_view_generation = generation
        retained_transforms = tuple(
            replace(value, matrix=rigid_body_view_matrix(value.matrix))
            if value.prim_path in self._rigid_body_view_paths else value
            for value in transforms if value.prim_path not in pending_transforms
        )
        retained_attributes = tuple(
            value
            for value in attributes
            if (value.prim_path, value.attribute) not in pending_attributes
        )
        if not self.ovrtx.activate(generation):
            return False
        self._native_view_generation = generation
        # Build plain immutable USD indexes on the serialized session owner,
        # before publishing ready. The first Blender slider callback reuses
        # them instead of paying a large stage/index scan on the UI thread.
        composition = getattr(self.ovrtx, "composition", None)
        if (composition is not None and composition.source_scene_path == generation.usd_path
                and any(identity.kind == "MATERIAL" for identity in generation.blender_prim_paths)):
            self._native_material_edit_resolver(generation)
        material_attributes = []
        from .view_update_stream import retain_view_value
        retained_materials = getattr(self, '_retained_material_edits', {})
        for key, edit in tuple(retained_materials.items()):
            if _edit_source_mapping(generation, edit) is None:
                # The candidate may remove this material and still fail later.
                # Keep its predecessor value available for rollback.
                continue
            mapped = self._resolve_native_view_target(edit)
            count = len(material_attributes)
            retain_view_value(mapped, attribute_sink=material_attributes.extend)
            if len(material_attributes) != count + 1:
                self.ovrtx.last_error = 'retained native material value is invalid'
                return False
        retained_attributes = (*retained_attributes, *material_attributes)
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
        self._prune_retained_material_edits(generation)
        self.last_activation_update = RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
            values_written=any(result.values_written for result in results),
            should_reset_refinement=any(
                result.should_reset_refinement for result in results
            ),
        )
        return True

    def _prune_retained_material_edits(self, generation: SceneGeneration) -> None:
        # Only the accepted/pending/native generations can still need replay.
        # Do not discard accepted identities while a candidate is being tried.
        owner = getattr(self, 'owner', None)
        generations = (
            generation,
            getattr(owner, 'current_generation', None),
            getattr(owner, 'pending_generation', None),
        )
        retained = getattr(self, '_retained_material_edits', {})
        for key, edit in tuple(retained.items()):
            if not any(item is not None and _edit_source_mapping(item, edit) is not None
                       for item in generations):
                retained.pop(key, None)

    def _resolve_native_view_target(self, intent):
        """Rebind queued authoring values by stable identity after native activation."""

        generation = self._native_view_generation
        if generation is None or not int(intent.provenance.get('blender_session_uid', 0) or 0):
            # Viewport camera and render-setting intents address composition
            # prims independently of the Blender authoring identity table.
            return intent
        mapping = _edit_source_mapping(generation, intent)
        kind = _canonical_blender_id_kind(str(intent.provenance.get('blender_id_kind', '') or ''))
        if mapping is None and kind == 'WORLD':
            uid = int(intent.provenance['blender_session_uid'])
            if uid != int(getattr(generation, 'world_session_uid', 0) or 0):
                return None
            result = self._native_material_edit_resolver(generation).resolve_world_dome()
            if result.value is None:
                raise scene_generation.SceneGenerationError(
                    f'native world target is unavailable: {result.error_reason}'
                )
            path = result.value.prim_path
            return replace(intent, usd_prim_path=path,
                usd_property_path=f'{path}.{intent.usd_attribute}')
        if mapping is None:
            # A structural deletion owns the disappearance of this old value.
            return None
        if intent.blender_property_path in {'matrix_world', 'viewport_camera_matrix'}:
            path = mapping.object_path
        else:
            path = mapping.schema_path or mapping.object_path
            kind = _canonical_blender_id_kind(str(intent.provenance.get('blender_id_kind', '') or ''))
            if kind == 'MATERIAL':
                resolver = self._native_material_edit_resolver(generation)
                result = resolver.resolve_material(
                    _MaterialEditIdentity(int(intent.provenance['blender_session_uid'])),
                    usd_attribute=intent.usd_attribute, property_name=intent.blender_property_path,
                )
                if result.value is None:
                    raise scene_generation.SceneGenerationError(
                        f'native material target is unavailable: {result.error_reason}'
                    )
                path = result.value.prim_path
        mapped = replace(intent, usd_prim_path=path,
            usd_property_path=(f'{path}.{intent.usd_attribute}' if intent.usd_attribute else ''))
        if (intent.blender_property_path == 'matrix_world'
                and path in getattr(self, '_rigid_body_view_paths', ())):
            from .ovphysx_to_ovrtx import rigid_body_view_matrix
            mapped = replace(mapped, value=rigid_body_view_matrix(mapped.value))
        if kind == 'MATERIAL':
            # Native OpenPBR paths are outside the stock mapping namespace.
            # Retain their verified Blender identity rather than those paths.
            retained = getattr(self, '_retained_material_edits', None)
            if retained is None:
                retained = self._retained_material_edits = {}
            retained[(int(intent.provenance['blender_session_uid']), intent.usd_attribute)] = mapped
        return mapped

    def _native_material_edit_resolver(self, generation):
        # Blender callbacks resolve against the accepted authoring generation;
        # the renderer can already own its successor. Build an independent,
        # bounded immutable composition index here, without reading Blender RNA.
        composition = getattr(getattr(self, 'ovrtx', None), 'composition', None)
        path = generation.usd_path
        if composition is not None and composition.source_scene_path == path:
            path = composition.composed_scene_path
        cached = getattr(self, '_native_material_context', None)
        if cached is not None and cached[0] is generation and cached[1] == path:
            return cached[2]
        resolver = UsdPrimResolver(
            material_source_scene_path=generation.usd_path,
            world_dome_path=next((mapping.schema_path or mapping.object_path
                for identity, mapping in generation.blender_prim_paths.items() if identity.kind == 'WORLD'), ''),
            material_paths_by_session_uid={identity.session_uid: mapping.schema_path or mapping.object_path
                for identity, mapping in generation.blender_prim_paths.items() if identity.kind == 'MATERIAL'},
        )
        resolver.scan(RenderRequest(input_usd_path=path))
        if resolver.diagnostics()['scan'].get('available', False):
            self._native_material_context = (generation, path, resolver)
        return resolver

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
            if getattr(self.scheduler, "has_pending_sim_updates", False) and not getattr(self.scheduler, "baked_playback_paused", False):
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
                if self.lifecycle_status != "open":
                    self.ovrtx.last_error = "activation superseded: runtime is closing"
                    return False
                if (getattr(self.scheduler, "has_pending_sim_updates", False) and not getattr(self.scheduler, "baked_playback_paused", False)) or getattr(
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
            # "closing" rejects admissions. Teardown then runs without the
            # admission lock: an activation holding the transport takes the
            # admission lock, and deactivation waits for that transport.
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
        with self._admission_lock:
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
        transport_lease: Any = None,
    ) -> None:
        self._runtime = runtime
        self._generation = generation
        self._previous_request = previous_request
        self._transport_lease = transport_lease

    @property
    def controller(self) -> Any:
        return self._runtime.ovrtx.controller

    @property
    def last_ensure_result(self) -> Any:
        return self._runtime.ovrtx.last_ensure_result

    def deactivate(self) -> str:
        try:
            return self._restore()
        finally:
            # Queue ownership spans activation, every final sample, and restore.
            # Background replay and other views retain intent until this release.
            lease, self._transport_lease = self._transport_lease, None
            if lease is not None:
                lease.__exit__(None, None, None)

    def _restore(self) -> str:
        if self._previous_request is None:
            # Nothing preceded the final render: keep its session loaded so the
            # next render reuses the healthy worker and native renderer. Scene
            # close, file load, runtime removal and Blender exit tear it down.
            return "stopped"
        try:
            activate = getattr(
                self._runtime, "activate_blocking", self._runtime.activate
            )
            activate(self._generation, self._previous_request)
        except Exception:
            fail_closed_runtime_reuse(self._runtime)
            return "failed"
        return "stopped"

_physics_snapshots: dict[int, Any] = {}
_physics_restart_pending: dict[int, Any] = {}
_physics_initial_bases: dict[int, dict[tuple[int, str], Any]] = {}
_viewport_generation_timers: set[int] = set()
# scene uid -> edit revision at which a deferred viewport reconcile failed
_blocked_revisions: dict[int, int] = {}
# scene uid -> Object physics identity -> session_uid of its first observed owner
_identity_owners: dict[int, dict[str, int]] = {}
_viewport_generation_errors: dict[int, str] = {}


# F12's render_init builds its request on Blender's render job thread while
# main-thread handlers keep observing edits; both mutate the session dicts.
_session_state_lock = threading.RLock()
_SESSION_STATE_MAIN_THREAD_WAIT_S = 30.0


class _SessionState:
    def __enter__(self) -> None:
        if threading.current_thread() is threading.main_thread():
            # The job thread holds this only while exporting; never hang the UI.
            if not _session_state_lock.acquire(timeout=_SESSION_STATE_MAIN_THREAD_WAIT_S):
                raise RuntimeError("scene generation state is busy with a final render")
        else:
            _session_state_lock.acquire()

    def __exit__(self, *_exc: Any) -> None:
        _session_state_lock.release()


def observe_physics_edits(scene: Any, *, explicit_physics_restart: bool = False) -> Any:
    """Snapshot on the main thread and queue intent, without export or RPC."""
    with _SessionState():
        return _observe_physics_edits(scene, explicit_physics_restart=explicit_physics_restart)


def observe_depsgraph_physics_edits(
    scene: Any, depsgraph: Any, *, updates: tuple[Any, ...] | None = None,
) -> Any:
    """Reuse a joint-free observation for proven shader-only notifications.

    Physics participants, authoring invalidation, unknown/geometry/pose changes
    and raw newly authored opinions keep the full capture. The periodic repair
    observer remains conservative even when a shader drag uses this fast path.
    ``updates`` is the caller's snapshot of ``depsgraph.updates`` (``None``
    reads the depsgraph); an empty batch still takes the full capture.
    """
    from .physics_edit_snapshot import PhysicsChanges, scene_authoring_observation, _authored
    uid = int(getattr(scene, "session_uid", 0) or 0)
    observation = _physics_view_observations.get(uid)
    if (observation is None or observation[1] or uid in _physics_authoring_dirty
            or uid in _physics_restart_pending or _bake_capture_owner is not None
            or scene_authoring_observation(scene) != observation[0]):
        return observe_physics_edits(scene)
    updates = tuple(getattr(depsgraph, "updates", ()) if updates is None else updates)
    if not updates:
        return observe_physics_edits(scene)
    for update in updates:
        source = getattr(update, "id", update)
        source = getattr(source, "original", source)
        kind = str(getattr(getattr(source, "bl_rna", None), "identifier", ""))
        if kind == "ShaderNodeTree":
            continue
        if (kind not in {"Material", "Mesh", "Object"}
                or getattr(update, "is_updated_geometry", None) is not False
                or getattr(update, "is_updated_transform", None) is not False
                or getattr(update, "is_updated_shading", None) is not True
                or (kind == "Object" and str(getattr(source, "type", "")) != "MESH")):
            return observe_physics_edits(scene)
        if _authored(getattr(getattr(source, "ov", None), "physics", None)):
            return observe_physics_edits(scene)
        keys = getattr(source, "keys", lambda: ())()
        if any(str(key).startswith(("pxr:usd:physics_", "pxr:physics:")) for key in keys):
            return observe_physics_edits(scene)
    return PhysicsChanges()


_physics_authoring_dirty: set[int] = set()
_physics_view_observations: dict[int, tuple[Any, bool]] = {}
_physics_idle_discovery: dict[int, Any] = {}


def _remember_physics_snapshot(uid: int, snapshot: Any) -> None:
    from .physics_edit_snapshot import PhysicsSnapshot
    _physics_snapshots[uid] = snapshot
    _physics_view_observations[uid] = (
        PhysicsSnapshot(tuple(row for row in snapshot.leaves if row[0] == 'SCENE')),
        any(row[0] != 'SCENE' for row in snapshot.leaves),
    )
    _physics_authoring_dirty.discard(uid)
    _physics_idle_discovery.pop(uid, None)


def _render_only_physics_current(scene):
    from .physics_edit_snapshot import scene_authoring_observation
    uid = int(getattr(scene, 'session_uid', 0) or 0)
    observation = _physics_view_observations.get(uid)
    return bool(observation is not None and not observation[1]
                and uid not in _physics_authoring_dirty
                and scene_authoring_observation(scene) == observation[0])


def observe_physics_depsgraph(scene, depsgraph, *, updates=None):
    """Camera-only navigation reuses a proven physics-free observation.

    ``updates`` is the caller's snapshot of ``depsgraph.updates``; ``None``
    reads the depsgraph (once). An empty batch is not camera-only.
    """
    from . import depsgraph_updates
    from .physics_edit_snapshot import PhysicsChanges
    if _render_only_physics_current(scene):
        updates = tuple(getattr(depsgraph, 'updates', ()) if updates is None else updates)
        # An empty batch proves nothing; fail closed like the shader fast path.
        camera_only = bool(updates)
        for update in updates:
            value = getattr(update, 'id', update)
            value = getattr(value, 'original', value)
            if (str(getattr(value, 'type', '')) != 'CAMERA'
                    and str(getattr(getattr(value, 'bl_rna', None), 'identifier', '')) != 'Camera'):
                camera_only = False
                break
        if camera_only:
            return PhysicsChanges()
    return observe_depsgraph_physics_edits(
        scene, depsgraph, **depsgraph_updates.keyword(observe_depsgraph_physics_edits, updates))


def note_physics_authoring_changed(owner: Any = None) -> None:
    """Latch RNA authoring changes without capture, export, or runtime work.

    Shared Object/Material/bone data can affect more than the UI's active scene.
    Conservatively invalidate scene identities, without traversing their objects.
    """
    if bpy is None:
        uid = int(getattr(owner, 'session_uid', 0) or 0)
        if uid:
            _physics_authoring_dirty.add(uid)
        return
    for scene in getattr(getattr(bpy, 'data', None), 'scenes', ()):
        uid = int(getattr(scene, 'session_uid', 0) or 0)
        if uid:
            _physics_authoring_dirty.add(uid)


def _observe_physics_edits(scene: Any, *, explicit_physics_restart: bool) -> Any:
    from .physics_edit_snapshot import capture, classify, PhysicsChanges, absorb_presentation
    if (_bake_capture_owner is not None or _baked_active(scene)
        or (not explicit_physics_restart
            and int(getattr(scene, "session_uid", 0) or 0) in _baked_playback_paused)):
        return PhysicsChanges()
    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid <= 0 or getattr(scene, "ov", None) is None:
        return PhysicsChanges()
    current = capture(scene)
    previous = _physics_snapshots.get(uid)
    if interactive_operator_state.interactive_edit_bridge_suppressed():
        if previous is not None:
            absorbed = absorb_presentation(previous, current)
            _remember_physics_snapshot(uid, absorbed)
            if absorbed != current:
                # Suppression advances only transient transform rows. Authoring
                # differences (including an explicit mark_dirty notification)
                # stay latched for the next unsuppressed observation.
                _physics_authoring_dirty.add(uid)
        return PhysicsChanges()
    _remember_physics_snapshot(uid, current)
    old_bases = {} if previous is None else {
        (row[1], row[2]): row[5] for row in previous.leaves if row[3] == "body_basis"
    }
    new_bases = {(row[1], row[2]): row[5] for row in current.leaves if row[3] == "body_basis"}
    retained = _physics_initial_bases.setdefault(uid, {})
    for key in tuple(retained):
        if key not in new_bases:
            del retained[key]
    for key, basis in new_bases.items():
        if key not in retained or basis != old_bases.get(key):
            retained[key] = basis
    if previous is None:
        return PhysicsChanges()
    changes = classify(previous, current)
    if changes.restart:
        _viewport_generation_errors.pop(uid, None)
        # This is a complete-scene barrier, not a joint-only USD export. Every
        # contacting participant remains present even outside the joint graph.
        _physics_restart_pending[uid] = changes
    if changes.redraw:
        for screen in getattr(getattr(bpy, "data", None), "screens", ()):
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    return changes


def absorb_physics_presentation() -> None:
    """Called before releasing presentation suppression, including delayed updates."""
    if bpy is not None:
        for scene in bpy.data.scenes:
            if int(scene.session_uid) in _physics_snapshots:
                observe_physics_edits(scene)


def _with_initial_physics_poses(scene: Any, compile_generation: Any) -> Any:
    """Export authored initial bases, restoring presentation even on failure.

    Bases are local to the authoring parent/rest hierarchy, so restoring a rig
    and its children does not bake their currently displayed world transforms.
    No RNA object references survive between observations or file loads.
    """
    retained = _physics_initial_bases.get(int(scene.session_uid), {})
    if not retained:
        return compile_generation()
    from mathutils import Matrix
    # Only presented bodies differ from their authored basis. Unchanged scenes
    # write nothing, so view_draw (which forbids ID writes) can still compile.
    pending = []
    for obj in scene.objects:
        for bone_name, target in [("", obj), *((bone.name, bone) for bone in getattr(getattr(obj, "pose", None), "bones", ()))]:
            basis = retained.get((int(obj.session_uid), bone_name))
            if basis is not None and basis != tuple(tuple(float(v) for v in row) for row in target.matrix_basis):
                pending.append((target, basis))
    if not pending:
        return compile_generation()
    restored = []
    try:
        for target, basis in pending:
            previous = target.matrix_basis.copy()
            try:
                target.matrix_basis = Matrix(basis)
            except AttributeError as exc:
                raise scene_generation.SceneGenerationError(
                    "Physics initial poses cannot be restored while drawing; "
                    "rewind or press Play to rebuild the scene"
                ) from exc
            restored.append((target, previous))
        for view_layer in scene.view_layers:
            view_layer.update()
        return compile_generation()
    finally:
        for target, basis in restored:
            target.matrix_basis = basis
        if restored:
            for view_layer in scene.view_layers:
                view_layer.update()


def repair_duplicate_identities(scene: Any, *, report: bool = True) -> list[str]:
    """Reidentify Blender copies of physics Objects; the original keeps its ID.

    Shift+D copies ``ov.physics.persistent_id``. Session uids only increase,
    so the Object first observed with an identity is the original. Without
    history (a file loaded with collisions) the lowest session uid keeps it.
    Object joint endpoints are Object pointers, so no joint is redirected.
    Main thread, writable contexts only; linked data keeps its explicit error.
    """
    from .physics_endpoints import EndpointError, repair_object_identity, require_writable
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owners = _identity_owners.setdefault(uid, {})
    from .physics_endpoints import initialize_owner
    for obj in getattr(scene, "objects", ()):
        physics = getattr(getattr(obj, "ov", None), "physics", None)
        # Owners authored through properties alone still need an identity.
        if (physics is not None and not physics.persistent_id and not physics.joint.enabled
                and (physics.rigid_body.schema_opinion != "INHERIT" or len(physics.colliders))):
            try:
                require_writable(obj)
            except EndpointError:
                continue
            initialize_owner(obj)
    groups: dict[str, list[Any]] = {}
    for obj in getattr(scene, "objects", ()):
        identity = getattr(getattr(getattr(obj, "ov", None), "physics", None), "persistent_id", "")
        if identity:
            groups.setdefault(identity, []).append(obj)
    repaired = []
    for identity, objects in groups.items():
        if len(objects) < 2:
            owners[identity] = int(objects[0].session_uid)
            continue
        known = owners.get(identity)
        keeper = next((o for o in objects if int(o.session_uid) == known), None)
        keeper = keeper or min(objects, key=lambda o: int(o.session_uid))
        owners[identity] = int(keeper.session_uid)
        for obj in sorted(objects, key=lambda o: int(o.session_uid)):
            if obj == keeper:
                continue
            try:
                require_writable(obj)
                changed = repair_object_identity(obj, scene)
            except EndpointError:
                # Linked or malformed identities keep their explicit build error.
                continue
            if changed:
                owners[obj.ov.physics.persistent_id] = int(obj.session_uid)
                repaired.append(obj.name)
    if repaired and report:
        from . import user_messages
        user_messages.report_info("Gave duplicated physics Objects their own identity: " + ", ".join(repaired),
                                  context=f"physics-identity:{uid}", dedup=False)
    return repaired


def _poll_physics_edits() -> float:
    # RNA collection add/remove through Python need not emit a depsgraph update.
    # Poll leaves on Blender's timer thread (the main thread), never a worker.
    if bpy is not None and _bake_capture_owner is None:
        for scene in bpy.data.scenes:
            runtime = _runtimes.get(int(scene.session_uid))
            if runtime is not None and runtime.playback_lock._presentation_records:
                from .physics_pose_presentation import enforce
                try:
                    # Dispatch already-pending artist edits before suppressing
                    # only the subsequent writes to physics-owned channels.
                    # That dispatch may release or replace this runtime.
                    if not interactive_operator_state.interactive_edit_bridge_suppressed():
                        bpy.context.view_layer.update()
                    runtime = _runtimes.get(int(scene.session_uid))
                    if runtime is not None and runtime.playback_lock._presentation_records:
                        with interactive_operator_state.suppress_interactive_edit_bridge():
                            enforce(runtime.playback_lock)
                            bpy.context.view_layer.update()
                except Exception as exc:
                    _record_playback_error(int(scene.session_uid), str(exc))
            if _render_only_physics_current(scene):
                from .physics_edit_snapshot import RenderOnlyDiscovery
                uid = int(scene.session_uid)
                discovery = _physics_idle_discovery.get(uid)
                if discovery is None:
                    discovery = _physics_idle_discovery[uid] = RenderOnlyDiscovery(scene)
                if not discovery.changed():
                    continue
            if getattr(scene, "ov", None) is not None:
                try:
                    repair_duplicate_identities(scene)
                except Exception:
                    # A raising timer is unregistered for the session; keep polling.
                    traceback.print_exc()
            try:
                observe_physics_edits(scene)
            except RuntimeError:
                # Busy with a long final-render export; poll again later.
                traceback.print_exc()
    return 0.2


def mark_scene_dirty(
    scene: Any,
    affected_ids: set[BlenderId] | frozenset[BlenderId] | tuple[BlenderId, ...] = (),
    *,
    defer_world_reconciliation: bool = False,
    update_provenance: Mapping[BlenderId, SceneUpdateProvenance] | None = None,
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
    queued = _dirty.setdefault(uid, set())
    records = _dirty_provenance.setdefault(uid, {})
    for identity in affected:
        incoming = (update_provenance or {}).get(identity, SceneUpdateProvenance())
        records[identity] = (records.get(identity, SceneUpdateProvenance()).merged(incoming)
                             if identity in queued else incoming)
    queued.update(affected)
    _viewport_generation_errors.pop(uid, None)
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
    """Reconcile dirty Blender IDs on a main-thread timer, never in a callback."""
    owner = _owners.get(uid)
    if (
        bpy is None
        or owner is None
        or owner.current_generation is None
        or uid in _reconciliation_timers
        or uid not in _dirty
    ):
        return
    blocked = _blocked_reconciliations.get(uid)
    if blocked is None:
        _blocked_revisions.pop(uid, None)  # Cleared elsewhere (e.g. activation).
    if blocked is not None and not all(identity.kind == "WORLD" for identity in newly_affected):
        # A failed reconcile retries after the next edit (mark_scene_dirty bumps
        # the revision) or once a failed addition is gone, never per redraw.
        # A blocked entry from another path (e.g. rejected activation) waits
        # for the next edit too.
        revision = _preparation_revisions.get(uid, 0)
        if (_blocked_revisions.setdefault(uid, revision) == revision
                and not _dirty_addition_removed(scene, uid, owner)):
            return
    elif blocked is not None:
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
            try:
                # The export's own selection/active writes flush as a depsgraph
                # update; evaluate under suppression so it is not a new edit.
                with interactive_operator_state.suppress_interactive_edit_bridge():
                    try:
                        generation_for_scene(scene)
                    finally:
                        for view_layer in getattr(scene, "view_layers", ()):
                            view_layer.update()
            except Exception as exc:
                # generation_for_scene retains the IDs as blocked; tell the user.
                _blocked_revisions[uid] = _preparation_revisions.get(uid, 0)
                from . import user_messages
                user_messages.report_error(f"Scene update failed: {type(exc).__name__}: {exc}",
                                           context=f"scene-reconcile:{uid}")
                return None
            if _blocked_revisions.pop(uid, None) is not None:
                from . import user_messages
                # Same context, so a later identical failure is reported again.
                user_messages.report_info("Scene updated; the previous scene update error is resolved",
                                          context=f"scene-reconcile:{uid}")
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


def _dirty_addition_removed(scene: Any, uid: int, owner: Any) -> bool:
    """A dirty Object that is neither in the scene nor mapped was deleted."""
    mapped = owner.current_generation.blender_prim_paths
    present = {blender_id(obj, "OBJECT") for obj in getattr(scene, "objects", ())}
    return any(identity.kind == "OBJECT" and identity not in present and identity not in mapped
               for identity in _dirty.get(uid, ()))


def _schedule_viewport_generation(scene: Any, uid: int) -> None:
    if uid in _viewport_generation_timers:
        return
    _viewport_generation_timers.add(uid)

    def construct() -> float | None:
        if uid not in _viewport_generation_timers:
            return None  # Closed since scheduling.
        try:
            _require_session_open()
            repair_duplicate_identities(scene)
            generation_for_scene(scene)
        except RuntimeServicesPreparingError:
            # File-load cleanup is asynchronous. Keep this one timer alive
            # until it settles; a transient wait must not latch a build error.
            return 0.1
        except Exception as exc:
            # Keep the error until a scene edit, so the redraw that shows it
            # does not schedule another failing build.
            _viewport_generation_errors[uid] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        else:
            _viewport_generation_errors.pop(uid, None)
        _viewport_generation_timers.discard(uid)
        try:
            scene.update_tag()  # Redraw rendered viewports with the result.
        except ReferenceError:
            pass
        return None

    try:
        bpy.app.timers.register(construct, first_interval=0.0)
    except Exception:
        _viewport_generation_timers.discard(uid)
        raise


def topology_identity_changed(scene: Any) -> bool:
    """Detect supported Blender IDs added to or removed from the generation."""

    return bool(topology_identity_changes(scene))


def topology_identity_changes_for_depsgraph(
    scene: Any, depsgraph: Any, *, updates: tuple[Any, ...] | None = None,
) -> set[BlenderId]:
    """Known camera edits cannot change the exported object/material inventory."""
    owner = _owners.get(int(getattr(scene, "session_uid", 0) or 0))
    generation = (None if owner is None else
                  getattr(owner, "pending_generation", None) or owner.current_generation)
    if generation is not None:
        saw_update = False
        for update in getattr(depsgraph, "updates", ()) if updates is None else updates:
            value = getattr(update, "id", update)
            value = getattr(value, "original", value)
            saw_update = True
            if str(getattr(value, "type", "")) == "CAMERA":
                if blender_id(value, "OBJECT") in generation.blender_prim_paths:
                    continue
            elif str(getattr(getattr(value, "bl_rna", None), "identifier", "")) == "Camera":
                continue
            break
        else:
            if saw_update:
                return set()
    return topology_identity_changes(scene)


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
        if str(getattr(obj, "type", "")) not in scene_generation.SUPPORTED_OBJECT_TYPES:
            continue
        identity = blender_id(obj, "OBJECT")
        if _is_editable_id(obj) or identity in mapped:
            result.add(identity)
        for material in scene_generation.object_export_materials(obj):
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


def _affected_blender_id(update: Any) -> BlenderId | None:
    value = getattr(update, "id", None)
    value = getattr(value, "original", value)
    if not _is_editable_id(value):
        return None
    identifier = _canonical_blender_id_kind(str(getattr(getattr(value, "bl_rna", None), "identifier", "")))
    if identifier == "OBJECT":
        kind = "OBJECT" if str(getattr(value, "type", "")) in scene_generation.GEOMETRY_TYPES | {"LIGHT"} else None
    elif identifier in scene_generation.GEOMETRY_TYPES | {"WORLD", "LIGHT", "MATERIAL"}:
        kind = identifier
    else:
        kind = None
    uid = int(getattr(value, "session_uid", 0) or 0)
    return BlenderId(kind, uid) if kind is not None and uid > 0 else None


def affected_blender_ids(depsgraph: Any, *, updates: tuple[Any, ...] | None = None) -> set[BlenderId]:
    return {identity for update in (getattr(depsgraph, "updates", ()) if updates is None else updates)
            if (identity := _affected_blender_id(update)) is not None}


def affected_blender_update_provenance(
    depsgraph: Any, *, updates: tuple[Any, ...] | None = None,
) -> dict[BlenderId, SceneUpdateProvenance]:
    records = {}
    context = (str(getattr(depsgraph, "mode", "")),
               int(getattr(getattr(depsgraph, "scene", None), "session_uid", 0) or 0),
               str(getattr(getattr(depsgraph, "view_layer", None), "name", "")))
    def flag(update, name):
        value = getattr(update, name, None)
        return value if value is True or value is False else None
    for update in getattr(depsgraph, "updates", ()) if updates is None else updates:
        identity = _affected_blender_id(update)
        if identity is None:
            continue
        note = SceneUpdateProvenance(flag(update, "is_updated_geometry"),
            flag(update, "is_updated_transform"), flag(update, "is_updated_shading"), context)
        records[identity] = records[identity].merged(note) if identity in records else note
    return records


def _canonical_blender_id_kind(identifier: str) -> str:
    identifier = str(identifier or "").upper()
    if identifier in {'TEXTCURVE', 'SURFACECURVE'}:
        return 'CURVE'
    if identifier == 'METABALL':
        return 'META'
    return "LIGHT" if identifier.endswith("LIGHT") else identifier




def _ensure_render_visibility_observer(scene):
    """Consume coalesced visibility changes outside the view_draw callback."""
    uid = int(getattr(scene, "session_uid", 0) or 0)
    register = getattr(getattr(getattr(bpy, "app", None), "timers", None), "register", None)
    owner = _owners.get(uid)
    generation = getattr(owner, "current_generation", None)
    if not hasattr(generation, "object_visibility"):
        return
    if uid <= 0 or uid in _visibility_observers or not callable(register):
        return
    from .object_visibility import VisibilityWatch
    watch = VisibilityWatch(scene, bpy)
    def forget():
        watch.close()
        _visibility_observers.pop(uid, None)
    def observe():
        if _visibility_observers.get(uid) is not observe:
            watch.close()
            return None
        if uid not in _owners:
            forget()
            return None
        try:
            if not is_authoring(scene) and watch.pending():
                watch.dirty = False
                with _SessionState():
                    _check_render_visibility(scene)
                    affected = set(_dirty.get(uid, ()))
                if affected:
                    _schedule_world_reconciliation(scene, uid, affected)
        except ReferenceError:
            # File loading releases Scene/ViewLayer RNA. load_pre normally closes
            # observers first; never retain a freed owner after another load path.
            forget()
            return None
        except ValueError as error:
            # A nonactive scene with several layers has no unambiguous owner.
            # Resume when its window owns an explicit active layer.
            if "explicit owning view layer" not in str(error):
                forget()
                raise
            watch.invalidate()
            return .25
        except Exception:
            watch.invalidate()
            raise
        return .25
    observe.close = watch.close
    _visibility_observers[uid] = observe
    register(observe, first_interval=.25)


def _check_render_visibility(scene):
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generation = None if owner is None else getattr(owner, "pending_generation", None) or owner.current_generation
    if generation is None:
        return
    # Compare against the generation being presented, so rejection retries and
    # layer switches remain observable without mutating Blender in view_draw.
    current = scene_generation._object_visibility(scene)
    previous = getattr(generation, "object_visibility", {})
    shadows = scene_generation._mesh_shadow_visibility(scene)
    old_shadows = getattr(generation, "shadow_visibility", {})
    changed = {key for key in current.keys() & previous.keys() if current[key] != previous[key]}
    shadow_changed = {key for key in shadows.keys() & old_shadows.keys() if shadows[key] != old_shadows[key]}
    changed.update(shadow_changed)
    if changed:
        # This observer compared visibility values, not mesh geometry. Preserve
        # that fact; merging must still retain any earlier geometry notification.
        mark_scene_dirty(scene, changed, update_provenance={
            key: SceneUpdateProvenance(False, False, False)
            for key in changed - shadow_changed})


def require_supported_render_generation(scene: Any, generation: SceneGeneration | None = None) -> None:
    """Reject cached candidate controls without committing a pending population."""
    with _SessionState():
        _require_supported_render_generation(scene, generation)


def _require_supported_render_generation(scene: Any, generation: SceneGeneration | None) -> None:
    uid = int(getattr(scene, 'session_uid', 0) or 0)
    owner = _owners.get(uid)
    revision = _preparation_revisions.get(uid, 0)
    failure = getattr(owner, '_render_admission_failure', None)
    if generation is None:
        if failure is not None and failure[0] == revision:
            raise scene_generation.SceneGenerationError(failure[1])
        return
    if not getattr(generation, 'unadmitted_rendering_overrides', ()):
        if owner is not None:
            owner._render_admission_failure = None
        return
    message = ('Blender matte and holdout overrides are unavailable in supported rendering; '
               'set both properties to Use Source (INHERIT).')
    if owner is not None:
        owner._render_admission_failure = (revision, message)
        if owner.pending_generation is generation:
            affected = _pending_affected.pop(uid, set()) | _dirty.get(uid, set())
            _dirty.setdefault(uid, set()).update(affected)
            _blocked_reconciliations[uid] = (message, set(affected))
            _blocked_revisions[uid] = revision
            owner.reject(generation)
    raise scene_generation.SceneGenerationError(message)


def generation_for_final_render(scene: Any) -> SceneGeneration:
    """Use evaluated per-frame geometry for the ordinary Blender final lane."""
    with _SessionState():
        generation_for_scene(scene)
        uid = _scene_uid(scene)
        owner = _owners[uid]
        _constructing.add(uid)
        try:
            with interactive_operator_state.suppress_interactive_edit_bridge():
                return owner.reconcile_final_frame(scene)
        finally:
            _constructing.discard(uid)


def generation_for_scene(
    scene: Any,
    *,
    work_root: str | Path | None = None,
    explicit_physics_restart: bool = False,
) -> SceneGeneration:
    """Return the current private generation for one Blender render callback."""

    _require_session_open()
    with _SessionState():
        from . import direct_scene_updates
        direct_scene_updates.require_projection_for_other_consumer(
            _owners.get(int(getattr(scene, "session_uid", 0) or 0)), scene)
        _check_render_visibility(scene)
        generation = _generation_for_scene(
            scene, work_root=work_root, explicit_physics_restart=explicit_physics_restart
        )
        if bpy is not None:
            from .native_scene_edit_batch import remember_generation_snapshot
            remember_generation_snapshot(scene,generation)
        return generation


def _generation_for_scene(
    scene: Any,
    *,
    work_root: str | Path | None,
    explicit_physics_restart: bool,
) -> SceneGeneration:
    _require_session_open()
    uid = _scene_uid(scene)
    if _baked_active(scene):
        _baked_playback_paused.add(uid)
    observe_physics_edits(scene, explicit_physics_restart=explicit_physics_restart)
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
                generation = _with_initial_physics_poses(scene, lambda: owner.replace(scene))
        finally:
            _constructing.discard(uid)
        if generation is not None:
            from .physics_edit_snapshot import capture
            if getattr(scene, "ov", None) is not None:
                _remember_physics_snapshot(uid, capture(scene))
            _physics_restart_pending.pop(uid, None)
            _mark_initial_generation_ready(uid, generation)
            return generation
    if getattr(owner, "pending_generation", None) is not None:
        return owner.reuse()
    if (not explicit_physics_restart
        and any(identity.kind == "PHYSICS" for identity in _dirty.get(uid, ()))):
        return owner.reuse()
    if uid in _physics_restart_pending:
        if not explicit_physics_restart:
            return owner.reuse()
        from .physics_authoring import require_scene_ready
        require_scene_ready(scene)
        # A PHYSICS barrier compiles the complete scene into a pending candidate;
        # runtime handoff must succeed before the existing owner accepts it.
        affected = {BlenderId("PHYSICS", uid)}
        with interactive_operator_state.suppress_interactive_edit_bridge():
            generation = _with_initial_physics_poses(scene, lambda: owner.reconcile(scene, affected))
        from .physics_edit_snapshot import capture
        _remember_physics_snapshot(uid, capture(scene))
        _physics_restart_pending.pop(uid, None)
        _dirty.pop(uid, None)
        if generation is not None:
            _pending_affected[uid] = affected
            return generation
        return owner.reuse()
    if getattr(owner, "pending_generation", None) is not None:
        return owner.reuse()
    if uid in _dirty:
        affected = _dirty.pop(uid)
        provenance = _dirty_provenance.pop(uid, {})
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
                generation = _with_initial_physics_poses(scene, lambda: owner.reconcile(scene, affected,
                    **({"update_provenance": provenance} if any(note.geometry is not None or note.transform is not None or note.shading is not None for note in provenance.values()) else {})))
        except Exception as exc:
            retained = _dirty.setdefault(uid, set())
            records = _dirty_provenance.setdefault(uid, {})
            for identity in affected:
                incoming = provenance.get(identity, SceneUpdateProvenance())
                records[identity] = records.get(identity, SceneUpdateProvenance()).merged(incoming) if identity in retained else incoming
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

    from . import direct_scene_updates
    direct_scene_updates.require_projection_for_other_consumer(
            _owners.get(int(getattr(scene, "session_uid", 0) or 0)), scene)
    _ensure_render_visibility_observer(scene)
    uid = int(getattr(scene, "session_uid", 0) or 0)
    if uid <= 0:
        return generation_for_scene(scene)
    owner = _owners.get(uid)
    if bpy is not None and (owner is None or owner.current_generation is None):
        # A complete export stamps temporary ID properties, which view_draw
        # forbids. Build on the main thread and report until it is accepted.
        error = _viewport_generation_errors.get(uid)
        if error is None:
            _schedule_viewport_generation(scene, uid)
            raise scene_generation.SceneGenerationPending("Building the scene generation")
        raise scene_generation.SceneGenerationError(error)
    if owner is None or owner.current_generation is None:
        return generation_for_scene(scene)
    if uid not in _dirty:
        # View redraws reuse immutable accepted state. Physics authoring is
        # observed by the depsgraph/property owner, not by every camera draw:
        # traversing registered default RNA groups here makes an unchanged
        # scene cost hundreds of milliseconds per thousand ordinary objects.
        # Explicit simulation/restart/final-render requests retain their full
        # generation_for_scene admission and observation.
        from .physics_edit_snapshot import scene_authoring_observation
        observation = _physics_view_observations.get(uid)
        # RNA property_unset and raw collection mutations do not call update.
        # Keep conservative full observation for physics participants, while
        # scene controls are cheap enough to compare directly every view draw.
        if (uid in _physics_authoring_dirty or observation is None
                or observation[1]
                or scene_authoring_observation(scene) != observation[0]):
            observe_physics_edits(scene)
        if _baked_active(scene):
            _baked_playback_paused.add(uid)
        return owner.reuse()
    # Reconciling exports and changes selection/active Object, which view_draw
    # forbids. Present the accepted/pending generation until the timer runs.
    _schedule_world_reconciliation(scene, uid, set(_dirty[uid]))
    return owner.reuse()


def viewport_joint_admission_clear(scene: Any) -> bool:
    """Reuse a proven joint-free observation, never an authored-physics guard.

    Property callbacks and undo/redo latch authoring invalidation; depsgraph
    observation captures membership and raw changes. Unknown/pending/dirty
    scenes and any authored participant retain full mandatory admission.
    Final render, simulation and bake callers never use this fast path.
    """
    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    observation = _physics_view_observations.get(uid)
    return bool(
        owner is not None and owner.current_generation is not None
        and uid not in _dirty and uid not in _physics_authoring_dirty
        and observation is not None and not observation[1]
    )


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
    _queue_owned_world_topology(scene, owner, edit, plan)
    return EditWorkflowResult(
        action=WorkflowAction.COMPOSE,
        status=EditStatus.QUEUED,
        reason="scene_generation_dirty",
        plan=plan,
    )


def current_generation_object_paths(scene: Any) -> dict[int, str]:
    """Resolve authored object identities without exporting or opening a stage."""
    owner = _owners.get(int(getattr(scene, "session_uid", 0) or 0))
    generation = None if owner is None else owner.current_generation
    if generation is None:
        return {}
    return {
        identity.session_uid: mapping.object_path
        for identity, mapping in generation.blender_prim_paths.items()
        if identity.kind == "OBJECT"
    }


def current_generation_edit_context(
    scene: Any,
) -> tuple[UsdPrimResolver | None, Iterable[Any]]:
    """Resolve untagged Blender values against the current scene generation."""

    uid = int(getattr(scene, "session_uid", 0) or 0)
    owner = _owners.get(uid)
    generation = None if owner is None else owner.current_generation
    if generation is None:
        return None, ()
    runtime = _runtimes.get(uid)
    controller = getattr(getattr(runtime, "ovrtx", None), "controller", None)
    spec = getattr(controller, "_spec", None)
    composition = getattr(spec, "ovrtx_scene_composition", None)
    scene_path = generation.usd_path
    if composition is not None and composition.source_scene_path == generation.usd_path:
        scene_path = composition.composed_scene_path
    cached = getattr(runtime, '_current_edit_context', None)
    if cached is not None and cached[0] is generation and cached[1] == scene_path:
        resolver = cached[2]
        resolver.begin_edit_callback()
        return resolver, getattr(scene, 'objects', ())
    resolver = UsdPrimResolver(
        world_dome_path=next((mapping.schema_path or mapping.object_path
            for identity, mapping in generation.blender_prim_paths.items() if identity.kind == 'WORLD'), ''),
        object_paths_by_session_uid={
            identity.session_uid: mapping.object_path
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT"
        },
        material_source_scene_path=generation.usd_path,
        material_paths_by_session_uid={
            identity.session_uid: mapping.schema_path or mapping.object_path
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "MATERIAL"
        },
        light_paths_by_object_session_uid={
            identity.session_uid: mapping.schema_path
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT" and mapping.blender_id_type == "LIGHT"
        },
        mesh_topology_change_resolver=lambda mesh: _mesh_topology_change(
            scene, generation, mesh
        ),
        material_graph_change_resolver=lambda material: _material_graph_change(generation, material),
        mesh_shading_unchanged_resolver=lambda mesh: _mesh_shading_unchanged(generation, mesh),
        unsupported_light_change_resolver=lambda light: _unsupported_light_changes(generation, light),
    )
    # Resolve live values against the accepted render composition, which can
    # replace stock PreviewSurface shaders with OpenPBR presentation shaders.
    # The immutable spec pointer is read without entering the render owner or
    # waiting on transport from Blender's depsgraph callback.
    request = RenderRequest(input_usd_path=scene_path)
    native = getattr(runtime, "_native_material_context", None)
    reuse = getattr(resolver, "reuse_scan_from", None)
    if not (native is not None and native[0] is generation and native[1] == scene_path
            and callable(reuse) and reuse(native[2], request)):
        resolver.scan(request)
    # One scene-owned snapshot. Generations/compositions are immutable, while
    # mesh loop validation depends on current Blender data and stays per edit.
    # Failed scans are retried rather than making a transient failure sticky.
    if runtime is not None and getattr(resolver, '_scan_diagnostics', {}).get('available', False):
        runtime._current_edit_context = (generation, scene_path, resolver)
    if runtime is not None and composition is not None and scene_path == composition.composed_scene_path:
        # One bounded ownership snapshot, replaced on each accepted context.
        # It carries verified binding correspondence, never arbitrary edit data.
        runtime._material_presentation_targets = (
            generation.usd_path, composition.composed_scene_path,
            dict(getattr(resolver, "_resolved_material_paths", {})),
        )
    return resolver, getattr(scene, "objects", ())


def _unsupported_light_changes(generation: SceneGeneration, light: Any) -> tuple[Mapping[str, Any], ...]:
    from .light_value_conversion import unsupported_field_changes
    previous = getattr(generation, "unsupported_light_values", {}).get(blender_id(light, "OBJECT"), {})
    return unsupported_field_changes(light, previous)


def _material_graph_change(generation: SceneGeneration, material: Any) -> Mapping[str, Any] | None:
    from .material_graph_fingerprint import material_graph_signature

    identity = blender_id(material, "MATERIAL")
    previous = getattr(generation, "material_graph_fingerprints", {}).get(identity)
    mapping = generation.blender_prim_paths.get(identity)
    if not previous or mapping is None:
        return None
    current = material_graph_signature(material)
    if previous == current:
        return None
    return {"usd_prim_path": mapping.schema_path or mapping.object_path,
            "previous_fingerprint": previous, "current_fingerprint": current}


def _mesh_shading_unchanged(generation: SceneGeneration, mesh: Any) -> bool:
    previous = getattr(generation, "mesh_shading_inputs", {}).get(blender_id(mesh, "OBJECT"))
    return previous is not None and previous == scene_generation._mesh_shading_inputs(mesh)


@dataclass(frozen=True)
class _ShaderUpdateContext:
    generation: SceneGeneration
    materials: tuple[Any, ...]
    mesh_owners: Mapping[int, tuple[Any, ...]]


def current_generation_shader_context(scene: Any) -> _ShaderUpdateContext | None:
    """Bounded main-thread RNA owner index, replaced with the generation.

    File-load teardown drops the scene runtime and every retained RNA owner.
    Graph traversal stays fresh; this indexes owners, never socket values/links.
    """
    uid = int(getattr(scene, "session_uid", 0) or 0)
    runtime = _runtimes.get(uid)
    owner = _owners.get(uid)
    generation = None if owner is None else owner.current_generation
    if (runtime is None or generation is None or uid in _dirty
            or getattr(owner, "pending_generation", None) is not None
            or getattr(generation, "scene_object_count", -1) != len(getattr(scene, "objects", ()))):
        return None
    cached = getattr(runtime, "_shader_update_context", None)
    if cached is not None and cached.generation is generation:
        return cached
    materials = {}; mesh_owners: dict[int, list[Any]] = {}
    for obj in getattr(scene, "objects", ()):
        if str(getattr(obj, "type", "")) != "MESH":
            continue
        data = getattr(obj, "data", None)
        mesh_owners.setdefault(int(getattr(data, "session_uid", 0) or 0), []).append(obj)
        slots = getattr(obj, "material_slots", None)
        bound = (tuple(getattr(slot, "material", None) for slot in slots)
                 if slots is not None else getattr(data, "materials", ()))
        for material in bound:
            if material is not None:
                identity = blender_id(material, "MATERIAL")
                if identity in generation.blender_prim_paths:
                    materials[identity] = material
    result = _ShaderUpdateContext(generation, tuple(materials.values()),
        {uid: tuple(objects) for uid, objects in mesh_owners.items()})
    runtime._shader_update_context = result
    return result


def shader_only_generation_update(
    scene: Any, depsgraph: Any, *, updates: tuple[Any, ...] | None = None,
) -> bool:
    """Prove unchanged membership and mesh controls before avoiding a scene scan.

    Updates that can never be shader-only (a camera move, any transform or
    geometry change) are rejected before the generation context is built:
    that context sizes ``scene.objects``, which Blender counts in O(scene).
    """
    updates = tuple(getattr(depsgraph, "updates", ()) if updates is None else updates)
    if not updates:
        return False
    if not all(_shader_only_candidate(update)[0] for update in updates):
        return False
    context = current_generation_shader_context(scene)
    if context is None:
        return False
    checked: set[int] = set()
    for update in updates:
        candidate, source, kind = _shader_only_candidate(update)
        if kind == "ShaderNodeTree":
            continue
        if not candidate:
            return False
        if kind == "Material":
            if blender_id(source, "MATERIAL") not in context.generation.blender_prim_paths:
                return False
            continue
        objects = ((source,) if kind == "Object" else
                   context.mesh_owners.get(int(getattr(source, "session_uid", 0) or 0), ()))
        if not objects:
            return False
        for obj in objects:
            uid = int(getattr(obj, "session_uid", 0) or 0)
            if uid in checked:
                continue
            if str(getattr(obj, "type", "")) != "MESH" or not _mesh_shading_unchanged(context.generation, obj):
                return False
            checked.add(uid)
    return True


def _shader_only_candidate(update: Any) -> tuple[bool, Any, str]:
    """Context-free part of the shader-only proof: (candidate, source, kind)."""
    source = getattr(update, "id", update)
    source = getattr(source, "original", source)
    kind = str(getattr(getattr(source, "bl_rna", None), "identifier", ""))
    if kind == "ShaderNodeTree":
        return True, source, kind
    candidate = (kind in {"Material", "Mesh", "Object"}
                 and getattr(update, "is_updated_geometry", None) is False
                 and getattr(update, "is_updated_transform", None) is False
                 and getattr(update, "is_updated_shading", None) is True)
    return candidate, source, kind


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
        if mapping is None or not (
            _mapping_owns_target(mapping, edit.usd_prim_path)
            or _accepted_material_presentation_owns_target(scene, generation, mapping, edit)
        ):
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
    for edit, result in zip(admitted, results):
        if result.accepted and result.plan is not None:
            _queue_owned_world_topology(scene, owner, edit, result.plan)
    if not runtime.viewport_ids and any(
        result.accepted for result in results
    ):
        _wake_preparation(uid)
    return results


def _queue_owned_world_topology(scene, owner, edit, plan) -> bool:
    """Reconcile accepted embedded World graph edits by verified owner identity."""
    if not plan.impact.scene_generation_replacement_requested:
        return False
    kind = _canonical_blender_id_kind(str(edit.provenance.get('blender_id_kind', '') or ''))
    generation = owner.current_generation
    if kind != 'WORLD' or generation is None or _edit_source_mapping(generation, edit) is None:
        return False
    identity = BlenderId('WORLD', int(edit.provenance['blender_session_uid']))
    return mark_scene_dirty(scene, {identity}, defer_world_reconciliation=True)


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


def _accepted_material_presentation_owns_target(scene, generation, mapping, edit) -> bool:
    if mapping.blender_id_type != "MATERIAL":
        return False
    runtime = _runtimes.get(_scene_uid(scene))
    snapshot = getattr(runtime, "_material_presentation_targets", None)
    spec = getattr(getattr(getattr(runtime, "ovrtx", None), "controller", None), "_spec", None)
    composition = getattr(spec, "ovrtx_scene_composition", None)
    if snapshot is None or composition is None:
        return False
    if (snapshot[:2] != (generation.usd_path, composition.composed_scene_path)
            or composition.source_scene_path != generation.usd_path):
        return False
    uid = int(edit.provenance.get("blender_session_uid", 0) or 0)
    root = snapshot[2].get(uid, "")
    return bool(root and (edit.usd_prim_path == root or edit.usd_prim_path.startswith(root + "/")))


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
        if _bake_capture_owner is not None:
            raise RuntimeError("physics bake capture owns the runtime")
        runtime = _runtimes.get(uid)
        _require_runtime_reuse(runtime)
        if runtime is None:
            if expected_runtime is not None:
                raise RuntimeError(
                    "viewport attachment attempted to replace the authoring runtime"
                )
            from .ovrtx_session_controller import OvrtxSessionController

            runtime = AuthoringGenerationRuntime(OvrtxSessionController(), owner)
            runtime.scene_uid = uid
            if uid in _baked_playback_paused:
                runtime.scheduler.baked_playback_paused = True
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
    _release_camera_capture_for_viewport(uid)
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
    _release_camera_capture_for_viewport(uid)
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
                # Retry after the next edit, including one made before any redraw.
                _blocked_revisions[uid] = _preparation_revisions.get(uid, 0)
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


def reserve_camera_capture(scene: Any, capture_owner: Any) -> None:
    """Reserve native ownership before preparing a camera batch on Blender."""
    uid = _scene_uid(scene)
    with _preparation_condition:
        if _bake_capture_owner is not None:
            raise RuntimeError("physics bake capture owns the runtime")
        previous = _camera_capture_owners.get(uid)
        if previous is not None and previous is not capture_owner:
            raise RuntimeError("another camera capture owns the scene")
        runtime = _runtimes.get(uid)
        _require_runtime_reuse(runtime)
        if runtime is not None and runtime.viewport_ids:
            raise RuntimeError("Detach rendered viewports before camera capture")
        _camera_capture_owners[uid] = capture_owner
        _preparation_pending.pop(uid, None)
        _preparation_condition.notify_all()


def release_camera_capture(scene_uid: int, capture_owner: Any) -> None:
    """Release only after the capture owner confirmed native teardown."""
    with _preparation_condition:
        if _camera_capture_owners.get(scene_uid) is capture_owner:
            _camera_capture_owners.pop(scene_uid, None)
            _preparation_condition.notify_all()


def activate_for_camera_capture(scene: Any, request: RenderRequest, *,
                                controller: Any, capture_owner: Any) -> OvrtxGenerationAdapter:
    """Transfer detached preparation, then activate without borrowing its RPC lane.

    Runs on the capture owner's native thread. Attached viewport runtimes are
    rejected before any activation. Detached preparation drains under its normal
    transport/prepare locks and remains fail-closed after unconfirmed teardown.
    """
    uid = _scene_uid(scene)
    with _preparation_condition:
        if _camera_capture_owners.get(uid) is not capture_owner:
            raise RuntimeError("camera capture ownership was not reserved")
        runtime = _runtimes.get(uid)
        _require_runtime_reuse(runtime)
        if runtime is not None and runtime.viewport_ids:
            raise RuntimeError("Detach rendered viewports before camera capture")
    if runtime is not None:
        # Drain prior preparation in the established lock order, then release
        # transport before close() joins scheduler threads. The capture lease
        # prevents a queued preparation from starting another native edit.
        with runtime.ovrtx.controller._serialized_transport(), runtime._preparation_lock:
            pass
        if not runtime.close():
            fail_closed_runtime_reuse(runtime)
            raise RuntimeError("Detached viewport teardown was not confirmed")
        with _preparation_condition:
            if _runtimes.get(uid) is runtime:
                _runtimes.pop(uid)
    owner = _owners.get(uid)
    if owner is None:
        raise RuntimeError("current scene generation owner is unavailable")
    generation, predecessor = _generation_for_request(owner, request)
    previous = getattr(capture_owner, 'generation_activation', None)
    adapter = (previous if previous is not None and previous.controller is controller
               else OvrtxGenerationAdapter(controller))
    adapter.update_request(request)
    try:
        revision = owner.retained_revision
        unchanged = (predecessor is None and getattr(capture_owner, 'retained_revision', None) == revision)
        if not adapter.activate(generation,
                retained_values=lambda: owner.retained_values_for(generation),
                skip_unchanged_retained=unchanged):
            raise RuntimeError(adapter.last_error or "camera generation activation failed")
        from . import direct_scene_updates
        adapter.activation_timings['direct_update'] = direct_scene_updates.apply(owner, generation, controller)
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
    capture_owner.generation_activation = adapter
    capture_owner.retained_revision = revision
    return adapter


def _release_camera_capture_for_viewport(uid: int) -> None:
    with _preparation_condition:
        capture = _camera_capture_owners.get(uid)
    if capture is not None:
        if capture.is_busy():
            raise RuntimeError("Camera capture is active; wait or cancel before rendered viewport startup")
        # The callback submits teardown to the retained capture thread. It may
        # not execute native calls on Blender's callback thread.
        if not capture.release_for_viewport():
            raise RuntimeError("Camera capture teardown is draining before viewport startup")
        with _preparation_condition:
            if uid in _camera_capture_owners:
                raise RuntimeError("Camera capture teardown was not confirmed")


def discard_final_render_request(scene: Any, request: RenderRequest) -> bool:
    """Settle only the canceled capture's unaccepted generation before resuming."""
    if not request.current_scene_generation:
        return False
    with _SessionState():
        uid = _scene_uid(scene)
        owner = _owners.get(uid)
        pending = None if owner is None else owner.pending_generation
        if pending is None or pending.usd_path != request.input_usd_path:
            return False
        owner.reject(pending)
        _dirty.setdefault(uid, set()).update(_pending_affected.pop(uid, set()))
        _blocked_reconciliations.pop(uid, None)
        return True


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
    if prepared is None and _bake_capture_owner is None:
        # Final renders share the scene-owned runtime so its worker and native
        # renderer survive between F12 renders, instead of paying SDK
        # verification and renderer creation for every render. A physics bake
        # capture owns runtime creation, so F12 then keeps a private worker.
        prepared = _runtime_for_owner(uid, owner)
    if prepared is not None:
        # The controller's FIFO transport gate is reentrant. Holding it across
        # the complete lease prevents retained edit replay from changing the
        # camera/population between final sample chunks.
        transport_lease = prepared.ovrtx.controller._serialized_transport()
        transport_lease.__enter__()
        try:
            # Bind restoration after older queued owners have finished.
            previous_request = prepared._request
            activate = getattr(prepared, "activate_blocking", prepared.activate)
            activate(
                generation,
                request,
                predecessor=predecessor,
            )
            if predecessor is not None:
                owner.accept(generation)
                _pending_affected.pop(uid, None)
                _blocked_reconciliations.pop(uid, None)
            return PreparedFinalRenderLease(prepared, generation, previous_request, transport_lease)
        except BaseException:
            try:
                if predecessor is not None:
                    owner.reject(generation)
                    _pending_affected.pop(uid, None)
                    _blocked_reconciliations.pop(uid, None)
            finally:
                transport_lease.__exit__(None, None, None)
            raise
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
            interactive_operator_state.clear_playback_lock_when_writable(
                candidate.playback_lock,
                reason="last_viewport_detached",
                still_wanted=lambda uid=uid, candidate=candidate: (
                    _runtimes.get(uid) is candidate and not candidate.viewport_ids
                ),
            )
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

    request_preparation_pause()
    deadline = time.monotonic() + 5.0
    for runtime in tuple(_runtimes.values()):
        if not runtime._preparation_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            fail_closed_runtime_reuse(runtime)
            return False
        runtime._preparation_lock.release()
    return True


def request_preparation_pause() -> None:
    """Close preparation admission immediately without waiting for native work."""
    global _preparation_paused
    with _preparation_condition:
        _preparation_paused = True
        _preparation_condition.notify_all()


def discard_unacknowledged_renderer_intent() -> int:
    """Call only after presentation and preparation owners have drained."""
    return sum(runtime.scheduler.discard_unacknowledged_view_updates()
               for runtime in tuple(_runtimes.values()))


def resume_preparation() -> None:
    global _preparation_paused
    with _preparation_condition:
        _preparation_paused = False
        if _preparation_pending:
            _start_preparation_worker_locked()
            _preparation_condition.notify_all()


def demand_physics_playback(scene: Any, *, apply_pending_anchors: bool = False) -> None:
    """Reconcile a barrier on Blender's thread, then prepare physics off-thread."""

    uid = _scene_uid(scene)
    if _baked_active(scene):
        raise RuntimeError("Deactivate baked Actions before requesting physics playback")
    _require_runtime_reuse(_runtimes.get(uid))
    # Before anchor commits, whose own repair would reidentify the originals.
    repair_duplicate_identities(scene)
    if apply_pending_anchors:
        from .physics_authoring import apply_pending_anchors_for_playback
        applied = apply_pending_anchors_for_playback(scene)
        if applied:
            print("[physics] Applied pending joint anchors for playback: " + ", ".join(applied))
    from .physics_authoring import require_scene_ready
    require_scene_ready(scene)
    generation_for_scene(scene, explicit_physics_restart=True)
    _clear_playback_error(uid)
    owner = _owners[uid]
    if uid not in _runtimes:
        _runtime_for_owner(uid, owner)
    # Keep automatic preparation and frame callbacks gated until every
    # synchronous admission step succeeds. Rejected demand must stay paused.
    _baked_playback_paused.discard(uid)
    _presentation_stopped.discard(uid)
    scheduler = getattr(_runtimes[uid], "scheduler", None)
    if scheduler is not None:
        scheduler.baked_playback_paused = False
    _wake_preparation(uid)


def diagnostics() -> dict[str, Any]:
    return {
        "physics_restart_pending_scene_uids": sorted(_physics_restart_pending),
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
        "playback_hold": playback_hold.diagnostics(),
    }


_presentation_epoch = 0
_presentation_stopped = set()
# Runtime keys stopped by the save in progress, resumed by save_post/_fail.
_save_stopped_presentations: set[int] = set()


def current_presentation_context(scene):
    from .physics_presentation_targets import presentation_targets
    owner = _owners.get(int(getattr(scene, "session_uid", 0) or 0))
    generation = None if owner is None else owner.current_generation
    return (None if generation is None else generation.number, _presentation_epoch,
            () if generation is None else presentation_targets(generation))


def presentation_stopped(scene):
    return _bake_capture_owner is not None or _baked_active(scene) or int(getattr(scene, "session_uid", 0) or 0) in (_presentation_stopped | _baked_playback_paused)


@persistent
def hold_physics_presentation(scene=None, *_unused):
    """Pause holds the displayed sample; save/undo still restore authoring."""
    global _presentation_epoch
    if _bake_capture_owner is not None or _baked_active(scene):
        return
    _presentation_epoch += 1
    uid = int(getattr(scene, "session_uid", 0) or 0)
    for key, runtime in tuple(_runtimes.items()):
        if (not uid or key == uid) and key not in _baked_playback_paused:
            _presentation_stopped.discard(key)
            runtime._wake_presentations()


@persistent
def restore_physics_presentation(scene=None, *_unused):
    global _presentation_epoch
    if _bake_capture_owner is not None:
        return
    _presentation_epoch += 1
    uid = int(getattr(scene, "session_uid", 0) or 0)
    for key, runtime in tuple(_runtimes.items()):
        if not uid or key == uid:
            runtime.playback_lock.clear(reason="physics_presentation_stopped")
            _presentation_stopped.add(key)
    if bpy is not None:
        bpy.context.view_layer.update()


@persistent
def presentation_frame_changed(scene, *_unused):
    if not _baked_active(scene) and int(scene.session_uid) not in _baked_playback_paused:
        _presentation_stopped.discard(int(scene.session_uid))


@persistent
def physics_frame_changed(scene, depsgraph=None, *_unused):
    """Latch animation write-back of physics opinions on a physics-free scene.

    F-curves and drivers write ``ov`` values to the original IDs during frame
    evaluation without an RNA update, and a frame change dispatches no
    ``depsgraph_update_post``. Probe the Objects this frame evaluated whose
    Action or drivers target ``ov`` paths (cached data-path verdicts); latch,
    never capture here. The walk over evaluated IDs is lazy and stops after
    ``_FRAME_PROBE_BUDGET_S`` (at least ``_FRAME_PROBE_MIN`` updates), so a
    heavily animated physics-free scene pays a bounded cost per frame; what it
    does not reach is left to idle discovery, which sees the written value.
    Off the main thread (animation render) RNA is not read: latch, since final
    render admission captures anyway.
    """
    if _bake_capture_owner is not None:
        return
    if threading.current_thread() is not threading.main_thread():
        note_physics_authoring_changed(scene)
        return
    from . import physics_live_kinematics
    if physics_live_kinematics.evaluating() or depsgraph is None:
        return
    try:
        if not _render_only_physics_current(scene):
            return  # unknown, latched, or participant scenes observe conservatively
        from .physics_edit_snapshot import animation_targets_physics, object_participates
        data_materials: dict = {}
        deadline = time.perf_counter() + _FRAME_PROBE_BUDGET_S
        # Lazy single pass over depsgraph.updates (not snapshotted: it may stop early).
        for seen, update in enumerate(getattr(depsgraph, "updates", ()), 1):
            source = getattr(update, "id", update)
            source = getattr(source, "original", source)
            if (str(getattr(getattr(source, "bl_rna", None), "identifier", "")) == "Object"
                    and animation_targets_physics(source)
                    and object_participates(source, data_materials)):
                note_physics_authoring_changed(scene)
                return
            if seen >= _FRAME_PROBE_MIN and time.perf_counter() >= deadline:
                return
    except ReferenceError:
        note_physics_authoring_changed(scene)


_FRAME_PROBE_BUDGET_S = 0.001
_FRAME_PROBE_MIN = 64


@persistent
def save_physics_presentation(scene=None, *_unused):
    if _bake_capture_owner is not None:
        _bake_capture_owner.cancel()
        _bake_capture_owner.close()
    already_stopped = set(_presentation_stopped)
    restore_physics_presentation(scene)
    # The file stores authored state; the viewport resumes after the write.
    _save_stopped_presentations.update(_presentation_stopped - already_stopped)


@persistent
def resume_saved_physics_presentation(_unused=None, *_args):
    """End the stop that ``save_pre`` imposed, after the save or its failure.

    Only runtimes the save itself stopped resume; a stop owned by the stop
    button, baked playback or a capture stays. The engine then re-prepares its
    newest frame, so the viewport presents without waiting for a scene edit.
    """
    global _presentation_epoch
    keys = set(_save_stopped_presentations)
    _save_stopped_presentations.clear()
    if not keys or _bake_capture_owner is not None:
        return
    resumed = set()
    for key in keys:
        runtime = _runtimes.get(key)
        if runtime is None or key in _baked_playback_paused or key not in _presentation_stopped:
            continue
        _presentation_stopped.discard(key)
        resumed.add(key)
    if not resumed:
        return
    _presentation_epoch += 1
    for key in resumed:
        _runtimes[key]._wake_presentations()
    if bpy is not None:
        from . import engine
        engine.resume_physics_presentation(resumed)


@persistent
def presentation_undo_pre(_unused=None):
    from . import engine
    engine.stop_viewport_render_threads_for_file_load()
    close(defer_native=True)


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


_CLOSE_TASK: dict[str, Any] | None = None
_CLOSE_LOCK = threading.Lock()
_MAX_CLOSE_FINALIZERS = 16
# Owners whose cleanup raised, and work directories a closed owner could not
# remove yet. Neither is reachable through _owners; every later close retries
# them without failing on them again (their close already reported failure).
_unclosed_owners: list[Any] = []
_deferred_cleanup_paths: list[Path] = []


def _close_owner(owner: Any) -> Path | None:
    if isinstance(owner, SceneGenerationOwner):
        return owner.close(defer_residue=True)
    owner.close()
    return None


def _retry_unreachable_cleanup() -> None:
    for owner in tuple(_unclosed_owners):
        try:
            residue = _close_owner(owner)
        except Exception:
            continue
        _unclosed_owners.remove(owner)
        if residue is not None:
            _deferred_cleanup_paths.append(residue)
    for path in tuple(_deferred_cleanup_paths):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            continue
        _deferred_cleanup_paths.remove(path)


def _close_scene_owners() -> list[Exception]:
    """Close every owner without a live runtime; return the cleanup errors.

    close() has already cleared the physics snapshots, so an owner leaves
    _owners before its cleanup runs: a failed cleanup must not leave an
    accepted generation reusable without the snapshot it was accepted with.
    """
    _retry_unreachable_cleanup()
    errors: list[Exception] = []
    for uid, owner in tuple(_owners.items()):
        if uid in _runtimes:
            continue
        del _owners[uid]
        try:
            residue = _close_owner(owner)
        except Exception as exc:
            _unclosed_owners.append(owner)
            errors.append(exc)
            continue
        if residue is not None:
            _deferred_cleanup_paths.append(residue)
            print(f"[ovrtx] scene generation cleanup deferred: {residue}", flush=True)
    return errors


def deferred_cleanup_paths() -> tuple[Path, ...]:
    return tuple(_deferred_cleanup_paths)


def _retain_close_finalizer(task: dict[str, Any], callback: Callable[[], Any] | None) -> None:
    if callback is None or callback in task["finalizers"] or callback in task["completed_finalizers"]:
        return
    if len(task["finalizers"]) + len(task["completed_finalizers"]) >= _MAX_CLOSE_FINALIZERS:
        raise RuntimeError("Scene cleanup finalizer capacity exceeded")
    task["finalizers"].append(callback)


def close_status() -> dict[str, Any]:
    return dict(_CLOSE_TASK["result"]) if _CLOSE_TASK is not None else {"status": "open"}


def wait_for_close(timeout: float) -> dict[str, Any]:
    """Wait, bounded, for an in-progress scene close and return its status.

    File load closes the previous scene off the main thread. A final render
    started right after the load waits here instead of refusing; the close's
    own cleanup thread never waits for itself.
    """
    task = _CLOSE_TASK
    if task is not None and task.get("thread") is not threading.current_thread():
        task["completion"].wait(timeout)
    return close_status()


def _require_session_open() -> None:
    status = close_status()["status"]
    if status == "closing":
        from .ovrtx_runtime_client import RuntimeServicesPreparingError
        raise RuntimeServicesPreparingError("Previous scene cleanup is still in progress")
    if status == "failed":
        raise RuntimeError("Previous scene cleanup failed; runtime reuse is disabled")


def close(*, defer_native: bool = False, after_native: Callable[[], Any] | None = None) -> None:
    global _presentation_epoch, _CLOSE_TASK
    with _CLOSE_LOCK:
        if _CLOSE_TASK is not None and not _CLOSE_TASK["completion"].is_set():
            _retain_close_finalizer(_CLOSE_TASK, after_native)
            return
        completion = threading.Event()
        task = {"completion": completion, "result": {"status": "closing"},
                "finalizers": [], "completed_finalizers": []}
        _retain_close_finalizer(task, after_native)
        _CLOSE_TASK = task
    try:
        capture_owner = _bake_capture_owner
        if capture_owner is not None:
            capture_owner.cancel()
        _presentation_epoch += 1
        _presentation_stopped.clear()
        _save_stopped_presentations.clear()
        _baked_playback_paused.clear()
        for future in _baked_drain_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("Scene closed during baked-playback drain"))
        _baked_drain_requests.clear()
        _physics_snapshots.clear()
        _physics_authoring_dirty.clear()
        _physics_view_observations.clear()
        _physics_idle_discovery.clear()
        _physics_restart_pending.clear()
        _physics_initial_bases.clear()
        _viewport_generation_timers.clear()
        _viewport_generation_errors.clear()
        _identity_owners.clear()
        _blocked_revisions.clear()
        from . import physics_live_kinematics
        physics_live_kinematics._state.clear()
        _invalidate_initial_generation()
        for runtime in tuple(_runtimes.values()):
            try:
                runtime.playback_lock.clear(reason="authoring_runtime_shutdown")
            except Exception:
                pass
        _dirty.clear()
        _dirty_provenance.clear()
        from .native_scene_edit_batch import _generation_snapshots
        _generation_snapshots.clear()
        _reconciling.clear()
        _blocked_reconciliations.clear()
        _pending_affected.clear()
        _constructing.clear()
        timers = getattr(getattr(bpy, "app", None), "timers", None)
        registered = getattr(timers, "is_registered", None)
        unregister = getattr(timers, "unregister", None)
        for observer in tuple(_visibility_observers.values()):
            cleanup = getattr(observer, 'close', None)
            if callable(cleanup):
                cleanup()
            if callable(registered) and callable(unregister) and registered(observer):
                unregister(observer)
        _visibility_observers.clear()
        _preparation_pending.clear()
        _preparation_revisions.clear()
        _playback_errors.clear()
        _rejected_unscoped_dirty_requests.clear()
        _accepted_world_dirty_requests.clear()
        _ignored_scene_updates.clear()
        _reconciliation_timers.clear()
    
    except Exception as exc:
        task["result"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        completion.set()
        raise
    # All RNA cleanup above happens on the caller. Retain Python owners until
    # old viewport/preparation work settles, even after unregister removes timers.
    from . import engine
    sidecars = tuple(engine._ENGINE_RUNTIMES.values())
    restart_task = engine._WORKER_RESTART

    def finish() -> None:
        try:
            if restart_task is not None and not restart_task["completion"].wait(engine.VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS):
                raise RuntimeError("Renderer restart did not settle before scene close")
            for sidecar in sidecars:
                pending = sidecar.get("cleanup_completion")
                if pending is not None:
                    if not pending.wait(engine.VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS):
                        raise RuntimeError("Viewport cleanup deadline exceeded")
                    if sidecar.get("stop_confirmed") is not True:
                        raise RuntimeError("Viewport cleanup failed")
            if capture_owner is not None:
                capture_owner.close()
            closed_uids = _stop_preparation_worker(close_runtimes=True)
            for uid in closed_uids:
                del _runtimes[uid]
            cleanup_errors = _close_scene_owners()

            if _runtimes:
                raise RuntimeError("Scene runtime cleanup was not confirmed")
            if cleanup_errors:
                raise cleanup_errors[0]
            errors = []
            while True:
                with _CLOSE_LOCK:
                    if not task["finalizers"]:
                        task["result"] = ({"status": "failed", "errors": errors}
                                          if errors else {"status": "closed"})
                        completion.set()
                        break
                    callback = task["finalizers"].pop(0)
                    task["completed_finalizers"].append(callback)
                try:
                    callback()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            task["result"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            completion.set()

    if defer_native:
        thread = threading.Thread(target=finish, name="ovrtx-scene-cleanup", daemon=True)
        task["thread"] = thread
        try:
            thread.start()
        except Exception as exc:
            task["result"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            completion.set()
    else:
        finish()


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
    close(defer_native=True)
    _load_pre_completed = True


@persistent
def load_post(_unused: Any = None) -> None:
    global _load_post_count, _load_pre_completed
    _load_post_count += 1
    _load_pre_completed = False


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
    owner = _owners.get(uid)
    if (getattr(owner, "current_generation", None) is not None
            and close_status()["status"] not in {"closing", "failed"}):
        # An accepted generation already serves this scene (for example one the
        # viewport built after a close). History steps latch physics authoring
        # and the depsgraph handler owns world changes, so no repair walk or
        # capture is scheduled here.
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

    def generate() -> float | None:
        global _initial_generation_diagnostics
        if revision != _initial_generation_revision:
            return None
        if close_status()["status"] == "closing":
            return 0.1
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
        cursor_waiting = False
        try:
            _require_session_open()
            if callable(cursor_set):
                cursor_set("WAIT")
                cursor_waiting = True
            repair_duplicate_identities(current_scene)
            generation = generation_for_scene(current_scene)
        except RuntimeServicesPreparingError:
            _initial_generation_diagnostics["status"] = "scheduled"
            return 0.1
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
            if cursor_waiting:
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
    if _baked_active(scene) or getattr(getattr(scene, "render", None), "engine", "") != "OVRTX_EXAMPLE":
        return
    _presentation_stopped.discard(int(scene.session_uid))
    try:
        demand_physics_playback(scene, apply_pending_anchors=True)
    except Exception as exc:
        uid = int(getattr(scene, "session_uid", 0) or 0)
        if uid > 0:
            _record_playback_error(uid, f"{type(exc).__name__}: {exc}")


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
    for name, callback in (("animation_playback_post", hold_physics_presentation),
                           ("save_pre", save_physics_presentation),
                           ("save_post", resume_saved_physics_presentation),
                           ("save_post_fail", resume_saved_physics_presentation),
                           ("undo_pre", presentation_undo_pre), ("redo_pre", presentation_undo_pre),
                           ("frame_change_post", presentation_frame_changed),
                           ("frame_change_post", physics_frame_changed)):
        handlers = getattr(module.app.handlers, name, None)
        if handlers is not None and callback not in handlers:
            handlers.append(callback)
    # First, so later frame handlers see the frame playback actually shows.
    frame_pre = getattr(module.app.handlers, "frame_change_pre", None)
    if frame_pre is not None and playback_hold.frame_change_pre not in frame_pre:
        frame_pre.insert(0, playback_hold.frame_change_pre)
    _start_preparation_worker()
    if not module.app.timers.is_registered(_poll_physics_edits):
        module.app.timers.register(_poll_physics_edits, first_interval=0.2, persistent=True)


def unregister_handlers(blender: Any | None = None, *, after_native: Callable[[], Any] | None = None) -> None:
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
    for name, callback in (("animation_playback_post", hold_physics_presentation),
                           ("save_pre", save_physics_presentation),
                           ("save_post", resume_saved_physics_presentation),
                           ("save_post_fail", resume_saved_physics_presentation),
                           ("undo_pre", presentation_undo_pre), ("redo_pre", presentation_undo_pre),
                           ("frame_change_post", presentation_frame_changed),
                           ("frame_change_post", physics_frame_changed)):
        handlers = getattr(module.app.handlers, name, None)
        if handlers is not None and callback in handlers:
            handlers.remove(callback)
    frame_pre = getattr(module.app.handlers, "frame_change_pre", None)
    if frame_pre is not None and playback_hold.frame_change_pre in frame_pre:
        frame_pre.remove(playback_hold.frame_change_pre)
    playback_hold.reset()
    if module.app.timers.is_registered(_poll_physics_edits):
        module.app.timers.unregister(_poll_physics_edits)
    close(defer_native=True, after_native=after_native)


def _wake_preparation(uid: int) -> None:
    global _preparation_thread
    if close_status()["status"] in {"closing", "failed"}:
        return
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


# The preparation thread closes every reusable runtime as it stops. Closing a
# runtime stops its native worker, which may wait 30 s for EOF-driven teardown
# plus terminate/kill fallbacks (direct_runtime_session._stop_process). The
# join must outlast those stops, or a healthy but slow worker stop is reported
# as unconfirmed cleanup and disables runtime reuse.
_PREPARATION_JOIN_SECONDS = 5.0
_RUNTIME_CLOSE_JOIN_SECONDS = 45.0


def _preparation_join_timeout() -> float:
    closing = sum(1 for runtime in tuple(_runtimes.values())
                  if not getattr(runtime, "reuse_blocked", False))
    return _PREPARATION_JOIN_SECONDS + _RUNTIME_CLOSE_JOIN_SECONDS * closing


def _stop_preparation_worker(*, close_runtimes: bool = False) -> set[int]:
    global _preparation_thread, _preparation_stopping
    with _preparation_condition:
        if close_runtimes:
            _start_preparation_worker_locked()
        thread = _preparation_thread
        _preparation_stopping = True
        _preparation_condition.notify_all()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=_preparation_join_timeout())
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


@performance_profile.profiled_thread("authoring-preparation")
@performance_profile.timed("background.authoring_preparation")
def _preparation_loop() -> None:
    while True:
        with _preparation_condition:
            _preparation_condition.wait_for(
                lambda: _preparation_stopping
                or bool(_baked_drain_requests)
                or (not _preparation_paused and bool(_preparation_pending))
            )
            if _preparation_stopping:
                runtimes = tuple(_runtimes.items())
                break
            if _baked_drain_requests:
                uid, future = next(iter(_baked_drain_requests.items()))
                del _baked_drain_requests[uid]
            else:
                future = None
            if future is not None:
                revision = 0
            else:
                uid, revision = next(iter(_preparation_pending.items()))
                del _preparation_pending[uid]
        runtime = _runtimes.get(uid)
        if future is not None:
            try:
                if runtime is not None:
                    held = []
                    try:
                        for lock in (runtime._preparation_lock, runtime.scheduler._tick_lock):
                            if not lock.acquire(timeout=5):
                                raise RuntimeError("Physics ownership drain exceeded its deadline")
                            held.append(lock)
                        status = runtime.ovphysx.deactivate()
                        if status not in {"stopped", "not_found"}:
                            runtime.reuse_blocked = True
                            raise RuntimeError("Physics teardown failed during baked activation")
                        runtime.scheduler._shutdown_controller(reset_timeline=True, reset_token=False)
                    finally:
                        for lock in reversed(held):
                            lock.release()
                if not future.done():
                    future.set_result(None)
            except Exception as exc:
                if runtime is not None:
                    runtime.reuse_blocked = True
                if not future.done():
                    future.set_exception(exc)
            continue
        owner = _owners.get(uid)
        generation = None if owner is None else owner.current_generation
        if (
            uid in _camera_capture_owners
            or runtime is None
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
            # A viewport ensure holds the transport before activate_blocking
            # takes the preparation lock. Use that order in the background
            # worker too, including physics creation and retained-value replay.
            with controller._serialized_transport(), runtime._preparation_lock:
                with _preparation_condition:
                    # Ownership may change while this worker waits for transport.
                    # Never initialize a stale/closing scene or race a viewport's
                    # activation of the same shared physics adapter.
                    if (
                        _preparation_stopping
                        or uid in _camera_capture_owners
                        or _runtimes.get(uid) is not runtime
                        or _owners.get(uid) is not owner
                        or owner.current_generation is not generation
                        or getattr(owner, "pending_generation", None) is not None
                        or runtime.viewport_ids
                        or uid in _dirty
                        or getattr(runtime, "lifecycle_status", "open") != "open"
                        or getattr(runtime, "reuse_blocked", False)
                    ):
                        continue
                    if _preparation_paused:
                        _preparation_pending[uid] = max(
                            revision, _preparation_pending.get(uid, 0)
                        )
                        continue
                if (
                    generation_requires_physics(generation)
                    and not getattr(runtime.scheduler, "baked_playback_paused", False)
                    and runtime.ovphysx.active_generation != generation.number
                    and not runtime.ovphysx.activate(
                        generation,
                        initial_conditions=owner.retained_values_for(generation)[2],
                    )
                ):
                    raise RuntimeError(
                        runtime.ovphysx.last_error or "OVPhysX preparation failed"
                    )
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
