# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal runtime scheduler for interactive OVRTX + OVPhysX playback."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import threading
import time
from typing import Any, Callable, Mapping

from .interactive_edit_planner import EditStatus, EditIntent
from .ovphysx_stage import (
    OvphysxStageController,
    OvphysxStageResult,
    OvphysxStageStatus,
)
from . import ovphysx_simulation
from .ovrtx_value_updates import (
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxUpdatePort,
    OvrtxValueUpdateResult,
)
from .ovphysx_to_ovrtx import translate_values
from .sim_update_stream import SimUpdateStream
from .view_update_stream import (
    ViewUpdateStream,
    combine_update_results,
    has_non_render_setting_failure,
    render_setting_write_rejection,
)
from .shared_stage_config import InteractiveSharedStageConfig
from .shared_stage_composition import BodyPose


class RuntimeTickStatus(str, Enum):
    NOT_ENABLED = "not_enabled"
    INITIAL = "initial"
    STEPPED = "stepped"
    PLAYBACK_ADVANCED = "playback_advanced"
    REUSED_LATEST = "reused_latest"
    COMPLETED = "completed"
    BUSY = "busy"
    FAILED = "failed"
    NOOP = "noop"


@dataclass(frozen=True)
class RuntimeTickRequest:
    input_usd_path: str
    now_ns: int | None = None
    timeline_controls_enabled: bool = False
    timeline_playing: bool = False
    timeline_frame: int = 1
    timeline_start: int = 1
    timeline_end: int = 1
    simulation_reset_token: int = 0


@dataclass(frozen=True)
class RuntimeTickResult:
    status: RuntimeTickStatus
    enabled: bool
    timeline_reset: bool = False
    stage_changed: bool = False
    values_written: bool = False
    should_reset_refinement: bool = False
    should_request_redraw: bool = False
    step_count: int = 0
    simulation_time_ns: int = 0
    generation: int = 0
    presentation_revision: int = 0
    applied_revision: int = 0
    physics_pose_set: tuple[BodyPose, ...] = ()
    complete_pose_projected: bool | None = None
    skipped_reason: str = ""
    #: Details of a rejected live render-setting write (task01-04 fallback), or
    #: ``None``. A worker that rejects a runtime render-setting write must not
    #: fail the tick (that would kill the viewport); the rejection is surfaced
    #: here so the render loop folds the RTPT values back into the composition
    #: digest and re-keys the session instead.
    render_setting_rejected: Mapping[str, Any] | None = None
    update: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditSubmissionResult:
    status: EditStatus
    reason: str
    physics_generation_reset: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status in {EditStatus.QUEUED, EditStatus.APPLIED}


def _sanitize_render_setting_failure(
    update: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """Strip a rejected live render-setting write of its fatal ``failed`` flag.

    A worker that rejects a runtime render-setting write must never kill the
    viewport (render-quality-color-controls task01-04 fallback). The render
    loop instead folds the RTPT values back into the composition digest and
    re-keys the session. Here the render-setting rejection is de-fataled
    (unless another lane also failed, which stays a genuine failure) and the
    rejection details are stamped under ``render_setting_rejected`` so the loop
    can trigger the re-key and the edit record can report ``applied_via: rekey``.
    """

    if not update:
        return (dict(update) if update else {}), None
    rejection = render_setting_write_rejection(update)
    if rejection is None:
        return dict(update), None
    sanitized = dict(update)
    if not has_non_render_setting_failure(update):
        sanitized["failed"] = False
        sanitized["skipped_reason"] = ""
    sanitized["render_setting_rejected"] = dict(rejection)
    return sanitized, rejection


def _extract_render_setting_rejected(
    update: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Recover the render-setting rejection marker from a combined update."""

    if not isinstance(update, Mapping):
        return None
    marker = update.get("render_setting_rejected")
    if isinstance(marker, Mapping):
        return marker
    for lane in update.get("updates", ()) or ():
        if isinstance(lane, Mapping) and isinstance(
            lane.get("render_setting_rejected"), Mapping
        ):
            return lane["render_setting_rejected"]
    return None


class RuntimeScheduler:
    """Owns the demo runtime loop policy between Blender, OVPhysX, and OVRTX."""

    def __init__(
        self,
        *,
        config_factory: Callable[[str], InteractiveSharedStageConfig] = InteractiveSharedStageConfig.from_env,
        controller_factory: Callable[[InteractiveSharedStageConfig], OvphysxStageController]
        | None = None,
        controller_provider: Callable[[], OvphysxStageController | None] | None = None,
        controller_reset: Callable[[], bool] | None = None,
        ovrtx_transform_sink: Callable[[tuple[Any, ...]], None] | None = None,
        ovrtx_attribute_sink: Callable[[tuple[Any, ...]], None] | None = None,
        ovphysx_initial_condition_sink: Callable[[tuple[BodyPose, ...]], None] | None = None,
    ) -> None:
        self._config_factory = config_factory
        self._controller_factory = controller_factory or OvphysxStageController
        self._controller_provider = controller_provider
        self._controller_reset = controller_reset
        self._controller: OvphysxStageController | None = None
        self._owns_controller = False
        self._simulation_spec: ovphysx_simulation.OvphysxSimulationSpec | None = None
        self._simulation_reuse_decision: dict[str, Any] = {}
        self._last_timeline_frame: int | None = None
        self._reset_token = 0
        self._next_update_monotonic_ns = 0
        self._view_updates = ViewUpdateStream(
            transform_sink=(
                (lambda values: ovrtx_transform_sink(tuple(values)))
                if ovrtx_transform_sink is not None else None
            ),
            attribute_sink=(
                (lambda values: ovrtx_attribute_sink(tuple(values)))
                if ovrtx_attribute_sink is not None else None
            ),
        )
        self._sim_updates = SimUpdateStream(
            value_sink=(
                (lambda values: ovphysx_initial_condition_sink(tuple(values)))
                if ovphysx_initial_condition_sink is not None else None
            ),
            retain_values=ovphysx_initial_condition_sink is None,
        )
        self._last_pose_projection: dict[str, Any] = {}
        self._pose_projection_count = 0
        self._pose_projection_failure_count = 0
        self._wake_hook: Callable[[], None] | None = None
        self._presentation_revision = 0
        self._applied_revision = 0
        self._presentation_revision_lock = threading.Lock()
        # one lock per scheduler is the whole concurrency policy;
        # split lanes only if shared-runtime tick contention is measured.
        self._tick_lock = threading.Lock()

    @property
    def has_pending_view_updates(self) -> bool:
        """True while queued view-authoritative edits await the next tick.

        Park-condition input for the latest-view render loop (task02-03):
        the loop must not park while value updates are pending application.
        """

        return self._view_updates.has_pending

    @property
    def has_pending_sim_updates(self) -> bool:
        """True while queued sim-authoritative edits await the next tick.

        Due-work input for the latest-view render loop (task02-07): an
        edit-submission wake with a pending sim edit must run one tick so
        the initial-condition value applies (the tick drains sim pendings
        whenever physics is enabled).
        """

        return self._sim_updates.has_pending

    def pending_view_targets(
        self,
    ) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
        return self._view_updates.pending_targets()

    def set_edit_wake_hook(self, hook: Callable[[], None] | None) -> None:
        """Install the render-loop wake hook for edits and pose publications.

        Fires after every queued edit and every complete physics pose
        publication so playback never depends on a separate scheduler tick.
        """

        self._wake_hook = hook
        wake = self._wake_for_presentation if hook is not None else None
        self._view_updates.set_wake_hook(wake)
        self._sim_updates.set_wake_hook(wake)
        if self._controller is not None:
            self._controller.set_pose_publication_wake_hook(wake)

    @property
    def presentation_revision(self) -> int:
        """Newest edit or pose publication every attached view must present."""

        with self._presentation_revision_lock:
            return self._presentation_revision

    @property
    def applied_revision(self) -> int:
        """Newest applied content change every attached view must reacquire."""

        with self._presentation_revision_lock:
            return self._applied_revision

    def _wake_for_presentation(self) -> None:
        with self._presentation_revision_lock:
            self._presentation_revision += 1
        wake_hook = self._wake_hook
        if wake_hook is not None:
            wake_hook()

    def note_applied_content(self) -> None:
        """Publish a successful out-of-band session/content mutation."""

        with self._presentation_revision_lock:
            self._applied_revision += 1
        wake_hook = self._wake_hook
        if wake_hook is not None:
            wake_hook()

    def tick_viewport(
        self,
        request: RuntimeTickRequest,
        *,
        ovrtx_updates: OvrtxUpdatePort,
        project_complete_pose: bool = False,
    ) -> RuntimeTickResult:
        with self._tick_lock:
            result = self._tick_viewport(
                request,
                ovrtx_updates=ovrtx_updates,
                project_complete_pose=project_complete_pose,
            )
            if getattr(result, "should_reset_refinement", False):
                self.note_applied_content()
            return (
                replace(
                    result,
                    presentation_revision=self.presentation_revision,
                    applied_revision=self.applied_revision,
                )
                if isinstance(result, RuntimeTickResult)
                else result
            )

    def _tick_viewport(
        self,
        request: RuntimeTickRequest,
        *,
        ovrtx_updates: OvrtxUpdatePort,
        project_complete_pose: bool = False,
    ) -> RuntimeTickResult:
        if self._controller_provider is not None:
            borrowed = self._controller_provider()
            if borrowed is None:
                update_result, render_setting_rejected = _sanitize_render_setting_failure(
                    self._view_updates.apply_pending(ovrtx_updates)
                )
                self._shutdown_controller(reset_timeline=True, reset_token=False)
                return RuntimeTickResult(
                    status=RuntimeTickStatus.FAILED
                    if update_result.get("failed", False)
                    else RuntimeTickStatus.NOT_ENABLED,
                    enabled=False,
                    values_written=bool(update_result.get("values_written", False)),
                    should_reset_refinement=bool(update_result.get("values_written", False)),
                    skipped_reason=str(update_result.get("skipped_reason", "")),
                    render_setting_rejected=render_setting_rejected,
                    update=_merge_update({}, update_result),
                )
            config = borrowed.config
        else:
            borrowed = None
            config = self._config_factory(request.input_usd_path)
        if not config.enabled:
            update_result, render_setting_rejected = _sanitize_render_setting_failure(
                self._view_updates.apply_pending(ovrtx_updates)
            )
            self._shutdown_controller(reset_timeline=True, reset_token=False)
            return RuntimeTickResult(
                status=RuntimeTickStatus.FAILED
                if update_result.get("failed", False)
                else RuntimeTickStatus.NOT_ENABLED,
                enabled=False,
                values_written=bool(update_result.get("values_written", False)),
                should_reset_refinement=bool(update_result.get("values_written", False)),
                skipped_reason=str(update_result.get("skipped_reason", "")),
                render_setting_rejected=render_setting_rejected,
                update=_merge_update({}, update_result),
            )

        timeline_reset = _timeline_should_reset(request, self._last_timeline_frame, self._reset_token)
        if borrowed is not None and timeline_reset and self._controller_reset is not None:
            if not self._controller_reset():
                self._reset_token = int(request.simulation_reset_token)
                return RuntimeTickResult(
                    status=RuntimeTickStatus.FAILED,
                    enabled=True,
                    timeline_reset=True,
                    skipped_reason="ovphysx_reset_failed",
                )
            borrowed = self._controller_provider()
            if borrowed is None:
                self._reset_token = int(request.simulation_reset_token)
                return RuntimeTickResult(
                    status=RuntimeTickStatus.FAILED,
                    enabled=True,
                    timeline_reset=True,
                    skipped_reason="ovphysx_reset_controller_unavailable",
                )
            config = borrowed.config
        controller = (
            self._borrow_controller(borrowed)
            if borrowed is not None
            else self._ensure_controller(config, explicit_reset=timeline_reset)
        )
        if timeline_reset:
            self._reset_token = int(request.simulation_reset_token)

        now_ns = time.monotonic_ns() if request.now_ns is None else int(request.now_ns)
        max_steps = _timeline_max_steps(request, controller.config.steps_per_update, controller.config.max_steps)
        update_results: list[Mapping[str, Any]] = []
        stage_result: OvphysxStageResult | None = None
        complete_pose_projected: bool | None = None
        if not controller.started and (
            project_complete_pose
            or self._sim_updates.has_pending
            or self._sim_updates.has_values
        ):
            starting_values = self._sim_updates.values_for_controller_start(controller_started=False)
            stage_result = controller.tick(
                max_steps=max_steps,
                initial_condition_values=starting_values,
            )
            if stage_result.status not in {OvphysxStageStatus.BUSY, OvphysxStageStatus.FAILED}:
                self._next_update_monotonic_ns = now_ns + controller.config.update_interval_ns
            initial_condition_update = self._sim_updates.record_controller_start(stage_result, controller)
            if initial_condition_update:
                update_results.append(initial_condition_update)
            if self._sim_updates.last_controller_result is not None:
                stage_result = self._sim_updates.last_controller_result
                self._next_update_monotonic_ns = 0
        if stage_result is not None and stage_result.status in {
            OvphysxStageStatus.BUSY,
            OvphysxStageStatus.FAILED,
        }:
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.INITIAL,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=combine_update_results(update_results),
            )
        sim_update = self._sim_updates.apply_pending(controller)
        if sim_update:
            update_results.append(sim_update)
            if self._sim_updates.last_controller_result is not None:
                stage_result = self._sim_updates.last_controller_result
            if sim_update.get("failed", False):
                return self._result_from_stage(
                    stage_result,
                    RuntimeTickStatus.BUSY
                    if sim_update.get("retryable", False)
                    else RuntimeTickStatus.FAILED,
                    request,
                    timeline_reset=timeline_reset,
                    controller=controller,
                    update_result=combine_update_results(update_results),
                )
            self._next_update_monotonic_ns = 0
        if project_complete_pose:
            poses = (
                stage_result.pose_set
                if stage_result is not None and stage_result.pose_set
                else controller.physics_pose_set(controller.current_simulation_time_ns)
            )
            projection_update = self._apply_pose_projection(
                poses, ovrtx_updates, complete=True
            )
            update_results.append(projection_update)
            complete_pose_projected = not bool(projection_update.get("failed", False))
            if projection_update.get("failed", False):
                return self._result_from_stage(
                    stage_result,
                    RuntimeTickStatus.FAILED,
                    request,
                    timeline_reset=timeline_reset,
                    controller=controller,
                    update_result=combine_update_results(update_results),
                    complete_pose_projected=False,
                )
        elif stage_result is not None and stage_result.dirty_paths:
            projection_update = self._project_dirty_poses(
                stage_result, ovrtx_updates
            )
            update_results.append(projection_update)
            if projection_update.get("failed", False):
                return self._result_from_stage(
                    stage_result,
                    RuntimeTickStatus.FAILED,
                    request,
                    timeline_reset=timeline_reset,
                    controller=controller,
                    update_result=combine_update_results(update_results),
                )
        view_update, _render_setting_rejected = _sanitize_render_setting_failure(
            self._view_updates.apply_pending(ovrtx_updates)
        )
        if view_update:
            update_results.append(view_update)
        update_result = combine_update_results(update_results)
        if update_result.get("failed", False):
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.FAILED,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=update_result,
                complete_pose_projected=complete_pose_projected,
            )
        if complete_pose_projected:
            self._commit_timeline_cursor(request)
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.NOOP,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=update_result,
                complete_pose_projected=True,
            )
        last_frame = self._last_timeline_frame
        should_step = _timeline_should_step(request, last_frame, controller.started)
        if not should_step:
            if request.timeline_controls_enabled and not request.timeline_playing:
                controller.stop_async_playback(wait=False, invalidate=True)
            self._commit_timeline_cursor(request)
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.NOOP,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=update_result,
                complete_pose_projected=complete_pose_projected,
            )

        if request.timeline_controls_enabled and request.timeline_playing:
            was_started = controller.started
            stage_result = controller.publish_latest_pose(max_steps=max_steps)
            projection_update = self._project_dirty_poses(
                stage_result, ovrtx_updates
            )
            if projection_update:
                update_results.append(projection_update)
            if stage_result.status != OvphysxStageStatus.BUSY:
                self._commit_timeline_cursor(request)
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.INITIAL
                if not was_started
                else RuntimeTickStatus.PLAYBACK_ADVANCED
                if stage_result.dirty_paths
                else RuntimeTickStatus.REUSED_LATEST,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=combine_update_results(update_results),
                complete_pose_projected=complete_pose_projected,
            )

        force = request.timeline_controls_enabled and not request.timeline_playing
        if request.timeline_controls_enabled and last_frame is not None and int(request.timeline_frame) > int(last_frame):
            force = True
        controller.stop_async_playback(wait=True, invalidate=True)
        controller.adopt_async_producer_position()
        if not force and now_ns < self._next_update_monotonic_ns:
            self._commit_timeline_cursor(request)
            return self._result_from_stage(
                stage_result,
                RuntimeTickStatus.NOOP,
                request,
                timeline_reset=timeline_reset,
                controller=controller,
                update_result=_merge_update(
                    combine_update_results(update_results),
                    {"skipped_reason": "not_due"},
                ),
                complete_pose_projected=complete_pose_projected,
            )
        was_started = controller.started
        stage_result = controller.tick(max_steps=max_steps)
        controller.sync_async_producer_position()
        projection_update = self._project_dirty_poses(
            stage_result, ovrtx_updates
        )
        if projection_update:
            update_results.append(projection_update)
        if stage_result.status != OvphysxStageStatus.BUSY:
            self._commit_timeline_cursor(request)
        if stage_result.status in {OvphysxStageStatus.OK, OvphysxStageStatus.COMPLETED}:
            self._next_update_monotonic_ns = now_ns + controller.config.update_interval_ns
        return self._result_from_stage(
            stage_result,
            RuntimeTickStatus.STEPPED if was_started else RuntimeTickStatus.INITIAL,
            request,
            timeline_reset=timeline_reset,
            controller=controller,
            update_result=combine_update_results(update_results),
            complete_pose_projected=complete_pose_projected,
        )

    def replay_retained_values(
        self,
        ovrtx_updates: OvrtxUpdatePort,
        transforms: tuple[OvrtxTransformValue, ...],
        attributes: tuple[OvrtxAttributeValue, ...],
    ) -> RuntimeTickResult:
        """Apply bounded scene-owned desired state during activation."""

        with self._tick_lock:
            try:
                if transforms:
                    result = ovrtx_updates.update_transforms(transforms)
                    if result.updated_count != len(transforms):
                        raise RuntimeError("OVRTX transform replay was incomplete")
                if attributes:
                    result = ovrtx_updates.update_attribute_values(attributes)
                    if result.updated_count != len(attributes):
                        raise RuntimeError("OVRTX attribute replay was incomplete")
            except Exception as exc:
                return RuntimeTickResult(
                    status=RuntimeTickStatus.FAILED,
                    enabled=True,
                    skipped_reason=(
                        f"retained_value_replay_failed:{type(exc).__name__}:{exc}"
                    ),
                )
            return RuntimeTickResult(
                status=RuntimeTickStatus.NOOP,
                enabled=True,
                values_written=bool(transforms or attributes),
                should_reset_refinement=bool(transforms or attributes),
            )

    def apply_pending_view_values(
        self,
        ovrtx_updates: OvrtxUpdatePort,
    ) -> RuntimeTickResult:
        """Drain queued desired view state before activation becomes ready."""

        with self._tick_lock:
            update, rejected = _sanitize_render_setting_failure(
                self._view_updates.apply_pending(ovrtx_updates)
            )
        failed = bool(update.get("failed", False))
        return RuntimeTickResult(
            status=RuntimeTickStatus.FAILED if failed else RuntimeTickStatus.NOOP,
            enabled=True,
            values_written=bool(update.get("values_written", False)),
            should_reset_refinement=bool(update.get("values_written", False)),
            skipped_reason=str(update.get("skipped_reason", "")),
            render_setting_rejected=rejected,
            update=update,
        )

    def apply_pending_sim_values(self, controller: Any) -> RuntimeTickResult:
        """Drain bounded SIM desired state before activation becomes ready."""

        with self._tick_lock:
            update = self._sim_updates.apply_pending(controller)
        failed = bool(update.get("failed", False))
        return RuntimeTickResult(
            status=(
                RuntimeTickStatus.BUSY
                if update.get("retryable", False)
                else RuntimeTickStatus.FAILED
                if failed
                else RuntimeTickStatus.NOOP
            ),
            enabled=True,
            values_written=bool(update.get("values_written", False)),
            should_reset_refinement=bool(update.get("values_written", False)),
            skipped_reason=str(update.get("skipped_reason", "")),
            update=update,
        )

    def submit_edit(self, intent: EditIntent) -> EditSubmissionResult:
        return self.submit_edits((intent,))[0]

    def submit_edits(
        self,
        intents: tuple[EditIntent, ...],
    ) -> tuple[EditSubmissionResult, ...]:
        """Validate then queue one complete edit group without partial effects."""

        if not intents:
            return ()
        streams = tuple(
            self._sim_updates
            if isinstance(intent, EditIntent) and intent.data_authority.value == "sim"
            else self._view_updates
            for intent in intents
        )
        invalid = next(
            (
                (intent, stream)
                for intent, stream in zip(intents, streams)
                if not isinstance(intent, EditIntent) or not stream.supports(intent)
            ),
            None,
        )
        if invalid is not None:
            intent, stream = invalid
            diagnostics = (
                stream.unsupported_result(intent)
                if isinstance(intent, EditIntent)
                else {}
            )
            reason = (
                str(diagnostics.get("skipped_reason", ""))
                or "unsupported:unresolved_edit"
            )
            return tuple(
                EditSubmissionResult(
                    status=EditStatus.UNSUPPORTED,
                    reason=reason,
                    diagnostics=diagnostics,
                )
                for _intent in intents
            )

        with self._tick_lock:
            diagnostics = tuple(
                stream.queue(intent, notify=False)
                for intent, stream in zip(intents, streams)
            )
            self._wake_for_presentation()
        return tuple(
            EditSubmissionResult(
                status=EditStatus.QUEUED,
                reason="queued",
                diagnostics=item,
            )
            for item in diagnostics
        )

    def diagnostics(self) -> Mapping[str, Any]:
        update_result = combine_update_results(
            [self._sim_updates.last_result, self._view_updates.last_result]
        )
        diagnostics = self._composition_diagnostics()
        diagnostics["last_edit_update"] = update_result
        diagnostics["sim_updates"] = self._sim_updates.diagnostics()
        diagnostics["ovphysx_simulation_reuse"] = dict(self._simulation_reuse_decision)
        diagnostics["last_pose_projection_application"] = dict(self._last_pose_projection)
        diagnostics["pose_projection_application_count"] = self._pose_projection_count
        diagnostics["pose_projection_application_failure_count"] = self._pose_projection_failure_count
        if self._controller is not None:
            diagnostics["physics_create_diagnostics"] = dict(
                self._controller.create_diagnostics or {}
            )
        return diagnostics

    def _composition_diagnostics(self) -> dict[str, Any]:
        if self._controller is None:
            return {"enabled": False}
        return self._controller.diagnostics()

    def shutdown(self) -> None:
        self._shutdown_controller(reset_timeline=True, reset_token=True)

    def _ensure_controller(
        self,
        config: InteractiveSharedStageConfig,
        *,
        explicit_reset: bool = False,
    ) -> OvphysxStageController:
        desired_spec = ovphysx_simulation.prepare(config)
        if self._controller is not None and self._simulation_spec is not None:
            decision = ovphysx_simulation.reuse_decision(
                self._simulation_spec,
                desired_spec,
                explicit_reset=explicit_reset,
                terminal_failure=self._controller.failed,
            )
            self._simulation_reuse_decision = {
                "reuse": decision.reuse,
                "reason": decision.reason,
            }
            if decision.reuse:
                self._controller.set_pose_publication_wake_hook(
                    self._wake_for_presentation if self._wake_hook is not None else None
                )
                return self._controller
        self._shutdown_controller(reset_timeline=True, reset_token=False)
        self._controller = self._controller_factory(config)
        self._controller.set_pose_publication_wake_hook(
            self._wake_for_presentation if self._wake_hook is not None else None
        )
        self._owns_controller = True
        self._simulation_spec = desired_spec
        if not self._simulation_reuse_decision:
            self._simulation_reuse_decision = {"reuse": False, "reason": "no_active_simulation"}
        return self._controller

    def _borrow_controller(self, controller: OvphysxStageController) -> OvphysxStageController:
        if self._controller is controller:
            return controller
        self._shutdown_controller(reset_timeline=True, reset_token=False)
        self._controller = controller
        self._controller.set_pose_publication_wake_hook(
            self._wake_for_presentation if self._wake_hook is not None else None
        )
        self._owns_controller = False
        self._simulation_spec = ovphysx_simulation.prepare(controller.config)
        self._simulation_reuse_decision = {"reuse": False, "reason": "authored_generation_changed"}
        return controller

    def _commit_timeline_cursor(self, request: RuntimeTickRequest) -> None:
        self._last_timeline_frame = int(request.timeline_frame)
        self._reset_token = int(request.simulation_reset_token)

    def _shutdown_controller(self, *, reset_timeline: bool, reset_token: bool) -> None:
        controller = self._controller
        owns_controller = self._owns_controller
        if controller is not None:
            controller.set_pose_publication_wake_hook(None)
        self._controller = None
        self._owns_controller = False
        self._simulation_spec = None
        self._next_update_monotonic_ns = 0
        if reset_timeline:
            self._last_timeline_frame = None
        if reset_token:
            self._reset_token = 0
        if controller is None or not owns_controller:
            return
        try:
            controller.shutdown()
        except Exception:
            pass


    def _project_dirty_poses(
        self,
        result: OvphysxStageResult,
        ovrtx_updates: OvrtxUpdatePort,
    ) -> Mapping[str, Any]:
        if not result.dirty_paths:
            return {}
        poses_by_path = {pose.prim_path: pose for pose in result.pose_set}
        poses = tuple(poses_by_path[path] for path in result.dirty_paths)
        return self._apply_pose_projection(poses, ovrtx_updates, complete=False)

    def _apply_pose_projection(
        self,
        poses: tuple[BodyPose, ...],
        ovrtx_updates: OvrtxUpdatePort,
        *,
        complete: bool,
    ) -> Mapping[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            if self._controller is None:
                raise RuntimeError("OVPhysX stage controller is unavailable")
            values = translate_values(poses, self._controller.config.body_scale)
            result = ovrtx_updates.update_transforms(values)
            error = _ovrtx_application_error(result, len(values))
            diagnostics = {
                **dict(result.diagnostics),
                "updated_count": result.updated_count,
                "pending_simulation_time_ns": result.pending_simulation_time_ns,
            }
        except Exception as exc:
            values = []
            diagnostics = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            error = str(diagnostics["error"])
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        failed = bool(error)
        if failed:
            diagnostics = {**diagnostics, "status": "error", "error": error}
            self._pose_projection_failure_count += 1
        else:
            self._pose_projection_count += 1
        application = {
            "reason": "complete_pose_projection" if complete else "incremental_pose_projection",
            "values_written": not failed and bool(values),
            "failed": failed,
            "skipped_reason": "pose_projection_application_error" if failed else "",
            "value_paths": [pose.prim_path for pose in poses] if not failed else [],
            "value_count": len(values) if not failed else 0,
            "application_ms": elapsed_ms,
            "result": diagnostics,
        }
        self._last_pose_projection = dict(application)
        return application

    def _result_from_stage(
        self,
        stage_result: OvphysxStageResult | None,
        ok_status: RuntimeTickStatus,
        request: RuntimeTickRequest,
        *,
        timeline_reset: bool,
        controller: OvphysxStageController,
        update_result: Mapping[str, Any] | None = None,
        complete_pose_projected: bool | None = None,
    ) -> RuntimeTickResult:
        if update_result is None:
            update_result = {}
        resolved_status = _runtime_status(stage_result, ok_status)
        if update_result.get("failed", False):
            resolved_status = RuntimeTickStatus.FAILED
        stage_changed = bool(stage_result and stage_result.dirty_paths)
        values_written = bool(update_result.get("values_written", False))
        should_request_redraw = (
            not controller.completed
            and (not request.timeline_controls_enabled or request.timeline_playing)
        )
        step_count = stage_result.step_count if stage_result else controller.step_count
        simulation_time_ns = (
            stage_result.simulation_time_ns
            if stage_result
            else controller.current_simulation_time_ns
        )
        generation = stage_result.generation if stage_result else controller.composition_generation
        pose_set = (
            stage_result.pose_set
            if stage_result is not None
            else controller.physics_pose_set(simulation_time_ns)
        )
        stage_update = _stage_result_mapping(stage_result)
        return RuntimeTickResult(
            status=resolved_status,
            enabled=True,
            timeline_reset=timeline_reset,
            stage_changed=stage_changed,
            values_written=values_written,
            should_reset_refinement=stage_changed or values_written,
            should_request_redraw=should_request_redraw,
            step_count=step_count,
            simulation_time_ns=simulation_time_ns,
            generation=generation,
            physics_pose_set=pose_set,
            complete_pose_projected=complete_pose_projected,
            render_setting_rejected=_extract_render_setting_rejected(update_result),
            skipped_reason=(
                str(update_result.get("skipped_reason", ""))
                or (
                    stage_result.reason
                    if stage_result is not None
                    and stage_result.status != OvphysxStageStatus.OK
                    else ""
                )
            ),
            update=_merge_update(stage_update, update_result),
        )

def _timeline_max_steps(
    request: RuntimeTickRequest,
    steps_per_update: int,
    configured_max_steps: int,
) -> int:
    if not request.timeline_controls_enabled:
        return max(1, int(configured_max_steps))
    frame_count = max(1, int(request.timeline_end) - int(request.timeline_start) + 1)
    return max(1, frame_count * max(1, int(steps_per_update)))


def _timeline_should_reset(
    request: RuntimeTickRequest,
    last_frame: int | None,
    last_reset_token: int,
) -> bool:
    if not request.timeline_controls_enabled:
        return False
    if int(request.simulation_reset_token) != int(last_reset_token):
        return True
    return last_frame is not None and int(request.timeline_frame) < int(last_frame)


def _timeline_should_step(
    request: RuntimeTickRequest,
    last_frame: int | None,
    controller_started: bool,
) -> bool:
    if not request.timeline_controls_enabled:
        return True
    if not controller_started:
        return True
    if request.timeline_playing:
        return True
    return last_frame is not None and int(request.timeline_frame) > int(last_frame)


def _runtime_status(
    result: OvphysxStageResult | None,
    ok_status: RuntimeTickStatus,
) -> RuntimeTickStatus:
    if result is None or result.status == OvphysxStageStatus.OK:
        return ok_status
    return {
        OvphysxStageStatus.COMPLETED: RuntimeTickStatus.COMPLETED,
        OvphysxStageStatus.BUSY: RuntimeTickStatus.BUSY,
        OvphysxStageStatus.FAILED: RuntimeTickStatus.FAILED,
    }[result.status]


def _stage_result_mapping(result: OvphysxStageResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "status": result.status.value,
        "reason": result.reason,
        "dirty_paths": list(result.dirty_paths),
        "step_count": result.step_count,
        "simulation_time_ns": result.simulation_time_ns,
        "composition_generation": result.generation,
    }


def _ovrtx_application_error(
    result: OvrtxValueUpdateResult,
    requested_count: int,
) -> str:
    if result.updated_count != requested_count:
        return f"OVRTX application accepted {result.updated_count} of {requested_count} values"
    diagnostics = result.diagnostics
    status = str(diagnostics.get("status", "")).lower()
    if status in {"error", "failed", "unavailable"}:
        return str(diagnostics.get("error") or status)
    grpc_status = str(diagnostics.get("grpc_status", "OK")).upper()
    if grpc_status != "OK":
        return f"OVRTX application gRPC status is {grpc_status}"
    return ""


def _merge_update(
    update: Mapping[str, Any],
    update_result: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(update)
    if update_result:
        merged["update_result"] = dict(update_result)
        for key in (
            "physics_generation_reset",
            "physics_generation_invalidated",
            "render_value_write_applied",
        ):
            if key in update_result:
                merged[key] = update_result[key]
        if update_result.get("values_written", False):
            attributes = tuple(str(value) for value in update_result.get("value_attributes", ()))
            merged["transform_updated"] = bool(
                update_result.get(
                    "transform_updated",
                    attributes and all(value in {"omni:xform", "xformOp:transform"} for value in attributes),
                )
            )
            merged["value_paths"] = list(update_result.get("value_paths", ()))
            merged["value_count"] = int(update_result.get("value_count", 0))
        if update_result.get("skipped_reason", ""):
            merged["skipped_reason"] = update_result["skipped_reason"]
    return merged
