# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive edit workflow orchestration for ADR 0009."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import time
from typing import Any, Mapping

from .edit_persistence import EditWriter, WriteRequest, WriteResult
from . import edit_records
from . import topology_edit_fallback
from .interactive_edit_planner import (
    EditMechanism,
    EditPersistence,
    EditShape,
    EditStatus,
    EditPlan,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from .runtime_scheduler import EditSubmissionResult, RuntimeScheduler
from .value_edit_conversion import (
    CLASSIFICATION_NON_RENDERING,
    CLASSIFICATION_SUPPORTED,
    CLASSIFICATION_TOPOLOGY,
    CLASSIFICATION_UNSUPPORTED,
    classification_for_unsupported_reason,
    classification_report_message,
    display_field_name,
    normalized_classification,
)


# diagnostics are not authoring state; retain a useful tail only.
# Stream records to an artifact writer if full-session audit history is needed.
DIAGNOSTIC_HISTORY_LIMIT = 256


class WorkflowAction(str, Enum):
    OBSERVATION = "observation"
    UPDATE = "update"
    COMPOSE = "compose"
    WRITE = "write"
    SELECTED_FOR_WRITE = "selected_for_write"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EditWorkflowResult:
    action: WorkflowAction
    status: EditStatus
    reason: str
    plan: EditPlan | None = None
    edits: tuple[InteractiveEdit, ...] = ()
    submission_result: EditSubmissionResult | None = None
    write_results: Mapping[str, WriteResult] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    #: Diagnostic classification (task04-07): one of the four normalized
    #: vocabulary values (supported / unsupported / non_rendering /
    #: topology), or "" when the result carries no single edit. Describes
    #: the edit family's classification, not the submission outcome —
    #: outcome stays on ``status``/``reason``.
    classification: str = ""
    #: User-visible one-line report, set at most once per
    #: (target, source field) per authoring or exact-stage session.
    #: Empty when nothing should be reported (supported edits, repeats of
    #: an already-reported key, or non-classification failures).
    user_report: str = ""

    @property
    def accepted(self) -> bool:
        return self.status in {EditStatus.QUEUED, EditStatus.APPLIED}


class InteractiveEditWorkflow:
    """Coordinates preview edits, updates, and selected writes."""

    def __init__(
        self,
        *,
        planner: InteractiveEditPlanner | None = None,
        runtime_scheduler: RuntimeScheduler | None = None,
        writer: EditWriter | None = None,
    ) -> None:
        self._planner = planner or InteractiveEditPlanner()
        self._runtime_scheduler = runtime_scheduler
        self._writer = writer
        self._selected_write_edits: list[InteractiveEdit] = []
        self._events: list[dict[str, Any]] = []
        self._edit_records: list[dict[str, Any]] = []
        self._event_count = 0
        self._edit_record_count = 0
        # Once-per-key user-visible report dedupe (task04-07). Keyed on
        # (usd_prim_path or blender_property_path, source field); the
        # workflow instance lives for one authoring or exact-stage session,
        # so the dedupe resets with that session.
        # Edit records are still written every time — only the
        # user-visible report({'INFO'}) is once-per-key, which also absorbs
        # drag spam (repeated depsgraph callbacks share the key).
        self._reported_user_keys: set[tuple[str, str]] = set()
        self._user_report_count = 0

    def preview_edit(self, edit: InteractiveEdit) -> EditWorkflowResult:
        plan = self._planner.plan(edit)
        if plan.mechanism == EditMechanism.UPDATE:
            if self._runtime_scheduler is None:
                return self._record_result(
                    EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.UNSUPPORTED,
                        reason="update_scheduler_unavailable",
                        plan=plan,
                        diagnostics=_plan_diagnostics(plan),
                    )
                )
            submission_result = self._runtime_scheduler.submit_edit(plan.to_intent())
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.UPDATE,
                    status=submission_result.status,
                    reason=submission_result.reason,
                    plan=plan,
                    submission_result=submission_result,
                    diagnostics={
                        **_plan_diagnostics(plan),
                        "scheduler": dict(submission_result.diagnostics),
                    },
                )
            )

        if (
            plan.mechanism == EditMechanism.COMPOSE
            and plan.impact.scene_generation_replacement_requested
        ):
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.COMPOSE,
                    status=EditStatus.QUEUED,
                    reason="scene_generation_dirty",
                    plan=plan,
                    diagnostics=_plan_diagnostics(plan),
                )
            )

        if plan.persistence == EditPersistence.WRITE:
            result = self._write_target(plan)
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.WRITE
                    if result.requested
                    else WorkflowAction.UNSUPPORTED,
                    status=_status_from_write_result(result),
                    reason=result.reason,
                    plan=plan,
                    write_results=(
                        {plan.usd_layer_id: result}
                        if self._writer is not None
                        else {}
                    ),
                    diagnostics={
                        **_plan_diagnostics(plan),
                        **_topology_fallback_diagnostics(plan, result),
                        "write": dict(result.diagnostics),
                    },
                )
            )

        return self._record_result(
            EditWorkflowResult(
                action=WorkflowAction.UNSUPPORTED,
                status=EditStatus.UNSUPPORTED,
                reason=plan.unsupported_reason or plan.reason,
                plan=plan,
                diagnostics=_plan_diagnostics(plan),
            )
        )

    def preview_edit_group(
        self,
        edits: tuple[InteractiveEdit, ...],
    ) -> tuple[EditWorkflowResult, ...]:
        """Plan the complete group, then submit all updates or none."""

        if not edits:
            return ()
        plans = tuple(self._planner.plan(edit) for edit in edits)
        mechanisms = {plan.mechanism for plan in plans}
        if mechanisms != {EditMechanism.UPDATE}:
            if len(mechanisms) == 1 and EditMechanism.NONE not in mechanisms:
                return tuple(self.preview_edit(edit) for edit in edits)
            return tuple(
                self._record_result(
                    EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.UNSUPPORTED,
                        reason="edit_group_rejected",
                        plan=plan,
                        diagnostics=_plan_diagnostics(plan),
                    )
                )
                for plan in plans
            )
        if self._runtime_scheduler is None:
            return tuple(
                self._record_result(
                    EditWorkflowResult(
                        action=WorkflowAction.UNSUPPORTED,
                        status=EditStatus.UNSUPPORTED,
                        reason="update_scheduler_unavailable",
                        plan=plan,
                        diagnostics=_plan_diagnostics(plan),
                    )
                )
                for plan in plans
            )
        submissions = self._runtime_scheduler.submit_edits(
            tuple(plan.to_intent() for plan in plans)
        )
        return tuple(
            self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.UPDATE,
                    status=submission.status,
                    reason=submission.reason,
                    plan=plan,
                    submission_result=submission,
                    diagnostics={
                        **_plan_diagnostics(plan),
                        "scheduler": dict(submission.diagnostics),
                    },
                )
            )
            for plan, submission in zip(plans, submissions)
        )

    def select_for_write(self, edit: InteractiveEdit) -> EditWorkflowResult:
        plan = self._planner.plan(edit)
        if (
            plan.persistence != EditPersistence.WRITE
            or plan.impact.scene_generation_replacement_requested
        ):
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.UNSUPPORTED,
                    status=EditStatus.UNSUPPORTED,
                    reason="edit_not_writeable",
                    plan=plan,
                    diagnostics=_plan_diagnostics(plan),
                )
            )
        self._selected_write_edits.append(edit)
        return self._record_result(
            EditWorkflowResult(
                action=WorkflowAction.SELECTED_FOR_WRITE,
                status=EditStatus.QUEUED,
                reason="selected_for_write",
                plan=plan,
                diagnostics={
                    **_plan_diagnostics(plan),
                    "pending_selected_write_count": len(self._selected_write_edits),
                },
            )
        )

    def write_selected_edits(self) -> EditWorkflowResult:
        selected_edits = tuple(self._selected_write_edits)
        if not selected_edits:
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.UNSUPPORTED,
                    status=EditStatus.UNSUPPORTED,
                    reason="no_selected_write_edits",
                    diagnostics=_selected_write_diagnostics((), pending_count=0),
                )
            )
        if self._writer is None:
            return self._record_result(
                EditWorkflowResult(
                    action=WorkflowAction.UNSUPPORTED,
                    status=EditStatus.UNSUPPORTED,
                    reason="writer_unavailable",
                    edits=selected_edits,
                    diagnostics=_selected_write_diagnostics(
                        selected_edits,
                        pending_count=len(self._selected_write_edits),
                    ),
                )
            )

        write_results: dict[str, WriteResult] = {}
        completed_ids: set[int] = set()
        for usd_layer_id, edits in _selected_write_groups(selected_edits):
            result = self._writer(
                WriteRequest(
                    edits=edits,
                    reason="selected_write",
                    usd_layer_id=usd_layer_id,
                )
            )
            write_results[usd_layer_id] = result
            if result.completed:
                completed_ids.update(id(edit) for edit in edits)
        if completed_ids:
            self._selected_write_edits = [
                edit for edit in self._selected_write_edits if id(edit) not in completed_ids
            ]
        result = _aggregate_write_results(write_results)
        return self._record_result(
            EditWorkflowResult(
                action=(
                    WorkflowAction.WRITE
                    if result.requested
                    else WorkflowAction.UNSUPPORTED
                ),
                status=_status_from_write_result(result),
                reason=result.reason,
                edits=selected_edits,
                write_results=write_results,
                diagnostics={
                    **_selected_write_diagnostics(
                        selected_edits,
                        pending_count=len(self._selected_write_edits),
                    ),
                    "write": dict(result.diagnostics),
                },
            )
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "event_count": self._event_count,
            "retained_event_count": len(self._events),
            "pending_selected_write_count": len(self._selected_write_edits),
            "edit_record_count": self._edit_record_count,
            "retained_edit_record_count": len(self._edit_records),
            "edit_records": list(self._edit_records),
            "events": list(self._events),
            "user_report_count": self._user_report_count,
            "user_reported_keys": sorted(self._reported_user_keys),
        }

    def record_selection_resolution(
        self,
        selection_resolution: Mapping[str, Any],
        *,
        reason: str = "selection_observation",
    ) -> dict[str, Any]:
        event_index = self._event_count + 1
        timestamp_ns = time.time_ns()
        record = edit_records.selection_observation_record(
            edit_id=f"edit-{event_index:06d}",
            timestamp_ns=timestamp_ns,
            selection_resolution=selection_resolution,
            reason=reason,
        )
        self._record_event(
            action=WorkflowAction.OBSERVATION,
            accepted=bool(record["accepted"]),
            reason=reason,
            timestamp_ns=timestamp_ns,
            diagnostics={"selection_resolution": dict(selection_resolution)},
            edit_records=[record],
        )
        return record

    def record_update_result(self, update: Mapping[str, Any]) -> int:
        matched = 0
        records: list[dict[str, Any]] = []
        for record in self._edit_records:
            if (
                record.get("action") == WorkflowAction.UPDATE.value
                and edit_records.update_matches_record(
                    record,
                    update,
                )
            ):
                records.append(edit_records.record_with_update_result(record, update))
                matched += 1
            else:
                records.append(record)
        self._edit_records = records
        return matched

    def _write_target(self, plan: EditPlan) -> WriteResult:
        if self._writer is None:
            return _unavailable_write_result("writer_unavailable")
        request = WriteRequest(
            edits=(plan.edit,),
            reason=plan.reason,
            usd_layer_id=plan.usd_layer_id,
        )
        return self._writer(request)

    def _record_result(self, result: EditWorkflowResult) -> EditWorkflowResult:
        result = self._classified_result(result)
        event_index = self._event_count + 1
        timestamp_ns = time.time_ns()
        records = edit_records.records_from_workflow_result(
            event_index=event_index,
            timestamp_ns=timestamp_ns,
            action=result.action.value,
            accepted=result.accepted,
            reason=result.reason,
            diagnostics={**dict(result.diagnostics), "result": result.status.value},
            plan=result.plan,
            edits=result.edits,
            submission_result=result.submission_result,
            write_results=result.write_results,
        )
        self._record_event(
            action=result.action,
            status=result.status,
            accepted=result.accepted,
            reason=result.reason,
            timestamp_ns=timestamp_ns,
            diagnostics={**dict(result.diagnostics), "result": result.status.value},
            edit_records=records,
        )
        return result

    def _classified_result(self, result: EditWorkflowResult) -> EditWorkflowResult:
        """Stamp the diagnostic classification and once-per-key user report.

        Classification (task04-07) uses the four-value normalized
        vocabulary and describes the edit family's classification;
        submission outcomes stay on ``status``/``reason``. The user-visible
        report is produced only for classification-originated results
        (policy ``unsupported_reason`` metadata, explicit ``classification``
        metadata, or family topology edits carrying
        ``topology_change_kinds``), at most once per
        (usd_prim_path or blender_property_path, source field) per
        workflow lifetime. The edit record is written every time.
        """

        edit = _result_edit(result)
        if edit is None:
            return result
        classification = _edit_classification(result, edit)
        message = _classification_user_message(result, edit, classification)
        user_report = ""
        if message:
            key = _user_report_key(edit)
            if key not in self._reported_user_keys:
                self._reported_user_keys.add(key)
                self._user_report_count += 1
                user_report = message
        diagnostics = dict(result.diagnostics)
        if classification:
            diagnostics["classification"] = classification
        return replace(
            result,
            classification=classification,
            user_report=user_report,
            diagnostics=diagnostics,
        )

    def _record_event(
        self,
        *,
        action: WorkflowAction,
        status: EditStatus | None = None,
        accepted: bool,
        reason: str,
        timestamp_ns: int,
        diagnostics: Mapping[str, Any],
        edit_records: list[dict[str, Any]],
    ) -> None:
        self._event_count += 1
        self._edit_record_count += len(edit_records)
        self._edit_records.extend(edit_records)
        if len(self._edit_records) > DIAGNOSTIC_HISTORY_LIMIT:
            del self._edit_records[: -DIAGNOSTIC_HISTORY_LIMIT]
        self._events.append(
            {
                "action": action.value,
                "result": status.value if status is not None else "",
                "accepted": bool(accepted),
                "reason": str(reason),
                "timestamp_ns": timestamp_ns,
                "diagnostics": dict(diagnostics),
                "edit_record_ids": [record["edit_id"] for record in edit_records],
            }
        )
        if len(self._events) > DIAGNOSTIC_HISTORY_LIMIT:
            del self._events[: -DIAGNOSTIC_HISTORY_LIMIT]


def _result_edit(result: EditWorkflowResult) -> InteractiveEdit | None:
    if result.plan is not None:
        return result.plan.edit
    if len(result.edits) == 1:
        return result.edits[0]
    return None


def _edit_classification(result: EditWorkflowResult, edit: InteractiveEdit) -> str:
    explicit = normalized_classification(edit.metadata.get("classification", ""))
    if explicit:
        return explicit
    if edit.shape == EditShape.TOPOLOGY or (
        result.plan is not None and result.plan.impact.topology_reasons
    ):
        return CLASSIFICATION_TOPOLOGY
    unsupported_reason = str(edit.metadata.get("unsupported_reason", "") or "")
    if unsupported_reason:
        return classification_for_unsupported_reason(unsupported_reason)
    # Infrastructure failures on supported routes (scheduler unavailable,
    # write failures, missing identities) keep the family classification:
    # the field is supported; the outcome is on status/reason.
    return CLASSIFICATION_SUPPORTED


def _classification_user_message(
    result: EditWorkflowResult,
    edit: InteractiveEdit,
    classification: str,
) -> str:
    explicit = bool(str(edit.metadata.get("classification", "") or ""))
    if classification == CLASSIFICATION_TOPOLOGY:
        # Only the value-edit families' topology edits report the route
        # (material graph, light form, world node graph/assignment —
        # they carry topology_change_kinds); generic scene topology stays
        # report-free to avoid noise on ordinary object adds/removes.
        if not (explicit or edit.metadata.get("topology_change_kinds")):
            return ""
    elif classification in (CLASSIFICATION_UNSUPPORTED, CLASSIFICATION_NON_RENDERING):
        if not (explicit or str(edit.metadata.get("unsupported_reason", "") or "")):
            return ""
    else:
        return ""
    return classification_report_message(
        classification,
        field=display_field_name(edit.blender_property_path),
    )


def _user_report_key(edit: InteractiveEdit) -> tuple[str, str]:
    return (
        str(edit.usd_prim_path or edit.blender_property_path or ""),
        str(edit.blender_property_path or ""),
    )


def _plan_diagnostics(plan: EditPlan) -> dict[str, Any]:
    return {
        "shape": plan.shape.value,
        "data_authority": plan.data_authority.value,
        "mechanism": plan.mechanism.value,
        "persistence": plan.persistence.value,
        "usd_layer_id": plan.usd_layer_id,
        "target": _target_details(plan.edit),
        "update_requested": plan.impact.update_requested,
        "write_requested": plan.impact.write_requested,
        "whole_scene_export_requested": plan.impact.whole_scene_export_requested,
        "whole_scene_export_avoided": plan.impact.whole_scene_export_avoided,
        "render_composition_identity_required": plan.impact.render_composition_identity_required,
        "render_session_reuse_expected": plan.impact.render_session_reuse_expected,
        "physics_generation_reset_expected": plan.impact.physics_generation_reset_expected,
        "target_identity_preserved": plan.impact.target_identity_preserved,
        "provenance_preserved": plan.impact.provenance_preserved,
        "topology_reasons": list(plan.impact.topology_reasons),
        "update_stream_rejected": plan.impact.update_stream_rejected,
        "session_rekey_expected": plan.impact.session_rekey_expected,
        "refinement_reset_expected": plan.impact.refinement_reset_expected,
        "scene_generation_replacement_requested": (
            plan.impact.scene_generation_replacement_requested
        ),
    }


def _topology_fallback_diagnostics(plan: EditPlan, result: WriteResult) -> dict[str, Any]:
    if not plan.impact.topology_reasons:
        return {}
    return {
        "topology_fallback": topology_edit_fallback.topology_rekey_diagnostics(
            reasons=plan.impact.topology_reasons,
            requested_write_path=result.path if result.requested else "",
            session_rekey_status="requested" if result.requested else "blocked",
            write_requested=result.requested,
        )
    }


def _selected_write_diagnostics(
    edits: tuple[InteractiveEdit, ...],
    *,
    pending_count: int,
) -> dict[str, Any]:
    return {
        "mechanism": EditMechanism.NONE.value,
        "persistence": EditPersistence.WRITE.value if edits else EditPersistence.NONE.value,
        "selected_edit_count": len(edits),
        "selected_targets": [_target_details(edit) for edit in edits],
        "pending_selected_write_count": pending_count,
        "whole_scene_export_requested": False,
        "whole_scene_export_avoided": True,
        "target_identity_preserved": bool(edits)
        and all(edit.has_edit_identity() for edit in edits),
        "provenance_preserved": bool(edits)
        and all(
            bool(edit.usd_layer_id or edit.provenance)
            for edit in edits
        ),
    }


def _unavailable_write_result(reason: str) -> WriteResult:
    return WriteResult(
        requested=False,
        completed=False,
        reason=reason,
        diagnostics={"unsupported_path": reason},
    )


def _target_details(edit: InteractiveEdit) -> dict[str, Any]:
    target = edit
    return {
        "shape": edit.shape.value,
        "data_authority": edit.data_authority.value,
        "usd_prim_path": target.usd_prim_path,
        "usd_attribute": target.usd_attribute,
        "usd_property_path": target.usd_property_path,
        "usd_layer_id": target.usd_layer_id,
        "blender_property_path": target.blender_property_path,
        "provenance": dict(target.provenance),
    }


def _selected_write_groups(
    edits: tuple[InteractiveEdit, ...],
) -> tuple[tuple[str, tuple[InteractiveEdit, ...]], ...]:
    grouped: dict[str, list[InteractiveEdit]] = {}
    for edit in edits:
        grouped.setdefault(edit.usd_layer_id, []).append(edit)
    return tuple(
        (usd_layer_id, tuple(group))
        for usd_layer_id, group in grouped.items()
    )


def _aggregate_write_results(results: Mapping[str, WriteResult]) -> WriteResult:
    if len(results) == 1:
        return next(iter(results.values()))
    completed_count = sum(1 for result in results.values() if result.completed)
    requested = any(result.requested for result in results.values())
    completed = completed_count == len(results)
    if completed:
        reason = "selected_write_completed"
    elif completed_count:
        reason = "selected_write_partial_failure"
    elif not requested:
        reason = "selected_write_unsupported"
    else:
        reason = "selected_write_failed"
    return WriteResult(
        requested=requested,
        completed=completed,
        reason=reason,
        diagnostics={
            "write_target_count": len(results),
            "completed_write_target_count": completed_count,
            "write_target_results": {
                usd_layer_id: {
                    "requested": result.requested,
                    "completed": result.completed,
                    "reason": result.reason,
                    "path": result.path,
                    "diagnostics": dict(result.diagnostics),
                }
                for usd_layer_id, result in results.items()
            },
        },
    )


def _status_from_write_result(result: WriteResult) -> EditStatus:
    if result.completed:
        return EditStatus.APPLIED
    if result.requested:
        return EditStatus.FAILED
    return EditStatus.UNSUPPORTED
