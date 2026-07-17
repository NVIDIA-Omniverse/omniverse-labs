# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact edit records for diagnostic artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .interactive_edit_planner import (
    EditPlan,
    InteractiveEdit,
    RENDER_SETTING_VALUE_SOURCE,
)


SCHEMA_VERSION = 1
ARTIFACT_ID = "ovrtx-edit-records"
_UPDATE_TARGET_PATH_KEYS = (
    "value_paths",
)


def records_from_workflow_result(
    *,
    event_index: int,
    timestamp_ns: int,
    action: str,
    accepted: bool,
    reason: str,
    diagnostics: Mapping[str, Any],
    plan: Any = None,
    edits: tuple[InteractiveEdit, ...] = (),
    submission_result: Any = None,
    write_results: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    edit_items = _edit_items(plan, edits)
    if not edit_items:
        return []

    records: list[dict[str, Any]] = []
    batch_id = _batch_id(event_index) if len(edit_items) > 1 else ""
    for edit_index, item in enumerate(edit_items):
        edit, item_accepted, item_reason = item
        item_write_result = (
            write_results.get(edit.usd_layer_id)
            if write_results is not None
            else None
        )
        if item_write_result is not None:
            record_accepted = bool(getattr(item_write_result, "completed", False))
            record_reason = str(getattr(item_write_result, "reason", reason))
            record_diagnostics = {
                **dict(diagnostics),
                "result": _write_result_name(item_write_result),
            }
        else:
            record_accepted = bool(accepted and item_accepted)
            record_reason = str(reason if item_accepted else item_reason)
            record_diagnostics = diagnostics
        records.append(
            _record(
                edit_id=_edit_id(event_index, edit_index, len(edit_items)),
                batch_id=batch_id,
                event_index=event_index,
                edit_index=edit_index,
                timestamp_ns=timestamp_ns,
                source=_source(edit),
                action=action,
                accepted=record_accepted,
                reason=record_reason,
                edit=edit,
                result_diagnostics=record_diagnostics,
                submission_result=submission_result,
                write_result=item_write_result,
            )
        )
    return records


def selection_observation_record(
    *,
    edit_id: str,
    timestamp_ns: int,
    selection_resolution: Mapping[str, Any],
    reason: str = "selection_observation",
) -> dict[str, Any]:
    unresolved = [
        str(item)
        for item in selection_resolution.get("unresolved_reasons", ())
        if str(item)
    ]
    accepted = not bool(selection_resolution.get("group_rejected", False))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "edit_id": str(edit_id),
        "timestamp_ns": int(timestamp_ns),
        "source": "selection_resolution",
        "action": "observation",
        "accepted": accepted,
        "result": "applied" if accepted else "unsupported",
        "reason": str(reason),
        "fail_reason": ";".join(unresolved) if not accepted else "",
        "selection_resolution": dict(selection_resolution),
        "shape": "",
        "data_authority": "",
        "target_identity": {},
        "usd_layer_id": "",
        "mechanism": "none",
        "persistence": "none",
        "identity": {
            "render_composition_identity_required": False,
            "target_identity_preserved": False,
            "provenance_preserved": False,
        },
        "render_session": {
            "reuse_expected": None,
            "rekey_expected": False,
            "refinement_reset_expected": False,
        },
        "physics": {
            "reset_expected": False,
            "generation_reset": False,
        },
        "values_written": False,
        "rendered_effect_observed": False,
        "operator_workflow_observed": bool(selection_resolution.get("status", "") == "resolved"),
        "artifacts": {},
    }


def record_with_update_result(
    record: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(record)
    update_values_written = bool(update.get("values_written", False))
    if update_values_written:
        updated["result"] = "applied"
    updated["values_written"] = bool(updated.get("values_written", False) or update_values_written)
    updated["rendered_effect_observed"] = bool(
        updated.get("rendered_effect_observed", False)
        or update.get("rendered_effect_observed", False)
    )

    physics = _mapping(updated.get("physics", {}))
    physics["generation_reset"] = bool(
        physics.get("generation_reset", False)
        or update.get("physics_generation_reset", False)
    )
    updated["physics"] = physics

    artifacts = _mapping(updated.get("artifacts", {}))
    artifacts.update(
        _artifacts(
            update_result=update,
            write_result=None,
            write_diagnostics={},
        )
    )
    updated["artifacts"] = artifacts

    render_setting = updated.get("render_setting")
    if isinstance(render_setting, Mapping):
        entry = dict(render_setting)
        attribute = str(entry.get("attribute", "") or "")
        rekey_attributes = _render_setting_rekey_attributes(update)
        if rekey_attributes and (not attribute or attribute in rekey_attributes):
            # The worker rejected the live runtime write, so the change is
            # applied by folding the RTPT values back into the composition
            # digest and re-keying the session (task01-04 fallback). The
            # recomposed session authors the value and warms up on its first
            # frame; the record states the provenance so evidence distinguishes
            # a live write from a re-key.
            entry["applied_via"] = "rekey"
        elif update_values_written and _update_wrote_attribute(update, attribute):
            # The write landed on the session-owning render thread and reset
            # refinement (``should_reset_refinement = values_written``), so the
            # post-write warm-up ran — the new value is active.
            entry["applied_on_thread"] = "render"
            entry["warmup_completed"] = True
        updated["render_setting"] = entry

    if bool(update.get("failed", False)):
        skipped_reason = str(update.get("skipped_reason", "") or "update_failed")
        updated["result"] = "failed"
        updated["fail_reason"] = skipped_reason
    return updated


def _render_setting_rekey_attributes(update: Mapping[str, Any]) -> set[str]:
    """Attributes whose live render-setting write the worker rejected.

    A rejected live write is folded back into the composition digest and the
    session is re-keyed (task01-04 fallback); the render loop / scheduler stamp
    the rejection under ``render_setting_rejected`` (top-level or on a combined
    update's lane). A record for one of these attributes reports
    ``applied_via: rekey``.
    """

    marker = update.get("render_setting_rejected")
    if not isinstance(marker, Mapping):
        for lane in update.get("updates", ()) or ():
            if isinstance(lane, Mapping) and isinstance(
                lane.get("render_setting_rejected"), Mapping
            ):
                marker = lane["render_setting_rejected"]
                break
    if not isinstance(marker, Mapping):
        return set()
    return {str(attribute) for attribute in marker.get("attributes", ()) or ()}


def _update_wrote_attribute(update: Mapping[str, Any], attribute: str) -> bool:
    """Whether the applied batch wrote ``attribute`` (per-attribute scope).

    ``update_matches_record`` matches on prim path only; every RTPT setting
    shares the one render-product path, so the warm-up confirmation
    additionally requires the record's attribute in the applied batch's
    ``value_attributes``. An update without ``value_attributes`` keeps the
    path-level match (no evidence to narrow on).
    """

    value_attributes = update.get("value_attributes")
    if not isinstance(value_attributes, (list, tuple)):
        return True
    if not attribute:
        return True
    return attribute in {str(item) for item in value_attributes}


def update_matches_record(
    record: Mapping[str, Any],
    update: Mapping[str, Any],
) -> bool:
    target = record.get("target_identity")
    if not isinstance(target, Mapping):
        return False
    record_path = str(target.get("usd_prim_path", "") or "")
    if not record_path:
        return False
    return record_path in _update_target_paths(update)


def _record(
    *,
    edit_id: str,
    batch_id: str,
    event_index: int,
    edit_index: int,
    timestamp_ns: int,
    source: str,
    action: str,
    accepted: bool,
    reason: str,
    edit: InteractiveEdit,
    result_diagnostics: Mapping[str, Any],
    submission_result: Any,
    write_result: Any,
) -> dict[str, Any]:
    scheduler_diagnostics = _mapping(getattr(submission_result, "diagnostics", {}))
    write_diagnostics = _mapping(getattr(write_result, "diagnostics", {}))
    values_written = bool(scheduler_diagnostics.get("values_written", False))
    if write_result is not None:
        values_written = values_written or bool(getattr(write_result, "completed", False))
    artifacts = _artifacts(
        update_result=scheduler_diagnostics,
        write_result=write_result,
        write_diagnostics=write_diagnostics,
    )
    record_result = str(result_diagnostics.get("result", "applied" if accepted else "unsupported"))
    if values_written and record_result == "queued":
        record_result = "applied"
    record = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "edit_id": edit_id,
        "batch_id": batch_id,
        "event_index": int(event_index),
        "edit_index": int(edit_index),
        "timestamp_ns": int(timestamp_ns),
        "source": source,
        "action": str(action),
        "accepted": bool(accepted),
        "result": record_result,
        "reason": str(reason),
        # Diagnostic classification vocabulary (task04-07): one of
        # supported / unsupported / non_rendering / topology, or "" for
        # records predating classification (selection observations).
        "classification": str(result_diagnostics.get("classification", "") or ""),
        "fail_reason": "" if accepted else str(reason),
        "selection_resolution": _selection_resolution(edit),
        "shape": edit.shape.value,
        "data_authority": edit.data_authority.value,
        "target_identity": _target_identity(edit),
        "usd_layer_id": edit.usd_layer_id,
        "mechanism": str(result_diagnostics.get("mechanism", "")),
        "persistence": str(result_diagnostics.get("persistence", "")),
        "identity": {
            "render_composition_identity_required": bool(
                result_diagnostics.get("render_composition_identity_required", False)
            ),
            "target_identity_preserved": bool(result_diagnostics.get("target_identity_preserved", False)),
            "provenance_preserved": bool(result_diagnostics.get("provenance_preserved", False)),
        },
        "render_session": {
            "reuse_expected": _optional_bool(result_diagnostics.get("render_session_reuse_expected")),
            "rekey_expected": bool(result_diagnostics.get("session_rekey_expected", False)),
            "refinement_reset_expected": bool(result_diagnostics.get("refinement_reset_expected", False)),
        },
        "physics": {
            "reset_expected": bool(result_diagnostics.get("physics_generation_reset_expected", False)),
            "generation_reset": bool(getattr(submission_result, "physics_generation_reset", False)),
        },
        "values_written": values_written,
        "rendered_effect_observed": bool(
            result_diagnostics.get("rendered_effect_observed", False)
            or scheduler_diagnostics.get("rendered_effect_observed", False)
        ),
        "operator_workflow_observed": bool(result_diagnostics.get("operator_workflow_observed", False)),
        "artifacts": artifacts,
    }
    render_setting = _render_setting_entry(edit)
    if render_setting is not None:
        record["render_setting"] = render_setting
    if not batch_id:
        record.pop("batch_id")
    return record


def _render_setting_entry(edit: InteractiveEdit) -> dict[str, Any] | None:
    """RTPT quality-change diagnostic entry (task01-05), or ``None``.

    A live quality change is morally a value edit against the render product,
    so it rides the existing edit-record path. This entry states the change is
    active: the authored attribute/value/dtype on the render product, whether
    it was applied live (the runtime write) or via session re-key, and the
    reset/warm-up (render at ``min_samples``, refine to ``max_samples``).
    ``warmup_completed`` starts ``False`` and is flipped by
    ``record_with_update_result`` when the render-thread apply confirms the
    write; ``applied_on_thread`` is filled at the same point.
    """

    provenance = edit.provenance
    if str(provenance.get("source", "")) != RENDER_SETTING_VALUE_SOURCE:
        return None
    return {
        "attribute": edit.usd_attribute,
        # ``value`` is the wire value actually sent to OVRTX; ``ui_value`` is the
        # artist-facing value from the slider (they differ for Max Bounces, which
        # adds the +2 camera-ray offset). Both are recorded so evidence states
        # what the artist set and what was sent.
        "value": edit.value,
        "ui_value": provenance.get("ui_value", edit.value),
        "dtype": str(provenance.get("dtype", "") or ""),
        "render_product_path": edit.usd_prim_path,
        # "live" runtime write (task01-04 primary route) unless a capability
        # probe fell back to session re-keying, in which case provenance says
        # so (``applied_via: rekey``) so evidence distinguishes the two.
        "applied_via": str(provenance.get("applied_via", "") or "live"),
        # A render-setting value edit always resets refinement on apply (the
        # warm-up); completion is confirmed when the write lands.
        "reset_requested": True,
        "applied_on_thread": "",
        "warmup_completed": False,
    }


def _edit_items(
    plan: Any,
    edits: tuple[InteractiveEdit, ...],
) -> tuple[tuple[InteractiveEdit, bool, str], ...]:
    if isinstance(plan, EditPlan):
        return ((plan.edit, True, ""),)
    return tuple((edit, True, "") for edit in edits)


def _edit_id(event_index: int, edit_index: int, edit_count: int) -> str:
    base = f"edit-{int(event_index):06d}"
    return base if edit_count == 1 else f"{base}-{int(edit_index):02d}"


def _batch_id(event_index: int) -> str:
    return f"batch-{int(event_index):06d}"


def _source(edit: InteractiveEdit) -> str:
    source = str(edit.metadata.get("source", "") or "")
    if source:
        return source
    return str(edit.provenance.get("source", "") or "interactive_edit_workflow")


def _target_identity(edit: InteractiveEdit) -> dict[str, Any]:
    target = edit
    return {
        "usd_prim_path": target.usd_prim_path,
        "usd_attribute": target.usd_attribute,
        "usd_property_path": target.usd_property_path,
        "usd_layer_id": target.usd_layer_id,
        "blender_property_path": target.blender_property_path,
        "provenance": dict(target.provenance),
    }


def _selection_resolution(edit: InteractiveEdit) -> dict[str, Any]:
    selection = edit.provenance.get("selection_resolution", {})
    return dict(selection) if isinstance(selection, Mapping) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _artifacts(
    *,
    update_result: Mapping[str, Any],
    write_result: Any,
    write_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    write_path = str(getattr(write_result, "path", "") or "")
    if write_path:
        artifacts["write_path"] = write_path
    usd_layer_path = str(write_diagnostics.get("usd_layer_path", "") or "")
    if usd_layer_path:
        artifacts["usd_layer_path"] = usd_layer_path
    update_paths = _update_artifact_paths(update_result)
    if update_paths:
        artifacts["update_paths"] = update_paths
    return artifacts


def _update_target_path_values(update: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for key in _UPDATE_TARGET_PATH_KEYS:
        paths.extend(str(path) for path in update.get(key, ()) if str(path))
    target = update.get("target")
    if isinstance(target, Mapping):
        path = str(target.get("usd_prim_path", "") or "")
        if path:
            paths.append(path)
    targets = update.get("targets")
    if isinstance(targets, (list, tuple)):
        for item in targets:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("usd_prim_path", "") or "")
            if path:
                paths.append(path)
    return tuple(paths)


def _update_artifact_paths(update: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for path in _update_target_path_values(update):
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _update_target_paths(update: Mapping[str, Any]) -> set[str]:
    return set(_update_target_path_values(update))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _write_result_name(result: Any) -> str:
    if bool(getattr(result, "completed", False)):
        return "applied"
    if bool(getattr(result, "requested", False)):
        return "failed"
    return "unsupported"
