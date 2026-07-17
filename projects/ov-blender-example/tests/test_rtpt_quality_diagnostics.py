# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTPT quality-change diagnostics (task01-05).

An applied live quality change is morally a value edit against the render
product, so it rides the existing edit-record / viewport-diagnostics path
(task04-* / blender-live-render) rather than a new reporting channel. These
plain (non-Blender) tests assert that a render-setting edit driven through the
real ``InteractiveEditWorkflow`` emits exactly one edit record carrying a
``render_setting`` diagnostic entry with the authored attribute, value, dtype,
render product path, applied-via, and reset/warm-up completion — and that the
warm-up flips only once the render-thread apply confirms the write.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import edit_records, rtpt_live_change  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import EditStatus  # noqa: E402
from ovrtx_blender_example.interactive_edit_workflow import (  # noqa: E402
    InteractiveEditWorkflow,
    WorkflowAction,
)
from ovrtx_blender_example.runtime_scheduler import EditSubmissionResult  # noqa: E402


RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"


class _FakeScheduler:
    """Accepts submitted value-update intents as queued (no worker)."""

    def __init__(self) -> None:
        self.intents: list[object] = []

    def submit_edit(self, intent: object) -> EditSubmissionResult:
        self.intents.append(intent)
        return EditSubmissionResult(
            status=EditStatus.QUEUED,
            reason="queued",
            diagnostics={},
        )


def _apply_change(workflow: InteractiveEditWorkflow, property_name: str, value: object):
    edit = rtpt_live_change.render_setting_edit(property_name, value, RENDER_PRODUCT)
    return workflow.preview_edit(edit)


def _render_setting_records(workflow: InteractiveEditWorkflow) -> list[dict]:
    return [
        record
        for record in workflow.diagnostics()["edit_records"]
        if "render_setting" in record
    ]


def test_applied_change_emits_one_render_setting_record_with_full_fields() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())

    result = _apply_change(workflow, "rtpt_max_bounces", 7)
    assert result.action == WorkflowAction.UPDATE

    records = _render_setting_records(workflow)
    assert len(records) == 1
    record = records[0]

    # The diagnostic record rides the existing edit-record schema.
    assert record["schema_version"] == edit_records.SCHEMA_VERSION
    assert record["artifact_id"] == edit_records.ARTIFACT_ID
    assert record["source"] == rtpt_live_change.RENDER_SETTING_VALUE_SOURCE

    entry = record["render_setting"]
    assert entry["attribute"] == "omni:rtx:rtpt:maxBounces"
    # The record carries the wire value sent to OVRTX (UI 7 -> wire 9) plus the
    # artist-facing UI value.
    assert entry["value"] == 9
    assert entry["ui_value"] == 7
    assert entry["dtype"] == "int32"
    assert entry["render_product_path"] == RENDER_PRODUCT
    # Task01-04 primary route is a live runtime write (not a session re-key).
    assert entry["applied_via"] == "live"
    assert entry["reset_requested"] is True
    # Warm-up is not yet confirmed until the render-thread apply lands.
    assert entry["applied_on_thread"] == ""
    assert entry["warmup_completed"] is False


def test_firefly_bool_change_records_bool_dtype() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())

    _apply_change(workflow, "rtpt_firefly_filter_enabled", False)

    entry = _render_setting_records(workflow)[0]["render_setting"]
    assert entry["attribute"] == "omni:rtx:rtpt:fireflyFilter:enabled"
    assert entry["value"] is False
    assert entry["dtype"] == "bool"


def test_render_thread_apply_flips_warmup_completion() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    _apply_change(workflow, "rtpt_max_bounces", 7)

    # The render-thread tick applies the write and resets refinement; the
    # tick's update_result is folded back into the matching edit record.
    matched = workflow.record_update_result(
        {
            "values_written": True,
            "value_paths": [RENDER_PRODUCT],
            "value_attributes": ["omni:rtx:rtpt:maxBounces"],
            "value_types": ["Int"],
            "targets": [{"usd_prim_path": RENDER_PRODUCT}],
        }
    )
    assert matched == 1

    entry = _render_setting_records(workflow)[0]["render_setting"]
    assert entry["warmup_completed"] is True
    assert entry["applied_on_thread"] == "render"


def test_unwritten_update_does_not_flip_warmup_completion() -> None:
    # A tick that did not write values (e.g. worker unavailable, apply
    # failed) must not claim the warm-up ran — the record stays unconfirmed.
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    _apply_change(workflow, "rtpt_max_bounces", 7)

    matched = workflow.record_update_result(
        {
            "values_written": False,
            "failed": True,
            "skipped_reason": "value_update_unavailable",
            "value_paths": [RENDER_PRODUCT],
            "value_attributes": ["omni:rtx:rtpt:maxBounces"],
            "targets": [{"usd_prim_path": RENDER_PRODUCT}],
        }
    )
    assert matched == 1

    entry = _render_setting_records(workflow)[0]["render_setting"]
    assert entry["warmup_completed"] is False
    assert entry["applied_on_thread"] == ""


def test_warmup_confirmation_is_scoped_to_the_written_attribute() -> None:
    # Every RTPT setting shares the one render-product prim path, so the
    # path-level record match alone would flip warm-up on records for other
    # attributes; the confirmation additionally requires the record's
    # attribute in the applied batch's value_attributes.
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    _apply_change(workflow, "rtpt_max_bounces", 7)
    _apply_change(workflow, "rtpt_max_volume_bounces", 9)

    workflow.record_update_result(
        {
            "values_written": True,
            "value_paths": [RENDER_PRODUCT],
            "value_attributes": ["omni:rtx:rtpt:maxVolumeBounces"],
            "value_types": ["Int"],
            "targets": [{"usd_prim_path": RENDER_PRODUCT}],
        }
    )

    entries = {
        record["render_setting"]["attribute"]: record["render_setting"]
        for record in _render_setting_records(workflow)
    }
    assert entries["omni:rtx:rtpt:maxVolumeBounces"]["warmup_completed"] is True
    assert entries["omni:rtx:rtpt:maxVolumeBounces"]["applied_on_thread"] == "render"
    # The maxBounces write has not been confirmed by any applied batch.
    assert entries["omni:rtx:rtpt:maxBounces"]["warmup_completed"] is False
    assert entries["omni:rtx:rtpt:maxBounces"]["applied_on_thread"] == ""


def test_each_applied_edit_event_produces_one_record() -> None:
    # A slider drag emits one edit per UI event, each a preview_edit; the
    # per-attribute latest-wins coalescing happens on the render thread
    # (task01-04). The record semantics match value edits: one record per
    # applied edit event (not one collapsed record and not one per render
    # tick).
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())

    _apply_change(workflow, "rtpt_max_bounces", 1)
    _apply_change(workflow, "rtpt_max_bounces", 2)
    _apply_change(workflow, "rtpt_max_volume_bounces", 9)

    records = _render_setting_records(workflow)
    assert len(records) == 3
    assert [r["render_setting"]["attribute"] for r in records] == [
        "omni:rtx:rtpt:maxBounces",
        "omni:rtx:rtpt:maxBounces",
        "omni:rtx:rtpt:maxVolumeBounces",
    ]


def test_non_render_setting_edit_has_no_render_setting_entry() -> None:
    # Only render-setting edits carry the RTPT diagnostic entry; ordinary
    # value edits are untouched.
    from ovrtx_blender_example.interactive_edit_planner import (
        DataAuthority,
        EditShape,
        InteractiveEdit,
        edit_location,
    )

    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Light",
            usd_attribute="inputs:intensity",
            provenance={"source": "light_value"},
        ),
        value=5.0,
    )
    workflow.preview_edit(edit)

    assert _render_setting_records(workflow) == []


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
