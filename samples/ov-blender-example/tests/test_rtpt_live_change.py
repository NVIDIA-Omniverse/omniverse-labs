# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live RTPT render-quality change application (task01-04).

A quality-setting change on a running viewport session is applied as a runtime
attribute write on the active ``RenderProduct``, executed on the session-owning
render thread, followed by the existing refinement restart (the warm-up). These
plain (non-Blender) tests exercise the value-update plumbing shared with the
material/light lanes:

- the dispatcher builds the exact documented attribute name and dtype;
- rapid successive changes coalesce latest-wins per attribute (slider drag);
- the render-thread apply targets the render product path with the exact dtype
  and, through the scheduler, resets refinement (values_written);
- no runtime write is produced when no session is active.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import rtpt_live_change  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    EditMechanism,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxSessionUpdatePort,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    RuntimeScheduler,
    RuntimeTickRequest,
    RuntimeTickStatus,
)
from ovrtx_blender_example.shared_stage_config import (  # noqa: E402
    InteractiveSharedStageConfig,
)
from ovrtx_blender_example.view_update_stream import ViewUpdateStream  # noqa: E402


RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"


class _RenderPort:
    def __init__(self) -> None:
        self.attribute_updates: list[list[OvrtxAttributeValue]] = []

    def update_attribute_values(self, values):
        self.attribute_updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


class _FakeRenderClient:
    def __init__(self) -> None:
        self.attribute_updates: list[list[OvrtxAttributeValue]] = []

    def update_transforms(self, _session, values):
        return OvrtxValueUpdateResult(len(values), 7 if values else None)

    def update_attribute_values(self, _session, values):
        self.attribute_updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


def _intent(property_name: str, value, *, render_product_path: str = RENDER_PRODUCT):
    edit = rtpt_live_change.render_setting_edit(
        property_name, value, render_product_path
    )
    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.UPDATE, plan.unsupported_reason
    return plan.to_intent()


def _disabled_config() -> InteractiveSharedStageConfig:
    return InteractiveSharedStageConfig(
        enabled=False,
        input_usd_path="/tmp/scene.usda",
        server="/tmp/ovphysx-bridge-server/_build/ovphysx-bridge-server",
        ovphysx_address="127.0.0.1:50094",
        ovphysx_worker_command="server",
        device="cpu",
        body_root="/World/PhysicsIsland/DynamicBodies",
        body_prims=("/World/PhysicsIsland/DynamicBodies/Cube_00",),
        physics_fps=60.0,
        update_fps=60.0,
        max_steps=2,
        body_scale=1.0,
        worker_log_path="/tmp/ovphysx-worker.log",
    )


# --- Dispatcher edit shape (attribute name + exact dtype) --------------------


def test_render_setting_edit_authors_int32_bounce_as_usd_int() -> None:
    edit = rtpt_live_change.render_setting_edit("rtpt_max_bounces", 7, RENDER_PRODUCT)

    assert edit.usd_prim_path == RENDER_PRODUCT
    assert edit.usd_attribute == "omni:rtx:rtpt:maxBounces"
    # The edit carries the wire value OVRTX consumes: Max Bounces UI 7 -> wire 9.
    assert edit.value == 9 and isinstance(edit.value, int)
    # The artist-facing UI value rides provenance for diagnostics.
    assert edit.provenance["ui_value"] == 7
    # int32 -> USD ``Int`` (32-bit), never ``Int64``.
    assert edit.provenance["value_type"] == "Int"
    assert edit.provenance["source"] == "rtpt_render_setting"


def test_render_setting_edit_authors_firefly_filter_as_bool() -> None:
    edit = rtpt_live_change.render_setting_edit(
        "rtpt_firefly_filter_enabled", False, RENDER_PRODUCT
    )

    assert edit.usd_attribute == "omni:rtx:rtpt:fireflyFilter:enabled"
    assert edit.value is False
    assert edit.provenance["value_type"] == "Bool"


def test_value_update_type_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError):
        rtpt_live_change.value_update_type("int64")


# --- No active session -> no runtime write ----------------------------------


def test_no_edit_when_no_active_session() -> None:
    # A change with no active session (no request) authors no runtime write.
    assert (
        rtpt_live_change.render_setting_edit_for_request(
            "rtpt_max_bounces", 7, None
        )
        is None
    )


def test_no_edit_when_request_has_no_render_product_path() -> None:
    request = SimpleNamespace(render_product_path="")
    assert (
        rtpt_live_change.render_setting_edit_for_request(
            "rtpt_max_bounces", 7, request
        )
        is None
    )


def test_edit_for_request_targets_the_active_render_product() -> None:
    request = SimpleNamespace(render_product_path=RENDER_PRODUCT)
    edit = rtpt_live_change.render_setting_edit_for_request(
        "rtpt_max_volume_bounces", 0, request
    )
    assert edit is not None
    assert edit.usd_prim_path == RENDER_PRODUCT
    assert edit.usd_attribute == "omni:rtx:rtpt:maxVolumeBounces"


def test_edit_for_request_resolves_render_product_from_a_real_request() -> None:
    # The engine resolves the render product path from its live
    # ``RenderRequest`` (``render_product_path`` = first selected/declared
    # sensor path); the live write must target that exact prim.
    from ovrtx_blender_example.render_requests import RenderRequest

    request = RenderRequest(
        selected_sensor_paths=(RENDER_PRODUCT,),
        sensor_paths=(RENDER_PRODUCT,),
    )
    edit = rtpt_live_change.render_setting_edit_for_request(
        "rtpt_max_bounces", 7, request
    )
    assert edit is not None
    assert edit.usd_prim_path == request.render_product_path == RENDER_PRODUCT
    assert edit.provenance["value_type"] == "Int"


# --- Render-thread apply: render product path, exact dtype ------------------


def test_stream_applies_render_setting_to_render_product_with_exact_dtypes() -> None:
    stream = ViewUpdateStream()
    port = _RenderPort()

    stream.queue(_intent("rtpt_max_bounces", 7))
    stream.queue(_intent("rtpt_firefly_filter_enabled", False))
    diagnostics = stream.apply_pending(port)

    # Max Bounces UI 7 is sent as wire 9 (+2 camera-ray offset).
    assert port.attribute_updates == [[
        OvrtxAttributeValue(RENDER_PRODUCT, "omni:rtx:rtpt:maxBounces", 9, "Int"),
        OvrtxAttributeValue(
            RENDER_PRODUCT, "omni:rtx:rtpt:fireflyFilter:enabled", False, "Bool"
        ),
    ]]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_paths"] == [RENDER_PRODUCT, RENDER_PRODUCT]
    assert diagnostics["value_attributes"] == [
        "omni:rtx:rtpt:maxBounces",
        "omni:rtx:rtpt:fireflyFilter:enabled",
    ]
    assert diagnostics["value_types"] == ["Int", "Bool"]
    assert diagnostics["accepted_by_worker"] is True


def test_stream_coalesces_rapid_changes_latest_wins_per_attribute() -> None:
    # A slider drag queues several edits for one attribute between ticks; the
    # client rejects duplicate targets in one batch, so the applied batch
    # keeps exactly one latest-wins value per attribute.
    stream = ViewUpdateStream()
    port = _RenderPort()

    stream.queue(_intent("rtpt_max_bounces", 1))
    stream.queue(_intent("rtpt_max_bounces", 2))
    stream.queue(_intent("rtpt_max_volume_bounces", 9))
    stream.queue(_intent("rtpt_max_bounces", 5))
    diagnostics = stream.apply_pending(port)

    # Latest-wins keeps Max Bounces UI 5 (wire 7); the volume sub-cap passes
    # through unchanged (wire 9).
    assert port.attribute_updates == [[
        OvrtxAttributeValue(RENDER_PRODUCT, "omni:rtx:rtpt:maxBounces", 7, "Int"),
        OvrtxAttributeValue(
            RENDER_PRODUCT, "omni:rtx:rtpt:maxVolumeBounces", 9, "Int"
        ),
    ]]
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["value_count"] == 2


def test_scheduler_applies_render_setting_and_resets_refinement() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: _disabled_config()
    )

    assert scheduler.submit_edit(_intent("rtpt_max_bounces", 7)).accepted
    result = scheduler.tick_viewport(
        RuntimeTickRequest(input_usd_path="/tmp/scene.usda"),
        ovrtx_updates=OvrtxSessionUpdatePort(render, "sim"),
    )

    assert result.status == RuntimeTickStatus.NOT_ENABLED
    assert result.values_written is True
    # The applied write resets refinement -> render at min_samples and refine
    # to max_samples (the warm-up) before the new frame is treated as converged.
    assert result.should_reset_refinement is True
    # Max Bounces UI 7 -> wire 9.
    assert render.attribute_updates == [[
        OvrtxAttributeValue(RENDER_PRODUCT, "omni:rtx:rtpt:maxBounces", 9, "Int"),
    ]]


def test_scheduler_reports_no_write_without_a_pending_change() -> None:
    render = _FakeRenderClient()
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: _disabled_config()
    )

    result = scheduler.tick_viewport(
        RuntimeTickRequest(input_usd_path="/tmp/scene.usda"),
        ovrtx_updates=OvrtxSessionUpdatePort(render, "sim"),
    )

    assert result.values_written is False
    assert result.should_reset_refinement is False
    assert render.attribute_updates == []
