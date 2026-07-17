# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live RTPT render-setting write rejection fallback (task01-04 follow-up).

The task01-04 primary route applies a live RTPT quality change as a runtime
attribute write on the session-owning render thread. A real OVRTX worker can
*reject* that write (the ``update_attribute_values`` RPC / the shared-stage
composition around it raises), which previously bubbled up as a
``RenderClientError: Shared-stage composition failed: render_setting_value_update_error``
and killed the whole viewport.

These tests pin the fallback:

* the scheduler never lets a rejected render-setting write fail the tick
  (the viewport survives);
* the render loop disables the live route for the session, folds the RTPT
  values back into the composition digest, and re-keys the session exactly
  once so the pending change still takes effect;
* the recomposed layer authors the new value;
* the edit record reports ``applied_via: rekey``;
* the failure is reported once through the user-messages bus as a WARNING,
  never a viewport-killing ERROR;
* a successful live write still works unchanged (happy-path regression).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (  # noqa: E402
    edit_records,
    ovrtx_scene_composition,
    ovrtx_session,
    rtpt_live_change,
    user_messages,
    viewport_handoff,
)
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    EditMechanism,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.interactive_edit_workflow import (  # noqa: E402
    InteractiveEditWorkflow,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxSessionUpdatePort,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    RuntimeScheduler,
    RuntimeTickRequest,
    RuntimeTickStatus,
)
from ovrtx_blender_example.view_update_stream import (  # noqa: E402
    ViewUpdateStream,
    has_non_render_setting_failure,
    render_setting_write_rejection,
)
from ovrtx_blender_example.viewport_handoff import (  # noqa: E402
    FRAME_STATUS_FAILED,
    FRAME_STATUS_FRAME,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (  # noqa: E402
    RENDER_SETTING_UNHONORED_REASON,
    LatestViewRenderLoop,
    SessionLifecycleHooks,
)


RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
MAX_BOUNCES_ATTR = "omni:rtx:rtpt:maxBounces"
WAIT_S = 5.0


# --- Shared helpers ---------------------------------------------------------


def _intent(property_name: str, value):
    edit = rtpt_live_change.render_setting_edit(property_name, value, RENDER_PRODUCT)
    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.UPDATE, plan.unsupported_reason
    return plan.to_intent()


def _disabled_config() -> SimpleNamespace:
    return SimpleNamespace(enabled=False)


class _RejectingPort:
    """A render port whose attribute-value write raises (worker rejection)."""

    def __init__(self) -> None:
        self.attribute_calls = 0

    def update_transforms(self, values):
        return OvrtxValueUpdateResult(len(values), 1 if values else None)

    def update_attribute_values(self, values):
        self.attribute_calls += 1
        raise RenderClientError("render_setting_value_update_error")


class _AcceptingPort:
    def __init__(self) -> None:
        self.batches: list[list[OvrtxAttributeValue]] = []

    def update_transforms(self, values):
        return OvrtxValueUpdateResult(len(values), 1 if values else None)

    def update_attribute_values(self, values):
        self.batches.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


# --- Section A: view_update_stream classification ---------------------------


def test_rejection_classifier_extracts_attributes_and_values() -> None:
    stream = ViewUpdateStream()
    stream.queue(_intent("rtpt_max_bounces", 7))
    result = stream.apply_pending(_RejectingPort())

    assert result["failed"] is True
    rejection = render_setting_write_rejection(result)
    assert rejection is not None
    assert rejection["attributes"] == [MAX_BOUNCES_ATTR]
    # The rejected write carries the wire value that was sent (UI 7 -> wire 9).
    assert rejection["values"] == [
        {"attribute": MAX_BOUNCES_ATTR, "value": 9, "value_type": "Int"}
    ]
    assert rejection["skipped_reason"] == "render_setting_value_update_error"
    assert rejection["render_product_paths"] == [RENDER_PRODUCT]
    # A render-setting-only rejection is not a "non-render-setting" failure.
    assert has_non_render_setting_failure(result) is False


def test_rejection_classifier_ignores_a_successful_write() -> None:
    stream = ViewUpdateStream()
    stream.queue(_intent("rtpt_max_bounces", 7))
    result = stream.apply_pending(_AcceptingPort())

    assert result["values_written"] is True
    assert render_setting_write_rejection(result) is None
    assert has_non_render_setting_failure(result) is False


# --- Section B: scheduler de-fatals the rejected write ----------------------


def test_scheduler_does_not_fail_the_tick_on_a_rejected_render_setting_write() -> None:
    scheduler = RuntimeScheduler(config_factory=lambda _path: _disabled_config())
    assert scheduler.submit_edit(_intent("rtpt_max_bounces", 7)).accepted

    port = _RejectingPort()
    result = scheduler.tick_viewport(
        RuntimeTickRequest(input_usd_path="/tmp/scene.usda"),
        ovrtx_updates=port,
    )

    # The worker rejected the write, but the tick is NOT failed: the viewport
    # survives (no RenderClientError bubbles out of the loop's apply site).
    assert result.status != RuntimeTickStatus.FAILED
    assert result.values_written is False
    assert result.should_reset_refinement is False
    assert port.attribute_calls == 1
    # The rejection is surfaced for the loop's re-key fallback (wire value 9).
    assert result.render_setting_rejected is not None
    assert result.render_setting_rejected["attributes"] == [MAX_BOUNCES_ATTR]
    assert result.render_setting_rejected["values"] == [
        {"attribute": MAX_BOUNCES_ATTR, "value": 9, "value_type": "Int"}
    ]


def test_scheduler_happy_path_write_is_unchanged() -> None:
    scheduler = RuntimeScheduler(config_factory=lambda _path: _disabled_config())
    assert scheduler.submit_edit(_intent("rtpt_max_bounces", 7)).accepted

    port = _AcceptingPort()
    result = scheduler.tick_viewport(
        RuntimeTickRequest(input_usd_path="/tmp/scene.usda"),
        ovrtx_updates=port,
    )

    assert result.values_written is True
    assert result.should_reset_refinement is True
    assert result.render_setting_rejected is None
    # UI 7 -> wire 9.
    assert port.batches == [
        [OvrtxAttributeValue(RENDER_PRODUCT, MAX_BOUNCES_ATTR, 9, "Int")]
    ]


# --- Section C: edit record reports applied_via: rekey ----------------------


class _FakeScheduler:
    def submit_edit(self, intent):
        from ovrtx_blender_example.interactive_edit_planner import EditStatus
        from ovrtx_blender_example.runtime_scheduler import EditSubmissionResult

        return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued", diagnostics={})


def _render_setting_entry(workflow: InteractiveEditWorkflow) -> dict:
    records = [
        record
        for record in workflow.diagnostics()["edit_records"]
        if "render_setting" in record
    ]
    assert len(records) == 1
    return records[0]["render_setting"]


def test_edit_record_reports_applied_via_rekey_when_write_rejected() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    workflow.preview_edit(
        rtpt_live_change.render_setting_edit("rtpt_max_bounces", 7, RENDER_PRODUCT)
    )
    # Pre-apply: the live route is still assumed.
    assert _render_setting_entry(workflow)["applied_via"] == "live"

    # The render thread's tick carries the de-fataled rejection marker.
    rejection = {
        "attributes": [MAX_BOUNCES_ATTR],
        "values": [{"attribute": MAX_BOUNCES_ATTR, "value": 7}],
        "skipped_reason": "render_setting_value_update_error",
    }
    matched = workflow.record_update_result(
        {
            "values_written": False,
            "value_paths": [RENDER_PRODUCT],
            "value_attributes": [MAX_BOUNCES_ATTR],
            "targets": [{"usd_prim_path": RENDER_PRODUCT}],
            "render_setting_rejected": rejection,
        }
    )
    assert matched == 1

    entry = _render_setting_entry(workflow)
    assert entry["applied_via"] == "rekey"
    # The change is applied by re-key, not the (rejected) live write, so warm-up
    # was not confirmed by the live apply.
    assert entry["warmup_completed"] is False


def test_edit_record_keeps_applied_via_live_on_a_successful_write() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_FakeScheduler())
    workflow.preview_edit(
        rtpt_live_change.render_setting_edit("rtpt_max_bounces", 7, RENDER_PRODUCT)
    )

    workflow.record_update_result(
        {
            "values_written": True,
            "value_paths": [RENDER_PRODUCT],
            "value_attributes": [MAX_BOUNCES_ATTR],
            "value_types": ["Int"],
            "targets": [{"usd_prim_path": RENDER_PRODUCT}],
        }
    )

    entry = _render_setting_entry(workflow)
    assert entry["applied_via"] == "live"
    assert entry["warmup_completed"] is True
    assert entry["applied_on_thread"] == "render"


# --- Section D: digest re-key authors the new value -------------------------


def _generated_request(tmp_path: Path, **changes) -> RenderRequest:
    request = RenderRequest(
        input_usd_path=str(tmp_path / "scene.usda"),
        current_scene_generation=True,
        sensor_paths=(RENDER_PRODUCT,),
        selected_sensor_paths=(RENDER_PRODUCT,),
        width=4,
        height=4,
        min_samples=1,
        max_samples=2,
        camera_prim_path="/World/OVRTXCamera",
        camera_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        worker_command="worker",
        native_client_module="client",
    )
    return replace(request, **changes)


def _render_product_block(spec: ovrtx_session.OvrtxSessionSpec) -> str:
    composition = spec.ovrtx_scene_composition
    record = next(
        item
        for item in composition.presentation_layers
        if item["source"] == "viewport_camera_projection"
    )
    text = Path(str(record["path"])).read_text(encoding="utf-8")
    marker = "def RenderProduct"
    assert marker in text, text
    return text[text.index(marker):]


def test_folding_rtpt_into_the_digest_rekeys_and_authors_the_new_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    # The live route excludes RTPT from the digest (a change keeps the session).
    live = ovrtx_session.build_spec(
        _generated_request(
            tmp_path, rtpt_value_route=True, rtpt_quality={"rtpt_max_bounces": 3}
        )
    )
    # The fallback flips the route off (RTPT rejoins the digest) and authors the
    # pending value: reuse_decision now replaces the session.
    rekey = ovrtx_session.build_spec(
        _generated_request(
            tmp_path, rtpt_value_route=False, rtpt_quality={"rtpt_max_bounces": 7}
        )
    )
    decision = ovrtx_session.reuse_decision(live, rekey)
    assert decision.reuse is False
    assert decision.reason == "scene_composition_changed"
    # rtpt_quality carries UI values; the authored line is the wire value
    # (UI 7 -> wire 9).
    assert "int omni:rtx:rtpt:maxBounces = 9" in _render_product_block(rekey)


# --- Section E: render-loop integration (threaded) --------------------------


def _matrix(tx: float):
    return (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _snapshot(tx: float = 2.0, **overrides) -> ViewSnapshot:
    fields = {
        "camera_matrix": _matrix(tx),
        "camera_prim_path": "/World/OVRTXCamera",
        "min_samples": 1,
        "max_samples": 2,
        "selected_sensor_paths": (RENDER_PRODUCT,),
        "width": 4,
        "height": 4,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


class _Client:
    """Fake srtx client; ``reject_attribute_values`` toggles worker rejection."""

    def __init__(self) -> None:
        self.simulation_id = "sim"
        self.starts = 0
        self.deletes = 0
        self.closed = 0
        self.render_calls = 0
        self.attribute_calls = 0
        self.reject_attribute_values = False
        self.attribute_batches: list[tuple] = []
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.starts += 1
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
        return RenderResult(
            width=4,
            height=4,
            rgba8=b"\x00\x00\x00\xff" * 16,
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        batch = tuple(values)
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def update_attribute_values(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        self.attribute_calls += 1
        if self.reject_attribute_values:
            raise RenderClientError("render_setting_value_update_error")
        self.attribute_batches.append(tuple(values))
        return OvrtxValueUpdateResult(len(values), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        self.deletes += 1
        return "stopped"

    def shutdown(self) -> None:
        self.closed += 1


class _RecordingSlot(LatestFrameSlot):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.published: list[FrameState] = []

    def publish(self, frame_state: FrameState) -> FrameState:
        stamped = super().publish(frame_state)
        with self._lock:
            self.published.append(stamped)
        return stamped

    def frames(self) -> list[FrameState]:
        with self._lock:
            return list(self.published)


class _LoopHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.client = _Client()
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work")
        )
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = _generated_request(
            tmp_path, rtpt_value_route=True, rtpt_quality={"rtpt_max_bounces": 3}
        )
        self.scheduler = RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = CameraRequestMailbox()
        self.slot = _RecordingSlot()
        self.ensure_calls: list[RenderRequest] = []

        def _ensure(request: RenderRequest) -> None:
            self.ensure_calls.append(request)
            self.controller.ensure(request)

        self.lifecycle = SessionLifecycleHooks(
            ensure_session=_ensure,
            replacement_reason=lambda request: self.controller.would_replace(request),
            retry_allowed=lambda: True,
        )
        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
            lifecycle=self.lifecycle,
        )


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="rtpt-fallback-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


def _fallback_warnings() -> list[user_messages.UserMessage]:
    return [
        message
        for message in user_messages.default_bus().take_pending()
        if message.level == user_messages.WARNING
        and "Live render-quality updates are not supported" in message.text
    ]


def test_rejected_live_write_survives_and_rekeys_once_authoring_new_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_messages.default_bus().reset()
    harness = _LoopHarness(monkeypatch, tmp_path)
    harness.client.reject_attribute_values = True

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        # Wait for the startup session + first refined frame.
        assert _wait_until(
            lambda: harness.client.starts >= 1
            and any(
                frame.status == FRAME_STATUS_FRAME
                and frame.completed_samples >= 2
                for frame in harness.slot.frames()
            )
        )
        assert harness.client.starts == 1

        # The artist changes Max Bounces to 7: the live write is dispatched to
        # the render thread (queued here as the engine would), and the worker
        # rejects it.
        assert harness.scheduler.submit_edit(_intent("rtpt_max_bounces", 7)).accepted
        assert _wait_until(
            lambda: harness.loop.diagnostics()["session_replacements"] >= 1
        )
        # The re-key restarts the session (the folded RTPT digest changed);
        # let the re-keyed session render.
        assert _wait_until(lambda: harness.client.starts >= 2)
        assert _wait_until(lambda: harness.client.render_calls >= 3)

    diagnostics = harness.loop.diagnostics()
    # The viewport never died: no failure publication, no loop failure latch.
    assert all(frame.status != FRAME_STATUS_FAILED for frame in harness.slot.frames())
    assert diagnostics["failures"] == 0
    assert diagnostics["last_failure_detail"] == ""
    # Live route disabled for the session; exactly one re-key.
    assert diagnostics["rtpt_live_route_supported"] is False
    assert diagnostics["render_setting_rejections"] >= 1
    assert diagnostics["session_replacements"] == 1
    assert diagnostics["last_resync_reason"] == RENDER_SETTING_UNHONORED_REASON
    assert harness.client.starts == 2

    # The re-key request folds RTPT back into the digest and carries the pending
    # value, so the recomposed layer authors it.
    assert len(harness.ensure_calls) == 2
    rekey_request = harness.ensure_calls[1]
    assert rekey_request.rtpt_value_route is False
    # The rejected wire value (9) is folded back to the artist-facing UI value
    # (7) in rtpt_quality; the recomposed layer authors the wire value again (9).
    assert rekey_request.rtpt_quality["rtpt_max_bounces"] == 7
    spec = ovrtx_session.build_spec(rekey_request)
    assert "int omni:rtx:rtpt:maxBounces = 9" in _render_product_block(spec)

    # Reported once through the user-messages bus as a WARNING (not an ERROR).
    warnings = _fallback_warnings()
    assert len(warnings) == 1


def test_successful_live_write_does_not_rekey_or_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_messages.default_bus().reset()
    harness = _LoopHarness(monkeypatch, tmp_path)
    harness.client.reject_attribute_values = False

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(
            lambda: harness.client.starts >= 1
            and any(
                frame.status == FRAME_STATUS_FRAME
                and frame.completed_samples >= 2
                for frame in harness.slot.frames()
            )
        )
        assert harness.scheduler.submit_edit(_intent("rtpt_max_bounces", 7)).accepted
        # The write lands on the render thread and resets refinement.
        assert _wait_until(lambda: harness.client.attribute_calls >= 1)
        assert _wait_until(
            lambda: any(
                batch and batch[0].attribute == MAX_BOUNCES_ATTR
                for batch in harness.client.attribute_batches
            )
        )
        # Give any (erroneous) replacement a chance to appear before asserting.
        time.sleep(0.1)

    diagnostics = harness.loop.diagnostics()
    assert diagnostics["rtpt_live_route_supported"] is True
    assert diagnostics["render_setting_rejections"] == 0
    assert diagnostics["session_replacements"] == 0
    assert harness.client.starts == 1
    assert all(frame.status != FRAME_STATUS_FAILED for frame in harness.slot.frames())
    assert _fallback_warnings() == []


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
