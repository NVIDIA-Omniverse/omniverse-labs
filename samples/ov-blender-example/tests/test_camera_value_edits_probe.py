# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera value edits with the live-honor capability probe (task04-05).

Fake-client coverage for every probe outcome: the user's first edit of a
camera value class is the probe (frame-digest comparison at equal samples
and the same snapshot key); honored classes stay on the live value route
for the session, an unhonored class folds its values into the OVRTX scene
composition digest so ``reuse_decision`` forces a background replacement
resync (never a silent no-op), and inconclusive probes retry on the next
edit of the class. A real-worker probe run is deferred to a runtime host
(no OVRTX bundle on this machine) — see memory/TASK-04-05.md.
"""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import camera_value_conversion
from ovrtx_blender_example import ovrtx_scene_composition
from ovrtx_blender_example import ovrtx_session
from ovrtx_blender_example import ovrtx_session_controller as controller_module
from ovrtx_blender_example import viewport_handoff
from ovrtx_blender_example.interactive_edit_planner import (
    DataAuthority,
    EditMechanism,
    EditShape,
    InteractiveEdit,
    InteractiveEditPlanner,
    edit_location,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult
from ovrtx_blender_example.render_requests import (
    ACTIVE_CAMERA_VIEW,
    CameraProjectionState,
    RenderRequest,
)
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler
from ovrtx_blender_example.value_edit_conversion import (
    STATUS_NON_RENDER,
    STATUS_SUPPORTED,
    STATUS_TOPOLOGY,
    STATUS_UNSUPPORTED,
    default_value_edit_conversion_policies,
)
from ovrtx_blender_example.view_update_stream import ViewUpdateStream
from ovrtx_blender_example.viewport_handoff import (
    FRAME_STATUS_FRAME,
    FRAME_STATUS_RESYNCING,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (
    CAMERA_VALUES_UNHONORED_REASON,
    LatestViewRenderLoop,
    SessionLifecycleHooks,
    render_result_digest,
)


WAIT_S = 5.0


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _matrix(tx: float) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _projection(**overrides) -> CameraProjectionState:
    fields = {
        "source": ACTIVE_CAMERA_VIEW,
        "focal_length": 50.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 24.0,
        "projection": "perspective",
        "clipping_range": (0.1, 100.0),
    }
    fields.update(overrides)
    return CameraProjectionState(**fields)


def _request(tmp_path: Path, **overrides) -> RenderRequest:
    fields = {
        "input_usd_path": str(tmp_path / "scene.usda"),
        "sensor_paths": ("/Render/Product",),
        "selected_sensor_paths": ("/Render/Product",),
        "width": 1,
        "height": 1,
        "min_samples": 1,
        "max_samples": 4,
        "camera_prim_path": "/World/Camera",
        "camera_matrix": _matrix(1.0),
        "camera_projection": _projection(),
        "worker_command": "worker",
        "native_client_module": "client",
    }
    fields.update(overrides)
    return RenderRequest(**fields)


def _snapshot(tx: float = 2.0, **overrides) -> ViewSnapshot:
    fields = {
        "camera_matrix": _matrix(tx),
        "camera_prim_path": "/World/Camera",
        "min_samples": 1,
        "max_samples": 4,
        "selected_sensor_paths": ("/Render/Product",),
        "width": 1,
        "height": 1,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


# --- Policy: probe classes and value parity ----------------------------------


def test_probe_class_partition_perspective_and_ortho() -> None:
    for name in ("focalLength", "horizontalAperture", "verticalAperture"):
        assert (
            camera_value_conversion.probe_class_for_attribute(name, "perspective")
            == camera_value_conversion.PROBE_CLASS_PROJECTION
        )
        assert (
            camera_value_conversion.probe_class_for_attribute(name, "orthographic")
            == camera_value_conversion.PROBE_CLASS_ORTHO
        )
    assert (
        camera_value_conversion.probe_class_for_attribute("clippingRange", "perspective")
        == camera_value_conversion.PROBE_CLASS_CLIP
    )
    # Non-probed attributes stay composition identity (replacement route).
    for name in (
        "projection",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fStop",
        "focusDistance",
    ):
        assert camera_value_conversion.probe_class_for_attribute(name) == ""


def test_usd_attribute_values_are_composed_camera_parity() -> None:
    projection = _projection(f_stop=200.0, focus_distance=10.0)
    composed = projection.usd_attributes()
    values = {
        attribute.name: attribute
        for attribute in camera_value_conversion.usd_attribute_values(projection)
    }
    # Exactly the probed attributes, carrying the composed values.
    assert set(values) == {
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "clippingRange",
    }
    for name, attribute in values.items():
        assert attribute.value == (
            tuple(composed[name]) if name == "clippingRange" else composed[name]
        )
        assert (
            attribute.value_type
            == camera_value_conversion.SUPPORTED_USD_ATTRIBUTES[name]
        )
        assert attribute.metadata["probe_class"] == (
            camera_value_conversion.PROBE_CLASS_CLIP
            if name == "clippingRange"
            else camera_value_conversion.PROBE_CLASS_PROJECTION
        )
    assert camera_value_conversion.usd_attribute_values(None) == ()


def test_ortho_projection_values_class_as_ortho() -> None:
    projection = _projection(projection="orthographic")
    values = camera_value_conversion.usd_attribute_values(projection)
    aperture_classes = {
        attribute.name: attribute.metadata["probe_class"]
        for attribute in values
        if attribute.name != "clippingRange"
    }
    assert set(aperture_classes.values()) == {
        camera_value_conversion.PROBE_CLASS_ORTHO
    }
    assert all(
        attribute.blender_property_path == "data.ortho_scale"
        for attribute in values
        if attribute.name in {"horizontalAperture", "verticalAperture"}
    )


def test_classify_field_statuses() -> None:
    perspective = SimpleNamespace(type="PERSP")
    ortho = SimpleNamespace(type="ORTHO")
    classify = camera_value_conversion.classify_field

    assert classify(perspective, "lens").status == STATUS_SUPPORTED
    assert classify(perspective, "lens").usd_attributes == ("focalLength",)
    assert classify(perspective, "sensor_width").usd_attributes == (
        "horizontalAperture",
        "verticalAperture",
    )
    assert classify(perspective, "clip_start").usd_attributes == ("clippingRange",)
    assert classify(ortho, "ortho_scale").status == STATUS_SUPPORTED
    assert classify(perspective, "ortho_scale").status == STATUS_UNSUPPORTED
    assert classify(ortho, "sensor_width").status == STATUS_UNSUPPORTED

    kind = classify(perspective, "type")
    assert kind.status == STATUS_TOPOLOGY
    assert kind.reason == camera_value_conversion.CAMERA_PROJECTION_KIND_CHANGED
    assert classify(perspective, "shift_x").status == STATUS_TOPOLOGY
    assert classify(perspective, "dof.aperture_fstop").status == STATUS_TOPOLOGY
    assert classify(perspective, "use_fake_user").status == STATUS_NON_RENDER
    assert classify(perspective, "mystery_field").status == STATUS_UNSUPPORTED


def test_camera_policy_registered_in_default_policies() -> None:
    policies = default_value_edit_conversion_policies()
    assert policies.camera is camera_value_conversion


def test_probe_state_machine_and_findings_hook() -> None:
    probe = camera_value_conversion.CameraValueProbe()
    all_classes = camera_value_conversion.CAMERA_VALUE_PROBE_CLASSES
    assert probe.value_route_classes() == all_classes
    assert probe.status("projection") == camera_value_conversion.PROBE_STATUS_UNKNOWN

    # Inconclusive keeps the class unknown (retries on the next edit).
    probe.record_inconclusive(
        "projection", camera_value_conversion.PROBE_INCONCLUSIVE_POSE_CHANGED
    )
    assert probe.status("projection") == camera_value_conversion.PROBE_STATUS_UNKNOWN
    assert probe.value_route_classes() == all_classes

    probe.begin_attempt("projection")
    probe.begin_attempt("projection")
    probe.record_result(
        "projection", honored=False, evidence={"pre_frame_digest": "aa"}
    )
    probe.record_result("clip", honored=True, evidence={})
    assert probe.status("projection") == camera_value_conversion.PROBE_STATUS_UNHONORED
    assert probe.status("clip") == camera_value_conversion.PROBE_STATUS_HONORED
    assert probe.value_route_classes() == ("clip", "ortho")

    findings = probe.unhonored_findings()
    assert len(findings) == 1
    finding = findings[0]
    assert finding["probe_class"] == "projection"
    assert finding["status"] == "open"
    assert finding["owner"] == camera_value_conversion.FINDINGS_OWNER
    assert finding["evidence"]["pre_frame_digest"] == "aa"
    assert finding["evidence"]["attempts"] == 2
    assert "camera-projection" in finding["ask"] or "projection" in finding["ask"]

    diagnostics = probe.diagnostics()
    assert diagnostics["unhonored_classes"] == ["projection"]
    assert diagnostics["classes"]["clip"]["status"] == "honored"
    assert len(diagnostics["unhonored_findings"]) == 1


# --- Composition digest folding ----------------------------------------------


def _compose(tmp_path: Path, projection: CameraProjectionState, route=()):
    return ovrtx_scene_composition.compose(
        source_scene_path=str(tmp_path / "scene.usda"),
        camera_prim_path="/World/Camera",
        sensor_paths=("/Render/Product",),
        width=8,
        height=8,
        camera_projection=projection,
        material_scene_layer=None,
        camera_value_route_classes=route,
    )


def test_default_route_keeps_camera_values_in_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    base = _compose(tmp_path, _projection())
    changed = _compose(tmp_path, _projection(focal_length=85.0))
    # Pre-task behavior pinned: without value routing (F12 and direct
    # requests), a focal length change is composition identity.
    assert base.digest != changed.digest


def test_value_route_classes_exclude_probed_values_from_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    route = camera_value_conversion.CAMERA_VALUE_PROBE_CLASSES
    base = _compose(tmp_path, _projection(), route)
    focal = _compose(tmp_path, _projection(focal_length=85.0), route)
    # The layer body always authors the current values regardless of the
    # digest routing (a replacement session composes them fresh); reading
    # immediately after the compose because same-digest composes share
    # the layer path and each rewrite carries its own values.
    layer_path = Path(str(focal.session_layer_identifiers[1]))
    assert "focalLength = 85" in layer_path.read_text(encoding="utf-8")
    clip = _compose(tmp_path, _projection(clipping_range=(0.5, 50.0)), route)
    # Value-routed classes stay out of session identity...
    assert base.digest == focal.digest
    assert base.digest == clip.digest
    # ...while non-probed projection fields remain composition identity.
    dof = _compose(tmp_path, _projection(f_stop=200.0, focus_distance=10.0), route)
    assert base.digest != dof.digest
    # Folding a class back (unhonored) changes identity with the values.
    folded_route = ("clip", "ortho")
    folded = _compose(tmp_path, _projection(focal_length=85.0), folded_route)
    refolded = _compose(tmp_path, _projection(focal_length=35.0), folded_route)
    assert folded.digest != base.digest
    assert folded.digest != refolded.digest


def test_build_spec_reads_route_classes_from_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    route = camera_value_conversion.CAMERA_VALUE_PROBE_CLASSES
    base = ovrtx_session.build_spec(
        _request(tmp_path, camera_value_route_classes=route)
    )
    focal_changed = ovrtx_session.build_spec(
        _request(
            tmp_path,
            camera_projection=_projection(focal_length=85.0),
            camera_value_route_classes=route,
        )
    )
    decision = ovrtx_session.reuse_decision(base, focal_changed)
    assert decision.reuse is True
    folded = ovrtx_session.build_spec(
        _request(
            tmp_path,
            camera_projection=_projection(focal_length=85.0),
            camera_value_route_classes=("clip", "ortho"),
        )
    )
    assert ovrtx_session.reuse_decision(base, folded).reason == (
        "scene_composition_changed"
    )


# --- Planner and update-stream lane -------------------------------------------


def _camera_value_edit(name: str, value, probe_class: str) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Camera",
            usd_attribute=name,
            blender_property_path="data.lens",
            provenance={
                "source": camera_value_conversion.VIEWPORT_CAMERA_PROJECTION_SOURCE,
                "probe_class": probe_class,
            },
        ),
        value=value,
        metadata={"probe_class": probe_class},
    )


def test_camera_value_edit_plans_update() -> None:
    plan = InteractiveEditPlanner().plan(
        _camera_value_edit("focalLength", 85.0, "projection")
    )
    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.impact.render_session_reuse_expected is True


class _RecordingPort:
    def __init__(self) -> None:
        self.attribute_batches: list[tuple] = []

    def update_transforms(self, values):
        raise AssertionError("camera value edits must use the attribute lane")

    def update_attribute_values(self, values):
        batch = tuple(values)
        self.attribute_batches.append(batch)
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)


def test_stream_applies_typed_camera_values() -> None:
    stream = ViewUpdateStream()
    port = _RecordingPort()
    for name, value, probe_class in (
        ("focalLength", 85.0, "projection"),
        ("clippingRange", (0.25, 250.0), "clip"),
    ):
        plan = InteractiveEditPlanner().plan(
            _camera_value_edit(name, value, probe_class)
        )
        stream.queue(plan.to_intent())
    result = stream.apply_pending(port)

    assert result["values_written"] is True
    assert result["value_attributes"] == ["focalLength", "clippingRange"]
    assert result["camera_value_probe_class"] == "projection"
    (batch,) = port.attribute_batches
    focal, clipping = batch
    assert focal.value == 85.0 and focal.value_type == "Float"
    assert clipping.value == [0.25, 250.0] and clipping.value_type == "Float2"
    assert focal.prim_path == "/World/Camera"


def test_stream_rejects_unsupported_camera_attribute() -> None:
    stream = ViewUpdateStream()
    port = _RecordingPort()
    plan = InteractiveEditPlanner().plan(
        _camera_value_edit("fStop", 2.8, "projection")
    )
    stream.queue(plan.to_intent())
    result = stream.apply_pending(port)
    assert result["values_written"] is False
    assert result["failed"] is True
    assert result["skipped_reason"] == "unsupported_camera_value_attribute"
    assert port.attribute_batches == []


# --- Render loop: the first edit is the probe ---------------------------------


class _Client:
    """Fake srtx client whose rendered pixels reflect applied camera values.

    ``honor_camera_values=True`` models a worker that honors live camera
    attribute writes: the frame content derives from the applied values,
    so a post-edit frame digests differently. ``False`` models the
    accepted-but-ignored behavior from the resize spike: the write is
    acknowledged but the pixels never change.
    """

    def __init__(self, *, honor_camera_values: bool) -> None:
        self.honor_camera_values = honor_camera_values
        self.simulation_id = "sim"
        self.starts = 0
        self.deletes = 0
        self.closed = 0
        self.render_calls = 0
        self.transform_update_batches: list[tuple] = []
        self.attribute_update_batches: list[tuple] = []
        self.applied_attribute_values: dict[tuple[str, str], object] = {}
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.starts += 1
        # A fresh session composes the scene fresh: honored-or-not, the
        # composed layer carries the current values (probe state resets
        # with the session content).
        self.applied_attribute_values = {}
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
        seed = (
            repr(sorted(self.applied_attribute_values.items())).encode("utf-8")
            + b":"
            + str(self.starts).encode("utf-8")
        )
        return RenderResult(
            width=1,
            height=1,
            rgba8=hashlib.sha256(seed).digest()[:4],
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        batch = tuple(values)
        self.transform_update_batches.append(batch)
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def update_attribute_values(
        self, simulation_id: str, values
    ) -> OvrtxValueUpdateResult:
        batch = tuple(values)
        self.attribute_update_batches.append(batch)
        if self.honor_camera_values:
            for value in batch:
                self.applied_attribute_values[(value.prim_path, value.attribute)] = (
                    value.value
                )
        return OvrtxValueUpdateResult(len(batch), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        self.deletes += 1
        return "stopped"

    def shutdown(self) -> None:
        self.closed += 1


class _RecordingSlot(LatestFrameSlot):
    def __init__(self) -> None:
        super().__init__()
        self._record_lock = threading.Lock()
        self.published: list[FrameState] = []

    def publish(self, frame_state: FrameState) -> FrameState:
        stamped = super().publish(frame_state)
        with self._record_lock:
            self.published.append(stamped)
        return stamped

    def frames(self) -> list[FrameState]:
        with self._record_lock:
            return list(self.published)

    def statuses(self) -> list[str]:
        return [frame.status for frame in self.frames()]


class _Harness:
    """Real controller/scheduler + lifecycle hooks + value-aware fake client."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        honor_camera_values: bool,
    ) -> None:
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work")
        )
        self.client = _Client(honor_camera_values=honor_camera_values)
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = _request(tmp_path)
        self.scheduler = RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = CameraRequestMailbox()
        self.slot = _RecordingSlot()
        self.ensure_calls: list[RenderRequest] = []

        def _ensure(request: RenderRequest) -> None:
            self.ensure_calls.append(request)
            self.controller.ensure(request)

        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
            lifecycle=SessionLifecycleHooks(
                ensure_session=_ensure,
                replacement_reason=lambda request: self.controller.would_replace(
                    request
                ),
                retry_allowed=lambda: True,
            ),
        )

    def set_projection(self, **overrides) -> None:
        self.base_request = replace(
            self.base_request, camera_projection=_projection(**overrides)
        )

    def frame_count(self) -> int:
        return sum(
            1 for frame in self.slot.frames() if frame.status == FRAME_STATUS_FRAME
        )

    def wait_refined(self, minimum_frames: int) -> None:
        assert _wait_until(
            lambda: self.frame_count() >= minimum_frames
            and any(
                frame.status == FRAME_STATUS_FRAME
                and frame.completed_samples >= self.base_request.max_samples
                for frame in self.slot.frames()[minimum_frames - 1 :]
            )
        ), f"never refined; publications: {self.slot.statuses()}"

    def probe_diagnostics(self) -> dict:
        return self.loop.diagnostics()["camera_value_probe"]

    def camera_attribute_names(self) -> list[str]:
        return [
            value.attribute
            for batch in self.client.attribute_update_batches
            for value in batch
        ]


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="camera-probe-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


def test_honored_class_stays_on_value_route_without_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)
        frames_before = harness.frame_count()

        # The user's first projection edit IS the probe.
        harness.set_projection(focal_length=85.0)
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_HONORED
        ), harness.probe_diagnostics()
        harness.wait_refined(frames_before + 3)

        # A later edit of the honored class re-applies without re-probing.
        harness.set_projection(focal_length=35.0)
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(
            lambda: harness.camera_attribute_names().count("focalLength") >= 2
        )

    # One session for the whole exchange: the value route never replaced.
    assert harness.client.starts == 1
    assert harness.client.deletes == 0
    assert len(harness.ensure_calls) == 1
    applied = [
        value.value
        for batch in harness.client.attribute_update_batches
        for value in batch
        if value.attribute == "focalLength"
    ]
    assert applied == [85.0, 35.0]
    diagnostics = harness.probe_diagnostics()
    assert diagnostics["classes"]["projection"]["attempts"] == 1
    evidence = diagnostics["classes"]["projection"]["evidence"]
    assert evidence["pre_frame_digest"] != evidence["post_frame_digest"]
    assert evidence["compared_samples"] == 1
    assert diagnostics["unhonored_findings"] == []
    # Probe state appears in the loop's session diagnostics.
    assert harness.loop.diagnostics()["camera_value_update_count"] == 2
    # The refined publications restart at min samples after each edit
    # (values_written -> refinement reset).
    statuses = harness.slot.statuses()
    assert FRAME_STATUS_RESYNCING not in statuses


def test_camera_probe_waits_for_its_comparison_sample_across_acquisitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    harness.base_request = replace(
        harness.base_request,
        min_samples=4,
        max_samples=4,
    )
    snapshot = _snapshot(2.0, min_samples=4, max_samples=4)

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        harness.wait_refined(4)
        frames_before = harness.frame_count()

        harness.set_projection(focal_length=85.0)
        harness.mailbox.write(snapshot)
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_HONORED
        ), harness.probe_diagnostics()
        harness.wait_refined(frames_before + 4)

    evidence = harness.probe_diagnostics()["classes"]["projection"]["evidence"]
    assert evidence["compared_samples"] == 4
    assert harness.probe_diagnostics()["classes"]["projection"][
        "last_inconclusive_reason"
    ] == ""


def test_unhonored_class_folds_into_digest_and_resyncs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=False)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        harness.set_projection(focal_length=85.0)
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_UNHONORED
        ), harness.probe_diagnostics()
        # The accepted-but-ignored write falls back to a background
        # replacement resync — the edit still renders (never a no-op).
        assert _wait_until(lambda: harness.client.starts >= 2)
        assert _wait_until(
            lambda: FRAME_STATUS_RESYNCING in harness.slot.statuses()
        )
        replacement_frames = harness.frame_count()

        # Further edits of the unhonored class skip the value route
        # entirely: the folded digest forces the replacement per edit.
        attribute_writes = len(harness.camera_attribute_names())
        harness.set_projection(focal_length=24.0)
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(lambda: harness.client.starts >= 3)
        assert _wait_until(lambda: harness.frame_count() > replacement_frames)

    assert harness.client.starts == 3
    # No further camera attribute writes after the unhonored conclusion.
    assert len(harness.camera_attribute_names()) == attribute_writes
    # The replacement ensure folded the class out of the value route.
    folded_request = harness.ensure_calls[1]
    assert "projection" not in folded_request.camera_value_route_classes
    assert set(folded_request.camera_value_route_classes) == {"clip", "ortho"}
    # The unhonored ensure carries the probe's replacement reason.
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["session_replacements"] >= 2
    probe = harness.probe_diagnostics()
    evidence = probe["classes"]["projection"]["evidence"]
    assert evidence["pre_frame_digest"] == evidence["post_frame_digest"]
    findings = probe["unhonored_findings"]
    assert len(findings) == 1
    assert findings[0]["owner"] == "Team Green"
    assert findings[0]["status"] == "open"
    assert findings[0]["probe_class"] == "projection"


def test_clip_class_probes_independently_of_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        harness.set_projection(clipping_range=(0.5, 50.0))
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["clip"]["status"]
            == camera_value_conversion.PROBE_STATUS_HONORED
        )

    probe = harness.probe_diagnostics()
    assert (
        probe["classes"]["projection"]["status"]
        == camera_value_conversion.PROBE_STATUS_UNKNOWN
    )
    assert harness.camera_attribute_names() == ["clippingRange"]
    assert harness.client.starts == 1


def test_edit_with_simultaneous_view_change_is_inconclusive_then_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        # Pose and focal length change together: the probe cannot compare
        # frames across snapshot keys, so it stays unknown — but the edit
        # itself still applies (no silent drop of the user's value).
        harness.set_projection(focal_length=85.0)
        harness.mailbox.write(_snapshot(9.0))
        assert _wait_until(
            lambda: "focalLength" in harness.camera_attribute_names()
        )
        assert _wait_until(
            lambda: bool(
                harness.probe_diagnostics()["classes"]["projection"][
                    "last_inconclusive_reason"
                ]
            )
        )
        probe = harness.probe_diagnostics()
        assert (
            probe["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_UNKNOWN
        )

        # The next isolated edit of the class retries and concludes.
        harness.wait_refined(6)
        harness.set_projection(focal_length=35.0)
        harness.mailbox.write(_snapshot(9.0))
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_HONORED
        ), harness.probe_diagnostics()

    assert harness.client.starts == 1


def test_two_classes_changing_together_defer_both_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        harness.set_projection(focal_length=85.0, clipping_range=(0.5, 50.0))
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(
            lambda: {"focalLength", "clippingRange"}
            <= set(harness.camera_attribute_names())
        )

    probe = harness.probe_diagnostics()
    for probe_class in ("projection", "clip"):
        assert (
            probe["classes"][probe_class]["status"]
            == camera_value_conversion.PROBE_STATUS_UNKNOWN
        )
        assert probe["classes"][probe_class]["last_inconclusive_reason"] == (
            camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS
        )
    assert harness.client.starts == 1


def test_non_probed_camera_fields_take_replacement_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        # DOF is composition identity (not a probe class): the change
        # replaces the session as before this task.
        harness.set_projection(f_stop=200.0, focus_distance=10.0)
        harness.mailbox.write(_snapshot(2.0))
        assert _wait_until(lambda: harness.client.starts >= 2)
        assert _wait_until(
            lambda: FRAME_STATUS_RESYNCING in harness.slot.statuses()
        )

    assert harness.camera_attribute_names() == []
    probe = harness.probe_diagnostics()
    assert probe["unhonored_classes"] == []


def test_unhonored_conclusion_resyncs_even_when_fully_refined(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With ``min_samples == max_samples`` the post-edit frame is already
    fully refined when the probe concludes unhonored — the scheduled
    replacement must still run promptly with no further input (the edit
    submission's wake latch plus the ensure-pending work/timeout policy),
    or the ignored edit would stay presented until the next user input."""

    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=False)
    harness.base_request = replace(
        harness.base_request, min_samples=1, max_samples=1
    )
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0, min_samples=1, max_samples=1))
        assert _wait_until(lambda: harness.frame_count() >= 1)

        harness.set_projection(focal_length=85.0)
        harness.base_request = replace(
            harness.base_request, min_samples=1, max_samples=1
        )
        harness.mailbox.write(_snapshot(2.0, min_samples=1, max_samples=1))
        # No further mailbox input: the unhonored conclusion alone must
        # drive the replacement resync.
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"]["status"]
            == camera_value_conversion.PROBE_STATUS_UNHONORED
        ), harness.probe_diagnostics()
        assert _wait_until(lambda: harness.client.starts >= 2)
        assert _wait_until(
            lambda: FRAME_STATUS_RESYNCING in harness.slot.statuses()
        )


def test_probe_defers_while_timeline_is_playing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review regression (task04-05): playback starting in the same tick
    as the edit can change frame content for non-camera reasons, so the
    probe defers (the edit still applies)."""

    harness = _Harness(monkeypatch, tmp_path, honor_camera_values=True)
    with _running(harness.loop):
        harness.mailbox.write(_snapshot(2.0))
        harness.wait_refined(3)

        harness.set_projection(focal_length=85.0)
        harness.mailbox.write(
            _snapshot(2.0, timeline_controls_enabled=True, timeline_playing=True)
        )
        assert _wait_until(
            lambda: "focalLength" in harness.camera_attribute_names()
        )
        assert _wait_until(
            lambda: harness.probe_diagnostics()["classes"]["projection"][
                "last_inconclusive_reason"
            ]
            == camera_value_conversion.PROBE_INCONCLUSIVE_PHYSICS_ACTIVE
        ), harness.probe_diagnostics()

    assert (
        harness.probe_diagnostics()["classes"]["projection"]["status"]
        == camera_value_conversion.PROBE_STATUS_UNKNOWN
    )


# --- Engine seam: authored ensure carries the loop's route classes ------------


def test_begin_probe_guards_record_inconclusive_reasons(tmp_path: Path) -> None:
    loop = LatestViewRenderLoop(
        mailbox=CameraRequestMailbox(),
        frame_slot=LatestFrameSlot(),
        controller=SimpleNamespace(),
        scheduler=SimpleNamespace(has_pending_view_updates=False),
        request_for_snapshot=lambda snapshot: snapshot,
    )
    probe = loop._camera_value_probe
    snapshot = _snapshot(2.0)
    loop._snapshot = snapshot
    loop._snapshot_key = snapshot.key

    # No baseline at all.
    assert loop._begin_camera_probe("projection", []) is None
    assert (
        probe.diagnostics()["classes"]["projection"]["last_inconclusive_reason"]
        == camera_value_conversion.PROBE_INCONCLUSIVE_NO_BASELINE
    )

    # Baseline captured at a different snapshot key: the pose changed.
    other = _snapshot(9.0)
    result = RenderResult(
        width=1,
        height=1,
        rgba8=b"\x01\x02\x03\x04",
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=0,
    )
    loop._baseline_min_frame = {"key": other.key, "samples": 1, "result": result}
    assert loop._begin_camera_probe("projection", []) is None
    assert (
        probe.diagnostics()["classes"]["projection"]["last_inconclusive_reason"]
        == camera_value_conversion.PROBE_INCONCLUSIVE_POSE_CHANGED
    )

    # Physics playback advancing content defers the probe.
    loop._baseline_min_frame = {"key": snapshot.key, "samples": 1, "result": result}
    loop._tick_should_request_redraw = True
    assert loop._begin_camera_probe("projection", []) is None
    assert (
        probe.diagnostics()["classes"]["projection"]["last_inconclusive_reason"]
        == camera_value_conversion.PROBE_INCONCLUSIVE_PHYSICS_ACTIVE
    )

    # All guards clear: the probe starts with the pre-edit frame.
    loop._tick_should_request_redraw = False
    context = loop._begin_camera_probe("projection", [])
    assert context is not None
    assert context["pre_result"] is result
    assert context["pre_samples"] == 1
    assert probe.status("projection") == camera_value_conversion.PROBE_STATUS_UNKNOWN


def test_render_result_digest_separates_content_and_shape() -> None:
    def _result(rgba8: bytes, width: int = 1) -> RenderResult:
        return RenderResult(
            width=width,
            height=1,
            rgba8=rgba8,
            completed_samples=1,
            session_completed_samples=1,
            simulation_time_ns=0,
        )

    assert render_result_digest(_result(b"aaaa")) == render_result_digest(
        _result(b"aaaa")
    )
    assert render_result_digest(_result(b"aaaa")) != render_result_digest(
        _result(b"bbbb")
    )
    assert render_result_digest(_result(b"aaaa", width=1)) != render_result_digest(
        _result(b"aaaa", width=2)
    )
