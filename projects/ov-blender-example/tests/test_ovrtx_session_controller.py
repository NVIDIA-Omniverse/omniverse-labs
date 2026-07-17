# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session_controller as controller_module
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult
from ovrtx_blender_example.render_requests import MaterialPresentationLayer, RenderRequest
from ovrtx_blender_example.runtime_scheduler import RuntimeTickResult, RuntimeTickStatus
from ovrtx_blender_example.viewport_render_thread import ViewportRenderThread


class _Client:
    def __init__(
        self,
        simulation_id: str,
        *,
        fail_start: bool = False,
        fail_render: bool = False,
        delete_status: str = "stopped",
    ) -> None:
        self.simulation_id = simulation_id
        self.fail_start = fail_start
        self.fail_render = fail_render
        self.delete_status = delete_status
        self.deletes = 0
        self.closed = 0
        self.starts = 0
        self.render_calls = 0
        self.call_thread_idents: list[int] = []
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings = {"native_render_ms": 2.5}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.call_thread_idents.append(threading.get_ident())
        self.starts += 1
        if self.fail_start:
            self.startup_diagnostics = {"render_worker": {"status": "failed"}}
            raise RenderClientError("start failed")
        return simulation_id or self.simulation_id

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.call_thread_idents.append(threading.get_ident())
        self.render_calls += 1
        if self.fail_render:
            raise RenderClientError("render failed")
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=42,
        )

    def shutdown(self) -> None:
        self.closed += 1

    def delete_simulation(self, simulation_id: str) -> str:
        self.call_thread_idents.append(threading.get_ident())
        assert simulation_id == self.simulation_id
        self.deletes += 1
        return self.delete_status


def _request(tmp_path: Path, **changes: object) -> RenderRequest:
    return replace(
        RenderRequest(
            input_usd_path=str(tmp_path / "scene.usda"),
            sensor_paths=("/Render/Product",),
            selected_sensor_paths=("/Render/Product",),
            width=1,
            height=1,
            min_samples=1,
            max_samples=4,
            camera_prim_path="/World/Camera",
            worker_command="worker",
            native_client_module="client",
        ),
        **changes,
    )


def _factory(monkeypatch: pytest.MonkeyPatch, clients: list[_Client]) -> None:
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda request: clients.pop(0),
    )


def test_suspend_deletes_the_simulation_and_the_next_ensure_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F12 borrows the single-simulation worker from the viewport session:
    suspend deletes the simulation but keeps the controller open, and the
    next ensure recreates the session as a replacement."""

    first = _Client("sim")
    second = _Client("sim")
    _factory(monkeypatch, [first, second])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)

    assert controller.suspend() == "stopped"

    # Simulation deleted and client shut down, but the controller stays
    # open and reports the replacement trigger the render loop probes.
    assert first.deletes == 1
    assert first.closed == 1
    assert controller.would_replace(request) == "no_active_session"
    # A second suspend with nothing active is a no-op.
    assert controller.suspend() == "not_found"

    result = controller.ensure(request)

    assert result.session_started is True
    assert second.starts == 1
    events = [event["event"] for event in controller.diagnostics()["lifecycle_events"]]
    assert events[-1] == "replaced"


def test_would_replace_probes_reuse_policy_without_touching_the_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)

    # No active session yet.
    assert controller.would_replace(request) == "no_active_session"

    controller.ensure(request)
    # Reusable request: no replacement, and the probe performed no RPCs.
    assert controller.would_replace(request) == ""
    # reuse_decision blockers surface as reasons (priority order intact).
    assert controller.would_replace(replace(request, width=2)) == "output_shape_changed"
    assert (
        controller.would_replace(replace(request, camera_prim_path="/World/Other"))
        == "camera_prim_changed"
    )
    assert (
        controller.would_replace(replace(request, worker_command="other-worker"))
        == "runtime_binding_changed"
    )
    # The probe mutated nothing: the original request still reuses and the
    # fake client saw exactly one session start.
    assert controller.would_replace(request) == ""
    assert client.starts == 1
    assert client.deletes == 0


def test_ensure_reuses_worker_across_replacement_and_shutdown_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_client = _Client("first")
    second = _Client("second")
    _factory(monkeypatch, [first_client, second])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)

    first = controller.ensure(request)
    reused = controller.ensure(request)
    reuse_timings = controller._ensure_timings_snapshot()
    replacement = controller.ensure(replace(request, width=2))

    assert first.session_started is True
    assert reused.session_started is False
    assert replacement.session_started is True
    assert reused.composition == first.composition
    assert replacement.composition != first.composition
    assert set(reuse_timings) == {
        "total_ms",
        "build_spec_ms",
        "reuse_decision_ms",
        "other_ms",
    }
    assert all(value >= 0.0 for value in reuse_timings.values())
    assert reuse_timings["total_ms"] == pytest.approx(
        reuse_timings["build_spec_ms"]
        + reuse_timings["reuse_decision_ms"]
        + reuse_timings["other_ms"]
    )
    assert first_client.closed == 0
    diagnostics = controller.diagnostics()
    assert diagnostics["simulation_id"] == "first"
    assert [event["event"] for event in diagnostics["lifecycle_events"]] == [
        "created",
        "stopped",
        "replaced",
    ]
    controller.shutdown()
    controller.shutdown()
    assert first_client.closed == 1
    assert second.closed == 0
    with pytest.raises(RenderClientError, match="shut down"):
        controller.ensure(request)


def test_lifecycle_diagnostics_retain_recent_events_without_growing_forever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)

    for index in range(100):
        controller.ensure(replace(request, width=2 + index))

    events = controller.diagnostics()["lifecycle_events"]
    assert len(events) < 201
    assert events[0]["event"] != "created"
    assert events[-1]["event"] == "replaced"


def test_unconfirmed_delete_retains_active_session_and_blocks_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Client("first", delete_status="failed")
    _factory(monkeypatch, [first])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)

    with pytest.raises(RenderClientError, match="deletion was not confirmed"):
        controller.ensure(replace(request, width=2))

    assert first.deletes == 1
    assert first.closed == 0
    assert controller.diagnostics()["active"] is True
    assert controller.shutdown() == "failed"


def test_projection_failure_invalidates_session_and_next_session_projects_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Client("first")
    second = _Client("second")
    third = _Client("third")
    _factory(monkeypatch, [first, second, third])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)
    controller.ensure(replace(request, width=2))
    projection_requests: list[bool] = []
    wakes: list[None] = []
    controller._attach_presentation(1, lambda: wakes.append(None), None)
    wakes.clear()
    revision = controller._session_revision

    result = controller.apply_runtime_updates(
        lambda port, project: (
            projection_requests.append(project)
            or RuntimeTickResult(
                status=RuntimeTickStatus.FAILED,
                enabled=True,
                complete_pose_projected=False,
            )
        )
    )

    assert result.complete_pose_projected is False
    assert projection_requests == [True]
    assert controller._session_revision > revision
    assert wakes == [None]
    invalidated_revision = controller._session_revision
    assert first.closed == 1
    with pytest.raises(RenderClientError, match="No active"):
        controller.render(request, additional_samples=1)
    wakes.clear()
    controller.ensure(replace(request, width=3))
    assert controller._session_revision > invalidated_revision
    assert wakes == [None]
    controller.apply_runtime_updates(
        lambda port, project: (
            projection_requests.append(project)
            or RuntimeTickResult(status=RuntimeTickStatus.NOOP, enabled=True)
        )
    )
    assert projection_requests == [True, True]


def test_projection_success_clears_replacement_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _factory(monkeypatch, [_Client("first"), _Client("second")])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)
    controller.ensure(replace(request, width=2))
    projection_requests: list[bool] = []

    for outcome in (True, None):
        controller.apply_runtime_updates(
            lambda port, project, outcome=outcome: (
                projection_requests.append(project)
                or RuntimeTickResult(
                    status=RuntimeTickStatus.NOOP,
                    enabled=True,
                    complete_pose_projected=outcome,
                )
            )
        )

    assert projection_requests == [True, False]


def test_render_acquires_explicit_samples_without_progress_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)

    first = controller.render(request, additional_samples=2)
    second = controller.render(request, additional_samples=3)
    controller.apply_runtime_updates(
        lambda port, project: RuntimeTickResult(
            status=RuntimeTickStatus.NOOP,
            enabled=True,
            should_reset_refinement=True,
        )
    )
    reset = controller.render(request, additional_samples=1)

    assert (
        first.completed_samples,
        second.completed_samples,
        reset.completed_samples,
    ) == (2, 3, 1)
    diagnostics = controller.diagnostics()
    assert "refinement" not in diagnostics
    assert diagnostics["lifecycle_events"][0]["event"] == "created"


def test_raw_acquisition_does_not_own_presentation_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    base = _request(tmp_path)
    first = replace(base, blender_signal={"engine_id": "pane-a"})
    second = replace(base, blender_signal={"engine_id": "pane-b"})
    controller.ensure(first)

    results = (
        controller.render(first, additional_samples=2),
        controller.render(second, additional_samples=1),
        controller.render(first, additional_samples=3),
    )

    assert [result.completed_samples for result in results] == [2, 1, 3]
    assert not {
        "_current_result",
        "_snapshot_key",
        "_snapshot_index",
        "_presentation_states",
    }.intersection(vars(controller))


def test_replacement_start_failure_closes_preserved_client_and_leaves_no_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Client("first")
    recovered = _Client("recovered")
    _factory(monkeypatch, [first, recovered])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)
    first.fail_start = True

    with pytest.raises(RenderClientError, match="start failed"):
        controller.ensure(replace(request, width=2))

    assert first.closed == 1
    assert controller.diagnostics()["active"] is False
    controller.ensure(replace(request, width=2))
    projection_requests: list[bool] = []
    controller.apply_runtime_updates(
        lambda port, project: (
            projection_requests.append(project)
            or RuntimeTickResult(status=RuntimeTickStatus.NOOP, enabled=True)
        )
    )
    assert projection_requests == [True]


def test_spec_construction_failure_preserves_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)

    def fail_build_spec(request: object) -> object:
        raise ValueError("build spec failed")

    monkeypatch.setattr(
        controller_module.ovrtx_session,
        "build_spec",
        fail_build_spec,
    )

    with pytest.raises(ValueError, match="build spec failed"):
        controller.ensure(replace(request, width=2))

    assert controller.diagnostics()["active"] is True
    assert client.closed == 0
    assert controller.render(request, additional_samples=1).completed_samples == 1


def test_scheduler_and_render_failures_preserve_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim", fail_render=True)
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)
    calls = 0

    def fail_update(port: object, project: bool) -> RuntimeTickResult:
        nonlocal calls
        calls += 1
        raise RuntimeError("scheduler failed")

    with pytest.raises(RuntimeError, match="scheduler failed"):
        controller.apply_runtime_updates(fail_update)
    with pytest.raises(RenderClientError, match="render failed"):
        controller.render(request, additional_samples=1)

    assert calls == 1
    assert controller.diagnostics()["active"] is True
    assert client.closed == 0


def test_operations_require_an_ensured_session() -> None:
    controller = controller_module.OvrtxSessionController()

    with pytest.raises(RenderClientError, match="No active"):
        controller.render(RenderRequest(), additional_samples=1)
    with pytest.raises(RenderClientError, match="No active"):
        controller.apply_runtime_updates(
            lambda port, project: RuntimeTickResult(
                status=RuntimeTickStatus.NOOP,
                enabled=True,
            )
        )


def test_diagnostics_returns_independent_nested_evidence() -> None:
    controller = controller_module.OvrtxSessionController()

    snapshot = controller.diagnostics()
    snapshot["startup"]["render_worker"]["status"] = "tampered"

    assert controller.diagnostics()["startup"]["render_worker"]["status"] == "not_started"


def test_hot_path_timing_snapshots_exclude_full_controller_diagnostics() -> None:
    controller = controller_module.OvrtxSessionController()
    controller._render_timings = {"native_timings": {"render_ms": 1.0}}
    controller._value_update_timings = {"native_timings": {"update_ms": 2.0}}

    render = controller._render_timings_snapshot()
    update = controller._value_update_timings_snapshot()
    render["native_timings"]["render_ms"] = 99.0
    update["native_timings"]["update_ms"] = 99.0

    assert controller._render_timings_snapshot() == {
        "native_timings": {"render_ms": 1.0}
    }
    assert controller._value_update_timings_snapshot() == {
        "native_timings": {"update_ms": 2.0}
    }


def test_controller_diagnostics_excludes_verbose_material_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _factory(monkeypatch, [_Client("sim")])
    controller = controller_module.OvrtxSessionController()
    entry = MaterialPresentationLayer(
        target_path="/World/Geom",
        layer_body='def Scope "OVRTX_Materials"\n{\n}\n',
        authored_properties=(("/World/Geom", "material:binding"),),
        digest_content={
            "source": "materialx_openpbr",
            "digest": "material-digest",
            "layer_body": 'def Scope "OVRTX_Materials"\n{\n}\n',
        },
        diagnostics={
            "source": "materialx_openpbr",
            "digest": "material-digest",
            "status": "generated",
            "materials": [{"node_inventory": [{"name": "Principled BSDF"}] * 100}],
        },
    )

    small_layer = replace(
        entry,
        diagnostics={
            "source": "materialx_openpbr",
            "digest": "material-digest",
            "status": "generated",
            "materials": [{"node_inventory": [{"name": "Principled BSDF"}]}],
        },
    )
    first = controller.ensure(_request(tmp_path, material_scene_layer=small_layer))
    first_composition = controller.diagnostics()["ovrtx_scene_composition"]
    reused = controller.ensure(_request(tmp_path, material_scene_layer=entry))

    second_composition = controller.diagnostics()["ovrtx_scene_composition"]
    record = second_composition["presentation_layers"][0]
    assert first.session_started is True
    assert reused.session_started is False
    assert first_composition == second_composition
    assert record["source"] == "materialx_openpbr"
    assert "materials" not in record


def test_all_client_calls_occur_on_the_adopted_render_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(controller_module.RPC_THREAD_GUARD_ENV, "1")
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    thread = ViewportRenderThread("controller-confinement")
    thread.start()
    try:
        owning_ident = thread.call(controller.adopt_owning_thread).result(5.0)
        ensured = thread.call(lambda: controller.ensure(request)).result(5.0)
        result = thread.call(
            lambda: controller.render(request, additional_samples=1)
        ).result(5.0)
        status = thread.call(controller.shutdown).result(5.0)
    finally:
        outcome = thread.stop()
    assert outcome["joined"] is True
    assert ensured.session_started is True
    assert result.completed_samples == 1
    assert status == "stopped"
    assert client.call_thread_idents  # start_session, render, delete
    assert set(client.call_thread_idents) == {owning_ident}
    assert owning_ident != threading.get_ident()
    rpc_thread = controller.diagnostics()["rpc_thread"]
    assert rpc_thread == {
        "owning_thread_ident": owning_ident,
        "adopted": True,
        "guard_active": True,
    }


def test_debug_guard_raises_on_foreign_thread_rpc_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(controller_module.RPC_THREAD_GUARD_ENV, "1")
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    controller.adopt_owning_thread(thread_ident=threading.get_ident() + 1)

    for operation in (
        lambda: controller.ensure(_request(tmp_path)),
        lambda: controller.render(_request(tmp_path), additional_samples=1),
        lambda: controller.apply_runtime_updates(
            lambda port, project: RuntimeTickResult(
                status=RuntimeTickStatus.NOOP,
                enabled=True,
            )
        ),
        controller.deactivate,
        controller.shutdown,
    ):
        with pytest.raises(
            controller_module.OvrtxThreadConfinementError,
            match="confined to the owning render thread",
        ):
            operation()

    assert client.starts == 0
    assert client.deletes == 0


def test_thread_guard_is_skipped_when_debug_env_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(controller_module.RPC_THREAD_GUARD_ENV, raising=False)
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    controller.adopt_owning_thread(thread_ident=threading.get_ident() + 1)

    ensured = controller.ensure(_request(tmp_path))

    assert ensured.session_started is True
    assert controller.diagnostics()["rpc_thread"]["guard_active"] is False
    assert controller.shutdown() == "stopped"


def test_confinement_error_flows_through_existing_render_client_error_path() -> None:
    assert issubclass(
        controller_module.OvrtxThreadConfinementError,
        RenderClientError,
    )


def test_requested_simulation_id_pins_every_session_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned simulation-ID lane (task05-01) rides every start_session.

    The F12 final render constructs its controller with the
    ``ovrtx-blender-final-<pid>`` lane so its session on the shared worker
    never collides with the viewport simulation; teardown deletes exactly
    that ID.
    """

    client = _Client("ovrtx-blender-final-123")
    starts: list[str | None] = []
    original_start = client.start_session

    def _recording_start(spec: object, simulation_id: str | None = None) -> str:
        starts.append(simulation_id)
        return original_start(spec, simulation_id)

    client.start_session = _recording_start
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController(
        simulation_id="ovrtx-blender-final-123"
    )

    ensured = controller.ensure(_request(tmp_path))

    assert ensured.session_started is True
    assert starts == ["ovrtx-blender-final-123"]
    assert controller.diagnostics()["simulation_id"] == "ovrtx-blender-final-123"
    assert controller.shutdown() == "stopped"
    assert client.deletes == 1


def test_default_controller_keeps_client_default_simulation_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    starts: list[str | None] = []
    original_start = client.start_session

    def _recording_start(spec: object, simulation_id: str | None = None) -> str:
        starts.append(simulation_id)
        return original_start(spec, simulation_id)

    client.start_session = _recording_start
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()

    controller.ensure(_request(tmp_path))

    assert starts == [None]
    assert controller.diagnostics()["simulation_id"] == "sim"


def test_serialized_presentations_acquire_transport_in_fifo_order() -> None:
    controller = controller_module.OvrtxSessionController()
    order: list[str] = []

    def _enter(label: str) -> None:
        with controller._serialized_transport():
            order.append(label)

    with controller._serialized_transport():
        first = threading.Thread(target=_enter, args=("first",), daemon=True)
        second = threading.Thread(target=_enter, args=("second",), daemon=True)
        first.start()
        deadline = time.monotonic() + 1.0
        while len(controller._transport_waiters) < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        second.start()
        deadline = time.monotonic() + 1.0
        while len(controller._transport_waiters) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(controller._transport_waiters) == 2

    first.join(1.0)
    second.join(1.0)
    assert not first.is_alive() and not second.is_alive()
    assert order == ["first", "second"]


def test_session_validation_keeps_publication_atomic_with_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client("sim")
    _factory(monkeypatch, [client])
    controller = controller_module.OvrtxSessionController()
    request = _request(tmp_path)
    controller.ensure(request)
    revision = controller._session_revision
    entered = threading.Event()
    finished = threading.Event()

    def _replace() -> None:
        entered.set()
        controller.ensure(replace(request, width=2))
        finished.set()

    with controller._validated_session(revision) as valid:
        replacement = threading.Thread(target=_replace, daemon=True)
        replacement.start()
        assert entered.wait(1.0)
        time.sleep(0.02)
        assert valid is True
        assert not finished.is_set()

    replacement.join(1.0)
    assert not replacement.is_alive()
    assert controller._session_revision != revision
