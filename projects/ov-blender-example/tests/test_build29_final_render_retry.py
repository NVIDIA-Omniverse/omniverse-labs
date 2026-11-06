# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""F12 retries a failed job once after a worker failure (build 29 review).

When the renderer worker is lost or reports a fatal device error and the
session's own recovery is exhausted, both F12 routes restart the whole
final-render job once: the current-scene route deactivates and re-activates
the generation runtime, and the dedicated-controller routes (standalone and
viewport-hosted) submit the job again with a fresh controller. A user cancel
is never retried, and a second failure is reported.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.direct_runtime_session import (  # noqa: E402
    RuntimeFatalDeviceError,
    RuntimeWorkerError,
    RuntimeWorkerLost,
    is_worker_failure,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402


WAIT_S = 5.0
SCENE_UID = 77
RESTARTING = ("ovrtx", "Restarting renderer")


def _worker_lost(message: str = "renderer render: [WinError 10054] connection reset") -> RenderClientError:
    """A lost worker as it reaches the engine: the client wraps the session error."""
    try:
        raise RuntimeWorkerLost(f"{message}; recovery failed: Renderer recovery budget exhausted (2)")
    except RuntimeWorkerLost as cause:
        try:
            raise RenderClientError(f"Public SDK render: {cause}") from cause
        except RenderClientError as error:
            return error


def _device_lost() -> RenderClientError:
    try:
        raise RuntimeFatalDeviceError("renderer render: VK_ERROR_DEVICE_LOST")
    except RuntimeFatalDeviceError as cause:
        try:
            raise RenderClientError(f"Public SDK render: {cause}") from cause
        except RenderClientError as error:
            return error


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def _matrix(tx: float) -> tuple[tuple[float, ...], ...]:
    return ((1.0, 0.0, 0.0, float(tx)), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _request(tmp_path: Path, **overrides) -> RenderRequest:
    fields = dict(
        input_usd_path=str(tmp_path / "scene.usda"),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=4,
        height=4,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        camera_matrix=None,
        worker_command="worker",
        native_client_module="client",
    )
    fields.update(overrides)
    return RenderRequest(**fields)


class _Client:
    """Fake renderer client; ``failures`` lists errors raised by successive final renders."""

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        self.started: list[str | None] = []
        self.render_calls = 0
        # Samples accumulated per simulation, as the worker reports them.
        self.accumulated: dict[str, int] = {}
        self.shutdowns = 0
        self.startup_diagnostics = {"render_worker": {"status": "running"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        self.started.append(simulation_id)
        self.accumulated.pop(simulation_id or "sim-viewport", None)
        return simulation_id or "sim-viewport"

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
        width = 2 if simulation_id == "sim-viewport" else 4
        if simulation_id != "sim-viewport" and self.failures:
            raise self.failures.pop(0)
        completed = self.accumulated.get(simulation_id, 0) + int(kwargs["additional_samples"])
        self.accumulated[simulation_id] = completed
        return RenderResult(
            width=width,
            height=width,
            rgba8=b"\x00" * (width * width * 4),
            completed_samples=completed,
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values):
        from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult

        return OvrtxValueUpdateResult(len(tuple(values)), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        return "stopped"

    def shutdown(self) -> None:
        self.shutdowns += 1


def _load_engine_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_build29_final_retry_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"

    class _FakeRenderEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stats: list[tuple[str, str]] = []
            self.reports: list[tuple[set[str], str]] = []
            self.test_break_result = False

        def update_stats(self, engine: str, message: str) -> None:
            self.stats.append((engine, message))

        def tag_redraw(self) -> None:
            pass

        def report(self, levels: set[str], message: str) -> None:
            self.reports.append((set(levels), message))

        def test_break(self) -> bool:
            return self.test_break_result

    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=SimpleNamespace()),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    from ovrtx_blender_example import bundled_runtime

    monkeypatch.setattr(bundled_runtime, "public_sdk_python", lambda: sys.executable)
    return module


def _f12_engine(module, monkeypatch: pytest.MonkeyPatch, request: RenderRequest):
    monkeypatch.setattr(
        module,
        "_render_callback_adapter",
        lambda engine_id="": SimpleNamespace(final_render=lambda depsgraph: request),
    )
    monkeypatch.setattr(module, "_final_render_color_presentation_from_scene", lambda scene: {})
    engine = module.OvrtxExampleRenderEngine()
    written: list[RenderResult] = []
    engine._write_blender_result = written.append
    engine._write_result_artifact = lambda *args, **kwargs: None
    return engine, written


def _errors(engine) -> list[str]:
    return [message for levels, message in engine.reports if "ERROR" in levels]


# --- Current-scene route ----------------------------------------------------


def _scene_route(module, monkeypatch, request, failures):
    """Fake generation runtime lease and job; returns the lifecycle counters."""

    monkeypatch.setattr(module.scene_generation_sessions, "owns_request", lambda *_: True)
    monkeypatch.setattr(module, "_end_active_viewport_sessions", lambda *_: True)
    lifecycle = {"activated": 0, "restored": 0, "shutdown": 0, "jobs": 0, "active": 0}
    composition = object()

    class Lease:
        controller = object()
        last_ensure_result = SimpleNamespace(composition=composition)

        def deactivate(self) -> str:
            lifecycle["restored"] += 1
            lifecycle["active"] -= 1
            return "stopped"

    def activate(*_args: object, **_kwargs: object) -> Lease:
        # Never activated twice without a deactivation in between.
        assert lifecycle["active"] == 0
        lifecycle["activated"] += 1
        lifecycle["active"] += 1
        return Lease()

    class Controller:
        def shutdown(self) -> None:
            lifecycle["shutdown"] += 1

    def run_job(render_request, *_args, **kwargs):
        lifecycle["jobs"] += 1
        if failures:
            raise failures.pop(0)
        return {
            "status": "completed",
            "result": SimpleNamespace(completed_samples=request.max_samples),
            "composition": composition,
        }

    monkeypatch.setattr(module.scene_generation_sessions, "activate_for_final_render", activate)
    monkeypatch.setattr(module, "OvrtxSessionController", Controller)
    monkeypatch.setattr(module, "_run_final_render_job", run_job)
    return lifecycle


# --- Dedicated-controller routes --------------------------------------------


def _standalone_route(module, monkeypatch, failures):
    clients: list[_Client] = []

    def new_client(_request):
        clients.append(_Client(failures))
        return clients[-1]

    monkeypatch.setattr(controller_module, "_runtime_client_from_request", new_client)
    return clients


@contextmanager
def _viewport_session(module, monkeypatch, client, request):
    """Running viewport engine for the scene (real thread and loop, fake client)."""

    monkeypatch.setattr(controller_module, "_runtime_client_from_request", lambda _request: client)
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(config_factory=lambda _path: SimpleNamespace(enabled=False))
    try:
        engine._begin_async_viewport_session(request)
        engine._viewport_scene = SimpleNamespace(session_uid=SCENE_UID)
        engine._camera_mailbox.write(module.viewport_handoff.snapshot_from_render_request(request))
        frame = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert frame is not None
        yield engine
    finally:
        engine._viewport_scene = None
        runtime = module._ENGINE_RUNTIMES[id(engine)]
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
        assert runtime["cleanup_completion"].wait(WAIT_S)


def _render(engine) -> None:
    engine.render(SimpleNamespace(scene=SimpleNamespace(session_uid=SCENE_UID)))


ROUTES = ("current_scene", "standalone", "viewport_hosted")


def _run_route(route, module, monkeypatch, tmp_path, failures, *, cancel=False):
    """Render F12 through ``route``; returns (engine, written, final job count)."""

    request = _request(tmp_path)
    if route == "current_scene":
        lifecycle = _scene_route(module, monkeypatch, request, failures)
        engine, written = _f12_engine(module, monkeypatch, request)
        engine.test_break_result = cancel
        _render(engine)
        # Every activation was deactivated and its controller shut down.
        assert lifecycle["activated"] == lifecycle["restored"] == lifecycle["shutdown"]
        return engine, written, lifecycle["jobs"]
    if route == "standalone":
        clients = _standalone_route(module, monkeypatch, failures)
        engine, written = _f12_engine(module, monkeypatch, request)
        engine.test_break_result = cancel
        _render(engine)
        # Each attempt owned a fresh controller and client, and shut it down.
        assert all(client.shutdowns == 1 for client in clients)
        return engine, written, len(clients)
    viewport_request = _request(tmp_path, width=2, height=2, camera_matrix=_matrix(1.0))
    client = _Client(failures)
    with _viewport_session(module, monkeypatch, client, viewport_request):
        before = sum(1 for simulation_id in client.started if simulation_id is not None)
        engine, written = _f12_engine(module, monkeypatch, request)
        engine.test_break_result = cancel
        _render(engine)
        jobs = sum(1 for simulation_id in client.started if simulation_id is not None) - before
    return engine, written, jobs


def test_worker_failure_classification() -> None:
    assert is_worker_failure(_worker_lost())
    assert is_worker_failure(_device_lost())
    assert not is_worker_failure(RenderClientError("final session refused"))
    try:
        raise RuntimeWorkerError("renderer render: invalid camera")
    except RuntimeWorkerError as cause:
        try:
            raise RenderClientError(str(cause)) from cause
        except RenderClientError as rejected:
            assert not is_worker_failure(rejected)
    assert not is_worker_failure(None)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("failure", (_worker_lost, _device_lost), ids=("transport", "fatal_device"))
def test_final_job_retried_once_on_transport_error(monkeypatch, tmp_path, route, failure) -> None:
    module = _load_engine_module(monkeypatch)
    engine, written, jobs = _run_route(route, module, monkeypatch, tmp_path, [failure()])

    assert jobs == 2
    assert len(written) == 1
    assert _errors(engine) == []
    assert RESTARTING in engine.stats
    # The restart is shown before the retried job's outcome.
    assert engine.stats.index(RESTARTING) < engine.stats.index(("ovrtx", "Done"))
    assert engine.stats[-1] == ("ovrtx", "Done")
    warnings = [message for levels, message in engine.reports if "WARNING" in levels]
    assert len(warnings) == 1 and "restarting" in warnings[0].lower()


@pytest.mark.parametrize("route", ROUTES)
def test_final_job_not_retried_after_cancel(monkeypatch, tmp_path, route) -> None:
    module = _load_engine_module(monkeypatch)
    engine, written, jobs = _run_route(route, module, monkeypatch, tmp_path, [_worker_lost()], cancel=True)

    assert jobs == 1
    assert written == []
    assert RESTARTING not in engine.stats
    assert not any("WARNING" in levels for levels, _message in engine.reports)


@pytest.mark.parametrize("route", ROUTES)
def test_final_job_second_failure_reported(monkeypatch, tmp_path, route) -> None:
    module = _load_engine_module(monkeypatch)
    failures = [_worker_lost("renderer render: first"), _worker_lost("renderer render: second"),
                _worker_lost("renderer render: third")]
    engine, written, jobs = _run_route(route, module, monkeypatch, tmp_path, failures)

    assert jobs == 2
    assert written == []
    assert RESTARTING in engine.stats
    errors = _errors(engine)
    assert len(errors) == 1
    assert "second" in errors[0]


@pytest.mark.parametrize("route", ROUTES)
def test_final_job_not_retried_for_other_errors(monkeypatch, tmp_path, route) -> None:
    module = _load_engine_module(monkeypatch)
    failures = [RenderClientError("final session refused")]
    engine, written, jobs = _run_route(route, module, monkeypatch, tmp_path, failures)

    assert jobs == 1
    assert written == []
    assert RESTARTING not in engine.stats
    assert len(_errors(engine)) == 1


def test_session_reports_an_idle_worker_exit_without_a_request(tmp_path) -> None:
    from ovrtx_blender_example.direct_runtime_session import RuntimeSession

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_direct_runtime_session import WORKER, kill_owned

    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    session = RuntimeSession(sys.executable, worker, "renderer", tmp_path / "renderer", timeout_s=2)
    session.restart_backoff_s = (0.0, 0.0)
    try:
        assert session.worker_exited() is False  # never spawned
        session.request("load", {"scene_path": "/scene.usda"})
        assert session.worker_exited() is False
        kill_owned(session)
        assert session.worker_exited() is True
        # The next render request recovers the worker.
        session.request("render", {"samples": 1})
        assert session.worker_exited() is False
        assert session.epoch == 2
    finally:
        session.close()
    assert session.worker_exited() is False  # closed
