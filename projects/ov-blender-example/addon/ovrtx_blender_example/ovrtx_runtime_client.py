# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX render runtime boundary for sessions, updates, and result readback."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable, MutableMapping, Mapping, Sequence

from . import bundled_runtime, color_presentation, ovrtx_gpu_lease
from . import native_client_support
from . import session_lifecycle
from .ovrtx_session import OvrtxSessionSpec
from .ovrtx_value_updates import (
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
    _attribute_values,
    _transform_values,
)
from .properties import DEFAULT_RENDER_PRODUCT_PATH


DEFAULT_RENDER_SIMULATION_STEP_NS = 10
# Attach-time sweep budget (blender-live-render task05-02): the bounded
# retry loop below runs only on the first attach to a worker endpoint in
# this process, never per session creation on an already-attached worker.
STARTUP_CLEANUP_ATTEMPTS = 12
STARTUP_CLEANUP_RETRY_DELAY_SECONDS = 2
STARTUP_CLEANUP_PAGE_LIMIT = 100
SIMULATION_ID_PREFIX = "ovrtx-blender-"
ATTACH_CLEANUP_SCOPE_FULL = "full"
ATTACH_CLEANUP_SCOPE_DEAD_PID = "dead_pid"
CONTROL_PLANE_RPC_TIMEOUT_SECONDS = 30
RENDER_READ_POLL_TIMEOUT_MS = 30_000
ASYNC_READ_CANCEL_DRAIN_TIMEOUT_SECONDS = 5.0
ASYNC_READ_CANCEL_POLL_INTERVAL_SECONDS = 0.001
RENDER_TIMEOUT_ENV = "OV_BLENDER_EXAMPLE_RENDER_TIMEOUT_S"
DEFAULT_CONTROL_PLANE_ADDRESS = "127.0.0.1"
DEFAULT_CONTROL_PLANE_PORT = "50051"
RENDER_NATIVE_CLIENT_LABEL = "Native ovrtx client"
MDL_SYSTEM_PATH_ENV = "MDL_SYSTEM_PATH"
ACTIVE_CUDA_GPUS_ENV = "OVRTX_ACTIVE_CUDA_GPUS"
RENDER_TRANSFORM_ATTRIBUTE = "omni:xform"
RENDER_TRANSFORM_VALUE_TYPE = "Matrix4d"
WORKER_ENDPOINT_PROBE_TIMEOUT_SECONDS = 0.25


class RenderClientError(RuntimeError):
    """Raised when the render client cannot produce a render result."""


class RuntimeServicesPreparingError(RenderClientError):
    """Runtime-services worker is (re)starting: a transient wait, not a failure.

    The viewport session start raises this while the runtime services respawn
    (e.g. a worker restart). The render loop treats it as a deferral -- hold the
    loading state and retry -- rather than a failure publication.
    """


@dataclass(frozen=True)
class RenderResult:
    width: int
    height: int
    rgba8: bytes
    completed_samples: int
    session_completed_samples: int
    simulation_time_ns: int
    render_output_simulation_time_ns: int = 0
    frame_format: str = color_presentation.FRAME_FORMAT_RGBA8
    frame_color_mode: str = color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    render_var: str = color_presentation.RENDER_VAR_LDR_COLOR
    linear_rgba16f: bytes = b""
    native_timings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderSubmission:
    """One exact-time render write that may be completed or discarded later."""

    owner_id: int
    submission_id: int
    session_token: int
    simulation_id: str
    selected_sensor_paths: tuple[str, ...]
    render_var: str
    simulation_time_ns: int
    completed_samples: int
    session_completed_samples: int
    selection_diagnostics: tuple[Mapping[str, Any], ...]
    write_diagnostics: tuple[Mapping[str, Any], ...]
    submitted_monotonic_ns: int


@dataclass(frozen=True)
class RenderReadTicket:
    """Opaque ownership token for one in-progress asynchronous framebuffer read."""

    owner_id: int
    read_id: int
    session_token: int
    simulation_id: str
    submission_id: int


@dataclass
class _OvrtxSessionCursor:
    simulation_time_ns: int = 0
    snapshot_completed_samples: int = 0
    session_completed_samples: int = 0


@dataclass
class _OvrtxSessionState:
    spec: OvrtxSessionSpec
    sensor_paths: tuple[str, ...]
    width: int
    height: int
    session_token: int
    cursor: _OvrtxSessionCursor = field(default_factory=_OvrtxSessionCursor)
    selected_render_var_paths: set[str] = field(default_factory=set)
    pending_submission_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _RenderNativeBindings:
    start_worker: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    create_simulation: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    write_world_state: Callable[[Any], Mapping[str, Any]]
    read_world_state: Callable[[Any], Mapping[str, Any]]
    begin_read_world_state: Callable[[Any], Any] | None
    build_write_world_state_columns: Callable[[Mapping[str, Any]], Any]
    build_render_sample_step: Callable[[Mapping[str, Any]], Any]
    build_attribute_values_update: Callable[[Mapping[str, Any]], Any]
    build_read_world_state_ldr_color: Callable[[Mapping[str, Any]], Any]
    build_read_world_state_hdr_color: Callable[[Mapping[str, Any]], Any] | None
    decode_ldr_color_frame: Callable[[Any, Any], Any]
    decode_hdr_color_frame: Callable[[Any, Any], Any] | None
    rpc_status_error: type[BaseException] | None


@dataclass
class _AsyncRenderReadState:
    """Incremental form of ``_read_color_frame`` for one exact-time read."""

    ticket: RenderReadTicket
    submission: RenderSubmission
    width: int
    height: int
    render_var_paths: tuple[str, ...]
    render_var: str
    build_read: Callable[[Mapping[str, Any]], Any]
    decode_frame: Callable[[Any, Any], Any]
    timeout_seconds: int
    deadline: float
    read_started_at: int
    path_index: int = 0
    iterator: str = ""
    path_frame: Mapping[str, Any] | None = None
    path_status: str | None = None
    selected_frame: Mapping[str, Any] | None = None
    request_handle: Any | None = None
    native_ticket: Any | None = None
    remaining_ms: int = 0
    poll_timeout_ms: int = 0
    read_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    read_poll_count: int = 0
    read_iterator_count: int = 0
    read_empty_ok_count: int = 0
    read_world_state_ms: float = 0.0
    read_iterator_world_state_ms: float = 0.0
    read_empty_ok_world_state_ms: float = 0.0
    read_success_world_state_ms: float = 0.0
    read_transient_status_count: int = 0
    read_transient_status_world_state_ms: float = 0.0


@dataclass(frozen=True)
class _ControlPlaneBindings:
    list_simulations: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    delete_simulation: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    close: Callable[[], None]


def _normalised_sensor_paths(paths: Any) -> tuple[str, ...]:
    if paths is None:
        return ()
    if isinstance(paths, str):
        candidate_paths = (paths,)
    else:
        try:
            candidate_paths = tuple(paths)
        except TypeError:
            return ()
    return tuple(str(path) for path in candidate_paths if str(path))


def _active_sensor_paths(sensor_paths: Sequence[str], selected_sensor_paths: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(path) for path in selected_sensor_paths if str(path))
    if selected:
        return selected
    sensors = tuple(str(path) for path in sensor_paths if str(path))
    return sensors or (DEFAULT_RENDER_PRODUCT_PATH,)


def _render_var_paths(
    selected_sensor_paths: Sequence[str],
    render_var: str,
) -> tuple[str, ...]:
    name = str(render_var or "").strip("/")
    if not name or "/" in name:
        raise RenderClientError(f"Invalid OVRTX render var name: {render_var}")
    return tuple(f"{str(path).rstrip('/')}/{name}" for path in selected_sensor_paths)


def _selected_color_frame(
    decoded: Any,
    render_var_paths: Sequence[str],
    render_var: str,
) -> Mapping[str, Any] | None:
    if decoded is None:
        return None
    if not isinstance(decoded, Mapping):
        raise RenderClientError(f"Native ovrtx client returned an invalid {render_var} decode result")
    frames = _result_value(decoded, "frames")
    if not isinstance(frames, Mapping):
        raise RenderClientError(f"Native ovrtx client returned no keyed {render_var} frames")
    for render_var_path in render_var_paths:
        frame = frames.get(render_var_path)
        if frame is None:
            continue
        if not isinstance(frame, Mapping):
            raise RenderClientError(
                f"Native ovrtx client returned an invalid {render_var} frame for "
                f"{render_var_path}"
            )
        return frame
    return None


def _endpoint_listening(endpoint: str, timeout_seconds: float = WORKER_ENDPOINT_PROBE_TIMEOUT_SECONDS) -> bool:
    """Whether something is already accepting connections on ``endpoint``.

    Used as the pre-launch half of the worker ownership signal: if the
    control-plane port is already served before this process launches
    anything, the serving worker is foreign (a prior crashed session's
    orphan or another Blender instance) — the native client will spawn a
    doomed duplicate whose ``worker_process_alive`` flag must not be read
    as ownership of the *serving* worker.
    """

    host, _, port_text = str(endpoint or "").rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return False
    if not host or not (0 < port <= 65535):
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


# Worker attach registry (blender-live-render task05-02): control-plane
# endpoints whose stale-simulation sweep already ran in this process.
# Session controllers construct a fresh OvrtxRuntimeClient per session
# replacement, so attach tracking must outlive individual clients.
_worker_attach_lock = threading.Lock()
_attached_worker_endpoints: set[str] = set()


def _reset_worker_attach_registry() -> None:
    """Forget swept worker endpoints (test isolation seam)."""

    with _worker_attach_lock:
        _attached_worker_endpoints.clear()


def _simulation_id_pid(simulation_id: str) -> int | None:
    """Parse the PID baked into this add-on's simulation ID convention.

    Session IDs are ``ovrtx-blender-<lane>-<pid>`` today (viewport lane:
    ``ovrtx-blender-viewport-<os.getpid()>``). Returns ``None`` for IDs
    outside the convention so foreign simulations are never candidates
    for the scoped attach sweep.
    """

    text = str(simulation_id or "")
    if not text.startswith(SIMULATION_ID_PREFIX):
        return None
    tail = text.rsplit("-", 1)[-1]
    if not tail or any(character not in "0123456789" for character in tail):
        return None
    pid = int(tail)
    return pid if pid > 0 else None


def _partition_stale_convention_simulations(
    simulation_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split listed simulations into (stale, kept) for the scoped sweep.

    Stale means: matches this add-on's PID-derived ID convention and the
    parsed PID is not a running local process. Everything else — foreign
    IDs, live PIDs (another running Blender), and this process's own
    PID — is kept.
    """

    stale: list[str] = []
    kept: list[str] = []
    for simulation_id in simulation_ids:
        pid = _simulation_id_pid(simulation_id)
        if pid is None or pid == os.getpid() or session_lifecycle.pid_running(pid):
            kept.append(str(simulation_id))
        else:
            stale.append(str(simulation_id))
    return stale, kept


class OvrtxRuntimeClient:
    """OVRTX render client through the generated native extension."""

    def __init__(self, *, worker_command: str, native_client_module: str) -> None:
        self._worker_command = str(worker_command)
        self._native_client_module = str(native_client_module)
        self._native_module: Any | None = None
        self._native_client: Any | None = None
        self._native_endpoint = ""
        self._native_bindings: _RenderNativeBindings | None = None
        self._control_plane_bindings: _ControlPlaneBindings | None = None
        self._session_states: dict[str, _OvrtxSessionState] = {}
        self._next_session_token = 1
        self._next_submission_id = 1
        self._next_read_id = 1
        self._active_render_read: _AsyncRenderReadState | None = None
        self._abort_cleanup = False
        self._gpu_lease: ovrtx_gpu_lease.OvrtxGpuLease | None = None
        self.last_render_timings: dict[str, Any] = {}
        self.last_value_update_timings: dict[str, Any] = {}
        self.last_delete_diagnostics: dict[str, Any] = {}
        self.startup_diagnostics: dict[str, Any] = {"render_worker": {"status": "not_started"}}
        self._worker_owned: bool | None = None

    @property
    def worker_owned(self) -> bool | None:
        """Whether this process launched (and can terminate) the serving worker.

        ``True``: the live worker on the control-plane endpoint is a process
        this add-on's native client spawned — ``shutdown()`` terminates it, so
        an in-app session restart relaunches it with the current worker startup
        config (``rtpt_worker_config``). ``False``: the serving worker is
        foreign (pre-existing orphan or another instance) — ``shutdown()``
        cannot and must not kill it (the native layer only ever terminates its
        own spawned process), so launch-time settings changes need that worker
        to exit by other means. ``None``: no session has been started yet.
        """

        return self._worker_owned

    def _evaluate_worker_ownership(self, native_client: Any) -> bool:
        """Pre-``start_worker`` half of the ownership signal.

        A worker this process spawned earlier and that is still alive
        (``check_health().worker_process_alive``) is ours. Otherwise the
        launch about to happen owns the serving worker only when nothing is
        already listening on the endpoint — if the port is served by a
        foreign worker, ``start_worker`` spawns a doomed duplicate whose
        ``worker_process_alive`` briefly reads ``True`` without the duplicate
        ever serving (verified against the real worker: the duplicate dies to
        our ``shutdown()`` while the foreign worker keeps serving).
        """

        health = getattr(native_client, "check_health", None)
        if callable(health):
            try:
                result = health()
            except Exception:
                result = None
            if isinstance(result, Mapping) and bool(result.get("worker_process_alive")):
                return True
        address = _srtx_server_address_from_worker_command(self._worker_command) or DEFAULT_CONTROL_PLANE_ADDRESS
        port = _srtx_server_port_from_worker_command(self._worker_command) or DEFAULT_CONTROL_PLANE_PORT
        return not _endpoint_listening(f"{address}:{port}")

    @property
    def startup_evidence(self) -> dict[str, Any]:
        return self.startup_diagnostics

    @startup_evidence.setter
    def startup_evidence(self, value: Mapping[str, Any]) -> None:
        self.startup_diagnostics = dict(value)

    def _record_startup_failure(
        self,
        *,
        error: str,
        logs: Mapping[str, Any],
        protocol_diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        failure = _render_worker_failure_diagnostics(
            worker_command=self._worker_command,
            error=error,
            logs=logs,
        )
        if protocol_diagnostics is not None:
            failure["protocol_diagnostics"] = dict(protocol_diagnostics)
        self.startup_diagnostics = {"render_worker": failure}

    def _ensure_gpu_lease(self, logs: Mapping[str, Any]) -> None:
        if self._gpu_lease is not None:
            return
        from . import runtime_services

        if runtime_services.owner.diagnostics()["status"] == "ready":
            return
        try:
            self._gpu_lease = ovrtx_gpu_lease.acquire(
                metadata={
                    "entrypoint": "OvrtxRuntimeClient.start_session",
                    "worker_command": self._worker_command,
                }
            )
        except ovrtx_gpu_lease.OvrtxGpuLeaseBusy as exc:
            self._record_startup_failure(
                error=str(exc),
                logs=logs,
            )
            self.startup_diagnostics["render_worker"]["ovrtx_gpu_lease"] = (
                ovrtx_gpu_lease.busy_diagnostics(exc)
            )
            raise RenderClientError(str(exc)) from exc

    def start_session(self, spec: OvrtxSessionSpec, simulation_id: str | None = None) -> str:
        from . import runtime_services

        service_status = runtime_services.owner.diagnostics()["status"]
        if service_status == "starting":
            raise RuntimeServicesPreparingError("Runtime services are still preparing")
        if service_status == "failed":
            raise RenderClientError(
                f"Runtime service preparation failed: {runtime_services.owner.diagnostics()['error']}"
            )
        log_paths = session_lifecycle.prepare_logs(os.environ)
        if not spec.ovrtx_scene_composition.composed_scene_path:
            self._record_startup_failure(
                error="No composed OVRTX scene path configured",
                logs=log_paths,
            )
            raise RenderClientError("No composed OVRTX scene path configured")
        if not self._worker_command:
            self._record_startup_failure(
                error="No managed ovrtx worker command configured",
                logs=log_paths,
            )
            raise RenderClientError("No managed ovrtx worker command configured")
        if not self._native_client_module:
            self._record_startup_failure(
                error="No native ovrtx client module configured",
                logs=log_paths,
            )
            raise RenderClientError("No native ovrtx client module configured")
        self._ensure_gpu_lease(log_paths)

        composed_usd_uri = Path(spec.ovrtx_scene_composition.composed_scene_path).expanduser().resolve().as_uri()
        sensor_paths = _normalised_sensor_paths(spec.sensor_paths)
        if not sensor_paths:
            sensor_paths = (DEFAULT_RENDER_PRODUCT_PATH,)
        old_worker_env: dict[str, str | None] = {}
        try:
            health_deadline = time.monotonic() + runtime_services.health_timeout_seconds()
            old_worker_env = _sanitize_worker_environment()
            old_worker_env.update(apply_worker_runtime_environment(os.environ, self._worker_command))
            native_client = self._import_native_client(self._native_client_module)
            owned_before_start = self._evaluate_worker_ownership(native_client)
            start_result = native_client_support.require_callable(
                native_client,
                "start_worker",
                client_label=RENDER_NATIVE_CLIENT_LABEL,
                error_type=RenderClientError,
            )(
                {
                    "worker_command": self._worker_command,
                    "wait_seconds": runtime_services._remaining_seconds(health_deadline),
                }
            )
            endpoint = _control_plane_endpoint(start_result, self._worker_command)
            self._ensure_client(native_client, endpoint)
            runtime_services.wait_for_serving(
                "OVRTX",
                endpoint,
                runtime_services._module_serving_probe(native_client),
                process_alive=runtime_services._process_alive(native_client, start_result),
                deadline=health_deadline,
            )
            bindings = self._require_bindings()
            self._worker_owned = owned_before_start and bool(
                _result_value(start_result, "worker_process_alive", False)
            )
            control_plane = _bind_official_control_plane(start_result, self._worker_command)
            self._control_plane_bindings = control_plane
            cleanup_diagnostics = self._cleanup_on_worker_attach(
                control_plane,
                endpoint=_control_plane_endpoint(start_result, self._worker_command),
                worker_launched=bool(_result_value(start_result, "worker_process_alive", False)),
            )
            result = _call_native_rpc(
                bindings,
                "CreateSimulation",
                bindings.create_simulation,
                {
                    "simulation_id": simulation_id or f"{SIMULATION_ID_PREFIX}viewport-{os.getpid()}",
                    "usd_file_uri": composed_usd_uri,
                    "sensors": [{"sensor_path": path} for path in sensor_paths],
                    "width": spec.width,
                    "height": spec.height,
                },
            )
            _restore_environment(old_worker_env)
        except RenderClientError as exc:
            _restore_environment(old_worker_env)
            diagnostics = native_client_support.exception_protocol_diagnostics(exc)
            self._record_startup_failure(
                error=str(exc),
                logs=log_paths,
                protocol_diagnostics=diagnostics,
            )
            self.shutdown(reset_startup_diagnostics=False)
            raise
        except Exception as exc:
            _restore_environment(old_worker_env)
            self._record_startup_failure(
                error=f"{type(exc).__name__}: {exc}",
                logs=log_paths,
            )
            self.shutdown(reset_startup_diagnostics=False)
            raise RenderClientError(f"Native ovrtx client failed before OVRTX session start: {exc}") from exc

        simulation_id = str(_result_value(result, "simulation_id", ""))
        if not simulation_id:
            self._record_startup_failure(
                error="Native ovrtx client returned no simulation id",
                logs=log_paths,
            )
            self.shutdown(reset_startup_diagnostics=False)
            raise RenderClientError("Native ovrtx client returned no simulation id")
        self.startup_diagnostics = {
            "render_worker": _render_worker_startup_diagnostics(
                native_client,
                simulation_id=simulation_id,
                worker_command=self._worker_command,
                logs=log_paths,
            )
        }
        self.startup_diagnostics["render_worker"]["cleanup"] = cleanup_diagnostics
        if self._gpu_lease is not None:
            self.startup_diagnostics["render_worker"]["ovrtx_gpu_lease"] = (
                self._gpu_lease.diagnostics()
            )
        self.startup_diagnostics["render_worker"]["worker_owned"] = self._worker_owned
        result_sensor_paths = _normalised_sensor_paths(_result_value(result, "sensor_paths")) or sensor_paths
        session_token = self._next_session_token
        self._next_session_token += 1
        self._session_states[simulation_id] = _OvrtxSessionState(
            spec=spec,
            sensor_paths=result_sensor_paths,
            width=int(_result_value(result, "width", spec.width)),
            height=int(_result_value(result, "height", spec.height)),
            session_token=session_token,
        )
        return simulation_id

    def render_result(
        self,
        simulation_id: str,
        *,
        selected_sensor_paths: Sequence[str],
        render_var: str,
        additional_samples: int,
    ) -> RenderResult:
        submission = self._submit_render_samples(
            simulation_id,
            selected_sensor_paths=selected_sensor_paths,
            render_var=render_var,
            additional_samples=additional_samples,
        )
        return self.complete_render_sample(submission)

    def supports_async_render_read(self) -> bool:
        """Whether the bound native client offers the complete async-read seam."""

        bindings = self._native_bindings
        return bool(bindings and callable(bindings.begin_read_world_state))

    def submit_render_sample(
        self,
        simulation_id: str,
        *,
        selected_sensor_paths: Sequence[str],
        render_var: str,
    ) -> RenderSubmission:
        """Submit one sample step without waiting for its framebuffer."""

        return self._submit_render_samples(
            simulation_id,
            selected_sensor_paths=selected_sensor_paths,
            render_var=render_var,
            additional_samples=1,
        )

    def begin_render_sample_read(
        self,
        submission: RenderSubmission,
    ) -> RenderReadTicket:
        """Begin nonblocking readback for one exact-time render submission."""

        if self._active_render_read is not None:
            raise RenderClientError("Another asynchronous render read is already active")
        bindings = self._require_bindings()
        if not callable(bindings.begin_read_world_state):
            raise RenderClientError(
                "Active native OVRTX client does not support asynchronous ReadWorldState"
            )
        state = self._claim_render_submission(submission)
        try:
            render_var = (
                submission.render_var or color_presentation.RENDER_VAR_LDR_COLOR
            )
            render_var_paths = _render_var_paths(
                submission.selected_sensor_paths,
                render_var,
            )
            if render_var == color_presentation.RENDER_VAR_HDR_COLOR:
                build_read = bindings.build_read_world_state_hdr_color
                decode_frame = bindings.decode_hdr_color_frame
                if build_read is None or decode_frame is None:
                    raise RenderClientError(
                        "Native ovrtx client does not support HdrColor readback"
                    )
            elif render_var == color_presentation.RENDER_VAR_LDR_COLOR:
                build_read = bindings.build_read_world_state_ldr_color
                decode_frame = bindings.decode_ldr_color_frame
            else:
                raise RenderClientError(
                    f"Unsupported OVRTX render var: {render_var}"
                )
            timeout_seconds = max(
                1,
                int(os.environ.get(RENDER_TIMEOUT_ENV, "600")),
            )
            ticket = RenderReadTicket(
                owner_id=id(self),
                read_id=self._next_read_id,
                session_token=submission.session_token,
                simulation_id=submission.simulation_id,
                submission_id=submission.submission_id,
            )
            self._next_read_id += 1
            read_state = _AsyncRenderReadState(
                ticket=ticket,
                submission=submission,
                width=state.width,
                height=state.height,
                render_var_paths=render_var_paths,
                render_var=render_var,
                build_read=build_read,
                decode_frame=decode_frame,
                timeout_seconds=timeout_seconds,
                deadline=time.monotonic() + timeout_seconds,
                read_started_at=time.perf_counter_ns(),
            )
            self._active_render_read = read_state
            self._start_async_render_read(read_state, bindings)
            return ticket
        except RenderClientError:
            self._active_render_read = None
            self._abort_cleanup = True
            raise
        except Exception as exc:
            self._active_render_read = None
            self._abort_cleanup = True
            raise RenderClientError(
                f"Native ovrtx client failed before OVRTX render result: {exc}"
            ) from exc

    def poll_render_sample_read(
        self,
        ticket: RenderReadTicket,
    ) -> RenderResult | None:
        """Advance one native async read without blocking; return a terminal frame."""

        read_state = self._render_read_state(ticket)
        bindings = self._require_bindings()
        try:
            result = self._advance_async_render_read(read_state, bindings)
            if result is not None:
                self._active_render_read = None
            return result
        except RenderClientError:
            self._active_render_read = None
            self._abort_cleanup = True
            raise
        except Exception as exc:
            self._active_render_read = None
            self._abort_cleanup = True
            raise RenderClientError(
                f"Native ovrtx client failed before OVRTX render result: {exc}"
            ) from exc

    def cancel_render_sample_read(self, ticket: RenderReadTicket) -> None:
        """Cancel, drain, and consume one active native async read ticket."""

        read_state = self._render_read_state(ticket)
        native_ticket = read_state.native_ticket
        cancel = getattr(native_ticket, "cancel", None)
        if not callable(cancel):
            self._active_render_read = None
            self._abort_cleanup = True
            raise RenderClientError(
                "Native asynchronous ReadWorldState ticket is missing callable cancel"
            )
        try:
            cancel()
        except Exception as exc:
            self._active_render_read = None
            self._abort_cleanup = True
            raise self._async_cancel_error(exc) from exc

        # Native cancellation is deliberately best-effort/nonblocking.  Keep
        # ownership until its callback is terminal so DeleteSimulation,
        # replacement, or a successor read cannot overtake the cancelled RPC.
        drain_deadline = (
            time.monotonic() + ASYNC_READ_CANCEL_DRAIN_TIMEOUT_SECONDS
        )
        while True:
            try:
                result = native_ticket.poll()
            except Exception as exc:
                bindings = self._native_bindings
                if (
                    bindings is not None
                    and bindings.rpc_status_error is not None
                    and isinstance(exc, bindings.rpc_status_error)
                ):
                    diagnostics = (
                        native_client_support.exception_protocol_diagnostics(exc)
                        or {}
                    )
                    self._active_render_read = None
                    if str(diagnostics.get("grpc_status", "")) == "CANCELLED":
                        return
                else:
                    self._active_render_read = None
                self._abort_cleanup = True
                raise self._async_cancel_error(exc) from exc
            if result is not None:
                # Completion won the cancellation race. Consuming and dropping
                # that response is terminal and intentionally publishes no frame.
                self._active_render_read = None
                return
            if time.monotonic() >= drain_deadline:
                # Retain the active state so shutdown may retry the drain; no
                # control-plane deletion is allowed beneath this RPC.
                self._abort_cleanup = True
                raise RenderClientError(
                    "Native asynchronous ReadWorldState cancellation did not "
                    "reach a terminal callback within "
                    f"{ASYNC_READ_CANCEL_DRAIN_TIMEOUT_SECONDS:.1f}s"
                )
            time.sleep(ASYNC_READ_CANCEL_POLL_INTERVAL_SECONDS)

    def _async_cancel_error(self, exc: BaseException) -> RenderClientError:
        bindings = self._native_bindings
        if (
            bindings is not None
            and bindings.rpc_status_error is not None
            and isinstance(exc, bindings.rpc_status_error)
        ):
            return native_client_support.rpc_status_exception(
                "ReadWorldState",
                exc,
                client_label=RENDER_NATIVE_CLIENT_LABEL,
                error_type=RenderClientError,
            )
        return RenderClientError(
            f"Native ovrtx client failed while cancelling render read: {exc}"
        )

    def _submit_render_samples(
        self,
        simulation_id: str,
        *,
        selected_sensor_paths: Sequence[str],
        render_var: str,
        additional_samples: int,
    ) -> RenderSubmission:
        if self._native_client is None:
            raise RenderClientError("No active OVRTX session")
        bindings = self._require_bindings()
        state = self._session_state(simulation_id)
        selected_sensor_paths = _active_sensor_paths(
            state.sensor_paths,
            _normalised_sensor_paths(selected_sensor_paths),
        )
        undeclared = [path for path in selected_sensor_paths if path not in state.sensor_paths]
        if undeclared:
            raise RenderClientError(
                "selected_sensor_paths must be declared by the OVRTX service: "
                + ", ".join(undeclared)
            )
        additional_samples = int(additional_samples)
        if additional_samples < 1:
            raise ValueError("additional_samples must be at least 1")
        cursor = state.cursor

        try:
            submitted_monotonic_ns = time.perf_counter_ns()
            render_var = render_var or color_presentation.RENDER_VAR_LDR_COLOR
            render_var_paths = _render_var_paths(selected_sensor_paths, render_var)
            if render_var == color_presentation.RENDER_VAR_HDR_COLOR:
                build_selection_read = bindings.build_read_world_state_hdr_color
                if build_selection_read is None:
                    raise RenderClientError("Native ovrtx client does not support HdrColor readback")
            elif render_var == color_presentation.RENDER_VAR_LDR_COLOR:
                build_selection_read = bindings.build_read_world_state_ldr_color
            else:
                raise RenderClientError(f"Unsupported OVRTX render var: {render_var}")
            selection_diagnostics: list[Mapping[str, Any]] = []
            next_simulation_time_ns = cursor.simulation_time_ns + DEFAULT_RENDER_SIMULATION_STEP_NS
            for render_var_path in render_var_paths:
                if render_var_path in state.selected_render_var_paths:
                    continue
                selection_handle = build_selection_read(
                    {
                        "simulation_id": simulation_id,
                        "render_var_paths": [render_var_path],
                        "simulation_time_ns": next_simulation_time_ns,
                        "timeout_ms": 0,
                        "width": state.width,
                        "height": state.height,
                        "completed_samples": cursor.snapshot_completed_samples,
                        "session_completed_samples": cursor.session_completed_samples,
                    }
                )
                try:
                    selection_result = _call_native_rpc(
                        bindings,
                        "ReadWorldState",
                        bindings.read_world_state,
                        selection_handle,
                    )
                except RenderClientError as exc:
                    diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {}
                    if str(diagnostics.get("grpc_status", "")) != "FAILED_PRECONDITION":
                        raise
                    selection_diagnostics.append(diagnostics)
                else:
                    selection_diagnostics.append(native_client_support.native_response_diagnostics(selection_result))
                state.selected_render_var_paths.add(render_var_path)
            step_results: list[Mapping[str, Any]] = []
            for _index in range(additional_samples):
                next_simulation_time_ns = cursor.simulation_time_ns + DEFAULT_RENDER_SIMULATION_STEP_NS
                write_handle = bindings.build_render_sample_step(
                    {
                        "simulation_id": simulation_id,
                        "simulation_time_ns": next_simulation_time_ns,
                    }
                )
                step_results.append(_call_native_rpc(bindings, "WriteWorldState", bindings.write_world_state, write_handle))
                cursor.simulation_time_ns = next_simulation_time_ns
                cursor.snapshot_completed_samples += 1
                cursor.session_completed_samples += 1
            submission_id = self._next_submission_id
            self._next_submission_id += 1
            state.pending_submission_ids.add(submission_id)
            return RenderSubmission(
                owner_id=id(self),
                submission_id=submission_id,
                session_token=state.session_token,
                simulation_id=str(simulation_id),
                selected_sensor_paths=tuple(selected_sensor_paths),
                render_var=render_var,
                simulation_time_ns=cursor.simulation_time_ns,
                completed_samples=cursor.snapshot_completed_samples,
                session_completed_samples=cursor.session_completed_samples,
                selection_diagnostics=tuple(dict(item) for item in selection_diagnostics),
                write_diagnostics=tuple(
                    native_client_support.native_response_diagnostics(item)
                    for item in step_results
                ),
                submitted_monotonic_ns=submitted_monotonic_ns,
            )
        except RenderClientError:
            self._abort_cleanup = True
            raise
        except Exception as exc:
            self._abort_cleanup = True
            raise RenderClientError(
                f"Native ovrtx client failed before OVRTX render submission: {exc}"
            ) from exc

    def complete_render_sample(self, submission: RenderSubmission) -> RenderResult:
        """Read and decode exactly one previously submitted render time."""

        if self._active_render_read is not None:
            raise RenderClientError("Another asynchronous render read is already active")
        state = self._claim_render_submission(submission)
        bindings = self._require_bindings()
        cursor = _OvrtxSessionCursor(
            simulation_time_ns=submission.simulation_time_ns,
            snapshot_completed_samples=submission.completed_samples,
            session_completed_samples=submission.session_completed_samples,
        )
        try:
            result = self._read_color_frame(
                bindings,
                submission.simulation_id,
                state,
                cursor,
                selected_sensor_paths=submission.selected_sensor_paths,
                render_var=submission.render_var,
            )
            native_completed_ns = time.perf_counter_ns()
            native_timings = _native_timings_from_result(result)
            native_timings["read_selection"] = [
                dict(item) for item in submission.selection_diagnostics
            ]
            native_timings["write_world_state"] = [
                dict(item) for item in submission.write_diagnostics
            ]
            result = dict(result)
            result["native_timings"] = native_timings
        except RenderClientError:
            self._abort_cleanup = True
            raise
        except Exception as exc:
            self._abort_cleanup = True
            raise RenderClientError(
                f"Native ovrtx client failed before OVRTX render result: {exc}"
            ) from exc

        try:
            convert_started_ns = time.perf_counter_ns()
            render_result = render_result_from_native(result, state.width, state.height)
            convert_completed_ns = time.perf_counter_ns()
        except Exception:
            self._abort_cleanup = True
            raise
        actual_time_ns = int(render_result.render_output_simulation_time_ns or 0)
        if actual_time_ns and actual_time_ns != submission.simulation_time_ns:
            self._abort_cleanup = True
            raise RenderClientError(
                "OVRTX render result time mismatch: "
                f"requested {submission.simulation_time_ns}, received {actual_time_ns}"
            )
        self.last_render_timings = {
            "native_render_ms": (
                native_completed_ns - submission.submitted_monotonic_ns
            )
            / 1_000_000.0,
            "result_convert_ms": (convert_completed_ns - convert_started_ns) / 1_000_000.0,
        }
        if native_timings:
            self.last_render_timings["native_timings"] = native_timings
        if submission.write_diagnostics:
            self.last_render_timings["write_world_state"] = [
                dict(item) for item in submission.write_diagnostics
            ]
        return render_result

    def discard_render_sample(self, submission: RenderSubmission) -> None:
        """Forget one speculative result without transferring its pixels."""

        self._claim_render_submission(submission)

    def _claim_render_submission(
        self, submission: RenderSubmission
    ) -> _OvrtxSessionState:
        if not isinstance(submission, RenderSubmission):
            raise TypeError("render submission must be a RenderSubmission")
        if submission.owner_id != id(self):
            raise RenderClientError("Render submission belongs to another client")
        state = self._session_state(submission.simulation_id)
        if submission.session_token != state.session_token:
            raise RenderClientError("Render submission belongs to an inactive session")
        if submission.submission_id not in state.pending_submission_ids:
            raise RenderClientError("Render submission was already completed or discarded")
        state.pending_submission_ids.remove(submission.submission_id)
        return state

    def _render_read_state(
        self,
        ticket: RenderReadTicket,
    ) -> _AsyncRenderReadState:
        if not isinstance(ticket, RenderReadTicket):
            raise TypeError("render read ticket must be a RenderReadTicket")
        if ticket.owner_id != id(self):
            raise RenderClientError("Render read ticket belongs to another client")
        read_state = self._active_render_read
        if read_state is None or read_state.ticket.read_id != ticket.read_id:
            raise RenderClientError("Render read ticket was already completed or cancelled")
        state = self._session_state(ticket.simulation_id)
        if ticket.session_token != state.session_token:
            raise RenderClientError("Render read ticket belongs to an inactive session")
        if ticket.submission_id != read_state.submission.submission_id:
            raise RenderClientError("Render read ticket does not match its submission")
        return read_state

    def _start_async_render_read(
        self,
        read_state: _AsyncRenderReadState,
        bindings: _RenderNativeBindings,
    ) -> None:
        remaining_ms = int((read_state.deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise RenderClientError(
                "Native ovrtx client returned no "
                f"{read_state.render_var} render result before deadline"
            )
        poll_timeout_ms = min(remaining_ms, RENDER_READ_POLL_TIMEOUT_MS)
        render_var_path = read_state.render_var_paths[read_state.path_index]
        submission = read_state.submission
        request: dict[str, Any] = {
            "simulation_id": submission.simulation_id,
            "render_var_paths": [render_var_path],
            "simulation_time_ns": submission.simulation_time_ns,
            "timeout_ms": poll_timeout_ms,
            "timeout_seconds": max(1, (poll_timeout_ms + 999) // 1000),
            "width": read_state.width,
            "height": read_state.height,
            "completed_samples": submission.completed_samples,
            "session_completed_samples": submission.session_completed_samples,
        }
        if read_state.iterator:
            request["iterator"] = read_state.iterator
        request_handle = read_state.build_read(request)
        begin = bindings.begin_read_world_state
        if not callable(begin):
            raise RenderClientError(
                "Active native OVRTX client does not support asynchronous ReadWorldState"
            )
        try:
            native_ticket = begin(request_handle)
        except Exception as exc:
            if (
                bindings.rpc_status_error is not None
                and isinstance(exc, bindings.rpc_status_error)
            ):
                raise native_client_support.rpc_status_exception(
                    "ReadWorldState",
                    exc,
                    client_label=RENDER_NATIVE_CLIENT_LABEL,
                    error_type=RenderClientError,
                ) from exc
            raise
        if not callable(getattr(native_ticket, "poll", None)) or not callable(
            getattr(native_ticket, "cancel", None)
        ):
            cancel = getattr(native_ticket, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            raise RenderClientError(
                "Native asynchronous ReadWorldState ticket must expose callable "
                "poll and cancel methods"
            )
        read_state.request_handle = request_handle
        read_state.native_ticket = native_ticket
        read_state.remaining_ms = remaining_ms
        read_state.poll_timeout_ms = poll_timeout_ms

    def _advance_async_render_read(
        self,
        read_state: _AsyncRenderReadState,
        bindings: _RenderNativeBindings,
    ) -> RenderResult | None:
        native_ticket = read_state.native_ticket
        poll = getattr(native_ticket, "poll", None)
        if not callable(poll):
            raise RenderClientError(
                "Native asynchronous ReadWorldState ticket is missing callable poll"
            )
        try:
            native_result = poll()
        except Exception as exc:
            if (
                bindings.rpc_status_error is not None
                and isinstance(exc, bindings.rpc_status_error)
            ):
                translated = native_client_support.rpc_status_exception(
                    "ReadWorldState",
                    exc,
                    client_label=RENDER_NATIVE_CLIENT_LABEL,
                    error_type=RenderClientError,
                )
                diagnostics = (
                    native_client_support.exception_protocol_diagnostics(translated)
                    or {}
                )
                if (
                    str(diagnostics.get("grpc_status", ""))
                    == "DEADLINE_EXCEEDED"
                    and read_state.remaining_ms > read_state.poll_timeout_ms
                ):
                    read_state.read_transient_status_count += 1
                    read_state.read_transient_status_world_state_ms += (
                        native_client_support.coerce_mapping_float(
                            diagnostics,
                            "elapsed_ms",
                            0.0,
                        )
                    )
                    read_state.read_diagnostics.append(dict(diagnostics))
                    self._start_async_render_read(read_state, bindings)
                    return None
                raise translated from exc
            raise
        if native_result is None:
            return None
        read_result = dict(native_result) if isinstance(native_result, Mapping) else {}
        read_state.read_poll_count += 1
        poll_ms = native_client_support.coerce_mapping_float(
            read_result,
            "read_world_state_ms",
            0.0,
        )
        read_state.read_world_state_ms += poll_ms
        read_state.read_diagnostics.append(
            native_client_support.native_response_diagnostics(read_result)
        )
        response_handle = read_result.get("response_handle")
        decoded = read_state.decode_frame(read_state.request_handle, response_handle)
        render_var_path = read_state.render_var_paths[read_state.path_index]
        frame = _selected_color_frame(
            decoded,
            [render_var_path],
            read_state.render_var,
        )
        if frame is not None:
            read_state.path_frame = frame
            read_state.read_success_world_state_ms += poll_ms
        statuses = (
            _result_value(decoded, "statuses", {})
            if isinstance(decoded, Mapping)
            else {}
        )
        status = statuses.get(render_var_path) if isinstance(statuses, Mapping) else None
        if status and str(status) != "OK":
            read_state.path_status = str(status)
        read_state.iterator = str(_result_value(read_result, "iterator", ""))
        if read_state.iterator:
            read_state.read_iterator_count += 1
            read_state.read_iterator_world_state_ms += poll_ms
            self._start_async_render_read(read_state, bindings)
            return None
        if read_state.path_status is not None:
            error = RenderClientError(
                "OVRTX render output "
                f"{render_var_path} terminated with {read_state.path_status}"
            )
            error.render_var_path = render_var_path  # type: ignore[attr-defined]
            error.render_status = read_state.path_status  # type: ignore[attr-defined]
            error.simulation_time_ns = read_state.submission.simulation_time_ns  # type: ignore[attr-defined]
            error.read_diagnostics = tuple(read_state.read_diagnostics)  # type: ignore[attr-defined]
            error.read_world_state_ms = read_state.read_world_state_ms  # type: ignore[attr-defined]
            raise error
        if read_state.path_frame is None:
            read_state.read_empty_ok_count += 1
            read_state.read_empty_ok_world_state_ms += poll_ms
            error = RenderClientError(
                "OVRTX render output "
                f"{render_var_path} sealed without {read_state.render_var} data"
            )
            error.render_var_path = render_var_path  # type: ignore[attr-defined]
            error.render_status = None  # type: ignore[attr-defined]
            error.simulation_time_ns = read_state.submission.simulation_time_ns  # type: ignore[attr-defined]
            error.read_diagnostics = tuple(read_state.read_diagnostics)  # type: ignore[attr-defined]
            error.read_world_state_ms = read_state.read_world_state_ms  # type: ignore[attr-defined]
            raise error
        if read_state.selected_frame is None:
            read_state.selected_frame = read_state.path_frame
        read_state.path_index += 1
        if read_state.path_index < len(read_state.render_var_paths):
            read_state.iterator = ""
            read_state.path_frame = None
            read_state.path_status = None
            self._start_async_render_read(read_state, bindings)
            return None
        return self._finish_async_render_read(read_state)

    def _finish_async_render_read(
        self,
        read_state: _AsyncRenderReadState,
    ) -> RenderResult:
        selected_frame = read_state.selected_frame
        if selected_frame is None:
            raise RenderClientError(
                "Native ovrtx client returned no "
                f"{read_state.render_var} render result"
            )
        result = dict(selected_frame)
        native_timings = _native_timings_from_result(result)
        native_timings.update(
            {
                "read_strategy": "long_poll",
                "read_transport": "async_ticket",
                "read_timeout_ms": read_state.timeout_seconds * 1000,
                "read_poll_count": read_state.read_poll_count,
                "read_iterator_count": read_state.read_iterator_count,
                "read_empty_ok_count": read_state.read_empty_ok_count,
                "read_world_state_ms": read_state.read_world_state_ms,
                "read_success_world_state_ms": read_state.read_success_world_state_ms,
                "read_iterator_world_state_ms": read_state.read_iterator_world_state_ms,
                "read_empty_ok_world_state_ms": read_state.read_empty_ok_world_state_ms,
                "read_transient_status_count": read_state.read_transient_status_count,
                "read_transient_status_world_state_ms": (
                    read_state.read_transient_status_world_state_ms
                ),
                "read_sleep_ms": 0.0,
                "read_empty_ok_sleep_ms": 0.0,
                "read_transient_status_sleep_ms": 0.0,
                "ldr_wait_ms": (
                    time.perf_counter_ns() - read_state.read_started_at
                )
                / 1_000_000.0,
                "read_world_state": read_state.read_diagnostics,
            }
        )
        submission = read_state.submission
        native_timings["read_selection"] = [
            dict(item) for item in submission.selection_diagnostics
        ]
        native_timings["write_world_state"] = [
            dict(item) for item in submission.write_diagnostics
        ]
        result["native_timings"] = native_timings
        native_completed_ns = time.perf_counter_ns()
        convert_started_ns = time.perf_counter_ns()
        render_result = render_result_from_native(
            result,
            read_state.width,
            read_state.height,
        )
        convert_completed_ns = time.perf_counter_ns()
        actual_time_ns = int(render_result.render_output_simulation_time_ns or 0)
        if actual_time_ns and actual_time_ns != submission.simulation_time_ns:
            raise RenderClientError(
                "OVRTX render result time mismatch: "
                f"requested {submission.simulation_time_ns}, received {actual_time_ns}"
            )
        self.last_render_timings = {
            "native_render_ms": (
                native_completed_ns - submission.submitted_monotonic_ns
            )
            / 1_000_000.0,
            "result_convert_ms": (
                convert_completed_ns - convert_started_ns
            )
            / 1_000_000.0,
            "native_timings": native_timings,
        }
        if submission.write_diagnostics:
            self.last_render_timings["write_world_state"] = [
                dict(item) for item in submission.write_diagnostics
            ]
        return render_result

    def update_transforms(
        self,
        simulation_id: str,
        transforms: Sequence[OvrtxTransformValue],
    ) -> OvrtxValueUpdateResult:
        values = _transform_values(transforms)
        attribute_values = tuple(
            attribute_value
            for value in values
            for attribute_value in (
                {
                    "prim_path": value.prim_path,
                    "attribute": RENDER_TRANSFORM_ATTRIBUTE,
                    "value": value.matrix,
                    "value_type": RENDER_TRANSFORM_VALUE_TYPE,
                },
                {
                    "prim_path": value.prim_path,
                    "attribute": "omni:resetXformStack",
                    "value": True,
                    "value_type": "Bool",
                },
            )
        )
        return self._apply_value_update(
            simulation_id,
            updated_count=len(values),
            attribute_values=attribute_values,
            error_context="OVRTX transform value write",
        )

    def update_attribute_values(
        self,
        simulation_id: str,
        values: Sequence[OvrtxAttributeValue],
    ) -> OvrtxValueUpdateResult:
        batch = _attribute_values(values)
        attribute_values = tuple(
            {
                "prim_path": value.prim_path,
                "attribute": value.attribute,
                "value": value.value,
                "value_type": value.value_type,
            }
            for value in batch
        )
        return self._apply_value_update(
            simulation_id,
            updated_count=len(batch),
            attribute_values=attribute_values,
            error_context="viewport attribute update",
        )

    def _apply_value_update(
        self,
        simulation_id: str,
        *,
        updated_count: int,
        attribute_values: Sequence[Mapping[str, Any]],
        error_context: str,
    ) -> OvrtxValueUpdateResult:
        if updated_count == 0:
            return OvrtxValueUpdateResult(0)
        if self._native_client is None:
            raise RenderClientError("No active OVRTX session")
        bindings = self._require_bindings()
        cursor = self._session_state(simulation_id).cursor

        try:
            native_started_ns = time.perf_counter_ns()
            result, pending_time_ns = self._write_update(
                bindings,
                cursor,
                simulation_id,
                bindings.build_attribute_values_update,
                {"attribute_values": list(attribute_values)},
            )
            result.update(
                {
                    "completed_samples": 0,
                    "session_completed_samples": cursor.session_completed_samples,
                }
            )
            native_completed_ns = time.perf_counter_ns()
            self._record_value_update_timings(result, native_completed_ns - native_started_ns)
            return OvrtxValueUpdateResult(
                updated_count=updated_count,
                pending_simulation_time_ns=pending_time_ns,
                diagnostics=result,
            )
        except RenderClientError:
            raise
        except Exception as exc:
            raise RenderClientError(
                f"Native ovrtx client failed before {error_context}: {exc}"
            ) from exc

    def _record_value_update_timings(self, result: Mapping[str, Any], elapsed_ns: int) -> None:
        native_timings = _native_timings_from_result(result)
        native_timings["write_world_state"] = native_client_support.native_response_diagnostics(result)
        self.last_value_update_timings = {
            "native_value_update_ms": elapsed_ns / 1_000_000.0,
            "native_timings": native_timings,
        }

    def _session_state(self, simulation_id: str) -> _OvrtxSessionState:
        try:
            return self._session_states[str(simulation_id)]
        except KeyError as exc:
            raise RenderClientError(f"Unknown OVRTX simulation ID: {simulation_id}") from exc

    def delete_simulation(self, simulation_id: str) -> str:
        simulation_id = str(simulation_id or "")
        if simulation_id not in self._session_states:
            self.last_delete_diagnostics = {"status": "not_found", "simulation_id": simulation_id}
            return "not_found"
        active_read = self._active_render_read
        if (
            active_read is not None
            and active_read.ticket.simulation_id == simulation_id
        ):
            self.cancel_render_sample_read(active_read.ticket)
        if self._abort_cleanup:
            return self.shutdown()
        bindings = self._control_plane_bindings
        if bindings is None:
            self.last_delete_diagnostics = {
                "status": "failed",
                "simulation_id": simulation_id,
                "error": "control_plane_unavailable",
            }
            return "failed"
        try:
            response = bindings.delete_simulation({"simulation_id": simulation_id})
        except RenderClientError as exc:
            diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {
                "error": str(exc),
                "request": {"simulation_id": simulation_id},
            }
            if str(diagnostics.get("grpc_status", "")) == "NOT_FOUND":
                self._session_states.pop(simulation_id, None)
                self.last_delete_diagnostics = {**dict(diagnostics), "status": "not_found"}
                return "not_found"
            self.last_delete_diagnostics = {**dict(diagnostics), "status": "failed"}
            return "failed"
        self._session_states.pop(simulation_id, None)
        self.last_delete_diagnostics = {
            **native_client_support.native_response_diagnostics(response),
            "status": "stopped",
        }
        return "stopped"

    def shutdown(
        self,
        *,
        reset_startup_diagnostics: bool = True,
        release_gpu_lease: bool = True,
    ) -> str:
        active_read = self._active_render_read
        if active_read is not None:
            try:
                self.cancel_render_sample_read(active_read.ticket)
            except RenderClientError:
                # Cancellation failure sets ``_abort_cleanup``. Continue into
                # native close without issuing DeleteSimulation beneath an RPC
                # whose terminal completion could not be confirmed.
                pass
        abort_cleanup = self._abort_cleanup
        simulation_ids = tuple(self._session_states)
        if abort_cleanup:
            statuses: list[str] = []
            self.last_delete_diagnostics = {
                "status": "skipped",
                "reason": "render_failed",
                "simulation_ids": list(simulation_ids),
            }
        else:
            statuses = [
                self.delete_simulation(simulation_id)
                for simulation_id in simulation_ids
            ]
            if "failed" in statuses:
                return "failed"
        native_client = self._native_client
        native_module = self._native_module
        control_plane = self._control_plane_bindings
        self._native_client = None
        self._native_endpoint = ""
        self._native_module = None
        self._native_bindings = None
        self._control_plane_bindings = None
        self._session_states.clear()
        self._abort_cleanup = False
        if reset_startup_diagnostics:
            self.startup_diagnostics = {"render_worker": {"status": "not_started"}}
        try:
            if control_plane is not None:
                control_plane.close()
        finally:
            try:
                if native_client is not None:
                    native_client.close()
            finally:
                shutdown = getattr(native_module, "shutdown", None)
                try:
                    from . import runtime_services

                    if shutdown is not None and not runtime_services.owner.owns_module(native_module):
                        shutdown()
                finally:
                    if release_gpu_lease:
                        self._release_gpu_lease()
        if native_client is None:
            if release_gpu_lease:
                self._release_gpu_lease()
            return "stopped" if statuses or abort_cleanup else "not_found"
        return "stopped" if statuses or abort_cleanup else "not_found"

    def _release_gpu_lease(self) -> None:
        lease = self._gpu_lease
        self._gpu_lease = None
        if lease is not None:
            lease.close()

    def _import_native_client(self, module_name: str) -> Any:
        if self._native_module is not None:
            return self._native_module
        self.shutdown(release_gpu_lease=False)
        native_module = importlib.import_module(module_name)
        self._native_module = native_module
        return self._native_module

    def _ensure_client(self, native_module: Any, endpoint: str) -> None:
        if self._native_client is not None and self._native_endpoint == endpoint:
            return
        if self._native_client is not None:
            self._native_client.close()
        self._native_client = native_module.Client(endpoint)
        self._native_endpoint = endpoint
        self._native_bindings = _bind_render_native_client(native_module, self._native_client)

    def _require_bindings(self) -> _RenderNativeBindings:
        if self._native_bindings is None:
            raise RenderClientError("Native ovrtx client bindings are not initialized")
        return self._native_bindings

    def _cleanup_on_worker_attach(
        self,
        bindings: _ControlPlaneBindings,
        *,
        endpoint: str,
        worker_launched: bool,
    ) -> dict[str, Any]:
        """Sweep stale simulations once per worker attach (task05-02).

        Session creation on an already-attached worker performs no
        list/delete sweep and no retry sleeps. The sweep runs on the
        first attach to a control-plane endpoint in this process, on
        the calling (RPC) thread:

        - Worker this add-on just launched (the native client owns a
          live worker process it spawned): full sweep — nothing else
          can own simulations on it.
        - Pre-existing worker (env-configured endpoint): scoped sweep
          of simulation IDs matching this add-on's PID-derived naming
          convention whose PID is dead (crash orphans).

        A failed sweep leaves the endpoint unattached so the next
        session start retries it.
        """

        endpoint = str(endpoint or "")
        with _worker_attach_lock:
            if endpoint in _attached_worker_endpoints:
                return {
                    "status": "skipped",
                    "reason": "worker_already_attached",
                    "endpoint": endpoint,
                    "deleted_count": 0,
                }
            scope = ATTACH_CLEANUP_SCOPE_FULL if worker_launched else ATTACH_CLEANUP_SCOPE_DEAD_PID
            diagnostics = self._clear_stale_simulations(bindings, scope=scope)
            _attached_worker_endpoints.add(endpoint)
            diagnostics["status"] = "swept"
            diagnostics["scope"] = scope
            diagnostics["endpoint"] = endpoint
            return diagnostics

    def _clear_stale_simulations(
        self,
        bindings: _ControlPlaneBindings,
        *,
        scope: str = ATTACH_CLEANUP_SCOPE_FULL,
    ) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        last_failure: dict[str, Any] | None = None
        last_error: RenderClientError | None = None
        for attempt_index in range(STARTUP_CLEANUP_ATTEMPTS):
            attempt = {"attempt": attempt_index + 1, "list": [], "delete": [], "kept": []}
            attempts.append(attempt)
            try:
                simulations = self._list_existing_simulations(bindings, attempt)
                if scope == ATTACH_CLEANUP_SCOPE_DEAD_PID:
                    simulations, kept = _partition_stale_convention_simulations(simulations)
                    attempt["kept"] = kept
                failures, delete_error = self._delete_existing_simulations(bindings, simulations, attempt)
                if delete_error is not None:
                    last_error = delete_error
            except RenderClientError as exc:
                failure = native_client_support.exception_protocol_diagnostics(exc) or {"error": str(exc)}
                last_failure = failure
                last_error = exc
                failures = [failure]
            if not failures:
                return _cleanup_summary(attempts)
            last_failure = failures[-1]
            if attempt_index + 1 < STARTUP_CLEANUP_ATTEMPTS:
                time.sleep(STARTUP_CLEANUP_RETRY_DELAY_SECONDS)
        error = RenderClientError("OVRTX cleanup failed before OVRTX session start")
        error.protocol_diagnostics = {  # type: ignore[attr-defined]
            "cleanup": _cleanup_summary(attempts),
            "last_failure": last_failure or {"error": "unknown cleanup failure"},
        }
        raise error from last_error

    def _list_existing_simulations(self, bindings: _ControlPlaneBindings, attempt: dict[str, Any]) -> list[str]:
        simulations: list[str] = []
        offset = 0
        while True:
            request = {"limit": STARTUP_CLEANUP_PAGE_LIMIT, "offset": offset}
            try:
                list_result = dict(bindings.list_simulations(request))
            except RenderClientError as exc:
                diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {"error": str(exc), "request": dict(request)}
                attempt["list"].append(diagnostics)
                raise
            attempt["list"].append(native_client_support.native_response_diagnostics(list_result))
            page = _result_value(list_result, "simulations", [])
            if not isinstance(page, Sequence) or isinstance(page, (str, bytes, bytearray)):
                return simulations
            page_ids = [str(simulation_id) for simulation_id in page if simulation_id]
            simulations.extend(page_ids)
            page_count = len(page_ids)
            total = native_client_support.coerce_mapping_int(list_result, "total", len(simulations))
            if page_count == 0 or page_count < STARTUP_CLEANUP_PAGE_LIMIT or len(simulations) >= total:
                return simulations
            offset += page_count

    def _delete_existing_simulations(
        self,
        bindings: _ControlPlaneBindings,
        simulations: Sequence[str],
        attempt: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RenderClientError | None]:
        failures: list[dict[str, Any]] = []
        last_error: RenderClientError | None = None
        for simulation_id in simulations:
            try:
                delete_result = bindings.delete_simulation({"simulation_id": str(simulation_id)})
            except RenderClientError as exc:
                diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {
                    "error": str(exc),
                    "request": {"simulation_id": str(simulation_id)},
                }
                if str(diagnostics.get("grpc_status")) == "NOT_FOUND":
                    diagnostics = dict(diagnostics)
                    diagnostics["deleted"] = True
                    diagnostics["not_found_race"] = True
                    attempt["delete"].append(diagnostics)
                    continue
                attempt["delete"].append(diagnostics)
                failures.append(diagnostics)
                last_error = exc
                continue
            attempt["delete"].append(native_client_support.native_response_diagnostics(delete_result))
        return failures, last_error

    def _read_color_frame(
        self,
        bindings: _RenderNativeBindings,
        simulation_id: str,
        state: _OvrtxSessionState,
        cursor: _OvrtxSessionCursor,
        *,
        selected_sensor_paths: Sequence[str],
        render_var: str,
    ) -> Mapping[str, Any]:
        render_var = render_var or color_presentation.RENDER_VAR_LDR_COLOR
        render_var_paths = _render_var_paths(selected_sensor_paths, render_var)
        if render_var == color_presentation.RENDER_VAR_HDR_COLOR:
            build_read = bindings.build_read_world_state_hdr_color
            decode_frame = bindings.decode_hdr_color_frame
            if build_read is None or decode_frame is None:
                raise RenderClientError("Native ovrtx client does not support HdrColor readback")
        elif render_var == color_presentation.RENDER_VAR_LDR_COLOR:
            build_read = bindings.build_read_world_state_ldr_color
            decode_frame = bindings.decode_ldr_color_frame
        else:
            raise RenderClientError(f"Unsupported OVRTX render var: {render_var}")
        timeout_seconds = max(1, int(os.environ.get(RENDER_TIMEOUT_ENV, "600")))
        deadline = time.monotonic() + timeout_seconds
        read_diagnostics: list[dict[str, Any]] = []
        read_started_at = time.perf_counter_ns()
        read_poll_count = 0
        read_iterator_count = 0
        read_empty_ok_count = 0
        read_world_state_ms = 0.0
        read_iterator_world_state_ms = 0.0
        read_empty_ok_world_state_ms = 0.0
        read_success_world_state_ms = 0.0
        read_transient_status_count = 0
        read_transient_status_world_state_ms = 0.0
        selected_frame: Mapping[str, Any] | None = None
        for render_var_path in render_var_paths:
            iterator = ""
            path_frame: Mapping[str, Any] | None = None
            path_status: str | None = None
            while True:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise RenderClientError(f"Native ovrtx client returned no {render_var} render result before deadline")
                poll_timeout_ms = min(remaining_ms, RENDER_READ_POLL_TIMEOUT_MS)
                read_request: dict[str, Any] = {
                    "simulation_id": simulation_id,
                    "render_var_paths": [render_var_path],
                    "simulation_time_ns": cursor.simulation_time_ns,
                    "timeout_ms": poll_timeout_ms,
                    "timeout_seconds": max(1, (poll_timeout_ms + 999) // 1000),
                    "width": state.width,
                    "height": state.height,
                    "completed_samples": cursor.snapshot_completed_samples,
                    "session_completed_samples": cursor.session_completed_samples,
                }
                if iterator:
                    read_request["iterator"] = iterator
                request_handle = build_read(read_request)
                try:
                    read_result = dict(
                        _call_native_rpc(bindings, "ReadWorldState", bindings.read_world_state, request_handle)
                    )
                except RenderClientError as exc:
                    diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {}
                    if (
                        str(diagnostics.get("grpc_status", "")) == "DEADLINE_EXCEEDED"
                        and remaining_ms > poll_timeout_ms
                    ):
                        read_transient_status_count += 1
                        read_transient_status_world_state_ms += native_client_support.coerce_mapping_float(
                            diagnostics,
                            "elapsed_ms",
                            0.0,
                        )
                        read_diagnostics.append(dict(diagnostics))
                        continue
                    raise
                read_poll_count += 1
                poll_ms = native_client_support.coerce_mapping_float(read_result, "read_world_state_ms", 0.0)
                read_world_state_ms += poll_ms
                read_diagnostics.append(native_client_support.native_response_diagnostics(read_result))
                response_handle = read_result.get("response_handle")
                decoded = decode_frame(request_handle, response_handle)
                frame = _selected_color_frame(decoded, [render_var_path], render_var)
                if frame is not None:
                    path_frame = frame
                    read_success_world_state_ms += poll_ms
                statuses = _result_value(decoded, "statuses", {}) if isinstance(decoded, Mapping) else {}
                status = statuses.get(render_var_path) if isinstance(statuses, Mapping) else None
                if status and str(status) != "OK":
                    path_status = str(status)
                iterator = str(_result_value(read_result, "iterator", ""))
                if iterator:
                    read_iterator_count += 1
                    read_iterator_world_state_ms += poll_ms
                    continue
                if path_status is not None:
                    error = RenderClientError(f"OVRTX render output {render_var_path} terminated with {path_status}")
                    error.render_var_path = render_var_path  # type: ignore[attr-defined]
                    error.render_status = path_status  # type: ignore[attr-defined]
                    error.simulation_time_ns = cursor.simulation_time_ns  # type: ignore[attr-defined]
                    error.read_diagnostics = tuple(read_diagnostics)  # type: ignore[attr-defined]
                    error.read_world_state_ms = read_world_state_ms  # type: ignore[attr-defined]
                    raise error
                if path_frame is not None:
                    if selected_frame is None:
                        selected_frame = path_frame
                    break
                read_empty_ok_count += 1
                read_empty_ok_world_state_ms += poll_ms
                error = RenderClientError(
                    f"OVRTX render output {render_var_path} sealed without {render_var} data"
                )
                error.render_var_path = render_var_path  # type: ignore[attr-defined]
                error.render_status = None  # type: ignore[attr-defined]
                error.simulation_time_ns = cursor.simulation_time_ns  # type: ignore[attr-defined]
                error.read_diagnostics = tuple(read_diagnostics)  # type: ignore[attr-defined]
                error.read_world_state_ms = read_world_state_ms  # type: ignore[attr-defined]
                raise error

        if selected_frame is None:
            raise RenderClientError(f"Native ovrtx client returned no {render_var} render result")
        result = dict(selected_frame)
        native_timings = _native_timings_from_result(result)
        native_timings["read_strategy"] = "long_poll"
        native_timings["read_timeout_ms"] = timeout_seconds * 1000
        native_timings["read_poll_count"] = read_poll_count
        native_timings["read_iterator_count"] = read_iterator_count
        native_timings["read_empty_ok_count"] = read_empty_ok_count
        native_timings["read_world_state_ms"] = read_world_state_ms
        native_timings["read_success_world_state_ms"] = read_success_world_state_ms
        native_timings["read_iterator_world_state_ms"] = read_iterator_world_state_ms
        native_timings["read_empty_ok_world_state_ms"] = read_empty_ok_world_state_ms
        native_timings["read_transient_status_count"] = read_transient_status_count
        native_timings["read_transient_status_world_state_ms"] = read_transient_status_world_state_ms
        native_timings["read_sleep_ms"] = 0.0
        native_timings["read_empty_ok_sleep_ms"] = 0.0
        native_timings["read_transient_status_sleep_ms"] = 0.0
        native_timings["ldr_wait_ms"] = (time.perf_counter_ns() - read_started_at) / 1_000_000.0
        native_timings["read_world_state"] = read_diagnostics
        result["native_timings"] = native_timings
        return result

    def _write_update(
        self,
        bindings: _RenderNativeBindings,
        cursor: _OvrtxSessionCursor,
        simulation_id: str,
        builder: Callable[[Mapping[str, Any]], Any],
        request: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        pending_time_ns = cursor.simulation_time_ns + DEFAULT_RENDER_SIMULATION_STEP_NS
        build_started_ns = time.perf_counter_ns()
        handle = builder(
            {
                "simulation_id": simulation_id,
                "simulation_time_ns": pending_time_ns,
                **request,
            }
        )
        request_build_ms = (time.perf_counter_ns() - build_started_ns) / 1_000_000.0
        result = dict(_call_native_rpc(bindings, "WriteWorldState", bindings.write_world_state, handle))
        result["request_build_ms"] = request_build_ms
        cursor.snapshot_completed_samples = 0
        return result, pending_time_ns


def _bind_render_native_client(native_module: Any, native_client: Any) -> _RenderNativeBindings:
    def require(name: str) -> Callable[..., Any]:
        return native_client_support.require_callable(
            native_module,
            name,
            client_label=RENDER_NATIVE_CLIENT_LABEL,
            error_type=RenderClientError,
        )

    capabilities_fn = require("capabilities")
    capabilities = capabilities_fn()
    if not isinstance(capabilities, Mapping):
        raise RenderClientError("Native ovrtx client capabilities() did not return a mapping")

    rpcs = native_client_support.capability_names(capabilities, "rpcs")
    required_rpcs = {"CreateSimulation", "WriteWorldState", "ReadWorldState"}
    missing_rpcs = sorted(name for name in required_rpcs if name not in rpcs)
    if missing_rpcs:
        raise RenderClientError(f"Native ovrtx client is missing RPC capabilities: {', '.join(missing_rpcs)}")

    generic_builders = native_client_support.capability_names(capabilities, "generic_builders")
    if "build_WriteWorldState_columns" not in generic_builders:
        raise RenderClientError("Native ovrtx client is missing generic build_WriteWorldState_columns capability")
    if "build_ReadWorldState_ldr_color" not in generic_builders:
        raise RenderClientError("Native ovrtx client is missing generic build_ReadWorldState_ldr_color capability")
    generic_write_builder = require("build_WriteWorldState_columns")
    generic_ldr_color_read_builder = require("build_ReadWorldState_ldr_color")
    generic_hdr_color_read_builder = (
        require("build_ReadWorldState_hdr_color")
        if "build_ReadWorldState_hdr_color" in generic_builders
        else None
    )
    async_rpcs = native_client_support.capability_names(capabilities, "async_rpcs")
    begin_read_world_state = None
    if "ReadWorldState" in async_rpcs:
        begin_read_world_state = native_client_support.optional_callable(
            native_client,
            "begin_ReadWorldState",
        )
        if begin_read_world_state is None:
            raise RenderClientError(
                "Native ovrtx client advertises asynchronous ReadWorldState "
                "but is missing callable begin_ReadWorldState"
            )

    return _RenderNativeBindings(
        start_worker=require("start_worker"),
        create_simulation=native_client_support.require_callable(
            native_client, "CreateSimulation", client_label=RENDER_NATIVE_CLIENT_LABEL, error_type=RenderClientError
        ),
        write_world_state=native_client_support.require_callable(
            native_client, "WriteWorldState", client_label=RENDER_NATIVE_CLIENT_LABEL, error_type=RenderClientError
        ),
        read_world_state=native_client_support.require_callable(
            native_client, "ReadWorldState", client_label=RENDER_NATIVE_CLIENT_LABEL, error_type=RenderClientError
        ),
        begin_read_world_state=begin_read_world_state,
        build_write_world_state_columns=generic_write_builder,
        build_render_sample_step=_builder_or_generic(
            native_module,
            "build_render_sample_step",
            generic_write_builder,
            _sample_step_write_request,
        ),
        build_attribute_values_update=_builder_or_generic(
            native_module,
            "build_attribute_values_update",
            generic_write_builder,
            _attribute_values_write_request,
        ),
        build_read_world_state_ldr_color=generic_ldr_color_read_builder,
        build_read_world_state_hdr_color=generic_hdr_color_read_builder,
        decode_ldr_color_frame=require("decode_ldr_color_frame"),
        decode_hdr_color_frame=native_client_support.optional_callable(native_module, "decode_hdr_color_frame"),
        rpc_status_error=native_client_support.rpc_status_error_type(
            native_module,
            client_label=RENDER_NATIVE_CLIENT_LABEL,
            error_type=RenderClientError,
        ),
    )


def _bind_official_control_plane(start_result: Mapping[str, Any], worker_command: str) -> _ControlPlaneBindings:
    endpoint = _control_plane_endpoint(start_result, worker_command)
    try:
        grpc_module = importlib.import_module("grpc")
        simulation_pb2 = importlib.import_module("srtx_protos.api.simulation.v1.simulation_pb2")
        simulation_pb2_grpc = importlib.import_module("srtx_protos.api.simulation.v1.simulation_pb2_grpc")
    except Exception as exc:
        raise RenderClientError(f"Official OVRTX control-plane gRPC modules are unavailable: {exc}") from exc

    channel = _create_grpc_channel(grpc_module, endpoint)
    stub = simulation_pb2_grpc.ControlPlaneServiceStub(channel)

    def list_simulations(request: Mapping[str, Any]) -> Mapping[str, Any]:
        diagnostics_request = {
            "limit": native_client_support.coerce_mapping_int(request, "limit", STARTUP_CLEANUP_PAGE_LIMIT),
            "offset": native_client_support.coerce_mapping_int(request, "offset", 0),
        }
        rpc_request = simulation_pb2.ListSimulationsRequest(**diagnostics_request)
        response, elapsed_ms = _call_official_control_plane_rpc(
            grpc_module,
            "ControlPlaneService.ListSimulations",
            stub.ListSimulations,
            rpc_request,
            diagnostics_request,
        )
        simulations = [
            str(getattr(simulation, "simulation_id", ""))
            for simulation in getattr(response, "simulations", ())
            if getattr(simulation, "simulation_id", "")
        ]
        return {
            **_grpc_success_diagnostics(
                "ControlPlaneService.ListSimulations",
                elapsed_ms=elapsed_ms,
                request=diagnostics_request,
            ),
            "simulations": simulations,
            "total": int(getattr(response, "total", len(simulations))),
        }

    def delete_simulation(request: Mapping[str, Any]) -> Mapping[str, Any]:
        simulation_id = str(request.get("simulation_id", ""))
        diagnostics_request = {"simulation_id": simulation_id}
        rpc_request = simulation_pb2.DeleteSimulationRequest(simulation_id=simulation_id)
        _response, elapsed_ms = _call_official_control_plane_rpc(
            grpc_module,
            "ControlPlaneService.DeleteSimulation",
            stub.DeleteSimulation,
            rpc_request,
            diagnostics_request,
        )
        return {
            **_grpc_success_diagnostics(
                "ControlPlaneService.DeleteSimulation",
                elapsed_ms=elapsed_ms,
                request=diagnostics_request,
            ),
            "simulation_id": simulation_id,
            "deleted": True,
        }

    def close() -> None:
        closer = getattr(channel, "close", None)
        if callable(closer):
            closer()

    return _ControlPlaneBindings(
        list_simulations=list_simulations,
        delete_simulation=delete_simulation,
        close=close,
    )


def _create_grpc_channel(grpc_module: Any, endpoint: str) -> Any:
    return grpc_module.insecure_channel(endpoint)


def _call_official_control_plane_rpc(
    grpc_module: Any,
    method: str,
    function: Callable[..., Any],
    request: Any,
    diagnostics_request: Mapping[str, Any],
) -> tuple[Any, float]:
    started_ns = time.perf_counter_ns()
    try:
        response = function(request, timeout=CONTROL_PLANE_RPC_TIMEOUT_SECONDS)
    except grpc_module.RpcError as exc:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        raise _render_grpc_rpc_error(method, exc, request=diagnostics_request, elapsed_ms=elapsed_ms) from exc
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return response, elapsed_ms


def _render_grpc_rpc_error(
    method: str,
    exc: BaseException,
    *,
    request: Mapping[str, Any],
    elapsed_ms: float,
) -> RenderClientError:
    diagnostics = _grpc_rpc_error_diagnostics(method, exc, request=request, elapsed_ms=elapsed_ms)
    status = str(diagnostics.get("grpc_status") or "UNKNOWN")
    message = f"OVRTX cleanup {method} failed with gRPC status {status}"
    detail = str(diagnostics.get("grpc_message") or "").strip()
    if detail:
        message = f"{message}: {detail}"
    error = RenderClientError(message)
    error.protocol_diagnostics = diagnostics  # type: ignore[attr-defined]
    return error


def _grpc_success_diagnostics(method: str, *, elapsed_ms: float, request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "method": method,
        "protocol_method": method,
        "code": "OK",
        "details": "",
        "grpc_status": "OK",
        "grpc_status_code": 0,
        "grpc_message": "",
        "elapsed_ms": elapsed_ms,
        "request": dict(request),
    }


def _grpc_rpc_error_diagnostics(
    method: str,
    exc: BaseException,
    *,
    request: Mapping[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    code = _call_error_method(exc, "code")
    code_name = _grpc_status_name(code)
    details = str(_call_error_method(exc, "details") or "")
    diagnostics: dict[str, Any] = {
        "ok": False,
        "method": method,
        "protocol_method": method,
        "code": code_name,
        "details": details,
        "grpc_status": code_name,
        "grpc_message": details,
        "elapsed_ms": elapsed_ms,
        "request": dict(request),
    }
    code_number = _grpc_status_code_number(code)
    if code_number is not None:
        diagnostics["grpc_status_code"] = code_number
    initial_metadata = _metadata_snapshot(_call_error_method(exc, "initial_metadata"))
    if initial_metadata is not None:
        diagnostics["initial_metadata"] = initial_metadata
    trailing_metadata = _metadata_snapshot(_call_error_method(exc, "trailing_metadata"))
    if trailing_metadata is not None:
        diagnostics["trailing_metadata"] = trailing_metadata
    return diagnostics


def _call_error_method(exc: BaseException, name: str) -> Any:
    method = getattr(exc, name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _grpc_status_name(code: Any) -> str:
    name = getattr(code, "name", None)
    if isinstance(name, str) and name:
        return name
    value = str(code or "UNKNOWN")
    if value.startswith("StatusCode."):
        return value.split(".", 1)[1]
    return value or "UNKNOWN"


def _grpc_status_code_number(code: Any) -> int | None:
    value = getattr(code, "value", None)
    if isinstance(value, tuple) and value:
        try:
            return int(value[0])
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_snapshot(metadata: Any) -> list[dict[str, str]] | None:
    if metadata is None:
        return None
    entries: list[dict[str, str]] = []
    try:
        iterator = iter(metadata)
    except TypeError:
        return None
    for entry in iterator:
        try:
            key, value = entry
        except (TypeError, ValueError):
            continue
        key_text = str(key)
        value_text = "<redacted>" if _sensitive_metadata_key(key_text) else str(value)
        entries.append({"key": key_text, "value": value_text})
    return entries


def _sensitive_metadata_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in ("TOKEN", "SECRET", "PASSWORD", "PASS", "API_KEY", "CREDENTIAL"))


def _control_plane_endpoint(start_result: Mapping[str, Any], worker_command: str) -> str:
    endpoint = str(start_result.get("endpoint", "")).strip()
    if endpoint:
        return endpoint
    address = _srtx_server_address_from_worker_command(worker_command) or DEFAULT_CONTROL_PLANE_ADDRESS
    port = _srtx_server_port_from_worker_command(worker_command) or DEFAULT_CONTROL_PLANE_PORT
    return f"{address}:{port}"


def _call_native_rpc(
    bindings: _RenderNativeBindings,
    name: str,
    function: Callable[[Any], Any],
    argument: Any,
) -> Mapping[str, Any]:
    return native_client_support.call_native_rpc(
        name,
        function,
        argument,
        rpc_status_error=bindings.rpc_status_error,
        client_label=RENDER_NATIVE_CLIENT_LABEL,
        error_type=RenderClientError,
    )


def _builder_or_generic(
    native_module: Any,
    builder_name: str,
    generic_write_builder: Callable[[Mapping[str, Any]], Any],
    fallback: Callable[[Callable[[Mapping[str, Any]], Any], Mapping[str, Any]], Any],
) -> Callable[[Mapping[str, Any]], Any]:
    builder = getattr(native_module, builder_name, None)
    if callable(builder):
        return builder

    def build_with_generic(request: Mapping[str, Any]) -> Any:
        return fallback(generic_write_builder, request)

    return build_with_generic


def _sample_step_write_request(
    generic_write_builder: Callable[[Mapping[str, Any]], Any],
    request: Mapping[str, Any],
) -> Any:
    simulation_time_ns = int(_required_value(request, "simulation_time_ns"))
    return generic_write_builder(
        {
            "simulation_id": str(_required_value(request, "simulation_id")),
            "simulation_time_ns": int(request.get("write_time_ns", simulation_time_ns)),
            "write": [{"update_simulation_time_ns": simulation_time_ns}],
        }
    )


def _attribute_values_write_request(
    generic_write_builder: Callable[[Mapping[str, Any]], Any],
    request: Mapping[str, Any],
) -> Any:
    attribute_values = _required_sequence(request, "attribute_values")
    groups: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for attribute_value in attribute_values:
        attribute = str(_required_value(attribute_value, "attribute"))
        value = _required_value(attribute_value, "value")
        value_type = str(attribute_value.get("value_type") or _infer_value_type(value))
        group = groups.setdefault((attribute, value_type), {"prim_paths": [], "values": []})
        group["prim_paths"].append(str(_required_value(attribute_value, "prim_path")))
        group["values"].append(value)

    write = []
    for (attribute, value_type), group in groups.items():
        write.append(
            {
                "keys": {"attribute": "usd-path", "values": group["prim_paths"]},
                "columns": [
                    {
                        "attribute": attribute,
                        "type": value_type,
                        "values": group["values"],
                    }
                ],
            }
        )
    return generic_write_builder(
        {
            "simulation_id": str(_required_value(request, "simulation_id")),
            "simulation_time_ns": int(_required_value(request, "simulation_time_ns")),
            "write": write,
        }
    )


def _required_sequence(mapping: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = _required_value(mapping, key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RenderClientError(f"{key} must be a sequence")
    items = list(value)
    if not items:
        raise RenderClientError(f"{key} must contain at least one value")
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise RenderClientError(f"{key}[{index}] must be a mapping")
    return items


def _required_value(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise RenderClientError(f"{key} is required")
    value = mapping[key]
    if value is None:
        raise RenderClientError(f"{key} is required")
    return value


def _infer_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, (int, float)):
        return "Float"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 3:
        return "Float3"
    raise RenderClientError("attribute values require value_type for non-scalar values")


def _cleanup_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delete_results: list[Mapping[str, Any]] = []
    list_results: list[Mapping[str, Any]] = []
    kept_ids: list[str] = []
    for attempt in attempts:
        delete_entries = attempt.get("delete", ())
        if isinstance(delete_entries, Sequence) and not isinstance(delete_entries, (str, bytes, bytearray)):
            delete_results.extend(entry for entry in delete_entries if isinstance(entry, Mapping))
        list_entries = attempt.get("list", ())
        if isinstance(list_entries, Sequence) and not isinstance(list_entries, (str, bytes, bytearray)):
            list_results.extend(entry for entry in list_entries if isinstance(entry, Mapping))
        kept_entries = attempt.get("kept", ())
        if isinstance(kept_entries, Sequence) and not isinstance(kept_entries, (str, bytes, bytearray)):
            kept_ids.extend(str(entry) for entry in kept_entries if str(entry) not in kept_ids)
    return {
        "attempts": [dict(attempt) for attempt in attempts],
        "list": [dict(entry) for entry in list_results],
        "delete": [dict(entry) for entry in delete_results],
        "kept": kept_ids,
        "deleted_count": sum(1 for entry in delete_results if str(entry.get("grpc_status", "OK")) in {"OK", "NOT_FOUND"}),
    }


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def render_result_from_native(result: Any, fallback_width: int, fallback_height: int) -> RenderResult:
    rgba8 = _result_value(result, "rgba8")
    if not isinstance(rgba8, (bytes, bytearray, memoryview)):
        raise RenderClientError("Native ovrtx client returned no RGBA8 render result")
    width = int(_result_value(result, "width", fallback_width))
    height = int(_result_value(result, "height", fallback_height))
    payload = bytes(rgba8)
    expected_size = width * height * 4
    if len(payload) != expected_size:
        raise RenderClientError(
            f"Native ovrtx client returned {len(payload)} RGBA bytes for a {width}x{height} render result"
        )
    completed_samples = int(_result_value(result, "completed_samples", 0))
    session_completed_samples = int(_result_value(result, "session_completed_samples", completed_samples))
    simulation_time_ns = int(_result_value(result, "simulation_time_ns", 0))
    render_output_simulation_time_ns = int(_result_value(result, "render_output_simulation_time_ns", 0))
    frame_format = str(
        _result_value(result, "frame_format", color_presentation.FRAME_FORMAT_RGBA8)
        or color_presentation.FRAME_FORMAT_RGBA8
    )
    frame_color_mode = str(
        _result_value(result, "frame_color_mode", color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR)
        or color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    )
    if frame_format != color_presentation.FRAME_FORMAT_RGBA8:
        if frame_format != color_presentation.FRAME_FORMAT_RGBA16F:
            raise RenderClientError(f"Native ovrtx client returned unsupported frame format: {frame_format}")
        if frame_color_mode != color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR:
            raise RenderClientError(f"Native ovrtx client returned unsupported frame color mode: {frame_color_mode}")
        linear_rgba16f = _result_value(result, "linear_rgba16f")
        if not isinstance(linear_rgba16f, (bytes, bytearray, memoryview)):
            raise RenderClientError("Native ovrtx client returned no HdrColor linear RGBA16F payload")
        linear_payload = bytes(linear_rgba16f)
        expected_linear_size = width * height * 8
        if len(linear_payload) != expected_linear_size:
            raise RenderClientError(
                f"Native ovrtx client returned {len(linear_payload)} RGBA16F bytes for a {width}x{height} render result"
            )
        linear_payload = _flip_payload_rows(linear_payload, width, height, 8)
    else:
        if frame_color_mode != color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR:
            raise RenderClientError(f"Native ovrtx client returned unsupported frame color mode: {frame_color_mode}")
        linear_payload = b""
    render_var = str(_result_value(result, "render_var", color_presentation.RENDER_VAR_LDR_COLOR) or "")
    native_timings = _native_timings_from_result(result)
    payload = _flip_payload_rows(payload, width, height, 4)
    return RenderResult(
        width=width,
        height=height,
        rgba8=payload,
        completed_samples=completed_samples,
        session_completed_samples=session_completed_samples,
        simulation_time_ns=simulation_time_ns,
        render_output_simulation_time_ns=render_output_simulation_time_ns,
        frame_format=frame_format,
        frame_color_mode=frame_color_mode,
        render_var=render_var or color_presentation.RENDER_VAR_LDR_COLOR,
        linear_rgba16f=linear_payload,
        native_timings=native_timings,
    )


def _native_timings_from_result(result: Any) -> dict[str, Any]:
    timings = _result_value(result, "native_timings")
    return dict(timings) if isinstance(timings, Mapping) else {}


def _render_worker_startup_diagnostics(
    native_client: Any,
    *,
    simulation_id: str,
    worker_command: str,
    logs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "status": "running",
        "simulation_id": simulation_id,
        "worker_command": worker_command,
        "worker_environment": worker_runtime_environment_evidence(worker_command),
        "logs": dict(logs or session_lifecycle.log_diagnostics()),
    }
    health = getattr(native_client, "check_health", None)
    if not callable(health):
        diagnostics["health_status"] = "unavailable"
        return diagnostics
    try:
        health_result = health()
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return diagnostics
    if isinstance(health_result, Mapping):
        diagnostics["health"] = dict(health_result)
        serving = health_result.get("serving")
        if serving is False:
            diagnostics["status"] = "failed"
            diagnostics["error"] = "render worker health reported serving=false"
        return diagnostics
    diagnostics["health_status"] = "unexpected_result"
    return diagnostics


def _render_worker_failure_diagnostics(
    *,
    worker_command: str,
    error: str,
    logs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "worker_command": worker_command,
        "worker_environment": worker_runtime_environment_evidence(worker_command),
        "error": error,
        "logs": dict(logs or session_lifecycle.log_diagnostics()),
    }


def _flip_rgba8_rows(payload: bytes, width: int, height: int) -> bytes:
    return _flip_payload_rows(payload, width, height, 4)


def _flip_payload_rows(payload: bytes, width: int, height: int, bytes_per_pixel: int) -> bytes:
    row_size = width * bytes_per_pixel
    return b"".join(payload[row * row_size : (row + 1) * row_size] for row in range(height - 1, -1, -1))


def _sanitize_worker_environment() -> dict[str, str | None]:
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not library_path:
        return {}
    entries = library_path.split(os.pathsep)
    kept = [entry for entry in entries if not entry.startswith("/snap/blender/")]
    if kept == entries:
        return {}
    previous = {"LD_LIBRARY_PATH": library_path}
    if kept:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    return previous


def apply_srtx_server_port_env(env: MutableMapping[str, str], worker_command: str) -> dict[str, str | None]:
    port = _srtx_server_port_from_worker_command(worker_command)
    if not port:
        return {}
    previous = env.get("SRTX_SERVER_PORT")
    env["SRTX_SERVER_PORT"] = port
    return {"SRTX_SERVER_PORT": previous}


def apply_worker_runtime_environment(env: MutableMapping[str, str], worker_command: str) -> dict[str, str | None]:
    previous = dict(apply_srtx_server_port_env(env, worker_command))
    configured_active_cuda_gpus = env.get(ACTIVE_CUDA_GPUS_ENV)
    active_cuda_gpus = (
        "0"
        if configured_active_cuda_gpus is None
        else configured_active_cuda_gpus.strip()
    )
    previous[ACTIVE_CUDA_GPUS_ENV] = env.get(ACTIVE_CUDA_GPUS_ENV)
    env[ACTIVE_CUDA_GPUS_ENV] = active_cuda_gpus
    plugins_path = worker_plugins_search_path_from_worker_command(worker_command)
    if plugins_path and os.name == "nt":
        current_path = env.get("PATH", "")
        updated_path = _prepend_env_path(current_path, plugins_path)
        if updated_path != current_path:
            previous["PATH"] = env.get("PATH")
            env["PATH"] = updated_path
    materialx_mdl_path = materialx_mdl_search_path_from_worker_command(worker_command)
    if not materialx_mdl_path:
        return previous
    current = env.get(MDL_SYSTEM_PATH_ENV, "")
    updated = _prepend_env_path(current, materialx_mdl_path)
    if updated == current:
        return previous
    previous[MDL_SYSTEM_PATH_ENV] = env.get(MDL_SYSTEM_PATH_ENV)
    env[MDL_SYSTEM_PATH_ENV] = updated
    return previous


def worker_runtime_environment_evidence(worker_command: str) -> dict[str, Any]:
    materialx_mdl_path = materialx_mdl_search_path_from_worker_command(worker_command)
    plugins_path = worker_plugins_search_path_from_worker_command(worker_command)
    return {
        "srtx_server_port": _srtx_server_port_from_worker_command(worker_command),
        "materialx_mdl_search_path": materialx_mdl_path,
        "materialx_mdl_search_path_configured": bool(materialx_mdl_path),
        "worker_plugins_search_path": plugins_path,
        "worker_plugins_search_path_configured": bool(plugins_path),
    }


def worker_plugins_search_path_from_worker_command(worker_command: str) -> str:
    """The worker package's staged ``plugins`` directory, when present.

    The pinned Windows worker exits during loader startup (0xC0000135)
    unless the monolithic USD DLL's Alembic/Imath/MaterialX dependencies
    are on the DLL search path; the release stages them under the worker
    package root's ``plugins`` directory rather than beside the executable
    (runtime measurements "Windows OVRTX 0.3 worker build still links the
    retired USD library name"). Prepending this directory to the child
    process PATH before ``start_worker`` is the add-on-side search-path
    mitigation that finding asks for; the Team Green staging fix remains
    the durable resolution.
    """

    package_root = _worker_package_root_from_worker_command(worker_command)
    if package_root is None:
        return ""
    path = package_root / "plugins"
    return str(path) if path.is_dir() else ""


def materialx_mdl_search_path_from_worker_command(worker_command: str) -> str:
    package_root = _worker_package_root_from_worker_command(worker_command)
    if package_root is None:
        return ""
    path = package_root / "library" / "materialx" / "mdl"
    return str(path) if path.is_dir() else ""


def _worker_package_root_from_worker_command(worker_command: str) -> Path | None:
    try:
        parts = bundled_runtime.parse_command(worker_command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part == "--package-root" and index + 1 < len(parts):
            return _worker_package_root_path(parts[index + 1])
        if part.startswith("--package-root="):
            return _worker_package_root_path(part.split("=", 1)[1])
    return None


def _worker_package_root_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else path.resolve()


def _prepend_env_path(current: str, path: str) -> str:
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if path in entries:
        return current
    return os.pathsep.join([path, *entries])


def _restore_environment(previous_values: Mapping[str, str | None]) -> None:
    for key, value in previous_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _srtx_server_port_from_worker_command(worker_command: str) -> str:
    try:
        parts = bundled_runtime.parse_command(worker_command)
    except ValueError:
        return ""
    for index, part in enumerate(parts):
        if part == "--port" and index + 1 < len(parts):
            return _valid_port(parts[index + 1])
        if part.startswith("--port="):
            return _valid_port(part.split("=", 1)[1])
    return ""


def _srtx_server_address_from_worker_command(worker_command: str) -> str:
    try:
        parts = bundled_runtime.parse_command(worker_command)
    except ValueError:
        return ""
    for index, part in enumerate(parts):
        if part == "--address" and index + 1 < len(parts):
            return _control_plane_address(parts[index + 1])
        if part.startswith("--address="):
            return _control_plane_address(part.split("=", 1)[1])
    return ""


def _control_plane_address(value: str) -> str:
    address = value.strip()
    if not address or address == "0.0.0.0":
        return ""
    return address


def _valid_port(value: str) -> str:
    try:
        port = int(value)
    except ValueError:
        return ""
    if 0 < port <= 65535:
        return str(port)
    return ""


__all__ = [
    "OvrtxRuntimeClient",
    "RenderClientError",
    "RenderReadTicket",
    "RenderResult",
    "RenderSubmission",
    "apply_worker_runtime_environment",
    "apply_srtx_server_port_env",
    "materialx_mdl_search_path_from_worker_command",
    "render_result_from_native",
]
