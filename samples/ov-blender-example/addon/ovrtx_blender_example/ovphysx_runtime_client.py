# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVPhysX runtime client boundary for shared-stage composition."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import bundled_runtime
from . import native_client_support
from .shared_stage_errors import SharedStageCompositionError
from .shared_stage_composition import BodyPose, BodyVelocity


DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH = Path("native")
DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE = "ovphysx_bridge_client"
UNKNOWN = "???"
INT64_MAX = 2**63 - 1
OVPHYSX_NATIVE_CLIENT_LABEL = "Native OVPhysX client"


@dataclass(frozen=True)
class _PhysicsNativeBindings:
    start_worker: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    connect: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    create_simulation: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    delete_simulation: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    write_world_state: Callable[[Any], Mapping[str, Any]]
    read_world_state: Callable[[Any], Mapping[str, Any]]
    build_write_world_state_step: Callable[[Mapping[str, Any]], Any]
    build_write_world_state_body_poses: Callable[[Mapping[str, Any]], Any]
    build_write_world_state_body_velocities: Callable[[Mapping[str, Any]], Any]
    build_read_world_state_body_states: Callable[[Mapping[str, Any]], Any]
    decode_body_states: Callable[[Any, Any], Mapping[str, Any]]
    rpc_status_error: type[BaseException] | None


class OvphysxRuntimeConfig(Protocol):
    input_usd_path: str
    ovphysx_address: str
    ovphysx_worker_command: str
    body_prims: Sequence[str]
    worker_log_path: str
    ovphysx_native_client_module: str
    ovphysx_native_client_path: str


class OvphysxRuntimeClient:
    transport = "native"
    topology_description = f"{DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE} native extension"
    worker_description = "managed by ovphysx_bridge_client native adapter"

    def __init__(
        self,
        config: OvphysxRuntimeConfig,
        simulation_id: str,
        native_module: Any | None = None,
    ) -> None:
        self.config = config
        self.simulation_id = simulation_id
        self._native_module = native_module
        self._native_client: Any | None = None
        self._native_endpoint = ""
        self._native_bindings: _PhysicsNativeBindings | None = None
        self._created = False
        self._started = False
        self.last_delete_diagnostics: dict[str, Any] = {}
        if native_module is not None:
            self._ensure_client(native_module, self.config.ovphysx_address)

    def start(self) -> None:
        if self._started:
            return
        from . import runtime_services

        services = runtime_services.owner.diagnostics()
        if services["status"] == "starting":
            raise SharedStageCompositionError("Runtime services are still preparing")
        if services["status"] == "failed":
            raise SharedStageCompositionError(
                f"Runtime service preparation failed: {services['error']}"
            )
        native_module = self._import_native_client()
        health_deadline = time.monotonic() + runtime_services.health_timeout_seconds()
        command = self.config.ovphysx_worker_command.strip()
        start_worker = native_client_support.optional_callable(native_module, "start_worker")
        connect = native_client_support.optional_callable(native_module, "connect")
        if command and start_worker is not None:
            worker_environment = _ovphysx_worker_environment(self.config)
            changed_environment = {
                name: os.environ.get(name)
                for name, value in worker_environment.items()
                if os.environ.get(name) != value
            }
            os.environ.update(worker_environment)
            request = {
                "worker_command": command,
                "address": self.config.ovphysx_address,
                "log_path": self.config.worker_log_path,
                "ready_timeout_ms": runtime_services._remaining_milliseconds(health_deadline),
            }
            try:
                connection = self._call_bound("start_worker", start_worker, request)
            finally:
                for name, previous in changed_environment.items():
                    if previous is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = previous
        elif connect is not None:
            connection = self._call_bound("connect", connect, {"address": self.config.ovphysx_address})
        else:
            raise SharedStageCompositionError(
                "Native OVPhysX client has neither start_worker nor connect"
            )
        try:
            endpoint = str(connection.get("address", self.config.ovphysx_address))
            self._ensure_client(native_module, endpoint)
            runtime_services.wait_for_serving(
                "OVPhysX",
                endpoint,
                lambda timeout: self._native_client.health("", timeout),
                process_alive=runtime_services._process_alive(native_module, connection),
                deadline=health_deadline,
            )
        except Exception as exc:
            self._close_client()
            if (
                command
                and start_worker is not None
                and not runtime_services.owner.owns_module(native_module)
            ):
                shutdown = getattr(native_module, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            if isinstance(exc, SharedStageCompositionError):
                raise
            error = SharedStageCompositionError(f"Native OVPhysX client health failed: {exc}")
            diagnostics = native_client_support.exception_protocol_diagnostics(exc)
            if diagnostics is not None:
                error.protocol_diagnostics = diagnostics  # type: ignore[attr-defined]
            raise error from exc
        self._started = True

    def create_simulation(self) -> Mapping[str, Any]:
        request = {
            "simulation_id": self.simulation_id,
            "usd_file_uri": _usd_file_uri(self.config.input_usd_path),
        }
        response = self._call_bound("CreateSimulation", self._require_bindings().create_simulation, request)
        self._created = True
        return _native_call_diagnostics("CreateSimulation", request, response)

    def advance_and_read_body_states(
        self,
        start_step_count: int,
        steps: int,
        timestep_ns: int,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        start_step_count, steps, timestep_ns, step_count, simulation_time_ns = _validate_advance_inputs(
            start_step_count,
            steps,
            timestep_ns,
        )
        request = {
            "simulation_id": self.simulation_id,
            "start_step_count": start_step_count,
            "steps": steps,
            "timestep_ns": timestep_ns,
            "prim_paths": list(self.config.body_prims),
        }
        started_ns = time.perf_counter_ns()
        bindings = self._require_bindings()
        step_results: list[dict[str, Any]] = []
        for index in range(steps):
            target_step_count = start_step_count + index + 1
            target_time_ns = target_step_count * timestep_ns
            handle = bindings.build_write_world_state_step(
                {
                    "simulation_id": self.simulation_id,
                    "simulation_time_ns": target_time_ns,
                }
            )
            step_results.append(dict(self._call_native_rpc("WriteWorldState", bindings.write_world_state, handle)))
        response = self._read_body_states_response(bindings, simulation_time_ns)
        step_timings = [native_client_support.coerce_mapping_float(result, "write_world_state_ms", 0.0) for result in step_results]
        response.update(
            {
                "start_step_count": int(start_step_count),
                "step_count": step_count,
                "simulation_time_ns": simulation_time_ns,
                "step_ms": sum(step_timings),
                "total_ms": sum(step_timings) + native_client_support.coerce_mapping_float(response, "read_ms", 0.0),
                "step_timings_ms": step_timings,
                "write_world_state": step_results,
            }
        )
        python_call_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        step_count = native_client_support.coerce_mapping_int(response, "step_count", step_count)
        simulation_time_ns = native_client_support.coerce_mapping_int(response, "simulation_time_ns", step_count * timestep_ns)
        states = _body_states_from_native_response(response, self.config.body_prims, simulation_time_ns)
        diagnostics = _native_call_diagnostics(
            "advance_and_read_body_states",
            request,
            response,
            states=states,
            python_call_ms=python_call_ms,
        )
        diagnostics["step_count"] = step_count
        diagnostics["simulation_time_ns"] = simulation_time_ns
        diagnostics["body_count"] = native_client_support.coerce_mapping_int(response, "body_count", len(states))
        return states, diagnostics

    def read_body_states(self, simulation_time_ns: int) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        request = {
            "simulation_id": self.simulation_id,
            "prim_paths": list(self.config.body_prims),
            "simulation_time_ns": int(simulation_time_ns),
        }
        started_ns = time.perf_counter_ns()
        response = self._read_body_states_response(self._require_bindings(), int(simulation_time_ns))
        python_call_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        response_simulation_time_ns = native_client_support.coerce_mapping_int(response, "simulation_time_ns", int(simulation_time_ns))
        states = _body_states_from_native_response(response, self.config.body_prims, response_simulation_time_ns)
        return states, _native_call_diagnostics(
            "ReadWorldState",
            request,
            response,
            states=states,
            python_call_ms=python_call_ms,
        )

    def write_body_poses(
        self,
        poses: Sequence[BodyPose],
        *,
        simulation_time_ns: int,
        reset: bool = False,
    ) -> Mapping[str, Any]:
        request = {
            "simulation_id": self.simulation_id,
            "simulation_time_ns": int(simulation_time_ns),
            "poses": [
                {
                    "prim_path": pose.prim_path,
                    "translate": list(pose.translate),
                    "orient": list(pose.orient),
                }
                for pose in poses
            ],
            "reset": bool(reset),
        }
        started_ns = time.perf_counter_ns()
        bindings = self._require_bindings()
        handle = bindings.build_write_world_state_body_poses(request)
        response = dict(self._call_native_rpc("WriteWorldState", bindings.write_world_state, handle))
        response["reset"] = bool(reset)
        python_call_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        diagnostics = _native_call_diagnostics(
            "WriteWorldState",
            request,
            response,
            python_call_ms=python_call_ms,
        )
        for key in ("ok", "status", "failed", "grpc_status", "error", "skipped_reason"):
            if key in response:
                diagnostics[key] = response[key]
        return diagnostics

    def write_body_velocities(
        self,
        velocities: Sequence[BodyVelocity],
        *,
        simulation_time_ns: int,
        reset: bool = False,
    ) -> Mapping[str, Any]:
        request = {
            "simulation_id": self.simulation_id,
            "simulation_time_ns": int(simulation_time_ns),
            "velocities": [
                {
                    "prim_path": velocity.prim_path,
                    "linear": list(velocity.linear),
                    "angular": list(velocity.angular),
                }
                for velocity in velocities
            ],
            "reset": bool(reset),
        }
        started_ns = time.perf_counter_ns()
        bindings = self._require_bindings()
        handle = bindings.build_write_world_state_body_velocities(request)
        response = dict(self._call_native_rpc("WriteWorldState", bindings.write_world_state, handle))
        response["reset"] = bool(reset)
        diagnostics = _native_call_diagnostics(
            "WriteWorldState",
            request,
            response,
            python_call_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
        )
        for key in ("ok", "status", "failed", "grpc_status", "error", "skipped_reason"):
            if key in response:
                diagnostics[key] = response[key]
        return diagnostics

    def delete_simulation(self) -> str:
        if not self._created:
            self.last_delete_diagnostics = {"status": "not_found"}
            return "not_found"
        request = {"simulation_id": self.simulation_id}
        try:
            response = self._call_bound(
                "DeleteSimulation",
                self._require_bindings().delete_simulation,
                request,
            )
        except Exception as exc:
            diagnostics = native_client_support.exception_protocol_diagnostics(exc) or {
                "error": f"{type(exc).__name__}: {exc}",
                "request": request,
            }
            if str(diagnostics.get("grpc_status", "")) == "NOT_FOUND":
                self._created = False
                self.last_delete_diagnostics = {**dict(diagnostics), "status": "not_found"}
                return "not_found"
            self.last_delete_diagnostics = {**dict(diagnostics), "status": "failed"}
            return "failed"
        self._created = False
        self.last_delete_diagnostics = {
            **native_client_support.native_response_diagnostics(response),
            "status": "stopped",
        }
        return "stopped"

    def shutdown(self) -> str:
        native_client = self._native_client
        status = self.delete_simulation()
        if status == "failed":
            return status
        self._started = False
        if native_client is not None:
            self._close_client()
        native_module = self._native_module
        if native_module is not None:
            shutdown = getattr(native_module, "shutdown", None)
            from . import runtime_services

            if callable(shutdown) and not runtime_services.owner.owns_module(native_module):
                shutdown()
        return status

    def _import_native_client(self) -> Any:
        if self._native_module is not None:
            return self._native_module
        module_path = self.config.ovphysx_native_client_path.strip()
        if module_path and module_path not in sys.path:
            sys.path.insert(0, module_path)
        module_name = self.config.ovphysx_native_client_module.strip() or DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE
        try:
            native_module = importlib.import_module(module_name)
            self._native_module = native_module
            self._ensure_client(native_module, self.config.ovphysx_address)
        except Exception as exc:
            raise SharedStageCompositionError(
                f"Native OVPhysX client import failed for {module_name}: {exc}"
            ) from exc
        return self._native_module

    def _ensure_client(self, native_module: Any, endpoint: str) -> None:
        if self._native_client is not None and self._native_endpoint == endpoint:
            return
        if self._native_client is not None:
            self._close_client()
        self._native_client = native_module.Client(endpoint)
        self._native_endpoint = endpoint
        self._native_bindings = _bind_physics_native_client(native_module, self._native_client)

    def _close_client(self) -> None:
        native_client = self._native_client
        self._native_client = None
        self._native_endpoint = ""
        self._native_bindings = None
        if native_client is not None:
            native_client.close()

    def _require_bindings(self) -> _PhysicsNativeBindings:
        if self._native_bindings is None:
            self._import_native_client()
        if self._native_bindings is None:
            raise SharedStageCompositionError("Native OVPhysX client bindings are not initialized")
        return self._native_bindings

    def _call_bound(
        self,
        name: str,
        function: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return native_client_support.call_native_rpc(
                name,
                function,
                dict(request),
                rpc_status_error=self._rpc_status_error_type(),
                client_label=OVPHYSX_NATIVE_CLIENT_LABEL,
                error_type=SharedStageCompositionError,
            )
        except Exception as exc:
            if isinstance(exc, SharedStageCompositionError):
                raise
            raise SharedStageCompositionError(f"Native OVPhysX client {name} failed: {exc}") from exc

    def _call_native_rpc(
        self,
        name: str,
        function: Callable[[Any], Any],
        argument: Any,
    ) -> Mapping[str, Any]:
        return native_client_support.call_native_rpc(
            name,
            function,
            argument,
            rpc_status_error=self._rpc_status_error_type(),
            client_label=OVPHYSX_NATIVE_CLIENT_LABEL,
            error_type=SharedStageCompositionError,
        )

    def _rpc_status_error_type(self) -> type[BaseException] | None:
        bindings = self._native_bindings
        return bindings.rpc_status_error if bindings is not None else None

    def _read_body_states_response(
        self,
        bindings: _PhysicsNativeBindings,
        simulation_time_ns: int,
    ) -> dict[str, Any]:
        request = {
            "simulation_id": self.simulation_id,
            "prim_paths": list(self.config.body_prims),
            "simulation_time_ns": int(simulation_time_ns),
        }
        read_handle = bindings.build_read_world_state_body_states(request)
        read_result = dict(self._call_native_rpc("ReadWorldState", bindings.read_world_state, read_handle))
        response_handle = read_result.get("response_handle")
        decoded = dict(bindings.decode_body_states(read_handle, response_handle))
        decoded.setdefault("simulation_time_ns", int(simulation_time_ns))
        decoded["read_world_state"] = native_client_support.native_response_diagnostics(read_result)
        decoded["read_ms"] = native_client_support.coerce_mapping_float(read_result, "read_world_state_ms", 0.0)
        return decoded


def _bind_physics_native_client(native_module: Any, native_client: Any) -> _PhysicsNativeBindings:
    def require(name: str) -> Callable[..., Any]:
        return native_client_support.require_callable(
            native_module,
            name,
            client_label=OVPHYSX_NATIVE_CLIENT_LABEL,
            error_type=SharedStageCompositionError,
        )

    capabilities_fn = require("capabilities")
    capabilities = capabilities_fn()
    if not isinstance(capabilities, Mapping):
        raise SharedStageCompositionError("Native OVPhysX client capabilities() did not return a mapping")

    rpcs = native_client_support.capability_names(capabilities, "rpcs")
    required_rpcs = {"CreateSimulation", "ListSimulations", "DeleteSimulation", "WriteWorldState", "ReadWorldState"}
    missing_rpcs = sorted(name for name in required_rpcs if name not in rpcs)
    if missing_rpcs:
        raise SharedStageCompositionError(f"Native OVPhysX client is missing RPC capabilities: {', '.join(missing_rpcs)}")

    generic_builders = native_client_support.capability_names(capabilities, "generic_builders")
    required_generic = {
        "build_WriteWorldState_step",
        "build_WriteWorldState_body_poses",
        "build_WriteWorldState_body_velocities",
        "build_ReadWorldState_body_states",
    }
    missing_generic = sorted(name for name in required_generic if name not in generic_builders)
    if missing_generic:
        raise SharedStageCompositionError(
            f"Native OVPhysX client is missing generic builder capabilities: {', '.join(missing_generic)}"
        )

    generic_step_builder = require("build_WriteWorldState_step")
    generic_body_pose_builder = require("build_WriteWorldState_body_poses")
    generic_body_velocity_builder = require("build_WriteWorldState_body_velocities")
    generic_body_state_builder = require("build_ReadWorldState_body_states")

    return _PhysicsNativeBindings(
        start_worker=native_client_support.optional_callable(native_module, "start_worker"),
        connect=native_client_support.optional_callable(native_module, "connect"),
        create_simulation=native_client_support.require_callable(
            native_client, "CreateSimulation", client_label=OVPHYSX_NATIVE_CLIENT_LABEL, error_type=SharedStageCompositionError
        ),
        delete_simulation=native_client_support.require_callable(
            native_client, "DeleteSimulation", client_label=OVPHYSX_NATIVE_CLIENT_LABEL, error_type=SharedStageCompositionError
        ),
        write_world_state=native_client_support.require_callable(
            native_client, "WriteWorldState", client_label=OVPHYSX_NATIVE_CLIENT_LABEL, error_type=SharedStageCompositionError
        ),
        read_world_state=native_client_support.require_callable(
            native_client, "ReadWorldState", client_label=OVPHYSX_NATIVE_CLIENT_LABEL, error_type=SharedStageCompositionError
        ),
        build_write_world_state_step=generic_step_builder,
        build_write_world_state_body_poses=generic_body_pose_builder,
        build_write_world_state_body_velocities=generic_body_velocity_builder,
        build_read_world_state_body_states=generic_body_state_builder,
        decode_body_states=require("decode_body_states"),
        rpc_status_error=native_client_support.rpc_status_error_type(
            native_module,
            client_label=OVPHYSX_NATIVE_CLIENT_LABEL,
            error_type=SharedStageCompositionError,
        ),
    )


def _validate_advance_inputs(
    start_step_count: Any,
    steps: Any,
    timestep_ns: Any,
) -> tuple[int, int, int, int, int]:
    start_step_count = _coerce_int("start_step_count", start_step_count)
    steps = _coerce_int("steps", steps)
    timestep_ns = _coerce_int("timestep_ns", timestep_ns)
    if start_step_count < 0:
        raise SharedStageCompositionError("start_step_count must be non-negative")
    if steps <= 0:
        raise SharedStageCompositionError("steps must be positive")
    if timestep_ns <= 0:
        raise SharedStageCompositionError("timestep_ns must be positive")
    if start_step_count > INT64_MAX or steps > INT64_MAX or timestep_ns > INT64_MAX:
        raise SharedStageCompositionError("advance inputs must fit int64")
    if start_step_count > INT64_MAX - steps:
        raise SharedStageCompositionError("start_step_count + steps overflows int64")
    step_count = start_step_count + steps
    if step_count > INT64_MAX // timestep_ns:
        raise SharedStageCompositionError("step_count * timestep_ns overflows int64")
    return start_step_count, steps, timestep_ns, step_count, step_count * timestep_ns


def _coerce_int(name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SharedStageCompositionError(f"{name} must be int") from exc


coerce_mapping_int = native_client_support.coerce_mapping_int
coerce_mapping_float = native_client_support.coerce_mapping_float


def _native_call_diagnostics(
    name: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    states: Sequence[Mapping[str, Any]] = (),
    python_call_ms: float | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "name": name,
        "transport": "native",
        "request": dict(request),
        "response": {key: value for key, value in response.items() if key != "states"},
    }
    if python_call_ms is not None:
        diagnostics["python_call_ms"] = float(python_call_ms)
    if states:
        diagnostics["body_count"] = len(states)
    for key in ("simulation_time_ns", "step_count", "step_ms", "read_ms", "total_ms", "step_timings_ms", "body_count"):
        if key in response:
            diagnostics[key] = response[key]
    return diagnostics


def _body_states_from_native_response(
    response: Mapping[str, Any],
    requested_paths: Sequence[str],
    simulation_time_ns: int,
) -> list[dict[str, Any]]:
    raw_states = response.get("states", [])
    if isinstance(raw_states, Mapping):
        items = list(raw_states.items())
    elif isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes, bytearray)):
        items = [(None, item) for item in raw_states]
    else:
        raise SharedStageCompositionError("Native OVPhysX body state response contains no states mapping or list")

    states: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        path_key, raw_state = item
        if not isinstance(raw_state, Mapping):
            raise SharedStageCompositionError("Native OVPhysX body state item is not a mapping")
        prim_path = _native_prim_path(raw_state, path_key, requested_paths, index)
        states.append(
            {
                "prim_path": prim_path,
                "simulation_time_ns": int(simulation_time_ns),
                "translate": _native_float3(raw_state, ("translate", "translation", "xformOp:translate", "position")),
                "orient": _native_quatf(raw_state, ("orient", "orientation", "xformOp:orient", "rotation")),
                "linear_velocity": _native_float3(raw_state, ("linear_velocity", "velocity", "physics:velocity")),
                "angular_velocity": _native_float3(
                    raw_state,
                    ("angular_velocity", "angularVelocity", "physics:angularVelocity"),
                ),
            }
        )
    return states


def _native_prim_path(
    state: Mapping[str, Any],
    path_key: Any,
    requested_paths: Sequence[str],
    index: int,
) -> str:
    for key in ("prim_path", "usd_path", "usd-path", "path"):
        value = state.get(key)
        if value:
            return str(value)
    if path_key:
        return str(path_key)
    if index < len(requested_paths):
        return str(requested_paths[index])
    return UNKNOWN


def _native_float3(state: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    value = _first_present(state, names)
    return _float3_value(value)


def _native_quatf(state: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    value = _first_present(state, names)
    if isinstance(value, Mapping):
        if value.get("found") is False:
            return {"found": False}
        if all(axis in value for axis in ("i", "j", "k")) and ("r" in value or "w" in value):
            return {
                "found": True,
                "i": float(value.get("i", 0.0)),
                "j": float(value.get("j", 0.0)),
                "k": float(value.get("k", 0.0)),
                "r": float(value.get("r", value.get("w", 1.0))),
            }
        if all(axis in value for axis in ("x", "y", "z")) and ("w" in value or "r" in value):
            return {
                "found": True,
                "i": float(value.get("x", 0.0)),
                "j": float(value.get("y", 0.0)),
                "k": float(value.get("z", 0.0)),
                "r": float(value.get("w", value.get("r", 1.0))),
            }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 4:
        return {
            "found": True,
            "i": float(value[0]),
            "j": float(value[1]),
            "k": float(value[2]),
            "r": float(value[3]),
        }
    return {"found": False}


def _float3_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        if value.get("found") is False:
            return {"found": False}
        if all(axis in value for axis in ("x", "y", "z")):
            return {
                "found": True,
                "x": float(value.get("x", 0.0)),
                "y": float(value.get("y", 0.0)),
                "z": float(value.get("z", 0.0)),
            }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 3:
        return {
            "found": True,
            "x": float(value[0]),
            "y": float(value[1]),
            "z": float(value[2]),
        }
    return {"found": False}


def _first_present(state: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in state:
            return state[name]
    return None


def _usd_file_uri(value: str) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser().resolve()
    return str(path) if bundled_runtime.current_platform_id() == "windows-x64" else path.as_uri()


def _ovphysx_worker_environment(config: OvphysxRuntimeConfig) -> dict[str, str]:
    env = os.environ.copy()
    library_entries: list[str] = []
    bundle = bundled_runtime.defaults(ovphysx_address=config.ovphysx_address, ovphysx_device="")
    env_ovphysx_root = _env_optional_path("OVPHYSX_ROOT")
    env_ovruntime_root = _env_optional_path("OVRUNTIME_ROOT")
    ovphysx_root = env_ovphysx_root or (Path(bundle.ovphysx_root) if bundle.ovphysx_root else None)
    ovruntime_root = env_ovruntime_root or (Path(bundle.ovruntime_root) if bundle.ovruntime_root else None)
    if ovphysx_root is not None:
        env["OVPHYSX_ROOT"] = str(ovphysx_root)
        env["OVPHYSX_LIB"] = str(ovphysx_root / ("bin" if bundled_runtime.current_platform_id() == "windows-x64" else "lib"))
        library_entries.extend([str(ovphysx_root / "lib"), str(ovphysx_root / "plugins")])
    if bundle.ovphysx_bridge_runtime_root:
        library_entries.extend(
            [
                str(Path(bundle.ovphysx_bridge_runtime_root) / "lib"),
                str(Path(bundle.ovphysx_bridge_runtime_root) / "lib64"),
            ]
        )
    if ovruntime_root is not None:
        env["OVRUNTIME_ROOT"] = str(ovruntime_root)
        library_entries.extend([str(ovruntime_root), str(ovruntime_root / "lib"), str(ovruntime_root / "lib64")])
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        library_entries.extend(value for value in existing.split(os.pathsep) if value)
    if library_entries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_entries)
    return env


def _env_optional_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


__all__ = [
    "DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE",
    "DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH",
    "OvphysxRuntimeClient",
    "OvphysxRuntimeConfig",
    "UNKNOWN",
    "coerce_mapping_float",
    "coerce_mapping_int",
]
