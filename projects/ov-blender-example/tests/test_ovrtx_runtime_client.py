# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from dataclasses import replace
import os
import sys
from types import ModuleType
from typing import Mapping
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (  # noqa: E402
    bundled_runtime,
    color_presentation,
    ovrtx_gpu_lease,
    ovrtx_runtime_client,
    ovrtx_session,
    session_lifecycle,
)
from ovrtx_blender_example.ovrtx_runtime_client import (  # noqa: E402
    ATTACH_CLEANUP_SCOPE_DEAD_PID,
    ATTACH_CLEANUP_SCOPE_FULL,
    OvrtxRuntimeClient,
    RenderClientError,
    render_result_from_native,
)
from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxSessionUpdatePort,
    OvrtxTransformValue,
)
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


class _FakeControlPlane:
    def __init__(self) -> None:
        self.simulations: list[str] = []
        self.delete_failures: dict[str, tuple[str, str]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.bind_count = 0
        self.closed_count = 0
        self.events: list[str] | None = None
        self.endpoints: list[str] = []

    def bindings(self) -> ovrtx_runtime_client._ControlPlaneBindings:
        self.bind_count += 1
        return ovrtx_runtime_client._ControlPlaneBindings(
            list_simulations=self.list_simulations,
            delete_simulation=self.delete_simulation,
            close=self.close,
        )

    def list_simulations(self, request: Mapping[str, object]) -> Mapping[str, object]:
        request_record = dict(request)
        self.calls.append(("ListSimulations", request_record))
        limit = int(request.get("limit", 100))
        offset = int(request.get("offset", 0))
        return {
            "protocol_method": "ControlPlaneService.ListSimulations",
            "request": request_record,
            "grpc_status": "OK",
            "code": "OK",
            "simulations": self.simulations[offset : offset + limit],
            "total": len(self.simulations),
        }

    def delete_simulation(self, request: Mapping[str, object]) -> Mapping[str, object]:
        request_record = dict(request)
        self.calls.append(("DeleteSimulation", request_record))
        simulation_id = str(request.get("simulation_id", ""))
        if self.events is not None:
            self.events.append(f"delete:{simulation_id}")
        failure = self.delete_failures.get(simulation_id)
        if failure is not None:
            status, details = failure
            error = RenderClientError(
                f"OVRTX cleanup ControlPlaneService.DeleteSimulation failed with gRPC status {status}"
            )
            error.protocol_diagnostics = {  # type: ignore[attr-defined]
                "protocol_method": "ControlPlaneService.DeleteSimulation",
                "request": request_record,
                "grpc_status": status,
                "code": status,
                "details": details,
            }
            raise error
        return {
            "protocol_method": "ControlPlaneService.DeleteSimulation",
            "request": request_record,
            "grpc_status": "OK",
            "code": "OK",
            "simulation_id": simulation_id,
            "deleted": True,
        }

    def close(self) -> None:
        self.closed_count += 1
        if self.events is not None:
            self.events.append("close")


@pytest.fixture(autouse=True)
def isolated_gpu_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ovrtx_gpu_lease.LOCK_DIR_ENV, str(tmp_path / "gpu-locks"))
    monkeypatch.setenv(ovrtx_gpu_lease.LEASE_ID_ENV, "test-gpu")


@pytest.fixture(autouse=True)
def fake_control_plane(monkeypatch) -> _FakeControlPlane:
    control = _FakeControlPlane()

    def _bind(start_result: Mapping[str, object], worker_command: str):
        control.endpoints.append(
            ovrtx_runtime_client._control_plane_endpoint(start_result, worker_command)
        )
        return control.bindings()

    monkeypatch.setattr(
        "ovrtx_blender_example.ovrtx_runtime_client._bind_official_control_plane",
        _bind,
    )
    return control


class _FakeNativeRenderModule(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.shutdown_called = False
        self.events: list[str] | None = None
        self.RpcStatusError = type(f"{name}_RpcStatusError", (RuntimeError,), {})
        self.client_endpoints: list[str] = []

    def Client(self, endpoint: str) -> object:
        self.client_endpoints.append(endpoint)
        module = self

        class Client:
            def close(self) -> None:
                module.calls.append(("close_client", {}))

            def __getattr__(self, name: str) -> object:
                return getattr(module, name)

        return Client()

    def rpc_status_error(self, diagnostics: dict[str, object]) -> BaseException:
        error = self.RpcStatusError(
            f"{diagnostics.get('protocol_method', 'RPC')} failed with {diagnostics.get('grpc_status', 'UNKNOWN')}"
        )
        error.protocol_diagnostics = dict(diagnostics)  # type: ignore[attr-defined]
        return error

    def capabilities(self) -> dict[str, object]:
        return {
            "rpcs": ["CreateSimulation", "ListSimulations", "DeleteSimulation", "WriteWorldState", "ReadWorldState"],
            "semantic_builders": [
                "build_render_sample_step",
            ],
            "generic_builders": [
                "build_WriteWorldState_columns",
                "build_ReadWorldState_ldr_color",
                "build_ReadWorldState_hdr_color",
            ],
            "request_handle_types": ["fake.WriteWorldStateRequest", "fake.ReadWorldStateRequest"],
            "response_helpers": ["decode_ldr_color_frame", "decode_hdr_color_frame"],
        }

    def check_health(self) -> dict[str, object]:
        return {
            "serving": True,
            "endpoint": "127.0.0.1:50051",
            "worker_process_alive": True,
        }

    def start_worker(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("start_worker", dict(request)))
        return {"serving": True, "endpoint": "127.0.0.1:50051", "worker_process_alive": True}

    def ListSimulations(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("ListSimulations", dict(request)))
        return {
            "simulations": [],
            "total": 0,
            "protocol_method": "ControlPlaneService.ListSimulations",
            "grpc_status": "OK",
        }

    def DeleteSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("DeleteSimulation", dict(request)))
        return {
            "simulation_id": request["simulation_id"],
            "deleted": True,
            "protocol_method": "ControlPlaneService.DeleteSimulation",
            "grpc_status": "OK",
        }

    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("CreateSimulation", dict(request)))
        return {
            "simulation_id": request["simulation_id"],
            "sensor_paths": [str(sensor["sensor_path"]) for sensor in request["sensors"]],  # type: ignore[index]
            "width": request["width"],
            "height": request["height"],
            "protocol_method": "ControlPlaneService.CreateSimulation",
            "grpc_status": "OK",
        }

    def build_render_sample_step(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_render_sample_step", dict(request)))
        return {"builder": "build_render_sample_step", "request": dict(request)}

    def build_WriteWorldState_columns(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_WriteWorldState_columns", dict(request)))
        return {"builder": "build_WriteWorldState_columns", "request": dict(request)}

    def build_ReadWorldState_ldr_color(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_ReadWorldState_ldr_color", dict(request)))
        return {"builder": "build_ReadWorldState_ldr_color", "request": dict(request)}

    def build_ReadWorldState_hdr_color(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_ReadWorldState_hdr_color", dict(request)))
        return {"builder": "build_ReadWorldState_hdr_color", "request": dict(request)}

    def WriteWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("WriteWorldState", dict(handle)))
        request = handle["request"]
        assert isinstance(request, dict)
        return {
            "simulation_id": request["simulation_id"],
            "simulation_time_ns": request["simulation_time_ns"],
            "written_operation_count": 1,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.WriteWorldState",
            "grpc_status": "OK",
            "native_timings": {"write_world_state_ms": 0.5} if handle["builder"] == "build_WriteWorldState_columns" else {},
        }

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("ReadWorldState", dict(handle)))
        return {
            "response_handle": {"source": "fake-read-response"},
            "result_column_count": 1,
            "has_iterator": False,
            "read_world_state_ms": 1.25,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_ldr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("decode_ldr_color_frame", {"read": dict(read_handle), "response": dict(response_handle)}))
        request = read_handle["request"]
        assert isinstance(request, dict)
        render_var_paths = tuple(request["render_var_paths"])
        render_var_path = render_var_paths[0]
        return {
            "frames": {
                render_var_path: {
                    "sensor_path": render_var_path,
                    "width": request["width"],
                    "height": request["height"],
                    "rgba8": bytes([255, 0, 0, 255, 0, 0, 255, 255]),
                    "completed_samples": request["completed_samples"],
                    "session_completed_samples": request["session_completed_samples"],
                    "simulation_time_ns": request["simulation_time_ns"],
                    "native_timings": {"render_ms": 1.25},
                }
            },
            "frame_count": 1,
            "render_var_paths": list(render_var_paths),
            "statuses": {render_var_path: "OK"},
            "simulation_id": request["simulation_id"],
            "simulation_time_ns": request["simulation_time_ns"],
        }

    def decode_hdr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("decode_hdr_color_frame", {"read": dict(read_handle), "response": dict(response_handle)}))
        request = read_handle["request"]
        assert isinstance(request, dict)
        render_var_paths = tuple(request["render_var_paths"])
        render_var_path = render_var_paths[0]
        return {
            "frames": {
                render_var_path: {
                    "sensor_path": render_var_path,
                    "width": request["width"],
                    "height": request["height"],
                    "rgba8": bytes([255, 255, 255, 255, 0, 0, 0, 255]),
                    "linear_rgba16f": bytes([0, 60, 0, 60, 0, 60, 0, 60, 0, 0, 0, 0, 0, 0, 0, 60]),
                    "frame_format": color_presentation.FRAME_FORMAT_RGBA16F,
                    "frame_color_mode": color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
                    "render_var": color_presentation.RENDER_VAR_HDR_COLOR,
                    "completed_samples": request["completed_samples"],
                    "session_completed_samples": request["session_completed_samples"],
                    "simulation_time_ns": request["simulation_time_ns"],
                    "native_timings": {"render_ms": 1.25},
                }
            },
            "frame_count": 1,
            "render_var_paths": list(render_var_paths),
            "statuses": {render_var_path: "OK"},
            "simulation_id": request["simulation_id"],
            "simulation_time_ns": request["simulation_time_ns"],
        }

    def shutdown(self) -> None:
        self.calls.append(("shutdown", {}))
        self.shutdown_called = True
        if self.events is not None:
            self.events.append("shutdown")


class _FakeAsyncReadTicket:
    def __init__(
        self,
        module: "_AsyncNativeRenderModule",
        handle: dict[str, object],
        source: str,
    ) -> None:
        self._module = module
        self._handle = handle
        self._source = source
        self._pending = True
        self._consumed = False
        self._cancel_requested = False
        self._cancel_pending_polls = int(module.cancel_pending_polls)

    def poll(self) -> Mapping[str, object] | None:
        assert not self._consumed
        request = self._handle["request"]
        assert isinstance(request, dict)
        self._module.calls.append(
            (
                "poll_ReadWorldState",
                {"request": dict(request), "source": self._source},
            )
        )
        if self._cancel_requested:
            if self._cancel_pending_polls > 0:
                self._cancel_pending_polls -= 1
                if self._module.events is not None:
                    self._module.events.append("cancel_pending")
                return None
            self._consumed = True
            if self._module.events is not None:
                self._module.events.append("cancel_terminal")
            raise self._module.rpc_status_error(
                {
                    "ok": False,
                    "protocol_method": "WorldStateService.ReadWorldState",
                    "grpc_status": "CANCELLED",
                    "grpc_status_code": 1,
                    "grpc_message": "cancelled for test",
                    "elapsed_ms": 0.25,
                    "request": {
                        "builder_name": self._handle["builder"],
                    },
                }
            )
        if self._pending:
            self._pending = False
            return None
        self._consumed = True
        return self._module.async_read_result(self._handle, self._source)

    def cancel(self) -> bool:
        assert not self._consumed
        if self._cancel_requested:
            return False
        self._cancel_requested = True
        request = self._handle["request"]
        assert isinstance(request, dict)
        self._module.calls.append(
            ("cancel_ReadWorldState", {"request": dict(request)})
        )
        if self._module.events is not None:
            self._module.events.append("cancel")
        return True


class _AsyncNativeRenderModule(_FakeNativeRenderModule):
    def __init__(
        self,
        name: str,
        *,
        sources: list[str] | None = None,
        cancel_pending_polls: int = 0,
    ) -> None:
        super().__init__(name)
        self.async_sources = list(sources or ["frame"])
        self.cancel_pending_polls = int(cancel_pending_polls)

    def capabilities(self) -> dict[str, object]:
        capabilities = dict(super().capabilities())
        capabilities["async_rpcs"] = ["ReadWorldState"]
        return capabilities

    def begin_ReadWorldState(
        self,
        handle: dict[str, object],
    ) -> _FakeAsyncReadTicket:
        source = self.async_sources.pop(0) if self.async_sources else "frame"
        self.calls.append(("begin_ReadWorldState", dict(handle)))
        return _FakeAsyncReadTicket(self, handle, source)

    def async_read_result(
        self,
        handle: dict[str, object],
        source: str,
    ) -> Mapping[str, object]:
        if source == "deadline":
            raise self.rpc_status_error(
                {
                    "ok": False,
                    "protocol_method": "WorldStateService.ReadWorldState",
                    "grpc_status": "DEADLINE_EXCEEDED",
                    "grpc_status_code": 4,
                    "grpc_message": "read deadline",
                    "elapsed_ms": float(
                        ovrtx_runtime_client.RENDER_READ_POLL_TIMEOUT_MS
                    ),
                    "request": {"builder_name": handle["builder"]},
                }
            )
        has_iterator = source == "iterator"
        return {
            "response_handle": {"source": source},
            "result_column_count": 0 if has_iterator else 1,
            "has_iterator": has_iterator,
            **({"iterator": "next-page"} if has_iterator else {}),
            "read_world_state_ms": 0.75 if has_iterator else 1.25,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_ldr_color_frame(
        self,
        read_handle: dict[str, object],
        response_handle: dict[str, object],
    ) -> dict[str, object]:
        if response_handle.get("source") == "iterator":
            request = read_handle["request"]
            assert isinstance(request, dict)
            return {
                "frames": {},
                "frame_count": 0,
                "render_var_paths": list(request["render_var_paths"]),
                "statuses": {},
                "simulation_id": request["simulation_id"],
                "simulation_time_ns": request["simulation_time_ns"],
            }
        decoded = super().decode_ldr_color_frame(read_handle, response_handle)
        if response_handle.get("source") == "mismatch":
            request = read_handle["request"]
            assert isinstance(request, dict)
            frame = next(iter(decoded["frames"].values()))  # type: ignore[union-attr]
            assert isinstance(frame, dict)
            frame["render_output_simulation_time_ns"] = (
                int(request["simulation_time_ns"]) + 10
            )
        return decoded


class _AdvertisedAsyncWithoutBeginNativeRenderModule(_FakeNativeRenderModule):
    def capabilities(self) -> dict[str, object]:
        capabilities = dict(super().capabilities())
        capabilities["async_rpcs"] = ["ReadWorldState"]
        return capabilities


class _AttributeValueBuilderNativeRenderModule(_FakeNativeRenderModule):
    def capabilities(self) -> dict[str, object]:
        capabilities = dict(super().capabilities())
        semantic_builders = list(capabilities.get("semantic_builders", ()))
        semantic_builders.append("build_attribute_values_update")
        capabilities["semantic_builders"] = semantic_builders
        return capabilities

    def build_attribute_values_update(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_attribute_values_update", dict(request)))
        return {"builder": "build_attribute_values_update", "request": dict(request)}


class _RejectingAttributeValueBuilderNativeRenderModule(_AttributeValueBuilderNativeRenderModule):
    def build_attribute_values_update(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_attribute_values_update", dict(request)))
        raise ValueError("Float2Array value has an invalid shape")


class _FailingNativeRenderModule(_FakeNativeRenderModule):
    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("CreateSimulation", dict(request)))
        raise RuntimeError("boom")


class _CreateSimulationStatusFailsNativeRenderModule(_FakeNativeRenderModule):
    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("CreateSimulation", dict(request)))
        raise self.rpc_status_error(
            {
                "ok": False,
                "protocol_method": "ControlPlaneService.CreateSimulation",
                "grpc_status": "UNAVAILABLE",
                "grpc_status_code": 14,
                "grpc_message": "worker unavailable",
                "elapsed_ms": 3.5,
                "request": {"simulation_id": request["simulation_id"]},
            }
        )


class _CreateSimulationPermanentPreconditionNativeRenderModule(_FakeNativeRenderModule):
    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("CreateSimulation", dict(request)))
        raise self.rpc_status_error(
            {
                "ok": False,
                "protocol_method": "ControlPlaneService.CreateSimulation",
                "grpc_status": "FAILED_PRECONDITION",
                "grpc_status_code": 9,
                "grpc_message": "simulation conflict",
                "elapsed_ms": 1.0,
                "request": {"simulation_id": request["simulation_id"]},
            }
        )


class _ReadWorldStateStatusFailsNativeRenderModule(_FakeNativeRenderModule):
    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        request = handle["request"]
        assert isinstance(request, dict)
        if request.get("timeout_ms") == 0:
            return super().ReadWorldState(handle)
        self.calls.append(("ReadWorldState", dict(handle)))
        raise self.rpc_status_error(
            {
                "ok": False,
                "protocol_method": "WorldStateService.ReadWorldState",
                "grpc_status": "DEADLINE_EXCEEDED",
                "grpc_status_code": 4,
                "grpc_message": "read deadline",
                "elapsed_ms": 1000.0,
                "request": {"builder_name": handle["builder"]},
            }
        )


class _ReadWorldStateDeadlineThenReadyNativeRenderModule(_ReadWorldStateStatusFailsNativeRenderModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.deadline_count = 0

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        request = handle["request"]
        assert isinstance(request, dict)
        if request.get("timeout_ms") != 0 and self.deadline_count == 0:
            self.deadline_count += 1
            return super().ReadWorldState(handle)
        return _FakeNativeRenderModule.ReadWorldState(self, handle)


class _EmptyThenReadyNativeRenderModule(_FakeNativeRenderModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.read_count = 0

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        request = handle["request"]
        assert isinstance(request, dict)
        if request.get("timeout_ms") == 0:
            return super().ReadWorldState(handle)
        self.read_count += 1
        if self.read_count > 1:
            return super().ReadWorldState(handle)
        self.calls.append(("ReadWorldState", dict(handle)))
        return {
            "response_handle": {"source": "empty-read-response"},
            "result_column_count": 0,
            "has_iterator": False,
            "read_world_state_ms": 0.75,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_ldr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object] | None:
        if response_handle.get("source") == "empty-read-response":
            self.calls.append(("decode_ldr_color_frame", {"read": dict(read_handle), "response": dict(response_handle)}))
            request = read_handle["request"]
            assert isinstance(request, dict)
            return {
                "frames": {},
                "frame_count": 0,
                "render_var_paths": list(request["render_var_paths"]),
                "statuses": {},
                "simulation_id": request["simulation_id"],
                "simulation_time_ns": request["simulation_time_ns"],
            }
        return super().decode_ldr_color_frame(read_handle, response_handle)


class _TerminalStatusNativeRenderModule(_EmptyThenReadyNativeRenderModule):
    def decode_ldr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object] | None:
        decoded = super().decode_ldr_color_frame(read_handle, response_handle)
        if response_handle.get("source") != "empty-read-response":
            return decoded
        assert isinstance(decoded, dict)
        render_var_path = str(decoded["render_var_paths"][0])
        decoded["statuses"] = {render_var_path: "INTERNAL"}
        return decoded


class _IteratorThenReadyNativeRenderModule(_FakeNativeRenderModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.read_count = 0

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        request = handle["request"]
        assert isinstance(request, dict)
        if request.get("timeout_ms") == 0:
            return super().ReadWorldState(handle)
        self.read_count += 1
        if self.read_count > 1:
            return super().ReadWorldState(handle)
        self.calls.append(("ReadWorldState", dict(handle)))
        return {
            "response_handle": {"source": "iterator-read-response"},
            "result_column_count": 0,
            "has_iterator": True,
            "iterator": "next-page",
            "read_world_state_ms": 0.75,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_ldr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object] | None:
        if response_handle.get("source") == "iterator-read-response":
            request = read_handle["request"]
            assert isinstance(request, dict)
            return {
                "frames": {},
                "frame_count": 0,
                "render_var_paths": list(request["render_var_paths"]),
                "statuses": {},
                "simulation_id": request["simulation_id"],
                "simulation_time_ns": request["simulation_time_ns"],
            }
        return super().decode_ldr_color_frame(read_handle, response_handle)


class _FrameThenTerminalStatusNativeRenderModule(_FakeNativeRenderModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.read_count = 0

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        request = handle["request"]
        assert isinstance(request, dict)
        if request.get("timeout_ms") == 0:
            return super().ReadWorldState(handle)
        self.read_count += 1
        self.calls.append(("ReadWorldState", dict(handle)))
        return {
            "response_handle": {"source": "frame-page" if self.read_count == 1 else "status-page"},
            "result_column_count": 1,
            "has_iterator": self.read_count == 1,
            **({"iterator": "status-page"} if self.read_count == 1 else {}),
            "read_world_state_ms": 1.0,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_ldr_color_frame(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object] | None:
        if response_handle.get("source") == "frame-page":
            return super().decode_ldr_color_frame(read_handle, response_handle)
        request = read_handle["request"]
        assert isinstance(request, dict)
        render_var_path = str(request["render_var_paths"][0])
        return {
            "frames": {},
            "frame_count": 0,
            "render_var_paths": [render_var_path],
            "statuses": {render_var_path: "INTERNAL"},
            "simulation_id": request["simulation_id"],
            "simulation_time_ns": request["simulation_time_ns"],
        }


class _ResolvedCreationNativeRenderModule(_FakeNativeRenderModule):
    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        result = super().CreateSimulation(request)
        result.update(
            {
                "sensor_paths": ["/Render/Resolved"],
                "width": 2,
                "height": 1,
            }
        )
        return result


class _OmittedCreationStateNativeRenderModule(_FakeNativeRenderModule):
    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        result = super().CreateSimulation(request)
        result.pop("sensor_paths", None)
        result.pop("width", None)
        result.pop("height", None)
        return result


def _request(
    tmp_path: Path,
    module_name: str,
    *,
    color_mode: str = color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
) -> RenderRequest:
    fixture = tmp_path / "scene.usda"
    fixture.write_text("#usda 1.0\n", encoding="utf-8")
    return RenderRequest(
        input_usd_path=str(fixture),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=1,
        height=2,
        max_samples=8,
        worker_command="worker",
        native_client_module=module_name,
        color_presentation=color_presentation.presentation_from_scene(None, requested_mode=color_mode),
    )


def _spec(request: RenderRequest) -> ovrtx_session.OvrtxSessionSpec:
    return ovrtx_session.build_spec(request)


def _client(module_name: str, *, worker_command: str = "worker") -> OvrtxRuntimeClient:
    return OvrtxRuntimeClient(
        worker_command=worker_command,
        native_client_module=module_name,
    )


def _start(client: OvrtxRuntimeClient, request: RenderRequest, *, simulation_id: str = "sim") -> str:
    return client.start_session(_spec(request), simulation_id=simulation_id)


def _render(
    client: OvrtxRuntimeClient,
    simulation_id: str,
    request: RenderRequest,
    *,
    additional_samples: int,
):
    return client.render_result(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=str(
            request.color_presentation.get(
                "render_var",
                color_presentation.RENDER_VAR_LDR_COLOR,
            )
        ),
        additional_samples=additional_samples,
    )


def _last_call(native_module: _FakeNativeRenderModule, name: str) -> dict[str, object]:
    for call_name, payload in reversed(native_module.calls):
        if call_name == name:
            return payload
    raise AssertionError(f"missing native call {name}")


def _path_from_file_uri(value: object) -> Path:
    parsed = urlparse(str(value))
    assert parsed.scheme == "file"
    return Path(url2pathname(parsed.path))


def test_render_request_owns_render_product_selection_rule() -> None:
    request = RenderRequest(
        sensor_paths=("/Render/Beauty", "/Render/Mask"),
        selected_sensor_paths=("/Render/Mask",),
    )
    fallback_request = RenderRequest(
        sensor_paths=("/Render/Beauty",),
        selected_sensor_paths=(),
    )

    assert request.render_product_path == "/Render/Mask"
    assert fallback_request.render_product_path == "/Render/Beauty"


def test_ovrtx_runtime_client_raw_native_render_result_and_updates(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_runtime"
    native_module = _FakeNativeRenderModule(module_name)
    lifecycle_events: list[str] = []
    native_module.events = lifecycle_events
    fake_control_plane.events = lifecycle_events
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setenv(session_lifecycle.WORKER_LOG_ENV, str(tmp_path / "worker.log"))
    monkeypatch.setenv(session_lifecycle.RENDERER_LOG_ENV, str(tmp_path / "renderer.log"))
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    startup_diagnostics = dict(client.startup_diagnostics)
    render_result = _render(client, simulation_id, request, additional_samples=2)
    camera_update = OvrtxSessionUpdatePort(client, simulation_id).update_transforms(
        [OvrtxTransformValue("/World/Camera", ((1.0, 0.0, 0.0, 0.0),))]
    )
    transform_update = client.update_transforms(
        simulation_id,
        [OvrtxTransformValue("/World/Cube", [[1.0, 0.0, 0.0, 0.0]])],
    )
    attribute_update = client.update_attribute_values(
        simulation_id,
        [OvrtxAttributeValue("/World/KeyLight", "inputs:intensity", 900.0, "Float")],
    )
    client.shutdown()

    assert native_module.calls[0][0] == "start_worker"
    # The OVRTX (ovsensors) client reports health through the module-level
    # check_health() gate, not a Client.health() RPC, so the next recorded native
    # call after the worker launch is CreateSimulation.
    assert native_module.calls[1][0] == "CreateSimulation"
    assert not any(name in {"ListSimulations", "DeleteSimulation"} for name, _payload in native_module.calls)
    assert fake_control_plane.bind_count == 1
    assert fake_control_plane.calls[0][0] == "ListSimulations"
    create_call = _last_call(native_module, "CreateSimulation")
    assert create_call["simulation_id"] == "sim"
    assert create_call["usd_file_uri"] == Path(request.input_usd_path).as_uri()
    assert create_call["sensors"] == [{"sensor_path": "/Render/Product"}]
    read_request = _last_call(native_module, "build_ReadWorldState_ldr_color")
    assert read_request["render_var_paths"] == ["/Render/Product/LdrColor"]
    assert simulation_id == "sim"
    assert startup_diagnostics["render_worker"]["status"] == "running"
    assert startup_diagnostics["render_worker"]["health"]["serving"] is True
    assert startup_diagnostics["render_worker"]["health"]["endpoint"] == "127.0.0.1:50051"
    assert startup_diagnostics["render_worker"]["cleanup"]["list"][0]["protocol_method"] == "ControlPlaneService.ListSimulations"
    assert startup_diagnostics["render_worker"]["cleanup"]["deleted_count"] == 0
    assert startup_diagnostics["render_worker"]["logs"]["worker_log"] == str(tmp_path / "worker.log")
    assert startup_diagnostics["render_worker"]["logs"]["renderer_log"] == str(tmp_path / "renderer.log")
    assert startup_diagnostics["render_worker"]["ovrtx_gpu_lease"]["status"] == "held"
    assert startup_diagnostics["render_worker"]["ovrtx_gpu_lease"]["gpu_id"] == "test-gpu"
    assert client.startup_diagnostics == {"render_worker": {"status": "not_started"}}
    assert render_result.width == 1
    assert render_result.height == 2
    assert render_result.rgba8 == bytes([0, 0, 255, 255, 255, 0, 0, 255])
    assert render_result.completed_samples == 2
    assert render_result.session_completed_samples == 2
    assert render_result.simulation_time_ns == 20
    assert render_result.frame_format == color_presentation.FRAME_FORMAT_RGBA8
    assert render_result.frame_color_mode == color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    assert render_result.native_timings["render_ms"] == 1.25
    assert render_result.native_timings["read_world_state"][0]["protocol_method"] == "WorldStateService.ReadWorldState"
    assert render_result.native_timings["write_world_state"][0]["protocol_method"] == "WorldStateService.WriteWorldState"
    assert "result_convert_ms" in client.last_render_timings
    assert client.last_render_timings["native_timings"]["render_ms"] == 1.25
    assert camera_update.diagnostics["builder_name"] == "build_WriteWorldState_columns"
    assert camera_update.updated_count == 1
    assert client.last_value_update_timings["native_timings"]["write_world_state_ms"] == 0.5
    assert (
        client.last_value_update_timings["native_timings"]["write_world_state"]["protocol_method"]
        == "WorldStateService.WriteWorldState"
    )
    assert client.last_value_update_timings["native_timings"]["write_world_state"][
        "request_build_ms"
    ] >= 0.0
    assert transform_update.updated_count == 1
    assert attribute_update.updated_count == 1
    assert _last_call(native_module, "WriteWorldState")["builder"] == "build_WriteWorldState_columns"
    assert native_module.calls[-1] == ("shutdown", {})
    assert native_module.shutdown_called is True
    assert native_module.client_endpoints == ["127.0.0.1:50051"]
    assert ("close_client", {}) in native_module.calls
    assert lifecycle_events == ["delete:sim", "close", "shutdown"]


def test_ovrtx_runtime_client_split_render_submits_ahead_and_reads_exact_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_split_render"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    native_module.calls.clear()

    first = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    second = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    first_result = client.complete_render_sample(first)
    second_result = client.complete_render_sample(second)

    timeline = [
        (name, payload["request"]["simulation_time_ns"])
        for name, payload in native_module.calls
        if (
            name == "WriteWorldState"
            and payload.get("builder") == "build_render_sample_step"
        )
        or (
            name == "ReadWorldState"
            and isinstance(payload.get("request"), dict)
            and int(payload["request"].get("timeout_ms", 0)) > 0
        )
    ]
    assert timeline == [
        ("WriteWorldState", 10),
        ("WriteWorldState", 20),
        ("ReadWorldState", 10),
        ("ReadWorldState", 20),
    ]
    assert (first.completed_samples, second.completed_samples) == (1, 2)
    assert (first_result.simulation_time_ns, second_result.simulation_time_ns) == (10, 20)
    assert (first_result.completed_samples, second_result.completed_samples) == (1, 2)


def test_ovrtx_runtime_client_split_render_discard_is_local_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_split_discard"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    native_module.calls.clear()

    discarded = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    client.discard_render_sample(discarded)
    retained = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    result = client.complete_render_sample(retained)

    positive_read_times = [
        int(payload["request"]["simulation_time_ns"])
        for name, payload in native_module.calls
        if name == "ReadWorldState"
        and isinstance(payload.get("request"), dict)
        and int(payload["request"].get("timeout_ms", 0)) > 0
    ]
    assert positive_read_times == [20]
    assert result.simulation_time_ns == 20
    with pytest.raises(RenderClientError, match="already completed or discarded"):
        client.complete_render_sample(retained)
    with pytest.raises(RenderClientError, match="already completed or discarded"):
        client.discard_render_sample(discarded)


def test_ovrtx_runtime_client_split_render_rejects_foreign_and_stale_tickets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_split_lifetime"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    submission = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )

    with pytest.raises(RenderClientError, match="another client"):
        _client(module_name).complete_render_sample(submission)

    assert client.delete_simulation(simulation_id) == "stopped"
    _start(client, request, simulation_id=simulation_id)
    with pytest.raises(RenderClientError, match="inactive session"):
        client.complete_render_sample(submission)


def test_async_render_read_preserves_exact_time_iterator_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_async_iterator"
    native_module = _AsyncNativeRenderModule(
        module_name,
        sources=["iterator", "frame"],
    )
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    native_module.calls.clear()

    submission = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    assert client.supports_async_render_read() is True
    ticket = client.begin_render_sample_read(submission)

    assert client.poll_render_sample_read(ticket) is None
    # Terminal iterator page starts a second native async RPC without blocking.
    assert client.poll_render_sample_read(ticket) is None
    assert client.poll_render_sample_read(ticket) is None
    result = client.poll_render_sample_read(ticket)

    assert result is not None
    assert result.simulation_time_ns == submission.simulation_time_ns == 10
    assert result.completed_samples == submission.completed_samples == 1
    assert result.native_timings["read_strategy"] == "long_poll"
    assert result.native_timings["read_transport"] == "async_ticket"
    assert result.native_timings["read_poll_count"] == 2
    assert result.native_timings["read_iterator_count"] == 1
    assert result.native_timings["read_world_state_ms"] == 2.0
    async_requests = [
        payload["request"]
        for name, payload in native_module.calls
        if name == "begin_ReadWorldState"
    ]
    assert [request["simulation_time_ns"] for request in async_requests] == [10, 10]
    assert "iterator" not in async_requests[0]
    assert async_requests[1]["iterator"] == "next-page"
    assert not any(
        name == "ReadWorldState"
        and int(payload["request"].get("timeout_ms", 0)) > 0
        for name, payload in native_module.calls
    )
    with pytest.raises(RenderClientError, match="already completed or cancelled"):
        client.poll_render_sample_read(ticket)


def test_async_read_allows_one_newer_camera_sample_without_losing_exact_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_async_camera_successor"
    native_module = _AsyncNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    native_module.calls.clear()

    first = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    first_ticket = client.begin_render_sample_read(first)
    assert client.poll_render_sample_read(first_ticket) is None

    camera_update = client.update_transforms(
        simulation_id,
        [
            OvrtxTransformValue(
                "/World/Camera",
                ((1.0, 0.0, 0.0, 5.0),),
            )
        ],
    )
    second = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )

    first_result = client.poll_render_sample_read(first_ticket)
    assert first_result is not None
    second_ticket = client.begin_render_sample_read(second)
    assert client.poll_render_sample_read(second_ticket) is None
    second_result = client.poll_render_sample_read(second_ticket)

    assert camera_update.pending_simulation_time_ns == 20
    assert first.simulation_time_ns == first_result.simulation_time_ns == 10
    assert second_result is not None
    assert second.simulation_time_ns == second_result.simulation_time_ns == 20
    timeline = [
        (
            name,
            int(payload["request"]["simulation_time_ns"]),
        )
        for name, payload in native_module.calls
        if name
        in {
            "WriteWorldState",
            "begin_ReadWorldState",
        }
        and isinstance(payload.get("request"), dict)
        and "simulation_time_ns" in payload["request"]
    ]
    assert timeline == [
        ("WriteWorldState", 10),
        ("begin_ReadWorldState", 10),
        ("WriteWorldState", 20),
        ("WriteWorldState", 20),
        ("begin_ReadWorldState", 20),
    ]


def test_async_render_read_restarts_transient_deadline_at_the_same_exact_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_async_deadline"
    native_module = _AsyncNativeRenderModule(
        module_name,
        sources=["deadline", "frame"],
    )
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    submission = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    ticket = client.begin_render_sample_read(submission)

    assert client.poll_render_sample_read(ticket) is None
    assert client.poll_render_sample_read(ticket) is None
    assert client.poll_render_sample_read(ticket) is None
    result = client.poll_render_sample_read(ticket)

    assert result is not None
    assert result.simulation_time_ns == 10
    assert result.native_timings["read_transient_status_count"] == 1
    assert (
        result.native_timings["read_transient_status_world_state_ms"]
        == ovrtx_runtime_client.RENDER_READ_POLL_TIMEOUT_MS
    )
    assert result.native_timings["read_world_state"][0]["grpc_status"] == (
        "DEADLINE_EXCEEDED"
    )
    begin_requests = [
        payload["request"]
        for name, payload in native_module.calls
        if name == "begin_ReadWorldState"
    ]
    assert [request["simulation_time_ns"] for request in begin_requests] == [10, 10]


def test_async_render_read_rejects_mismatched_render_output_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_async_time_mismatch"
    native_module = _AsyncNativeRenderModule(module_name, sources=["mismatch"])
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    submission = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    ticket = client.begin_render_sample_read(submission)

    assert client.poll_render_sample_read(ticket) is None
    with pytest.raises(RenderClientError, match="requested 10, received 20"):
        client.poll_render_sample_read(ticket)
    with pytest.raises(RenderClientError, match="already completed or cancelled"):
        client.poll_render_sample_read(ticket)


def test_async_render_read_cancel_precedes_simulation_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_async_cancel"
    native_module = _AsyncNativeRenderModule(
        module_name,
        cancel_pending_polls=2,
    )
    events: list[str] = []
    native_module.events = events
    fake_control_plane.events = events
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)
    events.clear()
    submission = client.submit_render_sample(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=color_presentation.RENDER_VAR_LDR_COLOR,
    )
    ticket = client.begin_render_sample_read(submission)

    assert client.delete_simulation(simulation_id) == "stopped"

    assert events == [
        "cancel",
        "cancel_pending",
        "cancel_pending",
        "cancel_terminal",
        "delete:sim",
    ]
    with pytest.raises(RenderClientError, match="already completed or cancelled"):
        client.cancel_render_sample_read(ticket)


def test_async_read_capability_requires_advertisement_and_begin_callable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fallback_name = "fake_ovsensors_worker_client_async_fallback"
    fallback_module = _FakeNativeRenderModule(fallback_name)
    monkeypatch.setitem(sys.modules, fallback_name, fallback_module)
    fallback_client = _client(fallback_name)
    _start(fallback_client, _request(tmp_path, fallback_name))
    assert fallback_client.supports_async_render_read() is False
    fallback_client.shutdown()

    broken_name = "fake_ovsensors_worker_client_async_broken"
    broken_module = _AdvertisedAsyncWithoutBeginNativeRenderModule(broken_name)
    monkeypatch.setitem(sys.modules, broken_name, broken_module)
    broken_client = _client(broken_name)
    with pytest.raises(RenderClientError, match="missing callable begin_ReadWorldState"):
        _start(broken_client, _request(tmp_path, broken_name))


def test_ovrtx_runtime_client_aborts_cleanup_after_render_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_RENDER_TIMEOUT_S", "1")
    module_name = "fake_ovsensors_worker_client_abort_cleanup"
    native_module = _ReadWorldStateStatusFailsNativeRenderModule(module_name)
    lifecycle_events: list[str] = []
    native_module.events = lifecycle_events
    fake_control_plane.events = lifecycle_events
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    with pytest.raises(RenderClientError, match="DEADLINE_EXCEEDED"):
        _render(client, simulation_id, request, additional_samples=1)

    assert client.delete_simulation(simulation_id) == "stopped"
    assert not any(name == "DeleteSimulation" for name, _request in fake_control_plane.calls)
    assert lifecycle_events == ["close", "shutdown"]
    assert fake_control_plane.closed_count == 1
    assert native_module.shutdown_called is True
    assert client.last_delete_diagnostics == {
        "status": "skipped",
        "reason": "render_failed",
        "simulation_ids": ["sim"],
    }
    assert client.delete_simulation(simulation_id) == "not_found"
    assert client.last_delete_diagnostics == {"status": "not_found", "simulation_id": "sim"}


def test_ovrtx_runtime_client_does_not_start_worker_when_gpu_lease_busy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_gpu_lease_busy"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    held = ovrtx_gpu_lease.acquire(
        metadata={"entrypoint": "other-session"},
        timeout_s=0,
    )
    client = _client(module_name)
    try:
        with pytest.raises(RenderClientError, match="OVRTX GPU lease is busy"):
            _start(client, request)
    finally:
        held.close()

    assert native_module.calls == []
    diagnostics = client.startup_diagnostics["render_worker"]
    assert diagnostics["status"] == "failed"
    assert diagnostics["ovrtx_gpu_lease"]["status"] == "busy"
    assert diagnostics["ovrtx_gpu_lease"]["owner"]["entrypoint"] == "other-session"


def test_ovrtx_runtime_client_releases_gpu_lease_on_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_gpu_lease_start_failure"
    native_module = _FailingNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    with pytest.raises(RenderClientError, match="failed before OVRTX session start"):
        _start(client, request)

    lease = ovrtx_gpu_lease.acquire(timeout_s=0)
    lease.close()


def test_ovrtx_runtime_client_reads_hdr_color_when_scene_linear_requested(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_hdr_color"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name, color_mode=color_presentation.MODE_SCENE_LINEAR_HDR)
    client = _client(module_name)

    simulation_id = _start(client, request)
    render_result = _render(client, simulation_id, request, additional_samples=1)

    assert _last_call(native_module, "build_ReadWorldState_hdr_color")["render_var_paths"] == ["/Render/Product/HdrColor"]
    assert _last_call(native_module, "decode_hdr_color_frame")["read"]["builder"] == "build_ReadWorldState_hdr_color"
    assert not any(name == "build_ReadWorldState_ldr_color" for name, _payload in native_module.calls)
    assert render_result.frame_format == color_presentation.FRAME_FORMAT_RGBA16F
    assert render_result.frame_color_mode == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    assert render_result.render_var == color_presentation.RENDER_VAR_HDR_COLOR
    assert render_result.linear_rgba16f == bytes([0, 0, 0, 0, 0, 0, 0, 60, 0, 60, 0, 60, 0, 60, 0, 60])
    assert render_result.rgba8 == bytes([0, 0, 0, 255, 255, 255, 255, 255])


def test_ovrtx_runtime_client_uses_generic_attribute_value_builder(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_generic_attribute_values"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    result = client.update_attribute_values(
        simulation_id,
        [OvrtxAttributeValue("/World/KeyLight", "inputs:intensity", 900.0, "Float")],
    )

    assert result.updated_count == 1
    generic_request = _last_call(native_module, "build_WriteWorldState_columns")
    assert generic_request["write"][0]["keys"] == {"attribute": "usd-path", "values": ["/World/KeyLight"]}
    assert generic_request["write"][0]["columns"][0] == {
        "attribute": "inputs:intensity",
        "type": "Float",
        "values": [900.0],
    }


def test_ovrtx_runtime_client_uses_native_attribute_value_builder_for_float2_array(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_native_attribute_values"
    native_module = _AttributeValueBuilderNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    result = client.update_attribute_values(
        simulation_id,
        [
            OvrtxAttributeValue(
                "/World/TexturedQuad",
                "primvars:st",
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                "Float2Array",
            )
        ],
    )

    builder_request = _last_call(native_module, "build_attribute_values_update")
    assert builder_request["attribute_values"] == [
        {
            "prim_path": "/World/TexturedQuad",
            "attribute": "primvars:st",
            "value_type": "Float2Array",
            "value": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        }
    ]
    assert _last_call(native_module, "WriteWorldState")["builder"] == "build_attribute_values_update"
    assert result.updated_count == 1
    assert result.pending_simulation_time_ns == 10
    assert result.diagnostics["builder_name"] == "build_attribute_values_update"
    assert "value_paths" not in result.diagnostics
    assert "value_types" not in result.diagnostics


def test_ovrtx_runtime_client_uses_one_semantic_request_for_transform_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_native_transform_values"
    native_module = _AttributeValueBuilderNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    client = _client(module_name)
    simulation_id = _start(client, _request(tmp_path, module_name))

    result = client.update_transforms(
        simulation_id,
        (
            OvrtxTransformValue("/World/A", [[1.0, 0.0, 0.0, 0.0]]),
            OvrtxTransformValue("/World/B", [[2.0, 0.0, 0.0, 0.0]]),
        ),
    )

    builder_request = _last_call(native_module, "build_attribute_values_update")
    assert builder_request["attribute_values"] == [
        {
            "prim_path": "/World/A",
            "attribute": "omni:xform",
            "value": [[1.0, 0.0, 0.0, 0.0]],
            "value_type": "Matrix4d",
        },
        {
            "prim_path": "/World/A",
            "attribute": "omni:resetXformStack",
            "value": True,
            "value_type": "Bool",
        },
        {
            "prim_path": "/World/B",
            "attribute": "omni:xform",
            "value": [[2.0, 0.0, 0.0, 0.0]],
            "value_type": "Matrix4d",
        },
        {
            "prim_path": "/World/B",
            "attribute": "omni:resetXformStack",
            "value": True,
            "value_type": "Bool",
        },
    ]
    assert result.updated_count == 2
    assert result.diagnostics["builder_name"] == "build_attribute_values_update"
    assert sum(name == "WriteWorldState" for name, _payload in native_module.calls) == 1


def test_ovrtx_runtime_client_native_value_validation_fails_before_rpc(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_rejecting_attribute_values"
    native_module = _RejectingAttributeValueBuilderNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    client = _client(module_name)
    simulation_id = _start(client, _request(tmp_path, module_name))

    with pytest.raises(RenderClientError, match="invalid shape"):
        client.update_attribute_values(
            simulation_id,
            [OvrtxAttributeValue("/World/Quad", "primvars:st", [0.0], "Float2Array")],
        )

    assert sum(name == "build_attribute_values_update" for name, _payload in native_module.calls) == 1
    assert not any(name == "WriteWorldState" for name, _payload in native_module.calls)


def test_ovrtx_runtime_client_treats_sealed_empty_read_as_terminal(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_terminal_empty"
    native_module = _EmptyThenReadyNativeRenderModule(module_name)
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    with pytest.raises(RenderClientError, match="sealed without LdrColor data"):
        _render(client, simulation_id, request, additional_samples=1)

    read_calls = [
        name
        for name, payload in native_module.calls
        if name == "ReadWorldState" and payload["request"].get("timeout_ms", 1) > 0
    ]
    assert len(read_calls) == 1
    assert sleeps == []


def test_ovrtx_runtime_client_surfaces_terminal_render_status(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_terminal_status"
    native_module = _TerminalStatusNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    with pytest.raises(RenderClientError, match="terminated with INTERNAL") as error:
        _render(client, simulation_id, request, additional_samples=1)

    assert error.value.render_status == "INTERNAL"  # type: ignore[attr-defined]
    assert error.value.render_var_path == "/Render/Product/LdrColor"  # type: ignore[attr-defined]


def test_ovrtx_runtime_client_continues_only_iterators_with_shared_deadline(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_iterator"
    native_module = _IteratorThenReadyNativeRenderModule(module_name)
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    render_result = _render(client, simulation_id, request, additional_samples=1)

    read_requests = [
        payload
        for name, payload in native_module.calls
        if name == "build_ReadWorldState_ldr_color" and payload.get("timeout_ms", 1) > 0
    ]
    assert len(read_requests) == 2
    assert "iterator" not in read_requests[0]
    assert read_requests[1]["iterator"] == "next-page"
    assert 0 < read_requests[1]["timeout_ms"] <= read_requests[0]["timeout_ms"]
    assert sleeps == []
    assert render_result.native_timings["read_strategy"] == "long_poll"
    assert render_result.native_timings["read_poll_count"] == 2
    assert render_result.native_timings["read_iterator_count"] == 1
    assert render_result.native_timings["read_sleep_ms"] == 0.0


def test_ovrtx_runtime_client_accumulates_terminal_status_across_iterator_pages(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_iterator_status"
    native_module = _FrameThenTerminalStatusNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    simulation_id = _start(client, request)
    with pytest.raises(RenderClientError, match="terminated with INTERNAL") as error:
        _render(client, simulation_id, request, additional_samples=1)

    assert error.value.render_status == "INTERNAL"  # type: ignore[attr-defined]
    assert len(error.value.read_diagnostics) == 2  # type: ignore[attr-defined]
    assert error.value.read_world_state_ms == 2.0  # type: ignore[attr-defined]


def test_ovrtx_runtime_client_reads_each_selected_render_var_with_exact_identity(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_multiple_outputs"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = replace(
        _request(tmp_path, module_name),
        sensor_paths=("/Render/Left", "/Render/Right"),
        selected_sensor_paths=("/Render/Left", "/Render/Right"),
    )
    client = _client(module_name)

    simulation_id = _start(client, request)
    render_result = _render(client, simulation_id, request, additional_samples=1)

    read_requests = [
        payload
        for name, payload in native_module.calls
        if name == "build_ReadWorldState_ldr_color" and payload.get("timeout_ms", 1) > 0
    ]
    assert [payload["render_var_paths"] for payload in read_requests] == [
        ["/Render/Left/LdrColor"],
        ["/Render/Right/LdrColor"],
    ]
    assert render_result.native_timings["read_poll_count"] == 2
    assert render_result.native_timings["read_sleep_ms"] == 0.0


def test_ovrtx_runtime_client_uses_service_resolved_creation_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_resolved_creation"
    native_module = _ResolvedCreationNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = replace(
        _request(tmp_path, module_name),
        selected_sensor_paths=("/Render/Resolved",),
    )
    spec = _spec(request)
    client = _client(module_name)

    simulation_id = client.start_session(spec, simulation_id="sim")
    result = _render(client, simulation_id, request, additional_samples=1)

    state = client._session_states[simulation_id]
    assert state.spec is spec
    assert state.sensor_paths == ("/Render/Resolved",)
    assert (state.width, state.height) == (2, 1)
    assert (result.width, result.height) == (2, 1)
    read_request = _last_call(native_module, "build_ReadWorldState_ldr_color")
    assert read_request["render_var_paths"] == ["/Render/Resolved/LdrColor"]
    assert (read_request["width"], read_request["height"]) == (2, 1)
    with pytest.raises(RenderClientError, match="declared by the OVRTX service"):
        client.render_result(
            simulation_id,
            selected_sensor_paths=("/Render/Product",),
            render_var="LdrColor",
            additional_samples=1,
        )


def test_ovrtx_runtime_client_falls_back_to_spec_when_creation_state_is_omitted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_omitted_creation"
    native_module = _OmittedCreationStateNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    spec = _spec(request)
    client = _client(module_name)

    simulation_id = client.start_session(spec, simulation_id="sim")
    state = client._session_states[simulation_id]

    assert state.spec is spec
    assert state.sensor_paths == spec.sensor_paths
    assert (state.width, state.height) == (spec.width, spec.height)


def test_ovrtx_runtime_client_preserves_failed_startup_diagnostics(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_start_failure"
    native_module = _FailingNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setenv(session_lifecycle.WORKER_LOG_ENV, str(tmp_path / "worker.log"))
    monkeypatch.setenv(session_lifecycle.RENDERER_LOG_ENV, str(tmp_path / "renderer.log"))
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    with pytest.raises(RenderClientError, match="failed before OVRTX session start"):
        _start(client, request)

    assert native_module.shutdown_called is True
    diagnostics = client.startup_diagnostics["render_worker"]
    assert diagnostics["status"] == "failed"
    assert "RuntimeError: boom" in diagnostics["error"]
    assert diagnostics["logs"]["worker_log"] == str(tmp_path / "worker.log")
    assert diagnostics["logs"]["renderer_log"] == str(tmp_path / "renderer.log")


def test_ovrtx_runtime_client_paginates_startup_cleanup(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_paged_cleanup"
    native_module = _FakeNativeRenderModule(module_name)
    fake_control_plane.simulations = ["sim-a", "sim-b", "sim-c"]
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.STARTUP_CLEANUP_PAGE_LIMIT", 2)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    _start(client, request)

    list_requests = [
        request
        for name, request in fake_control_plane.calls
        if name == "ListSimulations"
    ]
    assert list_requests == [
        {"limit": 2, "offset": 0},
        {"limit": 2, "offset": 2},
    ]
    delete_requests = [
        request["simulation_id"]
        for name, request in fake_control_plane.calls
        if name == "DeleteSimulation"
    ]
    assert delete_requests == [
        "sim-a",
        "sim-b",
        "sim-c",
    ]
    assert not any(name in {"ListSimulations", "DeleteSimulation"} for name, _payload in native_module.calls)
    cleanup = client.startup_diagnostics["render_worker"]["cleanup"]
    assert cleanup["deleted_count"] == 3
    assert len(cleanup["list"]) == 2
    assert len(cleanup["delete"]) == 3


def test_ovrtx_runtime_client_fails_startup_after_cleanup_retry_exhaustion(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_cleanup_delete_failure"
    native_module = _FakeNativeRenderModule(module_name)
    fake_control_plane.simulations = ["stale-sim"]
    fake_control_plane.delete_failures = {"stale-sim": ("UNAVAILABLE", "delete unavailable")}
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.STARTUP_CLEANUP_ATTEMPTS", 2)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    with pytest.raises(RenderClientError) as error:
        _start(client, request)

    assert isinstance(error.value.__cause__, RenderClientError)
    assert [name for name, _request in fake_control_plane.calls].count("DeleteSimulation") == 2
    assert not any(name in {"ListSimulations", "DeleteSimulation"} for name, _payload in native_module.calls)
    assert not any(name == "CreateSimulation" for name, _payload in native_module.calls)
    assert sleeps == [2]
    protocol_diagnostics = client.startup_diagnostics["render_worker"]["protocol_diagnostics"]
    assert len(protocol_diagnostics["cleanup"]["attempts"]) == 2
    assert protocol_diagnostics["last_failure"]["grpc_status"] == "UNAVAILABLE"
    assert protocol_diagnostics["last_failure"]["code"] == "UNAVAILABLE"
    assert protocol_diagnostics["last_failure"]["details"] == "delete unavailable"


def test_ovrtx_runtime_client_treats_cleanup_not_found_as_deleted_race(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_cleanup_not_found"
    native_module = _FakeNativeRenderModule(module_name)
    fake_control_plane.simulations = ["stale-sim"]
    fake_control_plane.delete_failures = {"stale-sim": ("NOT_FOUND", "already gone")}
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    _start(client, request)

    cleanup = client.startup_diagnostics["render_worker"]["cleanup"]
    assert cleanup["deleted_count"] == 1
    assert cleanup["delete"][0]["grpc_status"] == "NOT_FOUND"
    assert cleanup["delete"][0]["code"] == "NOT_FOUND"
    assert cleanup["delete"][0]["details"] == "already gone"
    assert cleanup["delete"][0]["not_found_race"] is True
    assert any(name == "CreateSimulation" for name, _payload in native_module.calls)


class _PreexistingWorkerNativeRenderModule(_FakeNativeRenderModule):
    def start_worker(self, request: dict[str, object]) -> dict[str, object]:
        result = dict(super().start_worker(request))
        result["worker_process_alive"] = False
        return result


class _ForeignWorkerDuplicateNativeRenderModule(_FakeNativeRenderModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = False

    def check_health(self) -> dict[str, object]:
        result = dict(super().check_health())
        result["worker_process_alive"] = self.started
        return result

    def start_worker(self, request: dict[str, object]) -> dict[str, object]:
        self.started = True
        return super().start_worker(request)


class _EndpointlessNativeRenderModule(_FakeNativeRenderModule):
    def start_worker(self, request: dict[str, object]) -> dict[str, object]:
        result = dict(super().start_worker(request))
        result.pop("endpoint", None)
        return result


class _SharedFakeGpuLease:
    """Stand-in lease for tests running two concurrent clients.

    The single-GPU lease is orthogonal to the attach-sweep contract these
    tests pin; a real second acquire in one process would raise
    ``OvrtxGpuLeaseBusy`` before the sweep logic runs.
    """

    def diagnostics(self) -> dict[str, object]:
        return {"status": "held", "lease_id": "pytest-shared"}

    def release(self) -> None:
        return None


def _allow_concurrent_gpu_leases(monkeypatch) -> None:
    monkeypatch.setattr(
        "ovrtx_blender_example.ovrtx_runtime_client.ovrtx_gpu_lease.acquire",
        lambda *args, **kwargs: _SharedFakeGpuLease(),
    )


def test_doomed_duplicate_launch_is_not_ownership(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_foreign_duplicate"
    native_module = _ForeignWorkerDuplicateNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr(
        ovrtx_runtime_client,
        "_endpoint_listening",
        lambda *_args, **_kwargs: True,
    )
    client = _client(module_name)

    _start(client, _request(tmp_path, module_name))

    assert client.worker_owned is False


def test_ovrtx_runtime_client_second_session_on_attached_worker_skips_sweep(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_attached_worker_skip"
    native_module = _FakeNativeRenderModule(module_name)
    sleeps: list[float] = []
    _allow_concurrent_gpu_leases(monkeypatch)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    request = _request(tmp_path, module_name)

    first_client = _client(module_name)
    _start(first_client, request, simulation_id="sim-1")
    control_plane_calls_after_attach = len(fake_control_plane.calls)

    second_client = _client(module_name)
    _start(second_client, request, simulation_id="sim-2")

    sweep_calls = [name for name, _request in fake_control_plane.calls]
    assert sweep_calls.count("ListSimulations") == 1
    assert sweep_calls.count("DeleteSimulation") == 0
    assert len(fake_control_plane.calls) == control_plane_calls_after_attach
    assert sleeps == []
    cleanup = second_client.startup_diagnostics["render_worker"]["cleanup"]
    assert cleanup == {
        "status": "skipped",
        "reason": "worker_already_attached",
        "endpoint": "127.0.0.1:50051",
        "deleted_count": 0,
    }
    # The first attach swept (full scope: this add-on launched the worker).
    first_cleanup = first_client.startup_diagnostics["render_worker"]["cleanup"]
    assert first_cleanup["status"] == "swept"
    assert first_cleanup["scope"] == ATTACH_CLEANUP_SCOPE_FULL
    assert first_cleanup["endpoint"] == "127.0.0.1:50051"


def test_ovrtx_runtime_client_scoped_attach_sweep_deletes_only_dead_pid_convention_ids(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_preexisting_scoped_sweep"
    native_module = _PreexistingWorkerNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    own_simulation = f"ovrtx-blender-viewport-{os.getpid()}"
    dead_simulation = "ovrtx-blender-viewport-4242"
    live_simulation = "ovrtx-blender-viewport-777"
    foreign_simulation = "someone-elses-simulation"
    stale_probe_simulation = "ovrtx-blender-performance-1234567890"
    fake_control_plane.simulations = [
        dead_simulation,
        live_simulation,
        own_simulation,
        foreign_simulation,
        stale_probe_simulation,
    ]
    live_pids = {777, os.getpid()}
    monkeypatch.setattr(
        "ovrtx_blender_example.session_lifecycle.pid_running",
        lambda pid: pid in live_pids,
    )
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    _start(client, request)

    delete_requests = [
        request["simulation_id"]
        for name, request in fake_control_plane.calls
        if name == "DeleteSimulation"
    ]
    assert delete_requests == [dead_simulation, stale_probe_simulation]
    cleanup = client.startup_diagnostics["render_worker"]["cleanup"]
    assert cleanup["status"] == "swept"
    assert cleanup["scope"] == ATTACH_CLEANUP_SCOPE_DEAD_PID
    assert cleanup["kept"] == [live_simulation, own_simulation, foreign_simulation]
    assert cleanup["deleted_count"] == 2


def test_ovrtx_runtime_client_failed_attach_sweep_retries_on_next_session_start(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_attach_sweep_retry"
    native_module = _FakeNativeRenderModule(module_name)
    fake_control_plane.simulations = ["stale-sim"]
    fake_control_plane.delete_failures = {"stale-sim": ("UNAVAILABLE", "delete unavailable")}
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.STARTUP_CLEANUP_ATTEMPTS", 1)
    request = _request(tmp_path, module_name)

    with pytest.raises(RenderClientError, match="cleanup failed"):
        _start(_client(module_name), request)

    fake_control_plane.delete_failures = {}
    retry_client = _client(module_name)
    _start(retry_client, request, simulation_id="sim-after-retry")

    sweep_calls = [name for name, _request in fake_control_plane.calls]
    assert sweep_calls.count("ListSimulations") == 2
    assert sweep_calls.count("DeleteSimulation") == 2
    cleanup = retry_client.startup_diagnostics["render_worker"]["cleanup"]
    assert cleanup["status"] == "swept"
    assert cleanup["deleted_count"] == 1


def test_ovrtx_runtime_client_attach_sweep_runs_once_per_worker_endpoint(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_per_endpoint_attach"
    native_module = _EndpointlessNativeRenderModule(module_name)
    _allow_concurrent_gpu_leases(monkeypatch)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)

    first_client = _client(module_name)
    _start(first_client, request, simulation_id="sim-a")
    second_client = _client(
        module_name,
        worker_command="/tmp/ovrtx-bridge-server --address 127.0.0.1 --port 50052",
    )
    second_client.start_session(_spec(request), simulation_id="sim-b")

    assert fake_control_plane.endpoints == ["127.0.0.1:50051", "127.0.0.1:50052"]
    sweep_calls = [name for name, _request in fake_control_plane.calls]
    assert sweep_calls.count("ListSimulations") == 2
    assert second_client.startup_diagnostics["render_worker"]["cleanup"]["status"] == "swept"
    assert second_client.startup_diagnostics["render_worker"]["cleanup"]["endpoint"] == "127.0.0.1:50052"


def test_session_replacement_on_attached_worker_pays_no_sweep_and_keeps_targeted_delete(
    monkeypatch,
    tmp_path: Path,
    fake_control_plane: _FakeControlPlane,
) -> None:
    module_name = "fake_ovsensors_worker_client_controller_replacement"
    native_module = _FakeNativeRenderModule(module_name)
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    request = _request(tmp_path, module_name)
    controller = OvrtxSessionController()

    first = controller.ensure(request)
    second = controller.ensure(replace(request, width=3))

    assert first.session_started is True
    assert second.session_started is True
    sweep_calls = [name for name, _request in fake_control_plane.calls]
    assert sweep_calls.count("ListSimulations") == 1
    delete_requests = [
        request["simulation_id"]
        for name, request in fake_control_plane.calls
        if name == "DeleteSimulation"
    ]
    # Break-before-make: the predecessor delete stays a targeted delete of
    # the replaced simulation, never a list/delete sweep.
    assert delete_requests == [f"ovrtx-blender-viewport-{os.getpid()}"]
    assert sleeps == []
    diagnostics = controller.diagnostics()
    assert diagnostics["startup"]["render_worker"]["cleanup"]["status"] == "skipped"
    assert diagnostics["startup"]["render_worker"]["cleanup"]["reason"] == "worker_already_attached"


def test_ovrtx_runtime_client_simulation_id_pid_parser() -> None:
    parse = ovrtx_runtime_client._simulation_id_pid

    assert parse(f"ovrtx-blender-viewport-{os.getpid()}") == os.getpid()
    assert parse("ovrtx-blender-performance-1234567890") == 1234567890
    assert parse("ovrtx-blender-viewport-") is None
    assert parse("ovrtx-blender-viewport-12a4") is None
    assert parse("ovrtx-blender-viewport-0") is None
    assert parse("someone-elses-simulation-123") is None
    assert parse("") is None


def test_ovrtx_runtime_client_preserves_rpc_status_startup_diagnostics(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_start_rpc_status_failure"
    native_module = _CreateSimulationStatusFailsNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)

    with pytest.raises(RenderClientError) as error:
        _start(client, request)

    assert error.value.protocol_diagnostics["protocol_method"] == "ControlPlaneService.CreateSimulation"  # type: ignore[attr-defined]
    diagnostics = client.startup_diagnostics["render_worker"]
    assert diagnostics["status"] == "failed"
    assert diagnostics["protocol_diagnostics"]["grpc_status"] == "UNAVAILABLE"
    assert diagnostics["protocol_diagnostics"]["request"] == {"simulation_id": "sim"}


def test_ovrtx_runtime_client_does_not_retry_permanent_create_precondition(monkeypatch, tmp_path: Path) -> None:
    module_name = "fake_ovsensors_worker_client_permanent_precondition"
    native_module = _CreateSimulationPermanentPreconditionNativeRenderModule(module_name)
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, module_name, native_module)
    monkeypatch.setattr("ovrtx_blender_example.ovrtx_runtime_client.time.sleep", sleeps.append)
    client = _client(module_name)

    with pytest.raises(RenderClientError) as error:
        _start(client, _request(tmp_path, module_name))

    assert "simulation conflict" in str(error.value)
    assert error.value.protocol_diagnostics["grpc_message"] == "simulation conflict"  # type: ignore[attr-defined]
    assert sum(name == "CreateSimulation" for name, _payload in native_module.calls) == 1
    assert sleeps == []


def test_ovrtx_runtime_client_exposes_rpc_status_diagnostics_on_read_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_RENDER_TIMEOUT_S", "1")
    module_name = "fake_ovsensors_worker_client_read_rpc_status_failure"
    native_module = _ReadWorldStateStatusFailsNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)

    with pytest.raises(RenderClientError) as error:
        _render(client, simulation_id, request, additional_samples=1)

    assert error.value.protocol_diagnostics["protocol_method"] == "WorldStateService.ReadWorldState"  # type: ignore[attr-defined]
    assert error.value.protocol_diagnostics["grpc_status"] == "DEADLINE_EXCEEDED"  # type: ignore[attr-defined]


def test_ovrtx_runtime_client_retries_bounded_read_deadline_within_logical_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_bounded_read_deadline"
    native_module = _ReadWorldStateDeadlineThenReadyNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    request = _request(tmp_path, module_name)
    client = _client(module_name)
    simulation_id = _start(client, request)

    result = _render(client, simulation_id, request, additional_samples=1)

    read_requests = [
        payload
        for name, payload in native_module.calls
        if name == "build_ReadWorldState_ldr_color" and payload.get("timeout_ms", 0) > 0
    ]
    assert [payload["timeout_ms"] for payload in read_requests] == [30_000, 30_000]
    assert [payload["timeout_seconds"] for payload in read_requests] == [30, 30]
    assert result.native_timings["read_transient_status_count"] == 1


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("input_usd_path", "No composed OVRTX scene path configured"),
        ("worker_command", "No managed ovrtx worker command configured"),
        ("native_client_module", "No native ovrtx client module configured"),
    ),
)
def test_ovrtx_runtime_client_records_validation_failure_diagnostics(
    monkeypatch,
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    monkeypatch.setenv(session_lifecycle.WORKER_LOG_ENV, str(tmp_path / "worker.log"))
    monkeypatch.setenv(session_lifecycle.RENDERER_LOG_ENV, str(tmp_path / "renderer.log"))
    request = replace(_request(tmp_path, "fake_unused_module"), **{field: ""})
    client = _client(
        request.native_client_module,
        worker_command=request.worker_command,
    )

    with pytest.raises(RenderClientError, match=message):
        _start(client, request)

    diagnostics = client.startup_diagnostics["render_worker"]
    assert diagnostics["status"] == "failed"
    assert diagnostics["error"] == message
    assert diagnostics["logs"]["worker_log"] == str(tmp_path / "worker.log")
    assert diagnostics["logs"]["renderer_log"] == str(tmp_path / "renderer.log")


def test_ovrtx_runtime_client_opens_resolution_composition_without_camera_projection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "fake_ovsensors_worker_client_no_composition"
    native_module = _FakeNativeRenderModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(work_dir))
    request = RenderRequest(
        input_usd_path=str(source),
        sensor_paths=("/Render/Product",),
        selected_sensor_paths=("/Render/Product",),
        width=1383,
        height=614,
        camera_prim_path="/World/Camera",
        worker_command="worker",
        native_client_module=module_name,
    )
    client = _client(module_name, worker_command=request.worker_command)

    _start(client, request)

    create_call = _last_call(native_module, "CreateSimulation")
    assert "/ovrtx-scene-" in create_call["usd_file_uri"]
    composition_paths = list(work_dir.glob("*.usda"))
    assert len(composition_paths) == 2
    root_path = _path_from_file_uri(create_call["usd_file_uri"])
    assert root_path in composition_paths
    presentation_path = next(path for path in composition_paths if path != root_path)
    assert presentation_path.name in root_path.read_text(encoding="utf-8")


def test_render_result_from_native_rejects_unsupported_frame_format() -> None:
    with pytest.raises(RenderClientError, match="unsupported frame format"):
        render_result_from_native(
            {
                "width": 1,
                "height": 1,
                "rgba8": b"\x00\x00\x00\xff",
                "frame_format": "linear_f32",
            },
            1,
            1,
        )


def test_render_result_from_native_rejects_unsupported_frame_color_mode() -> None:
    with pytest.raises(RenderClientError, match="unsupported frame color mode"):
        render_result_from_native(
            {
                "width": 1,
                "height": 1,
                "rgba8": b"\x00\x00\x00\xff",
                "frame_color_mode": "scene_linear",
            },
            1,
            1,
        )


def test_render_result_from_native_accepts_scene_linear_hdr_sidecar() -> None:
    result = render_result_from_native(
        {
            "width": 1,
            "height": 2,
            "rgba8": bytes([1, 2, 3, 255, 4, 5, 6, 255]),
            "linear_rgba16f": bytes([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]),
            "frame_format": color_presentation.FRAME_FORMAT_RGBA16F,
            "frame_color_mode": color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
            "render_var": color_presentation.RENDER_VAR_HDR_COLOR,
        },
        1,
        2,
    )

    assert result.frame_format == color_presentation.FRAME_FORMAT_RGBA16F
    assert result.frame_color_mode == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    assert result.render_var == color_presentation.RENDER_VAR_HDR_COLOR
    assert result.rgba8 == bytes([4, 5, 6, 255, 1, 2, 3, 255])
    assert result.linear_rgba16f == bytes([5, 0, 6, 0, 7, 0, 8, 0, 1, 0, 2, 0, 3, 0, 4, 0])


def test_render_result_from_native_rejects_hdr_without_linear_payload() -> None:
    with pytest.raises(RenderClientError, match="no HdrColor linear RGBA16F payload"):
        render_result_from_native(
            {
                "width": 1,
                "height": 1,
                "rgba8": b"\x00\x00\x00\xff",
                "frame_format": color_presentation.FRAME_FORMAT_RGBA16F,
                "frame_color_mode": color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
            },
            1,
            1,
        )
