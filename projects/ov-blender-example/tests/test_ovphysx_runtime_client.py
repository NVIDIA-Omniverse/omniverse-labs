# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_runtime_client import OvphysxRuntimeClient, _ovphysx_worker_environment, _usd_file_uri  # noqa: E402
from ovrtx_blender_example import bundled_runtime  # noqa: E402
from ovrtx_blender_example.shared_stage_composition import BodyPose, BodyVelocity  # noqa: E402
from ovrtx_blender_example.shared_stage_errors import SharedStageCompositionError  # noqa: E402


@dataclass(frozen=True)
class _Config:
    input_usd_path: str = "/tmp/stair_drop_ovrtx_ovphysx.usda"
    ovphysx_address: str = "127.0.0.1:50094"
    ovphysx_worker_command: str = "worker"
    body_prims: tuple[str, ...] = ("/World/PhysicsIsland/DynamicBodies/Cube_00",)
    worker_log_path: str = "/tmp/ovphysx-worker.log"
    ovphysx_native_client_module: str = "ovphysx_bridge_client"
    ovphysx_native_client_path: str = ""


class _FakeNativeOvphysxModule(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.environment_at_start: dict[str, str] = {}
        self.RpcStatusError = type(f"{name}_RpcStatusError", (RuntimeError,), {})
        self.client_endpoints: list[str] = []
        self.health_status = "SERVING"
        self.health_error: BaseException | None = None

    def Client(self, endpoint: str) -> object:
        self.client_endpoints.append(endpoint)
        module = self

        class Client:
            def health(self, service: str, timeout_s: float) -> str:
                module.calls.append(("health", {"service": service, "timeout_s": timeout_s}))
                if module.health_error is not None:
                    raise module.health_error
                return module.health_status

            def close(self) -> None:
                module.calls.append(("close", {}))

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
            "semantic_builders": [],
            "generic_builders": [
                "build_WriteWorldState_step",
                "build_WriteWorldState_body_poses",
                "build_WriteWorldState_body_velocities",
                "build_ReadWorldState_body_states",
            ],
            "request_handle_types": ["fake.WriteWorldStateRequest", "fake.ReadWorldStateRequest"],
            "response_helpers": ["decode_body_states"],
        }

    def start_worker(self, request: dict[str, object]) -> dict[str, object]:
        self.environment_at_start = os.environ.copy()
        self.calls.append(("start_worker", dict(request)))
        return {"status": "started", "address": "127.0.0.1:50094"}

    def connect(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("connect", dict(request)))
        return {"status": "connected", "address": request["address"]}

    def CreateSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("CreateSimulation", dict(request)))
        return {
            "status": "created",
            "simulation_id": request["simulation_id"],
            "protocol_method": "ControlPlaneService.CreateSimulation",
            "grpc_status": "OK",
        }

    def ListSimulations(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("ListSimulations", dict(request)))
        return {"simulations": [], "total": 0}

    def DeleteSimulation(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("DeleteSimulation", dict(request)))
        return {
            "status": "deleted",
            "simulation_id": request["simulation_id"],
            "protocol_method": "ControlPlaneService.DeleteSimulation",
            "grpc_status": "OK",
        }

    def build_WriteWorldState_step(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_WriteWorldState_step", dict(request)))
        return {"builder": "build_WriteWorldState_step", "request": dict(request)}

    def build_WriteWorldState_body_poses(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_WriteWorldState_body_poses", dict(request)))
        return {"builder": "build_WriteWorldState_body_poses", "request": dict(request)}

    def build_WriteWorldState_body_velocities(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_WriteWorldState_body_velocities", dict(request)))
        return {"builder": "build_WriteWorldState_body_velocities", "request": dict(request)}

    def build_ReadWorldState_body_states(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("build_ReadWorldState_body_states", dict(request)))
        return {"builder": "build_ReadWorldState_body_states", "request": dict(request)}

    def WriteWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("WriteWorldState", dict(handle)))
        request = handle["request"]
        assert isinstance(request, dict)
        response = {
            "simulation_id": request["simulation_id"],
            "simulation_time_ns": request["simulation_time_ns"],
            "written_operation_count": 1,
            "builder_name": handle["builder"],
            "write_world_state_ms": 1.0,
            "protocol_method": "WorldStateService.WriteWorldState",
            "grpc_status": "OK",
        }
        if "poses" in request:
            poses = request["poses"]
            assert isinstance(poses, list)
            response["body_count"] = len(poses)
        if "velocities" in request:
            velocities = request["velocities"]
            assert isinstance(velocities, list)
            response["body_count"] = len(velocities)
        return response

    def ReadWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("ReadWorldState", dict(handle)))
        return {
            "response_handle": {"source": "fake-read-response"},
            "result_column_count": 5,
            "read_world_state_ms": 2.5,
            "builder_name": handle["builder"],
            "protocol_method": "WorldStateService.ReadWorldState",
            "grpc_status": "OK",
        }

    def decode_body_states(self, read_handle: dict[str, object], response_handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("decode_body_states", {"read": dict(read_handle), "response": dict(response_handle)}))
        request = read_handle["request"]
        assert isinstance(request, dict)
        simulation_time_ns = int(request["simulation_time_ns"])
        y = 4.0 if simulation_time_ns else 5.0
        return {
            "states": {
                request["prim_paths"][0]: {
                    "translation": [1.0, y, 3.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "velocity": {"x": 0.0, "y": -1.0, "z": 0.0},
                    "angular_velocity": [0.0, 0.0, 1.0],
                }
            },
            "simulation_time_ns": simulation_time_ns,
            "body_count": len(request["prim_paths"]),
        }

    def shutdown(self) -> dict[str, object]:
        self.calls.append(("shutdown", {}))
        return {"status": "stopped"}


class _WriteWorldStateStatusFailsNativeOvphysxModule(_FakeNativeOvphysxModule):
    def WriteWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("WriteWorldState", dict(handle)))
        raise self.rpc_status_error(
            {
                "ok": False,
                "protocol_method": "WorldStateService.WriteWorldState",
                "grpc_status": "UNAVAILABLE",
                "grpc_status_code": 14,
                "grpc_message": "write unavailable",
                "elapsed_ms": 2.0,
                "request": {"builder_name": handle["builder"]},
            }
        )


class _WriteWorldStateReturnsErrorNativeOvphysxModule(_FakeNativeOvphysxModule):
    def WriteWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("WriteWorldState", dict(handle)))
        return {
            "status": "error",
            "failed": True,
            "grpc_status": "UNAVAILABLE",
            "body_count": 0,
            "error": "write rejected",
        }


class _WriteWorldStateReturnsSingleFieldRejection(_FakeNativeOvphysxModule):
    def __init__(self, name: str, response: dict[str, object]) -> None:
        super().__init__(name)
        self.response = response

    def WriteWorldState(self, handle: dict[str, object]) -> dict[str, object]:
        self.calls.append(("WriteWorldState", dict(handle)))
        return dict(self.response)


class _WriteWorldStateReturnsNone(_FakeNativeOvphysxModule):
    def WriteWorldState(self, handle: dict[str, object]) -> None:
        self.calls.append(("WriteWorldState", dict(handle)))
        return None


def _raw_client(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    module: _FakeNativeOvphysxModule | None = None,
) -> tuple[OvphysxRuntimeClient, _FakeNativeOvphysxModule, _Config]:
    native_module = module or _FakeNativeOvphysxModule(module_name)
    monkeypatch.setitem(sys.modules, module_name, native_module)
    config = _Config(ovphysx_native_client_module=module_name)
    return OvphysxRuntimeClient(config, "sim"), native_module, config


def _last_call(native_module: _FakeNativeOvphysxModule, name: str) -> dict[str, object]:
    for call_name, payload in reversed(native_module.calls):
        if call_name == name:
            return payload
    raise AssertionError(f"missing native call {name}")


def test_native_physics_client_normalizes_body_states_and_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    client, native_module, config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_normalize")

    client.start()
    create_diagnostics = client.create_simulation()
    states, read_diagnostics = client.read_body_states(0)

    assert native_module.calls[0][0] == "start_worker"
    assert type(native_module.calls[0][1]["ready_timeout_ms"]) is int
    assert native_module.client_endpoints == [config.ovphysx_address]
    assert native_module.calls[1][0] == "health"
    assert create_diagnostics["transport"] == "native"
    assert create_diagnostics["name"] == "CreateSimulation"
    assert create_diagnostics["response"]["protocol_method"] == "ControlPlaneService.CreateSimulation"
    assert create_diagnostics["request"]["usd_file_uri"] == _usd_file_uri(config.input_usd_path)
    assert read_diagnostics["transport"] == "native"
    assert read_diagnostics["name"] == "ReadWorldState"
    assert read_diagnostics["response"]["read_world_state"]["protocol_method"] == "WorldStateService.ReadWorldState"
    assert read_diagnostics["body_count"] == 1
    assert states == [
        {
            "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
            "simulation_time_ns": 0,
            "translate": {"found": True, "x": 1.0, "y": 5.0, "z": 3.0},
            "orient": {"found": True, "i": 0.0, "j": 0.0, "k": 0.0, "r": 1.0},
            "linear_velocity": {"found": True, "x": 0.0, "y": -1.0, "z": 0.0},
            "angular_velocity": {"found": True, "x": 0.0, "y": 0.0, "z": 1.0},
        }
    ]


def test_usd_file_uri_uses_native_windows_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "windows-x64")

    assert _usd_file_uri(str(tmp_path / "scene.usda")) == str((tmp_path / "scene.usda").resolve())


def test_native_physics_client_restores_parent_worker_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ovphysx_root = tmp_path / "ovphysx"
    ovruntime_root = tmp_path / "ovruntime"
    ovphysx_root.mkdir()
    ovruntime_root.mkdir()
    monkeypatch.setenv("OVPHYSX_ROOT", str(ovphysx_root))
    monkeypatch.setenv("OVRUNTIME_ROOT", str(ovruntime_root))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/parent/lib")
    client, native_module, _config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_environment")

    client.start()

    assert native_module.environment_at_start["OVPHYSX_ROOT"] == str(ovphysx_root)
    assert native_module.environment_at_start["OVRUNTIME_ROOT"] == str(ovruntime_root)
    assert native_module.environment_at_start["LD_LIBRARY_PATH"].endswith("/parent/lib")
    assert os.environ["OVPHYSX_ROOT"] == str(ovphysx_root)
    assert os.environ["OVRUNTIME_ROOT"] == str(ovruntime_root)
    assert os.environ["LD_LIBRARY_PATH"] == "/parent/lib"


def test_native_physics_client_reports_composite_advance_read_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    client, native_module, config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_advance")

    states, diagnostics = client.advance_and_read_body_states(2, 3, 100)

    assert [name for name, _payload in native_module.calls].count("WriteWorldState") == 3
    read_request = _last_call(native_module, "build_ReadWorldState_body_states")
    assert read_request == {
        "simulation_id": "sim",
        "prim_paths": list(config.body_prims),
        "simulation_time_ns": 500,
    }
    assert states[0]["translate"] == {"found": True, "x": 1.0, "y": 4.0, "z": 3.0}
    assert diagnostics["step_count"] == 5
    assert diagnostics["simulation_time_ns"] == 500
    assert diagnostics["body_count"] == 1
    assert diagnostics["step_ms"] == 3.0
    assert diagnostics["read_ms"] == 2.5
    assert diagnostics["total_ms"] == 5.5
    assert diagnostics["step_timings_ms"] == [1.0, 1.0, 1.0]


def test_native_physics_client_preserves_rpc_status_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "fake_ovphysx_worker_client_rpc_status_failure"
    native_module = _WriteWorldStateStatusFailsNativeOvphysxModule(module_name)
    client, native_module, _config = _raw_client(monkeypatch, module_name, native_module)

    with pytest.raises(SharedStageCompositionError) as error:
        client.advance_and_read_body_states(0, 1, 100)

    assert error.value.protocol_diagnostics["protocol_method"] == "WorldStateService.WriteWorldState"  # type: ignore[attr-defined]
    assert error.value.protocol_diagnostics["grpc_status"] == "UNAVAILABLE"  # type: ignore[attr-defined]
    assert [name for name, _payload in native_module.calls] == ["build_WriteWorldState_step", "WriteWorldState"]


@pytest.mark.parametrize(
    ("start_step_count", "steps", "timestep_ns", "message"),
    (
        (-1, 1, 100, "start_step_count must be non-negative"),
        (0, 0, 100, "steps must be positive"),
        (0, -1, 100, "steps must be positive"),
        (0, 1, 0, "timestep_ns must be positive"),
        (0, 1, -1, "timestep_ns must be positive"),
        (2**63 - 1, 1, 1, "start_step_count \\+ steps overflows int64"),
        (2**62, 1, 3, "step_count \\* timestep_ns overflows int64"),
    ),
)
def test_native_physics_client_rejects_invalid_advance_inputs_before_native_calls(
    monkeypatch: pytest.MonkeyPatch,
    start_step_count: int,
    steps: int,
    timestep_ns: int,
    message: str,
) -> None:
    client, native_module, _config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_invalid_advance")

    with pytest.raises(SharedStageCompositionError, match=message):
        client.advance_and_read_body_states(start_step_count, steps, timestep_ns)

    assert native_module.calls == []


def test_native_physics_client_writes_poses_and_shuts_down_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    client, native_module, _config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_write")
    pose = BodyPose(
        prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        translate=(1.0, 2.0, 3.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )

    client.create_simulation()
    write_diagnostics = client.write_body_poses([pose], simulation_time_ns=123, reset=True)
    client.shutdown()

    assert native_module.calls[-4:] == [
        (
            "WriteWorldState",
            {
                "builder": "build_WriteWorldState_body_poses",
                "request": {
                    "simulation_id": "sim",
                    "simulation_time_ns": 123,
                    "poses": [
                        {
                            "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
                            "translate": [1.0, 2.0, 3.0],
                            "orient": [0.0, 0.0, 0.0, 1.0],
                        }
                    ],
                    "reset": True,
                },
            },
        ),
        (
            "DeleteSimulation",
            {
                "simulation_id": "sim",
            },
        ),
        ("close", {}),
        ("shutdown", {}),
    ]
    assert native_module.calls[-5][0] == "build_WriteWorldState_body_poses"
    assert write_diagnostics["name"] == "WriteWorldState"
    assert write_diagnostics["response"]["protocol_method"] == "WorldStateService.WriteWorldState"
    assert write_diagnostics["body_count"] == 1
    assert write_diagnostics["simulation_time_ns"] == 123


def test_managed_physics_worker_requires_standard_health_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _FakeNativeOvphysxModule("fake_ovphysx_worker_client_legacy_health")
    module.health_error = module.rpc_status_error(
        {
            "protocol_method": "grpc.health.v1.Health.Check",
            "grpc_status": "UNIMPLEMENTED",
        }
    )
    client, native_module, _config = _raw_client(
        monkeypatch, module.__name__, module
    )

    with pytest.raises(SharedStageCompositionError, match="UNIMPLEMENTED"):
        client.start()

    assert _last_call(native_module, "health")["service"] == ""
    assert module.calls[-2:] == [("close", {}), ("shutdown", {})]


def test_native_physics_client_writes_typed_body_velocities(monkeypatch: pytest.MonkeyPatch) -> None:
    client, native_module, _config = _raw_client(monkeypatch, "fake_ovphysx_worker_client_velocity")
    velocity = BodyVelocity(
        "/World/PhysicsIsland/DynamicBodies/Cube_00",
        (4.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    diagnostics = client.write_body_velocities((velocity,), simulation_time_ns=124)

    assert _last_call(native_module, "build_WriteWorldState_body_velocities") == {
        "simulation_id": "sim",
        "simulation_time_ns": 124,
        "velocities": [{
            "prim_path": velocity.prim_path,
            "linear": [4.0, 0.0, 0.0],
            "angular": [0.0, 0.0, 1.0],
        }],
        "reset": False,
    }
    assert diagnostics["body_count"] == 1
    assert diagnostics["simulation_time_ns"] == 124


def test_native_physics_client_preserves_returned_write_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _WriteWorldStateReturnsErrorNativeOvphysxModule("fake_ovphysx_worker_client_rejected_write")
    client, _native_module, _config = _raw_client(
        monkeypatch,
        module.__name__,
        module,
    )
    pose = BodyPose(
        prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        translate=(1.0, 2.0, 3.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )

    diagnostics = client.write_body_poses([pose], simulation_time_ns=123)

    assert diagnostics["status"] == "error"
    assert diagnostics["failed"] is True
    assert diagnostics["grpc_status"] == "UNAVAILABLE"
    assert diagnostics["body_count"] == 0
    assert diagnostics["error"] == "write rejected"
    assert diagnostics["response"]["body_count"] == 0


@pytest.mark.parametrize(
    ("response", "key", "value"),
    [
        ({"error": "write rejected"}, "error", "write rejected"),
        ({"skipped_reason": "worker skipped write"}, "skipped_reason", "worker skipped write"),
        ({"ok": False}, "ok", False),
    ],
)
def test_native_physics_client_promotes_single_field_write_rejection(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, object],
    key: str,
    value: object,
) -> None:
    module_name = f"fake_ovphysx_worker_client_single_rejection_{key}"
    module = _WriteWorldStateReturnsSingleFieldRejection(module_name, response)
    client, _native_module, _config = _raw_client(monkeypatch, module_name, module)
    pose = BodyPose(
        prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        translate=(1.0, 2.0, 3.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )

    diagnostics = client.write_body_poses([pose], simulation_time_ns=123)

    assert diagnostics[key] == value


def test_native_physics_client_does_not_invent_acceptance_for_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "fake_ovphysx_worker_client_empty_write"
    module = _WriteWorldStateReturnsNone(module_name)
    client, _native_module, _config = _raw_client(monkeypatch, module_name, module)
    pose = BodyPose(
        prim_path="/World/PhysicsIsland/DynamicBodies/Cube_00",
        translate=(1.0, 2.0, 3.0),
        orient=(0.0, 0.0, 0.0, 1.0),
    )

    diagnostics = client.write_body_poses([pose], simulation_time_ns=123)

    assert "body_count" not in diagnostics
    assert "body_count" not in diagnostics["response"]


def test_native_import_failure_names_missing_module() -> None:
    client = OvphysxRuntimeClient(_Config(ovphysx_native_client_module="missing_ovphysx_worker_client"), "sim")

    with pytest.raises(SharedStageCompositionError, match="missing_ovphysx_worker_client"):
        client.start()


def test_ovphysx_worker_environment_uses_bundled_runtime_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    addon_root = tmp_path / "addon"
    ovphysx_bridge_root = addon_root / "runtime" / "ovphysx-bridge-server"
    ovphysx_root = ovphysx_bridge_root / "private" / "ovphysx-runtime"
    ovruntime_root = ovphysx_root
    ovphysx_root.mkdir(parents=True)
    monkeypatch.delenv("OVPHYSX_ROOT", raising=False)
    monkeypatch.delenv("OVRUNTIME_ROOT", raising=False)
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: addon_root)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    env = _ovphysx_worker_environment(SimpleNamespace(ovphysx_address="127.0.0.1:50094"))

    assert env["OVPHYSX_ROOT"] == str(ovphysx_root)
    assert env["OVPHYSX_LIB"] == str(ovphysx_root / "lib")
    assert env["OVRUNTIME_ROOT"] == str(ovruntime_root)
    assert str(ovphysx_root / "lib") in env["LD_LIBRARY_PATH"]
    assert str(ovphysx_bridge_root / "lib") in env["LD_LIBRARY_PATH"]
    assert str(ovphysx_bridge_root / "private" / "ovstage" / "bin") in env["LD_LIBRARY_PATH"]
    assert str(ovruntime_root / "lib") in env["LD_LIBRARY_PATH"]

    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "windows-x64")
    env = _ovphysx_worker_environment(SimpleNamespace(ovphysx_address="127.0.0.1:50094"))
    assert env["OVPHYSX_LIB"] == str(ovphysx_root / "bin")


def test_ovphysx_worker_environment_does_not_discover_sibling_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OVPHYSX_ROOT", raising=False)
    monkeypatch.delenv("OVRUNTIME_ROOT", raising=False)
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: tmp_path / "addon")
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    env = _ovphysx_worker_environment(SimpleNamespace(ovphysx_address="127.0.0.1:50094"))

    assert "OVPHYSX_ROOT" not in env
    assert "OVRUNTIME_ROOT" not in env
