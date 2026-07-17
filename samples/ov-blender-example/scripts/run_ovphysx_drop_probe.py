#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an OVPhysX drop/settle probe through ovphysx_client."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADDRESS = "127.0.0.1:50091"
DEFAULT_DEVICE = "cpu"
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "assets"
    / "demo_stair_drop_1280x720"
    / "usd"
    / "stair_drop_ovrtx_ovphysx.usda"
)
DEFAULT_BODY_PRIM = "/World/PhysicsIsland/DynamicBodies/Cube_00"
UNKNOWN = "???"

sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import bundled_runtime  # noqa: E402
from ovrtx_blender_example.ovphysx_runtime_client import (  # noqa: E402
    DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE,
    DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH,
    OvphysxRuntimeClient,
)
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": args.output_dir / "result.json",
        "worker_log": args.output_dir / "ovphysx-worker.log",
    }
    result = _base_result(args, paths)
    preflight = _preflight(args)
    result["preflight"] = preflight

    if preflight["blockers"]:
        result["status"] = "blocked-preflight"
        result["error"] = "OVPhysX native client prerequisites are missing."
        result["blockers"] = preflight["blockers"]
        _write_result(paths["result"], result)
        _print_result(result, paths["result"])
        return 2 if args.require_real else 0

    try:
        runtime_result = _run_probe(args, paths)
    except Exception as exc:  # noqa: BLE001 - diagnostics need the exact runtime failure.
        result["status"] = "failed-real"
        result["error"] = str(exc)
        result["failure_type"] = exc.__class__.__name__
        result["worker_log"] = str(paths["worker_log"])
        result["worker_log_tail"] = _read_tail(paths["worker_log"])
        _write_result(paths["result"], result)
        _print_result(result, paths["result"])
        return 1

    result.update(runtime_result)
    _write_result(paths["result"], result)
    _print_result(result, paths["result"])
    return 0


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        type=Path,
        default=_env_path("OV_BLENDER_EXAMPLE_OVPHYSX_SERVER"),
        help="Installed ovphysx-bridge-server binary.",
    )
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument(
        "--worker-command",
        default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND", ""),
        help="Command passed to ovphysx_bridge_client.start_worker.",
    )
    parser.add_argument(
        "--ovphysx-native-client-path",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH", str(DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH))),
    )
    parser.add_argument(
        "--ovphysx-native-client-module",
        default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_MODULE", DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE),
    )
    parser.add_argument("--address", default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS", DEFAULT_ADDRESS))
    parser.add_argument("--device", default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", DEFAULT_DEVICE), choices=("cpu", "gpu", "auto"))
    parser.add_argument("--box-prim", default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_BOX_PRIM", DEFAULT_BODY_PRIM))
    parser.add_argument("--sim-id", default=f"adr0015-native-drop-{int(time.time())}")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "artifacts" / "ovphysx-drop-probe")
    parser.add_argument("--ovphysx-root", type=Path, default=_env_path("OVPHYSX_ROOT"))
    parser.add_argument("--ovruntime-root", type=Path, default=_env_path("OVRUNTIME_ROOT"))
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Exit nonzero instead of writing a blocked artifact when prerequisites are missing.",
    )
    args = parser.parse_args(list(argv))
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    args.worker_command_is_override = bool(args.worker_command.strip())
    if not args.worker_command_is_override and args.server is None:
        parser.error("--server or --worker-command is required")
    args.fixture = args.fixture or DEFAULT_FIXTURE
    if not args.worker_command_is_override:
        args.worker_command = _default_worker_command(args.server, args.address, args.device)
    return args


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _default_worker_command(server: Path, address: str, device: str) -> str:
    return bundled_runtime.serialize_command(
        [str(server), "--listen", address, "--device", device]
    )


def _base_result(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": "ovphysx-native-drop-probe",
        "status": "running",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "runtime_artifacts_available": False,
        "topology": {
            "physics_worker": "managed by ovphysx_bridge_client native adapter",
            "container": False,
            "client_probe": "ovphysx_bridge_client native Python extension",
        },
        "native_client": {
            "module": args.ovphysx_native_client_module,
            "path": str(args.ovphysx_native_client_path),
        },
        "public_boundary": {
            "client_api": "ovphysx_bridge_client",
            "create_operation": "create_simulation",
            "read_operation": "read_body_states",
            "advance_operation": "advance_and_read_body_states",
            "cleanup_operation": "delete_simulation via OvphysxRuntimeClient.shutdown",
            "transform_attribute": "xformOp:translate",
            "linear_velocity_attribute": "physics:velocity",
            "angular_velocity_attribute": "physics:angularVelocity",
        },
        "scenario": {
            "type": "drop-step-read",
            "fixture": str(args.fixture),
            "rigid_body_prim_path": args.box_prim,
            "steps": args.steps,
            "fps": args.fps,
            "timestep_ns": int(1_000_000_000 / args.fps),
            "device": args.device,
        },
        "environment": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "nvidia_smi": _nvidia_smi(),
        },
        "arguments": {
            "output_dir": str(args.output_dir),
            "worker_command": args.worker_command,
            "worker_command_is_override": args.worker_command_is_override,
            "server": str(args.server) if args.server else UNKNOWN,
            "address": args.address,
            "fixture": str(args.fixture),
            "ovphysx_native_client_path": str(args.ovphysx_native_client_path),
            "ovphysx_native_client_module": args.ovphysx_native_client_module,
            "ovphysx_root": str(args.ovphysx_root) if args.ovphysx_root else UNKNOWN,
            "ovruntime_root": str(args.ovruntime_root) if args.ovruntime_root else UNKNOWN,
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "fixture": _path_check(args.fixture, "file"),
        "server_binary": _optional_path_check(args.server, "executable"),
        "worker_command": _command_check(args.worker_command),
        "ovphysx_native_extension": _extension_check(args.ovphysx_native_client_path, args.ovphysx_native_client_module),
        "ovphysx_root": _optional_path_check(args.ovphysx_root, "dir"),
        "ovruntime_root": _optional_path_check(args.ovruntime_root, "dir"),
    }
    required = ["fixture", "ovphysx_native_extension"]
    blockers = [
        f"{name} missing or invalid: {checks[name].get('path') or checks[name].get('command') or UNKNOWN}"
        for name in required
        if not checks[name].get("ok")
    ]
    if args.worker_command_is_override:
        if not checks["worker_command"].get("ok"):
            blockers.append(
                f"worker_command missing or invalid: {checks['worker_command'].get('path') or checks['worker_command'].get('command') or UNKNOWN}"
            )
    elif not checks["server_binary"].get("ok"):
        blockers.append(f"server_binary missing or invalid: {checks['server_binary']['path']}")
    elif not checks["worker_command"].get("ok"):
        blockers.append(
            f"worker_command missing or invalid: {checks['worker_command'].get('path') or checks['worker_command'].get('command') or UNKNOWN}"
        )
    return {"checks": checks, "blockers": blockers}


def _path_check(path: Path, kind: str) -> dict[str, Any]:
    ok = path.exists()
    if kind == "dir":
        ok = path.is_dir()
    elif kind == "file":
        ok = path.is_file()
    elif kind == "executable":
        ok = path.is_file() and os.access(path, os.X_OK)
    return {"path": str(path), "kind": kind, "ok": ok}


def _optional_path_check(path: Path | None, kind: str) -> dict[str, Any]:
    if path is None:
        return {"path": UNKNOWN, "kind": kind, "ok": False, "required": False}
    result = _path_check(path, kind)
    result["required"] = False
    return result


def _extension_check(directory: Path, module_name: str) -> dict[str, Any]:
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    expected = directory / f"{module_name}{suffix}"
    if suffix and expected.is_file():
        return {"path": str(expected), "kind": "file", "ok": True}
    fallback = sorted(directory.glob(f"{module_name}*.so"))
    return {
        "path": str(expected if suffix else directory / f"{module_name}.so"),
        "kind": "file",
        "fallback_matches": [str(path) for path in fallback],
        "ok": False,
    }


def _command_check(command: str) -> dict[str, Any]:
    return {"command": command, "kind": "command", "ok": bool(command.strip())}


def _run_probe(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    _apply_runtime_roots(args)
    config = _physics_config(args, paths["worker_log"])
    client = OvphysxRuntimeClient(config, args.sim_id)
    created = False
    try:
        client.start()
        create_response = client.create_simulation()
        created = True
        initial_states, initial_read = client.read_body_states(0)
        final_states, advance_read = client.advance_and_read_body_states(0, args.steps, config.timestep_ns)
    finally:
        client.shutdown()

    initial_state = _state_for_path(initial_states, args.box_prim)
    final_state = _state_for_path(final_states, args.box_prim)
    initial_position = _mapping_or_unknown(initial_state.get("translate") if initial_state else None)
    final_position = _mapping_or_unknown(final_state.get("translate") if final_state else None)
    final_velocity = _mapping_or_unknown(final_state.get("linear_velocity") if final_state else None)
    final_angular_velocity = _mapping_or_unknown(final_state.get("angular_velocity") if final_state else None)
    step_samples_ms = _float_sequence(advance_read.get("step_timings_ms"))
    read_samples_ms = _single_float_sequence(advance_read.get("read_ms"))
    return {
        "status": "pass-real",
        "runtime_artifacts_available": True,
        "worker": {
            "command": args.worker_command,
            "address": args.address,
            "log": str(paths["worker_log"]),
            "managed_as": "ovphysx_bridge_client.start_worker",
            "container": False,
        },
        "native_client_diagnostics": {
            "create": create_response,
            "initial_read": initial_read,
            "advance_and_read": advance_read,
            "cleanup": {
                "managed_by": "OvphysxRuntimeClient.shutdown",
                "delete_attempted": created,
            },
        },
        "body_states": {
            "initial": initial_states,
            "final": final_states,
        },
        "correctness": {
            "initial_translate": initial_position,
            "final_translate": final_position,
            "final_linear_velocity": final_velocity,
            "final_angular_velocity": final_angular_velocity,
            "moved_down_y": _moved_axis(initial_position, final_position, "y", direction=-1),
            "moved_down_z": _moved_axis(initial_position, final_position, "z", direction=-1),
            "moved_forward_y": _moved_axis(initial_position, final_position, "y", direction=1),
            "settled_velocity_candidate": _settled(final_velocity, threshold=0.05),
        },
        "performance": {
            "step_count": args.steps,
            "step_latency_ms": _summary(step_samples_ms),
            "read_latency_ms": _summary(read_samples_ms),
            "simulation_fps_by_step_rpc": _rate(step_samples_ms),
            "state_transport_included": True,
        },
        "worker_log_tail": _read_tail(paths["worker_log"]),
    }


def _physics_config(args: argparse.Namespace, worker_log: Path) -> InteractiveSharedStageConfig:
    return InteractiveSharedStageConfig(
        enabled=True,
        input_usd_path=str(args.fixture),
        server=str(args.server) if args.server else "",
        ovphysx_address=args.address,
        ovphysx_worker_command=args.worker_command,
        device=args.device,
        body_root=str(Path(args.box_prim).parent).replace("\\", "/"),
        body_prims=(args.box_prim,),
        physics_fps=args.fps,
        update_fps=args.fps,
        max_steps=args.steps,
        body_scale=1.0,
        worker_log_path=str(worker_log),
        ovphysx_native_client_module=args.ovphysx_native_client_module,
        ovphysx_native_client_path=str(args.ovphysx_native_client_path),
    )


def _apply_runtime_roots(args: argparse.Namespace) -> None:
    if args.ovphysx_root is not None:
        os.environ["OVPHYSX_ROOT"] = str(args.ovphysx_root)
    if args.ovruntime_root is not None:
        os.environ["OVRUNTIME_ROOT"] = str(args.ovruntime_root)


def _state_for_path(states: Sequence[Mapping[str, Any]], prim_path: str) -> Mapping[str, Any] | None:
    for state in states:
        if state.get("prim_path") == prim_path:
            return state
    return states[0] if states else None


def _mapping_or_unknown(value: Any) -> dict[str, Any] | str:
    return dict(value) if isinstance(value, Mapping) else UNKNOWN


def _moved_axis(initial: Mapping[str, Any] | str, final: Mapping[str, Any] | str, axis: str, *, direction: int) -> bool | str:
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        return UNKNOWN
    if initial.get("found") is False or final.get("found") is False:
        return UNKNOWN
    delta = float(final.get(axis, 0.0)) - float(initial.get(axis, 0.0))
    return delta > 0.0 if direction > 0 else delta < 0.0


def _settled(vector: Mapping[str, Any] | str, threshold: float) -> bool | str:
    if not isinstance(vector, Mapping) or vector.get("found") is False:
        return UNKNOWN
    magnitude = math.sqrt(float(vector.get("x", 0.0)) ** 2 + float(vector.get("y", 0.0)) ** 2 + float(vector.get("z", 0.0)) ** 2)
    return magnitude <= threshold


def _float_sequence(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    floats: list[float] = []
    for item in value:
        try:
            floats.append(float(item))
        except (TypeError, ValueError):
            continue
    return floats


def _single_float_sequence(value: Any) -> list[float]:
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _summary(samples_ms: Sequence[float]) -> dict[str, float | int | str]:
    if not samples_ms:
        return {"count": 0, "mean": UNKNOWN, "p50": UNKNOWN, "p95": UNKNOWN, "p99": UNKNOWN}
    ordered = sorted(samples_ms)
    return {
        "count": len(samples_ms),
        "mean": statistics.fmean(samples_ms),
        "p50": statistics.median(ordered),
        "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "p99": ordered[max(0, int(len(ordered) * 0.99) - 1)],
    }


def _rate(samples_ms: Sequence[float]) -> float | str:
    total_s = sum(samples_ms) / 1000.0
    return len(samples_ms) / total_s if total_s > 0 else UNKNOWN


def _nvidia_smi() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "error": "nvidia-smi not found"}
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr[-2048:] or completed.stdout[-2048:]}
    devices = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, name, driver_version, memory_total_mib = parts
        devices.append(
            {
                "index": _maybe_int(index),
                "name": name,
                "driver_version": driver_version,
                "memory_total_mib": _maybe_int(memory_total_mib),
            }
        )
    return {"available": True, "devices": devices}


def _maybe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _read_tail(path: Path, limit: int = 4096) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    payload = dict(result)
    payload["completed_at_ns"] = time.time_ns()
    payload["runtime_artifacts_available"] = payload.get("status") == "pass-real"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_result(result: Mapping[str, Any], path: Path) -> None:
    print(json.dumps({"status": result.get("status"), "result": str(path)}, indent=2, sort_keys=True))


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
