#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the first OVRTX + OVPhysX shared-stage composition probe."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
FIXTURES_ROOT = ROOT / "tests" / "fixtures"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(FIXTURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURES_ROOT))
DEFAULT_OVRTX_NATIVE_DIR = Path(
    os.environ.get("OVRTX_CLIENT_ROOT", ROOT / "out" / "missing-ovrtx-client")
)
DEFAULT_OVPHYSX_ADDRESS = "127.0.0.1:50094"
DEFAULT_FIXTURE_MANIFEST = FIXTURES_ROOT
DEFAULT_FIXTURE_ID = "demo_stair_drop_1280x720"
DEFAULT_RENDER_PRODUCT_PATH = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
DEFAULT_DYNAMIC_BODY_ROOT = "/World/PhysicsIsland/DynamicBodies"
DEFAULT_DEVICE = "cpu"
UNKNOWN = "???"

sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    OvphysxStageController,
)
from ovrtx_blender_example.physics_body_prims import discover_dynamic_body_prims  # noqa: E402
from ovrtx_blender_example.ovphysx_runtime_client import (  # noqa: E402
    DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE,
    DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH,
    OvphysxRuntimeClient,
)
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    RuntimeScheduler,
    RuntimeTickRequest,
    RuntimeTickResult,
    RuntimeTickStatus,
)
from ovrtx_blender_example import bundled_runtime, color_presentation  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import (  # noqa: E402
    OvrtxRuntimeClient,
    RenderResult,
)
from ovrtx_blender_example.ovrtx_value_updates import OvrtxSessionUpdatePort  # noqa: E402
from ovrtx_blender_example import ovrtx_session  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.shared_stage_composition import BodyPose, write_rgba_png  # noqa: E402
from ovrtx_blender_example.ovphysx_to_ovrtx import translate_values  # noqa: E402
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig  # noqa: E402
from ovrtx_probe_support import (  # noqa: E402
    native_extension_check,
    resolve_fixture,
    write_result,
)


def main(argv: Sequence[str] | None = None) -> int:
    _require_background_blender()
    args = _parse_args(_blender_args(sys.argv) if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": args.output_dir / "result.json",
        "initial_image": args.output_dir / "initial.png",
        "post_step_image": args.output_dir / "post-step.png",
        "ovphysx_worker_log": args.output_dir / "ovphysx-worker.log",
        "ovrtx_worker_log": args.output_dir / "ovrtx-worker.log",
    }
    result = _base_result(args, paths)
    preflight = _preflight(args)
    result["preflight"] = preflight
    if preflight["blockers"]:
        result["status"] = "blocked-preflight"
        result["error"] = "Shared-stage composition prerequisites are missing."
        result["blockers"] = preflight["blockers"]
        write_result(paths["result"], result)
        _print_result(paths["result"], result)
        return 2

    try:
        runtime = _run_probe(args, paths)
    except Exception as exc:  # noqa: BLE001 - diagnostics need the exact runtime failure.
        result["status"] = "failed-real"
        result["failure_type"] = exc.__class__.__name__
        result["error"] = str(exc)
        result["logs"] = {
            "ovphysx_worker_tail": _read_tail(paths["ovphysx_worker_log"]),
            "ovrtx_worker_tail": _read_tail(paths["ovrtx_worker_log"]),
        }
        write_result(paths["result"], result)
        _print_result(paths["result"], result)
        return 1

    result.update(runtime)
    write_result(paths["result"], result)
    _print_result(paths["result"], result)
    return 0 if result["status"] == "pass-real" else 1


def _blender_args(argv: Sequence[str]) -> list[str]:
    arguments = list(argv)
    try:
        separator = arguments.index("--")
    except ValueError as exc:
        raise RuntimeError(
            "pass probe arguments after Blender's -- separator"
        ) from exc
    return arguments[separator + 1 :]


def _require_background_blender() -> None:
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("run this probe inside background Blender") from exc
    if not bpy.app.background:
        raise RuntimeError("run this probe inside background Blender")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ovphysx-bridge-root",
        type=Path,
        default=_env_path("OVPHYSX_BRIDGE_ROOT"),
        help="Deployed ovphysx-bridge package root.",
    )
    parser.add_argument("--server", type=Path, default=None, help="ovphysx-bridge-server binary.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_FIXTURE_MANIFEST", str(DEFAULT_FIXTURE_MANIFEST))),
    )
    parser.add_argument("--fixture-id", default=DEFAULT_FIXTURE_ID)
    parser.add_argument("--fixture", type=Path, default=None, help="Direct fixture override for specialized diagnostics.")
    parser.add_argument("--ovrtx-native-dir", type=Path, default=DEFAULT_OVRTX_NATIVE_DIR)
    parser.add_argument(
        "--ovphysx-native-client-path",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH", str(DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH))),
    )
    parser.add_argument(
        "--ovphysx-native-client-module",
        default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_MODULE", DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE),
    )
    parser.add_argument("--ovphysx-address", default=DEFAULT_OVPHYSX_ADDRESS)
    parser.add_argument("--ovphysx-worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND", ""))
    parser.add_argument("--ovrtx-worker-command", default=_default_ovrtx_worker_command())
    parser.add_argument("--ovphysx-root", type=Path, default=_env_path("OVPHYSX_ROOT"))
    parser.add_argument("--ovruntime-root", type=Path, default=_env_path("OVRUNTIME_ROOT"))
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    parser.add_argument("--device", default=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", DEFAULT_DEVICE), choices=("cpu", "gpu", "auto"))
    parser.add_argument("--render-product-path", default=None)
    parser.add_argument("--camera-prim-path", default=None)
    parser.add_argument("--body-prim", action="append", default=None)
    parser.add_argument("--body-root", default=DEFAULT_DYNAMIC_BODY_ROOT)
    parser.add_argument("--sim-id", default=f"shared-stage-{int(time.time())}")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--initial-samples", type=int, default=1)
    parser.add_argument("--post-step-samples", type=int, default=1)
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "artifacts" / "shared-stage-composition")
    args = parser.parse_args(list(argv))
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.initial_samples <= 0 or args.post_step_samples <= 0:
        parser.error("--initial-samples and --post-step-samples must be positive")

    if args.server is None and args.ovphysx_bridge_root is not None:
        args.server = args.ovphysx_bridge_root / "bin" / "ovphysx-bridge-server"
    args.fixture_resolution_error = ""
    direct_fixture = args.fixture is not None
    if args.fixture is None:
        try:
            fixture = resolve_fixture(args.manifest, args.fixture_id)
        except (OSError, ValueError) as exc:
            args.fixture = args.manifest
            args.fixture_resolution_error = str(exc)
            args.render_product_path = args.render_product_path or DEFAULT_RENDER_PRODUCT_PATH
            args.camera_prim_path = args.camera_prim_path or ""
        else:
            args.fixture = Path(fixture["fixture_usd_path"])
            args.render_product_path = args.render_product_path or fixture["render_product_path"]
            args.camera_prim_path = args.camera_prim_path or fixture["camera_prim_path"]
    else:
        args.render_product_path = args.render_product_path or DEFAULT_RENDER_PRODUCT_PATH
        args.camera_prim_path = args.camera_prim_path or ""
    if direct_fixture and not args.camera_prim_path:
        parser.error("--camera-prim-path is required with a direct fixture")
    args.ovphysx_native_client_path = args.ovphysx_native_client_path.expanduser()
    packaged_runtime = (
        args.ovphysx_bridge_root / "private" / "ovphysx-runtime"
        if args.ovphysx_bridge_root is not None
        else None
    )
    args.ovphysx_root = args.ovphysx_root or packaged_runtime
    args.ovruntime_root = args.ovruntime_root or packaged_runtime
    args.body_root = args.body_root.strip()
    if not args.body_root:
        parser.error("--body-root must not be empty")
    if args.body_prim:
        args.body_prims = tuple(args.body_prim)
    elif args.fixture_resolution_error:
        args.body_prims = ()
    else:
        discovered = discover_dynamic_body_prims(str(args.fixture), args.body_root)
        if discovered:
            args.body_prims = discovered
        else:
            parser.error(f"no dynamic rigid bodies found under {args.body_root} in {args.fixture}")
    args.ovphysx_worker_command_is_override = bool(args.ovphysx_worker_command.strip())
    if not args.ovphysx_worker_command_is_override:
        args.ovphysx_worker_command = _default_ovphysx_worker_command(args.server, args.ovphysx_address, args.device)
    return args


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _default_ovrtx_worker_command() -> str:
    package_root = _env_path("OVRTX_BRIDGE_ROOT")
    if package_root is None:
        return ""
    executable = package_root / "bin" / "ovrtx-bridge-server"
    if not executable.is_file():
        return ""
    return bundled_runtime.serialize_command(
        [
            str(executable),
            "--address",
            "127.0.0.1",
            "--port",
            "50051",
            "--package-root",
            str(package_root),
        ]
    )


def _default_ovphysx_worker_command(server: Path, address: str, device: str) -> str:
    return bundled_runtime.serialize_command(
        [str(server), "--listen", address, "--device", device]
    )


def _base_result(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": "shared-stage-composition-probe",
        "status": "running",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "runtime_artifacts_available": False,
        "topology": {
            "stage_host": "demo-local Python RuntimeStageHost",
            "physics_worker": "managed by ovphysx_bridge_client native adapter",
            "physics_client": "ovphysx_bridge_client native extension",
            "render_worker": "managed local ovrtx-bridge-server subprocess",
            "render_client": "ovrtx_bridge_client native extension",
            "container": False,
        },
        "public_boundary": {
            "physics_client_api": "ovphysx_bridge_client",
            "physics_create_operation": "create_simulation",
            "physics_read_operation": "read_body_states",
            "physics_advance_operation": "advance_and_read_body_states",
            "render_update_attributes": ["usd-path", "omni:xform", "omni:resetXformStack"],
        },
        "deployment": {
            "ovphysx_bridge_root": (
                str(args.ovphysx_bridge_root)
                if args.ovphysx_bridge_root is not None
                else UNKNOWN
            ),
            "ovrtx_client_root": str(args.ovrtx_native_dir),
            "ovphysx_client_root": str(args.ovphysx_native_client_path),
        },
        "scenario": {
            "manifest": str(args.manifest),
            "fixture_id": args.fixture_id,
            "fixture": str(args.fixture),
            "render_product_path": args.render_product_path,
            "body_root": args.body_root,
            "body_prims": list(args.body_prims),
            "steps": args.steps,
            "fps": args.fps,
            "timestep_ns": _dt_ns(args),
            "width": args.width,
            "height": args.height,
            "body_scale": args.body_scale,
        },
        "arguments": {
            "output_dir": str(args.output_dir),
            "ovphysx_bridge_root": (
                str(args.ovphysx_bridge_root)
                if args.ovphysx_bridge_root is not None
                else UNKNOWN
            ),
            "server": str(args.server),
            "ovrtx_native_dir": str(args.ovrtx_native_dir),
            "ovphysx_native_client_path": str(args.ovphysx_native_client_path),
            "ovphysx_native_client_module": args.ovphysx_native_client_module,
            "ovphysx_worker_command": args.ovphysx_worker_command,
            "ovphysx_worker_command_is_override": args.ovphysx_worker_command_is_override,
            "ovrtx_worker_command": args.ovrtx_worker_command,
            "active_cuda_gpus": args.active_cuda_gpus,
            "ovphysx_address": args.ovphysx_address,
            "ovphysx_root": str(args.ovphysx_root) if args.ovphysx_root else UNKNOWN,
            "ovruntime_root": str(args.ovruntime_root) if args.ovruntime_root else UNKNOWN,
            "device": args.device,
            "sim_id": args.sim_id,
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "ovphysx_bridge_root": _optional_path_check(
            args.ovphysx_bridge_root, "dir"
        ),
        "fixture_catalog": {
            "kind": "fixture-manifest",
            "path": str(args.manifest),
            "fixture_id": args.fixture_id,
            "ok": not bool(args.fixture_resolution_error),
            "error": args.fixture_resolution_error,
        },
        "fixture": _path_check(args.fixture, "file"),
        "server_binary": _optional_path_check(args.server, "executable"),
        "worker_command": _command_check(args.ovphysx_worker_command),
        "ovphysx_native_extension": native_extension_check(args.ovphysx_native_client_path, args.ovphysx_native_client_module),
        "ovrtx_native_extension": native_extension_check(args.ovrtx_native_dir, "ovrtx_bridge_client"),
        "ovrtx_worker_command": _command_check(args.ovrtx_worker_command),
        "ovphysx_root": _optional_path_check(args.ovphysx_root, "dir"),
        "ovruntime_root": _optional_path_check(args.ovruntime_root, "dir"),
    }
    required = [
        "fixture_catalog",
        "fixture",
        "ovphysx_native_extension",
        "ovrtx_native_extension",
        "ovrtx_worker_command",
    ]
    blockers = [
        f"{name} missing or invalid: {checks[name].get('path') or checks[name].get('command') or UNKNOWN}"
        for name in required
        if not checks[name].get("ok")
    ]
    if args.ovphysx_worker_command_is_override:
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
    return {"kind": kind, "path": str(path), "ok": ok}


def _optional_path_check(path: Path | None, kind: str) -> dict[str, Any]:
    if path is None:
        return {"path": UNKNOWN, "kind": kind, "ok": False, "required": False}
    result = _path_check(path, kind)
    result["required"] = False
    return result


def _command_check(command: str) -> dict[str, Any]:
    return {"kind": "command", "command": command, "ok": bool(command.strip())}


def _run_probe(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    _add_ovrtx_client_paths(args.ovrtx_native_dir)
    render_client: OvrtxRuntimeClient | None = None
    active_cuda_gpus = str(args.active_cuda_gpus).strip()
    previous_active_cuda_gpus = os.environ.get("OVRTX_ACTIVE_CUDA_GPUS")
    if active_cuda_gpus:
        os.environ["OVRTX_ACTIVE_CUDA_GPUS"] = active_cuda_gpus

    _apply_runtime_roots(args)
    dt_ns = _dt_ns(args)
    target_ns = args.steps * dt_ns
    physics_sim_id = args.sim_id + "-physx"
    render_sim_id = args.sim_id + "-render"
    physics_config = _physics_config(args, paths["ovphysx_worker_log"])
    scheduler = RuntimeScheduler(
        config_factory=lambda _input_usd_path: physics_config,
        controller_factory=lambda config: OvphysxStageController(
            config,
            physics_client=OvphysxRuntimeClient(config, physics_sim_id),
            simulation_id=physics_sim_id,
        ),
    )
    render_simulation_id = ""
    initial_mutation: Any = None
    post_mutation: Any = None

    try:
        render_request = RenderRequest(
            input_usd_path=str(args.fixture),
            sensor_paths=(args.render_product_path,),
            selected_sensor_paths=(args.render_product_path,),
            worker_command=args.ovrtx_worker_command,
            width=args.width,
            height=args.height,
            min_samples=1,
            max_samples=max(args.initial_samples, args.post_step_samples),
            camera_prim_path=args.camera_prim_path,
            color_presentation=color_presentation.presentation_from_scene(
                None,
                requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
            ),
        )
        render_client = OvrtxRuntimeClient(
            worker_command=render_request.worker_command,
            native_client_module=render_request.native_client_module,
        )
        render_simulation_id = render_client.start_session(
            ovrtx_session.build_spec(render_request),
            simulation_id=render_sim_id,
        )
        ovrtx_updates = OvrtxSessionUpdatePort(render_client, render_simulation_id)
        initial_tick = scheduler.tick_viewport(
            RuntimeTickRequest(
                input_usd_path=str(args.fixture),
                now_ns=0,
                timeline_controls_enabled=False,
            ),
            ovrtx_updates=ovrtx_updates,
        )
        _raise_if_failed_tick(initial_tick, "initial shared runtime composition")
        initial_update = dict(initial_tick.update)
        initial_diagnostics = dict(scheduler.diagnostics())
        initial_mutation = initial_diagnostics.get("stage_host", {}).get("last_mutation")
        initial_read = dict(initial_diagnostics.get("last_physics_read_diagnostics") or {})
        physics_create = dict(initial_diagnostics.get("physics_create_diagnostics") or {})
        initial_states = _stage_pose_states(
            initial_tick.physics_pose_set,
            _mapping_int(initial_update, "simulation_time_ns", 0),
        )
        initial_render_result = _render_result(
            render_client,
            render_simulation_id,
            render_request,
            additional_samples=args.initial_samples,
        )

        post_tick = scheduler.tick_viewport(
            RuntimeTickRequest(
                input_usd_path=str(args.fixture),
                now_ns=physics_config.update_interval_ns,
                timeline_controls_enabled=False,
            ),
            ovrtx_updates=ovrtx_updates,
        )
        _raise_if_failed_tick(post_tick, "post-step shared runtime composition")
        post_update = dict(post_tick.update)
        post_diagnostics = dict(scheduler.diagnostics())
        post_mutation = post_diagnostics.get("stage_host", {}).get("last_mutation")
        advance_read = dict(
            post_diagnostics.get("last_physics_step_diagnostics")
            or post_diagnostics.get("last_physics_read_diagnostics")
            or {}
        )
        post_states = _stage_pose_states(
            post_tick.physics_pose_set,
            _mapping_int(post_update, "simulation_time_ns", target_ns),
        )
        changed_paths = list(post_update.get("dirty_paths", ()))
        transform_values = (
            translate_values(
                tuple(
                    pose for pose in post_tick.physics_pose_set
                    if pose.prim_path in changed_paths
                ),
                args.body_scale,
            )
            if changed_paths
            else []
        )
        value_update = (
            dict(post_diagnostics.get("last_pose_projection_application") or {})
            if changed_paths
            else {"skipped": True, "reason": "no changed pose paths"}
        )
        scheduler_diagnostics = scheduler.diagnostics()
        post_step_render_result = _render_result(
            render_client,
            render_simulation_id,
            render_request,
            additional_samples=args.post_step_samples,
        )
    finally:
        try:
            scheduler.shutdown()
        finally:
            try:
                if render_client is not None:
                    render_client.shutdown()
            finally:
                if active_cuda_gpus:
                    if previous_active_cuda_gpus is None:
                        os.environ.pop("OVRTX_ACTIVE_CUDA_GPUS", None)
                    else:
                        os.environ["OVRTX_ACTIVE_CUDA_GPUS"] = previous_active_cuda_gpus

    initial_png = write_rgba_png(
        paths["initial_image"],
        int(_value(initial_render_result, "width", 0)),
        int(_value(initial_render_result, "height", 0)),
        _rotate_rgba8_180(
            bytes(_value(initial_render_result, "rgba8", b"")),
            int(_value(initial_render_result, "width", 0)),
            int(_value(initial_render_result, "height", 0)),
        ),
    )
    post_png = write_rgba_png(
        paths["post_step_image"],
        int(_value(post_step_render_result, "width", 0)),
        int(_value(post_step_render_result, "height", 0)),
        _rotate_rgba8_180(
            bytes(_value(post_step_render_result, "rgba8", b"")),
            int(_value(post_step_render_result, "width", 0)),
            int(_value(post_step_render_result, "height", 0)),
        ),
    )
    moved_down_z = _moved_down_z(initial_states, post_states)
    render_results_differ = initial_png["sha256"] != post_png["sha256"]
    return {
        "status": (
            "pass-real"
            if changed_paths and moved_down_z and render_results_differ
            else "failed-real"
        ),
        "completed_at_ns": time.time_ns(),
        "runtime_artifacts_available": bool(changed_paths),
        "physics": {
            "worker": {
                "command": args.ovphysx_worker_command,
                "address": args.ovphysx_address,
                "log": str(paths["ovphysx_worker_log"]),
                "managed_as": "ovphysx_bridge_client.start_worker",
                "container": False,
            },
            "native_client": {
                "module": args.ovphysx_native_client_module,
                "path": str(args.ovphysx_native_client_path),
            },
            "create": physics_create,
            "initial_read": initial_read,
            "advance_and_read": advance_read,
            "cleanup": {
                "managed_by": "RuntimeScheduler.shutdown",
                "delete_attempted": True,
            },
            "initial_states": initial_states,
            "post_step_states": post_states,
            "step_operation": "advance_and_read_body_states",
            "final_simulation_time_ns": target_ns,
        },
        "stage_host": dict(scheduler_diagnostics.get("stage_host") or {}),
        "runtime_composition": {
            "entrypoint": "RuntimeScheduler.tick_viewport",
            "controller": "OvphysxStageController",
            "initial_tick": _tick_result_diagnostics(initial_tick),
            "post_step_tick": _tick_result_diagnostics(post_tick),
            "controller_diagnostics": scheduler_diagnostics,
        },
        "mutations": {
            "initial": _mutation_diagnostics(initial_mutation),
            "post_step": _mutation_diagnostics(post_mutation),
        },
        "render": {
            "session": {
                "simulation_id": render_simulation_id,
                "render_product_path": render_request.render_product_path,
                "width": render_request.width,
                "height": render_request.height,
            },
            "transform_values": [
                {"prim_path": value.prim_path, "matrix": value.matrix}
                for value in transform_values
            ],
            "value_update": value_update,
            "initial_result_read": _render_result_read_diagnostics(
                render_request, initial_render_result
            ),
            "post_step_result_read": _render_result_read_diagnostics(
                render_request, post_step_render_result
            ),
            "initial_image": initial_png,
            "post_step_image": post_png,
        },
        "correctness": {
            "changed_body_paths": changed_paths,
            "moved_down_z": moved_down_z,
            "render_results_differ": render_results_differ,
            "mutation_authority": "OVPhysX",
            "does_not_prove_zero_copy": True,
            "does_not_prove_direct_native_stage_sharing": True,
        },
        "logs": {
            "ovphysx_worker_tail": _read_tail(paths["ovphysx_worker_log"]),
            "ovrtx_worker_tail": _read_tail(paths["ovrtx_worker_log"]),
        },
    }


def _add_ovrtx_client_paths(native_dir: Path) -> None:
    native_path = Path(native_dir).expanduser()
    for path in (native_path, native_path.parent.parent):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _rotate_rgba8_180(payload: bytes, width: int, height: int) -> bytes:
    expected_size = int(width) * int(height) * 4
    if len(payload) != expected_size:
        raise ValueError(f"RGBA payload has {len(payload)} bytes, expected {expected_size}")
    return b"".join(payload[index : index + 4] for index in range(expected_size - 4, -1, -4))


def _physics_config(args: argparse.Namespace, worker_log: Path) -> InteractiveSharedStageConfig:
    return InteractiveSharedStageConfig(
        enabled=True,
        input_usd_path=str(args.fixture),
        server=str(args.server),
        ovphysx_address=args.ovphysx_address,
        ovphysx_worker_command=args.ovphysx_worker_command,
        device=args.device,
        body_root=args.body_root,
        body_prims=tuple(args.body_prims),
        physics_fps=args.fps,
        update_fps=args.fps / max(1, int(args.steps)),
        max_steps=args.steps,
        body_scale=args.body_scale,
        worker_log_path=str(worker_log),
        ovphysx_native_client_module=args.ovphysx_native_client_module,
        ovphysx_native_client_path=str(args.ovphysx_native_client_path),
    )


def _raise_if_failed_tick(result: RuntimeTickResult, label: str) -> None:
    if result.status not in {RuntimeTickStatus.FAILED, RuntimeTickStatus.BUSY}:
        return
    reason = result.skipped_reason or str(result.update.get("skipped_reason", ""))
    raise RuntimeError(f"{label} failed: {result.status.value} {reason}".strip())


def _stage_pose_states(poses: Sequence[BodyPose], simulation_time_ns: int) -> list[dict[str, Any]]:
    return [
        {
            "prim_path": pose.prim_path,
            "simulation_time_ns": int(simulation_time_ns),
            "translate": {
                "found": True,
                "x": float(pose.translate[0]),
                "y": float(pose.translate[1]),
                "z": float(pose.translate[2]),
            },
            "orient": {
                "found": True,
                "i": float(pose.orient[0]),
                "j": float(pose.orient[1]),
                "k": float(pose.orient[2]),
                "r": float(pose.orient[3]),
            },
        }
        for pose in poses
    ]


def _tick_result_diagnostics(result: RuntimeTickResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "enabled": result.enabled,
        "timeline_reset": result.timeline_reset,
        "stage_changed": result.stage_changed,
        "values_written": result.values_written,
        "should_reset_refinement": result.should_reset_refinement,
        "should_request_redraw": result.should_request_redraw,
        "step_count": result.step_count,
        "simulation_time_ns": result.simulation_time_ns,
        "generation": result.generation,
        "skipped_reason": result.skipped_reason,
        "update": dict(result.update),
    }


def _apply_runtime_roots(args: argparse.Namespace) -> None:
    if args.ovphysx_root is not None:
        os.environ["OVPHYSX_ROOT"] = str(args.ovphysx_root)
    if args.ovruntime_root is not None:
        os.environ["OVRUNTIME_ROOT"] = str(args.ovruntime_root)


def _mutation_diagnostics(mutation: Any) -> dict[str, Any]:
    if isinstance(mutation, Mapping):
        return {
            "authority": str(mutation.get("authority", "")),
            "simulation_time_ns": int(mutation.get("simulation_time_ns", 0)),
            "revision": int(mutation.get("revision", 0)),
            "dirty_paths": list(mutation.get("dirty_paths", ())),
        }
    return {
        "authority": mutation.authority,
        "simulation_time_ns": mutation.simulation_time_ns,
        "revision": mutation.revision,
        "dirty_paths": list(mutation.dirty_paths),
    }


def _render_result(
    render_client: Any,
    simulation_id: str,
    request: RenderRequest,
    *,
    additional_samples: int,
) -> RenderResult:
    return render_client.render_result(
        simulation_id,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=str(
            request.color_presentation.get(
                "render_var",
                color_presentation.RENDER_VAR_LDR_COLOR,
            )
        ),
        additional_samples=max(1, int(additional_samples)),
    )


def _render_result_read_diagnostics(
    request: RenderRequest, render_result: Any
) -> dict[str, Any]:
    presentation = color_presentation.diagnostics_from_request_result(
        request, render_result
    )
    presentation["result_render_var"] = str(
        _value(render_result, "render_var", "")
    )
    return {
        "width": int(_value(render_result, "width", 0)),
        "height": int(_value(render_result, "height", 0)),
        "completed_samples": int(_value(render_result, "completed_samples", 0)),
        "session_completed_samples": int(_value(render_result, "session_completed_samples", 0)),
        "simulation_time_ns": int(_value(render_result, "simulation_time_ns", 0)),
        "render_output_simulation_time_ns": int(_value(render_result, "render_output_simulation_time_ns", 0)),
        "color_presentation": presentation,
    }


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _moved_down_z(initial_states: Sequence[Mapping[str, Any]], post_states: Sequence[Mapping[str, Any]]) -> bool:
    initial_by_path = {str(state.get("prim_path")): state for state in initial_states}
    for state in post_states:
        prim_path = str(state.get("prim_path"))
        initial = initial_by_path.get(prim_path)
        if initial is None:
            continue
        initial_translate = initial.get("translate", {})
        post_translate = state.get("translate", {})
        if (
            isinstance(initial_translate, Mapping)
            and isinstance(post_translate, Mapping)
            and initial_translate.get("found")
            and post_translate.get("found")
            and float(post_translate["z"]) < float(initial_translate["z"])
        ):
            return True
    return False


def _dt_ns(args: argparse.Namespace) -> int:
    return int(1_000_000_000 / args.fps)


def _mapping_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _read_tail(path: Path, limit: int = 4096) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _print_result(path: Path, result: Mapping[str, Any]) -> None:
    payload = {"status": result.get("status"), "result": str(path)}
    if result.get("error"):
        payload["error"] = str(result["error"])
    print(json.dumps(payload, indent=2, sort_keys=True))


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
