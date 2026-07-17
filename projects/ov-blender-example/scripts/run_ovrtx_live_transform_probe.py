#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe stock Blender transform edits through the live OVRTX update path."""

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

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
ADDON_DIR = SCRIPTS_DIR.parent / "addon"
if str(ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(ADDON_DIR))

from ovrtx_probe_support import (  # noqa: E402
    BLENDER_COMMAND,
    DEFAULT_FIXTURE_MANIFEST,
    ROOT,
    UNKNOWN,
    default_native_client_path,
    default_worker_command,
    native_extension_check,
    read_json_object,
    resolve_executable,
    resolve_fixture,
    write_result,
)
from ovrtx_blender_example import color_presentation, ovrtx_session  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import OvrtxRuntimeClient  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


DEFAULT_TARGET_PRIM = "/World/PhysicsIsland/DynamicBodies/Cube_00"


def _start_probe_session(
    config: Mapping[str, Any],
    *,
    client_factory: Any = OvrtxRuntimeClient,
    simulation_id: str | None = None,
) -> tuple[Any, RenderRequest, str]:
    samples = int(config["samples"])
    request = RenderRequest(
        input_usd_path=config["input_usd_path"],
        sensor_paths=(config["render_product_path"],),
        selected_sensor_paths=(config["render_product_path"],),
        width=int(config["width"]),
        height=int(config["height"]),
        min_samples=samples,
        max_samples=samples,
        camera_prim_path=config["camera_prim_path"],
        worker_command=config["worker_command"],
        native_client_module=config["native_client_module"],
        color_presentation=color_presentation.presentation_from_scene(
            None,
            requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
        ),
    )
    client = client_factory(
        worker_command=request.worker_command,
        native_client_module=request.native_client_module,
    )
    session = client.start_session(
        ovrtx_session.build_spec(request),
        simulation_id=simulation_id
        or "ovrtx-live-transform-probe-" + str(time.time_ns()),
    )
    return client, request, session


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": args.output_dir / "result.json",
        "metrics": args.output_dir / "blender-metrics.json",
        "initial_image": args.output_dir / "initial.png",
        "post_edit_image": args.output_dir / "post-edit.png",
        "blender_log": args.output_dir / "blender.log",
        "worker_log": args.output_dir / "worker.log",
    }
    result = _base_result(args)
    runtime_started = False
    try:
        fixture = resolve_fixture(args.manifest, args.fixture_id)
        blender = resolve_executable(BLENDER_COMMAND)
        missing = _missing_inputs(args, fixture, blender)
        result["fixture"] = fixture
        result["blender"] = {"command": BLENDER_COMMAND, "executable": blender or UNKNOWN}
        if missing:
            result["status"] = "blocked-preflight"
            result["error"] = "Live transform probe prerequisites are missing."
            result["unresolved_values"] = missing
            write_result(paths["result"], result)
            _print_result(result, paths["result"])
            return 2

        paths["metrics"].unlink(missing_ok=True)
        runtime_started = True
        completed = _run_blender(args, fixture, blender, paths)
        result["blender_exit_status"] = completed.returncode
        result["blender_log"] = str(paths["blender_log"])
        result["worker_log"] = str(paths["worker_log"])
        metrics = read_json_object(paths["metrics"])
        if metrics:
            result["probe"] = metrics
        if completed.returncode != 0:
            result["status"] = "failed-real"
            result["error"] = _failure_message(metrics, "Blender live transform probe exited nonzero.")
            write_result(paths["result"], result)
            _print_result(result, paths["result"])
            return 1
        if not metrics:
            result["status"] = "failed-real"
            result["error"] = "Blender exited without writing live transform probe metrics."
            write_result(paths["result"], result)
            _print_result(result, paths["result"])
            return 1
        if metrics.get("status") != "pass":
            result["status"] = "failed-real"
            result["error"] = str(metrics.get("error", "live transform probe failed inside Blender"))
            write_result(paths["result"], result)
            _print_result(result, paths["result"])
            return 1

        result["status"] = "pass-real"
        result["runtime_artifacts_available"] = True
        write_result(paths["result"], result)
        _print_result(result, paths["result"])
        return 0
    except Exception as exc:
        result["status"] = "failed-real" if runtime_started else "blocked-preflight"
        result["error"] = str(exc)
        write_result(paths["result"], result)
        _print_result(result, paths["result"])
        return 1 if runtime_started else 2


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_FIXTURE_MANIFEST", str(DEFAULT_FIXTURE_MANIFEST))),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "artifacts" / "ovrtx-live-transform-probe")
    parser.add_argument("--fixture-id", default="demo_stair_drop_1280x720")
    parser.add_argument("--target-prim", default=DEFAULT_TARGET_PRIM)
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument("--native-client-module", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE", "ovrtx_bridge_client"))
    parser.add_argument(
        "--native-client-path",
        default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path()),
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--translate-y", type=float, default=-1.0)
    args = parser.parse_args(list(argv))
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not args.target_prim.startswith("/"):
        parser.error("--target-prim must be an absolute USD prim path")
    return args


def _base_result(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": "ovrtx-live-transform-probe",
        "status": "running",
        "runtime_artifacts_available": False,
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "arguments": {
            "manifest": str(args.manifest),
            "output_dir": str(args.output_dir),
            "fixture_id": args.fixture_id,
            "target_prim": args.target_prim,
            "active_cuda_gpus": args.active_cuda_gpus,
            "width": args.width,
            "height": args.height,
            "samples": args.samples,
            "translate_y": args.translate_y,
            "native_client_module": args.native_client_module,
            "native_client_path": args.native_client_path,
        },
    }


def _missing_inputs(args: argparse.Namespace, fixture: Mapping[str, str], blender: str) -> list[str]:
    missing: list[str] = []
    if not blender:
        missing.append("Blender executable: ???")
    if fixture["fixture_usd_path"] == UNKNOWN or not Path(fixture["fixture_usd_path"]).expanduser().is_file():
        missing.append("render USD test fixture local path: ???")
    if fixture["render_product_path"] == UNKNOWN:
        missing.append("fixture render product path: ???")
    if fixture["camera_prim_path"] == UNKNOWN:
        missing.append("fixture camera prim path: ???")
    if not args.worker_command.strip():
        missing.append("OVRTX worker command: ???")
    native_path = Path(str(args.native_client_path)).expanduser()
    if not native_path.is_dir():
        missing.append("native OVRTX client path: ???")
    if not args.native_client_module.strip():
        missing.append("native OVRTX client module: ???")
    elif blender and native_path.is_dir():
        try:
            suffix = _blender_extension_suffix(blender)
        except RuntimeError:
            missing.append("Blender native extension ABI: ???")
        else:
            if not native_extension_check(
                native_path,
                args.native_client_module,
                extension_suffix=suffix,
            )["ok"]:
                missing.append("native OVRTX client extension for Blender ABI: ???")
    return missing


def _blender_extension_suffix(blender: str) -> str:
    marker = "OVRTX_EXT_SUFFIX="
    completed = subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-expr",
            "import sysconfig; print('OVRTX_EXT_SUFFIX=' + str(sysconfig.get_config_var('EXT_SUFFIX') or ''))",
        ],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    suffix = next(
        (line[len(marker) :] for line in completed.stdout.splitlines() if line.startswith(marker)),
        "",
    )
    if completed.returncode != 0 or not suffix:
        raise RuntimeError("could not resolve Blender native extension ABI")
    return suffix


def _run_blender(
    args: argparse.Namespace,
    fixture: Mapping[str, str],
    blender: str,
    paths: Mapping[str, Path],
) -> subprocess.CompletedProcess[str]:
    expr = _blender_expr(
        {
            "root": str(ROOT),
            "native_client_path": args.native_client_path,
            "native_client_module": args.native_client_module,
            "input_usd_path": fixture["fixture_usd_path"],
            "render_product_path": fixture["render_product_path"],
            "camera_prim_path": fixture["camera_prim_path"],
            "target_prim": args.target_prim,
            "worker_command": args.worker_command,
            "width": args.width,
            "height": args.height,
            "samples": args.samples,
            "translate_y": args.translate_y,
            "metrics_path": str(paths["metrics"]),
            "initial_image_path": str(paths["initial_image"]),
            "post_edit_image_path": str(paths["post_edit_image"]),
        }
    )
    env = os.environ.copy()
    env["OV_BLENDER_EXAMPLE_WORKER_LOG"] = str(paths["worker_log"])
    if args.active_cuda_gpus:
        env["OVRTX_ACTIVE_CUDA_GPUS"] = args.active_cuda_gpus
    completed = subprocess.run(
        [blender, "--background", "--python-expr", expr],
        cwd=str(ROOT),
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    paths["blender_log"].write_text(completed.stdout, encoding="utf-8")
    return completed


def _blender_expr(config: Mapping[str, Any]) -> str:
    script = r'''
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ADDON_DIR = Path.cwd() / "addon"
if str(ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(ADDON_DIR))

CONFIG = json.loads(__CONFIG_JSON__)
SCRIPTS_DIR = Path(CONFIG["root"]) / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _sanitize_worker_environment():
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not library_path:
        return
    entries = library_path.split(os.pathsep)
    kept = [entry for entry in entries if not entry.startswith("/snap/blender/")]
    if kept == entries:
        return
    if kept:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items() if key != "rgba8"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _render_result_artifact_from_result(request, render_result, path):
    from ovrtx_blender_example import color_presentation
    from ovrtx_blender_example.shared_stage_composition import write_rgba_png

    payload = bytes(render_result.rgba8)
    png = write_rgba_png(Path(path), int(render_result.width), int(render_result.height), payload)
    presentation = color_presentation.diagnostics_from_request_result(
        request, render_result
    )
    presentation["result_render_var"] = str(render_result.render_var)
    return {
        **png,
        "rgba8_sha256": hashlib.sha256(payload).hexdigest(),
        "completed_samples": int(render_result.completed_samples),
        "session_completed_samples": int(render_result.session_completed_samples),
        "simulation_time_ns": int(render_result.simulation_time_ns),
        "color_presentation": presentation,
    }


def _render_result_artifact(render_client, session, request, path):
    from ovrtx_blender_example import color_presentation

    result = render_client.render_result(
        session,
        selected_sensor_paths=request.selected_sensor_paths,
        render_var=str(
            request.color_presentation.get(
                "render_var",
                color_presentation.RENDER_VAR_LDR_COLOR,
            )
        ),
        additional_samples=max(1, int(request.max_samples)),
    )
    return _render_result_artifact_from_result(request, result, path)


def _usd_rows_for_target():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(CONFIG["input_usd_path"]).expanduser().resolve()))
    if stage is None:
        raise RuntimeError("could not open fixture USD stage")
    prim = stage.GetPrimAtPath(CONFIG["target_prim"])
    if not prim or not prim.IsValid():
        raise RuntimeError(f"target prim not found: {CONFIG['target_prim']}")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    rows = [[float(transform[row][column]) for column in range(4)] for row in range(4)]
    rows[3][1] += float(CONFIG["translate_y"])
    return rows


def _blender_rows_from_usd_rows(rows):
    return [[rows[row][column] for row in range(4)] for column in range(4)]


from ovrtx_blender_example.ovrtx_value_updates import OvrtxSessionUpdatePort
from run_ovrtx_live_transform_probe import _start_probe_session


metrics = {
    "schema_version": 1,
    "artifact_id": "ovrtx-live-transform-probe-blender",
    "status": "running",
    "started_at_ns": time.time_ns(),
    "config": {
        key: value for key, value in CONFIG.items()
        if key not in {"worker_command"}
    },
}
render_client = None

try:
    root = Path(CONFIG["root"])
    sys.path.insert(0, str(root / "addon"))
    if CONFIG["native_client_path"]:
        sys.path.insert(0, CONFIG["native_client_path"])
    _sanitize_worker_environment()

    import bpy
    from mathutils import Matrix

    from ovrtx_blender_example import usd_paths as usd_paths
    from ovrtx_blender_example.blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
    from ovrtx_blender_example.interactive_edit_workflow import InteractiveEditWorkflow
    from ovrtx_blender_example.interactive_edit_planner import (
        DataAuthority,
        EditShape,
        edit_location,
        InteractiveEdit,
    )
    from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest

    render_client, render_request, session = _start_probe_session(CONFIG)
    session_id = session
    initial_render_result = _render_result_artifact(render_client, session, render_request, CONFIG["initial_image_path"])

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.object.empty_add(type="CUBE", location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = "OVRTX Live Transform Probe"
    obj[usd_paths.USD_LAYER_ID_PROP] = "/layers/live-transform-probe.usda"
    obj[usd_paths.USD_PRIM_PATH_PROP] = CONFIG["target_prim"]
    obj[usd_paths.BLENDER_PROPERTY_PATH_PROP] = "matrix_world"
    obj[usd_paths.DATA_AUTHORITY_PROP] = "view"

    captured = []

    def _capture_edits(_scene, depsgraph):
        captured.extend(build_interactive_edits_from_depsgraph(depsgraph))

    bpy.app.handlers.depsgraph_update_post.append(_capture_edits)
    try:
        target_usd_rows = _usd_rows_for_target()
        obj.matrix_world = Matrix(_blender_rows_from_usd_rows(target_usd_rows))
        bpy.context.view_layer.update()
    finally:
        try:
            bpy.app.handlers.depsgraph_update_post.remove(_capture_edits)
        except ValueError:
            pass

    edits = [
        edit for edit in captured
        if edit.usd_prim_path == CONFIG["target_prim"] and edit.data_authority.value == "view"
    ]
    if not edits:
        raise RuntimeError("Blender depsgraph did not produce a view value edit for the target prim")
    edit = edits[-1]
    scheduler = RuntimeScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)
    result = workflow.preview_edit(edit)
    tick = scheduler.tick_viewport(
        RuntimeTickRequest(
            input_usd_path=CONFIG["input_usd_path"],
        ),
        ovrtx_updates=OvrtxSessionUpdatePort(render_client, session),
    )
    update_result = dict(tick.update.get("update_result", {}))
    update_result_record_count = workflow.record_update_result(update_result)
    unsupported_result = workflow.preview_edit(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=CONFIG["target_prim"],
                provenance={"source": "live_transform_probe"},
            ),
            value={"label": "unsupported-live-validation"},
        )
    )
    post_edit_render_result = _render_result_artifact(render_client, session, render_request, CONFIG["post_edit_image_path"])
    update_result = scheduler.diagnostics().get("last_edit_update", {})
    workflow_diagnostics = workflow.diagnostics()
    edit_records = workflow_diagnostics.get("edit_records", [])
    correctness = {
        "depsgraph_edit_found": bool(edits),
        "workflow_accepted": bool(result.accepted),
        "workflow_unsupported_recorded": bool(not unsupported_result.accepted),
        "tick_values_written": bool(tick.values_written),
        "update_values_written": bool(update_result.get("values_written", False)),
        "edit_records_written": len(edit_records) >= 2,
        "per_edit_values_written_recorded": any(
            record.get("action") == "update"
            and record.get("result") == "applied"
            and bool(record.get("values_written", False))
            for record in edit_records
        ),
        "per_edit_unsupported_recorded": any(
            record.get("action") == "unsupported"
            and not bool(record.get("accepted", True))
            for record in edit_records
        ),
        "same_ovrtx_session": session == session_id,
        "whole_scene_export_avoided": not bool(result.diagnostics.get("whole_scene_export_requested", True)),
        "render_results_differ": (
            initial_render_result["rgba8_sha256"] != post_edit_render_result["rgba8_sha256"]
        ),
    }
    failures = [name for name, ok in correctness.items() if not ok]
    metrics.update(
        {
            "status": "pass" if not failures else "failed",
            "error": "" if not failures else "live transform correctness checks failed: " + ", ".join(failures),
            "session": _jsonable(session),
            "edit": _jsonable(edit),
            "workflow_result": _jsonable(result),
            "unsupported_workflow_result": _jsonable(unsupported_result),
            "update_result_record_count": update_result_record_count,
            "workflow_diagnostics": _jsonable(workflow_diagnostics),
            "tick_result": _jsonable(tick),
            "runtime_scheduler": _jsonable(scheduler.diagnostics()),
            "initial_render_result": initial_render_result,
            "post_edit_render_result": post_edit_render_result,
            "correctness": correctness,
            "target_usd_rows": target_usd_rows,
        }
    )
except BaseException as exc:
    metrics.update(
        {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    )
finally:
    if render_client is not None:
        try:
            metrics["shutdown"] = _jsonable(render_client.shutdown())
        except BaseException as exc:
            metrics["shutdown_error"] = str(exc)
    metrics["completed_at_ns"] = time.time_ns()
    Path(CONFIG["metrics_path"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if metrics["status"] != "pass":
    raise SystemExit(1)
'''
    return script.replace("__CONFIG_JSON__", repr(json.dumps(config, sort_keys=True)))


def _failure_message(metrics: Mapping[str, Any], fallback: str) -> str:
    if metrics.get("error"):
        return str(metrics["error"])
    return fallback


def _print_result(result: Mapping[str, Any], path: Path) -> None:
    probe = result.get("probe", {})
    correctness = probe.get("correctness", {}) if isinstance(probe, Mapping) else {}
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "result": str(path),
                "render_results_differ": correctness.get("render_results_differ"),
                "update_values_written": correctness.get("update_values_written"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
