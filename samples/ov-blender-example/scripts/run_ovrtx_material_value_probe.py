#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe same-session OVRTX updates for material value edits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_ovrtx_operator_seam_probe import _generated_red_cube_fixture  # noqa: E402
from ovrtx_probe_support import (  # noqa: E402
    BLENDER_COMMAND,
    default_native_client_path,
    default_worker_command,
    worker_command_for_port,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "out" / "artifacts" / "ovrtx-material-value-probe")
    parser.add_argument("--blender-command", default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument("--native-client-path", type=Path, default=Path(os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path())))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--port", type=int, default=0, help="override the worker command port; default uses the command as-is")
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = _generated_red_cube_fixture(args.output_dir / "material_value_probe.usda", args.width, args.height)
    worker_command = worker_command_for_port(args.worker_command, args.port) if args.port else args.worker_command
    paths = {
        "result": args.output_dir / "material-value-runtime.json",
        "setup": args.output_dir / "material_value_runtime_setup.py",
        "log": args.output_dir / "blender.log",
        "worker_log": args.output_dir / "worker.log",
    }
    config = {
        "repo": str(REPO),
        "native_client_path": str(args.native_client_path),
        "usd_path": fixture["fixture_usd_path"],
        "render_product_path": fixture["render_product_path"],
        "worker_command": worker_command,
        "width": args.width,
        "height": args.height,
        "result": str(paths["result"]),
        "worker_log": str(paths["worker_log"]),
    }
    paths["setup"].write_text(_setup_script(config), encoding="utf-8")

    env = os.environ.copy()
    if args.port:
        env["SRTX_SERVER_PORT"] = str(args.port)
    env["OV_BLENDER_EXAMPLE_WORKER_LOG"] = str(paths["worker_log"])
    if args.active_cuda_gpus:
        env["OVRTX_ACTIVE_CUDA_GPUS"] = str(args.active_cuda_gpus)
    completed = subprocess.run(
        [args.blender_command, "--background", "--python", str(paths["setup"])],
        cwd=str(REPO),
        env=env,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    paths["log"].write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        print(json.dumps({"status": "failed", "result": str(paths["result"]), "blender_log": str(paths["log"]), "returncode": completed.returncode}, indent=2, sort_keys=True))
        return completed.returncode
    if not paths["result"].exists():
        print(json.dumps({"status": "failed", "result": str(paths["result"]), "blender_log": str(paths["log"]), "error": "Probe did not write material-value-runtime.json."}, indent=2, sort_keys=True))
        return 1
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    print(json.dumps({"status": result.get("status"), "result": str(paths["result"]), "mean_abs_delta": result.get("mean_abs_delta"), "hash_changed": result.get("hash_changed")}, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pass" else 1


def _setup_script(config: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        from pathlib import Path
        import sys
        import time
        import traceback

        CONFIG = json.loads({json.dumps(json.dumps(config, sort_keys=True))})
        sys.path.insert(0, CONFIG["native_client_path"])
        sys.path.insert(0, str(Path(CONFIG["native_client_path"]).parents[1]))
        sys.path.insert(0, str(Path(CONFIG["repo"]) / "addon"))

        from types import SimpleNamespace

        import bpy
        from ovrtx_blender_example.blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
        from ovrtx_blender_example.interactive_edit_workflow import InteractiveEditWorkflow
        from ovrtx_blender_example import color_presentation, ovrtx_session
        from ovrtx_blender_example.ovrtx_runtime_client import (
            OvrtxRuntimeClient,
        )
        from ovrtx_blender_example.ovrtx_value_updates import OvrtxSessionUpdatePort
        from ovrtx_blender_example.render_requests import RenderRequest
        from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest
        from ovrtx_blender_example.usd_prim_resolver import UsdPrimResolver


        def _mean_rgb(frame):
            data = frame.rgba8
            pixel_count = max(1, len(data) // 4)
            return [
                sum(data[index] for index in range(channel, len(data), 4)) / pixel_count
                for channel in range(3)
            ]


        def _sha(frame):
            return hashlib.sha256(frame.rgba8).hexdigest()


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


        class _DepsgraphUpdate:
            def __init__(self, id_data):
                self.id = id_data


        class _Depsgraph:
            def __init__(self, updates):
                self.updates = updates


        def _set_input(node, name, value):
            socket = node.inputs.get(name)
            if socket is None:
                raise RuntimeError(f"Principled BSDF input {{name!r}} is missing")
            socket.default_value = value


        def _edited_blender_material():
            material = bpy.data.materials.new("TargetRed")
            material.use_nodes = True
            material.diffuse_color = (0.02, 0.08, 1.0, 1.0)
            principled = material.node_tree.nodes.get("Principled BSDF")
            if principled is None:
                raise RuntimeError("Principled BSDF node is missing")
            _set_input(principled, "Base Color", (0.02, 0.08, 1.0, 1.0))
            _set_input(principled, "Roughness", 0.9)
            _set_input(principled, "Metallic", 0.8)
            _set_input(principled, "IOR", 1.45)
            _set_input(principled, "Emission Color", (0.05, 0.1, 0.35, 1.0))
            _set_input(principled, "Emission Strength", 3.0)
            return material


        _sanitize_worker_environment()
        client = OvrtxRuntimeClient(
            worker_command=CONFIG["worker_command"],
            native_client_module="ovrtx_bridge_client",
        )
        result = {{
            "schema_version": 1,
            "artifact_id": "ovrtx-material-value-runtime-probe",
            "status": "running",
            "usd_path": CONFIG["usd_path"],
            "render_product_path": CONFIG["render_product_path"],
            "worker_command": CONFIG["worker_command"],
            "worker_log": CONFIG["worker_log"],
            "started_at_ns": time.time_ns(),
        }}
        try:
            request = RenderRequest(
                input_usd_path=CONFIG["usd_path"],
                sensor_paths=(CONFIG["render_product_path"],),
                selected_sensor_paths=(CONFIG["render_product_path"],),
                worker_command=CONFIG["worker_command"],
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                native_client_module="ovrtx_bridge_client",
                color_presentation=color_presentation.presentation_from_scene(
                    None,
                    requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
                ),
            )
            prim_resolver = UsdPrimResolver()
            prim_resolver.scan(request)
            simulation_id = client.start_session(ovrtx_session.build_spec(request), simulation_id="ovrtx-material-value-probe")
            baseline = client.render_result(
                simulation_id,
                selected_sensor_paths=request.selected_sensor_paths,
                render_var="LdrColor",
                additional_samples=1,
            )
            material = _edited_blender_material()
            edits = build_interactive_edits_from_depsgraph(
                _Depsgraph([_DepsgraphUpdate(material)]),
                usd_prim_resolver=prim_resolver,
            )
            scheduler = RuntimeScheduler(config_factory=lambda _input_usd_path: SimpleNamespace(enabled=False))
            workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)
            results = [workflow.preview_edit(edit) for edit in edits]
            tick = scheduler.tick_viewport(
                RuntimeTickRequest(
                    input_usd_path=CONFIG["usd_path"],
                    timeline_controls_enabled=False,
                ),
                ovrtx_updates=OvrtxSessionUpdatePort(client, simulation_id),
            )
            update = dict(tick.update.get("update_result", {{}}))
            material_values = [
                {{
                    "prim_path": edit.usd_prim_path,
                    "attribute": edit.usd_attribute,
                    "value": edit.value,
                    "value_type": edit.metadata.get("usd_value_type", ""),
                }}
                for edit in edits
            ]
            updated = client.render_result(
                simulation_id,
                selected_sensor_paths=request.selected_sensor_paths,
                render_var="LdrColor",
                additional_samples=1,
            )
            baseline_mean = _mean_rgb(baseline)
            updated_mean = _mean_rgb(updated)
            mean_abs_delta = sum(abs(a - b) for a, b in zip(baseline_mean, updated_mean)) / 3.0
            baseline_sha = _sha(baseline)
            updated_sha = _sha(updated)
            result.update(
                {{
                    "status": "pass" if baseline_sha != updated_sha and mean_abs_delta > 1.0 and tick.values_written else "failed",
                    "requested_material_values": material_values,
                    "adapter_edit_count": len(edits),
                    "workflow_results": [
                        {{
                            "accepted": bool(result.accepted),
                            "action": result.action.value,
                            "reason": result.reason,
                        }}
                        for result in results
                    ],
                    "workflow_diagnostics": workflow.diagnostics(),
                    "scheduler_tick": {{
                        "status": tick.status.value,
                        "values_written": bool(tick.values_written),
                        "should_reset_refinement": bool(tick.should_reset_refinement),
                    }},
                    "update": dict(update),
                    "baseline_sha256": baseline_sha,
                    "updated_sha256": updated_sha,
                    "hash_changed": baseline_sha != updated_sha,
                    "baseline_mean_rgb": baseline_mean,
                    "updated_mean_rgb": updated_mean,
                    "mean_abs_delta": mean_abs_delta,
                    "baseline_completed_samples": baseline.completed_samples,
                    "updated_completed_samples": updated.completed_samples,
                    "baseline_simulation_time_ns": baseline.simulation_time_ns,
                    "updated_simulation_time_ns": updated.simulation_time_ns,
                }}
            )
        except BaseException as exc:
            result.update(
                {{
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }}
            )
        finally:
            try:
                result["shutdown"] = client.shutdown()
            except BaseException as exc:
                result["shutdown_error"] = str(exc)
            result["completed_at_ns"] = time.time_ns()
        Path(CONFIG["result"]).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        """
    ).lstrip()


if __name__ == "__main__":
    raise SystemExit(main())
