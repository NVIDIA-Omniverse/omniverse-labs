#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe same-session OVRTX updates for light value edits."""

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

from ovrtx_probe_support import (  # noqa: E402
    BLENDER_COMMAND,
    default_native_client_path,
    default_worker_command,
    worker_command_for_port,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "out" / "artifacts" / "ovrtx-light-value-probe")
    parser.add_argument("--blender-command", default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument("--native-client-path", type=Path, default=Path(os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path())))
    parser.add_argument("--native-client-module", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE", "ovrtx_bridge_client"))
    parser.add_argument("--fixture-usd", type=Path, default=None)
    parser.add_argument("--render-product-path", default="")
    parser.add_argument("--target-light", default="/World/KeyLight")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--port", type=int, default=0, help="override the worker command port; default uses the command as-is")
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fixture_usd is None:
        fixture = _generated_light_probe_fixture(args.output_dir / "light_value_probe.usda", args.width, args.height)
    else:
        fixture = {
            "usd_path": str(args.fixture_usd),
            "render_product_path": args.render_product_path or "/Render/RGBDCamera",
        }
    worker_command = worker_command_for_port(args.worker_command, args.port) if args.port else args.worker_command
    paths = {
        "result": args.output_dir / "light-value-runtime.json",
        "setup": args.output_dir / "light_value_runtime_setup.py",
        "log": args.output_dir / "blender.log",
        "worker_log": args.output_dir / "worker.log",
    }
    config = {
        "repo": str(REPO),
        "native_client_path": str(args.native_client_path),
        "native_client_module": args.native_client_module,
        "usd_path": fixture["usd_path"],
        "render_product_path": fixture["render_product_path"],
        "target_light": args.target_light,
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
        print(json.dumps({"status": "failed", "result": str(paths["result"]), "blender_log": str(paths["log"]), "error": "Probe did not write light-value-runtime.json."}, indent=2, sort_keys=True))
        return 1
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    print(json.dumps({"status": result.get("status"), "result": str(paths["result"]), "mean_abs_delta": result.get("mean_abs_delta"), "hash_changed": result.get("hash_changed")}, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pass" else 1


def _generated_light_probe_fixture(path: Path, width: int, height: int) -> dict[str, str]:
    render_product_path = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
    path.write_text(
        textwrap.dedent(
            f"""
            #usda 1.0
            (
                defaultPrim = "World"
                metersPerUnit = 1
                upAxis = "Z"
            )

            def Xform "World"
            {{
                def Scope "Looks"
                {{
                    def Material "NeutralGrey"
                    {{
                        token outputs:surface.connect = </World/Looks/NeutralGrey/PreviewSurface.outputs:surface>
                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor = (0.6, 0.6, 0.6)
                            float inputs:roughness = 0.65
                            float inputs:metallic = 0
                            token outputs:surface
                        }}
                    }}
                }}

                def Cube "LitCube" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/NeutralGrey>
                    double size = 1.6
                    double3 xformOp:translate = (0, 0, 0.8)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}

                def Camera "Camera"
                {{
                    double3 xformOp:translate = (0, -5, 1.5)
                    quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                    float focalLength = 42
                    float horizontalAperture = 36
                    float verticalAperture = 20.25
                    float2 clippingRange = (0.05, 100)
                }}

                def DistantLight "KeyLight"
                {{
                    float inputs:intensity = 50000
                    color3f inputs:color = (1, 0.95, 0.9)
                    float inputs:angle = 0.53
                    bool inputs:normalize = false
                }}
            }}

            def "Render"
            {{
                def "OmniverseKit"
                {{
                    def "HydraTextures"
                    {{
                        def RenderProduct "ViewportTexture0"
                        {{
                            rel camera = </World/Camera>
                            token omni:rtx:rendermode = "RealTimePathTracing"
                            token omni:rtx:background:source:type = "color"
                            color3f omni:rtx:background:color = (0, 0, 0)
                            rel orderedVars = </Render/OmniverseKit/HydraTextures/ViewportTexture0/LdrColor>
                            uniform int2 resolution = ({int(width)}, {int(height)})
                            bool omni:rtx:autoExposure:enabled = false
                            bool omni:rtx:rt:ecoMode:enabled = false
                            def RenderVar "LdrColor"
                            {{
                                uniform string sourceName = "LdrColor"
                            }}
                        }}
                    }}
                }}
            }}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return {"usd_path": str(path), "render_product_path": render_product_path}


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

        from ovrtx_blender_example import color_presentation, ovrtx_session
        from ovrtx_blender_example.ovrtx_runtime_client import OvrtxRuntimeClient
        from ovrtx_blender_example.ovrtx_value_updates import OvrtxAttributeValue
        from ovrtx_blender_example.render_requests import RenderRequest


        def _mean_rgb(frame):
            data = frame.rgba8
            pixel_count = max(1, len(data) // 4)
            return [
                sum(data[index] for index in range(channel, len(data), 4)) / pixel_count
                for channel in range(3)
            ]


        def _sha(frame):
            return hashlib.sha256(frame.rgba8).hexdigest()


        client = OvrtxRuntimeClient(
            worker_command=CONFIG["worker_command"],
            native_client_module=CONFIG["native_client_module"],
        )
        result = {{
            "schema_version": 1,
            "artifact_id": "ovrtx-light-value-runtime-probe",
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
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                worker_command=CONFIG["worker_command"],
                native_client_module=CONFIG["native_client_module"],
                color_presentation=color_presentation.presentation_from_scene(
                    None,
                    requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
                ),
            )
            simulation_id = client.start_session(ovrtx_session.build_spec(request), simulation_id="ovrtx-light-value-probe")
            baseline = client.render_result(
                simulation_id,
                selected_sensor_paths=request.selected_sensor_paths,
                render_var="LdrColor",
                additional_samples=1,
            )
            requested_light_values = [
                {{
                    "prim_path": CONFIG["target_light"],
                    "attribute": "inputs:intensity",
                    "value": 100000000.0,
                    "value_type": "Float",
                }},
                {{
                    "prim_path": CONFIG["target_light"],
                    "attribute": "inputs:color",
                    "value": [0.05, 0.2, 1.0],
                    "value_type": "Color3f",
                }},
            ]
            light_values = [
                OvrtxAttributeValue(
                    value["prim_path"],
                    value["attribute"],
                    value["value"],
                    value["value_type"],
                )
                for value in requested_light_values
            ]
            update = client.update_attribute_values(simulation_id, light_values)
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
            accepted_count = update.updated_count
            result.update(
                {{
                    "status": "pass" if baseline_sha != updated_sha and mean_abs_delta > 1.0 and accepted_count == len(light_values) else "failed",
                    "requested_light_values": requested_light_values,
                    "accepted_light_value_count": accepted_count,
                    "update": {{
                        **dict(update.diagnostics),
                        "updated_count": update.updated_count,
                        "pending_simulation_time_ns": update.pending_simulation_time_ns,
                    }},
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
