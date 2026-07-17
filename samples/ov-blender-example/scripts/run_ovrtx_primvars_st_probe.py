#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe same-session OVRTX updates for Mesh primvars:st Float2Array values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import textwrap
from typing import Sequence
import zlib


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


TARGET_PRIM = "/World/TexturedQuad"
TARGET_ATTRIBUTE = "primvars:st"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "out" / "artifacts" / "ovrtx-primvars-st-probe")
    parser.add_argument("--blender-command", default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument(
        "--native-client-path",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path())),
    )
    parser.add_argument("--native-client-module", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE", "ovrtx_bridge_client"))
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--port", type=int, default=0, help="override the worker command port; default uses the command as-is")
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = _generated_primvars_st_fixture(args.output_dir / "primvars_st_probe.usda", args.width, args.height)
    worker_command = worker_command_for_port(args.worker_command, args.port) if args.port else args.worker_command
    paths = {
        "result": args.output_dir / "primvars-st-runtime.json",
        "setup": args.output_dir / "primvars_st_runtime_setup.py",
        "log": args.output_dir / "blender.log",
        "worker_log": args.output_dir / "worker.log",
        "baseline_image": args.output_dir / "baseline.png",
        "updated_image": args.output_dir / "updated.png",
    }
    config = {
        "repo": str(REPO),
        "native_client_path": str(args.native_client_path),
        "native_client_module": args.native_client_module,
        "usd_path": fixture["fixture_usd_path"],
        "texture_path": fixture["texture_path"],
        "render_product_path": fixture["render_product_path"],
        "target_prim": TARGET_PRIM,
        "target_attribute": TARGET_ATTRIBUTE,
        "worker_command": worker_command,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "result": str(paths["result"]),
        "worker_log": str(paths["worker_log"]),
        "baseline_image": str(paths["baseline_image"]),
        "updated_image": str(paths["updated_image"]),
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
        print(
            json.dumps(
                {
                    "status": "failed",
                    "result": str(paths["result"]),
                    "blender_log": str(paths["log"]),
                    "returncode": completed.returncode,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return completed.returncode
    if not paths["result"].exists():
        print(
            json.dumps(
                {
                    "status": "failed",
                    "result": str(paths["result"]),
                    "blender_log": str(paths["log"]),
                    "error": "Probe did not write primvars-st-runtime.json.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    flush_diagnostics = _worker_log_flush_diagnostics(paths["worker_log"], TARGET_PRIM, TARGET_ATTRIBUTE)
    result["worker_log_flush_diagnostics"] = flush_diagnostics
    result["renderer_flush_attempted"] = bool(flush_diagnostics["attempted"])
    result["renderer_flush_succeeded"] = bool(flush_diagnostics["succeeded"])
    result["rendered_effect_observed"] = bool(result.get("image_delta_pass", False))
    if result.get("status") == "pass" and not (result["renderer_flush_attempted"] and result["renderer_flush_succeeded"]):
        result["status"] = "failed"
        result["error"] = "Worker log did not record renderer flush attempt and success for primvars:st."
    paths["result"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": result.get("status"),
                "result": str(paths["result"]),
                "mean_abs_rgba_delta": result.get("mean_abs_rgba_delta"),
                "hash_changed": result.get("hash_changed"),
                "renderer_flush_attempted": result.get("renderer_flush_attempted"),
                "renderer_flush_succeeded": result.get("renderer_flush_succeeded"),
                "baseline_image": result.get("baseline_image"),
                "updated_image": result.get("updated_image"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "pass" else 1


def _generated_primvars_st_fixture(path: Path, width: int, height: int) -> dict[str, str]:
    texture_path = path.with_name("uv-quadrants.png")
    _write_quadrant_png(texture_path)
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
                    def Material "UvQuadrants"
                    {{
                        token outputs:surface.connect = </World/Looks/UvQuadrants/PreviewSurface.outputs:surface>

                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor.connect = </World/Looks/UvQuadrants/DiffuseTexture.outputs:rgb>
                            color3f inputs:emissiveColor.connect = </World/Looks/UvQuadrants/DiffuseTexture.outputs:rgb>
                            float inputs:roughness = 0.55
                            float inputs:metallic = 0
                            float inputs:opacity = 1
                            token outputs:surface
                        }}

                        def Shader "UVReader"
                        {{
                            uniform token info:id = "UsdPrimvarReader_float2"
                            string inputs:varname = "st"
                            float2 outputs:result
                        }}

                        def Shader "DiffuseTexture"
                        {{
                            uniform token info:id = "UsdUVTexture"
                            asset inputs:file = @{texture_path.name}@
                            token inputs:sourceColorSpace = "sRGB"
                            float4 inputs:fallback = (1, 0, 1, 1)
                            float2 inputs:st.connect = </World/Looks/UvQuadrants/UVReader.outputs:result>
                            token inputs:wrapS = "repeat"
                            token inputs:wrapT = "repeat"
                            float3 outputs:rgb
                        }}
                    }}
                }}

                def Mesh "TexturedQuad" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/UvQuadrants>
                    point3f[] points = [(-1, 0, 0), (1, 0, 0), (1, 0, 2), (-1, 0, 2)]
                    int[] faceVertexCounts = [4]
                    int[] faceVertexIndices = [0, 1, 2, 3]
                    normal3f[] normals = [(0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0)] (
                        interpolation = "faceVarying"
                    )
                    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
                        interpolation = "faceVarying"
                    )
                }}

                def Camera "Camera"
                {{
                    double3 xformOp:translate = (0, -4, 1)
                    quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                    float focalLength = 42
                    float horizontalAperture = 36
                    float verticalAperture = 20.25
                    float2 clippingRange = (0.05, 100)
                }}

                def DomeLight "AmbientDome"
                {{
                    float inputs:intensity = 900
                    color3f inputs:color = (1, 1, 1)
                }}

                def RectLight "KeyLight"
                {{
                    float inputs:intensity = 9000
                    color3f inputs:color = (1, 0.95, 0.9)
                    float inputs:width = 4
                    float inputs:height = 3
                    bool inputs:normalize = false
                    double3 xformOp:translate = (0, -2.5, 3.0)
                    quatd xformOp:orient = (0.9238795, 0.3826834, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
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
                            token omni:rtx:background:source:type = "sky"
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
    return {"usd_path": str(path), "texture_path": str(texture_path), "render_product_path": render_product_path}


def _write_quadrant_png(path: Path) -> None:
    width = 64
    height = 64
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            if x < width // 2 and y < height // 2:
                color = (240, 25, 25, 255)
            elif x >= width // 2 and y < height // 2:
                color = (25, 220, 70, 255)
            elif x < width // 2:
                color = (30, 80, 245, 255)
            else:
                color = (245, 220, 30, 255)
            if x in (width // 4, width // 2, width * 3 // 4) or y in (height // 4, height // 2, height * 3 // 4):
                color = (8, 8, 8, 255)
            pixels.extend(color)
    _write_rgba_png(path, width, height, bytes(pixels))


def _write_rgba_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA byte count does not match image dimensions")
    rows = b"".join(b"\x00" + rgba[y * width * 4 : (y + 1) * width * 4] for y in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _worker_log_flush_diagnostics(path: Path, prim_path: str, attribute: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    attempted_lines = [
        line
        for line in text.splitlines()
        if "renderer_flush_attempted" in line and prim_path in line and attribute in line
    ]
    succeeded_lines = [
        line
        for line in text.splitlines()
        if "renderer_flush_succeeded" in line and prim_path in line and attribute in line
    ]
    return {
        "worker_log": str(path),
        "attempted": bool(attempted_lines),
        "succeeded": bool(succeeded_lines),
        "attempted_count": len(attempted_lines),
        "succeeded_count": len(succeeded_lines),
        "target_prim": prim_path,
        "target_attribute": attribute,
    }


def _setup_script(config: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""
        import hashlib
        import json
        import os
        from pathlib import Path
        import struct
        import sys
        import time
        import traceback
        import zlib

        CONFIG = json.loads({json.dumps(json.dumps(config, sort_keys=True))})
        sys.path.insert(0, CONFIG["native_client_path"])
        sys.path.insert(0, str(Path(CONFIG["native_client_path"]).parents[1]))
        sys.path.insert(0, str(Path(CONFIG["repo"]) / "addon"))

        from ovrtx_blender_example import color_presentation, ovrtx_session
        from ovrtx_blender_example.ovrtx_runtime_client import OvrtxRuntimeClient
        from ovrtx_blender_example.ovrtx_value_updates import OvrtxAttributeValue
        from ovrtx_blender_example.render_requests import RenderRequest


        def _sha(frame):
            return hashlib.sha256(frame.rgba8).hexdigest()


        def _mean_abs_rgba_delta(left, right):
            pairs = zip(left.rgba8, right.rgba8)
            return sum(abs(int(a) - int(b)) for a, b in pairs) / max(1, min(len(left.rgba8), len(right.rgba8)))


        def _mean_rgb(frame):
            data = frame.rgba8
            pixel_count = max(1, len(data) // 4)
            return [
                sum(data[index] for index in range(channel, len(data), 4)) / pixel_count
                for channel in range(3)
            ]


        def _write_rgba_png(path, frame):
            rgba = frame.rgba8
            width = int(frame.width)
            height = int(frame.height)
            rows = b"".join(
                b"\\x00" + rgba[y * width * 4 : (y + 1) * width * 4]
                for y in range(height)
            )

            def chunk(kind, payload):
                return (
                    struct.pack(">I", len(payload))
                    + kind
                    + payload
                    + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
                )

            png = (
                b"\\x89PNG\\r\\n\\x1a\\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows))
                + chunk(b"IEND", b"")
            )
            Path(path).write_bytes(png)


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


        _sanitize_worker_environment()
        client = OvrtxRuntimeClient(
            worker_command=CONFIG["worker_command"],
            native_client_module=CONFIG["native_client_module"],
        )
        result = {{
            "schema_version": 1,
            "artifact_id": "ovrtx-primvars-st-runtime-probe",
            "status": "running",
            "usd_path": CONFIG["usd_path"],
            "texture_path": CONFIG["texture_path"],
            "render_product_path": CONFIG["render_product_path"],
            "worker_command": CONFIG["worker_command"],
            "worker_log": CONFIG["worker_log"],
            "target_prim": CONFIG["target_prim"],
            "target_attribute": CONFIG["target_attribute"],
            "value_type": "Float2Array",
            "element_count": 4,
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
            simulation_id = client.start_session(ovrtx_session.build_spec(request), simulation_id="ovrtx-primvars-st-probe")
            baseline = client.render_result(
                simulation_id,
                selected_sensor_paths=request.selected_sensor_paths,
                render_var="LdrColor",
                additional_samples=int(CONFIG["samples"]),
            )
            uv_values = [(0.125, 0.125), (0.125, 0.125), (0.125, 0.125), (0.125, 0.125)]
            update = client.update_attribute_values(
                simulation_id,
                [
                    OvrtxAttributeValue(
                        CONFIG["target_prim"],
                        CONFIG["target_attribute"],
                        uv_values,
                        "Float2Array",
                    )
                ],
            )
            updated = client.render_result(
                simulation_id,
                selected_sensor_paths=request.selected_sensor_paths,
                render_var="LdrColor",
                additional_samples=int(CONFIG["samples"]),
            )
            baseline_sha = _sha(baseline)
            updated_sha = _sha(updated)
            _write_rgba_png(CONFIG["baseline_image"], baseline)
            _write_rgba_png(CONFIG["updated_image"], updated)
            mean_abs_rgba_delta = _mean_abs_rgba_delta(baseline, updated)
            accepted_count = update.updated_count
            image_delta_pass = baseline_sha != updated_sha and mean_abs_rgba_delta > 4.0
            result.update(
                {{
                    "status": "pass" if image_delta_pass and accepted_count == 1 else "failed",
                    "requested_attribute_values": [
                        {{
                            "prim_path": CONFIG["target_prim"],
                            "attribute": CONFIG["target_attribute"],
                            "value_type": "Float2Array",
                            "value": uv_values,
                        }}
                    ],
                    "update": {{
                        **dict(update.diagnostics),
                        "updated_count": update.updated_count,
                        "pending_simulation_time_ns": update.pending_simulation_time_ns,
                    }},
                    "image_delta_pass": image_delta_pass,
                    "baseline_sha256": baseline_sha,
                    "updated_sha256": updated_sha,
                    "hash_changed": baseline_sha != updated_sha,
                    "baseline_mean_rgb": _mean_rgb(baseline),
                    "updated_mean_rgb": _mean_rgb(updated),
                    "mean_abs_rgba_delta": mean_abs_rgba_delta,
                    "baseline_image": CONFIG["baseline_image"],
                    "updated_image": CONFIG["updated_image"],
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
