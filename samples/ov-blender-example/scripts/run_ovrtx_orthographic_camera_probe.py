#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe OVRTX rendering of a normal USD orthographic camera."""

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


DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 180
DEFAULT_ORTHOGRAPHIC_FIXTURE_ROOT = REPO / "tests" / "fixtures" / "orthographic_camera_probe"
ORTHOGRAPHIC_FIXTURE_RENDER_PRODUCT_PATH = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
PROJECTIONS = ("perspective", "orthographic")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "out" / "artifacts" / "ovrtx-orthographic-camera-probe",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_ORTHOGRAPHIC_FIXTURE_ROOT,
        help="durable 320x180 USD test fixture corpus to render; nonstandard dimensions generate output fixtures",
    )
    parser.add_argument(
        "--generate-output-fixtures",
        action="store_true",
        help="generate USD fixtures under --output-dir instead of using the durable fixture corpus",
    )
    parser.add_argument("--blender-command", default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND))
    parser.add_argument("--worker-command", default=os.environ.get("OV_BLENDER_EXAMPLE_WORKER_COMMAND", default_worker_command()))
    parser.add_argument(
        "--native-client-path",
        type=Path,
        default=Path(os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", default_native_client_path())),
    )
    parser.add_argument("--native-client-module", default=os.environ.get("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE", "ovrtx_bridge_client"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--camera-profile",
        choices=sorted(_CAMERA_PROFILES),
        default="baseline",
        help="camera attribute profile to test",
    )
    parser.add_argument("--port", type=int, default=0, help="override the worker command port; default uses the command as-is")
    parser.add_argument("--active-cuda-gpus", default=os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", ""))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_source = "durable-fixture-root"
    if args.generate_output_fixtures or args.width != DEFAULT_WIDTH or args.height != DEFAULT_HEIGHT:
        fixture_source = "generated-output"
        fixture = _generated_orthographic_camera_fixture(
            args.output_dir / "orthographic_camera_probe.usda",
            args.width,
            args.height,
            args.camera_profile,
        )
    else:
        try:
            fixture = _orthographic_camera_fixture_from_root(args.fixture_root, args.camera_profile)
        except FileNotFoundError as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "classification": "missing-durable-fixture",
                        "fixture_root": str(args.fixture_root),
                        "camera_profile": args.camera_profile,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
    worker_command = worker_command_for_port(args.worker_command, args.port) if args.port else args.worker_command
    paths = {
        "result": args.output_dir / "orthographic-camera-runtime.json",
        "setup": args.output_dir / "orthographic_camera_probe_setup.py",
        "log": args.output_dir / "blender.log",
        "worker_log": args.output_dir / "worker.log",
    }
    config = {
        "repo": str(REPO),
        "native_client_path": str(args.native_client_path),
        "native_client_module": args.native_client_module,
        "perspective_usd_path": fixture["perspective_usd_path"],
        "orthographic_usd_path": fixture["orthographic_usd_path"],
        "fixture_source": fixture_source,
        "fixture_root": str(args.fixture_root),
        "render_product_path": fixture["render_product_path"],
        "perspective_camera_prim_path": fixture["perspective_camera_prim_path"],
        "orthographic_camera_prim_path": fixture["orthographic_camera_prim_path"],
        "orthographic_overlay_projection": _orthographic_overlay_projection(args.camera_profile),
        "worker_command": worker_command,
        "camera_profile": args.camera_profile,
        "width": args.width,
        "height": args.height,
        "samples": max(1, int(args.samples)),
        "result": str(paths["result"]),
        "worker_log": str(paths["worker_log"]),
        "work_dir": str(args.output_dir / "temporary-usd-layers"),
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
                    "error": "Probe did not write orthographic-camera-runtime.json.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "classification": result.get("classification"),
                "camera_profile": result.get("camera_profile"),
                "fixture_source": result.get("fixture_source"),
                "result": str(paths["result"]),
                "hash_changed": result.get("hash_changed"),
                "orthographic_equal_scale": result.get("orthographic_equal_scale"),
                "overlay_orthographic_equal_scale": result.get("overlay_orthographic_equal_scale"),
                "overlay_matches_durable_orthographic": result.get("overlay_matches_durable_orthographic"),
                "overlay_projection_generated": result.get("overlay_projection_generated"),
                "orthographic_width_ratio": result.get("orthographic_width_ratio"),
                "orthographic_overlay_width_ratio": result.get("orthographic_overlay_width_ratio"),
                "perspective_depth_scale": result.get("perspective_depth_scale"),
                "perspective_width_ratio": result.get("perspective_width_ratio"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "pass" else 1


def _generated_orthographic_camera_fixture(
    path: Path,
    width: int,
    height: int,
    camera_profile: str = "baseline",
) -> dict[str, str]:
    return _generated_orthographic_camera_fixture_with_profile(path, width, height, camera_profile)


def _orthographic_camera_fixture_from_root(fixture_root: Path, camera_profile: str) -> dict[str, str]:
    if camera_profile not in _CAMERA_PROFILES:
        raise ValueError(f"unknown camera profile: {camera_profile}")
    profile_dir = fixture_root / f"profile-{camera_profile}"
    perspective_path = profile_dir / "perspective.usda"
    orthographic_path = profile_dir / "orthographic.usda"
    missing = [path for path in (perspective_path, orthographic_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing durable orthographic fixture(s): " + ", ".join(str(path) for path in missing))
    camera_prim_path = _camera_prim_path_for_profile(camera_profile)
    return {
        "usd_path": str(perspective_path),
        "perspective_usd_path": str(perspective_path),
        "orthographic_usd_path": str(orthographic_path),
        "render_product_path": ORTHOGRAPHIC_FIXTURE_RENDER_PRODUCT_PATH,
        "perspective_render_product_path": ORTHOGRAPHIC_FIXTURE_RENDER_PRODUCT_PATH,
        "orthographic_render_product_path": ORTHOGRAPHIC_FIXTURE_RENDER_PRODUCT_PATH,
        "perspective_camera_prim_path": camera_prim_path,
        "orthographic_camera_prim_path": camera_prim_path,
    }


def _camera_prim_path_for_profile(camera_profile: str) -> str:
    return "/Camera" if _CAMERA_PROFILES[camera_profile].get("placement") == "root-matrix" else "/World/Camera"


def _orthographic_overlay_projection(camera_profile: str) -> dict[str, object]:
    profile = _CAMERA_PROFILES[camera_profile]
    return {
        "projection": "orthographic",
        "focalLength": profile["focal_length"],
        "horizontalAperture": profile["horizontal_aperture"],
        "verticalAperture": profile["vertical_aperture"],
        "clippingRange": [profile["clip_start"], profile["clip_end"]],
        "fStop": 0.0,
    }


_CAMERA_PROFILES = {
    "baseline": {
        "focal_length": 35.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 20.25,
        "clip_start": 0.05,
        "clip_end": 100.0,
    },
    "wide-clip": {
        "focal_length": 35.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 20.25,
        "clip_start": 1.0,
        "clip_end": 100000.0,
    },
    "tiny-near-wide-far": {
        "focal_length": 35.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 20.25,
        "clip_start": 0.001,
        "clip_end": 100000.0,
    },
    "large-aperture": {
        "focal_length": 35.0,
        "horizontal_aperture": 72.0,
        "vertical_aperture": 40.5,
        "clip_start": 0.05,
        "clip_end": 100.0,
    },
    "small-aperture": {
        "focal_length": 35.0,
        "horizontal_aperture": 12.0,
        "vertical_aperture": 6.75,
        "clip_start": 0.05,
        "clip_end": 100.0,
    },
    "root-matrix": {
        "focal_length": 35.0,
        "horizontal_aperture": 36.0,
        "vertical_aperture": 20.25,
        "clip_start": 0.05,
        "clip_end": 100.0,
        "placement": "root-matrix",
    },
}


def _generated_orthographic_camera_fixture_with_profile(
    path: Path,
    width: int,
    height: int,
    camera_profile: str,
) -> dict[str, str]:
    if camera_profile not in _CAMERA_PROFILES:
        raise ValueError(f"unknown camera profile: {camera_profile}")
    profile = _CAMERA_PROFILES[camera_profile]
    render_product_path = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
    text = textwrap.dedent(
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
                    def Material "NearRed"
                    {{
                        token outputs:surface.connect = </World/Looks/NearRed/PreviewSurface.outputs:surface>
                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor = (1, 0.02, 0.02)
                            float inputs:roughness = 0.35
                            token outputs:surface
                        }}
                    }}
                    def Material "FarBlue"
                    {{
                        token outputs:surface.connect = </World/Looks/FarBlue/PreviewSurface.outputs:surface>
                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor = (0.02, 0.12, 1)
                            float inputs:roughness = 0.35
                            token outputs:surface
                        }}
                    }}
                    def Material "CenterGreen"
                    {{
                        token outputs:surface.connect = </World/Looks/CenterGreen/PreviewSurface.outputs:surface>
                        def Shader "PreviewSurface"
                        {{
                            uniform token info:id = "UsdPreviewSurface"
                            color3f inputs:diffuseColor = (0.05, 0.9, 0.2)
                            float inputs:roughness = 0.35
                            token outputs:surface
                        }}
                    }}
                }}

                def Cube "NearRedCube" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/NearRed>
                    double size = 1.0
                    double3 xformOp:translate = (-1.25, -2.0, 0.6)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}

                def Cube "CenterGreenCube" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/CenterGreen>
                    double size = 1.0
                    double3 xformOp:translate = (0, 0, 0.6)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}

                def Cube "FarBlueCube" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/Looks/FarBlue>
                    double size = 1.0
                    double3 xformOp:translate = (1.25, 2.0, 0.6)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}

                def Camera "PerspectiveCamera"
                {{
                    double3 xformOp:translate = (0, -7.0, 1.4)
                    quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                    token projection = "perspective"
                    float focalLength = {profile["focal_length"]}
                    float horizontalAperture = {profile["horizontal_aperture"]}
                    float verticalAperture = {profile["vertical_aperture"]}
                    float2 clippingRange = ({profile["clip_start"]}, {profile["clip_end"]})
                    float fStop = 0
                }}

                def Camera "OrthographicCamera"
                {{
                    double3 xformOp:translate = (0, -7.0, 1.4)
                    quatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
                    token projection = "orthographic"
                    float focalLength = {profile["focal_length"]}
                    float horizontalAperture = {profile["horizontal_aperture"]}
                    float verticalAperture = {profile["vertical_aperture"]}
                    float2 clippingRange = ({profile["clip_start"]}, {profile["clip_end"]})
                    float fStop = 0
                }}

                def DistantLight "KeyLight"
                {{
                    float inputs:intensity = 50000
                    color3f inputs:color = (1, 0.96, 0.9)
                    float inputs:angle = 0.35
                }}

                def DomeLight "DomeLight"
                {{
                    float inputs:intensity = 1000
                    color3f inputs:color = (1, 1, 1)
                }}
            }}

            def "Render"
            {{
                def "OmniverseKit"
                {{
                    def "HydraTextures"
                    {{
                        def RenderProduct "PerspectiveTexture0"
                        {{
                            rel camera = </World/PerspectiveCamera>
                            token omni:rtx:rendermode = "RealTimePathTracing"
                            token omni:rtx:background:source:type = "color"
                            color3f omni:rtx:background:color = (0, 0, 0)
                            rel orderedVars = </Render/OmniverseKit/HydraTextures/PerspectiveTexture0/LdrColor>
                            uniform int2 resolution = ({int(width)}, {int(height)})
                            bool omni:rtx:autoExposure:enabled = false
                            bool omni:rtx:rt:ecoMode:enabled = false
                            def RenderVar "LdrColor"
                            {{
                                uniform string sourceName = "LdrColor"
                            }}
                        }}

                        def RenderProduct "OrthographicTexture0"
                        {{
                            rel camera = </World/OrthographicCamera>
                            token omni:rtx:rendermode = "RealTimePathTracing"
                            token omni:rtx:background:source:type = "color"
                            color3f omni:rtx:background:color = (0, 0, 0)
                            rel orderedVars = </Render/OmniverseKit/HydraTextures/OrthographicTexture0/LdrColor>
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
    ).lstrip()
    perspective_text = text.replace(
        'def RenderProduct "PerspectiveTexture0"',
        'def RenderProduct "ViewportTexture0"',
        1,
    )
    perspective_text = perspective_text.replace(
        'def Camera "PerspectiveCamera"',
        'def Camera "Camera"',
        1,
    ).replace(
        "rel camera = </World/PerspectiveCamera>",
        "rel camera = </World/Camera>",
        1,
    )
    orthographic_text = perspective_text.replace(
        'token projection = "perspective"',
        'token projection = "orthographic"',
        1,
    )
    camera_prim_path = "/World/Camera"
    if profile.get("placement") == "root-matrix":
        perspective_text = _with_root_matrix_camera(perspective_text, profile, "perspective")
        perspective_text = perspective_text.replace("rel camera = </World/Camera>", "rel camera = </Camera>", 1)
        orthographic_text = _with_root_matrix_camera(orthographic_text, profile, "orthographic")
        orthographic_text = orthographic_text.replace("rel camera = </World/Camera>", "rel camera = </Camera>", 1)
        camera_prim_path = "/Camera"
    orthographic_path = path.with_name(f"{path.stem}_orthographic{path.suffix}")
    path.write_text(perspective_text, encoding="utf-8")
    orthographic_path.write_text(orthographic_text, encoding="utf-8")
    return {
        "usd_path": str(path),
        "perspective_usd_path": str(path),
        "orthographic_usd_path": str(orthographic_path),
        "render_product_path": render_product_path,
        "perspective_render_product_path": render_product_path,
        "orthographic_render_product_path": render_product_path,
        "perspective_camera_prim_path": camera_prim_path,
        "orthographic_camera_prim_path": camera_prim_path,
    }


def _with_root_matrix_camera(text: str, profile: dict[str, object], projection: str) -> str:
    camera = textwrap.dedent(
        f"""
        def Camera "Camera"
        {{
            token projection = "{projection}"
            float focalLength = {profile["focal_length"]}
            float horizontalAperture = {profile["horizontal_aperture"]}
            float verticalAperture = {profile["vertical_aperture"]}
            float2 clippingRange = ({profile["clip_start"]}, {profile["clip_end"]})
            float fStop = 0
            matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, -7, 1.4, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}

        """
    ).lstrip()
    return text.replace('def Xform "World"', f"{camera}def Xform \"World\"", 1)


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
        os.environ["OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR"] = CONFIG["work_dir"]

        from ovrtx_blender_example import color_presentation
        from ovrtx_blender_example import ovrtx_scene_composition, ovrtx_session
        from ovrtx_blender_example.render_requests import ACTIVE_CAMERA_VIEW, CameraProjectionState, RenderRequest
        from ovrtx_blender_example.ovrtx_runtime_client import OvrtxRuntimeClient


        def _sha(frame):
            return hashlib.sha256(frame.rgba8).hexdigest()


        def _mean_rgb(frame):
            data = frame.rgba8
            pixel_count = max(1, len(data) // 4)
            return [
                sum(data[index] for index in range(channel, len(data), 4)) / pixel_count
                for channel in range(3)
            ]


        def _frame_stats(frame):
            data = frame.rgba8
            pixel_count = max(1, len(data) // 4)
            nonblack = 0
            min_x = frame.width
            min_y = frame.height
            max_x = -1
            max_y = -1
            for y in range(frame.height):
                row = y * frame.width * 4
                for x in range(frame.width):
                    offset = row + x * 4
                    if not (data[offset] > 12 or data[offset + 1] > 12 or data[offset + 2] > 12):
                        continue
                    nonblack += 1
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
            nonblack_bbox = {{"available": False, "pixel_count": 0}}
            if nonblack > 0:
                nonblack_bbox = {{
                    "available": True,
                    "x_min": min_x,
                    "x_max": max_x,
                    "y_min": min_y,
                    "y_max": max_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "pixel_count": nonblack,
                }}
            return {{
                "width": frame.width,
                "height": frame.height,
                "sha256": _sha(frame),
                "mean_rgb": _mean_rgb(frame),
                "nonblack_pixel_count": nonblack,
                "nonblack_pixel_ratio": nonblack / pixel_count,
                "nonblack_bbox": nonblack_bbox,
            }}


        def _color_bbox(frame, color):
            data = frame.rgba8
            min_x = frame.width
            min_y = frame.height
            max_x = -1
            max_y = -1
            count = 0
            for y in range(frame.height):
                row = y * frame.width * 4
                for x in range(frame.width):
                    offset = row + x * 4
                    red = data[offset]
                    green = data[offset + 1]
                    blue = data[offset + 2]
                    if color == "red":
                        matched = red > 45 and red > green * 1.35 and red > blue * 1.35
                    else:
                        matched = blue > 45 and blue > red * 1.2 and blue > green * 1.2
                    if not matched:
                        continue
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    count += 1
            if count <= 0:
                return {{
                    "available": False,
                    "pixel_count": 0,
                }}
            return {{
                "available": True,
                "x_min": min_x,
                "x_max": max_x,
                "y_min": min_y,
                "y_max": max_y,
                "width": max_x - min_x + 1,
                "height": max_y - min_y + 1,
                "pixel_count": count,
            }}


        def _write_frame_ppm(frame, path):
            data = frame.rgba8
            rgb = bytearray(frame.width * frame.height * 3)
            output_offset = 0
            for offset in range(0, len(data), 4):
                rgb[output_offset] = data[offset]
                rgb[output_offset + 1] = data[offset + 1]
                rgb[output_offset + 2] = data[offset + 2]
                output_offset += 3
            header = ("P6\\n" + str(frame.width) + " " + str(frame.height) + "\\n255\\n").encode("ascii")
            Path(path).write_bytes(header + bytes(rgb))


        def _width_ratio(render):
            red = render["red_bbox"]
            blue = render["blue_bbox"]
            if not red.get("available") or not blue.get("available"):
                return 0.0
            return red["width"] / max(1, blue["width"])


        def _camera_projection_state(projection):
            if not projection:
                return None
            clipping_range = projection.get("clippingRange")
            return CameraProjectionState(
                source=ACTIVE_CAMERA_VIEW,
                projection=str(projection["projection"]),
                focal_length=float(projection["focalLength"]),
                horizontal_aperture=float(projection["horizontalAperture"]),
                vertical_aperture=float(projection["verticalAperture"]),
                clipping_range=tuple(clipping_range) if clipping_range else None,
                f_stop=float(projection.get("fStop", 0.0) or 0.0),
                render_size=(int(CONFIG["width"]), int(CONFIG["height"])),
            )


        def _projection_overlay_generated(render):
            composition = render.get("composition", {{}})
            return any(
                record.get("source") == "viewport_camera_projection" and record.get("generated")
                for record in composition.get("presentation_layers", [])
            )


        def _render_camera(label, usd_path, camera_prim_path, projection=None):
            request = RenderRequest(
                input_usd_path=usd_path,
                sensor_paths=(CONFIG["render_product_path"],),
                selected_sensor_paths=(CONFIG["render_product_path"],),
                width=int(CONFIG["width"]),
                height=int(CONFIG["height"]),
                min_samples=int(CONFIG["samples"]),
                max_samples=int(CONFIG["samples"]),
                camera_prim_path=camera_prim_path,
                camera_projection=_camera_projection_state(projection),
                worker_command=CONFIG["worker_command"],
                native_client_module=CONFIG["native_client_module"],
                color_presentation=color_presentation.presentation_from_scene(
                    None,
                    requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
                ),
            )
            spec = ovrtx_session.build_spec(request)
            client = OvrtxRuntimeClient(
                worker_command=request.worker_command,
                native_client_module=request.native_client_module,
            )
            try:
                simulation_id = client.start_session(
                    spec,
                    simulation_id=f"ovrtx-orthographic-camera-probe-{{label}}",
                )
                frame = client.render_result(
                    simulation_id,
                    selected_sensor_paths=request.selected_sensor_paths,
                    render_var=str(
                        request.color_presentation.get(
                            "render_var",
                            color_presentation.RENDER_VAR_LDR_COLOR,
                        )
                    ),
                    additional_samples=max(1, int(request.max_samples)),
                )
                frame_path = Path(CONFIG["result"]).with_name(label + ".ppm")
                _write_frame_ppm(frame, frame_path)
                return {{
                    "label": label,
                    "session_simulation_id": simulation_id,
                    "sensor_paths": list(request.sensor_paths),
                    "selected_sensor_paths": list(request.selected_sensor_paths),
                    "render_product_path": CONFIG["render_product_path"],
                    "usd_path": usd_path,
                    "camera_prim_path": camera_prim_path,
                    "composition": ovrtx_scene_composition.diagnostics(
                        spec.ovrtx_scene_composition,
                        request=request,
                    ),
                    "render_acquisition": {{
                        "requested_samples": int(request.max_samples),
                        "native_samples": frame.completed_samples,
                    }},
                    "frame_path": str(frame_path),
                    "frame": _frame_stats(frame),
                    "red_bbox": _color_bbox(frame, "red"),
                    "blue_bbox": _color_bbox(frame, "blue"),
                    "completed_samples": frame.completed_samples,
                    "session_completed_samples": frame.session_completed_samples,
                    "simulation_time_ns": frame.simulation_time_ns,
                }}
            finally:
                client.shutdown()


        result = {{
            "schema_version": 1,
            "artifact_id": "ovrtx-orthographic-camera-probe",
            "status": "running",
            "classification": "running",
            "perspective_usd_path": CONFIG["perspective_usd_path"],
            "orthographic_usd_path": CONFIG["orthographic_usd_path"],
            "fixture_source": CONFIG.get("fixture_source", "unknown"),
            "fixture_root": CONFIG.get("fixture_root", ""),
            "render_product_path": CONFIG["render_product_path"],
            "worker_command": CONFIG["worker_command"],
            "worker_log": CONFIG["worker_log"],
            "camera_profile": CONFIG.get("camera_profile", "baseline"),
            "width": int(CONFIG["width"]),
            "height": int(CONFIG["height"]),
            "samples": int(CONFIG["samples"]),
            "started_at_ns": time.time_ns(),
        }}
        try:
            perspective = _render_camera(
                "perspective",
                CONFIG["perspective_usd_path"],
                CONFIG["perspective_camera_prim_path"],
            )
            orthographic = _render_camera(
                "orthographic",
                CONFIG["orthographic_usd_path"],
                CONFIG["orthographic_camera_prim_path"],
            )
            orthographic_overlay = _render_camera(
                "orthographic-overlay",
                CONFIG["perspective_usd_path"],
                CONFIG["perspective_camera_prim_path"],
                CONFIG["orthographic_overlay_projection"],
            )
            perspective_ratio = _width_ratio(perspective)
            orthographic_ratio = _width_ratio(orthographic)
            overlay_ratio = _width_ratio(orthographic_overlay)
            hash_changed = perspective["frame"]["sha256"] != orthographic["frame"]["sha256"]
            overlay_hash_changed = perspective["frame"]["sha256"] != orthographic_overlay["frame"]["sha256"]
            both_nonblank = (
                perspective["frame"]["nonblack_pixel_ratio"] > 0.0005
                and orthographic["frame"]["nonblack_pixel_ratio"] > 0.0005
                and orthographic_overlay["frame"]["nonblack_pixel_ratio"] > 0.0005
            )
            boxes_available = all(
                render[color].get("available")
                for render in (perspective, orthographic, orthographic_overlay)
                for color in ("red_bbox", "blue_bbox")
            )
            orthographic_equal_scale = 0.75 <= orthographic_ratio <= 1.35
            overlay_orthographic_equal_scale = 0.75 <= overlay_ratio <= 1.35
            overlay_matches_durable_orthographic = abs(overlay_ratio - orthographic_ratio) <= 0.15
            overlay_projection_generated = _projection_overlay_generated(orthographic_overlay)
            perspective_depth_scale = perspective_ratio >= max(1.2, orthographic_ratio + 0.2)
            passed = (
                both_nonblank
                and boxes_available
                and hash_changed
                and overlay_hash_changed
                and orthographic_equal_scale
                and overlay_orthographic_equal_scale
                and overlay_matches_durable_orthographic
                and overlay_projection_generated
                and perspective_depth_scale
            )
            result.update(
                {{
                    "status": "pass" if passed else "failed",
                    "classification": "orthographic-supported" if passed else "blocked-runtime-contract",
                    "hash_changed": hash_changed,
                    "both_nonblank": both_nonblank,
                    "boxes_available": boxes_available,
                    "orthographic_equal_scale": orthographic_equal_scale,
                    "overlay_orthographic_equal_scale": overlay_orthographic_equal_scale,
                    "overlay_matches_durable_orthographic": overlay_matches_durable_orthographic,
                    "overlay_projection_generated": overlay_projection_generated,
                    "perspective_depth_scale": perspective_depth_scale,
                    "orthographic_width_ratio": orthographic_ratio,
                    "orthographic_overlay_width_ratio": overlay_ratio,
                    "perspective_width_ratio": perspective_ratio,
                    "perspective": perspective,
                    "orthographic": orthographic,
                    "orthographic_overlay": orthographic_overlay,
                }}
            )
        except BaseException as exc:
            error_message = str(exc)
            classification = (
                "blocked-native-client"
                if type(exc).__name__ == "RenderClientError" and "missing callable" in error_message
                else "blocked-runtime-contract"
            )
            result.update(
                {{
                    "status": "failed",
                    "classification": classification,
                    "error": error_message,
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }}
            )
        finally:
            result["completed_at_ns"] = time.time_ns()
        Path(CONFIG["result"]).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        """
    ).lstrip()


if __name__ == "__main__":
    raise SystemExit(main())
