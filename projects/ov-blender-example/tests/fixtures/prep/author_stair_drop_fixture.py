#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author the standalone OVRTX + OVPhysX stair-drop demo fixture."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, MutableMapping, Sequence

import download_stair_drop_pbr_assets


FIXTURES_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = FIXTURES_ROOT.parent
REPO_ROOT = TESTS_ROOT.parent
ROOT = TESTS_ROOT
FIXTURE_ID = "demo_stair_drop_1280x720"
DEFAULT_RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
CAMERA_PATH = "/World/Camera"
UNKNOWN = "???"
PBR_TEXTURE_ROOT = "textures/ambientcg"
PBR_TEXTURE_PACKAGE = download_stair_drop_pbr_assets.TEXTURE_PACKAGE
PBR_METALNESS_ASSETS = {"Metal035", "Metal059C", "Concrete042B"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    usd_path = args.fixture_usd_path or (
        ROOT / "fixtures" / "assets" / FIXTURE_ID / "fixture" / "stair_drop_ovrtx_ovphysx.usda"
    )
    result_path = args.result or REPO_ROOT / "out" / "artifacts" / "stair-drop-authoring" / "result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "stair-drop-fixture-authoring",
        "status": "running",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "fixture_id": FIXTURE_ID,
        "manifest": str(args.manifest),
        "fixture_usd_path": str(usd_path),
    }

    try:
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        texture_infos = _pbr_texture_infos(usd_path.parent / PBR_TEXTURE_ROOT)
        usd_path.write_text(_fixture_usda(args.width, args.height), encoding="utf-8")
        sha256 = _sha256(usd_path)
        manifest = _load_manifest(args.manifest)
        fixture = _fixture_record(usd_path, sha256, args.width, args.height, texture_infos)
        _upsert_fixture(manifest, fixture)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

        result.update(
            {
                "status": "pass",
                "fixture_usd_sha256": sha256,
                "dynamic_body_root": "/World/PhysicsIsland/DynamicBodies",
                "dynamic_body_count": 12,
                "texture_count": len(texture_infos),
                "camera_prim_path": CAMERA_PATH,
                "render_product_prim_path": DEFAULT_RENDER_PRODUCT,
            }
        )
        return_code = 0
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
        return_code = 1

    result["completed_at_ns"] = time.time_ns()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path)}, indent=2, sort_keys=True))
    return return_code


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "fixtures" / "manifest.json")
    parser.add_argument("--fixture-usd-path", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


def _fixture_record(
    usd_path: Path,
    sha256: str,
    width: int,
    height: int,
    texture_infos: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    manifest_path = _manifest_path(usd_path)
    asset_files = [
        {
            "kind": "usd",
            "label": "Authored OVRTX/OVPhysX stair-drop USD",
            "render_ready": True,
            "physics_ready": True,
            "availability": "available",
            "local_path": manifest_path,
            "sha256": sha256,
            "generated_by": "tests/fixtures/prep/author_stair_drop_fixture.py",
        }
    ]
    for texture in texture_infos:
        asset = {
            "kind": "texture",
            "label": str(texture["label"]),
            "render_ready": True,
            "availability": "available",
            "local_path": _manifest_path(Path(texture["path"])),
            "sha256": str(texture["sha256"]),
            "generated_by": "tests/fixtures/prep/download_stair_drop_pbr_assets.py",
        }
        for key in ("source_url", "source_license", "license_url"):
            if key in texture:
                asset[key] = str(texture[key])
        asset_files.append(asset)
    return {
        "id": FIXTURE_ID,
        "display_name": "stair-drop-cubes OVRTX + OVPhysX Demo at 1280x720",
        "availability": "available",
        "capabilities": ["ovrtx", "ovphysx"],
        "target_resolution": {"width": width, "height": height},
        "fixture_usd_path": manifest_path,
        "fixture_usd_sha256": sha256,
        "camera_prim_path": CAMERA_PATH,
        "render_product_prim_path": DEFAULT_RENDER_PRODUCT,
        "asset_files": asset_files,
        "unresolved_values": [],
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _upsert_fixture(manifest: MutableMapping[str, Any], fixture: Mapping[str, Any]) -> None:
    fixtures = manifest.setdefault("fixtures", [])
    if not isinstance(fixtures, list):
        raise ValueError("manifest fixtures must be a list")
    for index, existing in enumerate(fixtures):
        if isinstance(existing, Mapping) and existing.get("id") == fixture["id"]:
            fixtures[index] = dict(fixture)
            return
    fixtures.append(dict(fixture))


def _fixture_usda(width: int, height: int) -> str:
    camera_matrix = _look_at_matrix((4.63, 7.56, 4.45), (0.0, -0.45, 1.85))
    return (
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        '    doc = "Standalone stair-drop demo fixture for OVRTX + OVPhysX composition. Twelve exact cube rigid bodies fall onto an explicit staircase and catch tray made only from box colliders."\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        ")\n\n"
        "def Xform \"World\"\n"
        "{\n"
        f"{_physics_scene_block()}"
        f"{_looks_block()}"
        f"{_physics_island_block()}"
        f"{_lighting_block()}"
        f"{_camera_block(camera_matrix)}"
        "}\n\n"
        f"{_render_block(width, height)}"
    )


def _physics_scene_block() -> str:
    return (
        '    def PhysicsScene "PhysicsScene"\n'
        "    {\n"
        "        vector3f physics:gravityDirection = (0, 0, -1)\n"
        "        float physics:gravityMagnitude = 9.81\n"
        "    }\n\n"
    )


def _looks_block() -> str:
    materials = [
        ("StairConcrete", (0.58, 0.58, 0.55), 0.72, 0.0, 1.0, "Concrete016"),
        ("TrayMetal", (0.34, 0.36, 0.38), 0.54, 1.0, 1.0, "Metal031"),
        ("CopperMetal", (0.9, 0.45, 0.18), 0.24, 1.0, 1.0, "Metal035"),
        ("ScarredTitanium", (0.55, 0.54, 0.50), 0.42, 1.0, 1.0, "Metal059C"),
        ("FineWood", (0.54, 0.32, 0.16), 0.48, 0.0, 1.0, "Wood025"),
        ("BlueWovenFabric", (0.05, 0.34, 0.62), 0.78, 0.0, 1.0, "Fabric017"),
        ("RedCarpet", (0.55, 0.08, 0.05), 0.88, 0.0, 1.0, "Fabric026"),
        ("VintageLeather", (0.34, 0.17, 0.08), 0.58, 0.0, 1.0, "Leather014"),
        ("BlueScratchedPlastic", (0.02, 0.16, 0.42), 0.38, 0.0, 1.0, "Plastic001"),
        ("BlackGymRubber", (0.02, 0.02, 0.02), 0.86, 0.0, 1.0, "Rubber004"),
        ("StoneTile", (0.72, 0.52, 0.36), 0.82, 0.0, 1.0, "Tiles027"),
        ("RustMeshConcrete", (0.45, 0.32, 0.24), 0.76, 0.3, 1.0, "Concrete042B"),
        ("TranslucentIce", (0.62, 0.9, 0.95), 0.16, 0.0, 0.48, "Ice001"),
        ("FrostedIce", (0.8, 0.95, 1.0), 0.34, 0.0, 0.58, "Ice003"),
    ]
    lines = ['    def Scope "Looks"\n', "    {\n"]
    for name, color, roughness, metallic, opacity, asset_id in materials:
        lines.append(_material_block(name, color, roughness, metallic, opacity, asset_id))
    lines.append("    }\n\n")
    return "".join(lines)


def _material_block(
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
    opacity: float,
    asset_id: str | None,
) -> str:
    opacity_line = f"                float inputs:opacity = {opacity:.4g}\n" if opacity < 1.0 else ""
    diffuse_line = (
        f"                color3f inputs:diffuseColor.connect = </World/Looks/{name}/ColorTexture.outputs:rgb>\n"
        if asset_id
        else f"                color3f inputs:diffuseColor = {_format_tuple(color)}\n"
    )
    roughness_line = (
        f"                float inputs:roughness.connect = </World/Looks/{name}/RoughnessTexture.outputs:r>\n"
        if asset_id
        else f"                float inputs:roughness = {roughness:.4g}\n"
    )
    metallic_line = (
        f"                float inputs:metallic.connect = </World/Looks/{name}/MetalnessTexture.outputs:r>\n"
        if asset_id in PBR_METALNESS_ASSETS
        else f"                float inputs:metallic = {metallic:.4g}\n"
    )
    normal_line = (
        f"                normal3f inputs:normal.connect = </World/Looks/{name}/NormalTexture.outputs:rgb>\n"
        if asset_id
        else ""
    )
    texture_block = _texture_block(name, color, roughness, metallic, asset_id) if asset_id else ""
    return (
        f'        def Material "{name}"\n'
        "        {\n"
        f"            token outputs:surface.connect = </World/Looks/{name}/PreviewSurface.outputs:surface>\n"
        '            def Shader "PreviewSurface"\n'
        "            {\n"
        '                uniform token info:id = "UsdPreviewSurface"\n'
        f"{diffuse_line}"
        f"{roughness_line}"
        f"{metallic_line}"
        f"{normal_line}"
        f"{opacity_line}"
        "                token outputs:surface\n"
        "            }\n"
        f"{texture_block}"
        "        }\n"
    )


def _texture_block(
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
    asset_id: str,
) -> str:
    texture_shaders = [
        _uv_texture_shader(
            name,
            "ColorTexture",
            _pbr_map_path(asset_id, "Color"),
            "sRGB",
            (color[0], color[1], color[2], 1.0),
            outputs=("rgb", "r", "g", "b", "a"),
        ),
        _uv_texture_shader(
            name,
            "RoughnessTexture",
            _pbr_map_path(asset_id, "Roughness"),
            "raw",
            (roughness, roughness, roughness, 1.0),
            outputs=("r", "rgb"),
        ),
        _uv_texture_shader(
            name,
            "NormalTexture",
            _pbr_map_path(asset_id, "NormalGL"),
            "raw",
            (0.5, 0.5, 1.0, 1.0),
            outputs=("rgb", "r", "g", "b", "a"),
        ),
    ]
    if asset_id in PBR_METALNESS_ASSETS:
        texture_shaders.append(
            _uv_texture_shader(
                name,
                "MetalnessTexture",
                _pbr_map_path(asset_id, "Metalness"),
                "raw",
                (metallic, metallic, metallic, 1.0),
                outputs=("r", "rgb"),
            )
        )
    return (
        '            def Shader "UVReader"\n'
        "            {\n"
        '                uniform token info:id = "UsdPrimvarReader_float2"\n'
        '                string inputs:varname = "st"\n'
        "                float2 outputs:result\n"
        "            }\n"
        + "".join(texture_shaders)
    )


def _uv_texture_shader(
    material_name: str,
    shader_name: str,
    texture_path: str,
    source_color_space: str,
    fallback: tuple[float, float, float, float],
    *,
    outputs: Sequence[str],
) -> str:
    lines = [
        f'            def Shader "{shader_name}"\n',
        "            {\n",
        '                uniform token info:id = "UsdUVTexture"\n',
        f"                asset inputs:file = @{texture_path}@\n",
        f'                token inputs:sourceColorSpace = "{source_color_space}"\n',
        f"                float4 inputs:fallback = {_format_tuple(fallback)}\n",
        f"                float2 inputs:st.connect = </World/Looks/{material_name}/UVReader.outputs:result>\n",
        '                token inputs:wrapS = "repeat"\n',
        '                token inputs:wrapT = "repeat"\n',
    ]
    for output in outputs:
        if output == "rgb":
            lines.append("                float3 outputs:rgb\n")
        else:
            lines.append(f"                float outputs:{output}\n")
    lines.append("            }\n")
    return "".join(lines)


def _pbr_map_path(asset_id: str, map_name: str) -> str:
    return f"{PBR_TEXTURE_ROOT}/{asset_id}/{asset_id}_{PBR_TEXTURE_PACKAGE}_{map_name}.jpg"


def _physics_island_block() -> str:
    lines = ['    def Xform "PhysicsIsland"\n', "    {\n"]
    lines.extend(_stair_blocks())
    lines.extend(_tray_blocks())
    lines.extend(_dynamic_body_blocks())
    lines.append("    }\n\n")
    return "".join(lines)


def _stair_blocks() -> list[str]:
    lines = ['        def Xform "Stairs"\n', "        {\n"]
    lines.append(
        _kinematic_cube_block(
            "StudioFloor",
            "/World/Looks/TrayMetal",
            translate=(0.0, 0.0, -0.05),
            scale=(5.0, 9.0, 0.10),
            indent="            ",
        )
    )
    step_count = 8
    step_depth = 0.76
    step_width = 3.4
    step_height = 0.26
    first_y = -2.45
    top_z = 2.28
    top_stop_depth = 0.16
    top_stop_y = first_y - step_depth * 0.5 - top_stop_depth * 0.5
    for index in range(step_count):
        top = top_z - index * 0.23
        y = first_y + index * (step_depth * 0.78)
        lines.append(
            _kinematic_cube_block(
                f"Step_{index:02d}",
                "/World/Looks/StairConcrete",
                translate=(0.0, y, top - step_height * 0.5),
                scale=(step_width, step_depth, step_height),
                indent="            ",
            )
        )
        lines.append(
            _kinematic_cube_block(
                f"LeftCurb_{index:02d}",
                "/World/Looks/StairConcrete",
                translate=(-1.78, y, top + 0.15),
                scale=(0.14, step_depth, 0.30),
                indent="            ",
            )
        )
        lines.append(
            _kinematic_cube_block(
                f"RightCurb_{index:02d}",
                "/World/Looks/StairConcrete",
                translate=(1.78, y, top + 0.15),
                scale=(0.14, step_depth, 0.30),
                indent="            ",
            )
        )
    lines.append(
        _kinematic_cube_block(
            "TopStop",
            "/World/Looks/StairConcrete",
            translate=(0.0, top_stop_y, 2.42),
            scale=(3.8, top_stop_depth, 0.28),
            indent="            ",
        )
    )
    lines.append("        }\n\n")
    return lines


def _tray_blocks() -> list[str]:
    lines = ['        def Xform "CatchTray"\n', "        {\n"]
    pieces = [
        ("TrayFloor", (0.0, 2.45, -0.05), (4.2, 2.0, 0.10)),
        ("TrayBack", (0.0, 3.48, 0.28), (4.2, 0.12, 0.66)),
        ("TrayLeft", (-2.08, 2.45, 0.24), (0.12, 2.1, 0.58)),
        ("TrayRight", (2.08, 2.45, 0.24), (0.12, 2.1, 0.58)),
    ]
    for name, translate, scale in pieces:
        lines.append(_kinematic_cube_block(name, "/World/Looks/TrayMetal", translate=translate, scale=scale, indent="            "))
    lines.append("        }\n\n")
    return lines


def _dynamic_body_blocks() -> list[str]:
    materials = [
        "CopperMetal",
        "ScarredTitanium",
        "FineWood",
        "BlueWovenFabric",
        "RedCarpet",
        "VintageLeather",
        "BlueScratchedPlastic",
        "BlackGymRubber",
        "StoneTile",
        "RustMeshConcrete",
        "TranslucentIce",
        "FrostedIce",
    ]
    positions = [
        (-0.90, -2.70, 3.45),
        (-0.30, -2.74, 3.55),
        (0.32, -2.66, 3.48),
        (0.92, -2.72, 3.60),
        (-0.64, -2.20, 3.78),
        (0.02, -2.16, 3.90),
        (0.68, -2.24, 3.82),
        (-0.96, -1.76, 4.08),
        (-0.28, -1.70, 4.18),
        (0.42, -1.78, 4.12),
        (0.98, -1.66, 4.22),
        (0.06, -2.96, 4.08),
    ]
    orientations = [
        (0.9914, 0.1305, 0.0, 0.0),
        (0.9848, 0.0, 0.1736, 0.0),
        (0.9763, 0.0, 0.0, 0.2164),
        (0.9962, -0.0872, 0.0, 0.0),
        (0.9659, 0.0, -0.2588, 0.0),
        (0.9877, 0.0, 0.0, -0.1564),
        (0.9914, 0.1305, 0.0, 0.0),
        (0.9848, 0.0, 0.1736, 0.0),
        (0.9763, 0.0, 0.0, 0.2164),
        (0.9962, -0.0872, 0.0, 0.0),
        (0.9659, 0.0, -0.2588, 0.0),
        (0.9877, 0.0, 0.0, -0.1564),
    ]
    velocities = [
        (-0.10, 1.67, 0.0),
        (0.06, 1.47, 0.0),
        (0.08, 1.86, 0.0),
        (-0.06, 1.38, 0.0),
        (0.12, 2.05, 0.0),
        (-0.08, 1.75, 0.0),
        (0.04, 1.56, 0.0),
        (0.10, 1.95, 0.0),
        (-0.12, 1.72, 0.0),
        (0.07, 1.52, 0.0),
        (-0.05, 1.79, 0.0),
        (0.00, 2.14, 0.0),
    ]
    angular_velocities = [
        (3.4, 0.6, -1.1),
        (-3.8, -0.4, 0.9),
        (4.2, 0.2, 1.3),
        (-3.2, 0.8, -1.4),
        (4.8, -0.5, 0.7),
        (-4.4, 0.3, 1.2),
        (3.6, -0.7, -0.9),
        (-5.0, 0.6, 1.0),
        (5.3, -0.4, -1.3),
        (-3.5, 0.5, 1.5),
        (4.6, 0.7, -0.8),
        (-4.9, -0.6, 1.1),
    ]
    lines = ['        def Xform "DynamicBodies"\n', "        {\n"]
    for index, (position, orientation, velocity, angular_velocity, material) in enumerate(
        zip(positions, orientations, velocities, angular_velocities, materials, strict=True)
    ):
        lines.append(_dynamic_cube_block(index, f"/World/Looks/{material}", position, orientation, velocity, angular_velocity))
    lines.append("        }\n")
    return lines


def _kinematic_cube_block(
    name: str,
    material_path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    indent: str,
) -> str:
    return (
        f'{indent}def Cube "{name}" (\n'
        f'{indent}    prepend apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]\n'
        f"{indent})\n"
        f"{indent}{{\n"
        f"{indent}    bool physics:kinematicEnabled = true\n"
        f"{indent}    rel material:binding = <{material_path}>\n"
        f"{indent}    double size = 1\n"
        f"{indent}    float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]\n"
        f"{_cube_st_primvar(indent)}"
        f"{indent}    double3 xformOp:translate = {_format_tuple(translate)}\n"
        f"{indent}    double3 xformOp:scale = {_format_tuple(scale)}\n"
        f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
        f"{indent}}}\n"
    )


def _dynamic_cube_block(
    index: int,
    material_path: str,
    translate: tuple[float, float, float],
    orient: tuple[float, float, float, float],
    velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float],
) -> str:
    prim_path = f"/World/PhysicsIsland/DynamicBodies/Cube_{index:02d}"
    return (
        f'            def Cube "Cube_{index:02d}" (\n'
        '                prepend apiSchemas = ["MaterialBindingAPI", "PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]\n'
        "            )\n"
        "            {\n"
        f'                custom string ovrtx:sourceUsdPath = "{prim_path}"\n'
        "                float physics:mass = 0.5\n"
        f"                vector3f physics:velocity = {_format_tuple(velocity)}\n"
        f"                vector3f physics:angularVelocity = {_format_tuple(angular_velocity)}\n"
        f"                rel material:binding = <{material_path}>\n"
        "                double size = 0.42\n"
        "                float3[] extent = [(-0.21, -0.21, -0.21), (0.21, 0.21, 0.21)]\n"
        f"{_cube_st_primvar('                ')}"
        f"                double3 xformOp:translate = {_format_tuple(translate)}\n"
        f"                quatd xformOp:orient = {_format_tuple(orient)}\n"
        '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n'
        "            }\n"
    )


def _cube_st_primvar(indent: str) -> str:
    face_st = (
        "[(0, 0), (1, 0), (1, 1), (0, 1), "
        "(0, 0), (1, 0), (1, 1), (0, 1), "
        "(0, 0), (1, 0), (1, 1), (0, 1), "
        "(0, 0), (1, 0), (1, 1), (0, 1), "
        "(0, 0), (1, 0), (1, 1), (0, 1), "
        "(0, 0), (1, 0), (1, 1), (0, 1)]"
    )
    return (
        f"{indent}    texCoord2f[] primvars:st = {face_st} (\n"
        f'{indent}        interpolation = "faceVarying"\n'
        f"{indent}    )\n"
    )


def _lighting_block() -> str:
    return (
        '    def DomeLight "StudioDome"\n'
        "    {\n"
        "        float inputs:intensity = 440\n"
        "        color3f inputs:color = (0.72, 0.78, 0.88)\n"
        '        token inputs:texture:format = "latlong"\n'
        "    }\n\n"
        '    def RectLight "KeyLight"\n'
        "    {\n"
        "        float inputs:intensity = 3200\n"
        "        color3f inputs:color = (1, 0.96, 0.88)\n"
        "        float inputs:width = 3.5\n"
        "        float inputs:height = 2.2\n"
        "        bool inputs:normalize = false\n"
        "        double3 xformOp:translate = (-2.8, -3.8, 6.4)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "    }\n\n"
        '    def RectLight "RimLight"\n'
        "    {\n"
        "        float inputs:intensity = 850\n"
        "        color3f inputs:color = (0.55, 0.72, 1)\n"
        "        float inputs:width = 4.0\n"
        "        float inputs:height = 2.0\n"
        "        bool inputs:normalize = false\n"
        "        double3 xformOp:translate = (3.0, 3.8, 4.2)\n"
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "    }\n\n"
    )


def _camera_block(matrix: tuple[tuple[float, float, float, float], ...]) -> str:
    return (
        '    def Camera "Camera"\n'
        "    {\n"
        "        float focalLength = 22\n"
        "        float horizontalAperture = 31.4\n"
        "        float verticalAperture = 17.6625\n"
        "        float2 clippingRange = (0.05, 100)\n"
        "        float fStop = 0\n"
        f"        matrix4d xformOp:transform = {_format_matrix4d(matrix)}\n"
        '        uniform token[] xformOpOrder = ["xformOp:transform"]\n'
        "    }\n\n"
    )


def _render_block(width: int, height: int) -> str:
    return (
        'def "Render"\n'
        "{\n"
        '    def "OmniverseKit"\n'
        "    {\n"
        '        def "HydraTextures"\n'
        "        {\n"
        '            def RenderProduct "ViewportTexture0"\n'
        "            {\n"
        f"                rel camera = <{CAMERA_PATH}>\n"
        '                token omni:rtx:rendermode = "RealTimePathTracing"\n'
        '                token omni:rtx:background:source:type = "sky"\n'
        "                rel orderedVars = </Render/OmniverseKit/HydraTextures/ViewportTexture0/LdrColor>\n"
        f"                uniform int2 resolution = ({width}, {height})\n"
        "                bool omni:rtx:indirectDiffuse:denoiser:enabled = true\n"
        "                bool omni:rtx:reflections:denoiser:enabled = true\n"
        "                bool omni:rtx:dlss:frameGeneration = false\n"
        "                bool omni:rtx:autoExposure:enabled = false\n"
        "                bool omni:rtx:rt:ecoMode:enabled = false\n"
        "                int omni:rtx:rtpt:maxVolumeBounces = 4\n"
        "                color3f omni:rtx:rt:ambientLight:color = (0.02, 0.02, 0.025)\n"
        '                def RenderVar "LdrColor"\n'
        "                {\n"
        '                    uniform string sourceName = "LdrColor"\n'
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _look_at_matrix(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    world_up = (0.0, 0.0, 1.0)
    forward = _normalize((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    back = (-forward[0], -forward[1], -forward[2])
    right = _normalize(_cross(world_up, back))
    up = _cross(back, right)
    return (
        (right[0], right[1], right[2], 0.0),
        (up[0], up[1], up[2], 0.0),
        (back[0], back[1], back[2], 0.0),
        (eye[0], eye[1], eye[2], 1.0),
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(component / length for component in vector)


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _format_tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(value):.6g}" for value in values) + ")"


def _format_matrix4d(matrix: Sequence[Sequence[float]]) -> str:
    return "(" + ", ".join(_format_tuple(row) for row in matrix) + ")"


def _pbr_texture_infos(texture_dir: Path) -> list[dict[str, str]]:
    index_path = texture_dir / "pbr-assets.json"
    if not index_path.is_file():
        raise ValueError(
            f"missing PBR texture index {index_path}; run tests/fixtures/prep/download_stair_drop_pbr_assets.py first"
        )
    with index_path.open("r", encoding="utf-8") as file:
        index = json.load(file)
    if not isinstance(index, Mapping):
        raise ValueError(f"PBR texture index must be an object: {index_path}")

    source_license = str(index.get("license", download_stair_drop_pbr_assets.AMBIENTCG_LICENSE))
    license_url = str(index.get("license_url", download_stair_drop_pbr_assets.AMBIENTCG_LICENSE_URL))
    infos: list[dict[str, str]] = []
    for asset in index.get("assets", []):
        if not isinstance(asset, Mapping):
            continue
        asset_id = str(asset.get("id", UNKNOWN))
        source_url = str(asset.get("source_url", f"https://ambientcg.com/a/{asset_id}"))
        maps = asset.get("maps", {})
        if not isinstance(maps, Mapping):
            continue
        for map_kind, map_info in sorted(maps.items()):
            if not isinstance(map_info, Mapping):
                continue
            path = _resolve_manifest_path(str(map_info.get("local_path", "")))
            if path is None or not path.is_file():
                raise ValueError(f"missing PBR texture map for {asset_id} {map_kind}: {map_info}")
            infos.append(
                {
                    "label": f"ambientCG {asset_id} {map_kind} map",
                    "path": str(path),
                    "sha256": _sha256(path),
                    "source_url": source_url,
                    "source_license": source_license,
                    "license_url": license_url,
                }
            )
    if not infos:
        raise ValueError(f"PBR texture index did not contain any texture maps: {index_path}")
    return infos


def _resolve_manifest_path(path: str) -> Path | None:
    if not path:
        return None
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return ROOT / parsed


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
