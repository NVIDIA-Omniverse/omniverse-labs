#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author one reproducible fixture from a Blender or USD source scene."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, MutableMapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_fixtures as fixtures


ROOT = Path(__file__).resolve().parents[3]
UNKNOWN = "???"
DEFAULT_RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    scene_id = args.scene_id or _default_scene_id(args.scene)
    output_dir = args.output_dir or ROOT / "out" / "fixtures" / scene_id
    manifest_path = args.manifest or output_dir / "manifest.json"
    result_path = args.result or output_dir / "result.json"

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "ov-blender-author-fixture",
        "status": "running",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "scene": str(args.scene),
        "scene_id": scene_id,
        "manifest": str(manifest_path),
    }

    try:
        scene = _resolve_scene(args.scene)
        output_dir.mkdir(parents=True, exist_ok=True)
        fixture = _fixture(scene, scene_id, args, output_dir)
        build_args = argparse.Namespace(
            asset_dir=output_dir / "assets",
            blender=fixtures.BLENDER_COMMAND,
            force=args.force,
            skip_blend_export=False,
            skip_download=True,
        )
        notes = fixtures._build_fixture(fixture, build_args)
        _absolutize_paths(fixture)
        manifest = _manifest(fixture)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")

        result.update(
            {
                "status": "pass",
                "fixture": _result_fixture(fixture),
                "inspection": notes.get("inspection", {}),
            }
        )
        _write_result(result_path, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "manifest": str(manifest_path),
                    "result": str(result_path),
                    "fixture_id": scene_id,
                    "fixture_usd_path": result["fixture"].get("fixture_usd_path", UNKNOWN),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
        _write_result(result_path, result)
        print(
            json.dumps(
                {"status": result["status"], "result": str(result_path), "error": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 1


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Source .blend, .usd, .usda, or .usdc scene.")
    parser.add_argument("--scene-id", default="", help="Fixture id to write into the generated manifest.")
    parser.add_argument("--display-name", default="", help="Human-readable scene name for the manifest.")
    parser.add_argument("--manifest", type=Path, help="Output manifest path.")
    parser.add_argument("--output-dir", type=Path, help="Directory for USD and result artifacts.")
    parser.add_argument("--result", type=Path, help="Output result JSON path.")
    parser.add_argument("--fixture-usd-path", type=Path, help="Authored fixture USD stage path.")
    parser.add_argument("--source-usd", type=Path, help="Exported source USD path for .blend inputs.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-prim-path", default="")
    parser.add_argument("--render-product-prim-path", default=DEFAULT_RENDER_PRODUCT)
    parser.add_argument("--force", action="store_true", help="Rewrite exported and usd scene artifacts.")
    args = parser.parse_args(list(argv))
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


def _resolve_scene(path: Path) -> Path:
    scene = path.expanduser().resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"scene not found: {scene}")
    suffix = scene.suffix.lower()
    if suffix != ".blend" and suffix not in USD_SUFFIXES:
        raise ValueError("scene must be a .blend, .usd, .usda, or .usdc file")
    return scene


def _fixture(scene: Path, scene_id: str, args: argparse.Namespace, output_dir: Path) -> MutableMapping[str, Any]:
    kind = "blend" if scene.suffix.lower() == ".blend" else "usd"
    source_asset: dict[str, Any] = {
        "kind": kind,
        "label": f"Source {kind.upper()} scene",
        "availability": "available",
        "local_path": fixtures._manifest_path(scene),
        "sha256": fixtures._sha256(scene),
    }
    if kind == "blend":
        source_usd = args.source_usd or output_dir / "source" / f"{scene_id}.usdc"
        source_asset["export_usd_path"] = fixtures._manifest_path(source_usd)

    usd_path = args.fixture_usd_path or output_dir / "fixture" / f"{scene_id}.usda"
    fixture: MutableMapping[str, Any] = {
        "id": scene_id,
        "display_name": args.display_name or scene.stem,
        "availability": "available",
        "capabilities": ["ovrtx"],
        "target_resolution": {"width": args.width, "height": args.height},
        "fixture_usd_path": fixtures._manifest_path(usd_path),
        "fixture_usd_sha256": UNKNOWN,
        "render_product_prim_path": args.render_product_prim_path,
        "asset_files": [source_asset],
    }
    if args.camera_prim_path:
        fixture["camera_prim_path"] = args.camera_prim_path
    return fixture


def _manifest(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": "ov-blender-scene-manifest",
        "unknown_marker": UNKNOWN,
        "fixtures": [fixture],
    }


def _absolutize_paths(fixture: MutableMapping[str, Any]) -> None:
    value = fixture.get("fixture_usd_path")
    if isinstance(value, str):
        fixture["fixture_usd_path"] = _absolute_manifest_path(value)
    for asset in fixture.get("asset_files", []):
        if not isinstance(asset, MutableMapping):
            continue
        for key in ("local_path", "export_usd_path"):
            value = asset.get(key)
            if isinstance(value, str):
                asset[key] = _absolute_manifest_path(value)


def _absolute_manifest_path(value: str) -> str:
    path = fixtures._resolve_manifest_path(value)
    return value if path is None else str(path.resolve())


def _result_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(fixture.get("id", UNKNOWN)),
        "fixture_usd_path": str(fixture.get("fixture_usd_path", UNKNOWN)),
        "fixture_usd_sha256": str(fixture.get("fixture_usd_sha256", UNKNOWN)),
        "camera_prim_path": str(fixture.get("camera_prim_path", UNKNOWN)),
        "render_product_prim_path": str(fixture.get("render_product_prim_path", UNKNOWN)),
        "unresolved_values": list(fixture.get("unresolved_values", [])),
    }


def _default_scene_id(scene: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", scene.stem.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "scene"


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    payload = dict(result)
    payload["completed_at_ns"] = time.time_ns()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
