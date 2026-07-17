#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize a cataloged .blend fixture through current-scene generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = ROOT / "tests" / "fixtures"
if str(FIXTURES_PATH) not in sys.path:
    sys.path.insert(0, str(FIXTURES_PATH))
DEFAULT_FIXTURE_MANIFEST = FIXTURES_PATH
BLENDER_COMMAND = os.environ.get("BLENDER_COMMAND", "blender")

from fixture_manifest import fixture_input, load_manifest  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "materialized-fixture.json"
    log_path = args.output_dir / "blender.log"
    selected = fixture_input(load_manifest(args.manifest), args.fixture_id)
    if selected["kind"] != "blend":
        raise ValueError(f"fixture {args.fixture_id} is not a Blender fixture")
    blender = _resolve_executable(BLENDER_COMMAND)
    if not blender:
        raise ValueError(f"Blender executable is unavailable: {BLENDER_COMMAND}")
    completed = subprocess.run(
        _blender_command(blender, _blender_expr(selected, args.output_dir, result_path)),
        cwd=ROOT,
        env=dict(os.environ),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_s,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        _write_json(
            result_path,
            {
                "schema_version": 1,
                "artifact_id": "ov-blender-example-materialized-blend-fixture",
                "status": "failed",
                "fixture": selected,
                "error": "Blender materialization exited nonzero.",
                "blender_exit_status": completed.returncode,
                "blender_log": str(log_path),
            },
        )
        return completed.returncode
    if not result_path.is_file():
        raise RuntimeError("Blender exited without writing materialization evidence")
    return 0


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FIXTURE_MANIFEST)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args(list(argv))


def _resolve_executable(command: str) -> str:
    parts = shlex.split(command)
    if not parts:
        return ""
    first = Path(parts[0]).expanduser()
    if first.is_absolute() or os.sep in str(first):
        return str(first) if first.is_file() and os.access(first, os.X_OK) else ""
    return shutil.which(parts[0]) or ""


def _blender_command(blender: str, expr: str) -> list[str]:
    return [blender, "--background", "--python-expr", expr]


def _blender_expr(
    selected: Mapping[str, Any],
    output_dir: Path,
    result_path: Path,
) -> str:
    return f"""
from pathlib import Path
import hashlib
import json
import sys

root = Path({str(ROOT)!r})
sys.path.insert(0, str(root / "addon"))

import bpy

bpy.ops.wm.open_mainfile(filepath={str(selected["path"])!r})

import ovrtx_blender_example
ovrtx_blender_example.register()

from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
from ovrtx_blender_example.scene_generation import SceneGenerationOwner

output_dir = Path({str(output_dir)!r})
owner = SceneGenerationOwner(output_dir / "scene-generations")
generation = owner.replace(bpy.context.scene)
if generation is None:
    raise RuntimeError("current scene generation did not create an initial generation")

request = BlenderRenderCallbackAdapter(
    generation_for_scene=lambda _scene: generation,
).final_render_from_scene(bpy.context.scene)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

generation_root = Path(generation.materialize_usd()).parent
runtime_files = []
for path in sorted(item for item in generation_root.rglob("*") if item.is_file()):
    runtime_files.append({{
        "path": str(path),
        "sha256": sha256(path),
    }})

result = {{
    "schema_version": 1,
    "artifact_id": "ov-blender-example-materialized-blend-fixture",
    "status": "pass",
    "fixture": {json.dumps(dict(selected), sort_keys=True)},
    "source": {{
        "blend_file": {str(selected["manifest_path"])!r},
        "selected_path": {str(selected["path"])!r},
        "blend_file_sha256": {str(selected["sha256"])!r},
    }},
    "generation": {{
        "digest": generation.digest,
        "usd_path": generation.materialize_usd(),
        "usd_sha256": sha256(generation.materialize_usd()),
        "base_usd_path": generation.base_usd_path,
        "base_usd_sha256": sha256(generation.base_usd_path),
        "runtime_files": runtime_files,
        "diagnostics": dict(generation.diagnostics),
    }},
    "render_request": {{
        "input_usd_path": request.input_usd_path,
        "camera_prim_path": request.camera_prim_path,
        "render_product_path": request.render_product_path,
        "width": request.width,
        "height": request.height,
    }},
}}
Path({str(result_path)!r}).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
"""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
