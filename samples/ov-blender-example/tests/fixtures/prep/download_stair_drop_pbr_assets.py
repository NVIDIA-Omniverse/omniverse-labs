#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download CC0 ambientCG PBR maps for the stair-drop demo fixture."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Sequence
import urllib.request
import zipfile


FIXTURES_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = FIXTURES_ROOT.parent
REPO_ROOT = TESTS_ROOT.parent
ROOT = TESTS_ROOT
FIXTURE_ID = "demo_stair_drop_1280x720"
AMBIENTCG_LICENSE_URL = "https://docs.ambientcg.com/license/"
AMBIENTCG_LICENSE = "Creative Commons CC0 1.0 Universal"
TEXTURE_PACKAGE = "1K-JPG"

ASSETS = [
    {"id": "Concrete016", "role": "stair light concrete"},
    {"id": "Metal031", "role": "catch tray gray metal"},
    {"id": "Metal035", "role": "copper metal cube"},
    {"id": "Metal059C", "role": "damaged titanium cube"},
    {"id": "Wood025", "role": "fine wood cube"},
    {"id": "Fabric017", "role": "blue woven fabric cube"},
    {"id": "Fabric026", "role": "red carpet cube"},
    {"id": "Leather014", "role": "scratched brown leather cube"},
    {"id": "Plastic001", "role": "scratched blue plastic cube"},
    {"id": "Rubber004", "role": "black gym rubber cube"},
    {"id": "Tiles027", "role": "rough stone tile cube"},
    {"id": "Concrete042B", "role": "rust mesh concrete cube"},
    {"id": "Ice001", "role": "translucent ice cube"},
    {"id": "Ice003", "role": "frosted ice cube"},
]

MAP_MARKERS = {
    "color": "color",
    "roughness": "roughness",
    "normalgl": "normal_gl",
    "metalness": "metalness",
    "ambientocclusion": "ambient_occlusion",
    "displacement": "displacement",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_dir = args.fixture_dir or ROOT / "fixtures" / "assets" / FIXTURE_ID
    result_path = args.result or REPO_ROOT / "out" / "artifacts" / "stair-drop-pbr-assets" / "result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "stair-drop-pbr-assets",
        "status": "running",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "fixture_id": FIXTURE_ID,
        "license": AMBIENTCG_LICENSE,
        "license_url": AMBIENTCG_LICENSE_URL,
        "assets": [],
    }

    try:
        asset_index, index_path = prepare_assets(fixture_dir, force=args.force)
        result["assets"] = asset_index["assets"]
        result["status"] = "pass"
        result["texture_index"] = str(index_path)
        return_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return_code = 1

    result["completed_at_ns"] = time.time_ns()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path)}, indent=2, sort_keys=True))
    return return_code


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def prepare_assets(
    fixture_dir: Path,
    *,
    force: bool = False,
    expected_archives: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Path]:
    downloads_dir = fixture_dir / "downloads" / "ambientcg"
    textures_dir = fixture_dir / "fixture" / "textures" / "ambientcg"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    textures_dir.mkdir(parents=True, exist_ok=True)
    assets = [
        _prepare_asset(
            asset,
            downloads_dir,
            textures_dir,
            force=force,
            expected_sha256=(expected_archives or {}).get(asset["id"]),
        )
        for asset in ASSETS
    ]
    asset_index = {
        "schema_version": 1,
        "source": "ambientCG",
        "license": AMBIENTCG_LICENSE,
        "license_url": AMBIENTCG_LICENSE_URL,
        "texture_package": TEXTURE_PACKAGE,
        "assets": assets,
    }
    index_path = textures_dir / "pbr-assets.json"
    index_path.write_text(json.dumps(asset_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return asset_index, index_path


def _prepare_asset(
    asset: dict[str, str],
    downloads_dir: Path,
    textures_dir: Path,
    *,
    force: bool,
    expected_sha256: str | None,
) -> dict[str, Any]:
    asset_id = asset["id"]
    archive = downloads_dir / f"{asset_id}_{TEXTURE_PACKAGE}.zip"
    url = f"https://ambientcg.com/get?file={archive.name}"
    if force or not archive.is_file():
        _download(url, archive, expected_sha256)
    elif expected_sha256 and _sha256(archive) != expected_sha256:
        raise ValueError(f"ambientCG archive SHA-256 mismatch: {archive}")

    asset_dir = textures_dir / asset_id
    if force and asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    maps: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            marker = _map_marker(member.filename)
            if marker is None:
                continue
            target = asset_dir / Path(member.filename).name
            _safe_extract_member(zip_file, member, target)
            maps[marker] = {
                "local_path": _manifest_path(target),
                "sha256": _sha256(target),
            }

    missing = sorted({"color", "roughness", "normal_gl"} - maps.keys())
    if missing:
        raise ValueError(f"{asset_id} missing expected PBR map(s): {', '.join(missing)}")

    return {
        "id": asset_id,
        "role": asset["role"],
        "source_url": f"https://ambientcg.com/a/{asset_id}",
        "download_url": url,
        "archive": {
            "local_path": _manifest_path(archive),
            "sha256": _sha256(archive),
        },
        "maps": maps,
    }


def _map_marker(filename: str) -> str | None:
    normalized = Path(filename).name.lower().replace("_", "").replace("-", "")
    if not normalized.endswith((".jpg", ".jpeg")):
        return None
    for marker, kind in MAP_MARKERS.items():
        if marker in normalized:
            return kind
    return None


def _safe_extract_member(zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zip_file.open(member, "r") as source, target.open("wb") as output:
        shutil.copyfileobj(source, output)


def _download(url: str, target: Path, expected_sha256: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ov-blender-example-fixtures/1.0"})
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if expected_sha256 and _sha256(temporary) != expected_sha256:
            raise ValueError(f"ambientCG archive SHA-256 mismatch: {target}")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
