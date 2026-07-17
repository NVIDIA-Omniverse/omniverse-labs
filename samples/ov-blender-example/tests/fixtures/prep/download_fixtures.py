#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare OVRTX Blender render fixtures."""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, MutableMapping, Sequence
import urllib.request
import zipfile

import author_stair_drop_fixture
import download_stair_drop_pbr_assets

FIXTURES_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = FIXTURES_ROOT.parent
REPO_ROOT = TESTS_ROOT.parent
ADDON_ROOT = REPO_ROOT / "addon"
ROOT = TESTS_ROOT
COMMITTED_RUNTIME_MANIFEST = FIXTURES_ROOT / "manifest.json"
UNKNOWN = "???"
DEFAULT_RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
BLENDER_COMMAND = "blender"
CLASSROOM_FIXTURE_ID = "perf_blender_classroom_1280x720"
FLATTENED_FIXTURE_IDS = {
    CLASSROOM_FIXTURE_ID,
}

if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))
if str(FIXTURES_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURES_ROOT))

from fixture_manifest import fixture_content_sha256  # noqa: E402
from ovrtx_blender_example import light_value_conversion as _light_policy  # noqa: E402


def prepare_spec(path: Path, *, force: bool = False) -> dict[str, Any]:
    """Prepare one adjacent fixture spec without rewriting it."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    preparation = spec.get("preparation")
    if not isinstance(preparation, Mapping):
        raise ValueError(f"fixture spec has no preparation object: {path}")
    fixture = copy.deepcopy(spec)
    fixture.update(copy.deepcopy(preparation))
    return _build_fixture(
        fixture,
        argparse.Namespace(
            asset_dir=FIXTURES_ROOT / "data",
            force=force,
            skip_download=False,
            skip_blend_export=False,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest = _load_manifest(args.manifest)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "fixture-download",
        "started_at_ns": time.time_ns(),
        "generated_at_utc": _utc_now(),
        "fixtures": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        selected_fixture_ids = set(args.fixture_id or [])
        usd_fixture_ids: set[str] = set()
        for fixture in manifest.get("fixtures", []):
            if not isinstance(fixture, MutableMapping):
                continue
            fixture_id = str(fixture.get("id", UNKNOWN))
            if selected_fixture_ids and fixture_id not in selected_fixture_ids:
                continue
            if "ovrtx" not in _fixture_capabilities(fixture):
                continue
            result["fixtures"].append(_build_fixture(fixture, args))
            usd_fixture_ids.add(fixture_id)
        missing_fixture_ids = sorted(selected_fixture_ids - usd_fixture_ids)
        if missing_fixture_ids:
            raise ValueError(f"requested fixture ids were not usd: {', '.join(missing_fixture_ids)}")

        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        result["status"] = "pass"
        return_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return_code = 1

    result["completed_at_ns"] = time.time_ns()
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(args.result)}, indent=2, sort_keys=True))
    return return_code


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Mutable prep manifest copy. The committed runtime fixture catalog is read-only.",
    )
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "fixtures" / "assets")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "out" / "artifacts" / "fixture-download")
    parser.add_argument("--result", type=Path, default=REPO_ROOT / "out" / "artifacts" / "fixture-download" / "result.json")
    parser.add_argument("--fixture-id", action="append", default=None, help="Only build the selected fixture id. May be repeated.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-blend-export", action="store_true")
    args = parser.parse_args(list(argv))
    if args.manifest.resolve() == COMMITTED_RUNTIME_MANIFEST.resolve():
        parser.error(
            "the committed runtime fixture catalog is read-only; "
            "copy the historical prep manifest to a separate path"
        )
    return args


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _build_fixture(fixture: MutableMapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    fixture_id = str(fixture.get("id", UNKNOWN))
    fixture_dir = args.asset_dir / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    notes: dict[str, Any] = {"id": fixture_id, "downloads": [], "exports": [], "inspection": {}}

    if fixture_id == author_stair_drop_fixture.FIXTURE_ID:
        return _build_authored_stair_drop_fixture(fixture, args, notes, fixture_dir)

    for asset in _asset_files(fixture):
        source_url = asset.get("source_url")
        if not source_url:
            continue
        source_url = str(source_url)
        local_path = _asset_local_path(asset)
        if local_path is not None:
            _mark_asset_available(asset, local_path)
            notes["downloads"].append(
                {"url": source_url, "path": _manifest_path(local_path), "sha256": asset["sha256"]}
            )
            continue
        if _is_omniverse_url(source_url):
            mirror_path = _declared_asset_path(asset)
            reason = "Omniverse source must be mirrored locally before prep"
            if mirror_path is not None:
                reason += f": {_manifest_path(mirror_path)}"
            _mark_asset_unavailable(asset, reason)
            notes["downloads"].append(
                {
                    "url": source_url,
                    "path": _manifest_path(mirror_path) if mirror_path is not None else UNKNOWN,
                    "status": "unavailable",
                    "missing_reason": reason,
                }
            )
            continue
        if args.skip_download:
            target = _download_target(fixture_dir, asset)
            reason = f"source file is not present and --skip-download was set: {_manifest_path(target)}"
            _mark_asset_unavailable(asset, reason)
            notes["downloads"].append(
                {
                    "url": source_url,
                    "path": _manifest_path(target),
                    "status": "skipped",
                    "missing_reason": reason,
                }
            )
            continue
        target = _download_target(fixture_dir, asset)
        _download(
            source_url,
            target,
            force=args.force,
            expected_sha256=asset.get("expected_archive_sha256"),
        )
        _validate_expected_sha256(
            target,
            asset.get("expected_archive_sha256"),
            label="downloaded archive",
        )
        archive_member = asset.get("archive_member")
        extract_dir = _asset_extract_dir(fixture_dir, asset)
        strip_components = _asset_strip_components(asset)
        extract_subtree = _asset_extract_subtree(asset)
        if archive_member or extract_dir is not None:
            extract_dir = extract_dir or fixture_dir / "source"
            _extract_zip(target, extract_dir, strip_components=strip_components, extract_subtree=extract_subtree)
            local_path = (
                extract_dir / _zip_member_relative_path(
                    str(archive_member),
                    strip_components=strip_components,
                    extract_subtree=extract_subtree,
                )
                if archive_member
                else target
            )
        else:
            local_path = target
        _mark_asset_available(asset, local_path)
        notes["downloads"].append({"url": source_url, "path": _manifest_path(local_path), "sha256": asset["sha256"]})

    source_blend = _source_blend_asset(fixture)
    if source_blend is not None and source_blend.get("local_path") != UNKNOWN:
        source_blend_path = _resolve_manifest_path(str(source_blend["local_path"]))
        source_usd = _normalized_source_usd(fixture, args, notes, source_blend, source_blend_path, "blend")
    else:
        source_usd_asset = _source_usd_asset(fixture)
        source_usd_path = _asset_local_path(source_usd_asset)
        source_usd = _normalized_source_usd(fixture, args, notes, source_usd_asset, source_usd_path, "usd")

    if source_usd is None or not source_usd.is_file():
        usd_path = _resolve_manifest_path(str(fixture.get("fixture_usd_path", "")))
        if usd_path and usd_path.is_file():
            _upsert_usd_asset(fixture, usd_path)
            fixture["availability"] = "available"
            _refresh_unresolved_values(fixture)
            notes["inspection"] = {
                "fixture_usd_path": _manifest_path(usd_path),
                "source": "authored-usd",
            }
            return notes
        fixture["availability"] = "unavailable"
        _refresh_unresolved_values(fixture)
        notes["inspection"] = {"error": "source USD unavailable"}
        return notes

    lighting_definition = _fixture_lighting_definition(fixture)
    render_product_definition = _fixture_render_product_definition(fixture)
    width, height = _fixture_resolution(fixture)
    source_info = _inspect_usd(BLENDER_COMMAND, source_usd)
    product_path = _choose_render_product(fixture, source_info)
    source_camera_path = _choose_camera(fixture, source_info, product_path)
    source_has_render_product = product_path in source_info.get("render_products", [])
    source_product_is_renderable = _render_product_targets_camera(source_info, product_path, source_camera_path)
    camera_details = source_info.get("camera_details", {}).get(source_camera_path)
    if camera_details is None:
        camera_details = _fallback_camera_details(source_info.get("bounds"))
    camera_definition = _fixture_camera_definition(fixture)
    force_usd_camera = camera_definition is not None
    if camera_definition is not None:
        camera_details = camera_definition
        camera_path = str(fixture.get("camera_prim_path", "/OvrtxCamera")) or "/OvrtxCamera"
    else:
        camera_details = _fixture_camera_details(fixture_id, camera_details)
        if source_product_is_renderable:
            camera_path = source_camera_path
        elif camera_details:
            camera_path = "/OvrtxCamera"
        else:
            camera_path = source_camera_path
    usd_path = _resolve_manifest_path(str(fixture.get("fixture_usd_path", "")))
    if not usd_path:
        usd_path = fixture_dir / "fixture" / f"{fixture_id}.usda"
    classroom_repairs = ""
    write_path = usd_path
    if fixture_id == CLASSROOM_FIXTURE_ID:
        if source_blend is None or source_blend.get("local_path") == UNKNOWN:
            raise RuntimeError("Classroom beauty repair requires the pinned source .blend")
        classroom_repairs = _classroom_repair_block(
            _resolve_manifest_path(str(source_blend["local_path"])),
            source_usd,
            source_lights=source_info.get("lights", []),
            lighting_definition=lighting_definition,
        )
    if fixture_id in FLATTENED_FIXTURE_IDS:
        write_path = usd_path.with_name(f".{usd_path.stem}.compose.usda")
    _write_usd_stage(
        write_path,
        source_usd,
        camera_path,
        product_path,
        camera_details=camera_details,
        add_camera=force_usd_camera
        or not source_product_is_renderable
        and (camera_details is not None or camera_path not in source_info.get("cameras", [])),
        add_render_product=force_usd_camera or not source_product_is_renderable,
        add_fixture_definitions=force_usd_camera or not source_product_is_renderable,
        lighting_definition=lighting_definition,
        render_product_definition=render_product_definition,
        width=width,
        height=height,
        source_lights=(
            [] if fixture_id == CLASSROOM_FIXTURE_ID else source_info.get("lights", [])
        ),
        source_lights_are_policy_converted=False,
        fixture_block=classroom_repairs,
    )
    if fixture_id in FLATTENED_FIXTURE_IDS:
        _flatten_usd_stage(write_path, usd_path)
        write_path.unlink(missing_ok=True)
        _remove_normalized_source_asset(fixture)
        source_usd.unlink(missing_ok=True)
    usd_info = _inspect_usd(BLENDER_COMMAND, usd_path)
    _upsert_usd_asset(fixture, usd_path)
    fixture["availability"] = "available"
    fixture["camera_prim_path"] = camera_path
    fixture["render_product_prim_path"] = product_path
    _refresh_unresolved_values(fixture)
    notes["inspection"] = {
        "source_usd": _manifest_path(source_usd),
        "fixture_usd_path": _manifest_path(usd_path),
        "fixture_usd_sha256": str(fixture["fixture_usd_sha256"]),
        "source_cameras": source_info.get("cameras", []),
        "source_lights": source_info.get("lights", []),
        "source_render_products": source_info.get("render_products", []),
        "source_bounds": source_info.get("bounds"),
        "source_selected_camera": source_camera_path,
        "selected_camera": camera_path,
        "selected_camera_source": "manifest" if force_usd_camera else "source",
        "selected_render_product": product_path,
        "usd_cameras": usd_info.get("cameras", []),
        "usd_render_products": usd_info.get("render_products", []),
    }
    return notes


def _build_authored_stair_drop_fixture(
    fixture: MutableMapping[str, Any],
    args: argparse.Namespace,
    notes: MutableMapping[str, Any],
    fixture_dir: Path,
) -> dict[str, Any]:
    width, height = _fixture_resolution(fixture)
    usd_path = _resolve_manifest_path(str(fixture.get("fixture_usd_path", "")))
    if not usd_path:
        usd_path = fixture_dir / "fixture" / "stair_drop_ovrtx_ovphysx.usda"
    if not args.skip_download:
        asset_index, texture_index_path = download_stair_drop_pbr_assets.prepare_assets(fixture_dir, force=args.force)
        notes["downloads"].append(
            {
                "source": "ambientCG",
                "license": asset_index["license"],
                "path": _manifest_path(texture_index_path),
                "asset_count": len(asset_index["assets"]),
            }
        )
    texture_infos = author_stair_drop_fixture._pbr_texture_infos(
        usd_path.parent / author_stair_drop_fixture.PBR_TEXTURE_ROOT
    )
    if args.force or not usd_path.is_file():
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        usd_path.write_text(author_stair_drop_fixture._fixture_usda(width, height), encoding="utf-8")
    sha256 = _sha256(usd_path)
    fixture.clear()
    fixture.update(author_stair_drop_fixture._fixture_record(usd_path, sha256, width, height, texture_infos))
    usd_info = _inspect_usd(BLENDER_COMMAND, usd_path)
    notes["inspection"] = {
        "source": "authored-usd",
        "fixture_usd_path": _manifest_path(usd_path),
        "usd_cameras": usd_info.get("cameras", []),
        "usd_render_products": usd_info.get("render_products", []),
        "usd_bounds": usd_info.get("bounds"),
        "dynamic_body_root": "/World/PhysicsIsland/DynamicBodies",
        "dynamic_body_count": 12,
    }
    return notes


def _fixture_capabilities(fixture: Mapping[str, Any]) -> list[str]:
    capabilities = fixture.get("capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} capabilities must be a list of strings")
    if not capabilities:
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} capabilities must not be empty")
    return capabilities


def _asset_files(fixture: MutableMapping[str, Any]) -> list[MutableMapping[str, Any]]:
    assets = fixture.setdefault("asset_files", [])
    if not isinstance(assets, list):
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} asset_files must be a list")
    if not all(isinstance(asset, MutableMapping) for asset in assets):
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} asset_files entries must be objects")
    return assets


def _download_target(fixture_dir: Path, asset: Mapping[str, Any]) -> Path:
    filename = str(asset.get("download_filename") or asset.get("expected_name") or Path(str(asset["source_url"])).name)
    if not filename or filename == UNKNOWN:
        filename = "fixture-download"
    return fixture_dir / "downloads" / filename


def _asset_extract_dir(fixture_dir: Path, asset: Mapping[str, Any]) -> Path | None:
    value = str(asset.get("extract_dir", ""))
    if not value or value == UNKNOWN:
        return None
    path = _resolve_manifest_path(value)
    return path or fixture_dir / "source"


def _asset_strip_components(asset: Mapping[str, Any]) -> int:
    value = asset.get("strip_components", 0)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strip_components must be an integer: {value!r}") from exc
    if result < 0:
        raise ValueError(f"strip_components must not be negative: {value!r}")
    return result


def _asset_extract_subtree(asset: Mapping[str, Any]) -> str | None:
    value = str(asset.get("extract_subtree", ""))
    if not value or value == UNKNOWN:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"extract_subtree must be a safe relative path: {value!r}")
    if len(path.parts) != 1:
        raise ValueError(f"extract_subtree currently supports one path component: {value!r}")
    return path.parts[0]


def _download(
    url: str,
    target: Path,
    *,
    force: bool,
    expected_sha256: Any = None,
) -> None:
    if target.is_file() and not force:
        _validate_expected_sha256(target, expected_sha256, label="downloaded file")
        return
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
        _validate_expected_sha256(temporary, expected_sha256, label="downloaded file")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extract_zip(
    archive: Path,
    target_dir: Path,
    *,
    strip_components: int = 0,
    extract_subtree: str | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        root = target_dir.resolve()
        for member in zip_file.infolist():
            relative_path = _zip_member_relative_path(
                member.filename,
                strip_components=strip_components,
                extract_subtree=extract_subtree,
            )
            if relative_path is None:
                continue
            destination = (target_dir / relative_path).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"unsafe zip member path: {member.filename}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def _zip_member_relative_path(
    filename: str,
    *,
    strip_components: int,
    extract_subtree: str | None,
) -> Path | None:
    path = PurePosixPath(filename)
    parts = path.parts
    if path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe zip member path: {filename}")
    if extract_subtree:
        try:
            subtree_index = parts.index(extract_subtree)
        except ValueError:
            return None
        parts = parts[subtree_index + 1 :]
    if len(parts) <= strip_components:
        return None
    return Path(*parts[strip_components:])


def _mark_asset_available(asset: MutableMapping[str, Any], path: Path) -> None:
    actual_sha256 = _sha256(path)
    _validate_expected_sha256(
        path,
        asset.get("expected_sha256"),
        label="extracted source",
        actual_sha256=actual_sha256,
    )
    asset["availability"] = "available"
    declared = str(asset.get("local_path", ""))
    asset["local_path"] = (
        declared if _resolve_manifest_path(declared) == path else _manifest_path(path)
    )
    asset["sha256"] = actual_sha256
    asset.pop("missing_reason", None)


def _validate_expected_sha256(
    path: Path,
    expected: Any,
    *,
    label: str,
    actual_sha256: str | None = None,
) -> None:
    expected_sha256 = str(expected or "")
    if not expected_sha256 or expected_sha256 == UNKNOWN:
        return
    actual = actual_sha256 or _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch for {_manifest_path(path)}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _mark_asset_unavailable(asset: MutableMapping[str, Any], reason: str) -> None:
    asset["availability"] = "unavailable"
    asset["missing_reason"] = reason
    asset.setdefault("sha256", UNKNOWN)


def _is_omniverse_url(value: str) -> bool:
    return value.startswith("omniverse://")


def _declared_asset_path(asset: Mapping[str, Any]) -> Path | None:
    return _resolve_manifest_path(str(asset.get("local_path", UNKNOWN)))


def _source_usd_asset(fixture: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    for asset in fixture.get("asset_files", []):
        if not isinstance(asset, MutableMapping):
            continue
        if (
            asset.get("kind") == "usd"
            and asset.get("render_ready") is not True
            and "generated_from" not in asset
        ):
            return asset
    return None


def _asset_local_path(asset: Mapping[str, Any] | None) -> Path | None:
    if asset is None:
        return None
    path = _resolve_manifest_path(str(asset.get("local_path", UNKNOWN)))
    if path and path.is_file():
        return path
    return None


def _source_blend_asset(fixture: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    for asset in fixture.get("asset_files", []):
        if isinstance(asset, MutableMapping) and asset.get("kind") == "blend":
            return asset
    return None


def _normalized_source_usd(
    fixture: Mapping[str, Any],
    args: argparse.Namespace,
    notes: MutableMapping[str, Any],
    source_asset: Mapping[str, Any] | None,
    source_path: Path | None,
    source_kind: str,
) -> Path | None:
    if source_asset is None or source_path is None:
        return None
    export_path = _resolve_manifest_path(str(source_asset.get("export_usd_path", "")))
    if export_path is None:
        return source_path if source_kind == "usd" else None
    if args.force or not export_path.is_file():
        if args.skip_blend_export:
            return None
        _export_with_prep_exporter(
            BLENDER_COMMAND,
            source_path,
            export_path,
            _fixture_resolution(fixture),
            source_kind=source_kind,
            evaluated_geometry=False,
            classroom_material_compatibility=(
                str(fixture.get("id")) == CLASSROOM_FIXTURE_ID
            ),
        )
    if not export_path.is_file():
        return None
    _reject_temp_texture_references(export_path)
    _upsert_source_usd_asset(fixture, export_path, source_asset)
    notes["exports"].append(
        {
            "source": source_asset.get("local_path", UNKNOWN),
            "usd": _manifest_path(export_path),
            "sha256": _sha256(export_path),
            "temp_texture_references": "none",
        }
    )
    return export_path


def _upsert_source_usd_asset(
    fixture: MutableMapping[str, Any],
    source_usd: Path,
    source_asset: Mapping[str, Any],
) -> None:
    label = "Stock Blender normalized USD"
    for asset in _asset_files(fixture):
        if asset.get("label") in {
            label,
            "Blender exporter normalized USD",
            "Blender source exported USD",
        }:
            target = asset
            break
    else:
        target = {"kind": "usd", "label": label}
        _asset_files(fixture).append(target)
    target.update(
        {
            "kind": "usd",
            "label": label,
            "generated_from": source_asset.get("local_path", UNKNOWN),
            "availability": "available",
            "local_path": _manifest_path(source_usd),
            "sha256": _sha256(source_usd),
        }
    )


def _upsert_usd_asset(fixture: MutableMapping[str, Any], usd_path: Path) -> None:
    label = "OVRTX render-ready USD"
    manifest_path = _manifest_path(usd_path)
    sha256 = _sha256(usd_path)
    fixture["fixture_usd_path"] = manifest_path
    fixture["fixture_usd_sha256"] = sha256
    fixture["fixture_content_sha256"] = fixture_content_sha256(usd_path)
    assets = _asset_files(fixture)
    for asset in assets:
        if asset.get("render_ready") is True:
            target = asset
            break
    else:
        target = {"kind": "usd", "label": label, "render_ready": True}
        assets.insert(0, target)
    target.update(
        {
            "kind": "usd",
            "label": label,
            "render_ready": True,
            "availability": "available",
            "local_path": manifest_path,
            "sha256": sha256,
        }
    )


def _reject_temp_texture_references(path: Path) -> None:
    pattern = b"ovrtx_textures"
    with path.open("rb") as file:
        previous = b""
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                return
            data = previous + chunk
            if pattern in data:
                raise ValueError(
                    f"{_manifest_path(path)} references the temp ovrtx texture cache; "
                    "set OVRTX_TEXTURE_OUTPUT_DIR during export so textures are durable"
                )
            previous = data[-len(pattern) :]


def _fixture_resolution(fixture: Mapping[str, Any]) -> tuple[int, int]:
    resolution = fixture.get("target_resolution", {})
    if not isinstance(resolution, Mapping):
        return 1280, 720
    try:
        return int(resolution.get("width", 1280)), int(resolution.get("height", 720))
    except (TypeError, ValueError):
        return 1280, 720


def _export_with_prep_exporter(
    blender: str,
    source_path: Path,
    usd_path: Path,
    resolution: tuple[int, int],
    *,
    source_kind: str,
    evaluated_geometry: bool = False,
    classroom_material_compatibility: bool = False,
) -> None:
    if usd_path.suffix.lower() != ".usda":
        raise ValueError(f"stock fixture export path must end in .usda: {_manifest_path(usd_path)}")
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = usd_path.with_name(
        f".{usd_path.stem}.tmp{usd_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)
    width, height = resolution
    script = f"""
from pathlib import Path

import bpy
from pxr import Sdf, Usd, UsdGeom

source_kind = {source_kind!r}
source_path = {str(source_path)!r}
target = Path({str(temporary_path)!r})
raw_target = target.with_name(f".{{target.stem}}.raw.usdc")
raw_target.unlink(missing_ok=True)

if source_kind == "blend":
    bpy.ops.wm.open_mainfile(filepath=source_path)
elif source_kind == "usd":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.wm.usd_import(filepath=source_path)
else:
    raise RuntimeError(f"unsupported source kind: {{source_kind}}")


def prepare_particle_hair():
    # Operator-free particle-hair export (mirrors the add-on's
    # scene_generation._temporary_particle_hair_curves): sample the evaluated
    # co_hair cache (authored hair_keys fallback) and author a Curves object via
    # the data API instead of bpy.ops.curves.convert_from_particle_system, whose
    # poll() fails outside an interactive context.
    import numpy as np

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for emitter in list(bpy.data.objects):
        if emitter.name not in bpy.context.view_layer.objects:
            continue
        systems = list(emitter.particle_systems)
        emitter_eval = emitter.evaluated_get(depsgraph)
        eval_systems = list(emitter_eval.particle_systems)
        for index, psys in enumerate(systems):
            settings = psys.settings
            if settings.type != "HAIR":
                continue
            source = eval_systems[index] if index < len(eval_systems) else psys
            render_step = int(getattr(settings, "render_step", 0) or 0)
            if render_step <= 0:
                render_step = int(getattr(settings, "hair_step", 1) or 1)
            render_step = min(max(render_step, 1), 20)
            max_step = 1 << render_step
            sample_count = max(3, min(max_step + 1, 64))
            root_width = max(settings.radius_scale * settings.root_radius, 0.001)
            tip_width = max(settings.radius_scale * settings.tip_radius, root_width * 0.05)
            if not getattr(settings, "use_close_tip", False):
                tip_width = max(tip_width, root_width * 0.2)

            inv = emitter_eval.matrix_world.inverted() if emitter_eval.matrix_world else None
            points = []
            vertex_counts = []
            parents = len(getattr(source, "particles", []))
            children = len(getattr(source, "child_particles", []))
            if parents and children and hasattr(source, "co_hair"):
                steps = np.rint(np.linspace(0, max_step, sample_count)).astype(int)
                for strand in range(parents + children):
                    strand_pts = []
                    total = 0.0
                    ok = True
                    for step in steps:
                        try:
                            co = source.co_hair(object=emitter_eval, particle_no=strand, step=int(step))
                        except Exception:
                            ok = False
                            break
                        if inv is not None:
                            co = inv @ co
                        strand_pts.append((co[0], co[1], co[2]))
                        total += abs(co[0]) + abs(co[1]) + abs(co[2])
                    if ok and total > 0.0:
                        points.extend(strand_pts)
                        vertex_counts.append(sample_count)
            if not vertex_counts:
                for particle in source.particles:
                    keys = particle.hair_keys
                    if len(keys) < 2:
                        continue
                    for key in keys:
                        co = key.co
                        points.append((co[0], co[1], co[2]))
                    vertex_counts.append(len(keys))
            if not vertex_counts:
                continue

            curves_data = bpy.data.hair_curves.new(emitter.name + " Fixture Hair")
            curves_data.add_curves(vertex_counts)
            flat = []
            for point in points:
                flat.extend((point[0], point[1], point[2]))
            curves_data.points.foreach_set("position", flat)
            radii = []
            for count in vertex_counts:
                for i in range(count):
                    t = i / (count - 1) if count > 1 else 0.0
                    radii.append((root_width * (1.0 - t) + tip_width * t) * 0.5)
            curves_data.points.foreach_set("radius", radii)

            curves = bpy.data.objects.new(curves_data.name, curves_data)
            curves.matrix_world = emitter.matrix_world.copy()
            bpy.context.scene.collection.objects.link(curves)
            material_index = max(0, int(settings.material) - 1)
            if material_index < len(emitter.material_slots):
                material = emitter.material_slots[material_index].material
                if material is not None:
                    curves.data.materials.append(material)


def prepare_classroom_deformed_geometry():
    if not {classroom_material_compatibility!r}:
        return
    # Stock skeleton export does not reproduce the posed blind slats in OVRTX.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not any(
            slot.material is not None and slot.material.name == "paintedBlind"
            for slot in obj.material_slots
        ):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        baked = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        obj.modifiers.clear()
        obj.data = baked


emission_strengths = {{}}
transparent_materials = set()
legacy_texture_materials = set()
classroom_complex_texture_files = {{
    "LeatherBook_cover": "base_leather.jpg",
    "beigePaint": "base_paintedPlasterWall.jpg",
    "beigePaintedPlastic": "base_paintedPlasterWall.jpg",
    "beigePaintedwood": "base_paintedPlasterWall.jpg",
    "beige_paintedPipe": "base_paintedPlasterWall.jpg",
    "crinkledPaper_paper": "crinkledPaper.png",
    "leatherCoat": "base_leather.jpg",
    "suitCase_leather": "base_leather.jpg",
    "varnishedWoodDoor": "base_paintedPlasterWall.jpg",
    "woodPlanks": "woodPlanks.jpg",
}}
classroom_roughness_overrides = {{"woodPlanks": 0.6223329901695251}}


def image_filename(image):
    if image is None:
        return ""
    path = image.filepath_raw or image.filepath or ""
    return path.replace("\\\\", "/").rstrip("/").rsplit("/", 1)[-1]


def connected_shader_nodes(surface):
    connected = []
    pending = [surface.links[0].from_node]
    while pending:
        node = pending.pop()
        if node in connected:
            continue
        connected.append(node)
        pending.extend(
            link.from_node
            for socket in node.inputs
            if socket.type == "SHADER"
            for link in socket.links
        )
    return connected


def image_socket_for_filename(socket, filename, seen=None):
    seen = set() if seen is None else seen
    for link in socket.links:
        source_socket = link.from_socket
        node = source_socket.node
        if node in seen:
            continue
        seen.add(node)
        image = node.image if node.type == "TEX_IMAGE" else None
        if image_filename(image) == filename:
            return node.outputs.get("Color") or source_socket
        for input_socket in node.inputs:
            image_socket = image_socket_for_filename(input_socket, filename, seen)
            if image_socket is not None:
                return image_socket
    return None


def prepare_classroom_materials():
    if not {classroom_material_compatibility!r}:
        return
    for material in bpy.data.materials:
        node_tree = material.node_tree
        if node_tree is None:
            continue
        outputs = [
            node
            for node in node_tree.nodes
            if node.type == "OUTPUT_MATERIAL" and node.is_active_output
        ]
        if not outputs:
            continue
        output = outputs[0]
        surface = output.inputs.get("Surface")
        if surface is None or not surface.is_linked:
            continue
        connected_nodes = connected_shader_nodes(surface)
        principled = next(
            (node for node in connected_nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        root_shader = surface.links[0].from_node
        fallback = None
        if principled is None and root_shader.type in {{"BSDF_DIFFUSE", "BSDF_GLOSSY", "BSDF_GLASS"}}:
            fallback = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            color = root_shader.inputs.get("Color")
            roughness = root_shader.inputs.get("Roughness")
            if color is not None:
                fallback.inputs["Base Color"].default_value = color.default_value
                if color.is_linked and color.links[0].from_node.type == "TEX_IMAGE":
                    node_tree.links.new(color.links[0].from_socket, fallback.inputs["Base Color"])
            if roughness is not None:
                fallback.inputs["Roughness"].default_value = roughness.default_value
            if root_shader.type == "BSDF_GLOSSY":
                fallback.inputs["Metallic"].default_value = 1.0
            elif root_shader.type == "BSDF_GLASS":
                fallback.inputs["Alpha"].default_value = 0.0
                fallback.inputs["IOR"].default_value = root_shader.inputs["IOR"].default_value
                material.surface_render_method = "DITHERED"
                transparent_materials.add(material.name)
        elif principled is None and root_shader.type == "GROUP":
            fallback = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            fallback.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            fallback.inputs["Roughness"].default_value = 0.5
        elif principled is None and root_shader.type == "MIX_SHADER":
            diffuse_nodes = [
                node for node in connected_nodes if node.type == "BSDF_DIFFUSE"
            ]
            glossy = next(
                (node for node in connected_nodes if node.type == "BSDF_GLOSSY"),
                None,
            )
            transparent = next(
                (node for node in connected_nodes if node.type == "BSDF_TRANSPARENT"),
                None,
            )
            direct_diffuse = next(
                (
                    (node, node.inputs["Color"].links[0].from_socket)
                    for node in diffuse_nodes
                    if node.inputs["Color"].is_linked
                    and node.inputs["Color"].links[0].from_node.type == "TEX_IMAGE"
                ),
                None,
            )
            expected_file = classroom_complex_texture_files.get(material.name)
            explicit_diffuse = None
            if direct_diffuse is None and expected_file is not None:
                explicit_diffuse = next(
                    (
                        (node, image_socket)
                        for node in diffuse_nodes
                        if (
                            image_socket := image_socket_for_filename(
                                node.inputs["Color"], expected_file
                            )
                        ) is not None
                    ),
                    None,
                )
            textured_diffuse = direct_diffuse or explicit_diffuse
            diffuse = textured_diffuse[0] if textured_diffuse else next(iter(diffuse_nodes), None)
            if diffuse is not None:
                fallback = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                fallback.inputs["Base Color"].default_value = diffuse.inputs["Color"].default_value
                image_socket = textured_diffuse[1] if textured_diffuse else None
                if image_socket is not None:
                    node_tree.links.new(image_socket, fallback.inputs["Base Color"])
                    if direct_diffuse is None:
                        legacy_texture_materials.add(material.name)
                matching_glossy = None
                if image_socket is not None:
                    image = image_socket.node.image
                    image_path_name = image_filename(image)
                    matching_glossy = next(
                        (
                            node
                            for node in connected_nodes
                            if node.type == "BSDF_GLOSSY"
                            and node.inputs.get("Color") is not None
                            and (
                                glossy_image := image_socket_for_filename(
                                    node.inputs["Color"], image_path_name
                                )
                            ) is not None
                            and glossy_image.node == image_socket.node
                        ),
                        None,
                    )
                roughness_source = matching_glossy or glossy
                if roughness_source is None:
                    roughness_source = next(
                        (
                            node
                            for node in connected_nodes
                            if node.type == "GROUP"
                            and node.inputs.get("Roughness") is not None
                        ),
                        None,
                    )
                fallback.inputs["Roughness"].default_value = (
                    roughness_source.inputs["Roughness"].default_value
                    if roughness_source is not None
                    else classroom_roughness_overrides.get(material.name, 0.5)
                )
                if transparent is not None:
                    fallback.inputs["Alpha"].default_value = 0.0
                    material.surface_render_method = "DITHERED"
                    transparent_materials.add(material.name)
            elif glossy is not None and transparent is not None:
                fallback = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                fallback.inputs["Base Color"].default_value = glossy.inputs["Color"].default_value
                fallback.inputs["Roughness"].default_value = glossy.inputs["Roughness"].default_value
                fallback.inputs["Alpha"].default_value = 0.0
                material.surface_render_method = "DITHERED"
                transparent_materials.add(material.name)
        if fallback is not None:
            while surface.links:
                node_tree.links.remove(surface.links[0])
            node_tree.links.new(fallback.outputs["BSDF"], surface)
            for node in connected_nodes:
                if node != fallback:
                    node_tree.nodes.remove(node)
            principled = fallback
        if principled is not None and (
            material.name == "worldMap" or material.name.startswith("drawing.")
        ):
            # The accepted workload renders these display surfaces neutral.
            base_color = principled.inputs.get("Base Color")
            if base_color is not None and base_color.is_linked:
                node_tree.links.remove(base_color.links[0])


def prepare_stock_materials():
    for material in bpy.data.materials:
        node_tree = material.node_tree
        if node_tree is None:
            continue
        outputs = [node for node in node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output]
        if not outputs:
            continue
        output = outputs[0]
        surface = output.inputs.get("Surface")
        volume = output.inputs.get("Volume")
        if (
            {evaluated_geometry!r}
            and material.name == "Smoke"
            and surface is not None
            and not surface.is_linked
            and volume is not None
            and volume.is_linked
        ):
            fallback = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            fallback.name = "Fixture Volume Surface"
            fallback.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            fallback.inputs["Roughness"].default_value = 0.5
            node_tree.links.new(fallback.outputs["BSDF"], surface)
        if surface is None or not surface.is_linked:
            continue

        connected_nodes = connected_shader_nodes(surface)
        principled = next(
            (node for node in connected_nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if principled is None:
            continue

        root_shader = surface.links[0].from_node
        shader_types = {{node.type for node in connected_nodes if any(output.type == "SHADER" for output in node.outputs)}}
        supported_mix_types = {{"MIX_SHADER", "BSDF_PRINCIPLED", "BSDF_TRANSPARENT", "BSDF_HAIR"}}
        if root_shader.type == "ADD_SHADER" or (
            root_shader.type == "MIX_SHADER" and shader_types <= supported_mix_types
        ):
            node_tree.links.new(principled.outputs["BSDF"], surface)

        base_color = principled.inputs.get("Base Color")
        if base_color is not None and base_color.is_linked:
            source_node = base_color.links[0].from_node
            if (
                source_node.type == "MIX"
                and source_node.data_type == "RGBA"
                and source_node.blend_type == "MIX"
            ):
                factor = next(
                    (socket for socket in source_node.inputs if socket.name == "Factor" and socket.is_linked),
                    None,
                )
                color_a = next(
                    (
                        socket
                        for socket in source_node.inputs
                        if socket.name == "A" and socket.type == "RGBA" and socket.is_linked
                    ),
                    None,
                )
                color_b = next(
                    (socket for socket in source_node.inputs if socket.name == "B" and socket.type == "RGBA"),
                    None,
                )
                if factor is not None and color_a is not None and color_b is not None and not color_b.is_linked:
                    node_tree.links.new(color_a.links[0].from_socket, base_color)

        transmission = principled.inputs.get("Transmission Weight")
        if transmission is not None and transmission.is_linked:
            invert = transmission.links[0].from_node
            invert_color = invert.inputs.get("Color") if invert.type == "INVERT" else None
            invert_factor = None
            if invert.type == "INVERT":
                invert_factor = invert.inputs.get("Factor") or invert.inputs.get("Fac")
            alpha = principled.inputs.get("Alpha")
            texture = invert_color.links[0].from_node if invert_color is not None and invert_color.is_linked else None
            image = texture.image if texture is not None and texture.type == "TEX_IMAGE" else None
            image_identity = f"{{image.name}} {{image.filepath}}".lower() if image is not None else ""
            if (
                invert_factor is not None
                and not invert_factor.is_linked
                and abs(float(invert_factor.default_value) - 1.0) <= 1.0e-6
                and "opacity" in image_identity
                and alpha is not None
            ):
                node_tree.links.new(invert_color.links[0].from_socket, alpha)
                node_tree.links.remove(transmission.links[0])
                transmission.default_value = 0.0

        legacy_emission = next(
            (node for node in connected_nodes if node.type == "EMISSION"),
            None,
        )
        emission_color = principled.inputs.get("Emission Color")
        emission_strength = principled.inputs.get("Emission Strength")
        if legacy_emission is not None and emission_color is not None:
            legacy_color = legacy_emission.inputs.get("Color")
            if legacy_color is not None and legacy_color.is_linked:
                node_tree.links.new(legacy_color.links[0].from_socket, emission_color)
            elif legacy_color is not None:
                emission_color.default_value = legacy_color.default_value
            if emission_strength is not None:
                strength = float(legacy_emission.inputs["Strength"].default_value)
                emission_strength.default_value = strength
                emission_strengths[material.name] = strength


prepare_particle_hair()
prepare_classroom_deformed_geometry()
prepare_classroom_materials()
prepare_stock_materials()
scene = bpy.context.scene
if scene.camera is None:
    cameras = sorted((obj for obj in scene.objects if obj.type == "CAMERA"), key=lambda obj: obj.name)
    if cameras:
        scene.camera = cameras[0]
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.resolution_percentage = 100

result = bpy.ops.wm.usd_export(
    filepath=str(raw_target),
    selected_objects_only=False,
    export_animation=False,
    export_hair=False,
    export_uvmaps=True,
    export_mesh_colors=True,
    export_normals=True,
    export_materials=True,
    export_armatures=True,
    export_shapekeys=True,
    use_instancing=False,
    evaluation_mode={"VIEWPORT" if evaluated_geometry or classroom_material_compatibility else "RENDER"!r},
    generate_preview_surface=True,
    generate_materialx_network=False,
    convert_orientation=False,
    export_textures_mode="NEW",
    overwrite_textures=False,
    relative_paths=True,
    root_prim_path="/World",
    export_custom_properties=True,
    author_blender_name=True,
    convert_world_material=True,
    export_meshes=True,
    export_lights=True,
    export_cameras=True,
    export_curves=True,
    export_points=True,
    export_volumes=True,
    export_subdivision={"TESSELLATE" if evaluated_geometry else "IGNORE" if classroom_material_compatibility else "BEST_MATCH"!r},
    triangulate_meshes={evaluated_geometry or classroom_material_compatibility!r},
    quad_method={"FIXED" if evaluated_geometry or classroom_material_compatibility else "SHORTEST_DIAGONAL"!r},
    ngon_method={"CLIP" if evaluated_geometry or classroom_material_compatibility else "BEAUTY"!r},
    convert_scene_units="METERS",
)
if set(result) != {{"FINISHED"}}:
    raise RuntimeError(f"stock Blender USD export failed: {{result}}")

raw_stage = Usd.Stage.Open(str(raw_target))
if raw_stage is None:
    raise RuntimeError(f"failed to open stock Blender USD export: {{raw_target}}")
if {classroom_material_compatibility!r}:
    for prim in raw_stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            prim.RemoveProperty("subdivisionScheme")
if {evaluated_geometry!r}:
    for prim in raw_stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDoubleSidedAttr(False).Set(False)
            for property_name in (
                "subdivisionScheme",
                "creaseIndices",
                "creaseLengths",
                "creaseSharpnesses",
            ):
                prim.RemoveProperty(property_name)
            continue
        if prim.GetTypeName() != "BasisCurves":
            continue
        curves = UsdGeom.BasisCurves(prim)
        vertex_counts = curves.GetCurveVertexCountsAttr().Get() or []
        if len(vertex_counts) >= 1000 and all(int(count) == 12 for count in vertex_counts):
            curves.GetTypeAttr().Set(UsdGeom.Tokens.cubic)
            curves.GetBasisAttr().Set(UsdGeom.Tokens.catmullRom)
for material_prim in raw_stage.Traverse():
    if material_prim.GetTypeName() != "Material":
        continue
    name_attr = material_prim.GetAttribute("userProperties:blender:data_name")
    material_name = name_attr.Get() if name_attr else None
    strength = emission_strengths.get(material_name)
    for shader_prim in material_prim.GetChildren():
        shader_id = shader_prim.GetAttribute("info:id")
        if not shader_id or shader_id.Get() != "UsdPreviewSurface":
            continue
        if material_name in transparent_materials:
            opacity = shader_prim.CreateAttribute(
                "inputs:opacity", Sdf.ValueTypeNames.Float, custom=False
            )
            opacity.ClearConnections()
            opacity.Set(0.0)
        if material_name in legacy_texture_materials:
            displacement_output = material_prim.GetAttribute("outputs:displacement")
            if displacement_output:
                displacement_output.ClearConnections()
            diffuse = shader_prim.GetAttribute("inputs:diffuseColor")
            connections = diffuse.GetConnections() if diffuse else []
            texture_prim = raw_stage.GetPrimAtPath(connections[0].GetPrimPath()) if connections else None
            texture_id = texture_prim.GetAttribute("info:id") if texture_prim else None
            if texture_id and texture_id.Get() == "UsdUVTexture":
                for property_name in ("inputs:bias", "inputs:scale", "inputs:sourceColorSpace"):
                    texture_prim.RemoveProperty(property_name)
                primvar_reader = next(
                    (
                        child
                        for child in Usd.PrimRange(material_prim)
                        if (
                            child.GetAttribute("info:id")
                            and child.GetAttribute("info:id").Get() == "UsdPrimvarReader_float2"
                        )
                    ),
                    None,
                )
                if primvar_reader is not None:
                    texture_prim.GetAttribute("inputs:st").SetConnections(
                        [primvar_reader.GetPath().AppendProperty("outputs:result")]
                    )
            shader_prim.RemoveProperty("inputs:displacement")
            for property_name in (
                "inputs:clearcoat",
                "inputs:clearcoatRoughness",
                "inputs:specular",
            ):
                shader_prim.RemoveProperty(property_name)
            if material_name not in transparent_materials:
                shader_prim.RemoveProperty("inputs:ior")
                shader_prim.RemoveProperty("inputs:opacity")
        if strength is None or abs(strength - 1.0) <= 1.0e-6:
            continue
        emissive = shader_prim.GetAttribute("inputs:emissiveColor")
        connections = emissive.GetConnections() if emissive else []
        if not connections:
            continue
        texture_prim = raw_stage.GetPrimAtPath(connections[0].GetPrimPath())
        texture_id = texture_prim.GetAttribute("info:id") if texture_prim else None
        if texture_id and texture_id.Get() == "UsdUVTexture":
            scale = texture_prim.CreateAttribute("inputs:scale", Sdf.ValueTypeNames.Float4, custom=False)
            scale.Set((strength, strength, strength, 1.0))
raw_stage.GetRootLayer().Save()
del raw_stage

source_layer = Sdf.Layer.FindOrOpen(str(raw_target))
if source_layer is None:
    raise RuntimeError(f"failed to open stock Blender USD export: {{raw_target}}")
target.unlink(missing_ok=True)
target_layer = Sdf.Layer.CreateNew(str(target))
if target_layer is None:
    raise RuntimeError(f"failed to create canonical fixture layer: {{target}}")
for key in source_layer.pseudoRoot.ListInfoKeys():
    target_layer.pseudoRoot.SetInfo(key, source_layer.pseudoRoot.GetInfo(key))


def canonical_uv_value(value):
    if isinstance(value, float):
        return round(value, 3)
    if type(value).__module__ in {{"pxr.Gf", "pxr.Vt"}}:
        return type(value)([canonical_uv_value(item) for item in value])
    return value


def is_redundant_curve_container(source_prim):
    children = list(source_prim.nameChildren.values())
    if source_prim.typeName != "Xform" or len(children) != 1:
        return False
    curve = children[0]
    if curve.typeName != "BasisCurves":
        return False
    for prop in source_prim.properties:
        if prop.name == "xformOpOrder":
            continue
        if not prop.name.startswith("xformOp:"):
            continue
        value = prop.default
        if prop.name == "xformOp:scale":
            if value is None or any(abs(float(item) - 1.0) > 1e-5 for item in value):
                return False
        elif value is None or any(abs(float(item)) > 1e-5 for item in value):
            return False
    parent = source_prim.nameParent
    if parent is None:
        return False
    sibling = parent.nameChildren.get(curve.name)
    if sibling is None or sibling.typeName != "BasisCurves":
        return False
    for name in ("curveVertexCounts", "points", "widths"):
        curve_property = source_layer.GetPropertyAtPath(curve.path.AppendProperty(name))
        sibling_property = source_layer.GetPropertyAtPath(sibling.path.AppendProperty(name))
        if curve_property is None or sibling_property is None:
            return False
        if curve_property.default != sibling_property.default:
            return False
    return True


def copy_prim(source_prim):
    if is_redundant_curve_container(source_prim):
        return
    target_prim = Sdf.CreatePrimInLayer(target_layer, source_prim.path)
    for key in source_prim.ListInfoKeys():
        target_prim.SetInfo(key, source_prim.GetInfo(key))
    for source_property in sorted(source_prim.properties, key=lambda item: item.name):
        if not Sdf.CopySpec(source_layer, source_property.path, target_layer, source_property.path):
            raise RuntimeError(f"failed to copy USD property {{source_property.path}}")
        if source_property.name == "primvars:st" and source_property.default is not None:
            target_property = target_layer.GetPropertyAtPath(source_property.path)
            target_property.default = canonical_uv_value(source_property.default)
    for child in sorted(source_prim.nameChildren.values(), key=lambda item: item.name):
        copy_prim(child)


for root_prim in sorted(source_layer.rootPrims, key=lambda item: item.name):
    copy_prim(root_prim)
target_layer.Save()
raw_target.unlink()
"""
    stdout = _run_blender_script(blender, script)
    if not temporary_path.is_file():
        raise RuntimeError(f"Blender fixture export did not write {temporary_path}\n{stdout}")
    os.replace(temporary_path, usd_path)


def _inspect_usd(blender: str, usd_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ovrtx-fixture-inspect-") as temp_dir:
        output = Path(temp_dir) / "inspection.json"
        script = f"""
import json
import math
from pathlib import Path
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open({str(usd_path)!r})
if stage is None:
    raise RuntimeError("failed to open USD")
cameras = []
camera_details = {{}}
lights = []
render_products = []
render_product_cameras = {{}}
default_prim = stage.GetDefaultPrim()
bbox_prim = default_prim if default_prim else stage.GetPseudoRoot()
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    useExtentsHint=True,
)
box = bbox_cache.ComputeWorldBound(bbox_prim).ComputeAlignedBox()
bounds = None
if not box.IsEmpty():
    min_v = box.GetMin()
    max_v = box.GetMax()
    bounds = {{
        "min": [float(min_v[i]) for i in range(3)],
        "max": [float(max_v[i]) for i in range(3)],
        "center": [float((min_v[i] + max_v[i]) * 0.5) for i in range(3)],
        "size": [float(max_v[i] - min_v[i]) for i in range(3)],
    }}

def _float_attr(prim, name, default):
    attr = prim.GetAttribute(name)
    value = attr.Get() if attr else None
    return float(value) if value is not None else float(default)

def _basis_length(matrix, row):
    return math.sqrt(sum(float(matrix[row][col]) * float(matrix[row][col]) for col in range(3)))

def _light_emitter_area(prim):
    type_name = prim.GetTypeName()
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    sx = _basis_length(matrix, 0)
    sy = _basis_length(matrix, 1)
    if type_name == "RectLight":
        return max(_float_attr(prim, "inputs:width", 1.0) * sx * _float_attr(prim, "inputs:height", 1.0) * sy, 1.0e-6)
    if type_name == "DiskLight":
        radius = _float_attr(prim, "inputs:radius", 0.5)
        return max(math.pi * (radius * sx) * (radius * sy), 1.0e-6)
    if type_name == "SphereLight":
        radius = max(_float_attr(prim, "inputs:radius", 0.5), 1.0e-3)
        return max(4.0 * math.pi * radius * radius, 1.0e-6)
    if type_name == "CylinderLight":
        radius = _float_attr(prim, "inputs:radius", 0.5) * sy
        length = _float_attr(prim, "inputs:length", 1.0) * sx
        return max(2.0 * math.pi * radius * length, 1.0e-6)
    return 1.0

for prim in stage.Traverse():
    type_name = prim.GetTypeName()
    path = str(prim.GetPath())
    if type_name == "Camera":
        cameras.append(path)
        camera = UsdGeom.Camera(prim)
        clipping = camera.GetClippingRangeAttr().Get()
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_details[path] = {{
            "clipping_range": [float(clipping[0]), float(clipping[1])] if clipping is not None else [0.1, 100000.0],
            "focal_length": float(camera.GetFocalLengthAttr().Get() or 24.0),
            "focus_distance": float(camera.GetFocusDistanceAttr().Get() or 0.0),
            "f_stop": float(camera.GetFStopAttr().Get() or 0.0),
            "horizontal_aperture": float(camera.GetHorizontalApertureAttr().Get() or 36.0),
            "transform": [[float(matrix[row][col]) for col in range(4)] for row in range(4)],
            "vertical_aperture": float(camera.GetVerticalApertureAttr().Get() or 20.25),
        }}
    elif type_name.endswith("Light"):
        intensity = None
        attr = prim.GetAttribute("inputs:intensity")
        if attr:
            value = attr.Get()
            if value is not None:
                intensity = float(value)
        lights.append({{
            "path": path,
            "type_name": type_name,
            "intensity": intensity,
            "emitter_area": _light_emitter_area(prim),
        }})
    elif type_name == "RenderProduct":
        render_products.append(path)
        rel = prim.GetRelationship("camera")
        if rel:
            render_product_cameras[path] = [str(target) for target in rel.GetTargets()]
Path({str(output)!r}).write_text(json.dumps({{
    "bounds": bounds,
    "camera_details": camera_details,
    "cameras": sorted(cameras),
    "lights": sorted(lights, key=lambda item: item["path"]),
    "render_products": sorted(render_products),
    "render_product_cameras": render_product_cameras,
}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
"""
        stdout = _run_blender_script(blender, script)
        if not output.is_file():
            raise RuntimeError(f"USD inspection did not write {output}\n{stdout}")
        return json.loads(output.read_text(encoding="utf-8"))


def _run_blender_script(blender: str, script: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as file:
        file.write(script)
        script_path = file.name
    try:
        completed = subprocess.run(
            [blender, "--background", "--factory-startup", "--threads", "1", "--disable-autoexec", "--python", script_path],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONHASHSEED": "0"},
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
        if completed.returncode != 0 or "Traceback (most recent call last):" in completed.stdout:
            raise RuntimeError(completed.stdout)
        return completed.stdout
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _choose_camera(fixture: Mapping[str, Any], inspection: Mapping[str, Any], render_product_path: str) -> str:
    cameras = [str(path) for path in inspection.get("cameras", [])]
    product_cameras = inspection.get("render_product_cameras", {}).get(render_product_path, [])
    for path in product_cameras:
        if str(path) in cameras:
            return str(path)
    candidate = str(fixture.get("camera_prim_path", UNKNOWN))
    if candidate != UNKNOWN and candidate in cameras:
        return candidate
    for path in cameras:
        if path.endswith("/Camera") or path.endswith("/Camera0"):
            return path
    if cameras:
        return sorted(cameras)[0]
    return "/OvrtxCamera"


def _choose_render_product(fixture: Mapping[str, Any], inspection: Mapping[str, Any]) -> str:
    products = [str(path) for path in inspection.get("render_products", [])]
    candidate = str(fixture.get("render_product_prim_path", DEFAULT_RENDER_PRODUCT))
    if candidate != UNKNOWN and candidate in products:
        return candidate
    if products:
        return sorted(products)[0]
    return candidate if candidate != UNKNOWN else DEFAULT_RENDER_PRODUCT


def _render_product_targets_camera(
    inspection: Mapping[str, Any],
    render_product_path: str,
    camera_path: str,
) -> bool:
    cameras = {str(path) for path in inspection.get("cameras", [])}
    if camera_path not in cameras:
        return False
    product_cameras = inspection.get("render_product_cameras", {}).get(render_product_path, [])
    return camera_path in {str(path) for path in product_cameras}


def _fixture_camera_definition(fixture: Mapping[str, Any]) -> dict[str, Any] | None:
    value = fixture.get("camera_definition")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} camera_definition must be an object")
    return dict(value)


def _fixture_camera_details(fixture_id: str, camera_details: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if camera_details is None:
        return None
    result = dict(camera_details)
    return result


def _fixture_definition(fixture: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = fixture.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"fixture {fixture.get('id', UNKNOWN)} {key} must be an object")
    return dict(value)


def _fixture_lighting_definition(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return _fixture_definition(fixture, "lighting_definition")


def _fixture_render_product_definition(fixture: Mapping[str, Any]) -> dict[str, Any]:
    definition = _fixture_definition(fixture, "render_product_definition")
    if "resolution" in definition:
        raise ValueError("render_product_definition must not duplicate target_resolution")
    return definition


def _write_usd_stage(
    usd_path: Path,
    source_usd: Path,
    camera_path: str,
    render_product_path: str,
    *,
    camera_details: Mapping[str, Any] | None,
    add_camera: bool,
    add_render_product: bool,
    add_fixture_definitions: bool,
    lighting_definition: Mapping[str, Any],
    render_product_definition: Mapping[str, Any],
    width: int,
    height: int,
    source_lights: Sequence[Mapping[str, Any]],
    source_lights_are_policy_converted: bool,
    fixture_block: str = "",
) -> None:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    camera_block = _camera_block(camera_path, camera_details, render_product_definition) if add_camera else ""
    dome_block = _dome_light_block(lighting_definition) if add_fixture_definitions else ""
    fill_light_block = _fill_light_block(lighting_definition) if add_fixture_definitions else ""
    override_tree: dict[str, Any] = {}
    if add_fixture_definitions and not bool(lighting_definition.get("skip_source_light_overrides", False)):
        _add_light_overrides(
            override_tree,
            source_lights,
            _source_light_intensity_multiplier(lighting_definition),
            apply_conversion_scale=not source_lights_are_policy_converted,
        )
    overrides = _emit_override_tree(override_tree, 0)
    render_block = _render_block(
        camera_path,
        render_product_path,
        lighting_definition,
        render_product_definition,
        width=width,
        height=height,
    ) if add_render_product else ""
    usd_path.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        "    metersPerUnit = 1.0\n"
        '    upAxis = "Z"\n'
        f"    subLayers = [@{_usd_sublayer_path(source_usd, usd_path)}@]\n"
        ")\n\n"
        f"{camera_block}"
        f"{dome_block}"
        f"{fill_light_block}"
        f"{overrides}"
        f"{fixture_block}"
        f"{render_block}",
        encoding="utf-8",
    )


def _classroom_repair_block(
    source_blend: Path | None,
    source_usd: Path,
    *,
    source_lights: Sequence[Mapping[str, Any]] = (),
    lighting_definition: Mapping[str, Any] | None = None,
) -> str:
    if source_blend is None or not source_blend.is_file():
        raise RuntimeError("Classroom beauty repair source .blend is unavailable")
    script = f"""
import json
import math
import sys

import bpy
from pxr import Usd, UsdGeom

sys.path.insert(0, {str(ADDON_ROOT)!r})
from ovrtx_blender_example import materialx_openpbr_conversion as conversion

bpy.ops.wm.open_mainfile(filepath={str(source_blend)!r})
scene_objects = set(bpy.context.scene.objects)
lights = []
for obj in sorted((obj for obj in bpy.data.objects if obj.type == "LIGHT" and obj.data.type == "SPOT" and obj not in scene_objects), key=lambda item: item.name):
    light = obj.data
    lights.append({{
        "name": obj.name,
        "color": list(light.color),
        "radius": float(light.shadow_soft_size),
        "cone_angle": math.degrees(float(light.spot_size)) * 0.5,
        "cone_softness": float(light.spot_blend),
        "matrix": [list(row) for row in obj.matrix_basis],
    }})

stage = Usd.Stage.Open({str(source_usd)!r})
if stage is None:
    raise RuntimeError("failed to open normalized Classroom USD")
curves = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.BasisCurves):
        continue
    curve = UsdGeom.BasisCurves(prim)
    point_count = sum(curve.GetCurveVertexCountsAttr().Get() or [])
    width_count = len(curve.GetWidthsAttr().Get() or [])
    curves.append({{"path": str(prim.GetPath()), "point_count": point_count, "width_count": width_count}})

material_names = ("blackBoardLight", "dayLight_portal", "ceillingLamp_light")
materials = []
for name in material_names:
    material = bpy.data.materials.get(name)
    if material is None:
        raise RuntimeError(f"missing Classroom emission material: {{name}}")
    materials.append(material)
conversion._EMISSION_LUMINANCE_SCALE = 360.0 * math.pi
identity = conversion._materialx_binding_identity({str(source_usd)!r})
used_identifiers = set()
material_blocks = []
binding_records = []
binding_targets = []
expected_strengths = {{"blackBoardLight": 1.0, "dayLight_portal": 20.0, "ceillingLamp_light": 2.0}}
for material in materials:
    emission_nodes = [node for node in material.node_tree.nodes if node.type == "EMISSION"]
    if len(emission_nodes) != 1:
        raise RuntimeError(f"expected one emission node in {{material.name}}, found {{len(emission_nodes)}}")
    values = conversion._emission_openpbr_values(emission_nodes[0])
    strength = float(values["emission_luminance"]) / conversion._EMISSION_LUMINANCE_SCALE
    if not math.isclose(strength, expected_strengths[material.name], rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"unexpected emission strength for {{material.name}}: {{strength}}")
    source_material_path = conversion._material_path_from_name(material.name, identity)
    targets = conversion._valid_target_paths(
        binding["binding_target"]
        for binding in identity.get("bindings", ())
        if binding["material_path"] == source_material_path
    )
    if not targets:
        raise RuntimeError(f"no exported binding targets for {{material.name}}")
    material_id = conversion._unique_identifier(conversion._sanitize_identifier(material.name), used_identifiers)
    material_path = f"/OVRTX_Materials/{{material_id}}"
    material_blocks.append(conversion._material_block_lines(material_id, values))
    binding_records.extend((target, material_path) for target in targets)
    binding_targets.extend(targets)
material_overlay = conversion._overlay_body(
    material_blocks,
    [],
)
print("CLASSROOM_REPAIRS=" + json.dumps({{
    "lights": lights,
    "curves": curves,
    "material_overlay": material_overlay,
    "material_count": len(material_blocks),
    "binding_targets": binding_targets,
    "material_bindings": binding_records,
}}, sort_keys=True))
"""
    stdout = _run_blender_script(BLENDER_COMMAND, script)
    marker = "CLASSROOM_REPAIRS="
    line = next((line for line in stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        raise RuntimeError("Classroom repair inspection produced no result")
    return _classroom_repair_block_from_inspection(
        json.loads(line[len(marker):]),
        source_lights=source_lights,
        source_light_multiplier=_source_light_intensity_multiplier(lighting_definition or {}),
    )


def _classroom_repair_block_from_inspection(
    inspection: Mapping[str, Any],
    *,
    source_lights: Sequence[Mapping[str, Any]] = (),
    source_light_multiplier: float = 1.0,
) -> str:
    lights = list(inspection.get("lights", ()))
    curves = list(inspection.get("curves", ()))
    binding_targets = list(inspection.get("binding_targets", ()))
    material_bindings = list(inspection.get("material_bindings", ()))
    material_overlay = str(inspection.get("material_overlay", "")).strip()
    thin = [curve for curve in curves if int(curve.get("point_count", 0)) == 2]
    thick = [curve for curve in curves if int(curve.get("point_count", 0)) == 9]
    if len(lights) != 5:
        raise RuntimeError(f"expected 5 orphaned Classroom spotlights, found {len(lights)}")
    if len(curves) != 56 or len(thin) != 48 or len(thick) != 8:
        raise RuntimeError(
            f"expected 56 Classroom blind curves (48 thin, 8 thick), found {len(curves)} ({len(thin)} thin, {len(thick)} thick)"
        )
    if (
        int(inspection.get("material_count", 0)) != 3
        or len(binding_targets) != 9
        or len(material_bindings) != 9
        or not material_overlay
    ):
        raise RuntimeError(
            "expected 3 Classroom OpenPBR emission materials and 9 binding targets"
        )

    blocks = [material_overlay, ""]
    for light in lights:
        name = str(light["name"]).replace(".", "_")
        color = _float_tuple(light["color"], 3, (1.0, 1.0, 1.0))
        blocks.extend(
            [
                f'def SphereLight "Classroom_{name}"',
                "{",
                "    float inputs:intensity = 300000",
                f"    color3f inputs:color = {_format_tuple(color)}",
                f"    float inputs:radius = {_format_number(float(light['radius']))}",
                "    bool inputs:normalize = false",
                "    bool inputs:enableColorTemperature = false",
                f"    float shaping:cone:angle = {_format_number(float(light['cone_angle']))}",
                f"    float shaping:cone:softness = {_format_number(float(light['cone_softness']))}",
                f"    matrix4d xformOp:transform = {_format_matrix4d(light['matrix'])}",
                '    uniform token[] xformOpOrder = ["xformOp:transform"]',
                "}",
                "",
            ]
        )

    override_tree: dict[str, Any] = {}
    _add_light_overrides(
        override_tree,
        source_lights,
        source_light_multiplier,
        apply_conversion_scale=True,
    )
    for target_path, material_path in material_bindings:
        _insert_override(
            override_tree,
            _path_parts(str(target_path)),
            f"rel material:binding = <{material_path}>",
        )
    for curve in curves:
        width = 0.0016 if int(curve["point_count"]) == 2 else 0.01
        width_count = int(curve["width_count"])
        if width_count != int(curve["point_count"]):
            raise RuntimeError(f"unexpected Classroom curve width interpolation at {curve['path']}")
        values = ", ".join(_format_number(width) for _ in range(width_count))
        _insert_override(override_tree, _path_parts(str(curve["path"])), f"float[] widths = [{values}]")
    blocks.append(_emit_override_tree(override_tree, 0).rstrip())
    return "\n".join(blocks).rstrip() + "\n\n"


def _flatten_usd_stage(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    canonical = output.with_name(f".{output.stem}.canonical{output.suffix}")
    temporary.unlink(missing_ok=True)
    canonical.unlink(missing_ok=True)
    script = f"""
import os
from pathlib import Path

from pxr import Sdf, Usd

stage = Usd.Stage.Open({str(source)!r})
if stage is None:
    raise RuntimeError("failed to open composed fixture")
layer = stage.Flatten()
if layer is None or not layer.Export({str(temporary)!r}):
    raise RuntimeError("failed to flatten fixture")
flat_stage = Usd.Stage.Open({str(temporary)!r})
if flat_stage is None:
    raise RuntimeError("failed to reopen flattened fixture")
fixture_root = Path({str(output.parent.parent)!r}).resolve()
for prim in flat_stage.Traverse():
    for attr in prim.GetAttributes():
        if attr.GetTypeName() != Sdf.ValueTypeNames.Asset or not attr.HasAuthoredValue():
            continue
        value = attr.Get()
        asset_path = Path(value.path)
        if not asset_path.is_absolute():
            continue
        resolved = asset_path.resolve()
        if not resolved.is_relative_to(fixture_root):
            raise RuntimeError(f"flattened fixture asset escapes fixture root: {{resolved}}")
        relative_asset_path = Path(os.path.relpath(resolved, {str(output.parent)!r})).as_posix()
        attr.Set(Sdf.AssetPath(relative_asset_path))
flat_stage.GetRootLayer().Save()
source_layer = flat_stage.GetRootLayer()
canonical_layer = Sdf.Layer.CreateNew({str(canonical)!r})
if canonical_layer is None:
    raise RuntimeError("failed to create canonical flattened fixture")
for key in source_layer.pseudoRoot.ListInfoKeys():
    if key == "documentation":
        continue
    canonical_layer.pseudoRoot.SetInfo(key, source_layer.pseudoRoot.GetInfo(key))


def copy_prim(source_prim):
    target_prim = Sdf.CreatePrimInLayer(canonical_layer, source_prim.path)
    for key in source_prim.ListInfoKeys():
        target_prim.SetInfo(key, source_prim.GetInfo(key))
    for source_property in sorted(source_prim.properties, key=lambda item: item.name):
        if not Sdf.CopySpec(source_layer, source_property.path, canonical_layer, source_property.path):
            raise RuntimeError(f"failed to copy flattened USD property {{source_property.path}}")
    for child in sorted(source_prim.nameChildren.values(), key=lambda item: item.name):
        copy_prim(child)


for root_prim in sorted(source_layer.rootPrims, key=lambda item: item.name):
    copy_prim(root_prim)
canonical_layer.Save()
del flat_stage
os.replace({str(canonical)!r}, {str(temporary)!r})
"""
    _run_blender_script(BLENDER_COMMAND, script)
    if not temporary.is_file():
        raise RuntimeError("fixture flatten did not write its output")
    os.replace(temporary, output)


def _remove_normalized_source_asset(fixture: MutableMapping[str, Any]) -> None:
    fixture["asset_files"] = [
        asset
        for asset in _asset_files(fixture)
        if asset.get("label")
        not in {
            "Stock Blender normalized USD",
            "Blender exporter normalized USD",
            "Blender source exported USD",
        }
    ]


def _fallback_camera_details(bounds: Any) -> dict[str, Any] | None:
    if not isinstance(bounds, Mapping):
        return None
    try:
        center = [float(value) for value in bounds["center"]]
        size = [float(value) for value in bounds["size"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(center) != 3 or len(size) != 3:
        return None
    extent = max(size)
    if not math.isfinite(extent) or extent <= 0:
        return None

    eye = (
        center[0] + extent * 0.95,
        center[1] - extent * 1.35,
        center[2] + extent * 0.65,
    )
    transform = _look_at_matrix(eye, center)
    if transform is None:
        return None
    return {
        "clipping_range": [max(0.1, extent * 0.001), extent * 12.0],
        "focal_length": 24.0,
        "horizontal_aperture": 36.0,
        "transform": transform,
        "vertical_aperture": 20.25,
    }


def _look_at_matrix(eye: Sequence[float], target: Sequence[float]) -> list[list[float]] | None:
    world_up = (0.0, 0.0, 1.0)
    forward = _normalize((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    if forward is None:
        return None
    back = (-forward[0], -forward[1], -forward[2])
    right = _normalize(_cross(world_up, back))
    if right is None:
        right = _normalize((1.0, 0.0, 0.0))
    up = _cross(back, right)
    return [
        [right[0], right[1], right[2], 0.0],
        [up[0], up[1], up[2], 0.0],
        [back[0], back[1], back[2], 0.0],
        [eye[0], eye[1], eye[2], 1.0],
    ]


def _normalize(vector: Sequence[float]) -> tuple[float, float, float] | None:
    length = math.sqrt(sum(float(component) * float(component) for component in vector))
    if not math.isfinite(length) or length <= 1e-9:
        return None
    return tuple(float(component) / length for component in vector)


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dome_light_block(lighting_definition: Mapping[str, Any]) -> str:
    name = str(lighting_definition.get("dome_name", "AmbientDome")) or "AmbientDome"
    intensity = float(lighting_definition.get("dome_intensity", 200.0))
    color = _float_tuple(lighting_definition.get("dome_color"), 3, (1.0, 1.0, 1.0))
    return (
        f'def DomeLight "{name}"\n'
        "{\n"
        f"    float inputs:intensity = {intensity:.12g}\n"
        f"    color3f inputs:color = {_format_tuple(color)}\n"
        '    token inputs:texture:format = "latlong"\n'
        "}\n\n"
    )


def _source_light_intensity_multiplier(lighting_definition: Mapping[str, Any]) -> float:
    value = lighting_definition.get("source_light_intensity_multiplier", 1.0)
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source_light_intensity_multiplier must be a number: {value!r}") from exc
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"source_light_intensity_multiplier must be positive: {value!r}")
    return multiplier


def _light_overrides_block(
    lights: Sequence[Mapping[str, Any]],
    multiplier: float,
    *,
    apply_conversion_scale: bool = True,
) -> str:
    tree: dict[str, Any] = {}
    _add_light_overrides(tree, lights, multiplier, apply_conversion_scale=apply_conversion_scale)
    return _emit_override_tree(tree, 0)


def _add_light_overrides(
    tree: dict[str, Any],
    lights: Sequence[Mapping[str, Any]],
    multiplier: float,
    *,
    apply_conversion_scale: bool,
) -> None:
    for light in lights:
        path = str(light.get("path", ""))
        type_name = str(light.get("type_name", ""))
        intensity = light.get("intensity")
        if not path.startswith("/") or intensity is None:
            continue
        try:
            value = float(intensity)
        except (TypeError, ValueError):
            continue
        if not _supports_light_intensity_override(type_name):
            continue
        scale = _light_intensity_scale(light) if apply_conversion_scale else 1.0
        if scale == 1.0 and multiplier == 1.0:
            continue
        _insert_override(tree, _path_parts(path), f"float inputs:intensity = {value * scale * multiplier:.6g}")
        _insert_override(tree, _path_parts(path), "bool inputs:normalize = false")
        _insert_override(tree, _path_parts(path), "bool inputs:enableColorTemperature = false")


def _supports_light_intensity_override(type_name: str) -> bool:
    return type_name in {"RectLight", "SphereLight", "DiskLight", "CylinderLight", "DistantLight"}


def _light_intensity_scale(light: Mapping[str, Any]) -> float:
    type_name = str(light.get("type_name", ""))
    if type_name == "RectLight":
        return _light_policy.MEASURED_LIGHT_SCALE / _light_emitter_area(light)
    if type_name == "SphereLight":
        return _light_policy.MEASURED_LIGHT_SCALE / _light_emitter_area(light)
    if type_name == "DiskLight":
        return _light_policy.MEASURED_LIGHT_SCALE / _light_emitter_area(light)
    if type_name == "CylinderLight":
        return _light_policy.MEASURED_LIGHT_SCALE / _light_emitter_area(light)
    if type_name == "DistantLight":
        return 4.0 * _light_policy.MEASURED_LIGHT_SCALE
    return 1.0


def _light_emitter_area(light: Mapping[str, Any]) -> float:
    try:
        area = float(light.get("emitter_area", 1.0))
    except (TypeError, ValueError):
        area = 1.0
    if not math.isfinite(area) or area <= 0.0:
        return _light_policy.MIN_EMITTER_AREA
    return max(area, _light_policy.MIN_EMITTER_AREA)


def _insert_override(tree: dict[str, Any], parts: Sequence[str], attribute_line: str) -> None:
    if not parts:
        return
    children = tree
    node: dict[str, Any] | None = None
    for part in parts:
        node = children.setdefault(part, {"attrs": [], "children": {}})
        children = node["children"]
    if node is not None:
        node["attrs"].append(attribute_line)


def _emit_override_tree(tree: Mapping[str, Any], depth: int) -> str:
    lines: list[str] = []
    for name in sorted(tree):
        node = tree[name]
        indent = "    " * depth
        lines.append(f'{indent}over "{name}"\n{indent}{{\n')
        for attribute in node["attrs"]:
            lines.append(f"{'    ' * (depth + 1)}{attribute}\n")
        lines.append(_emit_override_tree(node["children"], depth + 1))
        lines.append(f"{indent}}}\n")
    if depth == 0 and lines:
        lines.append("\n")
    return "".join(lines)


def _fill_light_block(lighting_definition: Mapping[str, Any]) -> str:
    fill = lighting_definition.get("camera_fill_light")
    if not isinstance(fill, Mapping):
        return ""
    translate = _float_tuple(fill.get("translate"), 3, (0.0, 0.0, 0.0))
    intensity = float(fill.get("intensity", 0.0))
    if intensity <= 0.0:
        return ""
    radius = float(fill.get("radius", 1.0))
    color = _float_tuple(fill.get("color"), 3, (1.0, 1.0, 1.0))
    return (
        'def SphereLight "CameraFillLight"\n'
        "{\n"
        f"    float inputs:intensity = {intensity:.12g}\n"
        f"    float inputs:radius = {radius:.12g}\n"
        f"    color3f inputs:color = {_format_tuple(color)}\n"
        f"    double3 xformOp:translate = {_format_tuple(translate)}\n"
        '    uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        "}\n\n"
    )


def _camera_block(
    camera_path: str,
    camera_details: Mapping[str, Any] | None = None,
    render_product_definition: Mapping[str, Any] | None = None,
) -> str:
    render_product_definition = render_product_definition or {}
    parts = _path_parts(camera_path)
    if not parts:
        parts = ["OvrtxCamera"]
    lines: list[str] = []
    indent = ""
    for parent in parts[:-1]:
        lines.append(f'{indent}over "{parent}"\n{indent}{{\n')
        indent += "    "
    if camera_details:
        clipping = camera_details.get("clipping_range", [0.1, 100000.0])
        auto_exposure = bool(render_product_definition.get("auto_exposure", False))
        if auto_exposure:
            lines.extend(
                [
                    f'{indent}def Camera "{parts[-1]}" (\n',
                    f'{indent}    prepend apiSchemas = ["OmniRtxCameraAutoExposureAPI_1", "OmniRtxCameraExposureAPI_1"]\n',
                    f"{indent})\n",
                ]
            )
        else:
            lines.append(f'{indent}def Camera "{parts[-1]}"\n')
        lines.extend(
            [
                f"{indent}{{\n",
                f"{indent}    float focalLength = {float(camera_details.get('focal_length', 24.0)):.12g}\n",
                f"{indent}    float horizontalAperture = {float(camera_details.get('horizontal_aperture', 36.0)):.12g}\n",
                f"{indent}    float verticalAperture = {float(camera_details.get('vertical_aperture', 20.25)):.12g}\n",
                f"{indent}    float2 clippingRange = ({float(clipping[0]):.12g}, {float(clipping[1]):.12g})\n",
                f"{indent}    float fStop = {float(camera_details.get('f_stop', 0.0)):.12g}\n",
                f"{indent}    float focusDistance = {float(camera_details.get('focus_distance', 0.0)):.12g}\n",
            ]
        )
        if auto_exposure:
            speed = float(render_product_definition.get("auto_exposure_speed", 3.5))
            lines.extend(
                [
                    f"{indent}    bool omni:rtx:autoExposure:enabled = 1\n",
                    f"{indent}    float omni:rtx:autoExposure:adaptationSpeed = {speed:.12g}\n",
                    f"{indent}    float exposure:responsivity = 1.1\n",
                    f"{indent}    float exposure:time = 0.02\n",
                ]
            )
        lines.extend(
            [
                f"{indent}    matrix4d xformOp:transform = {_format_matrix4d(camera_details.get('transform'))}\n",
                f'{indent}    uniform token[] xformOpOrder = ["xformOp:transform"]\n',
                f"{indent}}}\n",
            ]
        )
    else:
        lines.extend(
            [
                f'{indent}def Camera "{parts[-1]}"\n',
                f"{indent}{{\n",
                f"{indent}    float focalLength = 24\n",
                f"{indent}    float horizontalAperture = 36\n",
                f"{indent}    float verticalAperture = 20.25\n",
                f"{indent}    float2 clippingRange = (1, 100000)\n",
                f"{indent}    double3 xformOp:translate = (250, -250, 150)\n",
                f"{indent}    float3 xformOp:rotateXYZ = (70, 0, 135)\n",
                f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n',
                f"{indent}}}\n",
            ]
        )
    for depth in range(len(parts) - 2, -1, -1):
        lines.append(f"{'    ' * depth}}}\n")
    lines.append("\n")
    return "".join(lines)


def _format_matrix4d(value: Any) -> str:
    rows = value if isinstance(value, (list, tuple)) else None
    if not rows or len(rows) != 4:
        rows = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    formatted_rows = []
    for row in rows:
        formatted_rows.append("(" + ", ".join(_format_number(float(cell)) for cell in row) + ")")
    return "(" + ", ".join(formatted_rows) + ")"


def _render_block(
    camera_path: str,
    render_product_path: str,
    lighting_definition: Mapping[str, Any],
    render_product_definition: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> str:
    parts = _path_parts(render_product_path)
    if len(parts) < 2:
        raise ValueError(f"render product path must be absolute and nested: {render_product_path}")
    ambient = _float_tuple(lighting_definition.get("ambient_light_color"), 3, (0.0, 0.0, 0.0))
    auto_exposure = bool(render_product_definition.get("auto_exposure", False))
    background_source = str(render_product_definition.get("background_source_type", "domeLight")) or "domeLight"
    lines: list[str] = []
    final_specifier = "over" if bool(render_product_definition.get("over_existing_render_product", False)) else "def RenderProduct"
    render_var_specifier = "over" if final_specifier == "over" else "def RenderVar"
    for index, part in enumerate(parts):
        indent = "    " * index
        if index == len(parts) - 1:
            lines.extend(
                [
                    f'{indent}{final_specifier} "{part}"\n',
                    f"{indent}{{\n",
                    f"{indent}    rel camera = <{camera_path}>\n",
                    f'{indent}    token omni:rtx:rendermode = "RealTimePathTracing"\n',
                    f'{indent}    token omni:rtx:background:source:type = "{background_source}"\n',
                    f"{indent}    rel orderedVars = [<{render_product_path}/LdrColor>, <{render_product_path}/HdrColor>]\n",
                    f"{indent}    uniform int2 resolution = ({width}, {height})\n",
                    f"{indent}    bool omni:rtx:indirectDiffuse:denoiser:enabled = {_json_bool(bool(render_product_definition.get('indirect_diffuse_denoiser', True)))}\n",
                    f"{indent}    bool omni:rtx:reflections:denoiser:enabled = {_json_bool(bool(render_product_definition.get('reflections_denoiser', True)))}\n",
                    f"{indent}    bool omni:rtx:dlss:frameGeneration = {_json_bool(bool(render_product_definition.get('dlss_frame_generation', True)))}\n",
                    f"{indent}    bool omni:rtx:autoExposure:enabled = {_json_bool(auto_exposure)}\n",
                    f"{indent}    bool omni:rtx:rt:ecoMode:enabled = false\n",
                    f"{indent}    int omni:rtx:rtpt:maxVolumeBounces = 4\n",
                ]
            )
            if auto_exposure:
                speed = float(render_product_definition.get("auto_exposure_speed", 3.5))
                lines.append(f"{indent}    float omni:rtx:autoExposure:adaptationSpeed = {speed:.12g}\n")
            lines.append(f"{indent}    color3f omni:rtx:rt:ambientLight:color = {_format_tuple(ambient)}\n")
            tone_exposure = render_product_definition.get("tone_exposure_key")
            if tone_exposure is not None:
                lines.append(f"{indent}    float omni:rtx:post:tonemap:exposureKey = {float(tone_exposure):.12g}\n")
            grade_gain = render_product_definition.get("grade_gain")
            grade_saturation = render_product_definition.get("grade_saturation")
            if grade_gain is not None or grade_saturation is not None:
                lines.append(f"{indent}    bool omni:rtx:post:grade:enabled = 1\n")
            if grade_saturation is not None:
                lines.append(f"{indent}    float omni:rtx:post:grade:saturation = {float(grade_saturation):.12g}\n")
            if grade_gain is not None:
                lines.append(f"{indent}    float3 omni:rtx:post:grade:gain = {_format_tuple(_float_tuple(grade_gain, 3, (1.0, 1.0, 1.0)))}\n")
            lines.extend(
                [
                    f'{indent}    {render_var_specifier} "LdrColor"\n',
                    f"{indent}    {{\n",
                    f'{indent}        uniform string sourceName = "LdrColor"\n',
                    f"{indent}    }}\n",
                    f'{indent}    {render_var_specifier} "HdrColor"\n',
                    f"{indent}    {{\n",
                    f'{indent}        uniform string sourceName = "HdrColor"\n',
                    f"{indent}    }}\n",
                ]
            )
            lines.append(f"{indent}}}\n")
        else:
            lines.append(f'{indent}def "{part}"\n{indent}{{\n')
    for depth in range(len(parts) - 2, 0, -1):
        lines.append(f"{'    ' * depth}}}\n")
    lines.append("}\n")
    return "".join(lines)


def _json_bool(value: bool) -> str:
    return "true" if value else "false"


def _float_tuple(value: Any, count: int, fallback: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == count:
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            pass
    return tuple(float(item) for item in fallback)


def _format_tuple(values: Sequence[float]) -> str:
    return "(" + ", ".join(_format_number(float(value)) for value in values) + ")"


def _format_number(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _path_parts(path: str) -> list[str]:
    return [part for part in path.strip("/").split("/") if part]


def _refresh_unresolved_values(fixture: MutableMapping[str, Any]) -> None:
    unresolved: list[str] = []
    if fixture.get("fixture_usd_path", UNKNOWN) == UNKNOWN:
        unresolved.append("fixture_usd_path: ???")
    if fixture.get("fixture_usd_sha256", UNKNOWN) == UNKNOWN:
        unresolved.append("fixture_usd_sha256: ???")
    if fixture.get("camera_prim_path", UNKNOWN) == UNKNOWN:
        unresolved.append("camera_prim_path: ???")
    if fixture.get("render_product_prim_path", UNKNOWN) == UNKNOWN:
        unresolved.append("render_product_prim_path: ???")
    fixture["unresolved_values"] = unresolved


def _resolve_manifest_path(value: str) -> Path | None:
    if not value or value == UNKNOWN:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _manifest_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _usd_asset_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _usd_sublayer_path(source_path: Path, usd_path: Path) -> str:
    try:
        relative = os.path.relpath(source_path.absolute(), usd_path.absolute().parent)
    except ValueError:
        return _usd_asset_path(source_path)
    return relative.replace("\\", "/")


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
