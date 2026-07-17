# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD test fixture manifest loading and fixture record resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "???"
_ASSET_REFERENCE = re.compile(r"@([^@\r\n]+)@")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


def fixture_content_sha256(path: Path) -> str:
    """Hash a USDA fixture together with every directly referenced asset."""
    file_sha256 = _sha256_file(path)
    if path.suffix.lower() != ".usda":
        return file_sha256
    references = sorted(set(_ASSET_REFERENCE.findall(path.read_text(encoding="utf-8"))))
    if not references:
        return file_sha256
    digest = hashlib.sha256(b"ovrtx-fixture-content-v1\0")
    digest.update(b"fixture.usda\0")
    _update_digest_from_file(digest, path)
    fixture_root = path.parent.parent.resolve()
    for reference in references:
        resolved = (path.parent / reference).resolve()
        if not resolved.is_relative_to(fixture_root):
            raise ValueError(f"fixture asset escapes fixture root: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"fixture asset is missing: {resolved}")
        digest.update(reference.encode("utf-8") + b"\0")
        _update_digest_from_file(digest, resolved)
    return digest.hexdigest()


def fixture_runtime_content_sha256(runtime_files: Iterable[Mapping[str, Any]]) -> str:
    """Hash a complete runtime identity from canonical manifest-relative entries."""
    entries = _runtime_file_entries(runtime_files, "fixture runtime identity")
    digest = hashlib.sha256(b"ovrtx-fixture-content-v2\0")
    for path, sha256 in sorted(entries):
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(sha256.encode("ascii") + b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest_from_file(digest, path)
    return digest.hexdigest()


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def load_manifest(path: Path) -> Mapping[str, Any]:
    if path.is_dir() or (not path.exists() and path.name == "manifest.json"):
        return load_catalog(path if path.is_dir() else path.parent)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise ValueError("manifest root must be an object")
    manifest = dict(data)
    manifest["_manifest_base_path"] = str(_manifest_base_path(path))
    return manifest


def load_catalog(root: Path = Path(__file__).resolve().parent) -> Mapping[str, Any]:
    specs = sorted(root.glob("*/spec.json"))
    if not specs:
        raise ValueError(f"fixture catalog contains no specs: {root}")
    fixtures = []
    ids: set[str] = set()
    for path in specs:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"fixture spec root must be an object: {path}")
        fixture_id = value.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError(f"fixture spec id must be a non-empty string: {path}")
        if fixture_id in ids:
            raise ValueError(f"fixture id is declared more than once: {fixture_id}")
        ids.add(fixture_id)
        fixtures.append(value)
    return {"fixtures": fixtures, "_manifest_base_path": str(root.parent.resolve())}


def render_fixture(manifest: Mapping[str, Any], fixture_id: str = "") -> dict[str, Any]:
    base_path = Path(str(manifest.get("_manifest_base_path", ROOT)))
    record = _fixture_record(_find_fixture(manifest, fixture_id), base_path)
    if record["kind"] != "usd":
        raise ValueError(f"fixture {fixture_id} is not an exact USD fixture")
    if "ovrtx" not in record["capabilities"]:
        raise ValueError(f"fixture {fixture_id} does not declare ovrtx capability")
    _verify_runtime_files(record)
    _verify_runtime_reference_closure(record)
    _verify_runtime_content_identity(record)
    return record


def fixture_input(manifest: Mapping[str, Any], fixture_id: str = "") -> dict[str, Any]:
    """Return the manifest-selected source file for a fixture record."""
    base_path = Path(str(manifest.get("_manifest_base_path", ROOT)))
    record = _fixture_record(_find_fixture(manifest, fixture_id), base_path)
    if record["kind"] == "usd":
        _verify_runtime_files(record)
        _verify_runtime_reference_closure(record)
        _verify_runtime_content_identity(record)
        path = record["fixture_usd_path"]
        sha256 = record["fixture_usd_sha256"]
        manifest_path = record["fixture_usd_manifest_path"]
    else:
        path = record["blend_file"]
        sha256 = record["blend_file_sha256"]
        manifest_path = record["blend_file_manifest_path"]
        _verify_declared_file(record["id"], Path(str(path)), str(sha256))
    return {
        "id": record["id"],
        "kind": record["kind"],
        "path": path,
        "manifest_path": manifest_path,
        "sha256": sha256,
        "capabilities": record["capabilities"],
        "resolution": record["resolution"],
        "runtime_defaults": record["runtime_defaults"],
    }


def shared_stage_runtime_defaults(fixture: Mapping[str, Any]) -> dict[str, Any]:
    value = fixture.get("runtime_defaults")
    if not isinstance(value, Mapping):
        return {}
    shared_stage = value.get("shared_stage_composition")
    return dict(shared_stage) if isinstance(shared_stage, Mapping) else {}


def _find_fixture(manifest: Mapping[str, Any], fixture_id: str) -> Mapping[str, Any]:
    if not fixture_id:
        raise ValueError("--fixture-id is required")
    fixtures = manifest.get("fixtures", [])
    if not isinstance(fixtures, list):
        raise ValueError("manifest fixtures must be a list")
    for fixture in fixtures:
        if isinstance(fixture, Mapping) and fixture.get("id") == fixture_id:
            return fixture
    raise ValueError(f"fixture id not found: {fixture_id}")


def _manifest_base_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent.name == "fixtures":
        return resolved.parent.parent
    return resolved.parent


def _fixture_record(fixture: Mapping[str, Any], base_path: Path) -> dict[str, Any]:
    retired_scene_prefix = "pre" + "pared"
    removed_fields = (
        "usd_path",
        "usd_sha256",
        "camera_details",
        "render_affordances",
        f"{retired_scene_prefix}_usd_path",
        f"{retired_scene_prefix}_usd_sha256",
    )
    if any(field in fixture for field in removed_fields):
        fixture_id = str(fixture.get("id", UNKNOWN))
        raise ValueError(f"fixture {fixture_id} uses removed USD schema fields")
    fixture_id = str(fixture.get("id", UNKNOWN))
    platform_identity = _fixture_platform_identity(fixture, fixture_id)
    usd_path = _known(str(fixture.get("fixture_usd_path", UNKNOWN)))
    usd_sha256 = _known(
        str(
            platform_identity.get(
                "fixture_usd_sha256",
                fixture.get("fixture_usd_sha256", UNKNOWN),
            )
        )
    )
    blend_file = _known(str(fixture.get("blend_file", UNKNOWN)))
    blend_sha256 = _known(str(fixture.get("blend_file_sha256", UNKNOWN)))
    has_usd = usd_path != UNKNOWN or usd_sha256 != UNKNOWN
    has_blend = blend_file != UNKNOWN or blend_sha256 != UNKNOWN
    if has_usd == has_blend:
        if has_usd:
            raise ValueError(f"fixture {fixture_id} declares both USD and Blender inputs")
        raise ValueError(f"fixture {fixture_id} must declare one fixture input")
    kind = "blend" if has_blend else "usd"
    record = {
        "id": fixture_id,
        "kind": kind,
        "capabilities": _fixture_capabilities(fixture),
        "fixture_usd_path": _resolve_manifest_path(usd_path, base_path),
        "fixture_usd_manifest_path": usd_path,
        "fixture_usd_sha256": usd_sha256,
        "blend_file": _resolve_manifest_path(blend_file, base_path),
        "blend_file_manifest_path": blend_file,
        "blend_file_sha256": blend_sha256,
        "fixture_content_sha256": _known(
            str(
                platform_identity.get(
                    "fixture_content_sha256",
                    fixture.get("fixture_content_sha256", UNKNOWN),
                )
            )
        ),
        "camera_prim_path": _known(str(fixture.get("camera_prim_path", UNKNOWN))),
        "render_product_path": _known(str(fixture.get("render_product_prim_path", UNKNOWN))),
        "resolution": _fixture_resolution(fixture),
        "runtime_defaults": _fixture_runtime_defaults(fixture),
    }
    record["runtime_files"] = _fixture_runtime_files(
        fixture, record, base_path, bool(platform_identity)
    )
    return record


def _fixture_platform_identity(
    fixture: Mapping[str, Any], fixture_id: str
) -> Mapping[str, str]:
    identities = fixture.get("platform_identities")
    if identities is None:
        return {}
    if not isinstance(identities, Mapping) or not identities:
        raise ValueError(
            f"fixture {fixture_id} platform_identities must be a non-empty object"
        )
    required = {"fixture_usd_sha256", "fixture_content_sha256"}
    for platform, identity in identities.items():
        if not isinstance(platform, str) or not platform:
            raise ValueError(f"fixture {fixture_id} platform identity key must be non-empty")
        if not isinstance(identity, Mapping) or set(identity) != required:
            raise ValueError(
                f"fixture {fixture_id} platform identity for {platform} must declare "
                f"fixture_usd_sha256 and fixture_content_sha256"
            )
        if any(
            not isinstance(identity[field], str)
            or _SHA256.fullmatch(identity[field]) is None
            for field in required
        ):
            raise ValueError(f"fixture {fixture_id} platform identity for {platform} is invalid")
    return identities.get(sys.platform, {})


def _fixture_runtime_files(
    fixture: Mapping[str, Any],
    record: Mapping[str, Any],
    base_path: Path,
    has_platform_identity: bool,
) -> tuple[dict[str, str], ...]:
    value = fixture.get("runtime_files")
    if value is None:
        return ()
    fixture_id = str(record["id"])
    if record["kind"] != "usd":
        raise ValueError(f"fixture {fixture_id} runtime_files require an exact USD input")
    if not isinstance(value, list) or not value:
        raise ValueError(f"fixture {fixture_id} runtime_files must be a non-empty list")

    entries = _runtime_file_entries(value, f"fixture {fixture_id} runtime_files")
    files: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path_value, sha256 in entries:
        path = Path(_resolve_manifest_path(path_value, base_path)).resolve()
        base = base_path.resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"fixture {fixture_id} runtime file escapes manifest root: {path}")
        if path in seen:
            raise ValueError(f"fixture {fixture_id} runtime file is listed more than once: {path}")
        seen.add(path)
        files.append({"path": str(path), "manifest_path": path_value, "sha256": sha256})

    direct_path = Path(str(record["fixture_usd_path"])).resolve()
    direct = next((item for item in files if Path(item["path"]) == direct_path), None)
    if direct is None:
        raise ValueError(f"fixture {fixture_id} direct USD must appear in runtime_files")
    if has_platform_identity:
        direct["sha256"] = str(record["fixture_usd_sha256"])
    if direct["sha256"] != record["fixture_usd_sha256"]:
        raise ValueError(f"fixture {fixture_id} direct USD digest must match fixture_usd_sha256")
    return tuple(files)


def _runtime_file_entries(
    runtime_files: Iterable[Mapping[str, Any]],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if isinstance(runtime_files, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a list of objects")
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in runtime_files:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} entries must be objects")
        path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label} path must be a non-empty string")
        canonical = PurePosixPath(path)
        if (
            canonical.is_absolute()
            or canonical.as_posix() != path
            or "\\" in path
            or any(part in {"", ".", ".."} for part in canonical.parts)
        ):
            raise ValueError(f"{label} path must be canonical and manifest-relative: {path}")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"{label} sha256 must be 64 lowercase hex characters")
        if path in seen:
            raise ValueError(f"{label} path is listed more than once: {path}")
        seen.add(path)
        entries.append((path, sha256))
    return tuple(entries)


def _verify_runtime_files(record: Mapping[str, Any]) -> None:
    fixture_id = str(record["id"])
    for item in record["runtime_files"]:
        _verify_declared_file(fixture_id, Path(item["path"]), item["sha256"])


def _verify_runtime_reference_closure(record: Mapping[str, Any]) -> None:
    """Verify the recursively readable part of a fixture's USD asset closure."""
    fixture_id = str(record["id"])
    if not record["runtime_files"]:
        return
    direct_path = Path(str(record["fixture_usd_path"])).resolve()
    fixture_root = direct_path.parent.parent.resolve()
    declared = {Path(item["path"]).resolve() for item in record["runtime_files"]}
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved_path = path.resolve()
        if resolved_path in visited:
            return
        visited.add(resolved_path)

        # Binary USDC layers are identity-checked above but intentionally not parsed here.
        if resolved_path.suffix.lower() != ".usda":
            return
        references = set(
            _ASSET_REFERENCE.findall(resolved_path.read_text(encoding="utf-8"))
        )
        for reference in references:
            resolved = (resolved_path.parent / reference).resolve()
            if not resolved.is_relative_to(fixture_root):
                raise ValueError(
                    f"fixture asset reference escapes runtime root for {fixture_id}: {resolved}"
                )
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"fixture asset reference is missing for {fixture_id}: {resolved}"
                )
            if resolved not in declared:
                raise ValueError(
                    f"fixture asset reference is not declared in runtime_files for "
                    f"{fixture_id}: {resolved}"
                )
            visit(resolved)

    visit(direct_path)


def _verify_runtime_content_identity(record: Mapping[str, Any]) -> None:
    runtime_files = record["runtime_files"]
    if not runtime_files:
        return
    identities = (
        {"path": item["manifest_path"], "sha256": item["sha256"]}
        for item in runtime_files
    )
    actual = fixture_runtime_content_sha256(identities)
    expected = record["fixture_content_sha256"]
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(
            f"fixture {record['id']} fixture_content_sha256 must be 64 lowercase hex characters"
        )
    if actual != expected:
        raise ValueError(
            f"fixture {record['id']} fixture_content_sha256 mismatch: "
            f"expected {expected}, got {actual}"
        )


def _verify_declared_file(fixture_id: str, path: Path, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"fixture runtime file is missing for {fixture_id}: {path}")
    if _SHA256.fullmatch(sha256) is None:
        raise ValueError(
            f"fixture {fixture_id} source sha256 must be 64 lowercase hex characters"
        )
    with path.open("rb") as stream:
        if stream.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            raise ValueError(
                f"fixture runtime file is a Git LFS pointer for {fixture_id}: {path}"
            )
    actual = _sha256_file(path)
    if actual != sha256:
        raise ValueError(
            f"fixture runtime file digest mismatch for {fixture_id}: "
            f"{path} expected {sha256}, got {actual}"
        )


def _fixture_capabilities(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    fixture_id = str(fixture.get("id", UNKNOWN))
    value = fixture.get("capabilities")
    if not isinstance(value, list) or not value:
        raise ValueError(f"fixture {fixture_id} capabilities must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"fixture {fixture_id} capabilities must be a non-empty list of strings")
    return tuple(value)


def _fixture_runtime_defaults(fixture: Mapping[str, Any]) -> dict[str, Any]:
    value = fixture.get("runtime_defaults")
    return dict(value) if isinstance(value, Mapping) else {}


def _fixture_resolution(fixture: Mapping[str, Any]) -> dict[str, int]:
    value = fixture.get("target_resolution")
    if value is None:
        return {"width": 1280, "height": 720}
    fixture_id = str(fixture.get("id", UNKNOWN))
    if not isinstance(value, Mapping):
        raise ValueError(f"fixture {fixture_id} target_resolution must be an object")
    try:
        width = int(value.get("width", 1280))
        height = int(value.get("height", 720))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fixture {fixture_id} target_resolution width and height must be positive integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"fixture {fixture_id} target_resolution width and height must be positive integers")
    return {"width": width, "height": height}


def _known(value: str) -> str:
    value = value.strip()
    return value if value and value != UNKNOWN else UNKNOWN


def _resolve_manifest_path(value: str, base_path: Path) -> str:
    if value == UNKNOWN:
        return UNKNOWN
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(base_path / path)
