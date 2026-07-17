# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime bundle manifest loading for split extension releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RUNTIME_MANIFEST_NAME = "runtime-bundle-manifest.json"
RUNTIME_MANIFEST_PIN_NAME = "runtime-bundle-manifest.sha256"
RUNTIME_MANIFEST_KIND = "ov-blender-example-runtime-bundle"
REQUIRED_RUNTIME_COMPONENT_IDS = frozenset(
    {
        "ovrtx-bridge-server",
        "ovphysx-bridge-server",
        "ovrtx-bridge-client",
        "ovphysx-bridge-client",
    }
)


class RuntimeManifestError(RuntimeError):
    """Raised when a runtime manifest identity or document is invalid."""


@dataclass(frozen=True)
class RuntimeTarget:
    source: str
    target: str
    mode: str = ""


@dataclass(frozen=True)
class RuntimeComponent:
    id: str
    filename: str
    sha256: str
    size_bytes: int
    targets: tuple[RuntimeTarget, ...]
    executables: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeManifest:
    platform: str
    components: tuple[RuntimeComponent, ...]
    sha256: str


def load_manifest_pin(addon_root: Path) -> str:
    pin_path = addon_root / RUNTIME_MANIFEST_PIN_NAME
    try:
        value = pin_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeManifestError(f"runtime manifest pin missing: {pin_path}") from exc
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeManifestError(f"runtime manifest pin is invalid: {pin_path}")
    return value


def parse_manifest_bytes(data: bytes) -> RuntimeManifest:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeManifestError(f"runtime manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeManifestError("runtime manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeManifestError("runtime manifest schema_version must be 1")
    if payload.get("kind") != RUNTIME_MANIFEST_KIND:
        raise RuntimeManifestError(f"runtime manifest kind must be {RUNTIME_MANIFEST_KIND}")
    _require_fields(payload, {"schema_version", "kind", "platform", "components"}, "runtime manifest")
    components = _components(payload)
    return RuntimeManifest(
        platform=_required_string(payload, "platform"),
        components=tuple(components),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _components(payload: Mapping[str, Any]) -> list[RuntimeComponent]:
    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise RuntimeManifestError("runtime manifest must contain non-empty components")
    components = [_component(component) for component in raw_components]
    component_ids = [component.id for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise RuntimeManifestError("runtime manifest component ids must be unique")
    missing = REQUIRED_RUNTIME_COMPONENT_IDS - set(component_ids)
    if missing:
        raise RuntimeManifestError(
            "runtime manifest is missing required components: "
            + ", ".join(sorted(missing))
        )
    filenames = [component.filename for component in components]
    if len(filenames) != len(set(filenames)):
        raise RuntimeManifestError("runtime manifest component filenames must be unique")
    return components


def _component(payload: Any) -> RuntimeComponent:
    if not isinstance(payload, Mapping):
        raise RuntimeManifestError("runtime manifest components must be objects")
    _require_fields(
        payload,
        {"id", "filename", "sha256", "size_bytes", "targets", "executables"},
        "runtime manifest component",
    )
    size_bytes = payload.get("size_bytes")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise RuntimeManifestError("runtime component size_bytes must be a non-negative integer")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeManifestError("runtime component targets must be a non-empty list")
    executables = payload.get("executables")
    if not isinstance(executables, list) or not all(isinstance(item, str) for item in executables):
        raise RuntimeManifestError("runtime component executables must be a string list")
    return RuntimeComponent(
        id=_required_string(payload, "id"),
        filename=_required_filename(payload),
        sha256=_required_sha256(payload),
        size_bytes=size_bytes,
        targets=tuple(_target(target) for target in targets),
        executables=tuple(item.strip() for item in executables if item.strip()),
    )


def _required_filename(payload: Mapping[str, Any]) -> str:
    value = _required_string(payload, "filename")
    if Path(value).name != value or "\\" in value or value in {".", ".."}:
        raise RuntimeManifestError("runtime component filename must be a plain filename")
    return value


def _required_sha256(payload: Mapping[str, Any]) -> str:
    value = _required_string(payload, "sha256")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeManifestError("runtime component sha256 must be lowercase hexadecimal")
    return value


def _target(payload: Any) -> RuntimeTarget:
    if not isinstance(payload, Mapping):
        raise RuntimeManifestError("runtime component targets must be objects")
    allowed_fields = {"source", "target", "mode"}
    if not set(payload).issubset(allowed_fields):
        raise RuntimeManifestError("runtime component target contains unsupported fields")
    source = _required_string(payload, "source")
    target = _required_string(payload, "target")
    mode = payload.get("mode")
    return RuntimeTarget(source=source, target=target, mode=mode.strip() if isinstance(mode, str) else "")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeManifestError(f"runtime manifest missing required {key}")
    return value.strip()


def _require_fields(payload: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(payload) != fields:
        raise RuntimeManifestError(f"{label} fields must be {', '.join(sorted(fields))}")


__all__ = [
    "RUNTIME_MANIFEST_KIND",
    "RUNTIME_MANIFEST_NAME",
    "RUNTIME_MANIFEST_PIN_NAME",
    "REQUIRED_RUNTIME_COMPONENT_IDS",
    "RuntimeComponent",
    "RuntimeManifest",
    "RuntimeManifestError",
    "RuntimeTarget",
    "load_manifest_pin",
    "parse_manifest_bytes",
]
