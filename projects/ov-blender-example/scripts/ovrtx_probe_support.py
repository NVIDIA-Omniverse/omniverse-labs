# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared helpers for OVRTX runtime probes."""

import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "addon"
if str(ADDON) not in sys.path:
    sys.path.insert(0, str(ADDON))

from ovrtx_blender_example import bundled_runtime  # noqa: E402


BLENDER_COMMAND = os.environ.get("BLENDER_COMMAND", "blender")
DEFAULT_FIXTURE_MANIFEST = ROOT / "tests" / "fixtures"
if str(DEFAULT_FIXTURE_MANIFEST) not in sys.path:
    sys.path.insert(0, str(DEFAULT_FIXTURE_MANIFEST))
UNKNOWN = "???"


def default_worker_command() -> str:
    return bundled_runtime.defaults().worker_command


def default_native_client_path() -> str:
    return bundled_runtime.defaults().native_client_path


def worker_command_for_port(worker_command: str, port: int) -> str:
    return f"{worker_command.strip()} --port {port}".strip()


def resolve_executable(command: str) -> str:
    """Return the executable path for a command name or explicit path."""
    path = Path(command).expanduser()
    if path.is_absolute() or os.sep in command or (os.altsep and os.altsep in command):
        return str(path) if path.is_file() and os.access(path, os.X_OK) else ""
    return shutil.which(command) or ""


def native_extension_check(
    directory: Path, module_name: str, *, extension_suffix: str | None = None
) -> dict[str, Any]:
    """Describe whether the interpreter-specific native module is present."""
    suffix = (
        str(sysconfig.get_config_var("EXT_SUFFIX") or "")
        if extension_suffix is None
        else extension_suffix
    )
    expected = directory / f"{module_name}{suffix}"
    if suffix and expected.is_file():
        return {"kind": "file", "path": str(expected), "ok": True}
    fallback = sorted(directory.glob(f"{module_name}*.so"))
    return {
        "kind": "file",
        "path": str(expected if suffix else directory / f"{module_name}.so"),
        "fallback_matches": [str(path) for path in fallback],
        "ok": False,
    }


def resolve_fixture(manifest_path: Path, fixture_id: str) -> dict[str, Any]:
    """Resolve and verify one fixture from the committed fixture manifest."""
    from fixture_manifest import load_manifest, render_fixture

    return render_fixture(load_manifest(manifest_path), fixture_id)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when no artifact exists."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return dict(data) if isinstance(data, Mapping) else {}


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write a complete, deterministic probe result object."""
    payload = dict(result)
    payload.setdefault("completed_at_ns", time.time_ns())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
