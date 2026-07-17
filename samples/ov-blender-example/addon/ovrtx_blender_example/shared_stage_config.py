# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-stage runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from .physics_body_prims import DEFAULT_DYNAMIC_BODY_ROOT, discover_dynamic_body_prims
from .ovphysx_runtime_client import (
    DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE,
    DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH,
)
from . import bundled_runtime
from .shared_stage_errors import SharedStageCompositionError


DEFAULT_OVPHYSX_ADDRESS = "127.0.0.1:50094"
DEFAULT_OVPHYSX_SERVER = Path("ovphysx-bridge-server")
DEFAULT_DEVICE = "cpu"


@dataclass(frozen=True)
class InteractiveSharedStageConfig:
    enabled: bool
    input_usd_path: str
    server: str
    ovphysx_address: str
    ovphysx_worker_command: str
    device: str
    body_root: str
    body_prims: tuple[str, ...]
    physics_fps: float
    update_fps: float
    max_steps: int
    body_scale: float
    worker_log_path: str
    trace_log_path: str = ""
    ovphysx_native_client_module: str = DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE
    ovphysx_native_client_path: str = str(DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH)

    @classmethod
    def from_env(
        cls,
        input_usd_path: str,
        *,
        authored_body_prims: tuple[str, ...] | None = None,
    ) -> "InteractiveSharedStageConfig":
        enabled = (
            True
            if authored_body_prims is not None
            else _env_bool("OV_BLENDER_EXAMPLE_SHARED_STAGE")
        )
        bundle = bundled_runtime.defaults(
            ovphysx_address=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS", DEFAULT_OVPHYSX_ADDRESS),
            ovphysx_device=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", DEFAULT_DEVICE),
        )
        ovphysx_input_usd_path = os.environ.get(
            "OV_BLENDER_EXAMPLE_OVPHYSX_INPUT_USD_PATH",
            input_usd_path,
        )
        server_default = Path(bundle.ovphysx_server) if bundle.ovphysx_server else DEFAULT_OVPHYSX_SERVER
        server = _env_path("OV_BLENDER_EXAMPLE_OVPHYSX_SERVER", server_default)
        address = os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS", DEFAULT_OVPHYSX_ADDRESS)
        device = os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", DEFAULT_DEVICE)
        command = os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND", "").strip()
        if not command:
            command = bundle.ovphysx_worker_command or _default_ovphysx_worker_command(server, address, device)
        body_root = os.environ.get("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_ROOT", DEFAULT_DYNAMIC_BODY_ROOT).strip()
        if not body_root:
            body_root = DEFAULT_DYNAMIC_BODY_ROOT
        return cls(
            enabled=enabled,
            input_usd_path=ovphysx_input_usd_path,
            server=str(server),
            ovphysx_address=address,
            ovphysx_worker_command=command,
            device=device,
            body_root="" if authored_body_prims is not None else body_root,
            body_prims=(
                tuple(authored_body_prims)
                if authored_body_prims is not None
                else _env_body_prims(ovphysx_input_usd_path, body_root)
                if enabled
                else ()
            ),
            physics_fps=_env_float("OV_BLENDER_EXAMPLE_OVPHYSX_PHYSICS_FPS", 60.0),
            update_fps=_env_float("OV_BLENDER_EXAMPLE_SHARED_STAGE_UPDATE_FPS", 30.0),
            max_steps=max(1, _env_int("OV_BLENDER_EXAMPLE_OVPHYSX_MAX_STEPS", 240)),
            body_scale=_env_float("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_SCALE", 1.0),
            worker_log_path=os.environ.get("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_LOG", ""),
            trace_log_path=os.environ.get("OV_BLENDER_EXAMPLE_SHARED_STAGE_TRACE_LOG", ""),
            ovphysx_native_client_module=os.environ.get(
                "OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_MODULE",
                DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE,
            ).strip()
            or DEFAULT_OVPHYSX_NATIVE_CLIENT_MODULE,
            ovphysx_native_client_path=os.environ.get(
                "OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH",
                bundle.ovphysx_native_client_path or str(DEFAULT_OVPHYSX_NATIVE_CLIENT_PATH),
            ),
        )

    @property
    def timestep_ns(self) -> int:
        return int(1_000_000_000 / max(1.0, self.physics_fps))

    @property
    def update_interval_ns(self) -> int:
        return int(1_000_000_000 / max(1.0, self.update_fps))

    @property
    def steps_per_update(self) -> int:
        return max(1, int(math.ceil(max(1.0, self.physics_fps) / max(1.0, self.update_fps))))


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else default


def _env_body_prims(input_usd_path: str, root: str) -> tuple[str, ...]:
    value = os.environ.get("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS", "").strip()
    if value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.split(",")]
        if not isinstance(parsed, list):
            raise SharedStageCompositionError("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS must be a JSON list or comma-separated paths")
        body_prims = tuple(str(item) for item in parsed if str(item))
        if not body_prims:
            raise SharedStageCompositionError("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS did not contain any prim paths")
        return body_prims

    body_prims = discover_dynamic_body_prims(input_usd_path, root or DEFAULT_DYNAMIC_BODY_ROOT)
    if not body_prims:
        raise SharedStageCompositionError(
            f"No dynamic rigid bodies found under {root or DEFAULT_DYNAMIC_BODY_ROOT} in {input_usd_path}"
        )
    return body_prims


def _default_ovphysx_worker_command(server: Path, address: str, device: str) -> str:
    return bundled_runtime.serialize_command(
        [str(server), "--listen", address, "--device", device]
    )


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_DYNAMIC_BODY_ROOT",
    "DEFAULT_OVPHYSX_ADDRESS",
    "DEFAULT_OVPHYSX_SERVER",
    "InteractiveSharedStageConfig",
    "_default_ovphysx_worker_command",
]
