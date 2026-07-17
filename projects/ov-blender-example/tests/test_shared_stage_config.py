# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import bundled_runtime  # noqa: E402
from ovrtx_blender_example.shared_stage_config import (  # noqa: E402
    DEFAULT_DYNAMIC_BODY_ROOT,
    DEFAULT_OVPHYSX_ADDRESS,
    InteractiveSharedStageConfig,
)


_ENV_KEYS = (
    "OV_BLENDER_EXAMPLE_SHARED_STAGE",
    "OV_BLENDER_EXAMPLE_OVPHYSX_INPUT_USD_PATH",
    "OV_BLENDER_EXAMPLE_OVPHYSX_SERVER",
    "OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS",
    "OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE",
    "OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND",
    "OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_ROOT",
    "OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS",
    "OV_BLENDER_EXAMPLE_OVPHYSX_PHYSICS_FPS",
    "OV_BLENDER_EXAMPLE_SHARED_STAGE_UPDATE_FPS",
    "OV_BLENDER_EXAMPLE_OVPHYSX_MAX_STEPS",
    "OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_SCALE",
    "OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_LOG",
    "OV_BLENDER_EXAMPLE_SHARED_STAGE_TRACE_LOG",
    "OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_MODULE",
    "OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH",
)


def _clear_env(monkeypatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_shared_stage_config_from_env_disabled_uses_defaults(monkeypatch) -> None:
    _clear_env(monkeypatch)

    config = InteractiveSharedStageConfig.from_env("/tmp/source.usda")

    assert config.enabled is False
    assert config.input_usd_path == "/tmp/source.usda"
    assert config.ovphysx_address == DEFAULT_OVPHYSX_ADDRESS
    assert config.body_root == DEFAULT_DYNAMIC_BODY_ROOT
    assert config.body_prims == ()
    assert config.physics_fps == 60.0
    assert config.update_fps == 30.0
    assert config.timestep_ns == 16_666_666
    assert config.steps_per_update == 2
    assert config.max_steps == 240


def test_shared_stage_config_60hz_publication_uses_one_fixed_step(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_PHYSICS_FPS", "60")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_UPDATE_FPS", "60")

    config = InteractiveSharedStageConfig.from_env("/tmp/source.usda")

    assert config.timestep_ns == 16_666_666
    assert config.steps_per_update == 1


def test_shared_stage_config_ignores_removed_path_environment(monkeypatch) -> None:
    _clear_env(monkeypatch)
    removed_key = "OV_BLENDER_EXAMPLE_OVPHYSX_" + "FIXTURE_USD"
    monkeypatch.setenv(removed_key, "/tmp/removed.usda")

    config = InteractiveSharedStageConfig.from_env("/tmp/input.usda")

    assert config.input_usd_path == "/tmp/input.usda"


def test_shared_stage_config_uses_bundled_ovphysx_defaults(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    addon_root = tmp_path / "addon"
    server = addon_root / "bin" / "ovphysx-bridge-server"
    native = addon_root / "native"
    server.parent.mkdir(parents=True)
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    native.mkdir()
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: addon_root)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    config = InteractiveSharedStageConfig.from_env("/tmp/source.usda")

    assert config.server == str(server)
    assert config.ovphysx_native_client_path == str(native)


def test_shared_stage_config_from_env_parses_explicit_runtime_values(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    input_usd_path = tmp_path / "physics.usda"
    native_path = tmp_path / "native"
    server_path = tmp_path / "ovphysx-bridge-server" / "bin" / "ovphysx-bridge-server"
    worker_log_path = tmp_path / "ovphysx.log"
    trace_log_path = tmp_path / "shared-stage.jsonl"
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE", "true")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_INPUT_USD_PATH", str(input_usd_path))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_SERVER", str(server_path))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS", "127.0.0.1:50123")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE", "gpu")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND", "custom-ovphysx")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_ROOT", "/World/Bodies")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS", '["/World/Bodies/A", "/World/Bodies/B"]')
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_PHYSICS_FPS", "120")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_UPDATE_FPS", "30")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_MAX_STEPS", "12")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_SCALE", "2.5")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_LOG", str(worker_log_path))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_TRACE_LOG", str(trace_log_path))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_MODULE", "custom_ovphysx_client")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH", str(native_path))

    config = InteractiveSharedStageConfig.from_env("/tmp/source.usda")

    assert config.enabled is True
    assert config.input_usd_path == str(input_usd_path)
    assert config.server == str(server_path)
    assert config.ovphysx_address == "127.0.0.1:50123"
    assert config.device == "gpu"
    assert config.body_root == "/World/Bodies"
    assert config.body_prims == ("/World/Bodies/A", "/World/Bodies/B")
    assert config.physics_fps == 120.0
    assert config.update_fps == 30.0
    assert config.steps_per_update == 4
    assert config.timestep_ns == 8_333_333
    assert config.update_interval_ns == 33_333_333
    assert config.max_steps == 12
    assert config.body_scale == 2.5
    assert config.worker_log_path == str(worker_log_path)
    assert config.trace_log_path == str(trace_log_path)
    assert config.ovphysx_native_client_module == "custom_ovphysx_client"
    assert config.ovphysx_native_client_path == str(native_path)


def test_shared_stage_config_accepts_comma_separated_body_prims(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE", "1")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE_BODY_PRIMS", "/World/A, /World/B")

    config = InteractiveSharedStageConfig.from_env("/tmp/source.usda")

    assert config.body_prims == ("/World/A", "/World/B")
