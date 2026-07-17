# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovphysx_simulation  # noqa: E402
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig  # noqa: E402


def _config(**changes: object) -> InteractiveSharedStageConfig:
    config = InteractiveSharedStageConfig(
        enabled=True,
        input_usd_path="/tmp/physics.usda",
        server="/tmp/server",
        ovphysx_address="127.0.0.1:50094",
        ovphysx_worker_command="worker",
        device="cpu",
        body_root="/World/Bodies",
        body_prims=("/World/Bodies/Cube",),
        physics_fps=60.0,
        update_fps=30.0,
        max_steps=240,
        body_scale=1.0,
        worker_log_path="/tmp/worker.log",
    )
    return replace(config, **changes)


def test_prepare_excludes_non_simulation_configuration() -> None:
    spec = ovphysx_simulation.prepare(_config())
    changed = ovphysx_simulation.prepare(
        _config(
            body_prims=("/World/Bodies/Other",),
            physics_fps=120.0,
            update_fps=60.0,
            max_steps=480,
            body_scale=2.0,
            worker_log_path="/tmp/other.log",
        )
    )

    assert changed == spec


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"input_usd_path": "/tmp/other.usda"}, "physics_input_changed"),
        ({"ovphysx_address": "127.0.0.1:50095"}, "runtime_binding_changed"),
        ({"ovphysx_worker_command": "other-worker"}, "runtime_binding_changed"),
        ({"ovphysx_native_client_module": "other.module"}, "runtime_binding_changed"),
        ({"ovphysx_native_client_path": "/tmp/other.so"}, "runtime_binding_changed"),
    ],
)
def test_reuse_policy_replaces_changed_simulation_spec(
    changes: dict[str, object],
    reason: str,
) -> None:
    current = ovphysx_simulation.prepare(_config())
    desired = ovphysx_simulation.prepare(_config(**changes))

    assert ovphysx_simulation.reuse_decision(current, desired) == (
        ovphysx_simulation.OvphysxSimulationReuseDecision(False, reason)
    )


def test_reuse_policy_reason_priority() -> None:
    current = ovphysx_simulation.prepare(_config())
    changed = ovphysx_simulation.prepare(_config(input_usd_path="/tmp/other.usda"))

    assert ovphysx_simulation.reuse_decision(
        current, changed, explicit_reset=True, terminal_failure=True
    ).reason == "terminal_failure"
    assert ovphysx_simulation.reuse_decision(
        current, changed, explicit_reset=True
    ).reason == "explicit_reset"


def test_equal_spec_reuses_simulation() -> None:
    spec = ovphysx_simulation.prepare(_config())

    assert ovphysx_simulation.reuse_decision(spec, spec) == (
        ovphysx_simulation.OvphysxSimulationReuseDecision(True, "same_simulation")
    )
