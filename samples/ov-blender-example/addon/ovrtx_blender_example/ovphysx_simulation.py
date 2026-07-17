# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation and reuse policy for OVPhysX simulations."""

from __future__ import annotations

from dataclasses import dataclass

from .shared_stage_config import InteractiveSharedStageConfig


@dataclass(frozen=True)
class OvphysxSimulationSpec:
    input_usd_path: str
    address: str
    worker_command: str
    native_client_module: str
    native_client_path: str


@dataclass(frozen=True)
class OvphysxSimulationReuseDecision:
    reuse: bool
    reason: str


def prepare(config: InteractiveSharedStageConfig) -> OvphysxSimulationSpec:
    """Freeze only inputs that determine OVPhysX simulation identity."""

    return OvphysxSimulationSpec(
        input_usd_path=config.input_usd_path,
        address=config.ovphysx_address,
        worker_command=config.ovphysx_worker_command,
        native_client_module=config.ovphysx_native_client_module,
        native_client_path=config.ovphysx_native_client_path,
    )


def reuse_decision(
    current: OvphysxSimulationSpec,
    desired: OvphysxSimulationSpec,
    *,
    explicit_reset: bool = False,
    terminal_failure: bool = False,
) -> OvphysxSimulationReuseDecision:
    """Evaluate OVPhysX simulation reuse in diagnostic priority order."""

    if terminal_failure:
        return OvphysxSimulationReuseDecision(False, "terminal_failure")
    if explicit_reset:
        return OvphysxSimulationReuseDecision(False, "explicit_reset")
    if current.input_usd_path != desired.input_usd_path:
        return OvphysxSimulationReuseDecision(False, "physics_input_changed")
    if current != desired:
        return OvphysxSimulationReuseDecision(False, "runtime_binding_changed")
    return OvphysxSimulationReuseDecision(True, "same_simulation")


__all__ = [
    "OvphysxSimulationReuseDecision",
    "OvphysxSimulationSpec",
    "prepare",
    "reuse_decision",
]
