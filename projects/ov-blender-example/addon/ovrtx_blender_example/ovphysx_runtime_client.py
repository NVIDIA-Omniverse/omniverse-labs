# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared input validation for the external public SDK physics client."""
from __future__ import annotations
from typing import Any
from .shared_stage_errors import SharedStageCompositionError
INT64_MAX = (1 << 63) - 1
from .native_client_support import coerce_mapping_int
UNKNOWN = "unknown"

def _validate_advance_inputs(
    start_step_count: Any,
    steps: Any,
    timestep_ns: Any,
) -> tuple[int, int, int, int, int]:
    start_step_count = _coerce_int("start_step_count", start_step_count)
    steps = _coerce_int("steps", steps)
    timestep_ns = _coerce_int("timestep_ns", timestep_ns)
    if start_step_count < 0:
        raise SharedStageCompositionError("start_step_count must be non-negative")
    if steps <= 0:
        raise SharedStageCompositionError("steps must be positive")
    if timestep_ns <= 0:
        raise SharedStageCompositionError("timestep_ns must be positive")
    if start_step_count > INT64_MAX or steps > INT64_MAX or timestep_ns > INT64_MAX:
        raise SharedStageCompositionError("advance inputs must fit int64")
    if start_step_count > INT64_MAX - steps:
        raise SharedStageCompositionError("start_step_count + steps overflows int64")
    step_count = start_step_count + steps
    if step_count > INT64_MAX // timestep_ns:
        raise SharedStageCompositionError("step_count * timestep_ns overflows int64")
    return start_step_count, steps, timestep_ns, step_count, step_count * timestep_ns


def _coerce_int(name: str, value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SharedStageCompositionError(f"{name} must be int") from exc
