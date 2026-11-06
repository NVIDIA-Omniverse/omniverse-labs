# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared test isolation for suite-global add-on state."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_ovrtx_gpu_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OV_BLENDER_EXAMPLE_OVRTX_GPU_LOCK_DIR",
        str(tmp_path / "ovrtx-gpu-locks"),
    )
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_OVRTX_GPU_LEASE_ID", "pytest-gpu")
