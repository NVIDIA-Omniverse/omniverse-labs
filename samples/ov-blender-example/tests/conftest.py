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


@pytest.fixture(autouse=True)
def _reset_ovrtx_worker_attach_registry():
    """Isolate the per-process worker attach registry between tests.

    The OVRTX runtime client sweeps stale simulations only on the first
    attach to a worker endpoint in a process (blender-live-render
    task05-02). Tests exercising `start_session` share one pytest
    process, so the registry must reset around every test.
    """

    try:
        from ovrtx_blender_example import ovrtx_runtime_client
    except Exception:
        yield
        return
    ovrtx_runtime_client._reset_worker_attach_registry()
    yield
    ovrtx_runtime_client._reset_worker_attach_registry()
