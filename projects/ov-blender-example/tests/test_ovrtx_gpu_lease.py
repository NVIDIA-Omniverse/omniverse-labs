# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_gpu_lease  # noqa: E402


def test_gpu_lease_blocks_second_owner_and_reports_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ovrtx_gpu_lease.LOCK_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ovrtx_gpu_lease.LEASE_ID_ENV, "GPU-test")

    lease = ovrtx_gpu_lease.acquire(metadata={"entrypoint": "first"}, timeout_s=0)
    try:
        with pytest.raises(ovrtx_gpu_lease.OvrtxGpuLeaseBusy) as error:
            ovrtx_gpu_lease.acquire(metadata={"entrypoint": "second"}, timeout_s=0)
    finally:
        lease.close()

    assert error.value.gpu_id == "GPU-test"
    assert error.value.owner["entrypoint"] == "first"
    assert error.value.owner["gpu_id"] == "GPU-test"


def test_gpu_lease_probe_reports_busy_owner_without_taking_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ovrtx_gpu_lease.LOCK_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ovrtx_gpu_lease.LEASE_ID_ENV, "GPU-test")
    lease = ovrtx_gpu_lease.acquire(metadata={"entrypoint": "owner"}, timeout_s=0)
    try:
        status = ovrtx_gpu_lease.probe()
    finally:
        lease.close()

    assert status["status"] == "busy"
    assert status["owner"]["pid"] == os.getpid()
    assert "entrypoint=owner" in status["error"]
    assert ovrtx_gpu_lease.probe()["status"] == "available"


def test_gpu_lease_ignores_stale_metadata_when_lock_is_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ovrtx_gpu_lease.LOCK_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ovrtx_gpu_lease.LEASE_ID_ENV, "GPU-stale")
    metadata = tmp_path / "GPU-stale.lock.json"
    metadata.write_text('{"entrypoint": "stale"}\n', encoding="utf-8")

    lease = ovrtx_gpu_lease.acquire(metadata={"entrypoint": "fresh"}, timeout_s=0)
    try:
        current = json.loads(metadata.read_text(encoding="utf-8"))
    finally:
        lease.close()

    assert current["entrypoint"] == "fresh"


def test_gpu_lease_is_released_when_owner_process_exits_without_cleanup(
    tmp_path: Path,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "addon"),
        ovrtx_gpu_lease.LOCK_DIR_ENV: str(tmp_path),
        ovrtx_gpu_lease.LEASE_ID_ENV: "GPU-crash",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "from ovrtx_blender_example import ovrtx_gpu_lease; "
                "lease = ovrtx_gpu_lease.acquire(timeout_s=0); "
                "os._exit(0)"
            ),
        ],
        check=True,
        env=env,
    )

    lease = ovrtx_gpu_lease.acquire(
        metadata={"entrypoint": "fresh"},
        timeout_s=0,
        env=env,
    )
    try:
        assert lease.gpu_id == "GPU-crash"
    finally:
        lease.close()


def test_gpu_lease_allows_inherited_child_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ovrtx_gpu_lease.LOCK_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ovrtx_gpu_lease.LEASE_ID_ENV, "GPU-parent")

    lease = ovrtx_gpu_lease.acquire(metadata={"entrypoint": "parent"}, timeout_s=0)
    try:
        inherited = ovrtx_gpu_lease.acquire(
            env={**os.environ, **lease.child_environment()},
            timeout_s=0,
        )
        try:
            assert inherited.diagnostics()["status"] == "inherited"
        finally:
            inherited.close()

        with pytest.raises(ovrtx_gpu_lease.OvrtxGpuLeaseBusy):
            ovrtx_gpu_lease.acquire(metadata={"entrypoint": "other"}, timeout_s=0)
    finally:
        lease.close()


def test_gpu_lease_resolves_uuid_from_cuda_and_srtx_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ovrtx_gpu_lease,
        "_nvidia_smi_gpu_uuids",
        lambda: {"0": "GPU-zero", "2": "GPU-two"},
    )

    assert (
        ovrtx_gpu_lease.resolve_gpu_id({"CUDA_VISIBLE_DEVICES": "GPU-direct"})
        == "GPU-direct"
    )
    assert (
        ovrtx_gpu_lease.resolve_gpu_id({"CUDA_VISIBLE_DEVICES": "2"})
        == "GPU-two"
    )
    assert (
        ovrtx_gpu_lease.resolve_gpu_id({"OVRTX_ACTIVE_CUDA_GPUS": "2"})
        == "GPU-two"
    )
    assert ovrtx_gpu_lease.resolve_gpu_id({}) == "GPU-zero"
