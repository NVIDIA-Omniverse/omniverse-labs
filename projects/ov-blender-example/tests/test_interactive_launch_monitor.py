# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.interactive_launch_monitor import (  # noqa: E402
    viewport_artifact_readiness,
    wait_for_interactive_startup,
)


class _FakeProcess:
    def __init__(self, exit_status: int | None = None) -> None:
        self.exit_status = exit_status

    def poll(self) -> int | None:
        return self.exit_status


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_viewport_artifact_readiness_requires_running_viewport_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "viewport-preview.json"

    assert viewport_artifact_readiness(artifact).status == "missing"

    artifact.write_text('{"artifact_id": "other", "status": "running"}\n', encoding="utf-8")
    assert viewport_artifact_readiness(artifact).status == "unexpected_artifact"

    artifact.write_text('{"artifact_id": "ovrtx-viewport-preview", "status": "stopped"}\n', encoding="utf-8")
    stopped = viewport_artifact_readiness(artifact)
    assert stopped.ready is False
    assert stopped.status == "stopped"

    artifact.write_text('{"artifact_id": "ovrtx-viewport-preview", "status": "running"}\n', encoding="utf-8")
    running = viewport_artifact_readiness(artifact)
    assert running.ready is True
    assert running.status == "running"


def test_viewport_artifact_readiness_rejects_runtime_startup_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "viewport-preview.json"
    artifact.write_text(
        """
        {
          "artifact_id": "ovrtx-viewport-preview",
          "status": "running",
          "runtime_startup": {
            "render_worker": {
              "status": "failed",
              "error": "serving=false"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    readiness = viewport_artifact_readiness(artifact)

    assert readiness.ready is False
    assert readiness.status == "runtime_startup_failed"
    assert "render_worker:failed" in readiness.error


def test_wait_for_interactive_startup_reports_ready_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "viewport-preview.json"
    artifact.write_text('{"artifact_id": "ovrtx-viewport-preview", "status": "complete"}\n', encoding="utf-8")

    readiness = wait_for_interactive_startup(
        _FakeProcess(),
        artifact,
        timeout_s=1.0,
        sleep=lambda _duration: None,
    )

    assert readiness.ready is True
    assert readiness.status == "ready"
    assert readiness.viewport_status == "complete"


def test_wait_for_interactive_startup_fails_fast_on_runtime_startup_failure(tmp_path: Path) -> None:
    artifact = tmp_path / "viewport-preview.json"
    artifact.write_text(
        """
        {
          "artifact_id": "ovrtx-viewport-preview",
          "status": "running",
          "runtime_startup": {
            "render_worker": {
              "status": "endpoint_mismatch",
              "reason": "requested 50052 observed 50051"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    readiness = wait_for_interactive_startup(
        _FakeProcess(),
        artifact,
        timeout_s=100.0,
        sleep=lambda _duration: None,
    )

    assert readiness.ready is False
    assert readiness.status == "failed"
    assert readiness.viewport_status == "runtime_startup_failed"
    assert "endpoint_mismatch" in readiness.error


def test_wait_for_interactive_startup_reports_early_exit(tmp_path: Path) -> None:
    readiness = wait_for_interactive_startup(
        _FakeProcess(exit_status=17),
        tmp_path / "viewport-preview.json",
        timeout_s=1.0,
        sleep=lambda _duration: None,
    )

    assert readiness.ready is False
    assert readiness.status == "exited"
    assert readiness.blender_exit_status == 17
    assert "before viewport artifact" in readiness.error


def test_wait_for_interactive_startup_reports_timeout(tmp_path: Path) -> None:
    clock = _Clock()

    readiness = wait_for_interactive_startup(
        _FakeProcess(),
        tmp_path / "viewport-preview.json",
        timeout_s=0.25,
        poll_interval_s=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert readiness.ready is False
    assert readiness.status == "timed_out"
    assert readiness.viewport_status == "missing"
