# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readiness checks for launched interactive Blender validation sessions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol


READY_VIEWPORT_STATUSES = frozenset({"running", "complete"})


class PollableProcess(Protocol):
    def poll(self) -> int | None:
        """Return process exit status when exited, otherwise ``None``."""


@dataclass(frozen=True)
class ViewportArtifactReadiness:
    ready: bool
    status: str
    error: str = ""


@dataclass(frozen=True)
class InteractiveStartupReadiness:
    status: str
    ready: bool
    blender_exit_status: int | None = None
    viewport_status: str = ""
    error: str = ""


def viewport_artifact_readiness(path: Path) -> ViewportArtifactReadiness:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ViewportArtifactReadiness(False, "missing")
    except json.JSONDecodeError as exc:
        return ViewportArtifactReadiness(False, "invalid_json", str(exc))
    except OSError as exc:
        return ViewportArtifactReadiness(False, "unreadable", str(exc))
    if not isinstance(payload, Mapping):
        return ViewportArtifactReadiness(False, "invalid_payload")
    if payload.get("artifact_id") != "ovrtx-viewport-preview":
        return ViewportArtifactReadiness(False, "unexpected_artifact")
    runtime_startup = payload.get("runtime_startup")
    if isinstance(runtime_startup, Mapping):
        failure = _runtime_startup_failure(runtime_startup)
        if failure:
            return ViewportArtifactReadiness(False, "runtime_startup_failed", failure)
    status = str(payload.get("status", ""))
    return ViewportArtifactReadiness(status in READY_VIEWPORT_STATUSES, status or "missing_status")


def _runtime_startup_failure(runtime_startup: Mapping[str, Any]) -> str:
    for name in ("render_worker", "physics_worker"):
        worker = runtime_startup.get(name)
        if not isinstance(worker, Mapping):
            continue
        status = str(worker.get("status", ""))
        if status in {"failed", "endpoint_mismatch"}:
            error = str(worker.get("error", "") or worker.get("reason", ""))
            return f"{name}:{status}" + (f": {error}" if error else "")
    return ""


def wait_for_interactive_startup(
    process: PollableProcess,
    viewport_artifact_path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float = 0.1,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> InteractiveStartupReadiness:
    deadline = monotonic() + max(0.0, float(timeout_s))
    while True:
        readiness = viewport_artifact_readiness(viewport_artifact_path)
        if readiness.ready:
            return InteractiveStartupReadiness(
                status="ready",
                ready=True,
                viewport_status=readiness.status,
            )
        if readiness.status == "runtime_startup_failed":
            return InteractiveStartupReadiness(
                status="failed",
                ready=False,
                viewport_status=readiness.status,
                error=readiness.error,
            )
        returncode = process.poll()
        if returncode is not None:
            return InteractiveStartupReadiness(
                status="exited",
                ready=False,
                blender_exit_status=returncode,
                viewport_status=readiness.status,
                error="Blender exited before viewport artifact reached a running state.",
            )
        if monotonic() >= deadline:
            return InteractiveStartupReadiness(
                status="timed_out",
                ready=False,
                viewport_status=readiness.status,
                error="Timed out waiting for viewport artifact to reach a running state.",
            )
        sleep(min(poll_interval_s, max(0.0, deadline - monotonic())))
